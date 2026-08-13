# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator, Sequence
from tensordict import TensorDict

from .rollout_storage import RolloutStorage


class MultiCriticRolloutStorage:
    """Feed-forward rollout storage with an ordered objective axis.

    Rewards, values, returns, and objective advantages use shape ``[time, env, objective]``. The actor advantage is
    the fixed weighted sum of objective-wise advantages after each objective is normalized over the complete rollout.
    """

    Transition = RolloutStorage.Transition
    Batch = RolloutStorage.Batch

    def __init__(
        self,
        objective_names: Sequence[str],
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        """Allocate rollout buffers for a fixed ordered set of objectives."""
        if isinstance(objective_names, str):
            raise ValueError("objective_names must be an ordered sequence of names, not one string.")
        self.objective_names = tuple(objective_names)
        if not self.objective_names or any(not isinstance(name, str) or not name for name in self.objective_names):
            raise ValueError("objective_names must contain at least one non-empty string.")
        if len(set(self.objective_names)) != len(self.objective_names):
            raise ValueError("objective_names must be unique.")

        self.training_type = "rl"
        self.device = device
        self.num_objectives = len(self.objective_names)
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actions_shape = tuple(actions_shape)

        self.observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=device,
        )
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *self.actions_shape, device=device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, self.num_objectives, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.uint8, device=device)
        self.values = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.actor_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.distribution_params: tuple[torch.Tensor, ...] | None = None
        self.step = 0

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Add one complete vector-reward transition at the current step."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        required = (
            transition.observations,
            transition.actions,
            transition.rewards,
            transition.dones,
            transition.values,
            transition.actions_log_prob,
            transition.distribution_params,
        )
        if any(field is None for field in required):
            raise ValueError("Transition is incomplete.")

        objective_shape = (self.num_envs, self.num_objectives)
        if tuple(transition.rewards.shape) != objective_shape:  # type: ignore[union-attr]
            raise ValueError(
                f"Transition rewards must have shape {objective_shape}, got {tuple(transition.rewards.shape)}."  # type: ignore[union-attr]
            )
        if tuple(transition.values.shape) != objective_shape:  # type: ignore[union-attr]
            raise ValueError(
                f"Transition values must have shape {objective_shape}, got {tuple(transition.values.shape)}."  # type: ignore[union-attr]
            )

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)  # type: ignore[arg-type]
        self.rewards[self.step].copy_(transition.rewards)  # type: ignore[arg-type]
        self.dones[self.step].copy_(transition.dones.view(-1, 1))  # type: ignore[union-attr]
        self.values[self.step].copy_(transition.values)  # type: ignore[arg-type]
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))  # type: ignore[union-attr]

        distribution_params = transition.distribution_params
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(
                    self.num_transitions_per_env,
                    *parameter.shape,
                    dtype=parameter.dtype,
                    device=self.device,
                )
                for parameter in distribution_params  # type: ignore[union-attr]
            )
        if len(self.distribution_params) != len(distribution_params):  # type: ignore[arg-type]
            raise ValueError("Actor distribution parameter count changed during rollout collection.")
        for stored, parameter in zip(self.distribution_params, distribution_params):  # type: ignore[arg-type]
            stored[self.step].copy_(parameter)
        self.step += 1

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        reward_group_weights: torch.Tensor,
    ) -> None:
        """Compute objective-wise GAE and the fixed weighted actor advantage."""
        if self.step != self.num_transitions_per_env:
            raise RuntimeError(
                f"Rollout is incomplete: {self.step}/{self.num_transitions_per_env} transitions collected."
            )
        expected_values_shape = (self.num_envs, self.num_objectives)
        if tuple(last_values.shape) != expected_values_shape:
            raise ValueError(f"last_values must have shape {expected_values_shape}, got {tuple(last_values.shape)}.")
        if tuple(reward_group_weights.shape) != (self.num_objectives,):
            raise ValueError(
                f"reward_group_weights must have shape {(self.num_objectives,)}, "
                f"got {tuple(reward_group_weights.shape)}."
            )

        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        objective_advantages = self.returns - self.values
        mean = objective_advantages.mean(dim=(0, 1), keepdim=True)
        if self.num_envs * self.num_transitions_per_env > 1:
            # HoST uses torch.std()'s sample-standard-deviation semantics. Keep the correction explicit so the
            # reference-compatible normalization does not change if a framework default changes.
            std = objective_advantages.std(dim=(0, 1), keepdim=True, correction=1)
        else:
            std = torch.zeros_like(mean)
        normalized = (objective_advantages - mean) / (std + 1.0e-8)
        self.advantages.copy_(normalized)
        weights = reward_group_weights.to(device=self.device, dtype=self.advantages.dtype).view(1, 1, -1)
        self.actor_advantages.copy_((self.advantages * weights).sum(dim=-1, keepdim=True))

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Yield shared shuffled feed-forward batches for the actor and every critic."""
        if self.step != self.num_transitions_per_env:
            raise RuntimeError(
                f"Rollout is incomplete: {self.step}/{self.num_transitions_per_env} transitions collected."
            )
        if self.distribution_params is None:
            raise RuntimeError("No actor distribution parameters were recorded.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError("num_mini_batches exceeds the rollout batch size.")
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.actor_advantages.flatten(0, 1)
        old_distribution_params = tuple(parameter.flatten(0, 1) for parameter in self.distribution_params)

        for _ in range(num_epochs):
            for batch_index in range(num_mini_batches):
                start = batch_index * mini_batch_size
                stop = start + mini_batch_size
                selected = indices[start:stop]
                yield RolloutStorage.Batch(
                    observations=observations[selected],
                    actions=actions[selected],
                    values=values[selected],
                    advantages=advantages[selected],
                    returns=returns[selected],
                    old_actions_log_prob=old_actions_log_prob[selected],
                    old_distribution_params=tuple(parameter[selected] for parameter in old_distribution_params),
                )

    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Reject recurrent use until multiple critic hidden states have an explicit interface."""
        del num_mini_batches, num_epochs
        raise NotImplementedError("MultiCriticRolloutStorage supports feed-forward models only.")
        yield  # pragma: no cover

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0
