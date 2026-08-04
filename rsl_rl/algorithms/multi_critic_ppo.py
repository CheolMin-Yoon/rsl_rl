# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-weight multi-critic Proximal Policy Optimization.

This module keeps one actor and an ordered set of independent scalar critics. Each critic receives one objective reward,
computes its own complete-rollout GAE, and contributes a fixed weighted normalized advantage to the actor update.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import MultiCriticRolloutStorage, RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer

OBJECTIVE_REWARDS_KEY = "objective_rewards"
"""Key in ``extras`` containing rewards with shape ``[num_envs, num_objectives]``."""


class MultiCriticPPO:
    """PPO with one actor and ordered independent scalar critics."""

    actor: MLPModel
    """The actor model."""

    critics: nn.ModuleDict
    """Independent scalar critic models keyed by objective name."""

    def __init__(
        self,
        actor: MLPModel,
        critics: Mapping[str, MLPModel],
        storage: MultiCriticRolloutStorage,
        reward_group_weights: torch.Tensor | Sequence[float],
        objective_names: Sequence[str] | None = None,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the algorithm with an ordered objective identity and fixed weights."""
        critic_names = tuple(critics)
        if isinstance(objective_names, str):
            raise ValueError("objective_names must be an ordered sequence of names, not one string.")
        configured_objective_names = critic_names if objective_names is None else tuple(objective_names)
        if not configured_objective_names:
            raise ValueError("MultiCriticPPO requires at least one objective critic.")
        if critic_names != configured_objective_names or configured_objective_names != storage.objective_names:
            raise ValueError("critics and storage must use the same ordered objective names.")
        reward_group_weights_tensor = torch.as_tensor(reward_group_weights, dtype=torch.float32, device=device)
        if tuple(reward_group_weights_tensor.shape) != (len(configured_objective_names),):
            raise ValueError(
                f"reward_group_weights must have fixed shape {(len(configured_objective_names),)}, "
                f"got {tuple(reward_group_weights_tensor.shape)}."
            )
        if not torch.isfinite(reward_group_weights_tensor).all():
            raise ValueError("reward_group_weights must be finite.")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1].")
        if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be finite and in [0, 1].")
        if normalize_advantage_per_mini_batch:
            raise ValueError("MultiCriticPPO normalizes each objective over the complete rollout.")
        if rnd_cfg is not None:
            raise ValueError("MultiCriticPPO does not support RND; provide intrinsic reward as an explicit objective.")
        if symmetry_cfg is not None:
            raise ValueError("MultiCriticPPO does not currently support the symmetry extension.")
        if actor.is_recurrent or any(critic.is_recurrent for critic in critics.values()):
            raise ValueError("MultiCriticPPO supports feed-forward actor and critic models only.")

        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.objective_names = configured_objective_names
        self.actor = actor.to(self.device)
        self.critics = nn.ModuleDict(dict(critics)).to(self.device)
        self._check_independent_parameters()

        # Handles to uncompiled models for state_dict operations and export.
        self._raw_actor = self.actor
        self._raw_critics = self.critics

        # One optimizer and one parameter collection preserve a single global gradient clip.
        self._parameters = tuple(chain(self.actor.parameters(), self.critics.parameters()))
        self.optimizer = resolve_optimizer(optimizer)(self._parameters, lr=learning_rate)  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.reward_group_weights = reward_group_weights_tensor.detach().clone()
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = False

        # Keep the stock runner's optional extension attributes available.
        self.rnd = None
        self.symmetry = None
        self.intrinsic_rewards = None

    def _check_independent_parameters(self) -> None:
        """Reject parameter sharing between the actor and objective critics."""
        owners: dict[int, str] = {}
        for model_name, model in (("actor", self.actor), *self.critics.items()):
            for parameter in model.parameters():
                previous_owner = owners.setdefault(id(parameter), model_name)
                if previous_owner != model_name:
                    raise ValueError(
                        "MultiCriticPPO models must not share parameters; "
                        f"{previous_owner!r} and {model_name!r} overlap."
                    )

    def _compute_values(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate every scalar critic in objective order."""
        return torch.cat([self.critics[name](obs) for name in self.objective_names], dim=-1)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actor actions and record all objective values."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), None)
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self._compute_values(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(
            parameter.detach() for parameter in self.actor.output_distribution_params
        )
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one vector-reward environment step and update normalizers."""
        objective_rewards = extras.get(OBJECTIVE_REWARDS_KEY)
        expected_shape = (self.storage.num_envs, len(self.objective_names))
        if not isinstance(objective_rewards, torch.Tensor) or tuple(objective_rewards.shape) != expected_shape:
            raise ValueError(
                f"MultiCriticPPO requires extras[{OBJECTIVE_REWARDS_KEY!r}] with shape {expected_shape}, "
                f"got {getattr(objective_rewards, 'shape', None)}."
            )
        if not torch.is_floating_point(objective_rewards):
            raise ValueError(f"extras[{OBJECTIVE_REWARDS_KEY!r}] must be a floating-point tensor.")
        if not torch.isfinite(objective_rewards).all():
            raise ValueError(f"extras[{OBJECTIVE_REWARDS_KEY!r}] must contain only finite values.")
        if rewards.numel() != self.storage.num_envs:
            raise ValueError(f"rewards must contain {self.storage.num_envs} scalar environment rewards.")

        self.actor.update_normalization(obs)
        for critic in self.critics.values():
            critic.update_normalization(obs)  # type: ignore[attr-defined]

        self.transition.rewards = objective_rewards.clone().to(self.device)
        self.transition.dones = dones

        if "time_outs" in extras:
            time_outs = extras["time_outs"]
            if time_outs.numel() != self.storage.num_envs:
                raise ValueError(f"time_outs must contain {self.storage.num_envs} entries.")
            self.transition.rewards += self.gamma * self.transition.values * time_outs.to(self.device).view(-1, 1)  # type: ignore[operator]

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        for critic in self.critics.values():
            critic.reset(dones)  # type: ignore[attr-defined]

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute objective GAEs and the fixed weighted actor advantage."""
        last_values = self._compute_values(obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam, self.reward_group_weights)

    def update(self) -> dict[str, float]:
        """Run PPO epochs over shared samples and return mean losses."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_objective_value_losses = dict.fromkeys(self.objective_names, 0.0)
        batch_count = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self._compute_values(batch.observations)  # type: ignore[arg-type]
            distribution_params = tuple(parameter for parameter in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            # Compute KL divergence and adapt the learning rate.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.learning_rate = learning_rate.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            advantages = torch.squeeze(batch.advantages)  # type: ignore
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss is one mean over both sample and objective axes.
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)  # type: ignore
                value_errors = torch.max(
                    (values - batch.returns).pow(2),  # type: ignore[operator]
                    (value_clipped - batch.returns).pow(2),  # type: ignore[operator]
                )
            else:
                value_errors = (batch.returns - values).pow(2)  # type: ignore[operator]
            value_loss = value_errors.mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self._parameters, self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            objective_value_losses = value_errors.mean(dim=0)
            for objective_index, objective_name in enumerate(self.objective_names):
                mean_objective_value_losses[objective_name] += objective_value_losses[objective_index].item()
            batch_count += 1

        self.storage.clear()
        losses = {
            "value": mean_value_loss / batch_count,
            "surrogate": mean_surrogate_loss / batch_count,
            "entropy": mean_entropy / batch_count,
        }
        losses.update({
            f"value/{objective_name}": objective_loss / batch_count
            for objective_name, objective_loss in mean_objective_value_losses.items()
        })
        return losses

    def train_mode(self) -> None:
        """Set train mode for actor and critics."""
        self.actor.train()
        self.critics.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for actor and critics."""
        self.actor.eval()
        self.critics.eval()

    def save(self) -> dict:
        """Return all learnable state plus exact objective configuration."""
        return {
            "objective_names": self.objective_names,
            "reward_group_weights": self.reward_group_weights.cpu(),
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dicts": {
                objective_name: self._raw_critics[objective_name].state_dict()
                for objective_name in self.objective_names
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load selected state after validating exact objective identity and weights."""
        loaded_objective_names = tuple(loaded_dict["objective_names"])
        if loaded_objective_names != self.objective_names:
            raise ValueError(
                f"Checkpoint objectives {loaded_objective_names!r} do not match "
                f"required objectives {self.objective_names!r}."
            )
        loaded_weights = loaded_dict["reward_group_weights"].to(self.device)
        if not torch.equal(loaded_weights, self.reward_group_weights):
            raise ValueError("Checkpoint reward_group_weights do not match the fixed learner configuration.")

        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}

        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            critic_state_dicts = loaded_dict["critic_state_dicts"]
            if tuple(critic_state_dicts) != self.objective_names:
                raise ValueError("Checkpoint critic state order does not match objective order.")
            for objective_name in self.objective_names:
                self._raw_critics[objective_name].load_state_dict(critic_state_dicts[objective_name], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the uncompiled actor model for inference and export."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile the actor and every critic with ``torch.compile``."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critics = nn.ModuleDict({
            objective_name: compile_model(self._raw_critics[objective_name], mode)
            for objective_name in self.objective_names
        })

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MultiCriticPPO:
        """Construct the actor, ordered critics, vector storage, and fixed weights."""
        algorithm_cfg = cfg["algorithm"]
        alg_class: type[MultiCriticPPO] = resolve_callable(algorithm_cfg.pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        configured_objective_names = algorithm_cfg["objective_names"]
        if isinstance(configured_objective_names, str):
            raise ValueError("objective_names must be an ordered sequence of names, not one string.")
        objective_names = tuple(configured_objective_names)
        if not objective_names or any(not isinstance(name, str) or not name for name in objective_names):
            raise ValueError("objective_names must contain at least one non-empty string.")
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("objective_names must be unique.")
        if algorithm_cfg.pop("share_cnn_encoders", False):
            raise ValueError("MultiCriticPPO requires independent actor and critic parameters.")

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critics = {
            objective_name: critic_class(
                obs,
                cfg["obs_groups"],
                "critic",
                1,
                **copy.deepcopy(cfg["critic"]),
            ).to(device)
            for objective_name in objective_names
        }
        for objective_name, critic in critics.items():
            print(f"Critic Model [{objective_name}]: {critic}")

        storage = MultiCriticRolloutStorage(
            objective_names,
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
        )
        algorithm = alg_class(
            actor,
            critics,
            storage,
            device=device,
            **algorithm_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm

    def broadcast_parameters(self) -> None:
        """Broadcast actor and all critic parameters from rank zero."""
        model_params = [self._raw_actor.state_dict()]
        model_params.extend(self._raw_critics[name].state_dict() for name in self.objective_names)
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        for critic_index, objective_name in enumerate(self.objective_names, start=1):
            self._raw_critics[objective_name].load_state_dict(model_params[critic_index])

    def reduce_parameters(self) -> None:
        """Average actor and critic gradients across all GPUs."""
        gradients = [parameter.grad.view(-1) for parameter in self._parameters if parameter.grad is not None]
        all_gradients = torch.cat(gradients)
        torch.distributed.all_reduce(all_gradients, op=torch.distributed.ReduceOp.SUM)
        all_gradients /= self.gpu_world_size

        offset = 0
        for parameter in self._parameters:
            if parameter.grad is not None:
                numel = parameter.numel()
                parameter.grad.data.copy_(all_gradients[offset : offset + numel].view_as(parameter.grad.data))
                offset += numel
