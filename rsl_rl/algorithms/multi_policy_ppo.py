# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose multiple named policies from independent stock PPO algorithms."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from collections.abc import Mapping
from tensordict import TensorDict
from types import SimpleNamespace

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_callable


class _JointPolicy(nn.Module):
    """Concatenate deterministic policy outputs in declaration order."""

    def __init__(self, algorithms: tuple[PPO, ...]) -> None:
        super().__init__()
        self.actors = nn.ModuleList(algorithm.get_policy() for algorithm in algorithms)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Return the concatenated deterministic action."""
        return torch.cat([actor(obs) for actor in self.actors], dim=-1)

    @property
    def output_std(self) -> torch.Tensor:
        """Return all policy standard deviations as one vector."""
        return torch.cat([actor.output_std.reshape(-1) for actor in self.actors])


class MultiPolicyPPO:
    """Run one independent stock PPO for each named physical policy."""

    def __init__(
        self,
        algorithms: tuple[PPO, ...],
        policy_names: tuple[str, ...],
        observation_keys: tuple[tuple[str, ...], ...],
        num_envs: int,
    ) -> None:
        """Initialize the composite from already constructed PPO algorithms."""
        self.algorithms = algorithms
        self.policy_names = policy_names
        self.observation_keys = observation_keys
        self.num_envs = num_envs
        self.device = algorithms[0].device
        self.rnd = None
        self.symmetry = None
        self.intrinsic_rewards = None
        self._policy = _JointPolicy(algorithms)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample each policy action and concatenate them in declaration order."""
        actions = [
            algorithm.act(obs.select(*keys))
            for algorithm, keys in zip(self.algorithms, self.observation_keys, strict=True)
        ]
        return torch.cat(actions, dim=-1)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, object]
    ) -> None:
        """Route named policy rewards to the corresponding PPO transition."""
        policy_rewards = self._stack_policy_rewards(extras.get("policy_rewards"))
        for policy_index, (algorithm, keys) in enumerate(zip(self.algorithms, self.observation_keys, strict=True)):
            algorithm.process_env_step(
                obs.select(*keys),
                policy_rewards[:, policy_index],
                dones,
                extras,  # type: ignore[arg-type]
            )

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute returns independently for every policy."""
        for algorithm, keys in zip(self.algorithms, self.observation_keys, strict=True):
            algorithm.compute_returns(obs.select(*keys))

    def update(self) -> dict[str, float]:
        """Update every PPO and namespace its losses by policy name."""
        losses: dict[str, float] = {}
        for name, algorithm in zip(self.policy_names, self.algorithms, strict=True):
            losses.update({f"{name}/{loss_name}": value for loss_name, value in algorithm.update().items()})
        return losses

    @property
    def learning_rate(self) -> float:
        """Return the mean learning rate used by the independent PPOs."""
        return sum(algorithm.learning_rate for algorithm in self.algorithms) / len(self.algorithms)

    def train_mode(self) -> None:
        """Set every policy to training mode."""
        for algorithm in self.algorithms:
            algorithm.train_mode()

    def eval_mode(self) -> None:
        """Set every policy to evaluation mode."""
        for algorithm in self.algorithms:
            algorithm.eval_mode()

    def save(self) -> dict:
        """Return all policy states together with their ordered identity."""
        return {
            "policy_names": self.policy_names,
            "policy_state_dicts": {
                name: algorithm.save() for name, algorithm in zip(self.policy_names, self.algorithms, strict=True)
            },
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load policy states only when checkpoint identity exactly matches."""
        checkpoint_names = tuple(loaded_dict["policy_names"])
        states = loaded_dict["policy_state_dicts"]
        if checkpoint_names != self.policy_names or tuple(states) != self.policy_names:
            raise ValueError(
                f"Checkpoint policies {checkpoint_names!r} do not match required policies {self.policy_names!r}."
            )

        load_iteration = False
        for name, algorithm in zip(self.policy_names, self.algorithms, strict=True):
            load_iteration = algorithm.load(states[name], load_cfg, strict) or load_iteration
        return load_iteration

    def get_policy(self) -> _JointPolicy:
        """Return the deterministic joint inference policy."""
        return self._policy

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MultiPolicyPPO:
        """Construct independently configured stock PPOs in policy declaration order."""
        algorithm_cfg = cfg["algorithm"]
        alg_class: type[MultiPolicyPPO] = resolve_callable(algorithm_cfg.pop("class_name"))  # type: ignore
        algorithm_cfg.setdefault("rnd_cfg", None)
        algorithm_cfg.setdefault("symmetry_cfg", None)
        if algorithm_cfg["rnd_cfg"] is not None or algorithm_cfg["symmetry_cfg"] is not None:
            raise ValueError("MultiPolicyPPO does not support RND or symmetry extensions.")
        unexpected_algorithm_fields = tuple(
            field for field in algorithm_cfg if field not in {"rnd_cfg", "symmetry_cfg"}
        )
        if unexpected_algorithm_fields:
            raise ValueError(
                "MultiPolicyPPO algorithm options must be configured under each policy; "
                f"unexpected top-level fields: {unexpected_algorithm_fields}."
            )

        policies = cfg.get("policies")
        if not isinstance(policies, Mapping) or not policies:
            raise ValueError("MultiPolicyPPO requires at least one named policy.")

        policy_names = tuple(policies)
        if any(not isinstance(name, str) or not name for name in policy_names):
            raise ValueError("MultiPolicyPPO policy names must be non-empty strings.")

        action_dims: list[int] = []
        for name, policy in policies.items():
            if not isinstance(policy, Mapping):
                raise ValueError(f"Policy {name!r} must be a configuration mapping.")
            missing_fields = tuple(
                field for field in ("num_actions", "obs_groups", "actor", "critic", "algorithm") if field not in policy
            )
            if missing_fields:
                raise ValueError(f"Policy {name!r} is missing required fields: {missing_fields}.")
            unexpected_fields = tuple(
                field for field in policy if field not in {"num_actions", "obs_groups", "actor", "critic", "algorithm"}
            )
            if unexpected_fields:
                raise ValueError(f"Policy {name!r} has unsupported fields: {unexpected_fields}.")
            action_dim = policy["num_actions"]
            if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
                raise ValueError(f"Policy {name!r} num_actions must be a positive integer.")
            action_dims.append(action_dim)

        if sum(action_dims) != env.num_actions:
            raise ValueError(
                f"Policy action dimensions sum to {sum(action_dims)}, but the environment expects {env.num_actions}."
            )

        algorithms: list[PPO] = []
        observation_keys: list[tuple[str, ...]] = []
        for (name, policy), action_dim in zip(policies.items(), action_dims, strict=True):
            obs_groups = policy["obs_groups"]
            if not isinstance(obs_groups, Mapping) or set(obs_groups) != {"actor", "critic"}:
                raise ValueError(f"Policy {name!r} must define exactly actor and critic observation groups.")
            actor_groups = tuple(obs_groups["actor"])
            critic_groups = tuple(obs_groups["critic"])
            if not actor_groups or not critic_groups:
                raise ValueError(f"Policy {name!r} actor and critic observation groups cannot be empty.")
            keys = tuple(dict.fromkeys((*actor_groups, *critic_groups)))

            actor_cfg = _strip_none_model_options(policy["actor"])
            critic_cfg = _strip_none_model_options(policy["critic"])
            actor_class = resolve_callable(actor_cfg["class_name"])
            critic_class = resolve_callable(critic_cfg["class_name"])
            if getattr(actor_class, "is_recurrent", False) or getattr(critic_class, "is_recurrent", False):
                raise ValueError("MultiPolicyPPO does not support recurrent actor or critic models.")

            policy_algorithm_cfg = copy.deepcopy(policy["algorithm"])
            if policy_algorithm_cfg.get("class_name") != "PPO":
                raise ValueError(f"Policy {name!r} must use the stock RSL-RL PPO class.")
            if policy_algorithm_cfg.get("rnd_cfg") is not None or policy_algorithm_cfg.get("symmetry_cfg") is not None:
                raise ValueError("MultiPolicyPPO policy entries do not support RND or symmetry extensions.")

            ppo_cfg = {
                "num_steps_per_env": cfg["num_steps_per_env"],
                "obs_groups": {"actor": actor_groups, "critic": critic_groups},
                "actor": actor_cfg,
                "critic": critic_cfg,
                "algorithm": policy_algorithm_cfg,
                "multi_gpu": cfg["multi_gpu"],
                "torch_compile_mode": cfg.get("torch_compile_mode"),
            }
            ppo_env = SimpleNamespace(num_envs=env.num_envs, num_actions=action_dim)
            algorithms.append(
                PPO.construct_algorithm(obs.select(*keys), ppo_env, ppo_cfg, device)  # type: ignore[arg-type]
            )
            observation_keys.append(keys)

        return alg_class(tuple(algorithms), policy_names, tuple(observation_keys), env.num_envs)

    def broadcast_parameters(self) -> None:
        """Broadcast every policy's parameters to all workers."""
        for algorithm in self.algorithms:
            algorithm.broadcast_parameters()

    def _stack_policy_rewards(self, policy_rewards: object) -> torch.Tensor:
        """Validate named policy rewards and stack them in action order."""
        if not isinstance(policy_rewards, Mapping):
            raise ValueError("MultiPolicyPPO requires extras['policy_rewards'] to be a mapping.")

        expected_names = set(self.policy_names)
        actual_names = set(policy_rewards)
        if actual_names != expected_names:
            missing = tuple(name for name in self.policy_names if name not in actual_names)
            extra = tuple(name for name in policy_rewards if name not in expected_names)
            raise ValueError(f"Policy reward names do not match; missing={missing}, extra={extra}.")

        ordered_rewards: list[torch.Tensor] = []
        for name in self.policy_names:
            reward = policy_rewards[name]
            if not isinstance(reward, torch.Tensor):
                raise ValueError(f"Policy reward {name!r} must be a torch.Tensor.")
            if tuple(reward.shape) != (self.num_envs,):
                raise ValueError(
                    f"Policy reward {name!r} must have shape {(self.num_envs,)}, got {tuple(reward.shape)}."
                )
            if not reward.is_floating_point():
                raise ValueError(f"Policy reward {name!r} must use a floating dtype, got {reward.dtype}.")
            ordered_rewards.append(reward.to(self.device))

        stacked_rewards = torch.stack(ordered_rewards, dim=-1)
        if not torch.isfinite(stacked_rewards).all():
            raise ValueError("Policy rewards must contain only finite values.")
        return stacked_rewards


def _strip_none_model_options(model_cfg: Mapping[str, object]) -> dict:
    """Remove optional MJLab model fields that are inactive when set to None."""
    stripped = copy.deepcopy(dict(model_cfg))
    for option in ("cnn_cfg", "distribution_cfg"):
        if stripped.get(option) is None:
            stripped.pop(option, None)
    if stripped.get("rnn_type") is None:
        for option in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            stripped.pop(option, None)
    return stripped
