# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose ordered policies as one same-timestep autoregressive policy."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.multi_policy_ppo import MultiPolicyPPO
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv


def _action_observation_key(policy_name: str) -> str:
    """Return the private observation key carrying one preceding policy action."""
    return f"_sequential_multi_policy_{policy_name}_action"


def _policy_observation(
    obs: TensorDict,
    observation_keys: tuple[str, ...],
    dependencies: tuple[str, ...],
    actions: Mapping[str, torch.Tensor],
) -> TensorDict:
    """Select environment observations and append same-timestep preceding actions."""
    policy_obs = obs.select(*observation_keys)
    for dependency in dependencies:
        policy_obs.set(_action_observation_key(dependency), actions[dependency])
    return policy_obs


class _SequentialJointPolicy(nn.Module):
    """Evaluate deterministic policies in declaration order."""

    def __init__(
        self,
        algorithms: tuple[PPO, ...],
        policy_names: tuple[str, ...],
        observation_keys: tuple[tuple[str, ...], ...],
        policy_dependencies: tuple[tuple[str, ...], ...],
    ) -> None:
        super().__init__()
        self.actors = nn.ModuleList(algorithm.get_policy() for algorithm in algorithms)
        self.policy_names = policy_names
        self.observation_keys = observation_keys
        self.policy_dependencies = policy_dependencies

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Return deterministic actions conditioned on all preceding policy actions."""
        actions: dict[str, torch.Tensor] = {}
        for name, actor, keys, dependencies in zip(
            self.policy_names,
            self.actors,
            self.observation_keys,
            self.policy_dependencies,
            strict=True,
        ):
            actions[name] = actor(_policy_observation(obs, keys, dependencies, actions))
        return torch.cat([actions[name] for name in self.policy_names], dim=-1)

    @property
    def output_std(self) -> torch.Tensor:
        """Return all policy standard deviations as one vector."""
        return torch.cat([actor.output_std.reshape(-1) for actor in self.actors])


class SequentialMultiPolicyPPO(MultiPolicyPPO):
    """Run ordered stock PPOs with each actor conditioned on preceding actions."""

    def __init__(
        self,
        algorithms: tuple[PPO, ...],
        policy_names: tuple[str, ...],
        observation_keys: tuple[tuple[str, ...], ...],
        num_envs: int,
    ) -> None:
        """Initialize the autoregressive policy from independently optimized PPOs."""
        policy_dependencies = tuple(tuple(policy_names[:index]) for index in range(len(policy_names)))
        dependency_keys = tuple(
            tuple(_action_observation_key(dependency) for dependency in dependencies)
            for dependencies in policy_dependencies
        )
        for name, keys, required_keys in zip(policy_names, observation_keys, dependency_keys, strict=True):
            missing_keys = tuple(key for key in required_keys if key not in keys)
            if missing_keys:
                raise ValueError(f"Policy {name!r} is missing sequential action observations: {missing_keys}.")
        environment_observation_keys = tuple(
            tuple(key for key in keys if key not in required_keys)
            for keys, required_keys in zip(observation_keys, dependency_keys, strict=True)
        )

        super().__init__(algorithms, policy_names, environment_observation_keys, num_envs)
        self.policy_dependencies = policy_dependencies
        self._last_actions: dict[str, torch.Tensor] | None = None
        self._policy = _SequentialJointPolicy(
            algorithms,
            policy_names,
            environment_observation_keys,
            policy_dependencies,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample each action after injecting all same-timestep preceding actions."""
        actions: dict[str, torch.Tensor] = {}
        for name, algorithm, keys, dependencies in zip(
            self.policy_names,
            self.algorithms,
            self.observation_keys,
            self.policy_dependencies,
            strict=True,
        ):
            actions[name] = algorithm.act(_policy_observation(obs, keys, dependencies, actions))
        self._last_actions = actions
        return torch.cat([actions[name] for name in self.policy_names], dim=-1)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, object]
    ) -> None:
        """Store conditioned observations and route named rewards to each PPO."""
        policy_rewards = self._stack_policy_rewards(extras.get("policy_rewards"))
        actions = self._require_last_actions()
        for policy_index, (algorithm, keys, dependencies) in enumerate(
            zip(self.algorithms, self.observation_keys, self.policy_dependencies, strict=True)
        ):
            algorithm.process_env_step(
                _policy_observation(obs, keys, dependencies, actions),
                policy_rewards[:, policy_index],
                dones,
                extras,  # type: ignore[arg-type]
            )

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute each PPO return using its complete conditioned observation schema."""
        actions = self._require_last_actions()
        for algorithm, keys, dependencies in zip(
            self.algorithms,
            self.observation_keys,
            self.policy_dependencies,
            strict=True,
        ):
            algorithm.compute_returns(_policy_observation(obs, keys, dependencies, actions))

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> SequentialMultiPolicyPPO:
        """Construct stock PPOs with actor inputs augmented by all preceding actions."""
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)
        prepared_cfg = copy.deepcopy(cfg)
        policies = prepared_cfg.get("policies")
        if not isinstance(policies, Mapping) or not policies:
            raise ValueError("SequentialMultiPolicyPPO requires at least one named policy.")

        policy_names = tuple(policies)
        action_dims: dict[str, int] = {}
        for name, policy in policies.items():
            if not isinstance(policy, Mapping):
                raise ValueError(f"Policy {name!r} must be a configuration mapping.")
            action_dim = policy.get("num_actions")
            if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
                raise ValueError(f"Policy {name!r} num_actions must be a positive integer.")
            action_dims[name] = action_dim

        augmented_obs = obs.clone()
        for index, (name, policy) in enumerate(policies.items()):
            obs_groups = policy.get("obs_groups")
            if not isinstance(obs_groups, Mapping) or set(obs_groups) != {"actor", "critic"}:
                raise ValueError(f"Policy {name!r} must define exactly actor and critic observation groups.")
            actor_groups = tuple(obs_groups["actor"])
            if not actor_groups:
                raise ValueError(f"Policy {name!r} actor observation groups cannot be empty.")
            if actor_groups[0] not in obs:
                raise ValueError(f"Policy {name!r} actor observation {actor_groups[0]!r} was not provided.")

            dependency_keys: list[str] = []
            reference = obs[actor_groups[0]]
            for dependency in policy_names[:index]:
                dependency_key = _action_observation_key(dependency)
                if dependency_key in obs:
                    raise ValueError(
                        f"Sequential action observation {dependency_key!r} conflicts with an environment observation."
                    )
                augmented_obs.set(
                    dependency_key,
                    reference.new_zeros((env.num_envs, action_dims[dependency])),
                )
                dependency_keys.append(dependency_key)
            policy["obs_groups"] = {
                "actor": (*actor_groups, *dependency_keys),
                "critic": tuple(obs_groups["critic"]),
            }

        algorithm = MultiPolicyPPO.construct_algorithm(augmented_obs, env, prepared_cfg, device)
        if not isinstance(algorithm, SequentialMultiPolicyPPO):
            raise TypeError("SequentialMultiPolicyPPO construction resolved the wrong algorithm class.")
        return algorithm

    def _require_last_actions(self) -> dict[str, torch.Tensor]:
        """Return the latest actions after enforcing the stock runner lifecycle."""
        if self._last_actions is None:
            raise RuntimeError("SequentialMultiPolicyPPO.act() must be called before processing a rollout step.")
        return self._last_actions
