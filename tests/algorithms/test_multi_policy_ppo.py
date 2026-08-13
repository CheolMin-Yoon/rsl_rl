# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration and contract tests for MultiPolicyPPO."""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import MultiPolicyPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 4
LEG_ACTIONS = 2
ARM_ACTIONS = 3


class MultiPolicyEnv(VecEnv):
    """Minimal environment exposing different observations and rewards per policy."""

    def __init__(self) -> None:
        """Initialize a CPU-only vector environment."""
        self.num_envs = NUM_ENVS
        self.num_actions = LEG_ACTIONS + ARM_ACTIONS
        self.max_episode_length = 20
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        """Return distinct actor and critic observations for both policies."""
        return TensorDict(
            {
                "leg_actor": torch.randn(self.num_envs, 3),
                "leg_critic": torch.randn(self.num_envs, 5),
                "arm_actor": torch.randn(self.num_envs, 4),
                "arm_critic": torch.randn(self.num_envs, 6),
            },
            batch_size=[self.num_envs],
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance one step and return policy rewards in reversed mapping order."""
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected actions with shape {(self.num_envs, self.num_actions)}.")
        leg_reward = 1.0 - actions[:, :LEG_ACTIONS].square().mean(dim=-1)
        arm_reward = 0.5 - actions[:, LEG_ACTIONS:].abs().mean(dim=-1)
        rewards = leg_reward + arm_reward
        dones = torch.zeros(self.num_envs)
        extras = {
            "policy_rewards": {"arm": arm_reward, "leg": leg_reward},
            "time_outs": torch.zeros(self.num_envs),
        }
        return self.get_observations(), rewards, dones, extras


def _model_config(stochastic: bool) -> dict:
    """Return an MJLab-style model config containing inactive None options."""
    return {
        "class_name": "MLPModel",
        "hidden_dims": [8],
        "activation": "elu",
        "cnn_cfg": None,
        "distribution_cfg": ({"class_name": "GaussianDistribution", "init_std": 0.5} if stochastic else None),
        "rnn_type": None,
        "rnn_hidden_dim": 32,
        "rnn_num_layers": 1,
    }


def _ppo_config() -> dict:
    """Return the stock PPO options used independently by each policy."""
    return {
        "class_name": "PPO",
        "num_learning_epochs": 1,
        "num_mini_batches": 2,
        "schedule": "fixed",
        "normalize_advantage_per_mini_batch": False,
        "share_cnn_encoders": False,
    }


def _make_config() -> dict:
    """Return a two-policy configuration in the public dictionary format."""
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "algorithm": {"class_name": "MultiPolicyPPO"},
        "policies": {
            "leg": {
                "num_actions": LEG_ACTIONS,
                "obs_groups": {"actor": ("leg_actor",), "critic": ("leg_critic",)},
                "actor": _model_config(stochastic=True),
                "critic": _model_config(stochastic=False),
                "algorithm": _ppo_config(),
            },
            "arm": {
                "num_actions": ARM_ACTIONS,
                "obs_groups": {"actor": ("arm_actor",), "critic": ("arm_critic",)},
                "actor": _model_config(stochastic=True),
                "critic": _model_config(stochastic=False),
                "algorithm": _ppo_config(),
            },
        },
    }


def _build_runner() -> OnPolicyRunner:
    """Construct MultiPolicyPPO through the unmodified stock runner."""
    return OnPolicyRunner(MultiPolicyEnv(), _make_config(), log_dir=None, device="cpu")


def test_stock_runner_constructs_and_updates_independent_policies() -> None:
    """Stock OnPolicyRunner should complete one update with different policy dimensions."""
    runner = _build_runner()

    assert isinstance(runner.alg, MultiPolicyPPO)
    assert runner.alg.policy_names == ("leg", "arm")
    assert tuple(algorithm.actor.distribution.output_dim for algorithm in runner.alg.algorithms) == (
        LEG_ACTIONS,
        ARM_ACTIONS,
    )
    parameters_before = [
        {name: parameter.clone() for name, parameter in algorithm.get_policy().named_parameters()}
        for algorithm in runner.alg.algorithms
    ]

    runner.learn(num_learning_iterations=1)

    assert all(algorithm.storage.step == 0 for algorithm in runner.alg.algorithms)
    for before, algorithm in zip(parameters_before, runner.alg.algorithms, strict=True):
        assert any(
            not torch.equal(before[name], parameter) for name, parameter in algorithm.get_policy().named_parameters()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cross-device reward routing.")
def test_stock_runner_moves_cpu_policy_rewards_to_cuda_learner() -> None:
    """CPU environment rewards should reach every CUDA PPO storage without a custom runner."""
    runner = OnPolicyRunner(MultiPolicyEnv(), _make_config(), log_dir=None, device="cuda")

    runner.learn(num_learning_iterations=1)

    assert all(torch.device(algorithm.storage.device).type == "cuda" for algorithm in runner.alg.algorithms)


def test_action_concatenation_and_reward_routing_use_policy_identity() -> None:
    """Action order follows declaration order while reward mapping order is irrelevant."""
    runner = _build_runner()
    observations = runner.env.get_observations()
    deterministic_actions = runner.alg.get_policy()(observations)
    expected_actions = torch.cat(
        [
            algorithm.get_policy()(observations.select(*keys))
            for algorithm, keys in zip(runner.alg.algorithms, runner.alg.observation_keys, strict=True)
        ],
        dim=-1,
    )
    actions = runner.alg.act(observations)
    next_observations, rewards, dones, extras = runner.env.step(actions)

    runner.alg.process_env_step(next_observations, rewards, dones, extras)

    torch.testing.assert_close(deterministic_actions, expected_actions)
    assert actions.shape == (NUM_ENVS, LEG_ACTIONS + ARM_ACTIONS)
    for name, algorithm in zip(runner.alg.policy_names, runner.alg.algorithms, strict=True):
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], extras["policy_rewards"][name])


def test_checkpoint_round_trip_preserves_deterministic_joint_inference(tmp_path: Path) -> None:
    """Checkpoint load should preserve ordered identity and deterministic joint output."""
    runner = _build_runner()
    runner.learn(num_learning_iterations=1)
    observations = runner.env.get_observations()
    expected_actions = runner.get_inference_policy()(observations)
    checkpoint_path = tmp_path / "multi_policy.pt"

    runner.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    restored = _build_runner()
    restored.load(str(checkpoint_path))
    actual_actions = restored.get_inference_policy()(observations)

    assert tuple(checkpoint["policy_names"]) == ("leg", "arm")
    assert tuple(checkpoint["policy_state_dicts"]) == ("leg", "arm")
    torch.testing.assert_close(actual_actions, expected_actions)


def test_checkpoint_rejects_different_policy_order() -> None:
    """Checkpoint policy identity includes declaration order."""
    runner = _build_runner()
    checkpoint = runner.alg.save()
    checkpoint["policy_names"] = ("arm", "leg")

    with pytest.raises(ValueError, match="do not match"):
        runner.alg.load(checkpoint, load_cfg=None, strict=True)


def test_action_dimension_sum_must_match_environment() -> None:
    """Construction should reject policy actions that do not cover the environment action."""
    config = _make_config()
    config["policies"]["arm"]["num_actions"] += 1

    with pytest.raises(ValueError, match="environment expects"):
        OnPolicyRunner(MultiPolicyEnv(), config, log_dir=None, device="cpu")


@pytest.mark.parametrize(
    ("policy_rewards", "message"),
    [
        ({"leg": torch.zeros(NUM_ENVS)}, "names do not match"),
        (
            {"leg": torch.zeros(NUM_ENVS), "arm": torch.zeros(NUM_ENVS), "head": torch.zeros(NUM_ENVS)},
            "names do not match",
        ),
        ({"leg": torch.zeros(NUM_ENVS, 1), "arm": torch.zeros(NUM_ENVS)}, "must have shape"),
        ({"leg": torch.zeros(NUM_ENVS, dtype=torch.long), "arm": torch.zeros(NUM_ENVS)}, "floating dtype"),
        ({"leg": torch.full((NUM_ENVS,), torch.nan), "arm": torch.zeros(NUM_ENVS)}, "finite values"),
        ({"leg": torch.full((NUM_ENVS,), torch.inf), "arm": torch.zeros(NUM_ENVS)}, "finite values"),
    ],
)
def test_invalid_policy_rewards_are_rejected(policy_rewards: dict, message: str) -> None:
    """Named rewards must have exact names, shape, floating dtype, and finite values."""
    runner = _build_runner()
    observations = runner.env.get_observations()
    runner.alg.act(observations)

    with pytest.raises(ValueError, match=message):
        runner.alg.process_env_step(
            observations,
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {"policy_rewards": policy_rewards},
        )


def test_policy_reward_mapping_is_required() -> None:
    """A positional reward tensor is not accepted as a policy reward contract."""
    runner = _build_runner()
    observations = runner.env.get_observations()
    runner.alg.act(observations)

    with pytest.raises(ValueError, match="to be a mapping"):
        runner.alg.process_env_step(
            observations,
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {"policy_rewards": torch.zeros(NUM_ENVS, 2)},
        )
