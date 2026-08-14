# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration and contract tests for SequentialMultiPolicyPPO."""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import MultiPolicyPPO, SequentialMultiPolicyPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 4
CONTACT_ACTIONS = 2
JOINT_ACTIONS = 3


def _contact_message(obs: TensorDict, action: torch.Tensor) -> torch.Tensor:
    """Expose both the sampled proposal and a deterministic accepted preview."""
    del obs
    return torch.cat((action, 1.0 - action), dim=-1)


def _alternate_contact_message(obs: TensorDict, action: torch.Tensor) -> torch.Tensor:
    """Return the same shape under a deliberately different checkpoint identity."""
    del obs
    return torch.cat((action, action), dim=-1)


class SequentialPolicyEnv(VecEnv):
    """Minimal environment with categorical contact and continuous joint actions."""

    def __init__(self) -> None:
        """Initialize the fixed-shape mock environment."""
        self.num_envs = NUM_ENVS
        self.num_actions = CONTACT_ACTIONS + JOINT_ACTIONS
        self.max_episode_length = 20
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        """Return separate deployable observations and centralized critic states."""
        return TensorDict(
            {
                "contact_actor": torch.randn(self.num_envs, 3),
                "contact_critic": torch.randn(self.num_envs, 5),
                "joint_actor": torch.randn(self.num_envs, 4),
                "joint_critic": torch.randn(self.num_envs, 6),
            },
            batch_size=[self.num_envs],
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Return named rewards for the same combined transition."""
        contact_reward = 1.0 - actions[:, :CONTACT_ACTIONS].mean(dim=-1)
        joint_reward = 0.5 - actions[:, CONTACT_ACTIONS:].square().mean(dim=-1)
        rewards = contact_reward + joint_reward
        dones = torch.zeros(self.num_envs)
        extras = {
            "policy_rewards": {"joint": joint_reward, "contact": contact_reward},
            "time_outs": torch.zeros(self.num_envs),
        }
        return self.get_observations(), rewards, dones, extras


def _model_config(distribution_cfg: dict | None) -> dict:
    """Return a compact MJLab-style MLP model configuration."""
    return {
        "class_name": "MLPModel",
        "hidden_dims": [8],
        "activation": "elu",
        "cnn_cfg": None,
        "distribution_cfg": distribution_cfg,
        "rnn_type": None,
        "rnn_hidden_dim": 32,
        "rnn_num_layers": 1,
    }


def _ppo_config() -> dict:
    """Return stock PPO options for each sequential policy stage."""
    return {
        "class_name": "PPO",
        "num_learning_epochs": 1,
        "num_mini_batches": 2,
        "schedule": "fixed",
        "normalize_advantage_per_mini_batch": False,
        "share_cnn_encoders": False,
    }


def _make_config(*, transformed_message: bool = False) -> dict:
    """Return the public sequential policy configuration."""
    cfg = {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "algorithm": {"class_name": "SequentialMultiPolicyPPO"},
        "policies": {
            "contact": {
                "num_actions": CONTACT_ACTIONS,
                "obs_groups": {"actor": ("contact_actor",), "critic": ("contact_critic",)},
                "actor": _model_config({
                    "class_name": "CategoricalDistribution",
                    "num_categories": 2,
                }),
                "critic": _model_config(None),
                "algorithm": _ppo_config(),
            },
            "joint": {
                "num_actions": JOINT_ACTIONS,
                "obs_groups": {"actor": ("joint_actor",), "critic": ("joint_critic",)},
                "actor": _model_config({
                    "class_name": "GaussianDistribution",
                    "init_std": 0.5,
                }),
                "critic": _model_config(None),
                "algorithm": _ppo_config(),
            },
        },
    }
    if transformed_message:
        cfg["policy_messages"] = {
            "contact": {
                "dim": 2 * CONTACT_ACTIONS,
                "func": _contact_message,
            }
        }
    return cfg


def _build_runner() -> OnPolicyRunner:
    """Construct the sequential algorithm through the unmodified stock runner."""
    return OnPolicyRunner(SequentialPolicyEnv(), _make_config(), log_dir=None, device="cpu")


def test_stock_runner_updates_sequential_policies() -> None:
    """The stock runner should complete an update without changing its lifecycle."""
    runner = _build_runner()

    assert isinstance(runner.alg, SequentialMultiPolicyPPO)
    assert isinstance(runner.alg, MultiPolicyPPO)
    assert runner.alg.policy_names == ("contact", "joint")
    assert runner.alg.policy_dependencies == ((), ("contact",))

    runner.learn(num_learning_iterations=1)

    assert all(algorithm.storage.step == 0 for algorithm in runner.alg.algorithms)


def test_joint_policy_stores_same_timestep_contact_action() -> None:
    """The joint PPO should train on the exact contact sample that conditioned its action."""
    runner = _build_runner()
    observations = runner.env.get_observations()

    actions = runner.alg.act(observations)
    dependency_key = runner.alg.algorithms[1].actor.obs_groups[-1]

    assert dependency_key == "_sequential_multi_policy_contact_action"
    assert runner.alg.algorithms[1].actor.obs_dim == 4 + CONTACT_ACTIONS
    assert torch.all((actions[:, :CONTACT_ACTIONS] == 0.0) | (actions[:, :CONTACT_ACTIONS] == 1.0))
    torch.testing.assert_close(
        runner.alg.algorithms[1].transition.observations[dependency_key],
        actions[:, :CONTACT_ACTIONS],
    )

    next_observations, rewards, dones, extras = runner.env.step(actions)
    runner.alg.process_env_step(next_observations, rewards, dones, extras)
    torch.testing.assert_close(
        runner.alg.algorithms[1].storage.observations[0][dependency_key],
        actions[:, :CONTACT_ACTIONS],
    )


def test_joint_policy_stores_transformed_same_timestep_message() -> None:
    """A pure policy message should not replace the sampled environment action."""
    runner = OnPolicyRunner(
        SequentialPolicyEnv(),
        _make_config(transformed_message=True),
        log_dir=None,
        device="cpu",
    )
    observations = runner.env.get_observations()

    actions = runner.alg.act(observations)
    dependency_key = runner.alg.algorithms[1].actor.obs_groups[-1]
    contact_action = actions[:, :CONTACT_ACTIONS]
    expected_message = _contact_message(observations, contact_action)

    assert runner.alg.algorithms[1].actor.obs_dim == 4 + 2 * CONTACT_ACTIONS
    torch.testing.assert_close(
        runner.alg.algorithms[0].transition.actions,
        contact_action,
    )
    torch.testing.assert_close(
        runner.alg.algorithms[1].transition.observations[dependency_key],
        expected_message,
    )

    next_observations, rewards, dones, extras = runner.env.step(actions)
    runner.alg.process_env_step(next_observations, rewards, dones, extras)
    torch.testing.assert_close(
        runner.alg.algorithms[1].storage.observations[0][dependency_key],
        expected_message,
    )


def test_deterministic_inference_uses_transformed_message(tmp_path: Path) -> None:
    """Inference and checkpoint restore should preserve the message transform."""
    runner = OnPolicyRunner(
        SequentialPolicyEnv(),
        _make_config(transformed_message=True),
        log_dir=None,
        device="cpu",
    )
    observations = runner.env.get_observations()
    dependency_key = runner.alg.algorithms[1].actor.obs_groups[-1]

    contact_action = runner.alg.algorithms[0].get_policy()(observations.select(*runner.alg.observation_keys[0]))
    joint_observations = observations.select(*runner.alg.observation_keys[1])
    joint_observations.set(dependency_key, _contact_message(observations, contact_action))
    joint_action = runner.alg.algorithms[1].get_policy()(joint_observations)

    expected_actions = torch.cat((contact_action, joint_action), dim=-1)
    torch.testing.assert_close(runner.get_inference_policy()(observations), expected_actions)

    checkpoint_path = tmp_path / "transformed_sequential_multi_policy.pt"
    runner.save(str(checkpoint_path))
    restored = OnPolicyRunner(
        SequentialPolicyEnv(),
        _make_config(transformed_message=True),
        log_dir=None,
        device="cpu",
    )
    restored.load(str(checkpoint_path))
    torch.testing.assert_close(restored.get_inference_policy()(observations), expected_actions)

    mismatched_cfg = _make_config(transformed_message=True)
    mismatched_cfg["policy_messages"]["contact"]["func"] = _alternate_contact_message
    mismatched = OnPolicyRunner(SequentialPolicyEnv(), mismatched_cfg, log_dir=None, device="cpu")
    with pytest.raises(ValueError, match="policy messages"):
        mismatched.load(str(checkpoint_path))


def test_message_shape_is_validated_before_environment_step() -> None:
    """A declared message width must match the transform output exactly."""
    cfg = _make_config(transformed_message=True)
    cfg["policy_messages"]["contact"]["dim"] = CONTACT_ACTIONS
    runner = OnPolicyRunner(SequentialPolicyEnv(), cfg, log_dir=None, device="cpu")

    with pytest.raises(ValueError, match="message must have shape"):
        runner.alg.act(runner.env.get_observations())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for policy-message device validation.")
def test_transformed_message_completes_one_cuda_update() -> None:
    """The transform should preserve the learner device through one stock-runner update."""
    runner = OnPolicyRunner(
        SequentialPolicyEnv(),
        _make_config(transformed_message=True),
        log_dir=None,
        device="cuda",
    )

    runner.learn(num_learning_iterations=1)

    assert all(torch.device(algorithm.storage.device).type == "cuda" for algorithm in runner.alg.algorithms)


def test_deterministic_inference_is_sequential_and_survives_checkpoint(tmp_path: Path) -> None:
    """Inference should use contact argmax before joint mean and survive save/load."""
    runner = _build_runner()
    observations = runner.env.get_observations()
    dependency_key = runner.alg.algorithms[1].actor.obs_groups[-1]

    contact_action = runner.alg.algorithms[0].get_policy()(observations.select(*runner.alg.observation_keys[0]))
    joint_observations = observations.select(*runner.alg.observation_keys[1])
    joint_observations.set(dependency_key, contact_action)
    joint_action = runner.alg.algorithms[1].get_policy()(joint_observations)
    expected_actions = torch.cat((contact_action, joint_action), dim=-1)

    actual_actions = runner.get_inference_policy()(observations)
    torch.testing.assert_close(actual_actions, expected_actions)

    checkpoint_path = tmp_path / "sequential_multi_policy.pt"
    runner.save(str(checkpoint_path))
    restored = _build_runner()
    restored.load(str(checkpoint_path))
    torch.testing.assert_close(restored.get_inference_policy()(observations), expected_actions)


def test_rollout_step_requires_a_preceding_action_sample() -> None:
    """The sequential transition cannot be stored before dependency actions exist."""
    runner = _build_runner()
    observations = runner.env.get_observations()

    with pytest.raises(RuntimeError, match=r"act\(\) must be called"):
        runner.alg.process_env_step(
            observations,
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {
                "policy_rewards": {
                    "contact": torch.zeros(NUM_ENVS),
                    "joint": torch.zeros(NUM_ENVS),
                }
            },
        )
