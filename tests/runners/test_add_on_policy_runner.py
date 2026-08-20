# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stock OnPolicyRunner integration test for ADDPPO."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import ADDPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 4
NUM_STEPS = 4
OBS_DIM = 6
ACTION_DIM = 2
DISC_OBS_DIM = 3


class DummyADDEnv(VecEnv):
    """Minimal environment publishing transition-aligned ADD pairs."""

    def __init__(self) -> None:
        """Initialize fixed-shape CPU state for the runner smoke test."""
        self.num_envs = NUM_ENVS
        self.num_actions = ACTION_DIM
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}

    @property
    def unwrapped(self) -> DummyADDEnv:
        """Expose the stock VecEnv unwrapped interface."""
        return self

    def get_observations(self) -> TensorDict:
        """Return one flat actor/critic observation group."""
        return TensorDict(
            {"policy": torch.randn(NUM_ENVS, OBS_DIM)},
            batch_size=[NUM_ENVS],
            device="cpu",
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance one synthetic transition and publish its ADD pair."""
        assert actions.shape == (NUM_ENVS, ACTION_DIM)
        self.episode_length_buf += 1
        live = torch.randn(NUM_ENVS, DISC_OBS_DIM)
        return (
            self.get_observations(),
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {
                "add_live_obs": live,
                "add_reference_obs": live + 0.5,
                "time_outs": torch.zeros(NUM_ENVS),
            },
        )


def _train_cfg() -> dict:
    return {
        "num_steps_per_env": NUM_STEPS,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.2,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "ADDPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "entropy_coef": 0.0,
            "learning_rate": 1.0e-3,
            "schedule": "fixed",
            "desired_kl": None,
            "task_reward_weight": 0.0,
            "add_reward_weight": 1.0,
            "add_cfg": {
                "disc_obs_dim": DISC_OBS_DIM,
                "hidden_dims": [8],
                "disc_epochs": 1,
                "disc_batch_size": 2.0,
                "disc_buffer_size": 32,
                "disc_replay_samples": 4,
                "normalizer_samples": 100,
            },
        },
    }


def test_stock_runner_constructs_and_completes_one_add_update() -> None:
    """Unmodified OnPolicyRunner must construct ADDPPO and finish one update."""
    torch.manual_seed(17)
    runner = OnPolicyRunner(DummyADDEnv(), _train_cfg(), log_dir=None, device="cpu")
    assert isinstance(runner.alg, ADDPPO)
    actor_before = [parameter.detach().clone() for parameter in runner.alg.actor.parameters()]
    discriminator_before = [
        parameter.detach().clone() for parameter in runner.alg.add_discriminator.discriminator.parameters()
    ]

    runner.learn(num_learning_iterations=1)

    assert any(not torch.equal(before, after) for before, after in zip(actor_before, runner.alg.actor.parameters()))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(discriminator_before, runner.alg.add_discriminator.discriminator.parameters())
    )
    assert int(runner.alg.add_discriminator.policy_sample_count.item()) == NUM_ENVS * NUM_STEPS
    assert runner.alg.add_discriminator.at_update_boundary
