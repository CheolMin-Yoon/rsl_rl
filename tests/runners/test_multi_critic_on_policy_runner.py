# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for MultiCriticPPO through the stock OnPolicyRunner."""

import copy
import tempfile
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import MultiCriticPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

OBJECTIVE_NAMES = ("locomotion", "manipulation", "tracking")
REWARD_GROUP_WEIGHTS = (0.5, 1.5, 2.0)


class MultiCriticEnv(VecEnv):
    """Minimal vector environment exposing named objective rewards."""

    def __init__(self) -> None:
        """Initialize a CPU-only test environment."""
        self.num_envs = 4
        self.num_actions = 2
        self.max_episode_length = 20
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        """Return policy observations for every environment."""
        return TensorDict(
            {"policy": torch.randn(self.num_envs, 5)},
            batch_size=[self.num_envs],
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance one step and expose objective rewards in reverse name order."""
        self.episode_length_buf += 1
        observations = self.get_observations()
        locomotion_reward = 1.0 - actions.square().mean(dim=-1)
        manipulation_reward = actions[:, 0]
        tracking_reward = -actions[:, 1].abs()
        objective_rewards = {
            "tracking": tracking_reward,
            "manipulation": manipulation_reward,
            "locomotion": locomotion_reward,
        }
        rewards = locomotion_reward + manipulation_reward + tracking_reward
        dones = torch.zeros(self.num_envs)
        extras = {
            "objective_rewards": objective_rewards,
            "time_outs": torch.zeros(self.num_envs),
        }
        return observations, rewards, dones, extras


def _make_config() -> dict:
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "MultiCriticPPO",
            "objective_names": OBJECTIVE_NAMES,
            "reward_group_weights": REWARD_GROUP_WEIGHTS,
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "schedule": "fixed",
            "normalize_advantage_per_mini_batch": False,
            "share_cnn_encoders": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [8]},
    }


def test_stock_runner_constructs_learns_and_exports_multi_critic_policy() -> None:
    """The stock runner owns the complete lifecycle without a custom runner."""
    runner = OnPolicyRunner(MultiCriticEnv(), _make_config(), log_dir=None, device="cpu")

    assert isinstance(runner.alg, MultiCriticPPO)
    assert runner.alg.objective_names == OBJECTIVE_NAMES
    torch.testing.assert_close(runner.alg.reward_group_weights, torch.tensor(REWARD_GROUP_WEIGHTS))
    assert runner.cfg["algorithm"]["objective_names"] == OBJECTIVE_NAMES
    assert runner.cfg["algorithm"]["reward_group_weights"] == REWARD_GROUP_WEIGHTS

    actor_before = {name: parameter.clone() for name, parameter in runner.alg.actor.named_parameters()}
    runner.learn(num_learning_iterations=1)

    assert runner.alg.storage.step == 0
    assert any(
        not torch.equal(actor_before[name], parameter) for name, parameter in runner.alg.actor.named_parameters()
    )
    observations = runner.env.get_observations()
    assert runner.get_inference_policy()(observations).shape == (runner.env.num_envs, runner.env.num_actions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cross-device reward routing.")
def test_stock_runner_moves_cpu_objective_rewards_to_cuda_learner() -> None:
    """CPU objective rewards should reach CUDA storage without a custom runner."""
    runner = OnPolicyRunner(MultiCriticEnv(), _make_config(), log_dir=None, device="cuda")

    runner.learn(num_learning_iterations=1)

    assert torch.device(runner.alg.storage.device).type == "cuda"


def test_stock_runner_rejects_one_objective_name_string() -> None:
    """Runner configuration requires an ordered sequence rather than character splitting."""
    config = _make_config()
    config["algorithm"]["objective_names"] = "task"

    with pytest.raises(ValueError, match="not one string"):
        OnPolicyRunner(MultiCriticEnv(), config, log_dir=None, device="cpu")


def test_mjlab_style_config_without_optional_extension_keys_learns() -> None:
    """MJLab's base PPO dataclass omits RND and symmetry keys when unused."""
    config = _make_config()
    config["algorithm"].pop("rnd_cfg")
    config["algorithm"].pop("symmetry_cfg")
    runner = OnPolicyRunner(MultiCriticEnv(), config, log_dir=None, device="cpu")

    runner.learn(num_learning_iterations=1)

    assert runner.cfg["algorithm"]["rnd_cfg"] is None
    assert runner.cfg["algorithm"]["symmetry_cfg"] is None


def test_stock_runner_save_and_load_restores_actor_and_ordered_critics() -> None:
    """Runner checkpoints preserve all models and the objective identity used to train them."""
    runner = OnPolicyRunner(MultiCriticEnv(), _make_config(), log_dir=None, device="cpu")
    runner.learn(num_learning_iterations=1)
    actor_state = copy.deepcopy(runner.alg.get_policy().state_dict())
    critic_states = copy.deepcopy(runner.alg.critics.state_dict())

    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint_file:
        runner.save(checkpoint_file.name)
        checkpoint = torch.load(checkpoint_file.name, weights_only=False, map_location="cpu")
        assert tuple(checkpoint["objective_names"]) == OBJECTIVE_NAMES
        torch.testing.assert_close(checkpoint["reward_group_weights"], torch.tensor(REWARD_GROUP_WEIGHTS))

        restored = OnPolicyRunner(MultiCriticEnv(), _make_config(), log_dir=None, device="cpu")
        restored.load(checkpoint_file.name)

    for name, parameter in restored.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, actor_state[name])
    for name, parameter in restored.alg.critics.state_dict().items():
        torch.testing.assert_close(parameter, critic_states[name])
