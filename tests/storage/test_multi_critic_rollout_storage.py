# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for multi-critic rollout storage."""

import torch

import pytest

from rsl_rl.storage import MultiCriticRolloutStorage, RolloutStorage
from tests.conftest import make_obs


def test_objective_names_reject_one_string() -> None:
    """A name string is not silently expanded into character objectives."""
    obs = make_obs(num_envs=1, obs_dim=3)

    with pytest.raises(ValueError, match="not one string"):
        MultiCriticRolloutStorage("task", 1, 1, obs, [1])  # type: ignore[arg-type]


def test_complete_rollout_normalizes_each_objective_before_fixed_weighting() -> None:
    """Each objective is normalized over the complete rollout before actor aggregation."""
    objective_names = ("locomotion", "manipulation")
    obs = make_obs(num_envs=2, obs_dim=3)
    storage = MultiCriticRolloutStorage(objective_names, 2, 2, obs, [1])

    rewards = (
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        torch.tensor([[3.0, 30.0], [4.0, 40.0]]),
    )
    for reward in rewards:
        transition = RolloutStorage.Transition()
        transition.observations = obs
        transition.actions = torch.zeros(2, 1)
        transition.rewards = reward
        transition.dones = torch.zeros(2)
        transition.values = torch.zeros(2, 2)
        transition.actions_log_prob = torch.zeros(2)
        transition.distribution_params = (torch.zeros(2, 1), torch.ones(2, 1))
        storage.add_transition(transition)

    weights = torch.tensor([0.5, 2.0])
    storage.compute_returns(torch.zeros(2, 2), gamma=0.0, lam=0.0, reward_group_weights=weights)

    expected_normalized = torch.tensor([
        [[-1.161895, -1.161895], [-0.387298, -0.387298]],
        [[0.387298, 0.387298], [1.161895, 1.161895]],
    ])
    torch.testing.assert_close(storage.returns, torch.stack(rewards))
    torch.testing.assert_close(storage.advantages, expected_normalized, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(
        storage.actor_advantages,
        expected_normalized.mul(weights).sum(dim=-1, keepdim=True),
    )


def test_vector_gae_matches_a_hand_computed_recursive_example() -> None:
    """Every objective follows its own vectorized GAE recursion."""
    obs = make_obs(num_envs=1, obs_dim=3)
    storage = MultiCriticRolloutStorage(("task", "style"), 1, 3, obs, [1])
    rewards = ([1.0, 10.0], [2.0, 20.0], [3.0, 30.0])
    values = ([0.5, 5.0], [1.0, 10.0], [1.5, 15.0])

    for reward, value in zip(rewards, values):
        transition = RolloutStorage.Transition()
        transition.observations = obs
        transition.actions = torch.zeros(1, 1)
        transition.rewards = torch.tensor([reward])
        transition.dones = torch.zeros(1)
        transition.values = torch.tensor([value])
        transition.actions_log_prob = torch.zeros(1)
        transition.distribution_params = (torch.zeros(1, 1), torch.ones(1, 1))
        storage.add_transition(transition)

    storage.compute_returns(
        last_values=torch.tensor([[2.0, 20.0]]),
        gamma=0.9,
        lam=0.8,
        reward_group_weights=torch.ones(2),
    )

    expected_returns = torch.tensor([[[5.30272, 53.0272]], [[5.726, 57.26]], [[4.8, 48.0]]])
    torch.testing.assert_close(storage.returns, expected_returns)


def test_single_sample_rollout_has_finite_zero_normalized_advantage() -> None:
    """A degenerate complete rollout remains finite instead of producing an undefined sample standard deviation."""
    obs = make_obs(num_envs=1, obs_dim=3)
    storage = MultiCriticRolloutStorage(("task",), 1, 1, obs, [1])
    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.actions = torch.zeros(1, 1)
    transition.rewards = torch.tensor([[2.0]])
    transition.dones = torch.zeros(1)
    transition.values = torch.zeros(1, 1)
    transition.actions_log_prob = torch.zeros(1)
    transition.distribution_params = (torch.zeros(1, 1), torch.ones(1, 1))
    storage.add_transition(transition)

    storage.compute_returns(torch.zeros(1, 1), gamma=0.99, lam=0.95, reward_group_weights=torch.ones(1))

    torch.testing.assert_close(storage.advantages, torch.zeros_like(storage.advantages))
    torch.testing.assert_close(storage.actor_advantages, torch.zeros_like(storage.actor_advantages))
