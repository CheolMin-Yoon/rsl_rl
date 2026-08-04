# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for fixed-weight multi-critic PPO."""

import copy
import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict
from unittest.mock import patch

import pytest

from rsl_rl.algorithms import OBJECTIVE_REWARDS_KEY, MultiCriticPPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import MultiCriticRolloutStorage
from tests.conftest import make_obs

OBJECTIVE_NAMES = ("locomotion", "manipulation", "tracking")
REWARD_GROUP_WEIGHTS = torch.tensor([0.5, 1.5, 2.0])
NUM_ENVS = 4
NUM_STEPS = 3
NUM_ACTIONS = 2


def _build_algorithm(
    reward_group_weights: torch.Tensor = REWARD_GROUP_WEIGHTS,
    multi_gpu_cfg: dict | None = None,
    **algorithm_kwargs: object,
) -> tuple[MultiCriticPPO, TensorDict]:
    obs = make_obs(NUM_ENVS, obs_dim=5)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[8],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    )
    critics = {name: MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[8]) for name in OBJECTIVE_NAMES}
    storage = MultiCriticRolloutStorage(
        OBJECTIVE_NAMES,
        NUM_ENVS,
        NUM_STEPS,
        obs,
        [NUM_ACTIONS],
    )
    algorithm = MultiCriticPPO(
        actor,
        critics,
        storage,
        reward_group_weights,
        num_learning_epochs=1,
        num_mini_batches=1,
        schedule="fixed",
        multi_gpu_cfg=multi_gpu_cfg,
        **algorithm_kwargs,
    )
    return algorithm, obs


def _collect_rollout(algorithm: MultiCriticPPO, obs: TensorDict) -> None:
    for step in range(NUM_STEPS):
        algorithm.act(obs)
        env_index = torch.arange(NUM_ENVS, dtype=torch.float32)
        objective_rewards = torch.stack(
            (
                0.1 * env_index + step,
                -0.2 * env_index + 0.5 * step,
                0.3 * env_index - step,
            ),
            dim=-1,
        )
        algorithm.process_env_step(
            obs,
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {OBJECTIVE_REWARDS_KEY: objective_rewards},
        )
    algorithm.compute_returns(obs)


def test_update_uses_one_global_clip_and_means_value_loss_over_objectives() -> None:
    """Actor and critics share samples, one optimizer step, and one global norm clip."""
    algorithm, obs = _build_algorithm()
    _collect_rollout(algorithm, obs)

    batch = next(algorithm.storage.mini_batch_generator(1, 1))
    assert batch.values.shape[-1] == len(OBJECTIVE_NAMES)
    assert batch.returns.shape == batch.values.shape
    actor_before = copy.deepcopy(algorithm.actor.state_dict())
    critics_before = copy.deepcopy(algorithm.critics.state_dict())

    with patch(
        "rsl_rl.algorithms.multi_critic_ppo.nn.utils.clip_grad_norm_",
        wraps=nn.utils.clip_grad_norm_,
    ) as clip_grad_norm:
        losses = algorithm.update()

    assert clip_grad_norm.call_count == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    objective_mean = sum(losses[f"value/{name}"] for name in OBJECTIVE_NAMES) / len(OBJECTIVE_NAMES)
    assert losses["value"] == pytest.approx(objective_mean)
    assert algorithm.storage.step == 0
    assert any(not torch.equal(actor_before[name], value) for name, value in algorithm.actor.state_dict().items())
    assert any(not torch.equal(critics_before[name], value) for name, value in algorithm.critics.state_dict().items())


def test_timeout_bootstraps_each_objective_reward_from_its_critic() -> None:
    """The stable extras vector is stored and timeout bootstrapping stays objective-wise."""
    algorithm, obs = _build_algorithm()
    algorithm.act(obs)
    values = algorithm.transition.values.clone()
    objective_rewards = torch.arange(NUM_ENVS * len(OBJECTIVE_NAMES), dtype=torch.float32).reshape(
        NUM_ENVS, len(OBJECTIVE_NAMES)
    )
    time_outs = torch.tensor([1.0, 0.0, 1.0, 0.0])

    algorithm.process_env_step(
        obs,
        torch.full((NUM_ENVS,), -123.0),
        torch.ones(NUM_ENVS),
        {OBJECTIVE_REWARDS_KEY: objective_rewards, "time_outs": time_outs},
    )

    expected = objective_rewards + algorithm.gamma * values * time_outs.view(-1, 1)
    torch.testing.assert_close(algorithm.storage.rewards[0], expected)


def test_objective_reward_extras_require_exact_env_and_objective_axes() -> None:
    """Near-miss extras tensors cannot silently broadcast or reorder the objective axis."""
    algorithm, obs = _build_algorithm()
    algorithm.act(obs)

    with pytest.raises(ValueError, match=r"objective_rewards.*shape"):
        algorithm.process_env_step(
            obs,
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {OBJECTIVE_REWARDS_KEY: torch.zeros(NUM_ENVS, len(OBJECTIVE_NAMES) - 1)},
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_objective_reward_extras_reject_non_finite_values(non_finite: float) -> None:
    """The authoritative reward vector cannot bypass the runner's scalar reward checks."""
    algorithm, obs = _build_algorithm()
    algorithm.act(obs)
    objective_rewards = torch.zeros(NUM_ENVS, len(OBJECTIVE_NAMES))
    objective_rewards[0, 0] = non_finite

    with pytest.raises(ValueError, match="only finite values"):
        algorithm.process_env_step(
            obs,
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {OBJECTIVE_REWARDS_KEY: objective_rewards},
        )


def test_checkpoint_requires_exact_objective_order_and_fixed_weights() -> None:
    """A checkpoint cannot be resumed under a different objective identity or weighting."""
    algorithm, _ = _build_algorithm()
    checkpoint = algorithm.save()
    target, _ = _build_algorithm()
    assert target.load(checkpoint, None, True)

    wrong_weights = copy.deepcopy(checkpoint)
    wrong_weights["reward_group_weights"] = torch.ones(len(OBJECTIVE_NAMES))
    with pytest.raises(ValueError, match="reward_group_weights"):
        target.load(wrong_weights, None, True)

    wrong_order = copy.deepcopy(checkpoint)
    wrong_order["objective_names"] = tuple(reversed(OBJECTIVE_NAMES))
    with pytest.raises(ValueError, match="objectives"):
        target.load(wrong_order, None, True)


@pytest.mark.parametrize("shape", [(3, 1), (4, 3), (2,)])
def test_broadcastable_or_wrong_weight_shapes_are_rejected(shape: tuple[int, ...]) -> None:
    """Weights are one immutable objective vector, never an env/time-dependent tensor."""
    with pytest.raises(ValueError, match="fixed shape"):
        _build_algorithm(torch.ones(shape))


def test_non_finite_weights_are_rejected() -> None:
    """Checkpoint identity cannot contain non-finite objective weights."""
    with pytest.raises(ValueError, match="finite"):
        _build_algorithm(torch.tensor([0.5, float("nan"), 2.0]))


@pytest.mark.parametrize(
    ("algorithm_kwargs", "message"),
    [
        ({"normalize_advantage_per_mini_batch": True}, "complete rollout"),
        ({"rnd_cfg": {}}, "RND"),
        ({"symmetry_cfg": {}}, "symmetry"),
    ],
)
def test_ambiguous_stock_extensions_are_rejected_explicitly(algorithm_kwargs: dict[str, object], message: str) -> None:
    """Unsupported stock features fail during construction rather than degrading semantics."""
    with pytest.raises(ValueError, match=message):
        _build_algorithm(**algorithm_kwargs)


def test_multi_gpu_reduction_averages_one_actor_and_all_critic_gradients() -> None:
    """Distributed reduction covers the same global parameter collection as clipping."""
    algorithm, _ = _build_algorithm(multi_gpu_cfg={"global_rank": 0, "world_size": 2})
    parameters = tuple(chain(algorithm.actor.parameters(), algorithm.critics.parameters()))
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 2.0)

    with patch("torch.distributed.all_reduce") as all_reduce:
        algorithm.reduce_parameters()

    all_reduce.assert_called_once()
    for parameter in parameters:
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


def test_multi_gpu_broadcast_includes_actor_and_every_ordered_critic() -> None:
    """Runner synchronization broadcasts one state dictionary per owned model."""
    algorithm, _ = _build_algorithm(multi_gpu_cfg={"global_rank": 1, "world_size": 2})

    with patch("torch.distributed.broadcast_object_list") as broadcast_object_list:
        algorithm.broadcast_parameters()

    broadcast_object_list.assert_called_once()
    model_states = broadcast_object_list.call_args.args[0]
    assert len(model_states) == 1 + len(OBJECTIVE_NAMES)


def test_compile_keeps_raw_actor_for_policy_export_and_compiles_every_model() -> None:
    """Compilation wraps training models without changing checkpoint/export ownership."""
    algorithm, _ = _build_algorithm()
    raw_actor = algorithm.actor

    with patch(
        "rsl_rl.algorithms.multi_critic_ppo.compile_model",
        side_effect=lambda model, _mode: model,
    ) as compile_model:
        algorithm.compile("default")

    assert compile_model.call_count == 1 + len(OBJECTIVE_NAMES)
    assert algorithm.get_policy() is raw_actor
