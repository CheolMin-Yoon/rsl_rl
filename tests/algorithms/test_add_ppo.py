# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RSL-native PPO with an ADD auxiliary."""

from __future__ import annotations

import copy
import math
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import ADDPPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 2
NUM_STEPS = 4
OBS_DIM = 5
ACTION_DIM = 2
DISC_OBS_DIM = 3


def _observation(device: str = "cpu") -> TensorDict:
    return TensorDict(
        {"policy": torch.randn(NUM_ENVS, OBS_DIM, device=device)},
        batch_size=[NUM_ENVS],
        device=device,
    )


def _make_algorithm(
    *,
    task_reward_weight: float = 0.0,
    add_reward_weight: float = 1.0,
    add_cfg_overrides: dict | None = None,
    device: str = "cpu",
    **algorithm_overrides: object,
) -> ADDPPO:
    observation = _observation(device)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        observation,
        obs_groups,
        "actor",
        ACTION_DIM,
        hidden_dims=(16,),
        activation="elu",
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.2,
            "std_type": "scalar",
        },
    )
    critic = MLPModel(
        observation,
        obs_groups,
        "critic",
        1,
        hidden_dims=(16,),
        activation="elu",
    )
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, observation, [ACTION_DIM], device=device)
    add_cfg = {
        "disc_obs_dim": DISC_OBS_DIM,
        "hidden_dims": (8,),
        "disc_epochs": 1,
        "disc_batch_size": 2.0,
        "disc_buffer_size": 16,
        "disc_replay_samples": 2,
        "normalizer_samples": 100,
    }
    if add_cfg_overrides:
        add_cfg.update(add_cfg_overrides)
    algorithm_cfg = {
        "num_learning_epochs": 1,
        "num_mini_batches": 2,
        "entropy_coef": 0.0,
        "learning_rate": 1.0e-3,
        "schedule": "fixed",
        "desired_kl": None,
    }
    algorithm_cfg.update(algorithm_overrides)
    return ADDPPO(
        actor,
        critic,
        storage,
        add_cfg=add_cfg,
        task_reward_weight=task_reward_weight,
        add_reward_weight=add_reward_weight,
        device=device,
        **algorithm_cfg,
    )


def _extras(offset: float = 0.5, device: str = "cpu") -> dict[str, torch.Tensor]:
    live = torch.randn(NUM_ENVS, DISC_OBS_DIM, device=device)
    return {
        "add_live_obs": live,
        "add_reference_obs": live + offset,
        "time_outs": torch.zeros(NUM_ENVS, device=device),
    }


def _collect_rollout(algorithm: ADDPPO, observation: TensorDict | None = None) -> TensorDict:
    obs = _observation() if observation is None else observation
    for _ in range(NUM_STEPS):
        algorithm.act(obs)
        next_obs = _observation()
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        algorithm.process_env_step(next_obs, rewards, dones, _extras())
        obs = next_obs
    algorithm.compute_returns(obs)
    return obs


def test_process_env_step_mixes_learning_reward_without_mutating_logger_reward() -> None:
    """ADDPPO must train on the mixture while leaving stock logger reward intact."""
    algorithm = _make_algorithm(task_reward_weight=0.25, add_reward_weight=0.75)
    with torch.no_grad():
        for parameter in algorithm.add_discriminator.discriminator.parameters():
            parameter.zero_()
    observation = _observation()
    algorithm.act(observation)
    rewards = torch.full((NUM_ENVS,), 2.0)
    original_rewards = rewards.clone()

    algorithm.process_env_step(
        observation,
        rewards,
        torch.zeros(NUM_ENVS),
        _extras(),
    )

    expected = 0.25 * 2.0 + 0.75 * 2.0 * math.log(2.0)
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.full((NUM_ENVS,), expected))
    torch.testing.assert_close(rewards, original_rewards, rtol=0.0, atol=0.0)


def test_complete_rollout_updates_stock_ppo_and_add_discriminator() -> None:
    """A complete rollout must update both stock PPO and the ADD auxiliary."""
    torch.manual_seed(11)
    algorithm = _make_algorithm()
    policy_before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    discriminator_before = [
        parameter.detach().clone() for parameter in algorithm.add_discriminator.discriminator.parameters()
    ]
    _collect_rollout(algorithm)

    metrics = algorithm.update()

    assert any(not torch.equal(before, after) for before, after in zip(policy_before, algorithm.actor.parameters()))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(discriminator_before, algorithm.add_discriminator.discriminator.parameters())
    )
    assert algorithm.storage.step == 0
    assert algorithm.add_discriminator.at_update_boundary
    for key in ("value", "surrogate", "add/loss", "add/reward_mean", "add/replay_size"):
        assert key in metrics
        assert math.isfinite(metrics[key])


@pytest.mark.parametrize("missing_key", ["add_live_obs", "add_reference_obs"])
def test_missing_add_payload_is_rejected(missing_key: str) -> None:
    """Both named halves of the transition-aligned pair are mandatory."""
    algorithm = _make_algorithm()
    observation = _observation()
    algorithm.act(observation)
    extras = _extras()
    del extras[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        algorithm.process_env_step(observation, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), extras)


def test_non_finite_and_wrong_dtype_add_payloads_are_rejected() -> None:
    """ADD pair tensors must be finite float32 values."""
    for invalid in (
        torch.full((NUM_ENVS, DISC_OBS_DIM), float("nan")),
        torch.zeros(NUM_ENVS, DISC_OBS_DIM, dtype=torch.float64),
    ):
        algorithm = _make_algorithm()
        observation = _observation()
        algorithm.act(observation)
        extras = _extras()
        extras["add_live_obs"] = invalid
        with pytest.raises(ValueError):
            algorithm.process_env_step(observation, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), extras)


def test_checkpoint_round_trip_restores_policy_add_reward_and_replay() -> None:
    """A full checkpoint must restore inference plus all ADD learning state."""
    torch.manual_seed(13)
    algorithm = _make_algorithm()
    probe = _observation()
    _collect_rollout(algorithm)
    algorithm.update()
    algorithm.eval_mode()
    expected_action = algorithm.get_policy()(probe)
    live = torch.randn(NUM_ENVS, DISC_OBS_DIM)
    reference = torch.randn(NUM_ENVS, DISC_OBS_DIM)
    expected_reward = algorithm.add_discriminator.reward(live, reference)
    checkpoint = copy.deepcopy(algorithm.save())

    restored = _make_algorithm()
    assert restored.load(checkpoint, load_cfg=None, strict=True)
    restored.eval_mode()

    torch.testing.assert_close(restored.get_policy()(probe), expected_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(restored.add_discriminator.reward(live, reference), expected_reward, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored.add_discriminator.replay.pairs,
        algorithm.add_discriminator.replay.pairs,
        rtol=0.0,
        atol=0.0,
    )


def test_stock_actor_only_load_does_not_require_add_checkpoint_keys() -> None:
    """MJLab's stock ``{"actor": True}`` play load must skip all ADD state."""
    source = _make_algorithm()
    target = _make_algorithm()
    stock_checkpoint = source.save()
    stock_checkpoint.pop("add_checkpoint_identity")
    stock_checkpoint.pop("add_state_dict")
    stock_checkpoint.pop("add_optimizer_state_dict")

    target.load(
        stock_checkpoint,
        load_cfg={"actor": True},
        strict=True,
    )

    for key, value in source.actor.state_dict().items():
        torch.testing.assert_close(target.actor.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_partial_load_restores_add_only_when_explicitly_enabled() -> None:
    """A partial stock load may opt into ADD without changing full-resume semantics."""
    source = _make_algorithm()
    with torch.no_grad():
        for parameter in source.add_discriminator.discriminator.parameters():
            parameter.zero_()
    checkpoint = source.save()
    target = _make_algorithm()
    live = torch.randn(NUM_ENVS, DISC_OBS_DIM)
    reference = torch.randn(NUM_ENVS, DISC_OBS_DIM)

    target.load(
        checkpoint,
        load_cfg={"actor": True, "add": True},
        strict=True,
    )

    torch.testing.assert_close(
        target.add_discriminator.reward(live, reference),
        source.add_discriminator.reward(live, reference),
        rtol=0.0,
        atol=0.0,
    )


def test_checkpoint_save_requires_an_update_boundary() -> None:
    """Partial ADD rollouts must not be serialized as resumable checkpoints."""
    algorithm = _make_algorithm()
    observation = _observation()
    algorithm.act(observation)
    algorithm.process_env_step(observation, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), _extras())

    with pytest.raises(RuntimeError, match="update boundary"):
        algorithm.save()


def test_checkpoint_identity_rejects_different_reward_weights() -> None:
    """Reward mixture settings are part of ADD checkpoint identity."""
    source = _make_algorithm()
    checkpoint = source.save()
    target = _make_algorithm(task_reward_weight=0.5, add_reward_weight=0.5)

    with pytest.raises(RuntimeError, match="identity"):
        target.load(checkpoint, load_cfg=None, strict=True)


def test_checkpoint_resume_ignores_operational_add_runtime_options() -> None:
    """Eager/check settings are operational and may change across a full resume."""
    source = _make_algorithm(add_cfg_overrides={"runtime_backend": "eager", "runtime_finite_checks": True})
    checkpoint = copy.deepcopy(source.save())
    target = _make_algorithm(add_cfg_overrides={"runtime_backend": "auto", "runtime_finite_checks": False})

    assert target.load(checkpoint, load_cfg=None, strict=True)


def test_disabled_runtime_checks_keep_structural_validation_only() -> None:
    """The production switch removes content scans but retains shape and dtype checks."""
    algorithm = _make_algorithm(add_cfg_overrides={"runtime_finite_checks": False})
    observation = _observation()
    algorithm.act(observation)
    extras = _extras()
    extras["add_live_obs"] = torch.full((NUM_ENVS, DISC_OBS_DIM), float("nan"))
    algorithm.process_env_step(observation, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), extras)
    assert torch.isnan(algorithm.storage.rewards[0]).all()

    invalid_algorithm = _make_algorithm(add_cfg_overrides={"runtime_finite_checks": False})
    invalid_algorithm.act(observation)
    invalid_extras = _extras()
    invalid_extras["add_live_obs"] = torch.zeros(NUM_ENVS, DISC_OBS_DIM, dtype=torch.float64)
    with pytest.raises(ValueError, match="float32"):
        invalid_algorithm.process_env_step(
            observation,
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            invalid_extras,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reward storage parity requires CUDA")
def test_cuda_reward_graph_matches_eager_ppo_stored_reward_bitwise() -> None:
    """Operational reward replay must not change the scalar stored by stock PPO."""
    torch.manual_seed(29)
    eager = _make_algorithm(
        task_reward_weight=0.25,
        add_reward_weight=0.75,
        add_cfg_overrides={"runtime_backend": "eager", "runtime_finite_checks": False},
        device="cuda:0",
    )
    checkpoint = copy.deepcopy(eager.save())
    graph = _make_algorithm(
        task_reward_weight=0.25,
        add_reward_weight=0.75,
        add_cfg_overrides={"runtime_backend": "cuda_graph", "runtime_finite_checks": False},
        device="cuda:0",
    )
    graph.load(checkpoint, load_cfg=None, strict=True)

    observation = _observation("cuda:0")
    rewards = torch.randn(NUM_ENVS, device="cuda:0")
    dones = torch.zeros(NUM_ENVS, device="cuda:0")
    extras = _extras(device="cuda:0")
    eager.act(observation)
    graph.act(observation)
    eager.process_env_step(observation, rewards, dones, extras)
    graph.process_env_step(observation, rewards, dones, extras)

    assert torch.equal(graph.add_discriminator._reward_rollout[0], eager.add_discriminator._reward_rollout[0])
    assert torch.equal(graph.storage.rewards[0], eager.storage.rewards[0])


def test_unsupported_extensions_compile_and_multi_gpu_are_rejected() -> None:
    """Unimplemented lifecycle combinations must fail during construction or setup."""
    with pytest.raises(ValueError, match="RND"):
        _make_algorithm(rnd_cfg={"num_states": 1})
    with pytest.raises(ValueError, match="symmetry"):
        _make_algorithm(symmetry_cfg={"use_data_augmentation": False})
    with pytest.raises(ValueError, match="multi-GPU"):
        _make_algorithm(multi_gpu_cfg={"global_rank": 0, "world_size": 2})

    algorithm = _make_algorithm()
    with pytest.raises(ValueError, match=r"torch\.compile"):
        algorithm.compile("default")
    with pytest.raises(RuntimeError, match="multi-GPU"):
        algorithm.broadcast_parameters()
