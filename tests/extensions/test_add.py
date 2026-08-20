# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Adversarial Differential Discriminator extension."""

from __future__ import annotations

import copy
import math
import torch

import pytest

from rsl_rl.extensions.add import AdversarialDifferentialDiscriminator, DiffNormalizer

NUM_ENVS = 2
NUM_STEPS = 4
DISC_OBS_DIM = 3


def _make_add(**overrides: object) -> AdversarialDifferentialDiscriminator:
    config = {
        "disc_obs_dim": DISC_OBS_DIM,
        "num_envs": NUM_ENVS,
        "num_steps_per_env": NUM_STEPS,
        "hidden_dims": (8,),
        "disc_epochs": 1,
        "disc_batch_size": 2.0,
        "disc_buffer_size": 16,
        "disc_replay_samples": 2,
        "normalizer_samples": 100,
        "device": "cpu",
    }
    config.update(overrides)
    return AdversarialDifferentialDiscriminator(**config)


def _record_rollout(add: AdversarialDifferentialDiscriminator, offset: float = 0.5) -> None:
    for step in range(NUM_STEPS):
        live = torch.full((NUM_ENVS, DISC_OBS_DIM), float(step))
        reference = live + offset
        rewards = add.record_step(live, reference)
        assert rewards.shape == (NUM_ENVS,)
        assert torch.isfinite(rewards).all()


def test_diff_normalizer_preserves_zero_and_commits_pending_statistics() -> None:
    """Pending moments must not move ADD's exact zero positive class."""
    normalizer = DiffNormalizer(4, device="cpu")
    samples = torch.tensor([[-2.0, 2.0, 0.0, 4.0]])

    normalizer.record(samples)
    torch.testing.assert_close(normalizer.mean_abs, torch.ones(4))
    torch.testing.assert_close(normalizer.normalize(torch.zeros(1, 4)), torch.zeros(1, 4), rtol=0.0, atol=0.0)

    normalizer.update()
    torch.testing.assert_close(normalizer.mean_abs, torch.tensor([2.0, 2.0, 0.0, 4.0]))
    assert int(normalizer.count.item()) == 1
    assert int(normalizer.pending_count.item()) == 0
    assert torch.isfinite(normalizer.normalize(torch.ones(1, 4))).all()


def test_zero_logit_reward_matches_the_add_equation() -> None:
    """A zero logit must produce the source reward scale times log two."""
    add = _make_add()
    with torch.no_grad():
        for parameter in add.discriminator.parameters():
            parameter.zero_()
    live = torch.zeros(NUM_ENVS, DISC_OBS_DIM)
    reference = torch.ones_like(live)

    reward = add.reward(live, reference)
    expected = torch.full((NUM_ENVS,), 2.0 * math.log(2.0))
    torch.testing.assert_close(reward, expected)


def test_zero_discriminator_loss_matches_balanced_bce_oracle() -> None:
    """Zero weights reduce balanced positive/negative BCE to log two."""
    add = _make_add()
    with torch.no_grad():
        for parameter in add.discriminator.parameters():
            parameter.zero_()

    loss = add.compute_loss(torch.ones(4, DISC_OBS_DIM))

    torch.testing.assert_close(loss.total_loss, torch.tensor(math.log(2.0)))
    torch.testing.assert_close(loss.gradient_penalty, torch.tensor(0.0))
    torch.testing.assert_close(loss.logit_loss, torch.tensor(0.0))
    torch.testing.assert_close(loss.positive_accuracy, torch.tensor(0.0))
    torch.testing.assert_close(loss.negative_accuracy, torch.tensor(0.0))


def test_rollout_update_trains_discriminator_and_keeps_replay_pairs() -> None:
    """One update must train ADD, commit moments, and preserve pair identity."""
    torch.manual_seed(3)
    add = _make_add()
    parameters_before = [parameter.detach().clone() for parameter in add.discriminator.parameters()]
    _record_rollout(add, offset=0.5)

    assert int(add.diff_normalizer.count.item()) == 0
    assert int(add.diff_normalizer.pending_count.item()) == 0
    metrics = add.update()

    assert any(
        not torch.equal(before, after) for before, after in zip(parameters_before, add.discriminator.parameters())
    )
    assert add.at_update_boundary
    assert int(add.policy_sample_count.item()) == NUM_ENVS * NUM_STEPS
    assert int(add.diff_normalizer.count.item()) == NUM_ENVS * NUM_STEPS
    torch.testing.assert_close(add.diff_normalizer.mean_abs, torch.full((DISC_OBS_DIM,), 0.5))
    replay_pairs = add.replay.pairs[: add.replay.sample_count]
    torch.testing.assert_close(replay_pairs[:, 1] - replay_pairs[:, 0], torch.full_like(replay_pairs[:, 0], 0.5))
    assert metrics["updates"] == 2.0
    assert metrics["replay_size"] == float(NUM_ENVS * NUM_STEPS)


def test_full_replay_inserts_only_the_configured_subset_without_breaking_pairs() -> None:
    """A saturated replay accepts only the configured subset of later rollouts."""
    torch.manual_seed(5)
    add = _make_add()
    _record_rollout(add, offset=0.5)
    add.update()
    _record_rollout(add, offset=1.0)
    add.update()
    assert add.replay.is_full
    total_before = int(add.replay.total_samples.item())

    _record_rollout(add, offset=1.5)
    add.update()

    assert int(add.replay.total_samples.item()) == total_before + add.disc_replay_samples
    pair_differences = add.replay.pairs[:, 1] - add.replay.pairs[:, 0]
    allowed = torch.tensor([0.5, 1.0, 1.5])
    assert torch.isin(pair_differences, allowed).all()


def test_state_and_optimizer_round_trip_restore_reward_and_replay() -> None:
    """Module and optimizer states must restore ADD reward and replay exactly."""
    torch.manual_seed(7)
    add = _make_add()
    _record_rollout(add)
    add.update()
    state = copy.deepcopy(add.state_dict())
    optimizer_state = copy.deepcopy(add.optimizer.state_dict())

    clone = _make_add()
    clone.load_state_dict(state, strict=True)
    clone.optimizer.load_state_dict(optimizer_state)
    live = torch.randn(NUM_ENVS, DISC_OBS_DIM)
    reference = torch.randn(NUM_ENVS, DISC_OBS_DIM)

    torch.testing.assert_close(clone.reward(live, reference), add.reward(live, reference), rtol=0.0, atol=0.0)
    torch.testing.assert_close(clone.replay.pairs, add.replay.pairs, rtol=0.0, atol=0.0)
    assert int(clone.policy_sample_count.item()) == int(add.policy_sample_count.item())


def test_pair_validation_rejects_wrong_shape_dtype_and_non_finite_values() -> None:
    """ADD observations must obey the fixed float32 feature contract."""
    add = _make_add()
    valid = torch.zeros(NUM_ENVS, DISC_OBS_DIM)

    for invalid in (
        torch.zeros(NUM_ENVS, DISC_OBS_DIM + 1),
        valid.to(torch.float64),
        valid.clone().index_fill(1, torch.tensor([0]), float("nan")),
    ):
        with pytest.raises(ValueError):
            add.reward(valid, invalid)


def test_constructor_rejects_replay_smaller_than_one_rollout() -> None:
    """The source insertion lifecycle requires one rollout to fit in replay."""
    with pytest.raises(ValueError, match="complete rollout"):
        _make_add(disc_buffer_size=NUM_ENVS * NUM_STEPS - 1)


def test_runtime_backend_resolution_and_validation_on_cpu() -> None:
    """Auto remains eager on CPU while an explicit CUDA graph request fails loudly."""
    add = _make_add(runtime_backend="auto")
    assert add.runtime_backend_requested == "auto"
    assert add.runtime_backend == "eager"

    with pytest.raises(ValueError, match="requires a CUDA"):
        _make_add(runtime_backend="cuda_graph")


def test_runtime_checks_can_be_disabled_without_changing_checkpoint_identity() -> None:
    """Operational validation flags must not invalidate an otherwise identical resume."""
    source = _make_add(runtime_backend="eager", runtime_finite_checks=True)
    state = copy.deepcopy(source.state_dict())
    target = _make_add(runtime_backend="auto", runtime_finite_checks=False)

    target.load_state_dict(state, strict=True)

    invalid = torch.full((NUM_ENVS, DISC_OBS_DIM), float("nan"))
    reward = target.reward(invalid, torch.zeros_like(invalid))
    assert torch.isnan(reward).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity requires CUDA")
def test_cuda_reward_graph_is_bitwise_equal_and_does_not_alias_prior_rows() -> None:
    """Each fixed rollout graph must match eager and preserve previously returned rows."""
    torch.manual_seed(19)
    eager = _make_add(device="cuda", runtime_backend="eager")
    graph = _make_add(device="cuda", runtime_backend="cuda_graph")
    graph.load_state_dict(copy.deepcopy(eager.state_dict()), strict=True)

    first_graph_reward = None
    for step in range(NUM_STEPS):
        live = torch.randn(NUM_ENVS, DISC_OBS_DIM, device="cuda") + step
        reference = torch.randn(NUM_ENVS, DISC_OBS_DIM, device="cuda") - step
        eager_reward = eager.record_step(live, reference)
        graph_reward = graph.record_step(live, reference)
        assert torch.equal(graph_reward, eager_reward)
        if first_graph_reward is None:
            first_graph_reward = graph_reward.clone()
        else:
            assert torch.equal(graph._reward_rollout[0], first_graph_reward)

    assert torch.equal(graph._live_rollout, eager._live_rollout)
    assert torch.equal(graph._reference_rollout, eager._reference_rollout)
