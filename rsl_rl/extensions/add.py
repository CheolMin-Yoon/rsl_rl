# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial Differential Discriminator (ADD) auxiliary.

This is an RSL-RL-native implementation of the ADD discriminator contract from
MimicKit. It intentionally owns only the differential discriminator, paired
replay, delayed mean-absolute normalization, and discriminator optimization;
the policy learner remains stock RSL-RL PPO.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_optimizer

_STATE_VERSION = 1


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    non_negative: bool = False,
    positive: bool = False,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _clip_bound(value: object, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if math.isnan(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _hidden_dims(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("hidden_dims must be a sequence of positive integers.")
    result = tuple(value)
    if not result or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in result):
        raise ValueError("hidden_dims must contain only positive integers.")
    return result


def _resolve_runtime_backend(value: object, *, device: torch.device) -> tuple[str, str]:
    if not isinstance(value, str):
        raise TypeError("runtime_backend must be 'eager', 'auto', or 'cuda_graph'.")
    requested = value.lower()
    if requested not in {"eager", "auto", "cuda_graph"}:
        raise ValueError("runtime_backend must be 'eager', 'auto', or 'cuda_graph'.")
    resolved = "cuda_graph" if requested == "auto" and device.type == "cuda" else requested
    if resolved == "auto":
        resolved = "eager"
    if resolved == "cuda_graph" and device.type != "cuda":
        raise ValueError("runtime_backend='cuda_graph' requires a CUDA learner device.")
    return requested, resolved


class DiffNormalizer(nn.Module):
    """Scale differences by a delayed running mean absolute value.

    ADD's positive class is exactly zero, so this normalizer never recenters
    its inputs. ``record`` accumulates pending moments and ``update`` commits
    them after discriminator optimization, keeping one frozen snapshot for a
    complete rollout and update.
    """

    def __init__(
        self,
        shape: int | Sequence[int],
        *,
        min_diff: float = 1.0e-4,
        clip: float = math.inf,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        runtime_finite_checks: bool = True,
    ) -> None:
        """Initialize delayed mean-absolute statistics for the requested feature shape."""
        super().__init__()
        if isinstance(shape, int) and not isinstance(shape, bool):
            self.shape = (shape,)
        elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
            self.shape = tuple(shape)
        else:
            raise TypeError("shape must be an integer or a sequence of integers.")
        if not self.shape or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.shape
        ):
            raise ValueError("shape must contain only positive integers.")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype.")

        self.min_diff = _finite_float(min_diff, name="min_diff", positive=True)
        self.clip = _clip_bound(clip, name="clip")
        if not isinstance(runtime_finite_checks, bool):
            raise TypeError("runtime_finite_checks must be a boolean.")
        self.runtime_finite_checks = runtime_finite_checks
        self.register_buffer("count", torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer("mean_abs", torch.ones(self.shape, device=device, dtype=dtype))
        self.register_buffer("pending_count", torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer("pending_sum_abs", torch.zeros(self.shape, device=device, dtype=dtype))

    @property
    def device(self) -> torch.device:
        """Return the device that owns the running statistics."""
        return self.mean_abs.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating dtype used by the running statistics."""
        return self.mean_abs.dtype

    def _validate(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.ndim <= len(self.shape) or tuple(value.shape[-len(self.shape) :]) != self.shape:
            raise ValueError(f"{name} must have one or more batch axes followed by {self.shape}.")
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError(f"{name} must use dtype/device {self.dtype}/{self.device}.")
        if self.runtime_finite_checks and not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")
        return value.reshape(-1, *self.shape)

    @torch.no_grad()
    def record(self, value: torch.Tensor) -> None:
        """Accumulate absolute moments without changing the active snapshot."""
        samples = self._validate(value, name="diff normalizer samples")
        self.pending_count.add_(samples.shape[0])
        self.pending_sum_abs.add_(samples.abs().sum(dim=0))

    @torch.no_grad()
    def update(self) -> None:
        """Commit all pending moments to the active mean-absolute snapshot."""
        new_count = int(self.pending_count.item())
        if new_count == 0:
            return
        old_count = int(self.count.item())
        total_count = old_count + new_count
        self.mean_abs.mul_(old_count / total_count).add_(
            self.pending_sum_abs / new_count,
            alpha=new_count / total_count,
        )
        self.count.fill_(total_count)
        self.clear_pending()

    @torch.no_grad()
    def clear_pending(self) -> None:
        """Discard uncommitted moments without changing active statistics."""
        self.pending_count.zero_()
        self.pending_sum_abs.zero_()

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """Divide by the active mean absolute value without recentering."""
        samples = self._validate(value, name="diff normalizer input")
        normalized = (samples / self.mean_abs.clamp_min(self.min_diff)).clamp(-self.clip, self.clip)
        return normalized.reshape(value.shape)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor through the module call interface."""
        return self.normalize(value)

    def get_extra_state(self) -> dict[str, object]:
        """Return the configuration identity included in module checkpoints."""
        return {
            "version": _STATE_VERSION,
            "shape": self.shape,
            "min_diff": self.min_diff,
            "clip": self.clip,
            "dtype": str(self.dtype),
        }

    def set_extra_state(self, state: object) -> None:
        """Validate checkpoint configuration identity during state loading."""
        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError("DiffNormalizer identity does not match the configured instance.")


class _PermutationSampler:
    """Consume random permutations and reshuffle only after a complete cycle."""

    def __init__(self, sample_count: int, *, device: torch.device) -> None:
        self.sample_count = _positive_int(sample_count, name="sample_count")
        self.device = device
        self.permutation = torch.randperm(sample_count, device=device, dtype=torch.long)
        self.head = 0

    def sample(self, count: int) -> torch.Tensor:
        count = _positive_int(count, name="sample count")
        if count > self.sample_count:
            raise ValueError("sample count must not exceed the available samples.")
        if self.head + count <= self.sample_count:
            indices = self.permutation[self.head : self.head + count]
            self.head += count
            return indices
        before_wrap = self.permutation[self.head :]
        remainder = count - before_wrap.shape[0]
        self.permutation = torch.randperm(self.sample_count, device=self.device, dtype=torch.long)
        self.head = remainder
        return torch.cat((before_wrap, self.permutation[:remainder]), dim=0)


class _PairedReplayBuffer(nn.Module):
    """Bounded ring that keeps each live/reference pair under one index."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        *,
        device: torch.device | str,
        runtime_finite_checks: bool = True,
    ) -> None:
        super().__init__()
        self.capacity = _positive_int(capacity, name="replay capacity")
        self.observation_dim = _positive_int(observation_dim, name="observation_dim")
        if not isinstance(runtime_finite_checks, bool):
            raise TypeError("runtime_finite_checks must be a boolean.")
        self.runtime_finite_checks = runtime_finite_checks
        self.register_buffer(
            "pairs",
            torch.zeros(self.capacity, 2, self.observation_dim, device=device, dtype=torch.float32),
        )
        self.register_buffer("write_head", torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer("total_samples", torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer(
            "sample_permutation",
            torch.randperm(self.capacity, device=device, dtype=torch.long),
        )
        self.register_buffer("sample_head", torch.zeros((), device=device, dtype=torch.long))

    @property
    def device(self) -> torch.device:
        return self.pairs.device

    @property
    def sample_count(self) -> int:
        return min(int(self.total_samples.item()), self.capacity)

    @property
    def is_full(self) -> bool:
        return int(self.total_samples.item()) >= self.capacity

    def _validate_pair(self, live: torch.Tensor, reference: torch.Tensor) -> None:
        expected_tail = (self.observation_dim,)
        if (
            not isinstance(live, torch.Tensor)
            or not isinstance(reference, torch.Tensor)
            or live.ndim != 2
            or reference.shape != live.shape
            or tuple(live.shape[1:]) != expected_tail
        ):
            raise ValueError("replay live/reference tensors must have matching [samples, observation_dim] shapes.")
        if live.device != self.device or reference.device != self.device:
            raise ValueError(f"replay tensors must be on {self.device}.")
        if live.dtype != torch.float32 or reference.dtype != torch.float32:
            raise ValueError("replay tensors must use torch.float32.")
        if self.runtime_finite_checks and (
            not torch.isfinite(live).all() or not torch.isfinite(reference).all()
        ):
            raise ValueError("replay tensors must contain only finite values.")

    @torch.no_grad()
    def push(self, live: torch.Tensor, reference: torch.Tensor) -> None:
        self._validate_pair(live, reference)
        count = live.shape[0]
        if count <= 0 or count > self.capacity:
            raise ValueError(f"replay push count must be in [1, {self.capacity}].")
        values = torch.stack((live, reference), dim=1)
        start = int(self.write_head.item())
        first_count = min(count, self.capacity - start)
        self.pairs[start : start + first_count].copy_(values[:first_count])
        remainder = count - first_count
        if remainder:
            self.pairs[:remainder].copy_(values[first_count:])
        self.write_head.fill_((start + count) % self.capacity)
        self.total_samples.add_(count)

    @torch.no_grad()
    def _reset_sample_permutation(self) -> None:
        self.sample_permutation.copy_(torch.randperm(self.capacity, device=self.device, dtype=torch.long))
        self.sample_head.zero_()

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        count = _positive_int(count, name="replay sample count")
        if count > self.capacity:
            raise ValueError("replay sample count must not exceed replay capacity.")
        available = self.sample_count
        if available == 0:
            raise RuntimeError("cannot sample an empty ADD replay buffer.")
        head = int(self.sample_head.item())
        if head + count <= self.capacity:
            indices = self.sample_permutation[head : head + count]
            self.sample_head.add_(count)
        else:
            before_wrap = self.sample_permutation[head:]
            remainder = count - before_wrap.shape[0]
            self._reset_sample_permutation()
            indices = torch.cat((before_wrap, self.sample_permutation[:remainder]), dim=0)
            self.sample_head.fill_(remainder)
        sampled = self.pairs[torch.remainder(indices, available)]
        return sampled[:, 0], sampled[:, 1]

    def get_extra_state(self) -> dict[str, object]:
        return {
            "version": _STATE_VERSION,
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError("ADD replay identity does not match the configured instance.")


@dataclass(frozen=True)
class ADDDiscriminatorLoss:
    """Differentiable discriminator objective and detached diagnostics."""

    total_loss: torch.Tensor
    gradient_penalty: torch.Tensor
    logit_loss: torch.Tensor
    positive_accuracy: torch.Tensor
    negative_accuracy: torch.Tensor
    positive_logit_mean: torch.Tensor
    negative_logit_mean: torch.Tensor


class AdversarialDifferentialDiscriminator(nn.Module):
    """MimicKit-compatible ADD core for one RSL-RL PPO learner."""

    def __init__(
        self,
        disc_obs_dim: int,
        num_envs: int,
        num_steps_per_env: int,
        hidden_dims: Sequence[int] = (1024, 512),
        activation: str = "relu",
        disc_epochs: int = 2,
        disc_batch_size: float = 2.0,
        disc_buffer_size: int = 200_000,
        disc_replay_samples: int = 1000,
        disc_logit_reg: float = 0.01,
        disc_grad_penalty: float = 2.0,
        disc_reward_scale: float = 2.0,
        disc_optimizer: str = "sgd",
        disc_learning_rate: float = 2.5e-4,
        disc_momentum: float = 0.9,
        disc_weight_decay: float = 1.0e-4,
        normalizer_samples: int = 100_000_000,
        normalizer_clip: float = math.inf,
        normalizer_min_diff: float = 1.0e-4,
        runtime_backend: str = "eager",
        runtime_finite_checks: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the exact ADD core for one fixed-shape PPO rollout."""
        super().__init__()
        runtime_device = torch.device(device)
        self.runtime_backend_requested, self.runtime_backend = _resolve_runtime_backend(
            runtime_backend,
            device=runtime_device,
        )
        if not isinstance(runtime_finite_checks, bool):
            raise TypeError("runtime_finite_checks must be a boolean.")
        self.runtime_finite_checks = runtime_finite_checks

        self.disc_obs_dim = _positive_int(disc_obs_dim, name="disc_obs_dim")
        self.num_envs = _positive_int(num_envs, name="num_envs")
        self.num_steps_per_env = _positive_int(num_steps_per_env, name="num_steps_per_env")
        self.hidden_dims = _hidden_dims(hidden_dims)
        if not isinstance(activation, str) or not activation:
            raise ValueError("activation must be a non-empty string.")
        self.activation = activation
        self.disc_epochs = _positive_int(disc_epochs, name="disc_epochs")
        self.disc_batch_size = _finite_float(disc_batch_size, name="disc_batch_size", positive=True)
        self.disc_buffer_size = _positive_int(disc_buffer_size, name="disc_buffer_size")
        self.disc_replay_samples = _positive_int(disc_replay_samples, name="disc_replay_samples")
        self.disc_logit_reg = _finite_float(disc_logit_reg, name="disc_logit_reg", non_negative=True)
        self.disc_grad_penalty = _finite_float(
            disc_grad_penalty,
            name="disc_grad_penalty",
            non_negative=True,
        )
        self.disc_reward_scale = _finite_float(
            disc_reward_scale,
            name="disc_reward_scale",
            non_negative=True,
        )
        self.disc_learning_rate = _finite_float(
            disc_learning_rate,
            name="disc_learning_rate",
            positive=True,
        )
        self.disc_momentum = _finite_float(disc_momentum, name="disc_momentum", non_negative=True)
        self.disc_weight_decay = _finite_float(
            disc_weight_decay,
            name="disc_weight_decay",
            non_negative=True,
        )
        self.normalizer_samples = _positive_int(normalizer_samples, name="normalizer_samples")
        self.normalizer_clip = _clip_bound(normalizer_clip, name="normalizer_clip")
        self.normalizer_min_diff = _finite_float(
            normalizer_min_diff,
            name="normalizer_min_diff",
            positive=True,
        )
        if not isinstance(disc_optimizer, str) or not disc_optimizer:
            raise ValueError("disc_optimizer must be a non-empty string.")
        self.disc_optimizer_name = disc_optimizer.lower()

        rollout_size = self.num_envs * self.num_steps_per_env
        if self.disc_buffer_size < rollout_size:
            raise ValueError("disc_buffer_size must hold at least one complete rollout.")
        if self.disc_replay_samples > self.disc_buffer_size:
            raise ValueError("disc_replay_samples must not exceed disc_buffer_size.")
        batch_size = math.ceil(self.disc_batch_size * self.num_envs)
        if batch_size > rollout_size:
            raise ValueError("the discriminator batch size must not exceed one rollout.")

        self.discriminator = MLP(
            self.disc_obs_dim,
            1,
            self.hidden_dims,
            activation=self.activation,
        ).to(device)
        linear_layers = [module for module in self.discriminator if isinstance(module, nn.Linear)]
        for layer in linear_layers:
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(linear_layers[-1].weight, -1.0, 1.0)
        self.diff_normalizer = DiffNormalizer(
            self.disc_obs_dim,
            min_diff=self.normalizer_min_diff,
            clip=self.normalizer_clip,
            device=device,
            runtime_finite_checks=self.runtime_finite_checks,
        )
        self.replay = _PairedReplayBuffer(
            self.disc_buffer_size,
            self.disc_obs_dim,
            device=device,
            runtime_finite_checks=self.runtime_finite_checks,
        )

        optimizer_class = resolve_optimizer(self.disc_optimizer_name)
        optimizer_kwargs = {
            "lr": self.disc_learning_rate,
            "weight_decay": self.disc_weight_decay,
        }
        if self.disc_optimizer_name == "sgd":
            optimizer_kwargs["momentum"] = self.disc_momentum
        elif self.disc_momentum != 0.0:
            raise ValueError("disc_momentum is only supported with the SGD discriminator optimizer.")
        self.optimizer = optimizer_class(self.discriminator.parameters(), **optimizer_kwargs)  # type: ignore[arg-type]

        rollout_shape = (self.num_steps_per_env, self.num_envs, self.disc_obs_dim)
        self._live_rollout = torch.empty(rollout_shape, device=device, dtype=torch.float32)
        self._reference_rollout = torch.empty(rollout_shape, device=device, dtype=torch.float32)
        self._reward_rollout = torch.empty(
            self.num_steps_per_env,
            self.num_envs,
            device=device,
            dtype=torch.float32,
        )
        self._rollout_step = 0
        self.register_buffer("policy_sample_count", torch.zeros((), device=device, dtype=torch.long))
        self._normalizer_update_enabled: bool | None = None
        self._reward_graphs: list[torch.cuda.CUDAGraph] = []
        if self.runtime_backend == "cuda_graph":
            self._capture_reward_graphs()

    @property
    def device(self) -> torch.device:
        """Return the learner device shared by ADD state tensors."""
        return self.policy_sample_count.device

    @property
    def at_update_boundary(self) -> bool:
        """Return whether the auxiliary can be checkpointed safely."""
        return self._rollout_step == 0 and int(self.diff_normalizer.pending_count.item()) == 0

    @property
    def rollout_step(self) -> int:
        """Return the host-owned rollout position without synchronizing CUDA."""
        return self._rollout_step

    def _normalize_unchecked(self, value: torch.Tensor) -> torch.Tensor:
        return (value / self.diff_normalizer.mean_abs.clamp_min(self.normalizer_min_diff)).clamp(
            -self.normalizer_clip,
            self.normalizer_clip,
        )

    def _reward_unchecked(self, live: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        difference = reference - live
        logits = self.discriminator(self._normalize_unchecked(difference)).squeeze(-1)
        return -self.disc_reward_scale * torch.log((1.0 - torch.sigmoid(logits)).clamp_min(1.0e-4))

    @torch.no_grad()
    def _capture_reward_graphs(self) -> None:
        """Capture one bitwise reward graph for each fixed rollout row."""
        self._live_rollout.zero_()
        self._reference_rollout.zero_()
        self._reward_rollout.zero_()
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                self._reward_rollout[0].copy_(
                    self._reward_unchecked(self._live_rollout[0], self._reference_rollout[0])
                )
        capture_stream.synchronize()
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)

        pool = torch.cuda.graph_pool_handle()
        try:
            for step in range(self.num_steps_per_env):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                    self._reward_rollout[step].copy_(
                        self._reward_unchecked(self._live_rollout[step], self._reference_rollout[step])
                    )
                self._reward_graphs.append(graph)
        except Exception as error:
            self._reward_graphs.clear()
            raise RuntimeError("Failed to capture the ADD CUDA reward graphs.") from error

    def settings(self) -> dict[str, object]:
        """Return the serialized identity of the ADD auxiliary."""
        return {
            "disc_obs_dim": self.disc_obs_dim,
            "num_envs": self.num_envs,
            "num_steps_per_env": self.num_steps_per_env,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "disc_epochs": self.disc_epochs,
            "disc_batch_size": self.disc_batch_size,
            "disc_buffer_size": self.disc_buffer_size,
            "disc_replay_samples": self.disc_replay_samples,
            "disc_logit_reg": self.disc_logit_reg,
            "disc_grad_penalty": self.disc_grad_penalty,
            "disc_reward_scale": self.disc_reward_scale,
            "disc_optimizer": self.disc_optimizer_name,
            "disc_learning_rate": self.disc_learning_rate,
            "disc_momentum": self.disc_momentum,
            "disc_weight_decay": self.disc_weight_decay,
            "normalizer_samples": self.normalizer_samples,
            "normalizer_clip": self.normalizer_clip,
            "normalizer_min_diff": self.normalizer_min_diff,
        }

    def get_extra_state(self) -> dict[str, object]:
        """Return versioned ADD settings for module checkpoint identity."""
        return {"version": _STATE_VERSION, "settings": self.settings()}

    def set_extra_state(self, state: object) -> None:
        """Reject a checkpoint created for a different ADD configuration."""
        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError("ADD discriminator identity does not match the configured instance.")

    def _validate_pair(self, live: torch.Tensor, reference: torch.Tensor) -> None:
        expected = (self.num_envs, self.disc_obs_dim)
        for name, value in (("live", live), ("reference", reference)):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise ValueError(f"ADD {name} observations must have exact shape {expected}.")
            if value.device != self.device or value.dtype != torch.float32:
                raise ValueError(f"ADD {name} observations must use float32 on {self.device}.")
            if self.runtime_finite_checks and not torch.isfinite(value).all():
                raise ValueError(f"ADD {name} observations must contain only finite values.")

    def reward(self, live: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Evaluate the ADD style reward on ``reference - live``."""
        self._validate_pair(live, reference)
        with torch.no_grad():
            rewards = self._reward_unchecked(live, reference)
        if tuple(rewards.shape) != (self.num_envs,) or (
            self.runtime_finite_checks and not torch.isfinite(rewards).all()
        ):
            raise RuntimeError("ADD reward must contain one finite scalar per environment.")
        return rewards

    @torch.no_grad()
    def record_step(self, live: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Record one transition-aligned pair and return its frozen ADD reward."""
        self._validate_pair(live, reference)
        step = self._rollout_step
        if step >= self.num_steps_per_env:
            raise RuntimeError("ADD rollout buffer overflow.")
        if step == 0:
            self._normalizer_update_enabled = int(self.policy_sample_count.item()) < self.normalizer_samples
        self._live_rollout[step].copy_(live)
        self._reference_rollout[step].copy_(reference)
        if self.runtime_backend == "cuda_graph":
            self._reward_graphs[step].replay()
        else:
            self._reward_rollout[step].copy_(self._reward_unchecked(live, reference))
        rewards = self._reward_rollout[step]
        if self.runtime_finite_checks and not torch.isfinite(rewards).all():
            raise RuntimeError("ADD reward must contain one finite scalar per environment.")
        self._rollout_step += 1
        return rewards

    def require_complete_rollout(self) -> None:
        """Require all configured environment steps before an update."""
        if self._rollout_step != self.num_steps_per_env:
            raise RuntimeError("ADD update requires one complete rollout.")

    def compute_loss(self, difference: torch.Tensor) -> ADDDiscriminatorLoss:
        """Compute zero-positive BCE, two-sided gradient penalty, and logit L2."""
        if not isinstance(difference, torch.Tensor) or difference.ndim != 2 or difference.shape[1] != self.disc_obs_dim:
            raise ValueError(f"difference must have shape [samples, {self.disc_obs_dim}].")
        if difference.device != self.device or difference.dtype != torch.float32:
            raise ValueError(f"difference must use float32 on {self.device}.")
        if difference.shape[0] == 0 or (
            self.runtime_finite_checks and not torch.isfinite(difference).all()
        ):
            raise ValueError("difference must contain one or more finite samples.")

        positive = torch.zeros(1, self.disc_obs_dim, device=self.device, dtype=torch.float32).requires_grad_(True)
        negative = self.diff_normalizer(difference).detach().requires_grad_(True)
        positive_logits = self.discriminator(positive).squeeze(-1)
        negative_logits = self.discriminator(negative).squeeze(-1)

        positive_loss = nn.functional.binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
        )
        negative_loss = nn.functional.binary_cross_entropy_with_logits(
            negative_logits,
            torch.zeros_like(negative_logits),
        )
        total_loss = 0.5 * (positive_loss + negative_loss)

        positive_gradient = torch.autograd.grad(
            positive_logits,
            positive,
            grad_outputs=torch.ones_like(positive_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        negative_gradient = torch.autograd.grad(
            negative_logits,
            negative,
            grad_outputs=torch.ones_like(negative_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = 0.5 * (
            positive_gradient.square().sum(dim=-1).mean() + negative_gradient.square().sum(dim=-1).mean()
        )

        output_layer = next(module for module in reversed(self.discriminator) if isinstance(module, nn.Linear))
        logit_loss = output_layer.weight.square().sum()
        total_loss = total_loss + self.disc_grad_penalty * gradient_penalty + self.disc_logit_reg * logit_loss
        if self.runtime_finite_checks and not torch.isfinite(total_loss):
            raise RuntimeError("ADD discriminator loss is non-finite.")
        return ADDDiscriminatorLoss(
            total_loss=total_loss,
            gradient_penalty=gradient_penalty.detach(),
            logit_loss=logit_loss.detach(),
            positive_accuracy=(positive_logits.detach() > 0.0).float().mean(),
            negative_accuracy=(negative_logits.detach() < 0.0).float().mean(),
            positive_logit_mean=positive_logits.detach().mean(),
            negative_logit_mean=negative_logits.detach().mean(),
        )

    def _store_replay(self, live: torch.Tensor, reference: torch.Tensor) -> None:
        sample_count = live.shape[0]
        insert_count = min(sample_count, self.disc_replay_samples) if self.replay.is_full else sample_count
        indices = torch.randperm(sample_count, device=self.device, dtype=torch.long)[:insert_count]
        self.replay.push(live[indices], reference[indices])

    def update(self) -> dict[str, float]:
        """Train on current and paired replay differences, then commit normalization."""
        if not self.training:
            raise RuntimeError("ADD update requires training mode.")
        self.require_complete_rollout()
        live = self._live_rollout.view(-1, self.disc_obs_dim)
        reference = self._reference_rollout.view(-1, self.disc_obs_dim)
        difference = reference - live
        if bool(self._normalizer_update_enabled):
            self.diff_normalizer.record(difference)
        self._store_replay(live, reference)

        sample_count = live.shape[0]
        batch_size = math.ceil(self.disc_batch_size * self.num_envs)
        batch_count = math.ceil(sample_count / batch_size)
        update_count = batch_count * self.disc_epochs
        sampler = _PermutationSampler(sample_count, device=self.device)
        metric_names = (
            "loss",
            "gradient_penalty",
            "logit_loss",
            "positive_accuracy",
            "negative_accuracy",
            "positive_logit",
            "negative_logit",
        )
        metric_rows: list[torch.Tensor] = []
        for _ in range(update_count):
            indices = sampler.sample(batch_size)
            replay_live, replay_reference = self.replay.sample(batch_size)
            negative_difference = torch.cat(
                (reference[indices] - live[indices], replay_reference - replay_live),
                dim=0,
            )
            loss = self.compute_loss(negative_difference)
            self.optimizer.zero_grad()
            loss.total_loss.backward()
            self.optimizer.step()
            metric_rows.append(
                torch.stack(
                    (
                        loss.total_loss.detach(),
                        loss.gradient_penalty,
                        loss.logit_loss,
                        loss.positive_accuracy,
                        loss.negative_accuracy,
                        loss.positive_logit_mean,
                        loss.negative_logit_mean,
                    )
                )
            )

        if bool(self._normalizer_update_enabled):
            self.diff_normalizer.update()
        self.policy_sample_count.add_(sample_count)
        metric_tensor = torch.stack(metric_rows).to(torch.float64)
        summary_tensor = torch.stack(
            (
                self._reward_rollout.mean().to(torch.float64),
                self._reward_rollout.std(unbiased=False).to(torch.float64),
                self.replay.total_samples.clamp_max(self.replay.capacity).to(torch.float64),
                self.diff_normalizer.count.to(torch.float64),
                self.policy_sample_count.to(torch.float64),
            )
        )
        host_metrics = torch.cat((metric_tensor.flatten(), summary_tensor)).cpu()
        metric_values = host_metrics[:-summary_tensor.numel()].reshape(update_count, len(metric_names)).tolist()
        totals = {name: 0.0 for name in metric_names}
        for row in metric_values:
            for name, value in zip(metric_names, row, strict=True):
                totals[name] += float(value)
        summary = host_metrics[-summary_tensor.numel() :].tolist()
        reward_mean, reward_std, replay_size, normalizer_count, policy_samples = map(float, summary)
        self._rollout_step = 0
        self._normalizer_update_enabled = None
        return {
            **{name: value / update_count for name, value in totals.items()},
            "updates": float(update_count),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "replay_size": replay_size,
            "normalizer_count": normalizer_count,
            "policy_samples": policy_samples,
        }


__all__ = ["ADDDiscriminatorLoss", "AdversarialDifferentialDiscriminator", "DiffNormalizer"]
