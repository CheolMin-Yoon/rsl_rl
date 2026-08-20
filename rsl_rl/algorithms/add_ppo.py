# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO with an Adversarial Differential Discriminator reward."""

from __future__ import annotations

import math
import torch
from collections.abc import Mapping
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.extensions.add import AdversarialDifferentialDiscriminator
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

_CHECKPOINT_VERSION = 1


def _reward_weight(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _validate_module_state(module: torch.nn.Module, state: object, *, name: str, strict: bool) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"{name} checkpoint must be a mapping.")
    live = module.state_dict()
    if strict and set(state) != set(live):
        raise RuntimeError(f"{name} checkpoint keys do not match.")
    for key, live_value in live.items():
        if key not in state:
            raise RuntimeError(f"{name} checkpoint is missing {key!r}.")
        value = state[key]
        if isinstance(live_value, torch.Tensor):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != live_value.shape
                or value.dtype != live_value.dtype
            ):
                raise RuntimeError(f"{name} checkpoint tensor {key!r} has the wrong shape or dtype.")
        elif value != live_value:
            raise RuntimeError(f"{name} checkpoint identity {key!r} does not match.")


class ADDPPO(PPO):
    """Stock RSL-RL PPO with a MimicKit-compatible ADD auxiliary.

    The environment supplies transition-aligned, pre-reset live and reference
    discriminator observations. ADD converts their difference into a scalar
    reward while stock :class:`PPO` continues to own policy storage, GAE, and
    actor/critic optimization.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        add_cfg: Mapping[str, object],
        task_reward_weight: float = 0.0,
        add_reward_weight: float = 1.0,
        live_obs_key: str = "add_live_obs",
        reference_obs_key: str = "add_reference_obs",
        # Stock PPO parameters
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float | None = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize stock PPO and its independent ADD reward auxiliary."""
        if not isinstance(add_cfg, Mapping):
            raise TypeError("add_cfg must be a mapping.")
        if rnd_cfg is not None:
            raise ValueError("ADDPPO does not support RND in v1.")
        if symmetry_cfg is not None:
            raise ValueError("ADDPPO does not support symmetry in v1.")
        if multi_gpu_cfg is not None:
            raise ValueError("ADDPPO does not support unsynchronized multi-GPU training in v1.")
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("ADDPPO supports feed-forward actor and critic models only in v1.")
        if not isinstance(live_obs_key, str) or not live_obs_key:
            raise ValueError("live_obs_key must be a non-empty string.")
        if not isinstance(reference_obs_key, str) or not reference_obs_key:
            raise ValueError("reference_obs_key must be a non-empty string.")
        if live_obs_key == reference_obs_key:
            raise ValueError("live_obs_key and reference_obs_key must be distinct.")

        self.task_reward_weight = _reward_weight(task_reward_weight, name="task_reward_weight")
        self.add_reward_weight = _reward_weight(add_reward_weight, name="add_reward_weight")
        if self.task_reward_weight == 0.0 and self.add_reward_weight == 0.0:
            raise ValueError("task_reward_weight and add_reward_weight cannot both be zero.")
        self.live_obs_key = live_obs_key
        self.reference_obs_key = reference_obs_key

        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=None,
            symmetry_cfg=None,
            multi_gpu_cfg=None,
        )
        self.add_discriminator = AdversarialDifferentialDiscriminator(
            num_envs=storage.num_envs,
            num_steps_per_env=storage.num_transitions_per_env,
            device=device,
            **dict(add_cfg),  # type: ignore[arg-type]
        )

    def _extra_observation(self, extras: Mapping[str, object], key: str) -> torch.Tensor:
        value = extras.get(key)
        expected = (self.storage.num_envs, self.add_discriminator.disc_obs_dim)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"extras[{key!r}] must have exact shape {expected}.")
        if value.dtype != torch.float32:
            raise ValueError(f"extras[{key!r}] must use torch.float32.")
        if value.device != self.add_discriminator.device:
            raise ValueError(f"extras[{key!r}] must be on {self.add_discriminator.device}.")
        if self.add_discriminator.runtime_finite_checks and not torch.isfinite(value).all():
            raise ValueError(f"extras[{key!r}] must contain only finite values.")
        return value

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, object],
    ) -> None:
        """Mix ADD reward into PPO learning reward and record the paired payload."""
        if not isinstance(rewards, torch.Tensor) or tuple(rewards.shape) != (self.storage.num_envs,):
            raise ValueError(f"rewards must have exact shape {(self.storage.num_envs,)}.")
        if not rewards.is_floating_point() or (
            self.add_discriminator.runtime_finite_checks and not torch.isfinite(rewards).all()
        ):
            raise ValueError("rewards must contain finite floating values.")
        live = self._extra_observation(extras, self.live_obs_key)
        reference = self._extra_observation(extras, self.reference_obs_key)
        add_rewards = self.add_discriminator.record_step(live, reference)
        learning_rewards = self.task_reward_weight * rewards.to(self.device) + self.add_reward_weight * add_rewards
        super().process_env_step(obs, learning_rewards, dones, extras)  # type: ignore[arg-type]

    def update(self) -> dict[str, float]:
        """Run stock PPO first, then update the independent ADD discriminator."""
        self.add_discriminator.require_complete_rollout()
        losses = super().update()
        add_losses = self.add_discriminator.update()
        losses.update({f"add/{name}": value for name, value in add_losses.items()})
        return losses

    def train_mode(self) -> None:
        """Set the stock actor/critic and ADD discriminator to training mode."""
        super().train_mode()
        self.add_discriminator.train()

    def eval_mode(self) -> None:
        """Set the stock actor/critic and ADD discriminator to evaluation mode."""
        super().eval_mode()
        self.add_discriminator.eval()

    def _checkpoint_identity(self) -> dict[str, object]:
        return {
            "version": _CHECKPOINT_VERSION,
            "live_obs_key": self.live_obs_key,
            "reference_obs_key": self.reference_obs_key,
            "task_reward_weight": self.task_reward_weight,
            "add_reward_weight": self.add_reward_weight,
            "add_settings": self.add_discriminator.settings(),
        }

    def save(self) -> dict:
        """Save stock PPO plus ADD model, optimizer, replay, and normalizer state."""
        if not self.add_discriminator.at_update_boundary:
            raise RuntimeError("ADDPPO checkpoints may only be written at a completed update boundary.")
        saved = super().save()
        saved.update({
            "add_checkpoint_identity": self._checkpoint_identity(),
            "add_state_dict": self.add_discriminator.state_dict(),
            "add_optimizer_state_dict": self.add_discriminator.optimizer.state_dict(),
        })
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore a full ADD learner or permit an explicit actor-only load."""
        # Stock PPO treats a partial mapping such as ``{"actor": True}`` as
        # actor-only loading. MJLab play uses exactly that contract, often with
        # a different environment count, so ADD follows the same opt-in rule.
        load_add = load_cfg is None or bool(load_cfg.get("add", False))
        if load_add:
            if loaded_dict.get("add_checkpoint_identity") != self._checkpoint_identity():
                raise RuntimeError("ADD checkpoint identity does not match the configured learner.")
            _validate_module_state(
                self.add_discriminator,
                loaded_dict.get("add_state_dict"),
                name="ADD discriminator",
                strict=strict,
            )
            if not isinstance(loaded_dict.get("add_optimizer_state_dict"), Mapping):
                raise TypeError("ADD optimizer checkpoint must be a mapping.")

        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_add:
            self.add_discriminator.load_state_dict(loaded_dict["add_state_dict"], strict=strict)
            self.add_discriminator.optimizer.load_state_dict(loaded_dict["add_optimizer_state_dict"])
        return load_iteration

    def compile(self, mode: str | None = None) -> None:
        """Reject unvalidated compilation while preserving the stock no-op path."""
        if mode is not None:
            raise ValueError("ADDPPO does not support torch.compile in v1.")
        super().compile(None)

    def broadcast_parameters(self) -> None:
        """Reject distributed broadcast until ADD state synchronization is implemented."""
        raise RuntimeError("ADDPPO does not support multi-GPU parameter broadcast in v1.")


__all__ = ["ADDPPO"]
