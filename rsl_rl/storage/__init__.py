# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Storage for the learning algorithms."""

from .multi_critic_rollout_storage import MultiCriticRolloutStorage
from .rollout_storage import RolloutStorage

__all__ = ["MultiCriticRolloutStorage", "RolloutStorage"]
