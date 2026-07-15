# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU device utilities - single source of truth for Intel XPU hardware detection."""

from typing import Final

# Intel Arc Pro XPU families (architecturally identical - same runtime requirements)
# Add future Intel XPUs (b80, etc.) to this set.
XPU_FAMILIES: Final[set[str]] = {"b60", "b70"}

# Device selector env var for Level Zero runtime (Intel XPU)
LEVEL_ZERO_DEVICE_VAR: Final[str] = "ONEAPI_DEVICE_SELECTOR=level_zero:$GPU_LIST"
# Device selector env var for NVIDIA GPUs
CUDA_DEVICE_VAR: Final[str] = "CUDA_VISIBLE_DEVICES=$GPU_LIST"


def is_xpu_system(system_name: str | None) -> bool:
    """Return True if the system is an Intel XPU (Arc Pro family)."""
    if not system_name:
        return False
    return system_name.lower() in XPU_FAMILIES


def get_device_env_var(system_name: str | None) -> str:
    """Return the correct device env var for the given system.

    - Intel XPU (b60/b70/...): ONEAPI_DEVICE_SELECTOR=level_zero:$GPU_LIST
    - NVIDIA GPU (all others): CUDA_VISIBLE_DEVICES=$GPU_LIST
    """
    if is_xpu_system(system_name):
        return LEVEL_ZERO_DEVICE_VAR
    return CUDA_DEVICE_VAR


def get_accelerator_type(system_name: str | None) -> str:
    """Return 'xpu' for Intel XPU systems, 'nvidia' otherwise."""
    return "xpu" if is_xpu_system(system_name) else "nvidia"
