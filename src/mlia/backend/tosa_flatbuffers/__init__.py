# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tosa flatbuffers API module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.backend.tosa_flatbuffers.install import get_tosa_flatbuffers_installation

registry.register(
    "tosa-flatbuffers",
    BackendConfiguration(
        supported_advice=[],
        supported_systems=[
            System.LINUX_AMD64,
            System.LINUX_AARCH64,
            System.WINDOWS_AMD64,
            System.WINDOWS_AARCH64,
        ],
        backend_type=BackendType.WHEEL,
        installation=get_tosa_flatbuffers_installation(),
    ),
    pretty_name="Tosa Flatbuffers",
)
