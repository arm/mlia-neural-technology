# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tosa flatbuffers API module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import BackendRegistry
from mlia.backend.tosa_flatbuffers.install import get_tosa_flatbuffers_installation
from mlia.plugins.plugins import BackendPlugin


class TosaFlatBuffersPlugin(BackendPlugin):
    """Vela Backend Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the backend with the registry."""
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
                selectable=False,
            ),
            pretty_name="Tosa Flatbuffers",
        )
