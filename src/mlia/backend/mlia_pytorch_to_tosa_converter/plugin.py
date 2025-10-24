# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA Converter For PyTorch backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.mlia_pytorch_to_tosa_converter.install import (
    get_mlia_pytorch_to_tosa_backend_installation,
)
from mlia.backend.registry import BackendRegistry
from mlia.plugins.plugins import BackendPlugin


class MliaPytorchToTosaConverterPlugin(BackendPlugin):
    """TOSA converter for PyTorch Backend Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the backend with the registry."""
        registry.register(
            "mlia-pytorch-to-tosa-converter",
            BackendConfiguration(
                supported_advice=[],
                supported_systems=[System.LINUX_AMD64],
                backend_type=BackendType.WHEEL,
                installation=get_mlia_pytorch_to_tosa_backend_installation(),
            ),
            pretty_name="TOSA converter for PyTorch",
        )
