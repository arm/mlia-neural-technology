# SPDX-FileCopyrightText: Copyright 2023,2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""ML SDK Model Converter backend module."""
import logging

from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.ml_sdk_model_converter.install import (
    get_ml_sdk_model_converter_installation,
)
from mlia.backend.registry import BackendRegistry
from mlia.plugins.plugins import BackendPlugin

logger = logging.getLogger(__name__)


class MLSDKModelConverterPlugin(BackendPlugin):
    """ML SDK Model Converter Backend Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the backend with the registry."""
        registry.register(
            "ml-sdk-model-converter",
            BackendConfiguration(
                supported_advice=[],
                supported_systems=[System.LINUX_AMD64],
                backend_type=BackendType.CUSTOM,
                installation=get_ml_sdk_model_converter_installation(),
                selectable=False,
            ),
            pretty_name="ML SDK Model Converter",
        )
