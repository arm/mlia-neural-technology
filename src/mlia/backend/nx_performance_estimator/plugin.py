# SPDX-FileCopyrightText: Copyright 2023,2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Accelerator Performance Estimator backend configuration."""

import logging

from mlia.backend.config import BackendConfiguration, BackendType, System
from mlia.backend.nx_performance_estimator.config import CONFIG_TO_CLI_OPTION
from mlia.backend.nx_performance_estimator.install import (
    get_nx_performance_estimator_installation,
)
from mlia.backend.registry import BackendRegistry
from mlia.core.common import AdviceCategory
from mlia.plugins.plugins import BackendPlugin
from mlia.target.neural_technology.filtering import DEFAULT_COLLAPSE_RULES

logger = logging.getLogger(__name__)


class NXPerformanceEstimatorPlugin(BackendPlugin):
    """Neural Accelerator  Performance Estimator Backend Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the backend with the registry."""
        registry.register(
            "nx-performance-estimator",
            BackendConfiguration(
                supported_advice=[
                    AdviceCategory.PERFORMANCE,
                    AdviceCategory.COMPATIBILITY,
                ],
                supported_systems=[System.LINUX_AMD64, System.WINDOWS_AMD64],
                backend_type=BackendType.CUSTOM,
                installation=get_nx_performance_estimator_installation(),
                cli_options=CONFIG_TO_CLI_OPTION,
                default_collapse_rules=DEFAULT_COLLAPSE_RULES,
            ),
            pretty_name="Neural Accelerator Performance Estimator",
        )
