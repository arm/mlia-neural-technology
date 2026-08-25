# SPDX-FileCopyrightText: Copyright 2023, 2025-2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Technology target module."""

from mlia.plugins.plugins import TargetPlugin
from mlia.target.neural_technology.advisor import (
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.registry import TargetInfo, TargetRegistry


def _create_target_info() -> TargetInfo:
    """Build the Neural Technology target metadata."""
    return TargetInfo(
        supported_backends=["nx-performance-estimator"],
        default_backends=["nx-performance-estimator"],
        advisor_factory_func=configure_and_get_neural_technology_advisor,
        target_profile_cls=NeuralTechnologyConfiguration,
        supports_torch_module=True,
        torch_module_backend="nx-performance-estimator",
    )


class NeuralTechnologyTargetPlugin(TargetPlugin):
    """Neural Technology Target Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: TargetRegistry) -> None:
        """Register the target with the registry."""
        registry.register(
            "neural-technology",
            _create_target_info(),
        )
