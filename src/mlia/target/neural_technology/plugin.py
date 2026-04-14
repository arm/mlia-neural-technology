# SPDX-FileCopyrightText: Copyright 2023, 2025-2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Technology target module."""

import inspect
from pathlib import Path

from mlia.core.handlers import WorkflowEventsHandler
from mlia.plugins.plugins import TargetPlugin
from mlia.target.neural_technology.advisor import (
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.handlers import NeuralTechnologyEventHandler
from mlia.target.registry import TargetInfo, TargetRegistry


def _target_info_parameter_names() -> set[str]:
    """Return the supported TargetInfo constructor parameter names."""
    return set(inspect.signature(TargetInfo.__init__).parameters)


def _require_collect_only_handler_support() -> None:
    """Ensure the installed mlia core supports collect-only API handlers."""
    parameters = inspect.signature(WorkflowEventsHandler.__init__).parameters
    if "collect_only" not in parameters:
        raise RuntimeError(
            "The 'collect_only' option requires a newer version of mlia. "
            "Please upgrade mlia to use this feature."
        )


def _target_info_supports_event_handler_factory() -> bool:
    """Return whether the installed mlia core exposes API event handler hooks."""
    return "event_handler_factory" in _target_info_parameter_names()


def _target_info_supports_torch_module_fields() -> bool:
    """Return whether the installed mlia core supports torch-module metadata."""
    return (
        "supports_torch_module" in _target_info_parameter_names()
        and "torch_module_backend" in _target_info_parameter_names()
    )


def _create_target_info() -> TargetInfo:
    """Build TargetInfo while remaining compatible with older mlia cores."""
    kwargs = {
        "supported_backends": ["nx-performance-estimator"],
        "default_backends": ["nx-performance-estimator"],
        "advisor_factory_func": configure_and_get_neural_technology_advisor,
        "target_profile_cls": NeuralTechnologyConfiguration,
    }

    if _target_info_supports_event_handler_factory():
        kwargs["event_handler_factory"] = create_neural_technology_api_event_handler

    if _target_info_supports_torch_module_fields():
        kwargs["supports_torch_module"] = True
        kwargs["torch_module_backend"] = "nx-performance-estimator"

    return TargetInfo(**kwargs)


def create_neural_technology_api_event_handler(
    output_dir: Path | None,
) -> NeuralTechnologyEventHandler:
    """Create the Neural Technology event handler used by the Python API."""
    _require_collect_only_handler_support()
    return NeuralTechnologyEventHandler(output_dir, collect_only=True)


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
