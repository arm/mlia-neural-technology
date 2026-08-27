# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Technology advisor module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

from mlia.core.advisor import DefaultInferenceAdvisor, InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import Context, ExecutionContext
from mlia.core.data_analysis import DataAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.errors import ConfigurationError
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.data_collection import (
    NeuralTechnologyCompatibility,
    NeuralTechnologyPerformance,
)


class NeuralTechnologyInferenceAdvisor(DefaultInferenceAdvisor):
    """Neural Technology Inference Advisor."""

    @classmethod
    def name(cls) -> str:
        """Return name of the advisor."""
        return "neural_technology_inference_advisor"

    def get_collectors(self, context: Context) -> list[DataCollector]:
        """Return data collectors for model analysis."""
        backend = self._get_backends(context)[0]
        model = self.get_model(context)
        target_cfg = self._get_target_cfg(context)
        collectors: list[DataCollector] = []
        if context.category_enabled(AdviceCategory.PERFORMANCE):
            collectors.append(NeuralTechnologyPerformance(model, target_cfg, backend))
        if context.category_enabled(AdviceCategory.COMPATIBILITY):
            collectors.append(NeuralTechnologyCompatibility(model, target_cfg))
        return collectors

    def get_analyzers(self, context: Context) -> list[DataAnalyzer]:
        """Return list of data analyzers."""
        return []

    def get_pattern_analyzers(self, _context: Context) -> list:
        """Return list of the pattern analyzers."""
        return []

    def _get_target_cfg(self, context: Context) -> NeuralTechnologyConfiguration:
        """Get target configuration."""
        return NeuralTechnologyConfiguration.load_profile(
            self.get_target_profile(context), self._get_backend_options(context)
        )

    def _get_backends(self, context: Context) -> list[str]:
        """Get the selected backend."""
        return cast(
            list[str],
            self.get_parameter(
                self.name(),
                "backends",
                expected_type=list,
                expected=True,
                context=context,
            ),
        )

    def _get_backend_options(self, context: Context) -> dict[str, dict[str, Any]]:
        """Get backend option overrides."""
        return cast(
            dict[str, dict[str, Any]],
            self.get_parameter(
                self.name(),
                "backend_options",
                expected=False,
                expected_type=dict,
                context=context,
            )
            or {},
        )


def configure_and_get_neural_technology_advisor(
    context: ExecutionContext,
    target_profile: str | Path,
    model: str | Path,
    **extra_args: Any,
) -> InferenceAdvisor:
    """Create and configure Neural Technology advisor."""
    if context.config_parameters is None:
        context.config_parameters = _get_config_parameters(
            model, target_profile, **extra_args
        )

    return NeuralTechnologyInferenceAdvisor()


def _get_config_parameters(
    model: str | Path, target_profile: str | Path, **extra_args: Any
) -> dict[str, Any]:
    """Get configuration parameters for the advisor."""
    parameters: dict[str, Any] = {
        "model": str(model),
        "target_profile": target_profile,
    }
    backends: Sequence = extra_args.get("backends", [])
    if not backends:
        raise ConfigurationError("One backend is required but was not specified.")
    if len(backends) > 1:
        raise ConfigurationError(
            f"Only one backend is supported but {len(backends)} were provided: "
            f"{backends}"
        )
    parameters["backends"] = list(backends)
    parameters["backend_options"] = extra_args.get("backend_options", {})
    return {NeuralTechnologyInferenceAdvisor.name(): parameters}
