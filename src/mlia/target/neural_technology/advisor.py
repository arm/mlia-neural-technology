# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
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
    NeuralTechnologyProfilingData,
)
from mlia.utils.misc import summarize_list

PROFILING_DATA_BACKEND = "neural-technology-profiling-data"


class NeuralTechnologyInferenceAdvisor(DefaultInferenceAdvisor):
    """Neural Technology Inference Advisor."""

    @classmethod
    def name(cls) -> str:
        """Return name of the advisor."""
        return "neural_technology_inference_advisor"

    def get_collectors(self, context: Context) -> list[DataCollector]:
        """Return data collectors for model or profiling-data analysis."""
        backend = self._get_backends(context)[0]
        profiling_data = self._get_optional_paths(context, "profiling_data")
        if profiling_data is not None:
            return self._get_profiling_data_collectors(context, backend, profiling_data)

        model = self.get_model(context)
        target_cfg = self._get_target_cfg(context)
        collectors: list[DataCollector] = []
        if context.category_enabled(AdviceCategory.PERFORMANCE):
            collectors.append(NeuralTechnologyPerformance(model, target_cfg, backend))
        if context.category_enabled(AdviceCategory.COMPATIBILITY):
            collectors.append(NeuralTechnologyCompatibility(model, target_cfg))
        return collectors

    def _get_profiling_data_collectors(
        self, context: Context, backend: str, profiling_data: list[Path]
    ) -> list[DataCollector]:
        """Configure measured profiling data collection."""
        if backend != PROFILING_DATA_BACKEND:
            raise ConfigurationError(
                f"Backend '{backend}' does not support Neural Technology profiling data."
            )
        if context.category_enabled(AdviceCategory.COMPATIBILITY):
            raise ConfigurationError(
                "Backend does not support --compatibility with --profiling-data. "
                "Use --performance instead."
            )
        if not context.category_enabled(AdviceCategory.PERFORMANCE):
            raise ConfigurationError(
                "Neural Technology profiling data currently supports performance "
                "analysis only."
            )

        model = self._get_optional_path(context, "model")
        if model is not None and model.suffix.lower() != ".vgf":
            raise ConfigurationError(
                "Neural Technology profiling data can only be associated with a VGF model."
            )

        backend_options = self._get_backend_options(context).get(backend, {})
        if backend_options:
            options = summarize_list(sorted(backend_options))
            raise ConfigurationError(
                "Neural Technology profiling data does not support backend "
                f"options: {options}."
            )

        return [
            NeuralTechnologyProfilingData(
                profiling_data=profiling_data,
                target_profile=self.get_target_profile(context),
                model=model,
            )
        ]

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

    def _get_optional_path(self, context: Context, name: str) -> Path | None:
        """Return an optional configured filesystem path."""
        value = self.get_parameter(
            self.name(), name, expected=False, expected_type=str, context=context
        )
        if value is None:
            return None
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        return path

    def _get_optional_paths(self, context: Context, name: str) -> list[Path] | None:
        """Return optional configured filesystem paths in caller order."""
        values = self.get_parameter(
            self.name(), name, expected=False, expected_type=list, context=context
        )
        if values is None:
            return None
        paths = [Path(value) for value in values]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path {path} does not exist.")
        return paths


def configure_and_get_neural_technology_advisor(
    context: ExecutionContext,
    target_profile: str | Path,
    model: str | Path | None,
    **extra_args: Any,
) -> InferenceAdvisor:
    """Create and configure Neural Technology advisor."""
    if context.config_parameters is None:
        context.config_parameters = _get_config_parameters(
            model, target_profile, **extra_args
        )
    return NeuralTechnologyInferenceAdvisor()


def _get_config_parameters(
    model: str | Path | None,
    target_profile: str | Path,
    **extra_args: Any,
) -> dict[str, Any]:
    """Get configuration parameters for the advisor."""
    parameters: dict[str, Any] = {"target_profile": target_profile}
    if model is not None:
        parameters["model"] = str(model)
    if profiling_data := extra_args.get("profiling_data"):
        parameters["profiling_data"] = [str(path) for path in profiling_data]

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
