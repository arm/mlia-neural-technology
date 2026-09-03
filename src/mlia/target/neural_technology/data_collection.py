# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Data collection module for Neural Technology."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlia.backend.ml_sdk_model_converter.compat import (
    NXCompatibilityChecker,
    NXModelCompatibilityInfo,
    TOSAModel,
    VGFModel,
)
from mlia.backend.ml_sdk_model_converter.conversion import (
    get_front_end_output_subdir,
    transform_front_end_model,
)
from mlia.backend.neural_technology_profiling_data.profiling import (
    analyze_profiling_data,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceEstimator,
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.nx_utils.filesystem import (
    is_pte_file,
    is_pytorch_file,
    is_tosa_file,
    is_vgf_file,
)
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)

PERFORMANCE_ADVICE_MESSAGE = (
    "Please refer to the performance metrics shown in the report "
    "to find possible optimizations."
)
TFLITE_SOURCE_ATTRIBUTION_WARNING = (
    "Source operator attribution is unavailable for TFLite input converted to TOSA "
    "because tosa-converter-for-tflite does not preserve the exact original TFLite "
    "(subgraph_index, operator_index) provenance."
)


def _cli_arguments() -> list[str]:
    """Return CLI arguments without exposing the executable's parent path."""
    if not sys.argv:
        return []
    return [Path(sys.argv[0]).name, *sys.argv[1:]]


def _add_performance_advice(output: dict[str, Any]) -> None:
    """Add Neural Technology advice directly to each performance result."""
    for result in output.get("results", []):
        if not isinstance(result, dict) or result.get("kind") != "performance":
            continue
        result.setdefault("advice", []).append(
            {
                "id": "performance_metrics",
                "category": "performance",
                "severity": "info",
                "message": PERFORMANCE_ADVICE_MESSAGE,
            }
        )


def _is_tflite_file(model: Path) -> bool:
    return model.suffix == ".tflite"


def _is_supported_input_model(model: Path) -> bool:
    return any(
        [
            _is_tflite_file(model),
            is_tosa_file(model),
            is_vgf_file(model),
            is_pytorch_file(model),
            is_pte_file(model),
        ]
    )


def _get_front_end_output_dir(base_output_dir: Path, model: Path) -> Path:
    output_subdir = get_front_end_output_subdir(model)
    if output_subdir is None:
        return base_output_dir

    output_dir = base_output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _wrap_compatibility_model(model: Path) -> TOSAModel | VGFModel:
    if is_tosa_file(model):
        return TOSAModel(model)
    if is_vgf_file(model):
        return VGFModel(model)

    raise ConfigurationError(
        "Model conversion frontend output must be a TOSA or VGF file."
    )


@dataclass
class NXCompatibilityResult:
    """Wrapper for NX compatibility info with both legacy and standardized output."""

    legacy_info: NXModelCompatibilityInfo
    standardized_output: dict[str, Any] | None = None


@dataclass
class NXPerformanceResult:
    """Wrapper for NX performance metrics with both legacy and standardized output."""

    legacy_info: NXPerformanceEstimatorPerformanceMetrics
    standardized_output: dict[str, Any] | None = None


@dataclass
class NXProfilingDataResult:
    """Standardized output produced from measured profiling data."""

    standardized_output: dict[str, object]


class NeuralTechnologyProfilingData(ContextAwareDataCollector):
    """Collect measured Neural Technology profiling data."""

    def __init__(
        self,
        profiling_data: list[Path],
        target_profile: str | Path,
        model: Path | None,
    ) -> None:
        """Initialize the profiling data collector."""
        self.profiling_data = profiling_data
        self.target_profile = target_profile
        self.model = model

    def collect_data(self) -> NXProfilingDataResult:
        """Analyze profiling data through the normal MLIA collection workflow."""
        cli_args = _cli_arguments()
        standardized_output = analyze_profiling_data(
            target_profile=str(self.target_profile),
            profiling_data=self.profiling_data,
            categories={"performance"},
            model=str(self.model) if self.model is not None else None,
            cli_arguments=cli_args,
            output_dir=self.context.output_dir,
        )
        return NXProfilingDataResult(standardized_output=standardized_output)

    @classmethod
    def name(cls) -> str:
        """Return the collector name."""
        return "neural_technology_profiling_data"


class NeuralTechnologyPerformance(ContextAwareDataCollector):
    """Collect performance information."""

    def __init__(
        self, model: Path, cfg: NeuralTechnologyConfiguration, backend: str
    ) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg
        self.backend = backend

    def collect_data(
        self,
    ) -> NXPerformanceResult | NXPerformanceEstimatorPerformanceMetrics:
        """Run performance estimator."""
        if not _is_supported_input_model(self.model):
            raise ConfigurationError(
                "Input must be a TOSA, VGF, TFLite, PyTorch or PTE file."
            )
        operator_types_mapping: dict[str, str] = {}

        estimator: NXPerformanceEstimatorPerformanceEstimator
        if self.backend == "nx-performance-estimator":
            estimator = NXPerformanceEstimatorPerformanceEstimator(
                self.context.output_dir, self.cfg.backend_config, operator_types_mapping
            )
        else:
            raise ValueError(
                f"Backend '{self.backend}' is not supported for "
                f"target '{self.cfg.target}'."
            )

        with log_action("Checking performance..."):
            metrics = estimator.estimate(self.model)

        # Generate standardized output
        try:
            # Clean CLI arguments to use basename for executable
            cli_args = _cli_arguments()

            # Build target configuration
            target_config = {
                "target": self.cfg.target,
                "target_type": self.cfg.target,
                "profile_name": self.cfg.profile_name,
            }

            standardized = metrics.to_standardized_output(
                model_path=self.model,
                backend_name=self.backend,
                target_config=target_config,
                cli_arguments=cli_args,
            )
            _add_performance_advice(standardized)

            return NXPerformanceResult(
                legacy_info=metrics,
                standardized_output=standardized,
            )
        except Exception as exc:
            logger.warning(
                "Failed to generate standardized output for NX performance: %s",
                exc,
            )
            # Fall back to legacy metrics on any error
            return metrics

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "neural_technology_performance"


class NeuralTechnologyCompatibility(ContextAwareDataCollector):
    """Collect compatibility information."""

    def __init__(self, model: Path, cfg: NeuralTechnologyConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg

    def collect_data(
        self,
    ) -> NXCompatibilityResult | NXModelCompatibilityInfo:
        """Run performance estimator."""
        if not _is_supported_input_model(self.model):
            raise ConfigurationError(
                "Input must be a TOSA, VGF, TFLite, PyTorch or PTE file."
            )

        model: TOSAModel | VGFModel
        is_tflite = _is_tflite_file(self.model)
        if is_vgf_file(self.model):
            model = VGFModel(self.model)
        else:
            converted_model_path = transform_front_end_model(
                self.model,
                _get_front_end_output_dir(self.context.output_dir, self.model),
                enable_quantization=self.cfg.backend_config.get(
                    "nx-performance-estimator", {}
                ).get("enable_quantization", True),
                output_format="mlir-text" if is_tflite else None,
                emit_debug_info=True if is_tflite else None,
            )
            model = _wrap_compatibility_model(converted_model_path)

        checker = NXCompatibilityChecker(
            self.context.output_dir, self.cfg.backend_config
        )

        comp_info = checker.check_compatibility(model)

        # Generate standardized output
        try:
            # Clean CLI arguments to use basename for executable
            cli_args = _cli_arguments()

            # Build target configuration
            target_config = {
                "target": self.cfg.target,
                "target_type": self.cfg.target,
                "profile_name": self.cfg.profile_name,
            }

            # tosa-converter-for-tflite generates TOSA operator IDs but does not
            # retain the original TFLite indices needed for truthful attribution.
            # Preserve compatibility checks while deliberately withholding links.
            standardized = comp_info.to_standardized_output(
                model_path=self.model,
                target_config=target_config,
                backend_config=self.cfg.backend_config,
                cli_arguments=cli_args,
                source_operator_attribution_available=not is_tflite,
                warnings=[TFLITE_SOURCE_ATTRIBUTION_WARNING] if is_tflite else None,
            )

            return NXCompatibilityResult(
                legacy_info=comp_info,
                standardized_output=standardized,
            )
        except Exception as exc:
            logger.warning(
                "Failed to generate standardized output for NX compatibility: %s",
                exc,
            )
            # Fall back to legacy info on any error
            return comp_info

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "neural_technology_compatibility"
