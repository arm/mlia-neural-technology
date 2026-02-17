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
    PT2Model,
    TOSAModel,
    VGFModel,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceEstimator,
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.nn.tensorflow.tflite_graph import operator_names_to_types
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.utils.filesystem import is_pytorch_file, is_tosa_file, is_vgf_file
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


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
        if not any(
            [
                is_tflite_model(self.model),
                is_tosa_file(self.model),
                is_vgf_file(self.model),
                is_pytorch_file(self.model),
            ]
        ):
            raise ConfigurationError(
                "Input must be a TFLite, TOSA, VGF or PyTorch file."
            )

        if is_tflite_model(self.model):
            operator_types_mapping, _ = operator_names_to_types(model_path=self.model)
        else:
            operator_types_mapping = {}

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
            cli_args = [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []

            # Build target configuration
            target_config = {
                "target": self.cfg.target,
                "profile_name": self.cfg.target,
            }

            standardized = metrics.to_standardized_output(
                model_path=self.model,
                backend_name=self.backend,
                target_config=target_config,
                cli_arguments=cli_args,
            )

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
        model: Path | TOSAModel | VGFModel | PT2Model | None = None
        if is_tflite_model(self.model):
            model = self.model
        elif is_vgf_file(self.model):
            model = VGFModel(self.model)
        elif is_tosa_file(self.model):
            model = TOSAModel(self.model)
        elif is_pytorch_file(self.model):
            model = PT2Model(self.model)
        else:
            raise ConfigurationError(
                "Input must be a TFLite, TOSA, VGF or PyTorch file."
            )

        checker = NXCompatibilityChecker(self.context.output_dir)

        comp_info = checker.check_compatibility(model)

        # Generate standardized output
        try:
            # Clean CLI arguments to use basename for executable
            cli_args = [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []

            # Build target configuration
            target_config = {
                "target": self.cfg.target,
                "profile_name": self.cfg.target,
            }

            standardized = comp_info.to_standardized_output(
                model_path=self.model,
                target_config=target_config,
                backend_config=self.cfg.backend_config,
                cli_arguments=cli_args,
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
