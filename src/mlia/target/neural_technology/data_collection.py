# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Data collection module for Neural Technology."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.backend.ml_sdk_model_converter.compat import NXCompatibilityChecker
from mlia.backend.ml_sdk_model_converter.compat import NXModelCompatibilityInfo
from mlia.backend.ml_sdk_model_converter.compat import TOSAModel
from mlia.backend.ml_sdk_model_converter.compat import VGFModel
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceEstimator,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.nn.tensorflow.tflite_graph import operator_names_to_types
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.utils.filesystem import is_tosa_file
from mlia.utils.filesystem import is_vgf_file
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


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
    ) -> NXPerformanceEstimatorPerformanceMetrics:
        """Run performance estimator."""
        if not any(
            [
                is_tflite_model(self.model),
                is_tosa_file(self.model),
                is_vgf_file(self.model),
            ]
        ):
            raise ConfigurationError("Input must be a TFLite, TOSA or VGF file.")

        if is_tflite_model(self.model):
            operator_types_mapping = operator_names_to_types(model_path=self.model)
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
    ) -> NXModelCompatibilityInfo:
        """Run performance estimator."""
        model: Path | TOSAModel | VGFModel | None = None
        if is_tflite_model(self.model):
            model = self.model
        elif is_vgf_file(self.model):
            model = VGFModel(self.model)
        elif is_tosa_file(self.model):
            model = TOSAModel(self.model)
        else:
            raise ConfigurationError("Input must be a TFLite, TOSA or VGF file.")

        checker = NXCompatibilityChecker(self.context.output_dir)

        comp_info = checker.check_compatibility(model)

        return comp_info

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "neural_technology_compatibility"
