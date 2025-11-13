# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Technology data analysis module."""
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor


class NeuralTechnologyDataAnalyzer(FactExtractor):
    """Neural Technology data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:  # type: ignore[override]
        """Analyse the data."""

    @analyze_data.register
    def analyze_performance_estimator_performance(
        self, data_item: NXPerformanceEstimatorPerformanceMetrics
    ) -> None:
        """Analyse operator compatibility information."""
        self.add_fact(NXPerformanceEstimatorModelPerformanceAnalyzed(data_item))


@dataclass
class NXPerformanceEstimatorModelPerformanceAnalyzed(Fact):
    """Model performance was analyzed with the Neural Accelerator Performance Estimator."""  # pylint: disable=line-too-long

    metrics: NXPerformanceEstimatorPerformanceMetrics
