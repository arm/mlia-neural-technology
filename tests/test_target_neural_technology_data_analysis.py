# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology data analysis module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXPerformanceDatabaseParser,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.backend.nx_performance_estimator.statistics import (
    NXModelPerformanceStats,
    NXOperatorPerformanceStats,
)
from mlia.target.neural_technology.data_analysis import (
    NeuralTechnologyDataAnalyzer,
    NXPerformanceEstimatorModelPerformanceAnalyzed,
)


# mypy: disable-error-code=misc
@pytest.mark.parametrize(
    "analyzed_data",
    (
        NXPerformanceEstimatorModelPerformanceAnalyzed(
            NXPerformanceEstimatorPerformanceMetrics(
                backend_config=NXPerformanceEstimatorConfig(
                    "system.ini", "compiler.ini"
                ),
                performance_db_parser=NXPerformanceDatabaseParser(),
                stripe_performance_metrics={
                    "op0": MagicMock(spec=NXOperatorPerformanceStats)
                },
                chain_performance_metrics={
                    "chain0": MagicMock(spec=NXOperatorPerformanceStats)
                },
                model_performance_stats=MagicMock(spec=NXModelPerformanceStats),
            )
        ),
    ),
)
def test_neural_technology_data_analyzer(
    analyzed_data: NXPerformanceEstimatorModelPerformanceAnalyzed,
) -> None:
    """Test Neural Technology data analyzer."""
    analyzer = NeuralTechnologyDataAnalyzer()
    analyzer.analyze_data(analyzed_data.metrics)
    assert analyzer.get_analyzed_data() == [analyzed_data]
