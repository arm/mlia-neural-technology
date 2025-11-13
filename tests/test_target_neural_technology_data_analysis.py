# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology data analysis module."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXPerformanceDatabaseParser,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorOutputFiles,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.backend.nx_performance_estimator.statistics import NXOperatorPerformanceStats
from mlia.target.neural_technology.data_analysis import NeuralTechnologyDataAnalyzer
from mlia.target.neural_technology.data_analysis import (
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
                output_files=NXPerformanceEstimatorOutputFiles.from_output_dir(
                    Path("DOES_NOT_EXIST"), "TEST"
                ),
                performance_db_parser=NXPerformanceDatabaseParser(),
                performance_metrics={
                    "0": NXOperatorPerformanceStats(
                        op_id=[33],
                        op_cycles=15,
                        total_cycles=18,
                        memory={
                            "L1": {
                                "readBytes": 4,
                                "writeBytes": 6,
                                "trafficCycles": 43530,
                            },
                        },
                        utilization=[
                            {"sectionName": "OutputWriter", "cycles": 1},
                            {"sectionName": "VectorEngine", "cycles": 1},
                        ],
                        operators=["foo"],
                    )
                },
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
