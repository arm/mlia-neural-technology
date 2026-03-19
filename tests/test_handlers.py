# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology event handlers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.ml_sdk_model_converter.compat import NXModelCompatibilityInfo
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXPerformanceDatabaseParser,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.backend.nx_performance_estimator.statistics import NXOperatorPerformanceStats
from mlia.core.context import ExecutionContext
from mlia.core.events import CollectedDataEvent, ExecutionStartedEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.events import NeuralTechnologyAdvisorStartedEvent
from mlia.target.neural_technology.handlers import NeuralTechnologyEventHandler


@pytest.mark.parametrize(
    "handler, event",
    [
        (
            NeuralTechnologyEventHandler(),
            CollectedDataEvent(
                NXPerformanceEstimatorPerformanceMetrics(
                    backend_config=NXPerformanceEstimatorConfig(
                        system_config="", compiler_config=""
                    ),
                    performance_db_parser=NXPerformanceDatabaseParser(),
                    stripe_performance_metrics={
                        "op0": MagicMock(spec=NXOperatorPerformanceStats)
                    },
                    chain_performance_metrics={
                        "chain0": MagicMock(spec=NXOperatorPerformanceStats)
                    },
                    model_performance_stats=MagicMock(),
                )
            ),
        ),
        (
            NeuralTechnologyEventHandler(),
            CollectedDataEvent(NXModelCompatibilityInfo()),
        ),
    ],
)
def test_data_collection_event(
    handler: WorkflowEventsHandler, event: CollectedDataEvent
) -> None:
    """Coverage for the on_collected_data function."""
    handler.set_context(ExecutionContext())
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_collected_data(event)


def test_neural_technology_advisor_started(test_tflite_model: Path) -> None:
    """Coverage for the on_neural_technology_advisor_started function."""
    advisor_event = NeuralTechnologyAdvisorStartedEvent(
        model=test_tflite_model,
        device=NeuralTechnologyConfiguration(target="neural-technology"),
    )
    handler = NeuralTechnologyEventHandler()
    handler.set_context(ExecutionContext())
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_neural_technology_advisor_started(advisor_event)
