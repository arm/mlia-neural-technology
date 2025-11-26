# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the various event handlers."""
from pathlib import Path

import pytest

from mlia.backend.ml_sdk_model_converter.compat import NXModelCompatibilityInfo
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
from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo
from mlia.core.context import ExecutionContext
from mlia.core.events import CollectedDataEvent
from mlia.core.events import ExecutionStartedEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.events import NeuralTechnologyAdvisorStartedEvent
from mlia.target.neural_technology.handlers import NeuralTechnologyEventHandler
from mlia.target.tosa.config import TOSAConfiguration
from mlia.target.tosa.events import TOSAAdvisorStartedEvent
from mlia.target.tosa.handlers import TOSAEventHandler


@pytest.mark.parametrize(
    "handler, event",
    [
        (
            TOSAEventHandler(),
            CollectedDataEvent(
                TOSACompatibilityInfo(tosa_compatible=True, operators=[])
            ),
        ),
        (
            TOSAEventHandler(),
            CollectedDataEvent(
                TFLiteCompatibilityInfo(
                    status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR
                )
            ),
        ),
        (
            NeuralTechnologyEventHandler(),
            CollectedDataEvent(
                NXPerformanceEstimatorPerformanceMetrics(
                    backend_config=NXPerformanceEstimatorConfig(
                        system_config="", compiler_config=""
                    ),
                    output_files=NXPerformanceEstimatorOutputFiles(
                        debug_database=Path(), performance_database=Path()
                    ),
                    performance_db_parser=NXPerformanceDatabaseParser(),
                    performance_metrics={},
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


def test_tosa_advisor_started(test_tflite_model: Path) -> None:
    """Coverage for the on_tosa_advisor_started function."""
    advisor_event = TOSAAdvisorStartedEvent(
        model=test_tflite_model,
        target=TOSAConfiguration(target="tosa"),
        tosa_metadata=None,
    )
    handler = TOSAEventHandler()
    handler.set_context(ExecutionContext())
    handler.on_execution_started(ExecutionStartedEvent())
    with pytest.raises(RuntimeError):
        handler.on_tosa_advisor_started(advisor_event)


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
