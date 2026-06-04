# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology event handlers."""

import inspect
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
from mlia.core.events import (
    AdviceStageFinishedEvent,
    CollectedDataEvent,
    ExecutionStartedEvent,
)
from mlia.core.handlers import WorkflowEventsHandler
from mlia.core.output_schema import SCHEMA_VERSION, AdviceCategory, AdviceSeverity
from mlia.core.reporting import JSONReporter
from mlia.core.advice_generation import Advice
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.data_collection import (
    NXCompatibilityResult,
    NXPerformanceResult,
)
from mlia.target.neural_technology.events import NeuralTechnologyAdvisorStartedEvent
from mlia.target.neural_technology.handlers import NeuralTechnologyEventHandler


def _workflow_events_handler_supports_collect_only() -> bool:
    parameters = inspect.signature(WorkflowEventsHandler.__init__).parameters
    return "collect_only" in parameters


def _make_compatibility_result() -> NXCompatibilityResult:
    """Create a compatibility wrapper with minimal standardized output."""
    return NXCompatibilityResult(
        legacy_info=NXModelCompatibilityInfo(),
        standardized_output={"schema_version": SCHEMA_VERSION, "results": [{}]},
    )


def _make_performance_result() -> NXPerformanceResult:
    """Create a performance wrapper with minimal standardized output."""
    return NXPerformanceResult(
        legacy_info=NXPerformanceEstimatorPerformanceMetrics(
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
        ),
        standardized_output={"schema_version": SCHEMA_VERSION, "results": [{}]},
    )


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


def test_neural_technology_event_handler_collect_only_uses_json_reporter(
    tmp_path: Path,
) -> None:
    """Collect-only mode should use JSONReporter for in-memory output."""
    handler = NeuralTechnologyEventHandler(collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())

    assert isinstance(handler.reporter, JSONReporter)
    assert handler.collect_only is True


def test_neural_technology_event_handler_collect_only_skips_target_submission(
    tmp_path: Path,
) -> None:
    """Collect-only mode should not submit CLI-only target details."""
    handler = NeuralTechnologyEventHandler(collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    submit_mock = MagicMock(wraps=handler.reporter.submit)
    handler.on_neural_technology_advisor_started(
        NeuralTechnologyAdvisorStartedEvent(
            model=tmp_path / "model.tflite",
            device=NeuralTechnologyConfiguration(target="neural-technology"),
        )
    )
    submit_mock.assert_not_called()

    assert handler.reporter.missing_standardized_output is False
    handler.reporter.submit = submit_mock


def test_neural_technology_event_handler_collect_only_keeps_compatibility_in_memory(
    tmp_path: Path,
) -> None:
    """Collect-only mode should avoid sidecar files for compatibility output."""
    handler = NeuralTechnologyEventHandler(tmp_path, collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_collected_data(CollectedDataEvent(_make_compatibility_result()))
    handler.advice = [
        Advice(
            id="0",
            message="msg",
            severity=AdviceSeverity.INFO,
            category=AdviceCategory.COMPATIBILITY,
        )
    ]

    handler.on_advice_stage_finished(AdviceStageFinishedEvent())

    assert handler.output is not None
    assert not list(tmp_path.glob("*.json"))


def test_neural_technology_event_handler_collect_only_keeps_performance_in_memory(
    tmp_path: Path,
) -> None:
    """Collect-only mode should avoid sidecar files for performance output."""
    handler = NeuralTechnologyEventHandler(tmp_path, collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_collected_data(CollectedDataEvent(_make_performance_result()))
    handler.advice = [
        Advice(
            id="0",
            message="msg",
            severity=AdviceSeverity.INFO,
            category=AdviceCategory.PERFORMANCE,
        )
    ]

    handler.on_advice_stage_finished(AdviceStageFinishedEvent())

    assert handler.output is not None
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize(
    "data_item, advice_category, expected_category",
    [
        (
            _make_compatibility_result(),
            AdviceCategory.COMPATIBILITY,
            "compatibility",
        ),
        (
            _make_performance_result(),
            AdviceCategory.PERFORMANCE,
            "performance",
        ),
    ],
)
def test_neural_technology_event_handler_collect_only_adds_result_level_advice(
    tmp_path: Path,
    data_item: NXCompatibilityResult | NXPerformanceResult,
    advice_category: AdviceCategory,
    expected_category: str,
) -> None:
    """Collect-only output should use core result-level advice."""
    handler = NeuralTechnologyEventHandler(tmp_path, collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_collected_data(CollectedDataEvent(data_item))
    handler.advice = [
        Advice(
            id="0",
            message="msg",
            severity=AdviceSeverity.INFO,
            category=advice_category,
        )
    ]

    handler.on_advice_stage_finished(AdviceStageFinishedEvent())

    assert handler.output is not None
    result = handler.output["results"][0]
    assert "advices" not in result
    assert result["advice"] == [
        {
            "id": "0",
            "category": expected_category,
            "severity": "info",
            "message": "msg",
        }
    ]


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
