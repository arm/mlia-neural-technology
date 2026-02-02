# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology MLIA module."""

import re
from pathlib import Path
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
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.target.neural_technology.advice_generation import (
    NeuralTechnologyAdviceProducer,
)
from mlia.target.neural_technology.advisor import (
    NeuralTechnologyInferenceAdvisor,
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.data_analysis import (
    NXPerformanceEstimatorModelPerformanceAnalyzed,
)


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert (
        NeuralTechnologyInferenceAdvisor.name() == "neural_technology_inference_advisor"
    )


@pytest.mark.parametrize(
    "advice_category", [(AdviceCategory.PERFORMANCE), (AdviceCategory.COMPATIBILITY)]
)
def test_configure_and_get_neural_technology_advisor(
    test_tflite_model: Path, advice_category: AdviceCategory
) -> None:
    """Test Neural Technology advisor configuration."""
    ctx = ExecutionContext(advice_category={advice_category})

    advisor = configure_and_get_neural_technology_advisor(
        ctx,
        "neural-technology",
        test_tflite_model,
        backends=["nx-performance-estimator"],
    )
    workflow = advisor.configure(ctx)

    assert isinstance(advisor, NeuralTechnologyInferenceAdvisor)

    assert ctx.event_handlers is not None
    assert ctx.config_parameters == {
        "common_optimizations": {
            "optimizations": [
                [
                    {
                        "layers_to_optimize": None,
                        "optimization_target": 0.5,
                        "optimization_type": "pruning",
                    },
                    {
                        "layers_to_optimize": None,
                        "optimization_target": 32,
                        "optimization_type": "clustering",
                    },
                ]
            ],
            "rewrite_parameters": {
                "rewrite_specific_params": None,
                "train_params": None,
            },
        },
        "neural_technology_inference_advisor": {
            "backend_options": {},
            "backends": ["nx-performance-estimator"],
            "model": str(test_tflite_model),
            "target_profile": "neural-technology",
        },
    }

    assert isinstance(workflow, DefaultWorkflowExecutor)


def test_configure_and_get_neural_technology_advisor_invalid_backends(
    test_tflite_model: Path,
) -> None:
    """
    Test for raising an error when either no or too many backends are provided to
    configure_and_get_neural_technology_advisor function.
    """
    ctx = ExecutionContext(advice_category={AdviceCategory.PERFORMANCE})

    with pytest.raises(
        ConfigurationError, match="One backend is required but was not specified."
    ):
        configure_and_get_neural_technology_advisor(
            ctx,
            "neural-technology",
            test_tflite_model,
            backends=[],
        )

    backends = ["nx-performance-estimator", "vela"]
    with pytest.raises(
        ConfigurationError,
        match=f"Only one backend is supported but {len(backends)} "
        + f"were provided: {re.escape(str(backends))}",
    ):
        configure_and_get_neural_technology_advisor(
            ctx,
            "neural-technology",
            test_tflite_model,
            backends=backends,
        )


def test_neural_technology_advice_producer_produce_advice() -> None:
    """Test the produce_advice method on the NeuralTechnologyAdviceProducer."""
    ctx = ExecutionContext(advice_category={AdviceCategory.PERFORMANCE})
    producer = NeuralTechnologyAdviceProducer()
    producer.set_context(ctx)
    producer.produce_advice(
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
        )
    )
    assert len(producer.advice) == 1
    assert producer.advice[0].message == (
        "Please refer to the performance metrics shown in the report "
        "to find possible optimizations."
    )
