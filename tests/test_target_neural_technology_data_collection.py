# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology data collection."""

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any
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
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.data_collection import (
    NeuralTechnologyCompatibility,
    NeuralTechnologyPerformance,
)


@pytest.fixture(scope="session", name="test_tosa_model")
def fixture_test_tflite_no_act_model(test_models_path: Path) -> Path:
    """Return test TOSA model."""
    # For whatever reason defining this in conftest.py would remove the no act TFLite
    # model somehow
    return test_models_path / "model.tosa"


@pytest.mark.parametrize(
    "model_fixture, backend, expectation",
    [
        ("test_tflite_model", "nx-performance-estimator", does_not_raise()),
        ("test_tosa_model", "nx-performance-estimator", does_not_raise()),
        (
            "test_tflite_model",
            "cortex-a",
            pytest.raises(
                ValueError,
                match="Backend 'cortex-a' is not supported for target "
                + "'neural-technology'",
            ),
        ),
        (
            "test_keras_model",
            "nx-performance-estimator",
            pytest.raises(
                ConfigurationError,
                match="Input must be a TFLite, TOSA, VGF or PyTorch file.",
            ),
        ),
    ],
)
def test_neural_technology_performance_collect_data(
    model_fixture: str,
    backend: str,
    expectation: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Tests for the NeuralTechnologyPerformance class."""
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        + "NXPerformanceEstimatorPerformanceEstimator.estimate",
        MagicMock(
            return_value=NXPerformanceEstimatorPerformanceMetrics(
                backend_config=NXPerformanceEstimatorConfig(
                    system_config=NeuralTechnologyPerformance.name(), compiler_config=""
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
    )
    monkeypatch.setattr(
        "mlia.nn.tensorflow.tflite_graph.operator_names_to_types",
        MagicMock(return_value={}),
    )

    model = request.getfixturevalue(model_fixture)

    ntp = NeuralTechnologyPerformance(
        model,
        NeuralTechnologyConfiguration(
            target="neural-technology",
            backend_config={backend: {"system_config": "", "compiler_config": ""}},
        ),
        backend,
    )
    ntp.set_context(ExecutionContext(output_dir=tmp_path))

    with expectation:
        ntp.collect_data()


@pytest.mark.parametrize(
    "model_fixture, expectation",
    [
        ("test_tflite_model", does_not_raise()),
        ("test_tosa_mlir_model", does_not_raise()),
        ("test_vgf_model", does_not_raise()),
        (
            "test_keras_model",
            pytest.raises(
                ConfigurationError,
                match="Input must be a TFLite, TOSA, VGF or PyTorch file.",
            ),
        ),
    ],
)
def test_neural_technology_compatibility_collect_data(
    model_fixture: str,
    expectation: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Tests for the NeuralTechnologyCompatibility class."""
    mock_check_compatibility = MagicMock(return_value=None)
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        + "NXCompatibilityChecker.check_compatibility",
        mock_check_compatibility,
    )

    model = request.getfixturevalue(model_fixture)
    ntc = NeuralTechnologyCompatibility(
        model,
        NeuralTechnologyConfiguration(
            target="neural-technology",
            backend={
                "nx-performance-estimator": {
                    "system_config": NeuralTechnologyCompatibility.name(),
                    "compiler_config": "",
                }
            },
        ),
    )
    ntc.set_context(ExecutionContext(output_dir=tmp_path))

    with expectation:
        ntc.collect_data()
        mock_check_compatibility.assert_called_once()
