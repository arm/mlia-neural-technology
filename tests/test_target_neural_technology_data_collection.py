# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology data collection."""

from __future__ import annotations

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
    NXCompatibilityResult,
    NXPerformanceResult,
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
            "test_keras_model",
            "nx-performance-estimator",
            pytest.raises(
                ConfigurationError,
                match="Input must be a TOSA, VGF, TFLite or PyTorch file.",
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
    try:
        from mlia.nn.tensorflow import tflite_graph
    except ImportError:
        pytest.skip("mlia.nn.tensorflow.tflite_graph not available")

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
        tflite_graph,
        "operator_names_to_types",
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
                match="Input must be a TOSA, VGF, TFLite or PyTorch file.",
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

    def _fake_converter(
        _name: str,
        model: Path,
        output_dir: Path,
        *,
        enable_quantization: bool | None = None,
    ) -> Path:
        del enable_quantization
        output_dir.mkdir(exist_ok=True)
        tosa_path = output_dir / f"{model.stem}.tosa"
        tosa_path.touch()
        return tosa_path

    monkeypatch.setattr(
        "mlia.target.neural_technology.data_collection.run_named_converter",
        _fake_converter,
    )

    model = request.getfixturevalue(model_fixture)
    ntc = NeuralTechnologyCompatibility(
        model,
        NeuralTechnologyConfiguration(
            target="neural-technology",
            backend_config={
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


def test_neural_technology_performance_collect_data_preserves_profile_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Performance wrapper should keep the selected profile name in API output."""
    model = tmp_path / "model.tosa"
    model.write_text("tosa", encoding="utf-8")

    captured: dict[str, Any] = {}

    class FakeMetrics:
        def to_standardized_output(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"schema_version": "1.0.0", "results": [{}]}

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorPerformanceEstimator.estimate",
        MagicMock(return_value=FakeMetrics()),
    )
    monkeypatch.setattr("sys.argv", ["/tmp/mlia-script", "--flag"])

    collector = NeuralTechnologyPerformance(
        model,
        NeuralTechnologyConfiguration(
            target="neural-technology",
            profile_name="NX-peak-12SC-8NX-600MHz",
            backend_config={
                "nx-performance-estimator": {
                    "system_config": "",
                    "compiler_config": "",
                }
            },
        ),
        "nx-performance-estimator",
    )
    collector.set_context(ExecutionContext(output_dir=tmp_path))

    result = collector.collect_data()

    assert isinstance(result, NXPerformanceResult)
    assert captured["target_config"] == {
        "target": "neural-technology",
        "target_type": "neural-technology",
        "profile_name": "NX-peak-12SC-8NX-600MHz",
    }
    assert captured["cli_arguments"] == ["mlia-script", "--flag"]


def test_neural_technology_compatibility_collect_data_preserves_profile_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility wrapper should keep the selected profile name in API output."""
    model = tmp_path / "model.vgf"
    model.write_text("vgf", encoding="utf-8")

    captured: dict[str, Any] = {}

    class FakeCompatibilityInfo:
        def to_standardized_output(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"schema_version": "1.0.0", "results": [{}]}

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        "NXCompatibilityChecker.check_compatibility",
        MagicMock(return_value=FakeCompatibilityInfo()),
    )
    monkeypatch.setattr("sys.argv", ["/tmp/mlia-api", "--quiet"])

    collector = NeuralTechnologyCompatibility(
        model,
        NeuralTechnologyConfiguration(
            target="neural-technology",
            profile_name="NX-sustained-12SC-8NX-350MHz",
            backend_config={"nx-performance-estimator": {"system_config": ""}},
        ),
    )
    collector.set_context(ExecutionContext(output_dir=tmp_path))

    result = collector.collect_data()

    assert isinstance(result, NXCompatibilityResult)
    assert captured["target_config"] == {
        "target": "neural-technology",
        "target_type": "neural-technology",
        "profile_name": "NX-sustained-12SC-8NX-350MHz",
    }
    assert captured["backend_config"] == {
        "nx-performance-estimator": {"system_config": ""}
    }
    assert captured["cli_arguments"] == ["mlia-api", "--quiet"]
