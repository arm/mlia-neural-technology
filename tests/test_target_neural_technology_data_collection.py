# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology data collection."""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import mlia.core.output_schema as schema
from mlia.backend.ml_sdk_model_converter.compat import NXModelCompatibilityInfo
from mlia.backend.ml_sdk_model_converter.conversion import get_front_end_output_subdir
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOp, TosaOpType
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
from mlia.core.output_validation import validate_standardized_output
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.data_collection import (
    NXCompatibilityResult,
    NXPerformanceResult,
    NeuralTechnologyCompatibility,
    NeuralTechnologyPerformance,
    TFLITE_SOURCE_ATTRIBUTION_WARNING,
)


@pytest.fixture(scope="session", name="test_tosa_model")
def fixture_test_tflite_no_act_model(test_models_path: Path) -> Path:
    """Return test TOSA model."""
    # For whatever reason defining this in conftest.py would remove the no act TFLite
    # model somehow
    return test_models_path / "model.tosa"


def _neural_technology_config(
    *,
    system_config: str = "",
    compiler_config: str | None = "",
    profile_name: str = "neural-technology",
) -> NeuralTechnologyConfiguration:
    backend_config = {"system_config": system_config}
    if compiler_config is not None:
        backend_config["compiler_config"] = compiler_config

    return NeuralTechnologyConfiguration(
        target="neural-technology",
        profile_name=profile_name,
        backend_config={"nx-performance-estimator": backend_config},
    )


@pytest.mark.parametrize(
    ("model_name", "expected_subdir"),
    [
        ("model.tflite", "tflite-to-tosa"),
        ("model.pt2", "pt2-to-tosa"),
        ("model.pte", "pte-to-delegate"),
        ("model.tosa", None),
    ],
)
def test_get_front_end_output_subdir_returns_shared_conversion_location(
    tmp_path: Path,
    model_name: str,
    expected_subdir: str | None,
) -> None:
    """Frontend conversions should resolve output dirs from one shared helper."""
    assert get_front_end_output_subdir(tmp_path / model_name) == expected_subdir


@pytest.mark.parametrize(
    "model_fixture, backend, expectation",
    [
        ("test_tflite_model", "nx-performance-estimator", does_not_raise()),
        ("test_tosa_model", "nx-performance-estimator", does_not_raise()),
        ("test_pte_model", "nx-performance-estimator", does_not_raise()),
        (
            "test_keras_model",
            "nx-performance-estimator",
            pytest.raises(
                ConfigurationError,
                match="Input must be a TOSA, VGF, TFLite, PyTorch or PTE file.",
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
                cascade_performance_metrics={
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
        _neural_technology_config(),
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
        ("test_pte_model", does_not_raise()),
        (
            "test_keras_model",
            pytest.raises(
                ConfigurationError,
                match="Input must be a TOSA, VGF, TFLite, PyTorch or PTE file.",
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
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        + "NXCompatibilityChecker.check_compatibility",
        mock_check_compatibility,
    )

    def _fake_transform_front_end_model(
        model_path: Path,
        output_dir: Path,
        *,
        enable_quantization: bool | None = None,
        output_format: str | None = None,
        emit_debug_info: bool | None = None,
    ) -> Path:
        captured["model_path"] = model_path
        captured["output_dir"] = output_dir
        captured["enable_quantization"] = enable_quantization
        captured["output_format"] = output_format
        captured["emit_debug_info"] = emit_debug_info
        suffix = ".vgf" if model_path.suffix == ".pte" else ".tosa"
        output_path = output_dir / f"{model_path.stem}{suffix}"
        output_path.touch()
        return output_path

    monkeypatch.setattr(
        "mlia.target.neural_technology.data_collection.transform_front_end_model",
        _fake_transform_front_end_model,
    )

    model = request.getfixturevalue(model_fixture)
    ntc = NeuralTechnologyCompatibility(
        model,
        _neural_technology_config(
            system_config=NeuralTechnologyCompatibility.name(),
        ),
    )
    ntc.set_context(ExecutionContext(output_dir=tmp_path))

    with expectation:
        ntc.collect_data()
        mock_check_compatibility.assert_called_once()
        if model_fixture == "test_tflite_model":
            assert captured["model_path"] == model
            assert captured["output_dir"] == ntc.context.output_dir / "tflite-to-tosa"
            assert captured["output_format"] == "mlir-text"
            assert captured["emit_debug_info"] is True
        elif model_fixture == "test_pte_model":
            assert captured["model_path"] == model
            assert captured["output_dir"] == ntc.context.output_dir / "pte-to-delegate"


def test_neural_technology_compatibility_skips_frontend_conversion_for_vgf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VGF input should go straight to compatibility checking."""
    model = tmp_path / "model.vgf"
    model.write_text("vgf", encoding="utf-8")

    monkeypatch.setattr(
        "mlia.target.neural_technology.data_collection.transform_front_end_model",
        MagicMock(side_effect=AssertionError("frontend conversion should be skipped")),
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        "NXCompatibilityChecker.check_compatibility",
        MagicMock(return_value=None),
    )

    collector = NeuralTechnologyCompatibility(
        model,
        _neural_technology_config(
            system_config=NeuralTechnologyCompatibility.name(),
        ),
    )
    collector.set_context(ExecutionContext(output_dir=tmp_path))

    collector.collect_data()


def test_neural_technology_performance_accepts_pte_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Performance should pass PTE input through to the estimator."""
    model = tmp_path / "model.pte"
    model.write_bytes(b"pte")
    captured: dict[str, Any] = {}

    class FakeMetrics:
        def to_standardized_output(self, **kwargs: Any) -> dict[str, Any]:
            return {"schema_version": "1.0.0", "results": [{}], **kwargs}

    def fake_estimate(_self: object, model_path: Path) -> FakeMetrics:
        captured["model_path"] = model_path
        return FakeMetrics()

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorPerformanceEstimator.estimate",
        fake_estimate,
    )

    collector = NeuralTechnologyPerformance(
        model,
        _neural_technology_config(),
        "nx-performance-estimator",
    )
    collector.set_context(ExecutionContext(output_dir=tmp_path))

    collector.collect_data()

    assert captured["model_path"] == model


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
        _neural_technology_config(
            profile_name="custom-performance-profile",
        ),
        "nx-performance-estimator",
    )
    collector.set_context(ExecutionContext(output_dir=tmp_path))

    result = collector.collect_data()

    assert isinstance(result, NXPerformanceResult)
    assert captured["target_config"] == {
        "target": "neural-technology",
        "target_type": "neural-technology",
        "profile_name": "custom-performance-profile",
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
        _neural_technology_config(
            profile_name="custom-compatibility-profile",
            compiler_config=None,
        ),
    )
    collector.set_context(ExecutionContext(output_dir=tmp_path))

    result = collector.collect_data()

    assert isinstance(result, NXCompatibilityResult)
    assert captured["target_config"] == {
        "target": "neural-technology",
        "target_type": "neural-technology",
        "profile_name": "custom-compatibility-profile",
    }
    assert captured["backend_config"] == {
        "nx-performance-estimator": {"system_config": ""}
    }
    assert captured["cli_arguments"] == ["mlia-api", "--quiet"]
    assert captured["source_operator_attribution_available"] is True
    assert captured["warnings"] is None


def test_tflite_compatibility_uses_converted_operator_entities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Converted TFLite checks retain context without claiming source identity."""
    model = tmp_path / "model.tflite"
    model.write_bytes(b"tflite")
    converted_model = tmp_path / "converted.tosa"
    converted_model.write_bytes(b"tosa")

    compatibility = NXModelCompatibilityInfo()
    compatibility.add_lowered_to_tosa(
        TosaOp("ADD", "add_0", TosaOpType.INT),
        source_operator_id=schema.tosa_source_operator_id(0),
    )
    compatibility.add_lowering_error(
        "unsupported_1",
        "unsupported operation",
        source_operator_id=schema.tosa_source_operator_id(1),
    )

    monkeypatch.setattr(
        "mlia.target.neural_technology.data_collection.transform_front_end_model",
        MagicMock(return_value=converted_model),
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        "NXCompatibilityChecker.check_compatibility",
        MagicMock(return_value=compatibility),
    )

    collector = NeuralTechnologyCompatibility(model, _neural_technology_config())
    collector.set_context(ExecutionContext(output_dir=tmp_path))

    collected = collector.collect_data()

    assert isinstance(collected, NXCompatibilityResult)
    assert collected.standardized_output is not None
    result = collected.standardized_output["results"][0]
    assert result["warnings"] == [TFLITE_SOURCE_ATTRIBUTION_WARNING]
    assert result["entity_kinds"] == [{"id": "converted_operator"}]
    assert result["entities"] == [
        {
            "id": "converted_operator/0",
            "kind": "converted_operator",
            "name": "ADD",
            "placement": "NX",
            "attributes": {
                "index": 0,
                "compat_level": "TOSA",
                "display_label": "add_0",
                "tosa_op": "ADD",
                "converted_tosa_operator_id": "0",
            },
        },
        {
            "id": "converted_operator/1",
            "kind": "converted_operator",
            "name": "unsupported_1",
            "placement": "CPU",
            "attributes": {
                "index": 1,
                "compat_level": "Non-NX",
                "display_label": "unsupported_1",
                "converted_tosa_operator_id": "1",
            },
        },
    ]
    assert [check["entity_id"] for check in result["checks"]] == [
        "converted_operator/0",
        "converted_operator/1",
    ]
    assert [check["status"] for check in result["checks"]] == ["pass", "fail"]
    assert result["checks"][1]["details"] == {"error": "unsupported operation"}
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE]["value"] == 50.0
    validate_standardized_output(collected.standardized_output)
