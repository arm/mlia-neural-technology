# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ML SDK model converter compatibility output."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlia.core.output_schema as schema
import pytest
from mlia.backend.ml_sdk_model_converter.compat import (
    NXCompatibilityChecker,
    NXModelCompatibilityInfo,
    VGFModel,
)
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOp, TosaOpType
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.core.output_validation import validate_standardized_output


def _model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "model.tosamlir"
    model_path.write_bytes(b"test model content")
    return model_path


def _result_metrics(output: dict) -> dict[str, dict]:
    return {metric["name"]: metric for metric in output["results"][0]["metrics"]}


def test_compatibility_output_reports_nx_operator_percentage(tmp_path: Path) -> None:
    """Only NX placements count toward the accelerator operator percentage."""
    compatibility = NXModelCompatibilityInfo()
    compatibility.add_lowered_to_tosa(TosaOp("ADD", "/0_nx", TosaOpType.INT))
    compatibility.add_lowered_to_tosa(TosaOp("CONV2D", "/1_nx", TosaOpType.INT))
    compatibility.add_lowered_to_tosa(
        TosaOp("tosa.custom", "/2_shader_custom", TosaOpType.INT)
    )
    compatibility.add_lowered_to_tosa(TosaOp("MUL", "/3_shader_fp", TosaOpType.FLOAT))
    compatibility.add_lowering_error("/4_unsupported", "unsupported operation")

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    result = output["results"][0]
    metrics = _result_metrics(output)
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "value": 40.0,
        "unit": schema.UNIT_PERCENT,
    }
    assert [entity["placement"] for entity in result["entities"]] == [
        "nx",
        "nx",
        "ee",
        "ee",
        "cpu",
    ]
    assert result["entity_kinds"] == [{"id": "operator"}]
    assert [check["entity_id"] for check in result["checks"]] == [
        entity["id"] for entity in result["entities"]
    ]
    assert [check["status"] for check in result["checks"]] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "fail",
    ]
    assert result["status"] == "partial"
    validate_standardized_output(output)


def test_compatibility_output_reports_zero_percentage_for_only_shader_records(
    tmp_path: Path,
) -> None:
    """Shader-compatible records are successful but not accelerator-placed."""
    compatibility = NXModelCompatibilityInfo()
    compatibility.add_lowered_to_tosa(
        TosaOp("tosa.custom", "/0_shader_custom", TosaOpType.INT)
    )
    compatibility.add_lowered_to_tosa(TosaOp("ADD", "/1_shader_fp", TosaOpType.FLOAT))

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    metrics = _result_metrics(output)
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "value": 0.0,
        "unit": schema.UNIT_PERCENT,
    }
    assert output["results"][0]["status"] == "ok"


def test_add_estimator_placement() -> None:
    """Test add_estimator_placement method."""
    invalid_compatibility = NXModelCompatibilityInfo()
    with pytest.raises(ValueError, match="Unsupported estimator placement 'CPU'"):
        invalid_compatibility.add_estimator_placement("op_id_3_0", "loc", "CPU")

    compatibility = NXModelCompatibilityInfo()
    compatibility.add_estimator_placement("op_id_1_0", "loc0", "NX", op_type="Add")
    compatibility.add_estimator_placement("op_id_2_0", "loc1", "EE")

    assert compatibility.dump() == [
        {
            "compat_level": "TOSA",
            "location": "loc0",
            "placement": "NX",
            "type": "Add",
        },
        {
            "compat_level": "Shader",
            "location": "loc1",
            "placement": "EE",
        },
    ]


def test_compatibility_output_marks_operator_percentage_unavailable_without_records(
    tmp_path: Path,
) -> None:
    """Empty compatibility output has no placement data to summarize."""
    compatibility = NXModelCompatibilityInfo()

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    metrics = _result_metrics(output)
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "unit": schema.UNIT_PERCENT,
        "availability": "unavailable",
        "reason": "Accelerator operator placement data is not available.",
    }
    validate_standardized_output(output)


def test_tosa_unique_location_uses_explicit_unknown_prefix_for_empty_locations(
    tmp_path: Path,
) -> None:
    """Location-less public converter MLIR should not produce bare '_<id>' names."""
    checker = NXCompatibilityChecker(tmp_path)

    assert checker._get_tosa_unique_location("", 97) == "unknown_00097"
    assert checker._get_tosa_unique_location("unknown", 97) == "unknown_00097"


def test_vgf_compatibility_uses_estimator_debug_database(
    tmp_path: Path, monkeypatch
) -> None:
    """VGF compatibility should be recovered from debug database."""
    vgf_path = tmp_path / "model.vgf"
    vgf_path.write_text("vgf", encoding="utf-8")
    debug_db_path = tmp_path / "model_debug_database.dat"
    debug_db_path.touch()

    def fake_gc_run(output_root, backend_config, vgf_file, output_name):
        assert output_root == tmp_path
        assert backend_config.system_config == NXPerformanceEstimatorConfig.DEFAULT
        assert backend_config.compiler_config == NXPerformanceEstimatorConfig.DEFAULT
        assert vgf_file == vgf_path
        assert output_name == "model"
        return SimpleNamespace(debug_database=debug_db_path)

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.run_nx_performance_estimator",
        fake_gc_run,
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        "NXDebugDatabaseParser.parse_debug_database",
        lambda _self: {
            "tosa_op_id_to_api_labels": {"1": ["add0", "add1"]},
            "tosa_op_id_to_tosa_op": {"1": ["Add"]},
            "shader_op_id_to_api_labels": {"2": ["fp_add0", "fp_add1"]},
        },
    )

    checker = NXCompatibilityChecker(
        tmp_path,
        {
            "nx-performance-estimator": {
                "system_config": "",
                "compiler_config": "",
            }
        },
    )

    assert checker.check_compatibility(VGFModel(vgf_path)).dump() == [
        {
            "compat_level": "TOSA",
            "location": "add0",
            "placement": "NX",
            "type": "Add",
        },
        {
            "compat_level": "TOSA",
            "location": "add1",
            "placement": "NX",
            "type": "Add",
        },
        {
            "compat_level": "Shader",
            "location": "fp_add0",
            "placement": "EE",
        },
        {
            "compat_level": "Shader",
            "location": "fp_add1",
            "placement": "EE",
        },
    ]
