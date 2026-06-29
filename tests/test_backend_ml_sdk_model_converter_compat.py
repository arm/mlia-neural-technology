# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ML SDK model converter compatibility output."""

from __future__ import annotations

from pathlib import Path

import mlia.core.output_schema as schema
from mlia.backend.ml_sdk_model_converter.compat import (
    NXCompatibilityChecker,
    NXModelCompatibilityInfo,
)
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOp, TosaOpType
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
