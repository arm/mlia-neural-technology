# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ML SDK model converter compatibility output."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlia.core.output_schema as schema
import pytest
from mlia.backend.ml_sdk_model_converter.compat import (
    NXCompatibilityChecker,
    NXModelCompatibilityInfo,
    TOSAModel,
    VGFModel,
)
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOp, TosaOpType
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.statistics import NXPerformanceStats
from mlia.backend.nx_performance_estimator.vgf import GCPEVGFSegment, GCPEVGFSegments
from mlia.core.output_validation import validate_standardized_output


def _model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "model.tosamlir"
    model_path.write_bytes(b"test model content")
    return model_path


def _result_metrics(output: dict) -> dict[str, dict]:
    return {metric["name"]: metric for metric in output["results"][0]["metrics"]}


def _assert_direct_tosa_output(output: dict) -> None:
    """Assert direct TOSA records retain source entities and placements."""
    result = output["results"][0]
    entities = result["entities"]
    expected_source_ids = [schema.tosa_source_operator_id(i) for i in range(5)]

    assert [entity["id"] for entity in entities] == expected_source_ids
    assert [entity["placement"] for entity in entities] == [
        "NX",
        "NX",
        "EE",
        "EE",
        "CPU",
    ]
    assert [check["entity_id"] for check in result["checks"]] == expected_source_ids
    assert len({entity["id"] for entity in entities}) == 5
    validate_standardized_output(output)


def _direct_tosa_ops() -> dict[int, TosaOp]:
    """Return direct flatbuffer-style operations with repeated locations."""
    return {
        0: TosaOp("ADD", "shared", TosaOpType.INT),
        1: TosaOp("CONV2D", "shared", TosaOpType.INT),
        2: TosaOp("MUL", "unknown", TosaOpType.FLOAT),
        3: TosaOp("SUB", "unknown", TosaOpType.FLOAT),
        4: TosaOp("UNSUPPORTED", "unknown", TosaOpType.INT),
    }


def test_compatibility_output_reports_nx_operator_percentage(tmp_path: Path) -> None:
    """Only NX placements count toward the accelerator operator percentage."""
    compatibility = NXModelCompatibilityInfo()
    compatibility.add_lowered_to_tosa(
        TosaOp("ADD", "/0_nx", TosaOpType.INT),
        source_operator_id=schema.tosa_source_operator_id(0),
    )
    compatibility.add_lowered_to_tosa(
        TosaOp("CONV2D", "/1_nx", TosaOpType.INT),
        source_operator_id=schema.tosa_source_operator_id(1),
    )
    compatibility.add_lowered_to_tosa(
        TosaOp("tosa.custom", "/2_shader_custom", TosaOpType.INT),
        source_operator_id=schema.tosa_source_operator_id(2),
    )
    compatibility.add_lowered_to_tosa(
        TosaOp("MUL", "/3_shader_fp", TosaOpType.FLOAT),
        source_operator_id=schema.tosa_source_operator_id(3),
    )
    compatibility.add_lowering_error(
        "/4_unsupported",
        "unsupported operation",
        source_operator_id=schema.tosa_source_operator_id(4),
    )

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    result = output["results"][0]
    metrics = _result_metrics(output)
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "value": 40.0,
        "unit": schema.UNIT_PERCENT,
    }
    _assert_direct_tosa_output(output)
    assert [check["status"] for check in result["checks"]] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "fail",
    ]
    assert result["status"] == "partial"


def test_compatibility_output_reports_zero_percentage_for_only_shader_records(
    tmp_path: Path,
) -> None:
    """Shader-compatible records are successful but not accelerator-placed."""
    compatibility = NXModelCompatibilityInfo()
    compatibility.add_lowered_to_tosa(
        TosaOp("tosa.custom", "/0_shader_custom", TosaOpType.INT),
        source_operator_id=schema.tosa_source_operator_id(0),
    )
    compatibility.add_lowered_to_tosa(
        TosaOp("ADD", "/1_shader_fp", TosaOpType.FLOAT),
        source_operator_id=schema.tosa_source_operator_id(1),
    )

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    metrics = _result_metrics(output)
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "value": 0.0,
        "unit": schema.UNIT_PERCENT,
    }
    assert output["results"][0]["status"] == "ok"


def test_compatibility_output_uses_operator_type_for_entity_name(
    tmp_path: Path,
) -> None:
    """Compatibility entities should prefer semantic operator types as names."""
    compatibility = NXModelCompatibilityInfo()
    compatibility.add_estimator_placement(
        "record_0",
        "unknown_00000",
        "NX",
        op_type="ADD",
        source_operator_id="source_operator/segment_0/spirv-1",
    )

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    entity = output["results"][0]["entities"][0]
    assert entity["name"] == "ADD"
    assert entity["attributes"]["display_label"] == "unknown_00000"


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
            "display_label": "loc0",
            "placement": "NX",
            "type": "Add",
        },
        {
            "compat_level": "Shader",
            "display_label": "loc1",
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


def test_direct_tosa_flatbuffer_compatibility_preserves_source_entities(
    tmp_path: Path, monkeypatch
) -> None:
    """The direct .tosa checker should link all placements to parser IDs."""
    model_path = tmp_path / "model.tosa"
    model_path.write_bytes(b"flatbuffer boundary is mocked")
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.read_tosa_flatbuffer_ops",
        lambda path: _direct_tosa_ops(),
    )

    compatibility = NXCompatibilityChecker(tmp_path).check_compatibility(
        TOSAModel(model_path)
    )
    output = compatibility.to_standardized_output(model_path)

    _assert_direct_tosa_output(output)
    assert [record.display_label for record in compatibility.get_records()] == [
        "shared_0",
        "shared_1",
        "unknown_00002",
        "unknown_00003",
        "unknown_00004",
    ]


def test_direct_tosa_mlir_compatibility_preserves_source_entities(
    tmp_path: Path,
) -> None:
    """The direct .tosamlir checker should use numeric SSA result IDs."""
    model_path = tmp_path / "model.tosamlir"
    model_path.write_text(
        """#loc1 = loc(\"shared\")
module {
  %0 = \"tosa.add\"() : () -> tensor<1xi8> loc(#loc1)
  %1 = \"tosa.conv2d\"() : () -> tensor<1xi8> loc(#loc1)
  %2 = \"tosa.mul\"() : () -> tensor<1xf32>
  %3 = \"tosa.sub\"() : () -> tensor<1xf32>
  %4 = \"tosa.unsupported\"() : () -> tensor<1xi8>
}
""",
        encoding="utf-8",
    )

    compatibility = NXCompatibilityChecker(tmp_path).check_compatibility(
        TOSAModel(model_path)
    )
    output = compatibility.to_standardized_output(model_path)

    _assert_direct_tosa_output(output)
    assert [record.display_label for record in compatibility.get_records()] == [
        "shared_0",
        "shared_1",
        "unknown_00002",
        "unknown_00003",
        "unknown_00004",
    ]


def test_vgf_compatibility_uses_estimator_debug_database(
    tmp_path: Path, monkeypatch
) -> None:
    """VGF compatibility should be recovered from debug database."""
    vgf_path = tmp_path / "model.vgf"
    vgf_path.write_text("vgf", encoding="utf-8")
    debug_db_path = tmp_path / "model_debug_database.dat"
    debug_db_path.touch()
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.prepare_gcpe_compatible_vgfs",
        lambda *_args: GCPEVGFSegments(
            [
                GCPEVGFSegment(
                    segment_index=0,
                    segment_name="main",
                    path=vgf_path,
                    debug_names=SpirvDebugNameMap(),
                    structural_source_operator_ids=[],
                )
            ]
        ),
    )

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
            "display_label": "add0",
            "placement": "NX",
            "type": "Add",
        },
        {
            "compat_level": "TOSA",
            "display_label": "add1",
            "placement": "NX",
            "type": "Add",
        },
        {
            "compat_level": "Shader",
            "display_label": "fp_add0",
            "placement": "EE",
        },
        {
            "compat_level": "Shader",
            "display_label": "fp_add1",
            "placement": "EE",
        },
    ]


@pytest.mark.parametrize("spirv_ids", [["60"], ["60", "62"]])
def test_vgf_compatibility_emits_shared_provenance_entities(
    tmp_path: Path, monkeypatch, spirv_ids: list[str]
) -> None:
    """Compatibility should retain source, module, and code-stack provenance."""
    vgf_path = tmp_path / "model.vgf"
    vgf_path.write_text("vgf", encoding="utf-8")
    debug_db_path = tmp_path / "model_debug_database.dat"
    debug_db_path.touch()
    debug_label = json.dumps(
        {
            "aten_info": {
                "node_name": "aten_convolution_default",
                "operator_name": "aten.convolution.default",
            },
            "torch_info": {
                "nn_module_stack": {
                    "L['model']": ("model", "Model"),
                    "L['model'].features.0": ("model.features.0", "Conv2d"),
                },
                "stack_trace": [
                    '  File "E:\\XPK\\model.py", line 214, in forward',
                    '  File "/tmp/project/layer.py", line 156, in call',
                ],
            },
        }
    )
    source_ids = [f"source_operator/segment_7/spirv-{value}" for value in spirv_ids]
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={value: debug_label for value in spirv_ids},
        debug_name_to_spirv_ids={debug_label: spirv_ids},
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.prepare_gcpe_compatible_vgfs",
        lambda *_args: GCPEVGFSegments(
            [
                GCPEVGFSegment(
                    segment_index=7,
                    segment_name="main",
                    path=vgf_path,
                    debug_names=debug_names,
                    structural_source_operator_ids=[
                        *source_ids,
                        "source_operator/segment_7/spirv-61",
                    ],
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.run_nx_performance_estimator",
        lambda *_args, **_kwargs: SimpleNamespace(debug_database=debug_db_path),
    )
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["3"]},
        "chain_op_id_to_fused_op_ids": {"3": ["2"]},
        "fused_op_id_to_tosa_op_ids": {"2": ["1"]},
        "tosa_op_id_to_api_labels": {"1": [debug_label]},
        "tosa_op_id_to_tosa_op": {"1": ["Conv2D"]},
        "shader_op_id_to_api_labels": {},
    }
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        "NXDebugDatabaseParser.parse_debug_database",
        lambda _self: debug_db,
    )

    compatibility = NXCompatibilityChecker(tmp_path).check_compatibility(
        VGFModel(vgf_path)
    )
    output = compatibility.to_standardized_output(vgf_path)

    result = output["results"][0]
    entities = {entity["id"]: entity for entity in result["entities"]}
    source_id = "source_operator/segment_7/spirv-60"
    unreported_source_id = "source_operator/segment_7/spirv-61"
    model_module_id = "nn_module/L['model']"
    layer_module_id = "nn_module/L['model'].features.0"
    frame_entities = [
        entity for entity in entities.values() if entity["kind"] == "code_stack"
    ]
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=[],
        segment_index=7,
        debug_names=debug_names,
    )
    _, performance_locations, _, _, _ = performance_stats.track_op("0")

    assert source_ids == performance_locations[0]
    if len(source_ids) > 1:
        # A shared source label supports performance grouping, not a claim
        # about each candidate's individual placement or operation type.
        assert len(result["checks"]) == 1
        assert "entity_id" not in result["checks"][0]
        for resolved_id in source_ids:
            assert "placement" not in entities[resolved_id]
            assert "attributes" not in entities[resolved_id]
        validate_standardized_output(output)
        return
    assert {check.get("entity_id") for check in result["checks"]} == set(source_ids)
    for resolved_id in source_ids:
        assert entities[resolved_id]["name"] == "aten.convolution.default"
        assert entities[resolved_id]["placement"] == "NX"
    assert result["entity_kinds"] == [
        {
            "id": "nn_module",
            "parent_kinds": ["nn_module"],
            "child_kinds": ["nn_module", "source_operator"],
        }
    ]
    assert entities[source_id]["name"] == "aten.convolution.default"
    assert entities[source_id]["placement"] == "NX"
    assert entities[source_id]["parent_ids"] == [
        layer_module_id,
        frame_entities[-1]["id"],
    ]
    assert entities["nn_module/<root>"]["child_ids"] == [model_module_id]
    assert entities[model_module_id]["child_ids"] == [layer_module_id]
    assert entities[layer_module_id]["child_ids"] == source_ids
    assert [entity["name"] for entity in frame_entities] == [
        "model.py:214",
        "layer.py:156",
    ]
    assert frame_entities[0]["child_ids"] == [frame_entities[1]["id"]]
    assert frame_entities[1]["child_ids"] == source_ids
    assert result["checks"][0]["entity_id"] == source_id
    assert entities[unreported_source_id] == {
        "id": unreported_source_id,
        "kind": "source_operator",
        "name": unreported_source_id,
    }
    assert unreported_source_id not in {
        check.get("entity_id") for check in result["checks"]
    }
    validate_standardized_output(output)


@pytest.mark.parametrize("shader_label", ["shared", "shader"])
def test_shared_provenance_does_not_multiply_placement_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shader_label: str
) -> None:
    """Shared labels must not skew percentages or assign conflicting placements."""
    model_path = tmp_path / "model.vgf"
    model_path.write_bytes(b"model")
    (tmp_path / "debug.dat").touch()
    names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"60": "shared", "61": "shared", "62": "shader"},
        debug_name_to_spirv_ids={"shared": ["60", "61"], "shader": ["62"]},
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.prepare_gcpe_compatible_vgfs",
        lambda *_args: GCPEVGFSegments(
            [
                GCPEVGFSegment(
                    segment_index=0,
                    segment_name="main",
                    path=model_path,
                    debug_names=names,
                    structural_source_operator_ids=[
                        f"source_operator/segment_0/spirv-{value}"
                        for value in ["60", "61", "62"]
                    ],
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.run_nx_performance_estimator",
        lambda *_args, **_kwargs: SimpleNamespace(
            debug_database=tmp_path / "debug.dat"
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.NXDebugDatabaseParser.parse_debug_database",
        lambda _self: {
            "tosa_op_id_to_api_labels": {"1": ["shared"], "2": ["shared"]},
            "tosa_op_id_to_tosa_op": {"1": ["Conv2D"], "2": ["Rescale"]},
            "shader_op_id_to_api_labels": {"3": [shader_label]},
        },
    )
    compatibility = NXCompatibilityChecker(tmp_path).check_compatibility(
        VGFModel(model_path)
    )
    assert len(compatibility.get_records()) == 3
    output = compatibility.to_standardized_output(model_path)
    validate_standardized_output(output)
    result = output["results"][0]
    metric = next(
        item
        for item in result["metrics"]
        if item["name"] == "accelerator_operator_percentage"
    )
    assert metric["value"] == pytest.approx(100 * 2 / 3)
    assert len(result["checks"]) == 3
    for entity in result["entities"]:
        if entity["id"] in [
            "source_operator/segment_0/spirv-60",
            "source_operator/segment_0/spirv-61",
        ]:
            assert "placement" not in entity
            assert "attributes" not in entity
    assert all("entity_id" not in check for check in result["checks"][:2])


def test_vgf_compatibility_preserves_segment_scoped_source_ids(
    tmp_path: Path, monkeypatch
) -> None:
    """Multi-segment compatibility should use performance's canonical identities."""
    vgf_path = tmp_path / "model.vgf"
    vgf_path.write_text("vgf", encoding="utf-8")
    labels = [
        json.dumps({"aten_info": {"node_name": "shared", "operator_name": "Add"}}),
        json.dumps({"aten_info": {"node_name": "shared", "operator_name": "Mul"}}),
    ]
    segments = GCPEVGFSegments(
        [
            GCPEVGFSegment(
                segment_index=1,
                segment_name="left",
                path=tmp_path / "left.vgf",
                debug_names=SpirvDebugNameMap(
                    spirv_id_to_debug_name={"10": labels[0]},
                    debug_name_to_spirv_ids={labels[0]: ["10"]},
                ),
                structural_source_operator_ids=[],
            ),
            GCPEVGFSegment(
                segment_index=3,
                segment_name="right",
                path=tmp_path / "right.vgf",
                debug_names=SpirvDebugNameMap(
                    spirv_id_to_debug_name={"20": labels[1]},
                    debug_name_to_spirv_ids={labels[1]: ["20"]},
                ),
                structural_source_operator_ids=[],
            ),
        ]
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.prepare_gcpe_compatible_vgfs",
        lambda *_args: segments,
    )
    monkeypatch.setattr(
        NXCompatibilityChecker,
        "_run_estimator_debug_database",
        lambda _self, segment: {
            "tosa_op_id_to_api_labels": {
                "1": [labels[0] if segment.segment_index == 1 else labels[1]]
            },
            "tosa_op_id_to_tosa_op": {"1": ["Add"]},
            "shader_op_id_to_api_labels": {},
        },
    )

    compatibility = NXCompatibilityChecker(tmp_path).check_compatibility(
        VGFModel(vgf_path)
    )
    output = compatibility.to_standardized_output(vgf_path)

    source_ids = {
        entity["id"]
        for entity in output["results"][0]["entities"]
        if entity["kind"] == "source_operator"
    }
    assert source_ids == {
        "source_operator/segment_1/spirv-10",
        "source_operator/segment_3/spirv-20",
    }
    assert {
        check["entity_id"] for check in output["results"][0]["checks"]
    } == source_ids
    validate_standardized_output(output)
