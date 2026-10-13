# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NX performance estimator compatibility output."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlia.core.output_schema as schema
import pytest
from tosa_serializer import (
    TOSA_VERSION_MAJOR,
    TOSA_VERSION_MINOR,
    DType,
    Op,
    TosaOpLocation,
    TosaSerializer,
    TosaSerializerAttribute,
)
from mlia.backend.ml_sdk_model_converter.tosa_reader import (
    read_tosa_flatbuffer_ops,
    read_tosa_mlir_ops,
)
from mlia.backend.nx_performance_estimator.compat import (
    _collect_tosa_compiler_records,
    _CompilerRecord,
    NXCompatibilityChecker,
    NXModelCompatibilityInfo,
    TOSAModel,
    VGFModel,
)
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.provenance import (
    source_provenance_from_api_label,
)
from mlia.backend.nx_performance_estimator.statistics import NXPerformanceStats
from mlia.backend.nx_performance_estimator.vgf import GCPEVGFSegment, GCPEVGFSegments
from mlia.core.output_validation import validate_standardized_output


def _model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "model.tosamlir"
    model_path.write_bytes(b"test model content")
    return model_path


def _result_metrics(output: dict) -> dict[str, dict]:
    return {metric["name"]: metric for metric in output["results"][0]["metrics"]}


@pytest.mark.parametrize(
    ("placements", "expected_percentage", "expected_status"),
    [
        pytest.param(("NX", "NX"), 100.0, "ok", id="fully-compatible"),
        pytest.param(("NX", "EE"), 50.0, "ok", id="partially-nx-compatible"),
        pytest.param(("NX", None), 100.0, "ok", id="compiler-omitted-operation"),
    ],
)
@pytest.mark.parametrize("suffix", [".tosa", ".tosamlir"])
@pytest.mark.parametrize("original_label", ["", "shared", "layer_42", "conv(foo)"])
def test_tosa_compatibility_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    placements: tuple[str, str | None],
    expected_percentage: float,
    expected_status: str,
    suffix: str,
    original_label: str,
) -> None:
    """Report full and mixed NX/EE placement using unique compiler labels."""
    model = tmp_path / f"model{suffix}"
    if suffix == ".tosamlir":
        model.write_text(
            """module {
  func.func @main(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    %0 = tosa.abs %arg0 : (tensor<1xi32>) -> tensor<1xi32> loc(#loc1)
    %1 = tosa.abs %0 : (tensor<1xi32>) -> tensor<1xi32> loc(#loc2)
    return %1 : tensor<1xi32>
  }
}
"""
            + f"#loc1 = loc({json.dumps(original_label)})\n#loc2 = loc({json.dumps(original_label)})\n",
            encoding="utf-8",
        )
        reader = read_tosa_mlir_ops
    else:
        serializer = TosaSerializer()
        block = serializer.currRegion.currBasicBlock
        for i in range(3):
            block.addTensor(f"tensor{i}", [1], DType.INT32)
        block.addInput("tensor0")
        block.addOutput("tensor2")
        for i in range(2):
            attribute = TosaSerializerAttribute()
            attribute.AbsAttribute()
            serializer.addOperator(
                Op.ABS,
                [f"tensor{i}"],
                [f"tensor{i + 1}"],
                attribute,
                TosaOpLocation(original_label),
            )
        model.write_bytes(serializer.serialize())
        reader = read_tosa_flatbuffer_ops
    original_bytes = model.read_bytes()
    labels = [f"{original_label or 'unknown'}_{i}" for i in range(2)]
    checker = NXCompatibilityChecker(tmp_path)
    prepared = checker._prepare_tosa_model(model)
    assert [op.loc for op in reader(prepared).values()] == labels
    assert model.read_bytes() == original_bytes
    module = "mlia.backend.nx_performance_estimator.compat"
    reader_mock = Mock(wraps=reader)
    monkeypatch.setattr(f"{module}.{reader.__name__}", reader_mock)
    monkeypatch.setattr(f"{module}.get_ml_sdk_model_converter_path", lambda: tmp_path)
    vgf = tmp_path / "model.vgf"

    def convert(annotated: Path, output_dir: Path) -> Path:
        assert annotated != model
        assert annotated.suffix == suffix
        assert output_dir == tmp_path
        assert [op.loc for op in reader(annotated).values()] == labels
        assert model.read_bytes() == original_bytes
        return vgf

    monkeypatch.setattr(f"{module}.MLSDKModelConverter", Mock(return_value=convert))
    monkeypatch.setattr(
        f"{module}.prepare_gcpe_compatible_vgfs",
        lambda *_: [GCPEVGFSegment(0, "main", vgf, SpirvDebugNameMap(), [])],
    )
    debug_db = {
        table: {
            str(i): [label]
            for i, (label, placement) in enumerate(zip(labels, placements))
            if placement == backend
        }
        for backend, table in (
            ("NX", "tosa_op_id_to_api_labels"),
            ("EE", "shader_op_id_to_api_labels"),
        )
    }
    for label_table in debug_db.values():
        for compiler_labels in label_table.values():
            compiler_labels.append("")
    monkeypatch.setattr(
        checker, "_run_estimator_debug_database", Mock(return_value=debug_db)
    )

    compatibility = checker.check_compatibility(TOSAModel(model))
    assert [call.args[0] for call in reader_mock.call_args_list] == [model, prepared]
    assert all(
        record.display_label == (original_label or "unknown")
        for record in compatibility.get_records()
    )
    output = compatibility.to_standardized_output(model)

    result = output["results"][0]
    assert [backend["id"] for backend in output["backends"]] == [
        "ml-sdk-model-converter",
        "nx-performance-estimator",
    ]
    assert result["producer"] == "nx-performance-estimator"
    assert result["status"] == expected_status
    entities = {entity["id"]: entity for entity in result["entities"]}
    assert set(entities) == {schema.tosa_source_operator_id(i) for i in range(2)}
    for i, placement in enumerate(placements):
        assert entities[schema.tosa_source_operator_id(i)].get("placement") == placement
        assert entities[schema.tosa_source_operator_id(i)]["attributes"][
            "display_label"
        ] == (original_label or "unknown")
    assert {check["entity_id"] for check in result["checks"]} == {
        schema.tosa_source_operator_id(i)
        for i, placement in enumerate(placements)
        if placement is not None
    }
    assert all(check["status"] == "pass" for check in result["checks"])
    metric = _result_metrics(output)[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE]
    assert metric["value"] == expected_percentage
    assert not result.get("warnings")
    validate_standardized_output(output)


@pytest.mark.parametrize(
    ("major", "minor", "supported"),
    [
        pytest.param(TOSA_VERSION_MAJOR, TOSA_VERSION_MINOR, True, id="same-version"),
        pytest.param(TOSA_VERSION_MAJOR, 0, True, id="older-minor"),
        pytest.param(
            TOSA_VERSION_MAJOR, TOSA_VERSION_MINOR + 1, False, id="newer-minor"
        ),
        pytest.param(TOSA_VERSION_MAJOR - 1, 0, False, id="older-major"),
        pytest.param(TOSA_VERSION_MAJOR + 1, 0, False, id="newer-major"),
    ],
)
def test_prepare_tosa_model_validates_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    major: int,
    minor: int,
    supported: bool,
) -> None:
    """Accept only matching major versions and supported minor versions."""
    model = tmp_path / "versioned.tosa"
    original = TosaSerializer(
        targetMajor=major, targetMinor=minor, targetPatch=0, targetDraft=True
    ).serialize()
    model.write_bytes(original)
    serializer = Mock(wraps=TosaSerializer)
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.compat.TosaSerializer", serializer
    )
    checker = NXCompatibilityChecker(tmp_path)

    if supported:
        prepared = checker._prepare_tosa_model(model)
        assert prepared.is_file()
        serializer.assert_called_once()
    else:
        with pytest.raises(ValueError, match="Unsupported TOSA version"):
            checker._prepare_tosa_model(model)
        serializer.assert_not_called()
        assert not (tmp_path / "versioned_processed.tosa").exists()
    assert model.read_bytes() == original


def test_compatibility_output_reports_nx_operator_percentage(tmp_path: Path) -> None:
    """Only NX placements count toward the accelerator operator percentage."""
    compatibility = NXModelCompatibilityInfo()
    placements = ["NX", "NX", "EE", "EE", "EE"]
    source_ids = [schema.vgf_source_operator_id(0, i) for i in range(5)]
    for i, placement in enumerate(placements):
        compatibility.add_estimator_placement(
            f"record_{i}",
            f"operator_{i}",
            placement,
            source_operator_id=source_ids[i],
        )

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    result = output["results"][0]
    metrics = _result_metrics(output)
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "value": 40.0,
        "unit": schema.UNIT_PERCENT,
    }
    assert [entity["id"] for entity in result["entities"]] == source_ids
    assert [entity["placement"] for entity in result["entities"]] == placements
    assert [check["entity_id"] for check in result["checks"]] == source_ids
    assert all(check["status"] == "pass" for check in result["checks"])
    assert result["status"] == "ok"
    validate_standardized_output(output)


def test_compatibility_output_reports_zero_percentage_for_only_shader_records(
    tmp_path: Path,
) -> None:
    """Shader-compatible records are successful but not accelerator-placed."""
    placement = "EE"
    compatibility = NXModelCompatibilityInfo()
    compatibility.add_estimator_placement(
        "record_0",
        "custom_0",
        placement,
        source_operator_id=schema.vgf_source_operator_id(0, 0),
    )
    compatibility.add_estimator_placement(
        "record_1",
        "add_1",
        placement,
        source_operator_id=schema.vgf_source_operator_id(0, 1),
    )

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    metrics = _result_metrics(output)
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "value": 0.0,
        "unit": schema.UNIT_PERCENT,
    }
    assert output["results"][0]["status"] == "ok"
    assert all(
        entity["placement"] == placement for entity in output["results"][0]["entities"]
    )


def test_compatibility_output_uses_operator_type_for_entity_name(
    tmp_path: Path,
) -> None:
    """Compatibility entities should prefer semantic operator types as names."""
    compatibility = NXModelCompatibilityInfo()
    compatibility.add_estimator_placement(
        "record_0",
        "unknown",
        "NX",
        op_type="ADD",
        source_operator_id="source_operator/segment_0/spirv-1",
    )

    output = compatibility.to_standardized_output(_model_file(tmp_path))

    entity = output["results"][0]["entities"][0]
    assert entity["name"] == "ADD"
    assert entity["attributes"]["display_label"] == "unknown"
    assert compatibility.get_records()[0].display_label == "unknown"


def test_unattributed_converted_entity_ids_are_disjoint_and_stable(
    tmp_path: Path,
) -> None:
    """Use compiler keys, not positions, for entities without source identities."""
    compatibility = NXModelCompatibilityInfo()
    model = _model_file(tmp_path)
    for record_key in ("a_unknown", "b_unknown"):
        compatibility.add_estimator_placement(record_key, "unknown", "EE")
    compatibility.add_estimator_placement(
        "z_source", "known", "NX", "ADD", source_operator_id="source_operator/0"
    )

    output = compatibility.to_standardized_output(
        model, source_operator_attribution_available=False
    )

    validate_standardized_output(output)
    result = output["results"][0]
    expected_ids = {
        "converted_operator/unattributed/a_unknown",
        "converted_operator/unattributed/b_unknown",
        "converted_operator/0",
    }
    assert {entity["id"] for entity in result["entities"]} == expected_ids
    assert {check["entity_id"] for check in result["checks"]} == expected_ids
    assert len(result["entities"]) == len(result["checks"]) == 3
    for entity in result["entities"]:
        if entity["id"] == "converted_operator/0":
            assert entity["attributes"]["converted_tosa_operator_id"] == "0"
        else:
            assert "converted_tosa_operator_id" not in entity["attributes"]

    # An earlier-sorting record changes list indices, but not existing entity IDs.
    compatibility.add_estimator_placement("0_new", "unknown", "EE")
    updated = compatibility.to_standardized_output(
        model, source_operator_attribution_available=False
    )
    validate_standardized_output(updated)
    expected_ids.add("converted_operator/unattributed/0_new")
    assert {
        entity["id"] for entity in updated["results"][0]["entities"]
    } == expected_ids
    assert {
        check["entity_id"] for check in updated["results"][0]["checks"]
    } == expected_ids


def test_add_estimator_placement() -> None:
    """Test add_estimator_placement method."""
    invalid_compatibility = NXModelCompatibilityInfo()
    with pytest.raises(ValueError, match="Unsupported estimator placement 'CPU'"):
        invalid_compatibility.add_estimator_placement("op_id_3_0", "loc", "CPU")

    compatibility = NXModelCompatibilityInfo()
    compatibility.add_estimator_placement("op_id_1_0", "loc0", "NX", op_type="Add")
    compatibility.add_estimator_placement("op_id_2_0", "loc1", "EE")
    compatibility.add_estimator_placement("op_id_3_0", "loc2", "NX/EE")

    assert [
        (record.display_label, record.compat_level, record.placement, record.type)
        for record in compatibility.get_records()
    ] == [
        ("loc0", "TOSA", "NX", "Add"),
        ("loc1", "Shader", "EE", None),
        ("loc2", "TOSA/Shader", "NX/EE", None),
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
    converter = Mock()
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.compat.MLSDKModelConverter", converter
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.compat.prepare_gcpe_compatible_vgfs",
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
        "mlia.backend.nx_performance_estimator.compat.run_nx_performance_estimator",
        fake_gc_run,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.compat."
        "NXDebugDatabaseParser.parse_debug_database",
        lambda _self: {
            "tosa_op_id_to_api_labels": {
                "1": ["add0", "", "add1"],
            },
            "tosa_op_id_to_tosa_op": {"1": ["Add"]},
            "shader_op_id_to_api_labels": {
                "2": ["fp_add0", "", "fp_add1"],
            },
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

    compatibility = checker.check_compatibility(VGFModel(vgf_path))
    converter.assert_not_called()
    output = compatibility.to_standardized_output(
        vgf_path,
        backend_config={"ml-sdk-model-converter": {}, "nx-performance-estimator": {}},
    )
    assert [backend["id"] for backend in output["backends"]] == [
        "nx-performance-estimator"
    ]
    assert output["results"][0]["producer"] == "nx-performance-estimator"
    validate_standardized_output(output)
    assert [
        (record.display_label, record.compat_level, record.placement, record.type)
        for record in compatibility.get_records()
    ] == [
        ("add0", "TOSA", "NX", "Add"),
        ("add1", "TOSA", "NX", "Add"),
        ("fp_add0", "Shader", "EE", ""),
        ("fp_add1", "Shader", "EE", ""),
    ]


@pytest.mark.parametrize("suffix", [".vgf", ".tosamlir"])
@pytest.mark.parametrize("empty_labels", [[], [""], ["", " \t", ""]])
def test_compiler_record_without_labels_preserves_unattributed_placement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    empty_labels: list[str],
) -> None:
    """Count unlabelled EE operations without attempting source attribution."""
    model = tmp_path / f"model{suffix}"
    model.write_text(
        '%0 = tosa.abs %arg0 : (tensor<1xi32>) -> tensor<1xi32> loc("known")\n'
        '%1 = tosa.abs %0 : (tensor<1xi32>) -> tensor<1xi32> loc("unknown")\n',
        encoding="utf-8",
    )
    module = "mlia.backend.nx_performance_estimator.compat"
    provenance_mock = Mock(wraps=source_provenance_from_api_label)
    monkeypatch.setattr(f"{module}.source_provenance_from_api_label", provenance_mock)
    monkeypatch.setattr(f"{module}.get_ml_sdk_model_converter_path", lambda: tmp_path)
    vgf = tmp_path / "converted.vgf"
    monkeypatch.setattr(
        f"{module}.MLSDKModelConverter", Mock(return_value=Mock(return_value=vgf))
    )
    monkeypatch.setattr(
        f"{module}.prepare_gcpe_compatible_vgfs",
        lambda *_: [GCPEVGFSegment(0, "main", vgf, SpirvDebugNameMap(), [])],
    )
    checker = NXCompatibilityChecker(tmp_path)
    monkeypatch.setattr(
        checker,
        "_run_estimator_debug_database",
        Mock(
            return_value={
                "tosa_op_id_to_api_labels": {"1": ["known_0", ""]},
                "tosa_op_id_to_tosa_op": {"1": ["ABS"]},
                # Compiler IDs may overlap between placement tables.
                "shader_op_id_to_api_labels": {"1": empty_labels},
            }
        ),
    )

    compatibility = checker.check_compatibility(
        VGFModel(model) if suffix == ".vgf" else TOSAModel(model)
    )
    assert [call.args[0] for call in provenance_mock.call_args_list] == ["known_0"]
    records = compatibility.get_records()
    assert len(records) == 2
    unknown = next(record for record in records if record.placement == "EE")
    assert unknown.display_label == "unknown"
    assert unknown.source_operator_id is None
    output = compatibility.to_standardized_output(model)
    metric = _result_metrics(output)[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE]
    assert metric["value"] == 50.0
    checks = output["results"][0]["checks"]
    assert len(checks) == 2
    assert all(check["status"] == "pass" for check in checks)
    if suffix == ".tosamlir":
        assert {check.get("entity_id") for check in checks} == {
            schema.tosa_source_operator_id(0),
            None,
        }
    validate_standardized_output(output)


@pytest.mark.parametrize(
    ("placements", "op_types", "expected_type"),
    [
        (("NX", "NX"), [["ABS"], ["ABS"]], "ABS"),
        (("EE", "EE"), [["ABS"], ["ABS"]], "ABS"),
        (("NX", "EE", "NX"), [["ABS"], ["ABS"], ["ABS"]], "ABS"),
        (("EE", "NX", "EE"), [["ABS"], ["ABS"], ["ABS"]], "ABS"),
        (("NX", "NX"), [[], []], None),
        (("EE", "NX", "EE"), [[], ["ABS"], []], "ABS"),
        (
            ("NX", "NX", "EE", "NX", "EE"),
            [["ABS"], ["ABS"], ["ADD"], ["ABS"], ["MUL"]],
            "ABS/ADD/MUL",
        ),
        (
            ("EE", "NX", "EE"),
            [["ABS", "ADD", "ABS"], [], ["ADD", "MUL"]],
            "ABS/ADD/MUL",
        ),
    ],
)
def test_compiler_records_store_merged_placement_and_op_types(
    placements: tuple[str, ...],
    op_types: list[list[str]],
    expected_type: str | None,
) -> None:
    """Merge placement and types while preserving the first record's identity."""
    records: dict[int, _CompilerRecord] = {}
    unattributed: list[_CompilerRecord] = []
    for index, (placement, types) in enumerate(zip(placements, op_types, strict=True)):
        _collect_tosa_compiler_records(
            compiler_records=records,
            unattributed_records=unattributed,
            op_ids={str(index): ["abs_0"]},
            types={str(index): types},
            placement=placement,
            segment_index=index,
            location_to_ops={"abs_0": 0},
        )
        expected = "NX/EE" if len(set(placements[: index + 1])) > 1 else placement
        assert records[0].placement == expected

    assert records == {
        0: _CompilerRecord(
            record_key=f"segment_0_{placements[0]}_0_0",
            label="abs_0",
            placement=expected,
            op_type=expected_type,
        )
    }
    assert not unattributed


@pytest.mark.parametrize("source_attribution", [False, True])
@pytest.mark.parametrize("compiler_types_available", [False, True])
def test_tosa_compatibility_reports_merged_compiler_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_attribution: bool,
    compiler_types_available: bool,
) -> None:
    """Expose merged compiler types in output, falling back to the source type."""
    model = tmp_path / "model.tosamlir"
    model.write_text(
        "%0 = tosa.avg_pool2d %arg0, %arg1, %arg1 : "
        "(tensor<1x28x28x72xi8>, tensor<1xi8>, tensor<1xi8>) "
        '-> tensor<1x1x1x72xi8> loc("average_pool")\n',
        encoding="utf-8",
    )
    module = "mlia.backend.nx_performance_estimator.compat"
    vgf = tmp_path / "model.vgf"
    monkeypatch.setattr(f"{module}.get_ml_sdk_model_converter_path", lambda: tmp_path)
    monkeypatch.setattr(
        f"{module}.MLSDKModelConverter", Mock(return_value=Mock(return_value=vgf))
    )
    monkeypatch.setattr(
        f"{module}.prepare_gcpe_compatible_vgfs",
        lambda *_: [GCPEVGFSegment(0, "main", vgf, SpirvDebugNameMap(), [])],
    )
    checker = NXCompatibilityChecker(tmp_path)
    monkeypatch.setattr(
        checker,
        "_run_estimator_debug_database",
        Mock(
            return_value={
                "tosa_op_id_to_api_labels": {
                    "8": ["average_pool_0"],
                    "18": ["average_pool_0"],
                },
                "tosa_op_id_to_tosa_op": (
                    {"8": ["AvgPool"], "18": ["Rescale"]}
                    if compiler_types_available
                    else {}
                ),
            }
        ),
    )

    compatibility = checker.check_compatibility(TOSAModel(model))
    output = compatibility.to_standardized_output(
        model, source_operator_attribution_available=source_attribution
    )

    validate_standardized_output(output)
    result = output["results"][0]
    assert len(result["entities"]) == len(result["checks"]) == 1
    entity = result["entities"][0]
    expected_type = "AvgPool/Rescale" if compiler_types_available else "tosa.avg_pool2d"
    assert entity["name"] == expected_type
    assert entity["attributes"]["op_type"] == expected_type
    assert entity["attributes"]["display_label"] == "average_pool"


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
        "mlia.backend.nx_performance_estimator.compat.prepare_gcpe_compatible_vgfs",
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
        "mlia.backend.nx_performance_estimator.compat.run_nx_performance_estimator",
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
        "mlia.backend.nx_performance_estimator.compat."
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
        "mlia.backend.nx_performance_estimator.compat.prepare_gcpe_compatible_vgfs",
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
        "mlia.backend.nx_performance_estimator.compat.run_nx_performance_estimator",
        lambda *_args, **_kwargs: SimpleNamespace(
            debug_database=tmp_path / "debug.dat"
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.compat.NXDebugDatabaseParser.parse_debug_database",
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
        "mlia.backend.nx_performance_estimator.compat.prepare_gcpe_compatible_vgfs",
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
