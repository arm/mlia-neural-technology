# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for structured Neural Technology profiling captures."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast

import pytest

from mlia.backend.neural_technology_profiling_data.parser import (
    load_capture_input,
    parse_capture,
    parse_debug_database,
    parse_profiling_data,
    parse_statistics_file,
    parse_statistics_info,
)
from mlia.backend.neural_technology_profiling_data.plugin import (
    NeuralTechnologyProfilingDataPlugin,
)
from mlia.backend.neural_technology_profiling_data.profiling import (
    analyze_profiling_data,
)
from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.vgf import GCPEVGFSegment, GCPEVGFSegments
from mlia.backend.registry import BackendRegistry
from mlia.core.common import AdviceCategory
from mlia.core.errors import ConfigurationError
from mlia.core.output_validation import validate_standardized_output


def _u32_blob(words: list[int]) -> bytes:
    return b"".join(struct.pack("<I", word) for word in words)


def _statistics_info_blob(words: list[int]) -> bytes:
    payload = _u32_blob(words)
    return struct.pack("<III", 0x2211EEFF, 1, len(payload)) + payload


def _debug_database_blob(payload: str) -> bytes:
    encoded = payload.encode()
    return struct.pack("<III", 0x1122FFEE, 1, len(encoded)) + encoded


def _debug_database_payload(api_label: str = "TOSACONV2D_spirv_id_504") -> str:
    return (
        "id\ttosa_op\tapi_labels\n"
        f"10\tConv2D\t{api_label};\n"
        "----\n"
        "id\tapi_label\n"
        f"100\t{api_label}\n"
        "----\n"
        "id\ttosa_op_ids\n"
        "20\t10;\n"
        "----\n"
        "id\tfused_op_ids\n"
        "30\t20;\n"
        "----\n"
        "id\tcascade_op_ids\n"
        "40\t30;\n"
        "----\n"
        "id\top_id\tcascade_op_id\n"
        "0\t30\t40\n"
        "----\n"
        "id\tstripe_op_id\n"
        "0\t0\n"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "profile.toml"
    profile.write_text(
        'target = "neural-technology"\nprofile_name = "NX-measured"\n',
        encoding="utf-8",
    )
    return profile


def _pipeline_spec(
    pipeline_id: int,
    spirv: bytes,
    dispatches: list[tuple[int, int, int]],
    *,
    name: str | None = None,
    api_label: str = "TOSACONV2D_spirv_id_504",
) -> dict[str, Any]:
    return {
        "id": pipeline_id,
        "name": name or f"graph-{pipeline_id}",
        "spirv": spirv,
        "dispatches": dispatches,
        "api_label": api_label,
    }


def _write_capture(
    tmp_path: Path,
    pipelines: list[dict[str, Any]],
    *,
    mode: int = 0,
    device: str = "Mali-G1",
    vendor_id: int | None = 0x13B5,
    device_id: int | None = 2,
    root_name: str = "capture",
) -> Path:
    root = tmp_path / root_name
    root.mkdir()
    pipeline_refs = []
    for spec in pipelines:
        pipeline_id = spec["id"]
        pipeline_dir = root / f"pipeline_{pipeline_id:06d}"
        pipeline_dir.mkdir()
        debug_path = pipeline_dir / "debug_database.bin"
        debug_path.write_bytes(_debug_database_payload(spec["api_label"]).encode())
        info_path = pipeline_dir / "neural_statistics_info.bin"
        info_path.write_bytes(_u32_blob([0]))
        spirv_path = pipeline_dir / f"shader_module_{pipeline_id}.spv"
        spirv_path.write_bytes(spec["spirv"])

        sessions: dict[int, list[tuple[int, int]]] = {}
        for session_id, dispatch_id, executed_index in spec["dispatches"]:
            sessions.setdefault(session_id, []).append((dispatch_id, executed_index))
        session_refs = []
        for session_id, dispatch_specs in sessions.items():
            session_dir = pipeline_dir / f"session_{session_id:06d}"
            session_dir.mkdir()
            dispatch_refs = []
            for dispatch_id, executed_index in dispatch_specs:
                dispatch_dir = session_dir / f"dispatch_{dispatch_id:06d}"
                dispatch_dir.mkdir()
                statistics_path = dispatch_dir / f"statistics_mode{mode}.bin"
                statistics_path.write_bytes(
                    _u32_blob([10 + pipeline_id, 20, 30, *([0] * 13)])
                )
                dispatch_relative = (
                    f"pipeline_{pipeline_id:06d}/session_{session_id:06d}/"
                    f"dispatch_{dispatch_id:06d}"
                )
                _write_json(
                    dispatch_dir / "dispatch.json",
                    {
                        "schema_version": 2,
                        "id": dispatch_id,
                        "session_id": session_id,
                        "executed_index": executed_index,
                        "diagnostic_command_buffer_handle": None,
                        "artifacts": [
                            {
                                "path": f"{dispatch_relative}/{statistics_path.name}",
                                "type": "dispatch_statistics",
                                "size": statistics_path.stat().st_size,
                            }
                        ],
                    },
                )
                dispatch_refs.append(
                    {
                        "id": dispatch_id,
                        "path": f"{dispatch_relative}/dispatch.json",
                    }
                )
            session_relative = f"pipeline_{pipeline_id:06d}/session_{session_id:06d}"
            _write_json(
                session_dir / "session.json",
                {
                    "schema_version": 2,
                    "id": session_id,
                    "pipeline_id": pipeline_id,
                    "device_id": 0,
                    "friendly_name": f"session-{session_id}",
                    "diagnostic_handle": None,
                    "flags": 0,
                    "dispatches": dispatch_refs,
                },
            )
            session_refs.append(
                {"id": session_id, "path": f"{session_relative}/session.json"}
            )

        pipeline_relative = f"pipeline_{pipeline_id:06d}"
        _write_json(
            pipeline_dir / "pipeline.json",
            {
                "schema_version": 2,
                "id": pipeline_id,
                "device_id": 0,
                "friendly_name": spec["name"],
                "diagnostic_handle": None,
                "flags": 0,
                "pipeline_layout_handle": None,
                "resource_bindings": [],
                "vendor_options": None,
                "identifier_only": False,
                "foreign_processing_engine": False,
                "statistics_enabled": True,
                "shader": {
                    "module_id": pipeline_id,
                    "spirv_available": True,
                    "friendly_name": spec["name"],
                    "entry_point": "main",
                    "specialization_entries": [],
                    "specialization_data_size": 0,
                },
                "artifacts": [
                    {
                        "path": f"{pipeline_relative}/{debug_path.name}",
                        "type": "debug_database",
                        "size": debug_path.stat().st_size,
                    },
                    {
                        "path": f"{pipeline_relative}/{info_path.name}",
                        "type": "statistics_info",
                        "size": info_path.stat().st_size,
                    },
                    {
                        "path": f"{pipeline_relative}/{spirv_path.name}",
                        "type": "shader_module",
                        "size": spirv_path.stat().st_size,
                    },
                ],
                "sessions": session_refs,
            },
        )
        pipeline_refs.append(
            {"id": pipeline_id, "path": f"{pipeline_relative}/pipeline.json"}
        )

    _write_json(
        root / "capture.json",
        {
            "schema_version": 2,
            "status": "complete",
            "error": None,
            "capture": {
                "layer_name": "VK_LAYER_LGL_neural_statistics",
                "layer_version": "1.0.0",
                "commit_identity": "test",
                "layer_implementation_version": 1,
                "statistics_mode": mode,
                "dispatch_filter": "",
            },
            "devices": [
                {
                    "id": 0,
                    "name": device,
                    "vendor_id": vendor_id,
                    "device_id": device_id,
                    "driver_version": 3,
                    "api_version": 4,
                }
            ],
            "warnings": [],
            "pipelines": pipeline_refs,
        },
    )
    return root


def _dispatch(root: Path, pipeline_id: int, session_id: int, dispatch_id: int) -> Path:
    return (
        root
        / f"pipeline_{pipeline_id:06d}"
        / f"session_{session_id:06d}"
        / f"dispatch_{dispatch_id:06d}"
    )


def _vgf_segment(index: int, name: str, spirv: bytes) -> GCPEVGFSegment:
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"504": f"operator-{index}"},
        debug_name_to_spirv_ids={f"operator-{index}": ["504"]},
        op_ext_inst_spirv_ids=["504"],
    )
    return GCPEVGFSegment(
        segment_index=index,
        segment_name=name,
        path=Path(f"segment-{index}.vgf"),
        debug_names=debug_names,
        structural_source_operator_ids=[f"source_operator/segment_{index}/spirv-504"],
        structural_source_operator_names={
            f"source_operator/segment_{index}/spirv-504": f"operator-{index}"
        },
        spirv=spirv,
    )


def _mock_vgf(
    monkeypatch: pytest.MonkeyPatch,
    segments: list[GCPEVGFSegment],
    *,
    skipped: list[int] | None = None,
) -> None:
    monkeypatch.setattr(
        "mlia.backend.neural_technology_profiling_data.profiling."
        "prepare_gcpe_compatible_vgfs",
        lambda *_args: GCPEVGFSegments(segments, skipped),
    )


def test_parse_capture_follows_references_and_derives_device(tmp_path: Path) -> None:
    """The hierarchy and artifacts should be read only through metadata references."""
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 7)])])

    capture = parse_capture(root)

    assert capture.devices[0].profiling_device == "Mali-G1"
    assert capture.devices[0].name == "Mali-G1"
    assert capture.mode == 0
    assert capture.pipelines[0].spirv_path.read_bytes() == b"spirv-a"
    assert capture.dispatches[0].executed_index == 7
    assert (
        load_capture_input(_dispatch(root, 0, 0, 0)).dispatch == capture.dispatches[0]
    )


def test_flat_profiling_directory_is_not_accepted(tmp_path: Path) -> None:
    """The removed flat profiling layout must not be discovered heuristically."""
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "debug_database.bin").write_bytes(b"debug")
    (flat / "statistics_mode0.bin").write_bytes(_u32_blob([1] * 16))

    with pytest.raises(
        ConfigurationError, match="neither capture.json nor dispatch.json"
    ):
        load_capture_input(flat)


def test_parse_capture_rejects_escaping_reference(tmp_path: Path) -> None:
    """Parent and absolute paths must never be accepted from metadata."""
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    document = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    document["pipelines"][0]["path"] = "../pipeline.json"
    _write_json(root / "capture.json", document)

    with pytest.raises(ConfigurationError, match="parent component"):
        parse_capture(root)


def test_parse_capture_schema_error_recommends_compatible_layer(tmp_path: Path) -> None:
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    document = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    document["schema_version"] = 1
    _write_json(root / "capture.json", document)

    with pytest.raises(ConfigurationError) as error:
        parse_capture(root)

    assert "schema_version must be 2, got 1" in str(error.value)
    assert "Regenerate the capture with a compatible" in str(error.value)


def test_incomplete_capture_error_recommends_recapture(tmp_path: Path) -> None:
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    document = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    document["status"] = "incomplete"
    _write_json(root / "capture.json", document)

    with pytest.raises(ConfigurationError) as error:
        parse_capture(root)

    assert "Regenerate the capture" in str(error.value)
    assert "shut down cleanly" in str(error.value)


def test_parse_capture_accepts_text_statistics_info(tmp_path: Path) -> None:
    """The layer's text statistics-info representation should be supported."""
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    pipeline_dir = root / "pipeline_000000"
    binary = pipeline_dir / "neural_statistics_info.bin"
    text = pipeline_dir / "neural_statistics_info.txt"
    binary.rename(text)
    text.write_text("0\n", encoding="utf-8")
    document = json.loads((pipeline_dir / "pipeline.json").read_text(encoding="utf-8"))
    info = next(
        item for item in document["artifacts"] if item["type"] == "statistics_info"
    )
    info["path"] = "pipeline_000000/neural_statistics_info.txt"
    info["size"] = text.stat().st_size
    _write_json(pipeline_dir / "pipeline.json", document)

    capture = parse_capture(root)
    pipeline = capture.pipelines[0]
    parsed = parse_profiling_data(capture, pipeline, pipeline.dispatches[0])

    assert pipeline.statistics_info_path == text
    assert parsed.performance_database[0]["id"] == 0


def test_parse_profiling_data_preserves_selected_zero_counter_stripes(
    tmp_path: Path,
) -> None:
    """Statistics info identifies valid blocks, including zero-counter blocks."""
    root = _write_capture(
        tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])], mode=1
    )
    pipeline_dir = root / "pipeline_000000"
    info_path = pipeline_dir / "neural_statistics_info.bin"
    info_path.write_bytes(_u32_blob([0, 2]))

    dispatch_dir = pipeline_dir / "session_000000" / "dispatch_000000"
    statistics_path = dispatch_dir / "statistics_mode1.bin"
    words = [0] * (3 * 256 * 32)
    words[1] = 10
    statistics_path.write_bytes(_u32_blob(words))

    pipeline_document = json.loads(
        (pipeline_dir / "pipeline.json").read_text(encoding="utf-8")
    )
    # Keep the manifest's declared size in sync with the rewritten fixture artifact.
    next(
        item
        for item in pipeline_document["artifacts"]
        if item["type"] == "statistics_info"
    )["size"] = info_path.stat().st_size
    _write_json(pipeline_dir / "pipeline.json", pipeline_document)

    dispatch_document = json.loads(
        (dispatch_dir / "dispatch.json").read_text(encoding="utf-8")
    )
    dispatch_document["artifacts"][0]["size"] = statistics_path.stat().st_size
    _write_json(dispatch_dir / "dispatch.json", dispatch_document)

    capture = parse_capture(root)
    pipeline = capture.pipelines[0]
    parsed = parse_profiling_data(capture, pipeline, pipeline.dispatches[0])

    assert [row["id"] for row in parsed.performance_database] == [0, 2]
    assert [row["opCycles"] for row in parsed.performance_database] == [10, 0]

    info_path.write_bytes(_u32_blob([0, 3]))
    with pytest.raises(ConfigurationError, match="outside the statistics block range"):
        parse_profiling_data(capture, pipeline, pipeline.dispatches[0])


def test_parse_capture_derives_generation_from_layer_device_name(
    tmp_path: Path,
) -> None:
    """A real layer device name should retain its identity and select G2 decoding."""
    root = _write_capture(
        tmp_path,
        [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])],
        device="Mali-G2-Pro-NX MC1",
    )

    capture = parse_capture(root)

    assert capture.devices[0].profiling_device == "Mali-G2"
    assert capture.devices[0].name == "Mali-G2-Pro-NX MC1"


def test_parse_capture_rejects_unsupported_device(tmp_path: Path) -> None:
    """The parser must not silently assume a device generation."""
    root = _write_capture(
        tmp_path,
        [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])],
        device="Unknown accelerator",
    )

    with pytest.raises(ConfigurationError, match="Cannot derive profiling generation"):
        parse_capture(root)


def test_parse_capture_rejects_non_arm_generation_name(tmp_path: Path) -> None:
    """A generation-like name from another vendor must not be trusted."""
    root = _write_capture(
        tmp_path,
        [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])],
        device="Mali-G1",
        vendor_id=0x1234,
    )

    with pytest.raises(ConfigurationError, match="vendor_id"):
        parse_capture(root)


def test_parse_capture_allows_unreferenced_regular_entries(tmp_path: Path) -> None:
    """Unreferenced regular files and directories do not invalidate a capture."""
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    (root / "notes.txt").write_text("metadata", encoding="utf-8")
    extra_directory = root / "tool-output"
    extra_directory.mkdir()
    (extra_directory / "results.json").write_text("{}", encoding="utf-8")

    parse_capture(root)


def test_parse_capture_rejects_missing_referenced_file(tmp_path: Path) -> None:
    """Every file referenced by the capture manifests must be present."""
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    (root / "pipeline_000000" / "debug_database.bin").unlink()

    with pytest.raises(ConfigurationError, match="debug_database.bin"):
        parse_capture(root)


def test_parse_capture_rejects_case_mismatched_public_entry(tmp_path: Path) -> None:
    """On case-insensitive hosts, actual namespace spelling must still be exact."""
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    source = root / "pipeline_000000"
    renamed = root / "Pipeline_000000"
    source.rename(renamed)

    with pytest.raises(ConfigurationError, match="capture namespace"):
        parse_capture(root)


@pytest.mark.parametrize("directory", [False, True])
def test_parse_capture_rejects_orphan_symlink(tmp_path: Path, directory: bool) -> None:
    """Unreferenced symlinks must be rejected rather than skipped by traversal."""
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    target = tmp_path / ("outside" if directory else "outside.bin")
    if directory:
        target.mkdir()
    else:
        target.write_bytes(b"outside")
    try:
        (root / "orphan-link").symlink_to(target, target_is_directory=directory)
    except OSError as err:
        pytest.skip(f"Symlink creation is unavailable: {err}")

    with pytest.raises(ConfigurationError, match="symlink"):
        parse_capture(root)


def test_parse_debug_database_builds_relationship_maps(tmp_path: Path) -> None:
    path = tmp_path / "debug_database.bin"
    path.write_bytes(_debug_database_payload().encode())

    debug_database = parse_debug_database(path)

    assert debug_database["stripe_op_id_to_op_id"] == {"0": ["30"]}
    assert debug_database["stripe_op_id_to_cascade_op_id"] == {"0": ["40"]}


def test_parse_framed_debug_database(tmp_path: Path) -> None:
    """Real driver output may carry its versioned debug-database header."""
    path = tmp_path / "debug_database.bin"
    path.write_bytes(_debug_database_blob(_debug_database_payload()))

    debug_database = parse_debug_database(path)

    assert debug_database["stripe_op_id_to_op_id"] == {"0": ["30"]}


def test_parse_statistics_info_and_mode0(tmp_path: Path) -> None:
    info = tmp_path / "neural_statistics_info.bin"
    info.write_bytes(_u32_blob([3]))
    statistics = tmp_path / "statistics_mode0.bin"
    statistics.write_bytes(_u32_blob([10, 20, 30, *([0] * 13)]))

    assert parse_statistics_info(info) == [3]
    rows = parse_statistics_file(statistics, mode=0, device="Mali-G1")
    assert rows[0]["opCycles"] == 60
    assert rows[0]["totalCycles"] == 60


def test_parse_statistics_mode0_uses_g2_block_size(tmp_path: Path) -> None:
    """G2 mode-0 data should group 1024 tasks into each statistics block."""
    statistics = tmp_path / "statistics_mode0.bin"
    task = [1, *([0] * 15)]
    statistics.write_bytes(_u32_blob(task * 513))

    rows = parse_statistics_file(statistics, mode=0, device="Mali-G2")

    assert len(rows) == 1
    assert rows[0]["opCycles"] == 513


def test_parse_statistics_mode1_uses_g2_task_layout(tmp_path: Path) -> None:
    """G2 mode-1 data should use 64-word tasks and expose its extra counters."""
    statistics = tmp_path / "statistics_mode1.bin"
    task = [0] * 64
    task[1] = 10
    task[31] = 20
    task[32] = 30
    statistics.write_bytes(_u32_blob(task))

    rows = parse_statistics_file(statistics, mode=1, device="Mali-G2")

    assert len(rows) == 1
    assert rows[0]["opCycles"] == 10
    utilization = {
        entry["sectionName"]: entry["cycles"] for entry in rows[0]["Utilization"]
    }
    assert utilization["me_src0_stall"] == 20
    assert utilization["me_src1_stall"] == 30


def test_parse_framed_binary_statistics_info(tmp_path: Path) -> None:
    """Binary driver output may carry its versioned statistics-info header."""
    info = tmp_path / "neural_statistics_info.bin"
    info.write_bytes(_statistics_info_blob([3, 7]))

    assert parse_statistics_info(info) == [3, 7]


def test_parse_text_statistics_info(tmp_path: Path) -> None:
    """Text driver output contains unsigned IDs separated by ASCII delimiters."""
    info = tmp_path / "neural_statistics_info.txt"
    info.write_text("3, 7\n0x10;\0", encoding="utf-8")

    assert parse_statistics_info(info) == [3, 7, 16]


def test_parse_text_statistics_info_rejects_non_ids(tmp_path: Path) -> None:
    info = tmp_path / "neural_statistics_info.txt"
    info.write_text("3 stripe-7", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unsigned integer stripe IDs"):
        parse_statistics_info(info)


def test_analyze_without_vgf_uses_the_only_captured_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-dispatch capture should use its SPIR-V as model provenance."""
    root = _write_capture(
        tmp_path,
        [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])],
    )
    profile = _write_profile(tmp_path)
    debug_names = SpirvDebugNameMap(op_ext_inst_spirv_ids=["504"])
    monkeypatch.setattr(
        "mlia.backend.neural_technology_profiling_data.profiling."
        "read_spirv_debug_names_from_bytes",
        lambda _data: debug_names,
    )

    output_dir = tmp_path / "mlia-output"
    output = analyze_profiling_data(
        target_profile=str(profile),
        profiling_data=[root],
        categories={"performance"},
        output_dir=output_dir,
    )

    validate_standardized_output(output)
    assert (output_dir / "shader_module_0.spv").read_bytes() == b"spirv-a"
    assert output["model"]["format"] == "spv"
    assert output["model"]["name"] == "shader_module_0.spv"
    configuration = output["backends"][0]["configuration"]
    assert configuration["device"] == "Mali-G1"
    assert configuration["profiling_device"] == "Mali-G1"
    assert configuration["vendor_id"] == 0x13B5
    assert configuration["device_id"] == 2
    assert output["results"][0]["mode"] == "measured"


def test_analyze_without_vgf_rejects_multiple_dispatches(tmp_path: Path) -> None:
    root = _write_capture(
        tmp_path,
        [_pipeline_spec(0, b"spirv-a", [(0, index, index) for index in range(7)])],
    )

    with pytest.raises(ConfigurationError) as error:
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[root],
            categories={"performance"},
        )

    message = str(error.value)
    assert "capture contains 7 dispatches" in message
    assert "Set --profiling-data to an individual dispatch directory" in message
    for index in range(3):
        assert str(_dispatch(root, 0, 0, index)) in message
    for index in range(3, 7):
        assert str(_dispatch(root, 0, 0, index)) not in message
    assert "4 more omitted" in message


def test_multiple_dispatch_inputs_without_vgf_are_rejected(tmp_path: Path) -> None:
    root = _write_capture(
        tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0), (0, 1, 1)])]
    )

    with pytest.raises(ConfigurationError, match="exactly one individual dispatch"):
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[_dispatch(root, 0, 0, 0), _dispatch(root, 0, 0, 1)],
            categories={"performance"},
        )


def test_single_anchor_matches_other_segments_by_executed_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An anchor may identify a nonzero segment and search every peer session."""
    root = _write_capture(
        tmp_path,
        [
            _pipeline_spec(0, b"spirv-a", [(0, 0, 1), (1, 1, 7)], name="A"),
            _pipeline_spec(1, b"spirv-b", [(2, 2, 7)], name="B"),
        ],
    )
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(
        monkeypatch,
        [
            _vgf_segment(0, "segment-A", b"spirv-a"),
            _vgf_segment(1, "segment-B", b"spirv-b"),
        ],
    )

    output = analyze_profiling_data(
        target_profile=str(_write_profile(tmp_path)),
        profiling_data=[_dispatch(root, 1, 2, 2)],
        categories={"performance"},
        model=str(model),
    )

    assert output["model"]["name"] == "model.vgf"
    entities = {item["id"]: item for item in output["results"][0]["entities"]}
    assert entities["segment/0"]["name"] == "segment-A"
    assert entities["segment/1"]["name"] == "segment-B"
    assert output["context"]["runtime_configuration"]["profiling_data"] == [
        str(_dispatch(root, 0, 1, 1).resolve()),
        str(_dispatch(root, 1, 2, 2).resolve()),
    ]


def test_anchor_ambiguity_requires_explicit_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_capture(
        tmp_path,
        [
            _pipeline_spec(0, b"spirv-a", [(0, 0, 3)]),
            _pipeline_spec(1, b"spirv-b", [(1, 1, 3), (2, 2, 3)]),
        ],
    )
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(
        monkeypatch,
        [_vgf_segment(0, "A", b"spirv-a"), _vgf_segment(1, "B", b"spirv-b")],
    )

    with pytest.raises(ConfigurationError) as error:
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[_dispatch(root, 0, 0, 0)],
            categories={"performance"},
            model=str(model),
        )

    assert "Repeat --profiling-data once per graph segment" in str(error.value)
    assert "VGF order" in str(error.value)


def test_capture_root_with_vgf_selects_unique_matching_pipelines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unrelated pipelines should be ignored when every graph match is unique."""
    root = _write_capture(
        tmp_path,
        [
            _pipeline_spec(0, b"spirv-a", [(0, 0, 0)]),
            _pipeline_spec(1, b"spirv-b", [(1, 1, 0)]),
            _pipeline_spec(2, b"unrelated", [(2, 2, 0)]),
        ],
    )
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(
        monkeypatch,
        [_vgf_segment(0, "A", b"spirv-a"), _vgf_segment(1, "B", b"spirv-b")],
    )

    output = analyze_profiling_data(
        target_profile=str(_write_profile(tmp_path)),
        profiling_data=[root],
        categories={"performance"},
        model=str(model),
    )

    segment_ids = {
        item["id"]
        for item in output["results"][0]["entities"]
        if item["kind"] == "segment"
    }
    assert segment_ids == {"segment/0", "segment/1"}


def test_capture_root_with_ambiguous_vgf_pipeline_lists_dispatch_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ambiguous matching pipeline should suggest bounded dispatch paths."""
    root = _write_capture(
        tmp_path,
        [_pipeline_spec(0, b"spirv-a", [(0, index, index) for index in range(5)])],
    )
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(monkeypatch, [_vgf_segment(0, "A", b"spirv-a")])

    with pytest.raises(ConfigurationError) as error:
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[root],
            categories={"performance"},
            model=str(model),
        )

    message = str(error.value)
    assert "Available dispatch directories" in message
    for index in range(3):
        assert str(_dispatch(root, 0, 0, index)) in message
    for index in range(3, 5):
        assert str(_dispatch(root, 0, 0, index)) not in message
    assert "2 more omitted" in message


def test_vgf_segment_without_pipeline_match_explains_model_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"captured", [(0, 0, 0)])])
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(monkeypatch, [_vgf_segment(0, "A", b"different")])

    with pytest.raises(ConfigurationError) as error:
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[root],
            categories={"performance"},
            model=str(model),
        )

    message = str(error.value)
    assert "No captured pipeline SPIR-V matches VGF graph segment 0" in message
    assert "Verify that the VGF is the model used to produce this capture" in message
    assert "explicit" not in message


def test_explicit_dispatch_list_rejects_duplicate_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"spirv-a", [(0, 0, 0)])])
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(
        monkeypatch,
        [_vgf_segment(0, "A", b"spirv-a"), _vgf_segment(1, "B", b"spirv-a")],
    )
    dispatch = _dispatch(root, 0, 0, 0)

    with pytest.raises(ConfigurationError, match="Duplicate profiling dispatch"):
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[dispatch, dispatch],
            categories={"performance"},
            model=str(model),
        )


def test_explicit_dispatches_must_match_vgf_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_capture(
        tmp_path,
        [
            _pipeline_spec(0, b"spirv-a", [(0, 0, 0)]),
            _pipeline_spec(1, b"spirv-b", [(1, 1, 0)]),
        ],
    )
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(
        monkeypatch,
        [_vgf_segment(0, "A", b"spirv-a"), _vgf_segment(1, "B", b"spirv-b")],
    )

    with pytest.raises(ConfigurationError, match="does not match VGF graph segment 0"):
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[
                _dispatch(root, 1, 1, 1),
                _dispatch(root, 0, 0, 0),
            ],
            categories={"performance"},
            model=str(model),
        )


def test_duplicate_vgf_spirv_requires_explicit_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_capture(tmp_path, [_pipeline_spec(0, b"same", [(0, 0, 0)])])
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(
        monkeypatch,
        [_vgf_segment(0, "A", b"same"), _vgf_segment(1, "B", b"same")],
    )

    with pytest.raises(ConfigurationError, match="duplicate SPIR-V"):
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[root],
            categories={"performance"},
            model=str(model),
        )


def test_explicit_multi_segment_output_uses_shared_totals_and_original_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_capture(
        tmp_path,
        [
            _pipeline_spec(0, b"spirv-a", [(0, 0, 0)]),
            _pipeline_spec(1, b"spirv-b", [(1, 1, 0)]),
        ],
    )
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    _mock_vgf(
        monkeypatch,
        [
            _vgf_segment(2, "graph-two", b"spirv-a"),
            _vgf_segment(4, "graph-four", b"spirv-b"),
        ],
        skipped=[3],
    )

    output = analyze_profiling_data(
        target_profile=str(_write_profile(tmp_path)),
        profiling_data=[_dispatch(root, 0, 0, 0), _dispatch(root, 1, 1, 1)],
        categories={"performance"},
        model=str(model),
    )

    result = cast(dict[str, Any], output["results"][0])
    metrics = {
        item["name"]: item["value"] for item in result["metrics"] if "value" in item
    }
    assert metrics["total_cycles"] == 121
    assert metrics["compute_cycles"] == 121
    segments = {
        item["id"]: item["name"]
        for item in result["entities"]
        if item["kind"] == "segment"
    }
    assert segments == {"segment/2": "graph-two", "segment/4": "graph-four"}
    assert any("compute segment(s) 3" in warning for warning in result["warnings"])
    validate_standardized_output(output)


def test_analyze_rejects_multiple_captures(tmp_path: Path) -> None:
    first = _write_capture(
        tmp_path, [_pipeline_spec(0, b"a", [(0, 0, 0)])], root_name="first"
    )
    second = _write_capture(
        tmp_path, [_pipeline_spec(0, b"b", [(0, 0, 0)])], root_name="second"
    )
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")

    with pytest.raises(ConfigurationError, match="same capture"):
        analyze_profiling_data(
            target_profile=str(_write_profile(tmp_path)),
            profiling_data=[_dispatch(first, 0, 0, 0), _dispatch(second, 0, 0, 0)],
            categories={"performance"},
            model=str(model),
        )


def test_plugin_advertises_profiling_data_support() -> None:
    registry = BackendRegistry()

    NeuralTechnologyProfilingDataPlugin.register(registry)

    backend = registry.items["neural-technology-profiling-data"]
    assert backend.supported_advice == [AdviceCategory.PERFORMANCE]
    assert backend.selectable is False
    assert backend.supports_estimation is False
    assert backend.supports_profiling_data is True
