# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ETRecord/ETDump measured profiling integration."""

import base64
import json
import struct
from pathlib import Path
from unittest.mock import Mock

import pytest
from mlia.backend.executorch.capture import ETDumpEvent, ProfilingCapture
from mlia.backend.executorch.pte import PTEDelegate, PTEReader
from mlia.core import output_schema as schema
from mlia.core.errors import ConfigurationError
from mlia.core.output_projection import project_entity_breakdowns
from mlia.core.output_validation import validate_standardized_output

from mlia.backend.neural_technology_profiling_data import vgf_etdump, vgf_metadata
from mlia.backend.neural_technology_profiling_data.executorch_source import (
    node_category,
)
from mlia.nx_utils.debug_database import parse_debug_table
from mlia.backend.neural_technology_profiling_data.parser import (
    parse_debug_database as parse_capture_debug_database,
)


def test_etdump_debug_database_preserves_whitespace_header_mappings(tmp_path):
    """Both capture formats retain compiler associations with legacy headers."""
    sections = [
        "id tosa_op api_labels\n10\tConv2D\tTOSACONV2D_spirv_id_10;",
        "id api_label",
        "id tosa_op_ids\n2\t10;",
        "id fused_op_ids\n3\t2;",
        "id cascade_op_ids\n4\t3;",
        "id op_id cascade_op_id\n5\t3;\t4;",
        "id stripe_op_id\n0\t5;",
    ]
    payload = "\n----\n".join(sections).encode()
    database = tmp_path / "debug_database.dat"
    database.write_bytes(payload)
    capture = parse_capture_debug_database(database)
    etdump = vgf_metadata.parse_debug_database(payload)

    assert etdump["stripe_to_chain"] == capture["stripe_op_id_to_op_id"] == {"0": ["3"]}
    assert vgf_metadata.stripe_operations(etdump, 0) == [
        {
            "tosa_id": "10",
            "operator": "Conv2D",
            "api_labels": ["TOSACONV2D_spirv_id_10"],
        }
    ]


@pytest.mark.parametrize(
    "labels",
    [
        ['{"source": "first(); second()"}'],
        ['["first;second", {"source": "third;fourth"}]'],
        ['"first;second"'],
        ['{"source": "first(); second()"}', "fallback"],
        ["first", "second"],
        [],
    ],
)
def test_etrecord_debug_database_preserves_api_labels(labels):
    """Compiler associations retain complete JSON labels and ID list splitting."""
    sections = [
        "id\ttosa_op\tapi_labels\n10\tConv2D\t" + ";".join(labels) + ";",
        "id\tapi_label",
        "id\ttosa_op_ids\n2\t10;11;",
        "id\tfused_op_ids\n3\t2;",
        "id\tcascade_op_ids\n4\t3;",
        "id\top_id\tcascade_op_id\n5\t3;\t4;",
        "id\tstripe_op_id\n0\t5;",
    ]
    debug = vgf_metadata.parse_debug_database("\n----\n".join(sections).encode())
    operations = vgf_metadata.stripe_operations(debug, 0)
    assert operations[0]["api_labels"] == labels
    assert [operation["tosa_id"] for operation in operations] == ["10", "11"]


def test_etrecord_debug_table_preserves_singular_api_label():
    """The singular column holds one label, including literal delimiters."""
    label = "module.relu;;module.conv;1"
    assert parse_debug_table(["id\tapi_label", f"10\t{label}"])[0] == {
        "id": ["10"],
        "api_label": [label],
    }


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"000008 000009 000010", [8, 9, 10]),
        (b"000008, 0xA;\t0X0f\r\n16\0", [8, 10, 15, 16]),
    ],
)
def test_etdump_text_statistics_info_accepts_decimal_and_hex_ids(
    payload: bytes, expected: list[int]
) -> None:
    """Decimal padding and explicit hexadecimal prefixes preserve stripe IDs."""
    assert (
        vgf_metadata.parse_statistics_info({"raw_data": payload, "is_text": True})
        == expected
    )


@pytest.mark.parametrize(
    ("section_count", "with_separators", "versioned"),
    [
        # Check every truncation boundary in the raw, separated format.
        (0, True, False),
        (1, True, False),
        (2, True, False),
        (3, True, False),
        (4, True, False),
        (5, True, False),
        (6, True, False),
        # Check the remaining formats at the final truncation boundary.
        (6, False, False),
        (6, True, True),
        (6, False, True),
    ],
)
def test_etrecord_debug_database_rejects_missing_sections(
    section_count: int, with_separators: bool, versioned: bool
) -> None:
    """Every truncation boundary and supported format rejects missing tables."""
    separator = "\n----\n" if with_separators else "\n"
    payload = separator.join(["id"] * section_count).encode()
    if versioned:
        payload = struct.pack("<III", 0x1122FFEE, 1, len(payload)) + payload
    with pytest.raises(ValueError, match="VGF debug database.*expected.*sections"):
        vgf_metadata.parse_debug_database(payload)


@pytest.mark.parametrize("missing_section", range(7))
def test_etrecord_debug_database_rejects_empty_sections(missing_section: int) -> None:
    """Separators cannot disguise a missing section as an empty relationship map."""
    sections = ["id"] * 7
    sections[missing_section] = ""
    with pytest.raises(ValueError, match="VGF debug database.*sections"):
        vgf_metadata.parse_debug_database("\n----\n".join(sections).encode())


@pytest.mark.parametrize(
    ("with_separators", "trailing_separator", "versioned"),
    [
        (True, False, False),
        (True, True, False),
        (False, False, False),
        (True, False, True),
        (False, False, True),
    ],
)
def test_etrecord_debug_database_accepts_empty_tables_with_headers(
    with_separators: bool, trailing_separator: bool, versioned: bool
) -> None:
    """Header-only tables support legacy lines and optional final separators."""
    sections = [
        "id\ttosa_op\tapi_labels",
        "id\tapi_label",
        "id\ttosa_op_ids",
        "id\tfused_op_ids",
        "id\tcascade_op_ids",
        "id\top_id\tcascade_op_id",
        "id\tstripe_op_id",
    ]
    separator = "\n----\n" if with_separators else "\n"
    payload = separator.join(sections).encode()
    if trailing_separator:
        payload += b"\n----\n"
    if versioned:
        payload = struct.pack("<III", 0x1122FFEE, 1, len(payload)) + payload
    debug = vgf_metadata.parse_debug_database(payload)
    assert all(not relationships for relationships in debug.values())


@pytest.fixture
def empty_capture(monkeypatch):
    """Supply artifact data for tests substituting target mapping policies."""

    def read(dump, model, method=None, **_kwargs):
        return ProfilingCapture(
            method=method or "forward",
            etrecord_path=str(model),
            etdump_path=str(dump),
            events=(),
            nodes_by_handle={},
            instruction_handles={},
            source_graph={"method": method or "forward", "nodes": [], "edges": []},
        )

    monkeypatch.setattr(vgf_etdump, "read_profiling_capture", read)


def _encode_metadata(record):
    """Encode test VGF envelopes at the converter/target boundary."""
    if not isinstance(record, dict):
        return record
    result = dict(
        record,
        schema="executorch.vgf.neural_statistics",
        schema_version=record.get("schema_version", 1),
    )
    result["segments"] = []
    for segment in record.get("segments", []):
        encoded = dict(segment)
        for name in ("debug_database", "statistics_info", "statistics_memory"):
            if name in segment:
                blob = dict(segment[name])
                if "raw_data" in blob:
                    blob["data"] = base64.b64encode(blob.pop("raw_data")).decode(
                        "ascii"
                    )
                    blob["encoding"] = "base64"
                encoded[name] = blob
        result["segments"].append(encoded)
    return json.dumps(result)


@pytest.fixture
def vgf_profile(tmp_path, monkeypatch):
    """Exercise the public analyzer with detached captures and real mappings."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    dump = tmp_path / "run.etdump"
    dump.write_bytes(b"dump")
    sections = [
        "id\ttosa_op\tapi_labels\n10\tConv2D\tTOSACONV2D_spirv_id_10;",
        "id\tapi_label",
        "id\ttosa_op_ids\n2\t10;",
        "id\tfused_op_ids\n3\t2;",
        "id\tcascade_op_ids\n4\t3;",
        "id\top_id\tcascade_op_id\n5\t3;\t4;",
        "id\tstripe_op_id\n0\t5;",
    ]

    def segment(segment_id):
        return {
            "segment_id": segment_id,
            "debug_database": {
                "available": True,
                "raw_data": "\n----\n".join(sections).encode(),
            },
            "statistics_info": {
                "available": True,
                "raw_data": b"0",
                "is_text": True,
            },
            "statistics_memory": {
                "available": True,
                "raw_data": struct.pack("<II", 0, 100) + bytes(4 * (64 * 256 - 2)),
            },
        }

    def analyze(
        event_handles=((42,),),
        segments=None,
        target="aten.convolution",
        metadata_by_event=None,
        companion_model=None,
        mode=1,
        device="Mali-G2",
    ):
        node = {
            "name": "conv",
            "op": "call_function",
            "target": target,
            "debug_handle": 42,
            "category": node_category("conv", target),
            "graph_index": 0,
        }
        events = tuple(
            ETDumpEvent(
                "forward",
                "Execute",
                0,
                index,
                "VGF_NEURAL_STATISTICS",
                index,
                "VgfBackend",
                tuple(handles),
                None,
                tuple(
                    _encode_metadata(raw)
                    for raw in (
                        metadata_by_event[index]
                        if metadata_by_event is not None
                        else [
                            {
                                "segments": segments
                                if segments is not None
                                else [segment(7)]
                            }
                        ]
                    )
                ),
                None,
            )
            for index, handles in enumerate(event_handles)
        )
        capture = ProfilingCapture(
            method="forward",
            etrecord_path=str(model),
            etdump_path=str(dump),
            events=events,
            nodes_by_handle={42: [node]},
            instruction_handles=dict(
                enumerate(tuple(handles) for handles in event_handles)
            ),
            source_graph={"method": "forward", "nodes": [], "edges": []},
        )
        monkeypatch.setattr(
            vgf_etdump, "read_profiling_capture", lambda *_args, **_kwargs: capture
        )
        return vgf_etdump.analyze_etrecord_profiling_data(
            dump,
            model,
            target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
            companion_model=companion_model,
            mode=mode,
            device=device,
        )

    return analyze, segment


def test_vgf_companion_uses_segment_ordinals_and_preserves_producer_ids(
    vgf_profile, tmp_path, monkeypatch
) -> None:
    """Non-contiguous producer IDs still match the embedded VGF segment order."""
    from mlia.backend.neural_technology_profiling_data import executorch_artifacts

    analyze, segment = vgf_profile
    companion = tmp_path / "model.pte"
    companion.write_bytes(b"pte")
    delegate = PTEDelegate("forward", 0, 0, 0, "VgfBackend", b"vgf")
    reader = Mock(spec=PTEReader)
    reader.delegates.return_value = [delegate]
    reader.metadata.return_value = {}
    monkeypatch.setattr(executorch_artifacts, "PTEReader", lambda _path: reader)
    monkeypatch.setattr(
        executorch_artifacts,
        "vgf_segment_operators",
        lambda _payload: {0: {10}, 1: {10}},
    )

    output = analyze(segments=[segment(7), segment(11)], companion_model=companion)
    segments = [
        entity["attributes"]
        for entity in output["results"][0]["entities"]
        if entity["kind"] == "segment"
    ]
    associations = output["extensions"]["executorch"]["companion_model"][
        "operator_associations"
    ]
    for records in (segments, associations):
        assert sorted(
            (item["segment_index"], item["segment_id"]) for item in records
        ) == [(0, 7), (1, 11)]


def _vgf_v2_metadata(mode=1):
    """Use the interpretation metadata format proposed in ExecuTorch PR 23527."""
    return {
        "schema_version": 2,
        "statistics_mode": {
            "executorch_index": mode,
            "vulkan_value": mode + 1,
            "vulkan_name": f"VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS{mode}_ARM",
        },
        "target": {
            "available": True,
            "device_name": "Mali-G2-Pro-NX MC1",
            "vendor_id": 5045,
            "device_id": 7,
            "driver_version": 11,
        },
        "counter_layout": {
            "available": True,
            "schema_id": f"arm.mali-g2.neural-statistics.mode{mode}",
            "schema_version": 1,
            "endianness": "little",
            "word_type": "uint32",
            "words_per_task": 64,
            "leading_block_count": 1,
            "reason": "",
        },
    }


@pytest.mark.parametrize(
    "kind", ["legacy", "legacy-explicit", "recorded", "unavailable"]
)
def test_vgf_decoder_prefers_metadata_and_reports_fallback(vgf_profile, kind):
    """Recorded settings select the actual decoder and expose legacy assumptions."""
    analyze, segment = vgf_profile
    settings = _vgf_v2_metadata() if kind in ("recorded", "unavailable") else {}
    fallback = (
        {"mode": 0, "device": "Mali-G1"}
        if kind in ("legacy-explicit", "recorded")
        else {}
    )
    if kind == "unavailable":
        settings["target"]["available"] = False
        settings["counter_layout"] = {
            "available": False,
            "schema_id": "",
            "schema_version": 0,
            "endianness": "little",
            "word_type": "uint32",
            "words_per_task": 0,
            "leading_block_count": 0,
            "reason": "No registered layout",
        }
    captured_segment = segment(7)
    words = bytearray(captured_segment["statistics_memory"]["raw_data"])
    # Word 33 contributes to nx_active with G1's task layout or to mode-0
    # totals. G2 mode 1 retains the original 100 cycles.
    struct.pack_into("<I", words, 33 * 4, 13)
    captured_segment["statistics_memory"]["raw_data"] = words
    output = analyze(
        metadata_by_event=[[{**settings, "segments": [captured_segment]}]],
        **fallback,
    )
    expected_mode, expected_device, expected_cycles = (
        (0, "Mali-G1", 113) if kind == "legacy-explicit" else (1, "Mali-G2", 100)
    )
    fallback_fields = {
        "legacy": ["statistics_mode", "profiling_device"],
        "legacy-explicit": ["statistics_mode", "profiling_device"],
        "recorded": [],
        "unavailable": ["profiling_device", "counter_layout"],
    }[kind]
    assert (
        next(
            metric["value"]
            for metric in output["results"][0]["metrics"]
            if metric["name"] == "total_cycles"
        )
        == expected_cycles
    )
    configuration = output["backends"][0]["configuration"]
    assert configuration["statistics_mode"] == expected_mode
    assert configuration["device"] == expected_device
    assert configuration["decoder_fallback_fields"] == fallback_fields
    warnings = output["results"][0].get("warnings", [])
    assert any("decoder fallback" in str(warning) for warning in warnings) == bool(
        fallback_fields
    )
    mapping = output["extensions"]["etrecord_mapping"]
    assert mapping["statistics_mode"] == expected_mode
    assert mapping["device"] == expected_device


@pytest.mark.parametrize(
    "section, changes",
    [
        ("statistics_mode", {"executorch_index": True}),
        ("statistics_mode", {"vulkan_value": 1}),
        ("target", {"device_name": "Other GPU"}),
        ("counter_layout", {"schema_id": "unknown"}),
        ("counter_layout", {"endianness": "big"}),
        ("counter_layout", {"words_per_task": 32}),
        ("counter_layout", None),
    ],
)
def test_vgf_decoder_rejects_invalid_recorded_settings(vgf_profile, section, changes):
    """Malformed or unsupported v2 metadata must not select a fallback layout."""
    analyze, segment = vgf_profile
    settings = _vgf_v2_metadata()
    if changes is None:
        del settings[section]
    else:
        settings[section].update(changes)
    with pytest.raises(ConfigurationError, match="VGF decoder"):
        analyze(metadata_by_event=[[{**settings, "segments": [segment(7)]}]])


def test_vgf_decoder_rejects_declared_mode0_task_size(vgf_profile):
    """The producer's 64-word mode-0 declaration conflicts with our 16-word decoder."""
    analyze, segment = vgf_profile
    with pytest.raises(ConfigurationError, match="words_per_task"):
        analyze(metadata_by_event=[[{**_vgf_v2_metadata(0), "segments": [segment(7)]}]])


def test_vgf_decoder_rejects_conflicting_record_layouts(vgf_profile):
    """One result cannot silently combine counters from different layouts."""
    analyze, segment = vgf_profile
    other_mode = _vgf_v2_metadata(0)
    other_mode["counter_layout"]["words_per_task"] = 16
    with pytest.raises(ConfigurationError, match="Conflicting VGF decoder"):
        analyze(
            metadata_by_event=[
                [
                    {**_vgf_v2_metadata(), "segments": [segment(7)]},
                    {**other_mode, "segments": [segment(7)]},
                ]
            ]
        )


@pytest.fixture
def encoded_vgf_profile(vgf_profile):
    """Run the public analyzer with the serialized VGF JSON/base64 format."""
    analyze, segment = vgf_profile

    def metadata(cycles):
        decoded_segment = segment(7)
        decoded_segment["statistics_memory"]["raw_data"] = struct.pack(
            "<II", 0, cycles
        ) + bytes(4 * (64 * 256 - 2))
        return _encode_metadata({"segments": [decoded_segment]})

    return analyze, metadata


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema_version": 3}, "schema version"),
        ({"schema": "unrelated"}, "unexpected schema"),
        ({"segments": {}}, "segments must be a list"),
        ({"segments": [None]}, "segment must be an object"),
        ({"segments": [{"debug_database": None}]}, "blob must be an object"),
        (
            {"segments": [{"debug_database": {"available": True, "encoding": "hex"}}]},
            "blob encoding",
        ),
        (
            {
                "segments": [
                    {
                        "debug_database": {
                            "available": True,
                            "encoding": "base64",
                            "data": "!",
                        }
                    }
                ]
            },
            "Malformed base64",
        ),
    ],
)
def test_vgf_profile_rejects_invalid_metadata_envelopes(
    encoded_vgf_profile, monkeypatch, change, message
):
    """Corrupt or unsupported VGF envelopes cannot fall back to runtime totals."""
    analyze, metadata = encoded_vgf_profile
    payload = json.loads(metadata(100))
    payload.update(change)
    runtime = Mock(return_value=[])
    monkeypatch.setattr(vgf_etdump, "build_standard_runtime_mappings", runtime)
    with pytest.raises(ConfigurationError, match=message):
        analyze(metadata_by_event=[[json.dumps(payload)]])
    runtime.assert_not_called()


@pytest.mark.parametrize(
    ("missing", "record_index"),
    [
        (None, 0),
        (None, 1),
        (None, 2),
        ("", 1),
        (b"", 1),
        ("{}", 1),
        (b"\xff", 1),
        (17, 1),
    ],
    ids=[
        "missing-first",
        "missing-middle",
        "missing-last",
        "empty-text",
        "empty-bytes",
        "unrelated-json",
        "invalid-utf8",
        "invalid-type",
    ],
)
def test_vgf_profile_rejects_missing_repeated_payload(
    encoded_vgf_profile, monkeypatch, missing, record_index: int
) -> None:
    """Missing samples must not disappear or compact execution indices."""
    analyze, metadata = encoded_vgf_profile
    records = [metadata(100), metadata(200), metadata(300)]
    records[record_index] = missing
    runtime = Mock(return_value=[])
    monkeypatch.setattr(vgf_etdump, "build_standard_runtime_mappings", runtime)
    with pytest.raises(
        ConfigurationError,
        match=f"VGF_NEURAL_STATISTICS.*record {record_index}.*no decoded profiling records",
    ):
        analyze(metadata_by_event=[records])
    runtime.assert_not_called()


def test_vgf_profile_preserves_repeated_payload_indices(encoded_vgf_profile) -> None:
    """Complete repeated records retain their positions and per-execution means."""
    analyze, metadata = encoded_vgf_profile
    output = analyze(metadata_by_event=[[metadata(100), metadata(200), metadata(300)]])
    groups = output["extensions"]["etrecord_mapping"]["groups"]
    assert [group["record_index"] for group in groups] == [0, 1, 2]
    metrics = {
        metric["name"]: metric.get("value")
        for metric in output["results"][0]["metrics"]
    }
    assert metrics["execution_count"] == 3
    assert metrics["total_cycles"] == 200
    validate_standardized_output(output)


@pytest.mark.parametrize("event_handles", [((),), ((99,),), ((42,), (99,))])
def test_vgf_profile_rejects_unmatched_source_handles(vgf_profile, event_handles):
    """An unmatched VGF event cannot yield successful or partial model totals."""
    analyze, _segment = vgf_profile
    with pytest.raises(ConfigurationError, match="forward.*instruction.*debug handles"):
        analyze(event_handles=event_handles)


@pytest.mark.parametrize(
    ("empty_part", "valid_scope"),
    [
        ("records", "none"),
        ("records", "event"),
        ("segments", "none"),
        ("segments", "event"),
        ("segments", "record"),
        ("stripes", "none"),
        ("stripes", "event"),
        ("stripes", "record"),
        ("stripes", "segment"),
    ],
)
def test_vgf_profile_rejects_empty_metadata_without_runtime_fallback(
    vgf_profile, monkeypatch, empty_part: str, valid_scope: str
) -> None:
    """Empty VGF data cannot disappear behind runtime or partial VGF totals."""
    analyze, segment = vgf_profile
    valid_segment = segment(7)
    empty_segment = segment(8)
    empty_segment["statistics_info"]["raw_data"] = b""
    empty_segment["statistics_memory"]["raw_data"] = bytes(4 * 64 * 256)
    valid_record = {"segments": [valid_segment]}
    malformed_records = {
        "records": [],
        "segments": [{"segments": []}],
        "stripes": [{"segments": [empty_segment]}],
    }[empty_part]
    if valid_scope == "event":
        metadata_by_event = [[valid_record], malformed_records]
    elif valid_scope == "record":
        metadata_by_event = [[valid_record, *malformed_records]]
    elif valid_scope == "segment":
        metadata_by_event = [[{"segments": [valid_segment, empty_segment]}]]
    else:
        metadata_by_event = [malformed_records]
    runtime = Mock(return_value=[])
    monkeypatch.setattr(vgf_etdump, "build_standard_runtime_mappings", runtime)
    expected = {
        "records": "no decoded profiling records",
        "segments": "no segments",
        "stripes": "no stripe IDs",
    }[empty_part]
    with pytest.raises(ConfigurationError, match=f"VGF_NEURAL_STATISTICS.*{expected}"):
        analyze(
            event_handles=((42,),) * len(metadata_by_event),
            metadata_by_event=metadata_by_event,
        )
    runtime.assert_not_called()


@pytest.mark.parametrize(
    ("missing_blob", "absent", "with_valid_segment"),
    [
        ("debug_database", False, False),
        ("debug_database", True, False),
        ("statistics_info", False, False),
        ("statistics_info", True, False),
        ("statistics_memory", False, False),
        ("statistics_memory", True, False),
        ("statistics_memory", True, True),
    ],
)
def test_vgf_profile_rejects_missing_segment_blobs(
    vgf_profile, with_valid_segment, missing_blob, absent
):
    """Incomplete VGF data must fail rather than fall back or omit a segment."""
    analyze, segment = vgf_profile
    incomplete = segment(8)
    if absent:
        incomplete.pop(missing_blob)
    else:
        incomplete[missing_blob]["available"] = False
    segments = ([segment(7)] if with_valid_segment else []) + [incomplete]
    with pytest.raises(ConfigurationError, match=f"segment 8.*{missing_blob}"):
        analyze(segments=segments)


@pytest.mark.parametrize(
    ("with_valid_segment", "missing_section"),
    [
        pytest.param(False, 0, id="missing-first-section"),
        pytest.param(False, 6, id="missing-last-section"),
        pytest.param(True, 3, id="invalid-database-beside-valid-segment"),
    ],
)
def test_vgf_profile_rejects_incomplete_debug_database(
    vgf_profile, monkeypatch, with_valid_segment: bool, missing_section: int
) -> None:
    """Database errors propagate without runtime fallback or partial VGF totals."""
    analyze, segment = vgf_profile
    incomplete = segment(8)
    sections = incomplete["debug_database"]["raw_data"].split(b"\n----\n")
    del sections[missing_section]
    incomplete["debug_database"]["raw_data"] = b"\n----\n".join(sections)
    segments = ([segment(7)] if with_valid_segment else []) + [incomplete]
    runtime = Mock(return_value=[])
    monkeypatch.setattr(vgf_etdump, "build_standard_runtime_mappings", runtime)
    with pytest.raises(
        ConfigurationError, match="VGF debug database.*expected.*sections"
    ):
        analyze(segments=segments)
    runtime.assert_not_called()


@pytest.mark.parametrize("versioned", [False, True])
def test_vgf_profile_accepts_separator_free_debug_database(
    vgf_profile, versioned: bool
) -> None:
    """Legacy header-only databases still support measured ETDump analysis."""
    analyze, segment = vgf_profile
    legacy = segment(7)
    headers = [
        section.splitlines()[0]
        for section in legacy["debug_database"]["raw_data"].split(b"\n----\n")
    ]
    payload = b"\r\n\r\n".join(headers) + b"\r\n\0"
    if versioned:
        payload = struct.pack("<III", 0x1122FFEE, 1, len(payload)) + payload
    legacy["debug_database"]["raw_data"] = payload
    output = analyze(segments=[legacy])
    validate_standardized_output(output)
    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert metrics["total_cycles"]["value"] == 100


@pytest.mark.parametrize(
    ("text_id", "stripe_id"),
    [(b"000008", 8), (b"000010", 10), (b"0x8", 8), (b"0X8", 8)],
)
def test_vgf_profile_selects_textual_stripe_id(
    vgf_profile, text_id: bytes, stripe_id: int
) -> None:
    """Textual IDs select the intended statistics block in full analysis."""
    analyze, segment = vgf_profile
    profile = segment(7)
    profile["statistics_info"]["raw_data"] = text_id
    block = profile["statistics_memory"]["raw_data"]
    profile["statistics_memory"]["raw_data"] = bytes(stripe_id * len(block)) + block
    profile["debug_database"]["raw_data"] = profile["debug_database"][
        "raw_data"
    ].replace(b"\n0\t5;", f"\n{stripe_id}\t5;".encode())
    output = analyze(segments=[profile])
    validate_standardized_output(output)
    result = output["results"][0]
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert metrics["total_cycles"]["value"] == 100
    stripe = next(entity for entity in result["entities"] if entity["kind"] == "stripe")
    assert stripe["attributes"]["stripe_id"] == stripe_id


@pytest.mark.parametrize("target", ["aten.convolution", "aten.reshape"])
def test_complete_vgf_profile_preserves_source_associations_and_nx_placement(
    vgf_profile, target
):
    """NX work retains its placement with or without a source representative."""
    analyze, segment = vgf_profile
    output = analyze(segments=[segment(7), segment(8)], target=target)
    validate_standardized_output(output)
    result = output["results"][0]
    assert (
        next(
            metric["value"]
            for metric in result["metrics"]
            if metric["name"] == "total_cycles"
        )
        == 200
    )
    assert sum(entity["kind"] == "segment" for entity in result["entities"]) == 2
    assert any(
        entity["kind"] == "source_operator" for entity in result["entities"]
    ) == (target == "aten.convolution")
    execution_kinds = {"chain", "cascade", "stripe", "backend_operator"}
    entities = result["entities"]
    execution_entities = [
        entity for entity in entities if entity["kind"] in execution_kinds
    ]
    assert {entity["kind"] for entity in execution_entities} == execution_kinds
    assert all(entity.get("placement") == "NX" for entity in execution_entities)
    assert all(
        entity.get("placement") is None
        for entity in entities
        if entity["kind"] in {"segment", "source_operator"}
    )


def test_vgf_compiler_relationships_do_not_assign_operator_timings(vgf_profile):
    """Tiled stripes reuse compiler identities without copying their timings."""
    analyze, segment = vgf_profile
    captured = segment(7)
    captured["debug_database"]["raw_data"] = (
        captured["debug_database"]["raw_data"]
        .replace(
            b"10\tConv2D\tTOSACONV2D_spirv_id_10;",
            b"10\tConv2D\tTOSACONV2D_spirv_id_10;\n11\tRescale\tTOSARESCALE_spirv_id_20;",
        )
        .replace(b"2\t10;", b"2\t10;11;")
        .replace(b"0\t5;", b"0\t5;\n1\t5;")
    )
    captured["statistics_info"]["raw_data"] = b"0 1"
    captured["statistics_memory"]["raw_data"] *= 2
    output = analyze(segments=[captured])
    validate_standardized_output(output)
    result = output["results"][0]
    compiled = {
        entity["attributes"]["tosa_id"]: entity
        for entity in result["entities"]
        if entity["kind"] == "backend_operator"
    }
    assert set(compiled) == {"10", "11"}
    assert compiled["10"]["child_ids"] == ["source_operator/etrecord/forward/conv"]
    assert not compiled["11"].get("child_ids")
    assert compiled["10"]["attributes"]["mapping_confidence"]
    assert not {entity["id"] for entity in compiled.values()} & {
        breakdown["entity_id"] for breakdown in result["breakdowns"]
    }


def test_mode1_preserves_zero_statistics_block() -> None:
    """An inactive mode-1 block retains its stripe-alignment position."""
    words_per_block = 64 * 256

    assert vgf_metadata.parse_mode1([0] * words_per_block, "Mali-G2") == [
        {"op_cycles": 0, "total_cycles": 0, "counters": {}}
    ]


def test_mode0_preserves_zero_statistics_block() -> None:
    """An inactive mode-0 block retains its stripe-alignment position."""
    words_per_block = 16 * 1024

    rows = vgf_metadata.parse_mode0([0] * words_per_block, "Mali-G2")

    assert len(rows) == 1
    assert rows[0]["op_cycles"] == 0
    assert rows[0]["total_cycles"] == 0
    assert not any(rows[0]["counters"].values())


@pytest.mark.parametrize(
    ("operator", "labels", "expected"),
    [
        ("Conv2D", ["padding", "aten.add.default"], {"conv"}),
        ("Unknown", ["padding", "submodule", "multiple"], {"unknown"}),
        ("Unknown", ["TOSACONV2D_spirv_id_10"], {"conv"}),
        ("MatMul", [], {"matmul"}),
        ("Depthwise_Conv2D", [], {"conv"}),
        ("Avg_Pool2D", [], {"pool"}),
        ("Unknown", ["aten.add.default"], {"add"}),
    ],
)
def test_operation_categories_use_explicit_operator_and_token_boundaries(
    operator, labels, expected
) -> None:
    """Operator identity wins over labels; incidental substrings do not match."""
    assert (
        vgf_etdump.operation_categories({"operator": operator, "api_labels": labels})
        == expected
    )


@pytest.mark.parametrize(
    "target", ["aten.addmm.default", "aten.bmm.default", "aten.mm.default"]
)
def test_operator_mapping_links_matrix_operation_to_source(target: str) -> None:
    """VGF MatMul observations retain a representative matrix source node."""
    node = {
        "name": "aten_addmm_default",
        "debug_handle": 4,
        "target": target,
        "category": node_category("aten_addmm_default", target),
    }
    mapping = vgf_etdump.map_operator_group(
        [{"tosa_id": "42", "operator": "MatMul", "api_labels": []}],
        [node],
        ("Execute", 0, 0, 0),
        vgf_etdump.OperatorMappingState(),
    )
    assert mapping["nodes"] == [node]
    assert mapping["backend_operations"][0]["matched"]


def test_operator_mapping_reuses_representative_for_tiled_stripes() -> None:
    """Stripes for one backend operation share one source operator."""
    nodes = [
        {
            "graph_index": 1,
            "debug_handle": 4,
            "name": "conv_0",
            "op": "call_function",
            "target": "aten.convolution.default",
            "category": "conv",
        },
        {
            "graph_index": 2,
            "debug_handle": 8,
            "name": "conv_1",
            "op": "call_function",
            "target": "aten.convolution.default",
            "category": "conv",
        },
    ]
    state = vgf_etdump.OperatorMappingState()
    scope = ("Execute", 0, 0, 0)

    first = vgf_etdump.map_operator_group(
        [{"tosa_id": "42", "operator": "Conv2D", "api_labels": []}],
        nodes,
        scope,
        state,
    )
    tiled = vgf_etdump.map_operator_group(
        [{"tosa_id": "42", "operator": "Conv2D", "api_labels": []}],
        nodes,
        scope,
        state,
    )
    next_operation = vgf_etdump.map_operator_group(
        [{"tosa_id": "43", "operator": "Conv2D", "api_labels": []}],
        nodes,
        scope,
        state,
    )

    assert [node["name"] for node in first["nodes"]] == ["conv_0"]
    assert [node["name"] for node in tiled["nodes"]] == ["conv_0"]
    assert [node["name"] for node in next_operation["nodes"]] == ["conv_1"]


def test_operator_mapping_does_not_wrap_exhausted_candidates() -> None:
    """Extra backend operations remain delegate-only instead of mapping wrongly."""
    nodes = [
        {
            "graph_index": 1,
            "debug_handle": 4,
            "name": "conv",
            "op": "call_function",
            "target": "aten.convolution.default",
            "category": "conv",
        }
    ]
    state = vgf_etdump.OperatorMappingState()
    scope = ("Execute", 0, 0, 0)

    vgf_etdump.map_operator_group(
        [{"tosa_id": "42", "operator": "Conv2D", "api_labels": []}],
        nodes,
        scope,
        state,
    )
    exhausted = vgf_etdump.map_operator_group(
        [{"tosa_id": "43", "operator": "Conv2D", "api_labels": []}],
        nodes,
        scope,
        state,
    )

    assert exhausted["nodes"] == []
    assert exhausted["confidence"] == "delegate-only"


def test_etdump_mapping_links_chains_to_source_entities(
    tmp_path: Path, monkeypatch, empty_capture
) -> None:
    """Fused cycles remain on a partition mapped to all selected graph nodes."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    dump = tmp_path / "run.etdump"
    dump.write_bytes(b"dump")
    monkeypatch.setattr(
        vgf_etdump,
        "build_mappings",
        lambda *_args, **_kwargs: {
            "warning": "fused cycles",
            "groups": [
                {
                    "event_block": "Execute",
                    "instruction_id": 0,
                    "segment_index": 0,
                    "segment_id": 0,
                    "stripe_id": 1,
                    "chain_ids": ["42"],
                    "cascade_ids": ["7"],
                    "method": "forward",
                    "statistics": {
                        "op_cycles": 1249,
                        "total_cycles": 1249,
                        "counters": {"nx_active": 1249, "ce_active": 600},
                    },
                    "operator_group": {
                        "confidence": "high-level-high",
                        "strategy": "semantic anchor",
                        "backend_operations": [],
                        "nodes": [
                            {
                                "name": "conv",
                                "debug_handle": 4,
                                "target": "aten.convolution",
                                "category": "conv",
                            },
                            {
                                "name": "relu",
                                "debug_handle": 5,
                                "target": "aten.relu",
                                "category": "relu",
                            },
                        ],
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(
        vgf_etdump,
        "build_standard_runtime_mappings",
        lambda *_args, **_kwargs: [],
    )

    output = vgf_etdump.analyze_etrecord_profiling_data(
        dump,
        model,
        target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
    )

    result = output["results"][0]
    entities = {entity["id"]: entity for entity in result["entities"]}
    chain = next(entity for entity in result["entities"] if entity["kind"] == "chain")
    stripe = next(entity for entity in result["entities"] if entity["kind"] == "stripe")
    source_operator_ids = [
        "source_operator/etrecord/forward/conv",
        "source_operator/etrecord/forward/relu",
    ]
    assert chain["child_ids"] == [stripe["id"], *source_operator_ids]
    assert "child_ids" not in stripe
    assert all(
        entities[entity_id]["parent_ids"] == [chain["id"]]
        for entity_id in source_operator_ids
    )
    metrics = {metric["name"]: metric for metric in result["breakdowns"][0]["metrics"]}
    assert metrics["op_cycles"]["value"] == 1249
    assert metrics["nx_active_cycles"]["value"] == 1249
    assert metrics["ce_active_cycles"]["value"] == 600
    aggregate_breakdowns = {
        breakdown["entity_id"]: breakdown
        for breakdown in result["breakdowns"]
        if breakdown["entity_id"].startswith(("chain/", "cascade/"))
    }
    assert len(aggregate_breakdowns) == 2
    assert all(
        next(
            metric["value"]
            for metric in breakdown["metrics"]
            if metric["name"] == "op_cycles"
        )
        == 1249
        for breakdown in aggregate_breakdowns.values()
    )
    assert output["target"]["components"] == [
        {"type": "gpu", "family": "mali", "model": "nx"}
    ]
    assert result["entity_kinds"] == [
        kind.to_dict() for kind in vgf_etdump.NX_ENTITY_KINDS
    ]
    assert "model_explorer" not in output["extensions"]
    validate_standardized_output(output)


def test_delegate_only_vgf_mapping_omits_source_operator_hierarchy(
    tmp_path: Path, monkeypatch, empty_capture
) -> None:
    """Delegate-only VGF results do not advertise a source mapping view."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    dump = tmp_path / "run.etdump"
    dump.write_bytes(b"dump")
    monkeypatch.setattr(
        vgf_etdump,
        "build_mappings",
        lambda *_args, **_kwargs: {
            "warning": "delegate-only mapping",
            "groups": [
                {
                    "event_block": "Execute",
                    "instruction_id": 0,
                    "segment_index": 0,
                    "segment_id": 0,
                    "stripe_id": 1,
                    "chain_ids": ["42"],
                    "cascade_ids": ["7"],
                    "method": "forward",
                    "statistics": {
                        "op_cycles": 1249,
                        "total_cycles": 1249,
                        "counters": {"nx_active": 1249},
                    },
                    "operator_group": {
                        "confidence": "delegate-only",
                        "strategy": "no source candidate available",
                        "backend_operations": [],
                        "nodes": [],
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(
        vgf_etdump,
        "build_standard_runtime_mappings",
        lambda *_args, **_kwargs: [],
    )

    output = vgf_etdump.analyze_etrecord_profiling_data(
        dump,
        model,
        target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
    )

    assert {entity["kind"] for entity in output["results"][0]["entities"]} == {
        "segment",
        "cascade",
        "chain",
        "stripe",
    }
    assert "model_explorer" not in output["extensions"]
    validate_standardized_output(output)


@pytest.mark.parametrize(
    "execution_positions",
    [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ],
    ids=("records-in-one-block", "same-named-blocks"),
)
@pytest.mark.parametrize("zero_op_cycles", [False, True])
def test_repeated_vgf_executions_are_summarized_per_inference(
    tmp_path: Path,
    monkeypatch,
    empty_capture,
    execution_positions: tuple[tuple[int, int], tuple[int, int]],
    zero_op_cycles: bool,
) -> None:
    """Repeated records and same-named blocks remain separate executions."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    dump = tmp_path / "run.etdump"
    dump.write_bytes(b"dump")
    groups = []
    for (event_block_index, record_index), cycles in zip(
        execution_positions, (1000, 3000)
    ):
        groups.append(
            {
                "event_block": "Execute",
                "event_block_index": event_block_index,
                "record_index": record_index,
                "instruction_id": 0,
                "segment_index": 0,
                "segment_id": 0,
                "stripe_id": 1,
                "chain_ids": ["42"],
                "cascade_ids": ["7"],
                "method": "forward",
                "statistics": {
                    "op_cycles": 0 if zero_op_cycles else cycles,
                    "total_cycles": cycles,
                    "counters": (
                        {"nx_active": 0, "ce_active": cycles}
                        if zero_op_cycles
                        else {"nx_active": cycles // 2}
                    ),
                },
                "operator_group": {
                    "confidence": "high-level-high",
                    "strategy": "semantic anchor",
                    "backend_operations": [],
                    "nodes": [
                        {
                            "name": "conv",
                            "debug_handle": 4,
                            "target": "aten.convolution",
                            "category": "conv",
                        }
                    ],
                },
            }
        )
    monkeypatch.setattr(
        vgf_etdump,
        "build_mappings",
        lambda *_args, **_kwargs: {
            "warning": "fused cycles",
            "groups": groups,
        },
    )
    monkeypatch.setattr(
        vgf_etdump,
        "build_standard_runtime_mappings",
        lambda *_args, **_kwargs: [],
    )

    output = vgf_etdump.analyze_etrecord_profiling_data(
        dump,
        model,
        target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
    )

    result = output["results"][0]
    entities = {entity["id"]: entity for entity in result["entities"]}
    entity_ids_by_kind = {
        kind: [
            entity_id
            for entity_id, entity in entities.items()
            if entity["kind"] == kind
        ]
        for kind in ("stripe", "chain", "cascade")
    }
    assert all(len(entity_ids) == 1 for entity_ids in entity_ids_by_kind.values())
    chain = entities[entity_ids_by_kind["chain"][0]]
    cascade = entities[entity_ids_by_kind["cascade"][0]]
    assert chain["attributes"]["stripe_ids"] == [1]
    assert cascade["attributes"]["stripe_ids"] == [1]

    breakdowns = {
        breakdown["entity_id"]: {
            metric["name"]: metric["value"] for metric in breakdown["metrics"]
        }
        for breakdown in result["breakdowns"]
    }
    for kind in ("stripe", "chain", "cascade"):
        metrics = breakdowns[entity_ids_by_kind[kind][0]]
        assert metrics["op_cycles"] == (0 if zero_op_cycles else 2000)
        assert metrics["total_cycles"] == 2000
        assert metrics["total_cycles_min"] == 1000
        assert metrics["total_cycles_max"] == 3000
        assert metrics["total_cycles_p50"] == 2000
        assert metrics["execution_count"] == 2
        assert metrics["nx_active_cycles"] == (0 if zero_op_cycles else 1000)
        if zero_op_cycles:
            assert metrics["ce_active_cycles"] == 2000

    model_metrics = {
        metric["name"]: metric.get("value") for metric in result["metrics"]
    }
    assert {
        name: value for name, value in model_metrics.items() if value is not None
    } == {
        "total_cycles": 2000,
        "total_cycles_min": 1000,
        "total_cycles_max": 3000,
        "total_cycles_p50": 2000,
        "execution_count": 2,
    }
    assert output["backends"][0]["configuration"]["execution_count"] == 2
    validate_standardized_output(output)


@pytest.mark.parametrize(
    ("second_executions", "can_aggregate"),
    [
        (((0, 0), (0, 1)), True),
        (((1, 0), (1, 1)), False),
        (((0, 2), (0, 3)), False),
        (((0, 0),), False),
    ],
    ids=("aligned", "different-blocks", "different-records", "partial-overlap"),
)
@pytest.mark.parametrize("partition", ["instruction", "segment"])
def test_vgf_mean_projection_requires_matching_executions(
    tmp_path: Path,
    monkeypatch,
    empty_capture,
    second_executions,
    can_aggregate: bool,
    partition: str,
) -> None:
    """Cycle and counter means can only sum over identical execution coverage."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    dump = tmp_path / "run.etdump"
    dump.write_bytes(b"dump")
    groups = []
    for instruction_id, executions, cycles in (
        (0, ((0, 0), (0, 1)), 100),
        (1, second_executions, 300),
    ):
        for block_index, record_index in executions:
            groups.append(
                {
                    "event_block": "Execute",
                    "event_block_index": block_index,
                    "record_index": record_index,
                    "instruction_id": instruction_id
                    if partition == "instruction"
                    else 0,
                    "segment_index": instruction_id if partition == "segment" else 0,
                    "segment_id": instruction_id if partition == "segment" else 0,
                    "stripe_id": 0,
                    "chain_ids": ["0"],
                    "cascade_ids": ["0"],
                    "method": "forward",
                    "statistics": {
                        "op_cycles": cycles,
                        "total_cycles": cycles,
                        "counters": {"nx_active": cycles},
                    },
                    "operator_group": {
                        "confidence": "delegate-only",
                        "strategy": "no source candidate available",
                        "backend_operations": [],
                        "nodes": [],
                    },
                }
            )
    monkeypatch.setattr(
        vgf_etdump,
        "build_mappings",
        lambda *_args, **_kwargs: {"warning": "fused cycles", "groups": groups},
    )
    monkeypatch.setattr(
        vgf_etdump, "build_standard_runtime_mappings", lambda *_args, **_kwargs: []
    )
    output = vgf_etdump.analyze_etrecord_profiling_data(
        dump,
        model,
        target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
    )
    validate_standardized_output(output)
    model_metrics = {
        metric["name"]: metric for metric in output["results"][0]["metrics"]
    }
    expected_model_values = {
        "total_cycles": 400,
        "total_cycles_min": 400,
        "total_cycles_max": 400,
        "total_cycles_p50": 400,
        "execution_count": 2,
    }
    for name, expected in expected_model_values.items():
        if can_aggregate:
            assert model_metrics[name]["value"] == expected
        else:
            assert model_metrics[name].get("value") is None
            assert model_metrics[name]["availability"] == "unavailable"
            assert "coverage" in model_metrics[name]["reason"].lower()
    configuration = output["backends"][0]["configuration"]
    if can_aggregate:
        assert configuration["execution_count"] == 2
    else:
        assert "execution_count" not in configuration
        assert any(
            "coverage" in warning.lower()
            for warning in output["results"][0]["warnings"]
        )
    result = schema.Result.from_dict(output["results"][0])
    result.entities.append(
        schema.Entity(
            id="inference",
            kind="inference",
            name="Inference",
            child_ids=[
                entity.id for entity in result.entities if entity.kind == "segment"
            ],
        )
    )
    result.entity_kinds.append(
        schema.EntityKind(id="inference", child_kinds=["segment"])
    )
    projected = project_entity_breakdowns(result)
    mean_names = {"op_cycles", "total_cycles", "nx_active_cycles"}
    group_means = {
        metric.name: metric.value
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "inference"
        for metric in breakdown.metrics
        if metric.name in mean_names
    }
    assert group_means == (dict.fromkeys(mean_names, 400) if can_aggregate else {})
    stripe_ids = {entity.id for entity in result.entities if entity.kind == "stripe"}
    for name in mean_names:
        assert sorted(
            metric.value
            for breakdown in projected.breakdowns
            if breakdown.entity_id in stripe_ids
            for metric in breakdown.metrics
            if metric.name == name
        ) == [100, 300]


def test_etdump_without_vgf_statistics_uses_standard_runtime_result(
    tmp_path: Path, monkeypatch, empty_capture
) -> None:
    """Ordinary ETDump events are supported without VGF statistics."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    dump = tmp_path / "run.etdump"
    dump.write_bytes(b"dump")

    def no_vgf_statistics(*_args, **_kwargs):
        raise vgf_etdump.NoVGFProfilingDataError(
            "No executable VGF neural-statistics stripes were found"
        )

    monkeypatch.setattr(vgf_etdump, "build_mappings", no_vgf_statistics)
    monkeypatch.setattr(
        vgf_etdump,
        "build_standard_runtime_mappings",
        lambda *_args, **_kwargs: [
            {
                "node": {
                    "name": "conv",
                    "debug_handle": 4,
                    "target": "aten.convolution.default",
                    "category": "conv",
                },
                "backend": None,
                "instruction_ids": [0],
                "metrics": [
                    {
                        "name": "Runtime Total Avg",
                        "value": 1.5,
                        "unit": "ms",
                        "aggregation": "sum",
                    }
                ],
            }
        ],
    )

    output = vgf_etdump.analyze_etrecord_profiling_data(
        dump,
        model,
        target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
    )

    assert output["target"]["components"] == [
        {"type": "gpu", "family": "mali", "model": "nx"}
    ]
    assert output["backends"][0]["configuration"]["profiling_kind"] == (
        "standard-etdump"
    )
    result = output["results"][0]
    assert result["producer"] == "neural-technology-profiling-data"
    assert result["entities"][0]["id"] == ("source_operator/etrecord/forward/conv")
    model_metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert {"total_cycles", "inference_time", "compiled_size"} <= model_metrics.keys()
    assert all(
        metric["availability"] == "unavailable"
        and metric.get("value") is None
        and metric["reason"]
        for metric in model_metrics.values()
    )
    assert result["breakdowns"][0]["metrics"] == [
        {
            "name": "Runtime Total Avg",
            "value": 1.5,
            "unit": "ms",
            "aggregation": "sum",
        }
    ]
    validate_standardized_output(output)


def test_etdump_without_vgf_or_standard_runtime_events_is_rejected(
    tmp_path: Path, monkeypatch, empty_capture
) -> None:
    """An ETDump must contain at least one supported profiling event."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    dump = tmp_path / "run.etdump"
    dump.write_bytes(b"dump")

    def no_vgf_statistics(*_args, **_kwargs):
        raise vgf_etdump.NoVGFProfilingDataError(
            "No executable VGF neural-statistics stripes were found"
        )

    monkeypatch.setattr(vgf_etdump, "build_mappings", no_vgf_statistics)
    monkeypatch.setattr(
        vgf_etdump,
        "build_standard_runtime_mappings",
        lambda *_args, **_kwargs: [],
    )
    with pytest.raises(
        ConfigurationError,
        match="No ordinary ETDump runtime events matched ETRecord debug handles",
    ):
        vgf_etdump.analyze_etrecord_profiling_data(
            dump,
            model,
            target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
        )


def test_mixed_backend_etdump_partitions_vgf_and_standard_events(
    tmp_path: Path, monkeypatch, empty_capture
) -> None:
    """Non-VGF instructions remain visible beside VGF neural statistics."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    dump = tmp_path / "run.etdump"
    dump.write_bytes(b"dump")
    monkeypatch.setattr(
        vgf_etdump,
        "build_mappings",
        lambda *_args, **_kwargs: {
            "warning": "fused cycles",
            "profiled_instruction_ids": [0],
            "groups": [
                {
                    "event_block": "Execute",
                    "instruction_id": 0,
                    "segment_index": 0,
                    "segment_id": 0,
                    "stripe_id": 1,
                    "chain_ids": ["42"],
                    "cascade_ids": ["7"],
                    "method": "forward",
                    "statistics": {
                        "op_cycles": 1249,
                        "total_cycles": 1249,
                        "counters": {"nx_active": 1249},
                    },
                    "operator_group": {
                        "confidence": "high-level-high",
                        "strategy": "semantic anchor",
                        "backend_operations": [],
                        "nodes": [
                            {
                                "name": "vgf_conv",
                                "debug_handle": 4,
                                "target": "aten.convolution",
                                "category": "conv",
                            }
                        ],
                    },
                }
            ],
        },
    )
    excluded_instruction_ids = None

    def standard_mappings(*_args, **kwargs):
        nonlocal excluded_instruction_ids
        excluded_instruction_ids = kwargs["excluded_instruction_ids"]
        return [
            {
                "node": {
                    "name": "xnnpack_relu",
                    "debug_handle": 8,
                    "target": "aten.relu.default",
                    "category": "relu",
                },
                "backend": "XnnpackBackend",
                "instruction_ids": [1],
                "metrics": [
                    {
                        "name": "Runtime Total Avg",
                        "value": 0.5,
                        "unit": "ms",
                        "aggregation": "sum",
                    }
                ],
            }
        ]

    monkeypatch.setattr(
        vgf_etdump,
        "build_standard_runtime_mappings",
        standard_mappings,
    )

    output = vgf_etdump.analyze_etrecord_profiling_data(
        dump,
        model,
        target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
    )

    assert excluded_instruction_ids == {0}
    assert [backend["id"] for backend in output["backends"]] == [
        "neural-technology-profiling-data"
    ]
    assert output["backends"][0]["configuration"]["profiling_kind"] == (
        "mixed-vgf-and-standard-etdump"
    )
    assert len(output["results"]) == 2
    vgf_result, standard_result = output["results"]
    assert vgf_result["producer"] == "neural-technology-profiling-data"
    assert standard_result["producer"] == "neural-technology-profiling-data"
    assert standard_result["entities"] == [
        {
            "id": "source_operator/etrecord/forward/xnnpack_relu",
            "kind": "source_operator",
            "name": "xnnpack_relu",
            "attributes": {
                "debug_handle": 8,
                "target": "aten.relu.default",
                "category": "relu",
                "runtime_backend": "XnnpackBackend",
                "runtime_instruction_ids": [1],
                "runtime_attributions": [
                    {"backend": "XnnpackBackend", "instruction_ids": [1]}
                ],
            },
        }
    ]
    assert "model_explorer" not in output["extensions"]
    validate_standardized_output(output)


@pytest.mark.parametrize(
    "mode,device,words_per_block,stripe_ids",
    [
        # Each hardware layout retains referenced zero blocks. Sparse and
        # reordered IDs exercise selection independently of binary decoding.
        (0, "Mali-G1", 16 * 512, [0, 1, 2, 3]),
        (0, "Mali-G2", 16 * 1024, [0, 1, 2, 3]),
        (1, "Mali-G1", 32 * 256, [0, 1, 2, 3]),
        (1, "Mali-G2", 64 * 256, [0, 1, 2, 3]),
        (1, "Mali-G2", 64 * 256, [1, 2, 3]),
        (1, "Mali-G2", 64 * 256, [3, 1]),
    ],
)
def test_etdump_statistics_selects_blocks_by_stripe_id(
    mode, device, words_per_block, stripe_ids
):
    """Sparse, reordered stripe IDs exclude padding and preserve measured zeroes."""
    words = [0] * (4 * words_per_block)
    words[words_per_block + 1] = 11
    words[3 * words_per_block + 1] = 33
    rows = vgf_metadata.parse_etdump_statistics(
        words, stripe_ids, mode=mode, device=device
    )
    assert [row["total_cycles"] for row in rows] == [
        {0: 0, 1: 11, 2: 0, 3: 33}[stripe_id] for stripe_id in stripe_ids
    ]


@pytest.mark.parametrize(
    "stripe_ids,expected",
    [([1], "nonzero"), ([0, 1, 2], "outside"), ([0, 0, 1], "duplicate")],
)
def test_etdump_statistics_rejects_invalid_block_associations(stripe_ids, expected):
    """Selecting stripe IDs never discards nonzero data or invents missing blocks."""
    words = [0] * (2 * 64 * 256)
    words[1] = 5
    words[64 * 256 + 1] = 11
    with pytest.raises(ConfigurationError, match=expected):
        vgf_metadata.parse_etdump_statistics(
            words, stripe_ids, mode=1, device="Mali-G2"
        )


def test_runtime_fallback_reads_capture_once(monkeypatch):
    """Trying VGF metadata then ordinary timings reuses the detached capture."""
    node = {"name": "conv", "target": "aten.convolution.default", "debug_handle": 42}
    capture = ProfilingCapture(
        method="forward",
        etrecord_path="model.etrecord",
        etdump_path="run.etdump",
        events=(
            ETDumpEvent(
                "forward", "Execute", 0, 0, "conv", 0, None, (42,), None, (), (3.0, 5.0)
            ),
        ),
        nodes_by_handle={42: [node]},
        instruction_handles={0: (42,)},
        source_graph={},
    )
    reader = Mock(return_value=capture)
    monkeypatch.setattr(vgf_etdump, "read_profiling_capture", reader)
    metrics = vgf_etdump.load_etdump_metrics(Path("run.etdump"), Path("model.etrecord"))
    values = {
        metric.name: metric.value
        for metric in metrics.measured_observations[0].breakdowns[0].metrics
    }
    assert values["Runtime Total Avg"] == 4.0
    reader.assert_called_once()
