# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Decode VGF neural-statistics metadata and binary payloads without the SDK."""

import base64
import binascii
import json
import struct
from typing import Any, cast

from mlia.backend.neural_technology_profiling_data.parser import (
    ProfilingDevice,
    ProfilingMode,
    _profiling_device,
    parse_statistics_words,
    select_statistics_rows,
)
from mlia.nx_utils.debug_database import parse_debug_table

DEBUG_DATABASE_MAGIC = 0x1122FFEE
STATISTICS_INFO_MAGIC = 0x2211EEFF
ARTIFACT_VERSION = 1
SECTION_NAMES = (
    "tosa_ops",
    "api_labels",
    "fused_ops",
    "cascade_ops",
    "cascade_groups",
    "stripe_ops",
    "stripe_indexes",
)


def resolve_decoder_settings(
    record: dict[str, Any], fallback_mode: int, fallback_device: str
) -> tuple[int, str, list[str]]:
    """Read schema-v2 interpretation metadata or expose legacy assumptions."""
    fallback_fields: list[str] = []
    device: str | None = None
    if record["schema_version"] == 1:
        mode = fallback_mode
        fallback_fields.append("statistics_mode")
    else:
        for name in ("statistics_mode", "target", "counter_layout"):
            if not isinstance(record.get(name), dict):
                raise ValueError(f"VGF decoder metadata requires a {name} object")
        mode = record["statistics_mode"].get("executorch_index")
    if type(mode) is not int or mode not in (0, 1):
        raise ValueError("VGF decoder statistics_mode must be the integer 0 or 1")

    if record["schema_version"] == 2:
        mode_metadata = record["statistics_mode"]
        if (
            type(mode_metadata.get("vulkan_value")) is not int
            or mode_metadata["vulkan_value"] != mode + 1
            or mode_metadata.get("vulkan_name")
            != f"VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS{mode}_ARM"
        ):
            raise ValueError(
                "VGF decoder statistics_mode Vulkan fields disagree with executorch_index"
            )
        target = record["target"]
        layout = record["counter_layout"]
        for name, value in (("target", target), ("counter_layout", layout)):
            if type(value.get("available")) is not bool:
                raise ValueError(f"VGF decoder {name}.available must be boolean")
        if target["available"]:
            device_name = target.get("device_name")
            if not isinstance(device_name, str):
                raise ValueError("VGF decoder target.device_name must be a string")
            device = _profiling_device(
                device_name, target.get("vendor_id"), "VGF decoder target"
            )
        if layout["available"]:
            known_layouts = {
                f"arm.{generation.lower()}.neural-statistics.mode{mode}": generation
                for generation in ("Mali-G1", "Mali-G2")
            }
            schema_id = layout.get("schema_id")
            if not isinstance(schema_id, str) or schema_id not in known_layouts:
                raise ValueError(
                    f"Unsupported VGF decoder counter_layout.schema_id: {schema_id!r}"
                )
            layout_device = known_layouts[schema_id]
            if device is not None and device != layout_device:
                raise ValueError(
                    "VGF decoder counter_layout disagrees with target device"
                )
            device = layout_device
            expected_layout = {
                "schema_version": 1,
                "endianness": "little",
                "word_type": "uint32",
                "words_per_task": 16
                if mode == 0
                else (32 if device == "Mali-G1" else 64),
                "leading_block_count": 1,
            }
            for name, expected in expected_layout.items():
                value = layout.get(name)
                if type(value) is not type(expected) or value != expected:
                    raise ValueError(
                        f"Unsupported VGF decoder counter_layout.{name}: "
                        f"{value!r}; expected {expected!r}"
                    )
        else:
            fallback_fields.append("counter_layout")

    if device is None:
        device = fallback_device
        fallback_fields.append("profiling_device")
    if device not in ("Mali-G1", "Mali-G2"):
        raise ValueError("VGF decoder profiling_device must be Mali-G1 or Mali-G2")
    return (
        mode,
        device,
        [
            name
            for name in ("statistics_mode", "profiling_device", "counter_layout")
            if name in fallback_fields
        ],
    )


def decode_vgf_metadata(metadata: bytes | str | None) -> dict[str, Any]:
    """Validate one VGF execution and decode its available binary payloads."""
    if metadata is None:
        raise ValueError("VGF event has no decoded profiling records: missing metadata")
    try:
        payload = json.loads(metadata)
    except (TypeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "VGF event has no decoded profiling records: invalid JSON"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != (
        "executorch.vgf.neural_statistics"
    ):
        raise ValueError(
            "VGF event has no decoded profiling records: unexpected schema"
        )
    if type(payload.get("schema_version")) is not int or payload[
        "schema_version"
    ] not in (1, 2):
        raise ValueError("Unsupported VGF neural-statistics schema version")
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("VGF neural-statistics segments must be a list")
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("VGF neural-statistics segment must be an object")
        for name in ("debug_database", "statistics_info", "statistics_memory"):
            blob = segment.get(name, {})
            if not isinstance(blob, dict):
                raise ValueError(f"VGF {name} blob must be an object")
            if not blob.get("available", False):
                continue
            if blob.get("encoding") != "base64":
                raise ValueError(f"Unsupported VGF {name} blob encoding")
            try:
                blob["raw_data"] = base64.b64decode(blob.get("data", ""), validate=True)
            except (binascii.Error, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed base64 data in VGF {name} blob") from exc
    return payload


def extract_versioned_payload(data: bytes, expected_magic: int, name: str) -> bytes:
    """Extract and validate a versioned binary artifact payload."""
    if len(data) < 12:
        return data
    magic, version, payload_size = struct.unpack_from("<III", data)
    if magic != expected_magic:
        return data
    if version != ARTIFACT_VERSION:
        raise ValueError(f"{name} has unsupported version {version}")
    if payload_size != len(data) - 12:
        raise ValueError(
            f"{name} declares {payload_size} bytes but contains {len(data) - 12}"
        )
    return data[12:]


def parse_debug_database(data: bytes) -> dict[str, Any]:
    """Parse a VGF debug database and build its entity relationships."""
    payload = extract_versioned_payload(
        data, DEBUG_DATABASE_MAGIC, "VGF debug database"
    ).rstrip(b"\0")
    lines = [line.rstrip("\r\0") for line in payload.decode("utf-8").splitlines()]
    sections: list[list[str]] = []
    if "----" not in lines:
        # Legacy captures store each header-only table on a separate line.
        sections = [[line] for line in lines if line]
    else:
        current: list[str] = []
        for line in lines:
            if line == "----":
                sections.append(current)
                current = []
            elif line:
                current.append(line)
        if current:
            sections.append(current)

    if len(sections) < len(SECTION_NAMES):
        raise ValueError(
            "VGF debug database does not contain all expected profiling sections: "
            f"expected {len(SECTION_NAMES)}, found {len(sections)}"
        )
    empty_sections = [
        name for index, name in enumerate(SECTION_NAMES) if not sections[index]
    ]
    if empty_sections:
        raise ValueError(
            "VGF debug database has missing or empty sections: "
            f"{', '.join(empty_sections)}"
        )
    parsed = {
        name: parse_debug_table(sections[index])
        for index, name in enumerate(SECTION_NAMES)
    }
    return build_debug_relationships(parsed)


def rows_by_id(rows: list[dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    """Index debug-database rows by their first identifier."""
    result = {}
    for row in rows:
        identifiers = row.get("id", [])
        if identifiers:
            result[identifiers[0]] = row
    return result


def build_debug_relationships(sections: dict[str, Any]) -> dict[str, Any]:
    """Build stripe-to-compiler-operation relationships from debug sections."""
    tosa_ops = rows_by_id(sections["tosa_ops"])
    api_labels = rows_by_id(sections["api_labels"])
    fused_ops = rows_by_id(sections["fused_ops"])
    chain_ops = rows_by_id(sections["cascade_ops"])
    stripe_ops = rows_by_id(sections["stripe_ops"])
    stripe_indexes = rows_by_id(sections["stripe_indexes"])

    for identifier, row in api_labels.items():
        tosa_ops.setdefault(identifier, row)

    stripe_index_to_op = {}
    for identifier, row in stripe_indexes.items():
        stripe_ids = row.get("stripe_op_id", [])
        if stripe_ids:
            stripe_index_to_op[identifier] = stripe_ids[0]
        for stripe_index in row.get("stripe_op_indexs", []):
            stripe_index_to_op.setdefault(stripe_index, stripe_index)

    stripe_to_chain = {}
    stripe_to_cascade = {}
    for stripe_index, stripe_op_id in stripe_index_to_op.items():
        stripe = stripe_ops.get(stripe_op_id)
        if stripe is None:
            continue
        stripe_to_chain[stripe_index] = stripe.get("op_id", [])
        stripe_to_cascade[stripe_index] = stripe.get("cascade_op_id", [])

    return {
        "stripe_to_chain": stripe_to_chain,
        "stripe_to_cascade": stripe_to_cascade,
        "chain_to_fused": {
            identifier: row.get("fused_op_ids", [])
            for identifier, row in chain_ops.items()
        },
        "fused_to_tosa": {
            identifier: row.get("tosa_op_ids", [])
            for identifier, row in fused_ops.items()
        },
        "tosa_ops": tosa_ops,
    }


def stripe_operations(debug: dict[str, Any], stripe_id: int) -> list[dict[str, Any]]:
    """Return the ordered compiler operations associated with a stripe."""
    stripe_key = str(stripe_id)
    operations = []
    seen = set()
    for chain_id in debug["stripe_to_chain"].get(stripe_key, []):
        fused_ids = debug["chain_to_fused"].get(chain_id, [])
        if not fused_ids:
            fused_ids = [chain_id]
        for fused_id in fused_ids:
            tosa_ids = debug["fused_to_tosa"].get(fused_id, [])
            if not tosa_ids:
                tosa_ids = [fused_id]
            for tosa_id in tosa_ids:
                if tosa_id in seen:
                    continue
                seen.add(tosa_id)
                row = debug["tosa_ops"].get(tosa_id, {})
                operations.append(
                    {
                        "tosa_id": tosa_id,
                        "operator": (row.get("tosa_op") or ["BackendOp"])[0],
                        "api_labels": row.get("api_labels", []),
                    }
                )
    return operations


def parse_statistics_info(blob: dict[str, Any]) -> list[int]:
    """Decode stripe identifiers from a VGF statistics-info blob."""
    data = blob["raw_data"]
    if blob.get("is_text", False):
        text = data.decode("utf-8").rstrip("\0").replace(",", " ").replace(";", " ")
        return [
            int(token, 16) if token.lower().startswith("0x") else int(token, 10)
            for token in text.split()
        ]
    payload = extract_versioned_payload(data, STATISTICS_INFO_MAGIC, "Statistics info")
    if len(payload) % 4:
        raise ValueError("Statistics info size is not 4-byte aligned")
    return [value[0] for value in struct.iter_unpack("<I", payload)]


def decode_words(data: bytes) -> list[int]:
    """Decode little-endian 32-bit words from statistics memory."""
    if len(data) % 4:
        raise ValueError("Statistics memory size is not 4-byte aligned")
    return [value[0] for value in struct.iter_unpack("<I", data)]


def parse_etdump_statistics(
    words: list[int],
    stripe_ids: list[int],
    *,
    mode: int,
    device: str,
) -> list[dict[str, Any]]:
    """Decode and select ETDump blocks through the structured-capture parser."""
    rows = select_statistics_rows(
        parse_statistics_words(
            words,
            mode=cast(ProfilingMode, mode),
            device=cast(ProfilingDevice, device),
            included_zero_block_ids=frozenset(stripe_ids),
        ),
        stripe_ids,
    )
    return [
        {
            "op_cycles": row["opCycles"],
            "total_cycles": row["totalCycles"],
            "counters": {
                item["sectionName"]: item["cycles"] for item in row["Utilization"]
            },
        }
        for row in rows
    ]


def parse_mode1(words: list[int], device: str) -> list[dict[str, Any]]:
    """Decode all mode-1 blocks, including zero blocks, using common rules."""
    words_per_block = (32 if device == "Mali-G1" else 64) * 256
    block_count = (len(words) + words_per_block - 1) // words_per_block
    return parse_etdump_statistics(
        words, list(range(block_count)), mode=1, device=device
    )


def parse_mode0(words: list[int], device: str) -> list[dict[str, Any]]:
    """Decode all mode-0 blocks, including zero blocks, using common rules."""
    words_per_block = 16 * (512 if device == "Mali-G1" else 1024)
    block_count = (len(words) + words_per_block - 1) // words_per_block
    return parse_etdump_statistics(
        words, list(range(block_count)), mode=0, device=device
    )
