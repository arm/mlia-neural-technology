# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Publish VGF companion associations as ordinary measured-analysis results."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Use the vendored VGF 0.10.0 binding to avoid adding a separate dependency
# on upstream vgfpy.
from mlia._vendor.neural_technology import vgfpy
from mlia.backend.executorch.pte import PTEReader
from mlia.core.errors import ConfigurationError

from mlia.backend.neural_technology_profiling_data.executorch_source import (
    project_dataflow,
)
from mlia.backend.nx_performance_estimator.debug_info import (
    read_spirv_debug_names_from_bytes,
)
from mlia.backend.nx_performance_estimator.vgf import _decode_vgf, _segment_spirv


def vgf_segment_operators(payload: bytes) -> dict[int, set[int]]:
    """Index graph operation identities in each embedded VGF segment."""
    decoded = _decode_vgf(memoryview(payload), vgfpy, Path("companion.vgf"))
    return {
        index: {
            int(value)
            for value in read_spirv_debug_names_from_bytes(
                _segment_spirv(decoded, index)
            ).op_ext_inst_spirv_ids
        }
        for index in range(decoded.sequence.modelSequenceTableSize())
        if decoded.sequence.getSegmentType(index) == vgfpy.ModuleType.Graph
    }


def _ancestor(
    entity: dict[str, Any], entities: dict[str, Any], kind: str
) -> dict[str, Any] | None:
    pending = list(entity.get("parent_ids", []))
    seen = set()
    while pending:
        identity = pending.pop()
        if identity in seen:
            continue
        seen.add(identity)
        parent = entities.get(identity, {})
        if parent.get("kind") == kind:
            return parent
        pending.extend(parent.get("parent_ids", []))
    return None


def _operator_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", value.removeprefix("TOSA").lower())
    return "table" if key == "lut" else key


def _operation_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect compiler provenance while keeping source and backend IDs distinct."""
    entities = {entity["id"]: entity for entity in result.get("entities", [])}
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    for chain in entities.values():
        if chain["kind"] != "chain":
            continue
        segment = _ancestor(chain, entities, "segment")
        if segment is None:
            continue
        for child_id in chain.get("child_ids", []):
            stripe = entities.get(child_id, {})
            if stripe.get("kind") != "stripe":
                continue
            for operation in stripe.get("attributes", {}).get("backend_operations", []):
                for api_label in operation.get("api_labels", []):
                    match = re.fullmatch(r"(.+)_spirv_id_([0-9]+)", api_label)
                    if match is None:
                        continue
                    spirv_id = int(match.group(2))
                    record = records.setdefault(
                        (segment["id"], spirv_id, api_label),
                        {
                            "partition_id": segment["id"],
                            **segment.get("attributes", {}),
                            "spirv_id": spirv_id,
                            "api_label": api_label,
                            "chain_ids": [],
                            "chain_names": [],
                            "backend_operator_ids": [],
                            "matching_chain_ids": [],
                            "compiler_operations": [],
                        },
                    )
                    for key, value in (
                        ("chain_ids", chain["id"]),
                        ("chain_names", chain.get("name", chain["id"])),
                        ("compiler_operations", operation.get("operator", "BackendOp")),
                    ):
                        if value not in record[key]:
                            record[key].append(value)
                    if _operator_key(operation.get("operator", "")) == _operator_key(
                        match.group(1)
                    ):
                        if chain["id"] not in record["matching_chain_ids"]:
                            record["matching_chain_ids"].append(chain["id"])
                        entity_id = operation.get("entity_id")
                        if (
                            entity_id in entities
                            and entity_id not in record["backend_operator_ids"]
                        ):
                            record["backend_operator_ids"].append(entity_id)
    for record in records.values():
        candidates = (
            record["chain_ids"]
            if len(record["chain_ids"]) == 1
            else record["matching_chain_ids"]
        )
        record["chain_id"] = candidates[0] if len(candidates) == 1 else None
        del record["matching_chain_ids"]
    return [records[key] for key in sorted(records)]


def enrich_output(output: dict[str, Any], companion_model: Path | None = None) -> None:
    """Publish source dataflow and structural companion links before presentation."""
    metadata = output.setdefault("extensions", {}).setdefault(
        "executorch", {"schema_version": 1}
    )
    edges = metadata.get("source_graph", {}).get("edges", [])
    metadata["chain_relationships"] = [
        {"result_index": index, **relationship}
        for index, result in enumerate(output.get("results", []))
        for relationship in project_dataflow(result, edges, "chain")
    ]
    if companion_model is None:
        return
    try:
        reader = PTEReader(companion_model)
        delegates = reader.delegates()
        companion = reader.metadata()
        associations = []
        indexes = {}
        for result_index, result in enumerate(output.get("results", [])):
            for record in _operation_records(result):
                candidates = [
                    delegate
                    for delegate in delegates
                    if delegate.backend == "VgfBackend"
                    and delegate.method == record.get("method")
                    and delegate.instruction_id == record.get("instruction_id")
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"VGF partition {record['partition_id']!r} requires one companion delegate for "
                        f"method {record.get('method')!r}, instruction {record.get('instruction_id')}; found {len(candidates)}"
                    )
                delegate = candidates[0]
                if delegate.id not in indexes:
                    indexes[delegate.id] = vgf_segment_operators(delegate.payload)
                segment_index = int(record["segment_index"])
                if record["spirv_id"] not in indexes[delegate.id].get(
                    segment_index, set()
                ):
                    raise ValueError(
                        f"Companion delegate {delegate.id!r}, segment {segment_index} has no "
                        f"SPIR-V operation {record['spirv_id']} referenced by ETDump"
                    )
                associations.append(
                    {
                        **record,
                        "result_index": result_index,
                        "delegate_id": delegate.id,
                        "segment_index": segment_index,
                        "association": "structural-spirv-association",
                    }
                )
        companion["operator_associations"] = associations
        metadata["companion_model"] = companion
    except (
        ImportError,
        OSError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ConfigurationError(f"Unable to analyze companion model: {exc}") from exc
