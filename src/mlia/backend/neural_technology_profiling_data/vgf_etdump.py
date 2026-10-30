# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Map measured VGF ETDump events onto an ETRecord exported-program graph."""

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mlia.core import output_schema as schema
from mlia.core.errors import ConfigurationError
from mlia.core.performance import MeasuredPerformanceMetrics, PerformanceObservations

from mlia.nx_utils.boundaries.filesystem import (
    ensure_user_output_directory,
    write_user_output_bytes,
)
from mlia.nx_utils.performance import NX_ENTITY_KINDS, normalize_nx_observations

try:
    from mlia.backend.executorch.capture import ProfilingCapture
    from mlia.backend.executorch.etdump import read_profiling_capture
    from mlia.backend.executorch.etrecord import source_operator_id

    from mlia.backend.neural_technology_profiling_data.executorch_runtime import (
        build_standard_runtime_mappings,
        standard_runtime_metrics,
        standard_runtime_observations,
    )
except ImportError as exc:
    raise ConfigurationError(
        "Shared ExecuTorch profiling support is required for ETRecord/ETDump analysis. "
        "Install mlia-neural-technology[executorch]."
    ) from exc

from mlia.backend.neural_technology_profiling_data.executorch_source import (
    node_category,
)
from mlia.backend.neural_technology_profiling_data.vgf_metadata import (
    decode_vgf_metadata,
    decode_words,
    parse_debug_database,
    parse_etdump_statistics,
    parse_statistics_info,
    resolve_decoder_settings,
    stripe_operations,
)


@dataclass
class OperatorMappingState:
    """Track representative source nodes assigned to backend operations."""

    assigned_nodes: dict[tuple[Any, ...], set[str]] = field(default_factory=dict)
    backend_to_node: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    source_indexes: dict[tuple[Any, ...], dict[str, dict[str, dict[str, Any]]]] = field(
        default_factory=dict
    )


class NoVGFProfilingDataError(ValueError):
    """ETDump contains no VGF neural-statistics events."""


def operation_categories(operation: dict[str, Any]) -> set[str]:
    """Prefer explicit operator identity, falling back to complete label tokens."""
    rules = {
        "conv": ("conv2d", "depthwiseconv2d", "convolution"),
        "quantization": (
            "rescale",
            "cast",
            "quant",
            "dequant",
            "quantize",
            "dequantize",
        ),
        "relu": ("relu", "clamp"),
        "add": ("add",),
        "sub": ("sub",),
        "mul": ("mul",),
        "transpose": ("transpose", "permute"),
        "matmul": ("matmul", "fullyconnected", "linear"),
        "pool": ("pool", "pool2d", "avgpool2d", "maxpool2d"),
        "softmax": ("softmax",),
        "reshape": ("reshape", "view", "squeeze", "unsqueeze"),
        "reduction": ("reducesum", "sum", "mean"),
    }

    def classify(text: str) -> set[str]:
        return {
            category
            for category, needles in rules.items()
            if any(
                re.search(rf"(?<![a-z0-9])(?:tosa)?{needle}(?![a-z0-9])", text.lower())
                for needle in needles
            )
        }

    labels = []
    for label in operation.get("api_labels", []):
        try:
            metadata = json.loads(label)
        except ValueError:
            labels.append(label)
            continue
        # Structured labels contain source code and compiler histories. Only
        # their explicit operator identity is suitable for classification.
        if isinstance(metadata, dict):
            identity = metadata.get("aten_info", {})
            if isinstance(identity, dict):
                labels.append(
                    str(identity.get("operator_name", identity.get("node_name", "")))
                )
    return classify(operation["operator"]) or classify(" ".join(labels)) or {"unknown"}


def _provenance_node(
    operation: dict[str, Any], source_index: dict[str, dict[str, dict[str, Any]]]
) -> tuple[bool, dict[str, Any] | None]:
    """Require an unambiguous recorded name-and-target match."""
    has_provenance = False
    named_matches = {}
    for label in operation.get("api_labels", []):
        try:
            metadata = json.loads(label)
        except ValueError:
            continue
        if not isinstance(metadata, dict):
            continue
        info = metadata.get("torch_info", {})
        traces = info.get("node_trace", []) if isinstance(info, dict) else []
        if not isinstance(traces, list) or not traces:
            continue
        has_provenance = True
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            target = trace.get("target")
            name = trace.get("name")
            if not isinstance(target, str) or not isinstance(name, str):
                continue
            # ETRecord may qualify names with namespace/operator/overload,
            # while compiler histories retain the original FX operator name.
            parts = target.split(".")
            if len(parts) != 3 or not all(part.isidentifier() for part in parts):
                continue
            suffix = re.fullmatch(re.escape(parts[1]) + r"(_[0-9]+)?", name)
            qualified = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", target)
            qualified = qualified.replace(".", "_").lower()
            if suffix:
                qualified += suffix[1] or ""
            candidates = source_index.get(target, {})
            for candidate_name in (name, qualified if suffix else name):
                if candidate_name in candidates:
                    named_matches[candidate_name] = candidates[candidate_name]
    return (
        has_provenance,
        next(iter(named_matches.values())) if len(named_matches) == 1 else None,
    )


def map_operator_group(
    operations: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    scope: tuple[Any, ...],
    state: OperatorMappingState,
) -> dict[str, Any]:
    """Prefer compiler provenance; retain ordered matching for legacy labels."""
    if scope not in state.source_indexes:
        source_index: dict[str, dict[str, dict[str, Any]]] = {}
        for source_node in nodes:
            identity = re.search(r"\b\w+\.\w+\.\w+\b", source_node["target"])
            if identity is not None:
                source_index.setdefault(identity[0], {})[source_node["name"]] = (
                    source_node
                )
        state.source_indexes[scope] = source_index
    operation_entries = []
    recorded_nodes = []
    has_provenance = False
    for operation in operations:
        recorded, node = _provenance_node(operation, state.source_indexes[scope])
        has_provenance |= recorded
        recorded_nodes.append(node)
        operation_entries.append(
            {
                **operation,
                "categories": sorted(operation_categories(operation)),
                "matched": False,
            }
        )

    selected_node = next((node for node in recorded_nodes if node is not None), None)
    from_provenance = selected_node is not None
    assigned = state.assigned_nodes.setdefault(scope, set())
    if not has_provenance:
        for operation in operation_entries:
            categories = set(operation["categories"]) - {
                "quantization",
                "reshape",
                "unknown",
            }
            for category in sorted(categories):
                backend_key = (*scope, category, str(operation["tosa_id"]))
                selected_node = state.backend_to_node.get(backend_key)
                if selected_node is None:
                    selected_node = next(
                        (
                            node
                            for node in nodes
                            if node["category"] == category
                            and node["name"] not in assigned
                        ),
                        None,
                    )
                if selected_node is not None:
                    state.backend_to_node[backend_key] = selected_node
                    break
            if selected_node is not None:
                break

    selected = [selected_node] if selected_node is not None else []
    if selected_node is not None:
        assigned.add(selected_node["name"])
    for operation, recorded_node in zip(operation_entries, recorded_nodes):
        operation["matched"] = (
            recorded_node == selected_node and selected_node is not None
            if has_provenance
            else bool(
                selected_node and selected_node["category"] in operation["categories"]
            )
        )

    return {
        "mapping_type": "representative_source_operator",
        "confidence": "recorded-provenance"
        if from_provenance
        else "high-level-low"
        if selected
        else "delegate-only",
        "strategy": (
            "recorded compiler operator provenance"
            if from_provenance
            else "ordered backend-operation representative inside delegate"
            if selected
            else "no unambiguous source candidate available; retain delegate-level association"
        ),
        "primary_node": selected_node,
        "debug_handles": sorted({node["debug_handle"] for node in selected}),
        "nodes": selected,
        "backend_operations": operation_entries,
        "unmatched_backend_operations": [
            operation["operator"]
            for operation in operation_entries
            if not operation["matched"]
        ],
    }


def build_mappings(
    capture: ProfilingCapture,
    mode: int,
    device: str,
) -> dict[str, Any]:
    """Attribute decoded VGF stripes using detached recorded relationships."""
    method = capture.method
    groups = []
    decoder_settings: tuple[int, str] | None = None
    decoder_fallback_fields: set[str] = set()
    mapping_state = OperatorMappingState()

    for event in capture.events:
        if event.name != "VGF_NEURAL_STATISTICS":
            continue
        method = event.method
        event_block_index = event.event_block_index
        instruction_id = event.instruction_id
        if instruction_id is None:
            raise ValueError("VGF_NEURAL_STATISTICS has no instruction ID")
        handles = capture.instruction_handles.get(instruction_id, ())
        nodes = [
            {**node, "category": node_category(node["name"], node["target"])}
            for node in capture.nodes_for_instruction(instruction_id)
        ]
        if not nodes:
            raise ValueError(
                f"VGF_NEURAL_STATISTICS for method {method!r}, instruction "
                f"{instruction_id}, block {event_block_index} ({event.event_block!r}) has no "
                f"ETRecord graph nodes matching debug handles {sorted(handles)}. "
                "Use an ETDump and ETRecord from the same deployed program."
            )
        event_context = (
            f"VGF_NEURAL_STATISTICS for method {method!r}, instruction "
            f"{instruction_id}, block {event_block_index} ({event.event_block!r})"
        )
        if not event.raw_delegate_metadata:
            raise ValueError(
                f"{event_context} has no decoded profiling records. "
                "Cannot produce complete profiling totals."
            )

        for record_index, raw_metadata in enumerate(event.raw_delegate_metadata):
            # Decode executions individually: missing payloads must not shift
            # the indices that establish execution cohorts across instructions.
            try:
                record = decode_vgf_metadata(raw_metadata)
            except ValueError as exc:
                raise ValueError(
                    f"{event_context}, record {record_index}: {exc}"
                ) from exc
            record_mode, record_device, fallback_fields = resolve_decoder_settings(
                record, mode, device
            )
            record_settings = (record_mode, record_device)
            if decoder_settings is not None and record_settings != decoder_settings:
                raise ValueError(
                    f"Conflicting VGF decoder settings in {event_context}, "
                    f"record {record_index}: {record_settings!r}; "
                    f"previous records use {decoder_settings!r}"
                )
            decoder_settings = record_settings
            decoder_fallback_fields.update(fallback_fields)
            segments = record.get("segments", [])
            if not segments:
                raise ValueError(
                    f"{event_context}, record {record_index} has no segments. "
                    "Cannot produce complete profiling totals."
                )
            for segment_index, segment in enumerate(segments):
                segment_id = segment.get("segment_id", segment_index)
                debug_blob = segment.get("debug_database", {})
                info_blob = segment.get("statistics_info", {})
                memory_blob = segment.get("statistics_memory", {})
                missing_blobs = [
                    name
                    for name, blob in (
                        ("debug_database", debug_blob),
                        ("statistics_info", info_blob),
                        ("statistics_memory", memory_blob),
                    )
                    if not blob.get("available", False)
                ]
                if missing_blobs:
                    raise ValueError(
                        f"VGF_NEURAL_STATISTICS segment {segment_id} "
                        f"(method {method!r}, instruction {instruction_id}, "
                        f"block {event_block_index}, record {record_index}) "
                        f"has unavailable required blobs: {', '.join(missing_blobs)}. "
                        "Cannot produce complete profiling totals."
                    )

                debug = parse_debug_database(debug_blob["raw_data"])
                stripe_ids = parse_statistics_info(info_blob)
                if not stripe_ids:
                    raise ValueError(
                        f"{event_context}, record {record_index}, segment "
                        f"{segment_id} has no stripe IDs. "
                        "Cannot produce complete profiling totals."
                    )
                words = decode_words(memory_blob["raw_data"])
                performance_rows = parse_etdump_statistics(
                    words, stripe_ids, mode=record_mode, device=record_device
                )

                for stripe_id, performance in zip(stripe_ids, performance_rows):
                    operations = stripe_operations(debug, stripe_id)
                    scope = (event_block_index, instruction_id, segment_id)
                    groups.append(
                        {
                            "event_block": event.event_block,
                            "event_block_index": event_block_index,
                            "record_index": record_index,
                            "segment_index": segment_index,
                            "segment_id": segment_id,
                            "method": method,
                            "instruction_id": instruction_id,
                            "delegate_debug_handles": sorted(handles),
                            "stripe_id": stripe_id,
                            "chain_ids": debug["stripe_to_chain"].get(
                                str(stripe_id), []
                            ),
                            "cascade_ids": debug["stripe_to_cascade"].get(
                                str(stripe_id), []
                            ),
                            "statistics": performance,
                            "operator_group": map_operator_group(
                                operations,
                                nodes,
                                scope,
                                mapping_state,
                            ),
                        }
                    )

    if not groups:
        raise NoVGFProfilingDataError("No VGF neural-statistics events were found")
    assert decoder_settings is not None

    return {
        "schema_version": 1,
        "mapping_kind": "nx_stripe_to_representative_etrecord_operator",
        "warning": (
            "NX cycles belong to the stripe and are not per-node cycles. Source nodes "
            "use recorded compiler provenance when unambiguous, otherwise legacy ordered "
            "representatives or delegate-only associations."
        ),
        "source_etdump": str(Path(capture.etdump_path).resolve()),
        "source_etrecord": str(Path(capture.etrecord_path).resolve()),
        "statistics_mode": decoder_settings[0],
        "device": decoder_settings[1],
        "decoder_fallback_fields": [
            name
            for name in ("statistics_mode", "profiling_device", "counter_layout")
            if name in decoder_fallback_fields
        ],
        "profiled_instruction_ids": sorted(
            {group["instruction_id"] for group in groups}
        ),
        "source_graph": capture.source_graph,
        "groups": groups,
    }


def accumulate_statistics(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Add one stripe's counters to a chain or cascade aggregate."""
    target["op_cycles"] = target.get("op_cycles", 0) + int(source["op_cycles"])
    target["total_cycles"] = target.get("total_cycles", 0) + int(source["total_cycles"])
    counters = target.setdefault("counters", {})
    for name, value in source.get("counters", {}).items():
        counters[name] = counters.get(name, 0) + int(value)


def _average(values: list[int]) -> int | float:
    """Return an integer for integral means and a float otherwise."""
    average = statistics.fmean(values)
    return int(average) if average.is_integer() else average


def summarize_execution_statistics(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize corresponding entity statistics across inference executions."""
    if not samples:
        raise ValueError("At least one execution sample is required")

    op_cycles = [int(sample["op_cycles"]) for sample in samples]
    total_cycles = [int(sample["total_cycles"]) for sample in samples]
    counter_names = {name for sample in samples for name in sample.get("counters", {})}
    return {
        "op_cycles": _average(op_cycles),
        "total_cycles": _average(total_cycles),
        "total_cycles_min": min(total_cycles),
        "total_cycles_max": max(total_cycles),
        "total_cycles_p50": statistics.median(total_cycles),
        "execution_count": len(samples),
        "counters": {
            name: _average(
                [int(sample.get("counters", {}).get(name, 0)) for sample in samples]
            )
            for name in sorted(counter_names)
        },
    }


def statistics_metrics(
    statistics: dict[str, Any], execution_keys: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    """Convert measured stripe or aggregate statistics to output metrics."""
    # Core projection combines additive metrics only when their qualifiers match.
    # Equal sample counts alone do not establish identical execution coverage.
    qualifiers = {
        "etdump_execution_cohort": [
            {"block_index": block_index, "record_index": record_index}
            for block_index, record_index in sorted(execution_keys)
        ]
    }
    return [
        {
            "name": "op_cycles",
            "value": statistics["op_cycles"],
            "unit": "cycles",
            "aggregation": "sum",
            "qualifiers": qualifiers,
        },
        {
            "name": "total_cycles",
            "value": statistics["total_cycles"],
            "unit": "cycles",
            "aggregation": "sum",
            "qualifiers": qualifiers,
        },
        *[
            {
                "name": f"{name}_cycles",
                "value": value,
                "unit": "cycles",
                "aggregation": "sum",
                "qualifiers": qualifiers,
            }
            for name, value in statistics.get("counters", {}).items()
        ],
        *[
            {
                "name": name,
                "value": statistics[name],
                "unit": unit,
            }
            for name, unit in (
                ("total_cycles_min", "cycles"),
                ("total_cycles_max", "cycles"),
                ("total_cycles_p50", "cycles"),
                ("execution_count", "count"),
            )
            if name in statistics
        ],
    ]


def _register_backend_operations(
    group: dict[str, Any],
    entities: dict[str, dict[str, Any]],
    chain_id: str,
    scope: str,
    source_ids: list[str],
) -> list[dict[str, Any]]:
    """Give compiler operations shared identities without inventing timing."""
    operations = []
    association = group["operator_group"]
    for operation in association["backend_operations"]:
        tosa_id = str(operation.get("tosa_id") or "")
        if not tosa_id:
            operations.append(dict(operation))
            continue
        entity_id = (
            f"backend_operator/{quote(group['method'], safe='')}/"
            f"{scope}/{quote(tosa_id, safe='')}"
        )
        entity = entities.setdefault(
            entity_id,
            {
                "id": entity_id,
                "kind": "backend_operator",
                "placement": schema.PlacementType.NX.value,
                "name": operation["operator"],
                "parent_ids": [],
                "child_ids": [],
                "attributes": {
                    "tosa_id": tosa_id,
                    "operator": operation["operator"],
                    "api_labels": [],
                    "mapping_confidence": association["confidence"],
                    "mapping_strategy": association["strategy"],
                },
            },
        )
        if chain_id not in entity["parent_ids"]:
            entity["parent_ids"].append(chain_id)
        if entity_id not in entities[chain_id]["child_ids"]:
            entities[chain_id]["child_ids"].append(entity_id)
        for label in operation.get("api_labels", []):
            if label not in entity["attributes"]["api_labels"]:
                entity["attributes"]["api_labels"].append(label)
        # The compiler/source association is representative and heuristic. Only
        # operations actually matched by the adapter inherit its source links.
        if operation.get("matched"):
            for source_id in source_ids:
                if source_id not in entity["child_ids"]:
                    entity["child_ids"].append(source_id)
                if entity_id not in entities[source_id]["parent_ids"]:
                    entities[source_id]["parent_ids"].append(entity_id)
        operations.append({**operation, "entity_id": entity_id})
    return operations


def load_etdump_metrics(
    etdump_path: Path,
    etrecord_path: Path,
    *,
    method: str | None = None,
    mode: int = 1,
    device: str = "Mali-G2",
) -> MeasuredPerformanceMetrics:
    """Adapt an ETRecord/ETDump pair to common measured observations."""
    try:
        capture = read_profiling_capture(
            etdump_path,
            etrecord_path,
            method,
            method_option="--neural-technology-profiling-data.etrecord-method",
        )
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise ConfigurationError(
            f"Unable to read ETRecord/ETDump artifacts: {exc}"
        ) from exc
    return analyze_capture(capture, mode=mode, device=device)


def analyze_capture(
    capture: ProfilingCapture,
    *,
    mode: int = 1,
    device: str = "Mali-G2",
) -> MeasuredPerformanceMetrics:
    """Analyze detached artifact data without importing Torch or ExecuTorch."""
    method = capture.method
    etdump_path = Path(capture.etdump_path)
    etrecord_path = Path(capture.etrecord_path)
    try:
        mapping = build_mappings(capture, mode, device)
    except NoVGFProfilingDataError:
        try:
            standard_mappings = build_standard_runtime_mappings(capture)
            metrics = standard_runtime_metrics(etdump_path, standard_mappings, method)
            assert metrics.backend_configuration is not None
            metrics.backend_configuration["etrecord"] = str(etrecord_path)
            metrics.extensions["executorch"] = {
                "schema_version": 1,
                "source_graph": capture.source_graph,
            }
            metrics.measured_observations = [
                normalize_nx_observations(item)
                for item in metrics.measured_observations
            ]
            return metrics
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
            raise ConfigurationError(
                f"Unable to analyze standard ETDump profiling data: {exc}"
            ) from exc
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise ConfigurationError(
            f"Unable to analyze ETDump profiling data: {exc}"
        ) from exc

    profiled_instruction_ids = set(
        mapping.get(
            "profiled_instruction_ids",
            [group["instruction_id"] for group in mapping["groups"]],
        )
    )
    try:
        standard_mappings = build_standard_runtime_mappings(
            capture,
            excluded_instruction_ids=profiled_instruction_ids,
        )
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise ConfigurationError(
            f"Unable to analyze non-VGF ETDump profiling data: {exc}"
        ) from exc

    entities: dict[str, dict[str, Any]] = {}
    breakdowns = []
    entity_execution_statistics: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    model_execution_statistics: dict[tuple[int, int], dict[str, Any]] = {}
    partition_execution_keys: dict[tuple[str, int, int], set[tuple[int, int]]] = {}
    for group in mapping["groups"]:
        execution_key = (
            int(group.get("event_block_index", 0)),
            int(group.get("record_index", 0)),
        )
        partition_execution_keys.setdefault(
            (group["method"], group["instruction_id"], group["segment_id"]), set()
        ).add(execution_key)
        scope = (
            f"etrecord/{group['event_block']}/{group['instruction_id']}/"
            f"{group['segment_id']}"
        )
        segment_entity_id = f"segment/{scope}"
        chain_local_id = next(
            iter(group.get("chain_ids", [])), f"chain_{group['stripe_id']}"
        )
        cascade_local_id = next(
            iter(group.get("cascade_ids", [])), f"cascade_{group['stripe_id']}"
        )
        cascade_entity_id = f"cascade/{scope}/{cascade_local_id}"
        chain_entity_id = f"chain/{scope}/{chain_local_id}"
        stripe_entity_id = f"stripe/{scope}/{group['stripe_id']}"
        node_ids = [
            source_operator_id(group["method"], node["name"])
            for node in group["operator_group"]["nodes"]
        ]
        segment_entity = entities.setdefault(
            segment_entity_id,
            {
                "id": segment_entity_id,
                "kind": "segment",
                "name": f"{group['event_block']} segment {group['segment_id']}",
                "attributes": {
                    "method": group["method"],
                    "instruction_id": group["instruction_id"],
                    "segment_index": group["segment_index"],
                    "segment_id": group["segment_id"],
                },
                "child_ids": [],
            },
        )
        if cascade_entity_id not in segment_entity["child_ids"]:
            segment_entity["child_ids"].append(cascade_entity_id)
        cascade_entity = entities.setdefault(
            cascade_entity_id,
            {
                "id": cascade_entity_id,
                "kind": "cascade",
                "placement": schema.PlacementType.NX.value,
                "name": f"Cascade {cascade_local_id}",
                "parent_ids": [segment_entity_id],
                "child_ids": [],
                "attributes": {"stripe_ids": []},
            },
        )
        if chain_entity_id not in cascade_entity["child_ids"]:
            cascade_entity["child_ids"].append(chain_entity_id)
        if group["stripe_id"] not in cascade_entity["attributes"]["stripe_ids"]:
            cascade_entity["attributes"]["stripe_ids"].append(group["stripe_id"])
        chain_entity = entities.setdefault(
            chain_entity_id,
            {
                "id": chain_entity_id,
                "kind": "chain",
                "placement": schema.PlacementType.NX.value,
                "name": f"Chain {chain_local_id}",
                "parent_ids": [cascade_entity_id],
                "child_ids": [],
                "attributes": {"stripe_ids": []},
            },
        )
        if stripe_entity_id not in chain_entity["child_ids"]:
            chain_entity["child_ids"].append(stripe_entity_id)
        for node_id in node_ids:
            if node_id not in chain_entity["child_ids"]:
                chain_entity["child_ids"].append(node_id)
        if group["stripe_id"] not in chain_entity["attributes"]["stripe_ids"]:
            chain_entity["attributes"]["stripe_ids"].append(group["stripe_id"])
        entities.setdefault(
            stripe_entity_id,
            {
                "id": stripe_entity_id,
                "kind": "stripe",
                "placement": schema.PlacementType.NX.value,
                "name": f"NX stripe {group['stripe_id']}",
                "parent_ids": [chain_entity_id],
                "attributes": {
                    "stripe_id": group["stripe_id"],
                    "chain_id": chain_local_id,
                    "cascade_id": cascade_local_id,
                    "mapping_confidence": group["operator_group"]["confidence"],
                    "mapping_strategy": group["operator_group"]["strategy"],
                    "backend_operations": group["operator_group"]["backend_operations"],
                },
            },
        )
        for node in group["operator_group"]["nodes"]:
            entity_id = source_operator_id(group["method"], node["name"])
            entity = entities.setdefault(
                entity_id,
                {
                    "id": entity_id,
                    "kind": "source_operator",
                    "name": node["name"],
                    "parent_ids": [],
                    "attributes": {
                        "debug_handle": node["debug_handle"],
                        "target": node["target"],
                        "category": node["category"],
                    },
                },
            )
            if chain_entity_id not in entity["parent_ids"]:
                entity["parent_ids"].append(chain_entity_id)
        entities[stripe_entity_id]["attributes"]["backend_operations"] = (
            _register_backend_operations(
                group, entities, chain_entity_id, scope, node_ids
            )
        )
        for entity_id in (
            stripe_entity_id,
            chain_entity_id,
            cascade_entity_id,
        ):
            accumulate_statistics(
                entity_execution_statistics.setdefault(entity_id, {}).setdefault(
                    execution_key, {}
                ),
                group["statistics"],
            )
        op_cycles = int(group["statistics"]["op_cycles"])
        accumulate_statistics(
            model_execution_statistics.setdefault(execution_key, {}),
            {
                "op_cycles": op_cycles,
                "total_cycles": int(group["statistics"]["total_cycles"]),
                "counters": {},
            },
        )

    for entity_id, execution_statistics in entity_execution_statistics.items():
        breakdowns.append(
            {
                "id": f"etdump-{entity_id}",
                "entity_id": entity_id,
                "metrics": statistics_metrics(
                    summarize_execution_statistics(list(execution_statistics.values())),
                    list(execution_statistics),
                ),
            }
        )

    # Require matching cohorts across observed instructions and segments to
    # avoid averaging partial executions. This does not establish stripe
    # completeness; see docs/source/etrecord.md.
    execution_keys = set(model_execution_statistics)
    complete_coverage = all(
        keys == execution_keys for keys in partition_execution_keys.values()
    )
    model_statistics = (
        summarize_execution_statistics(list(model_execution_statistics.values()))
        if complete_coverage
        else None
    )
    incomplete_coverage_reason = (
        "VGF instructions or segments have different execution coverage; "
        "complete model inference totals cannot be established. "
        "Per-entity measurements remain available."
    )
    model_metrics = [
        {
            "name": name,
            "unit": unit,
            **(
                {"value": model_statistics[name]}
                if model_statistics is not None
                else {
                    "availability": "unavailable",
                    "reason": incomplete_coverage_reason,
                }
            ),
        }
        for name, unit in (
            ("total_cycles", "cycles"),
            ("total_cycles_min", "cycles"),
            ("total_cycles_max", "cycles"),
            ("total_cycles_p50", "cycles"),
            ("execution_count", "count"),
        )
    ]
    resolved_mode = mapping.get("statistics_mode", mode)
    resolved_device = mapping.get("device", device)
    fallback_fields = mapping.get("decoder_fallback_fields", [])
    warnings = [mapping["warning"]]
    if fallback_fields:
        warnings.append(
            "VGF decoder fallback: "
            f"{', '.join(fallback_fields)} missing from one or more profiling records. "
            f"Decoding assumes statistics mode {resolved_mode} on {resolved_device} "
            "for those fields; confirm that these settings match the capture."
        )
    if model_statistics is None:
        warnings.append(incomplete_coverage_reason)
    elif model_statistics["execution_count"] > 1:
        warnings.append(
            "Repeated VGF neural-statistics executions are summarized per "
            "inference using average cycles; minimum, maximum, P50, and "
            "execution count are also reported."
        )

    observations = PerformanceObservations(
        metrics=[schema.Metric.from_dict(item) for item in model_metrics],
        entities=[schema.Entity.from_dict(item) for item in entities.values()],
        breakdowns=[schema.Breakdown.from_dict(item) for item in breakdowns],
        entity_kinds=NX_ENTITY_KINDS,
        warnings=warnings,
    )
    metrics = MeasuredPerformanceMetrics(
        measured_observations=[observations],
        model_format="etrecord",
        backend_configuration={
            "etdump": str(etdump_path),
            "etrecord": str(etrecord_path),
            "statistics_mode": resolved_mode,
            "device": resolved_device,
            "decoder_fallback_fields": fallback_fields,
            "profiling_kind": "vgf-neural-statistics",
            "profiled_instruction_ids": sorted(profiled_instruction_ids),
            **(
                {"execution_count": model_statistics["execution_count"]}
                if model_statistics is not None
                else {}
            ),
        },
        runtime_configuration={
            "profiling_data": [str(etdump_path)],
            "etrecord_method": method,
        },
        extensions={
            "executorch": {
                "schema_version": 1,
                "source_graph": mapping.get("source_graph", {}),
            },
            "etrecord_mapping": mapping,
        },
    )
    if standard_mappings:
        metrics.backend_configuration.update(
            {
                "profiling_kind": "mixed-vgf-and-standard-etdump",
                "standard_runtime_excluded_instruction_ids": sorted(
                    profiled_instruction_ids
                ),
            }
        )
        metrics.measured_observations.append(
            standard_runtime_observations(
                standard_mappings,
                method,
                "Standard ETDump runtime events outside VGF-profiled "
                "instructions were mapped by backend, instruction, and debug handle.",
            )
        )
    metrics.measured_observations = [
        normalize_nx_observations(item) for item in metrics.measured_observations
    ]
    return metrics


def analyze_etrecord_profiling_data(
    etdump_path: Path,
    etrecord_path: Path,
    *,
    target_profile: str,
    cli_arguments: list[str] | None = None,
    method: str | None = None,
    mode: int = 1,
    device: str = "Mali-G2",
    companion_model: Path | None = None,
) -> dict[str, Any]:
    """Compatibility entry point using the common measured-profile pipeline."""
    from mlia.backend.neural_technology_profiling_data.profiling import (
        analyze_profiling_data,
    )

    return analyze_profiling_data(
        target_profile=target_profile,
        profiling_data=[etdump_path],
        categories={"performance"},
        model=str(etrecord_path),
        cli_arguments=cli_arguments,
        etrecord_method=method,
        companion_model=companion_model,
        statistics_mode=mode,
        profiling_device=device,
    )


def main() -> None:
    """Run the standalone ETRecord and ETDump mapping utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Map VGF NX statistics stripes in ETDump to high-level fused "
            "operator groups in ETRecord."
        )
    )
    parser.add_argument("etdump", type=Path, help="Path to ETDump")
    parser.add_argument("etrecord", type=Path, help="Path to ETRecord")
    parser.add_argument("--method", default=None, help="ETRecord method name")
    parser.add_argument("--statistics-mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", choices=("Mali-G1", "Mali-G2"), default="Mali-G2")
    parser.add_argument("--output", "-o", type=Path, help="Write JSON to this file")
    args = parser.parse_args()

    for path, name in ((args.etdump, "ETDump"), (args.etrecord, "ETRecord")):
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")

    try:
        capture = read_profiling_capture(
            args.etdump, args.etrecord, args.method, method_option="--method"
        )
        result = build_mappings(
            capture,
            args.statistics_mode,
            args.device,
        )
    except (KeyError, OSError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    text = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        ensure_user_output_directory(args.output.parent, parents=True)
        write_user_output_bytes(args.output, text.encode("utf-8"))
        print(f"Wrote {len(result['groups'])} stripe mapping(s) to {args.output}")


if __name__ == "__main__":
    main()
