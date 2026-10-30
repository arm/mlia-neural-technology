# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Attribute and aggregate ordinary ExecuTorch timings for Neural Technology."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mlia.core import output_schema as schema
from mlia.core.performance import MeasuredPerformanceMetrics, PerformanceObservations

from mlia.backend.executorch.capture import ETDumpEvent, ProfilingCapture
from mlia.backend.executorch.etrecord import source_operator_id
from mlia.backend.neural_technology_profiling_data.executorch_source import (
    node_category,
)

_RUNTIME_METRICS = (
    ("Runtime Total Avg", "avg"),
    ("Runtime Total P50", "p50"),
    ("Runtime Total Min", "min"),
    ("Runtime Total Max", "max"),
)
_RESERVED_FRAMEWORK_EVENT_NAMES = {
    "Method::init",
    "Method::execute",
    "Program::load_method",
}
_RUNTIME_WRAPPER_EVENT_NAMES = {
    "OPERATOR_CALL",
}


@dataclass(frozen=True)
class _RuntimeEvent:
    """One eligible recorded event before source associations duplicate it."""

    index: int
    name: str
    instruction_id: int | None
    backend: str | None
    handles: tuple[int, ...]
    samples: list[float]
    delegated: bool

    @property
    def call_scope(self) -> tuple[int | None, str | None, int]:
        """Identify one instruction's aligned executions within its block."""
        return self.instruction_id, self.backend, len(self.samples)


def _ordinary_runtime_events(
    block: list[ETDumpEvent], excluded_instruction_ids: set[int]
) -> list[_RuntimeEvent]:
    """Select timings under NT's runtime event and unit policy."""
    events = []
    for event in block:
        instruction_id = event.instruction_id
        if (
            event.samples is None
            or not event.debug_handles
            or instruction_id in excluded_instruction_ids
            or event.name in _RESERVED_FRAMEWORK_EVENT_NAMES
            or event.name in _RUNTIME_WRAPPER_EVENT_NAMES
        ):
            continue
        handles = tuple(sorted(set(event.debug_handles)))
        if event.time_unit != "ms":
            raise ValueError(f"Unsupported ETDump runtime time unit: {event.time_unit}")
        samples = list(event.samples)
        if not samples:
            raise ValueError("Cannot align empty ETDump runtime samples")
        events.append(
            _RuntimeEvent(
                event.event_index,
                event.name,
                instruction_id,
                event.backend,
                handles,
                samples,
                event.delegate_debug_identifier is not None,
            )
        )
    return events


def build_standard_runtime_mappings(
    capture: ProfilingCapture,
    excluded_instruction_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Apply NT timing attribution policy to recorded identities and samples."""
    method = capture.method
    excluded_instruction_ids = excluded_instruction_ids or set()
    samples_by_partition: dict[tuple[str | None, int], dict[str, Any]] = {}
    nodes_by_handle = capture.nodes_by_handle
    blocks: dict[int, list[ETDumpEvent]] = {}
    for event in capture.events:
        if event.event_block == "Execute":
            blocks.setdefault(event.event_block_index, []).append(event)
    for block_index, block in blocks.items():
        events = _ordinary_runtime_events(block, excluded_instruction_ids)
        detailed_handles: dict[tuple[int | None, str | None, int], set[int]] = {}
        for event in events:
            if (
                event.delegated
                and event.instruction_id is not None
                and event.name != "DELEGATE_CALL"
            ):
                detailed_handles.setdefault(event.call_scope, set()).update(
                    handle for handle in event.handles if nodes_by_handle.get(handle)
                )
        for event in events:
            # Detailed timings and their enclosing delegate interval overlap.
            # Keep wrapper fallback for handles without mapped detail in this
            # instruction/backend and aligned execution cohort within the block.
            handles = event.handles
            if event.name == "DELEGATE_CALL":
                covered = detailed_handles.get(event.call_scope, set())
                handles = tuple(handle for handle in handles if handle not in covered)
                if not handles:
                    continue
            # Keep the original associations: the wrapper interval still includes
            # covered nodes, so partial fallback remains a shared measurement.
            source_ids = {
                source_operator_id(method, node["name"])
                for handle in event.handles
                for node in nodes_by_handle.get(handle, [])
            }
            unmapped_handles = {
                handle for handle in event.handles if not nodes_by_handle.get(handle)
            }
            observation_id = (
                f"etdump-runtime/{quote(method, safe='-._~')}/"
                f"block/{block_index}/event/{event.index}"
            )
            for handle in handles:
                partition = samples_by_partition.setdefault(
                    (event.backend, handle),
                    {
                        "samples_by_block": {},
                        "event_count": 0,
                        "instruction_ids": set(),
                        "shared_source_operator_ids": set(),
                        "unresolved_debug_handles": set(),
                        "observation_ids": set(),
                        "delegate_wrapper_fallback": False,
                    },
                )
                # Inspector aligns events within an EventBlock by run signature.
                # Different blocks may represent distinct inputs or event sets;
                # their samples must be pooled rather than added by position.
                totals = partition["samples_by_block"].get(block_index)
                if totals is None:
                    partition["samples_by_block"][block_index] = event.samples.copy()
                else:
                    if len(totals) != len(event.samples):
                        raise ValueError(
                            "Cannot align ETDump runtime samples for debug handle "
                            f"{handle} in Execute block {block_index}: "
                            f"{len(totals)} and {len(event.samples)} executions"
                        )
                    partition["samples_by_block"][block_index] = [
                        total + sample
                        for total, sample in zip(totals, event.samples, strict=True)
                    ]
                partition["event_count"] += 1
                partition["observation_ids"].add(observation_id)
                # Unresolved work also makes the full interval shared, even
                # when only one source operator could be identified.
                partition["unresolved_debug_handles"].update(unmapped_handles)
                if len(source_ids) > 1 or unmapped_handles:
                    partition["shared_source_operator_ids"].update(source_ids)
                if event.name == "DELEGATE_CALL":
                    partition["delegate_wrapper_fallback"] = True
                if event.instruction_id is not None:
                    partition["instruction_ids"].add(event.instruction_id)

    mappings = []
    for (backend, debug_handle), partition in sorted(
        samples_by_partition.items(),
        key=lambda item: (
            item[0][0] or "",
            item[0][1],
        ),
    ):
        totals = [
            total
            for samples in partition["samples_by_block"].values()
            for total in samples
        ]
        summary = {
            "avg": statistics.mean(totals),
            "p50": statistics.median(totals),
            "min": min(totals),
            "max": max(totals),
        }
        nodes = nodes_by_handle.get(debug_handle, [])
        additive = not (
            partition["shared_source_operator_ids"]
            or partition["unresolved_debug_handles"]
        )
        execution_cohort = [
            {"block_index": block_index, "sample_count": len(samples)}
            for block_index, samples in sorted(partition["samples_by_block"].items())
        ]
        metrics = [
            {
                "name": name,
                "value": summary[attribute],
                "unit": "ms",
                # Shared measurements, percentiles and extrema are non-additive.
                **({"aggregation": "sum"} if additive and attribute == "avg" else {}),
                # Core projection only combines metrics with equal qualifiers.
                # Equal sample totals alone do not establish aligned executions.
                **(
                    {"qualifiers": {"etdump_execution_cohort": execution_cohort}}
                    if attribute == "avg"
                    else {}
                ),
            }
            for name, attribute in _RUNTIME_METRICS
        ]
        metrics.append(
            {
                "name": "Runtime Event Count",
                "value": partition["event_count"],
                "unit": "count",
                **({"aggregation": "sum"} if additive else {}),
            }
        )
        for node in nodes:
            mappings.append(
                {
                    "node": {
                        **node,
                        "category": node_category(node["name"], node["target"]),
                    },
                    "method": method,
                    "metrics": metrics,
                    "backend": backend,
                    "instruction_ids": sorted(partition["instruction_ids"]),
                    **(
                        {
                            "shared_source_operator_ids": sorted(
                                partition["shared_source_operator_ids"]
                            ),
                            "observation_ids": sorted(partition["observation_ids"]),
                            "unresolved_debug_handles": sorted(
                                partition["unresolved_debug_handles"]
                            ),
                        }
                        if not additive
                        else {}
                    ),
                    **(
                        {"delegate_wrapper_fallback": True}
                        if partition["delegate_wrapper_fallback"]
                        else {}
                    ),
                }
            )
    return mappings


def standard_runtime_observations(
    mappings: list[dict[str, Any]],
    method: str,
    warning: str,
) -> PerformanceObservations:
    """Normalize ordinary ETDump timing observations without cycle conversion."""
    entities: dict[str, dict[str, Any]] = {}
    attributions: dict[str, dict[str | None, set[int]]] = {}
    breakdowns = []
    for index, mapping in enumerate(mappings):
        node = mapping["node"]
        entity_id = source_operator_id(method, node["name"])
        entity = entities.setdefault(
            entity_id,
            {
                "id": entity_id,
                "kind": "source_operator",
                "name": node["name"],
                "attributes": {
                    "debug_handle": node["debug_handle"],
                    "target": node["target"],
                    "category": node["category"],
                },
            },
        )
        attributes = entity["attributes"]
        backend = mapping.get("backend")
        attributions.setdefault(entity_id, {}).setdefault(backend, set()).update(
            mapping.get("instruction_ids", [])
        )
        for key in (
            "shared_source_operator_ids",
            "observation_ids",
            "unresolved_debug_handles",
        ):
            attribute_key = f"runtime_{key}"
            values = set(attributes.get(attribute_key, []))
            values.update(mapping.get(key, []))
            if values:
                attributes[attribute_key] = sorted(values)
        if mapping.get("delegate_wrapper_fallback"):
            attributes["runtime_delegate_wrapper_fallback"] = True
        breakdowns.append(
            {
                "id": f"etdump-runtime/{method}/{node['name']}/{index}",
                "entity_id": entity_id,
                "metrics": mapping["metrics"],
                "qualifiers": {
                    "etrecord_method": method,
                    **({"runtime_backend": backend} if backend is not None else {}),
                },
            }
        )

    for entity_id, backends in attributions.items():
        attributes = entities[entity_id]["attributes"]
        attributes["runtime_attributions"] = [
            {"backend": backend, "instruction_ids": sorted(instruction_ids)}
            for backend, instruction_ids in sorted(
                backends.items(), key=lambda item: (item[0] is not None, item[0] or "")
            )
        ]
        instruction_ids = sorted(
            {instruction_id for ids in backends.values() for instruction_id in ids}
        )
        if instruction_ids:
            attributes["runtime_instruction_ids"] = instruction_ids
        # Preserve the original summary only when one backend is unambiguous.
        if len(backends) == 1 and (backend := next(iter(backends))) is not None:
            attributes["runtime_backend"] = backend

    return PerformanceObservations(
        warnings=[warning],
        entities=[schema.Entity.from_dict(item) for item in entities.values()],
        breakdowns=[schema.Breakdown.from_dict(item) for item in breakdowns],
    )


def standard_runtime_metrics(
    etdump_path: Path,
    mappings: list[dict[str, Any]],
    method: str,
) -> MeasuredPerformanceMetrics:
    """Normalize a capture containing ordinary runtime events only."""
    if not mappings:
        raise ValueError(
            "No ordinary ETDump runtime events matched ETRecord debug handles"
        )
    return MeasuredPerformanceMetrics(
        [
            standard_runtime_observations(
                mappings,
                method,
                "VGF neural statistics were not present. Ordinary ExecuTorch runtime "
                "events were mapped by instruction and debug handle.",
            )
        ],
        backend_configuration={
            "etdump": str(etdump_path),
            "profiling_kind": "standard-etdump",
        },
        runtime_configuration={
            "profiling_data": [str(etdump_path)],
            "etrecord_method": method,
        },
        model_format="etrecord",
    )
