# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""NX performance schema shared by estimated and measured analyses."""

from dataclasses import replace

from mlia.core import output_schema as schema
from mlia.core.performance import PerformanceObservations

NN_MODULE_ENTITY_KIND = "nn_module"
NX_PROVENANCE_ENTITY_KIND_DECLARATIONS = (
    schema.EntityKind(
        id=NN_MODULE_ENTITY_KIND,
        parent_kinds=[NN_MODULE_ENTITY_KIND],
        child_kinds=[NN_MODULE_ENTITY_KIND, schema.ENTITY_KIND_SOURCE_OPERATOR],
    ),
)
NX_ENTITY_KINDS = [
    schema.EntityKind(id="segment", child_kinds=["cascade", "source_operator"]),
    schema.EntityKind(id="cascade", parent_kinds=["segment"], child_kinds=["chain"]),
    schema.EntityKind(
        id="chain",
        parent_kinds=["cascade"],
        child_kinds=[
            "stripe",
            "source_operator",
            "performance_group",
            "backend_operator",
        ],
    ),
    schema.EntityKind(id="stripe", parent_kinds=["chain"]),
    schema.EntityKind(
        id="backend_operator", parent_kinds=["chain"], child_kinds=["source_operator"]
    ),
    schema.EntityKind(
        id="performance_group", parent_kinds=["chain"], child_kinds=["source_operator"]
    ),
    *NX_PROVENANCE_ENTITY_KIND_DECLARATIONS,
]
NX_MODEL_METRIC_UNITS = {
    "inference_time": "ms",
    "infs_per_sec": "inferences/s",
    "total_cycles": "cycles",
    "compute_cycles": "cycles",
    "cache_cycles": "cycles",
    "dram_cycles": "cycles",
    "compiled_size": "bytes",
    "cache_read_bytes": "bytes",
    "cache_write_bytes": "bytes",
    "dram_read_bytes": "bytes",
    "dram_write_bytes": "bytes",
    "dram_footprint": "bytes",
}
_UNAVAILABLE_REASON = "Backend output did not provide a numeric value for this metric."


def _nx_model_metric(name: str, value: int | float | None, unit: str) -> schema.Metric:
    if value is None:
        return schema.Metric(
            name=name,
            value=None,
            unit=unit,
            availability=schema.MetricAvailability.UNAVAILABLE,
            reason=_UNAVAILABLE_REASON,
        )
    return schema.Metric(name=name, value=value, unit=unit)


def normalize_nx_model_metrics(metrics: list[schema.Metric]) -> list[schema.Metric]:
    """Apply NX units, availability and derived metrics without estimator state."""
    by_name = {
        name: _nx_model_metric(name, None, unit)
        for name, unit in NX_MODEL_METRIC_UNITS.items()
    }
    by_name.update((metric.name, metric) for metric in metrics)
    inference_time = by_name["inference_time"].value
    if inference_time:
        by_name.setdefault(
            schema.METRIC_NAME_INFERENCES_PER_SECOND,
            schema.Metric(
                name=schema.METRIC_NAME_INFERENCES_PER_SECOND,
                value=1000 / inference_time,
                unit=schema.UNIT_INFERENCES_PER_SECOND,
            ),
        )
    compute_cycles = by_name["compute_cycles"].value
    total_cycles = by_name["total_cycles"].value
    utilization = (
        (compute_cycles / total_cycles * 100 if total_cycles else 0.0)
        if compute_cycles is not None and total_cycles is not None
        else None
    )
    by_name.setdefault(
        schema.METRIC_NAME_TARGET_UTILIZATION,
        _nx_model_metric(
            schema.METRIC_NAME_TARGET_UTILIZATION, utilization, schema.UNIT_PERCENT
        ),
    )
    return schema.ensure_standard_performance_metrics(list(by_name.values()))


def normalize_nx_observations(item: PerformanceObservations) -> PerformanceObservations:
    """Normalize model metrics while preserving measured scopes and relationships."""
    return replace(item, metrics=normalize_nx_model_metrics(item.metrics))
