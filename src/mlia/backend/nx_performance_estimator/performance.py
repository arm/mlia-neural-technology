# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for NX Performance Estimator performance estimation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlia
import mlia.core.output_schema as schema
from mlia.backend.ml_sdk_model_converter.conversion import MLSDKModelConverter
from mlia.backend.ml_sdk_model_converter.install import get_ml_sdk_model_converter_path
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.output_parsing import (
    DebugDatabaseContentsType,
    NXDebugDatabaseParser,
    NXPerformanceDatabaseParser,
    PerformanceDatabaseContentsType,
)
from mlia.backend.nx_performance_estimator.provenance import (
    NX_PROVENANCE_ENTITY_KIND_DECLARATIONS,
    NXProvenanceEntityBuilder,
)
from mlia.backend.nx_performance_estimator.runner import (
    NXPerformanceEstimatorOutputFiles,
    get_nx_resource_dir,
    run_nx_performance_estimator,
)
from mlia.backend.nx_performance_estimator.statistics import (
    NXModelCountMetricValue,
    NXModelFloatMetricValue,
    NXModelPerformanceStats,
    NXOperatorPerformanceStats,
    NXPerformanceStats,
)
from mlia.backend.nx_performance_estimator.vgf import (
    COMPUTE_SEGMENTS_SKIPPED_WARNING,
    GCPEVGFSegment,
    prepare_gcpe_compatible_vgfs,
)
from mlia.backend.repo import get_backend_repository
from mlia.core.performance import PerformanceEstimator
from mlia.nx_utils.filesystem import is_vgf_file
from mlia.utils.logging import log_action
from mlia.utils.misc import summarize_list

logger = logging.getLogger(__name__)

NX_ENTITY_KINDS = [
    schema.EntityKind(id="segment", child_kinds=["cascade", "source_operator"]),
    schema.EntityKind(id="cascade", parent_kinds=["segment"], child_kinds=["chain"]),
    schema.EntityKind(
        id="chain",
        parent_kinds=["cascade"],
        child_kinds=["source_operator", "performance_group"],
    ),
    schema.EntityKind(
        id="performance_group",
        parent_kinds=["chain"],
        child_kinds=["source_operator"],
    ),
    *NX_PROVENANCE_ENTITY_KIND_DECLARATIONS,
]


_BACKEND_METRIC_UNAVAILABLE_REASON = (
    "Backend output did not provide a numeric value for this metric."
)


def _metric_name_part(value: object) -> str:
    """Convert backend metric name fragments to lower snake_case."""
    text = str(value).strip()
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def _append_unique_items(target: list[str], items: list[str]) -> None:
    """Append string items while preserving first-seen order."""
    for item in items:
        if item and item not in target:
            target.append(item)


def _performance_group_entity_id(chain_id: str, operator_index: int) -> str:
    """Return an ID for one aggregate or unassociated performance record."""
    kind, segment_id, local_id = chain_id.split("/", maxsplit=2)
    if kind != "chain":
        raise ValueError(f"Expected a chain entity ID, got: {chain_id}")
    return f"performance_group/{segment_id}/chain_{local_id}/{operator_index}"


def _subgraph_entity_id(segment_id: str, debug_db_id: str) -> str:
    """Return a kind-namespaced entity ID for a segment-local debug DB ID."""
    kind, separator, local_id = debug_db_id.partition("_")
    if not separator or not local_id:
        raise ValueError(f"Invalid segment-local debug DB ID: {debug_db_id}")
    return f"{kind}/{segment_id}/{local_id}"


def _subgraph_display_name(segment_name: str, debug_db_id: str, kind: str) -> str:
    """Return a readable segment-local subgraph display name."""
    match = re.fullmatch(rf"{re.escape(kind.lower())}_(.+)", debug_db_id)
    suffix = match.group(1) if match else debug_db_id
    label = f"{kind} {suffix}"
    return f"{segment_name}/{label}" if segment_name else label


@dataclass
class NXSegmentPerformanceEntities:
    """Performance entity ids that belong to one VGF segment."""

    segment_index: int
    segment_name: str
    chain_ids: list[str]
    cascade_ids: list[str]
    structural_source_operator_ids: list[str] = field(default_factory=list)
    structural_source_operator_names: dict[str, str] = field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        """Return the standardized entity id for the segment."""
        return f"segment/{self.segment_index}"


@dataclass(frozen=True)
class NXSegmentPerformanceDatabase:
    """Parsed databases and provenance for one original model segment."""

    segment_index: int
    segment_name: str
    debug_database: DebugDatabaseContentsType
    performance_database: PerformanceDatabaseContentsType
    model_performance_stats: NXModelPerformanceStats
    debug_names: SpirvDebugNameMap | None = None
    structural_source_operator_ids: list[str] = field(default_factory=list)
    structural_source_operator_names: dict[str, str] = field(default_factory=dict)


@dataclass
class NXPerformanceEstimatorPerformanceMetrics:
    """NX Performance Estimator configuration and performance metrics."""

    backend_config: NXPerformanceEstimatorConfig
    performance_db_parser: NXPerformanceDatabaseParser | None
    chain_performance_metrics: dict[str, NXOperatorPerformanceStats]
    cascade_performance_metrics: dict[str, NXOperatorPerformanceStats]
    model_performance_stats: NXModelPerformanceStats
    segment_entities: list[NXSegmentPerformanceEntities] | None = None
    warnings: list[str] | None = None

    def _build_model_metric(
        self,
        name: str,
        value: NXModelCountMetricValue | NXModelFloatMetricValue,
        unit: str,
    ) -> schema.Metric:
        if value is None:
            return schema.Metric(
                name=name,
                value=None,
                unit=unit,
                availability=schema.MetricAvailability.UNAVAILABLE,
                reason=_BACKEND_METRIC_UNAVAILABLE_REASON,
            )

        return schema.Metric(name=name, value=value, unit=unit)

    def _build_target_utilization_metric(self) -> schema.Metric:
        compute_cycles = self.model_performance_stats.compute_cycles
        total_cycles = self.model_performance_stats.total_cycles
        if compute_cycles is None or total_cycles is None:
            return schema.Metric(
                name=schema.METRIC_NAME_TARGET_UTILIZATION,
                value=None,
                unit=schema.UNIT_PERCENT,
                availability=schema.MetricAvailability.UNAVAILABLE,
                reason=_BACKEND_METRIC_UNAVAILABLE_REASON,
            )

        return schema.Metric(
            name=schema.METRIC_NAME_TARGET_UTILIZATION,
            value=(compute_cycles / total_cycles * 100 if total_cycles else 0.0),
            unit=schema.UNIT_PERCENT,
        )

    def _build_breakdown_metrics(
        self, stats: NXOperatorPerformanceStats
    ) -> list[schema.Metric]:
        breakdown_metrics = [
            schema.Metric(
                name="total_cycles",
                value=stats.total_cycles,
                unit="cycles",
                aggregation=schema.AggregationType.SUM,
            ),
            schema.Metric(
                name="op_cycles",
                value=stats.op_cycles,
                unit="cycles",
                aggregation=schema.AggregationType.SUM,
            ),
        ]

        for mem_type, mem_data in stats.memory.items():
            metric_prefix = _metric_name_part(mem_type)
            breakdown_metrics.extend(
                [
                    schema.Metric(
                        name=f"{metric_prefix}_read_bytes",
                        value=mem_data["readBytes"],
                        unit="bytes",
                        aggregation=schema.AggregationType.SUM,
                    ),
                    schema.Metric(
                        name=f"{metric_prefix}_write_bytes",
                        value=mem_data["writeBytes"],
                        unit="bytes",
                        aggregation=schema.AggregationType.SUM,
                    ),
                    schema.Metric(
                        name=f"{metric_prefix}_traffic_cycles",
                        value=mem_data["trafficCycles"],
                        unit="cycles",
                        aggregation=schema.AggregationType.SUM,
                    ),
                ]
            )

        for util in stats.utilization:
            metric_prefix = _metric_name_part(util["sectionName"])
            breakdown_metrics.append(
                schema.Metric(
                    name=f"{metric_prefix}_cycles",
                    value=int(util["cycles"]),
                    unit="cycles",
                    aggregation=schema.AggregationType.SUM,
                )
            )

        return breakdown_metrics

    def to_standardized_output(
        self,
        model_path: Path,
        backend_name: str = "nx-performance-estimator",
        target_config: dict[str, Any] | None = None,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            backend_name: Name of the backend (default: 'nx-performance-estimator')
            target_config: Target configuration parameters
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run

        Returns:
            Dictionary representing standardized output
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend configuration
        backend_config_dict = {
            "system_config": Path(self.backend_config.system_config).stem
            if self.backend_config.system_config != NXPerformanceEstimatorConfig.DEFAULT
            else "default",
            "compiler_config": Path(self.backend_config.compiler_config).stem
            if self.backend_config.compiler_config
            != NXPerformanceEstimatorConfig.DEFAULT
            else "default",
        }

        # Create backend
        backend = schema.Backend(
            id=backend_name,
            name="Neural Accelerator (NX) Performance Estimator",
            version="unknown",  # NX version not readily available
            configuration=backend_config_dict,
        )

        # Extract target info from config
        target_config = target_config or {}
        target_type = target_config.get(
            "target_type", target_config.get("target", "neural-accelerator")
        )
        profile_name = target_config.get("profile_name", target_type)

        # Extract variant from system_config name if not default
        variant = None
        if self.backend_config.system_config != NXPerformanceEstimatorConfig.DEFAULT:
            variant = Path(self.backend_config.system_config).stem

        # Create target components - Neural Accelerator (NX)
        components = [
            schema.Component(
                type=schema.ComponentType.GPU,
                family="mali",
                model="nx",
                variant=variant,
            )
        ]

        # Create target
        target = schema.Target(
            profile_name=profile_name,
            target_type=target_type,
            components=components,
            configuration=target_config,
            description="Neural Accelerator (NX) performance estimation",
        )

        # Calculate model hash and size
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        model_size = model_path.stat().st_size

        # Determine model format (suffix without the dot)
        suffix = model_path.suffix.lower()
        model_format = suffix.lstrip(".") or "unknown"

        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
            size_bytes=model_size,
        )

        # Create context
        context = schema.Context(
            cli_arguments=cli_arguments or [],
            runtime_configuration=None,
            git=None,
            notes=None,
        )

        metric_units = {
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

        metrics = []
        for field_name, unit in metric_units.items():
            value = getattr(self.model_performance_stats, field_name)
            metrics.append(self._build_model_metric(field_name, value, unit))

        if self.model_performance_stats.inference_time:
            metrics.append(
                schema.Metric(
                    name=schema.METRIC_NAME_INFERENCES_PER_SECOND,
                    value=1000 / self.model_performance_stats.inference_time,
                    unit=schema.UNIT_INFERENCES_PER_SECOND,
                )
            )
        metrics.append(self._build_target_utilization_metric())
        metrics = schema.ensure_standard_performance_metrics(metrics)

        entities: list[schema.Entity] = []
        breakdowns: list[schema.Breakdown] = []
        segment_entities = self.segment_entities or []
        segment_name_by_id = {
            segment.entity_id: segment.segment_name for segment in segment_entities
        }
        chain_segment_by_id = {
            chain_id: segment.entity_id
            for segment in segment_entities
            for chain_id in segment.chain_ids
        }
        cascade_segment_by_id = {
            cascade_id: segment.entity_id
            for segment in segment_entities
            for cascade_id in segment.cascade_ids
        }
        # Segment membership includes structural VGF/SPIR-V operations that may
        # be omitted from GCPE's debug/performance database.
        segment_child_ids: dict[str, list[str]] = {
            segment.entity_id: [] for segment in segment_entities
        }
        provenance_builder = NXProvenanceEntityBuilder()
        performance_group_entities: dict[str, schema.Entity] = {}
        chain_entity_indexes: dict[str, int] = {}
        for chain_id, stats in self.chain_performance_metrics.items():
            entity_id = chain_id
            chain_entity_indexes[chain_id] = len(entities)
            segment_entity_id = chain_segment_by_id.get(entity_id)
            breakdown_metrics = self._build_breakdown_metrics(stats)
            child_ids = []
            for operator_index, operator in enumerate(stats.operators):
                source_operator_ids = list(
                    dict.fromkeys(
                        source_operator_id
                        for source_operator_id in operator["source_operator_ids"]
                        if source_operator_id
                    )
                )
                if not source_operator_ids:
                    continue
                if len(source_operator_ids) == 1:
                    association_entity_id = source_operator_ids[0]
                    child_ids.append(association_entity_id)
                    association_parent_ids = [entity_id]
                else:
                    association_entity_id = _performance_group_entity_id(
                        entity_id, operator_index
                    )
                    child_ids.append(association_entity_id)
                    association_parent_ids = [association_entity_id]
                    performance_group_entities[association_entity_id] = schema.Entity(
                        id=association_entity_id,
                        kind="performance_group",
                        name="/".join(operator["operator_types"]),
                        parent_ids=[entity_id],
                        child_ids=source_operator_ids,
                        attributes={
                            "operator_types": list(operator["operator_types"]),
                        },
                    )

                for source_operator_id in source_operator_ids:
                    provenance_builder.add_source_operator(
                        source_operator_id,
                        name="/".join(operator["operator_types"]),
                        placement=schema.PlacementType.NX.value,
                        attributes={
                            "operator_types": list(operator["operator_types"]),
                        },
                        parent_ids=association_parent_ids,
                        nn_module_stacks=operator.get("nn_module_stacks", []),
                        code_stacks=operator.get("code_stacks", []),
                    )

            entities.append(
                schema.Entity(
                    id=entity_id,
                    kind="chain",
                    name=_subgraph_display_name(
                        segment_name_by_id.get(segment_entity_id, "")
                        if segment_entity_id
                        else "",
                        entity_id.rsplit("/", maxsplit=1)[-1],
                        "Chain",
                    ),
                    parent_ids=[],
                    child_ids=child_ids,
                    attributes={"stripe_ids": list(stats.op_id)},
                )
            )
            breakdowns.append(
                schema.Breakdown(
                    entity_id=entity_id,
                    metrics=breakdown_metrics,
                    id=chain_id,
                    qualifiers={},
                )
            )

        for cascade_id, stats in self.cascade_performance_metrics.items():
            entity_id = cascade_id
            segment_entity_id = cascade_segment_by_id.get(entity_id)
            if segment_entity_id is not None:
                segment_child_ids.setdefault(segment_entity_id, []).append(entity_id)
            # Chain and cascade stats are both aggregated from stripe rows, where
            # the debug database explicitly maps each stripe to one chain and
            # one cascade. Use that shared stripe membership for hierarchy
            # instead of matching source locations, which can be blank or shared.
            cascade_stripe_ids = set(stats.op_id)
            child_ids = [
                chain_id
                for chain_id, chain_stats in self.chain_performance_metrics.items()
                if chain_segment_by_id.get(chain_id) == segment_entity_id
                and any(
                    stripe_id in cascade_stripe_ids for stripe_id in chain_stats.op_id
                )
            ]
            for chain_entity_index in (
                chain_entity_indexes[chain_id]
                for chain_id in child_ids
                if chain_id in chain_entity_indexes
            ):
                chain_entity = entities[chain_entity_index]
                for parent_id in chain_entity.parent_ids:
                    if parent_id in segment_child_ids:
                        segment_child_ids[parent_id] = [
                            child_id
                            for child_id in segment_child_ids[parent_id]
                            if child_id != chain_entity.id
                        ]
                entities[chain_entity_index] = schema.Entity(
                    id=chain_entity.id,
                    kind=chain_entity.kind,
                    name=chain_entity.name,
                    placement=chain_entity.placement,
                    parent_ids=[entity_id],
                    child_ids=chain_entity.child_ids,
                    attributes=chain_entity.attributes,
                    stack_trace=chain_entity.stack_trace,
                )
            breakdown_metrics = self._build_breakdown_metrics(stats)

            entities.append(
                schema.Entity(
                    id=entity_id,
                    kind="cascade",
                    name=_subgraph_display_name(
                        segment_name_by_id.get(segment_entity_id, "")
                        if segment_entity_id
                        else "",
                        entity_id.rsplit("/", maxsplit=1)[-1],
                        "Cascade",
                    ),
                    parent_ids=[segment_entity_id] if segment_entity_id else [],
                    child_ids=child_ids,
                    attributes={"stripe_ids": list(stats.op_id)},
                )
            )
            breakdowns.append(
                schema.Breakdown(
                    entity_id=entity_id,
                    metrics=breakdown_metrics,
                    id=cascade_id,
                    qualifiers={},
                )
            )

        entities.extend(provenance_builder.hierarchy_entities())
        entities.extend(performance_group_entities.values())

        for segment in segment_entities:
            for source_operator_id in segment.structural_source_operator_ids:
                if provenance_builder.has_source_operator(source_operator_id):
                    continue
                provenance_builder.add_source_operator(
                    source_operator_id,
                    name=segment.structural_source_operator_names.get(
                        source_operator_id, source_operator_id
                    ),
                    placement=None,
                    parent_ids=[segment.entity_id],
                )
                segment_child_ids.setdefault(segment.entity_id, []).append(
                    source_operator_id
                )

        entities.extend(provenance_builder.source_entities())

        for segment_entity_id in sorted(segment_child_ids):
            entities.append(
                schema.Entity(
                    id=segment_entity_id,
                    kind="segment",
                    name=segment_name_by_id.get(segment_entity_id, segment_entity_id),
                    child_ids=list(dict.fromkeys(segment_child_ids[segment_entity_id])),
                )
            )

        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer=backend.id,
            warnings=list(self.warnings or []),
            errors=[],
            metrics=metrics,
            mode=None,  # NX doesn't specify simulation/measured,
            breakdowns=breakdowns,
            entities=entities,
            entity_kinds=NX_ENTITY_KINDS,
        )
        return schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
            extensions={},
        ).to_dict()


def _sum_optional_numbers(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return None
    return left + right


def _max_optional_numbers(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _combine_model_performance_stats(
    left: NXModelPerformanceStats, right: NXModelPerformanceStats
) -> NXModelPerformanceStats:
    total_inference_time = _sum_optional_numbers(
        left.inference_time, right.inference_time
    )
    return NXModelPerformanceStats(
        compiled_size=_sum_optional_numbers(left.compiled_size, right.compiled_size),
        cache_cycles=_sum_optional_numbers(left.cache_cycles, right.cache_cycles),
        cache_read_bytes=_sum_optional_numbers(
            left.cache_read_bytes, right.cache_read_bytes
        ),
        cache_write_bytes=_sum_optional_numbers(
            left.cache_write_bytes, right.cache_write_bytes
        ),
        compute_cycles=_sum_optional_numbers(left.compute_cycles, right.compute_cycles),
        dram_cycles=_sum_optional_numbers(left.dram_cycles, right.dram_cycles),
        dram_read_bytes=_sum_optional_numbers(
            left.dram_read_bytes, right.dram_read_bytes
        ),
        dram_write_bytes=_sum_optional_numbers(
            left.dram_write_bytes, right.dram_write_bytes
        ),
        dram_footprint=_max_optional_numbers(left.dram_footprint, right.dram_footprint),
        inference_time=total_inference_time,
        infs_per_sec=(1000 / total_inference_time if total_inference_time else None),
        total_cycles=_sum_optional_numbers(left.total_cycles, right.total_cycles),
    )


def _prefix_segment_entity_stats(stats: dict, segment_id: str, kind: str) -> dict:
    return {
        _subgraph_entity_id(segment_id, f"{kind}_{key}"): value
        for key, value in stats.items()
    }


def _skipped_compute_segments_warnings(segment_indexes: list[int]) -> list[str]:
    if not segment_indexes:
        return []
    return [
        COMPUTE_SEGMENTS_SKIPPED_WARNING.format(
            segments=summarize_list(segment_indexes)
        )
    ]


def build_performance_metrics_from_segment_databases(
    *,
    backend_config: NXPerformanceEstimatorConfig,
    segment_databases: list[NXSegmentPerformanceDatabase],
    performance_db_parser: NXPerformanceDatabaseParser | None = None,
    warnings: list[str] | None = None,
) -> NXPerformanceEstimatorPerformanceMetrics:
    """Run shared correlation, aggregation, and total construction per segment."""
    if not segment_databases:
        raise ValueError("No graph segment performance databases were provided.")

    stats_per_chain: dict[str, NXOperatorPerformanceStats] = {}
    stats_per_cascade: dict[str, NXOperatorPerformanceStats] = {}
    model_performance_stats: NXModelPerformanceStats | None = None
    segment_entities: list[NXSegmentPerformanceEntities] = []

    for segment in segment_databases:
        perf_stats = NXPerformanceStats(
            debug_db=segment.debug_database,
            performance_db=segment.performance_database,
            segment_index=segment.segment_index,
            debug_names=segment.debug_names,
        )
        segment_id = f"segment_{segment.segment_index}"
        segment_chain_stats = _prefix_segment_entity_stats(
            perf_stats.process_stats_per_chain(), segment_id, "chain"
        )
        segment_cascade_stats = _prefix_segment_entity_stats(
            perf_stats.process_stats_per_cascade(), segment_id, "cascade"
        )
        stats_per_chain.update(segment_chain_stats)
        stats_per_cascade.update(segment_cascade_stats)
        segment_entities.append(
            NXSegmentPerformanceEntities(
                segment_index=segment.segment_index,
                segment_name=segment.segment_name,
                chain_ids=list(segment_chain_stats),
                cascade_ids=list(segment_cascade_stats),
                structural_source_operator_ids=list(
                    segment.structural_source_operator_ids
                ),
                structural_source_operator_names=dict(
                    segment.structural_source_operator_names
                ),
            )
        )
        model_performance_stats = (
            segment.model_performance_stats
            if model_performance_stats is None
            else _combine_model_performance_stats(
                model_performance_stats, segment.model_performance_stats
            )
        )

    if model_performance_stats is None:  # pragma: no cover - guarded above
        raise AssertionError("Expected model performance statistics.")
    return NXPerformanceEstimatorPerformanceMetrics(
        backend_config=backend_config,
        performance_db_parser=performance_db_parser,
        chain_performance_metrics=stats_per_chain,
        cascade_performance_metrics=stats_per_cascade,
        model_performance_stats=model_performance_stats,
        segment_entities=segment_entities,
        warnings=list(warnings or []),
    )


class NXPerformanceEstimatorPerformanceEstimator(
    PerformanceEstimator[Path | Any, NXPerformanceEstimatorPerformanceMetrics]
):
    """Performance estimator for the Neural Accelerator Performance Estimator."""

    resource_dir = get_nx_resource_dir()

    def __init__(
        self, output_dir: Path, backend_config: dict, operator_types_mapping: dict
    ) -> None:
        """Init performance estimator."""
        backend_options = dict(backend_config.get("nx-performance-estimator", {}))
        self.enable_quantization = backend_options.pop("enable_quantization", None)
        self.backend_config = NXPerformanceEstimatorConfig(**backend_options)
        self.backend_config.set_config_dir(self.resource_dir)
        self.output_dir = output_dir
        self.operator_types_mapping = operator_types_mapping

    def estimate(
        self,
        model: Path | Any,
    ) -> NXPerformanceEstimatorPerformanceMetrics:
        """Estimate performance."""
        with log_action("Getting the performance data..."):
            model_path = (
                Path(model.model_path) if hasattr(model, "model_path") else model
            )

            # Check the file extension to see if we've been given a vgf file
            if is_vgf_file(model_path):
                vgf_file = model_path
            # Otherwise try to convert the file to vgf
            else:
                vgf_file = self._run_ml_sdk_model_converter(model_path)

            gcpe_segments = prepare_gcpe_compatible_vgfs(
                vgf_file, self.output_dir / "gcpe-vgf-segments"
            )
            skipped_compute_segments = getattr(
                gcpe_segments, "skipped_compute_segments", []
            )
            return self._estimate_gcpe_segments(
                gcpe_segments, model_path.stem, skipped_compute_segments
            )

    def _estimate_gcpe_segments(
        self,
        gcpe_segments: list[GCPEVGFSegment],
        output_name: str,
        skipped_compute_segments: list[int] | None = None,
    ) -> NXPerformanceEstimatorPerformanceMetrics:
        if not gcpe_segments:
            raise ValueError("No VGF graph segments found for GCPE estimation.")

        perf_db_parser = None
        segment_databases: list[NXSegmentPerformanceDatabase] = []
        for gcpe_segment in gcpe_segments:
            segment_output_name = (
                output_name
                if len(gcpe_segments) == 1
                else f"{output_name}_segment_{gcpe_segment.segment_index}"
            )
            output = self._run_nx_performance_estimator(
                gcpe_segment.path, segment_output_name
            )
            perf_db_parser = NXPerformanceDatabaseParser(
                db_path=Path(output.performance_database)
            )
            performance_db = perf_db_parser.parse_performance_database()
            debug_names = getattr(gcpe_segment, "debug_names", None)
            debug_db = NXDebugDatabaseParser(
                Path(output.debug_database),
                known_api_labels=(
                    list(debug_names.debug_name_to_spirv_ids)
                    if debug_names is not None
                    else None
                ),
            ).parse_debug_database()
            segment_databases.append(
                NXSegmentPerformanceDatabase(
                    segment_index=gcpe_segment.segment_index,
                    segment_name=getattr(
                        gcpe_segment,
                        "segment_name",
                        f"segment_{gcpe_segment.segment_index}",
                    ),
                    debug_database=debug_db,
                    performance_database=performance_db,
                    model_performance_stats=NXModelPerformanceStats.read_from_json(
                        output.model_performance
                    ),
                    debug_names=debug_names,
                    structural_source_operator_ids=list(
                        getattr(gcpe_segment, "structural_source_operator_ids", [])
                    ),
                    structural_source_operator_names=dict(
                        getattr(gcpe_segment, "structural_source_operator_names", {})
                    ),
                )
            )

        metrics = build_performance_metrics_from_segment_databases(
            backend_config=self.backend_config,
            segment_databases=segment_databases,
            performance_db_parser=perf_db_parser,
            warnings=_skipped_compute_segments_warnings(skipped_compute_segments or []),
        )
        self.json_dump(
            metrics.chain_performance_metrics,
            self.output_dir / "nx_performance_statistics.json",
        )
        return metrics

    def _run_ml_sdk_model_converter(self, model_path: Path) -> Path:
        """Run the ML SDK Model Converter and return the path to the SPIR-V file."""
        vmc_path = get_ml_sdk_model_converter_path()
        if vmc_path is None:
            backend_repo = get_backend_repository()
            vmc_path, _ = backend_repo.get_backend_settings("ml-sdk-model-converter")
        output_dir = self.output_dir / "ml-sdk-model-converter"
        output_dir.mkdir(exist_ok=True)

        model_converter = MLSDKModelConverter(
            vmc_path, enable_quantization=self.enable_quantization
        )
        vgf_file = model_converter(model_path, output_dir)
        return vgf_file

    def _run_nx_performance_estimator(
        self, vgf_file: Path, output_name: str
    ) -> NXPerformanceEstimatorOutputFiles:
        """Run the NX Performance Estimator and return the output files."""
        return run_nx_performance_estimator(
            self.output_dir,
            self.backend_config,
            vgf_file,
            output_name,
        )

    def json_dump(self, stats_per_chain: dict, output_file_path: Path) -> None:
        """Make a json dump of the stats_per_chain dict."""
        json_serializable_stats = {
            key: obj.to_dict() for key, obj in stats_per_chain.items()
        }
        with output_file_path.open("w") as json_file:
            json.dump(json_serializable_stats, json_file, indent=4)
