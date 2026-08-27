# SPDX-FileCopyrightText: Copyright 2024-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module to track stripe-level statistics to TFLite granularity."""

import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, TypeAlias

from mlia.backend.ml_sdk_model_converter.tosa_reader import (
    TosaOp,
    read_tosa_flatbuffer_ops,
    read_tosa_mlir_ops,
)
from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.output_parsing import (
    DebugDatabaseContentsType,
    PerformanceDatabaseContentsType,
)
from mlia.backend.nx_performance_estimator.provenance import (
    code_stacks_from_api_labels,
    nn_module_stacks_from_api_labels,
    source_operator_id_from_api_label,
)

_SPIRV_ID_LABEL_RE = re.compile(r"^TOSA[A-Z0-9_]*_spirv_id_(\d+)$")
_TOSA_OP_TO_NX_OP_TYPE = {
    "add": "Add",
    "arithmetic_right_shift": "Asr",
    "avg_pool2d": "AvgPool",
    "clamp": "Clamp",
    "clz": "CLZ",
    "conv2d": "Conv2D",
    "depthwise_conv2d": "DepthwiseConv2D",
    "logical_left_shift": "SHL",
    "mul": "Mul",
    "reduce_max": "ReduceMax",
    "reduce_sum": "ReduceSum",
    "rescale": "Rescale",
    "resize": "Resize",
    "reshape": "Reshape",
    "sub": "Sub",
    "table": "LUT",
}

NXModelCountMetricValue: TypeAlias = int | None
NXModelFloatMetricValue: TypeAlias = float | None


def _read_model_count_metric_value(
    metric_data: dict, metric_name: str
) -> NXModelCountMetricValue:
    value = metric_data["value"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"Expected integer or null value for model performance metric "
            f"'{metric_name}', got {type(value).__name__}."
        )
    if isinstance(value, float):
        if not value.is_integer():
            raise TypeError(
                f"Expected integer or null value for model performance metric "
                f"'{metric_name}', got non-integral float."
            )
        return int(value)
    return value


def _read_model_float_metric_value(
    metric_data: dict, metric_name: str
) -> NXModelFloatMetricValue:
    value = metric_data["value"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"Expected numeric or null value for model performance metric "
            f"'{metric_name}', got {type(value).__name__}."
        )
    return float(value)


def _read_optional_model_count_metric_value(
    metric_container: dict, metric_name: str
) -> NXModelCountMetricValue:
    metric_data = metric_container.get(metric_name)
    if metric_data is None:
        return 0
    return _read_model_count_metric_value(metric_data, metric_name)


@dataclass
class NXModelPerformanceStats:
    """Defines performance stats for entire model."""

    compiled_size: NXModelCountMetricValue
    cache_cycles: NXModelCountMetricValue
    cache_read_bytes: NXModelCountMetricValue
    cache_write_bytes: NXModelCountMetricValue
    compute_cycles: NXModelCountMetricValue
    dram_cycles: NXModelCountMetricValue
    dram_read_bytes: NXModelCountMetricValue
    dram_write_bytes: NXModelCountMetricValue
    dram_footprint: NXModelCountMetricValue
    inference_time: NXModelFloatMetricValue
    infs_per_sec: NXModelFloatMetricValue
    total_cycles: NXModelCountMetricValue

    @classmethod
    def read_from_json(cls, path: Path) -> "NXModelPerformanceStats":
        """Parse performance stats from NX output JSON."""
        with open(path, encoding="utf-8") as file:
            data = json.load(file)

        network_perf = data["network_performance"]
        cache1 = network_perf.get("cache1", {})
        dram = network_perf.get("dram", {})

        return cls(
            compiled_size=_read_model_count_metric_value(
                data["compiled_size"], "compiled_size"
            ),
            cache_cycles=_read_optional_model_count_metric_value(cache1, "cycles"),
            cache_read_bytes=_read_optional_model_count_metric_value(
                cache1, "read_bytes"
            ),
            cache_write_bytes=_read_optional_model_count_metric_value(
                cache1, "write_bytes"
            ),
            compute_cycles=_read_model_count_metric_value(
                network_perf["compute_cycles"], "compute_cycles"
            ),
            dram_cycles=_read_optional_model_count_metric_value(dram, "cycles"),
            dram_read_bytes=_read_optional_model_count_metric_value(dram, "read_bytes"),
            dram_write_bytes=_read_optional_model_count_metric_value(
                dram, "write_bytes"
            ),
            dram_footprint=_read_model_count_metric_value(
                network_perf["dram_footprint"], "dram_footprint"
            ),
            inference_time=_read_model_float_metric_value(
                network_perf["inference_time"], "inference_time"
            ),
            infs_per_sec=_read_model_float_metric_value(
                network_perf["infs_per_sec"], "infs_per_sec"
            ),
            total_cycles=_read_model_count_metric_value(
                network_perf["total_cycles"], "total_cycles"
            ),
        )


@dataclass
class NXOperatorPerformanceStats:
    """Defines the performance stats for one operator."""

    op_id: list
    op_cycles: int
    total_cycles: int
    memory: dict
    utilization: list
    operators: list

    def to_dict(self) -> dict:
        """Convert one object to dictionary."""
        return {
            "stripe_ids": self.op_id,
            "op_cycles": self.op_cycles,
            "total_cycles": self.total_cycles,
            "memory": self.memory,
            "utilization": self.utilization,
            "operators": self.operators,
        }

    def sanitize_memory_fields(self) -> None:
        """Remove Internal memory fields as they are meaningless."""
        del self.memory["Internal"]

    def sanitize_utilization_fields(self) -> None:
        """Aggregrate and sanitize utilization fields."""
        utilization_dict = defaultdict(list)
        for util in self.utilization:
            try:
                section_name = util["sectionName"]
                hw_util = util["cycles"]
                utilization_dict[section_name].append(int(hw_util))
            except KeyError as exc:
                raise KeyError(
                    "sectionName or cycles missing from the utilization statistics."
                ) from exc

        self.utilization = []

        for util_k, util_v in utilization_dict.items():
            sum_v = sum(util_v)
            util = {
                "sectionName": util_k,
                "cycles": f"{sum_v}",
                "percentage": f"{round(sum_v / self.total_cycles * 100, 1)}%",
            }
            self.utilization.append(util)

    def merge(
        self,
        op_perf_stats: "NXOperatorPerformanceStats",
        *,
        require_same_operators: bool = True,
    ) -> None:
        """Merge statistics belonging to different emitted stripes."""
        if require_same_operators and self.operators != op_perf_stats.operators:
            raise ValueError("The same chain should map to the same location strings!")

        self.op_id.extend(op_perf_stats.op_id)
        self.op_cycles += op_perf_stats.op_cycles
        self.total_cycles += op_perf_stats.total_cycles

        for key, value in self.memory.items():
            try:
                to_merge = op_perf_stats.memory[key]
                merged_mem = {
                    "readBytes": value["readBytes"] + to_merge["readBytes"],
                    "writeBytes": value["writeBytes"] + to_merge["writeBytes"],
                    "trafficCycles": value["trafficCycles"] + to_merge["trafficCycles"],
                }
                self.memory[key] = merged_mem
            except KeyError as exc:
                raise KeyError(
                    "Missing key in memory statistics. Cannot merge."
                ) from exc

        if not require_same_operators:
            for operator in op_perf_stats.operators:
                if operator not in self.operators:
                    self.operators.append(operator)

        self.utilization.extend(op_perf_stats.utilization)
        self.sanitize_utilization_fields()


class NXPerformanceStats:
    """Class that contains the performance stats pertaining to a model."""

    def __init__(
        self,
        debug_db: DebugDatabaseContentsType,
        performance_db: PerformanceDatabaseContentsType,
        segment_index: int = 0,
        debug_names: SpirvDebugNameMap | None = None,
    ) -> None:
        """Initialize the class with the debug and performance database dictionaries."""
        self.debug_db: DebugDatabaseContentsType = debug_db
        self.performance_db: PerformanceDatabaseContentsType = performance_db
        self.segment_index = segment_index
        self.debug_names = debug_names

    def process_stats_per_chain(
        self,
    ) -> dict:
        """Get the performance stats per stripe.

        Tracks a stripe to its location string from the original TFLite model
        or TOSA model or the VGF file and collates its performance statistics.

        Returns a dictionary where the key is the chain op_id
        and the value is itself a dictionary containing the
        stripe op_id, operations and statistics

        """
        performance_stats_per_stripe = self.process_stats_per_stripe()
        performance_stats_per_chain: Dict[str, NXOperatorPerformanceStats] = {}
        for stripe_id, operator_stats in performance_stats_per_stripe.items():
            chain_op_id, *_ = self.track_op(stripe_op_id=stripe_id)

            if chain_op_id in performance_stats_per_chain:
                performance_stats_per_chain[chain_op_id].merge(operator_stats)
            else:
                performance_stats_per_chain[chain_op_id] = operator_stats

        return performance_stats_per_chain

    def process_stats_per_cascade(self) -> dict:
        """Aggregate emitted-stripe statistics by cascade op id."""
        performance_stats_per_stripe = self.process_stats_per_stripe()
        performance_stats_per_cascade: Dict[str, NXOperatorPerformanceStats] = {}
        for stripe_id, operator_stats in performance_stats_per_stripe.items():
            cascade_op_id = self.track_cascade(stripe_op_id=stripe_id)

            if cascade_op_id in performance_stats_per_cascade:
                performance_stats_per_cascade[cascade_op_id].merge(
                    operator_stats, require_same_operators=False
                )
            else:
                performance_stats_per_cascade[cascade_op_id] = operator_stats

        return performance_stats_per_cascade

    def process_stats_per_stripe(self) -> dict:
        """Get performance stats per op."""
        performance_stats_per_stripe: Dict[str, NXOperatorPerformanceStats] = {}
        for row in self.performance_db:
            row = copy.deepcopy(row)  # Make sure nested dicts are copied
            operators = []

            (
                _,
                source_operator_ids,
                operator_types,
                module_stacks,
                stack_traces,
            ) = self.track_op(stripe_op_id=str(row["id"]))

            for (
                source_operator_ids,
                operator_str,
                nn_module_stack,
                stack_trace,
            ) in zip(
                source_operator_ids,
                operator_types,
                module_stacks,
                stack_traces,
            ):
                operator = {
                    "source_operator_ids": [
                        source_operator_id
                        for source_operator_id in source_operator_ids
                        if source_operator_id is not None
                    ],
                    "operator_types": operator_str,
                }
                if nn_module_stack:
                    operator["nn_module_stacks"] = nn_module_stack
                if stack_trace:
                    operator["code_stacks"] = stack_trace
                operators.append(operator)

            stripe_stats = NXOperatorPerformanceStats(
                op_id=[str(row["id"])],
                op_cycles=row["opCycles"],
                total_cycles=row["totalCycles"],
                memory=row["Memory"],
                utilization=row["Utilization"],
                operators=operators,
            )

            stripe_stats.sanitize_memory_fields()
            stripe_stats.sanitize_utilization_fields()

            performance_stats_per_stripe[str(row["id"])] = stripe_stats

        return performance_stats_per_stripe

    def track_op(self, stripe_op_id: str) -> tuple[str, list, list, list, list]:
        """Track a stripe to canonical source IDs and presentation metadata."""
        chain_op_id = self.debug_db["stripe_op_id_to_op_id"][stripe_op_id]

        if len(chain_op_id) > 1:
            raise ValueError("There should be only one chain per stripe, found more!")

        fused_op_ids = self.debug_db["chain_op_id_to_fused_op_ids"][chain_op_id[0]]

        tosa_op_ids = []
        for fused_op_id in fused_op_ids:
            tosa_op_ids.extend(self.debug_db["fused_op_id_to_tosa_op_ids"][fused_op_id])

        source_operator_ids = []
        module_stacks = []
        stack_traces = []
        for tosa_op_id in tosa_op_ids:
            raw_api_labels = self.debug_db["tosa_op_id_to_api_labels"][tosa_op_id]
            source_operator_ids.append(
                [
                    source_operator_id_from_api_label(
                        api_label, self.segment_index, self.debug_names
                    )
                    for api_label in raw_api_labels
                ]
            )
            module_stacks.append(nn_module_stacks_from_api_labels(raw_api_labels))
            stack_traces.append(code_stacks_from_api_labels(raw_api_labels))

        operator_types = []
        for tosa_op_id in tosa_op_ids:
            operator_types.append(self.debug_db["tosa_op_id_to_tosa_op"][tosa_op_id])

        return (
            chain_op_id[0],
            source_operator_ids,
            operator_types,
            module_stacks,
            stack_traces,
        )

    def track_cascade(self, stripe_op_id: str) -> str:
        """Track a stripe to its cascade op id."""
        cascade_op_id = self.debug_db["stripe_op_id_to_cascade_op_id"][stripe_op_id]

        if len(cascade_op_id) > 1:
            raise ValueError("There should be only one cascade per stripe, found more!")

        return cascade_op_id[0]


def read_tosa_mlir_spirv_id_locations(
    tosa_mlir_file: Path,
    debug_db: DebugDatabaseContentsType,
) -> dict[str, str]:
    """Best-effort mapping from estimator SPIR-V placeholders to MLIR locations."""
    return read_tosa_spirv_id_locations(tosa_mlir_file, debug_db)


def read_tosa_spirv_id_locations(
    tosa_file: Path,
    debug_db: DebugDatabaseContentsType,
) -> dict[str, str]:
    """Best-effort mapping from estimator SPIR-V placeholders to MLIR locations.

    Public model-converter builds may not emit the MLGraph debug records that
    VGF-based location recovery needs. In that case, align the estimator TOSA op
    stream with the source TOSA op stream and recover the source locations
    for matching operators.
    """
    source_ops = _read_located_tosa_ops(tosa_file)
    estimator_ops = _read_estimator_tosa_ops(debug_db)
    matches = _longest_common_subsequence_matches(estimator_ops, source_ops)

    locations: dict[str, str] = {}
    estimator_locations: dict[int, str] = {}
    for estimator_index, estimator_op, source_op in matches:
        if not source_op.location:
            continue
        estimator_locations[estimator_index] = source_op.location
        if match := _SPIRV_ID_LABEL_RE.search(estimator_op.api_label):
            _set_better_location(locations, match[1], source_op.location)

    _fill_unmatched_spirv_locations(estimator_ops, estimator_locations, locations)

    return locations


def has_unresolved_spirv_id_locations(
    debug_db: DebugDatabaseContentsType,
    spirv_id_locations: dict[str, str],
) -> bool:
    """Check if estimator debug labels still need SPIR-V id location fallback."""
    labels = debug_db.get("tosa_op_id_to_api_labels", {})
    for api_labels in labels.values():
        for api_label in api_labels:
            if (match := _SPIRV_ID_LABEL_RE.search(api_label)) and match[
                1
            ] not in spirv_id_locations:
                return True
    return False


@dataclass(frozen=True)
class _LocatedTosaOp:
    op_type: str
    location: str


@dataclass(frozen=True)
class _EstimatorTosaOp:
    op_type: str
    api_label: str


def _read_located_tosa_ops(tosa_file: Path) -> list[_LocatedTosaOp]:
    if tosa_file.suffix == ".tosamlir":
        tosa_ops = read_tosa_mlir_ops(tosa_file)
    elif tosa_file.suffix == ".tosa":
        tosa_ops = read_tosa_flatbuffer_ops(tosa_file)
    else:
        return []

    return [
        _LocatedTosaOp(_to_nx_op_type(tosa_op), tosa_op.loc)
        for _, tosa_op in sorted(tosa_ops.items())
        if tosa_op.name.lower()
        not in {"const", "const_shape", "tosa.const", "tosa.const_shape"}
    ]


def _to_nx_op_type(tosa_op: TosaOp) -> str:
    op_name = tosa_op.name.removeprefix("tosa.").lower()
    return _TOSA_OP_TO_NX_OP_TYPE.get(op_name, op_name)


def _read_estimator_tosa_ops(
    debug_db: DebugDatabaseContentsType,
) -> list[_EstimatorTosaOp]:
    labels = debug_db.get("tosa_op_id_to_api_labels", {})
    types = debug_db.get("tosa_op_id_to_tosa_op", {})
    ops = []
    for tosa_op_id in sorted(labels, key=int):
        api_labels = labels[tosa_op_id]
        if not api_labels or not types.get(tosa_op_id):
            continue
        ops.append(_EstimatorTosaOp(types[tosa_op_id][0], api_labels[0]))
    return ops


def _longest_common_subsequence_matches(
    estimator_ops: list[_EstimatorTosaOp],
    source_ops: list[_LocatedTosaOp],
) -> list[tuple[int, _EstimatorTosaOp, _LocatedTosaOp]]:
    matches = []
    for estimator_index, source_index in _lcs_index_pairs(estimator_ops, source_ops):
        matches.append(
            (estimator_index, estimator_ops[estimator_index], source_ops[source_index])
        )
    return matches


def _lcs_index_pairs(
    estimator_ops: Sequence[_EstimatorTosaOp],
    source_ops: Sequence[_LocatedTosaOp],
    estimator_offset: int = 0,
    source_offset: int = 0,
) -> list[tuple[int, int]]:
    estimator_len = len(estimator_ops)
    source_len = len(source_ops)
    if estimator_len == 0 or source_len == 0:
        return []
    if estimator_len == 1:
        estimator_op = estimator_ops[0]
        for source_index, source_op in enumerate(source_ops):
            if estimator_op.op_type == source_op.op_type:
                return [(estimator_offset, source_offset + source_index)]
        return []

    estimator_midpoint = estimator_len // 2
    left_lengths = _lcs_prefix_lengths(estimator_ops[:estimator_midpoint], source_ops)
    right_lengths = _lcs_prefix_lengths(
        reversed(estimator_ops[estimator_midpoint:]), reversed(source_ops)
    )
    source_split = max(
        range(source_len + 1),
        key=lambda index: left_lengths[index] + right_lengths[source_len - index],
    )

    return [
        *_lcs_index_pairs(
            estimator_ops[:estimator_midpoint],
            source_ops[:source_split],
            estimator_offset,
            source_offset,
        ),
        *_lcs_index_pairs(
            estimator_ops[estimator_midpoint:],
            source_ops[source_split:],
            estimator_offset + estimator_midpoint,
            source_offset + source_split,
        ),
    ]


def _lcs_prefix_lengths(
    estimator_ops: Iterable[_EstimatorTosaOp],
    source_ops: Iterable[_LocatedTosaOp],
) -> list[int]:
    source_ops = list(source_ops)
    previous = [0] * (len(source_ops) + 1)

    for estimator_op in estimator_ops:
        current = [0]
        for source_index, source_op in enumerate(source_ops, start=1):
            if estimator_op.op_type == source_op.op_type:
                current.append(previous[source_index - 1] + 1)
            else:
                current.append(max(previous[source_index], current[-1]))
        previous = current

    return previous


def _fill_unmatched_spirv_locations(
    estimator_ops: list[_EstimatorTosaOp],
    estimator_locations: dict[int, str],
    locations: dict[str, str],
) -> None:
    if not estimator_locations:
        return

    for estimator_index, estimator_op in enumerate(estimator_ops):
        if not (match := _SPIRV_ID_LABEL_RE.search(estimator_op.api_label)):
            continue
        if match[1] in locations:
            continue

        if nearest_location := _nearest_estimator_location(
            estimator_index, estimator_locations
        ):
            _set_better_location(locations, match[1], nearest_location)


def _nearest_estimator_location(
    estimator_index: int,
    estimator_locations: dict[int, str],
) -> str | None:
    previous_indices = [
        index for index in estimator_locations if index < estimator_index
    ]
    next_indices = [index for index in estimator_locations if index > estimator_index]
    previous_index = max(previous_indices, default=None)
    next_index = min(next_indices, default=None)

    if previous_index is None:
        if next_index is None:
            return None
        return estimator_locations[next_index]
    if next_index is None:
        return estimator_locations[previous_index]

    previous_distance = estimator_index - previous_index
    next_distance = next_index - estimator_index
    if previous_distance <= next_distance:
        return estimator_locations[previous_index]
    return estimator_locations[next_index]


def _set_better_location(
    locations: dict[str, str],
    spirv_id: str,
    candidate: str,
) -> None:
    current = locations.get(spirv_id)
    if current is None or _location_score(candidate) > _location_score(current):
        locations[spirv_id] = candidate


def _location_score(location: str) -> int:
    if not location or location == "unknown":
        return 0
    if location == "StatefulPartitionedCall:0":
        return 1
    return 2
