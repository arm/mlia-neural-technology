# SPDX-FileCopyrightText: Copyright 2024-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Accelerator compatibility derived from performance estimator output."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
from dataclasses import dataclass, replace
from functools import singledispatchmethod
from pathlib import Path
from typing import Any

# from tosa-tools
from tosa import (
    TosaGraph,
)
from tosa_serializer import TOSA_VERSION_MAJOR, TOSA_VERSION_MINOR, TosaSerializer

import mlia
import mlia.core.output_schema as schema
from mlia.backend.ml_sdk_model_converter.conversion import (
    MLSDKModelConverter,
)
from mlia.backend.ml_sdk_model_converter.install import get_ml_sdk_model_converter_path
from mlia.backend.ml_sdk_model_converter.tosa_reader import (
    _RE_MLIR_OP_LINE,
    TosaOp,
    TosaOpType,
    read_tosa_flatbuffer_ops,
    read_tosa_mlir_ops,
    tosa_flatbuffers_available,
)
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.advice import (
    NXAdviceRecord,
    add_vgf_performance_advice,
)
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXDebugDatabaseParser,
)
from mlia.backend.nx_performance_estimator.provenance import (
    NXProvenanceEntityBuilder,
    source_provenance_from_api_label,
)
from mlia.backend.nx_performance_estimator.runner import (
    run_nx_performance_estimator,
    get_nx_resource_dir,
)
from mlia.backend.nx_performance_estimator.vgf import (
    GCPEVGFSegment,
    prepare_gcpe_compatible_vgfs,
)
from mlia.backend.repo import get_backend_repository
from mlia.nx_utils.boundaries.filesystem import write_persistent_state_bytes_with_notice
from mlia.utils.filesystem import sha256

logger = logging.getLogger(__name__)

_CONVERTED_OPERATOR_ENTITY_KIND = "converted_operator"
_TRIVIAL_OPS = {
    "tosa.const",
    "tosa.const_shape",
    "tosa.reshape",
    "CONST",
    "CONST_SHAPE",
    "RESHAPE",
    "Const",
    "ConstShape",
    "Reshape",
}


@dataclass
class TOSAModel:
    """TOSA model."""

    path: Path


@dataclass
class VGFModel:
    """VGF model."""

    path: Path


def _assign_synthetic_labels(tosa_ops: dict[int, TosaOp]) -> None:
    """Give every parsed operation a unique label, preserving its original prefix."""
    counts: dict[str, int] = {}
    for op in tosa_ops.values():
        prefix = op.loc or "unknown"
        index = counts.get(prefix, 0)
        op.loc = f"{prefix}_{index}"
        counts[prefix] = index + 1


@dataclass
class _CompilerRecord:
    """Compiler placement evidence and its identifying metadata."""

    record_key: str  # Unique key
    label: str  # Debug-database label, or "unknown" when no usable label exists
    placement: str  # NX, EE, or NX/EE after merging evidence.
    op_type: str | None  # Distinct compiler types joined by '/', if available.


def _collect_tosa_compiler_records(
    *,
    compiler_records: dict[int, _CompilerRecord],
    unattributed_records: list[_CompilerRecord],
    op_ids: dict[str, list[str]],
    types: dict[str, list[str]],
    placement: str,
    segment_index: int,
    location_to_ops: dict[str, int],
) -> None:
    """Index records by source ID while retaining all unattributed records."""
    for compiled_id, labels in op_ids.items():
        op_types = types.get(compiled_id, [])
        op_type = "/".join(dict.fromkeys(value for value in op_types if value)) or None
        if not any(label.strip() for label in labels):
            # Retain placement evidence without provenance.
            unattributed_records.append(
                _CompilerRecord(
                    record_key=f"segment_{segment_index}_{placement}_{compiled_id}_0",
                    label="unknown",
                    placement=placement,
                    op_type=op_type,
                )
            )
            continue
        for index, label in enumerate(labels):
            if not label.strip():
                continue
            # Resolve the unique source operator covered by `label`.
            # A single source_op_id may be covered by multiple
            # compiler records
            provenance = source_provenance_from_api_label(label)
            source_op_id = location_to_ops.get(label)
            if source_op_id is None:
                source_op_id = location_to_ops.get(provenance.display_label)

            record = _CompilerRecord(
                record_key=f"segment_{segment_index}_{placement}_{compiled_id}_{index}",
                label=label,
                placement=placement,
                op_type=op_type,
            )
            if source_op_id is None:
                unattributed_records.append(record)
            else:
                previous = compiler_records.setdefault(source_op_id, record)
                if previous.placement != placement:
                    # Keep the first record's identity and merge placement evidence.
                    previous.placement = "NX/EE"
                if op_type:
                    known_types = (
                        previous.op_type.split("/") if previous.op_type else []
                    )
                    previous.op_type = "/".join(
                        dict.fromkeys([*known_types, *op_type.split("/")])
                    )


@dataclass
class NXOperatorCompatibilityInfo:
    """Describes a particular operator's compatibility with NX."""

    display_label: str
    source_operator_id: str | None = None
    compat_level: str | None = None
    type: str | None = None
    tosa_op: str | None = None
    error: str | None = None
    placement: str | None = None
    name: str | None = None
    nn_module_stacks: list[list[dict[str, str]]] | None = None
    code_stacks: list[list[dict[str, str]]] | None = None


class NXModelCompatibilityInfo:
    """Contains information about a model's compatibility with NX."""

    _layer_map: dict[str, NXOperatorCompatibilityInfo]

    def __init__(self, *, model_converter_used: bool = False) -> None:
        """Initialize records and track whether the executed path used conversion."""
        self._layer_map = {}
        self.advice: list[NXAdviceRecord] = []
        self._model_converter_used = model_converter_used
        self._structural_source_operator_ids: list[str] = []
        self._structural_source_operator_names: dict[str, str] = {}
        self._structural_source_operator_display_labels: dict[str, str] = {}

    def add_structural_source_operators(
        self,
        source_operator_ids: list[str],
        *,
        source_operator_names: dict[str, str] | None = None,
        source_operator_display_labels: dict[str, str] | None = None,
    ) -> None:
        """Register the complete canonical source-operator population."""
        for source_operator_id in source_operator_ids:
            if source_operator_id not in self._structural_source_operator_ids:
                self._structural_source_operator_ids.append(source_operator_id)
        self._structural_source_operator_names.update(source_operator_names or {})
        self._structural_source_operator_display_labels.update(
            source_operator_display_labels or {}
        )

    def _find_or_create_record(
        self, record_key: str, display_label: str | None = None
    ) -> NXOperatorCompatibilityInfo:
        """Get or create one compatibility record by its internal key."""
        record: NXOperatorCompatibilityInfo | None = self._layer_map.get(record_key)
        if record is None:
            record = NXOperatorCompatibilityInfo(display_label or record_key)
            self._layer_map[record_key] = record
        elif display_label is not None:
            record.display_label = display_label
        return record

    def add_estimator_placement(
        self,
        record_key: str,
        display_label: str,
        placement: str,
        op_type: str | None = None,
        *,
        source_operator_id: str | None = None,
        name: str | None = None,
        nn_module_stacks: list[list[dict[str, str]]] | None = None,
        code_stacks: list[list[dict[str, str]]] | None = None,
    ) -> None:
        """Add an op placement recovered from NX performance estimator output."""
        record = self._find_or_create_record(record_key, display_label)
        record.source_operator_id = source_operator_id
        record.name = name
        record.nn_module_stacks = nn_module_stacks
        record.code_stacks = code_stacks
        if op_type is not None:
            record.type = op_type

        if placement == "NX":
            record.compat_level = "TOSA"
            record.placement = "NX"
        elif placement == "EE":
            record.compat_level = "Shader"
            record.placement = "EE"
        elif placement == "NX/EE":
            record.compat_level = "TOSA/Shader"
            record.placement = "NX/EE"
        else:
            raise ValueError(f"Unsupported estimator placement '{placement}'.")

    @property
    def layer_map(self) -> dict[str, NXOperatorCompatibilityInfo]:
        """Return compatibility records mapped by internal record keys."""
        return self._layer_map

    def get_records(self) -> list[NXOperatorCompatibilityInfo]:
        """Return an ordered list of records."""
        return [self.layer_map[loc] for loc in sorted(self.layer_map.keys())]

    def _build_accelerator_operator_percentage_metric(
        self,
        records: list[NXOperatorCompatibilityInfo],
    ) -> schema.Metric:
        """Build the accelerator operator percentage metric from placement records."""
        if not records:
            return schema.Metric(
                name=schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
                value=None,
                unit=schema.UNIT_PERCENT,
                availability=schema.MetricAvailability.UNAVAILABLE,
                reason="Accelerator operator placement data is not available.",
            )

        nx_operator_count = sum(1 for record in records if record.placement == "NX")
        return schema.Metric(
            name=schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
            value=nx_operator_count / len(records) * 100,
            unit=schema.UNIT_PERCENT,
        )

    def to_standardized_output(
        self,
        model_path: Path,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
        target_config: dict[str, Any] | None = None,
        backend_config: dict[str, Any] | None = None,
        source_operator_attribution_available: bool = True,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run
            target_config: Optional target configuration parameters
            backend_config: Optional backend configuration parameters
            source_operator_attribution_available: Whether converted records can be
                linked truthfully to source operator entities
            warnings: Optional result warnings

        Returns:
            Standardized output dictionary
        """
        # Generate run_id and timestamp if not provided
        # source_operator_attribution_available = False
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Conversion prepares the model; the estimator supplies placements.
        backend_config = backend_config or {}
        backends = []
        if self._model_converter_used:
            backends.append(
                schema.Backend(
                    id="ml-sdk-model-converter",
                    name="ML SDK Model Converter",
                    version="unknown",
                    configuration=backend_config.get("ml-sdk-model-converter", {}),
                )
            )
        estimator_backend = schema.Backend(
            id="nx-performance-estimator",
            name="Neural Accelerator (NX) Performance Estimator",
            version="unknown",
            configuration=backend_config.get("nx-performance-estimator", {}),
        )
        backends.append(estimator_backend)

        # Create target
        target_config = target_config or {}
        target_type = target_config.get(
            "target_type", target_config.get("target", "neural-accelerator")
        )
        profile_name = target_config.get("profile_name", target_type)
        gpu_component = schema.Component(
            type=schema.ComponentType.GPU,
            family="mali",
            model="nx",
        )

        target = schema.Target(
            profile_name=profile_name,
            target_type=target_type,
            components=[gpu_component],
            configuration=target_config,
            description="Neural Accelerator (NX) compatibility check",
        )

        # Create model
        model_hash = sha256(model_path)
        model_format = model_path.suffix.lstrip(".") if model_path.suffix else "unknown"
        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
        )

        # Create context
        context = schema.Context(
            cli_arguments=cli_arguments or [],
        )

        # Create checks and operator entities. Prefer canonical source operators when
        # the frontend preserves their identity; otherwise retain converted-operation
        # context in a result-local namespace without claiming source provenance.
        checks: list[schema.Check] = []
        provenance_builder = NXProvenanceEntityBuilder()
        converted_operator_entities: list[schema.Entity] = []
        record_items = sorted(self.layer_map.items())
        records = [record for _, record in record_items]

        for idx, (record_key, record) in enumerate(record_items):
            if record.compat_level in ("TOSA", "Shader", "TOSA/Shader"):
                placement = record.placement or schema.PlacementType.NX.value
                supported = True
            else:
                placement = schema.PlacementType.CPU.value
                supported = False

            entity_attrs = {
                "index": idx,
                "compat_level": record.compat_level,
                "display_label": record.display_label,
            }
            if record.type:
                entity_attrs["op_type"] = record.type
            if record.tosa_op:
                entity_attrs["tosa_op"] = record.tosa_op

            entity_name = (
                record.name or record.type or record.tosa_op or record.display_label
            )
            entity_id = None
            if source_operator_attribution_available:
                if record.source_operator_id is not None:
                    entity_id = provenance_builder.add_source_operator(
                        record.source_operator_id,
                        name=entity_name,
                        placement=placement,
                        attributes=entity_attrs,
                        nn_module_stacks=record.nn_module_stacks or [],
                        code_stacks=record.code_stacks or [],
                    )
            else:
                if record.source_operator_id is not None:
                    converted_id = record.source_operator_id.removeprefix(
                        f"{schema.ENTITY_KIND_SOURCE_OPERATOR}/"
                    )
                    entity_attrs["converted_tosa_operator_id"] = converted_id
                else:
                    # Keep unattributed identities stable and separate from TOSA IDs.
                    converted_id = f"unattributed/{record_key}"
                entity_id = f"{_CONVERTED_OPERATOR_ENTITY_KIND}/{converted_id}"
                converted_operator_entities.append(
                    schema.Entity(
                        id=entity_id,
                        kind=_CONVERTED_OPERATOR_ENTITY_KIND,
                        name=entity_name,
                        placement=placement,
                        attributes=entity_attrs,
                    )
                )

            if supported:
                status = schema.CheckStatus.PASS
                details: dict[str, Any] = {}
            else:
                status = schema.CheckStatus.FAIL
                details = {}
                if record.error:
                    details["error"] = record.error

            checks.append(
                schema.Check(
                    id=f"nx_support_{idx}",
                    status=status,
                    entity_id=entity_id,
                    details=details,
                )
            )

        if source_operator_attribution_available:
            for source_operator_id in self._structural_source_operator_ids:
                if provenance_builder.has_source_operator(source_operator_id):
                    continue
                display_label = self._structural_source_operator_display_labels.get(
                    source_operator_id
                )
                provenance_builder.add_source_operator(
                    source_operator_id,
                    name=self._structural_source_operator_names.get(
                        source_operator_id, source_operator_id
                    ),
                    placement=None,
                    attributes=(
                        {"display_label": display_label}
                        if display_label is not None
                        else None
                    ),
                )

        entities = [*provenance_builder.entities(), *converted_operator_entities]
        nx_entity_ids = {
            entity.id
            for entity in entities
            if entity.placement == schema.PlacementType.NX.value
        }
        entity_kinds = provenance_builder.entity_kind_declarations()
        if converted_operator_entities:
            entity_kinds.append(schema.EntityKind(id=_CONVERTED_OPERATOR_ENTITY_KIND))

        # Determine overall result status
        if not records:
            result_status = schema.ResultStatus.OK
        elif all(r.compat_level in ("TOSA", "Shader", "TOSA/Shader") for r in records):
            result_status = schema.ResultStatus.OK
        elif any(r.compat_level in ("TOSA", "Shader", "TOSA/Shader") for r in records):
            result_status = schema.ResultStatus.PARTIAL
        else:
            result_status = schema.ResultStatus.INCOMPATIBLE

        # Create result
        result = schema.Result(
            kind=schema.ResultKind.COMPATIBILITY,
            status=result_status,
            producer=estimator_backend.id,
            warnings=list(warnings or []),
            errors=[],
            metrics=[self._build_accelerator_operator_percentage_metric(records)],
            checks=checks,
            entities=entities,
            entity_kinds=entity_kinds,
            advice=[
                schema.Advice(
                    id=record.id,
                    category=schema.AdviceCategory(record.category),
                    severity=schema.AdviceSeverity(record.severity),
                    message=record.message,
                    affected_entity_ids=[record.entity_id],
                )
                for record in self.advice
                if record.entity_id is not None and record.entity_id in nx_entity_ids
            ],
        )

        return schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=backends,
            results=[result],
            extensions={},
        ).to_dict()


class NXCompatibilityChecker:
    """Checker operator compability for Neural Accelerator targets."""

    def __init__(
        self, output_dir: Path, backend_config: dict[str, Any] | None = None
    ) -> None:
        """Initialize the checker."""
        self.output_dir = output_dir
        backend_options = dict(
            (backend_config or {}).get("nx-performance-estimator", {})
        )
        backend_options["system_config"] = (
            backend_options.get("system_config") or "default"
        )
        backend_options["compiler_config"] = (
            backend_options.get("compiler_config") or "default"
        )
        self._backend_config = NXPerformanceEstimatorConfig(
            backend_options["system_config"],
            backend_options["compiler_config"],
            skip_trivial_ops=backend_options.get("skip_trivial_ops", False),
        )
        self._backend_config.set_config_dir(get_nx_resource_dir())

    @singledispatchmethod
    def check_compatibility(self, arg: Any) -> NXModelCompatibilityInfo:
        """Check compatibility, default implementation."""
        raise NotImplementedError(f"Compatibility not supported for {type(arg)}")

    def _get_tosa_unique_location(self, tosa_loc: str, op_id: int) -> str:
        if not tosa_loc or tosa_loc == "unknown":
            return f"unknown_{op_id:05d}"
        return f"{tosa_loc}_{op_id}"

    def _get_tosa_type(
        self, tosa_op: str, tosa_loc: str, tensor_types: dict
    ) -> TosaOpType | None:
        if tosa_op == "tosa.const_shape":
            return TosaOpType.TOSA_SPECIFIC
        tensor_type = tensor_types.get(tosa_loc)
        if tensor_type:
            if tensor_type.startswith("FLOAT"):
                return TosaOpType.FLOAT
            if tensor_type.startswith("INT"):
                return TosaOpType.INT
        return None

    def tosa_flatbuffer_input_supported(self) -> bool:
        """Are TOSA and TOSA-MLIR files supported."""
        return tosa_flatbuffers_available()

    @check_compatibility.register
    def _(self, vgf_model: VGFModel) -> NXModelCompatibilityInfo:
        """Check compatibility of every graph segment in a VGF model."""
        comp_info = NXModelCompatibilityInfo()
        segments = prepare_gcpe_compatible_vgfs(
            vgf_model.path, self.output_dir / "gcpe-vgf-segments"
        )
        for segment in segments:
            self._add_segment_advice(comp_info, segment)
            comp_info.add_structural_source_operators(
                segment.structural_source_operator_ids
            )
            debug_db = self._run_estimator_debug_database(segment)
            self._add_compatibility_from_debug_database(comp_info, debug_db, segment)
        return comp_info

    def _add_segment_advice(
        self,
        comp_info: NXModelCompatibilityInfo,
        segment: GCPEVGFSegment,
        location_to_ops: dict[str, int] | None = None,
    ) -> None:
        """Gather advice from a single-segment VGF and resolve its report IDs."""
        advice: list[NXAdviceRecord] = []
        add_vgf_performance_advice(segment.path, advice)
        tosa_entity_ids: dict[str, str] = {}
        if location_to_ops is not None:
            for spirv_id, label in segment.debug_names.spirv_id_to_debug_name.items():
                if not label.strip():
                    continue
                source_op_id = location_to_ops.get(label)
                if source_op_id is None:
                    provenance = source_provenance_from_api_label(label)
                    source_op_id = location_to_ops.get(provenance.display_label)
                if source_op_id is not None:
                    tosa_entity_ids[schema.vgf_source_operator_id(0, int(spirv_id))] = (
                        schema.tosa_source_operator_id(source_op_id)
                    )
        for record in advice:
            entity_id = record.entity_id
            if entity_id is not None:
                if location_to_ops is not None:
                    entity_id = tosa_entity_ids.get(entity_id)
                else:
                    entity_id = entity_id.replace(
                        "source_operator/segment_0/",
                        f"source_operator/segment_{segment.segment_index}/",
                        1,
                    )
            # Each generated file has local segment index zero. Restore the
            # original segment namespace so advice IDs remain unique.
            comp_info.advice.append(
                replace(
                    record,
                    id=record.id.replace(
                        "advice/segment_0/",
                        f"advice/segment_{segment.segment_index}/",
                        1,
                    ),
                    entity_id=entity_id,
                )
            )

    def _run_estimator_debug_database(
        self,
        segment: GCPEVGFSegment,
        known_api_labels: list[str] | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        """Run graph compiler flow and return parsed debug database output."""
        output_files = run_nx_performance_estimator(
            self.output_dir,
            self._backend_config,
            segment.path,
            segment.path.stem,
        )
        return NXDebugDatabaseParser(
            output_files.debug_database,
            known_api_labels=(
                known_api_labels
                if known_api_labels is not None
                else list(segment.debug_names.debug_name_to_spirv_ids)
            ),
        ).parse_debug_database()

    def _add_compatibility_from_debug_database(
        self,
        comp_info: NXModelCompatibilityInfo,
        debug_db: dict[str, dict[str, list[str]]],
        segment: GCPEVGFSegment,
    ) -> None:
        """Add one VGF segment's compatibility and canonical provenance."""
        self._add_estimator_records(
            comp_info,
            op_ids=debug_db.get("tosa_op_id_to_api_labels", {}),
            placement="NX",
            types=debug_db.get("tosa_op_id_to_tosa_op", {}),
            segment=segment,
        )
        self._add_estimator_records(
            comp_info,
            op_ids=debug_db.get("shader_op_id_to_api_labels", {}),
            placement="EE",
            types={},
            segment=segment,
        )

    def _add_estimator_records(
        self,
        comp_info: NXModelCompatibilityInfo,
        *,
        op_ids: dict[str, list[str]],
        placement: str,
        types: dict[str, list[str]],
        segment: GCPEVGFSegment,
    ) -> None:
        """Add compatibility records for one segment debug-database table."""
        for op_id, api_labels in op_ids.items():
            op_types = types.get(op_id, [])
            if (
                self._backend_config.skip_trivial_ops
                and op_types
                and all(op_type in _TRIVIAL_OPS for op_type in op_types)
            ):
                continue
            db_op_type = "/".join(
                dict.fromkeys(op_type for op_type in op_types if op_type)
            )
            if not any(label.strip() for label in api_labels):
                # Placement still counts even when no source can be attributed.
                comp_info.add_estimator_placement(
                    f"segment_{segment.segment_index}_op_id_{op_id}_{placement}_0",
                    "unknown",
                    placement,
                    op_type=db_op_type,
                )
                continue

            for idx, api_label in enumerate(api_labels):
                if not api_label.strip():
                    continue
                provenance = source_provenance_from_api_label(
                    api_label,
                    segment_index=segment.segment_index,
                    debug_names=segment.debug_names,
                )
                # Shared provenance supports performance grouping, but does not
                # establish each candidate's individual placement or type.
                # Keep one placement record per label without multiplying evidence.
                source_id = (
                    provenance.source_operator_ids[0]
                    if len(provenance.source_operator_ids) == 1
                    else None
                )
                comp_info.add_estimator_placement(
                    f"segment_{segment.segment_index}_op_id_{op_id}_{idx}",
                    provenance.display_label,
                    placement,
                    op_type=db_op_type,
                    source_operator_id=source_id,
                    name=provenance.name,
                    nn_module_stacks=provenance.nn_module_stacks,
                    code_stacks=provenance.code_stacks,
                )

    def _prepare_tosa_model(self, tosa_path: Path) -> Path:
        """Assign unique labels and write a matching converter input.

        Empty labels become unknown_<n>; existing labels become <label>_<n>.
        The processed model is the source of labels used for compiler attribution.
        """
        if tosa_path.suffix == ".tosamlir":
            tosa_ops = read_tosa_mlir_ops(tosa_path)
        else:
            version = TosaGraph.TosaGraph.GetRootAs(tosa_path.read_bytes()).Version()
            major, minor = version._Major(), version._Minor()
            if major != TOSA_VERSION_MAJOR or minor > TOSA_VERSION_MINOR:
                raise ValueError(
                    f"Unsupported TOSA version {major}.{minor} in {tosa_path}: "
                    f"tosa-tools requires major {TOSA_VERSION_MAJOR} and "
                    f"minor <= {TOSA_VERSION_MINOR}."
                )
            tosa_ops = read_tosa_flatbuffer_ops(tosa_path)
        # Unique labels let us recover a compiler record for each TOSA operation,
        # even when operations originally shared a label.
        _assign_synthetic_labels(tosa_ops)
        annotated_path = (
            self.output_dir / f"{tosa_path.stem}_processed{tosa_path.suffix}"
        )
        if tosa_path.suffix == ".tosamlir":
            lines = []
            for line in tosa_path.read_text(encoding="utf-8").splitlines(keepends=True):
                if match := _RE_MLIR_OP_LINE.search(line):
                    label = json.dumps(tosa_ops[int(match[1])].loc, ensure_ascii=False)
                    if match[4] is not None:
                        start, end = match.span(4)
                        line = line[:start] + label + line[end:]
                    else:
                        newline = "\n" if line.endswith("\n") else ""
                        line = f"{line.rstrip()} loc({label}){newline}"
                lines.append(line)
            write_persistent_state_bytes_with_notice(
                logger,
                "writing processed TOSA MLIR model",
                annotated_path,
                "".join(lines).encode("utf-8"),
            )
            return annotated_path

        serializer = TosaSerializer(
            targetMajor=version._Major(),
            targetMinor=version._Minor(),
            targetPatch=version._Patch(),
            targetDraft=version._Draft(),
        )

        # tosa-tools may print warning messages
        # to stdout, which pollutes mlia
        # standarized output. Redirect stdout to stderr
        fflush = ctypes.CDLL(None).fflush
        fflush.argtypes = [ctypes.c_void_p]
        fflush.restype = ctypes.c_int
        # flush any outstanding stdout
        sys.stdout.flush()
        fflush(None)
        stdout_fd = os.dup(1)
        try:
            os.dup2(2, 1)  # redirect stdout to stderr
            loaded = serializer.loadTosaFlatbuffer(str(tosa_path))
        finally:
            # Flush before restoring fd 1, including when loading raises.
            fflush(None)
            os.dup2(stdout_fd, 1)
            os.close(stdout_fd)
        if not loaded:
            raise RuntimeError(f"Failed to load TOSA model {tosa_path}")

        op_id = 0
        for region in serializer.regions:
            for block in region.basicBlocks:
                for operation in block.operators:
                    operation.location.text = tosa_ops[op_id].loc
                    op_id += 1

        write_persistent_state_bytes_with_notice(
            logger,
            "writing processed TOSA model",
            annotated_path,
            serializer.serialize(),
        )
        return annotated_path

    @check_compatibility.register
    def _(self, tosa_model: TOSAModel) -> NXModelCompatibilityInfo:
        """Check compatibility of a TOSA model."""
        tosa_path = self._prepare_tosa_model(tosa_model.path)
        is_mlir = tosa_path.suffix == ".tosamlir"

        tosa_ops = (
            read_tosa_mlir_ops(tosa_path)
            if is_mlir
            else read_tosa_flatbuffer_ops(tosa_path)
        )  # {unique_id : tosa_op}

        vmc_path = get_ml_sdk_model_converter_path()
        if vmc_path is None:
            backend_repo = get_backend_repository()
            vmc_path, _ = backend_repo.get_backend_settings("ml-sdk-model-converter")
        model_converter = MLSDKModelConverter(vmc_path)
        vgf_file = model_converter(tosa_path, self.output_dir)
        compat_info = NXModelCompatibilityInfo(model_converter_used=True)
        segments = prepare_gcpe_compatible_vgfs(
            vgf_file, self.output_dir / "gcpe-vgf-segments"
        )

        source_names = {
            schema.tosa_source_operator_id(op_id): op.name
            for op_id, op in tosa_ops.items()
            if not self._backend_config.skip_trivial_ops or op.name not in _TRIVIAL_OPS
        }  # e.g {'source_operator/16' : 'tosa.reshape'}
        # Keep original labels even when the compiler omits an operation.
        # Strip only our appended counter, preserving any original suffix.
        source_labels = {
            schema.tosa_source_operator_id(op_id): op.loc.rsplit("_", 1)[0]
            for op_id, op in tosa_ops.items()
            if not self._backend_config.skip_trivial_ops or op.name not in _TRIVIAL_OPS
        }

        compat_info.add_structural_source_operators(
            list(source_names),
            source_operator_names=source_names,
            source_operator_display_labels=source_labels,
        )

        location_to_ops: dict[str, int] = {
            op.loc: op_id for op_id, op in tosa_ops.items()
        }

        compiler_records: dict[int, _CompilerRecord] = {}
        unattributed_records: list[_CompilerRecord] = []

        for segment in segments:
            self._add_segment_advice(compat_info, segment, location_to_ops)
            debug_db = self._run_estimator_debug_database(
                segment, list(location_to_ops)
            )
            for placement, label_table, type_table in (
                ("NX", "tosa_op_id_to_api_labels", "tosa_op_id_to_tosa_op"),
                ("EE", "shader_op_id_to_api_labels", ""),
            ):
                _collect_tosa_compiler_records(
                    compiler_records=compiler_records,
                    unattributed_records=unattributed_records,
                    op_ids=debug_db.get(label_table, {}),
                    types=debug_db.get(type_table, {}),
                    placement=placement,
                    segment_index=segment.segment_index,
                    location_to_ops=location_to_ops,
                )

        for record in unattributed_records:
            if (
                self._backend_config.skip_trivial_ops
                and record.op_type
                and all(
                    op_type in _TRIVIAL_OPS for op_type in record.op_type.split("/")
                )
            ):
                continue
            compat_info.add_estimator_placement(
                record.record_key, record.label, record.placement, record.op_type
            )

        for source_op_id, record in compiler_records.items():
            source_id = schema.tosa_source_operator_id(source_op_id)
            if source_id not in source_names:  # e.g a skipped trivial op
                continue
            compat_info.add_estimator_placement(
                f"{record.record_key}_source_{source_op_id}",
                source_labels[source_id],
                record.placement,
                record.op_type or tosa_ops[source_op_id].name,  # EE has no op_type
                source_operator_id=source_id,
            )

        return compat_info
