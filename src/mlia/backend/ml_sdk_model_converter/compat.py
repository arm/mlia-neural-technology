# SPDX-FileCopyrightText: Copyright 2024-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Accelerator operator compatibility module."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from functools import singledispatchmethod
from pathlib import Path
from typing import Any

import mlia
import mlia.core.output_schema as schema
from mlia.backend.ml_sdk_model_converter.conversion import MLSDKModelConverterBase
from mlia.backend.ml_sdk_model_converter.tosa_reader import (
    TosaOp,
    TosaOpType,
    read_tosa_flatbuffer_ops,
    read_tosa_mlir_ops,
    tosa_flatbuffers_available,
)
from mlia.utils.filesystem import sha256

logger = logging.getLogger(__name__)

_SUPPORTED_TOSA_OPS = [
    # Tensor Operators
    "ARGMAX",
    "AVG_POOL2D",
    "CONV2D",
    "CONV3D",
    "DEPTHWISE_CONV2D",
    "FFT2D",
    "MATMUL",
    "MAX_POOL2D",
    "RFFT2D",
    "TRANSPOSE_CONV2D",
    # Activation Functions
    "CLAMP",
    "ERF",
    "SIGMOID",
    "TANH",
    # Elementwise Binary Operators
    "ADD",
    "ARITHMETIC_RIGHT_SHIFT",
    "BITWISE_AND",
    "BITWISE_OR",
    "BITWISE_XOR",
    "INTDIV",
    "LOGICAL_AND",
    "LOGICAL_LEFT_SHIFT",
    "LOGICAL_RIGHT_SHIFT",
    "LOGICAL_OR",
    "LOGICAL_XOR",
    "MAXIMUM",
    "MINIMUM",
    "MUL",
    "POW",
    "SUB",
    "TABLE",
    # Elementwise Unary Operators
    "ABS",
    "BITWISE_NOT",
    "CEIL",
    "CLZ",
    "COS",
    "EXP",
    "FLOOR",
    "LOG",
    "LOGICAL_NOT",
    "NEGATE",
    "RECIPROCAL",
    "RSQRT",
    "SIN",
    # Elementwise Ternary Operators
    "SELECT",
    # Comparison Operators
    "EQUAL",
    "GREATER",
    "GREATER_EQUAL",
    # Reduction Operators
    "REDUCE_ALL",
    "REDUCE_ANY",
    "REDUCE_MAX",
    "REDUCE_MIN",
    "REDUCE_PRODUCT",
    "REDUCE_SUM",
    # Data Layout
    "CONCAT",
    "PAD",
    "RESHAPE",
    "REVERSE",
    "SLICE",
    "TILE",
    "TRANSPOSE",
    # Scatter/Gather Operators
    "GATHER",
    "SCATTER",
    # Image Operators
    "RESIZE",
    # Type Conversion
    "CAST",
    "RESCALE",
    # Shape Operators
    "CONST_SHAPE",
    # Data Nodes
    "CONST",
]


@dataclass
class TOSAModel:
    """TOSA model."""

    path: Path


@dataclass
class VGFModel:
    """VGF model."""

    path: Path


@dataclass
class VMCCompatibilityLogReader:
    """Read log from VMC and extract low-level NX compatibility information."""

    _lowered_ops: dict[str, str]
    _lowering_errors: dict[str, str]

    def __init__(self) -> None:
        """Initialize cumulative fields and patterns."""
        self._lowered_ops = {}
        self._lowering_errors = {}
        self._loc_pattern = re.compile(
            r"(?:loc\(\"(\S+)\"\(\"([^\"]+)\"(?::\d+:\d+)?\)\))"
            r"|(?:loc\(\"(\S+)\"\))"
            r"|(?:loc\(fused\[(.*?)\]\))"
        )

        self._success_pattern = re.compile(r"^Successfully lowered: (\S*)\s+(at)? (.*)")
        self._error_pattern = re.compile(r"^\S+: error: (.*?): (.*?)$")

    def __call__(self, line: str) -> None:
        """Redirect output to the logger."""
        if match := self._success_pattern.match(line):
            lowered_op, filling, rest = match.group(1, 2, 3)
            if filling != "at":
                raise RuntimeError(f"Unrecognized log line: '{line}'")

            loc_string = self.parse_loc(rest)
            self._lowered_ops[loc_string] = lowered_op
        elif match := self._error_pattern.match(line):
            loc_part, error = match.group(1, 2)
            loc_string = self.parse_loc(loc_part)
            self._lowering_errors[loc_string] = error

    @property
    def lowered_ops(self) -> dict[str, str]:
        """Return a mapping of source(TF) to lowered(TOSA) operators."""
        return self._lowered_ops

    @property
    def lowering_errors(self) -> dict[str, str]:
        """Return a mapping of source(TF) to error messages of affected operators."""
        return self._lowering_errors

    def parse_loc(self, line: str) -> str:
        """Parse loc() expression in log strings."""
        if line.strip() == "loc(unknown)":
            return "unknown"  # handle unknowns gracefully

        loc_match = self._loc_pattern.match(line)
        if loc_match:
            if loc_match.group(1):  # nested: loc("op"("file"...))
                return loc_match.group(1)
            if loc_match.group(3):  # simple: loc("op")
                return loc_match.group(3)
            if loc_match.group(4):  # fused
                return loc_match.group(4).split(", ")[0].strip('"')

        raise RuntimeError(f"Can't find a valid location string in {line}")


class VMCCompatbilityChecker(MLSDKModelConverterBase):
    """Run the ML SDK Model Converter to check for NX compatibility."""

    def __init__(self, converter_path: Path) -> None:
        """Set up compatilibity checking for ML SDK Model Converter."""
        super().__init__(converter_path)
        self._compatibility_log_reader = VMCCompatibilityLogReader()
        self.output_consumers.append(self._compatibility_log_reader)

    def _extra_back_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC back-end."""
        return ["--emit-debug-info", "--experimental-analysis"]

    @property
    def compatibility_log_reader(self) -> VMCCompatibilityLogReader:
        """Return the log reader which holds to operator-level details."""
        return self._compatibility_log_reader


@dataclass
class NXOperatorCompatibilityInfo:
    """Describes a particular operator's compatibility with NX."""

    location: str
    compat_level: str | None = None
    type: str | None = None
    tosa_op: str | None = None
    error: str | None = None
    placement: str | None = None


class NXModelCompatibilityInfo:
    """Contains information about a model's compatibility with NX."""

    _layer_map: dict[str, NXOperatorCompatibilityInfo]

    def __init__(self, location_to_type: dict[str, str] | None = None) -> None:
        """Initialize the database."""
        self._layer_map = {}
        self._location_to_type = location_to_type or {}

    def _find_or_create_record(self, location: str) -> NXOperatorCompatibilityInfo:
        """Get a record for a particular model location, create if necessary."""
        record: NXOperatorCompatibilityInfo | None = self._layer_map.get(location)
        if record is None:
            record = NXOperatorCompatibilityInfo(location)
            record.type = self._location_to_type.get(location)
            self._layer_map[location] = record
        return record

    def add_lowered_to_tosa(self, tosa_op: TosaOp) -> None:
        """Add an op to the database, that was reported to be lowered to TOSA."""
        record = self._find_or_create_record(tosa_op.loc)
        record.tosa_op = tosa_op.name

        is_shader_op = tosa_op.name in ["tosa.custom", "CUSTOM"]
        # [TODO]: Replace with a proper check once FP support is added
        # to the performance estimator (MLIA-1640)
        is_fp_op = tosa_op.type == TosaOpType.FLOAT

        if is_shader_op or is_fp_op:
            record.compat_level = "Shader"
            record.placement = "EE"
        else:
            record.compat_level = "TOSA"
            record.placement = "NX"

    def add_lowering_error(self, location: str, error: str) -> None:
        """Add an op to the database, which can't be lowered due to some error."""
        record = self._find_or_create_record(location)
        record.error = error
        record.compat_level = "Non-NX"

    @property
    def layer_map(self) -> dict[str, NXOperatorCompatibilityInfo]:
        """Returns the underlying compatibilty records mapped to locations strings."""
        return self._layer_map

    def get_records(self) -> list[NXOperatorCompatibilityInfo]:
        """Return an ordered list of records."""
        return [self.layer_map[loc] for loc in sorted(self.layer_map.keys())]

    def to_standardized_output(
        self,
        model_path: Path,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
        target_config: dict[str, Any] | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run
            target_config: Optional target configuration parameters
            backend_config: Optional backend configuration parameters

        Returns:
            Standardized output dictionary
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend
        backend = schema.Backend(
            id="ml-sdk-model-converter",
            name="ML SDK Model Converter",
            version="unknown",
            configuration=backend_config or {},
        )

        # Create target
        target_type = (target_config or {}).get("target", "neural-accelerator")
        gpu_component = schema.Component(
            type=schema.ComponentType.GPU,
            family="mali",
            model="nx",
        )

        target = schema.Target(
            profile_name=target_type,
            target_type="gpu",
            components=[gpu_component],
            configuration=target_config or {},
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

        # Create checks and entities for each operator
        checks: list[schema.Check] = []
        entities: list[schema.Entity] = []

        for idx, record in enumerate(self.get_records()):
            entity_id = f"op_{idx}"

            # Determine placement based on compat level
            if record.compat_level in ("TOSA", "Shader"):
                placement = record.placement.lower() if record.placement else "nx"
                supported = True
            else:
                placement = "cpu"
                supported = False

            # Create entity for this operator
            entity_attrs = {
                "index": idx,
                "compat_level": record.compat_level,
            }
            if record.type:
                entity_attrs["op_type"] = record.type
            if record.tosa_op:
                entity_attrs["tosa_op"] = record.tosa_op

            entity = schema.Entity(
                scope=schema.OperatorScope.OPERATOR,
                name=record.location,
                location=record.location,
                placement=placement,
                id=entity_id,
                attributes=entity_attrs,
            )
            entities.append(entity)

            # Create check for NX compatibility
            if supported:
                status = schema.CheckStatus.PASS
                details: dict[str, Any] = {}
            else:
                status = schema.CheckStatus.FAIL
                details = {}
                if record.error:
                    details["error"] = record.error

            check = schema.Check(
                id=f"nx_support_{entity_id}",
                status=status,
                details=details,
            )
            checks.append(check)

        # Determine overall result status
        records = self.get_records()
        if not records:
            result_status = schema.ResultStatus.OK
        elif all(r.compat_level in ("TOSA", "Shader") for r in records):
            result_status = schema.ResultStatus.OK
        elif any(r.compat_level in ("TOSA", "Shader") for r in records):
            result_status = schema.ResultStatus.PARTIAL
        else:
            result_status = schema.ResultStatus.INCOMPATIBLE

        # Create result
        result = schema.Result(
            kind=schema.ResultKind.COMPATIBILITY,
            status=result_status,
            producer=backend.id,
            warnings=[],
            errors=[],
            checks=checks,
            entities=entities,
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

    def dump(self) -> list[dict]:
        """Dump info into a list of strings, for testing purposes."""

        def filter_no_values(k2v: dict) -> dict:
            return {k: v for k, v in sorted(k2v.items()) if v}

        return [filter_no_values(asdict(record)) for record in self.get_records()]


class NXCompatibilityChecker:
    """Checker operator compability for Neural Accelerator targets."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the checker."""
        self.output_dir = output_dir
        self.tosa_mlir_to_tosa_map = {
            f"tosa.{tosa_op.lower()}": tosa_op for tosa_op in _SUPPORTED_TOSA_OPS
        }

    @singledispatchmethod
    def check_compatibility(self, arg: Any) -> NXModelCompatibilityInfo:
        """Check compatibility, default implementation."""
        raise NotImplementedError(f"Compatibility not supported for {type(arg)}")

    def _get_tosa_unique_location(self, tosa_loc: str, op_id: int) -> str:
        return f"{tosa_loc}_{op_id}"

    def _get_supported_tosa_op_name(self, tosa_op: TosaOp, is_mlir: bool) -> str | None:
        if is_mlir:
            return self.tosa_mlir_to_tosa_map.get(tosa_op.name)
        return tosa_op.name if tosa_op.name in _SUPPORTED_TOSA_OPS else None

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
    def _(self, _vgf_model: VGFModel) -> NXModelCompatibilityInfo:
        """Check compatibility of a VGF model."""
        # Currently not supported as VGF models are not inherently NX compatible
        # and can contain incompatible operations, data types, etc.
        raise NotImplementedError(
            "Compatibility info is not supported yet for VGF models for this target."
        )

    @check_compatibility.register
    def _(self, tosa_model: TOSAModel) -> NXModelCompatibilityInfo:
        """Check compatibility of a TOSA model."""
        tosa_ops: dict[int, TosaOp] = {}
        tosa_path = tosa_model.path
        is_mlir: bool = False
        try:
            if tosa_path.suffix == ".tosamlir":
                tosa_ops = read_tosa_mlir_ops(tosa_path)
                is_mlir = True
            elif tosa_path.suffix == ".tosa":
                tosa_ops = read_tosa_flatbuffer_ops(tosa_path)
            else:
                raise ValueError(f"Unsupported file format '{tosa_path.suffix}'")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read TOSA operations from {tosa_path}: {exc}"
            ) from exc

        if not tosa_ops:
            logger.warning("Could not find any TOSA operations.")
            return NXModelCompatibilityInfo()

        location_to_op_names = {
            self._get_tosa_unique_location(tosa_op.loc, op_id): tosa_op.name
            for op_id, tosa_op in tosa_ops.items()
        }
        comp_info = NXModelCompatibilityInfo(location_to_op_names)
        for op_id, tosa_op in tosa_ops.items():
            op_name = self._get_supported_tosa_op_name(tosa_op, is_mlir)
            unique_location = self._get_tosa_unique_location(tosa_op.loc, op_id)
            if op_name:
                comp_info.add_lowered_to_tosa(
                    TosaOp(op_name, unique_location, tosa_op.type)
                )
            else:
                comp_info.add_lowering_error(unique_location, "unsupported operation")

        return comp_info
