# SPDX-FileCopyrightText: Copyright 2024-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Accelerator operator compatibility module."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict
from dataclasses import dataclass
from functools import singledispatchmethod
from pathlib import Path
from typing import Any

from mlia.backend.ml_sdk_model_converter.conversion import MLSDKModelConverterBase
from mlia.backend.ml_sdk_model_converter.tosa_reader import read_tosa_flatbuffer_ops
from mlia.backend.ml_sdk_model_converter.tosa_reader import read_tosa_mlir_ops
from mlia.backend.ml_sdk_model_converter.tosa_reader import tosa_flatbuffers_available
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOp
from mlia.backend.repo import get_backend_repository
from mlia.nn.tensorflow.tflite_graph import operator_names_to_types

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
]


@dataclass
class TOSAModel:
    """TOSA model."""

    path: Path


@dataclass
class VGFModel:
    """VGF model."""

    path: Path


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

    def add_lowered_to_tosa(self, location: str, tosa_op: str) -> None:
        """Add an op to the database, that was reported to be lowered to TOSA."""
        record = self._find_or_create_record(location)
        record.tosa_op = tosa_op
        is_shader_op = tosa_op in ["tosa.custom", "CUSTOM"]
        if is_shader_op:
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

    def tosa_flatbuffer_input_supported(self) -> bool:
        """Are TOSA and TOSA-MLIR files supported."""
        return tosa_flatbuffers_available()

    @check_compatibility.register
    def _(self, tflite_model_path: Path) -> NXModelCompatibilityInfo:
        """Run compabitlity check for TFLite using ML SDK Model Converter."""
        backend_repo = get_backend_repository()
        vmc_path, _ = backend_repo.get_backend_settings("ml-sdk-model-converter")
        output_dir = self.output_dir / "ml-sdk-model-converter"
        output_dir.mkdir()

        vmc = VMCCompatbilityChecker(vmc_path)
        vmc(tflite_model_path, output_dir)
        reader: VMCCompatibilityLogReader = vmc.compatibility_log_reader

        comp_info = NXModelCompatibilityInfo(operator_names_to_types(tflite_model_path))

        for lowered_op, tosa_op in reader.lowered_ops.items():
            comp_info.add_lowered_to_tosa(lowered_op, tosa_op)

        for location, error in reader.lowering_errors.items():
            comp_info.add_lowering_error(location, error)

        return comp_info

    @check_compatibility.register
    def _(self, _vgf_model: VGFModel) -> NXModelCompatibilityInfo:
        """Check compatibility of a VGF model."""
        return NXModelCompatibilityInfo()  # all VGF ops are supported

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

        location_to_types = {
            self._get_tosa_unique_location(tosa_op.loc, op_id): tosa_op.name
            for op_id, tosa_op in tosa_ops.items()
        }
        comp_info = NXModelCompatibilityInfo(location_to_types)
        for op_id, tosa_op in tosa_ops.items():
            op_name = self._get_supported_tosa_op_name(tosa_op, is_mlir)
            unique_location = self._get_tosa_unique_location(tosa_op.loc, op_id)
            if op_name:
                comp_info.add_lowered_to_tosa(unique_location, op_name)
            else:
                comp_info.add_lowering_error(unique_location, "unsupported operation")

        return comp_info
