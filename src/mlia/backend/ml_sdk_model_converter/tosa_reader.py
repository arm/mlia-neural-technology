# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""A tosa file parsing functionalities."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from mlia.backend.errors import BackendUnavailableError

_TOSA_FLATBUFFERS_AVAILABLE = True

try:
    from tosa_flatbuffers.tosa import (
        DType,  # pragma: no cover
        Op,  # pragma: no cover
        TosaGraph,
    )
except ImportError:  # pragma: no cover
    _TOSA_FLATBUFFERS_AVAILABLE = False  # pragma: no cover

logger = logging.getLogger(__name__)

# Capture locations: loc#1 = loc("model/layer") will have 1 and "model/layer" captured
_RE_MLIR_LOCATION = re.compile(r"^\s*(#loc\d+)\s*=\s*loc\((.*?)(?:\(#loc\))?\)\s*$")

# Capture ssa variable name, op name, return type and loc reference:
# %5 = "tosa.const"() <{values = ...> : () -> tensor<3x3x3x8xi8> loc(#loc1)
# will have 5, "tosa.const", "i8" and "#loc1" captured
# %44 = tosa.const_shape  {values = ...} : () -> !tosa.shape<4> loc(#loc1)
# will have 44, "tosa.const_shape", "tosa.shape" and "#loc1" captured
_RE_MLIR_OP_LINE = re.compile(
    r'^\s*%(\w+)\s*=\s*"?([\w.]+)"?.*?->\s*'
    r"(?:tensor<[\w*x]+x(\w+)>|!([\w.]+)<\d+>).*?loc\(([^)]*)\)\s*$"
)


class TosaOpType(Enum):
    """A TOSA operator type."""

    INT = auto()
    FLOAT = auto()
    TOSA_SPECIFIC = auto()  # e.g tosa.const_shape


@dataclass
class TosaOp:
    """A single TOSA operator."""

    name: str
    loc: str
    type: TosaOpType | None


def tosa_flatbuffers_available() -> bool:
    """Check if Tosa Flatbuffers backend is available."""
    return _TOSA_FLATBUFFERS_AVAILABLE


def _try_get_torch_fx_node_name(op_loc: str) -> str | None:
    torch_fx_node_name = None
    try:
        parsed = json.loads(op_loc)
        # Try direct node_name first (from mlia-pytorch-to-tosa-converter)
        torch_fx_node_name = parsed.get("node_name")
        # Fall back to nested aten_info.node_name (from debug_hook)
        if not torch_fx_node_name:
            torch_fx_node_name = parsed.get("aten_info", {}).get("node_name")
    except (
        json.JSONDecodeError,
        AttributeError,
    ):
        pass
    return torch_fx_node_name


def read_tosa_mlir_ops(tosa_mlir_file: Path) -> dict[int, TosaOp]:
    """Read TOSA operations from a .tosamlir file.

    Args:
        tosa_mlir_file(Path): A path to a .tosamlir file.

    Returns:
        A dictionary that maps op ids to TosaOp objects.
    """
    id_to_ops: dict[int, TosaOp] = {}
    id_to_loc_ref: dict[int, str] = {}
    loc_ref_to_loc: dict[str, str] = {}
    with open(str(tosa_mlir_file), encoding="utf-8") as file:
        for line in file:
            logger.debug("Parsing %s", line)
            if match := _RE_MLIR_OP_LINE.search(line):
                op_name = match[2]
                return_type = match[3] or match[4]  # Either tensor type or tosa type
                loc_ref = match[5]

                try:
                    op_id = int(match[1])
                except ValueError as err:
                    raise ValueError(
                        f"Failed to parse op {match[1], op_name}. "
                        "MLIR variables names must be numeric."
                    ) from err

                if op_id in id_to_ops:
                    raise ValueError(f"Tosa op with id {op_id} duplicated.")
                logger.debug("Adding op %s: %s", op_id, op_name)

                # Set empty locations temporarily
                if return_type.startswith("i"):
                    id_to_ops.update({op_id: TosaOp(op_name, "", TosaOpType.INT)})
                elif return_type.startswith("f") or return_type.startswith("bf"):
                    id_to_ops.update({op_id: TosaOp(op_name, "", TosaOpType.FLOAT)})
                elif return_type.startswith("tosa."):  # Tosa-specific type
                    id_to_ops.update(
                        {op_id: TosaOp(op_name, "", TosaOpType.TOSA_SPECIFIC)}
                    )
                else:
                    raise ValueError(f"Unsupported type: {return_type}")
                id_to_loc_ref.update({op_id: loc_ref})
                continue

            if match := _RE_MLIR_LOCATION.search(line):
                op_loc = match[2]
                if op_loc:
                    op_loc = _try_get_torch_fx_node_name(op_loc) or op_loc
                loc_ref_to_loc.update({match[1]: op_loc})
                continue

            logger.debug("Line %s ignored", line)

    for op_id, tosa_op in id_to_ops.items():
        tosa_op.loc = loc_ref_to_loc.get(id_to_loc_ref[op_id], "")

    return id_to_ops


def _get_tosa_flatbuffer_op_type(operation: Any, block: Any) -> TosaOpType | None:
    tensor_name_to_type: dict[str, int] = {}
    for t_idx in range(block.TensorsLength()):
        tensor = block.Tensors(t_idx)
        tensor_name_to_type[tensor.Name().decode("utf-8")] = tensor.Type()
    tosa_dtype_names = {
        v: k for k, v in DType.DType.__dict__.items() if isinstance(v, int)
    }

    #  https://gitlab.arm.com/tosa/tosa-tools/-/blob/v2025.11.0/serialization/schema/tosa.fbs?ref_type=tags#L26  # noqa: E501  # Line too long
    int_dtypes = {"INT4", "INT8", "INT16", "INT32", "INT48", "INT64", "MXINT8", "BOOL"}
    float_dtypes = {
        "FP32",
        "FP16",
        "BF16",
        "FP8E4M3",
        "FP8E5M2",
        "FP6E2M3",
        "FP6E3M2",
        "FP4E2M1",
        "FP8UE8M0",
    }
    tosa_specific_dtypes = {"SHAPE"}
    output_name = operation.Outputs(0).decode("utf-8")
    if operation.Op() == Op.Op.CONST_SHAPE:
        return TosaOpType.TOSA_SPECIFIC

    if output_name in tensor_name_to_type:
        dtype_val = tensor_name_to_type[output_name]
        dtype_name = tosa_dtype_names.get(dtype_val)
        if dtype_name in int_dtypes:
            return TosaOpType.INT
        if dtype_name in float_dtypes:
            return TosaOpType.FLOAT
        if dtype_name in tosa_specific_dtypes:
            return TosaOpType.TOSA_SPECIFIC
        raise ValueError(f"Unsupported type {dtype_val}")
    return None


def read_tosa_flatbuffer_ops(tosa_flatbuffer_file: Path) -> dict[int, TosaOp]:
    """Read TOSA operations from a flatbuffer file.

    Args:
        tosa_mlir_file(Path): A path to a .tosa flatbuffer file.

    Returns:
        A dictionary that maps op ids to TosaOp objects.
    """
    if not tosa_flatbuffers_available():
        raise BackendUnavailableError(
            "Tosa Flatbuffers backend not available", "tosa-flatbuffers"
        )

    # Create a mapping from Op enum values to their string names
    tosa_flatbuffer_ops = {
        v: k for k, v in Op.Op.__dict__.items() if isinstance(v, int)
    }

    with open(tosa_flatbuffer_file, "rb") as file:
        buf = bytearray(file.read())
    tosa_graph = TosaGraph.TosaGraph.GetRootAsTosaGraph(buf)
    tosa_ops = {}
    op_id = 0
    for i in range(tosa_graph.RegionsLength()):
        reg = tosa_graph.Regions(i)
        for j in range(reg.BlocksLength()):
            block = reg.Blocks(j)

            for k in range(block.OperatorsLength()):
                operation = block.Operators(k)
                # Try to read location if available (added in ExecuTorch 1.0.0+)
                try:
                    op_loc = operation.Location().Text().decode("utf-8")
                except AttributeError:
                    # Location not available in older TOSA files
                    op_loc = ""

                # Handle ExecuTorch JSON location format
                if op_loc:
                    op_loc = _try_get_torch_fx_node_name(op_loc) or op_loc
                else:
                    op_loc = "unknown"

                op_name = tosa_flatbuffer_ops[operation.Op()]

                # Assing output type as op's type
                op_type = None
                if operation.OutputsLength() > 0:
                    op_type = _get_tosa_flatbuffer_op_type(operation, block)

                tosa_ops.update({op_id: TosaOp(op_name, op_loc, op_type)})
                op_id += 1
    return tosa_ops
