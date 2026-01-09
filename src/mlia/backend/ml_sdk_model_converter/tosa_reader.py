# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""A tosa file parsing functionalities."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from mlia.backend.errors import BackendUnavailableError

_TOSA_FLATBUFFERS_AVAILABLE = True  # pylint: disable=invalid-name

try:
    from tosa_flatbuffers.tosa import TosaGraph
    from tosa_flatbuffers.tosa import Op  # pragma: no cover
except ImportError:  # pragma: no cover
    _TOSA_FLATBUFFERS_AVAILABLE = (  # pylint: disable=invalid-name
        False  # pragma: no cover
    )

logger = logging.getLogger(__name__)

# Capture locations: loc#1 = loc("model/layer") will have 1 and "model/layer" captured
_RE_MLIR_LOCATION = re.compile(r"^\s*(#loc\d+)\s*=\s*loc\((.*?)(?:\(#loc\))?\)\s*$")

# Capture ssa variable name, op name and loc reference:
# %5 = "tosa.const"() <{values = ...> : () -> tensor<3x3x3x8xi8> loc(#loc1)
# will have 5, "tosa.const()" and 1 captured
_RE_MLIR_OP_LINE = re.compile(
    r'^\s*%(\w+)\s*=\s*"?([\w.]+)"?(?:(?!\bloc\().)*?loc\(([^)]*)\)\s*$'
)


@dataclass
class TosaOp:
    """A single TOSA operator."""

    name: str
    loc: str


def tosa_flatbuffers_available() -> bool:
    """Check if Tosa Flatbuffers backend is available."""
    return _TOSA_FLATBUFFERS_AVAILABLE


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
                loc_ref = match[3]

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
                id_to_ops.update({op_id: TosaOp(op_name, "")})
                id_to_loc_ref.update({op_id: loc_ref})
                continue

            if match := _RE_MLIR_LOCATION.search(line):
                loc_ref_to_loc.update({match[1]: match[2]})
                continue

            logger.debug("Line %s ignored", line)

    for op_id, tosa_op in id_to_ops.items():
        tosa_op.loc = loc_ref_to_loc.get(id_to_loc_ref[op_id], "")

    return id_to_ops


# pylint: disable=too-many-locals
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
                op_loc = op_loc or "unknown"

                op_name = tosa_flatbuffer_ops[operation.Op()]
                tosa_ops.update({op_id: TosaOp(op_name, op_loc)})
                op_id += 1
    return tosa_ops
