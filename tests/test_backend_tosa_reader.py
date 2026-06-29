# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOSA model readers."""

from pathlib import Path

from mlia.backend.ml_sdk_model_converter.tosa_reader import (
    TosaOpType,
    read_tosa_mlir_ops,
)


def test_read_tosa_mlir_ops_handles_public_converter_text_without_locations(
    tmp_path: Path,
) -> None:
    """Public TFLite converter MLIR output may omit op locations."""
    mlir_file = tmp_path / "model.tosamlir"
    mlir_file.write_text(
        """
module {
  func.func @main(%arg0: tensor<1x8xi8>) -> tensor<1x8xi8> {
    %0 = tosa.const_shape  {values = dense<[1, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = tosa.clamp %arg0 {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x8xi8>) -> tensor<1x8xi8>
    return %2 : tensor<1x8xi8>
  }
}
""",
        encoding="utf-8",
    )

    ops = read_tosa_mlir_ops(mlir_file)

    assert [op.name for op in ops.values()] == [
        "tosa.const_shape",
        "tosa.const",
        "tosa.clamp",
    ]
    assert [op.loc for op in ops.values()] == ["", "", ""]
    assert [op.type for op in ops.values()] == [
        TosaOpType.TOSA_SPECIFIC,
        TosaOpType.INT,
        TosaOpType.INT,
    ]


def test_read_tosa_mlir_ops_handles_unsigned_integer_types(tmp_path: Path) -> None:
    """Unsigned integer MLIR tensor element types should be treated as integers."""
    mlir_file = tmp_path / "model.tosamlir"
    mlir_file.write_text(
        """
module {
  func.func @main(%arg0: tensor<1x8xui8>) -> tensor<1x8xui8> {
    %0 = tosa.rescale %arg0, %arg0, %arg0, %arg0, %arg0 {input_unsigned = true, output_unsigned = true, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8xui8>, tensor<1x8xui8>, tensor<1x8xui8>, tensor<1x8xui8>, tensor<1x8xui8>) -> tensor<1x8xui8> loc(#loc1)
    return %0 : tensor<1x8xui8>
  }
} loc(#loc)
#loc1 = loc("model/rescale"(#loc))
""",
        encoding="utf-8",
    )

    ops = read_tosa_mlir_ops(mlir_file)

    assert ops[0].type == TosaOpType.INT
    assert ops[0].loc == "model/rescale"


def test_read_tosa_mlir_ops_normalizes_debug_locations(tmp_path: Path) -> None:
    """Debug-info locations should be user-facing strings without MLIR quoting."""
    mlir_file = tmp_path / "model.tosamlir"
    mlir_file.write_text(
        """
module {
  func.func @main(%arg0: tensor<1x8xi8>) -> tensor<1x8xi8> {
    %0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> loc(#loc1)
    %1 = tosa.clamp %arg0 {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x8xi8>) -> tensor<1x8xi8> loc(#loc2)
    return %1 : tensor<1x8xi8>
  }
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("model/relu"(#loc))
""",
        encoding="utf-8",
    )

    ops = read_tosa_mlir_ops(mlir_file)

    assert ops[0].loc == ""
    assert ops[1].loc == "model/relu"
