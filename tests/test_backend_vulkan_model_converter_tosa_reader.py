# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for tosa reader."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.errors import BackendUnavailableError
from mlia.backend.ml_sdk_model_converter.tosa_reader import read_tosa_flatbuffer_ops
from mlia.backend.ml_sdk_model_converter.tosa_reader import read_tosa_mlir_ops
from mlia.backend.ml_sdk_model_converter.tosa_reader import tosa_flatbuffers_available
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOp


def _check_id_to_tosa_ops(id_to_tosa_ops: dict[int, TosaOp], expected_len: int) -> None:
    assert len(id_to_tosa_ops) == expected_len

    assert isinstance(id_to_tosa_ops, dict)

    # No id duplicates
    op_ids = list(id_to_tosa_ops.keys())
    assert len(op_ids) == len(set(op_ids))

    tosa_ops = list(id_to_tosa_ops.values())
    names = [op.name for op in tosa_ops]
    locations = [op.loc for op in tosa_ops]

    # Assert every element is a non-empty string
    assert all(isinstance(name, str) and name for name in names)
    assert all(isinstance(loc, str) and loc for loc in locations)


def test_read_tosa_mlir_model(
    test_tosa_mlir_model_with_length: tuple[Path, int]
) -> None:
    """Tests TOSA-MLIR file parser on a valid model
    (all locations defined, correct variable names, etc)."""
    model_path, expected_len = test_tosa_mlir_model_with_length
    id_to_tosa_ops = read_tosa_mlir_ops(model_path)
    _check_id_to_tosa_ops(id_to_tosa_ops, expected_len)


@pytest.mark.parametrize(
    "tosa_mlir_content, expected_err",
    [
        (
            """
%0 = "tosa.const"() : () -> tensor<256xi8> loc(#loc1)
%0 = "tosa.const"() : () -> tensor<3xi8> loc(#loc1)
%2 = "tosa.const"() : () -> tensor<3xi32> loc(#loc1)
%3 = "tosa.const"() : () -> tensor<1xi8> loc(#loc1)
            """,
            pytest.raises(ValueError, match="Tosa op with id 0 duplicated."),
        ),
        (
            """
%0 = "tosa.const"() : () -> tensor<256xi8> loc(#loc1)
%1 = "tosa.const"() : () -> tensor<3xi8> loc(#loc1)
%01 = "tosa.const"() : () -> tensor<3xi32> loc(#loc1)
%3 = "tosa.const"() : () -> tensor<1xi8> loc(#loc1)
            """,
            pytest.raises(ValueError, match="Tosa op with id 1 duplicated."),
        ),
        (
            """
%0 = "tosa.const"() : () -> tensor<256xi8> loc(#loc1)
%invalid_id = "tosa.const"() : () -> tensor<3xi8> loc(#loc1)
%2 = "tosa.const"() : () -> tensor<3xi32> loc(#loc1)
%3 = "tosa.const"() : () -> tensor<1xi8> loc(#loc1)
            """,
            pytest.raises(ValueError, match="Failed to parse op"),
        ),
    ],
)
def test_read_tosa_mlir_bad_mlir(
    tosa_mlir_content: str, expected_err: Any, tmp_path: Path
) -> None:
    """Tests if invalid input files are handled correctly."""

    tosa_mlir_file = tmp_path / "test.tosamlir"

    with open(tosa_mlir_file, "w", encoding="utf-8") as file:
        file.write(tosa_mlir_content)
    with expected_err:
        _ = read_tosa_mlir_ops(tosa_mlir_file)


def test_read_tosa_mlir_missing_loc(tmp_path: Path) -> None:
    """Tests if missing location results in an empty string being set in a TosaOP"""
    tosa_mlir_file = tmp_path / "test.tosamlir"

    with open(tosa_mlir_file, "w", encoding="utf-8") as file:
        file.write(
            """
%0 = "tosa.const"() : () -> tensor<256xi8> loc(#loc1)
%1 = "tosa.const"() : () -> tensor<3xi8> loc(#loc1)
%2 = "tosa.const"() : () -> tensor<3xi32> loc(#loc1)
%3 = "tosa.const"() : () -> tensor<1xi8> loc(#loc2)
#loc1 = loc("layer0")
        """
        )

    id_to_tosa_ops = read_tosa_mlir_ops(tosa_mlir_file)
    assert len(id_to_tosa_ops) == 4

    known_locations = [id_to_tosa_ops[op_id].loc for op_id in [0, 1, 2]]
    missing_location = id_to_tosa_ops[3]

    assert all(loc == '"layer0"' for loc in known_locations)
    assert missing_location.loc == ""


def test_read_tosa_flatbuffer_model(
    test_tosa_flatbuffer_model_with_length: tuple[Path, int]
) -> None:
    """Tests TOSA flatbuffer file parser."""
    if not tosa_flatbuffers_available():
        pytest.skip("Tosa Flatbuffers backend not available")
    model_path, expected_len = test_tosa_flatbuffer_model_with_length

    id_to_tosa_ops = read_tosa_flatbuffer_ops(model_path)
    _check_id_to_tosa_ops(id_to_tosa_ops, expected_len)


def test_tosa_flatbuffers_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test if BackendNotAvailable is raised with tosa-flatbuffers is not installed"""
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.tosa_reader.tosa_flatbuffers_available",
        MagicMock(return_value=False),
    )

    with pytest.raises(
        BackendUnavailableError, match="Tosa Flatbuffers backend not available"
    ):
        _ = read_tosa_flatbuffer_ops(Path("some_file.tosa"))
