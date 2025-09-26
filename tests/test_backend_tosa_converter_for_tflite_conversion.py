# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Neural Accelerator Graph Compiler config."""
from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

from mlia.backend.tosa_converter_for_tflite.conversion import TosaConverterForTflite
from mlia.utils.proc import Command


# mypy: disable-error-code=misc
@pytest.fixture(name="tosa_converter_for_tflite")
def fixture_tosa_converter_for_tflite() -> (
    Generator[TosaConverterForTflite, None, None]
):
    """Create a mock instance of the ML SDK Model Converter for testing."""
    vmc = TosaConverterForTflite()
    yield vmc


def test_tosa_converter_for_tflite_no_output_dir(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Test for class TosaConverterForTflite with an invalid output directory."""
    with pytest.raises(NotADirectoryError):
        tosa_converter_for_tflite(tmp_path / "model.tflite", tmp_path / "output")


def test_tosa_converter_for_tflite_front_end_fail(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for class TosaConverterForTflite when the front end execution fails."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.TosaConverterForTflite."
        "_create_converter_command",
        MagicMock(
            return_value=Command(
                [
                    "echo",
                    '"Faking a run of ML SDK Model Converter front end..."',
                ]
            )
        ),
    )

    with pytest.raises(FileNotFoundError):
        tosa_converter_for_tflite(tmp_path / "model.tflite", output_dir)


def test_tosa_converter_for_tflite_create_front_end_command(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Test for function _create_front_end_command of TosaConverterForTflite."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    in_file = tmp_path / "in"
    out_file = tmp_path / "out"

    cmd = tosa_converter_for_tflite._create_converter_command(  # pylint: disable=protected-access
        in_file, out_file
    )

    assert cmd.cmd
    assert all(isinstance(arg, str) for arg in cmd.cmd)
    assert str(in_file) in cmd.cmd
    assert str(out_file) in cmd.cmd
