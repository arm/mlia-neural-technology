# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Neural Accelerator Performance Estimator config."""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

from mlia.backend.ml_sdk_model_converter.conversion import MLSDKModelConverter
from mlia.utils.proc import Command


# mypy: disable-error-code=misc
@pytest.fixture(name="ml_sdk_model_converter")
def fixture_ml_sdk_model_converter(
    tmp_path: Path,
) -> Generator[MLSDKModelConverter, None, None]:
    """Create a mock instance of the ML SDK Model Converter for testing."""
    vmc = MLSDKModelConverter(tmp_path / "backend-ml-sdk-model-converter")
    yield vmc


def test_ml_sdk_model_converter_no_output_dir(
    ml_sdk_model_converter: MLSDKModelConverter,
    tmp_path: Path,
) -> None:
    """Test for class MLSDKModelConverter with an invalid output directory."""
    with pytest.raises(NotADirectoryError):
        ml_sdk_model_converter(tmp_path / "model.tflite", tmp_path / "output")


def test_ml_sdk_model_converter_success(
    ml_sdk_model_converter: MLSDKModelConverter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for class MLSDKModelConverter with successful execution."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.MLSDKModelConverter."
        "run_front_end",
        lambda _, __, tosa_file: Path(str(tosa_file)),
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.MLSDKModelConverter."
        "_create_back_end_command",
        lambda _, __, vgf_file: Command(["touch", str(vgf_file)]),
    )

    vgf_file = ml_sdk_model_converter(tmp_path / "model.tflite", output_dir)

    assert vgf_file.is_file()


def test_ml_sdk_model_converter_back_end_fail(
    ml_sdk_model_converter: MLSDKModelConverter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for class MLSDKModelConverter when the back end execution fails."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.MLSDKModelConverter."
        "run_front_end",
        lambda _, __, tosa_file: Path(str(tosa_file)),
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.MLSDKModelConverter."
        "_create_back_end_command",
        MagicMock(
            return_value=Command(
                [
                    "echo",
                    '"Faking a run of ML SDK Model Converter back end..."',
                ]
            )
        ),
    )

    with pytest.raises(FileNotFoundError):
        ml_sdk_model_converter(tmp_path / "model.tflite", output_dir)


def test_ml_sdk_model_converter_tosa_file_not_found(
    ml_sdk_model_converter: MLSDKModelConverter,
    tmp_path: Path,
) -> None:
    """Test for class MLSDKModelConverter throwing a FileNotFoundError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    tosa_file = "model2.tosamlir"
    with pytest.raises(
        FileNotFoundError,
        match="No output from the TosaConverterForTflite frontend found. "
        + f"File [A-Za-z0-9_/-]+{tosa_file} does not exist.",
    ):
        ml_sdk_model_converter(tmp_path / tosa_file, output_dir)


def test_ml_sdk_model_converter_create_back_end_command(
    ml_sdk_model_converter: MLSDKModelConverter,
    tmp_path: Path,
) -> None:
    """Test for function _create_back_end_command of MLSDKModelConverter."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    in_file = tmp_path / "in"
    out_file = tmp_path / "out"
    cmd = ml_sdk_model_converter._create_back_end_command(in_file, out_file)

    assert cmd.cmd
    assert all(isinstance(arg, str) for arg in cmd.cmd)
    assert cmd.cmd[0] == str(
        ml_sdk_model_converter.converter_path / ml_sdk_model_converter.BACK_END_EXE
    )
    assert str(in_file) in cmd.cmd
    assert str(out_file) in cmd.cmd
