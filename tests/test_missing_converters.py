# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for missing converter plugins."""

from pathlib import Path
from typing import Any

import pytest

from mlia.backend.ml_sdk_model_converter.conversion import (
    MLSDKModelConverterBase,
    run_named_converter,
)
from mlia.core.errors import ConfigurationError


def test_missing_tflite_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = MLSDKModelConverterBase(tmp_path)

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.load_converter_plugins",
        lambda registry: None,
    )

    with pytest.raises(ConfigurationError, match="mlia-converters-tflite"):
        converter.run_front_end(Path("model.tflite"), tmp_path)


def test_missing_pt2_converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converter = MLSDKModelConverterBase(tmp_path)

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.load_converter_plugins",
        lambda registry: None,
    )

    with pytest.raises(ConfigurationError, match="mlia-converters-pytorch"):
        converter.run_front_end(Path("model.pt2"), tmp_path)


def test_run_named_converter_passes_enable_quantization_when_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    expected_path = tmp_path / "supported.tosa"

    def fake_converter(
        model_file: Path,
        output_dir: Path,
        *,
        enable_quantization: bool | None = None,
    ) -> Path:
        captured["model_file"] = model_file
        captured["output_dir"] = output_dir
        captured["enable_quantization"] = enable_quantization
        return expected_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion._get_converter",
        lambda _name: fake_converter,
    )

    result = run_named_converter(
        "pt2_to_tosa",
        tmp_path / "model.pt2",
        tmp_path,
        enable_quantization=False,
    )

    assert result == expected_path
    assert captured == {
        "model_file": tmp_path / "model.pt2",
        "output_dir": tmp_path,
        "enable_quantization": False,
    }


def test_run_named_converter_allows_old_pt2_converter_for_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    expected_path = tmp_path / "legacy.tosa"

    def fake_converter(model_file: Path, output_dir: Path) -> Path:
        captured["model_file"] = model_file
        captured["output_dir"] = output_dir
        return expected_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion._get_converter",
        lambda _name: fake_converter,
    )

    result = run_named_converter(
        "pt2_to_tosa",
        tmp_path / "model.pt2",
        tmp_path,
        enable_quantization=True,
    )

    assert result == expected_path
    assert captured == {
        "model_file": tmp_path / "model.pt2",
        "output_dir": tmp_path,
    }


def test_run_named_converter_rejects_old_pt2_converter_for_no_ptq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_converter(model_file: Path, output_dir: Path) -> Path:
        return output_dir / f"{model_file.stem}.tosa"

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion._get_converter",
        lambda _name: fake_converter,
    )

    with pytest.raises(ConfigurationError, match="supports enable_quantization"):
        run_named_converter(
            "pt2_to_tosa",
            tmp_path / "model.pt2",
            tmp_path,
            enable_quantization=False,
        )


def test_run_named_converter_ignores_unsupported_flag_for_non_pt2_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_path = tmp_path / "legacy.tosa"
    captured: dict[str, Any] = {}

    def fake_converter(model_file: Path, output_dir: Path) -> Path:
        captured["model_file"] = model_file
        captured["output_dir"] = output_dir
        return expected_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion._get_converter",
        lambda _name: fake_converter,
    )

    result = run_named_converter(
        "tflite_to_tosa",
        tmp_path / "model.tflite",
        tmp_path,
        enable_quantization=False,
    )

    assert result == expected_path
    assert captured == {
        "model_file": tmp_path / "model.tflite",
        "output_dir": tmp_path,
    }
