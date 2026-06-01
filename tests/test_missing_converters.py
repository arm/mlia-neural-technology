# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for missing converter plugins."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.ml_sdk_model_converter.conversion import (
    MLSDKModelConverterBase,
    run_named_converter,
)
from mlia.core.errors import ConfigurationError


def _write_fake_model_converter(converter_path: Path) -> None:
    backend = converter_path / "model-converter"
    backend.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib\n"
        "import sys\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "output.touch()\n",
        encoding="utf-8",
    )
    backend.chmod(0o755)


def test_run_front_end_raises_when_tflite_converter_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = MLSDKModelConverterBase(tmp_path)
    input_model = tmp_path / "model.tflite"
    input_model.touch()

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.load_converter_plugins",
        lambda registry: None,
    )

    with pytest.raises(ConfigurationError, match="mlia-converters-tflite"):
        converter.run_front_end(input_model, tmp_path)


def test_run_front_end_raises_when_pt2_converter_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = MLSDKModelConverterBase(tmp_path)
    input_model = tmp_path / "model.pt2"
    input_model.touch()

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.load_converter_plugins",
        lambda registry: None,
    )

    with pytest.raises(ConfigurationError, match="mlia-converters-pytorch"):
        converter.run_front_end(input_model, tmp_path)


def test_run_front_end_raises_when_pte_converter_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = MLSDKModelConverterBase(tmp_path)
    input_model = tmp_path / "model.pte"
    input_model.touch()

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.load_converter_plugins",
        lambda registry: None,
    )

    with pytest.raises(ConfigurationError, match="mlia-converters-pytorch"):
        converter.run_front_end(input_model, tmp_path)


def test_run_front_end_raises_when_input_model_missing(tmp_path: Path) -> None:
    converter = MLSDKModelConverterBase(tmp_path)

    with pytest.raises(FileNotFoundError, match="Input model file does not exist"):
        converter.run_front_end(tmp_path / "model.pte", tmp_path)


@pytest.mark.parametrize("delegate_suffix", [".tosa", ".vgf"])
def test_converter_converts_pte_to_vgf(
    delegate_suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fake_model_converter(tmp_path)
    converter = MLSDKModelConverterBase(tmp_path)
    input_model = tmp_path / "model.pte"
    input_model.touch()
    vgf_path = tmp_path / "model.vgf"

    def fake_pte_converter(model_file: Path, output_dir: Path) -> Path:
        delegate_output = output_dir / f"{model_file.stem}{delegate_suffix}"
        delegate_output.touch()
        return delegate_output

    mock_pte_converter = MagicMock(side_effect=fake_pte_converter)

    def load_fake_converter(registry: Any) -> None:
        registry.register("pte_to_delegate", mock_pte_converter)

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.load_converter_plugins",
        load_fake_converter,
    )

    assert converter(input_model, tmp_path) == vgf_path
    assert vgf_path.is_file()
    mock_pte_converter.assert_called_once_with(input_model, tmp_path)


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
