# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for missing converter plugins."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.ml_sdk_model_converter import install as model_converter_install
from mlia.backend.ml_sdk_model_converter.conversion import (
    MLSDKModelConverterBase,
    run_named_converter,
)
from mlia.core.errors import ConfigurationError
from mlia.transformers.error import TransformerNotFoundError
from mlia.transformers.registry import TransformRequest


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


def test_get_ml_sdk_model_converter_path_uses_path_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PATH lookup should be preferred when model-converter is available."""
    converter_path = tmp_path / "model-converter"
    converter_path.touch()
    monkeypatch.setattr(
        model_converter_install.shutil,
        "which",
        lambda _name: str(converter_path),
    )

    assert model_converter_install.get_ml_sdk_model_converter_path() == tmp_path


def test_get_ml_sdk_model_converter_path_uses_scripts_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed console scripts may live in the Python scripts directory."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "model-converter").touch()
    monkeypatch.setattr(model_converter_install.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        model_converter_install.sysconfig,
        "get_path",
        lambda _name: str(scripts_dir),
    )

    assert model_converter_install.get_ml_sdk_model_converter_path() == scripts_dir


def test_get_ml_sdk_model_converter_path_returns_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing model-converter executable should report no installation path."""
    python_path = tmp_path / "bin" / "python"
    python_path.parent.mkdir()
    python_path.touch()
    monkeypatch.setattr(model_converter_install.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        model_converter_install.sysconfig,
        "get_path",
        lambda _name: None,
    )
    monkeypatch.setattr(model_converter_install.sys, "executable", str(python_path))

    assert model_converter_install.get_ml_sdk_model_converter_path() is None


def test_run_front_end_raises_when_tflite_converter_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = MLSDKModelConverterBase(tmp_path)
    input_model = tmp_path / "model.tflite"
    input_model.touch()

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        MagicMock(side_effect=TransformerNotFoundError("missing")),
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
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        MagicMock(side_effect=TransformerNotFoundError("missing")),
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
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        MagicMock(side_effect=TransformerNotFoundError("missing")),
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
    captured: dict[str, TransformRequest] = {}

    def fake_transform_model(req: TransformRequest) -> Path:
        captured["request"] = req
        delegate_output = req.output_dir / f"{input_model.stem}{delegate_suffix}"
        delegate_output.touch()
        return delegate_output

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        fake_transform_model,
    )

    assert converter(input_model, tmp_path) == vgf_path
    assert vgf_path.is_file()
    assert captured["request"] == TransformRequest(
        model=input_model,
        output_dir=tmp_path,
        target_format="delegate",
        transform_options={},
    )


def test_tflite_front_end_requests_mlir_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TFLite performance conversion should keep bytecode and text debug output."""
    model = tmp_path / "model.tflite"
    model.touch()
    bytecode_path = tmp_path / "model.tosa.mlirbc"
    text_path = tmp_path / "model.tosamlir"
    captured: list[dict[str, Any]] = []
    converter = MLSDKModelConverterBase(tmp_path)

    def fake_run_named_converter(
        name: str,
        model_file: Path,
        output_dir: Path,
        *,
        output_format: str | None = None,
        emit_debug_info: bool | None = None,
    ) -> Path:
        captured.append(
            {
                "name": name,
                "model_file": model_file,
                "output_dir": output_dir,
                "output_format": output_format,
                "emit_debug_info": emit_debug_info,
            }
        )
        output_path = bytecode_path if output_format == "mlir-bytecode" else text_path
        output_path.touch()
        return output_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.run_named_converter",
        fake_run_named_converter,
    )

    result = converter.run_front_end(model, tmp_path)

    assert result == bytecode_path
    assert captured == [
        {
            "name": "tflite_to_tosa",
            "model_file": model,
            "output_dir": tmp_path,
            "output_format": "mlir-bytecode",
            "emit_debug_info": True,
        },
        {
            "name": "tflite_to_tosa",
            "model_file": model,
            "output_dir": tmp_path,
            "output_format": "mlir-text",
            "emit_debug_info": True,
        },
    ]


def test_run_named_converter_forwards_enable_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_path = tmp_path / "supported.tosa"
    captured: dict[str, TransformRequest] = {}

    def fake_transform_model(req: TransformRequest) -> Path:
        captured["request"] = req
        return expected_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        fake_transform_model,
    )

    model = tmp_path / "model.pt2"
    result = run_named_converter(
        "pt2_to_tosa",
        model,
        tmp_path,
        enable_quantization=False,
    )

    assert result == expected_path
    assert captured["request"] == TransformRequest(
        model=model,
        output_dir=tmp_path,
        target_format="tosa",
        transform_options={"enable_quantization": False},
    )


def test_run_named_converter_maps_unknown_converter_to_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transform_model_mock = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        transform_model_mock,
    )

    with pytest.raises(
        ConfigurationError, match="Converter 'unknown' is not available"
    ):
        run_named_converter("unknown", tmp_path / "model.tflite", tmp_path)
    transform_model_mock.assert_not_called()


def test_run_named_converter_forwards_output_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Converter output format should be forwarded to transformer plugins."""
    expected_path = tmp_path / "model.tosa.mlirbc"
    captured: dict[str, TransformRequest] = {}

    def fake_transform_model(req: TransformRequest) -> Path:
        captured["request"] = req
        return expected_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        fake_transform_model,
    )

    model = tmp_path / "model.tflite"
    result = run_named_converter(
        "tflite_to_tosa",
        model,
        tmp_path,
        output_format="mlir-bytecode",
    )

    assert result == expected_path
    assert captured["request"] == TransformRequest(
        model=model,
        output_dir=tmp_path,
        target_format="tosa",
        transform_options={"output_format": "mlir-bytecode"},
    )


def test_run_named_converter_forwards_emit_debug_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Converter debug info should be forwarded to transformer plugins."""
    expected_path = tmp_path / "model.tosamlir"
    captured: dict[str, TransformRequest] = {}

    def fake_transform_model(req: TransformRequest) -> Path:
        captured["request"] = req
        return expected_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        fake_transform_model,
    )

    model = tmp_path / "model.tflite"
    result = run_named_converter(
        "tflite_to_tosa",
        model,
        tmp_path,
        emit_debug_info=True,
    )

    assert result == expected_path
    assert captured["request"] == TransformRequest(
        model=model,
        output_dir=tmp_path,
        target_format="tosa",
        transform_options={"emit_debug_info": True},
    )
