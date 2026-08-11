# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for missing converter plugins."""

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import call

import pytest

from mlia.backend.ml_sdk_model_converter import install as model_converter_install
from mlia.backend.ml_sdk_model_converter.conversion import (
    MLSDKModelConverterBase,
    build_front_end_transform_request,
    transform_front_end_model,
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

    with pytest.raises(ConfigurationError, match="mlia-converters-litert"):
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

    def fake_pte_converter(model_file: Path, output_dir: Path) -> Path:
        delegate_output = output_dir / f"{model_file.stem}{delegate_suffix}"
        delegate_output.touch()
        return delegate_output

    def fake_transform_model(request: TransformRequest) -> Path:
        captured["request"] = request
        return fake_pte_converter(request.model, request.output_dir)

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


def test_build_front_end_transform_request_passes_disable_quantization_for_pt2_to_tosa(
    tmp_path: Path,
) -> None:
    request = build_front_end_transform_request(
        tmp_path / "model.pt2",
        tmp_path,
        enable_quantization=False,
    )

    assert request == TransformRequest(
        model=tmp_path / "model.pt2",
        output_dir=tmp_path,
        target_format="tosa",
        transform_options={"enable_quantization": False},
    )


def test_build_front_end_transform_request_passes_enable_quantization_for_pt2_to_tosa(
    tmp_path: Path,
) -> None:
    request = build_front_end_transform_request(
        tmp_path / "model.pt2",
        tmp_path,
        enable_quantization=True,
    )

    assert request == TransformRequest(
        model=tmp_path / "model.pt2",
        output_dir=tmp_path,
        target_format="tosa",
        transform_options={"enable_quantization": True},
    )


def test_build_front_end_transform_request_omits_quantization_flag_when_unset_for_pt2_to_tosa(
    tmp_path: Path,
) -> None:
    request = build_front_end_transform_request(
        tmp_path / "model.pt2",
        tmp_path,
        enable_quantization=None,
    )

    assert request == TransformRequest(
        model=tmp_path / "model.pt2",
        output_dir=tmp_path,
        target_format="tosa",
        transform_options={},
    )


def test_build_front_end_transform_request_forwards_tflite_output_options(
    tmp_path: Path,
) -> None:
    request = build_front_end_transform_request(
        tmp_path / "model.tflite",
        tmp_path,
        output_format="mlir-bytecode",
        emit_debug_info=True,
    )

    assert request == TransformRequest(
        tmp_path / "model.tflite",
        tmp_path,
        "tosa",
        {"output_format": "mlir-bytecode", "emit_debug_info": True},
    )


def test_transform_front_end_model_dispatches_transform_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_path = tmp_path / "converted.tosa"
    captured: dict[str, Any] = {}

    def fake_transform_model(request: TransformRequest) -> Path:
        captured["request"] = request
        return expected_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        fake_transform_model,
    )

    result = transform_front_end_model(
        tmp_path / "model.pt2",
        tmp_path,
        enable_quantization=False,
    )

    assert result == expected_path
    assert captured["request"] == TransformRequest(
        model=tmp_path / "model.pt2",
        output_dir=tmp_path,
        target_format="tosa",
        transform_options={"enable_quantization": False},
    )


def test_transform_front_end_model_rejects_missing_pt2_transformer_for_no_ptq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        MagicMock(side_effect=TransformerNotFoundError("missing")),
    )

    with pytest.raises(ConfigurationError, match="mlia-converters-pytorch"):
        transform_front_end_model(
            tmp_path / "model.pt2",
            tmp_path,
            enable_quantization=False,
        )


def test_transform_front_end_model_ignores_unsupported_flag_for_non_pt2_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_path = tmp_path / "legacy.tosa"
    captured: dict[str, Any] = {}

    def fake_transform_model(request: TransformRequest) -> Path:
        captured["request"] = request
        return expected_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        fake_transform_model,
    )

    result = transform_front_end_model(
        tmp_path / "model.tflite",
        tmp_path,
        enable_quantization=False,
    )

    assert result == expected_path
    assert captured["request"] == TransformRequest(
        model=tmp_path / "model.tflite",
        output_dir=tmp_path,
        target_format="tosa",
        transform_options={},
    )


def test_tflite_front_end_requests_mlir_bytecode_and_debug_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.tflite"
    model.touch()
    bytecode_path = tmp_path / "model.tosa.mlirbc"
    text_path = tmp_path / "model.tosamlir"
    captured: list[TransformRequest] = []
    converter = MLSDKModelConverterBase(tmp_path)

    def fake_transform_model(request: TransformRequest) -> Path:
        captured.append(request)
        output_path = (
            bytecode_path
            if request.transform_options.get("output_format") == "mlir-bytecode"
            else text_path
        )
        output_path.touch()
        return output_path

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.transform_model",
        fake_transform_model,
    )

    assert converter.run_front_end(model, tmp_path) == bytecode_path
    assert captured == [
        TransformRequest(
            model=model,
            output_dir=tmp_path,
            target_format="tosa",
            transform_options={
                "output_format": "mlir-bytecode",
                "emit_debug_info": True,
            },
        ),
        TransformRequest(
            model=model,
            output_dir=tmp_path,
            target_format="tosa",
            transform_options={
                "output_format": "mlir-text",
                "emit_debug_info": True,
            },
        ),
    ]


def test_tflite_back_end_prefers_bytecode_and_falls_back_to_sibling_mlir_text(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.tflite"
    model.touch()
    bytecode_path = tmp_path / "model.tosa.mlirbc"
    bytecode_path.touch()
    text_path = tmp_path / "model.tosamlir"
    text_path.touch()
    vgf_path = tmp_path / "model.vgf"
    converter = MLSDKModelConverterBase(tmp_path)
    converter.run_front_end = MagicMock(return_value=bytecode_path)  # type: ignore[method-assign]
    converter.run_back_end = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            subprocess.CalledProcessError(255, ["model-converter"]),
            vgf_path,
        ]
    )

    assert converter(model, tmp_path) == vgf_path
    assert converter.run_back_end.call_args_list == [
        call(bytecode_path, tmp_path),
        call(text_path, tmp_path),
    ]
