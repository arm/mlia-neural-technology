# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for missing converter plugins."""

from pathlib import Path

import pytest

from mlia.backend.ml_sdk_model_converter.conversion import MLSDKModelConverterBase
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
