# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NX filesystem utilities."""

from pathlib import Path

import pytest

from mlia.nx_utils.filesystem import is_tosa_file


@pytest.mark.parametrize(
    "model",
    [
        Path("model.tosa"),
        Path("model.tosamlir"),
        Path("model.tosa.mlirbc"),
    ],
)
def test_is_tosa_file_accepts_tosa_formats(model: Path) -> None:
    assert is_tosa_file(model)


def test_is_tosa_file_rejects_non_tosa_format() -> None:
    assert not is_tosa_file(Path("model.tflite"))
