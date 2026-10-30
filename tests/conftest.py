# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest config for MLIA Neural Technology plugin tests."""

# mypy: disable-error-code=misc
import shutil
from pathlib import Path
from typing import Generator

import pytest

TEST_MODEL_H5_FILE = "unsupported_model.h5"
TEST_MODEL_TFLITE_FILE = "placeholder_model.tflite"
TEST_MODEL_TOSA_FILE = "model.tosa"
TEST_MODEL_TOSA_MLIR_FILE = "model.tosa.mlir"
TEST_MODEL_VGF_FILE = "model.vgf"
TEST_MODEL_PTE_FILE = "model.pte"


@pytest.fixture(scope="session", name="test_models_path")
def fixture_test_models_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Provide path to the test models."""
    tmp_path = tmp_path_factory.mktemp("models")

    (tmp_path / TEST_MODEL_H5_FILE).write_text("unsupported h5", encoding="utf-8")
    (tmp_path / TEST_MODEL_TFLITE_FILE).write_text(
        "placeholder tflite", encoding="utf-8"
    )
    (tmp_path / TEST_MODEL_TOSA_FILE).write_text("tosa", encoding="utf-8")
    (tmp_path / TEST_MODEL_TOSA_MLIR_FILE).write_text("tosa", encoding="utf-8")
    (tmp_path / TEST_MODEL_VGF_FILE).write_text("vgf", encoding="utf-8")
    (tmp_path / TEST_MODEL_PTE_FILE).write_bytes(b"pte")

    yield tmp_path

    shutil.rmtree(tmp_path)


@pytest.fixture(scope="session", name="test_resources_path")
def fixture_test_resources_path() -> Path:
    """Return test resources path."""
    return Path(__file__).parent / "test_resources"


@pytest.fixture(scope="session", name="test_h5_model")
def fixture_test_h5_model(test_models_path: Path) -> Path:
    """Return unsupported H5 model placeholder."""
    return test_models_path / TEST_MODEL_H5_FILE


@pytest.fixture(scope="session", name="test_tflite_model")
def fixture_test_tflite_model(test_models_path: Path) -> Path:
    """Return test TFLite model placeholder."""
    return test_models_path / TEST_MODEL_TFLITE_FILE


@pytest.fixture(scope="session", name="test_tosa_mlir_model")
def fixture_test_tosa_mlir_model(test_models_path: Path) -> Path:
    """Return test TOSA model."""
    return test_models_path / TEST_MODEL_TOSA_FILE


@pytest.fixture(scope="session", name="test_vgf_model")
def fixture_test_vgf_model(test_models_path: Path) -> Path:
    """Return test VGF model."""
    return test_models_path / TEST_MODEL_VGF_FILE


@pytest.fixture(scope="session", name="test_pte_model")
def fixture_test_pte_model(test_models_path: Path) -> Path:
    """Return test PTE model."""
    return test_models_path / TEST_MODEL_PTE_FILE
