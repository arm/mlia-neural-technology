# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Function-based tests for PyTorch file detection utility functions."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from mlia.utils.filesystem import is_pytorch_file
from mlia.utils.filesystem import is_tosa_file
from mlia.utils.filesystem import is_vgf_file


def test_pt2_file_detection() -> None:
    """Test detection of .pt2 files."""
    assert is_pytorch_file("model.pt2") is True
    assert is_pytorch_file("neural_network.pt2") is True
    assert is_pytorch_file("transformer.pt2") is True


def test_path_object_support() -> None:
    """Test that Path objects are supported."""
    assert is_pytorch_file(Path("model.pt2")) is True
    assert is_pytorch_file(Path("neural_network.pt2")) is True


def test_non_pytorch_files() -> None:
    """Test that non-PyTorch files are correctly identified."""
    non_pytorch_files = [
        "model.tflite",
        "model.onnx",
        "model.h5",
        "model.pb",
        "model.pt",  # Different extension
        "model.pth",  # Different extension
        "model.bin",
        "model.txt",
        "model.json",
    ]

    for filename in non_pytorch_files:
        assert (
            is_pytorch_file(filename) is False
        ), f"File {filename} incorrectly identified as PyTorch"


def test_files_without_extension() -> None:
    """Test files without extensions."""
    assert is_pytorch_file("model") is False
    assert is_pytorch_file("pytorch_model") is False
    assert is_pytorch_file("") is False


def test_files_with_multiple_extensions() -> None:
    """Test files with multiple extensions."""
    assert is_pytorch_file("model.backup.pt2") is True
    assert is_pytorch_file("model.v2.pt2") is True
    assert (
        is_pytorch_file("model.pt2.backup") is False
    )  # .backup is the final extension


def test_empty_and_edge_cases() -> None:
    """Test empty strings and edge cases."""
    assert is_pytorch_file("") is False
    assert is_pytorch_file("pt2") is False  # No dot
    assert is_pytorch_file(".") is False
    assert is_pytorch_file("..") is False


def test_special_characters_in_filename() -> None:
    """Test filenames with special characters."""
    special_files = [
        "model-v1.pt2",
        "model_final.pt2",
        "model (1).pt2",
        "model@latest.pt2",
        "my-model#1.pt2",
    ]

    for filename in special_files:
        assert (
            is_pytorch_file(filename) is True
        ), f"File {filename} should be recognized as PyTorch"


def test_unicode_filenames() -> None:
    """Test filenames with Unicode characters."""
    unicode_files = [
        "modèle.pt2",
        "模型.pt2",
        "モデル.pt2",
    ]

    for filename in unicode_files:
        assert (
            is_pytorch_file(filename) is True
        ), f"Unicode file {filename} should be recognized as PyTorch"


@pytest.mark.parametrize(
    "filename",
    [
        "model.pt2",
        "neural_network.pt2",
        "transformer_model_v2.pt2",
        "path/to/model.pt2",
        "/absolute/path/model.pt2",
        "model.checkpoint.pt2",
    ],
)
def test_valid_pytorch_files_parametrized(filename: str) -> None:
    """Test various valid PyTorch file patterns."""
    assert is_pytorch_file(filename) is True


@pytest.mark.parametrize(
    "filename",
    [
        "model.pt",
        "model.pth",
        "model.onnx",
        "model.tflite",
        "model.txt",
        "model",
        "",
        "model.pt2.backup",
    ],
)
def test_invalid_pytorch_files_parametrized(filename: str) -> None:
    """Test various invalid PyTorch file patterns."""
    assert is_pytorch_file(filename) is False


def test_suffix_extraction_behavior() -> None:
    """Test that the function correctly extracts file suffixes."""
    # Test that it uses the last suffix
    assert is_pytorch_file("model.tar.pt2") is True
    assert is_pytorch_file("model.pt2.gz") is False  # .gz is the last suffix

    # Test multiple dots
    assert is_pytorch_file("my.model.file.pt2") is True


def test_pathlib_integration() -> None:
    """Test integration with pathlib.Path objects."""
    # Test with different Path types
    assert is_pytorch_file(Path("model.pt2")) is True

    # Test with complex paths
    complex_path = Path("some") / "nested" / "path" / "model.pt2"
    assert is_pytorch_file(complex_path) is True


def test_function_signature_and_return_type() -> None:
    """Test function signature and return type."""
    sig = inspect.signature(is_pytorch_file)

    # Should have one parameter that accepts str or Path
    assert len(sig.parameters) == 1
    param = list(sig.parameters.values())[0]
    assert param.name == "model"

    # Return type should be bool (as string annotation)
    assert sig.return_annotation in ("bool", bool)


def test_consistency_with_other_file_type_functions() -> None:
    """Test consistency with other file type detection functions."""
    # All should have similar signatures
    assert callable(is_pytorch_file)
    assert callable(is_tosa_file)
    assert callable(is_vgf_file)

    # All should return bool for string input
    test_filename = "test.pt2"
    assert isinstance(is_pytorch_file(test_filename), bool)
    assert isinstance(is_tosa_file(test_filename), bool)
    assert isinstance(is_vgf_file(test_filename), bool)

    # Test mutual exclusivity
    pytorch_file = "model.pt2"
    tosa_file = "model.tosa"
    vgf_file = "model.vgf"

    assert is_pytorch_file(pytorch_file)
    assert not is_tosa_file(pytorch_file)
    assert not is_vgf_file(pytorch_file)

    assert not is_pytorch_file(tosa_file)
    assert is_tosa_file(tosa_file)
    assert not is_vgf_file(tosa_file)

    assert not is_pytorch_file(vgf_file)
    assert not is_tosa_file(vgf_file)
    assert is_vgf_file(vgf_file)
