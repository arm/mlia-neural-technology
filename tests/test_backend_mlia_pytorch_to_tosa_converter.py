# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to TOSA converter backend."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mlia.backend.install import PyPackageBackendInstallation
from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import (
    MliaPytorchToTosaConverter,
)
from mlia.backend.mlia_pytorch_to_tosa_converter.install import (
    get_mlia_pytorch_to_tosa_backend_installation,
)
from mlia.backend.registry import registry


def test_backend_registered() -> None:
    """Test backend is registered with correct name."""
    assert "mlia-pytorch-to-tosa-converter" in registry.items
    assert (
        registry.pretty_name("mlia-pytorch-to-tosa-converter")
        == "TOSA converter for pytorch"
    )


def test_installation_configured() -> None:
    """Test installation is properly configured."""
    installation = get_mlia_pytorch_to_tosa_backend_installation()
    assert isinstance(installation, PyPackageBackendInstallation)
    assert installation.name == "mlia-pytorch-to-tosa-converter"
    assert installation.could_be_installed
    assert installation.download_config is not None


def test_converter_validates_inputs() -> None:
    """Test converter validates input file and output directory."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Invalid file format
        txt_file = Path(tmpdir) / "model.txt"
        txt_file.write_text("test")
        with pytest.raises(ValueError, match="Only .pt2 files are supported"):
            # pylint: disable=protected-access
            converter._load_pytorch_model(txt_file)
            # pylint: enable=protected-access

        # Nonexistent output directory
        pt2_file = Path(tmpdir) / "model.pt2"
        pt2_file.write_text("test")
        with pytest.raises(NotADirectoryError):
            converter(pt2_file, Path(tmpdir) / "nonexistent")
