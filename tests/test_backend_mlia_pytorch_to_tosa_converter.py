# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to TOSA converter backend."""
from __future__ import annotations

from pathlib import Path
from subprocess import CalledProcessError  # nosec
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from mlia.backend.install import DownloadAndInstall
from mlia.backend.install import InstallationType
from mlia.backend.install import InstallFromPath
from mlia.backend.install import InstallFromVendorPackage
from mlia.backend.install import PyPackageBackendInstallation
from mlia.backend.mlia_pytorch_to_tosa_converter.install import (
    get_mlia_pytorch_to_tosa_backend_installation,
)
from mlia.backend.mlia_pytorch_to_tosa_converter.install import (
    PyTorchCPUBackendInstallation,
)
from mlia.backend.registry import registry
from mlia.core.errors import InternalError


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
    assert isinstance(installation, PyTorchCPUBackendInstallation)
    assert installation.name == "mlia-pytorch-to-tosa-converter"
    assert installation.could_be_installed
    assert installation.download_config is not None


def test_pytorch_cpu_backend_installation_instance() -> None:
    """Test PyTorchCPUBackendInstallation is a proper subclass."""
    installation = get_mlia_pytorch_to_tosa_backend_installation()
    assert isinstance(installation, PyPackageBackendInstallation)
    assert installation.description == "Tool to serialize and deserialize TOSA files"


@pytest.mark.parametrize(
    "install_type",
    [
        DownloadAndInstall(),
        InstallFromPath(backend_path=Path("/some/path")),
        InstallFromVendorPackage(),
    ],
)
@patch("subprocess.check_output")
@patch("mlia.backend.install.PyPackageBackendInstallation.install")
def test_pytorch_cpu_backend_install_calls_subprocess(
    mock_super_install: Mock, mock_subprocess: Mock, install_type: InstallationType
) -> None:
    """Test custom install method runs CPU-only pip install for all types."""
    installation = get_mlia_pytorch_to_tosa_backend_installation()

    # Mock successful subprocess call
    mock_subprocess.return_value = "Success"

    installation.install(install_type)

    # Verify subprocess was called with correct arguments
    mock_subprocess.assert_called_once()
    call_args = mock_subprocess.call_args[0][0]
    assert "pip" in call_args
    assert "install" in call_args
    assert "torch" in call_args
    assert "executorch" in call_args
    assert "torchao" in call_args
    assert "https://download.pytorch.org/whl/cpu" in call_args
    assert "--index-url" in call_args

    # Verify parent install was called
    mock_super_install.assert_called_once_with(install_type)


@patch("subprocess.check_output")
def test_pytorch_cpu_backend_install_subprocess_failure(mock_subprocess: Mock) -> None:
    """Test install method handles subprocess failures correctly."""
    installation = get_mlia_pytorch_to_tosa_backend_installation()

    mock_subprocess.side_effect = CalledProcessError(
        1, "pip install", output="Installation failed"
    )

    with pytest.raises(InternalError, match="Unable to install executorch and torchao"):
        installation.install(DownloadAndInstall())


def test_pytorch_cpu_backend_install_unsupported_type() -> None:
    """Test install method rejects unsupported installation types."""
    installation = get_mlia_pytorch_to_tosa_backend_installation()

    with patch.object(installation, "supports", return_value=False):
        with pytest.raises(ValueError, match="Insufficient configuration"):
            installation.install(DownloadAndInstall())
