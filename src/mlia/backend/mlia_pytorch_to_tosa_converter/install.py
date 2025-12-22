# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the TOSA Converter For Pytorch."""
from __future__ import annotations

import subprocess  # nosec
import sys

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import Installation
from mlia.backend.install import InstallationType
from mlia.backend.install import PyPackageBackendInstallation
from mlia.core.errors import InternalError
from mlia.utils.download import DownloadConfig


class PyTorchCPUBackendInstallation(PyPackageBackendInstallation):
    """Custom backend installation that uses CPU-only index.

    Due to large size of GPU versions of torch and related packages,
    this installation first installs CPU-only versions of dependencies,
    then proceeds with the normal installation of the TOSA serialization library.
    Uses regular PYPI index for dependencies that cant be found in the CPU-only index.
    """

    def install(self, install_type: InstallationType) -> None:
        """Install the backend with CPU-only dependencies."""
        if not self.supports(install_type):
            raise ValueError(
                f"Insufficient configuration for installation type {install_type}."
            )

        try:
            subprocess.check_output(  # nosec
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    "install",
                    "torch",
                    "executorch",
                    "torchao",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                    "--extra-index-url",
                    "https://pypi.org/simple",
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as err:
            raise InternalError(
                f"Unable to install executorch and torchao: {err.output}"
            ) from err

        # Then proceed with normal installation for the tosa wheel
        super().install(install_type)


def get_mlia_pytorch_to_tosa_backend_installation() -> Installation:
    """Get MLIA pytorch to TOSA backend whl."""
    return PyTorchCPUBackendInstallation(
        name="mlia-pytorch-to-tosa-converter",
        description="Tool to serialize and deserialize TOSA files",
        download_config=DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/tosa-tools/tosa_serialization_lib-0.0.0-cp310-cp310-linux_x86_64.whl"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "82dee9b2695c23ee309837a6f74d5ac363da032e629a480377f41ce496e94d3e"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        packages_to_install=[],
        packages_to_uninstall=[
            "tosa_serialization_lib",
            "torch",
            "executorch",
            "torchao",
        ],
        expected_packages=["tosa_serialization_lib", "executorch"],
        vendor_path="tosa-tools",
    )
