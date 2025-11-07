# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the TOSA Converter For Pytorch."""
from __future__ import annotations

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import Installation
from mlia.backend.install import PyPackageBackendInstallation
from mlia.utils.download import DownloadConfig


def get_mlia_pytorch_to_tosa_backend_installation() -> Installation:
    """Get MLIA pytorch to TOSA backend whl."""
    return PyPackageBackendInstallation(
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
        packages_to_install=["torch", "executorch", "torchao"],
        packages_to_uninstall=[
            "tosa_serialization_lib",
            "torch",
            "executorch",
            "torchao",
        ],
        expected_packages=["tosa_serialization_lib", "executorch"],
        vendor_path="tosa-tools",
    )
