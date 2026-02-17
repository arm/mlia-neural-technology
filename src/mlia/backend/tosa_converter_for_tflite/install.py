# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the TOSA Converter For Tflite."""

from __future__ import annotations

from mlia.backend.install import (
    Installation,
    PyPackageBackendInstallation,
    artifactory_credential_headers,
)
from mlia.utils.download import DownloadConfig


def get_tosa_converter_for_tflite_backend_installation() -> Installation:
    """Get TOSA converter for tflite backend whl."""
    return PyPackageBackendInstallation(
        name="tosa-converter-for-tflite",
        description="Tool to convert a tflite file to TOSA",
        download_config=DownloadConfig(
            url=(
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/tosa_converter_for_tflite/tosa_converter_for_tflite-2025.11.0.dev0-cp310-cp310-linux_x86_64.whl"
            ),
            sha256_hash=(
                "1732d72b8aa76a4eb8cc38480c0b38b335b38807140b192dcbc5a59d361c04aa"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        packages_to_install=[],
        packages_to_uninstall=["tosa-converter-for-tflite"],
        expected_packages=["tosa-converter-for-tflite"],
        vendor_path="tosa-converter-for-tflite",
    )
