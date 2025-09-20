# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the TOSA Converter For Tflite."""
from __future__ import annotations

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import Installation
from mlia.backend.install import PyPackageBackendInstallation
from mlia.utils.download import DownloadConfig


def get_tosa_converter_for_tflite_backend_installation() -> Installation:
    """Get TOSA converter for tflite backend whl."""
    return PyPackageBackendInstallation(
        name="tosa-converter-for-tflite",
        description="Tool to convert a tflite file to TOSA",
        download_config=DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/tosa_converter_for_tflite/tosa_converter_for_tflite-2025.11.0.dev0-cp39-cp39-linux_x86_64.whl"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "6470f98d50d30c91ac9b8dbc9c6cf67f1262755fbce97f135593088c0b4786fb"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        packages_to_install=[],
        packages_to_uninstall=["tosa-converter-for-tflite"],
        expected_packages=["tosa-converter-for-tflite"],
    )
