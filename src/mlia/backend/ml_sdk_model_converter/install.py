# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the installation of ML SDK Model Converter."""
from __future__ import annotations

from pathlib import Path

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import BackendInstallation
from mlia.backend.install import PackagePathChecker
from mlia.utils.download import DownloadConfig


def get_ml_sdk_model_converter_installation() -> BackendInstallation:
    """Get all information to install ML SDK Model Converter."""
    ml_sdk_model_converter_installation = BackendInstallation(
        name="ml-sdk-model-converter",
        description="ML SDK Model Converter",
        fvp_dir_name="ml-sdk-model-converter",
        download_config=DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/vulkan-model-converter/latest/ml-sdk-model-converter-backend-1.00.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "56d4c226d94e0aaa079bd4ea6d6b0c1951a615506ae9820942d53f9560be1677"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "model-converter",
            ],
        ),
        backend_installer=None,
        dependencies=["tosa-converter-for-tflite", "tosa-flatbuffers"],
        vendor_path=str(Path("ml-sdk-model-converter") / "model-converter"),
    )

    return ml_sdk_model_converter_installation
