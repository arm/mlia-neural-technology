# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the installation of ML SDK Model Converter."""

from __future__ import annotations


from mlia.backend.install import BackendInstallation, PackagePathChecker


def get_ml_sdk_model_converter_installation() -> BackendInstallation:
    """Get all information to install ML SDK Model Converter."""
    ml_sdk_model_converter_installation = BackendInstallation(
        name="ml-sdk-model-converter",
        description="ML SDK Model Converter",
        fvp_dir_name="ml-sdk-model-converter",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "model-converter",
            ],
        ),
        backend_installer=None,
        dependencies=[],
        vendor_path="ml-sdk-model-converter",
    )

    return ml_sdk_model_converter_installation
