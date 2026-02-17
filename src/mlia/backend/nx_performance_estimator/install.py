# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module for the installation of the Neural Accelerator Performance Estimator."""

from __future__ import annotations

from pathlib import Path

from mlia.backend.install import (
    BackendInstallation,
    PackagePathChecker,
    artifactory_credential_headers,
)
from mlia.utils.download import DownloadConfig


def get_nx_performance_estimator_installation() -> BackendInstallation:
    """Get all information to install the Neural Accelerator Performance Estimator."""
    nx_performance_estimator_installation = BackendInstallation(
        name="nx-performance-estimator",
        description="Neural Accelerator Performance Estimator",
        fvp_dir_name="nx-performance-estimator",
        download_config=DownloadConfig(
            url=(
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/nx-graph-compiler/r55p0_00eac0_mlia_4/graph_compiler_performance_estimator_r55p0_00eac0_mlia_4.tar.gz"  # noqa: E501
            ),
            sha256_hash=(
                "1705a76b4b3175531572004361593b3d8c6924c047026885e19e354d254fc9f1"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "graph-compiler-performance-estimator",
            ],
        ),
        backend_installer=None,
        dependencies=["ml-sdk-model-converter"],
        vendor_path=str(
            Path("nx-performance-estimator") / "graph-compiler-performance-estimator"
        ),
    )

    return nx_performance_estimator_installation
