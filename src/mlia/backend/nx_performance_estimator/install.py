# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module for the installation of the Neural Accelerator Performance Estimator."""

from __future__ import annotations


from mlia.backend.install import BackendInstallation, PackagePathChecker


def get_nx_performance_estimator_installation() -> BackendInstallation:
    """Get all information to install the Neural Accelerator Performance Estimator."""
    nx_performance_estimator_installation = BackendInstallation(
        name="nx-performance-estimator",
        description="Neural Accelerator Performance Estimator",
        fvp_dir_name="nx-performance-estimator",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "graph-compiler-performance-estimator",
            ],
        ),
        backend_installer=None,
        dependencies=["ml-sdk-model-converter", "tosa-flatbuffers"],
        vendor_path="nx-performance-estimator",
    )

    return nx_performance_estimator_installation
