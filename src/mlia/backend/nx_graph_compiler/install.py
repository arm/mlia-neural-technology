# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module for the installation of the Neural Accelerator Graph Compiler."""
from __future__ import annotations

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import BackendInstallation
from mlia.backend.install import PackagePathChecker
from mlia.utils.download import DownloadConfig


def get_nx_graph_compiler_installation() -> BackendInstallation:
    """Get all information to install the Neural Accelerator Graph Compiler."""
    nx_graph_compiler_installation = BackendInstallation(
        name="nx-graph-compiler",
        description="Neural Accelerator Graph Compiler",
        fvp_dir_name="nx-graph-compiler",
        download_config=DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/nx-graph-compiler/r55p0_00eac0_mlia_2/graph_compiler_performance_estimator_r55p0_00eac0_mlia_2.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "9140585e4a6bee147facdf78f91cc865c23dc127a68913f7fe837e1da4cc4e30"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "graph_compiler_performance_estimator_r55p0_00eac0_mlia_2",
            ],
            backend_subfolder="graph-compiler",
        ),
        backend_installer=None,
        dependencies=["vulkan-model-converter"],
    )

    return nx_graph_compiler_installation
