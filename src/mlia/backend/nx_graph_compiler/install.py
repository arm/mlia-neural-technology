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
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/nx-graph-compiler/latest/graph_compiler_drage_release_pe.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "bc1f4c8b97d9af87991a7bca3efe44f44def0af820e6046b4b82c95c126f4d15"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "graph_compiler_performance_estimator",
                "graph_compiler_performance_estimator-0.1.0",
            ],
            backend_subfolder="graph-compiler",
        ),
        backend_installer=None,
        dependencies=["vulkan-model-converter"],
    )

    return nx_graph_compiler_installation
