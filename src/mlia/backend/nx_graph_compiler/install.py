# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module for the installation of the Neural Accelerator Graph Compiler."""
from __future__ import annotations

from pathlib import Path

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
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/nx-graph-compiler/r55p0_00eac0_mlia_3/Graph_compiler_performance_estimator_r55p0_00eac0_mlia_3.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "2a97bb750d5dd3dd30ab1cfec5f4b75e51081dd9307a967b9d42fcd6639afd8c"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "Graph_compiler_performance_estimator_r55p0_00eac0_mlia_3",
            ],
        ),
        backend_installer=None,
        dependencies=["ml-sdk-model-converter"],
        vendor_path=str(Path("nx-graph-compiler") / "graph-compiler"),
    )

    return nx_graph_compiler_installation
