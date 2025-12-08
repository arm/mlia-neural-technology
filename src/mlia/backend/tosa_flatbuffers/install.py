# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for Tosa Flatbuffers package installations."""
from __future__ import annotations

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import Installation
from mlia.backend.install import PyPackageBackendInstallation
from mlia.utils.download import DownloadConfig


def get_tosa_flatbuffers_installation() -> Installation:
    """Get Tosa Flatbuffers installtion."""
    return PyPackageBackendInstallation(
        name="tosa-flatbuffers",
        description="Python API for TOSA flatbuffers.",
        download_config=DownloadConfig(
            # pylint: disable=line-too-long
            url="https://artifactory.arm.com:443/artifactory/ml-xpk.pypi/tosa-flatbuffers/tosa_flatbuffers-0.2.0-py3-none-any.whl",
            sha256_hash="ae93b849f635d401604dd391dcaa019bcf95201e0a9ceca07bbd570fbaf82f13",
            # pylint: enable=line-too-long
            header_gen_fn=artifactory_credential_headers,
        ),
        packages_to_install=[],  # don't use pypi index
        packages_to_uninstall=["tosa-flatbuffers"],
        expected_packages=["tosa-flatbuffers"],
        vendor_path="tosa-flatbuffers",
    )
