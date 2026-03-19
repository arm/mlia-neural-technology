# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for Tosa Flatbuffers package installations."""

from __future__ import annotations

from mlia.backend.install import Installation, PyPackageBackendInstallation


def get_tosa_flatbuffers_installation() -> Installation:
    """Get Tosa Flatbuffers installtion."""
    return PyPackageBackendInstallation(
        name="tosa-flatbuffers",
        description="Python API for TOSA flatbuffers.",
        download_config=None,
        packages_to_install=[],  # don't use pypi index
        packages_to_uninstall=["tosa-flatbuffers"],
        expected_packages=["tosa-flatbuffers"],
        vendor_path="tosa-flatbuffers",
    )
