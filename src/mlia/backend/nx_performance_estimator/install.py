# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the installation of the Neural Accelerator Performance Estimator."""

from __future__ import annotations

import sys
from pathlib import Path

from mlia.backend.install import BackendInstallation, EulaDocument, PackagePathChecker
from mlia.backend.nx_performance_estimator.installer import (
    DEFAULT_LICENSE_FILE,
    InstallerError,
    InstallerOptions,
    install_package,
)


def _resolve_nx_eula(_source: Path | None) -> list[EulaDocument]:
    """Read the separately packaged NX Performance Estimator EULA."""
    try:
        text = DEFAULT_LICENSE_FILE.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as err:
        raise RuntimeError(
            "Unable to read the NX Performance Estimator licence terms from "
            f"'{DEFAULT_LICENSE_FILE}'."
        ) from err
    if not text.lstrip("\ufeff").strip():
        raise RuntimeError(
            "The NX Performance Estimator licence terms are empty at "
            f"'{DEFAULT_LICENSE_FILE}'."
        )
    return [EulaDocument("Arm End User License Agreement", text)]


class NXPerformanceEstimatorInstaller:
    """Install an extracted NX performance estimator package."""

    def __call__(self, eula_agreement: bool, dist_dir: Path) -> Path:
        """Install the package payload and return its backend directory."""
        destination = dist_dir / "nx-performance-estimator"
        options = InstallerOptions(
            destination=destination,
            license_file=DEFAULT_LICENSE_FILE,
            eula_agreement=eula_agreement,
            interactive=not eula_agreement,
            force=True,
            quiet=True,
            prompt_to_proceed=False,
        )
        try:
            return install_package(dist_dir, options)
        except InstallerError as err:
            raise RuntimeError(
                "Unable to install the Neural Accelerator Performance Estimator."
            ) from err


def get_nx_performance_estimator_installation() -> BackendInstallation:
    """Get all information to install the Neural Accelerator Performance Estimator."""
    nx_performance_estimator_installation = BackendInstallation(
        name="nx-performance-estimator",
        description="Neural Accelerator Performance Estimator",
        fvp_dir_name="nx-performance-estimator",
        download_config=None,
        supported_platforms=["Linux", "Windows"],
        path_checker=PackagePathChecker(
            expected_files=[
                "graph-compiler-performance-estimator.exe"
                if sys.platform == "win32"
                else "graph-compiler-performance-estimator",
            ],
        ),
        backend_installer=NXPerformanceEstimatorInstaller(),
        vendor_path="nx-performance-estimator",
        requires_eula=True,
        eula_resolver=_resolve_nx_eula,
    )

    return nx_performance_estimator_installation
