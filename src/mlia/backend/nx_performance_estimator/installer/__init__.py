# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Platform-independent installer for NX performance estimator packages."""

from mlia.backend.nx_performance_estimator.installer.core import (
    DEFAULT_LICENSE_FILE,
    InstallerError,
    InstallerOptions,
    InvalidDestinationError,
    InvalidPackageError,
    LicenseNotAcceptedError,
    UserCancelledError,
    install_package,
)

__all__ = [
    "DEFAULT_LICENSE_FILE",
    "InstallerError",
    "InstallerOptions",
    "InvalidDestinationError",
    "InvalidPackageError",
    "LicenseNotAcceptedError",
    "UserCancelledError",
    "install_package",
]
