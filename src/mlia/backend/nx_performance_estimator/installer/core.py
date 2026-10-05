# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Core behavior for installing an NX estimator payload."""

from __future__ import annotations

import pydoc
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

EXECUTABLE = "graph-compiler-performance-estimator"
WINDOWS_EXECUTABLE = f"{EXECUTABLE}.exe"
DEFAULT_LICENSE_FILE = (
    Path(__file__).parents[3]
    / "_vendor/artifacts/nx-performance-estimator/license_terms/license_agreement.txt"
)


class InstallerError(RuntimeError):
    """Base error raised by the NX package installer."""

    exit_code = 1


class InvalidDestinationError(InstallerError):
    """Installation destination cannot be used."""

    exit_code = 4


class InvalidPackageError(InstallerError):
    """NX estimator package has an invalid structure."""

    exit_code = 8


class LicenseNotAcceptedError(InstallerError):
    """The user did not accept the contained EULA."""

    exit_code = 13


class UserCancelledError(InstallerError):
    """The user cancelled installation outside the EULA prompt."""

    exit_code = 15


@dataclass(frozen=True)
class InstallerOptions:
    """Options controlling NX estimator package installation."""

    destination: Path
    license_file: Path = DEFAULT_LICENSE_FILE
    eula_agreement: bool = False
    interactive: bool = True
    force: bool = False
    quiet: bool = False
    show_files: bool = False
    verbosity: int = 1
    prompt_to_proceed: bool = True


def _platform_executable() -> str:
    """Return the estimator executable expected on the current platform."""
    return WINDOWS_EXECUTABLE if sys.platform == "win32" else EXECUTABLE


def _validate_package(package_dir: Path, license_file: Path) -> tuple[str, Path]:
    """Validate package structure and return its EULA text and executable path."""
    if not package_dir.is_dir():
        raise InvalidPackageError(f"Missing package directory: {package_dir}.")
    try:
        license_text = license_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as err:
        raise InvalidPackageError(
            f"Unable to read licence file as UTF-8: {license_file}."
        ) from err
    if not license_text:
        raise InvalidPackageError(f"Missing or empty licence file: {license_file}.")

    estimator_executable = package_dir / _platform_executable()
    if not estimator_executable.is_file() or estimator_executable.is_symlink():
        raise InvalidPackageError(f"Missing {_platform_executable()} in {package_dir}.")
    executable_metadata = estimator_executable.stat()
    if executable_metadata.st_size == 0:
        raise InvalidPackageError(
            f"Estimator executable is empty: {estimator_executable}."
        )
    if sys.platform != "win32" and executable_metadata.st_mode & 0o111 == 0:
        raise InvalidPackageError(
            f"Estimator is not executable: {estimator_executable}."
        )
    return license_text, estimator_executable


def _ask(
    prompt: str,
    accepted: set[str],
    rejected: set[str],
    input_fn: Callable[[str], str],
) -> bool:
    """Ask a yes/no-style question until a recognized answer is supplied."""
    while True:
        try:
            answer = input_fn(prompt).strip().casefold()
        except (EOFError, StopIteration) as err:
            raise UserCancelledError(
                "Input ended before installation was confirmed."
            ) from err
        if answer in accepted:
            return True
        if answer in rejected:
            return False


def _write(message: str, options: InstallerOptions, output: TextIO) -> None:
    """Write informational output when enabled."""
    if not options.quiet and options.verbosity > 0:
        print(message, file=output)


def _obtain_eula_agreement(
    license_text: str,
    options: InstallerOptions,
    input_fn: Callable[[str], str],
    pager: Callable[[str], None],
) -> None:
    """Display and obtain agreement to the package EULA when needed."""
    if options.eula_agreement:
        return

    pager(license_text)
    if not options.interactive:
        raise LicenseNotAcceptedError(
            "You must agree to the contained EULA to install the product."
        )
    try:
        agreed = _ask(
            "Do you agree to the above terms and conditions? [yes/no]: ",
            {"yes"},
            {"no", "quit"},
            input_fn,
        )
    except UserCancelledError as err:
        raise LicenseNotAcceptedError(
            "Input ended before the contained EULA was accepted."
        ) from err
    if not agreed:
        raise LicenseNotAcceptedError(
            "You must agree to the contained EULA to install the product."
        )


def _confirm_installation(
    destination: Path,
    options: InstallerOptions,
    input_fn: Callable[[str], str],
) -> None:
    """Obtain general proceed and overwrite confirmations."""
    if options.prompt_to_proceed:
        if options.interactive:
            proceed = _ask(
                "Do you want to proceed with the installation? [yes/no, default yes]: ",
                {"", "yes", "y"},
                {"no", "n"},
                input_fn,
            )
            if not proceed:
                raise UserCancelledError("Installation cancelled by user.")

    if destination.exists() and not options.force:
        if not options.interactive:
            raise UserCancelledError(
                f"Destination already exists: {destination}. Use --force to replace it."
            )
        replace = _ask(
            f"Destination '{destination}' exists; replace it? [yes/no]: ",
            {"yes", "y"},
            {"no", "n", "quit", "q"},
            input_fn,
        )
        if not replace:
            raise UserCancelledError("Installation cancelled by user.")


def _reject_symlink_destination(destination: Path) -> None:
    """Reject a final destination component that is a symbolic link."""
    if destination.is_symlink():
        raise InvalidDestinationError(
            f"Installation destination cannot be a symbolic link: {destination}."
        )


def _normalize_destination(destination: Path) -> Path:
    """Resolve a destination's parent without following its final component."""
    _reject_symlink_destination(destination)
    try:
        return destination.parent.resolve() / destination.name
    except (OSError, RuntimeError) as err:
        raise InvalidDestinationError(
            f"Unable to resolve destination parent: {destination.parent}."
        ) from err


def _install_payload(
    estimator_executable: Path,
    destination: Path,
    options: InstallerOptions,
    output: TextIO,
) -> None:
    """Copy the executable through a temporary sibling installation directory."""
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise InvalidDestinationError(
            f"Unable to create destination parent: {parent}."
        ) from err
    if not parent.is_dir():
        raise InvalidDestinationError(
            f"Destination parent is not a directory: {parent}."
        )
    _reject_symlink_destination(destination)

    staging: Path | None = None
    backup = parent / f".{destination.name}-backup-{uuid.uuid4().hex}"
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))

        shutil.copy2(
            estimator_executable,
            staging / estimator_executable.name,
        )
        if destination.exists():
            destination.replace(backup)
        staging.replace(destination)
    except OSError as err:
        if backup.exists() and not destination.exists():
            try:
                backup.replace(destination)
            except OSError:
                pass
        raise InvalidDestinationError(
            f"Unable to install package into {destination}."
        ) from err
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and destination.exists():
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                backup.unlink(missing_ok=True)

    if options.show_files:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                print(path.relative_to(destination), file=output)


def install_package(
    package_dir: Path,
    options: InstallerOptions,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
    pager: Callable[[str], None] | None = None,
) -> Path:
    """Install an NX estimator package and return the destination path."""
    output = sys.stdout if output is None else output
    pager = pydoc.pager if pager is None else pager
    package_dir = package_dir.resolve()
    destination = _normalize_destination(options.destination)
    license_text, estimator_executable = _validate_package(
        package_dir,
        options.license_file.resolve(),
    )

    _confirm_installation(destination, options, input_fn)
    _obtain_eula_agreement(license_text, options, input_fn, pager)
    _write(
        f"Installing Neural Accelerator Performance Estimator to {destination}.",
        options,
        output,
    )
    _install_payload(estimator_executable, destination, options, output)
    _write("Installation completed successfully.", options, output)
    return destination
