# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the platform-independent NX backend package installer."""

from __future__ import annotations

import io
import stat
import sys
from pathlib import Path

import pytest

from mlia.backend.nx_performance_estimator.installer import (
    DEFAULT_LICENSE_FILE,
    InvalidDestinationError,
    InstallerOptions,
    InvalidPackageError,
    LicenseNotAcceptedError,
    UserCancelledError,
    install_package,
)
from mlia.backend.nx_performance_estimator.install import (
    NXPerformanceEstimatorInstaller,
)


def _executable() -> str:
    """Return the executable name for the test platform."""
    return (
        "graph-compiler-performance-estimator.exe"
        if sys.platform == "win32"
        else "graph-compiler-performance-estimator"
    )


def _create_package(root: Path) -> Path:
    package_dir = root / "package"
    package_dir.mkdir(parents=True)
    license_file = root / "license_terms" / "license_agreement.txt"
    license_file.parent.mkdir()
    license_file.write_text("Example licence terms\n", encoding="utf-8")
    executable = package_dir / _executable()
    executable.write_text("estimator", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    (package_dir / "additional-member.dat").write_text("extra", encoding="utf-8")
    return package_dir


def _license_file(package_dir: Path) -> Path:
    """Return the separately vendored test EULA."""
    return package_dir.parent / "license_terms" / "license_agreement.txt"


def test_install_package_prompts_for_eula(tmp_path: Path) -> None:
    """Display the external EULA and install after explicit agreement."""
    package_dir = _create_package(tmp_path)
    destination = tmp_path / "installed"
    output = io.StringIO()
    responses = iter(["yes"])

    result = install_package(
        package_dir,
        InstallerOptions(
            destination=destination,
            license_file=_license_file(package_dir),
            prompt_to_proceed=False,
        ),
        input_fn=lambda _prompt: next(responses),
        output=output,
        pager=lambda text: output.write(text),
    )

    assert result == destination
    assert "Example licence terms" in output.getvalue()
    assert (destination / _executable()).is_file()
    if sys.platform != "win32":
        assert (destination / _executable()).stat().st_mode & 0o111
    assert not (destination / "additional-member.dat").exists()


def test_install_package_preaccepted_eula_is_noninteractive(tmp_path: Path) -> None:
    """A caller-provided agreement should skip EULA display and prompting."""
    package_dir = _create_package(tmp_path)
    destination = tmp_path / "installed"
    output = io.StringIO()

    install_package(
        package_dir,
        InstallerOptions(
            destination=destination,
            license_file=_license_file(package_dir),
            eula_agreement=True,
            interactive=False,
            force=True,
            quiet=True,
            prompt_to_proceed=False,
        ),
        input_fn=lambda _prompt: pytest.fail("unexpected prompt"),
        output=output,
        pager=lambda _text: pytest.fail("unexpected EULA display"),
    )

    assert output.getvalue() == ""
    assert destination.is_dir()


@pytest.mark.parametrize("response", ["no", "quit"])
def test_install_package_rejected_eula_changes_nothing(
    tmp_path: Path, response: str
) -> None:
    """Rejecting the EULA must leave no partial installation."""
    package_dir = _create_package(tmp_path)
    destination = tmp_path / "installed"

    with pytest.raises(LicenseNotAcceptedError):
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                prompt_to_proceed=False,
            ),
            input_fn=lambda _prompt: response,
            output=io.StringIO(),
            pager=lambda _text: None,
        )

    assert not destination.exists()


def test_install_package_noninteractive_requires_preacceptance(tmp_path: Path) -> None:
    """Non-interactive installation must not infer EULA acceptance."""
    package_dir = _create_package(tmp_path)
    destination = tmp_path / "installed"
    output = io.StringIO()

    with pytest.raises(LicenseNotAcceptedError):
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                interactive=False,
                prompt_to_proceed=False,
            ),
            output=output,
            pager=lambda text: output.write(text),
        )

    assert "Example licence terms" in output.getvalue()
    assert not destination.exists()


def test_install_package_requires_force_to_replace_destination(tmp_path: Path) -> None:
    """Do not overwrite an installation without force or confirmation."""
    package_dir = _create_package(tmp_path)
    destination = tmp_path / "installed"
    destination.mkdir()
    marker = destination / "old"
    marker.write_text("old", encoding="utf-8")

    with pytest.raises(UserCancelledError):
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                eula_agreement=True,
                interactive=False,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )

    assert marker.is_file()

    install_package(
        package_dir,
        InstallerOptions(
            destination=destination,
            license_file=_license_file(package_dir),
            eula_agreement=True,
            interactive=False,
            force=True,
            prompt_to_proceed=False,
        ),
        output=io.StringIO(),
    )

    assert not marker.exists()
    assert (destination / _executable()).is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks only")
def test_install_package_rejects_symlink_destination(tmp_path: Path) -> None:
    """Do not replace the target of a destination symlink."""
    package_dir = _create_package(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("keep", encoding="utf-8")
    destination = tmp_path / "installed"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(InvalidDestinationError, match="symbolic link"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                eula_agreement=True,
                force=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )

    assert destination.is_symlink()
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks only")
def test_install_package_rejects_dangling_symlink_destination(
    tmp_path: Path,
) -> None:
    """Reject a dangling destination symlink without replacing it."""
    package_dir = _create_package(tmp_path)
    destination = tmp_path / "installed"
    destination.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(InvalidDestinationError, match="symbolic link"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                eula_agreement=True,
                force=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )

    assert destination.is_symlink()
    assert not destination.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks only")
def test_install_package_allows_symlinked_destination_parent(tmp_path: Path) -> None:
    """Resolve a symlinked parent while preserving the final destination name."""
    package_dir = _create_package(tmp_path)
    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    result = install_package(
        package_dir,
        InstallerOptions(
            destination=linked_parent / "installed",
            license_file=_license_file(package_dir),
            eula_agreement=True,
            force=True,
            prompt_to_proceed=False,
        ),
        output=io.StringIO(),
    )

    assert result == actual_parent / "installed"
    assert (result / _executable()).is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks only")
def test_install_package_rechecks_destination_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a destination changed into a symlink after initial validation."""
    package_dir = _create_package(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("keep", encoding="utf-8")
    destination = tmp_path / "installed"

    def replace_destination_with_symlink(
        checked_destination: Path,
        _options: InstallerOptions,
        _input_fn: object,
    ) -> None:
        checked_destination.symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.installer.core._confirm_installation",
        replace_destination_with_symlink,
    )

    with pytest.raises(InvalidDestinationError, match="symbolic link"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                eula_agreement=True,
                force=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )

    assert destination.is_symlink()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_install_package_lists_installed_files(tmp_path: Path) -> None:
    """List installed paths relative to the destination when requested."""
    package_dir = _create_package(tmp_path)
    output = io.StringIO()

    install_package(
        package_dir,
        InstallerOptions(
            destination=tmp_path / "installed",
            license_file=_license_file(package_dir),
            eula_agreement=True,
            interactive=False,
            force=True,
            quiet=True,
            show_files=True,
            prompt_to_proceed=False,
        ),
        output=output,
    )

    assert output.getvalue().splitlines() == [_executable()]


def test_install_package_validates_layout_before_prompting(tmp_path: Path) -> None:
    """Reject malformed packages before obtaining user agreement."""
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    license_file = tmp_path / "license_agreement.txt"
    license_file.write_text("terms", encoding="utf-8")

    with pytest.raises(InvalidPackageError, match="graph-compiler"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=tmp_path / "installed",
                license_file=license_file,
            ),
            input_fn=lambda _prompt: pytest.fail("unexpected prompt"),
            output=io.StringIO(),
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_install_package_rejects_non_executable_estimator(tmp_path: Path) -> None:
    """Reject a POSIX estimator that has no execute permission bits."""
    package_dir = _create_package(tmp_path)
    executable = package_dir / _executable()
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    destination = tmp_path / "installed"

    with pytest.raises(InvalidPackageError, match="not executable"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                eula_agreement=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )

    assert not destination.exists()


def test_install_package_rejects_empty_estimator(tmp_path: Path) -> None:
    """Reject a zero-byte estimator executable before installation."""
    package_dir = _create_package(tmp_path)
    (package_dir / _executable()).write_bytes(b"")
    destination = tmp_path / "installed"

    with pytest.raises(InvalidPackageError, match="executable is empty") as error_info:
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                eula_agreement=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )

    assert error_info.value.exit_code == 8
    assert not destination.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks only")
def test_install_package_rejects_symlinked_estimator(tmp_path: Path) -> None:
    """Reject an estimator executable supplied through a symbolic link."""
    package_dir = _create_package(tmp_path)
    estimator_executable = package_dir / _executable()
    estimator_executable.unlink()
    target = package_dir / "estimator-target"
    target.write_text("estimator", encoding="utf-8")
    estimator_executable.symlink_to(target)
    destination = tmp_path / "installed"

    with pytest.raises(InvalidPackageError, match="Missing graph-compiler"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                eula_agreement=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )

    assert not destination.exists()


@pytest.mark.parametrize("license_state", ["missing", "empty"])
def test_install_package_requires_valid_external_eula(
    tmp_path: Path, license_state: str
) -> None:
    """Do not install without meaningful external licence terms."""
    package_dir = _create_package(tmp_path)
    license_file = _license_file(package_dir)
    if license_state == "missing":
        license_file.unlink()
    else:
        license_file.write_text("", encoding="utf-8")

    with pytest.raises(InvalidPackageError, match="licence file"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=tmp_path / "installed",
                license_file=license_file,
            ),
            input_fn=lambda _prompt: pytest.fail("unexpected prompt"),
            output=io.StringIO(),
        )


def test_install_package_rejects_non_utf8_eula(tmp_path: Path) -> None:
    """Reject an EULA that is not valid UTF-8."""
    package_dir = _create_package(tmp_path)
    license_file = _license_file(package_dir)
    license_file.write_bytes(b"\xff")

    with pytest.raises(InvalidPackageError, match="licence file"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=tmp_path / "installed",
                license_file=license_file,
                eula_agreement=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )


def test_install_package_rejects_unreadable_eula(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an EULA that cannot be read."""
    package_dir = _create_package(tmp_path)
    license_file = _license_file(package_dir)
    original_read_text = Path.read_text

    def fail_for_license(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == license_file:
            raise PermissionError("not readable")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fail_for_license)

    with pytest.raises(InvalidPackageError, match="licence file"):
        install_package(
            package_dir,
            InstallerOptions(
                destination=tmp_path / "installed",
                license_file=license_file,
                eula_agreement=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )


def test_install_package_translates_staging_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Translate temporary-directory failures without cleanup errors."""
    package_dir = _create_package(tmp_path)
    destination = tmp_path / "installed"

    def fail_to_create_staging(**_kwargs: object) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.installer.core.tempfile.mkdtemp",
        fail_to_create_staging,
    )

    with pytest.raises(
        InvalidDestinationError, match="Unable to install"
    ) as error_info:
        install_package(
            package_dir,
            InstallerOptions(
                destination=destination,
                license_file=_license_file(package_dir),
                eula_agreement=True,
                force=True,
                prompt_to_proceed=False,
            ),
            output=io.StringIO(),
        )

    assert error_info.value.exit_code == 4
    assert not destination.exists()


@pytest.mark.parametrize("eula_agreement", [False, True])
def test_backend_adapter_calls_python_installer_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eula_agreement: bool,
) -> None:
    """Translate MLIA's installer callback into Python installer options."""
    captured: dict[str, object] = {}

    def fake_install(
        package_dir: Path,
        options: InstallerOptions,
    ) -> Path:
        captured["package_dir"] = package_dir
        captured["options"] = options
        return options.destination

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.install.install_package",
        fake_install,
    )

    result = NXPerformanceEstimatorInstaller()(eula_agreement, tmp_path)

    assert result == tmp_path / "nx-performance-estimator"
    assert captured["package_dir"] == tmp_path
    options = captured["options"]
    assert isinstance(options, InstallerOptions)
    assert options.eula_agreement is eula_agreement
    assert options.interactive is (not eula_agreement)
    assert options.force is True
    assert options.prompt_to_proceed is False
    assert options.license_file == DEFAULT_LICENSE_FILE
