# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology package build metadata."""

from __future__ import annotations

import hashlib
import importlib
import platform
import stat
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Generator, cast
from unittest.mock import Mock

import pytest
from packaging import tags

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

PROJECT_ROOT = Path(__file__).parents[1]
PLATFORM_SHA256_FILES = {
    "Linux": ".sha256.linux",
    "Windows": ".sha256.windows",
}


@pytest.fixture()
def hatch_build(monkeypatch: pytest.MonkeyPatch) -> Generator[ModuleType, None, None]:
    """Import hatch_build with local Hatchling shims restored after the test."""
    old_hatch_build = sys.modules.get("hatch_build")
    had_hatch_build = "hatch_build" in sys.modules

    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    for module_name in (
        "hatchling",
        "hatchling.builders",
        "hatchling.builders.hooks",
        "hatchling.builders.hooks.plugin",
        "hatchling.metadata",
        "hatchling.metadata.plugin",
    ):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    interface_module = ModuleType("hatchling.builders.hooks.plugin.interface")
    setattr(interface_module, "BuildHookInterface", object)
    monkeypatch.setitem(
        sys.modules,
        "hatchling.builders.hooks.plugin.interface",
        interface_module,
    )

    metadata_interface_module = ModuleType("hatchling.metadata.plugin.interface")
    setattr(metadata_interface_module, "MetadataHookInterface", object)
    monkeypatch.setitem(
        sys.modules,
        "hatchling.metadata.plugin.interface",
        metadata_interface_module,
    )

    sys.modules.pop("hatch_build", None)
    module = importlib.import_module("hatch_build")

    yield module

    if had_hatch_build:
        sys.modules["hatch_build"] = cast(ModuleType, old_hatch_build)
    else:
        sys.modules.pop("hatch_build", None)


def _read_expected_sha256(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8").strip()
    parts = content.split()
    assert len(parts) >= 2
    return parts[0], parts[1]


def _create_estimator_archive(path: Path, system: str) -> None:
    """Create a platform-specific estimator archive with an extra member."""
    source = path.parent / "archive-source"
    source.mkdir(parents=True, exist_ok=True)
    executable = (
        "graph-compiler-performance-estimator.exe"
        if system == "Windows"
        else "graph-compiler-performance-estimator"
    )
    executable_path = source / executable
    executable_path.write_text("estimator", encoding="utf-8")
    executable_path.chmod(executable_path.stat().st_mode | stat.S_IXUSR)
    (source / "metadata.txt").write_text("metadata", encoding="utf-8")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(executable_path, arcname=executable)
        archive.add(source / "metadata.txt", arcname="metadata.txt")


def test_metadata_hook_update_uses_commit_hash(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """README metadata links should target the current commit."""
    (tmp_path / "README.md").write_text(
        "[Docs](docs.md)\n![Image](image.png)\n[Section](#section)",
        encoding="utf-8",
    )
    (tmp_path / "docs.md").touch()
    (tmp_path / "image.png").touch()
    monkeypatch.setattr(
        hatch_build.subprocess,
        "run",
        Mock(
            side_effect=[
                Mock(stdout=f"{tmp_path}\n"),
                Mock(stdout="0123456789abcdef\n"),
            ]
        ),
    )
    hook = hatch_build.MetadataHook()
    hook.root = str(tmp_path)
    metadata = {"version": "0.1.0"}

    hook.update(metadata)

    assert metadata["readme"] == {
        "content-type": "text/markdown",
        "text": "[Docs](https://github.com/arm/mlia-neural-technology/"
        "blob/0123456789abcdef/docs.md)\n"
        "![Image](https://raw.githubusercontent.com/arm/"
        "mlia-neural-technology/0123456789abcdef/image.png)\n"
        "[Section](https://github.com/arm/mlia-neural-technology/"
        "blob/0123456789abcdef/README.md#section)",
    }


def test_metadata_hook_update_ignores_enclosing_worktree(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """README metadata should not use a revision from a parent worktree."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "README.md").write_text("[Docs](docs.md)", encoding="utf-8")
    (project_root / "docs.md").touch()
    monkeypatch.setattr(
        hatch_build.subprocess,
        "run",
        Mock(return_value=Mock(stdout=f"{tmp_path}\n")),
    )
    hook = hatch_build.MetadataHook()
    hook.root = str(project_root)
    metadata = {"version": "0.12.2"}

    hook.update(metadata)

    assert metadata["readme"] == {
        "content-type": "text/markdown",
        "text": "[Docs](https://github.com/arm/mlia-neural-technology/"
        "blob/v0.12.2/docs.md)",
    }


@pytest.mark.parametrize(
    ("version", "revision"),
    [
        ("0.12.2", "v0.12.2"),
        ("0.1.1.dev25+20a0c98", "20a0c98"),
    ],
)
def test_metadata_hook_update_falls_back_to_version_or_hash(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version: str,
    revision: str,
) -> None:
    """README links should use the version or its hash without Git metadata."""
    (tmp_path / "README.md").write_text("[Docs](docs.md)", encoding="utf-8")
    (tmp_path / "docs.md").touch()
    monkeypatch.setattr(
        hatch_build.subprocess,
        "run",
        Mock(side_effect=hatch_build.subprocess.CalledProcessError(1, "git")),
    )
    hook = hatch_build.MetadataHook()
    hook.root = str(tmp_path)
    metadata = {"version": version}

    hook.update(metadata)

    assert metadata["readme"] == {
        "content-type": "text/markdown",
        "text": f"[Docs](https://github.com/arm/mlia-neural-technology/"
        f"blob/{revision}/docs.md)",
    }


def test_pyproject_registers_readme_metadata_hook() -> None:
    """Building the package should rewrite links in packaged README metadata."""
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["tool"]["hatch"]["metadata"]["hooks"]["custom"] == {
        "path": "hatch_build.py"
    }
    assert "readme" in pyproject["project"]["dynamic"]
    assert "readme" not in pyproject["project"]


def test_wheel_includes_nx_estimator_archive_and_excludes_public_artifacts() -> None:
    """Wheel metadata should vendor only the platform's NX estimator archive."""
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    wheel_target = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

    vendor_dir = PROJECT_ROOT / "src/mlia/_vendor/artifacts/nx-performance-estimator"
    current_platform = platform.system()
    assert current_platform in PLATFORM_SHA256_FILES

    for sha256_filename in PLATFORM_SHA256_FILES.values():
        expected_sha, expected_name = _read_expected_sha256(
            vendor_dir / sha256_filename
        )
        assert expected_sha
        assert expected_name.endswith(".tar.gz")

    _, selected_name = _read_expected_sha256(
        vendor_dir / PLATFORM_SHA256_FILES[current_platform]
    )
    assert (vendor_dir / selected_name).is_file()
    assert (vendor_dir / "license_terms/license_agreement.txt").is_file()
    assert "src/mlia" in wheel_target["only-include"]

    excluded_paths = set(wheel_target["exclude"])
    assert "src/mlia/_vendor/artifacts/ml-sdk-model-converter/**" in excluded_paths
    assert "src/mlia/_vendor/artifacts/tosa-flatbuffers/**" in excluded_paths
    assert all("nx-performance-estimator" not in path for path in excluded_paths)


@pytest.mark.parametrize(
    ("host_platform", "wheel_platform"),
    [
        ("manylinux_2_38_x86_64", "manylinux_2_39_x86_64"),
        ("manylinux_2_39_x86_64", "manylinux_2_39_x86_64"),
        ("manylinux_2_40_x86_64", "manylinux_2_40_x86_64"),
    ],
)
def test_linux_wheel_enforces_glibc_baseline(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    host_platform: str,
    wheel_platform: str,
) -> None:
    """Linux wheels require Ubuntu 24.04's glibc or the newer build baseline."""
    monkeypatch.setattr(hatch_build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        tags,
        "sys_tags",
        lambda: iter(
            [
                tags.Tag("cp312", "cp312", "linux_x86_64"),
                tags.Tag("cp312", "cp312", host_platform),
            ]
        ),
    )
    monkeypatch.setattr(hatch_build, "ARTIFACTS", {})
    monkeypatch.setattr(
        hatch_build, "_build_vgfpy", lambda _root: tmp_path / "vgfpy.so"
    )
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(tmp_path)
    build_data: dict[str, object] = {}

    hook.initialize("0.0.0", build_data)

    assert build_data["tag"] == f"cp312-cp312-{wheel_platform}"


def test_linux_wheel_rejects_non_glibc_builds(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A musl build cannot be advertised as compatible with glibc."""
    monkeypatch.setattr(hatch_build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        tags,
        "sys_tags",
        lambda: iter([tags.Tag("cp312", "cp312", "musllinux_1_2_x86_64")]),
    )
    build_native = Mock()
    monkeypatch.setattr(hatch_build, "_build_vgfpy", build_native)
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(tmp_path)

    with pytest.raises(RuntimeError, match="glibc"):
        hook.initialize("0.0.0", {})

    build_native.assert_not_called()


def test_build_vgfpy_configures_and_builds_pinned_source(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The native helper should configure CMake and return its vgfpy output."""
    output_dir = tmp_path / hatch_build.VGF_BUILD_DIR / "_deps/vgf_library-build/src"
    output_dir.mkdir(parents=True)
    vgfpy = output_dir / (
        f"vgfpy{hatch_build.importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    vgfpy.write_bytes(b"native extension")
    calls: list[list[str]] = []

    def record_run(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)

    monkeypatch.setattr(hatch_build.subprocess, "run", record_run)

    assert hatch_build._build_vgfpy(tmp_path) == vgfpy
    build_root = tmp_path / hatch_build.VGF_BUILD_DIR
    assert calls == [
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build_root),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={hatch_build.sys.executable}",
            f"-DPython3_INCLUDE_DIR={hatch_build.sysconfig.get_path('include')}",
        ],
        [
            "cmake",
            "--build",
            str(build_root),
            "--target",
            "vgfpy",
            "--config",
            "Release",
        ],
    ]


def test_build_hook_force_includes_native_and_estimator_artifacts(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wheel must include vgfpy and the platform's estimator archive."""
    vgfpy = (
        PROJECT_ROOT / f"vgfpy{hatch_build.importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    monkeypatch.setattr(hatch_build, "_build_vgfpy", lambda _root: vgfpy)
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(PROJECT_ROOT)
    build_data: dict[str, object] = {}

    hook.initialize("0.0.0", build_data)

    force_include = build_data["force_include"]
    assert isinstance(force_include, dict)
    assert force_include[vgfpy.name] == vgfpy.name
    assert build_data["pure_python"] is False
    assert build_data["infer_tag"] is True
    vendor_dir = PROJECT_ROOT / "src/mlia/_vendor/artifacts/nx-performance-estimator"
    _, selected_name = _read_expected_sha256(
        vendor_dir / PLATFORM_SHA256_FILES[platform.system()]
    )
    archive = vendor_dir / selected_name
    assert force_include[str(archive.relative_to(PROJECT_ROOT))] == (
        f"mlia/_vendor/artifacts/nx-performance-estimator/{archive.name}"
    )


@pytest.mark.parametrize(
    ("system", "sha256_filename", "archive_name"),
    (
        ("Linux", ".sha256.linux", "estimator-linux.tar.gz"),
        ("Windows", ".sha256.windows", "estimator-windows.tar.gz"),
    ),
)
def test_build_hook_selects_platform_checksum_metadata(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    system: str,
    sha256_filename: str,
    archive_name: str,
) -> None:
    """The runtime hook should select and validate metadata for its platform."""
    vendor_dir = tmp_path / "src/mlia/_vendor/artifacts/nx-performance-estimator"
    vendor_dir.mkdir(parents=True)
    archive = vendor_dir / archive_name
    _create_estimator_archive(archive, system)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (vendor_dir / sha256_filename).write_text(
        f"{digest}  {archive_name}\n", encoding="utf-8"
    )

    vgfpy = tmp_path / f"vgfpy{hatch_build.importlib.machinery.EXTENSION_SUFFIXES[0]}"
    monkeypatch.setattr(hatch_build, "_build_vgfpy", lambda _root: vgfpy)
    monkeypatch.setattr(hatch_build.platform, "system", lambda: system)
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(tmp_path)
    build_data: dict[str, object] = {}

    hook.initialize("0.0.0", build_data)

    force_include = build_data["force_include"]
    assert isinstance(force_include, dict)
    assert force_include[str(archive.relative_to(tmp_path))] == (
        f"mlia/_vendor/artifacts/nx-performance-estimator/{archive_name}"
    )


def test_build_hook_rejects_estimator_archive_without_root_executable(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail a build whose estimator executable is not at the archive root."""
    vendor_dir = tmp_path / "src/mlia/_vendor/artifacts/nx-performance-estimator"
    vendor_dir.mkdir(parents=True)
    archive = vendor_dir / "estimator-linux.tar.gz"
    payload = tmp_path / "graph-compiler-performance-estimator"
    payload.write_text("estimator", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="payload/graph-compiler-performance-estimator")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (vendor_dir / ".sha256.linux").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        hatch_build, "_build_vgfpy", lambda _root: tmp_path / "vgfpy.so"
    )
    monkeypatch.setattr(hatch_build.platform, "system", lambda: "Linux")
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(tmp_path)

    with pytest.raises(RuntimeError, match="graph-compiler-performance-estimator"):
        hook.initialize("0.0.0", {})


def test_build_hook_rejects_empty_estimator_executable(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail a build whose root-level estimator executable is empty."""
    vendor_dir = tmp_path / "src/mlia/_vendor/artifacts/nx-performance-estimator"
    vendor_dir.mkdir(parents=True)
    archive = vendor_dir / "estimator-linux.tar.gz"
    executable = tmp_path / "graph-compiler-performance-estimator"
    executable.touch()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(executable, arcname=executable.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (vendor_dir / ".sha256.linux").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        hatch_build, "_build_vgfpy", lambda _root: tmp_path / "vgfpy.so"
    )
    monkeypatch.setattr(hatch_build.platform, "system", lambda: "Linux")
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(tmp_path)

    with pytest.raises(RuntimeError, match="graph-compiler-performance-estimator"):
        hook.initialize("0.0.0", {})


def test_build_hook_rejects_non_executable_estimator(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail a Linux build whose estimator has no execute permission bits."""
    vendor_dir = tmp_path / "src/mlia/_vendor/artifacts/nx-performance-estimator"
    vendor_dir.mkdir(parents=True)
    archive = vendor_dir / "estimator-linux.tar.gz"
    executable = tmp_path / "graph-compiler-performance-estimator"
    executable.write_text("estimator", encoding="utf-8")
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(executable, arcname=executable.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (vendor_dir / ".sha256.linux").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        hatch_build, "_build_vgfpy", lambda _root: tmp_path / "vgfpy.so"
    )
    monkeypatch.setattr(hatch_build.platform, "system", lambda: "Linux")
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(tmp_path)

    with pytest.raises(RuntimeError, match="not executable"):
        hook.initialize("0.0.0", {})


def test_build_hook_rejects_unsupported_platform(
    hatch_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unsupported systems should fail before selecting vendor metadata."""
    vgfpy = tmp_path / f"vgfpy{hatch_build.importlib.machinery.EXTENSION_SUFFIXES[0]}"
    monkeypatch.setattr(hatch_build, "_build_vgfpy", lambda _root: vgfpy)
    monkeypatch.setattr(hatch_build.platform, "system", lambda: "Darwin")
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(tmp_path)

    with pytest.raises(RuntimeError, match="Unsupported platform 'Darwin'"):
        hook.initialize("0.0.0", {})
