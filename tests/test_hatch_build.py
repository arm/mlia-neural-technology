# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology package build metadata."""

from __future__ import annotations

import hashlib
import importlib
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Generator, cast

import pytest

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
    ):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    interface_module = ModuleType("hatchling.builders.hooks.plugin.interface")
    setattr(interface_module, "BuildHookInterface", object)
    monkeypatch.setitem(
        sys.modules,
        "hatchling.builders.hooks.plugin.interface",
        interface_module,
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
    assert "src/mlia" in wheel_target["only-include"]

    excluded_paths = set(wheel_target["exclude"])
    assert "src/mlia/_vendor/artifacts/ml-sdk-model-converter/**" in excluded_paths
    assert "src/mlia/_vendor/artifacts/tosa-flatbuffers/**" in excluded_paths
    assert all("nx-performance-estimator" not in path for path in excluded_paths)


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
    archive.write_bytes(f"{system} estimator".encode())
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
