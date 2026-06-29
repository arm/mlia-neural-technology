# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology package build metadata."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Generator, cast

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).parents[1]


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
    """Wheel metadata should vendor only the NX estimator archive."""
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    wheel_target = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

    vendor_dir = PROJECT_ROOT / "src/mlia/_vendor/artifacts/nx-performance-estimator"
    expected_sha, expected_name = _read_expected_sha256(vendor_dir / ".sha256")

    assert expected_sha
    assert expected_name.endswith(".tar.gz")
    assert (vendor_dir / expected_name).is_file()
    assert "src/mlia" in wheel_target["only-include"]

    excluded_paths = set(wheel_target["exclude"])
    assert "src/mlia/_vendor/artifacts/ml-sdk-model-converter/**" in excluded_paths
    assert "src/mlia/_vendor/artifacts/tosa-flatbuffers/**" in excluded_paths
    assert all("nx-performance-estimator" not in path for path in excluded_paths)


def test_build_hook_force_includes_nx_estimator_archive(
    hatch_build: ModuleType,
) -> None:
    """The ignored NX estimator archive must be forced into built wheels."""
    hook = hatch_build.CustomBuildHook.__new__(hatch_build.CustomBuildHook)
    hook.root = str(PROJECT_ROOT)
    build_data: dict[str, object] = {}

    hook.initialize("0.0.0", build_data)

    force_include = build_data["force_include"]
    assert isinstance(force_include, dict)
    archive = next(
        PROJECT_ROOT.glob(
            "src/mlia/_vendor/artifacts/nx-performance-estimator/*.tar.gz"
        )
    )
    assert force_include[str(archive.relative_to(PROJECT_ROOT))] == (
        f"mlia/_vendor/artifacts/nx-performance-estimator/{archive.name}"
    )
