# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology backend installation metadata."""

import sys
from pathlib import Path

import pytest

from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.install import InstallFromVendorPackage
from mlia.backend.ml_sdk_model_converter.plugin import MLSDKModelConverterPlugin
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin
from mlia.backend.nx_performance_estimator.install import (
    NXPerformanceEstimatorInstaller,
)
from mlia.backend.nx_performance_estimator.installer import DEFAULT_LICENSE_FILE
from mlia.backend.registry import BackendRegistry
from mlia.backend.tosa_flatbuffers.plugin import TosaFlatBuffersPlugin


def test_nx_performance_estimator_only_manages_estimator_backend() -> None:
    """Public Python packages should be managed by normal project dependencies."""
    registry = BackendRegistry()
    NXPerformanceEstimatorPlugin.register(registry)
    MLSDKModelConverterPlugin.register(registry)
    TosaFlatBuffersPlugin.register(registry)

    nx_config = registry.items["nx-performance-estimator"]
    converter_config = registry.items["ml-sdk-model-converter"]
    flatbuffers_config = registry.items["tosa-flatbuffers"]

    assert nx_config.type == BackendType.CUSTOM
    assert converter_config.type == BackendType.CUSTOM
    assert flatbuffers_config.type == BackendType.WHEEL
    assert nx_config.installation is not None
    assert converter_config.installation is None
    assert flatbuffers_config.installation is None
    assert nx_config.installation.dependencies == []
    assert nx_config.installation.supports(InstallFromVendorPackage())
    assert nx_config.installation.requires_eula is True
    assert isinstance(
        nx_config.installation.backend_installer,
        NXPerformanceEstimatorInstaller,
    )
    assert nx_config.supported_systems == [
        System.LINUX_AMD64,
        System.WINDOWS_AMD64,
    ]


def test_nx_performance_estimator_exposes_packaged_eula() -> None:
    """Frontend metadata should expose the EULA used by the NX installer."""
    registry = BackendRegistry()
    NXPerformanceEstimatorPlugin.register(registry)
    installation = registry.items["nx-performance-estimator"].installation

    assert installation is not None
    assert installation.eula_resolver is not None
    documents = list(installation.eula_resolver(None))

    assert len(documents) == 1
    assert documents[0].title == "Arm End User License Agreement"
    assert documents[0].text == DEFAULT_LICENSE_FILE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "license_state", ["missing", "empty", "non-utf8", "bom-only", "bom-whitespace"]
)
def test_nx_eula_resolver_rejects_invalid_packaged_terms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, license_state: str
) -> None:
    """Do not offer frontend consent without readable, meaningful terms."""
    license_file = tmp_path / "license_agreement.txt"
    if license_state == "empty":
        license_file.write_text("", encoding="utf-8")
    elif license_state == "non-utf8":
        license_file.write_bytes(b"\xff")
    elif license_state == "bom-only":
        license_file.write_text("\ufeff", encoding="utf-8")
    elif license_state == "bom-whitespace":
        license_file.write_text("\ufeff \n\t", encoding="utf-8")
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.install.DEFAULT_LICENSE_FILE",
        license_file,
    )
    registry = BackendRegistry()
    NXPerformanceEstimatorPlugin.register(registry)
    installation = registry.items["nx-performance-estimator"].installation

    assert installation is not None
    assert installation.eula_resolver is not None
    with pytest.raises(RuntimeError, match="licence terms"):
        list(installation.eula_resolver(None))


def test_nx_eula_resolver_rejects_unreadable_packaged_terms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Report an actionable error when the packaged EULA cannot be opened."""
    license_file = tmp_path / "license_agreement.txt"
    license_file.write_text("terms", encoding="utf-8")
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.install.DEFAULT_LICENSE_FILE",
        license_file,
    )
    read_text = Path.read_text

    def fail_for_license(path: Path, *args: object, **kwargs: object) -> str:
        if path == license_file:
            raise PermissionError("unreadable")
        return read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", fail_for_license)
    registry = BackendRegistry()
    NXPerformanceEstimatorPlugin.register(registry)
    installation = registry.items["nx-performance-estimator"].installation

    assert installation is not None
    assert installation.eula_resolver is not None
    with pytest.raises(RuntimeError, match="licence terms"):
        list(installation.eula_resolver(None))


@pytest.mark.parametrize(
    ("platform_name", "expected_executable"),
    [
        ("linux", "graph-compiler-performance-estimator"),
        ("win32", "graph-compiler-performance-estimator.exe"),
    ],
)
def test_nx_performance_estimator_uses_platform_executable_name(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    expected_executable: str,
) -> None:
    """The backend installer should validate the binary invoked by the runner."""
    monkeypatch.setattr(sys, "platform", platform_name)

    registry = BackendRegistry()
    NXPerformanceEstimatorPlugin.register(registry)
    installation = registry.items["nx-performance-estimator"].installation

    assert installation is not None
    assert installation.supported_platforms == ["Linux", "Windows"]
    assert installation.path_checker.expected_files == [expected_executable]
