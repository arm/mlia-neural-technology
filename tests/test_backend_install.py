# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology backend installation metadata."""

import sys

import pytest

from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.install import InstallFromVendorPackage
from mlia.backend.ml_sdk_model_converter.plugin import MLSDKModelConverterPlugin
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin
from mlia.backend.nx_performance_estimator.install import (
    NXPerformanceEstimatorInstaller,
)
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
