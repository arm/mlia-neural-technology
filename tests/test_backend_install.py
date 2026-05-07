# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology backend installation metadata."""

from mlia.backend.config import BackendType
from mlia.backend.install import InstallFromVendorPackage
from mlia.backend.ml_sdk_model_converter.plugin import MLSDKModelConverterPlugin
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin
from mlia.backend.registry import BackendRegistry
from mlia.backend.tosa_flatbuffers.plugin import TosaFlatBuffersPlugin


def test_nx_performance_estimator_registers_vendored_dependencies() -> None:
    """NX backend should be auto-installable together with its dependencies."""
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
    assert converter_config.installation is not None
    assert flatbuffers_config.installation is not None
    assert nx_config.installation.dependencies == [
        "ml-sdk-model-converter",
        "tosa-flatbuffers",
    ]
    assert nx_config.installation.supports(InstallFromVendorPackage())
    assert converter_config.installation.supports(InstallFromVendorPackage())
    assert flatbuffers_config.installation.supports(InstallFromVendorPackage())
