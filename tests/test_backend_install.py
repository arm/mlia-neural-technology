# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology backend installation metadata."""

from mlia.backend.config import BackendType
from mlia.backend.install import InstallFromVendorPackage
from mlia.backend.ml_sdk_model_converter.plugin import MLSDKModelConverterPlugin
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin
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
