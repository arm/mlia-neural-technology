# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA Converter For Tflite backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.backend.tosa_converter_for_tflite.install import (
    get_tosa_converter_for_tflite_backend_installation,
)

registry.register(
    "tosa-converter-for-tflite",
    BackendConfiguration(
        supported_advice=[],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.WHEEL,
        installation=get_tosa_converter_for_tflite_backend_installation(),
        selectable=False,
    ),
    pretty_name="TOSA converter for tflite",
)
