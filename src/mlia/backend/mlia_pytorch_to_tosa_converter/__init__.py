# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA Converter For Pytorch backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.mlia_pytorch_to_tosa_converter.install import (
    get_mlia_pytorch_to_tosa_backend_installation,
)
from mlia.backend.registry import registry

registry.register(
    "mlia-pytorch-to-tosa-converter",
    BackendConfiguration(
        supported_advice=[],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.WHEEL,
        installation=get_mlia_pytorch_to_tosa_backend_installation(),
    ),
    pretty_name="TOSA converter for pytorch",
)
