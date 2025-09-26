# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""ML SDK Model Converter backend module."""
import logging

from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.ml_sdk_model_converter.install import (
    get_ml_sdk_model_converter_installation,
)
from mlia.backend.registry import registry

logger = logging.getLogger(__name__)

registry.register(
    "ml-sdk-model-converter",
    BackendConfiguration(
        supported_advice=[],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.CUSTOM,
        installation=get_ml_sdk_model_converter_installation(),
    ),
    pretty_name="ML SDK Model Converter",
)
