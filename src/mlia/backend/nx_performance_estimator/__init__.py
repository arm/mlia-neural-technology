# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Accelerator Performance Estimator backend module."""
import logging

from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.nx_performance_estimator.install import (
    get_nx_performance_estimator_installation,
)
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

logger = logging.getLogger(__name__)

registry.register(
    "nx-performance-estimator",
    BackendConfiguration(
        supported_advice=[AdviceCategory.PERFORMANCE, AdviceCategory.COMPATIBILITY],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.CUSTOM,
        installation=get_nx_performance_estimator_installation(),
    ),
    pretty_name="Neural Accelerator Performance Estimator",
)
