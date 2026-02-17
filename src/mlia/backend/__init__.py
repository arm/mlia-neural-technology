# SPDX-FileCopyrightText: Copyright 2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend module."""

from mlia.backend.registry import registry
from mlia.plugins.plugins import load_backend_plugins

load_backend_plugins(registry)
