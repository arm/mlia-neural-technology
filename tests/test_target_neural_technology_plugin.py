# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology target plugin metadata."""

from mlia.target.neural_technology.plugin import _create_target_info


def test_target_info_uses_current_core_metadata() -> None:
    """Target metadata should use the current core interface directly."""
    target_info = _create_target_info()

    assert target_info.supported_backends == [
        "nx-performance-estimator",
        "neural-technology-profiling-data",
    ]
    assert target_info.default_backends == ["nx-performance-estimator"]
    assert target_info.supports_torch_module is True
    assert target_info.torch_module_backend == "nx-performance-estimator"
