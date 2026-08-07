# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest-native MLIA e2e tests."""

from __future__ import annotations

import re
from pathlib import Path

from mlia.testing import e2e as mlia_e2e
from mlia.testing.e2e import COMMON_PATTERNS
from mlia.testing.e2e import COMPATIBILITY_PATTERNS

NEURAL_TECHNOLOGY_COMPATIBILITY_PATTERNS = (
    r".*│.*Operator location.*│.*",
    r".*│.*NX placement.*│.*",
    r".*│.*NX compatibility.*│.*",
)

NEURAL_TECHNOLOGY_PERFORMANCE_PATTERNS = (
    r".*Neural Accelerator raw performance report:.*",
)


def assert_matches(pattern: str, output: str) -> None:
    assert re.search(pattern, output), f"Pattern: {pattern}\n\n{output}"


@mlia_e2e.parametrize(mlia_e2e.E2E_COMPATIBILITY)
def test_e2e_compatibility(
    case: mlia_e2e.E2ECase,
    tmp_path: Path,
) -> None:
    result = mlia_e2e.run_case(case, workdir=tmp_path)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{case}\n\n{output}"
    for pattern in (*COMMON_PATTERNS, *COMPATIBILITY_PATTERNS):
        assert_matches(pattern, output)
    for pattern in NEURAL_TECHNOLOGY_COMPATIBILITY_PATTERNS:
        assert_matches(pattern, output)
    mlia_e2e.emit_e2e_results(result)


@mlia_e2e.parametrize(mlia_e2e.E2E_PERFORMANCE)
def test_e2e_performance(
    case: mlia_e2e.E2ECase,
    tmp_path: Path,
) -> None:
    result = mlia_e2e.run_case(case, workdir=tmp_path)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{case}\n\n{output}"
    for pattern in (*COMMON_PATTERNS, *NEURAL_TECHNOLOGY_PERFORMANCE_PATTERNS):
        assert_matches(pattern, output)
    mlia_e2e.emit_e2e_results(result)
