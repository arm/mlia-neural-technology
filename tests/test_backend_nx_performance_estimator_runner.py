# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for the shared NX performance estimator runner."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.nx_performance_estimator import runner
from mlia.backend.nx_performance_estimator.config import NXPerformanceEstimatorConfig


def test_run_nx_performance_estimator_reuses_existing_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A second analysis category should be able to reuse the estimator directory."""
    output_dir = tmp_path / "nx-performance-estimator"
    reused_output = output_dir / "model"
    reused_output.mkdir(parents=True)
    sentinel = reused_output / "existing.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    backend_repo = MagicMock()
    backend_repo.get_backend_settings.return_value = (tmp_path / "backend", {})
    monkeypatch.setattr(runner, "get_backend_repository", lambda: backend_repo)
    monkeypatch.setattr(runner, "get_nx_resource_dir", lambda: tmp_path / "resources")

    def run_estimator(_command: object, _consumers: object) -> None:
        for suffix in (
            "_debug_database.dat",
            "_performance_database.dat",
            "_network_performance_summary.json",
        ):
            (output_dir / f"model{suffix}").write_text("output", encoding="utf-8")

    monkeypatch.setattr(runner, "process_command_output", run_estimator)

    result = runner.run_nx_performance_estimator(
        tmp_path,
        NXPerformanceEstimatorConfig(
            system_config=NXPerformanceEstimatorConfig.DEFAULT,
            compiler_config=NXPerformanceEstimatorConfig.DEFAULT,
        ),
        tmp_path / "model.vgf",
        "model",
    )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    result.check_exists()
