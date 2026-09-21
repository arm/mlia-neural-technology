# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared NX performance estimator runner."""

from pathlib import Path
import subprocess
from unittest.mock import MagicMock

import pytest

from mlia.backend.nx_performance_estimator import runner
from mlia.backend.nx_performance_estimator.config import NXPerformanceEstimatorConfig
from mlia.core.errors import ConfigurationError, InternalError


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
    monkeypatch.setattr(runner, "validate_gcpe_compatible_vgf", lambda _: None)

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


def test_run_nx_performance_estimator_rejects_unshaped_input_before_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Validation must stop GCPE before backend setup or process execution."""
    validate = MagicMock(side_effect=ValueError("shape-specialize the VGF"))
    backend = MagicMock()
    process = MagicMock()
    monkeypatch.setattr(runner, "validate_gcpe_compatible_vgf", validate)
    monkeypatch.setattr(runner, "get_backend_repository", backend)
    monkeypatch.setattr(runner, "process_command_output", process)
    model = tmp_path / "unshaped.vgf"

    with pytest.raises(ConfigurationError, match="shape-specialize"):
        runner.run_nx_performance_estimator(
            tmp_path, NXPerformanceEstimatorConfig("default", "default"), model, "model"
        )

    validate.assert_called_once_with(model)
    backend.assert_not_called()
    process.assert_not_called()
    assert not (tmp_path / "nx-performance-estimator").exists()


@pytest.mark.parametrize("returncode", [-11, 1, 0xC0000005])
@pytest.mark.parametrize("has_diagnostic", [True, False])
def test_run_nx_performance_estimator_reports_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    has_diagnostic: bool,
) -> None:
    """Unexpected native failures retain context and the backend's diagnostic."""
    backend = MagicMock()
    backend.get_backend_settings.return_value = (tmp_path / "backend", {})
    monkeypatch.setattr(runner, "validate_gcpe_compatible_vgf", lambda _: None)
    monkeypatch.setattr(runner, "get_backend_repository", lambda: backend)
    monkeypatch.setattr(runner, "get_nx_resource_dir", lambda: tmp_path)
    cause = subprocess.CalledProcessError(returncode, ["gcpe"])

    def fail(_command, consumers):
        if has_diagnostic:
            for consumer in consumers:
                consumer("Backend diagnostic: could not compile graph\n")
        raise cause

    monkeypatch.setattr(runner, "process_command_output", fail)
    model = tmp_path / "shaped.vgf"
    with pytest.raises(InternalError) as error:
        runner.run_nx_performance_estimator(
            tmp_path, NXPerformanceEstimatorConfig("default", "default"), model, "model"
        )

    message = str(error.value)
    assert str(model) in message
    assert str(returncode) in message
    if has_diagnostic:
        assert "Backend diagnostic: could not compile graph" in message
    else:
        assert "No backend diagnostic was emitted" in message
    assert "report" in message.lower()
    assert error.value.__cause__ is cause
