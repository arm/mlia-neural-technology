# SPDX-FileCopyrightText: Copyright 2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Accelerator Performance Estimator config."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import mlia.backend.options as backend_options
from mlia.backend.registry import BackendRegistry
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)


def test_nx_performance_estimator_set_config_dir_abs_path(tmp_path: Path) -> None:
    """Test for set_config_dir method."""
    sys_cfg, compiler_cfg = Path(tmp_path), Path(tmp_path)
    cfg = NXPerformanceEstimatorConfig(sys_cfg, compiler_cfg)

    cfg.set_config_dir(tmp_path)  # absoulte path
    assert cfg.system_config == tmp_path
    assert cfg.compiler_config == tmp_path


def test_nx_performance_estimator_set_config_dir_relative() -> None:
    """Test for set_config_dir method."""
    sys_cfg, compiler_cfg = Path("system-config"), Path("compiler-config")
    cfg = NXPerformanceEstimatorConfig(sys_cfg, compiler_cfg)

    config_dir = Path("config-dir")
    cfg.set_config_dir(config_dir)  # relative path
    assert cfg.system_config == config_dir.joinpath("system-config")
    assert cfg.compiler_config == config_dir.joinpath("compiler-config")


def test_nx_performance_estimator_set_config_dir_default() -> None:
    """Test for set_config_dir method."""
    sys_cfg, compiler_cfg = "default", "default"
    cfg = NXPerformanceEstimatorConfig(sys_cfg, compiler_cfg)

    config_dir = Path("config-dir")
    cfg.set_config_dir(config_dir)  # relative path
    assert cfg.system_config == NXPerformanceEstimatorConfig.DEFAULT
    assert cfg.compiler_config == NXPerformanceEstimatorConfig.DEFAULT


@pytest.mark.parametrize("enabled", [False, True])
def test_skip_trivial_ops_cli_flag(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """Discover a valueless flag without changing the existing path options."""
    registry = BackendRegistry()
    NXPerformanceEstimatorPlugin.register(registry)
    registry.plugin_interface_versions["nx-performance-estimator"] = (
        NXPerformanceEstimatorPlugin.plugin_interface_version
    )
    monkeypatch.setattr(backend_options, "registry", registry)
    specs = backend_options.discover_backend_option_specs()
    options = {spec["config_key"]: spec["click_option"] for spec in specs}
    assert options["skip_trivial_ops"].is_flag
    for name in ("system_config", "compiler_config"):
        assert isinstance(options[name].type, click.Path)

    received: dict[str, object] = {}
    command = click.Command(
        "check", params=list(options.values()), callback=received.update
    )
    args = ["--nx-performance-estimator.skip-trivial-ops"] if enabled else []
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    assert received["nx_performance_estimator_skip_trivial_ops"] is (
        True if enabled else None
    )
    assert NXPerformanceEstimatorConfig("default", "default").skip_trivial_ops is False
