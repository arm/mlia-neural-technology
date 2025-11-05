# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Accelerator Graph Compiler config."""
from __future__ import annotations

from pathlib import Path

from mlia.backend.nx_graph_compiler.config import CONFIG_TO_CLI_OPTION
from mlia.backend.nx_graph_compiler.config import NXGraphCompilerConfig


def test_nx_graph_compiler_config() -> None:
    """Test for class NXGraphCompilerConfig."""
    sys_cfg, compiler_cfg = Path("system-config"), Path("compiler-config")
    cfg = NXGraphCompilerConfig(sys_cfg, compiler_cfg)
    assert cfg.system_config == sys_cfg
    assert cfg.system_config == sys_cfg

    assert set(CONFIG_TO_CLI_OPTION) == set(vars(cfg))


def test_nx_graph_compiler_set_config_dir_abs_path(tmp_path: Path) -> None:
    """Test for set_config_dir method."""
    sys_cfg, compiler_cfg = Path(tmp_path), Path(tmp_path)
    cfg = NXGraphCompilerConfig(sys_cfg, compiler_cfg)

    cfg.set_config_dir(tmp_path)  # absoulte path
    assert cfg.system_config == tmp_path
    assert cfg.compiler_config == tmp_path


def test_nx_graph_compiler_set_config_dir_relative() -> None:
    """Test for set_config_dir method."""
    sys_cfg, compiler_cfg = Path("system-config"), Path("compiler-config")
    cfg = NXGraphCompilerConfig(sys_cfg, compiler_cfg)

    config_dir = Path("config-dir")
    cfg.set_config_dir(config_dir)  # relative path
    assert cfg.system_config == config_dir.joinpath("system-config")
    assert cfg.compiler_config == config_dir.joinpath("compiler-config")


def test_nx_graph_compiler_set_config_dir_default() -> None:
    """Test for set_config_dir method."""
    sys_cfg, compiler_cfg = "default", "default"
    cfg = NXGraphCompilerConfig(sys_cfg, compiler_cfg)

    config_dir = Path("config-dir")
    cfg.set_config_dir(config_dir)  # relative path
    assert cfg.system_config == NXGraphCompilerConfig.DEFAULT
    assert cfg.compiler_config == NXGraphCompilerConfig.DEFAULT
