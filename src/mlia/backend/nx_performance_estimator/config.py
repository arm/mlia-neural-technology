# SPDX-FileCopyrightText: Copyright 2023,2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Accelerator Performance Estimator backend configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click

from mlia.backend.config import BackendCliOption

CONFIG_TO_CLI_OPTION: dict[str, BackendCliOption] = {
    "system_config": click.Option(
        ["--system-config"],
        default=None,
        type=click.Path(path_type=Path),
    ),
    "compiler_config": click.Option(
        ["--compiler-config"],
        default=None,
        type=click.Path(path_type=Path),
    ),
    "skip_trivial_ops": click.Option(
        ["--skip-trivial-ops"],
        is_flag=True,
        default=None,
    ),
}


@dataclass
class NXPerformanceEstimatorConfig:
    """Configuration for the Neural Accelerator Performance Estimator."""

    DEFAULT = Path("default")

    system_config: str | Path
    compiler_config: str | Path
    skip_trivial_ops: bool = False

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if not isinstance(self.system_config, Path):
            self.system_config = (
                Path(self.system_config)
                if self.system_config != "default"
                else "default"
            )
        if not isinstance(self.compiler_config, Path):
            self.compiler_config = (
                Path(self.compiler_config)
                if self.compiler_config != "default"
                else "default"
            )

    def set_config_dir(self, config_dir: Path) -> None:
        """Prepend config file paths (if relative) with the given config dir."""

        def make_absolute(config: str | Path) -> Path:
            if config == "default":
                return NXPerformanceEstimatorConfig.DEFAULT

            config_path = Path(config)
            if config_path.is_absolute():
                return config_path
            return config_dir / config_path

        self.system_config = make_absolute(self.system_config)
        self.compiler_config = make_absolute(self.compiler_config)
