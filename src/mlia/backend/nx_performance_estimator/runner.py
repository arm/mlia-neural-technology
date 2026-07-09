# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Shared graph-compiler runner for the NX Performance Estimator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.repo import get_backend_repository
from mlia.utils.filesystem import get_mlia_resource_dirs, get_mlia_resources
from mlia.utils.proc import Command, OutputLogger, process_command_output

logger = logging.getLogger(__name__)


@dataclass
class NXPerformanceEstimatorOutputFiles:
    """Collection of output files of the NX Performance Estimator."""

    debug_database: Path
    performance_database: Path
    model_performance: Path

    @classmethod
    def from_output_dir(
        cls, output_dir: Path, output_name: str
    ) -> NXPerformanceEstimatorOutputFiles:
        """Create files in the Neural Accelerator Performance Estimator output dir."""
        name_to_suffix = {
            "debug_database": "_debug_database.dat",
            "performance_database": "_performance_database.dat",
            "model_performance": "_network_performance_summary.json",
        }
        args = {
            name: output_dir / f"{output_name}{suffix}"
            for name, suffix in name_to_suffix.items()
        }
        return cls(**args)

    def check_exists(self) -> None:
        """Raise a FileNotFoundError if one of the files does not exist."""
        for path in vars(self).values():
            if isinstance(path, Path) and not path.is_file():
                raise FileNotFoundError(
                    f"Expected output file '{path}' of the Neural Accelerator "
                    "Performance Estimator does not exist."
                )


def run_nx_performance_estimator(
    output_root: Path,
    backend_config: NXPerformanceEstimatorConfig,
    vgf_file: Path,
    output_name: str,
) -> NXPerformanceEstimatorOutputFiles:
    """Run the NX performance estimator and return its output files."""
    backend_repo = get_backend_repository()
    gc_path, _ = backend_repo.get_backend_settings("nx-performance-estimator")
    output_dir = output_root / "nx-performance-estimator"
    output_dir.mkdir(exist_ok=True)
    output_name = output_name.replace(".", "_")
    output = output_dir / output_name
    system_config = backend_config.system_config
    compiler_config = backend_config.compiler_config

    output.mkdir()

    system_config_args = (
        []
        if system_config == NXPerformanceEstimatorConfig.DEFAULT
        else ["--system_config", str(system_config)]
    )

    compiler_config_args = (
        []
        if compiler_config == NXPerformanceEstimatorConfig.DEFAULT
        else ["--compiler_config", str(compiler_config)]
    )

    cmd = Command(
        cmd=[
            str(gc_path / "graph-compiler-performance-estimator"),
            "-i",
            str(vgf_file.resolve()),
            "-o",
            str(output.name),
            *system_config_args,
            *compiler_config_args,
        ],
        cwd=output_dir,
    )

    process_command_output(cmd, [OutputLogger(logger, logging.INFO)])

    output_files = NXPerformanceEstimatorOutputFiles.from_output_dir(
        output_dir, output_name
    )
    output_files.check_exists()
    return output_files


def get_nx_resource_dir() -> Path:
    """Get NX performance estimator resource directory."""
    for resources_dir in get_mlia_resource_dirs():
        candidate = resources_dir / "nx-performance-estimator"
        if candidate.exists():
            return candidate
    return get_mlia_resources() / "nx-performance-estimator"
