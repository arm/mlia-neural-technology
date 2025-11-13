# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for Neural Accelerator Performance Estimator performance estimation."""  # pylint: disable=line-too-long
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from mlia.backend.ml_sdk_model_converter.conversion import MLSDKModelConverter
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.output_parsing import NXDebugDatabaseParser
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXPerformanceDatabaseParser,
)
from mlia.backend.nx_performance_estimator.statistics import NXOperatorPerformanceStats
from mlia.backend.nx_performance_estimator.statistics import NXPerformanceStats
from mlia.backend.repo import get_backend_repository
from mlia.core.performance import PerformanceEstimator
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.filesystem import is_vgf_file
from mlia.utils.logging import log_action
from mlia.utils.proc import Command
from mlia.utils.proc import OutputLogger
from mlia.utils.proc import process_command_output

logger = logging.getLogger(__name__)


@dataclass
class NXPerformanceEstimatorOutputFiles:
    """Collection of output files of the Neural Accelerator Performance Estimator."""

    debug_database: Path
    performance_database: Path

    @classmethod
    def from_output_dir(
        cls, output_dir: Path, output_name: str
    ) -> NXPerformanceEstimatorOutputFiles:
        """Create files in the Neural Accelerator Performance Estimator output dir."""
        name_to_suffix = {
            "debug_database": "_debug_database.dat",
            "performance_database": "_performance_database.dat",
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


@dataclass
class NXPerformanceEstimatorPerformanceMetrics:
    """Neural Accelerator Performance Estimator configuration and performance metrics."""  # pylint: disable=line-too-long

    backend_config: NXPerformanceEstimatorConfig
    output_files: NXPerformanceEstimatorOutputFiles
    performance_db_parser: NXPerformanceDatabaseParser
    performance_metrics: dict[str, NXOperatorPerformanceStats]


class NXPerformanceEstimatorPerformanceEstimator(
    PerformanceEstimator[
        Union[Path, ModelConfiguration], NXPerformanceEstimatorPerformanceMetrics
    ]
):
    """Performance estimator for the Neural Accelerator Performance Estimator."""

    resource_dir = get_mlia_resources() / "nx-performance-estimator"

    def __init__(
        self, output_dir: Path, backend_config: dict, operator_types_mapping: dict
    ) -> None:
        """Init performance estimator."""
        self.backend_config = NXPerformanceEstimatorConfig(
            **backend_config.get("nx-performance-estimator", {})
        )
        self.backend_config.set_config_dir(self.resource_dir)
        self.output_dir = output_dir
        self.operator_types_mapping = operator_types_mapping

    def estimate(
        self,
        model: Path | ModelConfiguration,
    ) -> NXPerformanceEstimatorPerformanceMetrics:
        """Estimate performance."""
        with log_action("Getting the performance data..."):
            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            # Check the file extension to see if we've been given a vgf file
            if is_vgf_file(model_path):
                vgf_file = model_path
            # Otherwise try to convert the file to vgf
            else:
                vgf_file = self._run_ml_sdk_model_converter(model_path)

            output = self._run_nx_performance_estimator(vgf_file, model_path.stem)

            perf_db_parser = NXPerformanceDatabaseParser(
                db_path=Path(output.performance_database)
            )
            performance_db = perf_db_parser.parse_performance_database()

            ddb_parser = NXDebugDatabaseParser(Path(output.debug_database))
            debug_db = ddb_parser.parse_debug_database()

            perf_stats = NXPerformanceStats(
                debug_db=debug_db,
                performance_db=performance_db,
            )
            stats_per_chain = perf_stats.process_stats_per_chain()
            output_file_path = self.output_dir / "nx_performance_statistics.json"
            self.json_dump(stats_per_chain, output_file_path)

            return NXPerformanceEstimatorPerformanceMetrics(
                self.backend_config,
                output,
                perf_db_parser,
                stats_per_chain,
            )

    def _run_ml_sdk_model_converter(self, model_path: Path) -> Path:
        """Run the ML SDK Model Converter and return the path to the SPIR-V file."""
        backend_repo = get_backend_repository()
        vmc_path, _ = backend_repo.get_backend_settings("ml-sdk-model-converter")
        output_dir = self.output_dir / "ml-sdk-model-converter"
        output_dir.mkdir()

        model_converter = MLSDKModelConverter(vmc_path)
        vgf_file = model_converter(model_path, output_dir)
        return vgf_file

    def _run_nx_performance_estimator(
        self, vgf_file: Path, output_name: str
    ) -> NXPerformanceEstimatorOutputFiles:
        """Run the Neural Accelerator Performance Estimator and return the output files."""  # pylint: disable=line-too-long
        backend_repo = get_backend_repository()
        gc_path, _ = backend_repo.get_backend_settings("nx-performance-estimator")
        output_dir = self.output_dir / "nx-performance-estimator"
        output_dir.mkdir()
        # We need to specify the basename for the output files here, i.e. neither
        # the output directory or the specific output file.
        output_name = output_name.replace(".", "_")
        output = output_dir / output_name
        system_config = self.backend_config.system_config
        compiler_config = self.backend_config.compiler_config

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
                str(vgf_file),
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

    def json_dump(self, stats_per_chain: dict, output_file_path: Path) -> None:
        """Make a json dump of the stats_per_chain dict."""
        json_serializable_stats = {
            key: obj.to_dict() for key, obj in stats_per_chain.items()
        }
        with output_file_path.open("w") as json_file:
            json.dump(json_serializable_stats, json_file, indent=4)
