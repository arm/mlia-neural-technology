# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for NX Performance Estimator performance estimation."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import mlia
import mlia.core.output_schema as schema
from mlia.backend.ml_sdk_model_converter.conversion import MLSDKModelConverter
from mlia.backend.ml_sdk_model_converter.install import get_ml_sdk_model_converter_path
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.debug_locations import (
    resolve_spirv_id_locations,
)
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXDebugDatabaseParser,
    NXPerformanceDatabaseParser,
)
from mlia.backend.nx_performance_estimator.statistics import (
    NXModelCountMetricValue,
    NXModelFloatMetricValue,
    NXModelPerformanceStats,
    NXOperatorPerformanceStats,
    NXPerformanceStats,
)
from mlia.backend.repo import get_backend_repository
from mlia.core.performance import PerformanceEstimator
from mlia.nx_utils.filesystem import is_vgf_file
from mlia.utils.filesystem import get_mlia_resource_dirs, get_mlia_resources
from mlia.utils.logging import log_action
from mlia.utils.proc import Command, OutputLogger, process_command_output

logger = logging.getLogger(__name__)

_BACKEND_METRIC_UNAVAILABLE_REASON = (
    "Backend output did not provide a numeric value for this metric."
)


@dataclass
class NXPerformanceEstimatorOutputFiles:
    """Collection of output files of the Neural Accelerator Performance Estimator."""

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


@dataclass
class NXPerformanceEstimatorPerformanceMetrics:
    """NX Performance Estimator configuration and performance metrics."""

    backend_config: NXPerformanceEstimatorConfig
    performance_db_parser: NXPerformanceDatabaseParser
    stripe_performance_metrics: dict[str, NXOperatorPerformanceStats]
    chain_performance_metrics: dict[str, NXOperatorPerformanceStats]
    model_performance_stats: NXModelPerformanceStats

    def _build_model_metric(
        self,
        name: str,
        value: NXModelCountMetricValue | NXModelFloatMetricValue,
        unit: str,
    ) -> schema.Metric:
        if value is None:
            return schema.Metric(
                name=name,
                value=None,
                unit=unit,
                availability=schema.MetricAvailability.UNAVAILABLE,
                reason=_BACKEND_METRIC_UNAVAILABLE_REASON,
            )

        return schema.Metric(name=name, value=value, unit=unit)

    def _build_target_utilization_metric(self) -> schema.Metric:
        compute_cycles = self.model_performance_stats.compute_cycles
        total_cycles = self.model_performance_stats.total_cycles
        if compute_cycles is None or total_cycles is None:
            return schema.Metric(
                name=schema.METRIC_NAME_TARGET_UTILIZATION,
                value=None,
                unit=schema.UNIT_PERCENT,
                availability=schema.MetricAvailability.UNAVAILABLE,
                reason=_BACKEND_METRIC_UNAVAILABLE_REASON,
            )

        return schema.Metric(
            name=schema.METRIC_NAME_TARGET_UTILIZATION,
            value=(compute_cycles / total_cycles * 100 if total_cycles else 0.0),
            unit=schema.UNIT_PERCENT,
        )

    def _build_breakdown_metrics(
        self, stats: NXOperatorPerformanceStats
    ) -> list[schema.Metric]:
        breakdown_metrics = [
            schema.Metric(name="total_cycles", value=stats.total_cycles, unit="cycles"),
            schema.Metric(name="op_cycles", value=stats.op_cycles, unit="cycles"),
        ]

        for mem_type, mem_data in stats.memory.items():
            breakdown_metrics.extend(
                [
                    schema.Metric(
                        name=f"{mem_type.lower()}_read_bytes",
                        value=mem_data["readBytes"],
                        unit="bytes",
                    ),
                    schema.Metric(
                        name=f"{mem_type.lower()}_write_bytes",
                        value=mem_data["writeBytes"],
                        unit="bytes",
                    ),
                    schema.Metric(
                        name=f"{mem_type.lower()}_traffic_cycles",
                        value=mem_data["trafficCycles"],
                        unit="cycles",
                    ),
                ]
            )

        for util in stats.utilization:
            breakdown_metrics.append(
                schema.Metric(
                    name=f"{util['sectionName'].lower()}_cycles",
                    value=int(util["cycles"]),
                    unit="cycles",
                )
            )

        return breakdown_metrics

    def to_standardized_output(
        self,
        model_path: Path,
        backend_name: str = "nx-performance-estimator",
        target_config: dict[str, Any] | None = None,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            backend_name: Name of the backend (default: 'nx-performance-estimator')
            target_config: Target configuration parameters
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run

        Returns:
            Dictionary representing standardized output
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend configuration
        backend_config_dict = {
            "system_config": Path(self.backend_config.system_config).stem
            if self.backend_config.system_config != NXPerformanceEstimatorConfig.DEFAULT
            else "default",
            "compiler_config": Path(self.backend_config.compiler_config).stem
            if self.backend_config.compiler_config
            != NXPerformanceEstimatorConfig.DEFAULT
            else "default",
        }

        # Create backend
        backend = schema.Backend(
            id=backend_name,
            name="Neural Accelerator (NX) Performance Estimator",
            version="unknown",  # NX version not readily available
            configuration=backend_config_dict,
        )

        # Extract target info from config
        target_config = target_config or {}
        target_type = target_config.get(
            "target_type", target_config.get("target", "neural-accelerator")
        )
        profile_name = target_config.get("profile_name", target_type)

        # Extract variant from system_config name if not default
        variant = None
        if self.backend_config.system_config != NXPerformanceEstimatorConfig.DEFAULT:
            variant = Path(self.backend_config.system_config).stem

        # Create target components - Neural Accelerator (NX)
        components = [
            schema.Component(
                type=schema.ComponentType.GPU,
                family="mali",
                model="nx",
                variant=variant,
            )
        ]

        # Create target
        target = schema.Target(
            profile_name=profile_name,
            target_type=target_type,
            components=components,
            configuration=target_config,
            description="Neural Accelerator (NX) performance estimation",
        )

        # Calculate model hash and size
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        model_size = model_path.stat().st_size

        # Determine model format (suffix without the dot)
        suffix = model_path.suffix.lower()
        model_format = suffix.lstrip(".") or "unknown"

        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
            size_bytes=model_size,
        )

        # Create context
        context = schema.Context(
            cli_arguments=cli_arguments or [],
            runtime_configuration=None,
            git=None,
            notes=None,
        )

        metric_units = {
            "inference_time": "ms",
            "infs_per_sec": "inferences/s",
            "total_cycles": "cycles",
            "compute_cycles": "cycles",
            "cache_cycles": "cycles",
            "dram_cycles": "cycles",
            "compiled_size": "bytes",
            "cache_read_bytes": "bytes",
            "cache_write_bytes": "bytes",
            "dram_read_bytes": "bytes",
            "dram_write_bytes": "bytes",
            "dram_footprint": "bytes",
        }

        metrics = []
        for field_name, unit in metric_units.items():
            value = getattr(self.model_performance_stats, field_name)
            metrics.append(self._build_model_metric(field_name, value, unit))

        if self.model_performance_stats.inference_time:
            metrics.append(
                schema.Metric(
                    name=schema.METRIC_NAME_INFERENCES_PER_SECOND,
                    value=1000 / self.model_performance_stats.inference_time,
                    unit=schema.UNIT_INFERENCES_PER_SECOND,
                )
            )
        metrics.append(self._build_target_utilization_metric())
        metrics = schema.ensure_standard_performance_metrics(metrics)

        breakdowns = []
        for chain_name, stats in self.chain_performance_metrics.items():
            breakdown_metrics = self._build_breakdown_metrics(stats)

            breakdowns.append(
                schema.Breakdown(
                    scope=schema.OperatorScope.OPERATOR_CHAIN,
                    name=chain_name,
                    location=";".join(
                        [";".join(op["opLocation"]) for op in stats.operators]
                    ),
                    metrics=breakdown_metrics,
                    id=";".join(stats.op_id),
                    qualifiers={},
                )
            )

        for stripe_id, stats in self.stripe_performance_metrics.items():
            breakdown_metrics = self._build_breakdown_metrics(stats)

            breakdowns.append(
                schema.Breakdown(
                    scope=schema.OperatorScope.OPERATOR,
                    name=";".join([";".join(op["opType"]) for op in stats.operators]),
                    location=";".join(
                        [";".join(op["opLocation"]) for op in stats.operators]
                    ),
                    metrics=breakdown_metrics,
                    id=stripe_id,
                    qualifiers={},
                )
            )

        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer=backend.id,
            warnings=[],
            errors=[],
            metrics=metrics,
            mode=None,  # NX doesn't specify simulation/measured,
            breakdowns=breakdowns,
        )

        return schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
            extensions={},
        ).to_dict()


def _get_nx_resource_dir() -> Path:
    for resources_dir in get_mlia_resource_dirs():
        candidate = resources_dir / "nx-performance-estimator"
        if candidate.exists():
            return candidate
    return get_mlia_resources() / "nx-performance-estimator"


class NXPerformanceEstimatorPerformanceEstimator(
    PerformanceEstimator[Union[Path, Any], NXPerformanceEstimatorPerformanceMetrics]
):
    """Performance estimator for the Neural Accelerator Performance Estimator."""

    resource_dir = _get_nx_resource_dir()

    def __init__(
        self, output_dir: Path, backend_config: dict, operator_types_mapping: dict
    ) -> None:
        """Init performance estimator."""
        backend_options = dict(backend_config.get("nx-performance-estimator", {}))
        self.enable_quantization = backend_options.pop("enable_quantization", None)
        self.backend_config = NXPerformanceEstimatorConfig(**backend_options)
        self.backend_config.set_config_dir(self.resource_dir)
        self.output_dir = output_dir
        self.operator_types_mapping = operator_types_mapping

    def estimate(
        self,
        model: Path | Any,
    ) -> NXPerformanceEstimatorPerformanceMetrics:
        """Estimate performance."""
        with log_action("Getting the performance data..."):
            model_path = (
                Path(model.model_path) if hasattr(model, "model_path") else model
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
            spirv_id_locations = resolve_spirv_id_locations(
                model_path,
                vgf_file,
                debug_db,
            )

            perf_stats = NXPerformanceStats(
                debug_db=debug_db,
                performance_db=performance_db,
                spirv_id_locations=spirv_id_locations,
            )
            stats_per_chain = perf_stats.process_stats_per_chain()
            output_file_path = self.output_dir / "nx_performance_statistics.json"
            self.json_dump(stats_per_chain, output_file_path)

            stats_per_stripe = perf_stats.process_stats_per_stripe()

            model_performance_stats = NXModelPerformanceStats.read_from_json(
                output.model_performance
            )

            return NXPerformanceEstimatorPerformanceMetrics(
                self.backend_config,
                perf_db_parser,
                stats_per_stripe,
                stats_per_chain,
                model_performance_stats,
            )

    def _run_ml_sdk_model_converter(self, model_path: Path) -> Path:
        """Run the ML SDK Model Converter and return the path to the SPIR-V file."""
        vmc_path = get_ml_sdk_model_converter_path()
        if vmc_path is None:
            backend_repo = get_backend_repository()
            vmc_path, _ = backend_repo.get_backend_settings("ml-sdk-model-converter")
        output_dir = self.output_dir / "ml-sdk-model-converter"
        output_dir.mkdir(exist_ok=True)

        model_converter = MLSDKModelConverter(
            vmc_path, enable_quantization=self.enable_quantization
        )
        vgf_file = model_converter(model_path, output_dir)
        return vgf_file

    def _run_nx_performance_estimator(
        self, vgf_file: Path, output_name: str
    ) -> NXPerformanceEstimatorOutputFiles:
        """Run the NX Performance Estimator and return the output files."""
        backend_repo = get_backend_repository()
        gc_path, _ = backend_repo.get_backend_settings("nx-performance-estimator")
        output_dir = self.output_dir / "nx-performance-estimator"
        output_dir.mkdir(exist_ok=True)
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

    def json_dump(self, stats_per_chain: dict, output_file_path: Path) -> None:
        """Make a json dump of the stats_per_chain dict."""
        json_serializable_stats = {
            key: obj.to_dict() for key, obj in stats_per_chain.items()
        }
        with output_file_path.open("w") as json_file:
            json.dump(json_serializable_stats, json_file, indent=4)
