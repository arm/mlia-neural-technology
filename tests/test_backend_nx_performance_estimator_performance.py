# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Accelerator Performance Estimator performance estimation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mlia.core.output_schema as schema
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXModelPerformanceStats,
    NXPerformanceEstimatorOutputFiles,
    NXPerformanceEstimatorPerformanceEstimator,
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.backend.nx_performance_estimator.statistics import (
    NXOperatorPerformanceStats,
)
from mlia.core.output_validation import validate_standardized_output
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration


def test_nx_performance_estimator_output_files(tmp_path: Path) -> None:
    """Test for class NXPerformanceEstimatorConfig."""
    output_files = NXPerformanceEstimatorOutputFiles.from_output_dir(
        tmp_path, "model_xyz"
    )
    with pytest.raises(FileNotFoundError):
        output_files.check_exists()
    for file in vars(output_files).values():
        assert isinstance(file, Path)
        file.touch()
    output_files.check_exists()


def test_nx_performance_estimator_output_files_ignores_timing_database(
    tmp_path: Path,
) -> None:
    """Extra r56.1 timing database output should not affect required files."""
    output_files = NXPerformanceEstimatorOutputFiles.from_output_dir(
        tmp_path, "model_xyz"
    )
    for file in vars(output_files).values():
        file.touch()
    (tmp_path / "model_xyz_timing_database.dat").touch()

    output_files.check_exists()


@pytest.mark.parametrize("model_file", ("model.tflite", "model.vgf", "model.pte"))
def test_nx_performance_estimator_performance_estimator(
    tmp_path: Path, model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test class NXPerformanceEstimatorPerformanceEstimator."""
    neural_technology_cfg = NeuralTechnologyConfiguration.load_profile(
        "neural-technology"
    )
    pco_mock = MagicMock()
    mock_repo = MagicMock()
    mock_repo.get_backend_settings = MagicMock(return_value=(tmp_path / "backend", {}))
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.get_backend_repository",
        MagicMock(return_value=mock_repo),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.MLSDKModelConverter",
        MagicMock(return_value=MagicMock(return_value=tmp_path / "vgf_file")),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.process_command_output",
        pco_mock,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXPerformanceDatabaseParser",
        MagicMock(
            return_value=MagicMock(
                parse_performance_database=MagicMock(return_value=[])
            )
        ),
    )
    debug_db = {
        "tosa_op_id_to_api_labels": {"545": ["TOSAMUL_spirv_id_716"]},
        "tosa_op_id_to_tosa_op": {"545": ["Mul"]},
    }
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXDebugDatabaseParser",
        MagicMock(
            return_value=MagicMock(
                parse_debug_database=MagicMock(return_value=debug_db)
            )
        ),
    )
    resolve_locations_mock = MagicMock(return_value={"716": "model/real_mul"})
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.resolve_spirv_id_locations",
        resolve_locations_mock,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorOutputFiles.check_exists",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.statistics."
        "NXModelPerformanceStats.read_from_json",
        MagicMock(),
    )
    operator_types_mapping = {
        "Identity": "identity_op_type",
        "model/re_lu_7/Relu": "RELU",
    }

    estimator = NXPerformanceEstimatorPerformanceEstimator(
        tmp_path, neural_technology_cfg.backend_config, operator_types_mapping
    )
    metrics = estimator.estimate(tmp_path / model_file)
    assert isinstance(metrics.backend_config, NXPerformanceEstimatorConfig)
    expected_vgf_file = tmp_path / (
        "model.vgf" if model_file == "model.vgf" else "vgf_file"
    )
    resolve_locations_mock.assert_called_once_with(
        tmp_path / model_file,
        expected_vgf_file,
        debug_db,
    )

    json_dump_path = Path(tmp_path / "nx_performance_statistics.json")
    assert json_dump_path.exists()


def test_nx_performance_estimator_reads_tosa_fallback_for_missing_spirv_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOSA fallback should only run when VGF debug info leaves ids unresolved."""
    neural_technology_cfg = NeuralTechnologyConfiguration.load_profile(
        "neural-technology"
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.get_backend_repository",
        MagicMock(
            return_value=MagicMock(
                get_backend_settings=MagicMock(return_value=(tmp_path / "backend", {}))
            )
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.process_command_output",
        MagicMock(),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXPerformanceDatabaseParser",
        MagicMock(
            return_value=MagicMock(
                parse_performance_database=MagicMock(return_value=[])
            )
        ),
    )
    debug_db = {
        "tosa_op_id_to_api_labels": {
            "545": ["TOSAMUL_spirv_id_716"],
            "635": ["TOSARESCALE_spirv_id_999"],
        },
        "tosa_op_id_to_tosa_op": {
            "545": ["Mul"],
            "635": ["Rescale"],
        },
    }
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXDebugDatabaseParser",
        MagicMock(
            return_value=MagicMock(
                parse_debug_database=MagicMock(return_value=debug_db)
            )
        ),
    )
    resolve_locations_mock = MagicMock(
        return_value={
            "716": "model/real_mul",
            "999": "model/rescale",
        }
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.resolve_spirv_id_locations",
        resolve_locations_mock,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorOutputFiles.check_exists",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.statistics."
        "NXModelPerformanceStats.read_from_json",
        MagicMock(),
    )

    estimator = NXPerformanceEstimatorPerformanceEstimator(
        tmp_path, neural_technology_cfg.backend_config, {}
    )
    stats = MagicMock(
        process_stats_per_chain=MagicMock(return_value={}),
        process_stats_per_stripe=MagicMock(return_value={}),
    )
    stats_class = MagicMock(return_value=stats)
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXPerformanceStats",
        stats_class,
    )

    estimator.estimate(tmp_path / "model.vgf")

    resolve_locations_mock.assert_called_once_with(
        tmp_path / "model.vgf",
        tmp_path / "model.vgf",
        debug_db,
    )
    assert stats_class.call_args.kwargs["spirv_id_locations"] == {
        "716": "model/real_mul",
        "999": "model/rescale",
    }


def test_nx_performance_estimator_keeps_enable_quantization_out_of_config(
    tmp_path: Path,
) -> None:
    """Runtime converter options should not be forwarded into config dataclass."""
    estimator = NXPerformanceEstimatorPerformanceEstimator(
        tmp_path,
        {
            "nx-performance-estimator": {
                "system_config": "system.ini",
                "compiler_config": "compiler.ini",
                "enable_quantization": False,
            }
        },
        {},
    )

    assert estimator.enable_quantization is False
    assert estimator.backend_config.system_config.name == "system.ini"
    assert estimator.backend_config.compiler_config.name == "compiler.ini"


def test_nx_performance_estimator_prefers_public_model_converter_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed public model-converter should take precedence over backend repo."""
    public_converter_path = tmp_path / "public-bin"
    converter_instance = MagicMock(return_value=tmp_path / "model.vgf")
    converter_class = MagicMock(return_value=converter_instance)
    backend_repo_mock = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "get_ml_sdk_model_converter_path",
        MagicMock(return_value=public_converter_path),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.get_backend_repository",
        MagicMock(return_value=backend_repo_mock),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.MLSDKModelConverter",
        converter_class,
    )

    estimator = NXPerformanceEstimatorPerformanceEstimator(
        tmp_path,
        {
            "nx-performance-estimator": {
                "system_config": "default",
                "compiler_config": "default",
            }
        },
        {},
    )

    assert estimator._run_ml_sdk_model_converter(tmp_path / "model.tflite") == (
        tmp_path / "model.vgf"
    )
    backend_repo_mock.get_backend_settings.assert_not_called()
    converter_class.assert_called_once_with(
        public_converter_path, enable_quantization=None
    )
    converter_instance.assert_called_once_with(
        tmp_path / "model.tflite", tmp_path / "ml-sdk-model-converter"
    )


def test_nx_performance_estimator_falls_back_to_backend_model_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend repo model-converter should be used when no public executable exists."""
    backend_converter_path = tmp_path / "backend-bin"
    converter_instance = MagicMock(return_value=tmp_path / "model.vgf")
    converter_class = MagicMock(return_value=converter_instance)
    backend_repo_mock = MagicMock()
    backend_repo_mock.get_backend_settings.return_value = (backend_converter_path, {})
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "get_ml_sdk_model_converter_path",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.get_backend_repository",
        MagicMock(return_value=backend_repo_mock),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.MLSDKModelConverter",
        converter_class,
    )

    estimator = NXPerformanceEstimatorPerformanceEstimator(
        tmp_path,
        {
            "nx-performance-estimator": {
                "system_config": "default",
                "compiler_config": "default",
                "enable_quantization": False,
            }
        },
        {},
    )

    assert estimator._run_ml_sdk_model_converter(tmp_path / "model.tflite") == (
        tmp_path / "model.vgf"
    )
    backend_repo_mock.get_backend_settings.assert_called_once_with(
        "ml-sdk-model-converter"
    )
    converter_class.assert_called_once_with(
        backend_converter_path, enable_quantization=False
    )
    converter_instance.assert_called_once_with(
        tmp_path / "model.tflite", tmp_path / "ml-sdk-model-converter"
    )


def test_nx_performance_metrics_to_standardized_output(
    tmp_path: Path,
) -> None:
    """Test conversion of NX PerformanceMetrics to standardized output."""
    # Create mock backend config
    mock_config = NXPerformanceEstimatorConfig(
        system_config="default",
        compiler_config="default",
    )

    # Create mock performance database parser
    mock_parser = MagicMock()

    stats_chain_0 = NXOperatorPerformanceStats(
        op_id=["0", "1"],
        op_cycles=4500,
        total_cycles=5000,
        memory={
            "DRAM": {"readBytes": 1024, "writeBytes": 512, "trafficCycles": 300},
            "Cache": {"readBytes": 256, "writeBytes": 128, "trafficCycles": 100},
        },
        utilization=[
            {"sectionName": "Compute", "cycles": "3500", "percentage": "70.0%"},
            {"sectionName": "Memory", "cycles": "1000", "percentage": "20.0%"},
        ],
        operators=[
            {"opLocation": ["OpLocation0"], "opType": ["OpType0"]},
            {"opLocation": ["OpLocation1"], "opType": ["OpType1"]},
        ],
    )

    stats_chain_1 = NXOperatorPerformanceStats(
        op_id=["3", "4"],
        op_cycles=9000,
        total_cycles=10000,
        memory={
            "DRAM": {"readBytes": 2048, "writeBytes": 1024, "trafficCycles": 600},
            "Cache": {"readBytes": 512, "writeBytes": 256, "trafficCycles": 200},
        },
        utilization=[
            {"sectionName": "Compute", "cycles": "7000", "percentage": "70.0%"},
            {"sectionName": "Memory", "cycles": "2000", "percentage": "20.0%"},
        ],
        operators=[
            {"opLocation": ["OpLocation3"], "opType": ["OpType3"]},
            {"opLocation": ["OpLocation4"], "opType": ["OpType4"]},
        ],
    )

    stats_stripe = NXOperatorPerformanceStats(
        op_id=["0"],
        op_cycles=2700,
        total_cycles=3000,
        memory={
            "DRAM": {"readBytes": 512, "writeBytes": 256, "trafficCycles": 150},
            "Cache": {"readBytes": 128, "writeBytes": 64, "trafficCycles": 50},
        },
        utilization=[
            {"sectionName": "Compute", "cycles": "2100", "percentage": "70.0%"},
            {"sectionName": "Memory", "cycles": "600", "percentage": "20.0%"},
        ],
        operators=[{"opLocation": ["OpLocation0"], "opType": ["OpType0"]}],
    )

    model_performance = NXModelPerformanceStats(
        compiled_size=1024000,
        cache_cycles=2000,
        cache_read_bytes=512000,
        cache_write_bytes=256000,
        compute_cycles=8000,
        dram_cycles=5000,
        dram_read_bytes=2048000,
        dram_write_bytes=1024000,
        dram_footprint=4096000,
        inference_time=1.5,
        infs_per_sec=666.67,
        total_cycles=15000,
    )

    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=mock_config,
        performance_db_parser=mock_parser,
        stripe_performance_metrics={"0": stats_stripe},
        chain_performance_metrics={"chain_0": stats_chain_0, "chain_1": stats_chain_1},
        model_performance_stats=model_performance,
    )

    # Create a model file for hash computation
    model_file = tmp_path / "model.tosamlir"
    model_file.write_bytes(b"test nx model content")

    # Call to_standardized_output
    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    # Structure checks
    for key in ("schema_version", "backends", "target", "model", "context", "results"):
        assert key in output

    assert output["schema_version"] == schema.SCHEMA_VERSION

    # Backend checks
    assert len(output["backends"]) == 1
    backend = output["backends"][0]
    assert backend["id"] == "nx-performance-estimator"
    assert backend["name"] == "Neural Accelerator (NX) Performance Estimator"

    # Target/component checks
    components = output["target"]["components"]
    assert len(components) > 0
    assert any(c["type"] == "gpu" and c["model"] == "nx" for c in components)

    # Check target description mentions neural accelerator
    assert "neural accelerator" in output["target"]["description"].lower()

    # Results/metrics checks
    result = output["results"][0]
    assert result["kind"] == "performance"
    assert result["status"] == "ok"

    breakdowns = result["breakdowns"]
    assert len(breakdowns) == 3

    assert breakdowns == [
        {
            "scope": "operator_chain",
            "name": "chain_0",
            "location": "OpLocation0;OpLocation1",
            "metrics": [
                {"name": "total_cycles", "value": 5000, "unit": "cycles"},
                {"name": "op_cycles", "value": 4500, "unit": "cycles"},
                {"name": "dram_read_bytes", "value": 1024, "unit": "bytes"},
                {"name": "dram_write_bytes", "value": 512, "unit": "bytes"},
                {"name": "dram_traffic_cycles", "value": 300, "unit": "cycles"},
                {"name": "cache_read_bytes", "value": 256, "unit": "bytes"},
                {"name": "cache_write_bytes", "value": 128, "unit": "bytes"},
                {"name": "cache_traffic_cycles", "value": 100, "unit": "cycles"},
                {"name": "compute_cycles", "value": 3500, "unit": "cycles"},
                {"name": "memory_cycles", "value": 1000, "unit": "cycles"},
            ],
            "id": "0;1",
        },
        {
            "scope": "operator_chain",
            "name": "chain_1",
            "location": "OpLocation3;OpLocation4",
            "metrics": [
                {"name": "total_cycles", "value": 10000, "unit": "cycles"},
                {"name": "op_cycles", "value": 9000, "unit": "cycles"},
                {"name": "dram_read_bytes", "value": 2048, "unit": "bytes"},
                {"name": "dram_write_bytes", "value": 1024, "unit": "bytes"},
                {"name": "dram_traffic_cycles", "value": 600, "unit": "cycles"},
                {"name": "cache_read_bytes", "value": 512, "unit": "bytes"},
                {"name": "cache_write_bytes", "value": 256, "unit": "bytes"},
                {"name": "cache_traffic_cycles", "value": 200, "unit": "cycles"},
                {"name": "compute_cycles", "value": 7000, "unit": "cycles"},
                {"name": "memory_cycles", "value": 2000, "unit": "cycles"},
            ],
            "id": "3;4",
        },
        {
            "scope": "operator",
            "name": "OpType0",
            "location": "OpLocation0",
            "metrics": [
                {"name": "total_cycles", "value": 3000, "unit": "cycles"},
                {"name": "op_cycles", "value": 2700, "unit": "cycles"},
                {"name": "dram_read_bytes", "value": 512, "unit": "bytes"},
                {"name": "dram_write_bytes", "value": 256, "unit": "bytes"},
                {"name": "dram_traffic_cycles", "value": 150, "unit": "cycles"},
                {"name": "cache_read_bytes", "value": 128, "unit": "bytes"},
                {"name": "cache_write_bytes", "value": 64, "unit": "bytes"},
                {"name": "cache_traffic_cycles", "value": 50, "unit": "cycles"},
                {"name": "compute_cycles", "value": 2100, "unit": "cycles"},
                {"name": "memory_cycles", "value": 600, "unit": "cycles"},
            ],
            "id": "0",
        },
    ]

    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert len(metrics) == len(result["metrics"])

    assert metrics["inference_time"] == {
        "name": "inference_time",
        "value": 1.5,
        "unit": "ms",
    }
    assert metrics["infs_per_sec"] == {
        "name": "infs_per_sec",
        "value": 666.67,
        "unit": "inferences/s",
    }
    assert metrics["total_cycles"] == {
        "name": "total_cycles",
        "value": 15000,
        "unit": "cycles",
    }
    assert metrics["compute_cycles"] == {
        "name": "compute_cycles",
        "value": 8000,
        "unit": "cycles",
    }
    assert metrics["cache_cycles"] == {
        "name": "cache_cycles",
        "value": 2000,
        "unit": "cycles",
    }
    assert metrics["dram_cycles"] == {
        "name": "dram_cycles",
        "value": 5000,
        "unit": "cycles",
    }
    assert metrics["compiled_size"] == {
        "name": "compiled_size",
        "value": 1024000,
        "unit": "bytes",
    }
    assert metrics["cache_read_bytes"] == {
        "name": "cache_read_bytes",
        "value": 512000,
        "unit": "bytes",
    }
    assert metrics["cache_write_bytes"] == {
        "name": "cache_write_bytes",
        "value": 256000,
        "unit": "bytes",
    }
    assert metrics["dram_read_bytes"] == {
        "name": "dram_read_bytes",
        "value": 2048000,
        "unit": "bytes",
    }
    assert metrics["dram_write_bytes"] == {
        "name": "dram_write_bytes",
        "value": 1024000,
        "unit": "bytes",
    }
    assert metrics["dram_footprint"] == {
        "name": "dram_footprint",
        "value": 4096000,
        "unit": "bytes",
    }

    assert metrics[schema.METRIC_NAME_INFERENCES_PER_SECOND] == {
        "name": schema.METRIC_NAME_INFERENCES_PER_SECOND,
        "value": pytest.approx(1000 / 1.5),
        "unit": schema.UNIT_INFERENCES_PER_SECOND,
    }
    assert metrics[schema.METRIC_NAME_TARGET_UTILIZATION] == {
        "name": schema.METRIC_NAME_TARGET_UTILIZATION,
        "value": pytest.approx(8000 / 15000 * 100),
        "unit": schema.UNIT_PERCENT,
    }

    for metric_name, unit in (
        (schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE, schema.UNIT_PERCENT),
        (schema.METRIC_NAME_CPU_UTILIZATION, schema.UNIT_PERCENT),
        (schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY, schema.UNIT_BYTES),
        (schema.METRIC_NAME_AVERAGE_MEMORY, schema.UNIT_BYTES),
    ):
        metric = metrics[metric_name]
        assert metric["unit"] == unit
        assert metric["availability"] == "unavailable"
        assert "value" not in metric
        assert metric["reason"]


def test_nx_performance_metrics_standardized_output_validates(
    tmp_path: Path,
) -> None:
    """Test NX performance output against the MLIA output schema."""
    model_performance = NXModelPerformanceStats(
        compiled_size=1024000,
        cache_cycles=2000,
        cache_read_bytes=512000,
        cache_write_bytes=256000,
        compute_cycles=8000,
        dram_cycles=5000,
        dram_read_bytes=2048000,
        dram_write_bytes=1024000,
        dram_footprint=4096000,
        inference_time=1.5,
        infs_per_sec=666.67,
        total_cycles=15000,
    )
    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        stripe_performance_metrics={},
        chain_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.tosamlir"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    validate_standardized_output(output)


def test_nx_performance_metrics_handles_zero_total_cycles(tmp_path: Path) -> None:
    """Target utilization should follow the core zero-cycle formula."""
    model_performance = NXModelPerformanceStats(
        compiled_size=1024000,
        cache_cycles=2000,
        cache_read_bytes=512000,
        cache_write_bytes=256000,
        compute_cycles=8000,
        dram_cycles=5000,
        dram_read_bytes=2048000,
        dram_write_bytes=1024000,
        dram_footprint=4096000,
        inference_time=1.5,
        infs_per_sec=666.67,
        total_cycles=0,
    )
    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        stripe_performance_metrics={},
        chain_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.tosamlir"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert metrics[schema.METRIC_NAME_TARGET_UTILIZATION] == {
        "name": schema.METRIC_NAME_TARGET_UTILIZATION,
        "value": 0.0,
        "unit": schema.UNIT_PERCENT,
    }


def test_nx_performance_metrics_derives_standard_throughput_from_latency(
    tmp_path: Path,
) -> None:
    """Standard throughput should follow the single-inference latency formula."""
    model_performance = NXModelPerformanceStats(
        compiled_size=1024000,
        cache_cycles=2000,
        cache_read_bytes=512000,
        cache_write_bytes=256000,
        compute_cycles=8000,
        dram_cycles=5000,
        dram_read_bytes=2048000,
        dram_write_bytes=1024000,
        dram_footprint=4096000,
        inference_time=2.0,
        infs_per_sec=123.0,
        total_cycles=15000,
    )
    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        stripe_performance_metrics={},
        chain_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.tosamlir"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert metrics["infs_per_sec"] == {
        "name": "infs_per_sec",
        "value": 123.0,
        "unit": schema.UNIT_INFERENCES_PER_SECOND,
    }
    assert metrics[schema.METRIC_NAME_INFERENCES_PER_SECOND] == {
        "name": schema.METRIC_NAME_INFERENCES_PER_SECOND,
        "value": 500.0,
        "unit": schema.UNIT_INFERENCES_PER_SECOND,
    }


def test_nx_performance_metrics_marks_standard_throughput_unavailable_without_latency(
    tmp_path: Path,
) -> None:
    """Standard throughput should not be fabricated from zero latency."""
    model_performance = NXModelPerformanceStats(
        compiled_size=1024000,
        cache_cycles=2000,
        cache_read_bytes=512000,
        cache_write_bytes=256000,
        compute_cycles=8000,
        dram_cycles=5000,
        dram_read_bytes=2048000,
        dram_write_bytes=1024000,
        dram_footprint=4096000,
        inference_time=0.0,
        infs_per_sec=0.0,
        total_cycles=15000,
    )
    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        stripe_performance_metrics={},
        chain_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.tosamlir"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    metric = metrics[schema.METRIC_NAME_INFERENCES_PER_SECOND]
    assert metric["unit"] == schema.UNIT_INFERENCES_PER_SECOND
    assert metric["availability"] == "unavailable"
    assert "value" not in metric
    assert metric["reason"]


def test_nx_performance_metrics_marks_null_backend_values_unavailable(
    tmp_path: Path,
) -> None:
    """Null backend metric values should become availability-aware metrics."""
    model_performance = NXModelPerformanceStats(
        compiled_size=1024000,
        cache_cycles=2000,
        cache_read_bytes=512000,
        cache_write_bytes=256000,
        compute_cycles=8000,
        dram_cycles=5000,
        dram_read_bytes=None,
        dram_write_bytes=1024000,
        dram_footprint=4096000,
        inference_time=1.5,
        infs_per_sec=None,
        total_cycles=15000,
    )
    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        stripe_performance_metrics={},
        chain_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.tosamlir"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert metrics["infs_per_sec"] == {
        "name": "infs_per_sec",
        "unit": schema.UNIT_INFERENCES_PER_SECOND,
        "availability": "unavailable",
        "reason": "Backend output did not provide a numeric value for this metric.",
    }
    assert metrics["dram_read_bytes"] == {
        "name": "dram_read_bytes",
        "unit": schema.UNIT_BYTES,
        "availability": "unavailable",
        "reason": "Backend output did not provide a numeric value for this metric.",
    }
    validate_standardized_output(output)


def test_nx_performance_metrics_marks_utilization_unavailable_without_cycles(
    tmp_path: Path,
) -> None:
    """Target utilization should not be fabricated from missing cycle values."""
    model_performance = NXModelPerformanceStats(
        compiled_size=1024000,
        cache_cycles=2000,
        cache_read_bytes=512000,
        cache_write_bytes=256000,
        compute_cycles=None,
        dram_cycles=5000,
        dram_read_bytes=2048000,
        dram_write_bytes=1024000,
        dram_footprint=4096000,
        inference_time=1.5,
        infs_per_sec=666.67,
        total_cycles=15000,
    )
    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        stripe_performance_metrics={},
        chain_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.tosamlir"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert metrics[schema.METRIC_NAME_TARGET_UTILIZATION] == {
        "name": schema.METRIC_NAME_TARGET_UTILIZATION,
        "unit": schema.UNIT_PERCENT,
        "availability": "unavailable",
        "reason": "Backend output did not provide a numeric value for this metric.",
    }
    validate_standardized_output(output)
