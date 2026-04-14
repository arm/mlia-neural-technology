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


@pytest.mark.parametrize("model_file", ("model.tflite", "model.vgf"))
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
        MagicMock(),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXDebugDatabaseParser",
        MagicMock(),
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

    json_dump_path = Path(tmp_path / "nx_performance_statistics.json")
    assert json_dump_path.exists()


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

    assert result["metrics"] == [
        {"name": "inference_time", "value": 1.5, "unit": "ms"},
        {"name": "infs_per_sec", "value": 666.67, "unit": "inferences/s"},
        {"name": "total_cycles", "value": 15000, "unit": "cycles"},
        {"name": "compute_cycles", "value": 8000, "unit": "cycles"},
        {"name": "cache_cycles", "value": 2000, "unit": "cycles"},
        {"name": "dram_cycles", "value": 5000, "unit": "cycles"},
        {"name": "compiled_size", "value": 1024000, "unit": "bytes"},
        {"name": "cache_read_bytes", "value": 512000, "unit": "bytes"},
        {"name": "cache_write_bytes", "value": 256000, "unit": "bytes"},
        {"name": "dram_read_bytes", "value": 2048000, "unit": "bytes"},
        {"name": "dram_write_bytes", "value": 1024000, "unit": "bytes"},
        {"name": "dram_footprint", "value": 4096000, "unit": "bytes"},
    ]
