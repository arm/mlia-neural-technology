# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Accelerator Performance Estimator performance estimation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
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
    NXSegmentPerformanceEntities,
    _performance_group_entity_id,
)
from mlia.backend.nx_performance_estimator.statistics import (
    NXOperatorPerformanceStats,
)
from mlia.core.output_postprocessing import postprocess_standardized_output
from mlia.core.output_validation import validate_standardized_output
from mlia.core.settings import ApplicationSettings, FilteringSettings
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
        "mlia.backend.nx_performance_estimator.runner.get_backend_repository",
        MagicMock(return_value=mock_repo),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.MLSDKModelConverter",
        MagicMock(return_value=MagicMock(return_value=tmp_path / "vgf_file")),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.runner.process_command_output",
        pco_mock,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.runner.validate_gcpe_compatible_vgf",
        lambda _: None,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "prepare_gcpe_compatible_vgfs",
        MagicMock(
            side_effect=lambda path, _output_dir: [
                SimpleNamespace(
                    segment_index=0,
                    path=path,
                    structural_source_operator_ids=[
                        "source_operator/segment_0/spirv-structural"
                    ],
                    structural_source_operator_names={},
                    debug_names=SimpleNamespace(
                        debug_name_to_spirv_ids={"known-debug-label": ["1"]}
                    ),
                )
            ]
        ),
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


def test_nx_performance_estimator_prefixes_subgraph_ids_with_segment_and_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw debug DB ids should be namespaced by segment and subgraph kind."""
    stats_chain = NXOperatorPerformanceStats(
        op_id=["0"],
        op_cycles=4500,
        total_cycles=5000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/segment_0/spirv-0"],
                "operator_types": ["OpType0"],
            }
        ],
    )
    stats_cascade = NXOperatorPerformanceStats(
        op_id=["0"],
        op_cycles=2700,
        total_cycles=3000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/segment_0/spirv-0"],
                "operator_types": ["OpType0"],
            }
        ],
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

    class FakePerformanceStats:
        def __init__(self, *args, **kwargs):
            pass

        def process_stats_per_chain(self) -> dict[str, NXOperatorPerformanceStats]:
            return {"5300": stats_chain}

        def process_stats_per_cascade(self) -> dict[str, NXOperatorPerformanceStats]:
            return {"9114": stats_cascade}

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorPerformanceEstimator._run_nx_performance_estimator",
        lambda _self, _path, _output_name: SimpleNamespace(
            performance_database=tmp_path / "performance.dat",
            debug_database=tmp_path / "debug.dat",
            model_performance=tmp_path / "model.json",
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXPerformanceDatabaseParser",
        MagicMock(
            return_value=MagicMock(
                parse_performance_database=MagicMock(return_value=[])
            )
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXDebugDatabaseParser",
        MagicMock(
            return_value=MagicMock(parse_debug_database=MagicMock(return_value={}))
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance.NXPerformanceStats",
        FakePerformanceStats,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXModelPerformanceStats.read_from_json",
        MagicMock(return_value=model_performance),
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
    metrics = estimator._estimate_gcpe_segments(
        [
            SimpleNamespace(
                segment_index=0,
                segment_name="graph_segment_0",
                path=tmp_path / "segment.vgf",
                structural_source_operator_ids=[
                    "source_operator/segment_0/spirv-structural"
                ],
                structural_source_operator_names={},
            )
        ],
        "section_0",
    )

    assert set(metrics.chain_performance_metrics) == {"chain/segment_0/5300"}
    assert set(metrics.cascade_performance_metrics) == {"cascade/segment_0/9114"}
    assert metrics.segment_entities == [
        NXSegmentPerformanceEntities(
            segment_index=0,
            segment_name="graph_segment_0",
            chain_ids=["chain/segment_0/5300"],
            cascade_ids=["cascade/segment_0/9114"],
            structural_source_operator_ids=[
                "source_operator/segment_0/spirv-structural"
            ],
        )
    ]


def test_nx_performance_estimator_uses_platform_executable_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NX performance estimator command should use the platform executable name."""
    system_config = tmp_path / "system.ini"
    compiler_config = tmp_path / "compiler.ini"
    system_config.write_bytes(b"size=32kb\r\n")
    compiler_config.write_bytes(b"enableDirectStriping=true\r\n")
    commands = []
    mock_repo = MagicMock()
    mock_repo.get_backend_settings = MagicMock(return_value=(tmp_path / "backend", {}))
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.runner.get_backend_repository",
        MagicMock(return_value=mock_repo),
    )
    estimator = NXPerformanceEstimatorPerformanceEstimator(
        tmp_path,
        {
            "nx-performance-estimator": {
                "system_config": system_config,
                "compiler_config": compiler_config,
            }
        },
        {},
    )

    expected_output = NXPerformanceEstimatorOutputFiles.from_output_dir(
        tmp_path / "nx-performance-estimator", "model"
    )

    def record_command(command, _consumers):
        commands.append(command)
        for file in vars(expected_output).values():
            file.touch()

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.runner.process_command_output",
        record_command,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.runner.validate_gcpe_compatible_vgf",
        lambda _: None,
    )

    estimator._run_nx_performance_estimator(tmp_path / "model.vgf", "model")

    executable = Path(commands[0].cmd[0])
    expected_name = (
        "graph-compiler-performance-estimator.exe"
        if sys.platform == "win32"
        else "graph-compiler-performance-estimator"
    )
    assert executable == tmp_path / "backend" / expected_name

    if sys.platform == "win32":
        output_dir = tmp_path / "nx-performance-estimator"
        assert commands[0].cmd[-4:] == [
            "-s",
            str(output_dir / "system.ini"),
            "-c",
            str(output_dir / "compiler.ini"),
        ]
        assert (output_dir / "system.ini").read_bytes() == b"size=32kb\n"
        assert (
            output_dir / "compiler.ini"
        ).read_bytes() == b"enableDirectStriping=true\n"
    else:
        assert commands[0].cmd[-4:] == [
            "-s",
            str(system_config),
            "-c",
            str(compiler_config),
        ]


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
            "SystemCache": {"readBytes": 256, "writeBytes": 128, "trafficCycles": 100},
        },
        utilization=[
            {
                "sectionName": "ConvolutionEngine",
                "cycles": "3500",
                "percentage": "70.0%",
            },
            {"sectionName": "Memory", "cycles": "1000", "percentage": "20.0%"},
        ],
        operators=[
            {
                "source_operator_ids": ["source_operator/test-0"],
                "operator_types": ["OpType0"],
            },
            {
                "source_operator_ids": ["source_operator/test-1"],
                "operator_types": ["OpType1"],
            },
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
            {
                "source_operator_ids": ["source_operator/test-3"],
                "operator_types": ["OpType3"],
            },
            {
                "source_operator_ids": ["source_operator/test-4"],
                "operator_types": ["OpType4"],
            },
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
        operators=[
            {
                "source_operator_ids": ["source_operator/test-0"],
                "operator_types": ["OpType0"],
            }
        ],
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
        chain_performance_metrics={"chain_0": stats_chain_0, "chain_1": stats_chain_1},
        cascade_performance_metrics={"cascade_0": stats_stripe},
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

    assert all(
        metric["aggregation"] == "sum"
        for breakdown in breakdowns
        for metric in breakdown["metrics"]
    )
    breakdowns_without_aggregation = [
        {
            **breakdown,
            "metrics": [
                {key: value for key, value in metric.items() if key != "aggregation"}
                for metric in breakdown["metrics"]
            ],
        }
        for breakdown in breakdowns
    ]
    assert breakdowns_without_aggregation == [
        {
            "entity_id": "chain_0",
            "metrics": [
                {"name": "total_cycles", "value": 5000, "unit": "cycles"},
                {"name": "op_cycles", "value": 4500, "unit": "cycles"},
                {"name": "dram_read_bytes", "value": 1024, "unit": "bytes"},
                {"name": "dram_write_bytes", "value": 512, "unit": "bytes"},
                {"name": "dram_traffic_cycles", "value": 300, "unit": "cycles"},
                {"name": "system_cache_read_bytes", "value": 256, "unit": "bytes"},
                {"name": "system_cache_write_bytes", "value": 128, "unit": "bytes"},
                {"name": "system_cache_traffic_cycles", "value": 100, "unit": "cycles"},
                {"name": "convolution_engine_cycles", "value": 3500, "unit": "cycles"},
                {"name": "memory_cycles", "value": 1000, "unit": "cycles"},
            ],
            "id": "chain_0",
        },
        {
            "entity_id": "chain_1",
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
            "id": "chain_1",
        },
        {
            "entity_id": "cascade_0",
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
            "id": "cascade_0",
        },
    ]
    assert result["entities"] == [
        {
            "id": "chain_0",
            "kind": "chain",
            "name": "Chain 0",
            "parent_ids": ["cascade_0"],
            "child_ids": [
                "source_operator/test-0",
                "source_operator/test-1",
            ],
            "placement": "NX",
            "attributes": {"stripe_ids": ["0", "1"]},
        },
        {
            "id": "chain_1",
            "kind": "chain",
            "name": "Chain 1",
            "child_ids": [
                "source_operator/test-3",
                "source_operator/test-4",
            ],
            "placement": "NX",
            "attributes": {"stripe_ids": ["3", "4"]},
        },
        {
            "id": "cascade_0",
            "kind": "cascade",
            "name": "Cascade 0",
            "child_ids": ["chain_0"],
            "placement": "NX",
            "attributes": {"stripe_ids": ["0"]},
        },
        {
            "id": "source_operator/test-0",
            "kind": "source_operator",
            "name": "OpType0",
            "placement": "NX",
            "parent_ids": ["chain_0"],
            "attributes": {"operator_types": ["OpType0"]},
        },
        {
            "id": "source_operator/test-1",
            "kind": "source_operator",
            "name": "OpType1",
            "placement": "NX",
            "parent_ids": ["chain_0"],
            "attributes": {"operator_types": ["OpType1"]},
        },
        {
            "id": "source_operator/test-3",
            "kind": "source_operator",
            "name": "OpType3",
            "placement": "NX",
            "parent_ids": ["chain_1"],
            "attributes": {"operator_types": ["OpType3"]},
        },
        {
            "id": "source_operator/test-4",
            "kind": "source_operator",
            "name": "OpType4",
            "placement": "NX",
            "parent_ids": ["chain_1"],
            "attributes": {"operator_types": ["OpType4"]},
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


def test_nx_performance_metrics_adds_segment_subgraph_entities(
    tmp_path: Path,
) -> None:
    """Segmented VGF performance output should expose segment hierarchy."""
    stats_chain = NXOperatorPerformanceStats(
        op_id=["0"],
        op_cycles=4500,
        total_cycles=5000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/segment_7/spirv-0"],
                "operator_types": ["OpType0"],
            }
        ],
    )
    stats_cascade = NXOperatorPerformanceStats(
        op_id=["0"],
        op_cycles=2700,
        total_cycles=3000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/segment_7/spirv-0"],
                "operator_types": ["OpType0"],
            }
        ],
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
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        chain_performance_metrics={"chain/segment_7/5300": stats_chain},
        cascade_performance_metrics={"cascade/segment_7/9114": stats_cascade},
        model_performance_stats=model_performance,
        segment_entities=[
            NXSegmentPerformanceEntities(
                segment_index=7,
                segment_name="My VGF Segment",
                chain_ids=["chain/segment_7/5300"],
                cascade_ids=["cascade/segment_7/9114"],
                structural_source_operator_ids=[
                    "source_operator/segment_7/spirv-0",
                    "source_operator/segment_7/spirv-999",
                ],
                structural_source_operator_names={
                    "source_operator/segment_7/spirv-999": "Friendly structural op"
                },
            )
        ],
    )
    model_file = tmp_path / "model.vgf"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    entities = {entity["id"]: entity for entity in output["results"][0]["entities"]}
    assert entities["segment/7"] == {
        "id": "segment/7",
        "kind": "segment",
        "name": "My VGF Segment",
        "child_ids": [
            "cascade/segment_7/9114",
            "source_operator/segment_7/spirv-999",
        ],
    }
    assert entities["cascade/segment_7/9114"]["name"] == "My VGF Segment/Cascade 9114"
    assert entities["cascade/segment_7/9114"]["parent_ids"] == ["segment/7"]
    assert entities["cascade/segment_7/9114"]["child_ids"] == ["chain/segment_7/5300"]
    assert entities["chain/segment_7/5300"]["name"] == "My VGF Segment/Chain 5300"
    assert entities["chain/segment_7/5300"]["parent_ids"] == ["cascade/segment_7/9114"]
    assert entities["source_operator/segment_7/spirv-999"] == {
        "id": "source_operator/segment_7/spirv-999",
        "kind": "source_operator",
        "name": "Friendly structural op",
        "parent_ids": ["segment/7"],
    }
    validate_standardized_output(output)


def test_nx_performance_metrics_adds_nn_module_entities_from_debug_stack(
    tmp_path: Path,
) -> None:
    """ExecuTorch nn_module_stack entries should become DAG parent entities."""
    stats_chain = NXOperatorPerformanceStats(
        op_id=["0", "1"],
        op_cycles=4500,
        total_cycles=5000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/aten_convolution_default"],
                "operator_types": ["Conv2D"],
                "nn_module_stacks": [
                    [
                        {"tracer_key": "<root>", "name": "<root>"},
                        {
                            "tracer_key": "L['model'].features.0",
                            "name": "model.features.0",
                            "module_type": "Conv2d",
                        },
                    ]
                ],
            },
            {
                "source_operator_ids": ["source_operator/aten_relu_default"],
                "operator_types": ["Relu"],
                "nn_module_stacks": [
                    [
                        {"tracer_key": "<root>", "name": "<root>"},
                        {
                            "tracer_key": "L['model'].features.0",
                            "name": "model.features.0",
                            "module_type": "Conv2d",
                        },
                    ]
                ],
            },
        ],
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
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        chain_performance_metrics={"chain_0": stats_chain},
        cascade_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.pte"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    result = output["results"][0]
    assert result["entity_kinds"] == [
        {"id": "segment", "child_kinds": ["cascade", "source_operator"]},
        {
            "id": "cascade",
            "parent_kinds": ["segment"],
            "child_kinds": ["chain"],
        },
        {
            "id": "chain",
            "parent_kinds": ["cascade"],
            "child_kinds": ["source_operator", "performance_group"],
        },
        {
            "id": "performance_group",
            "parent_kinds": ["chain"],
            "child_kinds": ["source_operator"],
        },
        {
            "id": "nn_module",
            "parent_kinds": ["nn_module"],
            "child_kinds": ["nn_module", "source_operator"],
        },
    ]
    entities = {entity["id"]: entity for entity in result["entities"]}
    module_id = "nn_module/L['model'].features.0"
    root_module_id = "nn_module/<root>"
    assert entities[root_module_id] == {
        "id": root_module_id,
        "kind": "nn_module",
        "name": "<root>",
        "child_ids": [module_id],
    }
    assert entities[module_id] == {
        "id": module_id,
        "kind": "nn_module",
        "name": "model.features.0",
        "parent_ids": [root_module_id],
        "child_ids": [
            "source_operator/aten_convolution_default",
            "source_operator/aten_relu_default",
        ],
        "attributes": {"module_type": "Conv2d"},
    }
    assert entities["source_operator/aten_convolution_default"]["parent_ids"] == [
        "chain_0",
        module_id,
    ]
    assert entities["source_operator/aten_relu_default"]["parent_ids"] == [
        "chain_0",
        module_id,
    ]
    validate_standardized_output(output)


def test_nx_performance_metrics_adds_code_stack_entities_from_debug_stack(
    tmp_path: Path,
) -> None:
    """ExecuTorch stack_trace entries should become recursive frame entities."""
    stats_chain = NXOperatorPerformanceStats(
        op_id=["0", "1"],
        op_cycles=4500,
        total_cycles=5000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/aten_convolution_default"],
                "operator_types": ["Conv2D"],
                "code_stacks": [
                    [
                        {
                            "file": "E:\\XPK\\OPPO\\model.py",
                            "line": "214",
                            "function": "forward",
                        },
                        {
                            "file": "E:\\XPK\\OPPO\\IFNet_HDv3.py",
                            "line": "156",
                            "function": "forward",
                        },
                    ]
                ],
            },
            {
                "source_operator_ids": ["source_operator/aten_relu_default"],
                "operator_types": ["Relu"],
                "code_stacks": [
                    [
                        {
                            "file": "E:\\XPK\\OPPO\\model.py",
                            "line": "214",
                            "function": "forward",
                        },
                        {
                            "file": "E:\\XPK\\OPPO\\IFNet_HDv3.py",
                            "line": "156",
                            "function": "forward",
                        },
                    ]
                ],
            },
        ],
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
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        chain_performance_metrics={"chain_0": stats_chain},
        cascade_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.pte"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    entities = {entity["id"]: entity for entity in output["results"][0]["entities"]}
    frame_entities = [
        entity for entity in entities.values() if entity["kind"] == "code_stack"
    ]
    assert [entity["name"] for entity in frame_entities] == [
        "model.py:214",
        "IFNet_HDv3.py:156",
    ]

    root_frame, leaf_frame = frame_entities
    assert root_frame["child_ids"] == [leaf_frame["id"]]
    assert root_frame["attributes"] == {
        "file": "E:/XPK/OPPO/model.py",
        "line": 214,
        "function": "forward",
    }
    assert leaf_frame["parent_ids"] == [root_frame["id"]]
    assert leaf_frame["child_ids"] == [
        "source_operator/aten_convolution_default",
        "source_operator/aten_relu_default",
    ]
    assert leaf_frame["attributes"] == {
        "file": "E:/XPK/OPPO/IFNet_HDv3.py",
        "line": 156,
        "function": "forward",
    }
    assert entities["source_operator/aten_convolution_default"]["parent_ids"] == [
        "chain_0",
        leaf_frame["id"],
    ]
    assert entities["source_operator/aten_relu_default"]["parent_ids"] == [
        "chain_0",
        leaf_frame["id"],
    ]
    validate_standardized_output(output)


def test_nx_performance_metrics_preserves_divergent_code_stack_prefixes(
    tmp_path: Path,
) -> None:
    """The same frame under different callers should have distinct prefix entities."""
    stats_chain = NXOperatorPerformanceStats(
        op_id=["0"],
        op_cycles=4500,
        total_cycles=5000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/aten_add_default"],
                "operator_types": ["Add"],
                "code_stacks": [
                    [
                        {"file": "/tmp/left.py", "line": "1", "function": "forward"},
                        {"file": "/tmp/shared.py", "line": "9", "function": "call"},
                    ],
                    [
                        {"file": "/tmp/right.py", "line": "2", "function": "forward"},
                        {"file": "/tmp/shared.py", "line": "9", "function": "call"},
                    ],
                ],
            },
        ],
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
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        chain_performance_metrics={"chain_0": stats_chain},
        cascade_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.pte"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    entities = {entity["id"]: entity for entity in output["results"][0]["entities"]}
    shared_frames = [
        entity
        for entity in entities.values()
        if entity["kind"] == "code_stack" and entity["name"] == "shared.py:9"
    ]
    assert len(shared_frames) == 2
    assert entities["source_operator/aten_add_default"]["parent_ids"] == [
        "chain_0",
        *(entity["id"] for entity in shared_frames),
    ]
    validate_standardized_output(output)


def test_nx_performance_metrics_preserves_divergent_nn_module_paths(
    tmp_path: Path,
) -> None:
    """Operators with sibling module stacks should get multiple module parents."""
    stats_chain = NXOperatorPerformanceStats(
        op_id=["0"],
        op_cycles=4500,
        total_cycles=5000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/aten_add_default"],
                "operator_types": ["Add"],
                "nn_module_stacks": [
                    [
                        {"tracer_key": "<root>", "name": "<root>"},
                        {"tracer_key": "L['model']", "name": "model"},
                        {"tracer_key": "L['model'].left", "name": "model.left"},
                    ],
                    [
                        {"tracer_key": "<root>", "name": "<root>"},
                        {"tracer_key": "L['model']", "name": "model"},
                        {"tracer_key": "L['model'].right", "name": "model.right"},
                    ],
                ],
            },
        ],
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
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        chain_performance_metrics={"chain_0": stats_chain},
        cascade_performance_metrics={},
        model_performance_stats=model_performance,
    )
    model_file = tmp_path / "model.pte"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    entities = {entity["id"]: entity for entity in output["results"][0]["entities"]}
    root_id = "nn_module/<root>"
    model_id = "nn_module/L['model']"
    left_id = "nn_module/L['model'].left"
    right_id = "nn_module/L['model'].right"
    assert entities[root_id]["child_ids"] == [model_id]
    assert entities[model_id]["child_ids"] == [left_id, right_id]
    assert right_id not in entities[left_id].get("child_ids", [])
    assert entities["source_operator/aten_add_default"]["parent_ids"] == [
        "chain_0",
        left_id,
        right_id,
    ]
    validate_standardized_output(output)


def test_nx_performance_metrics_uses_stripe_mapping_for_cascade_children(
    tmp_path: Path,
) -> None:
    """Cascade hierarchy should come from stripe membership, not location overlap."""
    segment_24_chain = NXOperatorPerformanceStats(
        op_id=["144"],
        op_cycles=4500,
        total_cycles=5000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/segment_24/spirv-411", ""],
                "operator_types": ["OpType0"],
            }
        ],
    )
    segment_24_cascade = NXOperatorPerformanceStats(
        op_id=["144"],
        op_cycles=2700,
        total_cycles=3000,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/segment_24/spirv-411", ""],
                "operator_types": ["OpType0"],
            }
        ],
    )
    segment_32_chain = NXOperatorPerformanceStats(
        op_id=["19"],
        op_cycles=4600,
        total_cycles=5100,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/segment_32/spirv-409", ""],
                "operator_types": ["OpType1"],
            }
        ],
    )
    segment_32_cascade = NXOperatorPerformanceStats(
        op_id=["19"],
        op_cycles=2800,
        total_cycles=3100,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/segment_32/spirv-409", ""],
                "operator_types": ["OpType1"],
            }
        ],
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
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        chain_performance_metrics={
            "chain/segment_24/1664": segment_24_chain,
            "chain/segment_32/1668": segment_32_chain,
        },
        cascade_performance_metrics={
            "cascade/segment_24/2256": segment_24_cascade,
            "cascade/segment_32/2086": segment_32_cascade,
        },
        model_performance_stats=model_performance,
        segment_entities=[
            NXSegmentPerformanceEntities(
                segment_index=24,
                segment_name="graph_segment_12",
                chain_ids=["chain/segment_24/1664"],
                cascade_ids=["cascade/segment_24/2256"],
            ),
            NXSegmentPerformanceEntities(
                segment_index=32,
                segment_name="graph_segment_16",
                chain_ids=["chain/segment_32/1668"],
                cascade_ids=["cascade/segment_32/2086"],
            ),
        ],
    )
    model_file = tmp_path / "model.vgf"
    model_file.write_bytes(b"test nx model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="nx-performance-estimator",
        target_config={"target": "neural-accelerator"},
    )

    entities = {entity["id"]: entity for entity in output["results"][0]["entities"]}
    assert entities["cascade/segment_24/2256"]["child_ids"] == ["chain/segment_24/1664"]
    assert entities["chain/segment_24/1664"]["parent_ids"] == [
        "cascade/segment_24/2256"
    ]
    assert entities["cascade/segment_32/2086"]["child_ids"] == ["chain/segment_32/1668"]
    assert entities["chain/segment_32/1668"]["parent_ids"] == [
        "cascade/segment_32/2086"
    ]
    validate_standardized_output(output)


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
        chain_performance_metrics={},
        cascade_performance_metrics={},
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
        chain_performance_metrics={},
        cascade_performance_metrics={},
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
        chain_performance_metrics={},
        cascade_performance_metrics={},
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
        chain_performance_metrics={},
        cascade_performance_metrics={},
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
        chain_performance_metrics={},
        cascade_performance_metrics={},
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
        chain_performance_metrics={},
        cascade_performance_metrics={},
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


def test_nx_entity_ids_use_kind_namespaces() -> None:
    """NX hierarchy entities begin with their declared entity kind."""
    segment = NXSegmentPerformanceEntities(
        segment_index=7,
        segment_name="segment",
        chain_ids=[],
        cascade_ids=[],
    )

    assert segment.entity_id == "segment/7"
    assert _performance_group_entity_id("chain/segment_7/1178", 0) == (
        "performance_group/segment_7/chain_1178/0"
    )


def test_nx_performance_metrics_projects_breakdowns_across_entity_views(
    tmp_path: Path,
) -> None:
    """Core processing projects backend output once across entity views."""
    single_stats = NXOperatorPerformanceStats(
        op_id=["0"],
        op_cycles=45,
        total_cycles=50,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": ["source_operator/single"],
                "operator_types": ["Single"],
                "code_stacks": [
                    [
                        {"file": "/tmp/model.py", "line": "10", "function": "run"},
                        {"file": "/tmp/layer.py", "line": "20", "function": "run"},
                    ]
                ],
            }
        ],
    )
    multi_stats = NXOperatorPerformanceStats(
        op_id=["1"],
        op_cycles=70,
        total_cycles=80,
        memory={},
        utilization=[],
        operators=[
            {
                "source_operator_ids": [
                    "source_operator/multi-a",
                    "source_operator/multi-b",
                ],
                "operator_types": ["Fused"],
            }
        ],
    )
    model_performance = NXModelPerformanceStats(
        compiled_size=1024,
        cache_cycles=20,
        cache_read_bytes=512,
        cache_write_bytes=256,
        compute_cycles=80,
        dram_cycles=50,
        dram_read_bytes=2048,
        dram_write_bytes=1024,
        dram_footprint=4096,
        inference_time=1.5,
        infs_per_sec=666.67,
        total_cycles=150,
    )
    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default",
            compiler_config="default",
        ),
        performance_db_parser=MagicMock(),
        chain_performance_metrics={
            "chain/segment_0/single": single_stats,
            "chain/segment_0/multi": multi_stats,
        },
        cascade_performance_metrics={},
        model_performance_stats=model_performance,
        segment_entities=[
            NXSegmentPerformanceEntities(
                segment_index=0,
                segment_name="segment",
                chain_ids=[
                    "chain/segment_0/single",
                    "chain/segment_0/multi",
                ],
                cascade_ids=[],
                structural_source_operator_ids=["source_operator/structural"],
            )
        ],
    )
    model_file = tmp_path / "model.pte"
    model_file.write_bytes(b"test nx model content")

    backend_output = perf_metrics.to_standardized_output(model_path=model_file)
    assert "source_operator/single" not in {
        breakdown["entity_id"]
        for breakdown in backend_output["results"][0]["breakdowns"]
    }

    output = postprocess_standardized_output(
        backend_output,
        ApplicationSettings(filtering=FilteringSettings(collapse=())),
    )
    result = output["results"][0]
    entities = {entity["id"]: entity for entity in result["entities"]}
    breakdowns = {
        breakdown["entity_id"]: breakdown for breakdown in result["breakdowns"]
    }
    breakdown_entity_ids = [
        breakdown["entity_id"] for breakdown in result["breakdowns"]
    ]
    frame_ids = [
        entity_id
        for entity_id, entity in entities.items()
        if entity["kind"] == schema.ENTITY_KIND_CODE_STACK
    ]
    single_chain_id = "chain/segment_0/single"
    multi_chain_id = "chain/segment_0/multi"
    performance_group_id = _performance_group_entity_id(multi_chain_id, 0)

    def metrics_by_name(entity_id: str) -> dict[str, dict[str, object]]:
        return {metric["name"]: metric for metric in breakdowns[entity_id]["metrics"]}

    assert len(frame_ids) == 2
    single_chain_metrics = metrics_by_name(single_chain_id)
    multi_chain_metrics = metrics_by_name(multi_chain_id)
    assert metrics_by_name("source_operator/single") == single_chain_metrics
    assert all(
        metrics_by_name(frame_id) == single_chain_metrics for frame_id in frame_ids
    )
    assert metrics_by_name(performance_group_id) == multi_chain_metrics
    assert "segment/0" not in breakdowns
    assert "parent_ids" not in entities[single_chain_id]
    assert "parent_ids" not in entities[multi_chain_id]
    assert entities["segment/0"]["child_ids"] == ["source_operator/structural"]
    assert all(
        metric["aggregation"] == "sum"
        for breakdown in result["breakdowns"]
        for metric in breakdown["metrics"]
    )
    assert "source_operator/structural" not in breakdowns
    assert "source_operator/multi-a" not in breakdowns
    assert "source_operator/multi-b" not in breakdowns
    assert breakdown_entity_ids.count("source_operator/single") == 1
    validate_standardized_output(output)
