# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
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
    NXPerformanceEstimatorOutputFiles,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceEstimator,
)
from mlia.backend.nx_performance_estimator.performance import (
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
    operator_types_mapping = {
        "Identity": "identity_op_type",
        "model/re_lu_7/Relu": "RELU",
    }

    estimator = NXPerformanceEstimatorPerformanceEstimator(
        tmp_path, neural_technology_cfg.backend_config, operator_types_mapping
    )

    metrics = estimator.estimate(tmp_path / model_file)
    assert isinstance(metrics.backend_config, NXPerformanceEstimatorConfig)
    assert all(isinstance(file, Path) for file in vars(metrics.output_files).values())

    json_dump_path = Path(tmp_path / "nx_performance_statistics.json")
    assert json_dump_path.exists()


def test_nx_performance_metrics_to_standardized_output(tmp_path: Path) -> None:
    """Test conversion of NX PerformanceMetrics to standardized output."""
    # Create mock backend config
    mock_config = NXPerformanceEstimatorConfig(
        system_config="default",
        compiler_config="default",
    )

    # Create mock output files
    mock_output_files = NXPerformanceEstimatorOutputFiles(
        performance_database=tmp_path / "perf.db",
        debug_database=tmp_path / "debug.db",
    )

    # Create mock performance database parser
    mock_parser = MagicMock()

    # Create mock performance metrics with sample data
    mock_stats = MagicMock(spec=NXOperatorPerformanceStats)
    mock_stats.total_cycles = 5000

    perf_metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=mock_config,
        output_files=mock_output_files,
        performance_db_parser=mock_parser,
        performance_metrics={"chain_0": mock_stats, "chain_1": mock_stats},
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

    metrics = result["metrics"]
    metrics_dict = {m["name"]: m for m in metrics}

    # Check that metrics for both chains exist
    assert "chain_0_total_cycles" in metrics_dict
    assert "chain_1_total_cycles" in metrics_dict
    assert metrics_dict["chain_0_total_cycles"]["value"] == 5000.0
    assert metrics_dict["chain_1_total_cycles"]["value"] == 5000.0
    assert metrics_dict["chain_0_total_cycles"]["unit"] == "cycles"
