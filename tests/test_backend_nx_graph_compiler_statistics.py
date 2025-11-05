# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Accelerator Graph Compiler performance estimation."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mlia.backend.nx_graph_compiler.output_parsing import NXDebugDatabaseParser
from mlia.backend.nx_graph_compiler.output_parsing import NXPerformanceDatabaseParser
from mlia.backend.nx_graph_compiler.statistics import NXOperatorPerformanceStats
from mlia.backend.nx_graph_compiler.statistics import NXPerformanceStats


def test_nx_operator_performance_stats_to_dict() -> None:
    """Test expected dictionary of NXOperator."""

    op_stats = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "Internal": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
        ],
        operators=["foo"],
    )

    assert op_stats.to_dict() == {
        "stripe_ids": [33],
        "op_cycles": 15,
        "total_cycles": 18,
        "memory": {
            "Internal": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        "utilization": [
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
        ],
        "operators": ["foo"],
    }


def test_sanitize_memory_fields_expected_input() -> None:
    """Sanitize memory attribute of NXOperator."""
    op_stats = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "Internal": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
        ],
        operators=["foo"],
    )
    op_stats.sanitize_memory_fields()

    assert op_stats.memory == {
        "L1": {"readBytes": 4, "writeBytes": 6, "trafficCycles": 43530},
        "L2": {"readBytes": 5, "writeBytes": 7, "trafficCycles": 456570},
        "SystemCache": {
            "readBytes": 1,
            "writeBytes": 2,
            "trafficCycles": 3,
        },
        "DRAM": {
            "readBytes": 312369600,
            "writeBytes": 31104000,
            "trafficCycles": 7056279,
        },
    }


def test_sanitize_memory_fields_missing_memory_name_input() -> None:
    """Throw error when the memoryName key is absent."""
    op_stats = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "wrong_key": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
        ],
        operators=["foo"],
    )
    with pytest.raises(KeyError):
        op_stats.sanitize_memory_fields()


def test_sanitize_utilization_fields_expected_input() -> None:
    """Sanitize utilization attribute of the Neural Accelerator Operator."""
    op_stats = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=180,
        memory={
            "Internal": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 100},
            {"sectionName": "VectorEngine", "cycles": 80},
            {"sectionName": "OutputWriter", "cycles": 65},
            {"sectionName": "VectorEngine", "cycles": 87},
            {"sectionName": "ConvolutionEngine", "cycles": 87},
            {"sectionName": "WeightDecoder", "cycles": 100},
            {"sectionName": "InputReader", "cycles": 80},
            {"sectionName": "InputReader", "cycles": 80},
        ],
        operators=["foo"],
    )
    op_stats.sanitize_utilization_fields()

    assert op_stats.utilization == [
        {"sectionName": "OutputWriter", "cycles": "165", "percentage": "91.7%"},
        {"sectionName": "VectorEngine", "cycles": "167", "percentage": "92.8%"},
        {"sectionName": "ConvolutionEngine", "cycles": "87", "percentage": "48.3%"},
        {"sectionName": "WeightDecoder", "cycles": "100", "percentage": "55.6%"},
        {"sectionName": "InputReader", "cycles": "160", "percentage": "88.9%"},
    ]


def test_sanitize_utilization_fields_additional_utilization_field_input() -> None:
    """Sanitize utilization attribute when a new sectionName is added."""
    op_stats = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=180,
        memory={
            "Undefined": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "Internal": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 100},
            {"sectionName": "VectorEngine", "cycles": 80},
            {"sectionName": "OutputWriter", "cycles": 65},
            {"sectionName": "VectorEngine", "cycles": 87},
            {"sectionName": "ConvolutionEngine", "cycles": 87},
            {"sectionName": "Bar", "cycles": 1},
            {"sectionName": "Bar", "cycles": 1},
            {"sectionName": "Foo", "cycles": 1},
        ],
        operators=["foo"],
    )
    op_stats.sanitize_utilization_fields()

    assert op_stats.utilization == [
        {"sectionName": "OutputWriter", "cycles": "165", "percentage": "91.7%"},
        {"sectionName": "VectorEngine", "cycles": "167", "percentage": "92.8%"},
        {"sectionName": "ConvolutionEngine", "cycles": "87", "percentage": "48.3%"},
        {"sectionName": "Bar", "cycles": "2", "percentage": "1.1%"},
        {"sectionName": "Foo", "cycles": "1", "percentage": "0.6%"},
    ]


def test_sanitize_utilization_fields_missing_keys_input() -> None:
    """Throw error when the sectionName key is missing from the utilization attibute."""
    op_stats = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"bar": "OutputWriter", "foo": 1},
        ],
        operators=["foo"],
    )

    with pytest.raises(KeyError):
        op_stats.sanitize_utilization_fields()


def test_merge_nx_operator_performance_stats() -> None:
    """Merge the performance stats of two performance operators."""
    op_stats_1 = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
            {"sectionName": "ConvolutionEngine", "cycles": 0.875},
            {"sectionName": "WeightDecoder", "cycles": 1},
            {"sectionName": "InputReader", "cycles": 1},
        ],
        operators=["foo", "bar"],
    )

    op_stats_2 = NXOperatorPerformanceStats(
        op_id=[11],
        op_cycles=5345,
        total_cycles=465,
        memory={
            "L1": {
                "readBytes": 564,
                "writeBytes": 3,
                "trafficCycles": 876,
            },
            "L2": {
                "readBytes": 65,
                "writeBytes": 87,
                "trafficCycles": 985,
            },
            "SystemCache": {
                "readBytes": 12,
                "writeBytes": 34,
                "trafficCycles": 54,
            },
            "DRAM": {
                "readBytes": 5675656,
                "writeBytes": 567567,
                "trafficCycles": 6876,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
        ],
        operators=["foo", "bar"],
    )
    op_stats_1_copy = copy.deepcopy(op_stats_1)
    op_stats_2_copy = copy.deepcopy(op_stats_2)

    op_stats_1.merge(op_stats_2_copy)
    op_stats_2.merge(op_stats_1_copy)

    assert set(op_stats_1.op_id) == set(op_stats_2.op_id) == {11, 33}
    assert op_stats_1.op_cycles == op_stats_2.op_cycles == 5360
    assert op_stats_1.total_cycles == op_stats_2.total_cycles == 483
    assert op_stats_1.memory == op_stats_2.memory
    assert op_stats_1.utilization == op_stats_2.utilization
    assert op_stats_1.operators == op_stats_2.operators


def test_merge_different_location_strings_error() -> None:
    """Throw an error when the same chain is mapped to different location strings."""
    op_stats_1 = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
            {"sectionName": "OutputWriter", "cycles": 0.65625},
            {"sectionName": "VectorEngine", "cycles": 0.875},
            {"sectionName": "ConvolutionEngine", "cycles": 0.875},
            {"sectionName": "WeightDecoder", "cycles": 1},
            {"sectionName": "InputReader", "cycles": 1},
            {"sectionName": "InputReader", "cycles": 1},
        ],
        operators=[{"foo": "bar"}],
    )

    op_stats_2 = NXOperatorPerformanceStats(
        op_id=[11],
        op_cycles=5345,
        total_cycles=465,
        memory={
            "L1": {
                "readBytes": 564,
                "writeBytes": 3,
                "trafficCycles": 876,
            },
            "L2": {
                "readBytes": 65,
                "writeBytes": 87,
                "trafficCycles": 985,
            },
            "SystemCache": {
                "readBytes": 12,
                "writeBytes": 34,
                "trafficCycles": 54,
            },
            "DRAM": {
                "readBytes": 5675656,
                "writeBytes": 567567,
                "trafficCycles": 6876,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
        ],
        operators=[{"foo": "test"}],
    )

    with pytest.raises(ValueError):
        op_stats_1.merge(op_stats_2)


def test_merge_missing_memory_name_input() -> None:
    """Throw an error when the memoryName key is missing."""
    op_stats_1 = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
            {"sectionName": "OutputWriter", "cycles": 0.65625},
            {"sectionName": "VectorEngine", "cycles": 0.875},
            {"sectionName": "ConvolutionEngine", "cycles": 0.875},
            {"sectionName": "WeightDecoder", "cycles": 1},
            {"sectionName": "InputReader", "cycles": 1},
            {"sectionName": "InputReader", "cycles": 1},
        ],
        operators=["foo", "bar"],
    )

    op_stats_2 = NXOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "wrong_key": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "cycles": 1},
            {"sectionName": "VectorEngine", "cycles": 1},
        ],
        operators=["foo", "bar"],
    )

    with pytest.raises(KeyError):
        op_stats_1.merge(op_stats_2)


def test_process_stats_per_chain(test_resources_path: Path) -> None:
    """Test that we can find all location strings."""
    debug_db_file = str(
        test_resources_path / "nx/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )
    ddb_parser = NXDebugDatabaseParser(Path(debug_db_file))
    debug_db = ddb_parser.parse_debug_database()

    perf_db_file = str(
        test_resources_path
        / "nx/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )

    pdb_parser = NXPerformanceDatabaseParser(Path(perf_db_file))
    performance_db = pdb_parser.parse_performance_database()

    performance_stats = NXPerformanceStats(debug_db, performance_db)
    performance_stats_per_chain = performance_stats.process_stats_per_chain()

    # One chain per stripe, no accumulation of statistics
    performance_stats_chain_962 = NXOperatorPerformanceStats(
        op_id=["26"],
        op_cycles=6,
        total_cycles=224,
        memory={
            "L1": {"readBytes": 0, "writeBytes": 24, "trafficCycles": 0},
            "L2": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
            "SystemCache": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
            "DRAM": {"readBytes": 256, "writeBytes": 24, "trafficCycles": 8},
        },
        utilization=[
            {"sectionName": "InputReader", "cycles": "28", "percentage": "12.5%"},
            {"sectionName": "ConvolutionEngine", "cycles": "0", "percentage": "0.0%"},
            {"sectionName": "VectorEngine", "cycles": "6", "percentage": "2.7%"},
            {"sectionName": "TransformUnit", "cycles": "6", "percentage": "2.7%"},
            {"sectionName": "WeightDecoder", "cycles": "0", "percentage": "0.0%"},
            {"sectionName": "OutputWriter", "cycles": "12", "percentage": "5.4%"},
        ],
        operators=[
            {"opLocation": ["Identity"], "opType": ["Sub"]},
            {"opLocation": ["Identity"], "opType": ["Rescale"]},
        ],
    )

    assert performance_stats_per_chain["962"] == performance_stats_chain_962

    # One chain shared by two stripes, accumulation of statistics
    # Note: the debug db was edited manually to create this scenario
    performance_stats_per_chain_668 = NXOperatorPerformanceStats(
        op_id=["12", "13"],
        op_cycles=2080,
        total_cycles=9678,
        memory={
            "L1": {"readBytes": 0, "writeBytes": 17940, "trafficCycles": 111},
            "L2": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
            "SystemCache": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
            "DRAM": {"readBytes": 412308, "writeBytes": 17940, "trafficCycles": 8838},
        },
        utilization=[
            {"sectionName": "InputReader", "cycles": "9260", "percentage": "95.7%"},
            {
                "sectionName": "ConvolutionEngine",
                "cycles": "2080",
                "percentage": "21.5%",
            },
            {"sectionName": "VectorEngine", "cycles": "624", "percentage": "6.4%"},
            {"sectionName": "TransformUnit", "cycles": "0", "percentage": "0.0%"},
            {"sectionName": "WeightDecoder", "cycles": "2144", "percentage": "22.2%"},
            {"sectionName": "OutputWriter", "cycles": "832", "percentage": "8.6%"},
        ],
        operators=[
            {"opLocation": ["model/re_lu_6/Relu"], "opType": ["Conv2D"]},
            {"opLocation": ["model/re_lu_6/Relu"], "opType": ["Rescale"]},
        ],
    )

    assert performance_stats_per_chain["668"] == performance_stats_per_chain_668


def test_track_op(test_resources_path: Path) -> None:
    """Test that we can track location strings from a chain id."""
    debug_db_file = str(
        test_resources_path / "nx/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )
    ddb_parser = NXDebugDatabaseParser(Path(debug_db_file))
    debug_db = ddb_parser.parse_debug_database()

    perf_db_file = str(
        test_resources_path
        / "nx/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )

    pdb_parser = NXPerformanceDatabaseParser(Path(perf_db_file))
    performance_db = pdb_parser.parse_performance_database()
    performance_stats = NXPerformanceStats(
        debug_db=debug_db, performance_db=performance_db
    )

    chain_op_id, api_labels, operator_types = performance_stats.track_op("26")

    assert chain_op_id == "962"
    assert api_labels == [
        ["Identity"],
        ["Identity"],
    ]
    assert operator_types == [
        ["Sub"],
        ["Rescale"],
    ]

    chain_op_id, api_labels, operator_types = performance_stats.track_op("22")
    assert chain_op_id == "678"
    assert api_labels == [
        ["model/average_pooling2d/AvgPool"],
        ["model/average_pooling2d/AvgPool"],
        ["model/dense/BiasAdd"],
    ]
    assert operator_types == [
        ["AvgPool"],
        ["Rescale"],
        ["Reshape"],
    ]


def test_track_op_multiple_chains_per_stripe() -> None:
    """Test if ValueError is raised for multiple chains per stripe."""
    debug_db = {"stripe_op_id_to_op_id": {"26": ["123", "456"]}}
    performance_db = [{"key": None}]
    performance_stats = NXPerformanceStats(
        debug_db=debug_db, performance_db=performance_db
    )

    with pytest.raises(
        ValueError, match="There should be only one chain per stripe, " "found more!"
    ):
        _ = performance_stats.track_op("26")
