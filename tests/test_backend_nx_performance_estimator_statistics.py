# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Accelerator Performance Estimator performance estimation."""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import pytest

from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXDebugDatabaseParser,
    NXPerformanceDatabaseParser,
)
from mlia.backend.nx_performance_estimator.statistics import (
    NXModelPerformanceStats,
    NXOperatorPerformanceStats,
    NXPerformanceStats,
    read_tosa_mlir_spirv_id_locations,
)
from mlia.nx_utils.vgf_debug import (
    read_vgf_spirv_id_locations,
)


def _spirv_string_instruction(result_id: int, value: str) -> list[int]:
    encoded = value.encode("utf-8") + b"\0"
    padding = b"\0" * ((4 - len(encoded) % 4) % 4)
    word_count = (len(encoded) + len(padding)) // 4
    words = list(struct.unpack(f"<{word_count}I", encoded + padding))
    word_count = 2 + len(words)
    return [(word_count << 16) | 7, result_id, *words]


def _spirv_ext_inst_import_instruction(result_id: int, value: str) -> list[int]:
    encoded = value.encode("utf-8") + b"\0"
    padding = b"\0" * ((4 - len(encoded) % 4) % 4)
    word_count = (len(encoded) + len(padding)) // 4
    words = list(struct.unpack(f"<{word_count}I", encoded + padding))
    word_count = 2 + len(words)
    return [(word_count << 16) | 11, result_id, *words]


def _spirv_ext_inst_instruction(
    result_type: int,
    result_id: int,
    set_id: int,
    instruction: int,
    operands: list[int],
) -> list[int]:
    word_count = 5 + len(operands)
    return [
        (word_count << 16) | 12,
        result_type,
        result_id,
        set_id,
        instruction,
        *operands,
    ]


def _write_vgf_with_spirv_words(path: Path, module_words: list[int]) -> None:
    path.write_bytes(
        b"VGF1\0\0\0\0" + struct.pack(f"<{len(module_words)}I", *module_words)
    )


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
            {"source_operator_ids": [], "operator_types": ["Sub"]},
            {"source_operator_ids": [], "operator_types": ["Rescale"]},
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
            {"source_operator_ids": [], "operator_types": ["Conv2D"]},
            {"source_operator_ids": [], "operator_types": ["Rescale"]},
        ],
    )

    assert performance_stats_per_chain["668"] == performance_stats_per_chain_668


def test_process_stats_aggregates_stripes_by_chain_and_cascade_independently() -> None:
    """Raw stripe stats are independently aggregated by chain and cascade ids."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["100"], "1": ["200"], "2": ["100"]},
        "stripe_op_id_to_cascade_op_id": {"0": ["300"], "1": ["300"], "2": ["400"]},
        "chain_op_id_to_fused_op_ids": {"100": ["10"], "200": ["20"]},
        "fused_op_id_to_tosa_op_ids": {"10": ["1"], "20": ["2"]},
        "tosa_op_id_to_api_labels": {
            "1": ["TOSACONV2D_spirv_id_11"],
            "2": ["TOSAADD_spirv_id_22"],
        },
        "tosa_op_id_to_tosa_op": {"1": ["Conv2D"], "2": ["Add"]},
    }
    performance_db = [
        {
            "id": 0,
            "opCycles": 10,
            "totalCycles": 100,
            "Memory": {
                "Internal": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
                "DRAM": {"readBytes": 1, "writeBytes": 2, "trafficCycles": 3},
            },
            "Utilization": [{"sectionName": "VectorEngine", "cycles": 10}],
        },
        {
            "id": 1,
            "opCycles": 20,
            "totalCycles": 200,
            "Memory": {
                "Internal": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
                "DRAM": {"readBytes": 4, "writeBytes": 5, "trafficCycles": 6},
            },
            "Utilization": [{"sectionName": "VectorEngine", "cycles": 20}],
        },
        {
            "id": 2,
            "opCycles": 30,
            "totalCycles": 300,
            "Memory": {
                "Internal": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
                "DRAM": {"readBytes": 7, "writeBytes": 8, "trafficCycles": 9},
            },
            "Utilization": [{"sectionName": "VectorEngine", "cycles": 30}],
        },
    ]

    performance_stats = NXPerformanceStats(debug_db, performance_db)

    chain_stats = performance_stats.process_stats_per_chain()
    cascade_stats = performance_stats.process_stats_per_cascade()

    assert set(chain_stats) == {"100", "200"}
    assert chain_stats["100"].op_id == ["0", "2"]
    assert chain_stats["100"].op_cycles == 40
    assert chain_stats["100"].total_cycles == 400
    assert chain_stats["100"].memory == {
        "DRAM": {"readBytes": 8, "writeBytes": 10, "trafficCycles": 12}
    }
    assert chain_stats["100"].operators == [
        {
            "source_operator_ids": ["source_operator/segment_0/spirv-11"],
            "operator_types": ["Conv2D"],
        }
    ]

    assert set(cascade_stats) == {"300", "400"}
    assert cascade_stats["300"].op_id == ["0", "1"]
    assert cascade_stats["300"].op_cycles == 30
    assert cascade_stats["300"].total_cycles == 300
    assert cascade_stats["300"].memory == {
        "DRAM": {"readBytes": 5, "writeBytes": 7, "trafficCycles": 9}
    }
    assert cascade_stats["300"].operators == [
        {
            "source_operator_ids": ["source_operator/segment_0/spirv-11"],
            "operator_types": ["Conv2D"],
        },
        {
            "source_operator_ids": ["source_operator/segment_0/spirv-22"],
            "operator_types": ["Add"],
        },
    ]


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

    chain_op_id, api_labels, operator_types, module_stacks, stack_traces = (
        performance_stats.track_op("26")
    )

    assert chain_op_id == "962"
    assert api_labels == [[None], [None]]
    assert operator_types == [
        ["Sub"],
        ["Rescale"],
    ]

    chain_op_id, api_labels, operator_types, module_stacks, stack_traces = (
        performance_stats.track_op("22")
    )
    assert chain_op_id == "678"
    assert api_labels == [[None], [None], [None]]
    assert operator_types == [
        ["AvgPool"],
        ["Rescale"],
        ["Reshape"],
    ]


def test_track_op_maps_spirv_id_labels_to_vgf_locations() -> None:
    """Test that r56 SPIR-V id labels resolve to VGF debug locations."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["chain_0"]},
        "chain_op_id_to_fused_op_ids": {"chain_0": ["fused_0"]},
        "fused_op_id_to_tosa_op_ids": {"fused_0": ["545", "635"]},
        "tosa_op_id_to_api_labels": {
            "545": ["TOSAMUL_spirv_id_716"],
            "635": ["TOSARESCALE_spirv_id_999"],
        },
        "tosa_op_id_to_tosa_op": {
            "545": ["Mul"],
            "635": ["Rescale"],
        },
    }
    performance_stats = NXPerformanceStats(debug_db=debug_db, performance_db=[])

    chain_op_id, api_labels, operator_types, _, _ = performance_stats.track_op("0")

    assert chain_op_id == "chain_0"
    assert api_labels == [
        ["source_operator/segment_0/spirv-716"],
        ["source_operator/segment_0/spirv-999"],
    ]
    assert operator_types == [["Mul"], ["Rescale"]]


def test_tosa_mlir_locations_map_to_spirv_id_placeholders(tmp_path: Path) -> None:
    """TOSA MLIR locations should fill gaps when VGF debug metadata is absent."""
    tosa_mlir = tmp_path / "model.tosamlir"
    tosa_mlir.write_text(
        """
module {
  func.func @main(%arg0: tensor<1x8xi8>) -> tensor<1x8xi8> {
    %0 = tosa.conv2d %arg0, %arg0, %arg0, %arg0, %arg0 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>) -> tensor<1x8xi32> loc(#loc1)
    %1 = tosa.clamp %arg0 {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x8xi8>) -> tensor<1x8xi8> loc(#loc1)
    %2 = tosa.rescale %0, %arg0, %arg0, %arg0, %arg0 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8xi32>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>) -> tensor<1x8xi8> loc(#loc2)
    return %2 : tensor<1x8xi8>
  }
} loc(#loc)
#loc1 = loc("model/conv"(#loc))
#loc2 = loc("model/rescale"(#loc))
""",
        encoding="utf-8",
    )
    debug_db = {
        "tosa_op_id_to_api_labels": {
            "85": ["TOSACONV2D_spirv_id_202"],
            "96": ["TOSARESCALE_spirv_id_210"],
        },
        "tosa_op_id_to_tosa_op": {
            "85": ["Conv2D"],
            "96": ["Rescale"],
        },
    }

    assert read_tosa_mlir_spirv_id_locations(tosa_mlir, debug_db) == {
        "202": "model/conv",
        "210": "model/rescale",
    }


def test_tosa_mlir_locations_sort_estimator_ops_by_tosa_id(tmp_path: Path) -> None:
    """Estimator and source op streams should be aligned in numeric TOSA id order."""
    tosa_mlir = tmp_path / "model.tosamlir"
    tosa_mlir.write_text(
        """
module {
  func.func @main(%arg0: tensor<1x8xi8>) -> tensor<1x8xi8> {
    %0 = tosa.conv2d %arg0, %arg0, %arg0, %arg0, %arg0 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>) -> tensor<1x8xi32> loc(#loc1)
    %1 = tosa.rescale %0, %arg0, %arg0, %arg0, %arg0 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8xi32>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>) -> tensor<1x8xi8> loc(#loc2)
    return %1 : tensor<1x8xi8>
  }
} loc(#loc)
#loc1 = loc("model/conv"(#loc))
#loc2 = loc("model/rescale"(#loc))
""",
        encoding="utf-8",
    )
    debug_db = {
        "tosa_op_id_to_api_labels": {
            "96": ["TOSARESCALE_spirv_id_210"],
            "85": ["TOSACONV2D_spirv_id_202"],
        },
        "tosa_op_id_to_tosa_op": {
            "96": ["Rescale"],
            "85": ["Conv2D"],
        },
    }

    assert read_tosa_mlir_spirv_id_locations(tosa_mlir, debug_db) == {
        "202": "model/conv",
        "210": "model/rescale",
    }


def test_tosa_mlir_location_fallback_uses_nearest_lcs_match(
    tmp_path: Path,
) -> None:
    """Unmatched estimator SPIR-V ids should use the nearest LCS location."""
    tosa_mlir = tmp_path / "model.tosamlir"
    tosa_mlir.write_text(
        """
module {
  func.func @main(%arg0: tensor<1x8xi8>) -> tensor<1x8xi8> {
    %0 = tosa.conv2d %arg0, %arg0, %arg0, %arg0, %arg0 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>) -> tensor<1x8xi32> loc(#loc1)
    %1 = tosa.rescale %0, %arg0, %arg0, %arg0, %arg0 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8xi32>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>) -> tensor<1x8xi8> loc(#loc2)
    return %1 : tensor<1x8xi8>
  }
} loc(#loc)
#loc1 = loc("model/conv"(#loc))
#loc2 = loc("model/rescale"(#loc))
""",
        encoding="utf-8",
    )
    debug_db = {
        "tosa_op_id_to_api_labels": {
            "85": ["TOSACONV2D_spirv_id_202"],
            "91": ["TOSACLAMP_spirv_id_205"],
            "96": ["TOSARESCALE_spirv_id_210"],
        },
        "tosa_op_id_to_tosa_op": {
            "85": ["Conv2D"],
            "91": ["Clamp"],
            "96": ["Rescale"],
        },
    }

    assert read_tosa_mlir_spirv_id_locations(tosa_mlir, debug_db) == {
        "202": "model/conv",
        "205": "model/conv",
        "210": "model/rescale",
    }


def test_track_op_converts_spirv_id_suffix_labels() -> None:
    """Test that labels ending with SPIR-V-like suffixes become source locations."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["chain_0"]},
        "chain_op_id_to_fused_op_ids": {"chain_0": ["fused_0"]},
        "fused_op_id_to_tosa_op_ids": {"fused_0": ["545"]},
        "tosa_op_id_to_api_labels": {
            "545": ["model/foo_spirv_id_716"],
        },
        "tosa_op_id_to_tosa_op": {
            "545": ["Mul"],
        },
    }
    performance_stats = NXPerformanceStats(debug_db=debug_db, performance_db=[])

    _, api_labels, _, _, _ = performance_stats.track_op("0")

    assert api_labels == [["source_operator/segment_0/spirv-716"]]


def test_process_stats_per_chain_uses_resolved_spirv_locations() -> None:
    """Test final stats contain resolved locations, not r56 placeholders."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["chain_0"]},
        "chain_op_id_to_fused_op_ids": {"chain_0": ["fused_0"]},
        "fused_op_id_to_tosa_op_ids": {"fused_0": ["545"]},
        "tosa_op_id_to_api_labels": {
            "545": ["TOSAMUL_spirv_id_716"],
        },
        "tosa_op_id_to_tosa_op": {
            "545": ["Mul"],
        },
    }
    performance_db = [
        {
            "id": 0,
            "opCycles": 10,
            "totalCycles": 20,
            "Memory": {
                "Internal": {
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                "DRAM": {
                    "readBytes": 1,
                    "writeBytes": 2,
                    "trafficCycles": 3,
                },
            },
            "Utilization": [{"sectionName": "VectorEngine", "cycles": 10}],
        }
    ]
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=performance_db,
    )

    stats = performance_stats.process_stats_per_chain()

    assert stats["chain_0"].operators == [
        {
            "source_operator_ids": ["source_operator/segment_0/spirv-716"],
            "operator_types": ["Mul"],
        }
    ]


def test_read_vgf_spirv_id_locations(tmp_path: Path) -> None:
    """Test reading SPIR-V id to location mappings from VGF debug info."""
    mlgraph_debug_import_id = 10
    location_string_id = 20
    module_words = [
        0x07230203,
        0x00010500,
        0,
        1000,
        0,
        *_spirv_ext_inst_import_instruction(
            mlgraph_debug_import_id, "NonSemantic.MLGraph.DebugInfo.1"
        ),
        *_spirv_string_instruction(location_string_id, "model/real_mul"),
        *_spirv_ext_inst_instruction(
            result_type=1,
            result_id=30,
            set_id=mlgraph_debug_import_id,
            instruction=4,
            operands=[0, location_string_id, 716, 717],
        ),
    ]
    vgf_file = tmp_path / "model.vgf"
    _write_vgf_with_spirv_words(vgf_file, module_words)

    assert read_vgf_spirv_id_locations(vgf_file) == {
        "716": "model/real_mul",
        "717": "model/real_mul",
    }


def test_read_vgf_spirv_id_locations_normalizes_json_locations(
    tmp_path: Path,
) -> None:
    """Test JSON debug locations resolve to model node names."""
    mlgraph_debug_import_id = 10
    location_string_id = 20
    json_location = json.dumps(
        {
            "node_name": "model/real_mul",
            "other_debug_metadata": "ignored",
        }
    )
    module_words = [
        0x07230203,
        0x00010500,
        0,
        1000,
        0,
        *_spirv_ext_inst_import_instruction(
            mlgraph_debug_import_id, "NonSemantic.MLGraph.DebugInfo.1"
        ),
        *_spirv_string_instruction(location_string_id, json_location),
        *_spirv_ext_inst_instruction(
            result_type=1,
            result_id=30,
            set_id=mlgraph_debug_import_id,
            instruction=4,
            operands=[0, location_string_id, 716],
        ),
    ]
    vgf_file = tmp_path / "model.vgf"
    _write_vgf_with_spirv_words(vgf_file, module_words)

    assert read_vgf_spirv_id_locations(vgf_file) == {"716": "model/real_mul"}


def test_read_vgf_spirv_id_locations_normalizes_nested_json_locations(
    tmp_path: Path,
) -> None:
    """Test nested debug-hook JSON locations resolve to model node names."""
    mlgraph_debug_import_id = 10
    location_string_id = 20
    json_location = json.dumps({"aten_info": {"node_name": "model/real_mul"}})
    module_words = [
        0x07230203,
        0x00010500,
        0,
        1000,
        0,
        *_spirv_ext_inst_import_instruction(
            mlgraph_debug_import_id, "NonSemantic.MLGraph.DebugInfo.1"
        ),
        *_spirv_string_instruction(location_string_id, json_location),
        *_spirv_ext_inst_instruction(
            result_type=1,
            result_id=30,
            set_id=mlgraph_debug_import_id,
            instruction=4,
            operands=[0, location_string_id, 716],
        ),
    ]
    vgf_file = tmp_path / "model.vgf"
    _write_vgf_with_spirv_words(vgf_file, module_words)

    assert read_vgf_spirv_id_locations(vgf_file) == {"716": "model/real_mul"}


def test_read_vgf_spirv_id_locations_reads_multiple_spirv_modules(
    tmp_path: Path,
) -> None:
    """Test reading debug locations from multiple VGF-embedded SPIR-V modules."""
    mlgraph_debug_import_id = 10
    first_location_string_id = 20
    second_location_string_id = 21
    first_module_words = [
        0x07230203,
        0x00010500,
        0,
        1000,
        0,
        *_spirv_ext_inst_import_instruction(
            mlgraph_debug_import_id, "NonSemantic.MLGraph.DebugInfo.1"
        ),
        *_spirv_string_instruction(first_location_string_id, "model/conv"),
        *_spirv_ext_inst_instruction(
            result_type=1,
            result_id=30,
            set_id=mlgraph_debug_import_id,
            instruction=4,
            operands=[0, first_location_string_id, 716],
        ),
    ]
    second_module_words = [
        0x07230203,
        0x00010500,
        0,
        1000,
        0,
        *_spirv_ext_inst_import_instruction(
            mlgraph_debug_import_id, "NonSemantic.MLGraph.DebugInfo.1"
        ),
        *_spirv_string_instruction(second_location_string_id, "model/rescale"),
        *_spirv_ext_inst_instruction(
            result_type=1,
            result_id=30,
            set_id=mlgraph_debug_import_id,
            instruction=4,
            operands=[0, second_location_string_id, 999],
        ),
    ]
    vgf_file = tmp_path / "model.vgf"
    vgf_file.write_bytes(
        b"VGF1\0\0\0\0"
        + struct.pack(f"<{len(first_module_words)}I", *first_module_words)
        + b"PADDING\0"
        + struct.pack(f"<{len(second_module_words)}I", *second_module_words)
    )

    assert read_vgf_spirv_id_locations(vgf_file) == {
        "716": "model/conv",
        "999": "model/rescale",
    }


def test_read_vgf_spirv_id_locations_fallbacks(tmp_path: Path) -> None:
    """Test unsupported or malformed VGF inputs do not produce mappings."""
    no_spirv_file = tmp_path / "no_spirv.vgf"
    no_spirv_file.write_bytes(b"VGF1\0\0\0\0")
    truncated_file = tmp_path / "truncated.vgf"
    truncated_file.write_bytes(b"VGF1\0\0\0\0" + struct.pack("<I", 0x07230203))
    invalid_instruction_file = tmp_path / "invalid_instruction.vgf"
    _write_vgf_with_spirv_words(
        invalid_instruction_file,
        [
            0x07230203,
            0x00010500,
            0,
            1000,
            0,
            (5 << 16) | 7,
            1,
        ],
    )
    missing_import_file = tmp_path / "missing_import.vgf"
    _write_vgf_with_spirv_words(
        missing_import_file,
        [
            0x07230203,
            0x00010500,
            0,
            1000,
            0,
            *_spirv_string_instruction(20, "model/real_mul"),
            *_spirv_ext_inst_instruction(
                result_type=1,
                result_id=30,
                set_id=10,
                instruction=4,
                operands=[0, 20, 716],
            ),
        ],
    )

    assert read_vgf_spirv_id_locations(no_spirv_file) == {}
    assert read_vgf_spirv_id_locations(truncated_file) == {}
    assert read_vgf_spirv_id_locations(invalid_instruction_file) == {}
    assert read_vgf_spirv_id_locations(missing_import_file) == {}


def test_track_op_multiple_chains_per_stripe() -> None:
    """Test if ValueError is raised for multiple chains per stripe."""
    debug_db = {"stripe_op_id_to_op_id": {"26": ["123", "456"]}}
    performance_db = [{"key": None}]
    performance_stats = NXPerformanceStats(
        debug_db=debug_db, performance_db=performance_db
    )

    with pytest.raises(
        ValueError, match="There should be only one chain per stripe, found more!"
    ):
        _ = performance_stats.track_op("26")


def test_nx_model_performance_stats(tmp_path: Path) -> None:
    """Test NXModelPerformanceStats class."""
    model_performance = {
        "compiled_size": {"unit": "bytes", "value": 17180},
        "network_performance": {
            "cache1": {
                "cycles": {"unit": "cc", "value": 2},
                "read_bytes": {"unit": "bytes", "value": 0},
                "write_bytes": {"unit": "bytes", "value": 512},
            },
            "compute_cycles": {"unit": "cc", "value": 5520},
            "dram": {
                "cycles": {"unit": "cc", "value": 7838},
                "read_bytes": {"unit": "bytes", "value": 302428},
                "write_bytes": {"unit": "bytes", "value": 23808},
            },
            "dram_footprint": {"unit": "bytes", "value": 14784},
            "inference_time": {"unit": "ms", "value": 0.008937333710491657},
            "infs_per_sec": {"unit": "inf/s", "value": 111890.1953125},
            "total_cycles": {"unit": "cc", "value": 13406},
        },
    }

    json_path = tmp_path / "model_perf.json"
    with open(json_path, mode="w", encoding="utf-8") as file:
        json.dump(model_performance, file)

    model_performance_statistics = NXModelPerformanceStats.read_from_json(json_path)

    assert model_performance_statistics.compiled_size == 17180
    assert model_performance_statistics.cache_cycles == 2
    assert model_performance_statistics.cache_read_bytes == 0
    assert model_performance_statistics.cache_write_bytes == 512
    assert model_performance_statistics.compute_cycles == 5520
    assert model_performance_statistics.dram_cycles == 7838
    assert model_performance_statistics.dram_read_bytes == 302428
    assert model_performance_statistics.dram_write_bytes == 23808
    assert model_performance_statistics.dram_footprint == 14784
    assert model_performance_statistics.inference_time == 0.008937333710491657
    assert model_performance_statistics.infs_per_sec == 111890.1953125
    assert model_performance_statistics.total_cycles == 13406


def test_nx_model_performance_stats_preserves_null_values(tmp_path: Path) -> None:
    """Test that null backend metric values are preserved for output mapping."""
    model_performance = {
        "compiled_size": {"unit": "bytes", "value": 17180},
        "network_performance": {
            "cache1": {
                "cycles": {"unit": "cc", "value": None},
                "read_bytes": {"unit": "bytes", "value": 0},
                "write_bytes": {"unit": "bytes", "value": 512},
            },
            "compute_cycles": {"unit": "cc", "value": 5520},
            "dram": {
                "cycles": {"unit": "cc", "value": 7838},
                "read_bytes": {"unit": "bytes", "value": 302428},
                "write_bytes": {"unit": "bytes", "value": 23808},
            },
            "dram_footprint": {"unit": "bytes", "value": 14784},
            "inference_time": {"unit": "ms", "value": 0.008937333710491657},
            "infs_per_sec": {"unit": "inf/s", "value": None},
            "total_cycles": {"unit": "cc", "value": 13406},
        },
    }

    json_path = tmp_path / "model_perf.json"
    with open(json_path, mode="w", encoding="utf-8") as file:
        json.dump(model_performance, file)

    model_performance_statistics = NXModelPerformanceStats.read_from_json(json_path)

    assert model_performance_statistics.cache_cycles is None
    assert model_performance_statistics.infs_per_sec is None


def test_nx_model_performance_stats_preserves_domain_types(tmp_path: Path) -> None:
    """Test that parsed count and float metrics keep their domain types."""
    model_performance = {
        "compiled_size": {"unit": "bytes", "value": 17180.0},
        "network_performance": {
            "cache1": {
                "cycles": {"unit": "cc", "value": 2.0},
                "read_bytes": {"unit": "bytes", "value": 0.0},
                "write_bytes": {"unit": "bytes", "value": 512.0},
            },
            "compute_cycles": {"unit": "cc", "value": 5520.0},
            "dram": {
                "cycles": {"unit": "cc", "value": 7838.0},
                "read_bytes": {"unit": "bytes", "value": 302428.0},
                "write_bytes": {"unit": "bytes", "value": 23808.0},
            },
            "dram_footprint": {"unit": "bytes", "value": 14784.0},
            "inference_time": {"unit": "ms", "value": 1},
            "infs_per_sec": {"unit": "inf/s", "value": 111890},
            "total_cycles": {"unit": "cc", "value": 13406.0},
        },
    }

    json_path = tmp_path / "model_perf.json"
    with open(json_path, mode="w", encoding="utf-8") as file:
        json.dump(model_performance, file)

    model_performance_statistics = NXModelPerformanceStats.read_from_json(json_path)

    assert model_performance_statistics.compiled_size == 17180
    assert isinstance(model_performance_statistics.compiled_size, int)
    assert model_performance_statistics.cache_cycles == 2
    assert isinstance(model_performance_statistics.cache_cycles, int)
    assert model_performance_statistics.inference_time == 1.0
    assert isinstance(model_performance_statistics.inference_time, float)
    assert model_performance_statistics.infs_per_sec == 111890.0
    assert isinstance(model_performance_statistics.infs_per_sec, float)


def test_nx_model_performance_stats_rejects_non_numeric_metric_value(
    tmp_path: Path,
) -> None:
    """Test that unsupported backend metric value types fail at parse time."""
    model_performance = {
        "compiled_size": {"unit": "bytes", "value": "17180"},
        "network_performance": {
            "compute_cycles": {"unit": "cc", "value": 5520},
            "dram_footprint": {"unit": "bytes", "value": 14784},
            "inference_time": {"unit": "ms", "value": 0.008937333710491657},
            "infs_per_sec": {"unit": "inf/s", "value": 111890.1953125},
            "total_cycles": {"unit": "cc", "value": 13406},
        },
    }

    json_path = tmp_path / "model_perf.json"
    with open(json_path, mode="w", encoding="utf-8") as file:
        json.dump(model_performance, file)

    with pytest.raises(
        TypeError,
        match=(
            "Expected integer or null value for model performance metric "
            "'compiled_size', got str."
        ),
    ):
        NXModelPerformanceStats.read_from_json(json_path)


def test_nx_model_performance_stats_rejects_fractional_count_value(
    tmp_path: Path,
) -> None:
    """Test that fractional values fail for count-like backend metrics."""
    model_performance = {
        "compiled_size": {"unit": "bytes", "value": 17180.5},
        "network_performance": {
            "compute_cycles": {"unit": "cc", "value": 5520},
            "dram_footprint": {"unit": "bytes", "value": 14784},
            "inference_time": {"unit": "ms", "value": 0.008937333710491657},
            "infs_per_sec": {"unit": "inf/s", "value": 111890.1953125},
            "total_cycles": {"unit": "cc", "value": 13406},
        },
    }

    json_path = tmp_path / "model_perf.json"
    with open(json_path, mode="w", encoding="utf-8") as file:
        json.dump(model_performance, file)

    with pytest.raises(
        TypeError,
        match=(
            "Expected integer or null value for model performance metric "
            "'compiled_size', got non-integral float."
        ),
    ):
        NXModelPerformanceStats.read_from_json(json_path)


def test_track_op_converts_vgf_spirv_api_labels_to_source_locations() -> None:
    """VGF/SPIR-V GCPE labels are converted to stable source locations."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": ["TOSACONV2D_spirv_id_60"]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    performance_stats = NXPerformanceStats(debug_db=debug_db, performance_db=[])

    chain_op_id, api_labels, operator_types, module_stacks, stack_traces = (
        performance_stats.track_op("0")
    )

    assert chain_op_id == "72"
    assert api_labels == [["source_operator/segment_0/spirv-60"]]
    assert operator_types == [["Conv2D"]]
    assert module_stacks == [[]]
    assert stack_traces == [[]]


def test_track_op_maps_unique_debug_label_back_to_vgf_spirv_location() -> None:
    """Unique Graph DebugInfo labels recover canonical VGF/SPIR-V locations."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": ["model/re_lu_6/Relu"]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"60": "model/re_lu_6/Relu"},
        debug_name_to_spirv_ids={"model/re_lu_6/Relu": ["60"]},
    )
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=[],
        segment_index=7,
        debug_names=debug_names,
    )

    chain_op_id, api_labels, operator_types, module_stacks, stack_traces = (
        performance_stats.track_op("0")
    )

    assert chain_op_id == "72"
    assert api_labels == [["source_operator/segment_7/spirv-60"]]
    assert operator_types == [["Conv2D"]]
    assert module_stacks == [[]]
    assert stack_traces == [[]]


def test_track_op_omits_unmatched_debug_label_location() -> None:
    """Unmatched debug labels cannot safely recover a SPIR-V result id."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": ["model/re_lu_6/Relu"]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"60": "other"},
        debug_name_to_spirv_ids={"other": ["60"]},
    )
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=[],
        debug_names=debug_names,
    )

    _, api_labels, _, module_stacks, stack_traces = performance_stats.track_op("0")

    assert api_labels == [[None]]
    assert module_stacks == [[]]
    assert stack_traces == [[]]


def test_track_op_omits_ambiguous_debug_label_location() -> None:
    """Ambiguous debug labels cannot safely recover a SPIR-V result id."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": ["bob"]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"60": "bob", "61": "bob"},
        debug_name_to_spirv_ids={"bob": ["60", "61"]},
    )
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=[],
        debug_names=debug_names,
    )

    _, api_labels, _, module_stacks, stack_traces = performance_stats.track_op("0")

    assert api_labels == [[None]]
    assert module_stacks == [[]]
    assert stack_traces == [[]]


def test_track_op_rejects_fallback_spirv_id_absent_from_vgf() -> None:
    """Fallback labels cannot invent an operator outside the supplied VGF."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": ["TOSACONV2D_spirv_id_60"]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=[],
        debug_names=SpirvDebugNameMap(),
    )

    with pytest.raises(ValueError, match="not present in supplied VGF segment 0"):
        performance_stats.track_op("0")


def test_track_op_extracts_nn_module_stack_from_executorch_json_label() -> None:
    """ExecuTorch debug JSON exposes nn_module_stack, but not source locations."""
    debug_label = json.dumps(
        {
            "aten_info": {"node_name": "aten_convolution_default"},
            "torch_info": {
                "nn_module_stack": {
                    "L['model']": ("model", "Model"),
                    "L['model'].features.0": ("model.features.0", "Conv2d"),
                }
            },
        }
    )
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": [debug_label]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"60": debug_label},
        debug_name_to_spirv_ids={debug_label: ["60"]},
    )
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=[],
        debug_names=debug_names,
    )

    _, api_labels, operator_types, module_stacks, stack_traces = (
        performance_stats.track_op("0")
    )

    assert api_labels == [["source_operator/segment_0/spirv-60"]]
    assert operator_types == [["Conv2D"]]
    assert module_stacks == [
        [
            [
                {"tracer_key": "<root>", "name": "<root>"},
                {
                    "tracer_key": "L['model']",
                    "name": "model",
                    "module_type": "Model",
                },
                {
                    "tracer_key": "L['model'].features.0",
                    "name": "model.features.0",
                    "module_type": "Conv2d",
                },
            ]
        ]
    ]
    assert stack_traces == [[]]


def test_track_op_preserves_divergent_nn_module_stacks() -> None:
    """Multiple JSON labels should remain independent module-stack paths."""
    first_label = json.dumps(
        {
            "aten_info": {"node_name": "aten_convolution_default"},
            "torch_info": {
                "nn_module_stack": {
                    "L['model']": ("model", "Model"),
                    "L['model'].left": ("model.left", "Conv2d"),
                }
            },
        }
    )
    second_label = json.dumps(
        {
            "aten_info": {"node_name": "aten_relu_default"},
            "torch_info": {
                "nn_module_stack": {
                    "L['model']": ("model", "Model"),
                    "L['model'].right": ("model.right", "Relu"),
                }
            },
        }
    )
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": [first_label, second_label]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    performance_stats = NXPerformanceStats(debug_db=debug_db, performance_db=[])

    _, _, _, module_stacks, stack_traces = performance_stats.track_op("0")

    assert module_stacks == [
        [
            [
                {"tracer_key": "<root>", "name": "<root>"},
                {"tracer_key": "L['model']", "name": "model", "module_type": "Model"},
                {
                    "tracer_key": "L['model'].left",
                    "name": "model.left",
                    "module_type": "Conv2d",
                },
            ],
            [
                {"tracer_key": "<root>", "name": "<root>"},
                {"tracer_key": "L['model']", "name": "model", "module_type": "Model"},
                {
                    "tracer_key": "L['model'].right",
                    "name": "model.right",
                    "module_type": "Relu",
                },
            ],
        ]
    ]
    assert stack_traces == [[]]


def test_track_op_extracts_stack_trace_from_executorch_json_label() -> None:
    """ExecuTorch debug JSON stack_trace entries become Python frame records."""
    debug_label = json.dumps(
        {
            "aten_info": {"node_name": "aten_convolution_default"},
            "torch_info": {
                "stack_trace": [
                    '  File "E:\\XPK\\OPPO\\IFNet_HDv3.py", line 156, in forward',
                    "    return self.block(x)",
                    '  File "/tmp/project/model.py", line 214, in call',
                    "",
                    "No stack trace available",
                    "malformed",
                ]
            },
        }
    )
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": [debug_label]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    performance_stats = NXPerformanceStats(debug_db=debug_db, performance_db=[])

    _, _, _, module_stacks, stack_traces = performance_stats.track_op("0")

    assert module_stacks == [[]]
    assert stack_traces == [
        [
            [
                {
                    "file": "E:\\XPK\\OPPO\\IFNet_HDv3.py",
                    "line": "156",
                    "function": "forward",
                },
                {"file": "/tmp/project/model.py", "line": "214", "function": "call"},
            ]
        ]
    ]


def test_process_stats_per_stripe_includes_stack_trace_metadata() -> None:
    """Python stack traces are attached to operator records."""
    debug_label = json.dumps(
        {
            "torch_info": {
                "stack_trace": [
                    '  File "E:\\XPK\\OPPO\\IFNet_HDv3.py", line 156, in forward'
                ]
            }
        }
    )
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": [debug_label]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    performance_db = [
        {
            "id": 0,
            "opCycles": 10,
            "totalCycles": 20,
            "Memory": {
                "Internal": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
                "DRAM": {"readBytes": 1, "writeBytes": 2, "trafficCycles": 3},
            },
            "Utilization": [],
        }
    ]
    performance_stats = NXPerformanceStats(
        debug_db=debug_db, performance_db=performance_db
    )

    stripe_stats = performance_stats.process_stats_per_stripe()["0"]

    assert stripe_stats.operators == [
        {
            "source_operator_ids": [],
            "operator_types": ["Conv2D"],
            "code_stacks": [
                [
                    {
                        "file": "E:\\XPK\\OPPO\\IFNet_HDv3.py",
                        "line": "156",
                        "function": "forward",
                    }
                ]
            ],
        }
    ]


def test_track_op_preserves_empty_api_label_list() -> None:
    """No GCPE API labels should remain an empty location list."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": []},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=[],
        debug_names=SpirvDebugNameMap(
            spirv_id_to_debug_name={"60": "model/re_lu_6/Relu"},
            debug_name_to_spirv_ids={"model/re_lu_6/Relu": ["60"]},
        ),
    )

    _, api_labels, operator_types, module_stacks, stack_traces = (
        performance_stats.track_op("0")
    )

    assert api_labels == [[]]
    assert operator_types == [["Conv2D"]]
    assert module_stacks == [[]]
    assert stack_traces == [[]]


def test_process_stats_per_stripe_filters_unresolved_locations() -> None:
    """Unresolved debug labels are omitted from operator locations."""
    debug_db = {
        "stripe_op_id_to_op_id": {"0": ["72"]},
        "chain_op_id_to_fused_op_ids": {"72": ["52"]},
        "fused_op_id_to_tosa_op_ids": {"52": ["45"]},
        "tosa_op_id_to_api_labels": {"45": ["ambiguous"]},
        "tosa_op_id_to_tosa_op": {"45": ["Conv2D"]},
    }
    performance_db = [
        {
            "id": 0,
            "opCycles": 10,
            "totalCycles": 20,
            "Memory": {
                "Internal": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
                "DRAM": {"readBytes": 1, "writeBytes": 2, "trafficCycles": 3},
            },
            "Utilization": [],
        }
    ]
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"60": "ambiguous", "61": "ambiguous"},
        debug_name_to_spirv_ids={"ambiguous": ["60", "61"]},
    )
    performance_stats = NXPerformanceStats(
        debug_db=debug_db,
        performance_db=performance_db,
        debug_names=debug_names,
    )

    stripe_stats = performance_stats.process_stats_per_stripe()["0"]

    assert stripe_stats.operators == [
        {"source_operator_ids": [], "operator_types": ["Conv2D"]}
    ]
