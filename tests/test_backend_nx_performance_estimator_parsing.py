# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Accelerator Performance Estimator performance estimation."""

from __future__ import annotations

import csv
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any

import pytest

from mlia.backend.nx_performance_estimator.output_parsing import (
    NXDebugDatabaseParser,
    NXOutputParser,
    NXPerformanceDatabaseParser,
    SubtableColumnParser,
)


def test_load(test_resources_path: Path) -> None:
    """Load a file into a Neural Accelerator output parser."""
    perf_db_file = str(
        test_resources_path
        / "nx/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )
    debug_db_file = str(
        test_resources_path / "nx/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )
    parser = NXOutputParser()
    loaded_perf_db = parser.load(Path(perf_db_file))
    assert loaded_perf_db == parser.raw_xmlish
    loaded_debug_db = parser.load(Path(debug_db_file))
    assert loaded_debug_db == parser.raw_xmlish


def test_get_csv_reader() -> None:
    """Read string into csv."""
    contents = (
        '"id", "opCycles", "totalCycles", '
        '"memoryName;readBytes;writeBytes;trafficCycles", '
        '"sectionName;cycles"\n'
        "26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;"
        "SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;"
        "VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;"
        "TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;"
        "InputReader;0.0625;InputReader;0.25;\n"
        "25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;"
        "SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;"
        "VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;"
        "VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;\n"
    ).strip()
    parser = NXOutputParser()
    csv_reader = parser.get_csv_reader(table_data=contents)
    expected_csv_reader = csv.reader(contents.splitlines())

    for actual, expected in zip(csv_reader, expected_csv_reader):
        assert actual == expected


def test_get_csv_headers() -> None:
    """Extract the headers from a csv reader."""
    contents = (
        '"id", "opCycles", "totalCycles", '
        '"memoryName;readBytes;writeBytes;trafficCycles", '
        '"sectionName;cycles"\n'
        "26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;"
        "SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;"
        "VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;"
        "TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;"
        "InputReader;0.0625;InputReader;0.25;\n"
        "25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;"
        "SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;"
        "VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;"
        "VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;\n"
    ).strip()
    parser = NXOutputParser()
    csv_reader = parser.get_csv_reader(table_data=contents)
    csv_headers = parser.get_csv_headers(csv_reader=csv_reader)
    assert csv_headers == [
        "id",
        "opCycles",
        "totalCycles",
        "memoryName;readBytes;writeBytes;trafficCycles",
        "sectionName;cycles",
    ]


# mypy: disable-error-code=misc
@pytest.mark.parametrize(
    "variant",
    [
        "'field' ",
        '  "field" ',
        ' "field"',
        "field",
    ],
)
def test_extract_field(variant: str) -> None:
    """Test extract field."""
    parser = NXOutputParser()
    assert "field" == parser.extract_field(variant)


def test_extract_cdata() -> None:
    """Test util method to extract cdata."""
    parser = NXOutputParser()

    with pytest.raises(RuntimeError, match="No single CDATA section"):
        parser.extract_cdata("<![CDATA[a]]>....<![CDATA[b]]>")

    with pytest.raises(RuntimeError, match="No single CDATA section"):
        parser.extract_cdata("foo")

    assert "abc" == parser.extract_cdata(
        """line1
    line2
    <![CDATA[
    abc
    ]]>
    line3
    """
    )


def test_performance_database_parser_from_file(test_resources_path: Path) -> None:
    """Parse the whole file."""
    perf_db_file = str(
        test_resources_path
        / "nx/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )
    parser = NXPerformanceDatabaseParser(db_path=Path(perf_db_file))
    records = parser.parse_performance_database()
    assert len(records) == 38

    assert records[14] == {
        "Memory": {
            "Internal": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
            "L1": {"readBytes": 0, "writeBytes": 8280, "trafficCycles": 51},
            "L2": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
            "SystemCache": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0},
            "DRAM": {"readBytes": 24740, "writeBytes": 8280, "trafficCycles": 678},
        },
        "Utilization": [
            {"cycles": 573, "sectionName": "InputReader"},
            {"cycles": 247, "sectionName": "ConvolutionEngine"},
            {"cycles": 288, "sectionName": "VectorEngine"},
            {"cycles": 0, "sectionName": "TransformUnit"},
            {"cycles": 68, "sectionName": "WeightDecoder"},
            {"cycles": 384, "sectionName": "OutputWriter"},
        ],
        "id": 7,
        "opCycles": 288,
        "totalCycles": 1496,
    }


def test_register_sub_table() -> None:
    """Add a subtable to the performance db parser."""
    parser = NXPerformanceDatabaseParser()
    column_parsers = parser.register_sub_table(
        title="foo", header="bar", key_field="test"
    )
    assert column_parsers == parser.column_parsers


def test_parse_performance_database() -> None:
    """Testing with a CDATA xml body."""
    contents = (
        "<![CDATA[\n"
        '"id", "opCycles", "totalCycles", '
        '"memoryName;readBytes;writeBytes;trafficCycles", '
        '"sectionName;cycles"\n'
        "26, 18, 212, "
        "Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;"
        "SystemCache;0;0;0;DRAM;320;12;10;, "
        "OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;"
        "TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;"
        "InputReader;0.0625;InputReader;0.25;\n"
        "25, 4, 13, "
        "Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;"
        "SystemCache;0;0;0;DRAM;128;4;4;, "
        "OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;"
        "VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;"
        "InputReader;0.0625;\n"
        "]]>"
    )
    pdb_parser = NXPerformanceDatabaseParser()
    pdb_parser.raw_xmlish = contents

    assert pdb_parser.parse_performance_database() == [
        {
            "Memory": {
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
                "L1": {
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                "L2": {
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                "SystemCache": {
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                "DRAM": {
                    "readBytes": 320,
                    "writeBytes": 12,
                    "trafficCycles": 10,
                },
            },
            "Utilization": [
                {"cycles": 1, "sectionName": "OutputWriter"},
                {"cycles": 0.25, "sectionName": "VectorEngine"},
                {"cycles": 0.25, "sectionName": "VectorEngine"},
                {"cycles": 0.25, "sectionName": "VectorEngine"},
                {"cycles": 0.25, "sectionName": "TransformUnit"},
                {"cycles": 0.25, "sectionName": "TransformUnit"},
                {"cycles": 0.0625, "sectionName": "InputReader"},
                {"cycles": 0.0625, "sectionName": "InputReader"},
                {"cycles": 0.25, "sectionName": "InputReader"},
            ],
            "id": 26,
            "opCycles": 18,
            "totalCycles": 212,
        },
        {
            "Memory": {
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
                "L1": {
                    "readBytes": 0,
                    "writeBytes": 4,
                    "trafficCycles": 0,
                },
                "L2": {
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                "SystemCache": {
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                "DRAM": {
                    "readBytes": 128,
                    "writeBytes": 4,
                    "trafficCycles": 4,
                },
            },
            "Utilization": [
                {"cycles": 0.0625, "sectionName": "OutputWriter"},
                {"cycles": 0.125, "sectionName": "VectorEngine"},
                {"cycles": 0.125, "sectionName": "VectorEngine"},
                {"cycles": 0.125, "sectionName": "VectorEngine"},
                {"cycles": 0.125, "sectionName": "VectorEngine"},
                {"cycles": 0.0625, "sectionName": "InputReader"},
                {"cycles": 0.0625, "sectionName": "InputReader"},
            ],
            "id": 25,
            "opCycles": 4,
            "totalCycles": 13,
        },
    ]


@pytest.mark.parametrize(
    "contents, expected_result",
    [
        (
            (
                '"id", "opCycles", "totalCycles", '
                '"memoryName;readBytes;writeBytes;trafficCycles", '
                '"sectionName;cycles"\n'
                "26, 18, 212, "
                "Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;"
                "SystemCache;0;0;0;DRAM;320;12;10;, "
                "OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;"
                "VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;"
                "InputReader;0.0625;InputReader;0.0625;InputReader;0.25;\n"
                "25, 4, 13, "
                "Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;"
                "SystemCache;0;0;0;DRAM;128;4;4;, "
                "OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;"
                "VectorEngine;0.125;VectorEngine;0.125;"
                "InputReader;0.0625;InputReader;0.0625;\n"
            ).strip(),
            [
                {
                    "id": 26,
                    "opCycles": 18,
                    "totalCycles": 212,
                    "Memory": {
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
                        "L1": {
                            "readBytes": 0,
                            "writeBytes": 0,
                            "trafficCycles": 0,
                        },
                        "L2": {
                            "readBytes": 0,
                            "writeBytes": 0,
                            "trafficCycles": 0,
                        },
                        "SystemCache": {
                            "readBytes": 0,
                            "writeBytes": 0,
                            "trafficCycles": 0,
                        },
                        "DRAM": {
                            "readBytes": 320,
                            "writeBytes": 12,
                            "trafficCycles": 10,
                        },
                    },
                    "Utilization": [
                        {"sectionName": "OutputWriter", "cycles": 1},
                        {"sectionName": "VectorEngine", "cycles": 0.25},
                        {"sectionName": "VectorEngine", "cycles": 0.25},
                        {"sectionName": "VectorEngine", "cycles": 0.25},
                        {"sectionName": "TransformUnit", "cycles": 0.25},
                        {"sectionName": "TransformUnit", "cycles": 0.25},
                        {"sectionName": "InputReader", "cycles": 0.0625},
                        {"sectionName": "InputReader", "cycles": 0.0625},
                        {"sectionName": "InputReader", "cycles": 0.25},
                    ],
                },
                {
                    "id": 25,
                    "opCycles": 4,
                    "totalCycles": 13,
                    "Memory": {
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
                        "L1": {
                            "readBytes": 0,
                            "writeBytes": 4,
                            "trafficCycles": 0,
                        },
                        "L2": {
                            "readBytes": 0,
                            "writeBytes": 0,
                            "trafficCycles": 0,
                        },
                        "SystemCache": {
                            "readBytes": 0,
                            "writeBytes": 0,
                            "trafficCycles": 0,
                        },
                        "DRAM": {
                            "readBytes": 128,
                            "writeBytes": 4,
                            "trafficCycles": 4,
                        },
                    },
                    "Utilization": [
                        {"sectionName": "OutputWriter", "cycles": 0.0625},
                        {"sectionName": "VectorEngine", "cycles": 0.125},
                        {"sectionName": "VectorEngine", "cycles": 0.125},
                        {"sectionName": "VectorEngine", "cycles": 0.125},
                        {"sectionName": "VectorEngine", "cycles": 0.125},
                        {"sectionName": "InputReader", "cycles": 0.0625},
                        {"sectionName": "InputReader", "cycles": 0.0625},
                    ],
                },
            ],
        ),
        (
            """
            "id", "opCycles"
            26, "text"
            25, 4;
            """.strip(),
            [
                {"id": 26, "opCycles": "Invalid:text"},
                {"id": 25, "opCycles": "Invalid:4;"},
            ],
        ),
    ],
)
def test_make_parsed_db_performance_db(contents: str, expected_result: Any) -> None:
    """
    Test the performance_db has the required fields.
    """
    parser = NXPerformanceDatabaseParser()
    reader = parser.get_csv_reader(table_data=contents)
    headers = parser.get_csv_headers(csv_reader=reader)
    int_column_parsers = parser.set_column_parsers(headers=headers, content_type=int)
    parser.make_parsed_db(
        csv_reader=reader, headers=headers, column_parsers=int_column_parsers
    )

    assert parser.performance_db == expected_result


def test_debug_database_parser_from_file(test_resources_path: Path) -> None:
    """Parse the whole file."""
    debug_db_file = str(
        test_resources_path / "nx/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )
    parser = NXDebugDatabaseParser(Path(debug_db_file))
    records = parser.parse_debug_database()
    assert len(records) == 9
    assert records["fused_op_id_to_tosa_op_ids"]["584"] == ["372"]
    assert records["fused_op_id_to_tosa_op_ids"]["532"] == ["426", "500"]
    assert records["chain_op_id_to_fused_op_ids"]["690"] == [
        "642",
        "644",
        "562",
        "564",
        "568",
    ]
    assert records["tosa_op_id_to_api_labels"]["372"] == ["model/re_lu/Relu"]


def test_parse_debug_database() -> None:
    """Test the debug database has the required key-value pairs."""
    contents = (
        "<?xml version='1.0' encoding='utf-8' ?>\n"
        '<![CDATA[\n"id", "api_id"\n]]>\n'
        '</table>\n<table name="fused_op_id">\n'
        '<![CDATA[\n"id", "tosa_op_ids"\n'
        "531, 334;\n557, 335;\n499, 394;462;;\n]]>\n"
        '</table>\n<table name="chain_op_id">\n'
        '<![CDATA[\n"id", "fused_op_ids"\n'
        "603, 531;557;\n605, 533;559;\n607, 535;561;\n"
        "637, 589;591;509;511;515;\n]]>\n"
        '<table name="stripe_op_id">\n'
        '<![CDATA[\n"id", "chain_op_id", "cascade_op_id"\n'
        "0, 603, 1693;\n1, 605, 1691;\n]]>\n"
        "</table>\n</debug>"
    )
    parser = NXDebugDatabaseParser()
    parser.raw_xmlish = contents
    records = parser.parse_debug_database()
    print(records)
    assert len(records) == 4
    assert records["fused_op_id_to_tosa_op_ids"]["531"] == ["334"]
    assert records["fused_op_id_to_tosa_op_ids"]["499"] == ["394", "462"]
    assert records["chain_op_id_to_fused_op_ids"]["637"] == [
        "589",
        "591",
        "509",
        "511",
        "515",
    ]
    assert records["stripe_op_id_to_chain_op_id"]["0"] == ["603"]
    assert records["stripe_op_id_to_cascade_op_id"]["0"] == ["1693"]


def test_known_api_labels_do_not_affect_other_third_columns() -> None:
    """Known labels are used only when parsing an api_labels column."""
    contents = (
        "<debug>\n"
        '<table name="tosa_op_id">\n'
        '<![CDATA[\n"id", "tosa_op", "api_labels"\n'
        "62, Transpose, known;label;\n"
        "]]>\n"
        "</table>\n"
        '<table name="stripe_op_id">\n'
        '<![CDATA[\n"id", "op_id", "cascade_op_id"\n'
        "0, 1178, 1672;\n"
        "]]>\n"
        "</table>\n"
        "</debug>"
    )
    parser = NXDebugDatabaseParser(known_api_labels=["known;label"])
    parser.raw_xmlish = contents

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_api_labels"]["62"] == ["known;label"]
    assert records["stripe_op_id_to_op_id"]["0"] == ["1178"]
    assert records["stripe_op_id_to_cascade_op_id"]["0"] == ["1672"]


def test_parse_debug_database_uses_empty_list_for_blank_api_labels() -> None:
    """Blank api_labels cells mean the op has no source location."""
    contents = (
        "<debug>\n"
        '<table name="tosa_op_id">\n'
        '<![CDATA[\n"id", "tosa_op", "api_labels"\n'
        "878, Reinterleave, \n"
        "879, Conv2D, TOSACONV2D_spirv_id_411;\n"
        "]]>\n"
        "</table>\n"
        "</debug>"
    )
    parser = NXDebugDatabaseParser()
    parser.raw_xmlish = contents

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_tosa_op"] == {
        "878": ["Reinterleave"],
        "879": ["Conv2D"],
    }
    assert records["tosa_op_id_to_api_labels"] == {
        "878": [],
        "879": ["TOSACONV2D_spirv_id_411"],
    }


def test_parse_debug_database_invalid_num_db_headers() -> None:
    """Test error is raised if the debug database has too many headers."""
    contents = (
        "<?xml version='1.0' encoding='utf-8' ?>\n"
        '<![CDATA[\n"id", "api_id"\n]]>\n'
        '</table>\n<table name="fused_op_id">\n'
        '<![CDATA[\n"id", "api_id", "tosa_op_ids", '
        '"fused_op_ids"\n'
        "531, 334;\n557, 335;\n499, 394;462;;\n]]>\n"
        '</table>\n<table name="chain_op_id">\n'
        "</table>\n</debug>"
    )
    parser = NXDebugDatabaseParser()
    parser.raw_xmlish = contents
    with pytest.raises(RuntimeError, match="Unsupported number of headers"):
        parser.parse_debug_database()


def test_make_parsed_db_debug_db() -> None:
    """Test the debug database has the required key-value pairs."""
    contents = (
        "<?xml version='1.0' encoding='utf-8' ?>\n"
        '<![CDATA[\n"id", "api_id"\n]]>\n'
        '</table>\n<table name="fused_op_id">\n'
        '<![CDATA[\n"id", "tosa_op_ids"\n'
        "531, 334;\n557, 335;\n499, 394;462;;\n]]>\n"
        '</table>\n<table name="chain_op_id">\n'
        '<![CDATA[\n"id", "fused_op_ids"\n'
        "603, 531;557;\n605, 533;559;\n607, 535;561;\n"
        "637, 589;591;509;511;515;\n]]>\n"
        '<table name="stripe_op_id">\n'
        '<![CDATA[\n"id", "chain_op_id", "cascade_op_id"\n'
        "0, 603, 1693;\n1, 605, 1691;\n]]>\n"
        "</table>\n</debug>"
    )
    parser = NXDebugDatabaseParser()
    table_elements = contents.split('<table name="')[1:]
    for table_element in table_elements:
        table_name = table_element.split('">')[0]
        table_data = parser.extract_cdata(table_element)
        reader = parser.get_csv_reader(table_data=table_data)
        headers = parser.get_csv_headers(csv_reader=reader)
        headers[0] = table_name + "_to_" + headers[1]
        parser.make_parsed_db(csv_reader=reader, headers=headers)

    assert len(parser.debug_db) == 4
    assert parser.debug_db["fused_op_id_to_tosa_op_ids"]["531"] == ["334"]
    assert parser.debug_db["fused_op_id_to_tosa_op_ids"]["499"] == ["394", "462"]
    assert parser.debug_db["chain_op_id_to_fused_op_ids"]["637"] == [
        "589",
        "591",
        "509",
        "511",
        "515",
    ]
    assert parser.debug_db["stripe_op_id_to_chain_op_id"]["0"] == ["603"]


def test_make_parsed_db_debug_db_invalid_num_headers() -> None:
    """Test error is raised if the debug database has too many headers."""
    contents = (
        "<?xml version='1.0' encoding='utf-8' ?>\n"
        '<![CDATA[\n"id", "api_id"\n]]>\n'
        '</table>\n<table name="fused_op_id">\n'
        '<![CDATA[\n"id", "api_id", "tosa_op_ids", '
        '"fused_op_ids"\n'
        "531, 334;\n557, 335;\n499, 394;462;;\n]]>\n"
        '</table>\n<table name="chain_op_id">\n'
        "</table>\n</debug>"
    )
    parser = NXDebugDatabaseParser()
    table_elements = contents.split('<table name="')[1:]

    with pytest.raises(RuntimeError, match="Unsupported number of headers"):
        for table_element in table_elements:
            table_name = table_element.split('">')[0]
            table_data = parser.extract_cdata(table_element)
            reader = parser.get_csv_reader(table_data=table_data)
            headers = parser.get_csv_headers(csv_reader=reader)
            headers[0] = table_name + "_to_" + headers[1]
            parser.make_parsed_db(csv_reader=reader, headers=headers)


@pytest.mark.parametrize(
    "cell, expected_err, expected_result",
    [
        (
            "Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;"
            "DRAM;320;12;10;",
            does_not_raise(),
            [
                {
                    "memoryName": "Undefined",
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                {
                    "memoryName": "Internal",
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                {
                    "memoryName": "L1",
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                {
                    "memoryName": "L2",
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                {
                    "memoryName": "SystemCache",
                    "readBytes": 0,
                    "writeBytes": 0,
                    "trafficCycles": 0,
                },
                {
                    "memoryName": "DRAM",
                    "readBytes": 320,
                    "writeBytes": 12,
                    "trafficCycles": 10,
                },
            ],
        ),
        (
            "Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;"
            "DRAM;320;12;10;extra_val",
            pytest.raises(RuntimeError, match="Unmatched column entries"),
            None,
        ),
    ],
)
def test_subtable_column(cell: str, expected_err: Any, expected_result: Any) -> None:
    """Test the subtable columns have the expected labels."""
    parser = SubtableColumnParser(
        "Meminfo", "memoryName;readBytes;writeBytes;trafficCycles"
    )
    assert parser.sub_columns == [
        "memoryName",
        "readBytes",
        "writeBytes",
        "trafficCycles",
    ]
    with expected_err:
        result = parser(cell)
        assert result == expected_result


def test_column_parsers() -> None:
    """Test if column parsers are set up properly."""
    pdb = NXPerformanceDatabaseParser()
    parsers = pdb.column_parsers
    col1 = "memoryName;readBytes;writeBytes;trafficCycles"
    col2 = "sectionName;cycles"
    assert parsers == {
        col1: SubtableColumnParser("Memory", col1),
        col2: SubtableColumnParser("Utilization", col2),
    }
    assert parsers != {  # type mismatch
        col1: 1,
        col2: "string",
    }


def _debug_database_with_api_labels(value: str) -> str:
    """Return a minimal debug database containing one api_labels value."""
    return (
        "<debug>\n"
        '<table name="tosa_op_id">\n'
        '<![CDATA[\n"id", "tosa_op", "api_labels"\n'
        f"879, Conv2D, {value}\n"
        "]]>\n"
        "</table>\n"
        "</debug>"
    )


def test_parse_debug_database_preserves_known_label_containing_semicolons() -> None:
    """Semicolons inside a known label are not treated as delimiters."""
    label = '{"source": "first(); second()"}'
    parser = NXDebugDatabaseParser(known_api_labels=[label])
    parser.raw_xmlish = _debug_database_with_api_labels(f"{label};")

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_api_labels"]["879"] == [label]


def test_parse_debug_database_recovers_multiple_known_labels_in_order() -> None:
    """Multiple known labels retain their serialized order."""
    first = "prefix"
    second = '{"source": "left(); right()"}'
    parser = NXDebugDatabaseParser(known_api_labels=[second, first])
    parser.raw_xmlish = _debug_database_with_api_labels(f"{first};{second};")

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_api_labels"]["879"] == [first, second]


def test_parse_debug_database_prefers_longest_known_label() -> None:
    """Shared-prefix labels choose the longest complete match."""
    short = "shared"
    long = "shared;continued"
    parser = NXDebugDatabaseParser(known_api_labels=[short, long])
    parser.raw_xmlish = _debug_database_with_api_labels(f"{long};{short};")

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_api_labels"]["879"] == [long, short]


def test_parse_debug_database_preserves_fallback_labels_between_known_labels() -> None:
    """Unknown fallback labels are retained alongside complete known labels."""
    first = "first"
    second = "second"
    fallback = "backend-generated-fallback"
    parser = NXDebugDatabaseParser(known_api_labels=[first, second])
    parser.raw_xmlish = _debug_database_with_api_labels(f"{first};{fallback};{second};")

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_api_labels"]["879"] == [
        first,
        fallback,
        second,
    ]


def test_parse_debug_database_preserves_trailing_fallback_label() -> None:
    """A trailing fallback label without a delimiter is retained."""
    parser = NXDebugDatabaseParser(known_api_labels=["known"])
    parser.raw_xmlish = _debug_database_with_api_labels("known;fallback")

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_api_labels"]["879"] == ["known", "fallback"]


def test_parse_debug_database_mixes_semicolon_known_and_fallback_labels() -> None:
    """Known labels containing semicolons coexist with arbitrary fallback labels."""
    known = '{"source": "first(); second()"}'
    first_fallback = "TOSAMATMUL_spirv_id_45"
    second_fallback = "arbitrary fallback label"
    parser = NXDebugDatabaseParser(known_api_labels=[known])
    parser.raw_xmlish = _debug_database_with_api_labels(
        f"{first_fallback};{known};{second_fallback};"
    )

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_api_labels"]["879"] == [
        first_fallback,
        known,
        second_fallback,
    ]


def test_parse_debug_database_retains_legacy_semicolon_splitting_without_known_labels() -> (
    None
):
    """Callers without known labels retain legacy plain-text parsing."""
    parser = NXDebugDatabaseParser()
    parser.raw_xmlish = _debug_database_with_api_labels("first;second;")

    records = parser.parse_debug_database()

    assert records["tosa_op_id_to_api_labels"]["879"] == ["first", "second"]
