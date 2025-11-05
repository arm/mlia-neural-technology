# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Accelerator Graph Compiler performance estimation."""
from __future__ import annotations

import csv
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any

import pytest

from mlia.backend.nx_graph_compiler.output_parsing import NXDebugDatabaseParser
from mlia.backend.nx_graph_compiler.output_parsing import NXOutputParser
from mlia.backend.nx_graph_compiler.output_parsing import NXPerformanceDatabaseParser
from mlia.backend.nx_graph_compiler.output_parsing import SubtableColumnParser


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
    contents = """
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;cycles"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    """.strip()
    parser = NXOutputParser()
    csv_reader = parser.get_csv_reader(table_data=contents)
    expected_csv_reader = csv.reader(contents.splitlines())

    for actual, expected in zip(csv_reader, expected_csv_reader):
        assert actual == expected


def test_get_csv_headers() -> None:
    """Extract the headers from a csv reader."""
    contents = """
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;cycles"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    """.strip()
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
    contents = """
    <![CDATA[
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;cycles"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    ]]>
    """.strip()
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
    # pylint: disable=line-too-long
    assert records["tosa_op_id_to_api_labels"]["372"] == ["model/re_lu/Relu"]


def test_parse_debug_database() -> None:
    """Test the debug database has the required key-value pairs."""
    contents = """<?xml version='1.0' encoding='utf-8' ?>
    <![CDATA[\n"id", "api_id"\n]]>\n</table>\n<table name="fused_op_id">
    <![CDATA[\n"id", "tosa_op_ids"\n531, 334;\n557, 335;\n499, 394;462;;\n]]>\n</table>\n<table name="chain_op_id">
    <![CDATA[\n"id", "fused_op_ids"\n603, 531;557;\n605, 533;559;\n607, 535;561;\n637, 589;591;509;511;515;\n]]>\n<table name="stripe_op_id">
    <![CDATA[\n"id", "chain_op_id", "cascade_op_id"\n0, 603, 1693;\n1, 605, 1691;\n]]>
    </table>\n</debug>"""
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


def test_parse_debug_database_invalid_num_db_headers() -> None:
    """Test error is raised if the debug database has too many headers."""
    contents = """<?xml version='1.0' encoding='utf-8' ?>
    <![CDATA[\n"id", "api_id"\n]]>\n</table>\n<table name="fused_op_id">
    <![CDATA[\n"id", "api_id", "tosa_op_ids", "fused_op_ids"\n531, 334;\n557, 335;\n499, 394;462;;\n]]>\n</table>\n<table name="chain_op_id">
    </table>\n</debug>"""
    parser = NXDebugDatabaseParser()
    parser.raw_xmlish = contents
    with pytest.raises(RuntimeError, match="Unsupported number of headers"):
        parser.parse_debug_database()


def test_make_parsed_db_debug_db() -> None:
    """Test the debug database has the required key-value pairs."""
    contents = """<?xml version='1.0' encoding='utf-8' ?>
    <![CDATA[\n"id", "api_id"\n]]>\n</table>\n<table name="fused_op_id">
    <![CDATA[\n"id", "tosa_op_ids"\n531, 334;\n557, 335;\n499, 394;462;;\n]]>\n</table>\n<table name="chain_op_id">
    <![CDATA[\n"id", "fused_op_ids"\n603, 531;557;\n605, 533;559;\n607, 535;561;\n637, 589;591;509;511;515;\n]]>\n<table name="stripe_op_id">
    <![CDATA[\n"id", "chain_op_id", "cascade_op_id"\n0, 603, 1693;\n1, 605, 1691;\n]]>
    </table>\n</debug>"""
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
    contents = """<?xml version='1.0' encoding='utf-8' ?>
    <![CDATA[\n"id", "api_id"\n]]>\n</table>\n<table name="fused_op_id">
    <![CDATA[\n"id", "api_id", "tosa_op_ids", "fused_op_ids"\n531, 334;\n557, 335;\n499, 394;462;;\n]]>\n</table>\n<table name="chain_op_id">
    </table>\n</debug>"""
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
    parsers = pdb.column_parsers  # pylint: disable=protected-access
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
