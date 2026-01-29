# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology reporters."""
from functools import partial
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from mlia.backend.ml_sdk_model_converter.compat import NXModelCompatibilityInfo
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOp
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOpType
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.output_parsing import NXDebugDatabaseParser
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXPerformanceDatabaseParser,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.backend.nx_performance_estimator.statistics import NXModelPerformanceStats
from mlia.backend.nx_performance_estimator.statistics import NXOperatorPerformanceStats
from mlia.backend.nx_performance_estimator.statistics import NXPerformanceStats
from mlia.core.advice_generation import Advice
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity
from mlia.core.reporters import report_advice
from mlia.core.reporting import Table
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.reporters import neural_technology_formatters
from mlia.target.neural_technology.reporters import report_target
from mlia.utils.console import remove_ascii_codes


def test_report_target() -> None:
    """Test function report_target()."""
    report = report_target(
        NeuralTechnologyConfiguration.load_profile("neural-technology")
    )
    assert report.to_plain_text()


def assert_table_contents(report: Table, json: dict) -> None:
    """Assert that a given Table renders the expected JSON output."""
    assert isinstance(report, Table)
    assert report.to_json() == json


def assert_table_lines(report: Table, expected_lines: list) -> None:
    """Assert that a given Table renders the expected JSON output.

    In case of failure, it renders actual and expected textual tables in a form
    that's easy to overview and can directly be used as "golden" data in the test.
    """
    assert isinstance(report, Table)
    actual_lines = remove_ascii_codes(report.to_plain_text()).split("\n")

    def to_diff_string(lines: List[str]) -> str:
        test_line = [f'          "{line}",' for line in lines]
        return ("\n").join(test_line)

    actual = to_diff_string(actual_lines)
    expected = to_diff_string(expected_lines)
    assert actual_lines == expected_lines, f"Expected:\n{expected}\n\nActual:\n{actual}"


def test_nx_performance_estimator_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function neural_technology_formatters() with Neural Accelerator performance
    data."""

    performance_contents = """
    <![CDATA[
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;cycles"
    26, 1800, 212, Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;100;VectorEngine;25;VectorEngine;25;VectorEngine;25;TransformUnit;25;TransformUnit;25;InputReader;625;InputReader;625;InputReader;250;
    25, 4, 13, Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    ]]>
    """.strip()

    debug_contents = """
<?xml version='1.0' encoding='utf-8' ?>
<debug_database>
<regor_version>1.0.0</regor_version>
<table name="tosa_op_id">
<![CDATA[
"id", "tosa_op", "api_labels"
1077, DepthwiseConv2D, deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_relu/Relu6;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_10_depthwise/depthwise;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise/depthwise;
1078, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_relu/Relu6;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_10_depthwise/depthwise;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise/depthwise;
1079, Conv2D, deeplabv3plus_mbnV2__1080p/expanded_conv_8_project_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_9_project/Conv2D;deeplabv3plus_mbnV2__1080p/expanded_conv_8_project/Conv2D;
1080, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_project_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_9_project/Conv2D;deeplabv3plus_mbnV2__1080p/expanded_conv_8_project/Conv2D;
1083, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add;
1081, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add;
1084, Add, deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add;
1085, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add;
]]>
</table>
<table name="fused_op_id">
<![CDATA[
"id", "tosa_op_ids"
1361, 1077;
1473, 1078;
1363, 1079;
1475, 1080;
1299, 1083;1081;1084;1085;
]]>
</table>
<table name="chain_op_id">
<![CDATA[
"id", "fused_op_ids"
1613, 1361;1473;
1617, 1363;1475;1299;
]]>
</table>
<table name="stripe_op_id">
<![CDATA[
"id", "op_id", "cascade_op_id"
25, 1613, 2196
26, 1617, 2194
]]>
</table>
""".strip()
    performance_db_parser = NXPerformanceDatabaseParser()
    performance_db_parser.raw_xmlish = performance_contents
    performance_db = performance_db_parser.parse_performance_database()
    debug_db_parser = NXDebugDatabaseParser()
    debug_db_parser.raw_xmlish = debug_contents
    debug_db = debug_db_parser.parse_debug_database()

    sys_cfg, compiler_cfg = Path("system-config"), Path("compiler-config")
    cfg = NXPerformanceEstimatorConfig(sys_cfg, compiler_cfg)

    metrics = NXPerformanceEstimatorPerformanceMetrics(
        backend_config=cfg,
        performance_db_parser=performance_db_parser,
        stripe_performance_metrics={"op0": MagicMock(spec=NXOperatorPerformanceStats)},
        chain_performance_metrics=NXPerformanceStats(
            debug_db=debug_db,
            performance_db=performance_db,
        ).process_stats_per_chain(),
        model_performance_stats=MagicMock(spec=NXModelPerformanceStats),
    )

    monkeypatch.setattr("mlia.utils.console.Console", partial(Console, width=80))

    formatter = neural_technology_formatters(metrics)
    report = formatter(metrics)
    assert isinstance(report, Table)

    assert_table_lines(
        report,
        [
            # pylint: disable=C0301
            "Neural Accelerator raw performance report:",
            "┌────┬──────┬──────┬──────┬──────┬──────┬──────┬─────┬──────┬─────┬──────┬─────┐",
            "│    │ Ope… │ Ope… │ Ope… │ Tot… │ HW   │ Act… │ HW  │ Mem… │ Re… │ Wri… │ Tr… │",
            "│ ID │ Loc… │ Type │ Cyc… │ Cyc… │ Sec… │ Cyc… │ Ut… │ Name │ by… │ byt… │ cy… │",
            "╞════╪══════╪══════╪══════╪══════╪══════╪══════╪═════╪══════╪═════╪══════╪═════╡",
            "│ 25 │ dee… │ Dep… │ 4    │ 13   │ Out… │ 625  │ 48… │ L1   │ 0   │ 4    │ 0   │",
            "│    │ p/e… │ Res… │      │      │ Vec… │ 0    │ 0.… │ L2   │ 0   │ 0    │ 0   │",
            "│    │ se_… │      │      │      │ Inp… │ 0    │ 0.… │ Sys… │ 0   │ 0    │ 0   │",
            "│    │ us_… │      │      │      │      │      │     │ DRAM │ 128 │ 4    │ 4   │",
            "│    │ con… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ Bat… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ _mb… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ nv_… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ ;de… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ 0p/… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ ise… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ dee… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ p/e… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ se_… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ us_… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ con… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ Bat… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ _mb… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ nv_… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ ;de… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ 0p/… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ ise… │      │      │      │      │      │     │      │     │      │     │",
            "├────┼──────┼──────┼──────┼──────┼──────┼──────┼─────┼──────┼─────┼──────┼─────┤",
            "│ 26 │ dee… │ Con… │ 1800 │ 212  │ Out… │ 100  │ 47… │ L1   │ 0   │ 0    │ 0   │",
            "│    │ p/e… │ Res… │      │      │ Vec… │ 75   │ 35… │ L2   │ 0   │ 0    │ 0   │",
            "│    │ _BN… │ Res… │      │      │ Tra… │ 50   │ 23… │ Sys… │ 0   │ 0    │ 0   │",
            "│    │ lab… │ Res… │      │      │ Inp… │ 1500 │ 70… │ DRAM │ 320 │ 12   │ 10  │",
            "│    │ pan… │ Add  │      │      │      │      │     │      │     │      │     │",
            "│    │ v2D… │ Res… │      │      │      │      │     │      │     │      │     │",
            "│    │ 108… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ jec… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ dee… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ p/e… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ _BN… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ lab… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ pan… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ v2D… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ 108… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ jec… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ dee… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ p/e… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ dee… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ p/e… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ dee… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ p/e… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ dee… │      │      │      │      │      │     │      │     │      │     │",
            "│    │ p/e… │      │      │      │      │      │     │      │     │      │     │",
            "└────┴──────┴──────┴──────┴──────┴──────┴──────┴─────┴──────┴─────┴──────┴─────┘",
            # pylint: enable=C0301
        ],
    )


def test_nx_compatibility_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function neural_technology_formatters() with Neural Accelerator
    compatibility data."""

    comp_info = NXModelCompatibilityInfo({"/myop1": "COMP2D", "/myop4": "NMS"})
    comp_info.add_lowered_to_tosa(TosaOp("tosaop1", "/myop1", TosaOpType.INT))
    comp_info.add_lowered_to_tosa(TosaOp("tosa.custom", "/myop2", TosaOpType.INT))
    comp_info.add_lowered_to_tosa(TosaOp("tosaop3", "/myop3", TosaOpType.INT))
    comp_info.add_lowering_error("/myop4", "Error occured when lowering")

    formatter = neural_technology_formatters(comp_info)
    report = formatter(comp_info)
    assert isinstance(report, Table)

    monkeypatch.setattr("mlia.utils.console.Console", partial(Console, width=80))
    assert_table_lines(
        report,
        [
            # pylint: disable=C0301
            "Operators:",
            "┌───┬───────────────────┬───────────────┬──────────────┬──────────────────┐",
            "│ # │ Operator location │ Operator type │ NX placement │ NX compatibility │",
            "╞═══╪═══════════════════╪═══════════════╪══════════════╪══════════════════╡",
            "│ 1 │ /myop1            │ COMP2D        │ NX           │ TOSA             │",
            "├───┼───────────────────┼───────────────┼──────────────┼──────────────────┤",
            "│ 2 │ /myop2            │ Unknown       │ EE           │ Shader           │",
            "├───┼───────────────────┼───────────────┼──────────────┼──────────────────┤",
            "│ 3 │ /myop3            │ Unknown       │ NX           │ TOSA             │",
            "├───┼───────────────────┼───────────────┼──────────────┼──────────────────┤",
            "│ 4 │ /myop4            │ NMS           │ FAIL         │ Non-NX           │",
            "└───┴───────────────────┴───────────────┴──────────────┴──────────────────┘",
            # pylint: enable=C0301
        ],
    )


def test_neural_technology_formatters_advice_list() -> None:
    """Test neural_technology_formatters function with a list of Advice objects."""
    ret = neural_technology_formatters(
        [
            Advice(
                id="0",
                category=SchemaAdviceCategory.PERFORMANCE,
                severity=AdviceSeverity.INFO,
                message="Sample 1",
            )
        ]
    )
    assert ret is report_advice


def test_neural_technology_formatters_invalid_data() -> None:
    """Test neural_technology_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        neural_technology_formatters(12)


def test_neural_technology_configuration_verify() -> None:
    """
    Test that the verify function of class NeuralTechnologyConfiguration
    raises an error with an invalid target.
    """
    target = "AnyTarget"
    cfg = NeuralTechnologyConfiguration(target=target)
    with pytest.raises(
        ValueError, match=f"Wrong target {target} for Neural Technology configuration"
    ):
        cfg.verify()


# %%
