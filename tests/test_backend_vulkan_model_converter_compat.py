# SPDX-FileCopyrightText: Copyright 2024-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tflite_compat module."""

# ruff: noqa: E501  # Line too long - test data strings
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.ml_sdk_model_converter.compat import (
    NXCompatibilityChecker,
    NXModelCompatibilityInfo,
    NXOperatorCompatibilityInfo,
    PT2Model,
    TOSAModel,
    VGFModel,
    VMCCompatibilityLogReader,
)
from mlia.backend.ml_sdk_model_converter.tosa_reader import TosaOp, TosaOpType
from mlia.backend.tosa_converter_for_tflite.conversion import TosaConverterForTflite
from mlia.nn.tensorflow.config import get_model
from mlia.utils.proc import OutputConsumer


@pytest.mark.parametrize(
    "vmc_log, expected_ops, expected_errors",
    [
        (
            # CASE1: Framework Ops that don't lower to TOSA at all
            """
<unknown>:0: error: loc("tfl.custom"): failed to legalize operation 'tosa.custom' that was explicitly marked illegal
            """,
            {},
            {
                "tfl.custom": "failed to legalize operation 'tosa.custom' that was explicitly marked illegal"
            },
        ),
        (
            # CASE2: Framework Ops that are supported in the VGF but via generated/substitute shaders
            """
Successfully lowered: tosa.custom at loc("tfl.custom")
Successfully lowered: tosa.max_pool2d at loc("cs_ne_cs_model/quant_max_pooling2d/MaxPool")
            """,
            {
                "cs_ne_cs_model/quant_max_pooling2d/MaxPool": "tosa.max_pool2d",
                "tfl.custom": "tosa.custom",
            },
            {},
        ),
        (
            # CASE3: Framework Ops that successfully lowered to TOSA
            """
Successfully lowered: tfl.pseudo_qconst at loc("inference/coefficients/splat/conv5/convolution")
Successfully lowered: tfl.fully_connected at loc("inference/coefficients/global/fc2/MatMul;inference/coefficients/global/fc2/Relu;inference/coefficients/global/fc2/BiasAdd")
            """,
            {
                "inference/coefficients/global/fc2/MatMul;inference/coefficients/global/fc2/Relu;inference/coefficients/global/fc2/BiasAdd": "tfl.fully_connected",
                "inference/coefficients/splat/conv5/convolution": "tfl.pseudo_qconst",
            },
            {},
        ),
        (
            # CASE3/Fused:
            """
Successfully lowered: tfl.split_v at loc(fused["arm_nss_clampnet_v1_1/split/split", "arm_nss_clampnet_v1_1/split/split1"])
            """,
            {"arm_nss_clampnet_v1_1/split/split": "tfl.split_v"},
            {},
        ),
        (
            # Other errors:
            """
<unknown>:0: error: loc("model/tf.math.multiply_75/Mul1"): failed to materialize conversion for result #0 of operation 'tfl.broadcast_to' that remained live after conversion
            """,
            {},
            {
                "model/tf.math.multiply_75/Mul1": "failed to materialize conversion for result #0 of operation 'tfl.broadcast_to' that remained live after conversion"
            },
        ),
    ],
)
def test_vmc_log_parse_line(
    vmc_log: str, expected_ops: list, expected_errors: list
) -> None:
    """ "Test log parsing, mainly to extract compatibility info."""

    reader = VMCCompatibilityLogReader()

    for line in vmc_log.splitlines():
        reader(line.lstrip())

    assert reader.lowered_ops == expected_ops

    assert reader.lowering_errors == expected_errors


def test_vmc_log_parser_overall() -> None:
    """ "Test log parsing, mainly to extract compatibility info."""

    vmc_log = """
    CASE1: Framework Ops that don't lower to TOSA at all
    <unknown>:0: error: loc("tfl.custom"): failed to legalize operation 'tosa.custom' that was explicitly marked illegal

    CASE2: Framework Ops that are supported in the VGF but via generated/substitute shaders
    Successfully lowered: tosa.custom at loc("tfl.custom")
    Successfully lowered: tosa.max_pool2d at loc("cs_ne_cs_model/quant_max_pooling2d/MaxPool")

    CASE3: Framework Ops that successfully lowered to TOSA
    Successfully lowered: tfl.pseudo_qconst at loc("inference/coefficients/splat/conv5/convolution")
    Successfully lowered: tfl.fully_connected at loc("inference/coefficients/global/fc2/MatMul;inference/coefficients/global/fc2/Relu;inference/coefficients/global/fc2/BiasAdd")

    CASE3/Fused:
    Successfully lowered: tfl.split_v at loc(fused["arm_nss_clampnet_v1_1/split/split", "arm_nss_clampnet_v1_1/split/split1"])

    Other errors:
    <unknown>:0: error: loc("model/tf.math.multiply_75/Mul1"): failed to materialize conversion for result #0 of operation 'tfl.broadcast_to' that remained live after conversion
    """

    reader = VMCCompatibilityLogReader()

    for line in vmc_log.splitlines():
        reader(line.lstrip())

    assert reader.lowered_ops == {
        "arm_nss_clampnet_v1_1/split/split": "tfl.split_v",
        "cs_ne_cs_model/quant_max_pooling2d/MaxPool": "tosa.max_pool2d",
        (
            "inference/coefficients/global/fc2/MatMul;"
            "inference/coefficients/global/fc2/Relu;"
            "inference/coefficients/global/fc2/BiasAdd"
        ): "tfl.fully_connected",
        "inference/coefficients/splat/conv5/convolution": "tfl.pseudo_qconst",
        "tfl.custom": "tosa.custom",
    }

    assert reader.lowering_errors == {
        "tfl.custom": (
            "failed to legalize operation 'tosa.custom' that was explicitly "
            "marked illegal"
        ),
        "model/tf.math.multiply_75/Mul1": (
            "failed to materialize conversion for result #0 of operation "
            "'tfl.broadcast_to' that remained live after conversion"
        ),
    }


def test_parse_loc_simple() -> None:
    """Test parse loc() string, simple case."""
    loc = VMCCompatibilityLogReader().parse_loc(
        'loc("hierarchy/dotted.dashes-semi:location")'
    )
    assert loc == "hierarchy/dotted.dashes-semi:location"


def test_parse_loc_fused() -> None:
    """Test parse loc() string, fused case."""
    loc = VMCCompatibilityLogReader().parse_loc(
        'loc(fused["hierarchy/dotted.dashes-semi:loc", '
        '"hierarchy/dotted.dashes-semi:loc1"])'
    )
    assert loc == "hierarchy/dotted.dashes-semi:loc"


def test_parse_nested() -> None:
    """Test parse loc() string, fused case."""
    loc = VMCCompatibilityLogReader().parse_loc(
        'loc("arm_nss_clampnet_v1/quant_conv2d_5/Relu;'
        "arm_nss_clampnet_v1/quant_conv2d_5/BiasAdd;"
        "arm_nss_clampnet_v1/quant_conv2d_9/Conv2D;"
        "arm_nss_clampnet_v1/quant_conv2d_5/Conv2D;"
        'arm_nss_clampnet_v1/quant_conv2d_5/BiasAdd/ReadVariableOp"'
        '("/filepath/arm_nss_clampnet_v1-2160_3840-int8_qat.tflite":0:0))'
    )
    assert loc == (
        "arm_nss_clampnet_v1/quant_conv2d_5/Relu;"
        "arm_nss_clampnet_v1/quant_conv2d_5/BiasAdd;"
        "arm_nss_clampnet_v1/quant_conv2d_9/Conv2D;"
        "arm_nss_clampnet_v1/quant_conv2d_5/Conv2D;"
        "arm_nss_clampnet_v1/quant_conv2d_5/BiasAdd/ReadVariableOp"
    )


@pytest.mark.parametrize(
    "line",
    ["loc(unmatched", "loc(noquotes)", 'loc(fused["op1", "op2")', "foo", 'loc("a b")'],
)
def test_vmc_log_parser_invalid_loc(line: str) -> None:
    """Test log parsing, with invalid syntax."""

    with pytest.raises(Exception, match="Can't find a valid location string"):
        VMCCompatibilityLogReader().parse_loc(line)


def test_tosa_flatbuffer_input_supported(tmp_path: Path) -> None:
    """Tests the tosa_input_supported() function."""
    checker = NXCompatibilityChecker(tmp_path)

    try:
        import tosa_flatbuffers  # noqa: F401

        assert checker.tosa_flatbuffer_input_supported()
    except ImportError:
        assert not checker.tosa_flatbuffer_input_supported()


def test_check_compatibility_vgf(tmp_path: Path) -> None:
    """Test compatibility check for VGF inputs"""
    model_path = tmp_path / "model.vgf"
    model_path.touch()

    checker = NXCompatibilityChecker(tmp_path)
    # Currently not supported as VGF models are not inherently NX compatible
    # To be updated when support is added
    with pytest.raises(
        NotImplementedError,
        match="Compatibility info is not supported yet for VGF models for this target.",
    ):
        checker.check_compatibility(VGFModel(model_path))


def test_check_compatibility_pt2_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test PT2Model compatibility check with successful TOSA conversion."""
    model_path = tmp_path / "model.pt2"
    model_path.touch()

    # Mock converter will return this path - implementation will create the directory
    tosa_output = tmp_path / "mlia-pytorch-to-tosa" / "model.tosa"

    def mock_converter(*_args: Any, **_kwargs: Any) -> Path:
        # Converter creates the file when called
        tosa_output.parent.mkdir(parents=True, exist_ok=True)
        tosa_output.touch()
        return tosa_output

    monkeypatch.setattr(
        "mlia.backend.mlia_pytorch_to_tosa_converter.conversion.MliaPytorchToTosaConverter.__call__",
        mock_converter,
    )

    # Mock TOSA ops reading - successful conversion with supported ops
    mock_tosa_ops = {
        0: TosaOp(name="CONV2D", loc="layer1/conv", type=TosaOpType.INT),
        1: TosaOp(name="ADD", loc="layer2/add", type=TosaOpType.INT),
    }
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.read_tosa_flatbuffer_ops",
        MagicMock(return_value=mock_tosa_ops),
    )

    checker = NXCompatibilityChecker(tmp_path)
    result = checker.check_compatibility(PT2Model(model_path))

    # Should have records for converted ops
    records = result.get_records()
    assert len(records) == 2
    assert all(op.compat_level == "TOSA" for op in records)
    assert all(op.placement == "NX" for op in records)


def test_check_compatibility_pt2_conversion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test PT2Model compatibility check when TOSA conversion fails."""
    model_path = tmp_path / "model.pt2"
    model_path.touch()

    monkeypatch.setattr(
        "mlia.backend.mlia_pytorch_to_tosa_converter.conversion.MliaPytorchToTosaConverter.__call__",
        MagicMock(
            side_effect=RuntimeError(
                "PyTorch to TOSA conversion failed: unsupported operator"
            )
        ),
    )

    checker = NXCompatibilityChecker(tmp_path)
    result = checker.check_compatibility(PT2Model(model_path))

    records = result.get_records()
    assert len(records) == 1
    assert records[0].location == "model_conversion"
    assert records[0].compat_level == "Non-NX"
    assert records[0].error is not None
    assert "Failed to convert PyTorch model to TOSA" in records[0].error
    assert "unsupported operator" in records[0].error


def test_check_compatibility_pt2_with_unsupported_ops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test PT2Model compatibility when TOSA contains unsupported operations."""
    model_path = tmp_path / "model.pt2"
    model_path.touch()

    tosa_output = tmp_path / "mlia-pytorch-to-tosa" / "model.tosa"

    def mock_converter(*_args: Any, **_kwargs: Any) -> Path:
        tosa_output.parent.mkdir(parents=True, exist_ok=True)
        tosa_output.touch()
        return tosa_output

    monkeypatch.setattr(
        "mlia.backend.mlia_pytorch_to_tosa_converter.conversion.MliaPytorchToTosaConverter.__call__",
        mock_converter,
    )

    # Mock TOSA ops reading - mix of supported and unsupported ops
    mock_tosa_ops = {
        0: TosaOp(name="CONV2D", loc="layer1/conv", type=TosaOpType.INT),
        1: TosaOp(name="UNSUPPORTED_OP", loc="layer2/unsupported", type=None),
        2: TosaOp(name="ADD", loc="layer3/add", type=TosaOpType.INT),
    }
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.read_tosa_flatbuffer_ops",
        MagicMock(return_value=mock_tosa_ops),
    )

    checker = NXCompatibilityChecker(tmp_path)
    result = checker.check_compatibility(PT2Model(model_path))

    records = result.get_records()
    assert len(records) == 3

    supported_locs = [op.location for op in records if op.compat_level == "TOSA"]
    unsupported_locs = [op.location for op in records if op.compat_level == "Non-NX"]

    assert len(supported_locs) == 2
    assert len(unsupported_locs) == 1
    assert unsupported_locs[0] == "layer2/unsupported_1"


@pytest.mark.parametrize(
    "use_flatbuffer,"
    "tosa_ops,"
    "expected_nx_compatible_locations,"
    "expected_nx_incompatible_locations",
    [
        (
            False,
            {
                0: TosaOp("tosa.conv2d", "model/block0", type=TosaOpType.INT),
                1: TosaOp("tosa.avg_pool2d", "model/block0", type=TosaOpType.INT),
                2: TosaOp("tosa.tanh", "model/block1", type=TosaOpType.INT),
            },
            [
                "model/block0_0",
                "model/block0_1",
                "model/block1_2",
            ],
            [],
        ),
        (
            False,
            {
                0: TosaOp("tosa.conv2d", "model/block0", type=TosaOpType.INT),
                1: TosaOp("tosa.avg_pool2d", "model/block0", type=TosaOpType.INT),
                2: TosaOp("tosa.no_op", "model/block1", type=None),  # unknown op
            },
            [
                "model/block0_0",
                "model/block0_1",
            ],
            [
                "model/block1_2",
            ],
        ),
        (
            False,
            {
                0: TosaOp("tosa.conv2d", "model/block0", type=TosaOpType.INT),
                1: TosaOp("tosa.avg_pool2d", "model/block0", type=TosaOpType.INT),
                2: TosaOp(
                    "tosa.custom", "model/block1", type=TosaOpType.INT
                ),  # shader op
            },
            [
                "model/block0_0",
                "model/block0_1",
            ],
            [
                "model/block1_2",
            ],
        ),
        (
            True,
            {
                0: TosaOp("CONV2D", "model/block0", type=TosaOpType.INT),
                1: TosaOp("AVG_POOL2D", "model/block0", type=TosaOpType.INT),
                2: TosaOp("TANH", "model/block1", type=TosaOpType.INT),
            },
            [
                "model/block0_0",
                "model/block0_1",
                "model/block1_2",
            ],
            [],
        ),
        (
            True,
            {
                0: TosaOp("CONV2D", "model/block0", type=TosaOpType.INT),
                1: TosaOp("AVG_POOL2D", "model/block0", type=TosaOpType.INT),
                2: TosaOp("NO_OP", "model/block1", type=TosaOpType.INT),
            },
            [
                "model/block0_0",
                "model/block0_1",
            ],
            [
                "model/block1_2",
            ],
        ),
        (
            True,
            {
                0: TosaOp("CONV2D", "model/block0", type=TosaOpType.INT),
                1: TosaOp("AVG_POOL2D", "model/block0", type=TosaOpType.INT),
                2: TosaOp("CUSTOM", "model/block1", type=TosaOpType.INT),
            },
            [
                "model/block0_0",
                "model/block0_1",
            ],
            [
                "model/block1_2",
            ],
        ),
        (
            False,
            {},
            [],
            [],
        ),
    ],
)
def test_check_compatibility_tosa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_flatbuffer: bool,
    tosa_ops: dict[int, TosaOp],
    expected_nx_compatible_locations: list[str],
    expected_nx_incompatible_locations: list[str],
) -> None:
    """Test compatibility check for TOSA-MLIR inputs"""
    checker = NXCompatibilityChecker(tmp_path)

    if use_flatbuffer and not checker.tosa_flatbuffer_input_supported():
        pytest.skip("Tosa Flatbuffers not available.")

    model_file = "model.tosa" if use_flatbuffer else "model.tosamlir"
    model_path = tmp_path / model_file
    model_path.touch()

    mock_read_tosa_mlir_ops = MagicMock(return_value=tosa_ops)
    mock_read_tosa_function = "mlia.backend.ml_sdk_model_converter.compat." + (
        "read_tosa_flatbuffer_ops" if use_flatbuffer else "read_tosa_mlir_ops"
    )
    monkeypatch.setattr(
        mock_read_tosa_function,
        mock_read_tosa_mlir_ops,
    )
    records = checker.check_compatibility(TOSAModel(model_path)).get_records()
    mock_read_tosa_mlir_ops.assert_called_once()

    nx_compatible_locations = [op.location for op in records if op.placement == "NX"]
    nx_incompatible_locations = [op.location for op in records if op.placement != "NX"]

    assert nx_compatible_locations == expected_nx_compatible_locations
    assert nx_incompatible_locations == expected_nx_incompatible_locations


def test_check_compatibility_unsupported_input(
    tmp_path: Path,
    test_keras_model: Path,
) -> None:
    """Test compatibility check for unsupported input models"""
    checker = NXCompatibilityChecker(tmp_path)

    with pytest.raises(NotImplementedError, match="Compatibility not supported for"):
        checker.check_compatibility(get_model(test_keras_model))


def test_check_compatibility_tosa_bad_file_ext(
    tmp_path: Path,
) -> None:
    """Test compatibility check for bad file formats."""
    checker = NXCompatibilityChecker(tmp_path)
    model = tmp_path / "model.bad_ext"

    with pytest.raises(RuntimeError, match="Unsupported file format '.bad_ext'"):
        checker.check_compatibility(TOSAModel(model))


def test_check_compatibility_tosa_parsing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test compatibility check if parsing a TOSA file fails."""
    checker = NXCompatibilityChecker(tmp_path)
    model = tmp_path / "model.tosamlir"

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.read_tosa_mlir_ops",
        MagicMock(side_effect=Exception("Parsing failed")),
    )

    with pytest.raises(RuntimeError, match="Failed to read TOSA operations from"):
        checker.check_compatibility(TOSAModel(model))


def test_check_compatibility_tflite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test VMC compatibity check."""

    back_end_output = [
        (
            "Successfully lowered: tosa.rescale at loc("
            '"arm_nss_clampnet_v4/quant_conv2d_7/Relu"("model.tflite":0:0))'
        ),
        (
            '<unknown>:0: error: loc("model/tf.math.multiply_75/Mul1"): '
            "failed to materialize conversion for result #0 of"
            "operation 'tfl.broadcast_to' that remained live after conversion"
        ),
        "",
    ]

    def back_end_call(consumer: OutputConsumer, program: str, *args: list[str]) -> None:
        """Fake Backend call."""
        if not program.endswith("converter"):
            pytest.fail("Expected backend call")
        assert "--experimental-analysis" in args
        for line in back_end_output:
            consumer(line)

    def fake_tosa_converter_call(
        _: TosaConverterForTflite, tflite_file: Path, output_dir: Path
    ) -> Path:
        output_path = output_dir / f"{tflite_file.stem}.tosamlir"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return output_path

    monkeypatch.setattr(TosaConverterForTflite, "__call__", fake_tosa_converter_call)

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.conversion.process_command_output",
        lambda cmd, consumers: back_end_call(consumers[1], *cmd.cmd),
    )

    mock_repo = MagicMock()
    mock_repo.get_backend_settings = MagicMock(return_value=(tmp_path / "backend", {}))
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.get_backend_repository",
        MagicMock(return_value=mock_repo),
    )

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat.operator_names_to_types",
        MagicMock(
            return_value=(
                {
                    "model/tf.math.multiply_75/Mul1": "MUL",
                },
                {
                    "model/tf.math.multiply_75/Mul1": "INT8",
                },
            )
        ),
    )

    checker = NXCompatibilityChecker(tmp_path)

    result = checker.check_compatibility(Path("model.tflite"))

    assert result.dump() == [
        {
            "compat_level": "TOSA",
            "location": "arm_nss_clampnet_v4/quant_conv2d_7/Relu",
            "placement": "NX",
            "tosa_op": "tosa.rescale",
        },
        {
            "compat_level": "Non-NX",
            "error": "failed to materialize conversion for result #0 ofoperation "
            "'tfl.broadcast_to' that remained live after conversion",
            "location": "model/tf.math.multiply_75/Mul1",
            "type": "MUL",
        },
    ]


def test_nx_compatiblity_info() -> None:
    """Test Neural Accelerator CompatibilityInfo additions."""

    info = NXModelCompatibilityInfo()
    info.add_lowered_to_tosa(TosaOp("mytosa_op", "model/myloc1/op1", TosaOpType.INT))
    assert info.dump() == [
        {
            "compat_level": "TOSA",
            "location": "model/myloc1/op1",
            "placement": "NX",
            "tosa_op": "mytosa_op",
        },
    ]

    info.add_lowering_error("model/myloc2/op3", "Can't be lowered")
    info.add_lowered_to_tosa(
        TosaOp("tosa.custom", "model/myloc2/shader_op", TosaOpType.INT)
    )
    info.add_lowered_to_tosa(TosaOp("mytosa_op4", "model/myloc1/op4", TosaOpType.INT))
    info.add_lowered_to_tosa(
        TosaOp("tosa.conv2d", "model/myloc1/op5", TosaOpType.FLOAT)
    )

    expected_records = [
        NXOperatorCompatibilityInfo(
            location="model/myloc1/op1",
            compat_level="TOSA",
            type=None,
            tosa_op="mytosa_op",
            error=None,
            placement="NX",
        ),
        NXOperatorCompatibilityInfo(
            location="model/myloc2/op3",
            compat_level="Non-NX",
            type=None,
            tosa_op=None,
            error="Can't be lowered",
            placement=None,
        ),
        NXOperatorCompatibilityInfo(
            location="model/myloc2/shader_op",
            compat_level="Shader",
            type=None,
            tosa_op="tosa.custom",
            error=None,
            placement="EE",
        ),
        NXOperatorCompatibilityInfo(
            location="model/myloc1/op4",
            compat_level="TOSA",
            type=None,
            tosa_op="mytosa_op4",
            error=None,
            placement="NX",
        ),
        NXOperatorCompatibilityInfo(
            location="model/myloc1/op5",
            compat_level="Shader",
            type=None,
            tosa_op="tosa.conv2d",
            error=None,
            placement="EE",
        ),
    ]
    actual_records = info.get_records()
    assert len(actual_records) == len(expected_records)
    for record in actual_records:
        assert record in expected_records

    expected_dump = [
        {
            "location": "model/myloc1/op1",
            "compat_level": "TOSA",
            "tosa_op": "mytosa_op",
            "placement": "NX",
        },
        {
            "location": "model/myloc1/op4",
            "compat_level": "TOSA",
            "tosa_op": "mytosa_op4",
            "placement": "NX",
        },
        {
            "location": "model/myloc2/op3",
            "compat_level": "Non-NX",
            "error": "Can't be lowered",
        },
        {
            "location": "model/myloc2/shader_op",
            "compat_level": "Shader",
            "tosa_op": "tosa.custom",
            "placement": "EE",
        },
        {
            "location": "model/myloc1/op5",
            "compat_level": "Shader",
            "tosa_op": "tosa.conv2d",
            "placement": "EE",
        },
    ]
    actual_dump = info.dump()
    assert len(actual_dump) == len(expected_dump)
    for dumped in actual_dump:
        assert dumped in expected_dump


def test_unrecognized_log_line() -> None:
    """Test raising errors for unrecognized log lines."""

    line = "Successfully lowered: tosa.rescale  loc(unknown)"
    with pytest.raises(
        RuntimeError, match=f"Unrecognized log line: '{re.escape(line)}'"
    ):
        VMCCompatibilityLogReader()(line)


def test_handling_unknowns() -> None:
    """Test that unknowns are handled gracefully when parsing loc() expressions."""

    assert VMCCompatibilityLogReader().parse_loc("loc(unknown)") == "unknown"
