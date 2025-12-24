# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to TOSA converter conversion."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

import mlia.backend.mlia_pytorch_to_tosa_converter.conversion as conv_module
from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import DEFAULT_BASE_NAME
from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import (
    EXPECTED_OUTPUT_FILENAME,
)
from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import (
    MliaPytorchToTosaConverter,
)


def test_converter_validates_inputs() -> None:
    """Test converter validates input file and output directory."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Invalid file format
        txt_file = Path(tmpdir) / "model.txt"
        txt_file.write_text("test")
        with pytest.raises(ValueError, match="Only .pt2 files are supported"):
            # pylint: disable=protected-access
            converter._load_pytorch_model(txt_file)
            # pylint: enable=protected-access

        # Nonexistent output directory
        pt2_file = Path(tmpdir) / "model.pt2"
        pt2_file.write_text("test")
        with pytest.raises(NotADirectoryError):
            converter(pt2_file, Path(tmpdir) / "nonexistent")


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion._import_dependencies")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.torch")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TosaCompileSpec")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TOSAQuantizer")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.prepare_pt2e")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.convert_pt2e")
@patch(
    "mlia.backend.mlia_pytorch_to_tosa_converter.conversion."
    "to_edge_transform_and_lower"
)
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
@patch(
    "mlia.backend.mlia_pytorch_to_tosa_converter.conversion."
    "get_symmetric_quantization_config"
)
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TOSAPartitioner")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.EdgeCompileConfig")
# pylint: disable=too-many-arguments,too-many-locals
def test_full_conversion_process(
    _mock_edge_config: Mock,
    _mock_partitioner: Mock,
    mock_get_config: Mock,
    mock_move: Mock,
    mock_transform: Mock,
    mock_convert_pt2e: Mock,
    mock_prepare_pt2e: Mock,
    mock_quantizer: Mock,
    mock_compile_spec: Mock,
    mock_torch: Mock,
    _mock_import_deps: Mock,
) -> None:
    """Test complete conversion flow."""
    mock_exported_program = Mock()
    mock_exported_program.module.return_value = Mock()
    mock_exported_program.example_inputs = [Mock()]
    mock_torch.export.load.return_value = mock_exported_program
    mock_torch.export.export.return_value = Mock()

    mock_get_config.return_value = Mock()

    mock_compile_spec_inst = Mock()
    mock_compile_spec_inst.dump_intermediate_artifacts_to.return_value = (
        mock_compile_spec_inst
    )
    mock_compile_spec.return_value = mock_compile_spec_inst

    mock_quantizer.return_value = Mock()
    mock_prepare_pt2e.return_value = Mock()
    mock_convert_pt2e.return_value = Mock()

    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "model.pt2"
        input_file.write_text("test", encoding="utf-8")
        output_dir = Path(tmpdir)

        source_dir = output_dir / DEFAULT_BASE_NAME
        source_dir.mkdir()
        (source_dir / EXPECTED_OUTPUT_FILENAME).write_text("tosa", encoding="utf-8")

        mock_move.side_effect = lambda src, dst: Path(dst).write_text(
            "tosa", encoding="utf-8"
        )

        result = converter(input_file, output_dir)

        mock_torch.export.load.assert_called_once()
        mock_compile_spec.assert_called_once()
        mock_quantizer.assert_called_once()
        mock_transform.assert_called_once()
        assert result == output_dir / "model.tosa"


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.torch")
def test_load_model_failure(mock_torch: Mock) -> None:
    """Test model loading handles errors."""
    mock_torch.export.load.side_effect = RuntimeError("Load failed")
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "model.pt2"
        input_file.write_text("test", encoding="utf-8")

        with pytest.raises(ValueError, match="Failed to load PyTorch export file"):
            # pylint: disable=protected-access
            converter._load_pytorch_model(input_file)
            # pylint: enable=protected-access


@patch(
    "mlia.backend.mlia_pytorch_to_tosa_converter.conversion.to_edge_transform_and_lower"
)
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TOSAPartitioner")
def test_lowering_failure(_mock_partitioner: Mock, mock_transform: Mock) -> None:
    """Test TOSA lowering handles errors."""
    mock_transform.side_effect = RuntimeError("Lowering failed")

    converter = MliaPytorchToTosaConverter()

    with pytest.raises(RuntimeError, match="TOSA lowering failed"):
        # pylint: disable=protected-access
        converter._lower_to_tosa(Mock(), Mock())
        # pylint: enable=protected-access


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
def test_output_file_not_found(mock_move: Mock) -> None:
    """Test output file handling when file missing."""
    mock_move.side_effect = FileNotFoundError("Not found")
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(
            FileNotFoundError, match="Expected TOSA output file not found"
        ):
            # pylint: disable=protected-access
            converter._move_output_file(
                Path(tmpdir) / "model.pt2", Path(tmpdir), DEFAULT_BASE_NAME
            )
            # pylint: enable=protected-access


def test_import_dependencies_loads_modules() -> None:
    """Test that _import_dependencies actually loads all required modules."""
    original_flag = conv_module.DEPENDENCIES_LOADED
    conv_module.DEPENDENCIES_LOADED = False

    try:
        # pylint: disable=protected-access
        conv_module._import_dependencies()

        assert conv_module.torch is not None
        assert conv_module.get_symmetric_quantization_config is not None
        assert conv_module.TOSAQuantizer is not None
        assert conv_module.TosaCompileSpec is not None
        assert conv_module.TOSAPartitioner is not None
        assert conv_module.EdgeCompileConfig is not None
        assert conv_module.to_edge_transform_and_lower is not None
        assert conv_module.convert_pt2e is not None
        assert conv_module.prepare_pt2e is not None
        assert conv_module.DEPENDENCIES_LOADED is True

        conv_module._import_dependencies()
        # pylint: enable=protected-access
        assert conv_module.DEPENDENCIES_LOADED is True

    except ImportError:
        pytest.skip("PyTorch/executorch dependencies not available")
    finally:
        conv_module.DEPENDENCIES_LOADED = original_flag


def test_import_dependencies_raises_on_missing_torch() -> None:
    """Test _import_dependencies raises ImportError if torch is not available."""
    original_flag = conv_module.DEPENDENCIES_LOADED
    conv_module.DEPENDENCIES_LOADED = False

    original_torch = sys.modules.get("torch")
    if "torch" in sys.modules:
        sys.modules["torch"] = None  # type: ignore[assignment]

    try:
        with pytest.raises(ImportError):
            # pylint: disable=protected-access
            conv_module._import_dependencies()
            # pylint: enable=protected-access
    finally:
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        elif "torch" in sys.modules:
            del sys.modules["torch"]
        conv_module.DEPENDENCIES_LOADED = original_flag


def test_patch_node_visitor_success() -> None:
    """Test successful patching of NodeVisitor when executorch is available."""
    converter = MliaPytorchToTosaConverter()
    mock_node_visitor = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "executorch.backends.arm.operators.node_visitor": MagicMock(
                NodeVisitor=mock_node_visitor
            )
        },
    ):
        # pylint: disable=protected-access
        converter._patch_node_visitor_for_location()
        # pylint: enable=protected-access

        assert hasattr(mock_node_visitor, "_serialize_operator")

        # pylint: disable=protected-access
        patched_func = mock_node_visitor._serialize_operator
        # pylint: enable=protected-access
        mock_node_with_name = MagicMock()
        mock_node_with_name.name = "test_node"
        mock_node_without_name = None
        mock_tosa_graph = MagicMock()

        patched_func(
            None,
            mock_node_with_name,
            mock_tosa_graph,
            "tosa_op",
            ["input"],
            ["output"],
        )
        mock_tosa_graph.addOperator.assert_called_with(
            "tosa_op",
            inputs=["input"],
            outputs=["output"],
            attributes=None,
            location="test_node",
        )

        mock_tosa_graph.reset_mock()
        patched_func(None, mock_node_without_name, mock_tosa_graph, "tosa_op2", [], [])
        mock_tosa_graph.addOperator.assert_called_with(
            "tosa_op2",
            inputs=[],
            outputs=[],
            attributes=None,
            location="",
        )


def test_patch_node_visitor_import_error_logged() -> None:
    """Test that ImportError in patch_node_visitor is logged as warning."""
    converter = MliaPytorchToTosaConverter()

    executorch_modules = [k for k in sys.modules if k.startswith("executorch")]
    original_modules = {}
    for mod in executorch_modules:
        original_modules[mod] = sys.modules.pop(mod, None)

    try:
        with patch(
            "mlia.backend.mlia_pytorch_to_tosa_converter.conversion.logger"
        ) as mock_logger:
            # pylint: disable=protected-access
            converter._patch_node_visitor_for_location()
            # pylint: enable=protected-access

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0]
            assert "Could not patch NodeVisitor" in call_args[0]
    finally:
        for mod, val in original_modules.items():
            if val is not None:
                sys.modules[mod] = val


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion._import_dependencies")
def test_run_converter_validates_file_existence(_mock_import: Mock) -> None:
    """Test that _run_converter validates input file existence."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_file = Path(tmpdir) / "nonexistent.pt2"
        output_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            # pylint: disable=protected-access
            converter._run_converter(nonexistent_file, output_dir)
            # pylint: enable=protected-access


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion._import_dependencies")
def test_run_converter_validates_file_is_file(_mock_import: Mock) -> None:
    """Test that _run_converter validates input is a file not directory."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_file = Path(tmpdir) / "model.pt2"
        fake_file.mkdir()
        output_dir = Path(tmpdir)

        with pytest.raises(ValueError, match="Input path is not a file"):
            # pylint: disable=protected-access
            converter._run_converter(fake_file, output_dir)
            # pylint: enable=protected-access


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TosaCompileSpec")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TOSAQuantizer")
@patch(
    "mlia.backend.mlia_pytorch_to_tosa_converter.conversion."
    "get_symmetric_quantization_config"
)
def test_setup_quantization(
    mock_get_config: Mock,
    mock_quantizer: Mock,
    mock_compile_spec: Mock,
) -> None:
    """Test quantization setup creates compile spec and quantizer correctly."""
    mock_compile_spec_inst = Mock()
    mock_compile_spec_inst.dump_intermediate_artifacts_to.return_value = (
        mock_compile_spec_inst
    )
    mock_compile_spec.return_value = mock_compile_spec_inst

    mock_quantizer_inst = Mock()
    mock_quantizer.return_value = mock_quantizer_inst

    mock_operator_config = Mock()
    mock_get_config.return_value = mock_operator_config

    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_base"

        # pylint: disable=protected-access
        _compile_spec, _quantizer = converter._setup_quantization(output_dir, base_name)
        # pylint: enable=protected-access

        mock_compile_spec.assert_called_once()
        mock_compile_spec_inst.dump_intermediate_artifacts_to.assert_called_once()
        mock_quantizer.assert_called_once_with(mock_compile_spec_inst)
        mock_quantizer_inst.set_global.assert_called_once_with(mock_operator_config)


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.torch")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.prepare_pt2e")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.convert_pt2e")
def test_quantize_model(
    mock_convert: Mock,
    mock_prepare: Mock,
    mock_torch: Mock,
) -> None:
    """Test model quantization flow."""
    mock_graph_module = Mock()
    mock_quantizer = Mock()
    mock_example_inputs = [Mock()]

    mock_quantized_graph = Mock()
    mock_prepare.return_value = mock_quantized_graph
    mock_convert.return_value = mock_quantized_graph

    mock_exported = Mock()
    mock_torch.export.export.return_value = mock_exported

    converter = MliaPytorchToTosaConverter()

    # pylint: disable=protected-access
    result = converter._quantize_model(
        mock_graph_module, mock_quantizer, mock_example_inputs
    )
    # pylint: enable=protected-access

    mock_prepare.assert_called_once_with(mock_graph_module, mock_quantizer)
    mock_quantized_graph.assert_called_once_with(*mock_example_inputs)
    mock_convert.assert_called_once_with(mock_quantized_graph)
    mock_torch.export.export.assert_called_once_with(
        mock_quantized_graph, mock_example_inputs
    )
    assert result == mock_exported


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
def test_move_output_file_success(mock_move: Mock) -> None:
    """Test successful output file move."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = DEFAULT_BASE_NAME

        source_dir = output_dir / base_name
        source_dir.mkdir()
        (source_dir / EXPECTED_OUTPUT_FILENAME).write_text(
            "tosa data", encoding="utf-8"
        )

        mock_move.side_effect = lambda _src, dst: Path(dst).write_text(
            "tosa data", encoding="utf-8"
        )

        pytorch_file = Path(tmpdir) / "model.pt2"

        # pylint: disable=protected-access
        result = converter._move_output_file(pytorch_file, output_dir, base_name)
        # pylint: enable=protected-access

        assert result == output_dir / "model.tosa"
        mock_move.assert_called_once()


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
def test_move_output_file_target_not_created(mock_move: Mock) -> None:
    """Test error when target file isn't created after move."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = DEFAULT_BASE_NAME

        source_dir = output_dir / base_name
        source_dir.mkdir()
        (source_dir / EXPECTED_OUTPUT_FILENAME).write_text(
            "tosa data", encoding="utf-8"
        )

        mock_move.side_effect = lambda src, dst: None

        pytorch_file = Path(tmpdir) / "model.pt2"

        with pytest.raises(
            FileNotFoundError, match="No output from the TOSA Converter"
        ):
            # pylint: disable=protected-access
            converter._move_output_file(pytorch_file, output_dir, base_name)
            # pylint: enable=protected-access


@pytest.mark.parametrize(
    "example_inputs",
    [
        # Tuple format: (args, kwargs) -> extract args
        ([Mock()], {}),
        # List format: just return as-is
        [Mock()],
    ],
)
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion._import_dependencies")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.torch")
def test_load_pytorch_model_input_formats(
    mock_torch: Mock,
    _mock_import: Mock,
    example_inputs: Any,
) -> None:
    """Test loading PyTorch model handles different example_inputs formats."""
    mock_loaded = Mock()
    mock_graph_module = Mock()
    mock_loaded.module.return_value = mock_graph_module
    mock_loaded.example_inputs = example_inputs

    mock_torch.export.load.return_value = mock_loaded

    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        pt2_file = Path(tmpdir) / "model.pt2"
        pt2_file.write_text("test", encoding="utf-8")

        # pylint: disable=protected-access
        graph_module, result_inputs = converter._load_pytorch_model(pt2_file)
        # pylint: enable=protected-access

        assert graph_module == mock_graph_module
        if isinstance(example_inputs, tuple):
            assert result_inputs == example_inputs[0]
        else:
            assert result_inputs == example_inputs
