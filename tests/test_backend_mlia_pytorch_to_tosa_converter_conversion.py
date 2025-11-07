# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to TOSA converter conversion."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import DEFAULT_BASE_NAME
from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import (
    EXPECTED_OUTPUT_FILENAME,
)
from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import (
    MliaPytorchToTosaConverter,
)


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion._import_dependencies")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.torch")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TosaCompileSpec")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TOSAQuantizer")
@patch(
    "mlia.backend.mlia_pytorch_to_tosa_converter.conversion."
    "get_symmetric_quantization_config"
)
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.prepare_pt2e")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.convert_pt2e")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TOSAPartitioner")
@patch(
    "mlia.backend.mlia_pytorch_to_tosa_converter.conversion."
    "to_edge_transform_and_lower"
)
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.EdgeCompileConfig")
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
# pylint: disable=too-many-arguments,too-many-locals,unused-argument
def test_full_conversion_process(
    mock_move: Mock,
    _mock_edge_config: Mock,
    mock_transform: Mock,
    _mock_partitioner: Mock,
    mock_convert_pt2e: Mock,
    mock_prepare_pt2e: Mock,
    _mock_get_config: Mock,
    mock_quantizer: Mock,
    mock_compile_spec: Mock,
    mock_torch: Mock,
    _mock_import_deps: Mock,
) -> None:
    """Test complete conversion flow."""
    # Setup mocks
    mock_exported_program = Mock()
    mock_exported_program.module.return_value = Mock()
    mock_exported_program.example_inputs = [Mock()]
    mock_torch.export.load.return_value = mock_exported_program
    mock_torch.export.export.return_value = Mock()

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

        # Create expected output structure
        source_dir = output_dir / DEFAULT_BASE_NAME
        source_dir.mkdir()
        source_file = source_dir / EXPECTED_OUTPUT_FILENAME
        source_file.write_text("tosa", encoding="utf-8")

        mock_move.side_effect = lambda src, dst: Path(dst).write_text(
            "tosa", encoding="utf-8"
        )

        result = converter(input_file, output_dir)

        # Verify key steps were called
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


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.TOSAPartitioner")
@patch(
    "mlia.backend.mlia_pytorch_to_tosa_converter.conversion.to_edge_transform_and_lower"
)
@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.EdgeCompileConfig")
def test_lowering_failure(
    mock_edge_config: Mock,
    mock_transform: Mock,
    mock_partitioner: Mock,
) -> None:
    """Test TOSA lowering handles errors."""
    mock_partitioner.return_value = Mock()
    mock_edge_config.return_value = Mock()
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
