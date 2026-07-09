# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert TensorFlow Lite models with the ML SDK Model Converter."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from mlia.core.errors import ConfigurationError
from mlia.nx_utils.filesystem import (
    is_pte_file,
    is_pytorch_file,
    is_tosa_file,
    is_vgf_file,
)
from mlia.transformers.error import TransformerNotFoundError
from mlia.transformers.registry import TransformRequest, transform_model
from mlia.utils.logging import log_action
from mlia.utils.proc import (
    Command,
    OutputConsumer,
    OutputLogger,
    process_command_output,
)

logger = logging.getLogger(__name__)


def _is_tflite_file(model: Path) -> bool:
    return model.suffix == ".tflite"


def _converter_unavailable_error(model_file: Path) -> ConfigurationError:
    if _is_tflite_file(model_file):
        return ConfigurationError(
            "TFLite conversion requires the 'mlia-converters-tflite' plugin "
            "to be installed."
        )
    if is_pytorch_file(model_file) or is_pte_file(model_file):
        return ConfigurationError(
            "PyTorch conversion requires the 'mlia-converters-pytorch' plugin "
            "to be installed."
        )
    return ConfigurationError("Transformer for model is not available.")


def get_front_end_output_subdir(model_file: Path) -> str | None:
    """Return the standard output subdir for a converted frontend model."""
    if _is_tflite_file(model_file):
        return "tflite-to-tosa"
    if is_pytorch_file(model_file):
        return "pt2-to-tosa"
    if is_pte_file(model_file):
        return "pte-to-delegate"
    return None


def build_front_end_transform_request(
    model_file: Path,
    output_dir: Path,
    *,
    enable_quantization: bool | None = None,
    output_format: str | None = None,
    emit_debug_info: bool | None = None,
) -> TransformRequest:
    """Build the transformer request for a supported frontend model."""
    if not any(
        [
            _is_tflite_file(model_file),
            is_pytorch_file(model_file),
            is_pte_file(model_file),
        ]
    ):
        raise ConfigurationError("Input must be a TFLite, PyTorch or PTE file.")

    target_format = "delegate" if is_pte_file(model_file) else "tosa"
    transform_options: dict[str, Any] = {}
    if is_pytorch_file(model_file) and enable_quantization is not None:
        transform_options["enable_quantization"] = enable_quantization
    if output_format is not None:
        transform_options["output_format"] = output_format
    if emit_debug_info is not None:
        transform_options["emit_debug_info"] = emit_debug_info

    return TransformRequest(
        model=model_file,
        output_dir=output_dir,
        target_format=target_format,
        transform_options=transform_options,
    )


def transform_front_end_model(
    model_file: Path,
    output_dir: Path,
    *,
    enable_quantization: bool | None = None,
    output_format: str | None = None,
    emit_debug_info: bool | None = None,
) -> Path:
    """Convert a supported frontend input to TOSA/VGF or pass through TOSA."""
    if is_tosa_file(model_file):
        return model_file

    request = build_front_end_transform_request(
        model_file,
        output_dir,
        enable_quantization=enable_quantization,
        output_format=output_format,
        emit_debug_info=emit_debug_info,
    )
    try:
        return transform_model(request)
    except TransformerNotFoundError as err:
        raise _converter_unavailable_error(model_file) from err


class MLSDKModelConverterBase:
    """Wrapper class to run the ML SDK Model Converter."""

    BACK_END_EXE = "model-converter"

    def __init__(
        self, converter_path: Path, *, enable_quantization: bool | None = None
    ) -> None:
        """Set up some paths to run the ML SDK Model Converter."""
        self.converter_path = converter_path.resolve()
        self.enable_quantization = enable_quantization
        self.output_consumers: list[OutputConsumer] = [
            OutputLogger(logger, logging.INFO)
        ]

    def __call__(self, model_file: Path, output_dir: Path) -> Path:
        """
        Run the ML SDK Model Converter with the given model file.

        Returns the path of the VGF file created or extracted in the output dir.
        """
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                "ML SDK Model Converter."
            )
        with log_action("Running ML SDK Model Converter..."):
            logger.debug("ML SDK Model Converter path: %s", self.converter_path)

            converted_model_path = self.run_front_end(model_file, output_dir)
            if is_vgf_file(converted_model_path):
                vgf_file = converted_model_path
            elif is_tosa_file(converted_model_path):
                try:
                    vgf_file = self.run_back_end(converted_model_path, output_dir)
                except subprocess.CalledProcessError:
                    if not (
                        _is_tflite_file(model_file)
                        and converted_model_path.suffix == ".mlirbc"
                    ):
                        raise
                    vgf_file = self.run_back_end(
                        output_dir / f"{model_file.stem}.tosamlir",
                        output_dir,
                    )
            else:
                raise ConfigurationError(
                    "Model conversion frontend output must be a TOSA or VGF file."
                )

            logger.debug("Output file: %s", vgf_file)

        return vgf_file

    def run_front_end(self, model_file: Path, output_dir: Path) -> Path:
        """Convert supported frontend model formats to TOSA or VGF."""
        if not model_file.is_file():
            raise FileNotFoundError(f"Input model file does not exist: {model_file}")

        if _is_tflite_file(model_file):
            converted_model_path = transform_front_end_model(
                model_file,
                output_dir,
                output_format="mlir-bytecode",
                emit_debug_info=True,
            )
            transform_front_end_model(
                model_file,
                output_dir,
                output_format="mlir-text",
                emit_debug_info=True,
            )
        else:
            converted_model_path = transform_front_end_model(
                model_file,
                output_dir,
                enable_quantization=self.enable_quantization,
            )

        if not converted_model_path.is_file():
            raise FileNotFoundError(
                "No output from the model conversion frontend found. "
                f"File {converted_model_path} does not exist."
            )
        logger.debug(
            "Model conversion frontend of ML SDK Model Converter run "
            "successfully. See output: %s",
            converted_model_path,
        )

        return converted_model_path

    def _create_back_end_command(self, tosa_file: Path, vgf_file: Path) -> Command:
        """Create the command to run the front end."""
        cmd = Command(
            cmd=[
                str(self.converter_path / self.BACK_END_EXE),
                "-i",
                str(tosa_file),
                "-o",
                str(vgf_file),
                *self._extra_back_end_arguments(),
            ],
        )
        return cmd

    def run_back_end(self, tosa_file: Path, output_dir: Path) -> Path:
        """Run the backend and return the SPIR-V output archive."""
        vgf_file = output_dir / f"{tosa_file.stem}.vgf"
        cmd = self._create_back_end_command(tosa_file, vgf_file)
        process_command_output(cmd, self.output_consumers)

        return vgf_file

    def _extra_front_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC front-end."""
        return ["--text"]

    def _extra_back_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC back-end."""
        return []


class MLSDKModelConverter(MLSDKModelConverterBase):
    """Run the ML SDK Model Converter to produce a SPIR-v file."""

    def run_back_end(self, tosa_file: Path, output_dir: Path) -> Path:
        """Run the backend and return the SPIR-V output archive."""
        vgf_file = super().run_back_end(tosa_file, output_dir)

        if not vgf_file.is_file():
            raise FileNotFoundError(
                "No output from the ML SDK Model Converter backend found. "
                f"File {vgf_file} does not exist."
            )
        logger.debug(
            "Back end of ML SDK Model Converter run successfully. See output: %s",
            vgf_file,
        )

        return vgf_file

    def _extra_back_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC back-end."""
        return ["--emit-debug-info"]
