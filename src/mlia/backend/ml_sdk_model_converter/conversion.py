# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert TensorFlow Lite models with the ML SDK Model Converter."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Callable, cast

from mlia.core.errors import ConfigurationError
from mlia.plugins.converter_registry import ConverterRegistry
from mlia.plugins.plugins import load_converter_plugins
from mlia.nx_utils.filesystem import is_pytorch_file, is_tosa_file
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


ConverterFn = Callable[..., Path]


def _get_converter(name: str) -> ConverterFn:
    registry = ConverterRegistry()
    load_converter_plugins(registry)
    converter = registry.get(name)
    if converter is None:
        if name == "tflite_to_tosa":
            raise ConfigurationError(
                "TFLite conversion requires the 'mlia-converters-tflite' plugin "
                "to be installed."
            )
        if name == "pt2_to_tosa":
            raise ConfigurationError(
                "PyTorch conversion requires the 'mlia-converters-pytorch' plugin to be installed."
            )
        raise ConfigurationError(f"Converter '{name}' is not available.")
    return cast(ConverterFn, converter)


def _supports_enable_quantization(converter: ConverterFn) -> bool:
    parameters = inspect.signature(converter).parameters.values()
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or parameter.name == "enable_quantization"
        for parameter in parameters
    )


def run_named_converter(
    name: str,
    model_file: Path,
    output_dir: Path,
    *,
    enable_quantization: bool | None = None,
) -> Path:
    """Run a registered converter, passing enable_quantization when supported."""
    converter = _get_converter(name)
    if enable_quantization is not None and _supports_enable_quantization(converter):
        return converter(
            model_file,
            output_dir,
            enable_quantization=enable_quantization,
        )
    if name == "pt2_to_tosa" and enable_quantization is False:
        raise ConfigurationError(
            "PyTorch conversion requires an 'mlia-converters-pytorch' plugin "
            "version that supports enable_quantization. Please upgrade "
            "mlia-converters-pytorch."
        )
    return converter(model_file, output_dir)


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

    def __call__(self, tflite_file: Path, output_dir: Path) -> Path:
        """
        Run the ML SDK Model Converter with the given TensorFlow Lite file.

        Returns the path of the SPIR-V output archive created in the output dir.
        """
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                "ML SDK Model Converter."
            )
        with log_action("Running ML SDK Model Converter..."):
            logger.debug("ML SDK Model Converter path: %s", self.converter_path)

            tosa_file = self.run_front_end(tflite_file, output_dir)
            vgf_file = self.run_back_end(tosa_file, output_dir)

            logger.debug("Output file: %s", vgf_file)

        return vgf_file

    def run_front_end(self, model_file: Path, output_dir: Path) -> Path:
        """Run the TosaConverterForTflite frontend."""
        # Check the file extension to see if we've been given a tosa file
        if is_tosa_file(model_file):
            tosa_file = model_file
        # Otherwise try to convert the file to tosa
        elif is_pytorch_file(model_file):
            tosa_file = run_named_converter(
                "pt2_to_tosa",
                model_file,
                output_dir,
                enable_quantization=(
                    self.enable_quantization
                    if self.enable_quantization is not None
                    else True
                ),
            )
        elif _is_tflite_file(model_file):
            tosa_file = run_named_converter("tflite_to_tosa", model_file, output_dir)
        else:
            raise ConfigurationError("Input must be a TOSA, TFLite or PyTorch file.")

        if not tosa_file.is_file():
            raise FileNotFoundError(
                "No output from the TosaConverterForTflite frontend found. "
                f"File {tosa_file} does not exist."
            )
        logger.debug(
            "TosaConverterForTflite Frontend of ML SDK Model Converter run "
            + "successfully. See output: %s",
            tosa_file,
        )

        return tosa_file

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
