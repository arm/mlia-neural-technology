# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert TensorFlow Lite models with the TOSA Converter For Tflite."""

from __future__ import annotations

import logging
from pathlib import Path

from mlia.utils.logging import log_action
from mlia.utils.proc import (
    Command,
    OutputConsumer,
    OutputLogger,
    process_command_output,
)

logger = logging.getLogger(__name__)


class TosaConverterForTflite:
    """Wrapper class to run the TOSA Converter For Tflite."""

    CONVERTER_EXE = "tosa-converter-for-tflite"

    def __init__(self) -> None:
        """Set up some paths to run the TOSA Converter For Tflite."""
        self.output_consumers: list[OutputConsumer] = [
            OutputLogger(logger, logging.INFO)
        ]

    def __call__(self, tflite_file: Path, output_dir: Path) -> Path:
        """
        Run the TOSA Converter For Tflite with the given TensorFlow Lite file.

        Returns the path of the SPIR-V output archive created in the output dir.
        """
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                "TOSA Converter For Tflite."
            )
        if not tflite_file.is_file():
            raise FileNotFoundError(
                f"TensorFlow Lite model file '{tflite_file}' not found. "
                "Unable to run TOSA Converter For Tflite."
            )
        with log_action("Running TOSA Converter For Tflite..."):
            logger.debug("TOSA Converter For Tflite:")

            tosa_file = self._run_converter(tflite_file, output_dir)

            logger.debug("Output file: %s", tosa_file)

        return tosa_file

    def _create_converter_command(self, tflite_file: Path, tosa_file: Path) -> Command:
        """Create the command to run the TOSA Converter For Tflite."""
        cmd = Command(
            cmd=[
                str(self.CONVERTER_EXE),
                str(tflite_file),
                "-o",
                str(tosa_file),
                *self._extra_arguments(),
            ],
        )
        return cmd

    def _run_converter(self, tflite_file: Path, output_dir: Path) -> Path:
        """Run the TOSA Converter For Tflite and return the TOSA MLIR output file."""
        tosa_file = output_dir / f"{tflite_file.stem}.tosamlir"
        cmd = self._create_converter_command(tflite_file, tosa_file)
        process_command_output(cmd, self.output_consumers)

        if not tosa_file.is_file():
            raise FileNotFoundError(
                "No output from the TOSA Converter For Tflite found. "
                f"File {tosa_file} does not exist."
            )
        logger.debug(
            "TOSA Converter For Tflite run successfully. See output: %s",
            tosa_file,
        )

        return tosa_file

    def _extra_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the TCFT."""
        return ["--text"]
