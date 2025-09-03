# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert TensorFlow Lite models with the Vulkan Model Converter."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.backend.tosa_converter_for_tflite.conversion import TosaConverterForTflite
from mlia.utils.logging import log_action
from mlia.utils.proc import Command
from mlia.utils.proc import OutputConsumer
from mlia.utils.proc import OutputLogger
from mlia.utils.proc import process_command_output

logger = logging.getLogger(__name__)


class VulkanModelConverterBase:
    """Wrapper class to run the Vulkan Model Converter."""

    BACK_END_EXE = "model-converter"

    def __init__(self, converter_path: Path) -> None:
        """Set up some paths to run the Vulkan Model Converter."""
        self.converter_path = converter_path.resolve()
        self.output_consumers: list[OutputConsumer] = [
            OutputLogger(logger, logging.INFO)
        ]

    def __call__(self, tflite_file: Path, output_dir: Path) -> Path:
        """
        Run the Vulkan Model Converter with the given TensorFlow Lite file.

        Returns the path of the SPIR-V output archive created in the output dir.
        """
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                "Vulkan Model Converter."
            )
        with log_action("Running Vulkan Model Converter..."):
            logger.debug("Vulkan Model Converter path: %s", self.converter_path)

            tosa_file = self._run_front_end(tflite_file, output_dir)
            vgf_file = self._run_back_end(tosa_file, output_dir)

            logger.debug("Output file: %s", vgf_file)

        return vgf_file

    def _convert_file(self, tflite_file: Path, output_dir: Path) -> Path:
        """Run the TosaConverterForTflite to convert the tflite file to tosa."""
        model_converter = TosaConverterForTflite()
        return model_converter(tflite_file, output_dir)

    def _run_front_end(self, tflite_file: Path, output_dir: Path) -> Path:
        """Run the TosaConverterForTflite frontend."""
        tosa_file = self._convert_file(tflite_file, output_dir)

        if not tosa_file.is_file():
            raise FileNotFoundError(
                "No output from the TosaConverterForTflite frontend found. "
                f"File {tosa_file} does not exist."
            )
        logger.debug(
            "TosaConverterForTflite Frontend of Vulkan Model Converter run "
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

    def _run_back_end(self, tosa_file: Path, output_dir: Path) -> Path:
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


# pylint: disable=too-few-public-methods
class VulkanModelConverter(VulkanModelConverterBase):
    """Run the Vulkan Model Converter to produce a SPIR-v file."""

    def _run_back_end(self, tosa_file: Path, output_dir: Path) -> Path:
        """Run the backend and return the SPIR-V output archive."""
        vgf_file = super()._run_back_end(tosa_file, output_dir)

        if not vgf_file.is_file():
            raise FileNotFoundError(
                "No output from the Vulkan Model Converter backend found. "
                f"File {vgf_file} does not exist."
            )
        logger.debug(
            "Back end of Vulkan Model Converter run successfully. See output: %s",
            vgf_file,
        )

        return vgf_file

    def _extra_back_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC back-end."""
        return ["--emit-debug-info"]
