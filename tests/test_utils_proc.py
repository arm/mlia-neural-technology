# SPDX-FileCopyrightText: Copyright 2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for process management functions."""
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

from mlia.utils.proc import args_from_cfg
from mlia.utils.proc import Command
from mlia.utils.proc import process_command_output


def test_process_command_output() -> None:
    """Test function process_command_output."""
    command = Command(["echo", "-n", "sample message"])

    output_consumer = MagicMock()
    process_command_output(command, [output_consumer])

    output_consumer.assert_called_once_with("sample message")


def test_args_from_cfg() -> None:
    """Test function args_from_cfg."""

    @dataclass
    class Config:
        """Test configuration class."""

        alpha: int = 10
        beta: str = "string"
        gamma: Optional[str] = None

    cfg_to_arg = {"alpha": "NumAlphas", "beta": "BetaOption", "gamma": "NoOption"}
    assert args_from_cfg(Config(), cfg_to_arg) == [
        "NumAlphas",
        "10",
        "BetaOption",
        "string",
    ]
