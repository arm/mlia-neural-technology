# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Process boundary gateways for Neural Technology backends."""

from __future__ import annotations

import logging
from pathlib import Path

from mlia.transformers.registry import TransformRequest, transform_model
from mlia.utils.proc import Command, OutputConsumer, process_command_output

try:
    from mlia.utils.logging import log_boundary_action
except ImportError:  # pragma: no cover - compatibility with older MLIA cores

    def log_boundary_action(
        logger: logging.Logger, message: str, *args: object
    ) -> None:
        """Log a boundary action when the Core helper is unavailable."""
        logger.info(message, *args, extra={"boundary_action": True})


def transform_model_with_notice(
    logger: logging.Logger, request: TransformRequest
) -> Path:
    """Announce and run a registered model transformer."""
    log_boundary_action(
        logger,
        "Transforming model '%s'; derived output will be written under '%s'.",
        request.model,
        request.output_dir,
    )
    return transform_model(request)


def process_model_converter_output_with_notice(
    logger: logging.Logger,
    output_dir: Path,
    command: Command,
    consumers: list[OutputConsumer],
) -> None:
    """Announce and execute the ML SDK Model Converter backend."""
    log_boundary_action(
        logger,
        "Running ML SDK Model Converter; derived output will be written under '%s'.",
        output_dir,
    )
    process_command_output(command, consumers)


def process_performance_estimator_output_with_notice(
    logger: logging.Logger,
    output_dir: Path,
    command: Command,
    consumers: list[OutputConsumer],
) -> None:
    """Announce and execute the NX Performance Estimator."""
    log_boundary_action(
        logger,
        "Running NX Performance Estimator; derived output will be written under '%s'.",
        output_dir,
    )
    process_command_output(command, consumers)
