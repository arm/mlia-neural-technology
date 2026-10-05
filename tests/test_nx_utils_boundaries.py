# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology runtime boundary gateways."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mlia.core.errors import ConfigurationError
from mlia.nx_utils.boundaries import process
from mlia.nx_utils.boundaries.filesystem import materialize_user_output_file
from mlia.transformers.registry import TransformRequest
from mlia.utils.proc import Command


def test_transform_model_logs_before_action(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Model transformation is announced before invoking the transformer."""
    logger = logging.getLogger("test.neural.boundaries.transform")
    request = TransformRequest(
        model=tmp_path / "model.tflite",
        output_dir=tmp_path / "output",
        target_format="tosa",
        transform_options={},
    )
    expected = tmp_path / "output" / "model.tosa"

    def transform(transform_request: TransformRequest) -> Path:
        assert transform_request is request
        assert caplog.records
        assert getattr(caplog.records[-1], "boundary_action", False) is True
        return expected

    monkeypatch.setattr(process, "transform_model", transform)

    with caplog.at_level(logging.INFO, logger=logger.name):
        assert process.transform_model_with_notice(logger, request) == expected

    assert "Transforming model" in caplog.records[-1].message
    assert str(request.output_dir) in caplog.records[-1].message


@pytest.mark.parametrize(
    ("gateway_name", "notice"),
    [
        (
            "process_model_converter_output_with_notice",
            "Running ML SDK Model Converter",
        ),
        (
            "process_performance_estimator_output_with_notice",
            "Running NX Performance Estimator",
        ),
    ],
)
def test_process_gateway_logs_before_action_without_command_arguments(
    gateway_name: str,
    notice: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Backend execution is announced without exposing command arguments."""
    logger = logging.getLogger(f"test.neural.boundaries.{gateway_name}")
    command = Command(["tool", "--token", "secret-value"])

    def run(command_to_run: Command, _consumers: object) -> None:
        assert command_to_run is command
        assert caplog.records
        assert getattr(caplog.records[-1], "boundary_action", False) is True

    monkeypatch.setattr(process, "process_command_output", run)
    gateway = getattr(process, gateway_name)

    with caplog.at_level(logging.INFO, logger=logger.name):
        gateway(logger, tmp_path, command, [])

    assert notice in caplog.records[-1].message
    assert str(tmp_path) in caplog.records[-1].message
    assert "secret-value" not in caplog.records[-1].message


def test_materialize_user_output_file_is_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    """Materializing output preserves matching files and rejects conflicts."""
    source = tmp_path / "source" / "model.spv"
    source.parent.mkdir()
    source.write_bytes(b"model")
    destination = tmp_path / "output" / source.name

    assert materialize_user_output_file(source, destination) == destination
    assert destination.read_bytes() == b"model"
    assert materialize_user_output_file(source, destination) == destination

    destination.write_bytes(b"different")
    with pytest.raises(ConfigurationError, match="different contents"):
        materialize_user_output_file(source, destination)
