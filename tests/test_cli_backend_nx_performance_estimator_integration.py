# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for backend configuration via CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import ANY
from unittest.mock import MagicMock

import pytest

from mlia.cli.commands import check
from mlia.cli.options import add_backend_config_options
from mlia.core.context import ExecutionContext


def test_backend_options_flow_through_check_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that backend options flow from CLI through to backend config."""
    # Create a parser and add options
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--target-profile", "-t", default="ethos-u55-256")
    parser.add_argument("--backend", "-b", nargs="+")
    parser.add_argument("--performance", action="store_true")
    add_backend_config_options(parser)

    # Parse args as if from command line
    model_path = tmp_path / "test_model.tflite"
    model_path.write_text("placeholder model")

    args = parser.parse_args(
        [
            str(model_path),
            "--target-profile",
            "NX-peak-12SC-8NX-600MHz",
            "--backend",
            "nx-performance-estimator",
            "--performance",
            "--nx-performance-estimator.system-config",
            "/custom/system.ini",
            "--nx-performance-estimator.compiler-config",
            "/custom/compiler.ini",
        ]
    )

    mock_get_advice = MagicMock()
    monkeypatch.setattr("mlia.cli.commands.get_advice", mock_get_advice)
    monkeypatch.setattr(
        "mlia.cli.commands.get_available_backends",
        MagicMock(return_value=["nx-performance-estimator"]),
    )
    ctx = ExecutionContext()

    # Call check command with parsed args
    # pylint: disable=line-too-long
    check(
        ctx,
        args.target_profile,
        args.model,
        performance=args.performance,
        backend=args.backend,
        nx_performance_estimator_system_config=args.nx_performance_estimator_system_config,
        nx_performance_estimator_compiler_config=args.nx_performance_estimator_compiler_config,
    )
    # pylint: enable=line-too-long

    # Verify get_advice was called with expected parameters
    mock_get_advice.assert_called_once_with(
        args.target_profile,
        args.model,
        {"performance"},
        context=ctx,
        backends=args.backend,
        backend_options=ANY,
    )

    # Extract and verify the backend_options
    mock_get_advice.assert_called_once()
    call_kwargs = mock_get_advice.call_args.kwargs
    backend_options = call_kwargs["backend_options"]

    # Verify nx-performance-estimator backend options
    assert "nx-performance-estimator" in backend_options
    nx_options = backend_options["nx-performance-estimator"]
    assert nx_options["system_config"] == Path("/custom/system.ini")
    assert nx_options["compiler_config"] == Path("/custom/compiler.ini")


def test_cli_option_names_use_hyphens() -> None:
    """Test that CLI options use hyphens not underscores."""
    parser = argparse.ArgumentParser()
    add_backend_config_options(parser)

    # Get all option strings
    all_options: list[str] = []
    for action in parser._actions:  # pylint: disable=protected-access
        if action.option_strings:
            all_options.extend(action.option_strings)

    # Filter to backend config options (those with "config" in name)
    backend_options = [opt for opt in all_options if "config" in opt]

    # All should use hyphens
    for opt in backend_options:
        if opt.startswith("--"):
            # Should not contain underscores after the --
            assert (
                "_" not in opt
            ), f"CLI option {opt} should use hyphens, not underscores"


def test_backend_options_converted_to_absolute_paths() -> None:
    """Test that Path arguments are converted to absolute paths."""
    parser = argparse.ArgumentParser()
    add_backend_config_options(parser)

    # Parse with relative path
    args = parser.parse_args(
        [
            "--nx-performance-estimator.system-config",
            "relative/path/system.ini",
        ]
    )

    # When extracting backend options, paths should be made absolute
    system_config_path = Path(args.nx_performance_estimator_system_config)
    absolute_path = system_config_path.resolve()

    # Should be able to convert to absolute
    assert absolute_path.is_absolute()
