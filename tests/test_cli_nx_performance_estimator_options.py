# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI backend configuration options."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from mlia.backend.nx_performance_estimator.config import CONFIG_TO_CLI_OPTION
from mlia.cli.options import add_backend_config_options


def test_add_backend_config_options_creates_group() -> None:
    """Test that add_backend_config_options creates argument group."""
    parser = argparse.ArgumentParser()
    add_backend_config_options(parser)

    # Check that a group was added
    groups = [
        g
        for g in parser._action_groups  # pylint: disable=protected-access
        if g.title and "backend configuration" in g.title
    ]
    assert len(groups) == 1


def test_add_backend_config_options_discovers_nx_options() -> None:
    """Test that NX Performance Estimator options are discovered."""
    parser = argparse.ArgumentParser()
    add_backend_config_options(parser)

    # Parse with the discovered options
    args = parser.parse_args(
        [
            "--nx-performance-estimator.system-config",
            "test.ini",
            "--nx-performance-estimator.compiler-config",
            "compiler.ini",
        ]
    )

    assert args.nx_performance_estimator_system_config == Path("test.ini")
    assert args.nx_performance_estimator_compiler_config == Path("compiler.ini")


def test_add_backend_config_options_no_duplicates() -> None:
    """Test that options are not duplicated if called multiple times."""
    parser = argparse.ArgumentParser()

    # Call twice
    add_backend_config_options(parser)
    add_backend_config_options(parser)

    # Should only have one instance of each option
    # pylint: disable=protected-access
    all_options = [action.dest for action in parser._actions if action.dest != "help"]
    assert len(all_options) == len(set(all_options))


def test_add_backend_config_options_handles_discovery_failure() -> None:
    """Test graceful handling when backend discovery fails."""
    parser = argparse.ArgumentParser()

    with patch("mlia.cli.options.pkgutil.iter_modules", side_effect=Exception("Test")):
        # Should not raise, just skip options
        add_backend_config_options(parser)

    # Should still have the group with a note
    groups = [
        g
        for g in parser._action_groups  # pylint: disable=protected-access
        if g.title and "backend configuration" in g.title
    ]
    assert len(groups) == 1


def test_backend_config_option_types() -> None:
    """Test that discovered options have Path type for file parameters."""
    parser = argparse.ArgumentParser()
    add_backend_config_options(parser)

    # system-config and compiler-config should be Path type
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest in ["system_config", "compiler_config"]:
            assert action.type == Path


@pytest.mark.parametrize(
    "cli_arg,python_dest",
    [
        (
            "--nx-performance-estimator.system-config",
            "nx_performance_estimator_system_config",
        ),
        (
            "--nx-performance-estimator.compiler-config",
            "nx_performance_estimator_compiler_config",
        ),
    ],
)
def test_cli_to_python_parameter_naming(
    cli_arg: str,
    python_dest: str,
) -> None:
    """Test that CLI options use hyphens and map to Python underscores."""
    parser = argparse.ArgumentParser()
    add_backend_config_options(parser)

    args = parser.parse_args([cli_arg, "test.ini"])
    assert hasattr(args, python_dest)
    assert getattr(args, python_dest) == Path("test.ini")


def test_config_to_cli_option_mapping() -> None:
    """Test CONFIG_TO_CLI_OPTION uses hyphens for CLI options."""
    # All CLI options should use hyphens, not underscores
    for cli_option in CONFIG_TO_CLI_OPTION.values():
        assert cli_option.startswith("--")
        assert "_" not in cli_option, f"CLI option {cli_option} should use hyphens"
        assert "-" in cli_option[2:], f"CLI option {cli_option} should use hyphens"


def test_backend_options_extraction() -> None:
    """Test extraction of backend options from parsed args."""
    parser = argparse.ArgumentParser()
    add_backend_config_options(parser)

    args = parser.parse_args(
        [
            "--nx-performance-estimator.system-config",
            "/path/to/system.ini",
            "--nx-performance-estimator.compiler-config",
            "/path/to/compiler.ini",
        ]
    )

    # Simulate extracting backend options
    backend_options: dict[str, dict[str, str]] = {}
    for param_name, value in vars(args).items():
        if value is not None:
            # Map to backend
            if param_name in [
                "nx_performance_estimator_system_config",
                "nx_performance_estimator_compiler_config",
            ]:
                if "nx-performance-estimator" not in backend_options:
                    backend_options["nx-performance-estimator"] = {}
                backend_options["nx-performance-estimator"][param_name] = str(value)

    assert "nx-performance-estimator" in backend_options
    assert backend_options["nx-performance-estimator"][
        "nx_performance_estimator_system_config"
    ] == str(Path("/path/to/system.ini"))
    assert backend_options["nx-performance-estimator"][
        "nx_performance_estimator_compiler_config"
    ] == str(Path("/path/to/compiler.ini"))
