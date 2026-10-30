# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Technology profiling-data backend registration."""

import click

from mlia.backend.config import BackendConfiguration, BackendType
from mlia.backend.registry import BackendRegistry
from mlia.core.common import AdviceCategory
from mlia.plugins.plugins import BackendPlugin
from mlia.target.neural_technology.filtering import DEFAULT_COLLAPSE_RULES


class NeuralTechnologyProfilingDataPlugin(BackendPlugin):
    """Register the structured measured-data backend."""

    plugin_interface_version = "0.0.2"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the profiling-data performance backend."""
        registry.register(
            "neural-technology-profiling-data",
            BackendConfiguration(
                supported_advice=[AdviceCategory.PERFORMANCE],
                cli_options={
                    "etrecord_method": click.Option(
                        ["--etrecord-method"],
                        type=str,
                        default=None,
                        help="ETRecord method to associate with this measured capture.",
                    ),
                },
                supported_systems=None,
                backend_type=BackendType.BUILTIN,
                installation=None,
                selectable=False,
                supports_estimation=False,
                supports_profiling_data=True,
                default_collapse_rules=DEFAULT_COLLAPSE_RULES,
            ),
            pretty_name="Neural Technology Profiling Data",
        )
