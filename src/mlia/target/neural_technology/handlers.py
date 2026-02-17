# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Event handler."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mlia.backend.ml_sdk_model_converter.compat import NXModelCompatibilityInfo
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.target.neural_technology.data_collection import (
    NXCompatibilityResult,
    NXPerformanceResult,
)
from mlia.target.neural_technology.events import (
    NeuralTechnologyAdvisorEventHandler,
    NeuralTechnologyAdvisorStartedEvent,
)
from mlia.target.neural_technology.reporters import neural_technology_formatters

logger = logging.getLogger(__name__)


class NeuralTechnologyEventHandler(
    WorkflowEventsHandler, NeuralTechnologyAdvisorEventHandler
):
    """CLI event handler."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Init event handler."""
        super().__init__(neural_technology_formatters)
        self.output_dir = output_dir

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, NXPerformanceResult):
            # Save standardized output JSON if available
            if data_item.standardized_output and self.output_dir:
                try:
                    output_path = self.output_dir / "nx_performance.json"
                    with open(output_path, "w", encoding="utf-8") as file_handle:
                        json.dump(data_item.standardized_output, file_handle, indent=2)
                    logger.info("Saved NX performance output to %s", output_path)
                except Exception as exc:
                    logger.warning("Failed to save NX performance output: %s", exc)

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True, space=True)

        elif isinstance(data_item, NXCompatibilityResult):
            # Save standardized output JSON if available
            if data_item.standardized_output and self.output_dir:
                try:
                    output_path = self.output_dir / "nx_compatibility.json"
                    with open(output_path, "w", encoding="utf-8") as file_handle:
                        json.dump(data_item.standardized_output, file_handle, indent=2)
                    logger.info("Saved NX compatibility output to %s", output_path)
                except Exception as exc:
                    logger.warning("Failed to save NX compatibility output: %s", exc)

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True, space=True)

        elif isinstance(
            data_item,
            (
                NXPerformanceEstimatorPerformanceMetrics,
                NXModelCompatibilityInfo,
            ),
        ):
            self.reporter.submit(data_item, delay_print=True, space=True)

    def on_neural_technology_advisor_started(
        self, event: NeuralTechnologyAdvisorStartedEvent
    ) -> None:
        """Handle NeuralTechnologyAdvisorStarted event."""
        self.reporter.submit(event.device)
