# SPDX-FileCopyrightText: Copyright 2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Technology advice generation."""

from functools import singledispatchmethod

from mlia.core.advice_generation import FactBasedAdviceProducer, advice_category
from mlia.core.common import AdviceCategory, DataItem
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity
from mlia.target.neural_technology.data_analysis import (
    NXPerformanceEstimatorModelPerformanceAnalyzed,
)


class NeuralTechnologyAdviceProducer(FactBasedAdviceProducer):
    """Neural Technology advice producer."""

    @singledispatchmethod
    def produce_advice(self, _data_item: DataItem) -> None:  # type: ignore[override]
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.PERFORMANCE)
    def handle_nx_performance_estimator_performance_analyzed(
        self, _: NXPerformanceEstimatorModelPerformanceAnalyzed
    ) -> None:
        """Advice for NT performance estimated by the NX Performance Estimator."""
        self._point_to_performance_table()

    def _point_to_performance_table(self) -> None:
        """Create generic advice for Neural Technology performance."""
        self.add_advice(
            message=(
                "Please refer to the performance metrics shown in the report "
                "to find possible optimizations."
            ),
            category=SchemaAdviceCategory.PERFORMANCE,
            severity=AdviceSeverity.INFO,
        )
