# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for profiling data in the Neural Technology advisor workflow."""

from pathlib import Path
from typing import cast

import pytest
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError

from mlia.target.neural_technology.advisor import (
    NeuralTechnologyInferenceAdvisor,
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.data_collection import (
    NeuralTechnologyProfilingData,
)


def _profiling_context(
    tmp_path: Path,
    *,
    categories: set[AdviceCategory] | None = None,
    model: Path | None = None,
    companion_model: Path | None = None,
    backend_options: dict[str, dict[str, object]] | None = None,
) -> tuple[ExecutionContext, NeuralTechnologyInferenceAdvisor]:
    profiling_data = tmp_path / "profiling-data"
    profiling_data.mkdir()
    context = ExecutionContext(
        advice_category=categories or {AdviceCategory.PERFORMANCE},
        output_dir=tmp_path,
    )
    advisor = configure_and_get_neural_technology_advisor(
        context,
        "neural-technology",
        model,
        backends=["neural-technology-profiling-data"],
        profiling_data=[profiling_data],
        companion_model=companion_model,
        backend_options=backend_options or {},
    )
    return context, cast(NeuralTechnologyInferenceAdvisor, advisor)


def test_advisor_selects_profiling_data_collector_without_model(
    tmp_path: Path,
) -> None:
    """Profiling data should use the standard advisor collector list."""
    context, advisor = _profiling_context(tmp_path)

    collectors = advisor.get_collectors(context)

    assert len(collectors) == 1
    collector = cast(NeuralTechnologyProfilingData, collectors[0])
    assert collector.profiling_data == [tmp_path / "profiling-data"]
    assert collector.model is None


def test_advisor_rejects_non_vgf_model_with_profiling_data(tmp_path: Path) -> None:
    """Profiling correlation requires a VGF canonical model population."""
    model = tmp_path / "model.tflite"
    model.write_text("model", encoding="utf-8")
    context, advisor = _profiling_context(tmp_path, model=model)

    with pytest.raises(ConfigurationError, match="only be associated with a VGF"):
        advisor.get_collectors(context)


def test_advisor_rejects_profiling_compatibility(tmp_path: Path) -> None:
    """Unsupported measured compatibility requests should fail clearly."""
    context, advisor = _profiling_context(
        tmp_path,
        categories={AdviceCategory.PERFORMANCE, AdviceCategory.COMPATIBILITY},
    )

    with pytest.raises(ConfigurationError, match="does not support --compatibility"):
        advisor.get_collectors(context)


def test_advisor_rejects_profiling_backend_options(tmp_path: Path) -> None:
    """Unsupported profiling-data options should not be silently ignored."""
    context, advisor = _profiling_context(
        tmp_path,
        backend_options={"neural-technology-profiling-data": {"device": "Mali-G1"}},
    )

    with pytest.raises(ConfigurationError, match="does not support backend options"):
        advisor.get_collectors(context)


def test_companion_requires_measured_analysis(tmp_path: Path) -> None:
    """A companion input cannot be silently ignored by estimation."""
    model = tmp_path / "model.etrecord"
    model.touch()
    companion = tmp_path / "model.pte"
    companion.touch()
    context = ExecutionContext(
        advice_category={AdviceCategory.PERFORMANCE}, output_dir=tmp_path
    )
    advisor = configure_and_get_neural_technology_advisor(
        context,
        "neural-technology",
        model,
        backends=["nx-performance-estimator"],
        companion_model=companion,
    )
    with pytest.raises(ConfigurationError, match="requires ETRecord profiling data"):
        advisor.get_collectors(context)
