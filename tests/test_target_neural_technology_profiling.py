# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for profiling data in the Neural Technology advisor workflow."""

from pathlib import Path
import sys
from typing import cast
from unittest.mock import MagicMock

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
from mlia.target.neural_technology import data_collection
from mlia.target.neural_technology.advisor import (
    NeuralTechnologyInferenceAdvisor,
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.data_collection import (
    NXProfilingDataResult,
    NeuralTechnologyProfilingData,
)


def _profiling_context(
    tmp_path: Path,
    *,
    categories: set[AdviceCategory] | None = None,
    model: Path | None = None,
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


def test_advisor_passes_optional_model_to_profiling_data_collector(
    tmp_path: Path,
) -> None:
    """A source model should remain available for richer measured output."""
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    context, advisor = _profiling_context(tmp_path, model=model)

    collector = cast(NeuralTechnologyProfilingData, advisor.get_collectors(context)[0])

    assert collector.model == model


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


def test_profiling_data_collector_returns_standardized_workflow_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collector should expose standardized output to core collection."""
    profiling_data = tmp_path / "profiling-data"
    profiling_data.mkdir()
    model = tmp_path / "model.vgf"
    model.write_bytes(b"vgf")
    output: dict[str, object] = {"schema_version": "1.2.0", "results": []}
    analyze = MagicMock(return_value=output)
    monkeypatch.setattr(data_collection, "analyze_profiling_data", analyze)
    executable = tmp_path / "mlia"
    monkeypatch.setattr(sys, "argv", [str(executable), "check", "model.vgf"])
    collector = NeuralTechnologyProfilingData(
        profiling_data=[profiling_data],
        target_profile="neural-technology",
        model=model,
    )
    collector.set_context(ExecutionContext(output_dir=tmp_path))

    result = collector.collect_data()

    assert result == NXProfilingDataResult(standardized_output=output)
    analyze.assert_called_once_with(
        target_profile="neural-technology",
        profiling_data=[profiling_data],
        categories={"performance"},
        model=str(model),
        cli_arguments=["mlia", "check", "model.vgf"],
        output_dir=collector.context.output_dir,
    )
