# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology MLIA module."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.registry import registry as backend_registry
from mlia.backend.registry import BackendRegistry
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.target.neural_technology.advisor import (
    NeuralTechnologyInferenceAdvisor,
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.plugin import NeuralTechnologyTargetPlugin
from mlia.target.registry import TargetRegistry


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert (
        NeuralTechnologyInferenceAdvisor.name() == "neural_technology_inference_advisor"
    )


@pytest.mark.parametrize(
    "advice_category", [(AdviceCategory.PERFORMANCE), (AdviceCategory.COMPATIBILITY)]
)
def test_configure_and_get_neural_technology_advisor(
    test_tflite_model: Path, advice_category: AdviceCategory
) -> None:
    """Test Neural Technology advisor configuration."""
    ctx = ExecutionContext(advice_category={advice_category})

    advisor = configure_and_get_neural_technology_advisor(
        ctx,
        "neural-technology",
        test_tflite_model,
        backends=["nx-performance-estimator"],
    )
    workflow = advisor.configure(ctx)

    assert isinstance(advisor, NeuralTechnologyInferenceAdvisor)

    assert ctx.config_parameters["neural_technology_inference_advisor"] == {
        "backend_options": {},
        "backends": ["nx-performance-estimator"],
        "model": str(test_tflite_model),
        "target_profile": "neural-technology",
    }

    assert isinstance(workflow, DefaultWorkflowExecutor)


def test_configure_and_get_neural_technology_advisor_invalid_backends(
    test_tflite_model: Path,
) -> None:
    """
    Test for raising an error when either no or too many backends are provided to
    configure_and_get_neural_technology_advisor function.
    """
    ctx = ExecutionContext(advice_category={AdviceCategory.PERFORMANCE})

    with pytest.raises(
        ConfigurationError, match="One backend is required but was not specified."
    ):
        configure_and_get_neural_technology_advisor(
            ctx,
            "neural-technology",
            test_tflite_model,
            backends=[],
        )

    backends = ["nx-performance-estimator", "vela"]
    with pytest.raises(
        ConfigurationError,
        match=f"Only one backend is supported but {len(backends)} "
        + f"were provided: {re.escape(str(backends))}",
    ):
        configure_and_get_neural_technology_advisor(
            ctx,
            "neural-technology",
            test_tflite_model,
            backends=backends,
        )


def test_target_registry_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neural Technology target should register without event handlers."""
    monkeypatch.setitem(backend_registry.items, "nx-performance-estimator", MagicMock())
    registry = TargetRegistry()
    NeuralTechnologyTargetPlugin.register(registry)

    info = registry.items["neural-technology"]
    assert info.default_backends == ["nx-performance-estimator"]
    assert info.supports_torch_module is True
    assert info.torch_module_backend == "nx-performance-estimator"


def test_nx_performance_estimator_plugin_registers_cli_options() -> None:
    """NX backend plugin should expose API-discoverable backend options."""
    registry = BackendRegistry()
    NXPerformanceEstimatorPlugin.register(registry)

    assert registry.items["nx-performance-estimator"].cli_options == {
        "system_config": "--system-config",
        "compiler_config": "--compiler-config",
    }
