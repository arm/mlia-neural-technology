# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology MLIA module."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mlia.target.neural_technology.plugin
from mlia.backend.registry import registry as backend_registry
from mlia.backend.registry import BackendRegistry
from mlia.backend.nx_performance_estimator.config import (
    NXPerformanceEstimatorConfig,
)
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin
from mlia.backend.nx_performance_estimator.output_parsing import (
    NXPerformanceDatabaseParser,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXPerformanceEstimatorPerformanceMetrics,
)
from mlia.backend.nx_performance_estimator.statistics import (
    NXModelPerformanceStats,
    NXOperatorPerformanceStats,
)
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
from mlia.target.neural_technology.handlers import NeuralTechnologyEventHandler
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.target.neural_technology.advice_generation import (
    NeuralTechnologyAdviceProducer,
)
from mlia.target.neural_technology.advisor import (
    NeuralTechnologyInferenceAdvisor,
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.data_analysis import (
    NXPerformanceEstimatorModelPerformanceAnalyzed,
)
from mlia.target.neural_technology.plugin import (
    NeuralTechnologyTargetPlugin,
    create_neural_technology_api_event_handler,
)
from mlia.target.registry import TargetInfo, TargetRegistry


def _target_info_supports_event_handler_factory() -> bool:
    parameters = inspect.signature(TargetInfo.__init__).parameters
    return "event_handler_factory" in parameters


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

    assert ctx.event_handlers is not None
    assert ctx.config_parameters["neural_technology_inference_advisor"] == {
        "backend_options": {},
        "backends": ["nx-performance-estimator"],
        "model": str(test_tflite_model),
        "target_profile": "neural-technology",
    }

    assert isinstance(workflow, DefaultWorkflowExecutor)


def test_configure_and_get_neural_technology_advisor_preserves_existing_handlers(
    tmp_path: Path,
) -> None:
    """Preconfigured handlers should not be replaced by the advisor factory."""
    existing_handler = MagicMock()
    ctx = ExecutionContext(
        advice_category={AdviceCategory.PERFORMANCE},
        event_handlers=[existing_handler],
    )

    configure_and_get_neural_technology_advisor(
        ctx,
        "neural-technology",
        tmp_path / "model.tflite",
        backends=["nx-performance-estimator"],
    )

    assert ctx.event_handlers == [existing_handler]


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


@pytest.mark.skipif(
    not _target_info_supports_event_handler_factory(),
    reason="Installed mlia core does not expose event_handler_factory",
)
def test_target_registry_api_event_handler_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neural Technology target should register an API event handler factory."""
    monkeypatch.setitem(
        backend_registry.items,
        "nx-performance-estimator",
        MagicMock(),
    )
    registry = TargetRegistry()
    NeuralTechnologyTargetPlugin.register(registry)
    info = registry.items["neural-technology"]
    assert info.event_handler_factory is not None
    assert info.supports_torch_module is True
    assert info.torch_module_backend == "nx-performance-estimator"
    handler = info.event_handler_factory(None)
    assert isinstance(handler, NeuralTechnologyEventHandler)
    assert handler.collect_only is True


def test_target_registry_registers_without_event_handler_factory_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older mlia cores should register targets without API event handlers."""

    captured_kwargs: list[dict[str, object]] = []

    class _LegacyTargetInfo:
        def __init__(
            self,
            supported_backends: list[str],
            default_backends: list[str],
            advisor_factory_func: object,
            target_profile_cls: object,
            supports_torch_module: bool = False,
            torch_module_backend: str | None = None,
        ) -> None:
            self.supported_backends = supported_backends
            self.default_backends = default_backends
            self.advisor_factory_func = advisor_factory_func
            self.target_profile_cls = target_profile_cls
            self.supports_torch_module = supports_torch_module
            self.torch_module_backend = torch_module_backend
            captured_kwargs.append(
                {
                    "supported_backends": supported_backends,
                    "default_backends": default_backends,
                    "advisor_factory_func": advisor_factory_func,
                    "target_profile_cls": target_profile_cls,
                    "supports_torch_module": supports_torch_module,
                    "torch_module_backend": torch_module_backend,
                }
            )

    monkeypatch.setattr(
        mlia.target.neural_technology.plugin,
        "_target_info_supports_event_handler_factory",
        lambda: False,
    )
    monkeypatch.setattr(
        mlia.target.neural_technology.plugin,
        "_target_info_supports_torch_module_fields",
        lambda: False,
    )
    monkeypatch.setattr(
        mlia.target.neural_technology.plugin,
        "TargetInfo",
        _LegacyTargetInfo,
    )
    monkeypatch.setitem(
        backend_registry.items,
        "nx-performance-estimator",
        MagicMock(),
    )

    registry = TargetRegistry()
    NeuralTechnologyTargetPlugin.register(registry)

    assert tuple(registry.items) == ("neural-technology",)
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["supports_torch_module"] is False
    assert captured_kwargs[0]["torch_module_backend"] is None


def test_nx_performance_estimator_plugin_registers_cli_options() -> None:
    """NX backend plugin should expose API-discoverable backend options."""
    registry = BackendRegistry()
    NXPerformanceEstimatorPlugin.register(registry)

    assert registry.items["nx-performance-estimator"].cli_options == {
        "system_config": "--system-config",
        "compiler_config": "--compiler-config",
    }


def test_create_neural_technology_api_event_handler_requires_collect_only_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older mlia cores should fail with a clear upgrade message."""

    class _FakeSignature:
        parameters = {"self": object(), "formatter_resolver": object()}

    monkeypatch.setattr(
        mlia.target.neural_technology.plugin.inspect,
        "signature",
        lambda _obj: _FakeSignature(),
    )

    with pytest.raises(RuntimeError, match="Please upgrade mlia"):
        create_neural_technology_api_event_handler(None)


def test_neural_technology_advice_producer_produce_advice() -> None:
    """Test the produce_advice method on the NeuralTechnologyAdviceProducer."""
    ctx = ExecutionContext(advice_category={AdviceCategory.PERFORMANCE})
    producer = NeuralTechnologyAdviceProducer()
    producer.set_context(ctx)
    producer.produce_advice(
        NXPerformanceEstimatorModelPerformanceAnalyzed(
            NXPerformanceEstimatorPerformanceMetrics(
                backend_config=NXPerformanceEstimatorConfig(
                    "system.ini", "compiler.ini"
                ),
                performance_db_parser=NXPerformanceDatabaseParser(),
                stripe_performance_metrics={
                    "op0": MagicMock(spec=NXOperatorPerformanceStats)
                },
                chain_performance_metrics={
                    "chain0": MagicMock(spec=NXOperatorPerformanceStats)
                },
                model_performance_stats=MagicMock(spec=NXModelPerformanceStats),
            )
        )
    )
    assert len(producer.advice) == 1
    assert producer.advice[0].message == (
        "Please refer to the performance metrics shown in the report "
        "to find possible optimizations."
    )
