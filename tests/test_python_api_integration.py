# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused Python API integration tests for Neural Technology."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mlia.backend.registry import registry as backend_registry
from mlia.core.output_postprocessing import postprocess_standardized_output
from mlia.core.settings import ApplicationSettings, CollapseRule, FilteringSettings
from mlia.target.registry import registry as target_registry
from mlia.transformers.registry import TransformRequest, transformer_registry

from mlia.backend.ml_sdk_model_converter.plugin import MLSDKModelConverterPlugin
from mlia.backend.neural_technology_profiling_data.plugin import (
    NeuralTechnologyProfilingDataPlugin,
)
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin
from mlia.backend.tosa_flatbuffers.plugin import TosaFlatBuffersPlugin
from mlia.target.neural_technology.plugin import NeuralTechnologyTargetPlugin

mlia_api = importlib.import_module("mlia.api")
run_advisor = mlia_api.run_advisor
RUN_ADVISOR_SUPPORTS_ACCEPT_EULA = (
    "accept_eula" in inspect.signature(run_advisor).parameters
)

backend_registry_module = importlib.import_module("mlia.backend.registry")
target_registry_module = importlib.import_module("mlia.target.registry")

try:
    from mlia.backend.mlia_nn_module_to_pt2_exporter.exporter_plugin import (
        NNModuleToPt2ExporterPlugin,
    )
except ModuleNotFoundError:
    NNModuleToPt2ExporterPlugin = None


def _register_neural_technology_api_plugins(
    monkeypatch: pytest.MonkeyPatch,
    auto_install: Callable[..., None] | None = None,
) -> None:
    monkeypatch.setattr(backend_registry, "items", dict(backend_registry.items))
    monkeypatch.setattr(target_registry, "items", dict(target_registry.items))
    monkeypatch.setattr(transformer_registry, "items", dict(transformer_registry.items))
    monkeypatch.setattr(backend_registry_module, "_plugins_loaded", True)
    monkeypatch.setattr(target_registry_module, "_plugins_loaded", True)
    target_registry_module.profile.cache_clear()
    target_registry_module.create_target_profile.cache_clear()

    NXPerformanceEstimatorPlugin.register(backend_registry)
    NeuralTechnologyProfilingDataPlugin.register(backend_registry)
    MLSDKModelConverterPlugin.register(backend_registry)
    TosaFlatBuffersPlugin.register(backend_registry)
    NeuralTechnologyTargetPlugin.register(target_registry)
    if NNModuleToPt2ExporterPlugin is not None:
        NNModuleToPt2ExporterPlugin.register(transformer_registry)

    if auto_install is not None:
        monkeypatch.setattr(
            mlia_api,
            "ensure_backends_installed",
            auto_install,
            raising=False,
        )


def _write_profile(tmp_path: Path, profile_name: str) -> Path:
    profile = tmp_path / "neural_technology.toml"
    profile.write_text(
        "\n".join(
            [
                'target = "neural-technology"',
                f'profile_name = "{profile_name}"',
                "",
                "[backend.nx-performance-estimator]",
                'system_config = "profile-system.ini"',
                'compiler_config = "profile-compiler.ini"',
            ]
        ),
        encoding="utf-8",
    )
    return profile


def _fake_standardized_output(
    model_name: str, model_format: str, result_kind: str = "compatibility"
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "timestamp": "2026-07-24T12:00:00Z",
        "tool": {"name": "MLIA", "version": "1.0.0"},
        "target": {
            "profile_name": "test",
            "target_type": "neural-technology",
            "components": ["neural-technology"],
            "configuration": {},
        },
        "model": {
            "name": model_name,
            "format": model_format,
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "context": {"cli_arguments": ["mlia", "--debug"]},
        "backends": [{"name": "test", "version": "1.0.0"}],
        "results": [{"kind": result_kind, "status": "ok", "producer": "test"}],
    }


def test_run_advisor_compatibility_routes_backend_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_advisor should route backend option overrides into Neural Technology."""
    _register_neural_technology_api_plugins(
        monkeypatch, auto_install=lambda *_, **__: None
    )

    profile = _write_profile(tmp_path, "NX-demo")
    model = tmp_path / "model.vgf"
    model.write_text("vgf", encoding="utf-8")

    captured: dict[str, Any] = {}

    class FakeCompatibilityInfo:
        def to_standardized_output(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return _fake_standardized_output("model.vgf", "vgf")

    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        "NXCompatibilityChecker.check_compatibility",
        MagicMock(return_value=FakeCompatibilityInfo()),
    )

    output = run_advisor(
        "compatibility",
        str(profile),
        model,
        backends=["nx-performance-estimator"],
        backend_options={
            "nx-performance-estimator": {
                "system_config": "override-system.ini",
                "compiler_config": "override-compiler.ini",
            }
        },
        validation="off",
    )

    assert output["schema_version"] == "1.0.0"
    assert len(output["results"]) == 1
    assert output["results"][0].get("advice", []) == []
    assert captured["target_config"] == {
        "target": "neural-technology",
        "target_type": "neural-technology",
        "profile_name": "NX-demo",
    }
    assert captured["backend_config"] == {
        "nx-performance-estimator": {
            "system_config": "override-system.ini",
            "compiler_config": "override-compiler.ini",
        }
    }
    assert output["context"] == {}


def test_run_advisor_performance_routes_backend_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_advisor should pass backend option overrides into performance setup."""
    profile = _write_profile(tmp_path, "NX-peak")
    model = tmp_path / "model.tosa"
    model.write_text("tosa", encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_auto_install(
        backend_names: list[str], *, accept_eula: bool | None = None, **_kwargs: Any
    ) -> None:
        captured["auto_install"] = {
            "backend_names": backend_names,
            "accept_eula": accept_eula,
        }

    _register_neural_technology_api_plugins(monkeypatch, auto_install=fake_auto_install)

    class FakeMetrics:
        def to_standardized_output(self, **kwargs: Any) -> dict[str, Any]:
            captured["standardized_kwargs"] = kwargs
            return _fake_standardized_output("model.tosa", "tosa", "performance")

    def fake_init(
        self,
        output_dir: Path,
        backend_config: dict[str, Any],
        operator_types: dict[str, str],
    ) -> None:
        captured["output_dir"] = output_dir
        captured["backend_config"] = backend_config
        captured["operator_types"] = operator_types

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorPerformanceEstimator.__init__",
        fake_init,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorPerformanceEstimator.estimate",
        MagicMock(return_value=FakeMetrics()),
    )

    kwargs: dict[str, Any] = {}
    if RUN_ADVISOR_SUPPORTS_ACCEPT_EULA:
        kwargs["accept_eula"] = True

    output = run_advisor(
        "performance",
        str(profile),
        model,
        backends=["nx-performance-estimator"],
        backend_options={
            "nx-performance-estimator": {
                "system_config": "override-system.ini",
                "compiler_config": "override-compiler.ini",
            }
        },
        validation="off",
        **kwargs,
    )

    assert output["schema_version"] == "1.0.0"
    assert len(output["results"]) == 1
    assert output["results"][0]["advice"] == [
        {
            "id": "performance_metrics",
            "category": "performance",
            "severity": "info",
            "message": (
                "Please refer to the performance metrics shown in the report "
                "to find possible optimizations."
            ),
        }
    ]
    assert captured["backend_config"] == {
        "nx-performance-estimator": {
            "system_config": "override-system.ini",
            "compiler_config": "override-compiler.ini",
        }
    }
    if RUN_ADVISOR_SUPPORTS_ACCEPT_EULA:
        assert captured["auto_install"] == {
            "backend_names": ["nx-performance-estimator"],
            "accept_eula": True,
        }
    assert captured["standardized_kwargs"]["target_config"] == {
        "target": "neural-technology",
        "target_type": "neural-technology",
        "profile_name": "NX-peak",
    }
    assert output["context"] == {}


def test_run_advisor_routes_profiling_data_through_target_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public API should use the target advisor for measured data."""
    _register_neural_technology_api_plugins(
        monkeypatch, auto_install=lambda *_, **__: None
    )
    profile = _write_profile(tmp_path, "NX-measured")
    profiling_data = tmp_path / "profiling-data"
    profiling_data.mkdir()
    captured: dict[str, object] = {}

    def fake_analyze(**kwargs: object) -> dict[str, Any]:
        captured.update(kwargs)
        return _fake_standardized_output(
            "profiling-data", "profiling-data", "performance"
        )

    monkeypatch.setattr(
        "mlia.target.neural_technology.data_collection.analyze_profiling_data",
        fake_analyze,
    )

    output = run_advisor(
        "performance",
        str(profile),
        profiling_data=profiling_data,
        validation="off",
    )

    assert output["results"][0]["kind"] == "performance"
    cli_arguments = captured.pop("cli_arguments")
    assert isinstance(cli_arguments, list)
    assert cli_arguments
    output_dir = captured.pop("output_dir")
    assert isinstance(output_dir, Path)
    assert output_dir.name == "mlia-output"
    assert captured == {
        "target_profile": str(profile),
        "profiling_data": [profiling_data],
        "categories": {"performance"},
        "model": None,
    }


@pytest.mark.parametrize("collapse_enabled", [False, True])
def test_run_advisor_applies_centralized_collapse_and_projection_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collapse_enabled: bool,
) -> None:
    """The public NX API should apply core graph processing exactly once."""
    _register_neural_technology_api_plugins(
        monkeypatch, auto_install=lambda *_, **__: None
    )
    profile = _write_profile(tmp_path, "NX-centralized")
    model = tmp_path / "model.tosa"
    model.write_text("tosa", encoding="utf-8")
    collapse_rules = (
        (
            CollapseRule(
                kind="code_stack",
                attribute="file",
                globs=("vendor/*",),
            ),
        )
        if collapse_enabled
        else ()
    )
    settings = ApplicationSettings(filtering=FilteringSettings(collapse=collapse_rules))
    monkeypatch.setattr(
        mlia_api, "ApplicationSettings", MagicMock(return_value=settings)
    )

    captured: dict[str, Any] = {}

    class FakeMetrics:
        def to_standardized_output(self, **_kwargs: Any) -> dict[str, Any]:
            output = _fake_standardized_output("model.tosa", "tosa", "performance")
            output["results"] = [
                {
                    "kind": "performance",
                    "status": "ok",
                    "producer": "nx-performance-estimator",
                    "entity_kinds": [
                        {"id": "chain", "child_kinds": ["source_operator"]}
                    ],
                    "entities": [
                        {
                            "id": "target-stack",
                            "kind": "code_stack",
                            "name": "target.py:1",
                            "child_ids": ["A"],
                        },
                        {
                            "id": "generated-stack",
                            "kind": "code_stack",
                            "name": "generated.py:1",
                            "child_ids": ["G"],
                            "attributes": {"file": "vendor/generated.py"},
                        },
                        {
                            "id": "measured-chain",
                            "kind": "chain",
                            "name": "measured",
                            "child_ids": ["A", "G"],
                        },
                        {"id": "A", "kind": "source_operator", "name": "A"},
                        {"id": "G", "kind": "source_operator", "name": "G"},
                    ],
                    "breakdowns": [
                        {
                            "entity_id": "measured-chain",
                            "metrics": [
                                {"name": "cycles", "value": 10, "unit": "cycles"}
                            ],
                        }
                    ],
                }
            ]
            captured["backend_output"] = output
            return output

    def fake_init(
        self,
        output_dir: Path,
        backend_config: dict[str, Any],
        operator_types: dict[str, str],
    ) -> None:
        del self, output_dir, backend_config, operator_types

    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorPerformanceEstimator.__init__",
        fake_init,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.performance."
        "NXPerformanceEstimatorPerformanceEstimator.estimate",
        MagicMock(return_value=FakeMetrics()),
    )
    process = MagicMock(side_effect=postprocess_standardized_output)
    monkeypatch.setattr("mlia.core.advisor.postprocess_standardized_output", process)

    output = run_advisor(
        "performance",
        str(profile),
        model,
        backends=["nx-performance-estimator"],
        validation="off",
    )

    process.assert_called_once()
    settings = process.call_args.args[1]
    assert bool(settings.filtering.collapse) is collapse_enabled
    backend_result = captured["backend_output"]["results"][0]
    assert [item["entity_id"] for item in backend_result["breakdowns"]] == [
        "measured-chain"
    ]

    result = output["results"][0]
    entity_ids = {entity["id"] for entity in result["entities"]}
    breakdown_ids = [item["entity_id"] for item in result["breakdowns"]]
    assert ("generated-stack" not in entity_ids) is collapse_enabled
    assert breakdown_ids.count("measured-chain") == 1
    assert breakdown_ids.count("target-stack") == int(collapse_enabled)


def test_run_advisor_compatibility_accepts_torch_module_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_advisor should accept torch.nn.Module for Neural Technology."""
    if NNModuleToPt2ExporterPlugin is None:
        pytest.skip("mlia-converters-pytorch transformer plugin is not installed")

    _register_neural_technology_api_plugins(
        monkeypatch, auto_install=lambda *_, **__: None
    )

    profile = _write_profile(tmp_path, "NX-module")
    exported_paths: dict[str, Path] = {}
    captured: dict[str, Any] = {}

    class FakeModule:
        pass

    def fake_export(module: object, args: tuple[object, ...]) -> object:
        captured["exported_module"] = module
        captured["example_inputs"] = args
        return object()

    def fake_save(_exported: object, output_path: Path) -> None:
        output_path.write_text("pt2", encoding="utf-8")
        exported_paths["pt2"] = output_path

    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=FakeModule),
        export=types.SimpleNamespace(export=fake_export, save=fake_save),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def fake_transform_model(request: TransformRequest) -> Path:
        captured["export_request"] = request
        captured["example_inputs"] = request.transform_options["example_inputs"]
        fake_save(object(), request.output_dir / "model.pt2")
        return request.output_dir / "model.pt2"

    monkeypatch.setattr("mlia.api.transform_model", fake_transform_model)

    def fake_converter(
        model_path: Path,
        output_dir: Path,
        *,
        enable_quantization: bool | None = None,
    ) -> Path:
        captured["converter_input"] = model_path
        captured["converter_output_dir"] = output_dir
        captured["converter_enable_quantization"] = enable_quantization
        tosa_path = output_dir / "converted.tosa"
        tosa_path.write_text("tosa", encoding="utf-8")
        return tosa_path

    class FakeCompatibilityInfo:
        def to_standardized_output(self, **kwargs: Any) -> dict[str, Any]:
            captured["standardized_kwargs"] = kwargs
            return _fake_standardized_output("DemoModule", "pt2")

    def fake_transform_front_end_model(
        model_path: Path,
        output_dir: Path,
        *,
        enable_quantization: bool | None = None,
        **_kwargs: Any,
    ) -> Path:
        return fake_converter(
            model_path,
            output_dir,
            enable_quantization=enable_quantization,
        )

    monkeypatch.setattr(
        "mlia.target.neural_technology.data_collection.transform_front_end_model",
        fake_transform_front_end_model,
    )
    monkeypatch.setattr(
        "mlia.backend.ml_sdk_model_converter.compat."
        "NXCompatibilityChecker.check_compatibility",
        MagicMock(return_value=FakeCompatibilityInfo()),
    )

    module = FakeModule()
    output = run_advisor(
        "compatibility",
        str(profile),
        module,
        example_inputs=(object(),),
        backends=["nx-performance-estimator"],
        validation="off",
    )

    assert output["schema_version"] == "1.0.0"
    assert output["model"]["name"] == "FakeModule"
    assert exported_paths["pt2"].name == "model.pt2"
    assert captured["export_request"] == TransformRequest(
        model=module,
        output_dir=exported_paths["pt2"].parent,
        target_format="pt2",
        transform_options={
            "example_inputs": captured["example_inputs"],
            "enable_quantization": False,
        },
    )
    assert captured["converter_input"] == exported_paths["pt2"]
    assert captured["converter_enable_quantization"] is False
    assert captured["example_inputs"] != ()
    assert captured["standardized_kwargs"]["backend_config"] == {
        "nx-performance-estimator": {
            "system_config": "profile-system.ini",
            "compiler_config": "profile-compiler.ini",
            "enable_quantization": False,
        }
    }
