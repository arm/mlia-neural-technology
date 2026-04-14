# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Focused Python API integration tests for Neural Technology."""

from __future__ import annotations
from mlia.target.registry import registry as target_registry
from mlia.target.neural_technology.plugin import NeuralTechnologyTargetPlugin
from mlia.backend.registry import registry as backend_registry
from mlia.backend.nx_performance_estimator.plugin import NXPerformanceEstimatorPlugin

import importlib
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

mlia_api = importlib.import_module("mlia.api")
run_advisor = mlia_api.run_advisor

backend_registry_module = importlib.import_module("mlia.backend.registry")
target_registry_module = importlib.import_module("mlia.target.registry")


def _register_neural_technology_api_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_registry, "items", dict(backend_registry.items))
    monkeypatch.setattr(target_registry, "items", dict(target_registry.items))
    monkeypatch.setattr(backend_registry_module, "_plugins_loaded", True)
    monkeypatch.setattr(target_registry_module, "_plugins_loaded", True)
    target_registry_module.profile.cache_clear()
    target_registry_module.create_target_profile.cache_clear()

    NXPerformanceEstimatorPlugin.register(backend_registry)
    NeuralTechnologyTargetPlugin.register(target_registry)


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


def test_run_advisor_compatibility_routes_backend_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_advisor should route backend option overrides into Neural Technology."""
    _register_neural_technology_api_plugins(monkeypatch)

    profile = _write_profile(tmp_path, "NX-demo")
    model = tmp_path / "model.vgf"
    model.write_text("vgf", encoding="utf-8")

    captured: dict[str, Any] = {}

    class FakeCompatibilityInfo:
        def to_standardized_output(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "schema_version": "1.0.0",
                "model": {"name": "model.vgf", "format": "vgf"},
                "context": {"cli_arguments": ["mlia", "--debug"]},
                "results": [{}],
            }

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
    assert output["results"][0]["advices"] == []
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
    _register_neural_technology_api_plugins(monkeypatch)

    profile = _write_profile(tmp_path, "NX-peak")
    model = tmp_path / "model.tosa"
    model.write_text("tosa", encoding="utf-8")

    captured: dict[str, Any] = {}

    class FakeMetrics:
        def to_standardized_output(self, **kwargs: Any) -> dict[str, Any]:
            captured["standardized_kwargs"] = kwargs
            return {
                "schema_version": "1.0.0",
                "model": {"name": "model.tosa", "format": "tosa"},
                "context": {"cli_arguments": ["mlia", "--debug"]},
                "results": [{}],
            }

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
    )

    assert output["schema_version"] == "1.0.0"
    assert len(output["results"]) == 1
    assert output["results"][0]["advices"] == []
    assert captured["backend_config"] == {
        "nx-performance-estimator": {
            "system_config": "override-system.ini",
            "compiler_config": "override-compiler.ini",
        }
    }
    assert captured["standardized_kwargs"]["target_config"] == {
        "target": "neural-technology",
        "target_type": "neural-technology",
        "profile_name": "NX-peak",
    }
    assert output["context"] == {}


def test_run_advisor_compatibility_accepts_torch_module_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_advisor should accept torch.nn.Module for Neural Technology."""
    _register_neural_technology_api_plugins(monkeypatch)

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
            return {
                "schema_version": "1.0.0",
                "model": {"name": "DemoModule", "format": "pt2"},
                "context": {"cli_arguments": ["mlia", "--debug"]},
                "results": [{}],
            }

    monkeypatch.setattr(
        "mlia.target.neural_technology.data_collection.run_named_converter",
        lambda name, model_path, output_dir, enable_quantization=None: fake_converter(
            model_path,
            output_dir,
            enable_quantization=enable_quantization,
        ),
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
