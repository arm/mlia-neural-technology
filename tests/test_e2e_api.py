# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Python API end-to-end tests for Neural Technology."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest
from mlia import (
    list_backend_options,
    list_backends,
    list_target_profiles,
    list_targets,
    supported_backends,
)
from mlia.testing import e2e as mlia_e2e


NEURAL_TECHNOLOGY = "neural-technology"
NX_PEAK = "NX-peak-12SC-8NX-600MHz"
NX_SUSTAINED = "NX-sustained-12SC-8NX-350MHz"
NX_PERFORMANCE_ESTIMATOR = "nx-performance-estimator"

MLIA_API_E2E_MODEL = "MLIA_API_E2E_MODEL"
MLIA_API_E2E_PYTORCH_MODEL = "MLIA_API_E2E_PYTORCH_MODEL"
MLIA_API_E2E_SUITE = "MLIA_API_E2E_SUITE"
MLIA_API_E2E_TFLITE_MODEL = "MLIA_API_E2E_TFLITE_MODEL"
MLIA_API_E2E_TOSA_MODEL = "MLIA_API_E2E_TOSA_MODEL"
MLIA_API_E2E_VGF_MODEL = "MLIA_API_E2E_VGF_MODEL"
PYTORCH = "pytorch"
TFLITE = "tflite"

pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class ParityCase:
    """One CLI JSON versus Python API parity case."""

    name: str
    advice_category: str
    model_env_var: str
    suite: str
    target_profile: str
    backends: tuple[str, ...] = ()


PARITY_CASES = (
    ParityCase(
        name="tosa_compatibility_nx_performance_estimator",
        advice_category="compatibility",
        model_env_var=MLIA_API_E2E_TOSA_MODEL,
        suite=TFLITE,
        target_profile=NEURAL_TECHNOLOGY,
        backends=(NX_PERFORMANCE_ESTIMATOR,),
    ),
    ParityCase(
        name="tosa_performance_nx_performance_estimator",
        advice_category="performance",
        model_env_var=MLIA_API_E2E_TOSA_MODEL,
        suite=TFLITE,
        target_profile=NEURAL_TECHNOLOGY,
        backends=(NX_PERFORMANCE_ESTIMATOR,),
    ),
    # VGF compatibility is not included because the compatibility checker
    # currently raises NotImplementedError for VGF models.
    ParityCase(
        name="vgf_performance_nx_performance_estimator",
        advice_category="performance",
        model_env_var=MLIA_API_E2E_VGF_MODEL,
        suite=TFLITE,
        target_profile=NEURAL_TECHNOLOGY,
        backends=(NX_PERFORMANCE_ESTIMATOR,),
    ),
    ParityCase(
        name="tflite_compatibility_nx_performance_estimator",
        advice_category="compatibility",
        model_env_var=MLIA_API_E2E_TFLITE_MODEL,
        suite=TFLITE,
        target_profile=NEURAL_TECHNOLOGY,
        backends=(NX_PERFORMANCE_ESTIMATOR,),
    ),
    ParityCase(
        name="tflite_performance_nx_performance_estimator",
        advice_category="performance",
        model_env_var=MLIA_API_E2E_TFLITE_MODEL,
        suite=TFLITE,
        target_profile=NEURAL_TECHNOLOGY,
        backends=(NX_PERFORMANCE_ESTIMATOR,),
    ),
    ParityCase(
        name="pytorch_compatibility_nx_performance_estimator",
        advice_category="compatibility",
        model_env_var=MLIA_API_E2E_PYTORCH_MODEL,
        suite=PYTORCH,
        target_profile=NEURAL_TECHNOLOGY,
        backends=(NX_PERFORMANCE_ESTIMATOR,),
    ),
    ParityCase(
        name="pytorch_performance_nx_performance_estimator",
        advice_category="performance",
        model_env_var=MLIA_API_E2E_PYTORCH_MODEL,
        suite=PYTORCH,
        target_profile=NEURAL_TECHNOLOGY,
        backends=(NX_PERFORMANCE_ESTIMATOR,),
    ),
)


def _case_matches_active_suite(case: ParityCase) -> bool:
    """Return whether this parity case belongs to the active CI e2e suite."""
    active_suite = os.environ.get(MLIA_API_E2E_SUITE, "").strip()
    return not active_suite or active_suite == case.suite


def _active_suite_configured() -> bool:
    """Return whether the test is running for a configured CI suite."""
    return bool(os.environ.get(MLIA_API_E2E_SUITE, "").strip())


def _model_configuration_message(case: ParityCase) -> str:
    """Return suite-specific configuration guidance for missing artifacts."""
    if case.model_env_var == MLIA_API_E2E_TFLITE_MODEL:
        return f"Set {case.model_env_var} or {MLIA_API_E2E_MODEL} to run {case.name}."
    return f"Set {case.model_env_var} to run {case.name}."


def _fail_with_report(message: str) -> NoReturn:
    """Fail while also writing the message to captured e2e report output."""
    print(message)
    pytest.fail(message)


def _assert_reported(condition: bool, message: str) -> None:
    """Assert a condition with a message visible in the e2e PDF report."""
    if not condition:
        _fail_with_report(message)


def _configured_model_path(case: ParityCase) -> Path | None:
    """Resolve the configured model path from API e2e environment variables."""
    model_value = os.environ.get(case.model_env_var)
    if not model_value and case.model_env_var == MLIA_API_E2E_TFLITE_MODEL:
        model_value = os.environ.get(MLIA_API_E2E_MODEL)
    if not model_value:
        return None

    model = Path(model_value)
    if model.is_absolute():
        return model

    artifacts_root = os.environ.get(mlia_e2e.MLIA_E2E_ARTIFACTS)
    if artifacts_root:
        return Path(artifacts_root) / model
    return model


def _representative_model(case: ParityCase) -> Path:
    """Resolve the representative model used for API parity."""
    model = _configured_model_path(case)
    if model is None:
        if _active_suite_configured():
            _fail_with_report(_model_configuration_message(case))
        pytest.skip(_model_configuration_message(case))

    if not model.is_file():
        if not _active_suite_configured():
            pytest.skip(f"Representative API e2e model does not exist: {model}")
        _fail_with_report(f"Representative API e2e model does not exist: {model}")
    return model


def _format_backends(backends: tuple[str, ...]) -> str:
    """Return a compact backend list for e2e report output."""
    return ", ".join(backends) if backends else "<default>"


def _print_parity_case_summary(case: ParityCase, model: Path) -> None:
    """Print concise case details for the JUnit/PDF e2e report."""
    print(f"case: {case.name}")
    print(f"suite: {case.suite}")
    print(f"model: {model}")
    print(f"advice: {case.advice_category}")
    print(f"target_profile: {case.target_profile}")
    print(f"backends: {_format_backends(case.backends)}")


def _assert_json_matches_cli(
    case: ParityCase,
    model: Path,
    api_output: dict[str, Any],
    cli_output: dict[str, Any],
) -> None:
    """Assert API/CLI parity with enough context for the e2e report."""
    message = (
        f"Python API JSON did not match CLI JSON for {case.name}.\n"
        f"model: {model}\n"
        f"advice: {case.advice_category}\n"
        f"target_profile: {case.target_profile}\n"
        f"backends: {_format_backends(case.backends)}"
    )
    if api_output != cli_output:
        print(message)
    assert api_output == cli_output, message


def _cli_json_output(case: ParityCase, model: Path) -> dict[str, Any]:
    """Run the equivalent CLI command and return its standardized JSON output."""
    argv = [
        "mlia",
        "check",
        str(model),
        f"--{case.advice_category}",
        "--target-profile",
        case.target_profile,
        "--json",
    ]
    for backend in case.backends:
        argv.extend(["--backend", backend])

    result = subprocess.run(
        argv,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        _fail_with_report(
            f"CLI e2e command failed: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(_extract_json(result.stdout))


def _extract_json(output: str) -> str:
    """Extract the JSON object from command output that may include log lines."""
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        _fail_with_report(f"No JSON object found in output:\n{output}")
    return output[start : end + 1]


def _api_output(case: ParityCase, model: Path) -> dict[str, Any]:
    """Run the Python API in a subprocess and return its serialized JSON output."""
    payload = {
        "advice_category": case.advice_category,
        "target_profile": case.target_profile,
        "model": str(model),
        "backends": list(case.backends),
    }
    code = """
import json
import sys
from pathlib import Path

from mlia import ValidationMode, run_advisor

payload = json.loads(sys.argv[1])
kwargs = {}
if payload["backends"]:
    kwargs["backends"] = payload["backends"]
result = run_advisor(
    advice_category=payload["advice_category"],
    target_profile=payload["target_profile"],
    model=Path(payload["model"]),
    validation=ValidationMode.OFF,
    **kwargs,
)
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(payload)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        _fail_with_report(
            f"Python API e2e command failed for {case.name}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(_extract_json(result.stdout))


def _normalize_output(output: dict[str, Any]) -> dict[str, Any]:
    """Remove expected volatile CLI/API differences before parity comparison."""
    normalized = deepcopy(output)
    normalized.pop("run_id", None)
    normalized.pop("timestamp", None)

    context = normalized.get("context")
    if isinstance(context, dict):
        context.pop("cli_arguments", None)

    return normalized


def test_python_api_e2e_queries() -> None:
    """Check Neural Technology plugin surfaces through the public query APIs."""
    mlia_e2e.ensure_backends_available((NX_PERFORMANCE_ESTIMATOR,))

    targets = list_targets()
    target_names = {item["target"] for item in targets}
    _assert_reported(
        NEURAL_TECHNOLOGY in target_names,
        f"Expected target {NEURAL_TECHNOLOGY}; available targets: {sorted(target_names)}",
    )

    target = next(item for item in targets if item["target"] == NEURAL_TECHNOLOGY)
    target_profiles = set(target["profiles"])
    expected_profiles = {NEURAL_TECHNOLOGY, NX_PEAK, NX_SUSTAINED}
    _assert_reported(
        expected_profiles.issubset(target_profiles),
        f"Expected profiles {sorted(expected_profiles)}; got {sorted(target_profiles)}",
    )
    _assert_reported(
        NX_PERFORMANCE_ESTIMATOR in target["supported_backends"],
        f"Expected backend {NX_PERFORMANCE_ESTIMATOR}; "
        f"got {target['supported_backends']}",
    )
    _assert_reported(
        {"compatibility", "performance"}.issubset(set(target["supported_advice"])),
        f"Expected compatibility and performance advice; "
        f"got {target['supported_advice']}",
    )

    profiles = list_target_profiles()
    _assert_reported(
        NEURAL_TECHNOLOGY in profiles,
        f"Expected target profile group {NEURAL_TECHNOLOGY}; got {sorted(profiles)}",
    )
    profile_names = {item["name"] for item in profiles[NEURAL_TECHNOLOGY]}
    _assert_reported(
        expected_profiles.issubset(profile_names),
        f"Expected profile names {sorted(expected_profiles)}; "
        f"got {sorted(profile_names)}",
    )

    backends = {item["name"]: item for item in list_backends()}
    _assert_reported(
        NX_PERFORMANCE_ESTIMATOR in backends,
        f"Expected backend {NX_PERFORMANCE_ESTIMATOR}; "
        f"available backends: {sorted(backends)}",
    )
    _assert_reported(
        backends[NX_PERFORMANCE_ESTIMATOR]["installed"] is True,
        f"Expected {NX_PERFORMANCE_ESTIMATOR} to be installed; "
        f"got {backends[NX_PERFORMANCE_ESTIMATOR]}",
    )
    _assert_reported(
        backends[NX_PERFORMANCE_ESTIMATOR]["could_be_installed"] is True,
        f"Expected {NX_PERFORMANCE_ESTIMATOR} to be installable; "
        f"got {backends[NX_PERFORMANCE_ESTIMATOR]}",
    )

    backend_options = {
        item["backend"]: item["options"] for item in list_backend_options()
    }
    option_keys = {
        option["config_key"] for option in backend_options[NX_PERFORMANCE_ESTIMATOR]
    }
    _assert_reported(
        {"system_config", "compiler_config"}.issubset(option_keys),
        f"Expected backend options system_config and compiler_config; "
        f"got {sorted(option_keys)}",
    )

    supported = supported_backends(NEURAL_TECHNOLOGY)
    _assert_reported(
        NX_PERFORMANCE_ESTIMATOR in supported,
        f"Expected supported backend {NX_PERFORMANCE_ESTIMATOR}; got {supported}",
    )

    print("Neural Technology API query summary")
    print(f"target: {NEURAL_TECHNOLOGY}")
    print(f"profiles: {NEURAL_TECHNOLOGY}, {NX_PEAK}, {NX_SUSTAINED}")
    print(f"backend: {NX_PERFORMANCE_ESTIMATOR}")
    print("backend_options: system_config, compiler_config")
    print("supported_advice: compatibility, performance")
    print("result: query APIs returned the expected Neural Technology data")


@pytest.mark.parametrize("case", PARITY_CASES, ids=[case.name for case in PARITY_CASES])
def test_python_api_e2e_matches_cli_json(case: ParityCase) -> None:
    """Check normalized Python API output matches equivalent CLI JSON output."""
    if not _case_matches_active_suite(case):
        pytest.skip(f"Case belongs to the {case.suite} e2e suite.")

    model = _representative_model(case)
    mlia_e2e.ensure_backends_available(case.backends)

    cli_output = _normalize_output(_cli_json_output(case, model))
    api_output = _normalize_output(_api_output(case, model))

    _print_parity_case_summary(case, model)
    _assert_json_matches_cli(case, model, api_output, cli_output)
    print("result: Python API JSON matched CLI JSON")
