<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## 1. Comparison Setup

- [x] 1.1 Confirm the supported `.vgf` and `.tosa` model set and record the exact model paths used for comparison.
- [x] 1.2 Create a temporary runner or documented command sequence that can execute the current vendored estimator and the candidate `../misc/Graph_compiler_performance_estimator_test_mlia_2026_lts_r56_1` estimator without permanently replacing repository artifacts.
- [x] 1.3 Capture current-estimator raw outputs, MLIA `nx_performance_statistics.json`, and standardized MLIA performance output for each successful supported model, and record the current-estimator failure for `yolo_v3_tiny_darknet_fp32.vgf`.
- [x] 1.4 Capture candidate-estimator raw outputs, MLIA `nx_performance_statistics.json`, and standardized MLIA performance output for each supported model.
- [x] 1.5 Summarize observed differences by category: invocation/options, output filenames, debug database shape, performance database shape, network summary JSON shape, standardized output structure, and expected numeric drift.

## 2. Characterization Tests

- [x] 2.1 Add compact parser fixtures or generated test data for every changed candidate output shape discovered in the comparison.
- [x] 2.2 Confirm existing parser tests and comparison parsing cover r56.1 debug and performance database shapes; no parser code or parser-test changes were required.
- [x] 2.3 Add or update statistics tests in `tests/test_backend_nx_performance_estimator_statistics.py` for changed model summary JSON behavior.
- [x] 2.4 Add or update performance tests in `tests/test_backend_nx_performance_estimator_performance.py` for changed output filenames, command invocation, or model summary JSON fields.
- [x] 2.5 Keep malformed-output tests strict so missing required files or semantic fields still produce clear failures.

## 3. Runtime And Parser Implementation

- [x] 3.1 Update NX estimator invocation only if the candidate requires changed CLI options, working directory behavior, or output-name handling.
- [x] 3.2 Update `NXPerformanceEstimatorOutputFiles` only if the candidate changes required output filenames or locations.
- [x] 3.3 Confirm `NXPerformanceDatabaseParser` and `NXDebugDatabaseParser` already support observed r56.1 table/header shapes while preserving validation for unsupported malformed output.
- [x] 3.4 Update `NXModelPerformanceStats.read_from_json()` to support observed r56.1 network summary JSON changes while preserving required standardized metrics.
- [x] 3.5 Confirm `NXPerformanceStats` memory, utilization, and operator mapping logic already supports observed r56.1 semantic changes.
- [x] 3.6 Confirm `to_standardized_output()` still produces required model metrics and operator breakdowns for both `.vgf` and `.tosa` runs.
- [x] 3.7 Map r56.1 `TOSA*_spirv_id_*` debug database labels back to VGF MLGraph debug locations before producing performance breakdowns.

## 4. Artifact Integration

- [x] 4.1 Confirm the local candidate artifact name, format, and SHA-256 for the 2026 LTS r56.1 estimator; Artifactory URL wiring remains external to this repository change.
- [x] 4.2 Update `src/mlia/_vendor/artifacts/nx-performance-estimator/.sha256` and sidecar licensing metadata if required by the new artifact.
- [x] 4.3 Confirm `hatch_build.py` still resolves the estimator through the existing `graph-compiler-performance-estimator` artifact key and `VENDORED_ARTIFACTS_URLS` mapping.
- [x] 4.4 Verify backend installation still checks for `graph-compiler-performance-estimator` after artifact extraction.

## 5. Validation

- [x] 5.1 Run targeted NX estimator unit tests with `uv run pytest tests/test_backend_nx_performance_estimator_*.py`.
- [x] 5.2 Re-run the old-versus-new comparison after implementation and confirm candidate runs produce valid standardized MLIA output for each supported model.
- [x] 5.3 Run broader relevant regression tests if parser/runtime changes affect shared backend behavior; no broader shared-backend changes were required beyond the targeted NX estimator suite.
- [x] 5.4 Update the OpenSpec design or tasks if comparison results reveal additional required behavior not covered by the current spec.
