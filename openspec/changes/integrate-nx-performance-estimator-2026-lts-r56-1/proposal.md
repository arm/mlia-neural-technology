<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## Why

The bundled NX graph compiler performance estimator is moving to the 2026 LTS
r56.1 version, and MLIA must continue to produce usable performance output for
supported `.vgf` and `.tosa` models. The integration needs to account for any
runtime, output file, JSON, or database-format differences between the current
vendored estimator and the new one before the artifact source is switched.

## What Changes

- Compare current and new NX performance estimator outputs for supported
  `.vgf` and `.tosa` models from `../../models`.
- Use the provided candidate estimator at
  `../misc/Graph_compiler_performance_estimator_test_mlia_2026_lts_r56_1` as the
  local stand-in for the incoming artifact.
- Update MLIA's NX estimator integration to tolerate or normalize observed
  output differences while preserving standardized MLIA performance output.
- Update the vendored artifact metadata and build/download assumptions so the
  new estimator can be supplied through the existing artifact flow.
- Add regression coverage for parser, statistics, runtime invocation, and
  model-output differences discovered during comparison.

## Capabilities

### New Capabilities
- `nx-performance-estimator-version-integration`: Defines how MLIA integrates a
  new NX performance estimator artifact version while preserving supported model
  execution and standardized performance output.

### Modified Capabilities

None.

## Impact

- `hatch_build.py` and
  `src/mlia/_vendor/artifacts/nx-performance-estimator/`: artifact name, hash,
  and packaging/download expectations.
- `src/mlia/backend/nx_performance_estimator/`: estimator invocation, output
  parsing, model-level statistics, chain/stripe statistics, and standardized
  output conversion.
- `src/mlia/resources/nx-performance-estimator/`: system/compiler config
  compatibility if the new estimator changes accepted options or defaults.
- `tests/test_backend_nx_performance_estimator_*.py`: characterization and
  regression tests for old-versus-new output differences.
- External local inputs used for the integration spike:
  `../misc/Graph_compiler_performance_estimator_test_mlia_2026_lts_r56_1` and
  supported models under `../../models`.
