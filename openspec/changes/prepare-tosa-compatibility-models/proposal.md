<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## Why

TOSA compatibility now relies on NX Performance Estimator placement data. Empty
or repeated source locations prevent reliable attribution of compiler records,
and converted multi-region models can contain more graph segments than one GCPE
invocation accepts. Preparation must give each operation a unique compiler label
without modifying the input model or exposing synthetic counters in the report.

This change documents the TOSA implementation and preserves accounting for
estimator operations that lack usable labels.

## What Changes

- Move compatibility handling to `mlia.backend.nx_performance_estimator.compat`.
- Prepare both `.tosa` and `.tosamlir` copies with unique operation labels:
  empty locations become `unknown_<n>`; existing locations become `<label>_<n>`.
- Read operations from the processed model for attribution, keeping the readers
  free of synthetic-label generation.
- Parse quoted MLIR location labels without treating parentheses inside strings
  as location delimiters, preserving operations during preparation and rereading.
- Convert the processed model to VGF, using the backend-repository installation
  when the default converter lookup returns no path.
- Prepare single-graph VGF inputs and compile every returned graph segment.
- Store compiler evidence in a private `_CompilerRecord` dataclass. Key attributed
  records by TOSA source operation ID, merging placements and distinct compiler
  operation types across records and segments. Keep unattributed records separate.
- Expose merged compiler types in report entity names and `op_type` attributes,
  falling back to the parsed source TOSA type when compiler types are unavailable.
- Emit one compatibility check per attributed TOSA operation. Calculate the
  compatibility percentage from these consolidated records plus all unattributed
  records, rather than weighting source operations by compiler expansion count.
- Keep synthetic labels for attribution while removing their generated suffix
  from display labels. Render numbered unknown labels as `unknown`.
- Ignore blank labels when usable labels exist; otherwise preserve one
  unattributed `unknown` record per estimator operation so its placement still
  contributes to compatibility metrics.
- Use compiler-key-derived IDs under `converted_operator/unattributed/` for
  unattributed converted entities, avoiding numeric TOSA ID collisions and
  changes caused by record ordering.
- Attribute compatibility results to `nx-performance-estimator`; declare
  `ml-sdk-model-converter` only when the executed path used it.

## Capabilities

### New Capabilities

- `tosa-compatibility-preparation`: Prepare uniquely labelled TOSA models and
  attribute segment-level estimator placements to their source operations.

### Modified Capabilities

- None. The existing operator-percentage change remains separate; its formula
  is retained, but the TOSA record population is consolidated by source ID.

## Impact

- `src/mlia/backend/nx_performance_estimator/compat.py`
- `src/mlia/backend/ml_sdk_model_converter/tosa_reader.py`
- Neural Technology data collection and imports of the relocated module
- Compatibility and API tests, with updated mock targets and aggregation regressions
- Processed model copies and single-graph VGF files in the run output directory

The former compatibility-module import path is removed without a forwarding
alias. Original model files and the standardized output schema are unchanged.
