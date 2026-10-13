<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

This checklist records TOSA preparation, source-keyed aggregation, and
unattributed-record handling.
Checked implementation items were verified against the current source;
validation is tracked separately.

## 1. Preparation And Attribution

- [x] 1.1 Relocate compatibility handling and update imports and mock targets.
- [x] 1.2 Keep TOSA readers free of synthetic-label generation and file mutation.
- [x] 1.3 Assign prefix-based counters inside model preparation and write separate FlatBuffer and MLIR copies.
- [x] 1.4 Read operations from the processed model and use a one-label-to-one-ID mapping.
- [x] 1.5 Preserve original display prefixes and render numbered unknown labels as `unknown`.
- [x] 1.6 Parse parentheses inside quoted MLIR labels without truncating preparation spans or dropping operations on reread.

## 2. Compiler Execution And Reporting

- [x] 2.1 Use the converter backend-repository fallback when default lookup returns no path.
- [x] 2.2 Prepare and process each converted VGF graph segment with segment-scoped record keys.
- [x] 2.3 Combine NX and EE evidence for the same source ID as `NX/EE`.
- [x] 2.4 Ignore blank compiler labels when usable labels exist; otherwise preserve one unattributed estimator record in both compatibility paths.
- [x] 2.5 Declare only executed backends and keep the estimator as result producer.
- [x] 2.6 Calculate metrics from source-consolidated TOSA records and separate unattributed records, without incomplete-coverage handling.
- [x] 2.7 Use `_CompilerRecord` in source-keyed and unattributed collections; remove tuple indexing, the seen-ID set, and the separate placement map.
- [x] 2.8 Merge placement and distinct compiler op types directly into each attributed record while retaining its first key and label.
- [x] 2.9 Pass merged compiler types through to reported entity names and types, falling back to the source TOSA type only when needed.
- [x] 2.10 Derive unattributed converted entity IDs from compiler record keys in a separate namespace, without claiming TOSA IDs.

## 3. Existing Tests And Validation

- [x] 3.1 Update existing fully-NX and mixed-NX/EE scenarios for both formats, repeated or empty labels, unchanged input files, and processed-model reads.
- [x] 3.2 Add a separate parameterized regression test for unlabelled compiler records, including metric accounting and no provenance lookup.
- [x] 3.3 Pass the current non-slow suite with the debug attribution override disabled (383 passed, 5 skipped).
- [x] 3.4 Validate Markdown, YAML, SPDX headers, and diff whitespace for this change.
- [x] 3.5 Run `openspec validate prepare-tosa-compatibility-models --strict`.
- [x] 3.6 Cover duplicate source IDs across segments, separate unknown records, both placement arrival orders, repeated types, multiple types per entry, and missing types.
- [x] 3.7 Verify reader and compatibility tests (52 passed) and scoped Ruff, pydocstyle, and mypy checks after the final changes.
- [x] 3.8 Verify merged compiler types and missing-type fallback in source and converted entity output; run the average-pooling JSON reproducer.
- [x] 3.9 Verify unattributed ID uniqueness, stability under reordering, check references, and schema validation.
- [x] 3.10 Verify referenced and inline parenthesized labels round-trip through preparation and run a compiler smoke test with `conv(foo)`.

The updated artifacts pass strict OpenSpec validation. All implementation and
validation tasks are complete; the change is ready for archival review.
