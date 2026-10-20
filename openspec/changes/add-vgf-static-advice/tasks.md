<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

# Tasks

This retrospective checklist distinguishes implemented behavior from remaining
regression coverage. Checked items were verified against the branch diff or the
test run recorded below. Pending items do not authorize source changes as part of
this documentation request.

## 1. Instruction Visitation And Metadata

- [x] 1.1 Add read-only mapped VGF visitation with independent instruction bytes and segment indexes; verify the implementation against the visitor contract in the spec and its docstring.
- [ ] 1.2 Add real-VGF visitor tests for shared modules, multiple segments, retained callback bytes, malformed instructions, and callback failures; verify with the focused VGF test module.
- [x] 1.3 Track scalar constants, integer-array shapes, tensor sizes, and operand IDs per segment; verify the synthetic CONV2D instruction stream test and document the metadata design.
- [ ] 1.4 Add a regression proving large integer tensor composites are not traversed or stored as shape payloads while shape metadata remains available; verify it fails without the type guard and passes with it.

## 2. Static Advice Rules

- [x] 2.1 Implement kernel-area-dependent CONV2D alignment advice; verify aligned, misaligned, zero, and negative dimensions with `test_conv2d_advice` and document the factors in the spec.
- [x] 2.2 Implement shallow depthwise advice for input depths one through three; verify depth boundaries and missing/invalid shapes with `test_depthwise_conv2d_advice`.
- [x] 2.3 Detect skip edges above 128 KiB without mutating operands or traversing constants; verify direct, multi-hop, threshold, shared-ancestor, and constant-operand cases with `test_skip_connection_advice`.
- [x] 2.4 Preserve existing advice and derive segment-scoped IDs; verify the parameterized `test_performance_advice` cases.

## 3. Integration And Entity Attribution

- [x] 3.1 Gather performance advice from each prepared graph through `_add_segment_advice`; verify repeated local result IDs, non-contiguous segment indexes, skipped compute segments, and unlinked records in performance tests (29 passed on 2026-10-20).
- [x] 3.2 Gather compatibility advice from each prepared VGF and TOSA-converted segment; verify existing multi-segment compatibility scenarios and original-segment advice IDs.
- [x] 3.3 Resolve direct and structured compiler labels into TOSA entity IDs; verify mapped and unmatched labels in `test_tosa_compatibility_scenarios` for both input formats.
- [x] 3.4 Filter serialized advice to existing NX-only entities; verify compatibility placement scenarios, standardized performance advice assertions, and schema validation.
- [x] 3.5 Remove generic advice injection and preserve backend advice through the API; verify `test_run_advisor_performance_routes_backend_options` and document the empty-list migration behavior.

## 4. Integration Validation

- [x] 4.1 Run the advice, compatibility, performance, and Python API test modules together; verified 133 passed and 1 skipped on 2026-10-19 using `uv run --no-sync pytest --no-cov -q`.
- [x] 4.2 Validate the new Markdown directly with the configured PyMarkdown rules, YAML with the pre-commit hook, and SPDX headers by inspection; `openspec validate add-vgf-static-advice --strict` passes.

The repository-wide REUSE hook still fails on pre-existing untracked model and
capture files without license metadata. None of the new OpenSpec artifacts are
reported as missing copyright or licensing information.

## Workflow follow-up

- Review and complete the pending regression coverage before archival.
- Use `/opsx:apply add-vgf-static-advice` when ready to address the remaining tasks.
