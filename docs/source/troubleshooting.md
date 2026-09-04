<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Troubleshooting

## How to approach a failure

Neural Technology runs can fail in more than one place. Some failures belong to
the estimator itself. Others happen in the conversion path before estimation
starts. Some runs do not fail at all, but produce outputs that are thinner or
stranger than expected.

Because of that, the most useful first question is often not "What command did I
run?" but "Did the model reach the estimator stage at all?"

## General setup issues

### Target profile not found

If MLIA cannot find the Neural Technology target profile, treat that as a setup
problem first.

- Run `mlia target list` to confirm the profile name.
- Check that `mlia-neural-technology` is installed in the active environment.
- Use a file path if you are testing a custom target profile.

There is no value in debugging estimator output until the target profile itself
is being discovered correctly.

### Input-format confusion

This package can participate in workflows that begin from several different model
formats, so confusion about the input type can easily turn into confusion about
where the failure really belongs.

- Confirm whether the input is `.tflite`, `.pt2`, `.pte`, `.tosa`,
  `.tosamlir`, or `.vgf`
- Remember that some formats trigger automatic conversion before the estimator
  runs

If the model never reaches the estimator, the most useful clues are often in the
conversion stages rather than in the final backend.

## NX Performance Estimator issues

### Backend missing

If MLIA cannot find the estimator backend, confirm that the environment exposes
it first.

- Run `mlia backend list`.
- Install or reinstall with `mlia backend install nx-performance-estimator`.

### Config override errors

If a run fails only after you add custom configuration, reduce the command back
to the baseline path first.

- Re-check the paths passed to
  `--nx-performance-estimator.system-config` and
  `--nx-performance-estimator.compiler-config`
- Start without overrides first to confirm the packaged baseline flow works.

That tells you whether the issue is with the backend itself or with the
additional assumptions you introduced.

### Metrics look inconsistent

When the estimator does run but the result looks strange, start by checking
whether the inconsistency is really in the final numbers or in how you are
reading them.

- Compare `total_cycles` with the breakdowns that dominate `op_cycles`.
- Inspect memory-traffic metrics before assuming the estimator itself is wrong.
- Check whether the result shape looks complete before drawing conclusions from
  a single field.

## Conversion-path issues

### TFLite, PyTorch, or PTE flow fails before estimation

If the failure happens before the estimator produces output, treat it as part of
the conversion story first, not as an estimator problem.

- Simplify the test case with a smaller model or a known-good format.
- Check whether the model contains unsupported or awkward operations for the
  conversion chain
- Confirm which stage completed last before the failure appeared.

### Intermediate artifacts seem to be the problem

If generated artifacts look wrong or incomplete, inspect them before blaming the
final backend.

- Look in the run output directory for generated artifacts and logs.
- Verify that the earlier conversion stage completed before focusing on the
  estimator

This is often the fastest way to separate "the estimator gave a bad answer"
from "the estimator never received a clean input."

## Profiling-data issues

### Capture input is rejected

The measured backend accepts schema-version 2 structured captures, not the old
flat profiling directory. Check that metadata references resolve within one
capture and that every selected dispatch declares exactly one statistics
artifact matching the capture mode.

Without a VGF model, provide one dispatch directory, or a capture root containing
exactly one dispatch. With a VGF model, ensure captured pipeline SPIR-V matches
each graph segment. When automatic matching is ambiguous, repeat
`--profiling-data` once per graph segment in VGF order.

### Profiling categories or options are rejected

Measured profiling supports `--performance` only. It does not accept
compatibility analysis or NX estimator backend options. All explicitly selected
dispatches must belong to the same capture and logical device.

## When the run succeeds but the result still looks wrong

A successful run can still need troubleshooting. In that case, ask whether the
main issue looks like:

- Unexpectedly high total cost.
- Operator concentration in a few heavy steps.
- Unusually high traffic or memory movement.
- A result shape that suggests the pipeline did not complete cleanly.

That framing usually points you toward the right evidence much faster than
reading every output field equally.

## Escalation path

If you are unsure whether the problem belongs here or in a converter package:

1. Identify the original model format.
2. Confirm whether conversion completed.
3. Only blame the estimator after the conversion stage has clearly succeeded.
