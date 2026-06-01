<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# CLI Guide

Use the core `mlia` CLI for Neural Technology runs. This page keeps the focus on
the Neural Technology-specific command shapes rather than repeating the general
CLI guidance from core `mlia`.

## The standard estimator-driven workflow

Most Neural Technology-specific examples should start with a `.tosa` model and
the straightforward estimator path:

```bash
mlia check model.tosa \
  --target-profile neural-technology \
  --performance \
  --backend nx-performance-estimator
```

## A compatibility-oriented first pass

If you are unsure whether the model can pass through the expected Neural
Technology path cleanly, start with compatibility before leaning on performance
numbers.

```bash
mlia check model.tosa \
  --target-profile neural-technology \
  --compatibility \
  --backend nx-performance-estimator
```

## Backend-specific configuration

Use backend-specific overrides when you want to validate different hardware or
compiler assumptions.

```bash
mlia check model.tosa \
  --target-profile neural-technology \
  --performance \
  --backend nx-performance-estimator \
  --nx-performance-estimator.system-config ./system.ini \
  --nx-performance-estimator.compiler-config ./compiler.ini
```

## PyTorch-driven workflow

If the required converter plugins are installed, MLIA can also analyze a
PyTorch-originating model through the wider pipeline:

```bash
mlia check model.pt2 --target-profile neural-technology --performance
```

If the run fails early, the problem may belong to the conversion stages before
the estimator ever runs.

The same idea applies to TensorFlow Lite input. Use `.tosa` for the plain
plugin-only path, then install `mlia-converters-tflite` or
`mlia-converters-pytorch` when you want MLIA to accept `.tflite`, `.pt2`, or
`.pte` directly.

## Using JSON output

```bash
mlia check model.tosa \
  --target-profile neural-technology \
  --performance \
  --backend nx-performance-estimator \
  --json
```

## Practical debugging sequence

When a run is unclear, a useful sequence is:

1. Confirm the intended command with `mlia check --help`.
2. Run once with the standard estimator path.
3. Only then add config overrides or alternate input paths.
4. Inspect whether the failure belongs to conversion or estimation.

## When to use which path

- Use the plain estimator path for normal analysis.
- Use compatibility first when you are unsure whether the model can move through
  the expected flow cleanly.
- Use config overrides when validating system or compiler assumptions.
- Use `.pt2` or `.pte` inputs when you want to exercise the full
  PyTorch/ExecuTorch-to-analysis pipeline, not just the estimator stage, and
  have the PyTorch converter package installed.
