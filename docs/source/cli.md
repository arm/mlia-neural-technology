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
mlia check my_model.tosa \
  --target-profile neural-technology \
  --performance \
  --backend nx-performance-estimator
```

## A compatibility-oriented first pass

Use a compatibility check to confirm that a model is supported by the Neural
Technology workflow.

```bash
mlia check my_model.tosa \
  --target-profile neural-technology \
  --compatibility \
  --backend nx-performance-estimator
```

## Backend-specific configuration

Use backend-specific overrides when you want to validate different hardware or
compiler assumptions.

```bash
mlia check my_model.tosa \
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
mlia check my_model.pt2 --target-profile neural-technology --performance
```

If the run fails early, the problem may belong to the conversion stages before
the estimator ever runs.

The same idea applies to LiteRT / TensorFlow Lite `.tflite` input. Use `.tosa`
for the plugin-only path, then install `mlia-converters-litert` or
`mlia-converters-pytorch` when you want MLIA to accept `.tflite`, `.pt2`, or
`.pte` directly.

## Using JSON output

```bash
mlia check my_model.tosa \
  --target-profile neural-technology \
  --performance \
  --backend nx-performance-estimator \
  --json
```

## Using measured profiling data

The `neural-technology-profiling-data` backend reads schema-version 2 structured
captures produced by `VK_LAYER_LGL_neural_statistics`:

```text
capture.json
pipeline_<id>/pipeline.json
pipeline_<id>/session_<id>/session.json
pipeline_<id>/session_<id>/dispatch_<id>/dispatch.json
```

Metadata references, IDs, parent relationships, paths, artifact types, byte
sizes, capture status, and the capture device are validated before analysis.
Profiling input must use this structured hierarchy. Pipeline inputs must declare
`debug_database.bin`, either `neural_statistics_info.bin` or
`neural_statistics_info.txt`, and one `shader_module_<id>.spv`; each selected
dispatch must declare exactly one mode-matching statistics artifact.

Analyze a dispatch without a source model, or use a capture root only when it
contains exactly one dispatch:

```bash
mlia check \
  --target-profile neural-technology \
  --performance \
  --profiling-data ./capture/pipeline_000000/session_000000/dispatch_000000
```

With a VGF model, MLIA matches graph segments to captured pipelines by exact
SPIR-V bytes. A capture root can be selected automatically when every graph
segment has one unique matching pipeline and that pipeline has exactly one
dispatch. A single dispatch can instead act as an executed-index anchor for the
other uniquely matched pipelines:

```bash
mlia check my_model.vgf \
  --target-profile neural-technology \
  --performance \
  --profiling-data ./capture
```

For ambiguous or intentionally repeated captures, repeat `--profiling-data` with
one dispatch directory per VGF graph segment, in graph-segment order. Compute
segments are excluded consistently with estimator mode:

```bash
mlia check my_model.vgf \
  --target-profile neural-technology \
  --performance \
  --profiling-data ./capture/pipeline_000000/session_000000/dispatch_000000 \
  --profiling-data ./capture/pipeline_000001/session_000001/dispatch_000001
```

All explicit dispatches must belong to the same capture and match the
corresponding graph segment. Profiling data uses the same per-segment
correlation, aggregation, totals, warnings, entities, and standardized-output
flow as the NX Performance Estimator. Measured profiling supports
`--performance` only.

## Practical debugging sequence

When a run is unclear, a useful sequence is:

1. Confirm the intended command with `mlia check --help`.
2. Run once with the standard estimator path.
3. Add config overrides or alternate input paths as needed.
4. Inspect whether the failure belongs to conversion or estimation.

## When to use which path

- Use the plain estimator path for normal analysis.
- Use compatibility first when you are unsure whether the model can move through
  the expected flow cleanly.
- Use config overrides when validating system or compiler assumptions.
- Use `.pt2` or `.pte` inputs when you want to exercise the full
  PyTorch/ExecuTorch-to-analysis pipeline, not just the estimator stage, and
  have the PyTorch converter package installed.
