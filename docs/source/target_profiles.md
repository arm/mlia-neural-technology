<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Neural Technology Target Profiles

## Overview

This package provides the MLIA target plugin for Arm Neural Technology
analysis. It adds target-specific configuration, standardized-result collection,
and backend wiring for estimator and measured-profiling flows. Core MLIA owns
output post-processing and rendering.

## Bundled profiles

The package ships built-in profiles under
`src/mlia/resources/target_profiles/`:

- `neural-technology`
- `NX-peak-12SC-8NX-600MHz`
- `NX-sustained-12SC-8NX-350MHz`

These profiles are the starting point for MLIA analysis and can be referenced by
name from CLI commands.

## Supported input formats

This package participates in workflows for several model formats:

- LiteRT / TensorFlow Lite (`.tflite`).
- TOSA (`.tosa`, `.tosamlir`).
- VGF (`.vgf`).
- PyTorch export (`.pt2`).
- ExecuTorch PTE (`.pte`).

Not every format is consumed directly by the NX Performance Estimator. Some are
handled through conversion integrations in this package or through installed
converter plugins.

## Compatible converter plugins

Use these converter plugins when the input is not already in a directly usable
format for the Neural Technology flow:

- `mlia-converters-litert` for LiteRT / TensorFlow Lite (`.tflite`) inputs.
- `mlia-converters-pytorch` for PyTorch export (`.pt2`) and ExecuTorch PTE
  (`.pte`) inputs.

If you are documenting or demonstrating the Neural Technology plugin on its
own, a `.tosa` model is usually the clearest default example because it avoids
assuming those extra converter packages are installed.

## Typical usage

Compatibility check:

```bash
mlia check my_model.tosa --target-profile neural-technology \
  --compatibility --backend nx-performance-estimator
```

Performance analysis:

```bash
mlia check my_model.tosa --target-profile neural-technology \
  --performance --backend nx-performance-estimator
```

PyTorch export flow:

```bash
mlia check my_model.pt2 --target-profile neural-technology --performance
```

In the PyTorch case, MLIA can automatically invoke the PyTorch-to-TOSA
conversion path before the performance estimator runs, provided
`mlia-converters-pytorch` is installed.

LiteRT / TensorFlow Lite `.tflite` inputs can also flow through the Neural
Technology path, but they depend on `mlia-converters-litert` to prepare the
estimator input.

## Measured profiling data

Structured captures from `VK_LAYER_LGL_neural_statistics` can be analyzed without
a model when one dispatch is selected. A VGF model can instead correlate one or
more graph segments with captured pipelines. Profiling-data runs support
performance analysis only and use the built-in
`neural-technology-profiling-data` backend automatically.

See [CLI](cli.md) for the accepted capture layout and dispatch-selection rules.

## Configuration concepts

Neural Technology flows commonly depend on:

- Target profile selection.
- System configuration files.
- Compiler configuration files.
- Conversion steps between model formats and estimator inputs.

This package provides both the target-side logic and much of the backend
plumbing needed to keep that workflow together.

## Outputs

The target and selected analysis mode contribute standardized results containing:

- Model-level cycles, throughput, utilization, memory, and availability metrics.
- Per-chain and per-cascade breakdowns with explicit aggregation policies.
- Segment, source-operator, module, and source-code provenance entities.
- Measured-mode metadata when structured profiling captures are analyzed.
- Intermediate conversion or estimator artifacts used for diagnostics.

Core MLIA validates, post-processes, and renders these results. See
[Outputs and metrics](outputs_metrics.md) for the detailed structure.
