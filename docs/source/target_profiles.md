<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Neural Technology Target Profiles

## Overview

This package provides the MLIA target plugin for Arm Neural Technology
analysis. It adds target-specific configuration, reporting, and backend wiring
for flows that rely on the NX Performance Estimator and related conversion
backends.

Compared with Ethos-U, these targets are aimed at higher-throughput accelerator
flows and support a broader set of model-ingest paths through automatic
conversion.

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
handled through automatic conversion backends that are packaged in this package or
installed alongside it.

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
mlia check model.tosa --target-profile neural-technology --compatibility --backend nx-performance-estimator
```

Performance analysis:

```bash
mlia check model.tosa --target-profile neural-technology --performance --backend nx-performance-estimator
```

PyTorch export flow:

```bash
mlia check model.pt2 --target-profile neural-technology --performance
```

In the PyTorch case, MLIA can automatically invoke the PyTorch-to-TOSA
conversion path before the performance estimator runs, provided
`mlia-converters-pytorch` is installed.

LiteRT / TensorFlow Lite `.tflite` inputs can also flow through the Neural
Technology path, but they depend on `mlia-converters-litert` to prepare the
estimator input.

## Configuration concepts

Neural Technology flows commonly depend on:

- Target profile selection.
- System configuration files.
- Compiler configuration files.
- Conversion steps between model formats and estimator inputs.

This package provides both the target-side logic and much of the backend
plumbing needed to keep that workflow together.

## Outputs

The target and backend combination in this package contributes:

- Model-level cycle estimates.
- Per-operator statistics.
- Memory-traffic reporting.
- Generated intermediate artifacts used by downstream tools.

The exact outputs depend on the chosen backend and any automatic conversion path
that precedes it.
