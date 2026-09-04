<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Backends and Analysis Modes

This package provides one estimator backend, one measured-profiling backend, and
two internal conversion support integrations.

| Backend | Role |
| --- | --- |
| `nx-performance-estimator` | Estimates compatibility and performance from a model |
| `neural-technology-profiling-data` | Produces measured performance results from structured captures |
| `ml-sdk-model-converter` | Converts supported model formats for Neural Technology tools |
| `tosa-flatbuffers` | Provides TOSA FlatBuffers serialization support |

## NX Performance Estimator

The Python integration outside the proprietary directories is licensed under
Apache-2.0. All contents of the following directories are provided under
`LicenseRef-LICENSE`:

- `src/mlia/_vendor/artifacts/nx-performance-estimator/`
- `src/mlia/resources/nx-performance-estimator/`

See `LICENSES/LicenseRef-LICENSE.txt` in the repository root.

`nx-performance-estimator` is the primary model-analysis backend. Use it for:

- Model-level compatibility and performance results.
- Per-chain and per-cascade cycle breakdowns.
- Memory traffic and hardware-section utilisation metrics.
- Canonical source-operator, module, and source-code provenance.
- Control over the packaged system and compiler configuration.

Use it when you want:

A straightforward estimator run uses an input that this package can process
without an external framework converter:

```bash
mlia check model.tosa \
  --target-profile neural-technology \
  --performance \
  --backend nx-performance-estimator
```

LiteRT, PyTorch, and ExecuTorch inputs require their corresponding converter
plugins. VGF inputs can be analyzed directly.

### Configuration options

```bash
mlia check model.tosa \
  --target-profile neural-technology \
  --performance \
  --backend nx-performance-estimator \
  --nx-performance-estimator.system-config ./system.ini \
  --nx-performance-estimator.compiler-config ./compiler.ini
```

| Option | Meaning |
| --- | --- |
| `--nx-performance-estimator.system-config` | Override the system configuration file |
| `--nx-performance-estimator.compiler-config` | Override the graph-compiler configuration file |

## Measured profiling data

`neural-technology-profiling-data` analyzes schema-version 2 captures produced by
`VK_LAYER_LGL_neural_statistics`. Supplying `--profiling-data` switches the run
to measured mode and causes core MLIA to select this profiling-capable backend.
The backend is built in and normally does not need to be named explicitly.

Measured profiling supports performance analysis only. It accepts either:

- one dispatch directory without a model;
- a capture root or dispatch anchor associated with a VGF model; or
- one explicitly ordered dispatch directory per VGF graph segment.

The measured path uses the same correlation, aggregation, entity provenance,
and standardized-output construction as estimator mode. See [CLI](cli.md) for
the capture layout and selection rules.

## Conversion support integrations

`ml-sdk-model-converter` prepares supported inputs for the estimator, while
`tosa-flatbuffers` supplies lower-level TOSA serialization support. They use the
backend plugin mechanism for dependency integration, but they do not produce
independently selectable analysis results.

A useful model of the estimator path is:

1. MLIA receives a model in a supported format.
2. A framework converter runs when the original format requires one.
3. For non-VGF inputs, the ML SDK conversion path produces an estimator-ready
   VGF artifact. Existing VGF inputs proceed directly to segment preparation.
4. The NX Performance Estimator produces standardized results.
5. Core MLIA validates, post-processes, and renders those results.

## Cross-links

- See [Outputs and metrics](outputs_metrics.md) for result structure and entity
  provenance.
- See [CLI](cli.md) for estimator and measured-profiling examples.
- See [Troubleshooting](troubleshooting.md) for conversion, estimator, and
  capture-ingestion failures.
