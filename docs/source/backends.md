<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Backends and Conversion Flow

This package provides the Neural Technology backends used by MLIA.

| Backend | Role | Why it matters |
| --- | --- | --- |
| `nx-performance-estimator` | Primary analysis and performance-estimation backend | Produces the numbers most people care about |
| `ml-sdk-model-converter` | Conversion backend used in Neural Technology workflows | Prepares models for downstream Neural Technology tooling |
| `tosa-flatbuffers` | Low-level TOSA handling support | Supports serialization and interchange used by other parts of the flow |

## NX Performance Estimator

`nx-performance-estimator` is the main backend in this package. Focus on it when
you care about performance, operator cost, memory movement, or utilisation.

Use it when you want:

- Model-level performance estimates.
- Per-operator cycle data.
- Memory traffic and utilisation insight.
- Control over system and compiler configuration used for estimation.

The implementation lives under
`src/mlia/backend/nx_performance_estimator/`.

### Common CLI patterns

A straightforward estimator-driven run looks like this:

```bash
mlia check model.tflite --target-profile neural-technology --performance --backend nx-performance-estimator
```

A more explicit run that overrides the packaged setup looks like this:

```bash
mlia check model.tflite \
  --target-profile neural-technology \
  --performance \
  --backend nx-performance-estimator \
  --nx-performance-estimator.system-config ./system.ini \
  --nx-performance-estimator.compiler-config ./compiler.ini
```

### Useful signals from the estimator

The estimator is most useful when you want:

- The top-level cost of a run.
- Which operators dominate the result.
- Where memory traffic is becoming expensive.
- How a configuration change affects the estimate.

### Option table

| Option | Meaning |
| --- | --- |
| `--nx-performance-estimator.system-config` | Override the system configuration file |
| `--nx-performance-estimator.compiler-config` | Override the compiler configuration file |

## ML SDK Model Converter

`ml-sdk-model-converter` is part of the path that prepares models for Neural
Technology tooling. If this stage is wrong or incomplete, the estimator never
gets a clean input.

## TOSA FlatBuffers

`tosa-flatbuffers` is the low-level support layer that helps other parts of the
Neural Technology flow serialize or move TOSA-oriented data correctly.

## Conversion and estimation flow

A practical way to think about the Neural Technology workflow is:

1. MLIA receives a model in a supported input format.
2. Framework-specific conversion happens when needed.
3. The Neural Technology conversion path prepares artifacts for the packaged
   estimator flow.
4. `nx-performance-estimator` consumes the prepared input and produces the main
   analysis outputs.

## Cross-links

- See [outputs_metrics.md](outputs_metrics.md) for how to interpret
  estimator-oriented outputs and diagnostics
- See [cli.md](cli.md) for example commands and option usage.
- See [troubleshooting.md](troubleshooting.md) when the pipeline breaks between
  conversion and estimation
