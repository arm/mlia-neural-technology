<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Neural Technology

## Purpose

`mlia-neural-technology` packages the Neural Technology plugins used by MLIA.

## Included plugins

- Use `neural_technology` as the Neural Technology target plugin.
- Use `nx-performance-estimator` as the primary analysis backend.
- Use `ml-sdk-model-converter` as the Neural Technology conversion backend.
- Use `tosa-flatbuffers` as the low-level TOSA support backend.

Compatible converter plugins for upstream model formats include
`mlia-converters-litert` (for LiteRT / TensorFlow Lite `.tflite` models) and
`mlia-converters-pytorch`.

## Documentation Map

- [Target profiles](target_profiles.md)
- [Backends and conversion flow](backends.md)
- [Outputs and metrics](outputs_metrics.md)
- [CLI examples](cli.md)
- [Troubleshooting](troubleshooting.md)
- [Development](development.md)
