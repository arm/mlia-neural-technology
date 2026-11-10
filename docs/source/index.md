---
# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
sidebar_position: 1
slug: /documentation
---

# MLIA for Arm Mali with Neural Technology

## Purpose

`mlia-neural-technology` registers target and backend capabilities for supported
Arm Mali GPUs with Neural Technology in MLIA. These docs cover target profiles, model estimation,
measured profiling data, output interpretation, and troubleshooting.

## Supported plugins

Compatible converter plugins for upstream model formats include
`mlia-converters-litert` (for LiteRT / TensorFlow Lite `.tflite` models) and
`mlia-converters-pytorch`.

## Start here: Arm Learning Path

**[Estimate neural accelerator (NX) performance with ML Inference Advisor](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/mlia-nx-performance-estimator/)**

Follow this hands-on guide to install the plugin, analyze quantized MobileNet
models, check operator compatibility, and interpret NX Performance Estimator
reports.

## Documentation Map

- [Target profiles](target_profiles.md)
- [Backends and conversion flow](backends.md)
- [Outputs and metrics](outputs_metrics.md)
- [CLI examples](cli.md)
- [ETRecord and ETDump profiling](etrecord.md)
- [Download the Python API walkthrough notebook](neural_technology_api_walkthrough.ipynb)
- [Troubleshooting](troubleshooting.md)
- [Development](development.md)

For shared CLI, installation, and plugin architecture guidance, see the
[MLIA core documentation](https://developer.arm.com/dev-docs/mlia/).
