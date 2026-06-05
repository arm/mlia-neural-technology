<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Neural Technology Plugin

This package contains the MLIA target plugin for Arm Neural Technology flows
and packages the backend integrations used to estimate performance and work with
TOSA-related assets in that environment.

The package is distributed as `mlia-neural-technology` and contributes:

- The `neural-technology` target plugin.
- The `nx-performance-estimator` backend plugin.
- The `ml-sdk-model-converter` backend plugin.
- The `tosa-flatbuffers` backend plugin.
- Bundled Neural Technology target profiles and backend configuration assets.

## Table of Contents

- [Overview](#overview)
- [Supported targets](#supported-targets)
- [Backends in this package](#backends-in-this-package)
- [Installation](#installation)
- [Reporting bugs](#reporting-bugs)
- [Development Setup](#development-setup)
- [Common Commands](#common-commands)
- [Project Layout](#project-layout)
- [Documentation](#documentation)
- [Trademarks and copyrights](#trademarks-and-copyrights)

## Overview

`mlia-neural-technology` extends MLIA with Neural Technology target knowledge,
profile assets, and backend plugins used by NX-oriented performance and model
conversion workflows.

This package is responsible for the target-side integration and backend
wiring for Neural Technology analysis. It is typically installed alongside the
core `mlia` package and, depending on the workflow, other MLIA converter
plugins.

For upstream model formats, the compatible converter plugins are:

- `mlia-converters-tflite` for `.tflite` inputs.
- `mlia-converters-pytorch` for `.pt2` and `.pte` inputs.

## Supported targets

Bundled target profiles include:

- `neural-technology`
- `NX-peak-12SC-8NX-600MHz`
- `NX-sustained-12SC-8NX-350MHz`

These profiles are shipped under `src/mlia/resources/target_profiles/`.

## Backends in this package

### NX Performance Estimator

Provides the performance-estimation flow for Neural Technology targets and uses
bundled system and graph-compiler configuration assets.

### ML SDK Model Converter

Provides model-conversion support within the Neural Technology workflow and is
packaged here as a separate MLIA backend plugin.

### TOSA FlatBuffers

Provides the FlatBuffers-side integration used by TOSA-related parts of the
Neural Technology stack.

## Installation

Install into an environment that already contains `mlia`:

```bash
pip install mlia-neural-technology
```

A typical MLIA workflow then references one of the bundled profiles, for
example:

```bash
mlia check model.tosa --target-profile neural-technology
```

Use `.tosa` as the clearest default example for this package on its own. Direct
`.tflite`, `.pt2`, and `.pte` flows depend on the matching converter plugins being
installed.

The package depends on `mlia==0.11.0.dev28` and is intended to be used as part
of a wider MLIA installation rather than as a standalone CLI.

## Reporting bugs

Report bugs by creating GitHub issues. Use the
[`arm/mlia` issue tracker](https://github.com/arm/mlia/issues) by default.

Only open an issue in
[`arm/mlia-neural-technology`](https://github.com/arm/mlia-neural-technology/issues)
when the bug is clearly and specifically in this plugin.

## Development Setup

Create a local environment with the project and all development dependencies:

```bash
uv sync --group dev
```

If you only need the test dependencies:

```bash
uv sync --group test
```

This repository currently does not use a committed lock file (`uv.lock`).

## Common Commands

Run the quick test suite used in CI:

```bash
uv run pytest -m "not slow" tests/
```

Run the full test suite with coverage:

```bash
uv sync --group test
uv run pytest tests/
```

Run the local quality checks:

```bash
uv run pre-commit run --all-files
```

Build the package:

```bash
uv build
```

## Documentation

- `docs/source/neural_technology_api_walkthrough.ipynb`: walkthrough of the Neural Technology Python API flow for `torch.nn.Module` inputs.

## Project Layout

- `src/mlia/target/neural_technology/`: Neural Technology target integration and advisor logic.
- `src/mlia/backend/nx_performance_estimator/`: NX performance estimator backend.
- `src/mlia/backend/ml_sdk_model_converter/`: model-converter backend.
- `src/mlia/backend/tosa_flatbuffers/`: TOSA FlatBuffers backend.
- `src/mlia/resources/`: bundled target profiles and backend assets.
- `tests/`: unit tests covering plugins, CLI integration, and target/backend behaviour.

## Documentation

Additional package documentation lives in [docs/README.md](docs/README.md).

## Trademarks and copyrights

- Arm is a registered trademark or trademark of Arm Limited (or its subsidiaries) in the U.S. and/or elsewhere.
- TensorFlow is a trademark of Google LLC.
- PyTorch and ExecuTorch are trademarks of The Linux Foundation.
- Linux is the registered trademark of Linus Torvalds in the U.S. and elsewhere.
- Python is a registered trademark of the PSF.
