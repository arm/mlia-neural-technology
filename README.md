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
- The `nx-performance-estimator` model-analysis backend.
- The `neural-technology-profiling-data` measured-analysis backend.
- The internal `ml-sdk-model-converter` conversion integration.
- The internal `tosa-flatbuffers` serialization integration.
- Bundled Neural Technology target profiles and backend configuration assets.

## Table of Contents

- [Overview](#overview)
- [Supported targets](#supported-targets)
- [Backends in this package](#backends-in-this-package)
- [Installation](#installation)
- [Releases](#releases)
- [Reporting bugs](#reporting-bugs)
- [Development Setup](#development-setup)
- [Common Commands](#common-commands)
- [Project Layout](#project-layout)
- [Documentation](#documentation)
- [License](#license)
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

- `mlia-converters-litert` for LiteRT / TensorFlow Lite `.tflite` inputs.
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

### Measured profiling data

The built-in `neural-technology-profiling-data` backend analyzes structured
captures from `VK_LAYER_LGL_neural_statistics`. It is selected automatically
when `--profiling-data` is supplied and supports performance analysis only.

### Conversion support

`ml-sdk-model-converter` prepares models for downstream Neural Technology tools,
and `tosa-flatbuffers` supplies low-level TOSA serialization support. Both are
internal, non-selectable dependency integrations rather than analysis backends.

## Installation

The wheel requires CPython 3.12 (`>=3.12,<3.13`). On x86-64 Linux, it requires
Ubuntu 24.04 or newer, or another distribution with glibc 2.39 or newer.
Linux wheels use a `manylinux_2_39` platform tag so installers reject older
glibc versions. Builds on newer glibc versions retain that higher minimum.
The bundled native `vgfpy` extension also requires a matching CPython ABI.

Install into an environment that already contains `mlia`:

```bash
pip install mlia-neural-technology
```

A typical MLIA workflow then references one of the bundled profiles, for
example:

```bash
mlia check my_model.tosa --target-profile neural-technology
```

Measured profiling data can be analyzed without a source model when one dispatch
is selected:

```bash
mlia check --target-profile neural-technology --performance \
  --profiling-data ./capture/pipeline_000000/session_000000/dispatch_000000
```

Use `.tosa` as the clearest default model example for this package on its own.
Direct LiteRT / TensorFlow Lite `.tflite`, `.pt2`, and `.pte` flows depend on the
matching converter plugins being installed.

The supported Python range and required core MLIA dependency are maintained in
[`pyproject.toml`](pyproject.toml), which is the authoritative source for current
installation requirements. This package is intended to be used as part of a
wider MLIA installation rather than as a standalone CLI.

## Releases

Latest changes and release history can be found in
[MLIA Neural Technology releases](https://github.com/arm/mlia-neural-technology/releases).

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

## Project Layout

- `src/mlia/target/neural_technology/`: Neural Technology target integration and advisor logic.
- `src/mlia/backend/nx_performance_estimator/`: NX performance estimator backend.
- `src/mlia/backend/neural_technology_profiling_data/`: measured capture parsing
  and analysis.
- `src/mlia/backend/ml_sdk_model_converter/`: internal model-conversion support.
- `src/mlia/backend/tosa_flatbuffers/`: internal TOSA FlatBuffers support.
- `src/mlia/resources/`: bundled target profiles and backend assets.
- `tests/`: unit tests covering plugins, CLI integration, and target/backend behaviour.

## Documentation

Additional package documentation lives in [docs/README.md](docs/README.md). The
[Python API walkthrough](docs/source/neural_technology_api_walkthrough.ipynb)
covers `torch.nn.Module` inputs and structured Python results.

## License

The source code and configuration in this repository are licensed under the
Apache License 2.0 unless a file states otherwise. See
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) for the full license text.

All contents of these directories are proprietary and are provided under
[`LicenseRef-LICENSE`](LICENSES/LicenseRef-LICENSE.txt):

- `src/mlia/_vendor/artifacts/nx-performance-estimator/`
- `src/mlia/resources/nx-performance-estimator/`

This exception does not apply to the open-source Python integration outside
those directories.

## Trademarks and copyrights

- Arm is a registered trademark or trademark of Arm Limited (or its subsidiaries) in the U.S. and/or elsewhere.
- TensorFlow is a trademark of Google LLC.
- PyTorch and ExecuTorch are trademarks of The Linux Foundation.
- Linux is the registered trademark of Linus Torvalds in the U.S. and elsewhere.
- Python is a registered trademark of the PSF.
