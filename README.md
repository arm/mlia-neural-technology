<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Neural Technology Plugin

This package provides the Neural Technology target plugin and NX backends for MLIA.

## Requirements

- Python 3.10
- `uv`

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
- `src/mlia/backend/ml_sdk_model_converter/`: ML SDK model converter backend.
- `src/mlia/backend/tosa_flatbuffers/`: TOSA flatbuffers backend.
- `src/mlia/resources/`: bundled target profiles and backend assets.
- `tests/`: unit tests covering plugins, CLI integration, and target/backend behavior.
