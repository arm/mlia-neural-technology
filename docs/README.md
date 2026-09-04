<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Neural Technology Documentation

This directory contains the MkDocs content for the `mlia-neural-technology`
repository.

## Included pages

- `source/index.md`: documentation landing page
- `source/target_profiles.md`: bundled profiles, supported inputs, and usage notes
- `source/backends.md`: estimator-oriented backend behaviour and options
- `source/outputs_metrics.md`: output shapes and key metrics produced by Neural Technology workflows
- `source/cli.md`: practical CLI usage examples for common Neural Technology tasks
- `source/troubleshooting.md`: backend-specific troubleshooting notes
- `source/development.md`: local development, testing, and maintenance workflow
- `source/neural_technology_api_walkthrough.ipynb`: Python API walkthrough for
  `torch.nn.Module` inputs

## Build

Install the documentation dependencies in your environment, then build from the
repository root:

```bash
uv sync --no-install-project --only-group docs
uv run mkdocs build --strict
```

For local preview:

```bash
uv run mkdocs serve
```

The generated site will be written to `.mkdocs/site/`.

## Scope

These docs focus on the Neural Technology target, model estimation, measured
profiling data, conversion support, and standardized outputs packaged by this
split repo.

## Relationship to the core repo

Use the main `mlia` repo for shared CLI and architecture concepts. Use this
docs tree for Neural Technology-specific target, backend, metric, and
troubleshooting detail.
