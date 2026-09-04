<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Development

## What counts as development in this repo

This repository owns the Neural Technology target plus the NX-oriented backend
path. Development here often crosses target profiles, estimator configuration,
conversion integration, and resource files used by the analysis flow.

## Local setup

Use `uv` to create and sync the development environment:

```bash
uv sync --group dev
```

If you only need the test stack:

```bash
uv sync --group test
```

## Common commands

Run the quick CI-aligned test suite:

```bash
uv run pytest -m "not slow" tests/
```

Run the full test suite:

```bash
uv run pytest tests/
```

Run repository checks:

```bash
uv run pre-commit run --all-files
```

Build the package:

```bash
uv build
```

## What usually changes together

When you update one of these areas, review the adjacent layers as well:

- Target profiles and target logic.
- Estimator configuration parsing.
- Conversion backend integration.
- Resource files under `src/mlia/resources/`.
- Output, provenance, and troubleshooting docs for estimator and measured-data
  workflows.

## Good review questions

Before you consider a change complete, ask:

- Does the target still appear correctly through `mlia target list`?
- Do backend-specific CLI options still behave as documented?
- Did estimator config changes affect the documented examples?
- If conversion is involved, does the end-to-end workflow still make sense?
- Do estimator and measured-data results retain compatible entities,
  aggregations, totals, and warnings?

## Documentation expectations

When profiles, input formats, backend options, capture schemas, or provenance
rules change, update this repo's docs so estimator and measured-data behaviour
remain understandable on their own.
