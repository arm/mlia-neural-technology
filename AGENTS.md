<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Repository Guidelines

## Overview

This repository provides the MLIA Neural Technology target plugin and the
NX-oriented backend integrations. The Python package lives under `src/mlia`,
tests live under `tests`, and OpenSpec change proposals/specs live under
`openspec`.

## Working Rules

- Use `uv` for environment management, dependency execution, test execution, and
  builds. Do not reintroduce `tox`, `setup.cfg`, `pip install -e`, or ad hoc
  virtualenv instructions.
- Use test-driven development for behavior changes:
  - add or update a failing test or characterization test first
  - implement the minimum code needed to satisfy that test
  - run the narrowest relevant test target before broadening validation
- When working from OpenSpec, use a double-loop workflow:
  - outer loop: keep proposal, design, spec, and tasks aligned with the intended
    behavior
  - inner loop: write a failing public-behavior test or characterization test
    for each slice, make it pass, then refactor only after the test is green
- Keep changes scoped to the task. `.codex/`, `.github/skills/`, and
  `openspec/` may contain draft or workflow support files that should not always
  be staged or committed with code changes.
- Add or update tests when behavior changes, and prefer public-behavior tests
  over assertions tied to internal file shape.
- Keep new files compatible with REUSE and local copyright checks by following
  the SPDX header or `.license` sidecar pattern already used in this repo.
- If you touch packaging, CI, or dependencies, keep `pyproject.toml`,
  `.pre-commit-config.yaml`, and `.github/workflows/` aligned.

## Setup And Validation

Install development dependencies:

```bash
uv sync --group dev
```

If you only need test dependencies:

```bash
uv sync --group test
```

Common validation commands:

```bash
uv run pytest -m "not slow" tests/
uv run pre-commit run --all-files
uv build
```

For a full test run with coverage:

```bash
uv sync --group test
uv run pytest tests/
```

When running tools in this repo, prefer `uv run ...` even if another virtualenv
is active. This repository currently does not use a committed `uv.lock`.

## Repo Map

- `src/mlia/target/neural_technology/`: Neural Technology target plugin,
  profiles, and target-local logic.
- `src/mlia/backend/nx_performance_estimator/`: NX performance estimator
  backend registration, configuration, output parsing, statistics, and
  performance flow.
- `src/mlia/backend/ml_sdk_model_converter/`: ML SDK model-converter backend
  registration, conversion, TOSA reader, and compatibility helpers.
- `src/mlia/backend/tosa_flatbuffers/`: TOSA FlatBuffers backend integration.
- `src/mlia/nx_utils/`: shared filesystem and utility helpers for Neural
  Technology flows.
- `src/mlia/resources/`: bundled target profiles and backend configuration
  assets.
- `tests/`: regression coverage for plugins, CLI integration, Python API,
  backend behavior, parsing, statistics, and resources.
- `pre_commit_hooks/check_copyright_header.py`: local hook for current-year
  copyright checks.
- `.codex/skills/` and `.github/skills/`: local OpenSpec workflow skills.
- `openspec/`: change proposals, designs, tasks, and specs.

## OpenSpec Skills

Use the local OpenSpec skills when the task matches their ownership:

- `openspec-explore`: think through or clarify a change before implementation.
- `openspec-propose`: create a proposal, design, spec delta, and task list for a
  new change.
- `openspec-apply-change`: implement an accepted change and keep tasks aligned
  with the code.
- `openspec-archive-change`: archive a completed change after validation.

## Change Hygiene

- Prefer targeted test runs for the area you changed, then run broader
  validation if the change is cross-cutting.
- Review adjacent layers when changing target profiles, estimator
  configuration, conversion backend integration, resource files, or documented
  workflows.
- Before considering behavior complete, check whether target discovery, backend
  CLI options, estimator config examples, and conversion-driven end-to-end flows
  still make sense.
- Keep docs and examples executable from the repository root with `uv`.
- If a change should exclude `.codex/`, `.github/skills/`, or `openspec/`,
  verify the staged set before committing.
