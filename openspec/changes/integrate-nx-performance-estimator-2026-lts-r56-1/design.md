<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## Context

MLIA currently vendors the NX performance estimator metadata under
`src/mlia/_vendor/artifacts/nx-performance-estimator/`. The source archive is
resolved by `hatch_build.py` through the `graph-compiler-performance-estimator`
entry in `ARTIFACTS`, using `.sha256` to identify and verify the expected
archive and `VENDORED_ARTIFACTS_URLS` to provide the Artifactory URL at build
time.

At runtime, `get_nx_performance_estimator_installation()` declares the backend
as a vendored artifact and checks for a `graph-compiler-performance-estimator`
executable. `NXPerformanceEstimator` converts non-`.vgf`
models through `ml-sdk-model-converter`, invokes the estimator, expects three
output files, parses the debug and performance databases, reads the network
performance summary JSON, then maps the result to MLIA's standardized output
schema.

The candidate 2026 LTS r56.1 estimator is available locally as an unpacked ELF
executable at
`../misc/Graph_compiler_performance_estimator_test_mlia_2026_lts_r56_1`.

## Goals / Non-Goals

**Goals:**

- Establish an old-versus-new comparison workflow for all supported `.vgf` and
  `.tosa` models.
- Identify command-line, output-file, JSON, debug database, performance
  database, and value-shape differences introduced by the new estimator.
- Update MLIA so the 2026 LTS r56.1 estimator can be installed from the normal
  vendored artifact flow and used by the existing NX performance path.
- Preserve MLIA's standardized performance output contract for successful
  estimator runs.
- Add focused characterization/regression tests for every parser or statistics
  adaptation required by the observed differences.

**Non-Goals:**

- Replacing the ML SDK model converter integration.
- Adding support for model formats beyond the currently supported `.vgf` and
  `.tosa` paths.
- Treating raw cycle/performance value changes between estimator versions as a
  failure when the new estimator output is structurally valid.
- Committing the local `../misc` candidate executable as the final production
  artifact source unless it is repackaged into the agreed vendored format.

## Decisions

1. Compare outputs through MLIA-facing behavior first, then inspect raw files.

   The primary compatibility surface is MLIA's public performance output, not
   byte-for-byte equality of estimator internals. The comparison harness should
   run the same supported model set against the current vendored estimator and
   the candidate estimator, collect raw estimator files, and compare:
   `*_debug_database.dat`, `*_performance_database.dat`,
   `*_network_performance_summary.json`, `nx_performance_statistics.json`, and
   standardized MLIA output.

   Alternative considered: compare only raw estimator files. That would find
   structural differences but would miss whether MLIA already normalizes them or
   whether public output is actually affected.

2. Keep the existing artifact acquisition shape.

   The new estimator should continue to be represented by the existing
   `graph-compiler-performance-estimator` artifact key, `.sha256` metadata, and
   `VENDORED_ARTIFACTS_URLS`-driven download. The local candidate executable can
   be copied into a temporary backend location for comparison, but production
   integration should update the expected artifact name/hash once the official
   archive format is available.

   Alternative considered: add a second artifact key for the r56.1 estimator.
   That creates avoidable backend-selection complexity when the goal is a
   version replacement.

3. Make parsers tolerant only where the new output proves the old assumptions
   too narrow.

   Existing parser behavior should remain strict enough to detect malformed
   estimator output. Adaptations should target observed r56.1 differences such
   as renamed or missing JSON fields, additional memory sections, extra debug
   database tables, changed table headers, or changed output filenames. Tests
   should use compact fixtures derived from observed outputs rather than large
   full-model output files where possible.

   Alternative considered: loosen all parser validation to accept arbitrary
   missing data. That would make failures harder to diagnose and could silently
   degrade output quality.

4. Treat performance-value drift as expected unless it breaks MLIA semantics.

   New estimator versions can legitimately change cycle counts, memory traffic,
   compiled size, and inferred throughput. Regression checks should assert that
   required fields are present, typed, and mappable into standardized output,
   while comparison reports should summarize value differences for review.

   Alternative considered: lock expected numeric values for all models. That
   would make tests brittle across estimator revisions and host/toolchain
   changes.

## Risks / Trade-offs

- Candidate artifact format differs from final Artifactory artifact -> Keep the
  implementation path format-agnostic where practical and leave final `.sha256`
  update as a discrete task when the official archive is available.
- The new estimator may require different dynamic libraries or runtime
  environment variables -> Capture invocation failures separately from parser
  failures and document any required environment changes.
- Parser changes may accidentally hide invalid output -> Add tests for both
  supported r56.1 shapes and still-invalid malformed output.
- `.tosa` comparison also exercises model conversion -> Separate `.vgf` direct
  estimator failures from `.tosa` conversion-plus-estimator failures.

## Migration Plan

1. Build a temporary comparison setup that can run current and candidate
   estimators without permanently replacing the vendored artifact.
2. Run all supported `.vgf` and `.tosa` models through both versions and save
   raw outputs plus MLIA standardized outputs.
3. Classify differences into expected numeric drift, parser/statistics contract
   changes, invocation/config changes, and artifact metadata changes.
4. Add characterization tests for changed output shapes.
5. Update MLIA parser, statistics, invocation, config, and artifact metadata as
   required.
6. Validate with targeted NX estimator tests, then run the comparison again
   against the candidate estimator.

Rollback is to restore the previous `.sha256` artifact metadata and keep the
current parser/runtime behavior. Parser compatibility changes that remain
backward-compatible may not need rollback.

## Open Questions

- What Artifactory URL will provide the 2026 LTS r56.1 estimator artifact named
  by `.sha256`?
- Are any numeric performance deltas expected to be reviewed against acceptance
  thresholds, or is structural compatibility sufficient for this integration?
