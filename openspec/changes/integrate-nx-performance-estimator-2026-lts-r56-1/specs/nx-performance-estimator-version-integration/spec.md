<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## ADDED Requirements

### Requirement: Supported models are compared across estimator versions
MLIA SHALL provide a repeatable integration workflow for comparing the current
NX performance estimator with a candidate replacement estimator using supported
`.vgf` and `.tosa` models.

#### Scenario: Direct VGF comparison
- **WHEN** a supported `.vgf` model is included in the comparison set
- **THEN** MLIA runs both the current and candidate NX performance estimators for
  that model and records raw estimator outputs and MLIA standardized outputs for
  comparison

#### Scenario: TOSA comparison through conversion flow
- **WHEN** a supported `.tosa` model is included in the comparison set
- **THEN** MLIA runs the existing conversion flow before estimator execution and
  records both current and candidate estimator outputs for comparison

### Requirement: Candidate estimator output is normalized into MLIA output
MLIA SHALL parse candidate NX performance estimator outputs and produce the
standardized MLIA performance output for successful supported model runs.

#### Scenario: Candidate output contains compatible data
- **WHEN** the candidate estimator writes debug database, performance database,
  and network performance summary data with all required semantic fields
- **THEN** MLIA maps the data into model-level metrics and operator breakdowns in
  the standardized performance output

#### Scenario: Candidate output uses a supported changed shape
- **WHEN** the candidate estimator output differs from the previous estimator in
  an observed and supported file, table, or JSON shape
- **THEN** MLIA normalizes that shape without dropping required standardized
  performance metrics

#### Scenario: Candidate output uses SPIR-V id location labels
- **WHEN** the candidate estimator emits operator labels of the form
  `TOSA*_spirv_id_*` and the analyzed VGF contains MLGraph debug location
  metadata
- **THEN** MLIA maps those SPIR-V ids to the original VGF locations before
  producing raw and standardized performance breakdowns

### Requirement: Invalid estimator output remains diagnosable
MLIA SHALL report clear failures when required candidate estimator outputs are
missing, malformed, or semantically insufficient.

#### Scenario: Required output file is missing
- **WHEN** estimator execution completes without one of the required output files
- **THEN** MLIA raises an error that identifies the missing expected file

#### Scenario: Required performance data is absent
- **WHEN** estimator output lacks data required for model-level metrics or
  operator breakdown mapping
- **THEN** MLIA raises a parser or statistics error that identifies the missing
  required data instead of producing incomplete standardized output

### Requirement: Artifact replacement follows existing vendored flow
MLIA SHALL integrate the new NX performance estimator version through the
existing vendored artifact download and verification mechanism.

#### Scenario: Official artifact metadata is available
- **WHEN** the official 2026 LTS r56.1 estimator artifact name, URL, and SHA-256
  are available
- **THEN** MLIA uses the existing `graph-compiler-performance-estimator`
  artifact key, `.sha256` metadata, and build hook verification path to obtain
  the new estimator

#### Scenario: Local candidate executable is used before official packaging
- **WHEN** validating with the local candidate executable before official
  packaging is available
- **THEN** MLIA treats the candidate as a temporary comparison input and does not
  require a permanent second backend registration
