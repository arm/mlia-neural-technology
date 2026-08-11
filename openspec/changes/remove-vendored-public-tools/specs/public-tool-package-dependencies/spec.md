<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## ADDED Requirements

### Requirement: Public converter tools are package dependencies
MLIA plugin packages SHALL declare public Python package dependencies for tools
that are available from public package indexes.

#### Scenario: Neural Technology dependencies are installed
- **WHEN** `mlia-neural-technology` is installed
- **THEN** the `ai-ml-sdk-model-converter` and `tosa-tools` distributions are
  available through normal package dependency resolution

#### Scenario: TFLite converter dependency is installed
- **WHEN** `mlia-converters-litert` is installed
- **THEN** the `tosa-converter-for-tflite` distribution is available through
  normal package dependency resolution

### Requirement: Public tools are not vendored artifacts
MLIA plugin packages SHALL NOT download, force-include, or install private
vendored artifacts for the public model converter, TOSA FlatBuffers, or TFLite
converter packages.

#### Scenario: Building Neural Technology
- **WHEN** `mlia-neural-technology` is built
- **THEN** the build hook only resolves the vendored NX performance estimator
  artifact and does not require model-converter or TOSA FlatBuffers artifact
  URLs

#### Scenario: Building converter packages
- **WHEN** `mlia-converters-litert` or `mlia-converters-pytorch` is built
- **THEN** the wheel and sdist do not include vendored public tool wheels

### Requirement: Backend installs use public packages
MLIA backend installation metadata SHALL install public-package tools through
the active Python package manager instead of `InstallFromVendorPackage`.

#### Scenario: TFLite converter is missing at runtime
- **WHEN** neither the `tosa-converter-for-tflite` executable nor module can be
  resolved
- **THEN** MLIA installs the public `tosa-converter-for-tflite` package and
  retries executable/module resolution

#### Scenario: NX dependency installation remains supported
- **WHEN** the NX performance estimator backend is installed with dependencies
- **THEN** its public-package dependencies install through package metadata and
  the NX estimator itself remains installable from the vendored estimator
  artifact
