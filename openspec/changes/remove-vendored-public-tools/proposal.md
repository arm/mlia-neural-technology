<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## Why

The Neural Technology flow and converter plugins still carry private vendored
artifacts for tools that now have public package distributions. This keeps build
hooks tied to Artifactory-only inputs, forces runtime installation through
`InstallFromVendorPackage`, and makes converter package builds depend on wheel
files that should be resolved by normal Python package metadata.

## What Changes

- Use the public `ai-ml-sdk-model-converter` package for the ML SDK model
  converter backend instead of the vendored `ml-sdk-model-converter` tarball.
- Use the public `tosa-tools` distribution for TOSA FlatBuffers support instead
  of the vendored `tosa-flatbuffers` wheel.
- Use the public `tosa-converter-for-tflite` distribution in
  `mlia-converters-tflite` instead of the vendored converter wheel.
- Keep the NX performance estimator vendored through the existing artifact flow.
- Leave the PyTorch converter vendored `tosa-tools` artifact unchanged for this
  change because the current public `tosa-tools` release conflicts with
  `ethos-u-vela==5.0.0` over the `flatbuffers` dependency.
- Update installation metadata, build hooks, package metadata, and tests so the
  affected tools install through public package requirements.

## Capabilities

### New Capabilities

- `public-tool-package-dependencies`: Defines how MLIA plugin packages depend on
  public tool distributions rather than private vendored artifacts.

### Modified Capabilities

None.

## Impact

- `mlia-neural-technology/pyproject.toml` and `hatch_build.py`: public package
  dependencies and remaining vendored artifact build hook scope.
- `mlia-neural-technology/src/mlia/backend/ml_sdk_model_converter/`: install
  metadata and executable resolution for the public model converter package.
- `mlia-neural-technology/src/mlia/backend/tosa_flatbuffers/`: install
  metadata for the public TOSA tools package.
- `mlia-converters-tflite/pyproject.toml`, `hatch_build.py`, and
  `src/mlia/backend/tosa_converter_for_tflite/install.py`: public dependency
  and auto-install behavior.
- Vendor artifact directories and pre-commit exclusions for the removed
  artifacts.
- Targeted backend installation and converter runtime tests.
