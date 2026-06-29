<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## 1. Dependency Metadata

- [x] 1.1 Add public package dependencies to `mlia-neural-technology` and
  `mlia-converters-tflite`.
- [x] 1.2 Remove build hook artifact wiring for public packages while retaining
  the NX performance estimator vendor artifact.
- [x] 1.3 Remove wheel/sdist inclusion rules and vendor sidecars for removed
  public artifacts.

## 2. Runtime Installation

- [x] 2.1 Update Neural Technology model converter and TOSA FlatBuffers install
  metadata to use public package installs without `vendor_path`.
- [x] 2.2 Update the TFLite converter auto-install path to install the public
  package instead of `InstallFromVendorPackage`.
- [x] 2.3 Confirm PyTorch converter dependency preparation remains unchanged
  because public `tosa-tools` currently conflicts with `ethos-u-vela==5.0.0`.
- [x] 2.4 Keep NX performance estimator dependency declarations and vendored
  artifact support intact.

## 3. Tests

- [x] 3.1 Update backend installation tests to assert public-package install
  metadata and lack of vendor support.
- [x] 3.2 Update converter auto-install tests for public package install
  behavior.
- [x] 3.3 Add or adjust build-hook/package tests for removed artifact wiring if
  local tests cover it.

## 4. Validation

- [x] 4.1 Run targeted TFLite converter tests.
- [x] 4.2 Run targeted PyTorch converter dependency resolution check.
- [x] 4.3 Run targeted Neural Technology backend installation/API tests.
- [x] 4.4 Run package resolution or build validation where network and package
  availability permit.
