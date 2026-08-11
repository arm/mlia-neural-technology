<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## Overview

The affected packages should treat public tool distributions as normal Python
dependencies and reserve vendored artifact handling for tools that are still not
publicly consumable. The change removes private artifact download and wheel/tar
inclusion for `ml-sdk-model-converter`, `tosa-flatbuffers`, and
`tosa-converter-for-tflite`, while preserving the existing vendored NX
performance estimator path.

## Dependency Strategy

- `mlia-neural-technology` declares:
  - `ai-ml-sdk-model-converter` for the `model-converter` executable.
  - `tosa-tools` for the importable TOSA FlatBuffers package.
- `mlia-converters-litert` declares `tosa-converter-for-tflite`.

Public tool versions are pinned exactly so MLIA uses a known-compatible tool
set rather than silently mixing converter and compiler releases.

`mlia-converters-pytorch` is intentionally left out of the implementation slice:
the currently available public `tosa-tools` releases require a `flatbuffers`
version that conflicts with `ethos-u-vela==5.0.0`, so moving that repository to
the public package would make its dependency set unsatisfiable.

## Runtime Installation

Backends that wrap public Python packages should use
`PyPackageBackendInstallation` with `packages_to_install` populated and no
`vendor_path`. This preserves MLIA's backend install/uninstall semantics while
delegating package acquisition to the active package manager.

The TFLite converter still prefers an existing executable, then a Python module
entrypoint. If neither is available, it installs the public package through the
package manager and re-checks executable/module availability.

The ML SDK model converter remains registered as a backend because the NX
performance estimator reads its backend path from MLIA's backend repository. Its
installer records the installed package's executable path rather than a vendored
extracted tarball.

## Build Packaging

Build hooks should no longer download or force-include removed public artifacts.
`mlia-neural-technology/hatch_build.py` should keep only the
`graph-compiler-performance-estimator` artifact. The converter packages should
drop custom vendored artifact hooks when no other build-time artifact remains.

Source distributions should stop including `.sha256` sidecars for removed
artifacts.

## Validation

Targeted tests should cover:

- Backend registry metadata no longer reports support for
  `InstallFromVendorPackage` for public-package tools.
- Public package names are present in install metadata.
- TFLite converter auto-install uses a normal package installation request.
- NX performance estimator still supports vendored installation and still
  depends on the model converter and TOSA FlatBuffers backends.

## `raw_all.tflite` Investigation

The public toolchain rejects `/home/maksve01/models/raw_all.tflite` after the
TFLite converter generates MLIR bytecode because the generated module still
contains TensorFlow Lite dialect operations, specifically `tfl.greater`.
`ai-ml-sdk-model-converter==0.9.0` only registers the dialects it consumes for
TOSA/VGF lowering, so it rejects both bytecode and textual MLIR containing
unlowered `tfl` operations.

The cached vendored `tosa_converter_for_tflite-2025.11.0.dev0` and
`ml-sdk-model-converter-backend-1.00` artifacts were compared directly. They
are compatible with each other for representative models such as
`baseline_int8.tflite`, but `raw_all.tflite` is rejected there as well because
the old converter also leaves `tfl.greater` in the generated MLIR. The public
toolchain is therefore exposing a real model/operator lowering gap, not a
bytecode container issue. The compatibility-preserving fix belongs in
`tosa-converter-for-tflite` support for this TFLite op pattern, or in a clearer
MLIA diagnostic before invoking model-converter on non-pure TOSA MLIR.
