<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## Compared Inputs

Current estimator:
`src/mlia/_vendor/artifacts/nx-performance-estimator/graph_compiler_performance_estimator_r55p0_00eac0_mlia_4.tar.gz`,
extracted to `/tmp/mlia-nx-compare-old/graph-compiler-performance-estimator`
for comparison.

Candidate estimator:
`../misc/Graph_compiler_performance_estimator_test_mlia_2026_lts_r56_1`.

Supported models:

- `../../models/baseline_int8.vgf`
- `../../models/mobileNetV3_large_batch_fixed_int8.vgf`
- `../../models/yolo_v3_tiny_darknet_fp32.vgf`
- `../../models/mobilenet_debug.tosa`
- `../../models/old-tosa-arm_nss_clampnet_v5_1-int8_qat-1080_1920-TOSA-1_00+INT.tosa`

The `.tosa` models were converted with the existing ML SDK model converter
backend before estimator comparison.

Raw comparison outputs were written under `/tmp/mlia-nx-compare/old` and
`/tmp/mlia-nx-compare/new`.

Derived MLIA comparison outputs were written under:

- `/tmp/mlia-nx-compare/old/mlia-derived`
- `/tmp/mlia-nx-compare/new/mlia-derived`

Each successful raw-output run has:

- `nx_performance_statistics.json`
- `standardized_output.json`

The old estimator did not produce raw outputs for
`yolo_v3_tiny_darknet_fp32.vgf` because it failed with unsupported `Conv2D`, so
there are no old-estimator derived MLIA outputs for that model.

## Observed Differences

Invocation and output files:

- Existing `-i <vgf> -o <name>` invocation works for r56.1.
- Required output files keep the current names:
  - `<name>_debug_database.dat`
  - `<name>_performance_database.dat`
  - `<name>_network_performance_summary.json`
- r56.1 also writes `<name>_timing_database.dat`; MLIA does not need this file
  for the current performance output path.

Database shape:

- r56.1 performance and debug databases remain parseable by the existing
  `NXPerformanceDatabaseParser` and `NXDebugDatabaseParser`.
- Required debug mappings are still present:
  `stripe_op_id_to_op_id`, `chain_op_id_to_fused_op_ids`,
  `fused_op_id_to_tosa_op_ids`, `tosa_op_id_to_api_labels`, and
  `tosa_op_id_to_tosa_op`.
- r56.1 includes additional debug tables such as
  `command_streams_id_to_stripe_op_indexs`, `stripe_op_id_to_cascade_op_id`,
  and `shader_op_id_to_api_labels`; these do not affect current MLIA mapping.

Network summary JSON:

- The JSON schema remains compatible for normal successful models.
- For `yolo_v3_tiny_darknet_fp32.vgf`, r55 fails with unsupported `Conv2D`,
  while r56.1 succeeds and emits zero-cycle performance with
  `infs_per_sec.value` set to `null`.
- MLIA must normalize the `null` inference rate to a numeric metric value before
  producing standardized output.
- The normalized r56.1 standardized output for `yolo_v3_tiny_darknet_fp32.vgf`
  contains 12 numeric model metrics and no operator breakdowns because the
  r56.1 performance database contains no performance rows.

Numeric drift:

- Cycle counts, memory traffic, compiled size, inference time, and throughput
  differ between estimator versions as expected.
- These value changes are treated as estimator-version drift rather than MLIA
  failures when the output remains structurally valid.

Artifact replacement:

- The provided r56.1 executable was repackaged locally as
  `graph_compiler_performance_estimator_2026_lts_r56_1_mlia.tar.gz`.
- The archive extracts to `graph-compiler-performance-estimator`, preserving the
  backend installation check.
- `.sha256` now names that archive and hash so `hatch_build.py` continues to use
  the existing `graph-compiler-performance-estimator` artifact key.
