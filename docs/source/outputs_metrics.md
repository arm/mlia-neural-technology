<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Outputs and Metrics

## Overview

Neural Technology workflows produce standardized MLIA results from either:

- model-based estimation with `nx-performance-estimator`; or
- measured structured captures with `neural-technology-profiling-data`.

Both modes use the same metric, breakdown, entity, and provenance structure.
Conversion and estimator artifacts may also be written to the run output
directory for diagnostics.

## Standardized result shape

Metrics are objects in a list rather than a name-to-value mapping. Detailed
figures are stored in breakdowns linked to result-local entities:

```json
{
  "backends": [
    {"id": "nx-performance-estimator"}
  ],
  "results": [
    {
      "kind": "performance",
      "producer": "nx-performance-estimator",
      "metrics": [
        {"name": "total_cycles", "value": 456789, "unit": "cycles"},
        {"name": "inference_time", "value": 1.23, "unit": "ms"},
        {
          "name": "inferences_per_second",
          "value": 813.01,
          "unit": "inferences/s"
        }
      ],
      "entities": [
        {
          "id": "chain/segment_0/1",
          "kind": "chain",
          "name": "segment_0/Chain 1",
          "child_ids": ["source_operator/segment_0/spirv-17"],
          "placement": "NX"
        },
        {
          "id": "source_operator/segment_0/spirv-17",
          "kind": "source_operator",
          "name": "conv2d",
          "parent_ids": ["chain/segment_0/1"],
          "placement": "NX"
        }
      ],
      "breakdowns": [
        {
          "entity_id": "chain/segment_0/1",
          "metrics": [
            {
              "name": "op_cycles",
              "value": 400000,
              "unit": "cycles",
              "aggregation": "sum"
            },
            {
              "name": "internal_read_bytes",
              "value": 8192,
              "unit": "bytes",
              "aggregation": "sum"
            }
          ]
        }
      ]
    }
  ]
}
```

The complete output also contains schema, run, target, model, context, and tool
metadata owned by the shared MLIA schema.

## Model-level metrics

Estimator output can include:

- `total_cycles`
- `compute_cycles`
- `cache_cycles`
- `dram_cycles`
- `inference_time`
- `infs_per_sec`, preserving the backend field
- `inferences_per_second`, the standardized throughput metric
- `target_utilization`
- `compiled_size`
- cache and DRAM read/write byte counts
- `dram_footprint`

When the backend does not provide a model-level value, the metric is represented
as unavailable with a reason rather than with a fabricated number. Measured
captures therefore leave fields such as compiled size and inference time
unavailable when the capture does not contain them.

## Entity provenance

NX results describe several related views of the analyzed work:

- `segment` identifies a graph segment from the original VGF sequence when a
  VGF model is supplied. Capture-only profiling instead synthesizes segment `0`
  to represent the selected captured pipeline.
- `cascade` and `chain` identify estimator scheduling groups.
- `performance_group` connects an aggregate estimator record to multiple source
  operators when it cannot be attributed to one operation.
- `source_operator` identifies a canonical source operation.
- `nn_module` and `code_stack` preserve framework-module and source-code
  provenance when debug information provides it.

Core MLIA normalizes and validates these relationships. It may derive source-line
entities and project missing breakdowns to related views without replacing an
existing authoritative breakdown.

## Breakdown aggregation

NX cycle, byte, traffic, and hardware-section counters use
`aggregation: "sum"`. This declares that compatible breakdowns from distinct
accounting origins may be combined when core projects figures across the entity
graph. Existing breakdowns remain authoritative.

## Estimator versus measured mode

Estimator results contain performance estimates and currently omit the optional
`mode` field. Measured profiling results set `mode` to `measured` and record capture-device,
statistics-file, debug-database, and selected-dispatch information in backend and
runtime metadata.

Both modes share per-segment correlation, totals, warnings, entity provenance,
and breakdown aggregation. Compute-only VGF segments that are excluded from NX
analysis are reported through warnings.

## Generated artifacts

Depending on the input and mode, the output directory can contain:

- framework-conversion and ML SDK converter artifacts;
- GCPE-compatible VGF segment files;
- estimator performance and debug databases;
- `nx_performance_statistics.json`; and
- an effective model copied from a capture-only profiling run.

Use these artifacts to determine whether a failure occurred during conversion,
segment preparation, estimator execution, capture ingestion, or standardized
result construction.

## Cross-links

- See [Backends and analysis modes](backends.md) for backend ownership.
- See [CLI](cli.md) for commands that produce estimator and measured results.
- See [Troubleshooting](troubleshooting.md) for conversion, estimator, and
  profiling-data diagnostics.
