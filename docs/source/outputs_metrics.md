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

Model-level metrics are stored in `results[].metrics`. They summarize the complete
model rather than any individual layer or scheduling entity.

Estimator output includes backend-native metrics and standardized MLIA metrics.
When the backend cannot provide a standardized metric, the result still
contains an unavailable entry with a reason.

| Metric | Unit | Meaning |
| --- | --- | --- |
| `total_cycles` | cycles | Estimated end-to-end execution cycles. |
| `compute_cycles` | cycles | Estimated cycles spent performing NX compute. |
| `cache_cycles` | cycles | Estimated cycles attributed to traffic on the L1/cache path. |
| `dram_cycles` | cycles | Estimated cycles attributed to DRAM traffic. |
| `inference_time` | ms | Estimated time for one inference. |
| `infs_per_sec` | inferences/s | Estimator-native throughput field. |
| `inferences_per_second` | inferences/s | Standardized MLIA throughput, derived as `1000 / inference_time`. |
| `target_utilization` | % | Compute utilization, calculated as `compute_cycles / total_cycles * 100`. |
| `compiled_size` | bytes | Size reported for the compiled network. |
| `cache_read_bytes` | bytes | Estimated data traffic read through the L1/cache path. |
| `cache_write_bytes` | bytes | Estimated data traffic written through the L1/cache path. |
| `dram_read_bytes` | bytes | Estimated data traffic read from DRAM. |
| `dram_write_bytes` | bytes | Estimated data traffic written to DRAM. |
| `dram_footprint` | bytes | Estimated DRAM storage required by the network. |
| `accelerator_operator_percentage` | % | Percentage of model operators placed on the accelerator, when placement data is available. |
| `cpu_utilization` | % | CPU utilization, when the backend provides it. |
| `model_weight_memory` | bytes | Standardized model-weight memory metric. The NX estimator currently reports it as unavailable. |
| `peak_activation_memory` | bytes | Highest activation-memory usage during one inference, when the backend provides it. |
| `average_memory` | bytes | Average memory usage over the measurement window, when the backend provides it. |

The estimator calls the L1/cache fields `cache1` in its network performance
summary. MLIA exposes the model totals with the `cache_` prefix.

When the backend does not provide a model-level value, the metric is represented
as unavailable with a reason rather than with a fabricated number. Measured
captures therefore leave fields such as compiled size and inference time
unavailable when the capture does not contain them.

## Layer breakdown metrics

Layer breakdown metrics are stored in `results[].breakdowns[].metrics`. Each
breakdown is linked through `entity_id` to an entry in `results[].entities`.

The backend attaches its authoritative breakdowns to Graph Compiler chain and
cascade entities. Core MLIA may project those breakdowns to related entities,
such as source operators and provenance entities. It may also produce a segment
breakdown when the segment has complete, compatible coverage. Therefore, a
breakdown does not always correspond one-to-one with a framework layer. Use the
linked entity and its parent or child relationships to identify the source
operator or operators represented by the breakdown.

Layer breakdowns can include:

| Metrics | Unit | Meaning |
| --- | --- | --- |
| `op_cycles` | cycles | Sum of the estimator-provided `opCycles` values for the underlying stripes. This value is not necessarily the largest of all reported hardware-section cycle counts. |
| `total_cycles` | cycles | Sum of the total estimated cycles for the underlying stripes. |
| `l1_read_bytes`, `l1_write_bytes` | bytes | Estimated data traffic through the L1 path. |
| `l2_read_bytes`, `l2_write_bytes` | bytes | Estimated data traffic through the L2 path. |
| `system_cache_read_bytes`, `system_cache_write_bytes` | bytes | Estimated data traffic through the system-cache path. |
| `dram_read_bytes`, `dram_write_bytes` | bytes | Estimated data traffic through the DRAM path. |
| `l1_traffic_cycles`, `l2_traffic_cycles`, `system_cache_traffic_cycles`, `dram_traffic_cycles` | cycles | Estimated cycles attributed to traffic on each memory path. |
| `input_reader_cycles` | cycles | Estimated cycles for the input-reader section. |
| `convolution_engine_cycles` | cycles | Estimated cycles for the convolution-engine section. |
| `vector_engine_cycles` | cycles | Estimated cycles for the vector-engine section. |
| `transform_unit_cycles` | cycles | Estimated cycles for the transform-unit section. |
| `weight_decoder_cycles` | cycles | Estimated cycles for the weight-decoder section. |
| `output_writer_cycles` | cycles | Estimated cycles for the output-writer section. |

Memory and hardware-section names are normalized to lowercase snake case. For
example, `SystemCache` becomes the `system_cache_` prefix. In estimator mode,
breakdown values are estimates from the performance model rather than hardware
measurements. In measured profiling mode, breakdown values are derived from
captured hardware counters and are hardware measurements.

Do not add breakdowns from related entities together. Authoritative and
projected breakdowns at different hierarchy levels can describe overlapping
views of the same work.

### Interpreting L1/cache write bytes

At model level, `cache_write_bytes` is the total estimated traffic passing
through the L1 path. At layer-breakdown level, the corresponding metric is
`l1_write_bytes`.

These metrics do **not** mean that the compiler allocated data in L1 or
generated an output that is stored there. The performance model counts the
following traffic as L1 writes:

- writes whose destination is L2, because those writes pass through L1; and
- partial writes whose destination is DRAM, because those writes also pass
  through L1.

Consequently, an L1/cache write metric can be nonzero even when no output is
stored in L1. Memory-level byte counters describe traffic through a path, not
allocation or residency at that memory level.

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

Estimator results contain performance estimates and omit the optional `mode`
field. Measured profiling results set `mode` to `measured` and record capture
device, statistics files, debug databases, and selected dispatches in backend
and runtime metadata.

Both modes share per-segment correlation, totals, warnings, entity provenance,
and breakdown aggregation. Compute-only VGF segments that are excluded from NX
analysis are reported through warnings.

## Generated artifacts

Depending on the input and mode, the output directory can contain:

- framework-conversion and ML SDK converter artifacts;
- NX Performance Estimator-compatible VGF segment files;
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
