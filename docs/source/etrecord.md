---
# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
sidebar_position: 6
---

# ETRecord and ETDump Profiling

Use an ETRecord and its matching ETDump to analyze recorded ExecuTorch runtime
performance. For Neural Technology delegates, MLIA also decodes VGF profiling
into NX cycle measurements and a segment, cascade, chain and stripe hierarchy.
This workflow analyzes measured data; it does not estimate performance or run
compatibility analysis. For capture directories, see [Backends](backends.md).

## Install and run

From this repository checkout, install the optional ExecuTorch support:

```bash
uv sync --extra executorch
```

Supply these artifacts from the same deployed program:

- An ETRecord saved as `.etrecord` or `.bin`, containing the recorded model and
  its debug mappings.
- Exactly one ETDump saved as `.etdump` or `.etdp`, containing runtime events
  and, when collected, VGF profiling data.
- Optionally, the matching deployed `.pte` for companion graph associations.
  Companion operator associations currently use VGF profiling data; runtime-only
  observations do not provide these associations.

Run measured performance analysis:

```bash
uv run --extra executorch mlia check model.etrecord \
  --target-profile neural-technology \
  --performance \
  --profiling-data run.etdump \
  --output-dir mlia-etrecord-result
```

`--profiling-data` selects the measured backend automatically; do not specify
it with `--backend`. The standardized result is written to
`mlia-etrecord-result/mlia-output.json`. See
[Outputs and metrics](outputs_metrics.md) for the result format and metric
definitions.

For an ETRecord containing multiple methods, add the method used to collect the
dump, for example:

```text
--neural-technology-profiling-data.etrecord-method decode
```

Single-method records are selected automatically.

## View results in Model Explorer

Add the visualization plugin and its adapters to the command environment:

```bash
uv run --extra executorch --with "mlia-model-explorer[executorch]" \
  mlia check model.etrecord \
  --target-profile neural-technology \
  --performance \
  --profiling-data run.etdump \
  --output-dir mlia-etrecord-result \
  --companion-model model.pte \
  --model-explorer open
```

`open` generates a standalone visualization under the output directory and
opens it in the browser. The command reports the generated page's location.
Use `--model-explorer files` to generate it without opening a browser. Omit
`--companion-model` if no companion graph is needed.

Available views include source operators with runtime measurements, the NX
hierarchy and VGF backend operations. Supplying a companion PTE adds its VGF
graph and links to associated operations. Chain and cascade groups display VGF
cycle measurements; links to individual compiler or source operators do not
imply independent timing measurements for those operators.

Source-operator links may be incomplete or unavailable, and a fused chain may
not link every operation it contains. Compilation can rename, insert, combine
or reorder operations, leaving insufficient information to identify the
original source operator reliably. Such operations may have no source link,
even when the model contains an operation of the same type. Older captures
with limited source information may also show incorrect links.

Missing source links do not mean that profiling is absent: measurements remain
available on the recorded chains and cascades. Source placement and colouring
do not establish an individual operator's execution cost.

## Read the measurements

### VGF cycles and counters

VGF profiling provides cycle measurements and supported hardware counters for
segments, cascades, chains and stripes. `total_cycles` and `op_cycles` are
separate measurements; model cycle totals use decoded total cycles. Repeated
executions provide averages, execution counts and total-cycle minimum, maximum
and median values.

These cycles are derived from activity counters, not elapsed-time measurements.
Mode 0 sums section-active cycles across tasks. Mode 1 sums counters per core
and takes the largest core total for each counter; its `total_cycles` and
`op_cycles` both use the resulting `nx_active` value. Counters cannot reconstruct
idle gaps or overlap between stripes. See
[measured counter aggregation](outputs_metrics.md#aggregating-mode-0-neural-statistics)
before comparing them with estimator predictions or elapsed latency.

Measurements are summed within each recorded execution before those executions
are summarized. Averages can be combined across entities only when they cover
the same recorded executions. Minimum, maximum, median and execution counts are
not additive. When VGF instructions or segments have different execution
coverage, model cycle summaries and execution count are unavailable with a
warning; individual measurements remain available.

Matching instruction and segment coverage does not prove that every expected
stripe was captured. MLIA cannot distinguish a deliberately absent stripe from
a missing measurement using the available data. Different stripe lists are
therefore accepted, and totals describe the recorded observations. For example,
executions containing 400 and 100 observed cycles produce an average of 250
cycles even if the second execution lost a stripe. MLIA does not estimate
missing timings. Check capture completeness before treating these totals as
full-inference costs.

### Runtime timings

Ordinary runtime timings are reported in milliseconds as `Runtime Total Avg`,
`Runtime Total P50`, `Runtime Total Min` and `Runtime Total Max`.
`Runtime Event Count` counts contributing event records, not inference samples.
Independent events associated with one source operator are summed per execution
before calculating summaries. Group averages require matching execution
coverage. P50, minimum and maximum cannot be added across operators.

An event may cover several operators. Each mapped operator retains the full
shared timing, with non-additive averages and event counts. The timing must not
be interpreted as an independent cost for each operator. If some debug handles
cannot be mapped, the mapped operators still retain the non-additive timing;
`runtime_unresolved_debug_handles` lists the unresolved handles. Summaries that
mix shared and independent observations remain non-additive.
Events with no mapped source operators do not produce per-operator timings.

Detailed delegate timings take precedence over an enclosing delegate-call
interval for the same instruction, backend and executions. The full enclosing
interval remains a fallback for operators without mapped details. When it spans
multiple operators or unresolved handles, it is non-additive. It is not divided
among operators or added to covered operators' detailed timings.
Detailed timings may exclude delegate overhead and unprofiled work.

Mixed dumps keep ordinary runtime timings and VGF cycle measurements separate.
Runtime events for instructions with VGF profiling are excluded from the
ordinary-runtime result to avoid duplicate accounting. Source timings do not
establish end-to-end latency or model cycle totals.

## Supported decoder layouts

VGF decoding supports statistics modes 0 and 1 for Mali-G1 and Mali-G2. Supported
layouts use little-endian 32-bit unsigned words and one leading block:

| Device | Mode 0 words per task | Mode 1 words per task |
| --- | --- | --- |
| Mali-G1 | 16 | 32 |
| Mali-G2 | 16 | 64 |

Schema-v2 VGF metadata supplies the statistics mode and, when available, device
and counter-layout information. The supported declared counter-layout schema
version is 1. MLIA uses recorded settings and rejects unsupported layouts or
conflicting settings. A declared Mali-G2 mode-0 layout with 64 words per task is
not supported by the current 16-word mode-0 decoder.

Legacy schema-v1 VGF metadata does not record these settings. The CLI defaults
to mode 1 and Mali-G2 when settings are unavailable. Python callers can supply
`statistics_mode` and `profiling_device` as fallbacks. The output records the
effective settings and `decoder_fallback_fields`; warnings identify assumptions
that could not be checked against the capture. Selecting a target profile does
not change these defaults. Successful parsing alone does not verify that an
assumed layout matches the capture.

## Companion matching and Python results

`--companion-model model.pte` checks available VGF operator associations for
structural compatibility using the selected method, deployed instruction, VGF
segment and SPIR-V result IDs. It assumes you supplied the matching PTE. These
identifiers can also occur in another compiled model, so matching does not prove
that the PTE produced the capture. Operations without the required profiling
relationships have no companion operator association.

The result records a SHA-256 digest of the PTE bytes analyzed. It is not compared
with a producer-recorded identity in the capture and does not verify capture
provenance. Before generating the companion visualization, Model Explorer
compares the file's current digest with the recorded digest and rejects a
mismatch. Analysis does not monitor later file changes.

Companion associations and profiling results are also available without Model
Explorer through the Python API:

```python
from mlia import run_advisor

output = run_advisor(
    "performance", "neural-technology", "model.etrecord",
    profiling_data="run.etdump", companion_model="model.pte",
)
companion = output["extensions"]["executorch"]["companion_model"]
```

Omit `companion_model` when it is not needed. Supplying it does not enable
visualization. VGF delegate profiling is supported; Ethos-U-specific profiling
is not currently implemented in this workflow.

## Troubleshooting

- For an optional-package import error, install the named extra or adapter in
  the environment running the command.
- For missing method or graph mappings, check that the ETRecord, ETDump and PTE
  belong to the same deployed program and selected method. VGF events without
  matching model nodes fail analysis.
- Missing VGF debug databases, statistics information or statistics memory;
  empty or unrecognized VGF payloads; and malformed or misaligned samples fail
  analysis. These failures do not silently fall back to ordinary runtime timing.
- For decoder errors, check the recorded layout and any fallback assumptions
  against the capture settings.
- For companion mismatches, check the supplied PTE and method. If the file has
  changed since analysis, run analysis again. An unavailable delegate graph or
  unsupported adapter produces an error identifying the dependency or graph.

For capture-directory issues, see [Troubleshooting](troubleshooting.md).
