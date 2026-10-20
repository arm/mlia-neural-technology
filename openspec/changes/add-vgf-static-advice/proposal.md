<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

# Proposal

## Why

The generic suggestion to inspect performance metrics does not identify model
changes that could improve NX utilization. Static analysis of VGF-embedded TOSA
operations can provide actionable, entity-linked advice in both performance and
compatibility reports.

## What Changes

- Add a memory-mapped VGF instruction visitor and segment-local metadata tracking
  without retaining tensor constant payloads as shape data.
- Recommend input/output depth alignment for CONV2D according to kernel area.
- Recommend considering regular CONV2D for depthwise input depths below four.
- Detect direct skip edges with an alternate operation path and a known tensor
  size greater than 128 KiB; explain their potential poor cache utilization.
- Gather performance and compatibility advice from every prepared graph segment.
- Preserve segment-scoped advice identities and translate converted TOSA advice
  through debug labels to the report's source entity IDs.
- Serialize advice only for existing entities placed exclusively on NX.
- Remove the unconditional `performance_metrics` advice entry from target data
  collection; forward backend-produced advice instead. Reports can have no advice.

## Capabilities

### New Capabilities

- `nx-static-model-advice`: Derive static optimization advice from VGF operations
  and publish it with valid NX entity attribution.

### Modified Capabilities

None. There are no published specs under `openspec/specs`; the existing TOSA
preparation change remains separate.

## Impact

- `src/mlia/backend/nx_performance_estimator/advice.py` owns the advice rules.
- `vgf.py` adds instruction visitation; `compat.py` and `performance.py` integrate
  generation, attribution, and standardized serialization.
- Neural Technology data collection stops injecting generic advice.
- Advice, compatibility, performance, and Python API tests cover the new behavior.
- No new dependencies, CLI flags, or output schema fields are required.
