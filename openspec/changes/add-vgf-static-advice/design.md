<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

# Design

## Context

See `proposal.md` for motivation and the reviewed branch scope. VGF modules
contain TOSA extended instructions, tensor types, constants, and debug metadata.
The estimator supplies placements separately. Existing standardized entities and
advice fields are sufficient; no schema extension is needed.

## Goals / Non-Goals

Goals:

- Share static analysis between compatibility and performance flows.
- Keep instruction visitation file-backed and retain only advice metadata.
- Preserve segment identities and source attribution through conversion.
- Treat shape-dependent recommendations as heuristics, not compatibility errors.

Non-goals:

- Rewrite models, adjust channels automatically, or predict a numerical speedup.
- Infer unknown dimensions, trace through arbitrary non-TOSA operations, or
  establish provenance when compiler labels cannot resolve a source entity.
- Fix whole-file allocations in GCPE preparation or the standalone debug reader.
- Add new dependencies, configuration flags, or generic fallback advice.

## Decisions

### Visit bounded SPIR-V buffers with a reusable callback

`for_each_vgf_op` maps the VGF read-only, obtains each segment's module view, and
copies one instruction at a time for a `(segment_index, instruction_bytes)`
callback. Owned bytes allow callback retention after the mapping closes. Keeping
the callback interface independent from advice rules permits other consumers.
Whole-file reads and unpacking the entire module into Python objects were rejected
for visitation because they unnecessarily amplify memory use.

The visitor checks headers and instruction bounds and propagates callback errors.
It includes compute segments and revisits shared modules per referencing segment;
rule selection and NX placement filtering remain separate responsibilities.

### Maintain segment-local metadata, not tensor payloads

`_SpirvSegmentState` stores imported TOSA sets, scalar constants, integer-array
shape types, tensor shapes and sizes, value metadata, and operand relationships.
`_SpirvId` aliases `int` for ID-indexed maps and sets without runtime conversions.

The parser inspects at most the first fifteen words for ordinary metadata and
convolution dispatch. Shape composites are expanded only after checking their
integer-array type and declared length. Tensor constant constituents must not be
mistaken for dimensions. Tensor sizes use rounded-up bit counts from known
positive shapes and scalar element widths.

### Keep rule-specific callbacks and operation-aware graph traversal

CONV2D and depthwise CONV2D have separate callbacks, dispatched only for their
matching TOSA opcode. The spec defines their thresholds; regular convolution
rejects any non-positive weight dimension, while depthwise advice requires a
known rank-four shape and input depth from one through three.

Skip analysis retains operand sets unchanged. It tests membership in the TOSA
result map at each traversal step rather than mutating those sets to remove
constant operands. For each candidate direct parent, a backwards search from the
other inputs determines whether an alternate path exists. A visited set bounds
repeated exploration. Only known outputs larger than 128 KiB produce advice.

### Carry attribution in records and resolve it before serialization

`NXAdviceRecord` carries ID, category, severity, message, and optional entity ID.
Convolution advice IDs use `advice/segment_<index>/spirv_<id>`; skip advice appends
`/skip_connection_<parent_id>` to the merge's advice ID. Initial entity IDs use
the existing canonical VGF source-operator format.

Performance and compatibility generation run on each prepared graph file because
conversion and partitioning can expose operations there. Skipped compute segments
are not analyzed for advice. Each generated file has local segment zero, so
`_add_segment_advice` rewrites advice IDs and VGF entity references into the
original segment namespace without mutating the gathered records. Performance
retains VGF entity IDs; compatibility additionally resolves TOSA source IDs below.

For converted TOSA compatibility, segment debug names resolve the local SPIR-V ID
to a prepared source location, with a decoded provenance-label fallback. An exact
source mapping produces a TOSA entity ID; otherwise attribution remains absent.
This avoids presenting a VGF ID that does not exist in the TOSA result.

### Filter using the finalized result entities

Both serializers collect IDs whose placement is exactly NX and emit only advice
linked to that set. Filtering at serialization uses final placement evidence and
excludes EE, mixed NX/EE, unresolved, and missing entities. Advice is categorized
as performance information even inside a compatibility result.

The target collector forwards backend advice instead of injecting a generic
suggestion. This makes empty advice meaningful and avoids maintaining two
independent advice-generation layers.
