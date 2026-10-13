<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## Context

Compatibility is derived from GCPE debug databases, not converter lowering logs.
The converter prepares a VGF for TOSA inputs, while direct VGF inputs bypass it.
The compatibility module therefore belongs to the NX Performance Estimator
backend. This design captures the TOSA implementation and corrects accounting
for estimator operations without usable labels.

Compiler expansion can produce several records for one TOSA operation. Emitting
each as a separate converted entity duplicates entity IDs and overweights that
operation in compatibility metrics. Consolidate evidence before reporting.

## Goals / Non-Goals

**Goals:**

- Give every parsed TOSA operation a unique label in the compiler input.
- Keep source files unchanged and derive attribution from the processed model.
- Support the existing FlatBuffer and textual MLIR input formats.
- Handle converted models with multiple graph segments.
- Separate internal attribution keys from user-facing display labels.
- Report only the backends actually used by the compatibility path.

**Non-Goals:**

- Splitting TOSA into one-operation models or removing unsupported operations.
- Implementing a general MLIR parser or supporting non-numeric SSA identifiers.
- Changing VGF structural source IDs or performance metric calculation.
- Restoring incomplete-coverage status, warnings, or metric suppression.

## Decisions

### Preparation owns synthetic labels

`_prepare_tosa_model()` reads the original model and invokes the separate
`_assign_synthetic_labels()` helper. Counters start at zero for each prefix in
operation iteration order. The prefix is the original location, or `unknown`
when that location is empty. An existing literal `unknown` label shares the
same counter as empty locations, avoiding collisions.

For example, repeated `layer` locations become `layer_0` and `layer_1`; empty
locations become `unknown_0` and `unknown_1`. An original `layer_42` becomes
`layer_42_0`, not `layer_0`.

FlatBuffers are loaded with `tosa-tools` serialization support, preserving the
input schema version and assigning labels in reader traversal order. MLIR
operation locations are replaced or inserted in a text copy. Both formats are
written as `<stem>_processed<suffix>` in the run output directory through the
existing filesystem boundary.

Readers only parse supplied locations. They leave missing locations empty and
do not assign counters or modify files. The MLIR reader also resolves inline
locations so it can read the annotated model.

The operation-line matcher consumes quoted location strings, including escapes,
as units. Parentheses inside those strings are label characters, not the closing
delimiter of `loc(...)`. Preserve capture group 4 as the complete location
content so preparation can replace it without truncation. For example, both
`loc(#loc1)` with `#loc1 = loc("conv(foo)")` and inline `loc("conv(foo)")`
round-trip through preparation as `loc("conv(foo)_0")`. Retain all parsed
operation IDs and leave the original file unchanged. This does not add general
support for nested MLIR location expressions outside quoted strings.

### Read the compiler input back for attribution

The TOSA compatibility handler first prepares the model, then reads `tosa_ops`
from the returned path. `location_to_ops` is a `dict[str, int]`: each unique
location identifies exactly one source operation. Lookup checks for `None`
explicitly because operation ID zero is valid.

Alternative rejected: keep labels only in memory. Those labels would not occur
in the compiler debug database. Re-reading the processed model also avoids
depending on mutation of a dictionary supplied by the caller.

### Conversion and graph-segment execution are separate stages

Resolve the converter from `get_ml_sdk_model_converter_path()`, falling back to
the backend repository when that returns `None`. Pass the resolved path to
`MLSDKModelConverter` and convert the processed model.

Pass the converted VGF through `prepare_gcpe_compatible_vgfs()`. Run the estimator
once per returned graph segment, using segment-specific output names and record
keys. Reused compiler IDs in different segments must not overwrite records.
Existing segment preparation behavior, including compute-segment skipping,
is retained.

### Source-keyed records hold aggregated compiler evidence

`_CompilerRecord` stores `record_key`, `label`, `placement`, and optional
`op_type`. `_collect_tosa_compiler_records()` updates the supplied
`compiler_records: dict[int, _CompilerRecord]` and
`unattributed_records: list[_CompilerRecord]` in place.

Each source ID, including zero, has at most one attributed record across all
segments. Keep the first contributing record's key and label. Merge NX and EE
into `NX/EE` in that record, regardless of arrival order; subsequent evidence
must not revert it. No separate placement map or seen-ID set is needed.

Merge all distinct non-empty compiler types from each debug-table entry and
subsequent records into a slash-separated `op_type`, in first-seen order, such
as `ABS/ADD/MUL`. Missing types neither erase known types nor introduce a
placeholder; all-missing types remain `None`. Repeated types appear only once.
Pass the aggregated compiler type into compatibility reporting so entity names
and `op_type` attributes expose values such as `AvgPool/Rescale`. Fall back to
the parsed source TOSA operation's name only when compiler types are unavailable.
Source IDs and display labels continue to identify the original TOSA operation;
the merged type describes the compiler operations implementing it. This applies
to both source-attributed and converted-entity output.

Unmatched non-empty labels and empty-label fallback records remain separate
unattributed records. Do not group them under `None`, and do not infer placement
or source identity from their operation types.

Iterate the source-keyed mapping once to produce compatibility records and
checks. This avoids duplicate entities for repeated attributed source IDs at
record construction, rather than deduplicating entities during rendering.

### Unattributed converted entities use a separate stable namespace

For converted-output rendering, attributed TOSA entities retain
`converted_operator/<source-id>`. Unattributed entities instead use
`converted_operator/unattributed/<record_key>`, derived from their unique
compiler record keys, not display labels or list positions. Adding an
earlier-sorting record must not change existing entity IDs. Checks reference
these same IDs. Preserve the existing sorted record order and metric population.

Do not attach `converted_tosa_operator_id` to an unattributed entity: its compiler
record key is not a known TOSA source ID. Source-attributed output continues to
leave unmatched checks unlinked rather than fabricate source entities.

### Keep machine labels out of display labels

Use the complete synthetic label for source lookup. For an attributed TOSA
record, remove only the final generated counter for display. Thus
`layer_42_0` displays as `layer_42`. An originally empty location prepared as
`unknown_0` displays as `unknown`; do not strip any original numeric suffix.

Empty and whitespace-only compiler labels are ignored when usable labels exist.
If an estimator operation has no usable labels, retain exactly one unattributed
record displayed as `unknown`, with its actual placement and operation type.
Do not look up that fallback label as a source identity. Apply this rule on both
TOSA and VGF paths. Non-empty unmatched labels retain the existing unlinked-record
behavior; no source identity is fabricated for them.

### Backend metadata follows execution

`NXModelCompatibilityInfo` tracks whether the converter was used. Direct VGF
compatibility declares only `nx-performance-estimator`. Converted TOSA paths
declare both backends. The estimator is always the compatibility result
producer, and each declared backend receives its own configuration.

### Calculate metrics from consolidated compatibility records

All-NX records produce 100% accelerator placement. Mixed NX/EE records remain
supported but produce a lower accelerator percentage. `NX/EE` records are not
NX-only records for the numerator. Empty record sets retain the existing
unavailable-metric behavior.

For TOSA, count each attributed source operation once, plus every unattributed
record. For example, three NX compiler records for source A and one EE record
for source B produce two checks and 50%, not four checks and 75%. Direct VGF
record construction is unchanged.

Unattributed estimator records participate in the same calculation: one labelled
NX operation and one unlabelled EE operation yield 50%, not 100%. Missing labels
are distinct from missing compiler evidence: the former must not discard known
placement data.

Unique source labels do not prove that the compiler emitted placement evidence
for every source operation. The explicitly removed incomplete-coverage handling
is not reintroduced: missing evidence does not independently change status or
suppress a percentage calculated from available records.

## Risks / Trade-offs

- Compiler optimization can omit source operations. A record-based percentage
  is not proof of complete source coverage; this limitation is accepted here.
- MLIR annotation depends on the syntax supported by the current reader.
  Broader syntax support requires a separate change.
- Annotation rewrites location metadata in the processed model. Preserve the
  input bytes and strip only generated suffixes in the report.
- Moving the module breaks imports of its old internal path. Update repository
  imports and mock targets together; do not add a compatibility alias.
- Aggregation keeps the first compiler key and label, merged placement, and
  distinct types, not every contributing compiler record's identity or count.

## Migration Plan

The implementation is already present. Existing callers use the relocated
module. Confirm the current tests and OpenSpec artifacts, then archive this
change through the normal OpenSpec workflow after review. No model migration
or repository-variable update is required.

## Open Questions

- None for this scope. Complete source-coverage accounting remains deferred.

## Local Validation State

The earlier forced `source_operator_attribution_available = False` debug override
is now commented out. Reporting honors the supplied attribution flag; regression
coverage exercises both source-attributed and converted-entity output.
