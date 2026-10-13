<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

## ADDED Requirements

### Requirement: Preparation creates uniquely labelled model copies

TOSA compatibility SHALL prepare a separate model copy before conversion for
both supported FlatBuffer and MLIR inputs. Preparation SHALL assign unique
locations without modifying the original file or changing source operation IDs.
Readers SHALL NOT generate synthetic locations themselves.

#### Scenario: Operations have empty locations

- **WHEN** parsed operations have empty locations
- **THEN** preparation assigns `unknown_0`, `unknown_1`, and subsequent counter
  values within the `unknown` prefix
- **AND** the processed model contains those same locations

#### Scenario: Operations share an existing location

- **WHEN** multiple operations have location `layer`
- **THEN** preparation assigns `layer_0`, `layer_1`, and subsequent counter values
- **AND** an existing numeric suffix is preserved before the new suffix is added

#### Scenario: Synthetic and existing prefixes coincide

- **WHEN** empty locations and literal `unknown` locations coexist
- **THEN** preparation uses one counter for that prefix and produces no duplicate
  labels

### Requirement: Quoted MLIR location delimiters are preserved

MLIR reading and preparation SHALL treat parentheses inside quoted location
labels as label content, not as location delimiters. Both referenced and inline
quoted labels SHALL retain their operations and exact label content through
synthetic-label assignment and processed-model rereading.

#### Scenario: A referenced label contains parentheses

- **WHEN** two operations reference `#loc1 = loc("conv(foo)")`
- **THEN** both operations are present after preparation with locations
  `conv(foo)_0` and `conv(foo)_1`
- **AND** the input file is unchanged

#### Scenario: An inline label contains parentheses

- **WHEN** an operation uses inline `loc("conv(foo)")`, including labels with
  nested or unmatched parentheses inside the quotes
- **THEN** reading and replacing the location preserves the complete label
- **AND** the operation remains present when the processed file is reread

### Requirement: Attribution uses the processed model

The compatibility handler SHALL retrieve `tosa_ops` from the processed model.
Each processed label SHALL map to a single integer source operation ID.

#### Scenario: The first source operation is reported

- **WHEN** a compiler label maps to source operation ID zero
- **THEN** the resulting check is attributed to that operation rather than being
  treated as an unmatched label

### Requirement: Converted VGF graph segments are processed individually

Converted TOSA models SHALL use the same GCPE-compatible segment preparation as
direct VGF inputs. Every returned graph segment SHALL be processed separately,
with segment-scoped compiler record keys.

#### Scenario: Conversion produces multiple graph segments

- **WHEN** conversion produces a VGF containing multiple graph segments
- **THEN** each GCPE invocation receives a prepared single-graph input
- **AND** repeated compiler IDs across segments do not overwrite records

#### Scenario: One source operation has mixed placement evidence

- **WHEN** compiler records attribute both NX and EE placement to one source ID
- **THEN** its single attributed compatibility record uses placement `NX/EE`

### Requirement: Compiler records aggregate by TOSA source operation ID

TOSA compatibility SHALL store one `_CompilerRecord` per attributed source ID
across segments, retaining its first key and label. Records SHALL contain
`record_key`, `label`, `placement`, and optional `op_type`. Mixed placement SHALL
merge to `NX/EE`; distinct non-empty types SHALL merge in first-seen order,
separated by `/`. Unattributed records SHALL remain separate.

#### Scenario: Compiler expansion repeats one source operation

- **WHEN** several compiler records in one or more segments map to the same
  source operation ID
- **THEN** the collector stores one attributed record and emits one check
- **AND** repeated references to that source do not create duplicate entity IDs

#### Scenario: Placement changes and then repeats

- **WHEN** one source receives NX, EE, and then NX evidence, or the reverse order
- **THEN** its stored placement becomes `NX/EE` and stays `NX/EE`

#### Scenario: Compiler operation types differ

- **WHEN** successive entries for one source report `ABS`, `ADD`, `ABS`, and `MUL`
- **THEN** the stored compiler type is `ABS/ADD/MUL`
- **AND** multiple types within an individual debug entry are also retained

#### Scenario: Compiler operation types are missing

- **WHEN** missing type information precedes or follows a known type
- **THEN** the known type is retained without a placeholder
- **AND** an operation with no known compiler types retains `None`

#### Scenario: Multiple records cannot be attributed

- **WHEN** two compiler records cannot be mapped to a source operation ID
- **THEN** both remain distinct records, checks, and metric contributions

### Requirement: Reports expose aggregated compiler operation types

For attributed TOSA records, compatibility reports SHALL use the merged compiler
type for entity names and `op_type` attributes when available. Otherwise they
SHALL use the parsed source TOSA operation type. Source IDs and display labels
SHALL remain tied to the source operation. This behavior SHALL apply whether
output uses source entities or converted entities.

#### Scenario: Average pooling lowers to multiple compiler types

- **WHEN** `AvgPool` and `Rescale` compiler records map to the same TOSA operation
- **THEN** its reported entity name and `op_type` are `AvgPool/Rescale`
- **AND** it still contributes a single compatibility check

#### Scenario: The compiler provides no operation types

- **WHEN** no compiler type is available for a parsed `tosa.avg_pool2d` operation
- **THEN** its reported name and `op_type` fall back to `tosa.avg_pool2d`

### Requirement: Converter lookup supports repository installations

TOSA conversion SHALL use the backend-repository converter settings when the
default converter-path lookup returns `None`.

#### Scenario: The default converter lookup has no path

- **WHEN** `get_ml_sdk_model_converter_path()` returns `None`
- **THEN** compatibility resolves `ml-sdk-model-converter` through the backend
  repository and passes that resolved path to the converter

### Requirement: Display labels omit generated counters

Attributed TOSA records SHALL preserve unique internal labels while removing
only their generated final counter from display labels. Operations whose original
locations were empty SHALL be displayed as `unknown`.

#### Scenario: An original label already ends in a number

- **WHEN** preparation changes `layer_42` to `layer_42_0`
- **THEN** the report displays `layer_42` and retains the unique attribution key

#### Scenario: An operation had no original location

- **WHEN** the attributed compiler label is `unknown_0`
- **THEN** its display label is `unknown`

### Requirement: Unlabelled estimator operations retain placement records

TOSA and VGF compatibility SHALL ignore empty and whitespace-only compiler
labels when usable labels exist. When an estimator operation has no usable
labels, compatibility SHALL retain exactly one unattributed record displayed as
`unknown`, preserving the operation's placement and type without inventing a
source association.

#### Scenario: A compiler entry contains valid and blank labels

- **WHEN** a compiler entry contains a valid label alongside blank labels
- **THEN** only the valid label contributes a compatibility record, check, or
  percentage contribution

#### Scenario: An estimator operation has no usable labels

- **WHEN** an estimator operation's label list is empty or contains only empty
  or whitespace-only strings
- **THEN** exactly one unattributed record contributes to checks and placement
  metrics for that operation
- **AND** its display label is `unknown`
- **AND** compatibility does not parse provenance or look up source operations
  for the missing label or its display fallback

#### Scenario: An unlabelled EE operation accompanies a labelled NX operation

- **WHEN** the estimator reports one labelled NX operation and one unlabelled
  EE operation
- **THEN** both operations contribute records and accelerator placement is 50%

### Requirement: Unattributed converted entity IDs are stable and disjoint

Converted-output rendering SHALL derive unattributed entity IDs from their
compiler record keys under `converted_operator/unattributed/`, separately from
numeric TOSA source IDs. These IDs SHALL NOT depend on list indices or display
labels. Checks SHALL reference the corresponding entity IDs, and unattributed
entities SHALL NOT claim a `converted_tosa_operator_id`.

#### Scenario: An unattributed record is first in the sorted record list

- **WHEN** an unattributed record has index zero and another record refers to
  TOSA source operation zero
- **THEN** the attributed entity retains `converted_operator/0`
- **AND** the unattributed entity uses its compiler-key-derived ID without a
  collision or loss of either check

#### Scenario: Record positions change

- **WHEN** a new record sorts before existing unattributed records
- **THEN** their entity IDs remain unchanged and their checks reference those IDs

#### Scenario: Unattributed records share a display label

- **WHEN** distinct compiler record keys both have display label `unknown`
- **THEN** their converted entities remain distinct

### Requirement: Backend declarations reflect actual execution

Compatibility results SHALL name `nx-performance-estimator` as their producer.
They SHALL declare the converter only when the executed path used it.

#### Scenario: Direct VGF compatibility bypasses conversion

- **WHEN** the input is handled directly as VGF
- **THEN** the backend list contains only `nx-performance-estimator`

#### Scenario: TOSA compatibility invokes conversion

- **WHEN** a prepared TOSA input is converted before estimator execution
- **THEN** both `ml-sdk-model-converter` and `nx-performance-estimator` are
  declared, each with its own configuration

### Requirement: Placement metrics count consolidated TOSA records

Compatibility SHALL calculate accelerator percentage from available records,
counting only NX-only placements in the numerator. For TOSA inputs, attributed
operations SHALL each contribute once, while all unattributed records SHALL
contribute separately. Direct VGF accounting SHALL remain unchanged. This change
SHALL NOT add incomplete-source-coverage status or warnings.

#### Scenario: One source expands into several NX compiler operations

- **WHEN** three NX compiler records map to source A and one EE record maps to B
- **THEN** compatibility emits two checks and accelerator placement is 50%

#### Scenario: Every reported operation is placed on NX

- **WHEN** all available compatibility records have placement `NX`
- **THEN** compatibility is supported and accelerator placement is 100%

#### Scenario: Reported operations are split between NX and EE

- **WHEN** available records include NX and EE placements
- **THEN** both placements remain supported and only NX records contribute to
  the accelerator numerator

#### Scenario: A parsed operation has no compiler placement evidence

- **WHEN** the compiler omits placement evidence for a parsed operation
- **THEN** compatibility does not invent a CPU placement or an unsupported check
- **AND** the omission does not independently introduce an incomplete result or
  make an otherwise available percentage unavailable
