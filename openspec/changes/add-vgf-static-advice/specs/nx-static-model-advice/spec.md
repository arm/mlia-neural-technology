<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

# Spec Delta

## Purpose

Provide actionable NX optimization advice from static VGF model structure,
linked to the affected operations in compatibility and performance reports.

## ADDED Requirements

### Requirement: Visit instructions with bounded file access

The VGF instruction visitor SHALL visit each segment's SPIR-V instructions in
model-sequence order, excluding module headers. It SHALL provide the segment
index and independently owned instruction bytes without copying the whole file.
Shared modules SHALL be visited for each referencing segment.

#### Scenario: Independent segment identities

- **WHEN** two segments reference a SPIR-V module containing the same result IDs
- **THEN** callbacks receive those instructions separately for each segment index
- **AND** retained instruction bytes remain usable after visitation completes

#### Scenario: Invalid input

- **WHEN** the input is empty, a segment lacks SPIR-V code, or a module header or instruction length is invalid
- **THEN** visitation raises a descriptive error rather than processing invalid bytes

#### Scenario: Callback failure

- **WHEN** a callback raises an exception
- **THEN** visitation propagates the exception to its caller

### Requirement: Avoid interpreting tensor payloads as shapes

Static advice analysis SHALL resolve tensor dimensions through integer shape
arrays and their scalar constant references. It SHALL NOT expand tensor constant
payloads into Python shape collections. Missing shape information SHALL NOT be
replaced with invented dimensions.

#### Scenario: Quantized weight tensor

- **WHEN** an integer tensor constant contains many scalar constant references
- **THEN** its payload is not treated as a shape vector
- **AND** its declared tensor type can still supply its shape for advice

### Requirement: Recommend CONV2D depth alignment

For CONV2D with known positive four-dimensional weights, analysis SHALL recommend
input and output channel counts divisible by 64 for kernel area 1, by 32 for
areas 2 through 3, by 16 for areas 4 through 7, and by 8 for areas 8 and above.
It SHALL emit a single record containing only the recommendations that apply.

#### Scenario: Both depths need alignment

- **WHEN** a 3-by-3 kernel has input and output depth 5
- **THEN** one advice record identifies both depths and recommends multiples of 8

#### Scenario: Aligned or unresolved convolution

- **WHEN** both depths meet the alignment rule, or any weight dimension is missing, zero, or negative
- **THEN** no depth-alignment advice is emitted for that operation

### Requirement: Recommend regular convolution for shallow depthwise inputs

For depthwise CONV2D with a known four-dimensional weight shape and input depth
from 1 through 3, analysis SHALL recommend considering regular CONV2D for hardware
utilization. It SHALL emit no such advice when input depth is unknown, non-positive,
or at least 4.

#### Scenario: Shallow feature map

- **WHEN** a depthwise operation has input depth 2
- **THEN** its advice suggests considering regular CONV2D

#### Scenario: Sufficient depth

- **WHEN** a depthwise operation has input depth 4
- **THEN** no shallow-depth advice is emitted

### Requirement: Identify large skip connections

Analysis SHALL identify a skip edge when a merge directly consumes a TOSA result
that is also reachable backwards through another operation input. It SHALL emit
advice only when the bypassed result's known size exceeds 131072 bytes. Advice
SHALL identify the endpoints and size, warn of potential poor cache utilization,
and link to the merge operation.

#### Scenario: Large alternate path

- **WHEN** a merge consumes A and B, B depends directly or transitively on A, and A produces 184320 bytes
- **THEN** advice identifies the skip from A to the merge and warns about large skip connections

#### Scenario: No qualifying skip

- **WHEN** inputs are independent, only share an ancestor, or the bypassed size is unknown or at most 131072 bytes
- **THEN** no skip-connection advice is emitted

#### Scenario: Constant operands are not operations

- **WHEN** candidate paths include weight, shape, or other non-operation operand IDs
- **THEN** these IDs are excluded from traversal and cannot establish a skip connection
- **AND** the stored operand relationships remain unchanged

### Requirement: Gather advice across graph segments

Performance and compatibility analysis SHALL gather advice from every prepared
graph segment, including segments produced by TOSA conversion, and SHALL NOT
analyze skipped compute segments for advice. Advice identities SHALL preserve
the original segment namespace even when generated files use local segment zero.

#### Scenario: Reused result IDs in split files

- **WHEN** two prepared graph files both contain a qualifying operation with result ID 42
- **THEN** performance and compatibility advice IDs distinguish their original segments
- **AND** direct VGF entity references use the corresponding original segment indexes

### Requirement: Attribute converted TOSA advice to source entities

For converted TOSA compatibility, advice SHALL resolve VGF operation debug labels
to prepared TOSA source locations, using decoded provenance display labels when
an exact match is absent. Unmatched labels SHALL NOT invent source attribution.

#### Scenario: Direct or structured source label

- **WHEN** an operation debug label directly matches a prepared source location or its decoded display label matches
- **THEN** the advice is linked to that TOSA source entity rather than a VGF-only ID

#### Scenario: Missing source mapping

- **WHEN** a VGF advice operation cannot be mapped to a TOSA source entity
- **THEN** its standardized advice is omitted

### Requirement: Publish advice only for NX entities

Standardized compatibility and performance results SHALL publish advice only when
its affected entity exists in that result and has placement exactly NX. Each
published record SHALL carry its advice ID, performance category, info severity,
message, and affected entity ID. EE, mixed NX/EE, unplaced, missing, and unlinked
records SHALL be omitted.

#### Scenario: Mixed placements

- **WHEN** qualifying advice records reference NX, EE, NX/EE, and unplaced entities
- **THEN** only records linked to NX entities appear in standardized advice

### Requirement: Preserve backend advice without a generic fallback

Neural Technology data collection SHALL forward backend-produced advice without
injecting the generic performance-metrics suggestion. An empty advice list SHALL
be valid when no attributable NX optimization advice applies.

#### Scenario: Python API forwarding

- **WHEN** the performance backend returns a standardized advice list
- **THEN** the Python API preserves that list without adding `performance_metrics`
