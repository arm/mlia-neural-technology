## ADDED Requirements

### Requirement: Compatibility output reports accelerator operator percentage

Neural Technology SHALL emit `accelerator_operator_percentage` in compatibility
standardized output when operator placement records are available.

#### Scenario: NX operations are counted as accelerator-placed

- **WHEN** compatibility output contains operation records with placement `NX`
- **THEN** `accelerator_operator_percentage` counts those records in the
  numerator and emits the metric with unit `%`.

#### Scenario: EE or Shader operations are not counted as accelerator-placed

- **WHEN** compatibility output contains operation records routed through
  EE/Shader support
- **THEN** `accelerator_operator_percentage` includes those records in the
  denominator but excludes them from the numerator.

#### Scenario: Unsupported operations remain in the denominator

- **WHEN** compatibility output contains unsupported operation records
- **THEN** `accelerator_operator_percentage` includes those records in the
  denominator but excludes them from the numerator.

#### Scenario: Empty compatibility records are unavailable

- **WHEN** compatibility output has no operation records
- **THEN** `accelerator_operator_percentage` is emitted as an availability-aware
  metric entry with unit `%` and a reason.

### Requirement: Performance output remains unavailable without placement source

Neural Technology SHALL NOT derive `accelerator_operator_percentage` for
performance standardized output from unrelated performance data.

#### Scenario: Performance output lacks placement data

- **WHEN** Neural Technology builds standardized performance output without a
  trustworthy operator placement percentage source
- **THEN** `accelerator_operator_percentage` remains an availability-aware
  metric entry.
