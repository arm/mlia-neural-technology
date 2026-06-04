## ADDED Requirements

### Requirement: Neural Technology preserves existing performance metrics

Neural Technology SHALL preserve existing NX Performance Estimator result metrics when adding standardized performance fields.

#### Scenario: Existing throughput metric remains present

- **WHEN** Neural Technology has a model-level `infs_per_sec` value
- **THEN** the performance result includes the existing `infs_per_sec` metric with unit `inferences/s`.

#### Scenario: Existing model metrics remain present

- **WHEN** Neural Technology builds standardized performance output
- **THEN** existing model-level NX Performance Estimator metrics remain present in the performance result.

#### Scenario: Existing model metric value is null

- **WHEN** NX Performance Estimator output provides a model-level metric with a null value
- **THEN** Neural Technology emits that metric as an availability-aware entry with its existing metric name and unit.

### Requirement: Neural Technology integrates with core standardized performance fields

Neural Technology SHALL use the core MLIA standardized performance metric contract when building standardized performance results.

#### Scenario: Core helper fills missing standardized metrics

- **WHEN** Neural Technology passes a performance result metric list through the core standardized performance metric helper
- **THEN** supplied numeric values are preserved and missing standardized metrics are represented as availability-aware metric entries.

#### Scenario: Helper is called explicitly

- **WHEN** Neural Technology builds standardized performance output
- **THEN** Neural Technology explicitly calls the core helper rather than relying on core reporting to modify the result later.

### Requirement: Neural Technology maps only trustworthy source values

Neural Technology SHALL emit numeric standardized metrics only when NX Performance Estimator source data matches the standardized metric semantics.

#### Scenario: Inference throughput source is available

- **WHEN** Neural Technology has a non-zero `inference_time` value in milliseconds for a performance result
- **THEN** Neural Technology emits `inferences_per_second` as `1000 / inference_time` with unit `inferences/s`.

#### Scenario: Existing throughput metric remains independent

- **WHEN** Neural Technology emits the standardized `inferences_per_second` metric
- **THEN** Neural Technology preserves the existing backend-specific `infs_per_sec` metric.

#### Scenario: Neural engine utilization source is available

- **WHEN** Neural Technology has suitable `compute_cycles` and `total_cycles` counters for a performance result
- **THEN** Neural Technology emits `target_utilization` with value `(compute_cycles / total_cycles) * 100 if total_cycles else 0.0` and unit `%`.

#### Scenario: CPU utilization source is absent

- **WHEN** Neural Technology has no trustworthy source for CPU utilization
- **THEN** Neural Technology emits `cpu_utilization` as an availability-aware metric entry with unit `%` and a reason.

#### Scenario: Accelerator operator percentage source is absent

- **WHEN** Neural Technology has no trustworthy operator placement percentage source available to the performance output
- **THEN** Neural Technology emits `accelerator_operator_percentage` as an availability-aware metric entry with unit `%` and a reason.

#### Scenario: Peak activation memory source is not confirmed

- **WHEN** no Neural Technology source value is confirmed to have peak activation memory semantics
- **THEN** Neural Technology emits `peak_activation_memory` as an availability-aware metric entry with unit `bytes` and a reason.

#### Scenario: Average memory source is absent

- **WHEN** Neural Technology has no trustworthy average-over-window memory source
- **THEN** Neural Technology emits `average_memory` as an availability-aware metric entry with unit `bytes` and a reason.

### Requirement: Neural Technology validation covers standardized payload behavior

Neural Technology SHALL include tests that prove the standardized performance payload behavior.

#### Scenario: Result-level standardized metrics are tested

- **WHEN** Neural Technology standardized performance output is tested
- **THEN** tests assert the result-level standardized metrics and availability-aware entries, not only backend-specific metrics.

#### Scenario: Existing metric preservation is tested

- **WHEN** Neural Technology standardized performance output is tested
- **THEN** tests assert that existing backend-specific metrics are preserved after standardized result metrics are added.
