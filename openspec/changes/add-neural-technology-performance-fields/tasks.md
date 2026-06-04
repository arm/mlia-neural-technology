## 1. Contract Alignment

- [x] 1.1 Confirm this plugin change depends on the core `mlia` standardized performance metric helper rather than redefining schema behavior locally.
- [x] 1.2 Confirm the NX Performance Estimator source mapping for `inferences_per_second` uses `1000 / inference_time`.
- [x] 1.3 Confirm the NX Performance Estimator source mapping for `target_utilization`.
- [x] 1.4 Confirm that `dram_footprint` is not treated as `peak_activation_memory` without matching source semantics.

## 2. Tests

- [x] 2.1 Add or update tests proving existing NX Performance Estimator metrics remain present.
- [x] 2.2 Add or update tests proving `inferences_per_second` is derived from `inference_time` while `infs_per_sec` is preserved.
- [x] 2.3 Add or update tests proving `target_utilization` is calculated from `compute_cycles` and `total_cycles`.
- [x] 2.4 Add or update tests proving missing standard performance metrics are emitted as availability-aware entries.
- [x] 2.5 Add a plugin-level test proving the standardized output validates against the MLIA output schema.

## 3. Implementation

- [x] 3.1 Add standardized metric aliases for trustworthy NX Performance Estimator source values.
- [x] 3.2 Pass result-level metrics through the core standardized performance metric helper.
- [x] 3.3 Preserve existing backend-specific metric names and units.
- [ ] 3.4 Update dependency metadata once the core helper is available from a published `mlia` package.

## 4. Validation

- [x] 4.1 Run OpenSpec validation for this change.
- [x] 4.2 Run the targeted Neural Technology performance output tests.
- [x] 4.3 Run formatting, linting, and copyright checks for touched files.
- [x] 4.4 Update the cross-repo implementation plan with implementation status and validation evidence.
