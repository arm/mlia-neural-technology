## Why

Core MLIA is defining standardized performance fields for AI Portal relevant output, but the Neural Technology plugin owns the NX Performance Estimator source values. The plugin needs its own change so it can map trustworthy estimator output into the core contract without moving plugin-specific extraction logic into core MLIA.

## What Changes

- Integrate the NX Performance Estimator standardized output path with the core standardized performance metric helper once that helper is available from `mlia`.
- Preserve existing Neural Technology model-level metrics, including backend-specific metrics such as `infs_per_sec`.
- Add standard MLIA metric entries for NX source values whose semantics match the core contract.
- Emit availability-aware entries for standard performance fields that the NX Performance Estimator cannot currently provide.
- Add plugin-specific tests for source-value mapping, unavailable entries, and existing metric preservation.

## Capabilities

### New Capabilities

- `neural-technology-standard-performance-fields`: Neural Technology integration for MLIA standardized performance fields and availability-aware metric entries.

### Modified Capabilities

- None.

## Impact

- `src/mlia/backend/nx_performance_estimator/performance.py`
- Neural Technology performance output tests, especially `tests/test_backend_nx_performance_estimator_performance.py`
- Dependency on the core `mlia` change that provides schema `1.1.0`, standardized metric names/units, availability-aware metric entries, and the shared helper
- Possible follow-up release or dependency-floor work once the core helper is available from a published `mlia` package
