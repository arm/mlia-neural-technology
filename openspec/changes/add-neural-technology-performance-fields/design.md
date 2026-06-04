## Context

The core `mlia` change `add-ai-portal-performance-fields` defines a standardized JSON contract for AI Portal relevant performance fields. This repository owns the Neural Technology target plugin and the NX Performance Estimator backend path, so it must decide which estimator source values can populate those standardized metrics and where the output must explicitly mark values unavailable.

The current NX Performance Estimator output already emits model-level metrics, including `inference_time`, `infs_per_sec`, cycle counts, compiled size, memory traffic, and `dram_footprint`. It also emits operator and operator-chain breakdowns. This plugin change should add standard MLIA metric names alongside existing plugin-specific metrics rather than renaming or removing existing output.

## Goals / Non-Goals

**Goals:**

- Prove the core standardized performance metric contract against the Neural Technology performance path.
- Call the core helper explicitly once the helper is available from `mlia`.
- Preserve existing Neural Technology result metrics when adding standardized fields.
- Populate standardized metrics from NX Performance Estimator source values only when the source semantics match the core contract.
- Emit availability-aware metric entries for standardized fields that Neural Technology cannot provide.
- Add Neural Technology-specific tests for source mapping, unavailable entries, and existing metric preservation.

**Non-Goals:**

- Do not define the MLIA output schema or standard metric names in this plugin repository.
- Do not fabricate CPU utilization, average memory, peak activation memory, or operator placement values from unrelated estimator fields.
- Do not rename or remove existing backend-specific metrics such as `infs_per_sec`.
- Do not add a separate Perf-AI-shaped report.

## Decisions

### Neural Technology calls the core helper explicitly

The core `mlia` helper should be called from the NX Performance Estimator result construction path after the plugin has built its model-level metric list. This keeps the shared contract in core while making plugin-owned source extraction explicit.

The helper should receive the metrics Neural Technology can provide and fill missing standardized metrics as availability-aware entries. The plugin should not duplicate the core unavailable-fill rules.

Alternative considered: rely on core reporting to normalize every performance result automatically. The core spec rejected that for the initial implementation, and this plugin should follow the explicit helper-call model.

### Existing metrics remain stable

The plugin should keep existing NX Performance Estimator metric names and units in the result-level metric list. Standardized MLIA metric entries are additive aliases where the source value has suitable semantics.

This avoids breaking current consumers while allowing consumers of the MLIA standardized metric set to use the generic names.

### Source mapping is metric-specific

NX Performance Estimator source values should map to standardized MLIA metric names as follows:

| Standard metric | NX source | Initial behavior |
| --- | --- | --- |
| `inferences_per_second` | `NXModelPerformanceStats.inference_time` | Emit `1000 / inference_time` with unit `inferences/s` when latency is non-zero, while preserving the existing backend-specific `infs_per_sec` metric. |
| `target_utilization` | `compute_cycles` and `total_cycles` | Emit `(compute_cycles / total_cycles) * 100 if total_cycles else 0.0` with unit `%`. |
| `cpu_utilization` | None currently identified | Emit unavailable with unit `%`. |
| `accelerator_operator_percentage` | None currently identified in performance output | Emit unavailable with unit `%`. |
| `peak_activation_memory` | No confirmed peak activation memory source in current NX Performance Estimator model-level output | Emit unavailable with unit `bytes`. |
| `average_memory` | No confirmed average-over-window memory source in current NX Performance Estimator model-level output | Emit unavailable with unit `bytes`. |

The implementation should not map `dram_footprint` to `peak_activation_memory` or `average_memory` solely because it is a memory value. The source requirements call for result-level memory metrics, but Neural Technology should emit unavailable entries until a source is confirmed to match the peak activation or average-over-window semantics.

The standardized throughput value should follow the source requirement formula based on single-inference latency. The existing `infs_per_sec` metric remains in the output as backend-specific data, but it is not the source of truth for the standardized MLIA metric.

### Interpretation notes use existing result fields when needed

If the plugin emits limitations needed to interpret standardized performance metrics, they should use result warnings. Structured availability reasons should remain on availability-aware metric entries.

## Risks / Trade-offs

- Core helper availability may depend on a released `mlia` version. Mitigation: land this after the core change is available and update dependency floors through the normal MLIA cross-repo process.
- `dram_footprint` may be tempting but semantically wrong for `peak_activation_memory` or `average_memory`. Mitigation: keep these fields unavailable in this change, and require explicit semantic confirmation before any later numeric mapping.
- Standardized throughput overlaps with the existing backend-specific `infs_per_sec` concept. Mitigation: keep both names, derive the standard metric from `inference_time`, and preserve `infs_per_sec` for existing output compatibility.
- This integration does not complete every backend or every MLIA target. Mitigation: keep this OpenSpec scoped to Neural Technology and rely on separate plugin-owned changes where needed.

## Migration Plan

Implement this after the core `mlia` change provides schema `1.1.0`, standardized metric constants or names, availability-aware metric support, and the shared helper.

Plugin implementation should then update the NX Performance Estimator output path, tests, and dependency metadata as needed. If the helper is only available from an unreleased core commit, use the normal MLIA cross-repo dependency process rather than adding a local compatibility copy.
