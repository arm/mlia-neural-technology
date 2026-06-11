## Why

Standardized Neural Technology output needs
`accelerator_operator_percentage`, but the performance path currently has no
trustworthy source for operator placement. The compatibility path already
records whether each converted operation is placed on NX, EE, or remains
unsupported, so it can provide the percentage with clear semantics.

## What Changes

- Add an `accelerator_operator_percentage` metric to Neural Technology
  compatibility standardized output.
- Calculate the metric as the percentage of compatibility records placed on NX
  out of all compatibility records.
- Treat EE/Shader-routed records as compatible but not accelerator-placed for
  this metric.
- Treat unsupported records as part of the denominator and not
  accelerator-placed.
- Emit an availability-aware metric when no compatibility records are present.
- Preserve the existing performance-output behavior, where
  `accelerator_operator_percentage` remains unavailable unless a trustworthy
  performance source is added later.

## Capabilities

### New Capabilities

- `compatibility-accelerator-operator-percentage`: Neural Technology
  compatibility output reports the percentage of converted operations placed on
  the accelerator.

### Modified Capabilities

- None.

## Impact

- `src/mlia/backend/ml_sdk_model_converter/compat.py`
- Neural Technology compatibility output tests
- OpenSpec change artifacts under
  `openspec/changes/add-neural-technology-operator-percentage/`
