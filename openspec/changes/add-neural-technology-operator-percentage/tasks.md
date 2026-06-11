## 1. Compatibility Output Tests

- [x] 1.1 Map the existing compatibility standardized output code and tests.
- [x] 1.2 Add focused tests for NX, EE/Shader, unsupported, and empty-record accelerator percentage behavior.
- [x] 1.3 Add or preserve a focused assertion that performance output leaves `accelerator_operator_percentage` unavailable without placement data.

## 2. Implementation

- [x] 2.1 Add compatibility metric construction for `accelerator_operator_percentage`.
- [x] 2.2 Keep existing compatibility checks, entity placement output, and result statuses unchanged.
- [x] 2.3 Avoid changing NX Performance Estimator behavior for this metric.

## 3. Validation

- [x] 3.1 Run OpenSpec validation for the change.
- [x] 3.2 Run the targeted compatibility and performance-output tests.
- [x] 3.3 Run formatting, linting, and copyright checks for touched files.
- [x] 3.4 Review the diff for scope and public/private boundary issues.
