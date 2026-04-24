<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Outputs and Metrics

## Overview

Neural Technology workflows in MLIA can produce both preparation artifacts and
final estimation results. That makes the output story slightly richer than a
single "metrics page" might suggest: some outputs exist to help the estimator
run, while others are the actual results users inspect and compare.

## Typical output forms

Common output forms include:

- Console summaries printed by MLIA.
- JSON output when `--json` is used.
- Intermediate converted artifacts consumed by downstream Neural Technology
  backends
- Estimator-generated statistics files in the run output directory.

It helps to think of those outputs in two groups:

- Conversion-stage outputs that prove the pipeline prepared the model correctly.
- Estimator outputs that explain performance behaviour.

## Example JSON shape

A simplified result shape might look like this:

```json
{
  "target": {"profile": "neural-technology"},
  "backends": [{"name": "nx-performance-estimator"}],
  "results": [
    {
      "metrics": {
        "totalCycles": 456789,
        "opCycles": 400000
      }
    }
  ]
}
```

Real output can include more detail, but the key point is that the result keeps
the target and backend context next to the metrics, which makes it easier to
compare runs later.

## Main estimator metrics

When the workflow uses `nx-performance-estimator`, the most useful metrics
usually include:

- `totalCycles`
- `opCycles`
- Per-operator memory statistics such as `readBytes` and `writeBytes`.
- Traffic-oriented cycle metrics such as `trafficCycles`.
- Hardware-section utilisation statistics.

These numbers help you answer three practical questions:

- How expensive is the run overall?
- Where is the cost concentrated?
- Is the dominant problem compute work, data movement, or a broader scheduling
  effect?

## How to read the estimator output

Start with `totalCycles`. It gives the quickest top-level answer about whether
the result looks broadly acceptable, slightly concerning, or clearly too high.

Then compare the large `opCycles` contributors. That usually reveals whether the
problem is concentrated in a few operators or spread more evenly across the
model. Concentrated cost is often the easiest kind of cost to act on, because it
suggests a smaller set of layers or operator families to inspect first.

After that, look at the memory-related signals. `readBytes`, `writeBytes`, and
traffic-oriented counters help distinguish compute-heavy behaviour from
bandwidth-heavy behaviour. If the model is dominated by movement rather than
operator work, the next optimization idea should usually focus on layout,
representation, or conversion-path effects rather than on arithmetic alone.

## Conversion-related outputs

Some Neural Technology workflows produce artifacts that are not themselves the
final user-facing result. They exist because the estimator expects the model in
a particular prepared form.

Those artifacts matter most when the pipeline is being validated or debugged. If
the estimator output is missing, malformed, or clearly inconsistent with what
you expected, the conversion-stage artifacts are often the best place to check
whether the pipeline reached the estimator cleanly.

## What to do when numbers look bad

If `totalCycles` is high, do not immediately try to optimize the whole model.
Look first for the few operators contributing the largest share of `opCycles`.
That usually gives a better starting point than reacting to the headline number
alone.

If memory traffic dominates, treat that as a signal that the model may be paying
more for data movement than for compute. In those cases, layout choices,
intermediate representation, or configuration can matter as much as the raw
operator set.

If the estimator output is missing altogether, or if the metrics look thinner
than expected, do not assume the estimator itself is broken. It can simply mean
that an earlier conversion stage did not complete in the way the estimator
needed.

## A practical review pattern

A useful review sequence is:

1. Check whether the workflow produced the expected estimator output at all.
2. Read the top-level total first.
3. Identify the heaviest operator contributors.
4. Use memory and traffic metrics to decide whether the next investigation
   should focus on compute or movement.
5. Only then return to the earlier conversion artifacts if the result shape
   still looks suspicious.

This keeps the page grounded in real troubleshooting and tuning behaviour,
rather than treating the metrics as a list to memorize.

## Cross-links

- See [backends.md](backends.md) for backend roles and the wider pipeline story.
- See [cli.md](cli.md) for commands that produce these outputs.
- See [troubleshooting.md](troubleshooting.md) for conversion-versus-estimator
  debugging guidance.
