# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""NX normalization supports core measured results without estimator state."""

from pathlib import Path

from mlia.core import output_schema as schema
from mlia.core.output_validation import validate_standardized_output
from mlia.core.performance import MeasuredPerformanceMetrics, PerformanceObservations

from mlia.nx_utils.performance import normalize_nx_observations


def _standardized_output(metrics, model):
    return metrics.to_standardized_output(
        model,
        target=schema.Target(
            target_type="neural-technology",
            profile_name="NX-measured",
            components=[
                schema.Component(
                    type=schema.ComponentType.GPU, family="mali", model="nx"
                )
            ],
            configuration={},
        ),
        backend=schema.Backend(
            id="neural-technology-profiling-data",
            name="Neural Technology Profiling Data",
            version="experimental",
            configuration={},
        ),
    )


def test_nx_metrics_preserves_measured_cycles_and_separate_runtime_scope(
    tmp_path: Path,
):
    """Measured observations reuse NX availability rules without inventing totals."""
    source_id = "source_operator/etrecord/decode/conv"
    metrics = MeasuredPerformanceMetrics(
        [
            normalize_nx_observations(item)
            for item in [
                PerformanceObservations(
                    metrics=[
                        schema.Metric(name="total_cycles", value=1000.5, unit="cycles"),
                        schema.Metric(name="execution_count", value=2, unit="count"),
                    ],
                ),
                PerformanceObservations(
                    entities=[
                        schema.Entity(id=source_id, kind="source_operator", name="conv")
                    ],
                    breakdowns=[
                        schema.Breakdown(
                            entity_id=source_id,
                            metrics=[
                                schema.Metric(
                                    name="Runtime Total Avg", value=1.5, unit="ms"
                                ),
                            ],
                        )
                    ],
                ),
            ]
        ]
    )
    metrics.backend_configuration = {"profiling_kind": "mixed-vgf-and-standard-etdump"}
    metrics.runtime_configuration = {"etrecord_method": "decode"}
    metrics.extensions = {"etrecord_mapping": {"method": "decode"}}
    model = tmp_path / "model.bin"
    model.write_bytes(b"record")
    metrics.model_format = "etrecord"
    output = _standardized_output(metrics, model)
    validate_standardized_output(output)
    first, runtime = output["results"]
    first_metrics = {item["name"]: item for item in first["metrics"]}
    assert first_metrics["total_cycles"]["value"] == 1000.5
    assert first_metrics["execution_count"]["value"] == 2
    assert first_metrics["compiled_size"]["availability"] == "unavailable"
    runtime_metrics = {item["name"]: item for item in runtime["metrics"]}
    assert runtime_metrics["total_cycles"]["availability"] == "unavailable"
    assert runtime_metrics["inference_time"]["availability"] == "unavailable"
    assert runtime["breakdowns"][0]["metrics"] == [
        {"name": "Runtime Total Avg", "value": 1.5, "unit": "ms"},
    ]
    assert all(item["mode"] == "measured" for item in output["results"])
    assert output["context"]["runtime_configuration"] == {"etrecord_method": "decode"}
    assert output["extensions"] == metrics.extensions
    assert output["model"]["format"] == "etrecord"


def test_measured_model_metrics_reuse_nx_derived_metrics(tmp_path: Path):
    """Measured model latency and cycles use the existing NX derivation rules."""
    metrics = MeasuredPerformanceMetrics(
        [
            normalize_nx_observations(
                PerformanceObservations(
                    metrics=[
                        schema.Metric(name="inference_time", value=2.0, unit="ms"),
                        schema.Metric(name="compute_cycles", value=75, unit="cycles"),
                        schema.Metric(name="total_cycles", value=100, unit="cycles"),
                    ]
                )
            )
        ]
    )
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"record")
    output = _standardized_output(metrics, model)
    values = {item["name"]: item for item in output["results"][0]["metrics"]}
    assert values[schema.METRIC_NAME_INFERENCES_PER_SECOND]["value"] == 500.0
    assert values[schema.METRIC_NAME_TARGET_UTILIZATION]["value"] == 75.0
