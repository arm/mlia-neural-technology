# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Neural Technology's ExecuTorch timing attribution."""

import statistics
from dataclasses import replace

import pytest
from mlia.core import output_schema as schema
from mlia.core.output_projection import project_entity_breakdowns

from mlia.backend.executorch.capture import ETDumpEvent, ProfilingCapture
from mlia.backend.executorch.etrecord import source_operator_id
from mlia.backend.neural_technology_profiling_data.executorch_runtime import (
    build_standard_runtime_mappings,
    standard_runtime_observations,
)


def _runtime_event(
    name, handles, samples, *, instruction=0, backend="TestBackend", delegate_id=None
):
    """Build detached event data without SDK objects or artifact readers."""
    return ETDumpEvent(
        method="forward",
        event_block="Execute",
        event_block_index=0,
        event_index=0,
        name=name,
        instruction_id=instruction,
        backend=backend,
        debug_handles=tuple(handles),
        delegate_debug_identifier=delegate_id,
        raw_delegate_metadata=(),
        samples=tuple(samples) if samples is not None else None,
    )


@pytest.fixture
def runtime_mappings():
    """Analyze scoped capture data, retaining aliases and repeated handles."""

    def build(event_blocks, handles=(42, 43), excluded=None):
        nodes = [
            {
                "name": f"conv{index}",
                "debug_handle": handle,
                "target": "aten.convolution.default",
                "category": "conv",
            }
            for index, handle in enumerate(handles)
        ]
        capture = ProfilingCapture(
            method="forward",
            etrecord_path="model.etrecord",
            etdump_path="run.etdump",
            events=tuple(
                replace(event, event_block_index=block_index, event_index=index)
                for block_index, events in enumerate(event_blocks)
                for index, event in enumerate(events)
            ),
            nodes_by_handle={
                handle: [node for node in nodes if node["debug_handle"] == handle]
                for handle in handles
            },
            instruction_handles={},
            source_graph={},
        )
        return build_standard_runtime_mappings(capture, excluded)

    return build


def _project_runtime_observations(observations, source_ids=None):
    """Project runtime measurements to a containing group via the core schema."""
    result = schema.Result(
        kind=schema.ResultKind.PERFORMANCE,
        status=schema.ResultStatus.OK,
        producer="executorch",
        entities=[
            schema.Entity(
                id="group",
                kind="runtime_group",
                name="Group",
                child_ids=source_ids
                if source_ids is not None
                else [entity.id for entity in observations.entities],
            ),
            *observations.entities,
        ],
        breakdowns=observations.breakdowns,
        entity_kinds=[
            schema.EntityKind(id="runtime_group", child_kinds=["source_operator"]),
            schema.EntityKind(id="source_operator", parent_kinds=["runtime_group"]),
        ],
    )
    return project_entity_breakdowns(result)


@pytest.fixture
def runtime_result(runtime_mappings):
    """Analyze source timings and project them to a containing runtime group."""

    def analyze(events):
        mappings = runtime_mappings(
            [
                [
                    _runtime_event(
                        "delegate_op",
                        [handle],
                        samples,
                        instruction=instruction,
                        backend=backend,
                        delegate_id=instruction if backend is not None else None,
                    )
                    for backend, instruction, handle, samples in events
                ]
            ]
        )
        observations = standard_runtime_observations(
            mappings, "forward", "test capture"
        )
        return observations, _project_runtime_observations(observations)

    return analyze


@pytest.mark.parametrize(
    "samples",
    [
        [[1, 1, 100], [100, 1, 1]],
        [[1, 100], [100, 1]],
    ],
)
def test_runtime_statistics_use_per_execution_totals(runtime_mappings, samples):
    """Percentiles and extrema describe the whole operator in each execution."""
    mappings = runtime_mappings(
        [
            [
                _runtime_event(f"op{i}", [42], values, instruction=i, backend=None)
                for i, values in enumerate(samples)
            ]
        ],
        handles=(42,),
    )
    totals = [sum(run) for run in zip(*samples, strict=True)]
    assert {metric["name"]: metric["value"] for metric in mappings[0]["metrics"]} == {
        "Runtime Total Avg": statistics.mean(totals),
        "Runtime Total P50": statistics.median(totals),
        "Runtime Total Min": min(totals),
        "Runtime Total Max": max(totals),
        "Runtime Event Count": 2,
    }


def test_runtime_statistics_keep_separate_event_blocks(runtime_mappings):
    """Distinct input cohorts contribute observations, not additive averages."""
    mappings = runtime_mappings(
        [
            [
                _runtime_event("op0", [42], [1, 1], backend=None),
                _runtime_event("op1", [42], [2, 2], instruction=1, backend=None),
            ],
            [_runtime_event("op0", [42], [100], backend=None)],
        ],
        handles=(42,),
    )
    metrics = {metric["name"]: metric["value"] for metric in mappings[0]["metrics"]}
    assert metrics["Runtime Total Avg"] == pytest.approx(106 / 3)
    assert metrics["Runtime Total P50"] == 3
    assert metrics["Runtime Total Min"] == 3
    assert metrics["Runtime Total Max"] == 100


def test_runtime_statistics_reject_unaligned_samples(runtime_mappings):
    """Incomplete per-event runs cannot produce trustworthy operator totals."""
    with pytest.raises(ValueError, match="align.*runtime samples"):
        runtime_mappings(
            [
                [
                    _runtime_event("op0", [42], [1, 2], backend=None),
                    _runtime_event("op1", [42], [3], instruction=1, backend=None),
                ]
            ],
            handles=(42,),
        )


@pytest.mark.parametrize("name", ["fused", "DELEGATE_CALL"])
def test_runtime_partial_mappings_retain_non_additive_timings(runtime_mappings, name):
    """Retain the interval while making unresolved attribution visible."""
    mappings = runtime_mappings(
        [[_runtime_event(name, [42, 43], [3, 3], delegate_id=7)]],
        handles=(42,),
    )
    assert len(mappings) == 1
    mapping = mappings[0]
    assert mapping["unresolved_debug_handles"] == [43]
    assert mapping["metrics"][0]["value"] == 3
    assert all("aggregation" not in metric for metric in mapping["metrics"])
    observations = standard_runtime_observations(mappings, "forward", "test")
    assert observations.entities[0].attributes["runtime_unresolved_debug_handles"] == [
        43
    ]
    assert all(
        metric.aggregation is None
        for breakdown in observations.breakdowns
        for metric in breakdown.metrics
    )


@pytest.mark.parametrize("handles", [(42, 43), (42, 42)])
def test_shared_runtime_events_are_non_additive(runtime_mappings, handles):
    """One observation associated with two source nodes is not two timings."""
    mappings = runtime_mappings(
        [[_runtime_event("fused", list(set(handles)), [3, 3], delegate_id=7)]],
        handles=handles,
    )
    assert len(mappings) == 2
    source_ids = [source_operator_id("forward", f"conv{i}") for i in range(2)]
    for mapping in mappings:
        metrics = {metric["name"]: metric for metric in mapping["metrics"]}
        assert metrics["Runtime Total Avg"]["value"] == 3
        assert metrics["Runtime Event Count"]["value"] == 1
        assert all("aggregation" not in metric for metric in metrics.values())
        assert mapping["shared_source_operator_ids"] == source_ids
    assert mappings[0]["observation_ids"] == mappings[1]["observation_ids"]
    observations = standard_runtime_observations(mappings, "forward", "test")
    for entity in observations.entities:
        assert entity.attributes["runtime_shared_source_operator_ids"] == source_ids
        assert (
            entity.attributes["runtime_observation_ids"]
            == mappings[0]["observation_ids"]
        )


def test_runtime_repeated_handle_does_not_duplicate_measurement(
    runtime_mappings,
):
    mappings = runtime_mappings(
        [[_runtime_event("conv", [42, 42], [3, 3])]], handles=(42,)
    )
    metrics = {metric["name"]: metric for metric in mappings[0]["metrics"]}
    assert metrics["Runtime Total Avg"] == {
        "name": "Runtime Total Avg",
        "value": 3,
        "unit": "ms",
        "aggregation": "sum",
        "qualifiers": {
            "etdump_execution_cohort": [{"block_index": 0, "sample_count": 2}]
        },
    }
    assert metrics["Runtime Event Count"]["value"] == 1


def test_runtime_shared_and_independent_samples_remain_non_additive(
    runtime_mappings,
):
    mappings = runtime_mappings(
        [
            [
                _runtime_event("fused", [42, 43], [3, 3]),
                _runtime_event("conv", [42], [2, 2], instruction=1),
            ]
        ]
    )
    averages = []
    for mapping in mappings:
        metrics = {metric["name"]: metric for metric in mapping["metrics"]}
        averages.append(metrics["Runtime Total Avg"]["value"])
        assert "aggregation" not in metrics["Runtime Total Avg"]
        assert "aggregation" not in metrics["Runtime Event Count"]
    assert averages == [5, 3]


@pytest.mark.parametrize("detail_first", [False, True])
def test_runtime_delegate_wrapper_yields_to_matching_detail(
    runtime_mappings, detail_first
):
    events = [
        _runtime_event("DELEGATE_CALL", [42], [3, 3]),
        _runtime_event("conv", [42], [2, 2], delegate_id=7),
    ]
    mappings = runtime_mappings([events[::-1] if detail_first else events])
    metrics = {metric["name"]: metric for metric in mappings[0]["metrics"]}
    assert metrics["Runtime Total Avg"]["value"] == 2
    assert metrics["Runtime Event Count"]["value"] == 1
    assert metrics["Runtime Total Avg"]["aggregation"] == "sum"


@pytest.mark.parametrize("detail", ["absent", "no-timing", "unmapped", "framework"])
def test_runtime_delegate_wrapper_retains_fallback(runtime_mappings, detail):
    events = [_runtime_event("DELEGATE_CALL", [42], [3, 3])]
    if detail != "absent":
        events.append(
            _runtime_event(
                "Method::execute" if detail == "framework" else "conv",
                [99] if detail == "unmapped" else [42],
                None if detail == "no-timing" else [2, 2],
                delegate_id=7,
            )
        )
    mappings = runtime_mappings([events], handles=(42,))
    metrics = {metric["name"]: metric for metric in mappings[0]["metrics"]}
    assert metrics["Runtime Total Avg"]["value"] == 3
    assert metrics["Runtime Event Count"]["value"] == 1
    observations = standard_runtime_observations(mappings, "forward", "test")
    assert (
        observations.entities[0].attributes["runtime_delegate_wrapper_fallback"] is True
    )


@pytest.mark.parametrize(
    "difference",
    ["block", "instruction", "backend", "not-delegated", "unknown-instruction"],
)
def test_runtime_delegate_wrapper_does_not_use_unrelated_detail(
    runtime_mappings, difference
):
    wrapper = _runtime_event(
        "DELEGATE_CALL",
        [42],
        [3, 3],
        instruction=None if difference == "unknown-instruction" else 0,
    )
    detail = _runtime_event(
        "conv",
        [43],
        [2, 2],
        instruction=1
        if difference == "instruction"
        else None
        if difference == "unknown-instruction"
        else 0,
        backend="OtherBackend" if difference == "backend" else "TestBackend",
        delegate_id=None if difference == "not-delegated" else 7,
    )
    mappings = runtime_mappings(
        [[wrapper], [detail]] if difference == "block" else [[wrapper, detail]]
    )
    assert {
        mapping["node"]["name"]: next(
            metric["value"]
            for metric in mapping["metrics"]
            if metric["name"] == "Runtime Total Avg"
        )
        for mapping in mappings
    } == {"conv0": 3, "conv1": 2}


def test_runtime_wrapper_alignment_and_vgf_exclusion_are_preserved(
    runtime_mappings,
):
    events = [
        _runtime_event("DELEGATE_CALL", [42], [3, 3]),
        _runtime_event("conv", [42], [2], delegate_id=7),
    ]
    with pytest.raises(ValueError, match="align.*runtime samples"):
        runtime_mappings([events])
    assert runtime_mappings([events], excluded={0}) == []


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("backends", [("BackendA", "BackendB"), (None, "BackendA")])
def test_runtime_observations_preserve_multiple_backend_attributions(
    runtime_result, backends: tuple[str | None, str | None], reverse: bool
) -> None:
    """One canonical source node retains all backend timings and instruction IDs."""
    first, second = backends
    events = [
        (first, 9, 42, [1, 3]),
        (second, 3, 42, [2, 4]),
        (first, 7, 42, [5, 7]),
    ]
    observations, projected = runtime_result(
        list(reversed(events)) if reverse else events
    )
    assert len(observations.entities) == 1
    attributes = observations.entities[0].attributes
    assert attributes["runtime_attributions"] == [
        {"backend": first, "instruction_ids": [7, 9]},
        {"backend": second, "instruction_ids": [3]},
    ]
    assert attributes["runtime_instruction_ids"] == [3, 7, 9]
    assert "runtime_backend" not in attributes
    assert [breakdown.qualifiers for breakdown in observations.breakdowns] == [
        {
            "etrecord_method": "forward",
            **({"runtime_backend": backend} if backend is not None else {}),
        }
        for backend in backends
    ]
    group_averages = {
        breakdown.qualifiers.get("runtime_backend"): metric.value
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "group"
        for metric in breakdown.metrics
        if metric.name == "Runtime Total Avg"
    }
    assert group_averages == {first: 8, second: 3}


@pytest.mark.parametrize("backend", [None, "BackendA"])
def test_runtime_projection_combines_different_instructions_within_one_backend(
    runtime_result, backend: str | None
) -> None:
    """Instruction attribution does not obstruct independent operator totals."""
    observations, projected = runtime_result(
        [(backend, 9, 42, [1, 3]), (backend, 3, 43, [2, 4])]
    )
    assert [entity.id for entity in observations.entities] == [
        "source_operator/etrecord/forward/conv0",
        "source_operator/etrecord/forward/conv1",
    ]
    for breakdown in observations.breakdowns:
        assert all(
            metric.unit == "ms"
            for metric in breakdown.metrics
            if metric.name.startswith("Runtime Total")
        )
    for entity, instruction in zip(observations.entities, (9, 3), strict=True):
        assert entity.attributes["runtime_attributions"] == [
            {"backend": backend, "instruction_ids": [instruction]}
        ]
        assert entity.attributes["runtime_instruction_ids"] == [instruction]
        assert entity.attributes.get("runtime_backend") == backend
    group_breakdown = next(
        breakdown
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "group"
    )
    assert group_breakdown.qualifiers == {
        "etrecord_method": "forward",
        **({"runtime_backend": backend} if backend is not None else {}),
    }
    assert {metric.name: metric.value for metric in group_breakdown.metrics} == {
        "Runtime Total Avg": 5,
        "Runtime Event Count": 2,
    }


@pytest.mark.parametrize(
    ("source_handles", "detail_handles", "detail_first"),
    [
        ((42, 43), [], False),
        ((42, 43), [[42]], False),
        ((42, 43), [[42]], True),
        ((42, 43), [[42], [43]], False),
        ((42, 43), [[42, 43]], False),
        ((42, 43), [[99]], False),
        ((42, 42), [], False),
        ((42, 42), [[42]], False),
    ],
    ids=[
        "no-detail",
        "partial-first-handle",
        "partial-first-handle-detail-first",
        "separate-complete-detail",
        "shared-complete-detail",
        "unmapped-detail",
        "aliased-handle-without-detail",
        "aliased-handle-with-detail",
    ],
)
def test_runtime_wrapper_fallback_preserves_uncovered_handles(
    runtime_mappings,
    source_handles: tuple[int, int],
    detail_first: bool,
    detail_handles: list[list[int]],
) -> None:
    """A shared wrapper supplies fallback only for handles without mapped detail."""
    wrapper = _runtime_event(
        "DELEGATE_CALL",
        sorted(set(source_handles)),
        [10.0, 20.0],
        backend="ExampleBackend",
    )
    details = [
        _runtime_event(
            "delegate_op",
            handles,
            [1.0, 3.0],
            backend="ExampleBackend",
            delegate_id=index,
        )
        for index, handles in enumerate(detail_handles)
    ]
    mappings = runtime_mappings(
        [[*details, wrapper] if detail_first else [wrapper, *details]],
        handles=source_handles,
    )
    assert [mapping["node"]["name"] for mapping in mappings] == ["conv0", "conv1"]
    observations = standard_runtime_observations(mappings, "forward", "test capture")
    for mapping, breakdown in zip(mappings, observations.breakdowns, strict=True):
        handle = mapping["node"]["debug_handle"]
        fallback = not any(handle in handles for handles in detail_handles)
        metrics = {metric["name"]: metric for metric in mapping["metrics"]}
        assert metrics["Runtime Total Avg"]["value"] == (15.0 if fallback else 2.0)
        assert metrics["Runtime Total P50"]["value"] == (15.0 if fallback else 2.0)
        assert metrics["Runtime Event Count"]["value"] == 1
        assert mapping.get("delegate_wrapper_fallback", False) == fallback
        if fallback or source_handles == (42, 42):
            # Wrapper fallback and aliased nodes share one observation.
            # Removing covered handles cannot make that interval additive.
            assert all("aggregation" not in metric for metric in metrics.values())
            assert all(metric.aggregation is None for metric in breakdown.metrics)
            assert mapping["shared_source_operator_ids"] == [
                "source_operator/etrecord/forward/conv0",
                "source_operator/etrecord/forward/conv1",
            ]


@pytest.mark.parametrize(
    ("blocks", "expected_average"),
    [
        ([[(42, [1, 1]), (43, [2, 2])]], 3),
        ([[(42, [1, 1]), (43, [2, 2])], [(42, [4]), (43, [8])]], 6),
        ([[(42, [1])], [(43, [2])]], None),
        ([[(42, [1]), (43, [2])], [(42, [5])]], None),
        ([[(42, [1, 1]), (43, [2])]], None),
        ([[(42, [1, 1]), (43, [2])], [(42, [4]), (43, [8, 8])]], None),
        ([[(42, [1]), (43, [2])], [(44, [9])]], 3),
    ],
    ids=[
        "aligned",
        "aligned-pooled-blocks",
        "disjoint-blocks",
        "partially-shared-blocks",
        "different-sample-counts",
        "equal-totals-different-block-weights",
        "unrelated-operator-with-different-coverage",
    ],
)
def test_runtime_average_projection_requires_matching_execution_coverage(
    runtime_mappings, blocks, expected_average
):
    """Only averages measured over the same executions can form a group total."""
    mappings = runtime_mappings(
        [
            [
                _runtime_event(
                    f"op{handle}",
                    [handle],
                    samples,
                    instruction=handle,
                    backend=None,
                )
                for handle, samples in events
            ]
            for events in blocks
        ],
        handles=(42, 43, 44),
    )
    observations = standard_runtime_observations(mappings, "forward", "review capture")
    projected = _project_runtime_observations(
        observations,
        [source_operator_id("forward", f"conv{i}") for i in (0, 1)],
    )
    group_metrics = {
        metric.name: metric.value
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "group"
        for metric in breakdown.metrics
    }
    assert group_metrics.get("Runtime Total Avg") == expected_average
    assert group_metrics["Runtime Event Count"] == sum(
        handle in (42, 43) for block in blocks for handle, _ in block
    )
    for mapping in mappings:
        average = next(
            m for m in mapping["metrics"] if m["name"] == "Runtime Total Avg"
        )
        assert average["qualifiers"]["etdump_execution_cohort"] == [
            {"block_index": block_index, "sample_count": len(samples)}
            for block_index, events in enumerate(blocks)
            for handle, samples in events
            if handle == mapping["node"]["debug_handle"]
        ]
