# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for VGF performance advice."""

import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from mlia.backend.nx_performance_estimator.advice import (
    NXAdviceRecord,
    _add_skip_connection_advice,
    _conv2d_callback,
    _spirv_op_callback,
    _SpirvSegmentState,
    add_vgf_performance_advice,
)


@pytest.mark.parametrize("segments", [[], [0], [0, 0, 1]])
def test_performance_advice(segments: list[int]) -> None:
    """Use extended-instruction result IDs and ignore non-operation metadata."""
    path = Path("model.vgf")
    existing = NXAdviceRecord("existing", "performance", "info", "Existing advice")
    assert existing.entity_id is None
    advice = [existing]

    def visit(vgf_path, callback):
        assert vgf_path == path
        for index, segment in enumerate(segments):
            callback(segment, struct.pack("<I", 1 << 16))  # OpNop has no result ID.
            callback(segment, struct.pack("<3I", (3 << 16) | 5, 123, 0))  # OpName
            callback(segment, struct.pack("<5I", (5 << 16) | 12, 1, 449 + index, 10, 6))
            # Import TOSA and define weights with shape [64, 1, 1, 32].
            name = b"TOSA.001000.1\0\0\0"
            callback(segment, struct.pack("<2I", (6 << 16) | 11, 10) + name)
            instructions = [
                ((4 << 16) | 21, 2, 32, 0),  # OpTypeInt
                ((4 << 16) | 43, 2, 3, 4),  # Rank
                ((4 << 16) | 43, 2, 4, 64),  # Output depth
                ((4 << 16) | 43, 2, 5, 1),  # Kernel dimensions
                ((4 << 16) | 43, 2, 6, 32),  # Input depth
                ((4 << 16) | 28, 7, 2, 3),  # Integer array of rank dimensions
                ((7 << 16) | 44, 7, 8, 4, 5, 5, 6),  # Shape
                ((5 << 16) | 4163, 9, 2, 3, 8),  # OpTypeTensorARM
                ((3 << 16) | 46, 9, 50),  # Weight value
                ((15 << 16) | 12, 9, 449 + index, 10, 2, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0),
            ]
            for words in instructions:
                callback(segment, struct.pack(f"<{len(words)}I", *words))

    with patch(
        "mlia.backend.nx_performance_estimator.advice.for_each_vgf_op",
        side_effect=visit,
    ):
        result = add_vgf_performance_advice(path, advice)

    assert result is None
    assert advice[0] is existing
    assert len(advice) == len(segments) + 1


@pytest.mark.parametrize(
    ("dimensions", "expected_message"),
    [
        ((1, 1, 64, 64), ""),
        (
            (1, 1, 32, 64),
            "Input depth (currently 32) should be a multiple of 64 "
            "for maximum hardware utilization.",
        ),
        (
            (1, 1, 64, 32),
            "Output depth (currently 32) should be a multiple of 64 "
            "for maximum hardware utilization.",
        ),
        (
            (1, 1, 32, 48),
            "Input depth (currently 32) should be a multiple of 64 "
            "for maximum hardware utilization. "
            "Output depth (currently 48) should be a multiple of 64 "
            "for maximum hardware utilization.",
        ),
        (
            (2, 1, 16, 32),
            "Input depth (currently 16) should be a multiple of 32 "
            "for maximum hardware utilization.",
        ),
        (
            (1, 3, 32, 16),
            "Output depth (currently 16) should be a multiple of 32 "
            "for maximum hardware utilization.",
        ),
        (
            (2, 2, 8, 16),
            "Input depth (currently 8) should be a multiple of 16 "
            "for maximum hardware utilization.",
        ),
        (
            (3, 2, 16, 8),
            "Output depth (currently 8) should be a multiple of 16 "
            "for maximum hardware utilization.",
        ),
        ((3, 2, 16, 32), ""),
        (
            (4, 2, 4, 8),
            "Input depth (currently 4) should be a multiple of 8 "
            "for maximum hardware utilization.",
        ),
        (
            (3, 3, 8, 4),
            "Output depth (currently 4) should be a multiple of 8 "
            "for maximum hardware utilization.",
        ),
        ((0, 3, 8, 8), ""),
        ((-1, 3, 8, 8), ""),
        ((3, -1, 8, 8), ""),
        ((3, 3, -1, 8), ""),
        ((3, 3, 8, -1), ""),
    ],
)
def test_conv2d_advice(
    dimensions: tuple[int, int, int, int], expected_message: str
) -> None:
    """Check depth-alignment advice for (kw, kh, in_depth, out_depth)."""
    kw, kh, in_depth, out_depth = dimensions
    state = _SpirvSegmentState(
        tosa_sets={10},
        value_shapes={50: (out_depth, kh, kw, in_depth)},
    )
    words = ((15 << 16) | 12, 1, 449, 10, 2, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0)
    advice: list[NXAdviceRecord] = []

    _conv2d_callback(0, words, state, advice)

    assert [record.message for record in advice] == (
        [expected_message] if expected_message else []
    )


@pytest.mark.parametrize(
    ("parents", "output_size", "expected_count"),
    [
        pytest.param({1: set(), 2: {1}, 3: {1, 2}}, 131073, 1, id="skip-connection"),
        pytest.param(
            {1: {99}, 2: {1, 99}, 3: {1, 2, 99}},
            131073,
            1,
            id="skip-with-constant-operands",
        ),
        pytest.param(
            {1: {99}, 2: {1, 99}},
            131073,
            0,
            id="shared-constant-is-not-a-skip",
        ),
        pytest.param(
            {1: set(), 2: {1}, 3: {2}, 4: {1, 3}},
            131073,
            1,
            id="multi-hop-skip-connection",
        ),
        pytest.param({1: set(), 2: {1}, 3: {1, 2}}, 131072, 0, id="at-threshold"),
        pytest.param({1: set(), 2: {1}, 3: {1, 2}}, 131071, 0, id="below-threshold"),
        pytest.param({1: set(), 2: {1}, 3: {1, 2}}, None, 0, id="unknown-size"),
        pytest.param(
            {1: set(), 2: set(), 3: {1, 2}}, 131073, 0, id="independent-parents"
        ),
        pytest.param(
            {1: set(), 2: {1}, 3: {1}, 4: {2, 3}},
            131073,
            0,
            id="shared-ancestor-is-not-a-skip",
        ),
    ],
)
def test_skip_connection_advice(
    parents: dict[int, set[int]], output_size: int | None, expected_count: int
) -> None:
    """Report only large skip tensors, not independent or parallel branches."""
    state = _SpirvSegmentState(
        parents={op_id: set(inputs) for op_id, inputs in parents.items()},
        value_sizes=(
            {op_id: output_size for op_id in parents} if output_size is not None else {}
        ),
    )
    advice: list[NXAdviceRecord] = []

    if output_size is not None:
        state.value_sizes[99] = output_size  # Large constant, not a TOSA result.
    _add_skip_connection_advice(0, state, advice)

    assert state.parents == parents
    assert len(advice) == expected_count
    for record in advice:
        assert "poor cache utilization" in record.message
        assert f"{output_size} bytes" in record.message


@pytest.mark.parametrize(
    ("weight_shape", "expected_count"),
    [
        pytest.param((3, 3, 1, 8), 1, id="depth-1-multiplier-8"),
        pytest.param((3, 3, 2, 1), 1, id="depth-2"),
        pytest.param((3, 3, 3, 1), 1, id="depth-3"),
        pytest.param((3, 3, 4, 1), 0, id="depth-at-threshold"),
        pytest.param((3, 3, 8, 1), 0, id="depth-above-threshold"),
        pytest.param((3, 3, 0, 1), 0, id="zero-depth"),
        pytest.param((3, 3, -1, 1), 0, id="dynamic-depth"),
        pytest.param(None, 0, id="unknown-shape"),
        pytest.param((3, 3, 3), 0, id="invalid-rank"),
    ],
)
def test_depthwise_conv2d_advice(
    weight_shape: tuple[int, ...] | None, expected_count: int
) -> None:
    """Recommend regular CONV2D only for known input depths from one to three."""
    state = _SpirvSegmentState(tosa_sets={10}, value_shapes={50: weight_shape})
    # DEPTHWISE_CONV2D uses instruction 4 and weight operand 50.
    words = ((15 << 16) | 12, 1, 449, 10, 4, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0)
    advice: list[NXAdviceRecord] = []

    _spirv_op_callback(0, struct.pack("<15I", *words), advice, {0: state})

    assert len(advice) == expected_count
    for record in advice:
        assert "Consider using regular CONV2D" in record.message
        assert record.entity_id == "source_operator/segment_0/spirv-449"
