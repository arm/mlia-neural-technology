# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Advice generation for NX Performance Estimator results."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from functools import partial
from math import prod
from pathlib import Path
from typing import TypeAlias

import mlia.core.output_schema as schema
from mlia.backend.nx_performance_estimator.vgf import for_each_vgf_op

_OP_EXT_INST_IMPORT = 11
_OP_EXT_INST = 12
_OP_TYPE_INT = 21
_OP_TYPE_FLOAT = 22
_OP_TYPE_ARRAY = 28
_OP_CONSTANT = 43
_OP_CONSTANT_COMPOSITE = 44
_OP_TYPE_TENSOR_ARM = 4163
_TOSA_CONV2D = 2
_TOSA_DEPTHWISE_CONV2D = 4
_TOSA_INSTRUCTION_SET = "TOSA.001000.1"
_SKIP_CONNECTION_SIZE_THRESHOLD = 128 * 1024
# Tensor-producing instructions relevant to constant and computed weights:
# constants, loads, composite extracts, copies, graph constants/inputs, and TOSA.
_TENSOR_VALUE_OPCODES = {12, 43, 44, 46, 61, 81, 83, 4181, 4184, 4461}


@dataclass
class NXAdviceRecord:
    """A single advice record."""

    id: str
    category: str
    severity: str
    message: str
    entity_id: str | None = None


_SpirvId: TypeAlias = int


@dataclass
class _SpirvSegmentState:
    """A helper struct for SPIR-V module lookups while traversals."""

    tosa_sets: set[_SpirvId] = field(default_factory=set)
    integer_types: dict[_SpirvId, tuple[int, bool]] = field(default_factory=dict)
    element_bit_widths: dict[_SpirvId, int] = field(default_factory=dict)
    integers: dict[_SpirvId, int] = field(default_factory=dict)
    shape_array_lengths: dict[_SpirvId, int] = field(default_factory=dict)
    shapes: dict[_SpirvId, tuple[int, ...]] = field(default_factory=dict)
    tensor_shapes: dict[_SpirvId, tuple[int, ...] | None] = field(default_factory=dict)
    value_shapes: dict[_SpirvId, tuple[int, ...] | None] = field(default_factory=dict)
    tensor_sizes: dict[_SpirvId, int] = field(default_factory=dict)
    value_sizes: dict[_SpirvId, int] = field(default_factory=dict)
    parents: dict[_SpirvId, set[_SpirvId]] = field(default_factory=dict)


def _spirv_op_callback(
    segment_idx: int,
    operation: bytes,
    out_advice: list[NXAdviceRecord],
    segment_state: dict[int, _SpirvSegmentState],
) -> None:
    if len(operation) < 4:
        return
    header = struct.unpack_from("<I", operation)[0]
    word_count = header >> 16
    if word_count * 4 != len(operation):
        return
    opcode = header & 0xFFFF
    # Inspect only the prefix needed for metadata, not weight payloads.
    words = struct.unpack_from(f"<{min(word_count, 15)}I", operation)
    state = segment_state.setdefault(segment_idx, _SpirvSegmentState())

    if opcode == _OP_EXT_INST_IMPORT and word_count >= 3:
        # words: [header, result_id, ...NUL-terminated UTF-8 name packed into words].
        _header, result_id, *_name_words = words
        name = operation[8:].split(b"\0", 1)[0].decode("utf-8", errors="replace")
        if name == _TOSA_INSTRUCTION_SET:
            state.tosa_sets.add(result_id)
    elif opcode == _OP_TYPE_INT and word_count == 4:
        # words: [header, type_id, bit_width, signedness (0=unsigned, 1=signed)].
        _header, type_id, bit_width, signedness = words
        state.integer_types[type_id] = (bit_width, bool(signedness))
        state.element_bit_widths[type_id] = bit_width
    elif opcode == _OP_TYPE_FLOAT and word_count >= 3:
        # words: [header, type_id, bit_width, optional floating-point encoding].
        _header, type_id, bit_width, *_encoding = words
        state.element_bit_widths[type_id] = bit_width
    elif opcode == _OP_CONSTANT and word_count >= 4:
        # words: [header, type_id, result_id, ...literal value words].
        # A 64-bit integer uses two value words, least-significant word first.
        _header, type_id, result_id, *_value_words = words
        integer_type = state.integer_types.get(type_id)
        if integer_type is not None:
            width, signed = integer_type
            if 0 < width <= 64 and word_count == 3 + (width + 31) // 32:
                value = int.from_bytes(operation[12:], "little") & ((1 << width) - 1)
                if signed and value & (1 << (width - 1)):
                    value -= 1 << width
                state.integers[result_id] = value
    elif opcode == _OP_TYPE_ARRAY and word_count == 4:
        # words: [header, type_id, element_type_id, length_constant_id].
        # OpTypeTensorARM shapes are integer arrays, not tensor constants.
        _header, type_id, element_type_id, length_id = words
        length = state.integers.get(length_id)
        if element_type_id in state.integer_types and length is not None and length > 0:
            state.shape_array_lengths[type_id] = length
    elif opcode == _OP_CONSTANT_COMPOSITE and word_count >= 3:
        # words: [header, type_id, result_id, ...constituent_ids].
        _header, type_id, result_id, *_constituent_ids = words
        # Check the shape-array type before traversing operands: quantized tensor
        # constants also contain integer IDs, but their payloads can be enormous.
        if state.shape_array_lengths.get(type_id) == word_count - 3:
            dimensions = []
            for (dimension_id,) in struct.iter_unpack("<I", memoryview(operation)[12:]):
                # Shape operands reference scalar constants, not literal dimensions.
                if dimension_id not in state.integers:
                    break
                dimensions.append(state.integers[dimension_id])
            else:
                state.shapes[result_id] = tuple(dimensions)
    elif opcode == _OP_TYPE_TENSOR_ARM and word_count >= 3:
        # words: [header, tensor_type_id, element_type_id, rank_id?, shape_id?].
        # Rank and shape are optional references to constant instructions.
        _header, tensor_type_id, element_type_id, *shape_operands = words
        shape = None
        if word_count == 5:
            rank_id, shape_id = shape_operands
            candidate_shape = state.shapes.get(shape_id)
            if candidate_shape is not None and state.integers.get(rank_id) == len(
                candidate_shape
            ):
                shape = candidate_shape
        state.tensor_shapes[tensor_type_id] = shape
        bit_width = state.element_bit_widths.get(element_type_id)
        if (
            shape is not None
            and all(dimension > 0 for dimension in shape)
            and bit_width is not None
            and bit_width > 0
        ):
            state.tensor_sizes[tensor_type_id] = (prod(shape) * bit_width + 7) // 8

    if opcode in _TENSOR_VALUE_OPCODES and word_count >= 3:
        # words: [header, result_type_id, result_id, ...opcode-specific operands].
        _header, result_type_id, result_id, *_operands = words
        state.value_shapes[result_id] = state.tensor_shapes.get(result_type_id)
        if result_type_id in state.tensor_sizes:
            state.value_sizes[result_id] = state.tensor_sizes[result_type_id]

    if opcode != _OP_EXT_INST or word_count < 5:
        return

    _header, _result_type_id, result_id, set_id, instruction, *_ = words
    if set_id not in state.tosa_sets:
        return
    # TOSA operands are IDs. Read all of them, including those beyond the prefix
    # used for CONV2D, then discard non-TOSA producers after reading the segment.
    state.parents[result_id] = set(
        struct.unpack_from(f"<{word_count - 5}I", operation, 20)
    )
    if word_count == 15:
        if instruction == _TOSA_CONV2D:
            _conv2d_callback(segment_idx, words, state, out_advice)
        elif instruction == _TOSA_DEPTHWISE_CONV2D:
            _depthwise_conv2d_callback(segment_idx, words, state, out_advice)


def _add_skip_connection_advice(
    segment_idx: int,
    state: _SpirvSegmentState,
    out_advice: list[NXAdviceRecord],
) -> None:
    """Report direct TOSA edges that also have an indirect path to the merge."""
    for op_id, parents in state.parents.items():
        if len(parents) < 2:
            continue
        for parent_id in sorted(parents):
            # Constants and metadata operands are not TOSA operations.
            if parent_id not in state.parents:
                continue
            # The bypassed parent's output stays live along the longer path.
            # Unknown sizes cannot establish that the threshold is exceeded.
            output_size = state.value_sizes.get(parent_id)
            if output_size is None or output_size <= _SKIP_CONNECTION_SIZE_THRESHOLD:
                continue
            # Walk backwards from the other inputs, not from the merge itself:
            # reaching parent_id proves that its direct edge bypasses a path.
            pending = list(parents - {parent_id})
            visited: set[int] = set()
            while pending:
                predecessor_id = pending.pop()
                if predecessor_id not in state.parents:
                    continue
                if predecessor_id == parent_id:
                    out_advice.append(
                        NXAdviceRecord(
                            id=(
                                f"advice/segment_{segment_idx}/spirv_{op_id}"
                                f"/skip_connection_{parent_id}"
                            ),
                            category="performance",
                            severity="info",
                            message=(
                                f"Skip connection detected from SPIR-V operation "
                                f"{parent_id} to {op_id}. "
                                f"Its output is {output_size} bytes (>128 KiB). "
                                "Large skip connections may cause poor cache "
                                "utilization."
                            ),
                            entity_id=schema.vgf_source_operator_id(segment_idx, op_id),
                        )
                    )
                    break
                if predecessor_id not in visited:
                    visited.add(predecessor_id)
                    pending.extend(state.parents[predecessor_id] - visited)


def _conv2d_callback(
    segment_idx: int,
    words: tuple[int, ...],
    state: _SpirvSegmentState,
    out_advice: list[NXAdviceRecord],
) -> None:
    """Append depth-alignment advice for a fifteen-word TOSA CONV2D instruction."""
    # CONV2D words: [header, result_type_id, result_id, set_id, instruction (=2),
    # pad_id, stride_id, dilation_id, acc_type_id, local_bound_id, input_id,
    # weight_id, bias_id, input_zp_id, weight_zp_id]. Weight is at index 11.
    (
        _header,
        _result_type_id,
        result_id,
        set_id,
        instruction,
        _pad_id,
        _stride_id,
        _dilation_id,
        _acc_type_id,
        _local_bound_id,
        _input_id,
        weight_id,
        _bias_id,
        _input_zp_id,
        _weight_zp_id,
    ) = words
    if set_id not in state.tosa_sets or instruction != _TOSA_CONV2D:
        return
    weight_shape = state.value_shapes.get(weight_id)
    if (
        weight_shape is None
        or len(weight_shape) != 4
        or any(dimension <= 0 for dimension in weight_shape)
    ):
        return

    output_channels, kernel_height, kernel_width, input_channels = weight_shape
    kernel_size = kernel_height * kernel_width
    depth_factor = 8
    if kernel_size == 1:
        depth_factor = 64
    elif kernel_size < 4:
        depth_factor = 32
    elif kernel_size < 8:
        depth_factor = 16

    message = ""
    if input_channels % depth_factor != 0:
        message += f"Input depth (currently {input_channels}) should be a multiple of {depth_factor} for maximum hardware utilization. "
    if output_channels % depth_factor != 0:
        message += f"Output depth (currently {output_channels}) should be a multiple of {depth_factor} for maximum hardware utilization. "
    message = message.strip()

    if message:
        message
        advice_id = f"advice/segment_{segment_idx}/spirv_{result_id}"
        out_advice.append(
            NXAdviceRecord(
                id=advice_id,
                category="performance",
                severity="info",
                message=message,
                entity_id=schema.vgf_source_operator_id(segment_idx, result_id),
            )
        )


def _depthwise_conv2d_callback(
    segment_idx: int,
    words: tuple[int, ...],
    state: _SpirvSegmentState,
    out_advice: list[NXAdviceRecord],
) -> None:
    """Suggest regular CONV2D for depthwise convolutions with input depth below four."""
    # DEPTHWISE_CONV2D uses the same operands as CONV2D, but its weight shape
    # is [KH, KW, C, M], where C is input depth and M is the channel multiplier.
    (
        _header,
        _result_type_id,
        result_id,
        set_id,
        instruction,
        _pad_id,
        _stride_id,
        _dilation_id,
        _acc_type_id,
        _local_bound_id,
        _input_id,
        weight_id,
        _bias_id,
        _input_zp_id,
        _weight_zp_id,
    ) = words
    if set_id not in state.tosa_sets or instruction != _TOSA_DEPTHWISE_CONV2D:
        return
    weight_shape = state.value_shapes.get(weight_id)
    if weight_shape is None or len(weight_shape) != 4:
        return
    _kernel_height, _kernel_width, input_depth, _channel_multiplier = weight_shape
    if not 0 < input_depth < 4:
        return
    out_advice.append(
        NXAdviceRecord(
            id=f"advice/segment_{segment_idx}/spirv_{result_id}",
            category="performance",
            severity="info",
            message=(
                "Using depthwise convolution on shallow feature maps. Consider using regular CONV2D for maximum hardware utilization."
            ),
            entity_id=schema.vgf_source_operator_id(segment_idx, result_id),
        )
    )


def add_vgf_performance_advice(
    vgf_path: Path,
    advice: list[NXAdviceRecord],
) -> None:
    """Append advice from VGF."""
    segment_state: dict[int, _SpirvSegmentState] = {}
    callback = partial(
        _spirv_op_callback,
        out_advice=advice,
        segment_state=segment_state,
    )
    for_each_vgf_op(vgf_path, callback)
    for segment_idx, state in segment_state.items():
        _add_skip_connection_advice(segment_idx, state, advice)
