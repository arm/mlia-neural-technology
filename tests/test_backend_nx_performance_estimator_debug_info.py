# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NX Performance Estimator SPIR-V debug info helpers."""

from __future__ import annotations

import struct
from pathlib import Path

from mlia.backend.nx_performance_estimator.debug_info import (
    read_spirv_debug_names_from_bytes,
    read_vgf_spirv_debug_names,
)


def _spirv_string_instruction(result_id: int, value: str) -> list[int]:
    encoded = value.encode("utf-8") + b"\0"
    padding = b"\0" * ((4 - len(encoded) % 4) % 4)
    word_count = (len(encoded) + len(padding)) // 4
    words = list(struct.unpack(f"<{word_count}I", encoded + padding))
    return [((2 + len(words)) << 16) | 7, result_id, *words]


def _spirv_ext_inst_import_instruction(result_id: int, value: str) -> list[int]:
    encoded = value.encode("utf-8") + b"\0"
    padding = b"\0" * ((4 - len(encoded) % 4) % 4)
    word_count = (len(encoded) + len(padding)) // 4
    words = list(struct.unpack(f"<{word_count}I", encoded + padding))
    return [((2 + len(words)) << 16) | 11, result_id, *words]


def _spirv_ext_inst_instruction(
    result_type: int,
    result_id: int,
    set_id: int,
    instruction: int,
    operands: list[int],
) -> list[int]:
    return [
        ((5 + len(operands)) << 16) | 12,
        result_type,
        result_id,
        set_id,
        instruction,
        *operands,
    ]


def _spirv_module(*instructions: list[int]) -> bytes:
    words = [0x07230203, 0x00010500, 0, 1000, 0]
    for instruction in instructions:
        words.extend(instruction)
    return struct.pack(f"<{len(words)}I", *words)


def test_read_spirv_debug_names_reads_debug_operation_names() -> None:
    """Graph debug operation names map to referenced SPIR-V result ids."""
    data = _spirv_module(
        _spirv_ext_inst_import_instruction(10, "NonSemantic.Graph.DebugInfo.1"),
        _spirv_string_instruction(20, "model/re_lu_6/Relu"),
        _spirv_ext_inst_instruction(
            result_type=1,
            result_id=30,
            set_id=10,
            instruction=2,
            operands=[0, 20, 411, 412],
        ),
    )

    debug_names = read_spirv_debug_names_from_bytes(data)

    assert debug_names.spirv_id_to_debug_name == {
        "411": "model/re_lu_6/Relu",
        "412": "model/re_lu_6/Relu",
    }
    assert debug_names.debug_name_to_spirv_ids == {"model/re_lu_6/Relu": ["411", "412"]}
    assert debug_names.op_ext_inst_spirv_ids == []
    assert debug_names.spirv_ids == ["411", "412"]


def test_read_spirv_debug_names_reads_tosa_op_ext_inst_result_ids() -> None:
    """TOSA OpExtInst result ids are captured even without graph debug names."""
    data = _spirv_module(
        _spirv_ext_inst_import_instruction(10, "TOSA.001000.1"),
        _spirv_ext_inst_instruction(
            result_type=1,
            result_id=449,
            set_id=10,
            instruction=42,
            operands=[1, 2, 3],
        ),
    )

    debug_names = read_spirv_debug_names_from_bytes(data)

    assert debug_names.op_ext_inst_spirv_ids == ["449"]
    assert debug_names.spirv_id_to_debug_name == {}
    assert debug_names.debug_name_to_spirv_ids == {}
    assert debug_names.spirv_ids == ["449"]


def test_read_spirv_debug_names_combines_tosa_result_ids_and_debug_names() -> None:
    """SPIR-V ids keep debug names when present and stay structural otherwise."""
    data = _spirv_module(
        _spirv_ext_inst_import_instruction(10, "TOSA.001000.1"),
        _spirv_ext_inst_import_instruction(11, "NonSemantic.Graph.DebugInfo.1"),
        _spirv_string_instruction(20, "model/cat"),
        _spirv_ext_inst_instruction(1, 486, 10, 42, [1, 2, 3]),
        _spirv_ext_inst_instruction(1, 449, 10, 43, [1, 2, 3]),
        _spirv_ext_inst_instruction(1, 30, 11, 2, [0, 20, 486]),
    )

    debug_names = read_spirv_debug_names_from_bytes(data)

    assert debug_names.op_ext_inst_spirv_ids == ["486", "449"]
    assert debug_names.spirv_id_to_debug_name == {"486": "model/cat"}
    assert debug_names.debug_name_to_spirv_ids == {"model/cat": ["486"]}
    assert debug_names.spirv_ids == ["486", "449"]


def test_read_spirv_debug_names_ignores_modules_without_graph_debug_import() -> None:
    """No graph debug import means no debug-name mapping."""
    data = _spirv_module(
        _spirv_ext_inst_import_instruction(10, "Other.DebugInfo.1"),
        _spirv_string_instruction(20, "model/re_lu_6/Relu"),
        _spirv_ext_inst_instruction(
            result_type=1,
            result_id=30,
            set_id=10,
            instruction=2,
            operands=[0, 20, 411],
        ),
    )

    debug_names = read_spirv_debug_names_from_bytes(data)

    assert debug_names.spirv_id_to_debug_name == {}
    assert debug_names.debug_name_to_spirv_ids == {}


def test_read_vgf_spirv_debug_names_reads_multiple_embedded_modules(
    tmp_path: Path,
) -> None:
    """VGF byte streams may contain multiple SPIR-V modules."""
    first_module = _spirv_module(
        _spirv_ext_inst_import_instruction(10, "NonSemantic.Graph.DebugInfo.1"),
        _spirv_string_instruction(20, "first"),
        _spirv_ext_inst_instruction(1, 30, 10, 2, [0, 20, 411]),
    )
    second_module = _spirv_module(
        _spirv_ext_inst_import_instruction(10, "NonSemantic.Graph.DebugInfo.1"),
        _spirv_string_instruction(20, "second"),
        _spirv_ext_inst_instruction(1, 30, 10, 2, [0, 20, 512]),
    )
    vgf_file = tmp_path / "model.vgf"
    vgf_file.write_bytes(b"VGF1\0\0\0\0" + first_module + b"padding" + second_module)

    debug_names = read_vgf_spirv_debug_names(vgf_file)

    assert debug_names.spirv_id_to_debug_name == {"411": "first", "512": "second"}
    assert debug_names.debug_name_to_spirv_ids == {"first": ["411"], "second": ["512"]}
