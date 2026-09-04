# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Read graph debug operation labels from SPIR-V modules."""

from __future__ import annotations

import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_SPIRV_MAGIC = 0x07230203
_SPIRV_MAGIC_BYTES = struct.pack("<I", _SPIRV_MAGIC)

_OP_STRING = 7
_OP_EXT_INST_IMPORT = 11
_OP_EXT_INST = 12

_NON_SEMANTIC_GRAPH_DEBUG_INFO = "NonSemantic.Graph.DebugInfo.1"
_TOSA_EXT_INST_IMPORT_NAME = "TOSA.001000.1"
_DEBUG_OPERATION = 2
_DEBUG_OPERATION_NAME_OPERAND_INDEX = 1
_DEBUG_OPERATION_FIRST_INSTRUCTION_OPERAND_INDEX = 2


@dataclass(frozen=True)
class SpirvDebugNameMap:
    """Bidirectional mapping between SPIR-V result ids and debug operation names."""

    spirv_id_to_debug_name: dict[str, str] = field(default_factory=dict)
    debug_name_to_spirv_ids: dict[str, list[str]] = field(default_factory=dict)
    op_ext_inst_spirv_ids: list[str] = field(default_factory=list)

    @property
    def spirv_ids(self) -> list[str]:
        """Return all known graph operation SPIR-V result ids in first-seen order."""
        return list(
            dict.fromkeys([*self.op_ext_inst_spirv_ids, *self.spirv_id_to_debug_name])
        )


def read_vgf_spirv_debug_names(vgf_file: Path) -> SpirvDebugNameMap:
    """Read graph debug operation names from VGF-embedded SPIR-V modules."""
    return read_spirv_debug_names_from_bytes(vgf_file.read_bytes())


def read_spirv_debug_names_from_bytes(data: bytes) -> SpirvDebugNameMap:
    """Read graph debug operation names from one or more SPIR-V byte streams."""
    spirv_id_to_debug_name: dict[str, str] = {}
    op_ext_inst_spirv_ids: list[str] = []
    for module_start in _spirv_module_offsets(data):
        module = data[module_start:]
        word_count = len(module) // 4
        if word_count < 5:
            continue

        words = struct.unpack_from(f"<{word_count}I", module)
        if words[0] != _SPIRV_MAGIC:
            continue

        module_debug_names, module_op_ext_inst_ids = _read_spirv_module_graph_ops(words)
        spirv_id_to_debug_name.update(module_debug_names)
        op_ext_inst_spirv_ids.extend(
            spirv_id
            for spirv_id in module_op_ext_inst_ids
            if spirv_id not in op_ext_inst_spirv_ids
        )

    debug_name_to_spirv_ids: dict[str, list[str]] = defaultdict(list)
    for spirv_id, debug_name in spirv_id_to_debug_name.items():
        debug_name_to_spirv_ids[debug_name].append(spirv_id)

    return SpirvDebugNameMap(
        spirv_id_to_debug_name=spirv_id_to_debug_name,
        debug_name_to_spirv_ids=dict(debug_name_to_spirv_ids),
        op_ext_inst_spirv_ids=op_ext_inst_spirv_ids,
    )


def _spirv_module_offsets(data: bytes) -> list[int]:
    offsets = []
    search_start = 0
    while (offset := data.find(_SPIRV_MAGIC_BYTES, search_start)) >= 0:
        offsets.append(offset)
        search_start = offset + len(_SPIRV_MAGIC_BYTES)
    return offsets


def _read_spirv_module_graph_ops(
    words: Sequence[int],
) -> tuple[dict[str, str], list[str]]:
    strings: dict[int, str] = {}
    debug_instruction_sets: set[int] = set()
    tosa_instruction_sets: set[int] = set()
    spirv_id_to_debug_name: dict[str, str] = {}
    op_ext_inst_spirv_ids: list[str] = []

    instruction_index = 5
    while instruction_index < len(words):
        word = words[instruction_index]
        word_count = word >> 16
        opcode = word & 0xFFFF

        if word_count == 0 or instruction_index + word_count > len(words):
            break

        operands = words[instruction_index + 1 : instruction_index + word_count]
        if opcode == _OP_STRING and len(operands) >= 2:
            strings[operands[0]] = _decode_spirv_string(operands[1:])
        elif opcode == _OP_EXT_INST_IMPORT and len(operands) >= 2:
            import_name = _decode_spirv_string(operands[1:])
            if import_name == _NON_SEMANTIC_GRAPH_DEBUG_INFO:
                debug_instruction_sets.add(operands[0])
            elif import_name == _TOSA_EXT_INST_IMPORT_NAME:
                tosa_instruction_sets.add(operands[0])
        elif opcode == _OP_EXT_INST and len(operands) >= 4:
            result_id = str(operands[1])
            set_id = operands[2]
            instruction = operands[3]
            if (
                set_id in tosa_instruction_sets
                and result_id not in op_ext_inst_spirv_ids
            ):
                op_ext_inst_spirv_ids.append(result_id)
            if (
                set_id in debug_instruction_sets
                and instruction == _DEBUG_OPERATION
                and len(operands) >= 6
            ):
                _record_debug_operation(strings, operands[4:], spirv_id_to_debug_name)

        instruction_index += word_count

    return spirv_id_to_debug_name, op_ext_inst_spirv_ids


def _record_debug_operation(
    strings: dict[int, str],
    debug_operands: Sequence[int],
    spirv_id_to_debug_name: dict[str, str],
) -> None:
    if len(debug_operands) <= _DEBUG_OPERATION_NAME_OPERAND_INDEX:
        return

    debug_name_id = debug_operands[_DEBUG_OPERATION_NAME_OPERAND_INDEX]
    debug_name = strings.get(debug_name_id)
    if not debug_name:
        return

    for spirv_id in debug_operands[_DEBUG_OPERATION_FIRST_INSTRUCTION_OPERAND_INDEX:]:
        if spirv_id not in strings:
            spirv_id_to_debug_name.setdefault(str(spirv_id), debug_name)


def _decode_spirv_string(words: Sequence[int]) -> str:
    raw_string = struct.pack(f"<{len(words)}I", *words)
    return raw_string.split(b"\0", 1)[0].decode("utf-8", errors="replace")
