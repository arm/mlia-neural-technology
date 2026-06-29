# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Read debug location metadata from VGF-embedded SPIR-V modules."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

from mlia.nx_utils.debug_metadata import normalize_debug_location

_SPIRV_MAGIC = 0x07230203
_SPIRV_MAGIC_BYTES = struct.pack("<I", _SPIRV_MAGIC)

_OP_STRING = 7
_OP_EXT_INST_IMPORT = 11
_OP_EXT_INST = 12

_MLGRAPH_DEBUG_IMPORT = "NonSemantic.MLGraph.DebugInfo.1"
_MLGRAPH_DEBUG_OPERATION = 4


def read_vgf_spirv_id_locations(vgf_file: Path) -> dict[str, str]:
    """Read mapping from SPIR-V result ids to original model locations."""
    data = vgf_file.read_bytes()
    spirv_id_locations: dict[str, str] = {}

    for module_start in _spirv_module_offsets(data):
        module = data[module_start:]
        word_count = len(module) // 4
        if word_count < 5:
            continue

        words = struct.unpack_from(f"<{word_count}I", module)
        if words[0] != _SPIRV_MAGIC:
            continue

        for spirv_id, location in _read_spirv_id_locations(words).items():
            spirv_id_locations.setdefault(spirv_id, location)

    return spirv_id_locations


def _spirv_module_offsets(data: bytes) -> list[int]:
    offsets = []
    search_start = 0
    while (offset := data.find(_SPIRV_MAGIC_BYTES, search_start)) >= 0:
        offsets.append(offset)
        search_start = offset + len(_SPIRV_MAGIC_BYTES)
    return offsets


def _read_spirv_id_locations(words: Sequence[int]) -> dict[str, str]:
    """Read MLGraph debug operation records from a SPIR-V word stream."""
    strings: dict[int, str] = {}
    mlgraph_debug_imports: set[int] = set()
    spirv_id_locations: dict[str, str] = {}

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
            if import_name == _MLGRAPH_DEBUG_IMPORT:
                mlgraph_debug_imports.add(operands[0])
        elif opcode == _OP_EXT_INST and len(operands) >= 5:
            set_id = operands[2]
            instruction = operands[3]
            ext_operands = operands[4:]
            if (
                set_id in mlgraph_debug_imports
                and instruction == _MLGRAPH_DEBUG_OPERATION
            ):
                _record_mlgraph_debug_operation(
                    strings, ext_operands, spirv_id_locations
                )

        instruction_index += word_count

    return spirv_id_locations


def _decode_spirv_string(words: Sequence[int]) -> str:
    raw_string = struct.pack(f"<{len(words)}I", *words)
    return raw_string.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _record_mlgraph_debug_operation(
    strings: dict[int, str],
    operands: Sequence[int],
    spirv_id_locations: dict[str, str],
) -> None:
    for operand_index, operand in enumerate(operands):
        location = normalize_debug_location(strings.get(operand))
        if not location:
            continue

        for spirv_id in operands[operand_index + 1 :]:
            if spirv_id not in strings:
                spirv_id_locations.setdefault(str(spirv_id), location)
        break
