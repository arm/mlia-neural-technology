# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""VGF helpers for the NX performance estimator."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import vgfpy  # type: ignore[import-untyped]

from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.debug_info import read_vgf_spirv_debug_names
from mlia.backend.nx_performance_estimator.provenance import (
    source_provenance_from_api_label,
)
import mlia.core.output_schema as schema
from mlia.utils.misc import summarize_list

logger = logging.getLogger(__name__)

COMPUTE_SEGMENTS_SKIPPED_WARNING = (
    "VGF compute segment(s) {segments} will not be estimated by the NX "
    "Performance Estimator and will not contribute to performance totals."
)


@dataclass(frozen=True)
class GCPEVGFSegment:
    """A VGF input that GCPE can process for one original VGF segment."""

    segment_index: int
    segment_name: str
    path: Path
    debug_names: SpirvDebugNameMap
    structural_source_operator_ids: list[str]
    structural_source_operator_names: dict[str, str] = field(default_factory=dict)
    spirv: bytes = b""


class GCPEVGFSegments(list[GCPEVGFSegment]):
    """Prepared GCPE VGF segment inputs and skipped segment metadata."""

    def __init__(
        self,
        segments: list[GCPEVGFSegment],
        skipped_compute_segments: list[int] | None = None,
    ) -> None:
        """Initialize graph segments and original compute-segment indexes."""
        super().__init__(segments)
        self.skipped_compute_segments = skipped_compute_segments or []


def read_vgf_segment_name(path: Path, segment_index: int = 0) -> str:
    """Return the model-sequence name for one VGF segment."""
    decoded = _decode_vgf(memoryview(path.read_bytes()), vgfpy, path)
    segment_count = decoded.sequence.modelSequenceTableSize()
    if segment_index < 0 or segment_index >= segment_count:
        raise ValueError(
            f"VGF segment index {segment_index} is outside the available range "
            f"0..{segment_count - 1}."
        )
    return str(decoded.sequence.getSegmentName(segment_index))


def prepare_gcpe_compatible_vgfs(path: Path, output_dir: Path) -> GCPEVGFSegments:
    """Return single-segment VGF files suitable for GCPE.

    GCPE's VGF command-line path currently asserts that the input VGF contains
    exactly one graph segment. For single-segment VGFs this returns the original
    file. For multi-segment VGFs this decodes the original VGF and writes one
    temporary single-segment wrapper VGF per graph segment.
    """
    return _prepare_gcpe_compatible_vgfs(path, output_dir, vgfpy)


def validate_gcpe_compatible_vgf(path: Path) -> None:
    """Validate that a VGF can be converted into GCPE-compatible inputs."""
    _validate_gcpe_compatible_vgf(path, vgfpy)


def _prepare_gcpe_compatible_vgfs(
    path: Path, output_dir: Path, vgfpy_module: Any
) -> GCPEVGFSegments:
    data = memoryview(path.read_bytes())
    decoded = _decode_vgf(data, vgfpy_module, path)
    segment_count = decoded.sequence.modelSequenceTableSize()

    if segment_count == 1:
        segment_type = decoded.sequence.getSegmentType(0)
        if segment_type == vgfpy_module.ModuleType.Graph:
            debug_names = read_vgf_spirv_debug_names(path)
            return GCPEVGFSegments(
                [
                    GCPEVGFSegment(
                        segment_index=0,
                        segment_name=str(decoded.sequence.getSegmentName(0)),
                        path=path,
                        debug_names=debug_names,
                        structural_source_operator_ids=(
                            _segment_structural_source_operator_ids(0, debug_names)
                        ),
                        structural_source_operator_names=(
                            _segment_structural_source_operator_names(0, debug_names)
                        ),
                        spirv=_segment_spirv(decoded, 0),
                    )
                ]
            )
        elif segment_type == vgfpy_module.ModuleType.Compute:
            _warn_compute_segments_skipped([0])
            return GCPEVGFSegments([], skipped_compute_segments=[0])
        else:
            raise _unsupported_segment_type(path, 0, segment_type)

    output_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    skipped_compute_segments = []
    for segment_index in range(segment_count):
        segment_type = decoded.sequence.getSegmentType(segment_index)
        if segment_type == vgfpy_module.ModuleType.Graph:
            segment_path = output_dir / f"{path.stem}_segment_{segment_index}.vgf"
            _write_single_segment_vgf(
                decoded, segment_index, segment_path, vgfpy_module
            )
            debug_names = read_vgf_spirv_debug_names(segment_path)
            segments.append(
                GCPEVGFSegment(
                    segment_index=segment_index,
                    segment_name=str(decoded.sequence.getSegmentName(segment_index)),
                    path=segment_path,
                    debug_names=debug_names,
                    structural_source_operator_ids=(
                        _segment_structural_source_operator_ids(
                            segment_index, debug_names
                        )
                    ),
                    structural_source_operator_names=(
                        _segment_structural_source_operator_names(
                            segment_index, debug_names
                        )
                    ),
                    spirv=_segment_spirv(decoded, segment_index),
                )
            )
        elif segment_type == vgfpy_module.ModuleType.Compute:
            skipped_compute_segments.append(segment_index)
        else:
            raise _unsupported_segment_type(path, segment_index, segment_type)

    if skipped_compute_segments:
        _warn_compute_segments_skipped(skipped_compute_segments)

    return GCPEVGFSegments(segments, skipped_compute_segments)


def _segment_structural_source_operator_ids(
    segment_index: int, debug_names: SpirvDebugNameMap
) -> list[str]:
    """Return canonical VGF source-operator IDs for one original segment."""
    return [
        schema.vgf_source_operator_id(segment_index, int(spirv_id))
        for spirv_id in debug_names.spirv_ids
    ]


def _segment_structural_source_operator_names(
    segment_index: int, debug_names: SpirvDebugNameMap
) -> dict[str, str]:
    """Return available friendly names keyed by canonical source-operator ID."""
    names = {}
    for spirv_id, debug_name in debug_names.spirv_id_to_debug_name.items():
        if not debug_name:
            continue
        provenance = source_provenance_from_api_label(debug_name)
        names[schema.vgf_source_operator_id(segment_index, int(spirv_id))] = (
            provenance.name or provenance.display_label
        )
    return names


def _validate_gcpe_compatible_vgf(path: Path, vgfpy_module: Any) -> None:
    data = memoryview(path.read_bytes())
    decoded = _decode_vgf(data, vgfpy_module, path)
    segment_count = decoded.sequence.modelSequenceTableSize()

    for segment_index in range(segment_count):
        segment_type = decoded.sequence.getSegmentType(segment_index)
        if segment_type == vgfpy_module.ModuleType.Graph:
            continue
        elif segment_type == vgfpy_module.ModuleType.Compute:
            continue
        else:
            raise _unsupported_segment_type(path, segment_index, segment_type)


def _unsupported_segment_type(
    path: Path, segment_index: int, segment_type: Any
) -> ValueError:
    return ValueError(
        "GCPE currently supports only graph and skipped compute VGF segments; "
        f"'{path}' has segment {segment_index} with type {segment_type!s}."
    )


def _warn_compute_segments_skipped(segment_indexes: list[int]) -> None:
    logger.warning(
        COMPUTE_SEGMENTS_SKIPPED_WARNING.format(
            segments=summarize_list(segment_indexes)
        )
    )


@dataclass
class _DecodedVGF:
    data: memoryview
    header: Any
    modules: Any
    sequence: Any
    resources: Any
    constants: Any


def _decode_vgf(data: memoryview, vgfpy_module: Any, path: Path) -> _DecodedVGF:
    header = _create_header_decoder(data, vgfpy_module)
    if header is None or not header.IsValid():
        raise ValueError(f"Invalid VGF file: {path}")

    modules = _create_section_decoder(
        vgfpy_module.CreateModuleTableDecoder,
        data[header.GetModuleTableOffset() :],
        header.GetModuleTableSize(),
    )
    sequence = _create_section_decoder(
        vgfpy_module.CreateModelSequenceTableDecoder,
        data[header.GetModelSequenceTableOffset() :],
        header.GetModelSequenceTableSize(),
    )
    resources = _create_section_decoder(
        vgfpy_module.CreateModelResourceTableDecoder,
        data[header.GetModelResourceTableOffset() :],
        header.GetModelResourceTableSize(),
    )
    constants = _create_section_decoder(
        vgfpy_module.CreateConstantDecoder,
        data[header.GetConstantsOffset() :],
        header.GetConstantsSize(),
    )

    if sequence is None:
        raise ValueError(f"Invalid VGF model sequence table: {path}")
    if modules is None:
        raise ValueError(f"Invalid VGF module table: {path}")
    if resources is None:
        raise ValueError(f"Invalid VGF model resource table: {path}")
    if constants is None:
        raise ValueError(f"Invalid VGF constant table: {path}")

    return _DecodedVGF(data, header, modules, sequence, resources, constants)


def _create_header_decoder(data: memoryview, vgfpy_module: Any) -> Any:
    try:
        return vgfpy_module.CreateHeaderDecoder(data)
    except TypeError:
        return vgfpy_module.CreateHeaderDecoder(
            data, vgfpy_module.HeaderSize(), len(data)
        )


def _create_section_decoder(factory: Any, data: memoryview, size: int) -> Any:
    section = data[:size]
    try:
        return factory(section)
    except TypeError:
        return factory(section, size)


def _segment_spirv(decoded: _DecodedVGF, segment_index: int) -> bytes:
    """Return the exact little-endian SPIR-V bytes for one decoded graph segment."""
    module_index = decoded.sequence.getSegmentModuleIndex(segment_index)
    if not decoded.modules.isSPIRV(module_index) or not decoded.modules.hasSPIRVCode(
        module_index
    ):
        raise ValueError(f"VGF segment {segment_index} does not contain SPIR-V code")
    return _spirv_words_to_bytes(decoded.modules.getSPIRVModuleCode(module_index))


def _write_single_segment_vgf(
    decoded: _DecodedVGF, segment_index: int, output_path: Path, vgfpy_module: Any
) -> None:
    if not hasattr(vgfpy_module, "CreateEncoder"):
        raise RuntimeError(
            "vgfpy does not provide the encoder API required to split VGFs"
        )

    encoder = vgfpy_module.CreateEncoder(
        decoded.header.GetEncoderVulkanHeadersVersion()
    )
    module_index = decoded.sequence.getSegmentModuleIndex(segment_index)
    if not decoded.modules.isSPIRV(module_index) or not decoded.modules.hasSPIRVCode(
        module_index
    ):
        raise ValueError(f"VGF segment {segment_index} does not contain SPIR-V code")

    module = encoder.AddModule(
        decoded.sequence.getSegmentType(segment_index),
        str(decoded.modules.getModuleName(module_index)),
        str(decoded.modules.getModuleEntryPoint(module_index)),
        _view_to_list(decoded.modules.getSPIRVModuleCode(module_index)),
    )

    resources: dict[int, Any] = {}
    binding_slots: dict[tuple[int, int], Any] = {}

    def add_resource(mrt_index: int) -> Any:
        if mrt_index in resources:
            return resources[mrt_index]

        category = decoded.resources.getCategory(mrt_index)
        vk_format = decoded.resources.getVkFormat(mrt_index)
        shape = _view_to_list(decoded.resources.getTensorShape(mrt_index))
        strides = _view_to_list(decoded.resources.getTensorStride(mrt_index))

        if category == vgfpy_module.ResourceCategory.Input:
            descriptor_type = decoded.resources.getDescriptorType(mrt_index)
            resource = encoder.AddInputResource(
                descriptor_type, vk_format, shape, strides
            )
        elif category == vgfpy_module.ResourceCategory.Output:
            descriptor_type = decoded.resources.getDescriptorType(mrt_index)
            resource = encoder.AddOutputResource(
                descriptor_type, vk_format, shape, strides
            )
        elif category == vgfpy_module.ResourceCategory.Intermediate:
            descriptor_type = decoded.resources.getDescriptorType(mrt_index)
            resource = encoder.AddIntermediateResource(
                descriptor_type, vk_format, shape, strides
            )
        elif category == vgfpy_module.ResourceCategory.Constant:
            resource = encoder.AddConstantResource(vk_format, shape, strides)
        else:
            raise ValueError(f"Unsupported VGF resource category {category!s}")

        resources[mrt_index] = resource
        return resource

    def copy_binding_slot(binding_handle: Any, slot_index: int) -> Any:
        binding = decoded.sequence.getBindingSlotBinding(binding_handle, slot_index)
        mrt_index = decoded.sequence.getBindingSlotMrtIndex(binding_handle, slot_index)
        key = (binding, mrt_index)
        if key not in binding_slots:
            binding_slots[key] = encoder.AddBindingSlot(
                binding, add_resource(mrt_index)
            )
        return binding_slots[key]

    descriptor_refs = []
    for descriptor_index in range(
        decoded.sequence.getSegmentDescriptorSetInfosSize(segment_index)
    ):
        handle = decoded.sequence.getDescriptorBindingSlotsHandle(
            segment_index, descriptor_index
        )
        descriptor_refs.append(
            encoder.AddDescriptorSetInfo(
                [
                    copy_binding_slot(handle, slot_index)
                    for slot_index in range(decoded.sequence.getBindingsSize(handle))
                ]
            )
        )

    input_handle = decoded.sequence.getSegmentInputBindingSlotsHandle(segment_index)
    input_refs = [
        copy_binding_slot(input_handle, slot_index)
        for slot_index in range(decoded.sequence.getBindingsSize(input_handle))
    ]
    output_handle = decoded.sequence.getSegmentOutputBindingSlotsHandle(segment_index)
    output_refs = [
        copy_binding_slot(output_handle, slot_index)
        for slot_index in range(decoded.sequence.getBindingsSize(output_handle))
    ]

    segment_constant_indexes = _view_to_list(
        decoded.sequence.getSegmentConstantIndexes(segment_index)
    )
    constant_refs = _copy_constants_preserving_indexes(
        decoded, segment_constant_indexes, add_resource, encoder
    )

    push_constant_handle = decoded.sequence.getSegmentPushConstRange(segment_index)
    push_constant_refs = [
        encoder.AddPushConstRange(
            decoded.sequence.getPushConstRangeStageFlags(
                push_constant_handle, range_index
            ),
            decoded.sequence.getPushConstRangeOffset(push_constant_handle, range_index),
            decoded.sequence.getPushConstRangeSize(push_constant_handle, range_index),
        )
        for range_index in range(
            decoded.sequence.getPushConstRangesSize(push_constant_handle)
        )
    ]

    dispatch_shape = _view_to_list(
        decoded.sequence.getSegmentDispatchShape(segment_index)
    )
    if len(dispatch_shape) != 3:
        dispatch_shape = [0, 0, 0]

    encoder.AddSegmentInfo(
        module,
        str(decoded.sequence.getSegmentName(segment_index)),
        descriptor_refs,
        input_refs,
        output_refs,
        [constant_refs[index] for index in segment_constant_indexes],
        dispatch_shape,
        push_constant_refs,
    )
    encoder.AddModelSequenceInputsOutputs()
    encoder.Finish()

    with output_path.open("wb") as output:
        if not encoder.WriteTo(output):
            raise RuntimeError(f"Failed to write single-segment VGF: {output_path}")


def _copy_constants_preserving_indexes(
    decoded: _DecodedVGF,
    segment_constant_indexes: list[int],
    add_resource: Any,
    encoder: Any,
) -> dict[int, Any]:
    if not segment_constant_indexes:
        return {}

    constant_refs = {}
    for constant_index in range(max(segment_constant_indexes) + 1):
        mrt_index = decoded.constants.getConstantMrtIndex(constant_index)
        resource = add_resource(mrt_index)
        sparsity_dimension = (
            decoded.constants.getConstantSparsityDimension(constant_index)
            if decoded.constants.isSparseConstant(constant_index)
            else -1
        )
        constant_ref = encoder.AddConstant(
            resource,
            _view_to_bytes(decoded.constants.getConstant(constant_index)),
            sparsity_dimension,
        )
        if _ref_value(constant_ref) != constant_index:
            raise ValueError(
                "Unable to preserve VGF graph constant indexes while splitting "
                f"segment; expected {constant_index}, got {_ref_value(constant_ref)}."
            )
        constant_refs[constant_index] = constant_ref

    return constant_refs


def _ref_value(ref: Any) -> int:
    return int(getattr(ref, "reference", ref))


def _view_to_list(view: Any) -> list:
    if view is None:
        return []
    return list(view)


def _spirv_words_to_bytes(view: Any) -> bytes:
    """Pack a VGF decoder's uint32 SPIR-V view as raw little-endian bytes."""
    words = _view_to_list(view)
    return struct.pack(f"<{len(words)}I", *words)


def _view_to_bytes(view: Any) -> bytes:
    if view is None:
        return b""
    return bytes(view)
