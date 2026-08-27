# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NX Performance Estimator VGF helpers."""

from __future__ import annotations

from array import array
import json
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.vgf import _decode_vgf
from mlia.backend.nx_performance_estimator.vgf import _prepare_gcpe_compatible_vgfs
from mlia.backend.nx_performance_estimator.vgf import (
    _segment_structural_source_operator_ids,
    _segment_structural_source_operator_names,
)
from mlia.backend.nx_performance_estimator.vgf import _validate_gcpe_compatible_vgf
from mlia.backend.nx_performance_estimator.vgf import prepare_gcpe_compatible_vgfs


class _FakeModuleType:
    Graph = "Graph"
    Compute = "Compute"


class _FakeResourceCategory:
    Input = "Input"
    Output = "Output"
    Intermediate = "Intermediate"
    Constant = "Constant"


def _fake_header() -> SimpleNamespace:
    return SimpleNamespace(
        IsValid=MagicMock(return_value=True),
        GetEncoderVulkanHeadersVersion=MagicMock(return_value=123),
        GetModuleTableOffset=MagicMock(return_value=0),
        GetModuleTableSize=MagicMock(return_value=1),
        GetModelSequenceTableOffset=MagicMock(return_value=0),
        GetModelSequenceTableSize=MagicMock(return_value=1),
        GetModelResourceTableOffset=MagicMock(return_value=0),
        GetModelResourceTableSize=MagicMock(return_value=1),
        GetConstantsOffset=MagicMock(return_value=0),
        GetConstantsSize=MagicMock(return_value=1),
    )


def _fake_vgfpy(decoder: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        HeaderSize=MagicMock(return_value=0),
        CreateHeaderDecoder=MagicMock(return_value=_fake_header()),
        CreateModuleTableDecoder=MagicMock(
            return_value=SimpleNamespace(
                isSPIRV=MagicMock(return_value=True),
                hasSPIRVCode=MagicMock(return_value=True),
                getSPIRVModuleCode=MagicMock(
                    return_value=memoryview(
                        array("I", [0x07230203, 0x00010600, 0x12345678, 0xDEADBEEF])
                    )
                ),
            )
        ),
        CreateModelSequenceTableDecoder=MagicMock(return_value=decoder),
        CreateModelResourceTableDecoder=MagicMock(return_value=SimpleNamespace()),
        CreateConstantDecoder=MagicMock(return_value=SimpleNamespace()),
        ModuleType=_FakeModuleType,
        ResourceCategory=_FakeResourceCategory,
    )


def test_segment_structural_source_operator_ids_use_all_spirv_debug_operation_ids() -> (
    None
):
    """Segment ownership can include SPIR-V ops absent from GCPE debug DB stats."""
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"484": "CONCAT", "496": "CONCAT"}
    )

    assert _segment_structural_source_operator_ids(7, debug_names) == [
        "source_operator/segment_7/spirv-484",
        "source_operator/segment_7/spirv-496",
    ]


def test_segment_structural_source_operator_names_decode_friendly_json_name() -> None:
    """Structured VGF debug labels should expose their friendly operator name."""
    debug_label = json.dumps(
        {
            "aten_info": {
                "node_name": "aten_cat_default",
                "operator_name": "aten.cat.default",
            }
        }
    )
    debug_names = SpirvDebugNameMap(spirv_id_to_debug_name={"484": debug_label})

    assert _segment_structural_source_operator_names(7, debug_names) == {
        "source_operator/segment_7/spirv-484": "aten.cat.default"
    }


def test_validate_gcpe_compatible_vgf_allows_multiple_graph_segments(
    tmp_path: Path,
) -> None:
    """Multi-segment VGFs are valid if every segment is a graph segment."""
    model = tmp_path / "model.vgf"
    model.write_bytes(b"fake vgf")
    decoder = SimpleNamespace(
        modelSequenceTableSize=MagicMock(return_value=2),
        getSegmentType=MagicMock(return_value=_FakeModuleType.Graph),
    )

    _validate_gcpe_compatible_vgf(model, _fake_vgfpy(decoder))


def test_validate_gcpe_compatible_vgf_allows_compute_segments(tmp_path: Path) -> None:
    """Compute VGF segments are skipped by the GCPE integration."""
    model = tmp_path / "model.vgf"
    model.write_bytes(b"fake vgf")
    decoder = SimpleNamespace(
        modelSequenceTableSize=MagicMock(return_value=2),
        getSegmentType=MagicMock(
            side_effect=lambda segment_index: [
                _FakeModuleType.Graph,
                _FakeModuleType.Compute,
            ][segment_index]
        ),
    )

    _validate_gcpe_compatible_vgf(model, _fake_vgfpy(decoder))


def test_validate_gcpe_compatible_vgf_rejects_unknown_segment(tmp_path: Path) -> None:
    """Unknown VGF segment types are rejected before invoking GCPE."""
    model = tmp_path / "model.vgf"
    model.write_bytes(b"fake vgf")
    decoder = SimpleNamespace(
        modelSequenceTableSize=MagicMock(return_value=2),
        getSegmentType=MagicMock(
            side_effect=lambda segment_index: [
                _FakeModuleType.Graph,
                "Unknown",
            ][segment_index]
        ),
    )

    with pytest.raises(ValueError, match="segment 1"):
        _validate_gcpe_compatible_vgf(model, _fake_vgfpy(decoder))


def test_prepare_gcpe_compatible_vgfs_keeps_single_graph_segment(
    tmp_path: Path,
) -> None:
    """A single graph segment can be passed to GCPE without rewriting."""
    model = tmp_path / "model.vgf"
    model.write_bytes(b"fake vgf")
    decoder = SimpleNamespace(
        modelSequenceTableSize=MagicMock(return_value=1),
        getSegmentType=MagicMock(return_value=_FakeModuleType.Graph),
        getSegmentName=MagicMock(return_value="graph_segment"),
        getSegmentModuleIndex=MagicMock(return_value=0),
    )

    segments = _prepare_gcpe_compatible_vgfs(
        model, tmp_path / "segments", _fake_vgfpy(decoder)
    )

    assert len(segments) == 1
    assert segments[0].segment_index == 0
    assert segments[0].segment_name == "graph_segment"
    assert segments[0].path == model
    assert segments[0].spirv == bytes.fromhex("03022307 00060100 78563412 efbeadde")


class _Ref:
    def __init__(self, reference: int) -> None:
        self.reference = reference


class _Encoder:
    def __init__(self) -> None:
        self.modules: list[tuple[object, ...]] = []
        self.resources: list[tuple[object, ...]] = []
        self.bindings: list[tuple[object, ...]] = []
        self.descriptors: list[object] = []
        self.constants: list[tuple[object, ...]] = []
        self.segments: list[tuple[object, ...]] = []
        self.model_io = None
        self.finished = False

    def AddModule(self, *args):  # noqa: N802, ANN002, ANN003
        self.modules.append(args)
        return _Ref(len(self.modules) - 1)

    def AddInputResource(self, *args):  # noqa: N802, ANN002, ANN003
        return self._add_resource("input", args)

    def AddOutputResource(self, *args):  # noqa: N802, ANN002, ANN003
        return self._add_resource("output", args)

    def AddIntermediateResource(self, *args):  # noqa: N802, ANN002, ANN003
        return self._add_resource("intermediate", args)

    def AddConstantResource(self, *args):  # noqa: N802, ANN002, ANN003
        return self._add_resource("constant", args)

    def _add_resource(self, category, args):  # noqa: ANN001, ANN202
        self.resources.append((category, args))
        return _Ref(len(self.resources) - 1)

    def AddBindingSlot(self, *args):  # noqa: N802, ANN002, ANN003
        self.bindings.append(args)
        return _Ref(len(self.bindings) - 1)

    def AddDescriptorSetInfo(self, bindings):  # noqa: N802, ANN001, ANN202
        self.descriptors.append(bindings)
        return _Ref(len(self.descriptors) - 1)

    def AddConstant(self, resource, data, sparsity_dimension=-1):  # noqa: N802, ANN001, ANN202
        self.constants.append((resource, data, sparsity_dimension))
        return _Ref(len(self.constants) - 1)

    def AddPushConstRange(self, *args):  # noqa: N802, ANN002, ANN003
        return _Ref(0)

    def AddSegmentInfo(self, *args):  # noqa: N802, ANN002, ANN003
        self.segments.append(args)
        return _Ref(len(self.segments) - 1)

    def AddModelSequenceInputsOutputs(self, *args):  # noqa: N802, ANN002, ANN003
        self.model_io = args

    def Finish(self):  # noqa: N802, ANN202
        self.finished = True

    def WriteTo(self, output):  # noqa: N802, ANN001, ANN202
        output.write(b"split vgf")
        return True


class _Handle:
    def __init__(self, bindings: list[tuple[int, int]]) -> None:
        self.bindings = bindings


class _Sequence:
    def __init__(self, segment_types: list[str] | None = None) -> None:
        self.segment_types = segment_types or [
            _FakeModuleType.Graph,
            _FakeModuleType.Graph,
        ]
        self.desc = [_Handle([(0, 2), (1, 3)]), _Handle([(0, 4), (1, 5)])]
        self.inputs = [_Handle([(0, 2)]), _Handle([(0, 4)])]
        self.outputs = [_Handle([(1, 3)]), _Handle([(1, 5)])]

    def modelSequenceTableSize(self):  # noqa: N802, ANN202
        return 2

    def getSegmentType(self, segment_index):  # noqa: N802, ANN001, ANN202
        return self.segment_types[segment_index]

    def getSegmentModuleIndex(self, segment_index):  # noqa: N802, ANN001, ANN202
        return segment_index

    def getSegmentDescriptorSetInfosSize(self, _segment_index):  # noqa: N802, ANN001, ANN202
        return 1

    def getDescriptorBindingSlotsHandle(self, segment_index, _descriptor_index):  # noqa: N802, ANN001, ANN202
        return self.desc[segment_index]

    def getSegmentInputBindingSlotsHandle(self, segment_index):  # noqa: N802, ANN001, ANN202
        return self.inputs[segment_index]

    def getSegmentOutputBindingSlotsHandle(self, segment_index):  # noqa: N802, ANN001, ANN202
        return self.outputs[segment_index]

    def getBindingsSize(self, handle):  # noqa: N802, ANN001, ANN202
        return len(handle.bindings)

    def getBindingSlotBinding(self, handle, slot_index):  # noqa: N802, ANN001, ANN202
        return handle.bindings[slot_index][0]

    def getBindingSlotMrtIndex(self, handle, slot_index):  # noqa: N802, ANN001, ANN202
        return handle.bindings[slot_index][1]

    def getSegmentConstantIndexes(self, segment_index):  # noqa: N802, ANN001, ANN202
        return [segment_index]

    def getSegmentPushConstRange(self, _segment_index):  # noqa: N802, ANN001, ANN202
        return _Handle([])

    def getPushConstRangesSize(self, _handle):  # noqa: N802, ANN001, ANN202
        return 0

    def getSegmentDispatchShape(self, _segment_index):  # noqa: N802, ANN001, ANN202
        return [0, 0, 0]

    def getSegmentName(self, segment_index):  # noqa: N802, ANN001, ANN202
        return f"segment_{segment_index}"


class _Modules:
    def isSPIRV(self, _module_index):  # noqa: N802, ANN001, ANN202
        return True

    def hasSPIRVCode(self, _module_index):  # noqa: N802, ANN001, ANN202
        return True

    def getModuleName(self, module_index):  # noqa: N802, ANN001, ANN202
        return f"module_{module_index}"

    def getModuleEntryPoint(self, _module_index):  # noqa: N802, ANN001, ANN202
        return "main"

    def getSPIRVModuleCode(self, module_index):  # noqa: N802, ANN001, ANN202
        words = [
            [0x07230203, 0x00010600, 0x12345678, 0xDEADBEEF],
            [0x07230203, 0x00010600, 0x89ABCDEF, 0xCAFEBABE],
        ][module_index]
        return memoryview(array("I", words))


class _Resources:
    def getCategory(self, mrt_index):  # noqa: N802, ANN001, ANN202
        return {
            0: _FakeResourceCategory.Constant,
            1: _FakeResourceCategory.Constant,
            2: _FakeResourceCategory.Input,
            3: _FakeResourceCategory.Output,
            4: _FakeResourceCategory.Input,
            5: _FakeResourceCategory.Output,
        }[mrt_index]

    def getDescriptorType(self, _mrt_index):  # noqa: N802, ANN001, ANN202
        return 1

    def getVkFormat(self, _mrt_index):  # noqa: N802, ANN001, ANN202
        return 2

    def getTensorShape(self, mrt_index):  # noqa: N802, ANN001, ANN202
        return [mrt_index + 1]

    def getTensorStride(self, _mrt_index):  # noqa: N802, ANN001, ANN202
        return []


class _Constants:
    def getConstantMrtIndex(self, constant_index):  # noqa: N802, ANN001, ANN202
        return constant_index

    def isSparseConstant(self, _constant_index):  # noqa: N802, ANN001, ANN202
        return False

    def getConstant(self, constant_index):  # noqa: N802, ANN001, ANN202
        return bytes([constant_index])


class _FakeVGFModule:
    ModuleType = _FakeModuleType
    ResourceCategory = _FakeResourceCategory

    def __init__(self, segment_types: list[str] | None = None) -> None:
        self.encoders: list[_Encoder] = []
        self.segment_types = segment_types

    def CreateHeaderDecoder(self, _data):  # noqa: N802, ANN001, ANN202
        return _fake_header()

    def CreateModuleTableDecoder(self, _data):  # noqa: N802, ANN001, ANN202
        return _Modules()

    def CreateModelSequenceTableDecoder(self, _data):  # noqa: N802, ANN001, ANN202
        return _Sequence(self.segment_types)

    def CreateModelResourceTableDecoder(self, _data):  # noqa: N802, ANN001, ANN202
        return _Resources()

    def CreateConstantDecoder(self, _data):  # noqa: N802, ANN001, ANN202
        return _Constants()

    def CreateEncoder(self, _vk_header_version):  # noqa: N802, ANN001, ANN202
        encoder = _Encoder()
        self.encoders.append(encoder)
        return encoder


def test_prepare_gcpe_compatible_vgfs_splits_multi_segment_vgf(
    tmp_path: Path,
) -> None:
    """Multi-segment VGFs are rewritten as one wrapper VGF per segment."""
    model = tmp_path / "model.vgf"
    model.write_bytes(b"fake vgf")
    vgfpy = _FakeVGFModule()

    segments = _prepare_gcpe_compatible_vgfs(model, tmp_path / "segments", vgfpy)

    assert [(segment.segment_index, segment.path.name) for segment in segments] == [
        (0, "model_segment_0.vgf"),
        (1, "model_segment_1.vgf"),
    ]
    assert [segment.path.read_bytes() for segment in segments] == [b"split vgf"] * 2
    assert [encoder.modules[0][1] for encoder in vgfpy.encoders] == [
        "module_0",
        "module_1",
    ]
    assert [encoder.segments[0][1] for encoder in vgfpy.encoders] == [
        "segment_0",
        "segment_1",
    ]
    assert [encoder.constants[-1][1] for encoder in vgfpy.encoders] == [
        b"\x00",
        b"\x01",
    ]


def test_prepare_gcpe_compatible_vgfs_skips_compute_segments(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Compute segments are not wrapped for GCPE and are reported as skipped."""
    model = tmp_path / "model.vgf"
    model.write_bytes(b"fake vgf")
    vgfpy = _FakeVGFModule([_FakeModuleType.Graph, _FakeModuleType.Compute])

    segments = _prepare_gcpe_compatible_vgfs(model, tmp_path / "segments", vgfpy)

    assert [(segment.segment_index, segment.path.name) for segment in segments] == [
        (0, "model_segment_0.vgf"),
    ]
    assert segments.skipped_compute_segments == [1]
    assert len(vgfpy.encoders) == 1
    assert "compute segment(s) 1 will not be estimated" in caplog.text


def test_real_multi_segment_vgf_splits_into_expected_single_segment_outputs(
    tmp_path: Path,
) -> None:
    """A generated real multi-segment VGF splits into valid single-segment outputs."""
    vgfpy = pytest.importorskip("vgfpy")
    source = tmp_path / "generated_multi_graph.vgf"

    words = [
        [0x07230203, 0x00010600, 0x12345678, 0xDEADBEEF],
        [0x07230203, 0x00010600, 0x89ABCDEF, 0xCAFEBABE],
    ]
    encoder = vgfpy.CreateEncoder(123)
    graph_0 = encoder.AddModule(vgfpy.ModuleType.Graph, "graph_0", "main_0", words[0])
    graph_1 = encoder.AddModule(vgfpy.ModuleType.Graph, "graph_1", "main_1", words[1])
    input_0 = encoder.AddInputResource(1, 2, [1], [])
    output_0 = encoder.AddOutputResource(1, 2, [1], [])
    input_1 = encoder.AddInputResource(1, 2, [2], [])
    output_1 = encoder.AddOutputResource(1, 2, [2], [])
    binding_0 = encoder.AddBindingSlot(0, input_0)
    binding_1 = encoder.AddBindingSlot(1, output_0)
    binding_2 = encoder.AddBindingSlot(0, input_1)
    binding_3 = encoder.AddBindingSlot(1, output_1)
    desc_0 = encoder.AddDescriptorSetInfo([binding_0, binding_1])
    desc_1 = encoder.AddDescriptorSetInfo([binding_2, binding_3])
    encoder.AddSegmentInfo(
        graph_0, "segment_0", [desc_0], [binding_0], [binding_1], [], [0, 0, 0], []
    )
    encoder.AddSegmentInfo(
        graph_1, "segment_1", [desc_1], [binding_2], [binding_3], [], [0, 0, 0], []
    )
    encoder.Finish()
    with source.open("wb") as output:
        assert encoder.WriteTo(output)

    segments = prepare_gcpe_compatible_vgfs(source, tmp_path / "segments")

    assert [(segment.segment_index, segment.path.name) for segment in segments] == [
        (0, "generated_multi_graph_segment_0.vgf"),
        (1, "generated_multi_graph_segment_1.vgf"),
    ]
    assert [segment.spirv for segment in segments] == [
        struct.pack("<4I", *module_words) for module_words in words
    ]

    expected_module_names = ["graph_0", "graph_1"]
    for segment, expected_module_name in zip(segments, expected_module_names):
        data = memoryview(segment.path.read_bytes())
        decoded = _decode_vgf(data, vgfpy, segment.path)
        sequence = decoded.sequence
        modules = decoded.modules

        assert sequence.modelSequenceTableSize() == 1
        assert sequence.getSegmentType(0) == vgfpy.ModuleType.Graph
        module_index = sequence.getSegmentModuleIndex(0)
        assert modules.getModuleName(module_index) == expected_module_name
