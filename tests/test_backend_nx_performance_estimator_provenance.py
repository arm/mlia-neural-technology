# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology source provenance identity resolution."""

from __future__ import annotations

import json

import pytest

from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.provenance import (
    source_operator_ids_from_api_label,
    source_provenance_from_api_label,
)


def test_source_location_uses_explicit_spirv_suffix() -> None:
    """An explicit SPIR-V suffix supplies canonical VGF coordinates."""
    assert source_operator_ids_from_api_label(
        "TOSACONV2D_spirv_id_504", segment_index=3
    ) == ["source_operator/segment_3/spirv-504"]


@pytest.mark.parametrize("spirv_ids", [["437"], ["437", "438"], ["437", "438", "437"]])
@pytest.mark.parametrize("segment_index", [0, 7])
def test_source_location_preserves_all_vgf_debug_name_matches(
    spirv_ids: list[str], segment_index: int
) -> None:
    """Shared source locations retain every distinct segment-scoped identity."""
    api_label = json.dumps({"aten_info": {"node_name": "tosa_conv2d_default"}})
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={spirv_id: api_label for spirv_id in spirv_ids},
        debug_name_to_spirv_ids={api_label: spirv_ids},
    )
    expected = [f"source_operator/segment_{segment_index}/spirv-437"]
    if "438" in spirv_ids:
        expected.append(f"source_operator/segment_{segment_index}/spirv-438")

    assert (
        source_operator_ids_from_api_label(
            api_label, segment_index=segment_index, debug_names=debug_names
        )
        == expected
    )
    provenance = source_provenance_from_api_label(
        api_label, segment_index=segment_index, debug_names=debug_names
    )
    assert provenance.source_operator_ids == expected
    assert provenance.display_label == "tosa_conv2d_default"


@pytest.mark.parametrize(
    "api_label",
    [
        "arbitrary-backend-label",
        json.dumps({"aten_info": {"node_name": "tosa_conv2d_default"}}),
    ],
)
def test_source_location_does_not_invent_identity(api_label: str) -> None:
    """Presentation labels cannot become source-operator identities."""
    assert source_operator_ids_from_api_label(api_label) == []
    assert source_provenance_from_api_label(api_label).source_operator_ids == []


def test_source_location_leaves_missing_debug_name_unresolved() -> None:
    """A label absent from the supplied name map has no source identity."""
    debug_names = SpirvDebugNameMap(debug_name_to_spirv_ids={"other": ["437"]})
    assert source_operator_ids_from_api_label("label", debug_names=debug_names) == []


def test_explicit_spirv_suffix_takes_precedence_over_shared_name() -> None:
    """An explicit result ID is more specific than shared label provenance."""
    api_label = "TOSACONV2D_spirv_id_504"
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"504": api_label, "505": api_label},
        debug_name_to_spirv_ids={api_label: ["504", "505"]},
    )
    assert source_operator_ids_from_api_label(api_label, debug_names=debug_names) == [
        "source_operator/segment_0/spirv-504"
    ]


def test_source_location_rejects_spirv_id_outside_supplied_vgf() -> None:
    """An explicit suffix cannot invent an operator outside the supplied VGF."""
    debug_names = SpirvDebugNameMap(op_ext_inst_spirv_ids=["504"])
    with pytest.raises(ValueError, match="not present in supplied VGF segment 3"):
        source_operator_ids_from_api_label(
            "TOSACONV2D_spirv_id_999", segment_index=3, debug_names=debug_names
        )
