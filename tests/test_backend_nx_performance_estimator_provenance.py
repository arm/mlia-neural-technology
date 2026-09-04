# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Neural Technology source provenance identity resolution."""

from __future__ import annotations

import json

import pytest

from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap
from mlia.backend.nx_performance_estimator.provenance import (
    source_operator_id_from_api_label,
)


def test_source_location_uses_explicit_spirv_suffix() -> None:
    """An explicit SPIR-V suffix supplies canonical VGF coordinates."""
    assert (
        source_operator_id_from_api_label("TOSACONV2D_spirv_id_504", segment_index=3)
        == "source_operator/segment_3/spirv-504"
    )


def test_source_location_uses_unique_vgf_debug_name() -> None:
    """An exact unique VGF debug-name match supplies canonical coordinates."""
    api_label = json.dumps({"aten_info": {"node_name": "tosa_conv2d_default"}})
    debug_names = SpirvDebugNameMap(
        spirv_id_to_debug_name={"437": api_label},
        debug_name_to_spirv_ids={api_label: ["437"]},
    )

    assert (
        source_operator_id_from_api_label(
            api_label, segment_index=0, debug_names=debug_names
        )
        == "source_operator/segment_0/spirv-437"
    )


@pytest.mark.parametrize(
    "api_label",
    [
        "arbitrary-backend-label",
        json.dumps({"aten_info": {"node_name": "tosa_conv2d_default"}}),
    ],
)
def test_source_location_does_not_invent_identity(api_label: str) -> None:
    """Presentation labels cannot become source-operator identities."""
    assert source_operator_id_from_api_label(api_label) is None


@pytest.mark.parametrize("spirv_ids", [[], ["437", "438"]])
def test_source_location_rejects_unresolved_debug_name(
    spirv_ids: list[str],
) -> None:
    """Missing and ambiguous VGF debug-name matches remain unresolved."""
    debug_names = SpirvDebugNameMap(debug_name_to_spirv_ids={"label": spirv_ids})

    assert source_operator_id_from_api_label("label", debug_names=debug_names) is None


def test_source_location_rejects_spirv_id_outside_supplied_vgf() -> None:
    """An explicit suffix cannot invent an operator outside the supplied VGF."""
    debug_names = SpirvDebugNameMap(op_ext_inst_spirv_ids=["504"])

    with pytest.raises(ValueError, match="not present in supplied VGF segment 3"):
        source_operator_id_from_api_label(
            "TOSACONV2D_spirv_id_999",
            segment_index=3,
            debug_names=debug_names,
        )
