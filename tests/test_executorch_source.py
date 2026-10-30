# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target-owned source classification and analysis relationship projection."""

import pytest

from mlia.backend.neural_technology_profiling_data.executorch_source import (
    node_category,
    project_dataflow,
)


@pytest.mark.parametrize(
    ("name", "target", "expected"),
    [
        ("broadcast_to", "aten.broadcast_to.default", "other"),
        ("broadcast_tensors", "aten.broadcast_tensors.default", "other"),
        ("forecast", "aten.add.Tensor", "add"),
        ("cast", "aten.cast.default", "quantization"),
        ("cast_1", "custom.cast.default", "quantization"),
        ("convert", "custom::cast", "quantization"),
        (
            "quantize_per_tensor_1",
            "quantized_decomposed.quantize_per_tensor.default",
            "quantization",
        ),
        (
            "dequantize_per_channel",
            "quantized_decomposed.dequantize_per_channel.default",
            "quantization",
        ),
        ("rescale_1", "tosa.rescale.default", "quantization"),
        ("aten_addmm_default", "aten.addmm.default", "matmul"),
        ("aten_bmm_default", "aten.bmm.default", "matmul"),
        ("aten_mm_default", "aten.mm.default", "matmul"),
        ("conv2d_bias", "aten.add.Tensor", "add"),
        ("cast_input", "aten.relu.default", "relu"),
        ("submodule", "custom.submodule.default", "other"),
        ("max_pool2d", "aten.max_pool2d_with_indices.default", "pool"),
        ("adaptive_avg_pool2d", "aten.adaptive_avg_pool2d.default", "pool"),
        (
            "aten_addmm_default",
            "<EdgeOpOverload: aten.addmm.default>: schema = aten::addmm(Tensor self, Tensor mat1, Tensor mat2) -> Tensor",
            "matmul",
        ),
    ],
)
def test_node_category_matches_explicit_operator_tokens(
    name: str, target: str, expected: str
) -> None:
    """Explicit operation tokens take priority over misleading source labels."""
    assert node_category(name, target) == expected


def test_dataflow_projection_crosses_unprofiled_nodes_without_inventing_hierarchy():
    """Dataflow relationships remain separate from entity containment."""
    result = {
        "entities": [
            {"id": "chain-a", "kind": "chain", "child_ids": ["source-a"]},
            {"id": "chain-b", "kind": "chain", "child_ids": ["source-b"]},
            {"id": "source-a", "kind": "source_operator"},
            {"id": "source-b", "kind": "source_operator"},
        ]
    }
    edges = [
        {"source_id": "source-a", "target_id": "unprofiled"},
        {"source_id": "unprofiled", "target_id": "source-b"},
        {"source_id": "unprofiled", "target_id": "source-a"},
    ]
    assert project_dataflow(result, edges, "chain") == [
        {"source_id": "chain-a", "target_id": "chain-b"}
    ]
