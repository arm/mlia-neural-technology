# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Companion relationships are analysis output, independent of visualization."""

import hashlib
from unittest.mock import Mock

import pytest
from mlia.backend.executorch.pte import PTEDelegate, PTEReader
from mlia.core.errors import ConfigurationError

from mlia.backend.neural_technology_profiling_data import (
    executorch_artifacts as artifacts,
)


@pytest.fixture
def companion_reader(monkeypatch):
    """Supply the converter's detached delegate records at the target boundary."""
    delegates = [
        PTEDelegate("forward", 0, index, index, "VgfBackend", b"vgf")
        for index in range(2)
    ]
    reader = Mock(spec=PTEReader)
    reader.delegates.return_value = delegates
    reader.metadata.return_value = {
        "sha256": hashlib.sha256(b"pte").hexdigest(),
        "delegates": [delegate.metadata() for delegate in delegates],
    }
    monkeypatch.setattr(artifacts, "PTEReader", Mock(return_value=reader))


@pytest.mark.parametrize("operator_ids", [{42, 43}, {99}], ids=["matching", "missing"])
def test_companion_requires_instruction_and_spirv_compatibility(
    monkeypatch, tmp_path, companion_reader, operator_ids
):
    """Instruction identity selects the delegate; its graph must contain the op."""
    pte = tmp_path / "model.pte"
    pte.write_bytes(b"pte")
    monkeypatch.setattr(
        artifacts, "vgf_segment_operators", lambda _payload: {0: operator_ids}
    )
    output = {
        "extensions": {"executorch": {"source_graph": {"edges": []}}},
        "results": [
            {
                "entities": [
                    {
                        "id": "segment",
                        "kind": "segment",
                        "attributes": {
                            "method": "forward",
                            "instruction_id": 1,
                            "segment_index": 0,
                        },
                    },
                    {"id": "cascade", "kind": "cascade", "parent_ids": ["segment"]},
                    {
                        "id": "chain",
                        "kind": "chain",
                        "name": "Chain",
                        "parent_ids": ["cascade"],
                        "child_ids": ["stripe"],
                    },
                    {
                        "id": "stripe",
                        "kind": "stripe",
                        "attributes": {
                            "backend_operations": [
                                {
                                    "operator": "Conv2D",
                                    "tosa_id": "3",
                                    "api_labels": ["TOSACONV2D_spirv_id_42"],
                                    "entity_id": "compiled",
                                }
                            ]
                        },
                    },
                    {"id": "compiled", "kind": "backend_operator"},
                ]
            }
        ],
    }
    if 42 not in operator_ids:
        with pytest.raises(ConfigurationError, match="42"):
            artifacts.enrich_output(output, pte)
        return

    artifacts.enrich_output(output, pte)
    companion = output["extensions"]["executorch"]["companion_model"]
    assert companion["sha256"] == hashlib.sha256(b"pte").hexdigest()
    association = companion["operator_associations"][0]
    assert association["delegate_id"] == companion["delegates"][1]["id"]
    assert association["spirv_id"] == 42
    assert association["association"] == "structural-spirv-association"
    assert association["backend_operator_ids"] == ["compiled"]
    assert association["chain_id"] == "chain"
