# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classify source operators and project dataflow for target analysis."""

import re
from typing import Any


def node_category(name: str, target: str) -> str:
    """Classify an exported-program node for heuristic matching."""
    rules = (
        ("conv", ("convolution", "conv2d")),
        ("quantization", ("quantize", "dequantize", "rescale", "cast")),
        ("relu", ("relu", "clamp")),
        ("add", ("add",)),
        ("sub", ("sub",)),
        ("mul", ("mul",)),
        ("transpose", ("transpose", "permute")),
        ("matmul", ("matmul", "linear", "mm", "bmm", "addmm", "addbmm", "baddbmm")),
        ("pool", ("pool", "pool1d", "pool2d", "pool3d")),
        ("softmax", ("softmax",)),
        ("reshape", ("reshape", "view", "squeeze", "unsqueeze")),
        ("reduction", ("sum", "mean")),
    )
    # Explicit operator identity takes precedence over a generated source name.
    # Whole tokens distinguish AddMM from Add and broadcast operations from cast.
    for text in (target, name):
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        for category, needles in rules:
            if any(needle in tokens for needle in needles):
                return category
    return "other"


def project_dataflow(
    result: dict[str, Any], edges: list[dict[str, str]], kind: str
) -> list[dict[str, str]]:
    """Follow source edges to the next associated entity, with cycle protection."""
    entities = {entity["id"]: entity for entity in result.get("entities", [])}
    owners: dict[str, set[str]] = {}
    for entity in entities.values():
        if entity["kind"] != kind:
            continue
        for child_id in entity.get("child_ids", []):
            if entities.get(child_id, {}).get("kind") == "source_operator":
                owners.setdefault(child_id, set()).add(entity["id"])
    for entity in entities.values():
        if entity["kind"] == "source_operator":
            owners.setdefault(entity["id"], set()).update(
                parent
                for parent in entity.get("parent_ids", [])
                if entities.get(parent, {}).get("kind") == kind
            )
    consumers: dict[str, list[str]] = {}
    for edge in edges:
        consumers.setdefault(edge["source_id"], []).append(edge["target_id"])
    relationships: set[tuple[str, str]] = set()
    for source_id, source_owners in owners.items():
        for owner in source_owners:
            pending = list(consumers.get(source_id, []))
            visited = set()
            while pending:
                target_id = pending.pop()
                if target_id in visited:
                    continue
                visited.add(target_id)
                different = owners.get(target_id, set()) - {owner}
                if different:
                    relationships.update((owner, target) for target in different)
                else:
                    pending.extend(consumers.get(target_id, []))
    return [
        {"source_id": source, "target_id": target}
        for source, target in sorted(relationships)
    ]
