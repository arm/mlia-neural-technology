# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Helpers for source location metadata emitted by ML debug tooling."""

from __future__ import annotations

import json


def get_debug_node_name(location: str | None) -> str | None:
    """Return a node name from JSON debug metadata, if present."""
    if not location:
        return None

    try:
        parsed = json.loads(location)
    except json.JSONDecodeError:
        return None

    try:
        return parsed.get("node_name") or parsed.get("aten_info", {}).get("node_name")
    except AttributeError:
        return None


def normalize_debug_location(location: str | None) -> str | None:
    """Normalize JSON debug metadata to a user-facing location string."""
    return get_debug_node_name(location) or location
