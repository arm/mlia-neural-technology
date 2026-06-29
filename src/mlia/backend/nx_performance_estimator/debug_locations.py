# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Resolve NX Performance Estimator debug labels to source model locations."""

from __future__ import annotations

from pathlib import Path

from mlia.backend.nx_performance_estimator.output_parsing import (
    DebugDatabaseContentsType,
)
from mlia.backend.nx_performance_estimator.statistics import (
    has_unresolved_spirv_id_locations,
    read_tosa_spirv_id_locations,
)
from mlia.nx_utils.vgf_debug import read_vgf_spirv_id_locations


def resolve_spirv_id_locations(
    model_path: Path,
    vgf_file: Path,
    debug_db: DebugDatabaseContentsType,
) -> dict[str, str]:
    """Resolve estimator SPIR-V placeholder ids to source model locations."""
    spirv_id_locations = read_vgf_spirv_id_locations(vgf_file)
    if not has_unresolved_spirv_id_locations(debug_db, spirv_id_locations):
        return spirv_id_locations

    for spirv_id, location in _read_tosa_fallback_locations(
        model_path,
        vgf_file,
        debug_db,
    ).items():
        spirv_id_locations.setdefault(spirv_id, location)

    return spirv_id_locations


def _read_tosa_fallback_locations(
    model_path: Path,
    vgf_file: Path,
    debug_db: DebugDatabaseContentsType,
) -> dict[str, str]:
    """Read fallback SPIR-V id locations from TOSA text, if available."""
    candidates = []
    if model_path.suffix in {".tosamlir", ".tosa"}:
        candidates.append(model_path)
    candidates.extend(vgf_file.parent.glob("*.tosamlir"))

    for candidate in candidates:
        if candidate.is_file():
            return read_tosa_spirv_id_locations(candidate, debug_db)
    return {}
