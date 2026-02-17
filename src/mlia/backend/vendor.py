# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for vendor."""

from __future__ import annotations

from pathlib import Path


def vendor_artifact_path(backend_folder: str) -> Path | None:
    """Return mlia/_vendor/artifacts/<backend_folder> if it exists, else None."""
    # <repo>/src/mlia/backend/vendor.py -> .../mlia (package root)
    pkg_root = Path(__file__).resolve().parents[1]
    vendor = pkg_root / "_vendor" / "artifacts" / backend_folder
    return vendor if vendor.exists() else None
