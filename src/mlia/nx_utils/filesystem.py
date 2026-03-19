# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""NX plugin filesystem utilities."""

from pathlib import Path


def is_tosa_file(model: str | Path) -> bool:
    """Check if path contains tosa file."""
    model_path = Path(model)

    return model_path.suffix in {".tosamlir", ".tosa"}


def is_vgf_file(model: str | Path) -> bool:
    """Check if path contains vgf file."""
    model_path = Path(model)

    return model_path.suffix == ".vgf"


def is_pytorch_file(model: str | Path) -> bool:
    """Check if path contains a pt2 file."""
    model_path = Path(model)

    return model_path.suffix in {".pt2"}
