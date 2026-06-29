# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the installation of ML SDK Model Converter."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import sysconfig

MODEL_CONVERTER_EXE = "model-converter"


def get_ml_sdk_model_converter_path() -> Path | None:
    """Return the installed model converter executable directory."""
    resolved = shutil.which(MODEL_CONVERTER_EXE)
    if resolved:
        return Path(resolved).parent

    script_dir = sysconfig.get_path("scripts")
    candidate_dirs = [Path(sys.executable).parent]
    if script_dir:
        candidate_dirs.insert(0, Path(str(script_dir)))

    for directory in candidate_dirs:
        exe_path = directory / MODEL_CONVERTER_EXE
        if exe_path.is_file():
            return exe_path.parent

    return None
