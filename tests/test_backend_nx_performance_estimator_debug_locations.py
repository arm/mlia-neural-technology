# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NX Performance Estimator debug location resolution."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.nx_performance_estimator.debug_locations import (
    resolve_spirv_id_locations,
)


def test_resolve_spirv_id_locations_keeps_complete_vgf_mappings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete VGF debug metadata should not trigger the TOSA fallback."""
    debug_db = {
        "tosa_op_id_to_api_labels": {"545": ["TOSAMUL_spirv_id_716"]},
    }
    read_vgf_locations_mock = MagicMock(return_value={"716": "model/real_mul"})
    read_tosa_locations_mock = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.debug_locations."
        "read_vgf_spirv_id_locations",
        read_vgf_locations_mock,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.debug_locations."
        "read_tosa_spirv_id_locations",
        read_tosa_locations_mock,
    )

    assert resolve_spirv_id_locations(
        tmp_path / "model.tflite",
        tmp_path / "model.vgf",
        debug_db,
    ) == {"716": "model/real_mul"}

    read_vgf_locations_mock.assert_called_once_with(tmp_path / "model.vgf")
    read_tosa_locations_mock.assert_not_called()


def test_resolve_spirv_id_locations_fills_missing_ids_from_tosa_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOSA fallback should fill gaps without replacing VGF-derived locations."""
    debug_db = {
        "tosa_op_id_to_api_labels": {
            "545": ["TOSAMUL_spirv_id_716"],
            "635": ["TOSARESCALE_spirv_id_999"],
        },
    }
    tosa_mlir = tmp_path / "converted.tosamlir"
    tosa_mlir.touch()
    read_vgf_locations_mock = MagicMock(return_value={"716": "model/real_mul"})
    read_tosa_locations_mock = MagicMock(
        return_value={
            "716": "wrong/fallback",
            "999": "model/rescale",
        }
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.debug_locations."
        "read_vgf_spirv_id_locations",
        read_vgf_locations_mock,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_performance_estimator.debug_locations."
        "read_tosa_spirv_id_locations",
        read_tosa_locations_mock,
    )

    assert resolve_spirv_id_locations(
        tmp_path / "model.tflite",
        tmp_path / "model.vgf",
        debug_db,
    ) == {
        "716": "model/real_mul",
        "999": "model/rescale",
    }

    read_tosa_locations_mock.assert_called_once_with(tosa_mlir, debug_db)
