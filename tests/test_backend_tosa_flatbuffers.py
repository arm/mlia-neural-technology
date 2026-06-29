# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TOSA FlatBuffers reader backend."""

from mlia.backend.ml_sdk_model_converter import tosa_reader


def test_public_tosa_tools_flatbuffers_import_is_available() -> None:
    """The public tosa-tools package should provide readable TOSA modules."""
    assert tosa_reader.tosa_flatbuffers_available()
