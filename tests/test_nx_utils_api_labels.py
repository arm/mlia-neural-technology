# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for source-independent API-label recovery."""

import json

import pytest

from mlia.nx_utils.api_labels import ApiLabelParser


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "first(); second()", "nested": ["left;right"]},
        ["first;second", {"quoted": 'a";b'}],
        "quoted;label",
    ],
)
@pytest.mark.parametrize("known_labels", [None, [], ["known;label"]])
def test_json_fallback_preserves_payload_without_known_source_names(
    payload: object, known_labels: list[str] | None
) -> None:
    """JSON strings and containers remain atomic in both ingestion paths."""
    label = json.dumps(payload)
    parser = ApiLabelParser(known_labels)
    assert parser.parse(f"before; {label} ;after;") == ["before", label, "after"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", []),
        (";;;", []),
        ("plain;fallback", ["plain", "fallback"]),
        ("{broken;fallback;", ["{broken", "fallback"]),
        ('{"valid": 1}suffix;fallback', ['{"valid": 1}suffix', "fallback"]),
    ],
)
def test_plain_or_malformed_fallback_labels_are_retained(
    value: str, expected: list[str]
) -> None:
    """Unknown fields retain the existing plain-delimiter fallback behavior."""
    assert ApiLabelParser().parse(value) == expected


@pytest.mark.parametrize("whitespace", [" ", "\t", " \r\n\t "])
def test_delimiter_whitespace_does_not_split_known_labels(whitespace: str) -> None:
    """Formatting after a delimiter must not prevent a complete name match."""
    parser = ApiLabelParser(["known;label"])
    assert parser.parse(f"first;{whitespace}known;label;") == [
        "first",
        "known;label",
    ]


def test_delimiter_whitespace_preserves_whitespace_inside_known_labels() -> None:
    """Only delimiter-adjacent whitespace is skipped, not content in a name."""
    label = "module one;module two"
    assert ApiLabelParser([label]).parse(f"first; \t{label};") == ["first", label]


def test_known_label_matching_takes_precedence_over_json_decoding() -> None:
    """A complete known name can include JSON plus additional semicolon text."""
    label = '{"name": "conv"};rescale;'
    assert ApiLabelParser([label]).parse(f"{label};fallback") == [label, "fallback"]
