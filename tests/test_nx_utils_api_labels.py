# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for source-independent API-label decoding."""

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
def test_legacy_json_labels_remain_atomic(
    payload: object, known_labels: list[str] | None
) -> None:
    """JSON inside a legacy label is not an outer array encoding."""
    label = json.dumps(payload)
    parser = ApiLabelParser(known_labels)
    assert parser.parse(f"before; {label} ;after;") == ["before", label, "after"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", []),
        (";;;", []),
        ("plain;fallback;", ["plain", "fallback"]),
        ("{broken;fallback;", ["{broken", "fallback"]),
        ('{"valid": 1}suffix;fallback;', ['{"valid": 1}suffix', "fallback"]),
    ],
)
def test_legacy_fallback_labels_are_retained(value: str, expected: list[str]) -> None:
    """Unknown legacy fields retain plain-delimiter fallback behavior."""
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


def test_known_legacy_label_matching_precedes_json_payload_recovery() -> None:
    """A complete legacy name can include JSON plus semicolon text."""
    label = '{"name": "conv"};rescale;'
    assert ApiLabelParser([label]).parse(f"{label};fallback;") == [label, "fallback"]


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["None"],
        ["2:0", "4:0"],
        ["a;b", "c", "a;b"],
        ["", "[]", " leading and trailing "],
        ['quote "', "backslash \\", "tab\tnewline\nreturn\r", "\0", "操作"],
        ['{"source": "first(); second()"}', '["nested","array"]'],
    ],
)
@pytest.mark.parametrize("known_labels", [None, ["None", "a", "b", "c"]])
def test_json_arrays_preserve_exact_labels(
    labels: list[str], known_labels: list[str] | None
) -> None:
    """Decode only the outer list, preserving contents, order and duplicates."""
    encoded = json.dumps(labels, ensure_ascii=False)
    assert ApiLabelParser(known_labels).parse(encoded) == labels


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[];", ["[]"]),
        ('["a","b"];', ['["a","b"]']),
        ("[]", []),
        ('["[]"]', ["[]"]),
        ('["[\\"a\\",\\"b\\"]"]', ['["a","b"]']),
        (' \t["a;b", ""]\r\n', ["a;b", ""]),
        ("old; \r\n", ["old"]),
        (" \t\r\n", []),
    ],
)
def test_encoding_detection_preserves_literal_json_labels(
    value: str, expected: list[str]
) -> None:
    """The trailing legacy delimiter distinguishes a literal array label."""
    assert ApiLabelParser().parse(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "plain",
        "first;second",
        "{}",
        '"label"',
        "null",
        '["a"',
        '["a",]',
        "not-json]",
        "[1]",
        "[null]",
        "[true]",
        '[["nested"]]',
        '[{"name": "op"}]',
        '["ok", 1]',
    ],
)
def test_unsupported_or_invalid_encodings_are_rejected(value: str) -> None:
    """Malformed new data must not fall back to legacy label splitting."""
    with pytest.raises(ValueError, match="API label"):
        ApiLabelParser().parse(value)


def test_new_encoding_does_not_use_known_label_recovery() -> None:
    """A source name equal to the wire text must not override its encoding."""
    value = '["a;b","c"]'
    assert ApiLabelParser([value, "a;b;c"]).parse(value) == ["a;b", "c"]
