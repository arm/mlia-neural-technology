# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared API-label recovery for estimated and measured debug databases."""

from __future__ import annotations

import json
from collections.abc import Iterable


class ApiLabelParser:
    """Decode JSON label arrays and recover names from legacy delimited fields."""

    def __init__(self, known_api_labels: Iterable[str] | None = None) -> None:
        """Index known names once, preferring the longest complete match."""
        self._known_labels = sorted(
            {label for label in known_api_labels or () if label},
            key=len,
            reverse=True,
        )
        self._json_decoder = json.JSONDecoder()

    def parse(self, value: str) -> list[str]:
        """Decode the complete field using its wire-format terminator.

        Legacy nonempty lists end in a semicolon, including a list whose only
        label is the literal string ``[]``. New lists are JSON arrays of strings.
        Empty legacy fields contain no labels. Invalid encodings are rejected
        rather than reinterpreted as source names.
        """
        value = value.strip()
        if not value:
            return []
        if value.endswith(";"):
            return self._parse_semicolon_labels(value)
        if not value.endswith("]"):
            raise ValueError(
                "Unsupported API label encoding: expected a semicolon-terminated "
                "list or a JSON array of strings."
            )
        try:
            labels = json.loads(value)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON API label array: {err.msg}.") from err
        if not isinstance(labels, list) or any(
            not isinstance(label, str) for label in labels
        ):
            raise ValueError("API labels must be a JSON array of strings.")
        return labels

    def _parse_semicolon_labels(self, value: str) -> list[str]:
        """Recover complete known names and JSON payloads in the legacy format.

        Semicolons inside names are unescaped. Prefer complete SPIR-V names
        (MLCE-1937), then atomic JSON payloads, then plain delimiter splitting.
        """
        labels: list[str] = []
        cursor = 0
        while cursor < len(value):
            while cursor < len(value) and (
                value[cursor] == ";" or value[cursor].isspace()
            ):
                cursor += 1
            if cursor >= len(value):
                break

            matched = next(
                (
                    label
                    for label in self._known_labels
                    if value.startswith(label, cursor)
                    and (
                        cursor + len(label) == len(value)
                        or value[cursor + len(label)] == ";"
                    )
                ),
                None,
            )
            if matched is not None:
                labels.append(matched)
                cursor += len(matched)
                continue

            end = self._fallback_end(value, cursor)
            label = value[cursor:end].strip()
            if label:
                labels.append(label)
            cursor = end + 1
        return labels

    def _fallback_end(self, value: str, cursor: int) -> int:
        start = cursor
        while start < len(value) and value[start].isspace():
            start += 1
        if start < len(value) and value[start] in '{["':
            try:
                _, end = self._json_decoder.raw_decode(value, start)
            except json.JSONDecodeError:
                pass
            else:
                while end < len(value) and value[end].isspace():
                    end += 1
                if end == len(value) or value[end] == ";":
                    return end
        delimiter = value.find(";", cursor)
        return len(value) if delimiter < 0 else delimiter
