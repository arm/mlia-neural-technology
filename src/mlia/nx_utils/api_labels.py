# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared API-label recovery for estimated and measured debug databases."""

from __future__ import annotations

import json
from collections.abc import Iterable


class ApiLabelParser:
    """Recover full SPIR-V names from unescaped semicolon-delimited fields."""

    def __init__(self, known_api_labels: Iterable[str] | None = None) -> None:
        """Index known names once, preferring the longest complete match."""
        self._known_labels = sorted(
            {label for label in known_api_labels or () if label},
            key=len,
            reverse=True,
        )
        self._json_decoder = json.JSONDecoder()

    def parse(self, value: str) -> list[str]:
        """Recover known labels, JSON payloads and plain fallback labels.

        Debug producers concatenate labels without escaping semicolons inside
        the names. Match complete SPIR-V names before interpreting delimiters
        (MLCE-1937). Unknown JSON payloads remain intact even without source
        names; other unknown values use semicolon-delimited fallback labels.
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
