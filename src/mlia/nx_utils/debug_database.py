# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Decode compiler debug tables shared by structured and ETDump captures."""

from mlia.nx_utils.api_labels import ApiLabelParser


def parse_debug_table(
    lines: list[str], api_label_parser: ApiLabelParser | None = None
) -> list[dict[str, list[str]]]:
    """Read tab-delimited rows with tab or legacy whitespace header columns."""
    if not lines:
        return []
    columns = lines[0].split()
    label_parser = api_label_parser or ApiLabelParser()
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        values = line.split("\t")
        row = {}
        for index, column in enumerate(columns):
            value = (values[index] if index < len(values) else "").rstrip("\0\r")
            if column == "api_labels":
                row[column] = label_parser.parse(value)
            elif column == "api_label":
                row[column] = [value] if value else []
            else:
                row[column] = [
                    item.strip() for item in value.split(";") if item.strip()
                ]
        rows.append(row)
    return rows
