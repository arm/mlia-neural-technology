# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""NX plugin misc utilities."""

from typing import Any, Dict, List, Union


def list_to_dict(list_mapping: list, key_field: Any) -> Union[Dict, List]:
    """Convert a list to a dict with key key_field."""
    if key_field:
        dict_mapping = {}
        for list_map in list_mapping:
            try:
                new_key = list_map[key_field]
                dict_mapping[new_key] = {
                    key: value for key, value in list_map.items() if key != key_field
                }
            except KeyError as exc:
                raise KeyError("The key_field isn't present in all dicts.") from exc
        return dict_mapping

    return list_mapping


def dict_to_list(dict_mapping: dict, key_field: Any) -> list:
    """Convert a dict to a list of dicts."""
    output = []
    for key, value in dict_mapping.items():
        out = {**value}
        out[key_field] = key
        output.append(out)

    return output
