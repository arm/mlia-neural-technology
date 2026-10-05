# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared Neural Technology source provenance entities and debug-label parsing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import mlia.core.output_schema as schema
from mlia.backend.nx_performance_estimator.debug_info import SpirvDebugNameMap

NN_MODULE_ENTITY_KIND = "nn_module"
NX_PROVENANCE_ENTITY_KIND_DECLARATIONS = (
    schema.EntityKind(
        id=NN_MODULE_ENTITY_KIND,
        parent_kinds=[NN_MODULE_ENTITY_KIND],
        child_kinds=[NN_MODULE_ENTITY_KIND, schema.ENTITY_KIND_SOURCE_OPERATOR],
    ),
)

_SPIRV_API_LABEL_RE = re.compile(r"_spirv_id_(\d+)$")
_STACK_TRACE_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>.+?)", line (?P<line>\d+)(?:, in (?P<function>.*))?\s*$'
)
_STACK_TRACE_SENTINELS = {"No stack trace available"}


@dataclass(frozen=True)
class NXSourceProvenance:
    """Canonical identity and presentation metadata from one debug label."""

    source_operator_ids: list[str]
    display_label: str
    name: str | None = None
    nn_module_stacks: list[list[dict[str, str]]] = field(default_factory=list)
    code_stacks: list[list[dict[str, str]]] = field(default_factory=list)


def source_provenance_from_api_label(
    api_label: str,
    segment_index: int = 0,
    debug_names: SpirvDebugNameMap | None = None,
) -> NXSourceProvenance:
    """Decode one GCPE API label into canonical source provenance."""
    decoded = _json_api_label(api_label)
    source_operator_ids = source_operator_ids_from_api_label(
        api_label, segment_index, debug_names
    )
    display_label = api_label
    name = None
    if decoded is not None:
        aten_info = decoded.get("aten_info")
        if isinstance(aten_info, Mapping):
            node_name = aten_info.get("node_name")
            operator_name = aten_info.get("operator_name")
            if isinstance(node_name, str) and node_name:
                display_label = node_name
            if isinstance(operator_name, str) and operator_name:
                name = operator_name
        module_stack = _nn_module_stack_from_decoded_label(decoded)
        code_stack = _stack_trace_from_decoded_label(decoded)
        return NXSourceProvenance(
            source_operator_ids=source_operator_ids,
            display_label=display_label,
            name=name,
            nn_module_stacks=[module_stack] if module_stack else [],
            code_stacks=[code_stack] if code_stack else [],
        )
    return NXSourceProvenance(source_operator_ids, display_label)


def source_operator_ids_from_api_label(
    api_label: str,
    segment_index: int = 0,
    debug_names: SpirvDebugNameMap | None = None,
) -> list[str]:
    """Resolve a label to every matching canonical VGF source-operator ID.

    Several lowered operations may retain the same source location. Preserve
    that shared provenance instead of selecting one operation or dropping it.
    """
    match = _SPIRV_API_LABEL_RE.search(api_label)
    if match:
        spirv_id = match.group(1)
        if debug_names is not None and spirv_id not in debug_names.spirv_ids:
            raise ValueError(
                f"Debug label {api_label!r} references SPIR-V result ID {spirv_id}, "
                f"which is not present in supplied VGF segment {segment_index}."
            )
        return [schema.vgf_source_operator_id(segment_index, int(spirv_id))]

    if debug_names is not None:
        matching_spirv_ids = debug_names.debug_name_to_spirv_ids.get(api_label, [])
        return list(
            dict.fromkeys(
                schema.vgf_source_operator_id(segment_index, int(spirv_id))
                for spirv_id in matching_spirv_ids
            )
        )
    return []


def nn_module_stacks_from_api_labels(
    api_labels: list[str],
) -> list[list[dict[str, str]]]:
    """Extract all distinct ExecuTorch module stacks from API labels."""
    stacks: list[list[dict[str, str]]] = []
    for api_label in api_labels:
        decoded = _json_api_label(api_label)
        stack = _nn_module_stack_from_decoded_label(decoded) if decoded else []
        if stack and stack not in stacks:
            stacks.append(stack)
    return stacks


def code_stacks_from_api_labels(
    api_labels: list[str],
) -> list[list[dict[str, str]]]:
    """Extract all distinct ExecuTorch Python stack traces from API labels."""
    stacks: list[list[dict[str, str]]] = []
    for api_label in api_labels:
        decoded = _json_api_label(api_label)
        stack = _stack_trace_from_decoded_label(decoded) if decoded else []
        if stack and stack not in stacks:
            stacks.append(stack)
    return stacks


@dataclass
class _EntityState:
    id: str
    kind: str
    name: str
    placement: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_entity(self) -> schema.Entity:
        return schema.Entity(
            id=self.id,
            kind=self.kind,
            name=self.name,
            placement=self.placement,
            parent_ids=self.parent_ids,
            child_ids=self.child_ids,
            attributes=self.attributes,
        )


class NXProvenanceEntityBuilder:
    """Build canonical source, module, and code-stack entity hierarchies."""

    def __init__(self) -> None:
        """Initialize an empty provenance graph."""
        self._states: dict[str, _EntityState] = {}

    def has_source_operator(self, source_operator_id: str) -> bool:
        """Return whether a canonical source operator has already been added."""
        _validate_source_operator_id(source_operator_id)
        return source_operator_id in self._states

    def add_source_operator(
        self,
        source_operator_id: str,
        *,
        name: str,
        placement: str | None,
        attributes: Mapping[str, Any] | None = None,
        parent_ids: Sequence[str] = (),
        nn_module_stacks: Iterable[list[dict[str, str]]] = (),
        code_stacks: Iterable[list[dict[str, str]]] = (),
    ) -> str:
        """Add one source operator and all available provenance parents."""
        _validate_source_operator_id(source_operator_id)
        source = self._states.get(source_operator_id)
        if source is None:
            source = _EntityState(
                id=source_operator_id,
                kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                name=name,
                placement=placement,
                parent_ids=list(parent_ids),
                attributes=dict(attributes or {}),
            )
            self._states[source_operator_id] = source
        else:
            if source.placement is None:
                source.placement = placement
            elif placement is not None and source.placement != placement:
                raise ValueError(
                    "Conflicting placements for source operator "
                    f"'{source_operator_id}': '{source.placement}' and "
                    f"'{placement}'."
                )
            _append_unique_items(source.parent_ids, parent_ids)
            for key, value in (attributes or {}).items():
                source.attributes.setdefault(key, value)

        for module_stack in nn_module_stacks:
            self._add_module_hierarchy(source_operator_id, module_stack)
        for code_stack in code_stacks:
            self._add_code_hierarchy(source_operator_id, code_stack)
        return source_operator_id

    def hierarchy_entities(self) -> list[schema.Entity]:
        """Return module and code-stack entities in stable insertion order."""
        return [
            state.to_entity()
            for state in self._states.values()
            if state.kind != schema.ENTITY_KIND_SOURCE_OPERATOR
        ]

    def source_entities(self) -> list[schema.Entity]:
        """Return source operators in stable insertion order."""
        return [
            state.to_entity()
            for state in self._states.values()
            if state.kind == schema.ENTITY_KIND_SOURCE_OPERATOR
        ]

    def entities(self) -> list[schema.Entity]:
        """Return the unique provenance graph in stable hierarchy-first order."""
        return [*self.hierarchy_entities(), *self.source_entities()]

    def entity_kind_declarations(self) -> list[schema.EntityKind]:
        """Return declarations required by custom provenance kinds in the graph."""
        kinds = {state.kind for state in self._states.values()}
        return [
            declaration
            for declaration in NX_PROVENANCE_ENTITY_KIND_DECLARATIONS
            if declaration.id in kinds
        ]

    def _add_module_hierarchy(
        self, source_id: str, module_stack: list[dict[str, str]]
    ) -> None:
        previous_id = None
        for module in module_stack:
            module_id = f"{NN_MODULE_ENTITY_KIND}/{module['tracer_key']}"
            state = self._states.get(module_id)
            if state is None:
                attributes = (
                    {"module_type": module["module_type"]}
                    if "module_type" in module
                    else {}
                )
                state = _EntityState(
                    id=module_id,
                    kind=NN_MODULE_ENTITY_KIND,
                    name=module["name"],
                    attributes=attributes,
                )
                self._states[module_id] = state
            if previous_id is not None:
                self._link(previous_id, module_id)
            previous_id = module_id
        if previous_id is not None:
            self._link(previous_id, source_id)

    def _add_code_hierarchy(
        self, source_id: str, code_stack: list[dict[str, str]]
    ) -> None:
        prefix: list[dict[str, str]] = []
        previous_id = None
        for frame in code_stack:
            prefix.append(frame)
            frame_id = _code_stack_frame_prefix_id(prefix)
            state = self._states.get(frame_id)
            if state is None:
                attributes: dict[str, Any] = {
                    "file": frame["file"].replace("\\", "/"),
                    "line": int(frame["line"]),
                }
                if "function" in frame:
                    attributes["function"] = frame["function"]
                state = _EntityState(
                    id=frame_id,
                    kind=schema.ENTITY_KIND_CODE_STACK,
                    name=f"{_path_basename(frame['file'])}:{frame['line']}",
                    attributes=attributes,
                )
                self._states[frame_id] = state
            if previous_id is not None:
                self._link(previous_id, frame_id)
            previous_id = frame_id
        if previous_id is not None:
            self._link(previous_id, source_id)

    def _link(self, parent_id: str, child_id: str) -> None:
        _append_unique_items(self._states[parent_id].child_ids, [child_id])
        _append_unique_items(self._states[child_id].parent_ids, [parent_id])


def _json_api_label(api_label: str) -> dict[str, Any] | None:
    try:
        decoded = json.loads(api_label)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _nn_module_stack_from_decoded_label(
    decoded: Mapping[str, Any],
) -> list[dict[str, str]]:
    torch_info = decoded.get("torch_info")
    if not isinstance(torch_info, Mapping):
        return []
    value = torch_info.get("nn_module_stack")
    if not value or isinstance(value, str):
        return []

    raw_entries: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        raw_entries = [(str(key), entry) for key, entry in value.items()]
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            if isinstance(entry, Mapping):
                tracer_key = entry.get("tracer_key", entry.get("key", str(index)))
                if isinstance(tracer_key, str):
                    raw_entries.append((tracer_key, entry))
            elif (
                isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], str)
            ):
                raw_entries.append((entry[0], entry[1:]))

    entries = [
        {
            "tracer_key": tracer_key,
            "name": _module_name(entry, tracer_key),
            **(
                {"module_type": module_type}
                if (module_type := _module_type(entry))
                else {}
            ),
        }
        for tracer_key, entry in raw_entries
    ]
    if entries and entries[0]["tracer_key"] != "<root>":
        entries.insert(0, {"tracer_key": "<root>", "name": "<root>"})
    return entries


def _stack_trace_from_decoded_label(
    decoded: Mapping[str, Any],
) -> list[dict[str, str]]:
    torch_info = decoded.get("torch_info")
    if not isinstance(torch_info, Mapping):
        return []
    value = torch_info.get("stack_trace")
    if isinstance(value, str):
        lines = value.splitlines()
    elif isinstance(value, list):
        lines = [line for line in value if isinstance(line, str)]
    else:
        return []

    frames = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped in _STACK_TRACE_SENTINELS:
            continue
        match = _STACK_TRACE_FRAME_RE.match(line)
        if not match:
            continue
        frame = {"file": match.group("file"), "line": match.group("line")}
        if function := match.group("function"):
            frame["function"] = function
        frames.append(frame)
    return frames


def _module_name(value: Any, tracer_key: str) -> str:
    if isinstance(value, Mapping):
        for key in ("qualified_name", "name", "module_name"):
            if isinstance((name := value.get(key)), str) and name:
                return name
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return value[0]
    elif isinstance(value, str) and value:
        return value
    return tracer_key


def _module_type(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("module_type", "type", "class_name"):
            if isinstance((module_type := value.get(key)), str) and module_type:
                return module_type
    elif isinstance(value, (list, tuple)) and len(value) > 1:
        if isinstance(value[1], str) and value[1]:
            return value[1]
    return None


def _code_stack_frame_prefix_id(stack_prefix: list[dict[str, str]]) -> str:
    parts = [
        ":".join(
            (
                frame["file"].replace("\\", "/"),
                frame.get("line", ""),
                frame.get("function", ""),
            )
        )
        for frame in stack_prefix
    ]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    leaf = stack_prefix[-1]
    name = f"{_path_basename(leaf['file'])}:{leaf['line']}"
    safe_name = re.sub(r"[^A-Za-z0-9_.:-]+", "_", name).strip("_") or "frame"
    return f"{schema.ENTITY_KIND_CODE_STACK}/{safe_name}-{digest}"


def _path_basename(value: str) -> str:
    windows_name = PureWindowsPath(value).name
    posix_name = PurePosixPath(value).name
    return windows_name if len(windows_name) < len(posix_name) else posix_name


def _validate_source_operator_id(source_operator_id: str) -> None:
    """Reject noncanonical source-operator identities at the entity boundary."""
    prefix = f"{schema.ENTITY_KIND_SOURCE_OPERATOR}/"
    if not source_operator_id.startswith(prefix) or source_operator_id == prefix:
        raise ValueError(
            f"Invalid canonical source-operator ID: {source_operator_id!r}."
        )


def _append_unique_items(target: list[str], items: Iterable[str]) -> None:
    for item in items:
        if item and item not in target:
            target.append(item)
