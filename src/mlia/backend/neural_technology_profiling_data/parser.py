# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Parse structured Neural Technology statistics captures."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from typing import Any, Iterable, Literal

from mlia.backend.nx_performance_estimator.output_parsing import (
    DebugDatabaseContentsType,
    PerformanceDatabaseContentsType,
)
from mlia.core.errors import ConfigurationError
from mlia.utils.misc import summarize_list

ProfilingMode = Literal[0, 1]
ProfilingDevice = Literal["Mali-G1", "Mali-G2"]

_SCHEMA_VERSION = 2
_ARM_VENDOR_ID = 0x13B5
_DEVICE_GENERATION_RE = re.compile(r"^Mali-G([12])(?:$|[- ])")
_INDEX_RE = re.compile(r"^(pipeline|session|dispatch)_(\d{6,})$")
_FORBIDDEN_CONTAINER_NAMES = {"pipelines", "sessions", "dispatches"}
_OLD_FLAT_RE = re.compile(r"pipeline_0x[0-9a-f]+.*\.bin$", re.IGNORECASE)
_DEBUG_DATABASE_MAGIC = 0x1122FFEE
_STATISTICS_INFO_MAGIC = 0x2211EEFF
_CAPTURE_ARTIFACT_VERSION = 1
_BINARY_HEADER_SIZE = 12
_TEXT_INTEGER_RE = re.compile(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)")

_CAPTURE_KEYS = {
    "schema_version",
    "status",
    "error",
    "capture",
    "devices",
    "warnings",
    "pipelines",
}
_CAPTURE_SETTINGS_KEYS = {
    "layer_name",
    "layer_version",
    "commit_identity",
    "layer_implementation_version",
    "statistics_mode",
    "dispatch_filter",
}
_DEVICE_KEYS = {"id", "name", "vendor_id", "device_id", "driver_version", "api_version"}
_PIPELINE_KEYS = {
    "schema_version",
    "id",
    "friendly_name",
    "diagnostic_handle",
    "flags",
    "pipeline_layout_handle",
    "resource_bindings",
    "vendor_options",
    "identifier_only",
    "foreign_processing_engine",
    "device_id",
    "statistics_enabled",
    "shader",
    "artifacts",
    "sessions",
}
_SHADER_KEYS = {
    "module_id",
    "spirv_available",
    "friendly_name",
    "entry_point",
    "specialization_entries",
    "specialization_data_size",
}
_SESSION_KEYS = {
    "schema_version",
    "id",
    "pipeline_id",
    "device_id",
    "friendly_name",
    "diagnostic_handle",
    "flags",
    "dispatches",
}
_DISPATCH_KEYS = {
    "schema_version",
    "id",
    "session_id",
    "executed_index",
    "diagnostic_command_buffer_handle",
    "artifacts",
}
_REFERENCE_KEYS = {"id", "path"}
_ARTIFACT_KEYS = {"path", "type", "size"}
_ARTIFACT_TYPES = {
    "debug_database",
    "statistics_info",
    "shader_module",
    "dispatch_statistics",
}

_SECTION_NAMES = (
    "tosa_ops",
    "api_labels",
    "fused_ops",
    "cascade_ops",
    "cascade_groups",
    "stripe_ops",
    "stripe_indexes",
)

_MODE1_LABELS_G1: tuple[str | None, ...] = (
    None,
    "nx_active",
    "dl_active",
    "ir_active",
    "wf_active",
    "ce_active",
    "ve_active",
    "tu_active",
    "ow_active",
    "task_init_stall",
    "rd_ll_stall",
    "rd_hl_stall",
    "dma_wr_stall",
    "ir_idx_stall",
    "ce_src0_stall",
    "ce_src1_stall",
    "ve_src0_ir_stall",
    "ve_src0_ce_stall",
    "ve_src0_ve_stall",
    "ve_src0_tu_stall",
    "ve_src1_ir_stall",
    "ve_src1_ve_stall",
    "ve_src1_tu_stall",
    "tu_src0_ir_stall",
    "tu_src0_ve_stall",
    "tu_src0_tu_stall",
    "ow_src0_ir_stall",
    "ow_src0_ve_stall",
    "ow_src0_tu_stall",
    "ow_idx_stall",
)
_MODE1_LABELS_G2 = (
    *_MODE1_LABELS_G1,
    "reserved",
    "me_src0_stall",
    "me_src1_stall",
)


@dataclass(frozen=True)
class CaptureDevice:
    """One logical device declared by a structured capture."""

    id: int
    profiling_device: ProfilingDevice
    name: str
    vendor_id: int | None
    hardware_device_id: int | None


@dataclass(frozen=True)
class CaptureDispatch:
    """One referenced dispatch in a structured capture."""

    id: int
    session_id: int
    executed_index: int
    directory: Path
    statistics_path: Path


@dataclass(frozen=True)
class CaptureSession:
    """One referenced pipeline session."""

    id: int
    pipeline_id: int
    device_id: int
    directory: Path
    dispatches: tuple[CaptureDispatch, ...]


@dataclass(frozen=True)
class CapturePipeline:
    """One referenced pipeline and its static artifacts."""

    id: int
    device_id: int
    friendly_name: str
    directory: Path
    shader_module_id: int
    spirv_path: Path
    debug_database_path: Path
    statistics_info_path: Path
    sessions: tuple[CaptureSession, ...]

    @property
    def dispatches(self) -> tuple[CaptureDispatch, ...]:
        """Return all dispatches in session/reference order."""
        return tuple(
            dispatch for session in self.sessions for dispatch in session.dispatches
        )


@dataclass(frozen=True)
class StructuredCapture:
    """Validated capture hierarchy rooted by capture.json references."""

    root: Path
    devices: tuple[CaptureDevice, ...]
    mode: ProfilingMode
    warnings: tuple[str, ...]
    pipelines: tuple[CapturePipeline, ...]

    @property
    def dispatches(self) -> tuple[CaptureDispatch, ...]:
        """Return all dispatches in pipeline/session/reference order."""
        return tuple(
            dispatch for pipeline in self.pipelines for dispatch in pipeline.dispatches
        )

    def device_by_id(self, device_id: int) -> CaptureDevice:
        """Return a captured logical device by its capture-local ID."""
        for device in self.devices:
            if device.id == device_id:
                return device
        raise AssertionError(f"Capture device {device_id} is not declared.")

    def device_for_pipeline(self, pipeline: CapturePipeline) -> CaptureDevice:
        """Return the logical device used to create a pipeline."""
        return self.device_by_id(pipeline.device_id)

    def pipeline_for_dispatch(self, dispatch: CaptureDispatch) -> CapturePipeline:
        """Return the pipeline that owns a validated dispatch."""
        for pipeline in self.pipelines:
            if any(candidate.id == dispatch.id for candidate in pipeline.dispatches):
                return pipeline
        raise AssertionError(f"Dispatch {dispatch.id} has no owning pipeline.")


@dataclass(frozen=True)
class CaptureInput:
    """A user input resolved to a capture root or referenced dispatch."""

    capture: StructuredCapture
    dispatch: CaptureDispatch | None


@dataclass(frozen=True)
class ProfilingDataFiles:
    """Files selected for one measured pipeline dispatch."""

    capture_root: Path
    pipeline_dir: Path
    dispatch_dir: Path
    stats_paths: dict[ProfilingMode, Path]
    debug_database_path: Path
    statistics_info_path: Path
    source_spirv_path: Path
    device: ProfilingDevice

    @property
    def primary_mode(self) -> ProfilingMode:
        """Return the capture-declared statistics mode."""
        return next(iter(self.stats_paths))

    @property
    def primary_stats_path(self) -> Path:
        """Return the selected dispatch statistics artifact."""
        return self.stats_paths[self.primary_mode]


@dataclass(frozen=True)
class ParsedProfilingData:
    """Measured databases normalized for shared NX processing."""

    files: ProfilingDataFiles
    debug_database: DebugDatabaseContentsType
    performance_database: PerformanceDatabaseContentsType


def load_capture_input(path: Path) -> CaptureInput:
    """Resolve one directory input and validate its entire referenced capture."""
    input_path = path.expanduser()
    if not input_path.is_dir():
        raise ConfigurationError(
            f"Profiling data path '{input_path}' must be a capture or dispatch directory."
        )
    if _path_is_link(input_path):
        raise ConfigurationError(
            f"Profiling data directory '{input_path}' is a symlink or reparse point."
        )

    if (input_path / "capture.json").is_file():
        return CaptureInput(parse_capture(input_path), None)
    if (input_path / "dispatch.json").is_file():
        capture_root = _capture_root_from_dispatch_dir(input_path)
        capture = parse_capture(capture_root)
        resolved = input_path.resolve(strict=True)
        matches = [
            dispatch
            for dispatch in capture.dispatches
            if dispatch.directory.resolve(strict=True) == resolved
        ]
        if len(matches) != 1:
            raise ConfigurationError(
                f"Dispatch directory '{input_path}' is not uniquely referenced by "
                f"capture '{capture_root}'."
            )
        return CaptureInput(capture, matches[0])
    raise ConfigurationError(
        f"Profiling data directory '{input_path}' contains neither capture.json "
        "nor dispatch.json. Point --profiling-data to the capture root or to an "
        "individual pipeline/session/dispatch directory."
    )


def parse_capture(root: Path) -> StructuredCapture:
    """Parse and strictly validate a complete structured capture."""
    capture_root = root.expanduser()
    if not capture_root.is_dir() or _path_is_link(capture_root):
        raise ConfigurationError(
            f"Capture root '{capture_root}' is not an ordinary directory."
        )
    capture_root = capture_root.resolve(strict=True)
    capture = _read_document(capture_root, "capture.json", _CAPTURE_KEYS)
    if capture.get("status") != "complete":
        raise ConfigurationError(
            f"capture.json status must be 'complete', got {capture.get('status')!r}. "
            "Regenerate the capture and ensure the application and statistics "
            "layer shut down cleanly."
        )
    if capture.get("error") is not None:
        raise ConfigurationError("Complete capture.json must have a null error field.")

    settings = _require_dict(capture.get("capture"), "capture.json.capture")
    _require_keys(settings, _CAPTURE_SETTINGS_KEYS, "capture.json.capture")
    _require_nonempty_string(
        settings.get("layer_name"), "capture.json.capture.layer_name"
    )
    for field in ("layer_version", "commit_identity", "dispatch_filter"):
        _require_string(settings.get(field), f"capture.json.capture.{field}")
    _require_nonnegative_int(
        settings.get("layer_implementation_version"),
        "capture.json.capture.layer_implementation_version",
    )
    mode_value = settings.get("statistics_mode")
    if mode_value not in (0, 1) or isinstance(mode_value, bool):
        raise ConfigurationError("capture.json.capture.statistics_mode must be 0 or 1.")
    mode = mode_value

    devices_value = capture.get("devices")
    if not isinstance(devices_value, list) or not devices_value:
        raise ConfigurationError("capture.json.devices must be a nonempty array.")
    devices: list[CaptureDevice] = []
    device_ids: set[int] = set()
    for index, raw_device in enumerate(devices_value):
        label = f"capture.json.devices[{index}]"
        device_data = _require_dict(raw_device, label)
        _require_keys(device_data, _DEVICE_KEYS, label)
        logical_id = _require_nonnegative_int(device_data.get("id"), f"{label}.id")
        if logical_id in device_ids:
            raise ConfigurationError(
                f"{label} duplicates logical device ID {logical_id}."
            )
        device_ids.add(logical_id)
        device_name = _require_nonempty_string(device_data.get("name"), f"{label}.name")
        numeric_device_fields: dict[str, int | None] = {}
        for field in _DEVICE_KEYS - {"id", "name"}:
            value = device_data.get(field)
            numeric_device_fields[field] = (
                None
                if value is None
                else _require_nonnegative_int(value, f"{label}.{field}")
            )
        vendor_id = numeric_device_fields["vendor_id"]
        devices.append(
            CaptureDevice(
                id=logical_id,
                profiling_device=_profiling_device(device_name, vendor_id, label),
                name=device_name,
                vendor_id=vendor_id,
                hardware_device_id=numeric_device_fields["device_id"],
            )
        )

    warnings_value = capture.get("warnings")
    if not isinstance(warnings_value, list) or not all(
        isinstance(item, str) for item in warnings_value
    ):
        raise ConfigurationError("capture.json.warnings must be an array of strings.")

    pipeline_refs = _reference_list(capture, "pipelines", "capture.json")
    pipeline_ids: set[int] = set()
    session_ids: set[int] = set()
    dispatch_ids: set[int] = set()
    pipelines: list[CapturePipeline] = []
    for pipeline_id, relative in pipeline_refs:
        expected = f"pipeline_{pipeline_id:06d}/pipeline.json"
        _require_exact_path(relative, expected, f"pipeline reference {pipeline_id}")
        if pipeline_id in pipeline_ids:
            raise ConfigurationError(f"Duplicate global pipeline ID {pipeline_id}.")
        pipeline_ids.add(pipeline_id)
        pipelines.append(
            _parse_pipeline(
                capture_root,
                pipeline_id,
                relative,
                mode,
                device_ids,
                session_ids,
                dispatch_ids,
            )
        )

    structured_capture = StructuredCapture(
        root=capture_root,
        devices=tuple(devices),
        mode=mode,
        warnings=tuple(warnings_value),
        pipelines=tuple(pipelines),
    )
    _validate_capture_namespace(structured_capture)
    return structured_capture


def parse_profiling_data(
    capture: StructuredCapture,
    pipeline: CapturePipeline,
    dispatch: CaptureDispatch,
) -> ParsedProfilingData:
    """Parse one selected dispatch into shared estimator database shapes."""
    if pipeline not in capture.pipelines or dispatch not in pipeline.dispatches:
        raise ConfigurationError(
            "Selected profiling dispatch is not part of the capture."
        )
    files = ProfilingDataFiles(
        capture_root=capture.root,
        pipeline_dir=pipeline.directory,
        dispatch_dir=dispatch.directory,
        stats_paths={capture.mode: dispatch.statistics_path},
        debug_database_path=pipeline.debug_database_path,
        statistics_info_path=pipeline.statistics_info_path,
        source_spirv_path=pipeline.spirv_path,
        device=capture.device_for_pipeline(pipeline).profiling_device,
    )
    performance_database = parse_statistics_file(
        files.primary_stats_path,
        mode=files.primary_mode,
        device=files.device,
    )
    stripe_ids = parse_statistics_info(files.statistics_info_path)
    if len(stripe_ids) != len(performance_database):
        raise ConfigurationError(
            f"Statistics info contains {len(stripe_ids)} stripe IDs but mode "
            f"{files.primary_mode} statistics produced "
            f"{len(performance_database)} performance rows."
        )
    for row, stripe_id in zip(performance_database, stripe_ids):
        row["id"] = stripe_id
    debug_database = _complete_debug_database(
        parse_debug_database(files.debug_database_path),
        [str(row["id"]) for row in performance_database],
    )
    return ParsedProfilingData(files, debug_database, performance_database)


def _parse_pipeline(
    root: Path,
    pipeline_id: int,
    relative: str,
    mode: ProfilingMode,
    device_ids: set[int],
    global_session_ids: set[int],
    global_dispatch_ids: set[int],
) -> CapturePipeline:
    document = _read_document(root, relative, _PIPELINE_KEYS)
    label = relative
    if document.get("id") != pipeline_id:
        raise ConfigurationError(f"{label}.id does not match its reference.")
    device_id = _require_nonnegative_int(
        document.get("device_id"), f"{label}.device_id"
    )
    if device_id not in device_ids:
        raise ConfigurationError(f"{label}.device_id references an unknown device.")
    friendly_name = _require_string(
        document.get("friendly_name"), f"{label}.friendly_name"
    )
    _validate_handle(document.get("diagnostic_handle"), f"{label}.diagnostic_handle")
    _validate_handle(
        document.get("pipeline_layout_handle"), f"{label}.pipeline_layout_handle"
    )
    _require_nonnegative_int(document.get("flags"), f"{label}.flags")
    vendor_options = document.get("vendor_options")
    if vendor_options is not None:
        _require_string(vendor_options, f"{label}.vendor_options")
    for field in ("identifier_only", "foreign_processing_engine", "statistics_enabled"):
        if not isinstance(document.get(field), bool):
            raise ConfigurationError(f"{label}.{field} must be boolean.")
    if not document["statistics_enabled"]:
        raise ConfigurationError(f"{label} does not have statistics enabled.")
    _validate_bindings(document.get("resource_bindings"), f"{label}.resource_bindings")

    shader = _require_dict(document.get("shader"), f"{label}.shader")
    _require_keys(shader, _SHADER_KEYS, f"{label}.shader")
    module_id = _require_nonnegative_int(
        shader.get("module_id"), f"{label}.shader.module_id"
    )
    if shader.get("spirv_available") is not True:
        raise ConfigurationError(f"{label}.shader.spirv_available must be true.")
    _require_string(shader.get("friendly_name"), f"{label}.shader.friendly_name")
    _require_string(shader.get("entry_point"), f"{label}.shader.entry_point")
    _require_nonnegative_int(
        shader.get("specialization_data_size"),
        f"{label}.shader.specialization_data_size",
    )
    _validate_specialization_entries(
        shader.get("specialization_entries"),
        f"{label}.shader.specialization_entries",
    )

    pipeline_dir_relative = str(PurePosixPath(relative).parent)
    artifacts = _artifact_map(
        root,
        pipeline_dir_relative,
        document,
        allowed={"debug_database", "statistics_info", "shader_module"},
        label=label,
    )
    for artifact_type in ("debug_database", "statistics_info", "shader_module"):
        if len(artifacts.get(artifact_type, [])) != 1:
            raise ConfigurationError(
                f"{label} must declare exactly one {artifact_type} artifact."
            )
    debug_path = artifacts["debug_database"][0]
    if debug_path.name != "debug_database.bin":
        raise ConfigurationError(f"{label} debug database must be debug_database.bin.")
    info_path = artifacts["statistics_info"][0]
    if info_path.name not in {
        "neural_statistics_info.bin",
        "neural_statistics_info.txt",
    }:
        raise ConfigurationError(
            f"{label} statistics info must be neural_statistics_info.bin or "
            "neural_statistics_info.txt."
        )
    spirv_path = artifacts["shader_module"][0]
    expected_spirv = f"shader_module_{module_id}.spv"
    if spirv_path.name != expected_spirv:
        raise ConfigurationError(
            f"{label} shader artifact must be named {expected_spirv}."
        )

    sessions = []
    for session_id, session_relative in _reference_list(document, "sessions", label):
        expected = f"{pipeline_dir_relative}/session_{session_id:06d}/session.json"
        _require_exact_path(
            session_relative, expected, f"session reference {session_id}"
        )
        if session_id in global_session_ids:
            raise ConfigurationError(f"Duplicate global session ID {session_id}.")
        global_session_ids.add(session_id)
        sessions.append(
            _parse_session(
                root,
                pipeline_id,
                session_id,
                session_relative,
                mode,
                device_id,
                global_dispatch_ids,
            )
        )
    return CapturePipeline(
        id=pipeline_id,
        device_id=device_id,
        friendly_name=friendly_name,
        directory=_resolve_regular(root, pipeline_dir_relative, directory=True),
        shader_module_id=module_id,
        spirv_path=spirv_path,
        debug_database_path=debug_path,
        statistics_info_path=info_path,
        sessions=tuple(sessions),
    )


def _parse_session(
    root: Path,
    pipeline_id: int,
    session_id: int,
    relative: str,
    mode: ProfilingMode,
    pipeline_device_id: int,
    global_dispatch_ids: set[int],
) -> CaptureSession:
    document = _read_document(root, relative, _SESSION_KEYS)
    if document.get("id") != session_id:
        raise ConfigurationError(f"{relative}.id does not match its reference.")
    if document.get("pipeline_id") != pipeline_id:
        raise ConfigurationError(f"{relative}.pipeline_id does not match its parent.")
    device_id = _require_nonnegative_int(
        document.get("device_id"), f"{relative}.device_id"
    )
    if device_id != pipeline_device_id:
        raise ConfigurationError(f"{relative}.device_id does not match its pipeline.")
    _require_string(document.get("friendly_name"), f"{relative}.friendly_name")
    _validate_handle(document.get("diagnostic_handle"), f"{relative}.diagnostic_handle")
    _require_nonnegative_int(document.get("flags"), f"{relative}.flags")
    session_dir_relative = str(PurePosixPath(relative).parent)
    dispatches = []
    for dispatch_id, dispatch_relative in _reference_list(
        document, "dispatches", relative
    ):
        expected = f"{session_dir_relative}/dispatch_{dispatch_id:06d}/dispatch.json"
        _require_exact_path(
            dispatch_relative, expected, f"dispatch reference {dispatch_id}"
        )
        if dispatch_id in global_dispatch_ids:
            raise ConfigurationError(f"Duplicate global dispatch ID {dispatch_id}.")
        global_dispatch_ids.add(dispatch_id)
        dispatches.append(
            _parse_dispatch(root, session_id, dispatch_id, dispatch_relative, mode)
        )
    return CaptureSession(
        id=session_id,
        pipeline_id=pipeline_id,
        device_id=device_id,
        directory=_resolve_regular(root, session_dir_relative, directory=True),
        dispatches=tuple(dispatches),
    )


def _parse_dispatch(
    root: Path,
    session_id: int,
    dispatch_id: int,
    relative: str,
    mode: ProfilingMode,
) -> CaptureDispatch:
    document = _read_document(root, relative, _DISPATCH_KEYS)
    if document.get("id") != dispatch_id:
        raise ConfigurationError(f"{relative}.id does not match its reference.")
    if document.get("session_id") != session_id:
        raise ConfigurationError(f"{relative}.session_id does not match its parent.")
    executed_index = _require_nonnegative_int(
        document.get("executed_index"), f"{relative}.executed_index"
    )
    _validate_handle(
        document.get("diagnostic_command_buffer_handle"),
        f"{relative}.diagnostic_command_buffer_handle",
    )
    dispatch_dir_relative = str(PurePosixPath(relative).parent)
    artifacts = _artifact_map(
        root,
        dispatch_dir_relative,
        document,
        allowed={"dispatch_statistics"},
        label=relative,
    )
    if len(artifacts.get("dispatch_statistics", [])) != 1:
        raise ConfigurationError(
            f"{relative} must declare exactly one dispatch_statistics artifact."
        )
    statistics_path = artifacts["dispatch_statistics"][0]
    expected_name = f"statistics_mode{mode}.bin"
    if statistics_path.name != expected_name:
        raise ConfigurationError(
            f"{relative} dispatch statistics must be named {expected_name}."
        )
    return CaptureDispatch(
        id=dispatch_id,
        session_id=session_id,
        executed_index=executed_index,
        directory=_resolve_regular(root, dispatch_dir_relative, directory=True),
        statistics_path=statistics_path,
    )


def _capture_root_from_dispatch_dir(dispatch_dir: Path) -> Path:
    parts = (
        dispatch_dir.name,
        dispatch_dir.parent.name,
        dispatch_dir.parent.parent.name,
    )
    expected = ("dispatch", "session", "pipeline")
    for name, prefix in zip(parts, expected):
        match = _INDEX_RE.fullmatch(name)
        if match is None or match.group(1) != prefix:
            raise ConfigurationError(
                f"Dispatch directory '{dispatch_dir}' is not in the expected "
                "pipeline_<id>/session_<id>/dispatch_<id> hierarchy. Point "
                "--profiling-data to an individual dispatch directory emitted by "
                "the statistics layer."
            )
    root = dispatch_dir.parent.parent.parent
    if not (root / "capture.json").is_file():
        raise ConfigurationError(
            f"Dispatch directory '{dispatch_dir}' has no capture.json ancestor. "
            "Choose a dispatch directory inside a complete statistics capture."
        )
    return root


def _profiling_device(name: str, vendor_id: int | None, label: str) -> ProfilingDevice:
    """Derive the NX statistics layout from an Arm Vulkan device name."""
    if vendor_id != _ARM_VENDOR_ID:
        raise ConfigurationError(
            f"Cannot derive profiling generation from {label}: vendor_id must be "
            f"Arm's 0x{_ARM_VENDOR_ID:04X}, got {vendor_id!r}."
        )
    match = _DEVICE_GENERATION_RE.match(name)
    if match is None:
        raise ConfigurationError(
            f"Cannot derive profiling generation from {label}.name {name!r}; "
            "expected an Arm device name beginning with Mali-G1 or Mali-G2."
        )
    return "Mali-G1" if match.group(1) == "1" else "Mali-G2"


def _stat_is_link(entry_stat: os.stat_result) -> bool:
    """Return whether an entry is a symlink or Windows reparse point."""
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(entry_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(entry_stat.st_mode) or bool(file_attributes & reparse_flag)


def _path_is_link(path: Path) -> bool:
    """Return whether an existing path is a symlink or Windows reparse point."""
    try:
        return _stat_is_link(path.lstat())
    except OSError:
        return False


def _validate_capture_namespace(capture: StructuredCapture) -> None:
    """Require the complete capture root to match the referenced public namespace."""
    expected_files, expected_directories = _expected_capture_namespace(capture)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    errors: list[str] = []

    def walk_error(error: OSError) -> None:
        errors.append(f"cannot traverse capture namespace: {error}")

    for current, directory_names, file_names in os.walk(
        capture.root, topdown=True, followlinks=False, onerror=walk_error
    ):
        current_path = Path(current)
        for name in list(directory_names):
            path = current_path / name
            relative = path.relative_to(capture.root).as_posix()
            try:
                entry_stat = path.lstat()
            except OSError as err:
                errors.append(
                    f"cannot inspect capture namespace entry {relative}: {err}"
                )
                directory_names.remove(name)
                continue
            if _stat_is_link(entry_stat):
                errors.append(
                    "symlink or reparse point is forbidden in capture namespace: "
                    f"{relative}"
                )
                directory_names.remove(name)
                continue
            if not stat.S_ISDIR(entry_stat.st_mode):
                errors.append(
                    f"non-directory entry appears as a capture directory: {relative}"
                )
                directory_names.remove(name)
                continue
            actual_directories.add(relative)
            errors.extend(_namespace_name_errors(name, relative))

        for name in file_names:
            path = current_path / name
            relative = path.relative_to(capture.root).as_posix()
            try:
                entry_stat = path.lstat()
            except OSError as err:
                errors.append(
                    f"cannot inspect capture namespace entry {relative}: {err}"
                )
                continue
            if _stat_is_link(entry_stat):
                errors.append(
                    "symlink or reparse point is forbidden in capture namespace: "
                    f"{relative}"
                )
            elif not stat.S_ISREG(entry_stat.st_mode):
                errors.append(f"non-regular capture namespace entry: {relative}")
            else:
                actual_files.add(relative)
            errors.extend(_namespace_name_errors(name, relative))

    if actual_files != expected_files:
        unexpected = summarize_list(sorted(actual_files - expected_files)) or "none"
        missing = summarize_list(sorted(expected_files - actual_files)) or "none"
        errors.append(
            f"public file set mismatch: unexpected={unexpected}, missing={missing}"
        )
    if actual_directories != expected_directories:
        unexpected = (
            summarize_list(sorted(actual_directories - expected_directories)) or "none"
        )
        missing = (
            summarize_list(sorted(expected_directories - actual_directories)) or "none"
        )
        errors.append(
            f"public directory set mismatch: unexpected={unexpected}, missing={missing}"
        )
    if errors:
        raise ConfigurationError(
            "Invalid capture namespace: " + summarize_list(errors, separator="; ")
        )


def _expected_capture_namespace(
    capture: StructuredCapture,
) -> tuple[set[str], set[str]]:
    files = {"capture.json"}
    directories: set[str] = set()
    for pipeline in capture.pipelines:
        pipeline_directory = f"pipeline_{pipeline.id:06d}"
        directories.add(pipeline_directory)
        files.update(
            {
                f"{pipeline_directory}/pipeline.json",
                f"{pipeline_directory}/debug_database.bin",
                f"{pipeline_directory}/{pipeline.statistics_info_path.name}",
                f"{pipeline_directory}/shader_module_{pipeline.shader_module_id}.spv",
            }
        )
        for session in pipeline.sessions:
            session_directory = f"{pipeline_directory}/session_{session.id:06d}"
            directories.add(session_directory)
            files.add(f"{session_directory}/session.json")
            for dispatch in session.dispatches:
                dispatch_directory = f"{session_directory}/dispatch_{dispatch.id:06d}"
                directories.add(dispatch_directory)
                files.update(
                    {
                        f"{dispatch_directory}/dispatch.json",
                        f"{dispatch_directory}/statistics_mode{capture.mode}.bin",
                    }
                )
    return files, directories


def _namespace_name_errors(name: str, relative: str) -> list[str]:
    errors = []
    normalized = name.casefold()
    if normalized in _FORBIDDEN_CONTAINER_NAMES:
        errors.append(f"forbidden container name in capture namespace: {relative}")
    if normalized.endswith(".tmp"):
        errors.append(f"temporary residue remains in capture namespace: {relative}")
    if normalized.startswith(".capture-internal-"):
        errors.append(
            f"internal publication residue remains in capture namespace: {relative}"
        )
    if _OLD_FLAT_RE.fullmatch(name):
        errors.append(
            f"old flat profiling artifact remains in capture namespace: {relative}"
        )
    return errors


def _read_document(
    root: Path, relative: str, expected_keys: set[str]
) -> dict[str, Any]:
    path = _resolve_regular(root, relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ConfigurationError(f"Cannot parse '{relative}': {err}") from err
    document = _require_dict(value, relative)
    _require_keys(document, expected_keys, relative)
    if document.get("schema_version") != _SCHEMA_VERSION:
        raise ConfigurationError(
            f"{relative}.schema_version must be {_SCHEMA_VERSION}, got "
            f"{document.get('schema_version')!r}. Regenerate the capture with a "
            "compatible neural statistics layer."
        )
    return document


def _reference_list(
    document: dict[str, Any], key: str, label: str
) -> list[tuple[int, str]]:
    value = document.get(key)
    if not isinstance(value, list):
        raise ConfigurationError(f"{label}.{key} must be an array.")
    result = []
    ids: set[int] = set()
    paths: set[str] = set()
    for index, raw_reference in enumerate(value):
        ref_label = f"{label}.{key}[{index}]"
        reference = _require_dict(raw_reference, ref_label)
        _require_keys(reference, _REFERENCE_KEYS, ref_label)
        ref_id = _require_nonnegative_int(reference.get("id"), f"{ref_label}.id")
        relative = _relative_posix(reference.get("path"), f"{ref_label}.path")
        if ref_id in ids:
            raise ConfigurationError(f"{ref_label} duplicates ID {ref_id}.")
        if relative in paths:
            raise ConfigurationError(f"{ref_label} duplicates path '{relative}'.")
        ids.add(ref_id)
        paths.add(relative)
        result.append((ref_id, relative))
    return result


def _artifact_map(
    root: Path,
    owner_relative: str,
    document: dict[str, Any],
    *,
    allowed: set[str],
    label: str,
) -> dict[str, list[Path]]:
    value = document.get("artifacts")
    if not isinstance(value, list):
        raise ConfigurationError(f"{label}.artifacts must be an array.")
    artifacts: dict[str, list[Path]] = {}
    paths: set[str] = set()
    descriptors: set[tuple[str, str, int]] = set()
    for index, raw_descriptor in enumerate(value):
        item_label = f"{label}.artifacts[{index}]"
        descriptor = _require_dict(raw_descriptor, item_label)
        _require_keys(descriptor, _ARTIFACT_KEYS, item_label)
        relative = _relative_posix(descriptor.get("path"), f"{item_label}.path")
        artifact_type = descriptor.get("type")
        if artifact_type not in _ARTIFACT_TYPES:
            raise ConfigurationError(
                f"{item_label}.type has unknown value {artifact_type!r}."
            )
        if artifact_type not in allowed:
            raise ConfigurationError(
                f"{item_label} has artifact type {artifact_type!r}, which is not "
                "valid for this document."
            )
        size = _require_nonnegative_int(descriptor.get("size"), f"{item_label}.size")
        if size == 0:
            raise ConfigurationError(f"{item_label}.size must be positive.")
        identity = (relative, artifact_type, size)
        if identity in descriptors or relative in paths:
            raise ConfigurationError(
                f"{item_label} duplicates an artifact descriptor/path."
            )
        descriptors.add(identity)
        paths.add(relative)
        expected_prefix = owner_relative + "/"
        if (
            not relative.startswith(expected_prefix)
            or "/" in relative[len(expected_prefix) :]
        ):
            raise ConfigurationError(
                f"{item_label}.path must name a direct artifact of '{owner_relative}'."
            )
        artifact_path = _resolve_regular(root, relative)
        if artifact_path.stat().st_size != size:
            raise ConfigurationError(
                f"{item_label}.size declares {size} bytes but '{relative}' has "
                f"{artifact_path.stat().st_size} bytes."
            )
        artifacts.setdefault(artifact_type, []).append(artifact_path)
    return artifacts


def _resolve_exact_child(parent: Path, component: str, relative: str) -> Path:
    """Resolve one path component while preserving its exact published spelling."""
    try:
        names = {entry.name for entry in parent.iterdir()}
    except OSError:
        return parent / component
    if component in names:
        return parent / component
    mismatched = sorted(
        name for name in names if name.casefold() == component.casefold()
    )
    if mismatched:
        raise ConfigurationError(
            "Invalid capture namespace: path component "
            f"'{component}' in '{relative}' has mismatched spelling "
            f"{mismatched}."
        )
    return parent / component


def _resolve_regular(root: Path, relative: str, *, directory: bool = False) -> Path:
    canonical = _relative_posix(relative, "path")
    current = root
    for component in canonical.split("/"):
        current = _resolve_exact_child(current, component, relative)
        if _path_is_link(current):
            raise ConfigurationError(
                f"Capture path '{relative}' contains a symlink or reparse point."
            )
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as err:
        raise ConfigurationError(
            f"Capture path '{relative}' does not exist or escapes the capture root."
        ) from err
    if directory:
        if not resolved.is_dir():
            raise ConfigurationError(f"Capture path '{relative}' is not a directory.")
    elif not resolved.is_file():
        raise ConfigurationError(f"Capture path '{relative}' is not a regular file.")
    return resolved


def _relative_posix(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{label} must be a nonempty relative POSIX path.")
    if "\0" in value or "\\" in value or value.startswith(("/", "//")):
        raise ConfigurationError(f"{label} is not a canonical relative POSIX path.")
    if re.match(r"^[A-Za-z]:", value):
        raise ConfigurationError(f"{label} must not be drive-qualified.")
    components = value.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise ConfigurationError(
            f"{label} contains an empty, dot, or parent component."
        )
    canonical = PurePosixPath(*components).as_posix()
    if canonical != value or PurePosixPath(value).is_absolute():
        raise ConfigurationError(f"{label} is not canonical.")
    return canonical


def _require_exact_path(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ConfigurationError(f"{label} must be '{expected}', got '{actual}'.")


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be an object.")
    return value


def _require_keys(document: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(document)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ConfigurationError(
            f"{label} has invalid fields; missing={missing}, unknown={extra}."
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{label} must be a string.")
    return value


def _require_nonempty_string(value: object, label: str) -> str:
    result = _require_string(value, label)
    if not result:
        raise ConfigurationError(f"{label} must be nonempty.")
    return result


def _require_nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigurationError(f"{label} must be a nonnegative integer.")
    return value


def _validate_handle(value: object, label: str) -> None:
    if value is not None:
        _require_nonnegative_int(value, label)


def _validate_bindings(value: object, label: str) -> None:
    if not isinstance(value, list):
        raise ConfigurationError(f"{label} must be an array.")
    seen: set[tuple[int, int, int]] = set()
    for index, raw_binding in enumerate(value):
        item_label = f"{label}[{index}]"
        binding = _require_dict(raw_binding, item_label)
        keys = {"descriptor_set", "binding", "array_element"}
        _require_keys(binding, keys, item_label)
        identity = (
            _require_nonnegative_int(
                binding["descriptor_set"], f"{item_label}.descriptor_set"
            ),
            _require_nonnegative_int(binding["binding"], f"{item_label}.binding"),
            _require_nonnegative_int(
                binding["array_element"], f"{item_label}.array_element"
            ),
        )
        if identity in seen:
            raise ConfigurationError(f"{item_label} duplicates a resource binding.")
        seen.add(identity)


def _validate_specialization_entries(value: object, label: str) -> None:
    if not isinstance(value, list):
        raise ConfigurationError(f"{label} must be an array.")
    seen: set[tuple[int, int, int]] = set()
    for index, raw_entry in enumerate(value):
        item_label = f"{label}[{index}]"
        entry = _require_dict(raw_entry, item_label)
        keys = {"constant_id", "offset", "size"}
        _require_keys(entry, keys, item_label)
        identity = (
            _require_nonnegative_int(entry["constant_id"], f"{item_label}.constant_id"),
            _require_nonnegative_int(entry["offset"], f"{item_label}.offset"),
            _require_nonnegative_int(entry["size"], f"{item_label}.size"),
        )
        if identity in seen:
            raise ConfigurationError(f"{item_label} duplicates a specialization entry.")
        seen.add(identity)


def parse_debug_database(path: Path) -> DebugDatabaseContentsType:
    """Parse a binary debug database into the legacy MLIA relationship maps."""
    payload_bytes = _extract_versioned_payload(
        path.read_bytes(),
        expected_magic=_DEBUG_DATABASE_MAGIC,
        description="Debug database",
    )
    payload = payload_bytes.rstrip(b"\0").decode("utf-8", errors="strict")
    sections = _parse_debug_sections(payload)
    tosa_ops = _rows_by_id(sections["tosa_ops"])
    fused_ops = _rows_by_id(sections["fused_ops"])
    chain_ops = _rows_by_id(sections["cascade_ops"])
    stripe_ops = _rows_by_id(sections["stripe_ops"])
    stripe_indexes = _rows_by_id(sections["stripe_indexes"])

    stripe_index_to_op = {
        index: row["stripe_op_id"][0]
        for index, row in stripe_indexes.items()
        if row.get("stripe_op_id")
    }
    for row in stripe_indexes.values():
        for stripe_index in row.get("stripe_op_indexs", []):
            stripe_index_to_op.setdefault(stripe_index, stripe_index)

    stripe_to_chain: dict[str, list[str]] = {}
    stripe_to_cascade: dict[str, list[str]] = {}
    for stripe_index, stripe_op_id in stripe_index_to_op.items():
        stripe = stripe_ops.get(stripe_op_id)
        if stripe is None:
            continue
        stripe_to_chain[stripe_index] = stripe.get("op_id", [])
        stripe_to_cascade[stripe_index] = stripe.get("cascade_op_id", [])

    return {
        "stripe_op_id_to_op_id": stripe_to_chain,
        "stripe_op_id_to_cascade_op_id": stripe_to_cascade,
        "chain_op_id_to_fused_op_ids": {
            op_id: row.get("fused_op_ids", []) for op_id, row in chain_ops.items()
        },
        "fused_op_id_to_tosa_op_ids": {
            op_id: row.get("tosa_op_ids", []) for op_id, row in fused_ops.items()
        },
        "tosa_op_id_to_api_labels": {
            op_id: row.get("api_labels", []) for op_id, row in tosa_ops.items()
        },
        "tosa_op_id_to_tosa_op": {
            op_id: row.get("tosa_op", []) for op_id, row in tosa_ops.items()
        },
    }


def _complete_debug_database(
    debug_database: DebugDatabaseContentsType,
    stripe_ids: list[str],
) -> DebugDatabaseContentsType:
    """Fill absent debug mappings with stable stripe-local identities."""
    stripe_to_chain = debug_database["stripe_op_id_to_op_id"]
    stripe_to_cascade = debug_database["stripe_op_id_to_cascade_op_id"]
    chains = debug_database["chain_op_id_to_fused_op_ids"]
    fused_ops = debug_database["fused_op_id_to_tosa_op_ids"]
    api_labels = debug_database["tosa_op_id_to_api_labels"]
    tosa_types = debug_database["tosa_op_id_to_tosa_op"]

    for stripe_id in stripe_ids:
        chain_id = (stripe_to_chain.get(stripe_id) or [f"chain_{stripe_id}"])[0]
        cascade_id = (stripe_to_cascade.get(stripe_id) or [f"cascade_{stripe_id}"])[0]
        fused_id = (chains.get(chain_id) or [f"fused_{stripe_id}"])[0]
        tosa_id = (fused_ops.get(fused_id) or [f"tosa_{stripe_id}"])[0]
        stripe_to_chain.setdefault(stripe_id, [chain_id])
        stripe_to_cascade.setdefault(stripe_id, [cascade_id])
        chains.setdefault(chain_id, [fused_id])
        fused_ops.setdefault(fused_id, [tosa_id])
        api_labels.setdefault(tosa_id, [f"profiling-stripe/{stripe_id}"])
        tosa_types.setdefault(tosa_id, ["NeuralOp"])
    return debug_database


def parse_statistics_info(path: Path) -> list[int]:
    """Parse a captured binary or textual stripe-index list."""
    if path.suffix.lower() == ".txt":
        return _parse_text_statistics_info(path)

    data = _extract_versioned_payload(
        path.read_bytes(),
        expected_magic=_STATISTICS_INFO_MAGIC,
        description="Statistics info",
    )
    return _decode_words(data)


def _parse_text_statistics_info(path: Path) -> list[int]:
    """Parse unsigned stripe IDs separated by ASCII whitespace or punctuation."""
    try:
        text = path.read_text(encoding="utf-8").rstrip("\0")
    except (OSError, UnicodeDecodeError) as err:
        raise ConfigurationError(
            f"Cannot parse textual statistics info: {err}"
        ) from err

    normalized = re.sub(r"[,;]", " ", text)
    tokens = normalized.split()
    if not tokens:
        raise ConfigurationError("Textual statistics info is empty.")
    if any(_TEXT_INTEGER_RE.fullmatch(token) is None for token in tokens):
        raise ConfigurationError(
            "Textual statistics info must contain only unsigned integer stripe IDs."
        )

    stripe_ids = [
        int(token, 16) if token.lower().startswith("0x") else int(token, 10)
        for token in tokens
    ]
    if any(stripe_id > 0xFFFFFFFF for stripe_id in stripe_ids):
        raise ConfigurationError(
            "Textual statistics info contains a stripe ID outside uint32 range."
        )
    return stripe_ids


def parse_statistics_file(
    path: Path,
    *,
    mode: ProfilingMode,
    device: ProfilingDevice,
) -> PerformanceDatabaseContentsType:
    """Parse one raw statistics file into GCPE-compatible stripe rows."""
    words = _decode_words(path.read_bytes())
    if mode == 0:
        tasks_per_block = 512 if device == "Mali-G1" else 1024
        return _parse_mode0(words, tasks_per_block)
    words_per_task = 32 if device == "Mali-G1" else 64
    labels = _MODE1_LABELS_G1 if device == "Mali-G1" else _MODE1_LABELS_G2
    return _parse_mode1(words, words_per_task, labels)


def _parse_mode0(
    words: list[int], tasks_per_block: int
) -> PerformanceDatabaseContentsType:
    words_per_task = 16
    _validate_statistics_words(words, words_per_task, "Mode0")
    rows: PerformanceDatabaseContentsType = []
    task_count = len(words) // words_per_task
    for block_index, block_start in enumerate(range(0, task_count, tasks_per_block)):
        section_totals = [0] * words_per_task
        block_end = min(block_start + tasks_per_block, task_count)
        for task_index in range(block_start, block_end):
            task = words[
                task_index * words_per_task : (task_index + 1) * words_per_task
            ]
            if not any(task):
                continue
            section_totals = [
                total + value for total, value in zip(section_totals, task)
            ]
        if not any(section_totals):
            continue
        total_cycles = sum(section_totals)
        rows.append(
            _performance_row(
                block_index,
                op_cycles=total_cycles,
                total_cycles=total_cycles,
                counters={
                    f"section_{index}": value
                    for index, value in enumerate(section_totals)
                },
            )
        )
    return rows


def _parse_mode1(
    words: list[int],
    words_per_task: int,
    labels: tuple[str | None, ...],
) -> PerformanceDatabaseContentsType:
    _validate_statistics_words(words, words_per_task, "Mode1")
    tasks_per_block = 256
    rows: PerformanceDatabaseContentsType = []
    task_count = len(words) // words_per_task
    for block_index, block_start in enumerate(range(0, task_count, tasks_per_block)):
        counter_totals: dict[str, int] = {}
        block_end = min(block_start + tasks_per_block, task_count)
        for task_index in range(block_start, block_end):
            task = words[
                task_index * words_per_task : (task_index + 1) * words_per_task
            ]
            if not any(task):
                continue
            for word_index in range(1, min(words_per_task, len(labels))):
                label = labels[word_index]
                if label is None or label == "reserved":
                    continue
                counter_totals[label] = counter_totals.get(label, 0) + task[word_index]
        if not counter_totals:
            continue
        op_cycles = counter_totals.get("nx_active", 0)
        total_cycles = op_cycles or sum(counter_totals.values())
        rows.append(
            _performance_row(
                block_index,
                op_cycles=op_cycles,
                total_cycles=total_cycles,
                counters=counter_totals,
            )
        )
    return rows


def _performance_row(
    stripe_id: int,
    *,
    op_cycles: int,
    total_cycles: int,
    counters: dict[str, int],
) -> dict[str, object]:
    return {
        "id": stripe_id,
        "opCycles": op_cycles,
        "totalCycles": total_cycles,
        "Memory": {"Internal": {"readBytes": 0, "writeBytes": 0, "trafficCycles": 0}},
        "Utilization": [
            {"sectionName": name, "cycles": value} for name, value in counters.items()
        ],
    }


def _extract_versioned_payload(
    data: bytes,
    *,
    expected_magic: int,
    description: str,
) -> bytes:
    """Extract a current versioned payload, or return an unframed payload."""
    if len(data) < _BINARY_HEADER_SIZE:
        return data
    magic, version, payload_size = struct.unpack_from("<III", data)
    if magic != expected_magic:
        return data
    if version != _CAPTURE_ARTIFACT_VERSION:
        raise ConfigurationError(
            f"{description} uses unsupported format version {version}."
        )
    available_size = len(data) - _BINARY_HEADER_SIZE
    if payload_size != available_size:
        raise ConfigurationError(
            f"{description} declares a {payload_size}-byte payload but contains "
            f"{available_size} bytes."
        )
    return data[_BINARY_HEADER_SIZE:]


def _decode_words(data: bytes) -> list[int]:
    if len(data) % 4:
        raise ConfigurationError("Statistics file size is not 4-byte aligned.")
    return [word[0] for word in struct.iter_unpack("<I", data)]


def _validate_statistics_words(
    words: list[int], words_per_task: int, mode_name: str
) -> None:
    if not words:
        raise ConfigurationError("Statistics file is empty.")
    if len(words) % 16:
        raise ConfigurationError(
            f"{mode_name} statistics size is not aligned to 64 bytes."
        )
    if len(words) % words_per_task:
        raise ConfigurationError(
            f"{mode_name} statistics size is not a multiple of the expected task size."
        )


def _parse_debug_sections(
    payload: str,
) -> dict[str, list[dict[str, list[str]]]]:
    lines = [line.rstrip("\r\0") for line in payload.splitlines()]
    if "----" not in lines:
        raw_sections = [[line] for line in lines if line]
    else:
        raw_sections = []
        current: list[str] = []
        for line in lines:
            if line == "----":
                if current:
                    raw_sections.append(current)
                    current = []
                continue
            current.append(line)
        if current:
            raw_sections.append(current)
    if len(raw_sections) < len(_SECTION_NAMES):
        raise ConfigurationError(
            "Debug database does not contain all expected profiling sections."
        )
    return {
        name: _parse_debug_section(lines)
        for name, lines in zip(_SECTION_NAMES, raw_sections)
    }


def _parse_debug_section(lines: list[str]) -> list[dict[str, list[str]]]:
    if not lines:
        return []
    columns = lines[0].split("\t") if "\t" in lines[0] else lines[0].split()
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        rows.append(
            {
                column: _parse_debug_cell(values[index] if index < len(values) else "")
                for index, column in enumerate(columns)
            }
        )
    return rows


def _parse_debug_cell(value: str) -> list[str]:
    value = value.rstrip("\0\r")
    if not value:
        return []
    if value.endswith(";"):
        return _split_semicolon_list(value)
    return [value]


def _split_semicolon_list(value: str) -> list[str]:
    """Split top-level semicolons while preserving JSON label contents."""
    items: list[str] = []
    start = 0
    depth = 0
    quoted = False
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if quoted and character == "\\":
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if character in "[{":
            depth += 1
        elif character in "]}":
            depth = max(depth - 1, 0)
        elif character == ";" and depth == 0:
            item = value[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    final = value[start:].strip()
    if final:
        items.append(final)
    return items


def _rows_by_id(
    rows: Iterable[dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    return {row["id"][0]: row for row in rows if row.get("id") and row["id"][0]}
