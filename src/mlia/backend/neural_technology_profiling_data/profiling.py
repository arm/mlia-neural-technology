# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Integrate structured measured captures with shared NX performance processing."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from mlia.backend.neural_technology_profiling_data.parser import (
    CaptureDevice,
    CaptureDispatch,
    CaptureInput,
    CapturePipeline,
    ParsedProfilingData,
    StructuredCapture,
    load_capture_input,
    parse_profiling_data,
)
from mlia.backend.nx_performance_estimator.config import NXPerformanceEstimatorConfig
from mlia.backend.nx_performance_estimator.debug_info import (
    SpirvDebugNameMap,
    read_spirv_debug_names_from_bytes,
)
from mlia.backend.nx_performance_estimator.performance import (
    NXSegmentPerformanceDatabase,
    _skipped_compute_segments_warnings,
    build_performance_metrics_from_segment_databases,
)
from mlia.backend.nx_performance_estimator.statistics import NXModelPerformanceStats
from mlia.backend.nx_performance_estimator.vgf import (
    GCPEVGFSegment,
    _segment_structural_source_operator_ids,
    _segment_structural_source_operator_names,
    prepare_gcpe_compatible_vgfs,
)
from mlia.core.errors import ConfigurationError
from mlia.core.output_schema import ModeType
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.utils.misc import summarize_list

BACKEND_ID = "neural-technology-profiling-data"


@dataclass(frozen=True)
class _SelectedSegment:
    segment_index: int
    segment_name: str
    debug_names: SpirvDebugNameMap
    structural_source_operator_ids: list[str]
    structural_source_operator_names: dict[str, str]
    pipeline: CapturePipeline
    dispatch: CaptureDispatch


def analyze_profiling_data(
    *,
    target_profile: str,
    profiling_data: list[Path],
    categories: set[str],
    model: str | None = None,
    cli_arguments: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Analyze structured measured captures through shared NX processing."""
    model_path = Path(model) if model is not None else None
    if model_path is not None and model_path.suffix.lower() != ".vgf":
        raise ConfigurationError(
            "Neural Technology profiling data can only be associated with a VGF model."
        )
    if "compatibility" in categories:
        raise ConfigurationError(
            "Backend does not support --compatibility with --profiling-data. "
            "Use --performance instead."
        )
    if "performance" not in categories:
        raise ConfigurationError(
            "Neural Technology profiling data currently supports performance "
            "analysis only."
        )
    if not profiling_data:
        raise ConfigurationError("At least one profiling data directory is required.")

    inputs = [load_capture_input(path) for path in profiling_data]
    capture = _require_same_capture(inputs)
    if model_path is None:
        selected = _select_without_vgf(inputs)
        source_path = selected[0].pipeline.spirv_path
        skipped_compute_segments: list[int] = []
    else:
        selected, skipped_compute_segments = _select_with_vgf(
            inputs, capture, model_path
        )
        source_path = model_path

    capture_device = _selected_device(capture, selected)
    parsed_segments = [
        parse_profiling_data(capture, item.pipeline, item.dispatch) for item in selected
    ]
    segment_databases = [
        NXSegmentPerformanceDatabase(
            segment_index=item.segment_index,
            segment_name=item.segment_name,
            debug_database=parsed.debug_database,
            performance_database=parsed.performance_database,
            model_performance_stats=_model_performance_stats(
                parsed.performance_database
            ),
            debug_names=item.debug_names,
            structural_source_operator_ids=item.structural_source_operator_ids,
            structural_source_operator_names=item.structural_source_operator_names,
        )
        for item, parsed in zip(selected, parsed_segments)
    ]
    warnings = list(capture.warnings)
    warnings.extend(_skipped_compute_segments_warnings(skipped_compute_segments))
    metrics = build_performance_metrics_from_segment_databases(
        backend_config=NXPerformanceEstimatorConfig(
            system_config="default", compiler_config="default"
        ),
        segment_databases=segment_databases,
        warnings=warnings,
    )

    if model_path is None and output_dir is not None:
        _materialize_effective_model(source_path, output_dir)

    target = NeuralTechnologyConfiguration.load_profile(target_profile)
    output = metrics.to_standardized_output(
        model_path=source_path,
        backend_name=BACKEND_ID,
        target_config={
            "target": target.target,
            "target_type": target.target,
            "profile_name": target.profile_name,
        },
        cli_arguments=cli_arguments,
    )
    _apply_measured_metadata(output, capture, capture_device, parsed_segments)
    return output


def _materialize_effective_model(source_path: Path, output_dir: Path) -> Path:
    """Atomically copy the capture-only effective model into the output root."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source_path.name
    if destination.exists():
        if (
            not destination.is_file()
            or destination.read_bytes() != source_path.read_bytes()
        ):
            raise ConfigurationError(
                f"Cannot materialize effective model '{destination}': an existing "
                "entry has different contents."
            )
        return destination

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{source_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _require_same_capture(inputs: list[CaptureInput]) -> StructuredCapture:
    first = inputs[0].capture
    first_root = first.root.resolve(strict=True)
    for item in inputs[1:]:
        if item.capture.root.resolve(strict=True) != first_root:
            raise ConfigurationError(
                "All --profiling-data dispatch directories must belong to the same "
                "capture root. Choose dispatch directories from one capture."
            )
    return first


def _select_without_vgf(inputs: list[CaptureInput]) -> list[_SelectedSegment]:
    if len(inputs) != 1:
        raise ConfigurationError(
            "Multiple --profiling-data directories require a supplied VGF model. "
            "Without a VGF, provide exactly one individual dispatch directory."
        )
    capture_input = inputs[0]
    dispatch = capture_input.dispatch
    if dispatch is None:
        dispatches = capture_input.capture.dispatches
        if len(dispatches) != 1:
            choices = summarize_list((item.directory for item in dispatches), limit=3)
            raise ConfigurationError(
                f"The capture contains {len(dispatches)} dispatches, so MLIA cannot "
                "choose one without a VGF model. Set --profiling-data to an "
                "individual dispatch directory. Available dispatch directories: "
                f"{choices}."
            )
        dispatch = dispatches[0]
    pipeline = capture_input.capture.pipeline_for_dispatch(dispatch)
    spirv = pipeline.spirv_path.read_bytes()
    debug_names = read_spirv_debug_names_from_bytes(spirv)
    return [
        _SelectedSegment(
            segment_index=0,
            segment_name=pipeline.friendly_name or pipeline.directory.name,
            debug_names=debug_names,
            structural_source_operator_ids=_segment_structural_source_operator_ids(
                0, debug_names
            ),
            structural_source_operator_names=(
                _segment_structural_source_operator_names(0, debug_names)
            ),
            pipeline=pipeline,
            dispatch=dispatch,
        )
    ]


def _select_with_vgf(
    inputs: list[CaptureInput],
    capture: StructuredCapture,
    model_path: Path,
) -> tuple[list[_SelectedSegment], list[int]]:
    with TemporaryDirectory(prefix="mlia-vgf-segments-") as temp_dir:
        vgf_segments = prepare_gcpe_compatible_vgfs(model_path, Path(temp_dir))
        segments = list(vgf_segments)
        skipped = list(getattr(vgf_segments, "skipped_compute_segments", []))
    if not segments:
        raise ConfigurationError("The supplied VGF contains no graph segments.")
    if any(not segment.spirv for segment in segments):
        raise ConfigurationError(
            "Unable to extract exact SPIR-V bytes for every VGF graph segment."
        )

    if len(inputs) > 1:
        dispatches = []
        seen: set[Path] = set()
        if len(inputs) != len(segments):
            raise ConfigurationError(
                f"Explicit profiling dispatch count {len(inputs)} does not match "
                f"VGF graph segment count {len(segments)}. Repeat --profiling-data "
                f"exactly {len(segments)} times, once per graph segment in VGF order."
            )
        for capture_input in inputs:
            if capture_input.dispatch is None:
                raise ConfigurationError(
                    "When --profiling-data is repeated, every entry must be an "
                    "individual dispatch directory. Replace capture-root entries and "
                    "provide one dispatch per VGF graph segment, in VGF order."
                )
            directory = capture_input.dispatch.directory.resolve(strict=True)
            if directory in seen:
                raise ConfigurationError(
                    f"Duplicate profiling dispatch directory '{directory}'. Provide "
                    "one distinct dispatch directory per VGF graph segment."
                )
            seen.add(directory)
            dispatches.append(capture_input.dispatch)
        return (
            [
                _selected_vgf_segment(
                    segment,
                    _require_dispatch_matches_segment(capture, dispatch, segment),
                    dispatch,
                )
                for segment, dispatch in zip(segments, dispatches)
            ],
            skipped,
        )

    capture_input = inputs[0]
    if capture_input.dispatch is None:
        return _select_vgf_from_capture_root(capture, segments), skipped
    return _select_vgf_from_anchor(capture, segments, capture_input.dispatch), skipped


def _select_vgf_from_capture_root(
    capture: StructuredCapture, segments: list[GCPEVGFSegment]
) -> list[_SelectedSegment]:
    _reject_duplicate_segment_spirv(segments)
    selected = []
    for segment in segments:
        pipeline = _unique_pipeline_for_spirv(
            capture, segment.spirv, segment.segment_index
        )
        if len(pipeline.dispatches) != 1:
            choices = summarize_list(
                (dispatch.directory for dispatch in pipeline.dispatches), limit=3
            )
            raise ConfigurationError(
                f"Pipeline {pipeline.id} matching VGF segment {segment.segment_index} "
                f"has {len(pipeline.dispatches)} dispatches, so MLIA cannot choose one. "
                "Set --profiling-data to an individual dispatch directory for any "
                "segment, or repeat it once per graph segment in VGF order. "
                f"Available dispatch directories: {choices}."
            )
        selected.append(
            _selected_vgf_segment(segment, pipeline, pipeline.dispatches[0])
        )
    return selected


def _select_vgf_from_anchor(
    capture: StructuredCapture,
    segments: list[GCPEVGFSegment],
    anchor: CaptureDispatch,
) -> list[_SelectedSegment]:
    _reject_duplicate_segment_spirv(segments)
    anchor_pipeline = capture.pipeline_for_dispatch(anchor)
    anchor_spirv = anchor_pipeline.spirv_path.read_bytes()
    matching_segments = [
        segment for segment in segments if segment.spirv == anchor_spirv
    ]
    if len(matching_segments) != 1:
        raise ConfigurationError(
            "The selected dispatch pipeline does not match exactly one VGF graph "
            "segment. Verify that this VGF produced the capture, or repeat "
            "--profiling-data once per graph segment in VGF order."
        )
    anchor_segment_index = matching_segments[0].segment_index

    selected = []
    for segment in segments:
        pipeline = _unique_pipeline_for_spirv(
            capture, segment.spirv, segment.segment_index
        )
        if segment.segment_index == anchor_segment_index:
            if pipeline.id != anchor_pipeline.id:
                raise ConfigurationError(
                    "Anchor dispatch resolved to an inconsistent pipeline."
                )
            dispatch = anchor
        else:
            candidates = [
                dispatch
                for dispatch in pipeline.dispatches
                if dispatch.executed_index == anchor.executed_index
            ]
            if len(candidates) != 1:
                raise ConfigurationError(
                    f"Automatic matching found {len(candidates)} dispatches for VGF "
                    f"segment {segment.segment_index} in pipeline {pipeline.id} with "
                    f"executed_index {anchor.executed_index}. Repeat --profiling-data "
                    "once per graph segment, using explicit dispatch directories in "
                    "VGF order."
                )
            dispatch = candidates[0]
        selected.append(_selected_vgf_segment(segment, pipeline, dispatch))
    return selected


def _reject_duplicate_segment_spirv(segments: list[GCPEVGFSegment]) -> None:
    seen: set[bytes] = set()
    for segment in segments:
        if segment.spirv in seen:
            raise ConfigurationError(
                "The VGF contains graph segments with duplicate SPIR-V, so MLIA "
                "cannot distinguish them automatically. Repeat --profiling-data "
                "once per graph segment, using explicit dispatch directories in "
                "VGF order."
            )
        seen.add(segment.spirv)


def _unique_pipeline_for_spirv(
    capture: StructuredCapture, spirv: bytes, segment_index: int
) -> CapturePipeline:
    matches = [
        pipeline
        for pipeline in capture.pipelines
        if pipeline.spirv_path.read_bytes() == spirv
    ]
    if not matches:
        raise ConfigurationError(
            f"No captured pipeline SPIR-V matches VGF graph segment {segment_index}. "
            "Verify that the VGF is the model used to produce this capture."
        )
    if len(matches) > 1:
        raise ConfigurationError(
            f"VGF graph segment {segment_index} matches {len(matches)} captured "
            "pipelines, so MLIA cannot choose one automatically. Repeat "
            "--profiling-data once per graph segment, using explicit dispatch "
            "directories in VGF order."
        )
    return matches[0]


def _require_dispatch_matches_segment(
    capture: StructuredCapture,
    dispatch: CaptureDispatch,
    segment: GCPEVGFSegment,
) -> CapturePipeline:
    pipeline = capture.pipeline_for_dispatch(dispatch)
    if pipeline.spirv_path.read_bytes() != segment.spirv:
        raise ConfigurationError(
            f"Dispatch {dispatch.id} pipeline SPIR-V does not match VGF graph "
            f"segment {segment.segment_index}. Ensure repeated --profiling-data "
            "values are ordered to match the VGF graph segments."
        )
    return pipeline


def _selected_vgf_segment(
    segment: GCPEVGFSegment,
    pipeline: CapturePipeline,
    dispatch: CaptureDispatch,
) -> _SelectedSegment:
    return _SelectedSegment(
        segment_index=segment.segment_index,
        segment_name=segment.segment_name,
        debug_names=segment.debug_names,
        structural_source_operator_ids=list(segment.structural_source_operator_ids),
        structural_source_operator_names=dict(segment.structural_source_operator_names),
        pipeline=pipeline,
        dispatch=dispatch,
    )


def _selected_device(
    capture: StructuredCapture, selected: list[_SelectedSegment]
) -> CaptureDevice:
    """Require one logical device for all dispatches in an inference."""
    device_ids = {item.pipeline.device_id for item in selected}
    if len(device_ids) != 1:
        raise ConfigurationError(
            "Selected profiling dispatches span multiple logical devices. Choose "
            "dispatch directories whose pipeline.json files reference the same "
            "device_id."
        )
    return capture.device_by_id(next(iter(device_ids)))


def _model_performance_stats(
    performance_database: list[dict[str, Any]],
) -> NXModelPerformanceStats:
    compute_cycles = sum(int(row["opCycles"]) for row in performance_database)
    total_cycles = sum(int(row["totalCycles"]) for row in performance_database)
    return NXModelPerformanceStats(
        compiled_size=None,
        cache_cycles=None,
        cache_read_bytes=None,
        cache_write_bytes=None,
        compute_cycles=compute_cycles,
        dram_cycles=None,
        dram_read_bytes=None,
        dram_write_bytes=None,
        dram_footprint=None,
        inference_time=None,
        infs_per_sec=None,
        total_cycles=total_cycles,
    )


def _apply_measured_metadata(
    output: dict[str, object],
    capture: StructuredCapture,
    device: CaptureDevice,
    parsed_segments: list[ParsedProfilingData],
) -> None:
    backends = output.get("backends")
    if isinstance(backends, list) and backends and isinstance(backends[0], dict):
        backends[0].update(
            {
                "id": BACKEND_ID,
                "name": "Neural Technology Profiling Data",
                "version": "experimental",
                "configuration": {
                    "device": device.name,
                    "profiling_device": device.profiling_device,
                    "vendor_id": device.vendor_id,
                    "device_id": device.hardware_device_id,
                    "mode": capture.mode,
                    "capture_root": str(capture.root),
                    "statistics_files": [
                        str(parsed.files.primary_stats_path)
                        for parsed in parsed_segments
                    ],
                    "debug_database_files": [
                        str(parsed.files.debug_database_path)
                        for parsed in parsed_segments
                    ],
                },
            }
        )

    context = output.get("context")
    if isinstance(context, dict):
        context["runtime_configuration"] = {
            "profiling_data": [
                str(parsed.files.dispatch_dir) for parsed in parsed_segments
            ]
        }

    target = output.get("target")
    if isinstance(target, dict):
        target["description"] = "Neural Technology measured profiling data"

    results = output.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        results[0]["mode"] = ModeType.MEASURED.value
