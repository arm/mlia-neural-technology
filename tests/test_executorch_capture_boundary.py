# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target analysis must work after the SDK has left the artifact boundary."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("profiling_kind", ["runtime", "vgf", "mixed"])
def test_runtime_capture_analysis_without_torch_or_executorch(profiling_kind):
    """A fresh process can attribute samples using only detached capture data."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import base64
import dataclasses
import importlib.abc
import json
import struct
import sys

class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "executorch"}:
            raise AssertionError(f"Target analysis imported {fullname}")

sys.meta_path.insert(0, BlockSDK())
from mlia.core.performance import MeasuredPerformanceMetrics
from mlia.backend.executorch.capture import ETDumpEvent, ProfilingCapture
from mlia.backend.neural_technology_profiling_data.vgf_etdump import analyze_capture

node = {"name": "relu", "debug_handle": 42, "target": "aten.relu.default",
        "graph_index": 0, "op": "call_function"}
capture = ProfilingCapture(
    method="forward", etrecord_path="model.etrecord", etdump_path="run.etdump",
    events=(ETDumpEvent("forward", "Execute", 0, 0, "aten::relu", 0, None,
                        (42,), None, (), (1.0, 3.0)),),
    nodes_by_handle={42: [node]}, instruction_handles={0: (42,)},
    source_graph={"method": "forward", "nodes": [], "edges": []},
)
profiling_kind = sys.argv[1]
if profiling_kind != "runtime":
    sections = [
        "id\\ttosa_op\\tapi_labels\\n10\\tConv2D\\tTOSACONV2D_spirv_id_10;",
        "id\\tapi_label", "id\\ttosa_op_ids\\n2\\t10;",
        "id\\tfused_op_ids\\n3\\t2;", "id\\tcascade_op_ids\\n4\\t3;",
        "id\\top_id\\tcascade_op_id\\n5\\t3;\\t4;", "id\\tstripe_op_id\\n0\\t5;",
    ]
    def blob(data, **attributes):
        return dict(available=True, encoding="base64",
                    data=base64.b64encode(data).decode("ascii"), **attributes)
    metadata = json.dumps(dict(
        schema="executorch.vgf.neural_statistics", schema_version=1,
        segments=[dict(segment_id=7,
            debug_database=blob("\\n----\\n".join(sections).encode()),
            statistics_info=blob(b"0", is_text=True),
            statistics_memory=blob(struct.pack("<II", 0, 100) + bytes(4*(64*256-2))))],
    ))
    vgf_event = ETDumpEvent("forward", "Execute", 0, 1, "VGF_NEURAL_STATISTICS",
                           1, "VgfBackend", (42,), None, (metadata,), None)
    capture = dataclasses.replace(
        capture, events=(vgf_event,) if profiling_kind == "vgf" else (*capture.events, vgf_event),
        instruction_handles={0: (42,), 1: (42,)},
    )
metrics = analyze_capture(capture)
assert isinstance(metrics, MeasuredPerformanceMetrics)
if profiling_kind == "runtime":
    assert metrics.measured_observations[0].breakdowns[0].metrics[0].value == 2.0
else:
    assert next(metric.value for metric in metrics.measured_observations[0].metrics
                if metric.name == "total_cycles") == 100
    assert len(metrics.measured_observations) == (2 if profiling_kind == "mixed" else 1)
assert not any(name.split(".")[0] in {"torch", "executorch"} for name in sys.modules)
""",
            profiling_kind,
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
