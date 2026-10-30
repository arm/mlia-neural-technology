# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Optional integration of PyTorch conversion and measured ExecuTorch profiling."""

import importlib
import os
import struct
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from mlia.core.output_validation import validate_standardized_output
from mlia.transformers.registry import (
    TransformRequest,
    transform_model,
    transformer_registry,
)

from mlia.backend.neural_technology_profiling_data.vgf_etdump import (
    analyze_etrecord_profiling_data,
)


@pytest.mark.slow
@pytest.mark.parametrize("method_name", ["forward", "decode"])
def test_pytorch_export_and_etdump_analysis_share_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method_name: str
) -> None:
    """Export, execute and profile a real model with both plugins installed."""
    try:
        version("mlia-converters-pytorch")
    except PackageNotFoundError:
        if os.environ.get("MLIA_REQUIRE_PYTORCH_E2E") == "1":
            pytest.fail("Combined integration requires mlia-converters-pytorch")
        pytest.skip("Install the executorch extra and test dependency group")

    import torch
    from executorch.devtools.etdump.schema_flatcc import (
        ETDumpFlatCC,
        Event,
        ProfileEvent,
        RunData,
    )
    from executorch.devtools.etdump.serialize import serialize_to_etdump_flatcc
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    exporter = importlib.import_module(
        "mlia.backend.mlia_nn_module_to_pt2_exporter.exporter_plugin"
    )
    monkeypatch.setattr(transformer_registry, "items", dict(transformer_registry.items))
    exporter.NNModuleToPt2ExporterPlugin.register(transformer_registry)

    model = torch.nn.Sequential(torch.nn.ReLU()).eval()
    inputs = (torch.tensor([-2.0, 0.0, 3.0]),)
    pt2_path = transform_model(
        TransformRequest(
            model=model,
            output_dir=tmp_path,
            target_format="pt2",
            transform_options={"example_inputs": inputs},
        )
    )
    assert pt2_path.is_file()

    edge_program = to_edge_transform_and_lower(
        {method_name: torch.export.load(pt2_path)}, generate_etrecord=True
    )
    executable = edge_program.to_executorch()
    record_path = tmp_path / "model.etrecord"
    executable.get_etrecord().save(str(record_path))
    pte_path = tmp_path / "model.pte"
    pte_path.write_bytes(executable.buffer)

    runtime_program = Runtime.get().load_program(str(pte_path))
    runtime_method = runtime_program.load_method(method_name)
    for _ in range(2):
        outputs = runtime_method.execute(inputs)
        torch.testing.assert_close(outputs[0], model(*inputs))

    dump_path = tmp_path / "run.etdump"
    # Released runtime wheels need not enable on-device event tracing. Serialize
    # controlled samples for the single ReLU instruction using the real schema;
    # this exercises Inspector and source mapping without mocking either parser.
    dump_data = serialize_to_etdump_flatcc(
        ETDumpFlatCC(
            version=0,
            run_data=[
                RunData(
                    name="Execute",
                    bundled_input_index=0,
                    allocators=[],
                    events=[
                        Event(
                            profile_event=ProfileEvent(
                                name="aten::relu",
                                chain_index=0,
                                instruction_id=0,
                                delegate_debug_id_int=-1,
                                delegate_debug_id_str=None,
                                delegate_debug_metadata=None,
                                start_time=1_000_000,
                                end_time=1_000_000 + duration,
                            ),
                            allocation_event=None,
                            debug_event=None,
                        )
                    ],
                )
                for duration in (1_000_000, 2_000_000)
            ],
        )
    )
    # Inspector reads size-prefixed ETDump; the serializer emits a plain buffer.
    dump_path.write_bytes(struct.pack("<I", len(dump_data)) + dump_data)
    output = analyze_etrecord_profiling_data(
        dump_path,
        record_path,
        target_profile="src/mlia/resources/target_profiles/neural-technology.toml",
    )
    validate_standardized_output(output)
    assert output["backends"][0]["configuration"]["profiling_kind"] == (
        "standard-etdump"
    )
    result = output["results"][0]
    source_ids = {
        entity["id"]
        for entity in result["entities"]
        if entity["kind"] == "source_operator"
    }
    assert source_ids
    assert all(
        identity.startswith(f"source_operator/etrecord/{method_name}/")
        for identity in source_ids
    )
    assert any(
        breakdown["entity_id"] in source_ids
        and any(
            metric.get("unit") == "ms" and metric["value"] == pytest.approx(1.5)
            for metric in breakdown["metrics"]
        )
        for breakdown in result["breakdowns"]
    )
