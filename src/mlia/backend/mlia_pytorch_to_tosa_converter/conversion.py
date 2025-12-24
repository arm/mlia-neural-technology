# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert PyTorch models to TOSA format using the PyTorch to TOSA converter."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from mlia.utils.logging import log_action
from mlia.utils.proc import OutputConsumer
from mlia.utils.proc import OutputLogger

# Lazy imports - populated on first use to avoid import errors during backend discovery
torch: Any = None
get_symmetric_quantization_config: Any = None
TOSAQuantizer: Any = None
TosaCompileSpec: Any = None
TOSAPartitioner: Any = None
EdgeCompileConfig: Any = None
to_edge_transform_and_lower: Any = None
convert_pt2e: Any = None
prepare_pt2e: Any = None
DEPENDENCIES_LOADED = False


def _import_dependencies() -> None:
    """Import runtime dependencies. Raises ImportError if not available.

    This function populates module-level variables to avoid import errors
    during backend discovery, while still allowing the imports to happen
    lazily when actually needed.
    """
    # pylint: disable=global-statement,import-outside-toplevel
    # Justification: Lazy loading pattern to defer heavy imports until runtime
    global torch, get_symmetric_quantization_config, TOSAQuantizer
    global TosaCompileSpec, TOSAPartitioner, EdgeCompileConfig
    global to_edge_transform_and_lower, convert_pt2e, prepare_pt2e
    global DEPENDENCIES_LOADED

    if DEPENDENCIES_LOADED:
        return

    import torch as _torch

    torch = _torch
    from executorch.backends.arm.quantizer import (
        get_symmetric_quantization_config as _get_config,
        TOSAQuantizer as _TOSAQuantizer,
    )
    from executorch.backends.arm.tosa.compile_spec import (
        TosaCompileSpec as _TosaCompileSpec,
    )
    from executorch.backends.arm.tosa.partitioner import (
        TOSAPartitioner as _TOSAPartitioner,
    )
    from executorch.exir import EdgeCompileConfig as _EdgeCompileConfig
    from executorch.exir import to_edge_transform_and_lower as _to_edge
    from torchao.quantization.pt2e.quantize_pt2e import (
        convert_pt2e as _convert_pt2e,
        prepare_pt2e as _prepare_pt2e,
    )

    get_symmetric_quantization_config = _get_config
    TOSAQuantizer = _TOSAQuantizer
    TosaCompileSpec = _TosaCompileSpec
    TOSAPartitioner = _TOSAPartitioner
    EdgeCompileConfig = _EdgeCompileConfig
    to_edge_transform_and_lower = _to_edge
    convert_pt2e = _convert_pt2e
    prepare_pt2e = _prepare_pt2e
    DEPENDENCIES_LOADED = True


logger = logging.getLogger(__name__)

# Constants for TOSA conversion
DEFAULT_TOSA_TARGET = "TOSA-1.0+INT"
DEFAULT_BASE_NAME = "tosa_simple"
EXPECTED_OUTPUT_FILENAME = "output_tag1_TOSA-1.0+INT.tosa"


class MliaPytorchToTosaConverter:
    """The TOSA Converter For PyTorch class."""

    def __init__(self) -> None:
        """Set up output consumers for the TOSA Converter For PyTorch."""
        self.output_consumers: list[OutputConsumer] = [
            OutputLogger(logger, logging.INFO)
        ]

    def __call__(self, pytorch_file: Path, output_dir: Path) -> Path:
        """
        Run the TOSA Converter For PyTorch with the given PyTorch file.

        Returns the path of the TOSA output file created in the output dir.
        """
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                "TOSA Converter For PyTorch."
            )
        with log_action("Running TOSA Converter For PyTorch..."):
            logger.debug("TOSA Converter For PyTorch:")

            tosa_file = self._run_converter(pytorch_file, output_dir)

            logger.debug("Output file: %s", tosa_file)

        return tosa_file

    def _load_pytorch_model(self, pytorch_file: Path) -> tuple[Any, Any]:
        """Load PyTorch model and extract graph module and example inputs."""
        if pytorch_file.suffix != ".pt2":
            raise ValueError(
                "Unsupported model file type. Only .pt2 files are supported."
            )

        # Bandit complains here as torch.export.load() uses pickle internally which
        # can execute arbitrary code. However this is designed to convert .pt2
        # files so we have to load them.
        try:
            loaded = torch.export.load(
                pytorch_file
            )  # nosec B614  # type: ignore[union-attr]
        except Exception as exc:
            raise ValueError(
                f"Failed to load PyTorch export file {pytorch_file}: {exc}"
            ) from exc

        # Get the GraphModule from ExportedProgram
        graph_module = loaded.module(check_guards=False)
        full_example_inputs = loaded.example_inputs

        # Convert from (args, kwargs) format to just args
        example_inputs = (
            full_example_inputs[0]
            if isinstance(full_example_inputs, tuple)
            else full_example_inputs
        )

        return graph_module, example_inputs

    def _setup_quantization(self, output_dir: Path, base_name: str) -> tuple[Any, Any]:
        """Set up TOSA compilation spec and quantizer."""
        # Create a compilation spec describing the target for
        # configuring the quantizer. Dump intermediate artifacts
        # (TOSA flat buffers) to specified location
        compile_spec = TosaCompileSpec(
            DEFAULT_TOSA_TARGET
        ).dump_intermediate_artifacts_to(str(output_dir / base_name))

        # Patch NodeVisitor to include node names as location info for debugging
        self._patch_node_visitor_for_location()

        # Create and configure quantizer to use a symmetric quantization config
        # globally on all nodes
        quantizer = TOSAQuantizer(compile_spec)
        operator_config = get_symmetric_quantization_config()
        quantizer.set_global(operator_config)

        return compile_spec, quantizer

    def _patch_node_visitor_for_location(self) -> None:
        """Patch NodeVisitor node names as TOSA operator locations."""
        try:
            # pylint: disable=import-outside-toplevel
            from executorch.backends.arm.operators.node_visitor import (
                NodeVisitor,
            )

            # pylint: disable=unused-argument
            def _serialize_operator_with_node_name(  # type: ignore[no-untyped-def]
                self,
                node,
                tosa_graph,
                tosa_op,
                inputs,
                outputs,
                attributes=None,
            ):
                # Use node name as location for traceability
                op_location = node.name if node else ""

                tosa_graph.addOperator(
                    tosa_op,
                    inputs=inputs,
                    outputs=outputs,
                    attributes=attributes,
                    location=op_location,
                )

            # pylint: disable=protected-access
            NodeVisitor._serialize_operator = _serialize_operator_with_node_name
            logger.debug("Patched NodeVisitor to include node names in TOSA locations")

        except ImportError as exc:
            logger.warning("Could not patch NodeVisitor: %s", exc)

    def _quantize_model(
        self, graph_module: Any, quantizer: Any, example_inputs: Any
    ) -> Any:
        """Perform post-training quantization on the model."""
        quantized_graph_module = prepare_pt2e(graph_module, quantizer)
        quantized_graph_module(
            *example_inputs
        )  # Calibrate the graph module with the example input
        quantized_graph_module = convert_pt2e(quantized_graph_module)

        # Create a new exported program using the quantized_graph_module
        return torch.export.export(quantized_graph_module, example_inputs)

    def _lower_to_tosa(self, lowered_exported_program: Any, compile_spec: Any) -> None:
        """Lower the exported program to the TOSA backend."""
        partitioner = TOSAPartitioner(compile_spec)

        try:
            _ = to_edge_transform_and_lower(
                lowered_exported_program,
                partitioner=[partitioner],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
        except Exception as exc:
            logger.error("Error during lowering to TOSA: %s", exc)
            logger.debug("Full traceback:", exc_info=True)
            raise RuntimeError(f"TOSA lowering failed: {exc}") from exc

    def _move_output_file(
        self, pytorch_file: Path, output_dir: Path, base_name: str
    ) -> Path:
        """Move the generated TOSA file to the output location."""
        tosa_file = output_dir / f"{pytorch_file.stem}.tosa"
        source_file = output_dir / base_name / EXPECTED_OUTPUT_FILENAME

        try:
            shutil.move(str(source_file), str(tosa_file))
        except FileNotFoundError as fnfe:
            raise FileNotFoundError(
                f"Expected TOSA output file not found at {source_file}. "
                "TOSA conversion may have failed."
            ) from fnfe

        if not tosa_file.is_file():
            raise FileNotFoundError(
                "No output from the TOSA Converter For PyTorch found. "
                f"File {tosa_file} does not exist."
            )

        logger.debug(
            "TOSA Converter For PyTorch run successfully. See output: %s", tosa_file
        )
        return tosa_file

    def _run_converter(self, pytorch_file: Path, output_dir: Path) -> Path:
        """Run the TOSA Converter For PyTorch and return the TOSA MLIR output file."""
        # Import dependencies at runtime
        _import_dependencies()

        # Validate input file
        if not pytorch_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {pytorch_file}")
        if not pytorch_file.is_file():
            raise ValueError(f"Input path is not a file: {pytorch_file}")

        # Step 1: Load the model
        graph_module, example_inputs = self._load_pytorch_model(pytorch_file)

        # Step 2: Set up quantization
        compile_spec, quantizer = self._setup_quantization(
            output_dir, DEFAULT_BASE_NAME
        )

        # Step 3: Quantize the model
        lowered_exported_program = self._quantize_model(
            graph_module, quantizer, example_inputs
        )

        # Step 4: Lower to TOSA backend
        self._lower_to_tosa(lowered_exported_program, compile_spec)

        # Step 5: Move output file to final location
        return self._move_output_file(pytorch_file, output_dir, DEFAULT_BASE_NAME)
