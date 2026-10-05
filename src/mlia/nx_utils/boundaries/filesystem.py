# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem boundary gateways for Neural Technology runtime output."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, BinaryIO

from mlia.core.errors import ConfigurationError


def ensure_user_output_directory(path: Path, *, parents: bool = False) -> None:
    """Create a user-requested output directory when needed."""
    path.mkdir(parents=parents, exist_ok=True)


def write_user_output_bytes(path: Path, content: bytes) -> None:
    """Write bytes to a user-requested output file."""
    path.write_bytes(content)


def write_user_output_json(path: Path, value: Any) -> None:
    """Write JSON to a user-requested output file."""
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=4)


def write_user_output_binary(path: Path, writer: Callable[[BinaryIO], bool]) -> bool:
    """Write binary user output through the supplied stream writer."""
    with path.open("wb") as output:
        return writer(output)


@contextmanager
def temporary_directory(*, prefix: str) -> Generator[Path, None, None]:
    """Create and clean up a temporary runtime directory."""
    with TemporaryDirectory(prefix=prefix) as temp_dir:
        yield Path(temp_dir)


def materialize_user_output_file(source: Path, destination: Path) -> Path:
    """Atomically copy a file to a user-requested output location."""
    ensure_user_output_directory(destination.parent, parents=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            raise ConfigurationError(
                f"Cannot materialize effective model '{destination}': an existing "
                "entry has different contents."
            )
        return destination

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
