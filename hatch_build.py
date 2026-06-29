# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Hatch build hook to download vendored artifacts at build time."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from typing import Literal, TypedDict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ENV_URLS = "VENDORED_ARTIFACTS_URLS"
ENV_USER = "UV_INDEX_INTERNAL_USERNAME"
ENV_TOKEN = "UV_INDEX_INTERNAL_PASSWORD"


class ArtifactSpec(TypedDict, total=False):
    """Spec describing a vendored artifact."""

    vendor_dir: Path
    type: Literal["tar", "whl"]


ARTIFACTS: dict[str, ArtifactSpec] = {
    "graph-compiler-performance-estimator": {
        "vendor_dir": Path("mlia/_vendor/artifacts/nx-performance-estimator"),
        "type": "tar",
    }
}


def _read_expected_sha256(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise RuntimeError(f"Empty sha256 file: {path}")
    parts = content.split()
    if len(parts) < 2:
        raise RuntimeError(f"Invalid sha256 file format: {path}")
    return parts[0], parts[1]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    username = os.environ.get(ENV_USER)
    token = os.environ.get(ENV_TOKEN)
    if not username or not token:
        raise RuntimeError(
            f"Missing Artifactory credentials. Set {ENV_USER} and {ENV_TOKEN}."
        )

    req = urllib.request.Request(url)
    req.add_header("Username", username)
    req.add_header("X-JFrog-Art-Api", token)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=30) as resp, dest.open("wb") as out:
        for chunk in iter(lambda: resp.read(1024 * 1024), b""):
            out.write(chunk)


def _download_and_verify(url: str, dest: Path, expected_sha: str) -> None:
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    _download_file(url, tmp_path)
    actual_sha = _file_sha256(tmp_path)
    if actual_sha != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded vendor artifact hash mismatch. "
            f"Expected {expected_sha}, got {actual_sha}."
        )
    tmp_path.replace(dest)


def _load_vendor_urls() -> dict[str, str]:
    raw = os.environ.get(ENV_URLS, "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{ENV_URLS} must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{ENV_URLS} must be a JSON object mapping keys to URLs.")
    return data


class CustomBuildHook(BuildHookInterface):
    """Download vendored artifacts before building."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Ensure vendored artifacts are present with expected hashes."""
        del version
        force_include: dict[str, str] = build_data.setdefault("force_include", {})

        root = Path(self.root)
        src_root = root / "src"
        base_root = src_root if (src_root / "mlia").exists() else root
        alt_root = root if base_root is src_root else src_root

        urls = _load_vendor_urls()
        for key, spec in ARTIFACTS.items():
            vendor_dir = base_root / spec["vendor_dir"]
            sha_path = vendor_dir / ".sha256"
            if not sha_path.exists():
                alt_vendor_dir = alt_root / spec["vendor_dir"]
                alt_sha = alt_vendor_dir / ".sha256"
                if alt_sha.exists():
                    vendor_dir = alt_vendor_dir
                    sha_path = alt_sha
                else:
                    raise FileNotFoundError(f"Missing sha256 file: {sha_path}")
            expected_sha, expected_name = _read_expected_sha256(sha_path)
            archive_path = vendor_dir / expected_name
            target_rel = f"mlia/_vendor/artifacts/{vendor_dir.name}/{expected_name}"
            try:
                source_rel = archive_path.relative_to(root)
            except ValueError:
                try:
                    source_rel = archive_path.relative_to(src_root)
                except ValueError:
                    source_rel = archive_path
            force_include[str(source_rel)] = target_rel

            if archive_path.exists():
                actual_sha = _file_sha256(archive_path)
                if actual_sha == expected_sha:
                    continue
                archive_path.unlink()

            url = urls.get(key)
            if not url:
                raise RuntimeError(
                    f"Missing vendor URL for '{key}'. Set {ENV_URLS} with a "
                    "JSON map including this key."
                )

            if spec["type"] not in {"whl", "tar"}:
                raise RuntimeError(f"Unsupported vendor artifact type: {spec['type']}")

            _download_and_verify(url, archive_path, expected_sha)
