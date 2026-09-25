# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Hatch build hook to download vendored artifacts at build time."""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import urllib.request
from pathlib import Path
from string import Template
from typing import Literal, TypedDict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface
from packaging import tags

ENV_URLS = "VENDORED_ARTIFACTS_URLS"
ENV_USER = "UV_INDEX_INTERNAL_USERNAME"
ENV_TOKEN = "UV_INDEX_INTERNAL_PASSWORD"


def _replace_markdown_relative_paths(path: Path, file_name: str, revision: str) -> str:
    """Replace relative Markdown paths with links to the GitHub revision."""
    md_url = Template("https://github.com/arm/mlia-neural-technology/blob/$rev/$link")
    img_url = Template(
        "https://raw.githubusercontent.com/arm/mlia-neural-technology/$rev/$link"
    )
    md_link_pattern = r"(!?\[.+?\]\((.+?)\))"

    content = path.joinpath(file_name).read_text(encoding="utf-8")
    for match, link in re.findall(md_link_pattern, content):
        if link.startswith("#") or path.joinpath(link).exists():
            if link.startswith("#"):
                new_url = md_url.substitute(rev=revision, link=file_name + link)
            else:
                template = img_url if match[0] == "!" else md_url
                new_url = template.substitute(rev=revision, link=link)
            target = f"({link})"
            content = content.replace(
                match,
                match.replace(target, f"({new_url})", 1),
            )
    return content


def _get_revision(root: Path, fallback: str) -> str:
    """Return the current commit hash, or fallback when Git is unavailable."""
    try:
        worktree = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        # The source package placed inside a git repo,
        # but not a git repo itself
        if Path(worktree.stdout.strip()).resolve() != root.resolve():
            return fallback

        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback

    return revision.stdout.strip() or fallback


def _get_revision_fallback(version: str) -> str:
    """Extract the commit hash from a development version when available."""
    match = re.fullmatch(r"\d+\.\d+\.\d+\.dev\d+\+([A-Za-z0-9]+)", version)
    return match.group(1) if match else f"v{version}"


class MetadataHook(MetadataHookInterface):
    """Rewrite relative links in the packaged README metadata."""

    def update(self, metadata: dict) -> None:
        """Replace the README path with its rewritten Markdown content."""
        root = Path(self.root)
        version = str(metadata.get("version", ""))
        revision_tag = _get_revision(root, _get_revision_fallback(version))
        readme = _replace_markdown_relative_paths(root, "README.md", revision_tag)
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": readme,
        }


class ArtifactSpec(TypedDict):
    """Spec describing a vendored artifact."""

    vendor_dir: Path
    type: Literal["tar", "whl"]
    sha256_by_platform: dict[str, str]


ARTIFACTS: dict[str, ArtifactSpec] = {
    "graph-compiler-performance-estimator": {
        "vendor_dir": Path("mlia/_vendor/artifacts/nx-performance-estimator"),
        "type": "tar",
        "sha256_by_platform": {
            "Linux": ".sha256.linux",
            "Windows": ".sha256.windows",
        },
    }
}

VGF_BUILD_DIR = Path(".native/build")
MINIMUM_GLIBC = (2, 39)  # Ubuntu 24.04


def _linux_wheel_tag() -> str:
    """Preserve the native Python ABI and enforce the minimum Linux baseline."""
    for tag in tags.sys_tags():
        match = re.fullmatch(r"manylinux_(\d+)_(\d+)_(.+)", tag.platform)
        if match:
            # A newer build host may introduce newer glibc requirements.
            # Never advertise a baseline older than that host supports.
            major, minor = max(MINIMUM_GLIBC, (int(match[1]), int(match[2])))
            return str(
                tags.Tag(
                    tag.interpreter,
                    tag.abi,
                    f"manylinux_{major}_{minor}_{match[3]}",
                )
            )
    raise RuntimeError("Linux wheels must be built on a glibc-based system.")


def _find_built_vgfpy(build_root: Path) -> Path:
    """Find the vgfpy extension produced by the upstream CMake build."""
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    matches = sorted(
        path
        for path in build_root.rglob("vgfpy*")
        if path.is_file()
        and path.name.startswith("vgfpy")
        and path.name.endswith(suffixes)
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find a built vgfpy extension under {build_root}."
        )

    for path in matches:
        if "vgf_library-build" in path.parts:
            return path
    return matches[0]


def _build_vgfpy(root: Path) -> Path:
    """Temporarily build vgfpy from source until PyPI provides usable wheels."""
    build_root = root / VGF_BUILD_DIR
    subprocess.run(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_root),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython3_INCLUDE_DIR={sysconfig.get_path('include')}",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_root),
            "--target",
            "vgfpy",
            "--config",
            "Release",
        ],
        check=True,
    )
    return _find_built_vgfpy(build_root)


def _read_expected_sha256(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise RuntimeError(f"Empty sha256 file: {path}")
    parts = content.split()
    if len(parts) < 2:
        raise RuntimeError(f"Invalid sha256 file format: {path}")
    return parts[0], parts[1]


def _platform_sha256_filename(spec: ArtifactSpec) -> str:
    current_platform = platform.system()
    try:
        return spec["sha256_by_platform"][current_platform]
    except KeyError as exc:
        supported = ", ".join(sorted(spec["sha256_by_platform"]))
        raise RuntimeError(
            f"Unsupported platform '{current_platform}'. Supported platforms: {supported}."
        ) from exc


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
    """Build native bindings and download vendored artifacts before packaging."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Prepare native and downloaded artifacts for the selected target."""
        del version
        force_include: dict[str, str] = build_data.setdefault("force_include", {})

        root = Path(self.root)
        if getattr(self, "target_name", "wheel") == "wheel":
            if platform.system() == "Linux":
                build_data["tag"] = _linux_wheel_tag()
            vgfpy = _build_vgfpy(root)
            force_include[str(vgfpy.relative_to(root))] = vgfpy.name
            build_data["pure_python"] = False
            build_data["infer_tag"] = True

        src_root = root / "src"
        base_root = src_root if (src_root / "mlia").exists() else root
        alt_root = root if base_root is src_root else src_root

        urls = _load_vendor_urls()
        for key, spec in ARTIFACTS.items():
            sha256_filename = _platform_sha256_filename(spec)
            vendor_dir = base_root / spec["vendor_dir"]
            sha_path = vendor_dir / sha256_filename
            if not sha_path.exists():
                alt_vendor_dir = alt_root / spec["vendor_dir"]
                alt_sha = alt_vendor_dir / sha256_filename
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
