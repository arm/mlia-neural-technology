# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Build multiple wheel variants with different configurations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec
import sys
import tempfile
from pathlib import Path
from typing import Any


class VariantBuilder:
    """Handles building wheel variants with different configurations."""

    def __init__(
        self,
        mlia_root: Path,
        config_base_dir: Path | None = None,
        platform: str = "manylinux2014_x86_64",
    ):
        """
        Initialize the variant builder.

        Args:
            mlia_root: Root directory of the mlia repository
            config_base_dir: Base directory containing custom configurations
                           (defaults to ../mlia-tools/mlia-gc-sys-configs)
            platform: Platform name for the wheel (default: manylinux2014_x86_64)
        """
        self.mlia_root = mlia_root.resolve(strict=True)
        self.platform = platform

        # Set default config directory
        if config_base_dir is None:
            self.config_base_dir = (
                self.mlia_root.parent / "mlia-tools" / "mlia-gc-sys-configs"
            )
        else:
            self.config_base_dir = config_base_dir.resolve(strict=True)

        # Backup directory for original files
        self.backup_dir: Path | None = None

        # Track backed up paths for restoration
        self.backed_up_paths: dict[Path, Path] = {}

        # Track files added by variants for cleanup
        self._current_variant_files: list[Path] = []

    def load_variant_config(self, variant: str) -> dict[str, Any]:
        """
        Load variant configuration from JSON file.

        Args:
            variant: Name of the variant

        Returns:
            Dictionary with variant configuration

        Raises:
            FileNotFoundError: If variant_config.json not found
            json.JSONDecodeError: If JSON is invalid
        """
        variant_dir = self.config_base_dir / variant
        config_file = variant_dir / "variant_config.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Each variant directory must contain a 'variant_config.json' file."
            )

        with open(config_file, encoding="utf-8") as file:
            config: dict[str, Any] = json.load(file)

        # Validate required fields
        if "copy_paths" not in config:
            raise ValueError(
                f"Invalid configuration in {config_file}: missing 'copy_paths' field"
            )

        return config

    def get_available_variants(self) -> list[str]:
        """
        Get list of available configuration variants.

        Returns:
            List of variant names
        """
        if not self.config_base_dir.exists():
            return []

        variants = []
        for item in self.config_base_dir.iterdir():
            if item.is_dir():
                # Check if it has a variant_config.json file
                if (item / "variant_config.json").exists():
                    variants.append(item.name)

        return sorted(variants)

    def backup_resources(self, paths_to_backup: list[Path] | None = None) -> None:
        """
        Backup the current resources that might be overwritten.

        Args:
            paths_to_backup: List of paths to backup (relative to mlia_root).
                           If None, backs up all known resource directories.
        """
        if self.backup_dir is None:
            self.backup_dir = Path(tempfile.mkdtemp(prefix="mlia_backup_"))
            print(f"Backing up resources to {self.backup_dir}")

        if paths_to_backup is None:
            # Default backup paths
            paths_to_backup = [
                Path("src/mlia/resources/nx-performance-estimator"),
                Path("src/mlia/resources/target_profiles"),
            ]

        for rel_path in paths_to_backup:
            src_path = self.mlia_root / rel_path
            if src_path.exists():
                backup_path = self.backup_dir / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                if src_path.is_dir():
                    shutil.copytree(src_path, backup_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, backup_path)

                self.backed_up_paths[rel_path] = backup_path
                print(f"  Backed up: {rel_path}")

    def restore_resources(self) -> None:
        """Restore the original resources from backup."""
        if self.backup_dir is None or not self.backup_dir.exists():
            print("No backup to restore")
            return

        print(f"Restoring resources from {self.backup_dir}")

        for rel_path, backup_path in self.backed_up_paths.items():
            if not backup_path.exists():
                continue

            dest_path = self.mlia_root / rel_path

            # Remove existing destination if it exists
            if dest_path.exists():
                if dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()

            # Restore from backup
            if backup_path.is_dir():
                shutil.copytree(backup_path, dest_path)
            else:
                shutil.copy2(backup_path, dest_path)

            print(f"  Restored: {rel_path}")

        # Clean up backup directory
        shutil.rmtree(self.backup_dir)
        self.backup_dir = None
        self.backed_up_paths.clear()

    def apply_variant_config(self, variant: str) -> None:
        """
        Apply a specific variant configuration based on variant_config.json.

        Args:
            variant: Name of the variant to apply
        """
        variant_dir = self.config_base_dir / variant

        if not variant_dir.exists():
            raise ValueError(f"Variant directory not found: {variant_dir}")

        # Load configuration from JSON
        config = self.load_variant_config(variant)

        print(f"Applying configuration for variant: {variant}")
        if "description" in config:
            print(f"  Description: {config['description']}")

        # Track files added by this variant for potential cleanup
        variant_files = []

        # Process each copy path from configuration
        for copy_spec in config["copy_paths"]:
            src_rel = copy_spec["src"]
            dst_rel = copy_spec["dst"]

            src_path = variant_dir / src_rel
            dst_path = self.mlia_root / dst_rel

            if not src_path.exists():
                print(f"  ⚠ Warning: Source path not found: {src_path}")
                continue

            # Create destination directory if it doesn't exist
            dst_path.mkdir(parents=True, exist_ok=True)

            # Copy files (adding to existing files, not replacing)
            if src_path.is_dir():
                # Copy all files from source directory
                copied_count = 0
                for file in src_path.iterdir():
                    if file.is_file():
                        dest_file = dst_path / file.name
                        shutil.copy2(file, dest_file)
                        variant_files.append(dest_file)
                        copied_count += 1
                print(f"  ✓ Copied {copied_count} file(s) from {src_rel} to {dst_rel}")
            else:
                # Copy single file
                dest_file = dst_path / src_path.name
                shutil.copy2(src_path, dest_file)
                variant_files.append(dest_file)
                print(f"  ✓ Copied {src_path.name} to {dst_rel}")

        # Store variant files for cleanup (could be used later if needed)
        self._current_variant_files = variant_files

    def clean_variant_files(self) -> None:
        """Clean files added by the last variant configuration."""
        if self._current_variant_files:
            print("Cleaning variant-specific files...")
            for file_path in self._current_variant_files:
                if file_path.exists():
                    file_path.unlink()
                    print(f"  Removed: {file_path.relative_to(self.mlia_root)}")
            self._current_variant_files = []

    def build_wheel(self, variant: str | None = None) -> int:
        """
        Build a wheel, optionally with a specific variant tag.

        Args:
            variant: Variant name to tag the wheel with (None for default)

        Returns:
            Exit code from the build process
        """
        print("-" * 70)
        if variant:
            print(f"Building wheel for variant: {variant}")
            tag_suffix = variant
        else:
            print("Building default wheel")
            tag_suffix = None

        # Clean build directories to avoid caching issues
        build_dirs = [
            self.mlia_root / "build",
            self.mlia_root / "src" / "mlia.egg-info",
        ]
        for build_dir in build_dirs:
            if build_dir.exists():
                shutil.rmtree(build_dir)
                print(f"  Cleaned: {build_dir.relative_to(self.mlia_root)}")

        # Set up environment
        env = os.environ.copy()
        if tag_suffix:
            env["MLIA_CUSTOM_TAG_SUFFIX"] = tag_suffix

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            f"--config-setting=--build-option=--plat-name={self.platform}",
        ]

        # Run build
        result = subprocess.run(  # nosec
            cmd,
            cwd=self.mlia_root,
            env=env,
            check=False,
        )

        if result.returncode != 0:
            print(f"Build failed for variant: {variant or 'default'}")
            return result.returncode

        print(f"Successfully built wheel for variant: {variant or 'default'}")
        return 0

    def build_all_variants(
        self, variants: list[str] | None = None, include_default: bool = True
    ) -> dict[str, Any]:
        """
        Build wheels for all specified variants.

        Args:
            variants: List of variant names to build (None = auto-discover)
            include_default: Also build a default wheel without any variant config

        Returns:
            Dictionary with build results
        """
        if variants is None:
            variants = self.get_available_variants()

        if not variants and not include_default:
            print("No variants found to build")
            return {"success": False, "builds": [], "failed": []}

        results: dict[str, Any] = {
            "success": True,
            "builds": [],
            "failed": [],
        }

        # Collect all paths that need to be backed up from variant configs
        paths_to_backup = set()
        for variant in variants:
            try:
                config = self.load_variant_config(variant)
                for copy_spec in config["copy_paths"]:
                    dst_rel = copy_spec["dst"]
                    paths_to_backup.add(Path(dst_rel))
            except Exception as error:
                print(f"Warning: Could not load config for {variant}: {error}")

        try:
            # Backup original resources
            if paths_to_backup:
                self.backup_resources(list(paths_to_backup))
            else:
                self.backup_resources()  # Use defaults

            # Build default wheel first if requested
            if include_default:
                print("\n" + "=" * 70)
                print("Building DEFAULT wheel")
                print("=" * 70)

                exit_code = self.build_wheel(variant=None)
                if exit_code == 0:
                    results["builds"].append("default")
                else:
                    results["failed"].append("default")
                    results["success"] = False

            # Build each variant
            for variant in variants:
                print("\n" + "=" * 70)
                print(f"Building VARIANT wheel: {variant}")
                print("=" * 70)

                try:
                    # Clean variant-specific files from previous build (if any)
                    self.clean_variant_files()

                    # Apply variant configuration (adds to defaults)
                    self.apply_variant_config(variant)

                    # Build wheel
                    exit_code = self.build_wheel(variant=variant)

                    if exit_code == 0:
                        results["builds"].append(variant)
                    else:
                        results["failed"].append(variant)
                        results["success"] = False

                except Exception as error:
                    print(f"Error building variant {variant}: {error}")
                    results["failed"].append(variant)
                    results["success"] = False

        finally:
            # Always restore original state
            self.restore_resources()

        return results


def main() -> int:
    """Build multiple wheel variants with different configurations."""
    parser = argparse.ArgumentParser(
        description="Build multiple wheel variants with different configurations"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help=(
            "Base directory containing configurations "
            "(default: ../mlia-tools/mlia-gc-sys-configs)"
        ),
    )
    parser.add_argument(
        "--platform",
        default="manylinux2014_x86_64",
        help="Platform name for the wheel (default: manylinux2014_x86_64)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Specific variants to build (default: all available)",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available variants and exit",
    )
    parser.add_argument(
        "--no-default",
        action="store_true",
        help="Skip building the default wheel (only build variants)",
    )

    args = parser.parse_args()

    # Get mlia root directory - go up from build_tools to repo root
    mlia_root = Path(__file__).parent.parent

    # Create builder
    builder = VariantBuilder(
        mlia_root=mlia_root,
        config_base_dir=args.config_dir,
        platform=args.platform,
    )

    # Handle list-variants command
    if args.list_variants:
        variants = builder.get_available_variants()
        if variants:
            print("Available variants:")
            for variant in variants:
                print(f"  - {variant}")
        else:
            print("No variants found")
            if args.config_dir:
                print(f"Searched in: {builder.config_base_dir}")
            else:
                print(f"Default search path: {builder.config_base_dir}")
                print("Use --config-dir to specify a different location")
        return 0

    # Build wheels
    results = builder.build_all_variants(
        variants=args.variants,
        include_default=not args.no_default,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("BUILD SUMMARY")
    print("=" * 70)

    if results["builds"]:
        print(f"Successfully built {len(results['builds'])} wheel(s):")
        for build in results["builds"]:
            print(f"  - {build}")

    if results["failed"]:
        print(f"Failed to build {len(results['failed'])} wheel(s):")
        for failed in results["failed"]:
            print(f"  - {failed}")

    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
