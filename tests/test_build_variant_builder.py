# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the variant builder module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import from build_tools directory (not installed package)

sys.path.insert(0, str(Path(__file__).parent.parent / "build_tools"))

from variant_builder import VariantBuilder  # noqa: E402


@pytest.fixture
def temp_workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """
    Create a temporary workspace with mlia root and config directories.

    Returns:
        Tuple of (mlia_root, config_base_dir, resources_dir, target_profiles_dir)
    """
    mlia_root = tmp_path / "mlia"
    config_base_dir = tmp_path / "mlia-tools" / "mlia-gc-sys-configs"

    # Create mlia directory structure
    resources_dir = (
        mlia_root / "src" / "mlia" / "resources" / "nx-performance-estimator"
    )
    resources_dir.mkdir(parents=True)

    target_profiles_dir = mlia_root / "src" / "mlia" / "resources" / "target_profiles"
    target_profiles_dir.mkdir(parents=True)

    # Create some default resource files
    (resources_dir / "default-system-config.ini").write_text("test\n")
    (resources_dir / "default-compiler.ini").write_text("test\n")
    (target_profiles_dir / "default-profile.toml").write_text("test\n")

    # Create config base directory
    config_base_dir.mkdir(parents=True)

    return mlia_root, config_base_dir, resources_dir, target_profiles_dir


@pytest.fixture
def sample_variant_config(
    temp_workspace: tuple[Path, Path, Path, Path],
) -> tuple[Path, dict[str, Any]]:
    """
    Create a sample variant configuration.

    Returns:
        Tuple of (variant_dir, config_dict)
    """
    _, config_base_dir, _, _ = temp_workspace
    variant_name = "test_variant"
    variant_dir = config_base_dir / variant_name
    variant_dir.mkdir(parents=True)

    # Create variant configuration with both config paths
    config = {
        "name": variant_name,
        "description": "Test variant for unit tests",
        "copy_paths": [
            {
                "src": "configurations",
                "dst": "src/mlia/resources/nx-performance-estimator",
            },
            {
                "src": "target_profile",
                "dst": "src/mlia/resources/target_profiles",
            },
        ],
    }

    # Write config file
    config_file = variant_dir / "variant_config.json"
    config_file.write_text(json.dumps(config, indent=2))

    # Create source files to copy (matching real file types)
    configs_dir = variant_dir / "configurations"
    configs_dir.mkdir()
    (configs_dir / "variant-system-config.ini").write_text(
        "[VARIANT]\nsystem = config\n"
    )
    (configs_dir / "variant-compiler.ini").write_text("[VARIANT]\ncompiler = config\n")

    target_dir = variant_dir / "target_profile"
    target_dir.mkdir()
    (target_dir / "variant-profile.toml").write_text("[variant]\nprofile = config\n")

    return variant_dir, config


class TestVariantBuilder:
    """Test VariantBuilder class."""

    def test_init_default_config_dir(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test initialization with default config directory."""
        mlia_root, _, _, _ = temp_workspace

        builder = VariantBuilder(mlia_root)

        assert builder.mlia_root == mlia_root.resolve()
        assert builder.config_base_dir == (
            mlia_root.parent / "mlia-tools" / "mlia-gc-sys-configs"
        )
        assert builder.platform == "manylinux2014_x86_64"
        assert builder.backup_dir is None
        assert not builder.backed_up_paths
        assert not builder._current_variant_files

    def test_init_custom_config_dir(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test initialization with custom config directory."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)

        assert builder.mlia_root == mlia_root.resolve()
        assert builder.config_base_dir == config_base_dir.resolve()

    def test_init_custom_platform(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test initialization with custom platform."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        builder = VariantBuilder(
            mlia_root, config_base_dir=config_base_dir, platform="linux_aarch64"
        )

        assert builder.platform == "linux_aarch64"

    def test_load_variant_config_success(
        self,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test loading a valid variant configuration."""
        mlia_root, config_base_dir, _, _ = temp_workspace
        _, expected_config = sample_variant_config

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        config = builder.load_variant_config("test_variant")

        assert config["name"] == expected_config["name"]
        assert config["description"] == expected_config["description"]
        assert config["copy_paths"] == expected_config["copy_paths"]

    def test_load_variant_config_missing_file(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test loading a non-existent variant configuration."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            builder.load_variant_config("nonexistent")

    def test_load_variant_config_invalid_json(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test loading a variant with invalid JSON."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        # Create variant with invalid JSON
        variant_dir = config_base_dir / "bad_variant"
        variant_dir.mkdir(parents=True)
        config_file = variant_dir / "variant_config.json"
        config_file.write_text("{ invalid json }")

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)

        with pytest.raises(json.JSONDecodeError):
            builder.load_variant_config("bad_variant")

    def test_load_variant_config_missing_copy_paths(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test loading a variant without copy_paths field."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        # Create variant without copy_paths
        variant_dir = config_base_dir / "incomplete_variant"
        variant_dir.mkdir(parents=True)
        config_file = variant_dir / "variant_config.json"
        config_file.write_text(json.dumps({"name": "incomplete"}))

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)

        with pytest.raises(ValueError, match="missing 'copy_paths' field"):
            builder.load_variant_config("incomplete_variant")

    def test_get_available_variants(
        self,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test getting list of available variants."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        # Create additional variant
        variant2_dir = config_base_dir / "variant2"
        variant2_dir.mkdir()
        (variant2_dir / "variant_config.json").write_text(
            json.dumps({"copy_paths": []})
        )

        # Create directory without config (should be ignored)
        (config_base_dir / "not_a_variant").mkdir()

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        variants = builder.get_available_variants()

        assert sorted(variants) == ["test_variant", "variant2"]

    def test_get_available_variants_empty(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test getting variants when none are available."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        variants = builder.get_available_variants()

        assert variants == []

    def test_get_available_variants_missing_config_dir(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test getting variants when config directory doesn't exist."""
        mlia_root, _, _, _ = temp_workspace
        nonexistent_dir = mlia_root.parent / "nonexistent"

        with pytest.raises(FileNotFoundError):
            builder = VariantBuilder(mlia_root, config_base_dir=nonexistent_dir)
            variants = builder.get_available_variants()

            assert not variants

    def test_backup_default_paths(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test backing up default resource paths."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        builder.backup_resources()

        # Check backup was created
        assert builder.backup_dir is not None
        assert builder.backup_dir.exists()

        # Check files were backed up from both directories
        nx_backup_path = builder.backed_up_paths[
            Path("src/mlia/resources/nx-performance-estimator")
        ]
        assert (nx_backup_path / "default-system-config.ini").exists()
        assert (nx_backup_path / "default-compiler.ini").exists()

        target_backup_path = builder.backed_up_paths[
            Path("src/mlia/resources/target_profiles")
        ]
        assert (target_backup_path / "default-profile.toml").exists()

    def test_backup_custom_paths(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test backing up custom resource paths."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        # Create custom file to backup
        custom_dir = mlia_root / "custom"
        custom_dir.mkdir()
        custom_file = custom_dir / "file.txt"
        custom_file.write_text("test content")

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        builder.backup_resources([Path("custom")])

        # Check backup
        assert Path("custom") in builder.backed_up_paths
        backup_path = builder.backed_up_paths[Path("custom")]
        assert (backup_path / "file.txt").read_text() == "test content"

    def test_restore_resources(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test restoring backed up resources."""
        mlia_root, config_base_dir, resources_dir, target_profiles_dir = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)

        builder.backup_resources(
            [
                Path("src/mlia/resources/nx-performance-estimator"),
                Path("src/mlia/resources/target_profiles"),
            ]
        )
        backup_dir = builder.backup_dir
        assert backup_dir is not None

        # Modify original files in both directories
        (resources_dir / "default-system-config.ini").write_text("modified")
        (target_profiles_dir / "default-profile.toml").write_text("modified")

        # Restore
        builder.restore_resources()

        # Check restoration
        assert not backup_dir.exists()
        assert builder.backup_dir is None
        assert not builder.backed_up_paths

        # Check both directories were restored
        resources_content = (resources_dir / "default-system-config.ini").read_text()
        assert "test\n" in resources_content

        target_content = (target_profiles_dir / "default-profile.toml").read_text()
        assert "test\n" in target_content

    def test_restore_resources_no_backup(
        self,
        temp_workspace: tuple[Path, Path, Path, Path],
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test restoring when no backup exists."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        builder.restore_resources()

        captured = capsys.readouterr()
        assert "No backup to restore" in captured.out

    def test_apply_variant_config(
        self,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test applying a variant configuration."""
        mlia_root, config_base_dir, resources_dir, target_profiles_dir = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        builder.apply_variant_config("test_variant")

        # Check variant files were copied (in addition to defaults)
        assert (resources_dir / "default-system-config.ini").exists()
        assert (resources_dir / "variant-system-config.ini").exists()
        assert (resources_dir / "variant-compiler.ini").exists()
        assert (target_profiles_dir / "default-profile.toml").exists()
        assert (target_profiles_dir / "variant-profile.toml").exists()

        assert len(builder._current_variant_files) == 3
        assert (
            resources_dir / "variant-system-config.ini"
            in builder._current_variant_files
        )
        assert (
            target_profiles_dir / "variant-profile.toml"
            in builder._current_variant_files
        )

    def test_apply_variant_config_missing_source(
        self,
        temp_workspace: tuple[Path, Path, Path, Path],
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test applying variant with missing source files."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        # Create variant with non-existent source
        variant_dir = config_base_dir / "bad_variant"
        variant_dir.mkdir()
        config = {
            "name": "bad_variant",
            "copy_paths": [{"src": "nonexistent", "dst": "src/mlia/resources"}],
        }
        (variant_dir / "variant_config.json").write_text(json.dumps(config))

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        builder.apply_variant_config("bad_variant")

        captured = capsys.readouterr()
        assert "Warning: Source path not found" in captured.out

    def test_clean_variant_files(
        self,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test cleaning variant-specific files."""
        mlia_root, config_base_dir, resources_dir, target_profiles_dir = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)

        # Apply variant
        builder.apply_variant_config("test_variant")
        variant_file = resources_dir / "variant-system-config.ini"
        variant_profile = target_profiles_dir / "variant-profile.toml"
        assert variant_file.exists()
        assert variant_profile.exists()

        # Clean variant files
        builder.clean_variant_files()

        # Check variant files removed, defaults preserved
        assert not variant_file.exists()
        assert not variant_profile.exists()
        assert (resources_dir / "default-system-config.ini").exists()
        assert (target_profiles_dir / "default-profile.toml").exists()
        assert not builder._current_variant_files

    def test_clean_variant_files_empty(
        self, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test cleaning when no variant files are tracked."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        builder.clean_variant_files()  # Should not raise

        assert not builder._current_variant_files

    @patch("subprocess.run")
    @patch("shutil.rmtree")
    def test_build_wheel_default(
        self,
        mock_rmtree: Mock,
        mock_run: Mock,
        temp_workspace: tuple[Path, Path, Path, Path],
    ) -> None:
        """Test building a default wheel."""
        mlia_root, config_base_dir, _, _ = temp_workspace
        mock_run.return_value.returncode = 0

        # Create build directories to be cleaned
        build_dir = mlia_root / "build"
        build_dir.mkdir()
        egg_info = mlia_root / "src" / "mlia.egg-info"
        egg_info.mkdir(parents=True)

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        exit_code = builder.build_wheel(variant=None)

        assert exit_code == 0
        assert mock_rmtree.call_count == 2

        # Check subprocess was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.kwargs["cwd"] == mlia_root
        assert "MLIA_CUSTOM_TAG_SUFFIX" not in call_args.kwargs["env"]

    @patch("subprocess.run")
    @patch("shutil.rmtree")
    def test_build_wheel_variant(
        self,
        mock_rmtree: Mock,
        mock_run: Mock,
        temp_workspace: tuple[Path, Path, Path, Path],
    ) -> None:
        """Test building a variant wheel."""
        mlia_root, config_base_dir, _, _ = temp_workspace
        mock_run.return_value.returncode = 0

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        exit_code = builder.build_wheel(variant="test_variant")

        assert exit_code == 0
        call_args = mock_run.call_args
        assert call_args.kwargs["env"]["MLIA_CUSTOM_TAG_SUFFIX"] == "test_variant"

    @patch("subprocess.run")
    def test_build_wheel_failure(
        self, mock_run: Mock, temp_workspace: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test handling build failure."""
        mlia_root, config_base_dir, _, _ = temp_workspace
        mock_run.return_value.returncode = 1

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        exit_code = builder.build_wheel(variant="test_variant")

        assert exit_code == 1

    @patch.object(VariantBuilder, "build_wheel")
    @patch.object(VariantBuilder, "apply_variant_config")
    def test_build_all_variants_success(
        self,
        mock_apply: Mock,
        mock_build: Mock,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test building all variants successfully."""
        mlia_root, config_base_dir, _, _ = temp_workspace
        mock_build.return_value = 0

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        results = builder.build_all_variants(variants=["test_variant"])

        # Check results
        assert results["success"] is True
        assert "default" in results["builds"]
        assert "test_variant" in results["builds"]
        assert not results["failed"]

        # Check build was called for default and variant
        assert mock_build.call_count == 2

    @patch.object(VariantBuilder, "build_wheel")
    def test_build_all_variants_no_default(
        self,
        mock_build: Mock,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test building variants without default."""
        mlia_root, config_base_dir, _, _ = temp_workspace
        mock_build.return_value = 0

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        results = builder.build_all_variants(
            variants=["test_variant"], include_default=False
        )

        # Check only variant was built
        assert results["builds"] == ["test_variant"]
        assert mock_build.call_count == 1

    @patch.object(VariantBuilder, "build_wheel")
    def test_build_all_variants_failure(
        self,
        mock_build: Mock,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test handling build failures."""
        mlia_root, config_base_dir, _, _ = temp_workspace
        mock_build.return_value = 1

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        results = builder.build_all_variants(variants=["test_variant"])

        # Check failures recorded
        assert results["success"] is False
        assert "default" in results["failed"]
        assert "test_variant" in results["failed"]
        assert not results["builds"]

    @patch.object(VariantBuilder, "restore_resources")
    @patch.object(VariantBuilder, "apply_variant_config")
    def test_build_all_variants_exception_handling(
        self,
        mock_apply: Mock,
        mock_restore: Mock,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test that resources are restored even on exception."""
        mlia_root, config_base_dir, _, _ = temp_workspace
        mock_apply.side_effect = RuntimeError("Build error")

        builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
        results = builder.build_all_variants(variants=["test_variant"])

        # Check restore was called despite exception
        mock_restore.assert_called_once()

        # Check failure recorded
        assert results["success"] is False
        assert "test_variant" in results["failed"]

    def test_build_all_variants_auto_discover(
        self,
        temp_workspace: tuple[Path, Path, Path, Path],
        sample_variant_config: tuple[Path, dict[str, Any]],
    ) -> None:
        """Test auto-discovering variants."""
        mlia_root, config_base_dir, _, _ = temp_workspace

        with patch.object(VariantBuilder, "build_wheel", return_value=0):
            builder = VariantBuilder(mlia_root, config_base_dir=config_base_dir)
            results = builder.build_all_variants(variants=None)

            # Check auto-discovered variant was built
            assert "test_variant" in results["builds"]


class TestMain:
    """Test main CLI function."""

    @patch("variant_builder.VariantBuilder")
    @patch("sys.argv", ["variant_builder.py", "--list-variants"])
    def test_main_list_variants(
        self,
        mock_builder_class: Mock,
        temp_workspace: tuple[Path, Path, Path],
    ) -> None:
        """Test listing variants through CLI."""
        from variant_builder import main

        mock_builder = MagicMock()
        mock_builder.get_available_variants.return_value = ["variant1", "variant2"]
        mock_builder_class.return_value = mock_builder

        exit_code = main()

        assert exit_code == 0
        mock_builder.get_available_variants.assert_called_once()

    @patch("variant_builder.VariantBuilder")
    @patch("sys.argv", ["variant_builder.py"])
    def test_main_build_all(self, mock_builder_class: Mock) -> None:
        """Test building all variants through CLI."""
        from variant_builder import main

        mock_builder = MagicMock()
        mock_builder.build_all_variants.return_value = {
            "success": True,
            "builds": ["default", "variant1"],
            "failed": [],
        }
        mock_builder_class.return_value = mock_builder

        exit_code = main()

        assert exit_code == 0
        mock_builder.build_all_variants.assert_called_once()

    @patch("variant_builder.VariantBuilder")
    @patch("sys.argv", ["variant_builder.py", "--no-default"])
    def test_main_no_default(self, mock_builder_class: Mock) -> None:
        """Test building without default wheel."""
        from variant_builder import main

        mock_builder = MagicMock()
        mock_builder.build_all_variants.return_value = {
            "success": True,
            "builds": ["variant1"],
            "failed": [],
        }
        mock_builder_class.return_value = mock_builder

        exit_code = main()

        assert exit_code == 0
        call_args = mock_builder.build_all_variants.call_args
        assert call_args.kwargs["include_default"] is False

    @patch("variant_builder.VariantBuilder")
    @patch(
        "sys.argv",
        ["variant_builder.py", "--variants", "variant1", "variant2"],
    )
    def test_main_specific_variants(self, mock_builder_class: Mock) -> None:
        """Test building specific variants."""
        from variant_builder import main

        mock_builder = MagicMock()
        mock_builder.build_all_variants.return_value = {
            "success": True,
            "builds": ["variant1", "variant2"],
            "failed": [],
        }
        mock_builder_class.return_value = mock_builder

        exit_code = main()

        assert exit_code == 0
        call_args = mock_builder.build_all_variants.call_args
        assert call_args.kwargs["variants"] == ["variant1", "variant2"]

    @patch("variant_builder.VariantBuilder")
    @patch("sys.argv", ["variant_builder.py", "--platform", "linux_aarch64"])
    def test_main_custom_platform(self, mock_builder_class: Mock) -> None:
        """Test specifying custom platform."""
        from variant_builder import main

        mock_builder = MagicMock()
        mock_builder.build_all_variants.return_value = {
            "success": True,
            "builds": [],
            "failed": [],
        }
        mock_builder_class.return_value = mock_builder

        main()

        # Check builder was created with custom platform
        call_args = mock_builder_class.call_args
        assert call_args.kwargs["platform"] == "linux_aarch64"

    @patch("variant_builder.VariantBuilder")
    @patch(
        "sys.argv",
        ["variant_builder.py", "--config-dir", "/custom/path"],
    )
    def test_main_custom_config_dir(self, mock_builder_class: Mock) -> None:
        """Test specifying custom config directory."""
        from variant_builder import main

        mock_builder = MagicMock()
        mock_builder.build_all_variants.return_value = {
            "success": True,
            "builds": [],
            "failed": [],
        }
        mock_builder_class.return_value = mock_builder

        main()

        # Check builder was created with custom config dir
        call_args = mock_builder_class.call_args
        assert call_args.kwargs["config_base_dir"] == Path("/custom/path")

    @patch("variant_builder.VariantBuilder")
    @patch("sys.argv", ["variant_builder.py"])
    def test_main_build_failure(self, mock_builder_class: Mock) -> None:
        """Test handling build failure in main."""
        from variant_builder import main

        mock_builder = MagicMock()
        mock_builder.build_all_variants.return_value = {
            "success": False,
            "builds": ["variant1"],
            "failed": ["variant2"],
        }
        mock_builder_class.return_value = mock_builder

        exit_code = main()

        assert exit_code == 1
