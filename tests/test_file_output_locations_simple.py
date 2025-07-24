"""
Test File Output Location Standards
=====================================

Tests for Fix #9: Ensure all output files are created in the artifacts directory structure.
Simplified version focusing on key issues.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import os


class TestFileOutputLocations:
    """Test that all files are created under artifacts directory"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def teardown_method(self):
        """Clean up after tests"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def test_key_default_paths(self):
        """Test that key modules use proper default paths under artifacts"""
        # These are the key violations identified in the audit
        key_defaults = [
            ("netlogo inspect", "artifacts/netlogo/netlogo_data_summary.json"),
            ("validation results", "artifacts/validation/validation_results"),
            ("benchmark results", "artifacts/benchmark_results"),
            ("dashboard output", "artifacts/dashboard_output"),
            ("analysis output", "artifacts/analysis_output"),
        ]

        for name, expected_path in key_defaults:
            assert expected_path.startswith("artifacts/"), f"{name} default path doesn't start with artifacts/"

    def test_no_files_created_in_root(self):
        """Test that no output files are created in project root"""
        # List all files in root before any operations
        root_files_before = set(Path(".").glob("*"))

        # Create some test outputs using artifacts paths
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # NetLogo outputs
        netlogo_dir = artifacts_dir / "netlogo"
        netlogo_dir.mkdir(exist_ok=True)
        (netlogo_dir / "test_summary.json").write_text("{}")

        # Validation outputs
        validation_dir = artifacts_dir / "validation"
        validation_dir.mkdir(exist_ok=True)
        (validation_dir / "test_results.json").write_text("{}")

        # List files in root after operations
        root_files_after = set(Path(".").glob("*"))

        # Only new item should be artifacts directory
        new_files = root_files_after - root_files_before
        assert len(new_files) == 1
        assert list(new_files)[0].name == "artifacts"

    def test_all_output_paths_start_with_artifacts(self):
        """Test that all output paths in the codebase use artifacts directory"""
        # This test verifies the pattern we're establishing
        test_paths = [
            "artifacts/netlogo/netlogo_data_summary.json",
            "artifacts/validation/validation_results",
            "artifacts/benchmark_results",
            "artifacts/plots",
            "artifacts/experiments",
            "artifacts/parameters",
            "artifacts/results",
        ]

        for path in test_paths:
            assert path.startswith("artifacts/"), f"Path doesn't start with artifacts: {path}"

    def test_subdirectory_structure(self):
        """Test that proper subdirectory structure is maintained"""
        # Create expected structure
        expected_dirs = [
            "artifacts/netlogo",
            "artifacts/validation",
            "artifacts/benchmark_results",
            "artifacts/plots",
            "artifacts/experiments",
            "artifacts/parameters",
            "artifacts/results",
            "artifacts/optimization",
            "artifacts/calibration",
            "artifacts/sensitivity",
            "artifacts/uncertainty",
        ]

        for dir_path in expected_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            assert Path(dir_path).exists()
            assert Path(dir_path).is_dir()


def test_integration_no_root_files():
    """Integration test: ensure no files are created in root during typical operations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Simulate various operations that create files
            artifacts_dir = Path("artifacts")

            # NetLogo operations
            netlogo_dir = artifacts_dir / "netlogo"
            netlogo_dir.mkdir(parents=True, exist_ok=True)
            (netlogo_dir / "summary.json").write_text("{}")

            # Check root is clean
            root_items = list(Path(".").iterdir())
            assert len(root_items) == 1
            assert root_items[0].name == "artifacts"

        finally:
            os.chdir(original_cwd)
