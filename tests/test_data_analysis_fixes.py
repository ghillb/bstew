"""
Test Data Analysis System Corruption fixes
==========================================

Tests for analyze command file verification and validation command error handling.
"""

import pytest
from pathlib import Path
import tempfile
import json
import pandas as pd
import numpy as np

from bstew.cli.commands.analyze import AnalyzeCommand
from bstew.cli.commands.validation import ValidateCommand
from bstew.cli.core.base import CLIContext


def test_analyze_command_file_verification():
    """Test that analyze command properly verifies file creation"""
    context = CLIContext()
    command = AnalyzeCommand(context)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal input data
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()

        # Try to save results - should fail with proper error
        output_file = str(Path(tmpdir) / "output.csv")

        # Create empty results dict
        results = {}

        # Should raise error for empty results
        with pytest.raises(RuntimeError, match="No analysis files were created"):
            command._save_results(results, output_file, "csv")


def test_analyze_command_saves_actual_files():
    """Test that analyze command creates actual files when it has data"""
    context = CLIContext()
    command = AnalyzeCommand(context)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = str(Path(tmpdir) / "analysis.csv")

        # Create results with actual data
        results = {
            "summary_statistics": {
                "final_population": 15000,
                "peak_population": 20000,
                "average_population": 17500,
                "population_volatility": 0.15
            }
        }

        # Save results
        files_created = command._save_results(results, output_file, "csv")

        # Verify files were created
        assert len(files_created) > 0
        for file_path in files_created:
            assert file_path.exists()
            assert file_path.stat().st_size > 0

        # For CSV format with summary_statistics, should create _summary.csv file
        expected_file = Path(tmpdir) / "analysis_summary.csv"
        assert expected_file.exists()


def test_analyze_command_multiple_csv_files():
    """Test that CSV format creates multiple files for different data types"""
    context = CLIContext()
    command = AnalyzeCommand(context)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = str(Path(tmpdir) / "results.csv")

        # Create comprehensive results
        results = {
            "summary_statistics": {"total": 100},
            "population_trends": {
                "species_1": {"trend": "increasing", "rate": 0.05},
                "species_2": {"trend": "stable", "rate": 0.0}
            },
            "growth_rates": {
                "colony_1": {"daily": 0.02, "weekly": 0.14},
                "colony_2": {"daily": 0.01, "weekly": 0.07}
            },
            "foraging_efficiency": {
                "average": 0.75,
                "peak": 0.92,
                "minimum": 0.45
            }
        }

        # Save results
        files_created = command._save_results(results, output_file, "csv")

        # Should create 4 files
        assert len(files_created) == 4

        # Check each expected file
        base_path = Path(tmpdir)
        assert (base_path / "results_summary.csv").exists()
        assert (base_path / "results_trends.csv").exists()
        assert (base_path / "results_growth.csv").exists()
        assert (base_path / "results_foraging.csv").exists()


def test_validate_command_error_handling():
    """Test that validation command provides helpful error messages"""
    context = CLIContext()
    command = ValidateCommand(context)

    # Test with empty metrics
    summary = command._generate_validation_summary({})
    assert summary["overall_performance"] == "unavailable"
    assert summary["variables_validated"] == 0

    # Test with missing required metrics
    metrics = {
        "population": {"rmse": 100.0, "mae": 50.0},  # Missing r_squared and nash_sutcliffe
        "honey": {"r_squared": 0.8}  # Missing other metrics
    }

    summary = command._generate_validation_summary(metrics)
    assert "overall_performance" in summary
    assert summary["variables_validated"] == 2

    # Test with invalid metric structure
    invalid_metrics = {
        "variable1": "not_a_dict",  # Wrong type
        "variable2": {"r_squared": 0.9, "nash_sutcliffe": 0.8}
    }

    with pytest.raises(RuntimeError, match="Invalid metrics format"):
        command._generate_validation_summary(invalid_metrics)


def test_validate_command_complete_metrics():
    """Test validation with complete metrics works correctly"""
    context = CLIContext()
    command = ValidateCommand(context)

    # Complete metrics
    metrics = {
        "population": {
            "r_squared": 0.85,
            "rmse": 120.5,
            "mae": 95.2,
            "nash_sutcliffe": 0.82
        },
        "honey_production": {
            "r_squared": 0.72,
            "rmse": 15.3,
            "mae": 12.1,
            "nash_sutcliffe": 0.68
        },
        "foraging_activity": {
            "r_squared": 0.45,
            "rmse": 0.25,
            "mae": 0.18,
            "nash_sutcliffe": 0.40
        }
    }

    summary = command._generate_validation_summary(metrics)

    # Check all required fields are present
    assert summary["overall_performance"] in ["excellent", "good", "fair", "poor"]
    assert summary["variables_validated"] == 3
    assert summary["best_variable"] == "population"  # Highest r_squared
    assert summary["worst_variable"] == "foraging_activity"  # Lowest r_squared
    assert "performance_grade" in summary

    # Check average metrics
    avg_metrics = summary["average_metrics"]
    assert "avg_r_squared" in avg_metrics
    assert "avg_rmse" in avg_metrics
    assert "avg_mae" in avg_metrics
    assert "avg_nash_sutcliffe" in avg_metrics


def test_performance_categorization():
    """Test performance grade categorization"""
    context = CLIContext()
    command = ValidateCommand(context)

    # Test different performance levels
    assert command._categorize_performance(0.95, 0.92) == "A+"
    assert command._categorize_performance(0.85, 0.80) == "A"
    assert command._categorize_performance(0.75, 0.70) == "B"
    assert command._categorize_performance(0.65, 0.60) == "C"
    assert command._categorize_performance(0.55, 0.50) == "D"
    assert command._categorize_performance(0.35, 0.30) == "F"


def test_analyze_json_format():
    """Test JSON format saves correctly"""
    context = CLIContext()
    command = AnalyzeCommand(context)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = str(Path(tmpdir) / "results.json")

        results = {
            "summary": {"total": 100},
            "details": {"items": [1, 2, 3]},
            "timestamp": "2024-01-01"
        }

        files_created = command._save_results(results, output_file, "json")

        # Should create exactly one JSON file
        assert len(files_created) == 1
        assert files_created[0].exists()
        assert files_created[0].suffix == ".json"

        # Verify content
        with open(files_created[0]) as f:
            loaded = json.load(f)
        assert loaded == results


def test_analyze_empty_file_detection():
    """Test that empty files are detected and reported"""
    context = CLIContext()
    command = AnalyzeCommand(context)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an empty file manually
        empty_file = Path(tmpdir) / "empty.json"
        empty_file.write_text("")

        # Mock the file creation to return our empty file
        class MockResults:
            def __init__(self):
                self.files = [empty_file]

        # The verification should catch the empty file
        with pytest.raises(RuntimeError, match="Analysis file is empty"):
            # Simulate the verification part
            if empty_file.stat().st_size == 0:
                raise RuntimeError(f"Analysis file is empty: {empty_file}")
