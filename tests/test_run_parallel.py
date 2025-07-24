"""
Tests for run command parallel execution
========================================

Tests the --parallel and --max-workers options for the run command.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from src.bstew.cli.commands.run import RunCommand
from src.bstew.cli.core.base import CLIContext


class TestRunParallel:
    """Test parallel execution functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.context = CLIContext()
        self.command = RunCommand(self.context)
        self.test_dir = Path(tempfile.mkdtemp())

        # Create test config
        self.config_path = self.test_dir / "test_config.yaml"
        self.config_path.write_text("""
simulation:
  duration_days: 2
  timestep: 1.0
  random_seed: 42
  output_frequency: 1

colony:
  species: bombus_terrestris
  initial_population:
    workers: 10
    foragers: 5
    drones: 1
    queens: 1

environment:
  landscape_width: 50
  landscape_height: 50
  cell_size: 10.0
""")

    def teardown_method(self):
        """Clean up test files"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_parallel_validation_success(self):
        """Test validation passes for valid parallel parameters"""
        errors = self.command.validate_inputs(
            config=str(self.config_path),
            parallel=True,
            max_workers=2,
            replicates=3,
            output="test_output"
        )
        assert len(errors) == 0

    def test_parallel_validation_requires_multiple_replicates(self):
        """Test validation fails when parallel=True but replicates=1"""
        errors = self.command.validate_inputs(
            config=str(self.config_path),
            parallel=True,
            max_workers=2,
            replicates=1,
            output="test_output"
        )
        assert len(errors) == 1
        assert "parallel execution requires replicates > 1" in errors[0]

    def test_parallel_validation_max_workers_positive(self):
        """Test validation fails for non-positive max_workers"""
        errors = self.command.validate_inputs(
            config=str(self.config_path),
            parallel=True,
            max_workers=0,
            replicates=3,
            output="test_output"
        )
        assert len(errors) == 1
        assert "max_workers must be a positive integer" in errors[0]

    def test_parallel_validation_replicates_positive(self):
        """Test validation fails for non-positive replicates"""
        errors = self.command.validate_inputs(
            config=str(self.config_path),
            parallel=False,
            replicates=0,
            output="test_output"
        )
        assert len(errors) == 1
        assert "replicates must be a positive integer" in errors[0]

    def test_parallel_execution_mock(self):
        """Test parallel execution with minimal validation"""
        output_path = self.test_dir / "parallel_output"

        # We'll allow the actual execution but test the basic functionality
        # Since parallel execution uses ProcessPoolExecutor, it's harder to mock
        result = self.command.execute(
            config=str(self.config_path),
            output=str(output_path),
            parallel=True,
            max_workers=2,
            replicates=2,  # Reduced for test speed
            quiet=True  # Avoid progress bar issues in tests
        )

        assert result.success is True
        assert "Parallel simulation completed" in result.message
        assert result.data["replicates"] == 2
        assert "results" in result.data

        # Check that execution mode is parallel
        assert result.data["results"]["execution_mode"] == "parallel"

    def test_sequential_execution_multiple_replicates(self):
        """Test sequential execution with multiple replicates"""
        output_path = self.test_dir / "sequential_output"

        # Mock the _run_simulation method to avoid actual simulation
        with patch.object(self.command, '_run_simulation') as mock_run:
            mock_run.return_value = {
                "final_population": 15,
                "max_population": 20,
                "total_honey_produced": 10.5,
                "foraging_efficiency": 0.8,
                "colony_survival": True,
                "simulation_days": 2,
            }

            result = self.command.execute(
                config=str(self.config_path),
                output=str(output_path),
                parallel=False,
                replicates=3,
                quiet=True
            )

            assert result.success is True
            assert "Sequential simulation completed" in result.message
            assert result.data["replicates"] == 3
            assert "results" in result.data

            # Check that _run_simulation was called 3 times
            assert mock_run.call_count == 3

    def test_single_simulation_default(self):
        """Test single simulation (default behavior)"""
        output_path = self.test_dir / "single_output"

        # Mock the _run_simulation_with_progress method
        with patch.object(self.command, '_run_simulation_with_progress') as mock_run:
            mock_run.return_value = {
                "final_population": 15,
                "max_population": 20,
                "total_honey_produced": 10.5,
                "foraging_efficiency": 0.8,
                "colony_survival": True,
                "simulation_days": 2,
            }

            result = self.command.execute(
                config=str(self.config_path),
                output=str(output_path),
                replicates=1,  # Default behavior
                quiet=True
            )

            assert result.success is True
            assert "Simulation completed successfully" in result.message
            assert "replicates" not in result.data  # Single simulation doesn't mention replicates
            assert "results" in result.data

    def test_aggregate_replicate_results(self):
        """Test aggregation of results from multiple replicates"""
        # Test data with different values
        results = [
            {
                "final_population": 10,
                "max_population": 15,
                "total_honey_produced": 5.0,
                "foraging_efficiency": 0.7,
                "colony_survival": True,
                "replicate_id": 1
            },
            {
                "final_population": 20,
                "max_population": 25,
                "total_honey_produced": 8.0,
                "foraging_efficiency": 0.9,
                "colony_survival": True,
                "replicate_id": 2
            },
            {
                "final_population": 15,
                "max_population": 20,
                "total_honey_produced": 6.5,
                "foraging_efficiency": 0.8,
                "colony_survival": False,
                "replicate_id": 3
            }
        ]

        aggregated = self.command._aggregate_replicate_results(results)

        # Check mean calculations
        assert aggregated["final_population_mean"] == 15.0  # (10+20+15)/3
        assert aggregated["max_population_mean"] == 20.0  # (15+25+20)/3
        assert aggregated["total_honey_produced_mean"] == 6.5  # (5.0+8.0+6.5)/3
        assert abs(aggregated["foraging_efficiency_mean"] - 0.8) < 0.001  # (0.7+0.9+0.8)/3, accounting for float precision

        # Check survival rate
        assert aggregated["colony_survival_rate"] == 2/3  # 2 out of 3 survived

        # Check that individual results are preserved
        assert "individual_results" in aggregated
        assert len(aggregated["individual_results"]) == 3

    def test_max_workers_auto_detection(self):
        """Test that max_workers is auto-detected when not specified"""
        output_path = self.test_dir / "auto_workers_output"

        with patch('src.bstew.cli.commands.run._run_single_replicate') as mock_replicate:
            mock_replicate.return_value = {
                "final_population": 15,
                "max_population": 20,
                "total_honey_produced": 10.5,
                "foraging_efficiency": 0.8,
                "colony_survival": True,
                "simulation_days": 2,
                "output_directory": "test_output"
            }

            # Test with max_workers=None (should auto-detect)
            result = self.command.execute(
                config=str(self.config_path),
                output=str(output_path),
                parallel=True,
                max_workers=None,  # Should auto-detect
                replicates=4,
                quiet=True
            )

            assert result.success is True
            # Should have used some number of workers (likely CPU count or replicate count)
            assert "max_workers" in result.data["results"]

    def test_unique_output_directories_per_replicate(self):
        """Test that each replicate gets a unique output directory"""
        output_path = self.test_dir / "unique_output"

        with patch.object(self.command, '_run_simulation') as mock_run:
            # Track the configurations passed to _run_simulation
            configs_called = []

            def capture_config(config, *args, **kwargs):
                configs_called.append(config)
                return {
                    "final_population": 15,
                    "max_population": 20,
                    "total_honey_produced": 10.5,
                    "foraging_efficiency": 0.8,
                    "colony_survival": True,
                    "simulation_days": 2,
                }

            mock_run.side_effect = capture_config

            result = self.command.execute(
                config=str(self.config_path),
                output=str(output_path),
                parallel=False,
                replicates=3,
                quiet=True
            )

            assert result.success is True
            assert len(configs_called) == 3

            # Check that each config has a unique output directory
            output_dirs = [
                config.get("output", {}).get("output_directory", "")
                for config in configs_called
            ]

            assert len(set(output_dirs)) == 3  # All unique
            assert all("replicate_" in output_dir for output_dir in output_dirs)

    def test_unique_seeds_per_replicate(self):
        """Test that each replicate gets a unique random seed"""
        output_path = self.test_dir / "unique_seeds"

        with patch.object(self.command, '_run_simulation') as mock_run:
            # Track the configurations passed to _run_simulation
            configs_called = []

            def capture_config(config, *args, **kwargs):
                configs_called.append(config)
                return {
                    "final_population": 15,
                    "max_population": 20,
                    "total_honey_produced": 10.5,
                    "foraging_efficiency": 0.8,
                    "colony_survival": True,
                    "simulation_days": 2,
                }

            mock_run.side_effect = capture_config

            result = self.command.execute(
                config=str(self.config_path),
                output=str(output_path),
                parallel=False,
                replicates=3,
                quiet=True
            )

            assert result.success is True
            assert len(configs_called) == 3

            # Check that each config has a unique random seed
            seeds = [
                config.get("simulation", {}).get("random_seed")
                for config in configs_called
            ]

            assert len(set(seeds)) == 3  # All unique
            # Seeds should start from the original seed (42) plus replicate index
            assert seeds[0] == 42  # First replicate should be base seed
            assert all(isinstance(seed, int) for seed in seeds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
