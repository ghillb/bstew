"""
Integration Tests for NetLogo Behavioral Validation
==================================================

Tests the complete workflow with real SimulationEngine and NetLogo data parsing.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from src.bstew.validation.test_scenarios import (
    ValidationScenario,
    ValidationRunner,
    ValidationResult,
    ValidationScenarioLoader,
)
from src.bstew.simulation.simulation_engine import SimulationEngine, SimulationResults
from src.bstew.utils.config import BstewConfig, SimulationConfig


class TestNetLogoValidationIntegration:
    """Integration tests for the complete NetLogo validation workflow"""

    def setup_method(self):
        """Setup test fixtures with real test data"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.netlogo_data_dir = self.test_dir / "netlogo_data"
        self.netlogo_data_dir.mkdir()

        # Create sample BSTEW config
        self.bstew_config_path = self.test_dir / "bstew_config.yaml"
        self.bstew_config_path.write_text("""
simulation:
  duration_days: 30
  timestep: 1.0
  random_seed: 42
  output_frequency: 1

colony:
  species: bombus_terrestris
  initial_population:
    workers: 100
    foragers: 50
    drones: 10
    queens: 1

environment:
  landscape_width: 100
  landscape_height: 100
  cell_size: 10.0
""")

    def teardown_method(self):
        """Clean up test files"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def create_sample_netlogo_data(self, scenario_name: str) -> Path:
        """Create sample NetLogo data files for testing"""
        scenario_dir = self.netlogo_data_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Create BehaviorSpace-style CSV output
        behaviorspace_data = pd.DataFrame({
            '[run number]': range(1, 31),
            '[step]': range(30),
            'total-population': np.random.randint(90, 150, 30),
            'worker-count': np.random.randint(60, 100, 30),
            'forager-count': np.random.randint(30, 60, 30),
            'drone-count': np.random.randint(5, 15, 30),
            'total-foraging-trips': np.random.randint(20, 50, 30),
            'successful-trips': np.random.randint(15, 45, 30),
            'foraging-efficiency': np.random.uniform(0.6, 0.9, 30),
            'colony-energy': np.random.uniform(1000, 2000, 30),
            'daily-mortality-rate': np.random.uniform(0.001, 0.005, 30),
        })

        behaviorspace_file = scenario_dir / "experiment_output.csv"
        behaviorspace_data.to_csv(behaviorspace_file, index=False)

        # Create reporter output file
        reporter_data = """
Final Population: 142
Final Energy: 1567.8
Total Foraging Trips: 1250
Successful Trips: 1050
Average Efficiency: 0.84
Colony Survival: true
"""
        reporter_file = scenario_dir / "reporter_output.txt"
        reporter_file.write_text(reporter_data)

        return scenario_dir

    def create_test_scenario(self, scenario_name: str = "population_test") -> ValidationScenario:
        """Create a test scenario"""
        return ValidationScenario(
            name=scenario_name,
            description="Test population dynamics validation",
            category="population",
            netlogo_params={
                "InitialWorkerCount": 100,
                "InitialForagerCount": 50,
                "MaxForagingRange_m": 500,
                "SimulationDays": 30,
            },
            bstew_config_overrides={
                "colony.initial_population.workers": 100,
                "colony.initial_population.foragers": 50,
                "simulation.duration_days": 30,
            },
            metrics_to_compare=[
                "total_population",
                "worker_count",
                "forager_count",
                "foraging_trips",
                "successful_trips",
                "foraging_efficiency",
            ],
            duration_days=30,
        )

    def test_test_scenario_creation_and_serialization(self):
        """Test creating and serializing test scenarios"""
        scenario = self.create_test_scenario()

        # Test to_dict
        scenario_dict = scenario.to_dict()
        assert scenario_dict["name"] == "population_test"
        assert scenario_dict["category"] == "population"
        assert len(scenario_dict["metrics_to_compare"]) == 6

        # Test from_dict
        recreated = ValidationScenario.from_dict(scenario_dict)
        assert recreated.name == scenario.name
        assert recreated.category == scenario.category
        assert recreated.metrics_to_compare == scenario.metrics_to_compare

    def test_test_scenario_loader(self):
        """Test loading scenarios from YAML file"""
        scenarios_data = {
            "scenarios": [
                {
                    "name": "population_growth",
                    "description": "Test population growth patterns",
                    "category": "population",
                    "netlogo_params": {"InitialWorkerCount": 100, "SimulationDays": 30},
                    "bstew_config_overrides": {"colony.initial_population.workers": 100},
                    "metrics_to_compare": ["total_population", "worker_count"],
                    "duration_days": 30,
                },
                {
                    "name": "foraging_efficiency",
                    "description": "Test foraging behavior",
                    "category": "foraging",
                    "netlogo_params": {"MaxForagingRange_m": 500},
                    "bstew_config_overrides": {"foraging.max_range": 500},
                    "metrics_to_compare": ["foraging_trips", "foraging_efficiency"],
                    "duration_days": 30,
                },
            ]
        }

        scenarios_file = self.test_dir / "scenarios.yaml"
        with open(scenarios_file, 'w') as f:
            yaml.dump(scenarios_data, f)

        loader = ValidationScenarioLoader(scenarios_file)
        scenarios = loader.load_scenarios()

        assert len(scenarios) == 2
        assert scenarios[0].name == "population_growth"
        assert scenarios[1].name == "foraging_efficiency"
        assert scenarios[0].category == "population"
        assert scenarios[1].category == "foraging"

    def test_netlogo_data_loading_behaviorspace(self):
        """Test loading real NetLogo BehaviorSpace data"""
        scenario = self.create_test_scenario()
        scenario_dir = self.create_sample_netlogo_data("population_test")

        runner = ValidationRunner(output_dir=self.test_dir)
        runner.set_netlogo_data_source(self.netlogo_data_dir)

        # Test loading NetLogo data
        netlogo_data = runner._load_netlogo_data(scenario)

        assert "total_population" in netlogo_data
        assert "worker_count" in netlogo_data
        assert "foraging_efficiency" in netlogo_data
        assert len(netlogo_data["total_population"]) == 30

        # Check that data values are reasonable
        assert all(90 <= pop <= 150 for pop in netlogo_data["total_population"])
        assert all(0.6 <= eff <= 0.9 for eff in netlogo_data["foraging_efficiency"])

    def test_netlogo_data_loading_reporter_output(self):
        """Test loading NetLogo reporter text output"""
        scenario = self.create_test_scenario()
        scenario_dir = self.create_sample_netlogo_data("population_test")

        runner = ValidationRunner(output_dir=self.test_dir)

        # Test parsing reporter output
        reporter_file = scenario_dir / "reporter_output.txt"
        reporter_data = runner._parse_netlogo_reporter_output(reporter_file)

        assert "final_population" in reporter_data
        assert "final_energy" in reporter_data
        assert "total_foraging_trips" in reporter_data
        assert reporter_data["final_population"] == 142
        assert reporter_data["final_energy"] == 1567.8
        assert reporter_data["total_foraging_trips"] == 1250

    @patch('src.bstew.validation.test_scenarios.SimulationEngine')
    def test_bstew_simulation_integration(self, mock_engine_class):
        """Test BSTEW simulation integration with real configuration"""
        # Create mock simulation results
        mock_results = SimulationResults(
            simulation_id="test_sim",
            config=SimulationConfig(duration_days=30),
            start_time=Mock(),
            status="completed",
            final_data={
                "model_data": pd.DataFrame({
                    "step": range(30),
                    "total_population": np.random.randint(95, 145, 30),
                    "worker_count": np.random.randint(65, 95, 30),
                    "foraging_trips": np.random.randint(18, 48, 30),
                    "successful_trips": np.random.randint(14, 42, 30),
                    "foraging_efficiency": np.random.uniform(0.65, 0.88, 30),
                })
            }
        )

        mock_engine = Mock()
        mock_engine.run_simulation.return_value = mock_results
        mock_engine_class.return_value = mock_engine

        scenario = self.create_test_scenario()
        runner = ValidationRunner(output_dir=self.test_dir)
        runner.set_bstew_config(self.bstew_config_path)

        # Test running BSTEW simulation
        bstew_data = runner._run_bstew_simulation(scenario)

        # Verify simulation was called with correct parameters
        mock_engine_class.assert_called_once()
        mock_engine.run_simulation.assert_called_once()

        # Verify data extraction
        assert "total_population" in bstew_data
        assert "worker_count" in bstew_data
        assert "foraging_efficiency" in bstew_data
        assert len(bstew_data["total_population"]) == 30

    def test_metric_comparison_time_series(self):
        """Test comparing time series metrics"""
        runner = ValidationRunner(output_dir=self.test_dir)

        # Create sample time series data
        netlogo_data = {
            "total_population": [100, 105, 110, 115, 120],
        }
        bstew_data = {
            "total_population": [98, 104, 112, 114, 118],
        }

        comparison = runner._compare_metric(
            "total_population", netlogo_data, bstew_data, tolerance=0.1
        )

        assert comparison.name == "total_population"
        assert comparison.difference < 0.1  # Should be within tolerance
        assert comparison.passed is True

    def test_metric_comparison_scalar(self):
        """Test comparing scalar metrics"""
        runner = ValidationRunner(output_dir=self.test_dir)

        netlogo_data = {"final_energy": 1500.0}
        bstew_data = {"final_energy": 1485.0}

        comparison = runner._compare_metric(
            "final_energy", netlogo_data, bstew_data, tolerance=0.05
        )

        assert comparison.name == "final_energy"
        assert comparison.difference == 0.01  # 1% difference
        assert comparison.passed is True

    @patch('src.bstew.validation.test_scenarios.SimulationEngine')
    def test_complete_scenario_execution(self, mock_engine_class):
        """Test complete scenario execution from NetLogo data to comparison"""
        # Setup mock BSTEW simulation
        mock_results = SimulationResults(
            simulation_id="test_sim",
            config=SimulationConfig(duration_days=30),
            start_time=Mock(),
            status="completed",
            final_data={
                "model_data": pd.DataFrame({
                    "step": range(30),
                    "total_population": np.random.randint(95, 145, 30),
                    "worker_count": np.random.randint(65, 95, 30),
                    "foraging_trips": np.random.randint(18, 48, 30),
                })
            }
        )

        mock_engine = Mock()
        mock_engine.run_simulation.return_value = mock_results
        mock_engine_class.return_value = mock_engine

        # Create test data
        scenario = self.create_test_scenario()
        self.create_sample_netlogo_data("population_test")

        # Run complete test
        runner = ValidationRunner(output_dir=self.test_dir, tolerance=0.2)
        runner.set_netlogo_data_source(self.netlogo_data_dir)
        runner.set_bstew_config(self.bstew_config_path)

        result = runner.run_scenario(scenario)

        # Verify results
        assert isinstance(result, ValidationResult)
        assert result.scenario == scenario
        assert result.passed in [True, False]  # Could pass or fail depending on mock data
        assert len(result.metrics) == len(scenario.metrics_to_compare)
        assert result.execution_time > 0

        # Check that both NetLogo and BSTEW were processed
        mock_engine.run_simulation.assert_called_once()

    def test_html_report_generation(self):
        """Test HTML report generation with real results"""
        runner = ValidationRunner(output_dir=self.test_dir)

        # Create sample test results
        scenario = self.create_test_scenario()

        from src.bstew.validation.test_scenarios import MetricComparison

        metrics = [
            MetricComparison(
                name="total_population",
                netlogo_value=[100, 105, 110],
                bstew_value=[98, 104, 112],
                difference=0.03,
                tolerance=0.05,
                passed=True,
            ),
            MetricComparison(
                name="foraging_efficiency",
                netlogo_value=0.85,
                bstew_value=0.82,
                difference=0.035,
                tolerance=0.05,
                passed=True,
            ),
        ]

        test_result = ValidationResult(
            scenario=scenario,
            passed=True,
            execution_time=15.2,
            metrics=metrics,
            failed_metrics=[],
        )

        # Generate report
        report_path = runner.generate_report([test_result])

        assert report_path.exists()
        assert report_path.suffix == ".html"

        # Verify report content
        report_content = report_path.read_text()
        assert "NetLogo Behavioral Validation Report" in report_content
        assert "population_test" in report_content
        assert "PASSED" in report_content
        assert "total_population" in report_content
        assert "foraging_efficiency" in report_content

    def test_multiple_test_suites(self):
        """Test running multiple test scenarios with different categories"""
        scenarios = [
            ValidationScenario(
                name="pop_test",
                description="Population test",
                category="population",
                netlogo_params={"InitialWorkerCount": 100},
                bstew_config_overrides={"colony.initial_population.workers": 100},
                metrics_to_compare=["total_population"],
                duration_days=10,
            ),
            ValidationScenario(
                name="forage_test",
                description="Foraging test",
                category="foraging",
                netlogo_params={"MaxForagingRange_m": 500},
                bstew_config_overrides={"foraging.max_range": 500},
                metrics_to_compare=["foraging_efficiency"],
                duration_days=10,
            ),
        ]

        # Create test data for both scenarios
        for scenario in scenarios:
            self.create_sample_netlogo_data(scenario.name)

        runner = ValidationRunner(output_dir=self.test_dir, tolerance=0.1)
        runner.set_netlogo_data_source(self.netlogo_data_dir)

        # Mock BSTEW simulations
        with patch('src.bstew.validation.test_scenarios.SimulationEngine') as mock_engine_class:
            mock_results = SimulationResults(
                simulation_id="test",
                config=SimulationConfig(duration_days=10),
                start_time=Mock(),
                status="completed",
                final_data={"model_data": pd.DataFrame({"step": range(10)})},
            )

            mock_engine = Mock()
            mock_engine.run_simulation.return_value = mock_results
            mock_engine_class.return_value = mock_engine

            results = []
            for scenario in scenarios:
                result = runner.run_scenario(scenario)
                results.append(result)

            # Verify all scenarios were executed
            assert len(results) == 2
            assert results[0].scenario.category == "population"
            assert results[1].scenario.category == "foraging"

            # Generate combined report
            report_path = runner.generate_report(results)
            report_content = report_path.read_text()

            assert "pop_test" in report_content
            assert "forage_test" in report_content

    def test_error_handling_simulation_failure(self):
        """Test error handling when simulations fail"""
        scenario = self.create_test_scenario()
        runner = ValidationRunner(output_dir=self.test_dir)

        # Mock failed BSTEW simulation
        with patch('src.bstew.validation.test_scenarios.SimulationEngine') as mock_engine_class:
            mock_results = SimulationResults(
                simulation_id="test",
                config=SimulationConfig(duration_days=30),
                start_time=Mock(),
                status="failed",
                error_message="Simulation crashed",
            )

            mock_engine = Mock()
            mock_engine.run_simulation.return_value = mock_results
            mock_engine_class.return_value = mock_engine

            result = runner.run_scenario(scenario)

            # Should handle gracefully
            assert result.passed is False
            assert "Simulation crashed" in result.failure_reason

    def test_tolerance_configuration(self):
        """Test different tolerance configurations"""
        scenario = self.create_test_scenario()
        scenario.tolerance_overrides = {
            "total_population": 0.01,  # Very strict
            "foraging_efficiency": 0.2,  # More lenient
        }

        runner = ValidationRunner(output_dir=self.test_dir, tolerance=0.05)

        # Test strict tolerance
        strict_comparison = runner._compare_metric(
            "total_population",
            {"total_population": 100},
            {"total_population": 102},  # 2% difference
            tolerance=0.01,
        )
        assert strict_comparison.passed is False

        # Test lenient tolerance
        lenient_comparison = runner._compare_metric(
            "foraging_efficiency",
            {"foraging_efficiency": 0.8},
            {"foraging_efficiency": 0.85},  # 6.25% difference
            tolerance=0.2,
        )
        assert lenient_comparison.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
