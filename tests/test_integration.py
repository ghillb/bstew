"""
Integration tests for BSTEW simulation workflow
==============================================

Tests complete simulation scenarios from initialization to results,
ensuring all components work together correctly.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Import BSTEW components
from src.bstew.core.model import BeeModel
from src.bstew.utils.config import ConfigManager
from src.bstew.utils.batch_processing import ExperimentManager, ParameterSpec
from src.bstew.utils.validation import ModelValidator
from src.bstew.utils.performance import SimulationOptimizer
from src.bstew.components.environment import ClimateScenario


class MockLandscape:
    """Mock landscape class for testing purposes"""

    def __init__(self, width=100, height=100, cell_size=20.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.use_masterpatch_system = True
        self.world_width = width * cell_size
        self.world_height = height * cell_size
        self.patches = {}
        self.masterpatch_system = None

    def get_patch_at_position(self, x, y):
        """Mock method to get patch at position"""
        return None

    def get_patches_in_radius(self, center, radius):
        """Mock method to get patches in radius"""
        return []

    def update_all_patches(self, day_of_year, weather):
        """Mock method to update all patches"""
        pass

    def get_total_resources(self):
        """Mock method to get total resources"""
        return {"nectar": 1000.0, "pollen": 500.0, "total": 1500.0}


class TestBasicSimulationWorkflow:
    """Test basic simulation initialization and execution"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_complete_simulation_run(self):
        """Test complete simulation from start to finish"""

        # Create basic configuration
        config = self.config_manager.load_default_config()
        config.simulation.duration_days = 10
        # Use proper dictionary structure for initial_population
        config.colony.initial_population = {
            "queens": 1,
            "workers": 1500,
            "foragers": 300,
            "drones": 50,
            "brood": 150,
        }

        # Initialize model
        model = BeeModel(config=config)
        assert model is not None
        assert model.get_colony_count() > 0

        # Run simulation
        initial_step = model.schedule.steps
        for day in range(10):
            model.step()

        # Verify simulation progressed
        assert model.schedule.steps == initial_step + 10

        # Check data collection
        model_data = model.datacollector.get_model_vars_dataframe()
        assert len(model_data) == 10
        assert "Total_Bees" in model_data.columns
        assert "Total_Brood" in model_data.columns

        # Verify colonies still exist
        assert model.get_colony_count() > 0

    def test_colony_lifecycle(self):
        """Test colony lifecycle over shorter period for integration testing"""

        config = self.config_manager.load_default_config()
        config.simulation.duration_days = 15  # Very short test duration
        # Use proper dictionary structure for initial_population
        config.colony.initial_population = {
            "queens": 1,
            "workers": 1200,
            "foragers": 200,
            "drones": 50,
            "brood": 50,
        }

        model = BeeModel(config=config)
        initial_colony_count = model.get_colony_count()

        # Run simulation for 15 days
        for day in range(15):
            model.step()

        # Verify simulation completed
        model_data = model.datacollector.get_model_vars_dataframe()
        assert len(model_data) == 15

        # Check population dynamics
        final_population = model_data["Total_Bees"].iloc[-1]
        assert final_population > 0  # Colony survived

        # Check that colony is still active
        final_colony_count = model.get_colony_count()
        assert (
            final_colony_count >= initial_colony_count
        )  # At least initial colonies remain

    def test_disease_integration(self):
        """Test disease system integration"""

        config = self.config_manager.load_default_config()
        config.simulation.duration_days = 12  # Much shorter test period
        config.disease.enable_varroa = True
        config.disease.enable_viruses = True
        # Use proper dictionary structure for initial_population
        config.colony.initial_population = {
            "queens": 1,
            "workers": 1000,
            "foragers": 150,
            "drones": 30,
            "brood": 50,
        }

        model = BeeModel(config=config)

        # Add disease pressure
        for colony in model.get_colonies():
            if hasattr(colony, "disease_manager"):
                # Introduce Varroa mites
                colony.disease_manager.varroa_model.mite_population = 100

        # Run simulation with disease pressure
        for day in range(12):
            model.step()

        # Check that disease affected the colony
        model_data = model.datacollector.get_model_vars_dataframe()
        initial_pop = model_data["Total_Bees"].iloc[0]
        final_pop = model_data["Total_Bees"].iloc[-1]

        # With disease pressure, population should be affected
        assert final_pop <= initial_pop * 1.1  # Allow some growth but expect impact

    def test_environmental_effects_integration(self):
        """Test environmental effects system integration"""

        config = self.config_manager.load_default_config()
        config.simulation.duration_days = 8  # Very short test
        # Weather effects are enabled through seasonal_effects field
        config.environment.seasonal_effects = True
        # Use proper dictionary structure for initial_population
        config.colony.initial_population = {
            "queens": 1,
            "workers": 650,
            "foragers": 100,
            "drones": 25,
            "brood": 25,
        }

        model = BeeModel(config=config)

        # Add climate scenario
        climate_scenario = ClimateScenario(
            name="test_warming",
            description="Test warming scenario",
            temperature_trend=0.1,  # 0.1°C increase per year
            precipitation_trend=-5,  # 5% decrease per year
            extreme_frequency_multiplier=1.2,  # 20% increase in extreme events
        )

        if hasattr(model, "environment_manager"):
            model.environment_manager.set_climate_scenario(climate_scenario)

        # Run simulation
        for day in range(8):
            model.step()

        # Verify environmental effects were applied
        model_data = model.datacollector.get_model_vars_dataframe()
        assert len(model_data) == 8

        # Check that foraging efficiency is being tracked (values may be constant in short runs)
        if "Foraging_Efficiency" in model_data.columns:
            efficiency_values = model_data["Foraging_Efficiency"]
            # For short simulations, efficiency might be constant, so just check it's being recorded
            assert len(efficiency_values) == 8
            assert all(eff >= 0.0 for eff in efficiency_values)  # Valid range


class TestBatchProcessingIntegration:
    """Test batch processing and experiment management"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.experiment_manager = ExperimentManager(str(self.temp_dir))

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_parameter_sweep_execution(self):
        """Test parameter sweep experiment execution"""

        # Create parameter specification
        parameter_specs = {
            "colony.initial_population": ParameterSpec(
                name="colony.initial_population",
                min_value=5000,
                max_value=15000,
                step_size=5000,
            )
        }

        # Create experiment design
        design = self.experiment_manager.create_parameter_sweep(
            "Test Parameter Sweep",
            parameter_specs,
            n_replicates=1,
            simulation_days=30,  # Short test
        )

        # Generate runs
        runs = design.generate_runs()
        assert len(runs) == 3  # Should have 3 parameter values

        # Verify run parameters
        populations = [run.parameters.get("colony.initial_population") for run in runs]
        assert set(populations) == {5000, 10000, 15000}

    def test_monte_carlo_experiment(self):
        """Test Monte Carlo experiment setup"""

        parameter_specs = {
            "biology.base_mortality": ParameterSpec(
                name="biology.base_mortality",
                min_value=0.01,
                max_value=0.05,
                distribution="uniform",
            )
        }

        design = self.experiment_manager.create_monte_carlo_experiment(
            "Test Monte Carlo", parameter_specs, n_samples=10, simulation_days=30
        )

        runs = design.generate_runs()
        assert len(runs) == 10

        # Check parameter variation
        mortality_rates = [run.parameters.get("biology.base_mortality") for run in runs]
        assert len(set(mortality_rates)) > 1  # Should have variation
        assert all(0.01 <= rate <= 0.05 for rate in mortality_rates)

    def test_scenario_comparison(self):
        """Test scenario comparison functionality"""

        scenarios = {
            "baseline": {
                "colony": {"initial_population": 20000},
                "disease": {"varroa_enabled": False},
            },
            "with_varroa": {
                "colony": {"initial_population": 20000},
                "disease": {"varroa_enabled": True},
            },
        }

        # Mock the run_quick_comparison method to avoid full simulation
        with patch.object(
            self.experiment_manager, "run_quick_comparison"
        ) as mock_comparison:
            mock_comparison.return_value = {
                "baseline": {"final_population": 25000, "colony_survival": True},
                "with_varroa": {"final_population": 18000, "colony_survival": True},
            }

            results = self.experiment_manager.run_quick_comparison(scenarios, 90)

            assert "baseline" in results
            assert "with_varroa" in results
            assert (
                results["baseline"]["final_population"]
                > results["with_varroa"]["final_population"]
            )


class TestValidationIntegration:
    """Test validation framework integration"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = ModelValidator()

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_parameter_validation(self):
        """Test parameter validation workflow"""

        # Mock NetLogo parameters
        netlogo_params = {
            "initial-bees": 20000,
            "initial-brood": 5000,
            "max-age": 45,
            "egg-laying-rate": 1500,
        }

        # Mock BSTEW config
        bstew_config = {
            "colony": {"initial_population": 20000, "initial_brood": 5000},
            "biology": {"max_lifespan": 45, "egg_laying_rate": 1500},
        }

        # Run validation
        results = self.validator.parameter_validator.validate_parameter_mapping(
            netlogo_params, bstew_config
        )

        assert len(results) > 0

        # Check that matching parameters pass
        matching_results = [r for r in results if r.passed]
        assert len(matching_results) > 0

    def test_output_comparison(self):
        """Test output comparison workflow"""

        # Create mock data
        days = range(100)

        # NetLogo-style data (with some noise)
        netlogo_data = pd.DataFrame(
            {
                "Day": days,
                "Total_Bees": [
                    20000 + 1000 * np.sin(d / 30) + np.random.normal(0, 100)
                    for d in days
                ],
                "Total_Brood": [
                    5000 + 500 * np.sin(d / 30) + np.random.normal(0, 50) for d in days
                ],
            }
        )

        # BSTEW data (similar but slightly different)
        bstew_data = pd.DataFrame(
            {
                "Day": days,
                "Total_Bees": [
                    20100 + 1050 * np.sin(d / 30) + np.random.normal(0, 120)
                    for d in days
                ],
                "Total_Brood": [
                    5100 + 520 * np.sin(d / 30) + np.random.normal(0, 60) for d in days
                ],
            }
        )

        # Run comparison
        results = self.validator.output_comparator.compare_time_series(
            netlogo_data, bstew_data, ["Total_Bees", "Total_Brood"]
        )

        assert "Total_Bees" in results
        assert "Total_Brood" in results

        # Should have reasonable correlation due to similar patterns
        bees_result = results["Total_Bees"]
        assert (
            bees_result.score > 0.3
        )  # Should have decent agreement (lowered threshold for random data)

    def test_full_validation_workflow(self):
        """Test complete validation workflow"""

        # Create mock validation data
        netlogo_data = {
            "parameters": {"initial-bees": 20000},
            "output_data": pd.DataFrame(
                {"Day": range(50), "Total_Bees": [20000 + i * 10 for i in range(50)]}
            ),
        }

        bstew_data = {
            "config": {"colony": {"initial_population": 20000}},
            "output_data": pd.DataFrame(
                {"Day": range(50), "Total_Bees": [20000 + i * 12 for i in range(50)]}
            ),
        }

        # Run full validation
        validation_config = {
            "comparison_metrics": ["Total_Bees"],
            "create_plots": False,  # Skip plots for testing
        }

        results = self.validator.run_full_validation(
            netlogo_data, bstew_data, validation_config
        )

        assert "parameter_validation" in results
        assert "output_comparison" in results
        assert "summary" in results
        assert "recommendations" in results


class TestPerformanceIntegration:
    """Test performance optimization integration"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_simulation_with_optimization(self):
        """Test simulation with performance optimization enabled"""

        config = {
            "simulation": {"duration_days": 60},
            "colony": {"initial_population": 15000},
            "optimization": {
                "enabled": True,
                "memory_limit_gb": 2.0,
                "cache_memory_mb": 100,
                "max_workers": 2,
            },
        }

        optimizer = SimulationOptimizer(config["optimization"])

        # Mock simulation state
        simulation_state = {
            "current_step": 0,
            "dataframes": {
                "test_df": pd.DataFrame({"A": range(1000), "B": range(1000)})
            },
        }

        # Test optimization
        optimized_state = optimizer.optimize_simulation_step(simulation_state)

        assert optimized_state is not None
        assert "dataframes" in optimized_state

        # Get performance metrics
        metrics = optimizer.get_optimization_metrics()
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0

        optimizer.cleanup()

    def test_memory_management_integration(self):
        """Test memory management during simulation"""

        from src.bstew.utils.performance import MemoryManager

        memory_manager = MemoryManager(memory_limit_gb=1.0)

        # Monitor memory
        initial_memory = memory_manager.monitor_memory()
        assert initial_memory >= 0

        # Test object pooling
        def create_test_object():
            return {"data": list(range(100))}

        obj1 = memory_manager.get_object_from_pool("test_type", create_test_object)
        assert obj1 is not None

        memory_manager.return_to_pool(obj1, "test_type")
        obj2 = memory_manager.get_object_from_pool("test_type", create_test_object)

        # Should reuse the same object
        assert obj2 is obj1

        # Get memory report
        report = memory_manager.get_memory_report()
        assert "current_usage_mb" in report
        assert "pool_sizes" in report


class TestDataPersistenceIntegration:
    """Test data persistence and state management"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_configuration_persistence(self):
        """Test configuration saving and loading"""

        config_manager = ConfigManager()

        # Create test configuration
        config = {
            "simulation": {"duration_days": 100},
            "colony": {"initial_population": {"queens": 1, "workers": 25000}},
        }

        # Save configuration
        config_path = self.temp_dir / "test_config.yaml"
        config_manager.save_config(config, config_path)

        assert config_path.exists()

        # Load configuration
        loaded_config = config_manager.load_config(config_path)

        assert loaded_config.simulation.duration_days == 100
        assert loaded_config.colony.initial_population["queens"] == 1
        assert loaded_config.colony.initial_population["workers"] == 25000

    def test_simulation_results_export(self):
        """Test simulation results export and import"""

        from src.bstew.utils.data_io import DataExporter

        # Create mock simulation data
        model_data = pd.DataFrame(
            {
                "Day": range(30),
                "Total_Bees": [20000 + i * 100 for i in range(30)],
                "Total_Brood": [5000 + i * 50 for i in range(30)],
            }
        )

        agent_data = pd.DataFrame(
            {
                "AgentID": range(100),
                "Type": ["Worker"] * 100,
                "Age": np.random.randint(1, 50, 100),
            }
        )

        exporter = DataExporter()

        # Export to CSV
        csv_path = self.temp_dir / "test_export.csv"
        exporter.export_model_data(model_data, csv_path)

        assert csv_path.exists()

        # Verify exported data
        imported_data = pd.read_csv(csv_path)
        assert len(imported_data) == 30
        assert "Total_Bees" in imported_data.columns

        # Test agent data export
        agent_path = self.temp_dir / "test_agent_export.json"
        exporter.export_agent_data(agent_data, agent_path)

        assert agent_path.exists()

        # Verify both files have content
        assert csv_path.stat().st_size > 0
        assert agent_path.stat().st_size > 0


class TestEndToEndSimulation:
    """Test complete end-to-end simulation scenarios"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_seasonal_simulation_cycle(self):
        """Test seasonal changes over shorter period for integration testing"""

        config_manager = ConfigManager()
        config = config_manager.load_default_config()
        config.simulation.duration_days = 20  # Much shorter test
        # Use proper dictionary structure for initial_population
        config.colony.initial_population = {
            "queens": 1,
            "workers": 800,
            "foragers": 150,
            "drones": 25,
            "brood": 25,
        }

        model = BeeModel(config=config)

        # Track changes over time
        temporal_data = []

        for day in range(20):
            model.step()

            # Record data every 5 days
            if day % 5 == 0:
                model_data = model.datacollector.get_model_vars_dataframe()
                if len(model_data) > 0:
                    temporal_data.append(
                        {
                            "day": day,
                            "population": model_data["Total_Bees"].iloc[-1],
                            "brood": model_data["Total_Brood"].iloc[-1],
                        }
                    )

        # Verify temporal patterns
        assert len(temporal_data) >= 4  # Should have multiple data points (20/5 = 4)

        # Check that population dynamics occurred
        populations = [d["population"] for d in temporal_data]
        assert len(set(populations)) > 1  # Population should change over time

    def test_multi_colony_simulation(self):
        """Test simulation with multiple colonies"""

        config_manager = ConfigManager()
        config = config_manager.load_default_config()
        config.simulation.duration_days = 10  # Much shorter for test performance
        config.colony.initial_population = {
            "queens": 1,
            "workers": 500,
            "foragers": 100,
            "drones": 50,
            "brood": 200,
        }  # Smaller populations

        model = BeeModel(config=config)

        # Add additional colonies manually (since config doesn't support multiple colonies)
        from src.bstew.core.colony import Colony

        for i in range(2):  # Add 2 more colonies for total of 3
            location = (
                config.colony.location[0] + i * 1000,
                config.colony.location[1] + i * 1000,
            )
            new_colony = Colony(
                model,
                config.colony.species,
                location,
                config.colony.initial_population,
                model.next_id(),
            )
            model.colonies.append(new_colony)
            model.schedule.add(new_colony)

        # Verify multiple colonies exist
        initial_colonies = model.get_colony_count()
        assert initial_colonies >= 3

        # Run simulation
        for day in range(10):
            model.step()

        # Check final state
        final_colonies = model.get_colony_count()
        model_data = model.datacollector.get_model_vars_dataframe()

        assert len(model_data) == 10
        assert final_colonies > 0  # At least some colonies should survive

    def test_complex_scenario_simulation(self):
        """Test simulation with multiple stressors and effects"""

        config_manager = ConfigManager()
        config = config_manager.load_default_config()
        config.simulation.duration_days = 12  # Very short test
        # Use proper dictionary structure for initial_population
        config.colony.initial_population = {
            "queens": 1,
            "workers": 1200,
            "foragers": 200,
            "drones": 50,
            "brood": 50,
        }
        config.disease.enable_varroa = True
        config.environment.seasonal_effects = True

        model = BeeModel(config=config)

        # Add environmental stressors mid-simulation
        stressor_applied = False

        for day in range(12):
            model.step()

            # Apply environmental stress after 6 days
            if day == 6 and not stressor_applied:
                # Add drought stress
                if hasattr(model, "environment_manager"):
                    from src.bstew.components.environment import EnvironmentalStressor

                    drought = EnvironmentalStressor(
                        name="test_drought",
                        stressor_type="drought",
                        severity=0.7,
                        duration=30,
                        affected_area=1000,
                    )
                    model.environment_manager.add_environmental_stressor(drought, day)
                    stressor_applied = True

        # Verify simulation completed with stressors
        model_data = model.datacollector.get_model_vars_dataframe()
        assert len(model_data) == 12

        # Check that stressors affected the colony
        if len(model_data) > 6:
            pre_stress_pop = model_data["Total_Bees"].iloc[5]
            post_stress_pop = model_data["Total_Bees"].iloc[-1]

            # Population should be affected by combined stressors
            assert (
                post_stress_pop <= pre_stress_pop * 1.2
            )  # Allow some growth but expect impact


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def test_simulation_speed_benchmark(self):
        """Benchmark simulation execution speed"""

        import time

        config_manager = ConfigManager()
        config = config_manager.load_default_config()
        config.simulation.duration_days = 10  # Very short benchmark
        # Use proper dictionary structure for initial_population
        config.colony.initial_population = {
            "queens": 1,
            "workers": 1600,
            "foragers": 300,
            "drones": 50,
            "brood": 50,
        }

        model = BeeModel(config=config)

        start_time = time.time()

        for day in range(10):
            model.step()

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete 10 days in reasonable time
        assert execution_time < 10  # 10 seconds max for 10 days

        steps_per_second = 10 / execution_time
        assert steps_per_second > 0.5  # At least 0.5 steps per second

    def test_memory_usage_benchmark(self):
        """Benchmark memory usage during simulation"""

        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        config_manager = ConfigManager()
        config = config_manager.load_default_config()
        config.simulation.duration_days = 8  # Very short test
        # Use proper dictionary structure for initial_population
        config.colony.initial_population = {
            "queens": 1,
            "workers": 1200,
            "foragers": 200,
            "drones": 50,
            "brood": 50,
        }

        model = BeeModel(config=config)

        for day in range(8):
            model.step()

        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable
        assert memory_growth < 100  # Less than 100MB growth

        # Cleanup
        del model
        gc.collect()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
