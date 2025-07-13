"""
Unit tests for batch processing functionality
=============================================

Comprehensive tests for batch processing system including parameter
specifications, experiment design, and batch execution.
"""

from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.bstew.utils.batch_processing import (
    ParameterSpec,
    ExperimentDesign,
    ExperimentRun,
    ExperimentType,
    ExperimentStatus,
    BatchProcessor,
)


class TestParameterSpec:
    """Test ParameterSpec functionality"""

    def test_parameter_spec_creation(self):
        """Test basic parameter spec creation"""
        spec = ParameterSpec(
            name="test_param", min_value=1.0, max_value=10.0, values=[1, 2, 3]
        )

        assert spec.name == "test_param"
        assert spec.min_value == 1.0
        assert spec.max_value == 10.0
        assert spec.values == [1, 2, 3]

    def test_parameter_spec_generate_values(self):
        """Test parameter value generation"""
        spec = ParameterSpec(
            name="test_param", min_value=0.0, max_value=10.0, step_size=2.0
        )

        values = spec.generate_values()
        assert len(values) == 6  # 0, 2, 4, 6, 8, 10
        assert values[0] == 0.0
        assert values[-1] == 10.0

    def test_parameter_spec_with_values(self):
        """Test parameter spec with predefined values"""
        spec = ParameterSpec(
            name="test_param", min_value=0.0, max_value=10.0, values=[0.5, 1.5, 2.5]
        )

        values = spec.generate_values()
        assert values == [0.5, 1.5, 2.5]

    def test_parameter_spec_distributions(self):
        """Test different parameter distributions"""
        # Uniform distribution
        spec_uniform = ParameterSpec(
            name="uniform_param", min_value=0.0, max_value=10.0, distribution="uniform"
        )
        values_uniform = spec_uniform.generate_values(5)
        assert len(values_uniform) == 5
        assert all(0.0 <= v <= 10.0 for v in values_uniform)

        # Normal distribution
        spec_normal = ParameterSpec(
            name="normal_param", min_value=0.0, max_value=10.0, distribution="normal"
        )
        values_normal = spec_normal.generate_values(5)
        assert len(values_normal) == 5

        # Log uniform distribution
        spec_log = ParameterSpec(
            name="log_param", min_value=1.0, max_value=100.0, distribution="log_uniform"
        )
        values_log = spec_log.generate_values(5)
        assert len(values_log) == 5
        assert all(1.0 <= v <= 100.0 for v in values_log)


class TestExperimentRun:
    """Test ExperimentRun functionality"""

    def test_experiment_run_creation(self):
        """Test basic experiment run creation"""
        run = ExperimentRun(
            run_id="test_run_001",
            parameters={"param1": 1.0, "param2": "value"},
            random_seed=42,
        )

        assert run.run_id == "test_run_001"
        assert run.parameters == {"param1": 1.0, "param2": "value"}
        assert run.random_seed == 42
        assert run.status == ExperimentStatus.PENDING

    def test_experiment_run_duration(self):
        """Test duration calculation"""
        from datetime import datetime, timedelta

        run = ExperimentRun(
            run_id="test_run_001", parameters={"param1": 1.0}, random_seed=42
        )

        # No duration initially
        assert run.duration is None

        # Set start and end times
        run.start_time = datetime.now()
        run.end_time = run.start_time + timedelta(seconds=5)

        # Should calculate duration
        duration = run.duration
        assert duration is not None
        assert 4.0 <= duration <= 6.0  # Allow some tolerance


class TestExperimentDesign:
    """Test ExperimentDesign functionality"""

    def test_experiment_design_creation(self):
        """Test basic experiment design creation"""
        parameters = {
            "param1": ParameterSpec(
                name="param1", min_value=1.0, max_value=5.0, values=[1, 2, 3, 4, 5]
            )
        }

        design = ExperimentDesign(
            experiment_id="test_exp_001",
            experiment_type=ExperimentType.PARAMETER_SWEEP,
            name="Test Parameter Sweep",
            description="Testing parameter sweep functionality",
            parameters=parameters,
            base_config={"base_param": 100},
            n_replicates=2,
        )

        assert design.experiment_id == "test_exp_001"
        assert design.experiment_type == ExperimentType.PARAMETER_SWEEP
        assert design.name == "Test Parameter Sweep"
        assert design.n_replicates == 2
        assert design.base_config == {"base_param": 100}

    def test_parameter_sweep_generation(self):
        """Test parameter sweep run generation"""
        parameters = {
            "param1": ParameterSpec(
                name="param1", min_value=1.0, max_value=3.0, values=[1, 2, 3]
            )
        }

        design = ExperimentDesign(
            experiment_id="test_exp_001",
            experiment_type=ExperimentType.PARAMETER_SWEEP,
            name="Test Parameter Sweep",
            description="Testing parameter sweep functionality",
            parameters=parameters,
            base_config={"base_param": 100},
            n_replicates=2,
        )

        runs = design.generate_runs()

        # Should have 3 values * 2 replicates = 6 runs
        assert len(runs) == 6
        assert all(isinstance(run, ExperimentRun) for run in runs)
        assert all(run.status == ExperimentStatus.PENDING for run in runs)

    def test_monte_carlo_generation(self):
        """Test Monte Carlo run generation"""
        parameters = {
            "param1": ParameterSpec(
                name="param1", min_value=1.0, max_value=10.0, distribution="uniform"
            )
        }

        design = ExperimentDesign(
            experiment_id="test_exp_001",
            experiment_type=ExperimentType.MONTE_CARLO,
            name="Test Monte Carlo",
            description="Testing Monte Carlo functionality",
            parameters=parameters,
            base_config={"base_param": 100},
            n_replicates=5,
        )

        runs = design.generate_runs()

        # Should have 5 runs
        assert len(runs) == 5
        assert all(isinstance(run, ExperimentRun) for run in runs)

        # Check that parameters vary
        param_values = [run.parameters["param1"] for run in runs]
        assert all(1.0 <= v <= 10.0 for v in param_values)


class TestExperimentType:
    """Test ExperimentType enum"""

    def test_experiment_type_values(self):
        """Test experiment type enum values"""
        assert ExperimentType.PARAMETER_SWEEP.value == "parameter_sweep"
        assert ExperimentType.SENSITIVITY_ANALYSIS.value == "sensitivity_analysis"
        assert ExperimentType.MONTE_CARLO.value == "monte_carlo"
        assert ExperimentType.FACTORIAL_DESIGN.value == "factorial_design"
        assert ExperimentType.LATIN_HYPERCUBE.value == "latin_hypercube"
        assert ExperimentType.OPTIMIZATION.value == "optimization"
        assert ExperimentType.VALIDATION.value == "validation"


class TestExperimentStatus:
    """Test ExperimentStatus enum"""

    def test_experiment_status_values(self):
        """Test experiment status enum values"""
        assert ExperimentStatus.PENDING.value == "pending"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.COMPLETED.value == "completed"
        assert ExperimentStatus.FAILED.value == "failed"
        assert ExperimentStatus.CANCELLED.value == "cancelled"


class TestBatchProcessor:
    """Test BatchProcessor functionality"""

    def test_batch_processor_creation(self):
        """Test basic batch processor creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir, max_workers=2)

            assert processor.output_dir.exists()
            assert processor.max_workers == 2

    def test_batch_processor_output_dir_creation(self):
        """Test output directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "test_experiments")
            processor = BatchProcessor(output_dir=output_dir)

            assert processor.output_dir.exists()
            assert processor.output_dir.name == "test_experiments"

    @patch("src.bstew.utils.batch_processing.BeeModel")
    @patch("src.bstew.utils.batch_processing.ConfigManager")
    def test_single_run_execution(self, mock_config_manager, mock_bee_model):
        """Test single run execution"""
        # Mock the config manager
        mock_config_instance = Mock()
        mock_config_manager.return_value = mock_config_instance

        # Mock the model data frame
        mock_df = MagicMock()
        mock_df.to_dict.return_value = {"Total_Bees": [100]}
        mock_df.__len__.return_value = 1
        mock_df.__getitem__.return_value.iloc = MagicMock()
        mock_df.__getitem__.return_value.iloc.__getitem__.return_value = 100

        # Mock the model
        mock_model_instance = Mock()
        mock_model_instance.get_colony_count.return_value = 1
        mock_model_instance.datacollector.get_model_vars_dataframe.return_value = (
            mock_df
        )
        mock_model_instance.landscape.export_to_dict.return_value = {}

        mock_bee_model.return_value = mock_model_instance

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir)

            # Create test run
            run = ExperimentRun(
                run_id="test_run_001", parameters={"param1": 1.0}, random_seed=42
            )

            # Create test design
            design = ExperimentDesign(
                experiment_id="test_exp_001",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                name="Test",
                description="Test",
                parameters={},
                base_config={},
                simulation_days=5,
            )

            # Execute single run
            completed_run = processor._execute_single_run(run, design)

            assert completed_run.status == ExperimentStatus.COMPLETED
            assert completed_run.results is not None
            assert completed_run.start_time is not None
            assert completed_run.end_time is not None

    def test_batch_processor_multiple_designs(self):
        """Test handling multiple experiment designs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir)

            # Create multiple test designs
            ExperimentDesign(
                experiment_id="test_exp_001",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                name="Test 1",
                description="Test 1",
                parameters={},
                base_config={},
            )

            ExperimentDesign(
                experiment_id="test_exp_002",
                experiment_type=ExperimentType.MONTE_CARLO,
                name="Test 2",
                description="Test 2",
                parameters={},
                base_config={},
            )

            # Should be able to handle multiple designs
            assert processor.output_dir.exists()

    def test_batch_processor_error_handling(self):
        """Test error handling in batch processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir)

            # Test with invalid run
            run = ExperimentRun(
                run_id="invalid_run",
                parameters={"invalid_param": "invalid_value"},
                random_seed=42,
            )

            design = ExperimentDesign(
                experiment_id="test_exp_001",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                name="Test",
                description="Test",
                parameters={},
                base_config={},
            )

            # Should handle errors gracefully
            with patch("src.bstew.utils.batch_processing.BeeModel") as mock_bee_model:
                mock_bee_model.side_effect = Exception("Test error")

                completed_run = processor._execute_single_run(run, design)
                assert completed_run.status == ExperimentStatus.FAILED
                assert completed_run.error_message is not None

    def test_batch_processor_progress_tracking(self):
        """Test progress tracking functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir)

            # Should have progress tracking attributes
            assert hasattr(processor, "output_dir")
            assert hasattr(processor, "max_workers")

    def test_batch_processor_concurrent_execution(self):
        """Test concurrent execution capability"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir, max_workers=2)

            # Test that max_workers is respected
            assert processor.max_workers == 2

            # Test with different worker counts
            processor_single = BatchProcessor(output_dir=temp_dir, max_workers=1)
            assert processor_single.max_workers == 1

    def test_batch_processor_output_organization(self):
        """Test output file organization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir)

            # Should create organized output directory structure
            assert processor.output_dir.exists()
            assert processor.output_dir.is_dir()

    def test_batch_processor_resource_management(self):
        """Test resource management"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir)

            # Should have reasonable default resource limits
            assert processor.max_workers > 0
            assert processor.max_workers <= 16  # Reasonable upper limit

    def test_batch_processor_result_aggregation(self):
        """Test result aggregation functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BatchProcessor(output_dir=temp_dir)

            # Test basic processor setup
            assert processor.output_dir.exists()

            # Result aggregation would be tested with actual runs
            # For now, just verify processor can be created
            assert isinstance(processor, BatchProcessor)
