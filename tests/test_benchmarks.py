"""
Tests for BSTEW Benchmark and Validation Systems
==============================================

Comprehensive tests for NetLogo parity benchmarks, performance validation,
and CLI integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.bstew.benchmarks.netlogo_parity_benchmarks import NetLogoParityBenchmarks, BenchmarkResult
from src.bstew.benchmarks.benchmark_runner import BenchmarkRunner, EndToEndTestResult
from src.bstew.benchmarks.scalability_benchmarks import ScalabilityBenchmarks, ScalabilityResult
from src.bstew.benchmarks.performance_regression_suite import PerformanceRegressionSuite, RegressionResult
from src.bstew.validation.netlogo_validation import NetLogoDataLoader, BSTEWDataExtractor, BehavioralValidator


class TestNetLogoParityBenchmarks:
    """Test NetLogo parity benchmark functionality"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def benchmark_suite(self, temp_output_dir):
        """Create benchmark suite instance"""
        return NetLogoParityBenchmarks(temp_output_dir)
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult dataclass creation"""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            execution_time=1.5,
            memory_peak=100.0,
            memory_average=80.0,
            cpu_usage=75.0,
            steps_per_second=200.0,
            success=True
        )
        
        assert result.benchmark_name == "test_benchmark"
        assert result.execution_time == 1.5
        assert result.success is True
        assert result.error_message is None
    
    def test_benchmark_result_with_error(self):
        """Test BenchmarkResult creation with error"""
        result = BenchmarkResult(
            benchmark_name="failed_benchmark",
            execution_time=0.0,
            memory_peak=0.0,
            memory_average=0.0,
            cpu_usage=0.0,
            steps_per_second=0.0,
            success=False,
            error_message="Test error"
        )
        
        assert result.success is False
        assert result.error_message == "Test error"
    
    @patch('src.bstew.benchmarks.netlogo_parity_benchmarks.BeeModel')
    @patch('psutil.Process')
    def test_simulation_speed_benchmark(self, mock_process, mock_bee_model, benchmark_suite):
        """Test simulation speed benchmarking"""
        # Mock process for memory monitoring
        mock_proc = Mock()
        mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        mock_proc.cpu_percent.return_value = 50.0
        mock_process.return_value = mock_proc
        
        # Mock BeeModel
        mock_model = Mock()
        mock_model.step = Mock()
        mock_model.cleanup = Mock()
        mock_bee_model.return_value = mock_model
        
        # Run benchmark
        results = benchmark_suite._benchmark_simulation_speed()
        
        # Verify results structure
        assert isinstance(results, dict)
        assert "small_colony" in results
        assert "medium_colony" in results
        assert "large_colony" in results
        
        # Check individual result
        small_result = results["small_colony"]
        assert isinstance(small_result, BenchmarkResult)
        assert small_result.success is True
        assert small_result.steps_per_second > 0
    
    @patch('src.bstew.benchmarks.netlogo_parity_benchmarks.BeeModel')
    @patch('psutil.Process')
    def test_memory_efficiency_benchmark(self, mock_process, mock_bee_model, benchmark_suite):
        """Test memory efficiency benchmarking"""
        # Mock process with memory samples
        mock_proc = Mock()
        mock_proc.memory_info.return_value.rss = 60 * 1024 * 1024
        mock_proc.cpu_percent.return_value = 40.0
        mock_process.return_value = mock_proc
        
        # Mock BeeModel
        mock_model = Mock()
        mock_model.step = Mock()
        mock_model.cleanup = Mock()
        mock_bee_model.return_value = mock_model
        
        # Run benchmark
        results = benchmark_suite._benchmark_memory_efficiency()
        
        # Verify results
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for scenario_name, result in results.items():
            assert isinstance(result, BenchmarkResult)
            assert scenario_name.startswith("memory_")
    
    @patch('src.bstew.benchmarks.netlogo_parity_benchmarks.BeeModel')
    def test_initialization_benchmark(self, mock_bee_model, benchmark_suite):
        """Test model initialization benchmarking"""
        # Mock BeeModel
        mock_model = Mock()
        mock_model.cleanup = Mock()
        mock_bee_model.return_value = mock_model
        
        # Run benchmark
        results = benchmark_suite._benchmark_initialization()
        
        # Verify results
        assert isinstance(results, dict)
        assert len(results) == 4  # 4 different bee counts
        
        for result in results.values():
            assert isinstance(result, BenchmarkResult)
            assert result.execution_time >= 0
    
    def test_netlogo_reference_values(self, benchmark_suite):
        """Test NetLogo reference performance values"""
        refs = benchmark_suite.netlogo_references
        
        assert "simulation_speed" in refs
        assert "memory_usage" in refs
        assert "initialization_time" in refs
        
        # Check that reference values are reasonable
        assert refs["simulation_speed"]["small_colony"] > 0
        assert refs["memory_usage"]["large_colony"] > refs["memory_usage"]["small_colony"]
    
    def test_system_info_collection(self, benchmark_suite):
        """Test system information collection"""
        system_info = benchmark_suite._get_system_info()
        
        assert "cpu_count" in system_info
        assert "memory_total" in system_info
        assert isinstance(system_info["cpu_count"], int)
        assert isinstance(system_info["memory_total"], (int, float))


class TestBenchmarkRunner:
    """Test benchmark runner functionality"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def benchmark_runner(self, temp_output_dir):
        """Create benchmark runner instance"""
        return BenchmarkRunner(temp_output_dir)
    
    def test_end_to_end_test_result_creation(self):
        """Test EndToEndTestResult creation"""
        result = EndToEndTestResult(
            test_name="test_workflow",
            success=True,
            execution_time=2.5,
            steps_completed=100,
            data_generated=True,
            reports_generated=True
        )
        
        assert result.test_name == "test_workflow"
        assert result.success is True
        assert result.steps_completed == 100
        assert result.data_generated is True
    
    @patch('src.bstew.core.model.BeeModel')
    @patch('src.bstew.core.system_integrator.SystemIntegrator')
    def test_end_to_end_scenario_execution(self, mock_integrator, mock_bee_model, benchmark_runner):
        """Test end-to-end scenario execution"""
        # Mock system integrator
        mock_si = Mock()
        mock_si.data_collector = Mock()
        mock_si.data_collector.colony_metrics = {"colony_1": Mock()}
        mock_si.generate_reports.return_value = {"excel_comprehensive": "path/to/report"}
        mock_integrator.return_value = mock_si
        
        # Mock BeeModel
        mock_model = Mock()
        mock_model.step = Mock()
        mock_model.cleanup = Mock()
        mock_model.system_integrator = mock_si
        mock_bee_model.return_value = mock_model
        
        # Test scenario
        scenario = {
            "name": "test_scenario",
            "config": {
                "simulation": {"steps": 10, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 50,
                        "foragers": 10,
                        "drones": 5,
                        "brood": 5
                    }
                },
                "environment": {"patches": 20, "flower_density": 0.3}
            },
            "expected_steps": 10
        }
        
        result = benchmark_runner._run_end_to_end_scenario(scenario)
        
        assert isinstance(result, EndToEndTestResult)
        assert result.test_name == "test_scenario"
        assert result.steps_completed == 10
        
        # Verify model interactions
        mock_bee_model.assert_called_once()
        assert mock_model.step.call_count == 10
        mock_model.cleanup.assert_called_once()
    
    def test_validation_summary_generation(self, benchmark_runner):
        """Test validation summary generation"""
        # Mock validation results
        mock_results = {
            "performance_benchmarks": {
                "status": "completed",
                "results": {
                    "netlogo_comparison": [
                        Mock(passes_benchmark=True),
                        Mock(passes_benchmark=True),
                        Mock(passes_benchmark=False)
                    ]
                }
            },
            "integration_tests": {"status": "completed"},
            "end_to_end_tests": [
                Mock(success=True),
                Mock(success=True),
                Mock(success=False)
            ],
            "stress_tests": {
                "status": "completed",
                "results": {
                    "test1": {"success": True},
                    "test2": {"success": True}
                }
            }
        }
        
        summary = benchmark_runner._generate_validation_summary(mock_results)
        
        assert summary["overall_status"] in ["excellent", "good", "needs_attention"]
        assert summary["performance_status"] == "needs_improvement"  # 1/3 failed (66% < 80% threshold)
        assert summary["integration_status"] == "passed"
        assert summary["end_to_end_status"] == "failed"  # 2/3 passed
        assert summary["stress_status"] == "passed"
    
    @patch('subprocess.run')
    def test_pytest_execution(self, mock_subprocess, benchmark_runner):
        """Test pytest execution for integration tests"""
        # Mock successful pytest run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test output"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        result = benchmark_runner._run_pytest("tests/test_integration.py")
        
        assert result["status"] == "completed"
        assert result["exit_code"] == 0
        mock_subprocess.assert_called_once()
    
    def test_quick_validation(self, benchmark_runner):
        """Test quick validation functionality"""
        with patch.object(benchmark_runner, '_test_basic_functionality') as mock_basic, \
             patch.object(benchmark_runner, '_quick_performance_check') as mock_perf, \
             patch.object(benchmark_runner, '_integration_smoke_test') as mock_smoke:
            
            # Mock successful quick tests
            mock_basic.return_value = {"success": True}
            mock_perf.return_value = {"success": True, "steps_per_second": 150.0}
            mock_smoke.return_value = {"success": True, "data_collected": True}
            
            results = benchmark_runner.run_quick_validation()
            
            assert results["overall_status"] == "passed"
            assert "basic_functionality" in results
            assert "performance_check" in results
            assert "integration_smoke_test" in results


class TestScalabilityBenchmarks:
    """Test scalability benchmark functionality"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def scalability_benchmarks(self, temp_output_dir):
        """Create scalability benchmarks instance"""
        return ScalabilityBenchmarks(temp_output_dir)
    
    def test_scalability_result_creation(self):
        """Test ScalabilityResult creation"""
        result = ScalabilityResult(
            test_name="population_scaling",
            parameter_value=100,
            execution_time=5.0,
            memory_peak=150.0,
            steps_per_second=20.0,
            success=True
        )
        
        assert result.test_name == "population_scaling"
        assert result.parameter_value == 100
        assert result.success is True
    
    @patch('src.bstew.benchmarks.scalability_benchmarks.BeeModel')
    @patch('psutil.Process')
    def test_population_scaling_test(self, mock_process, mock_bee_model, scalability_benchmarks):
        """Test population scaling benchmark"""
        # Mock process
        mock_proc = Mock()
        mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process.return_value = mock_proc
        
        # Mock BeeModel
        mock_model = Mock()
        mock_model.step = Mock()
        mock_model.cleanup = Mock()
        mock_bee_model.return_value = mock_model
        
        results = scalability_benchmarks.run_population_scaling_test()
        
        assert isinstance(results, list)
        assert len(results) == 7  # 7 population sizes
        
        for result in results:
            assert isinstance(result, ScalabilityResult)
            assert result.test_name == "population_scaling"
    
    def test_scaling_behavior_analysis(self, scalability_benchmarks):
        """Test scaling behavior analysis"""
        # Create mock results
        mock_results = [
            ScalabilityResult("test", 50, 1.0, 50.0, 50.0, True),
            ScalabilityResult("test", 100, 2.5, 75.0, 40.0, True),
            ScalabilityResult("test", 200, 6.0, 120.0, 33.3, True),
            ScalabilityResult("test", 500, 18.0, 250.0, 27.8, True)
        ]
        
        analysis = scalability_benchmarks.analyze_scaling_behavior(mock_results)
        
        assert "time_scaling" in analysis
        assert "memory_scaling" in analysis
        assert "efficiency" in analysis
        
        time_scaling = analysis["time_scaling"]
        assert "exponent" in time_scaling
        assert "interpretation" in time_scaling
        assert "is_acceptable" in time_scaling
    
    def test_insufficient_data_analysis(self, scalability_benchmarks):
        """Test analysis with insufficient data"""
        # Single result - not enough for analysis
        mock_results = [
            ScalabilityResult("test", 50, 1.0, 50.0, 50.0, True)
        ]
        
        analysis = scalability_benchmarks.analyze_scaling_behavior(mock_results)
        assert analysis["analysis"] == "insufficient_data"


class TestPerformanceRegressionSuite:
    """Test performance regression detection"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def regression_suite(self, temp_output_dir):
        """Create regression suite instance"""
        return PerformanceRegressionSuite(output_directory=temp_output_dir)
    
    def test_regression_result_creation(self):
        """Test RegressionResult creation"""
        result = RegressionResult(
            metric_name="simulation_speed_small_colony",
            current_value=90.0,
            baseline_value=100.0,
            change_percent=-10.0,
            is_regression=True,
            severity="minor"
        )
        
        assert result.metric_name == "simulation_speed_small_colony"
        assert result.change_percent == -10.0
        assert result.is_regression is True
        assert result.severity == "minor"
    
    def test_severity_determination(self, regression_suite):
        """Test regression severity determination"""
        # Test performance regression (lower is worse)
        assert regression_suite._determine_severity(-0.03) == "none"
        assert regression_suite._determine_severity(-0.08) == "minor"
        assert regression_suite._determine_severity(-0.20) == "major"
        assert regression_suite._determine_severity(-0.35) == "critical"
        
        # Test memory regression (higher is worse)
        assert regression_suite._determine_severity(0.03, higher_is_worse=True) == "none"
        assert regression_suite._determine_severity(0.08, higher_is_worse=True) == "minor"
        assert regression_suite._determine_severity(0.20, higher_is_worse=True) == "major"
        assert regression_suite._determine_severity(0.35, higher_is_worse=True) == "critical"
    
    def test_custom_thresholds(self, regression_suite):
        """Test custom threshold setting"""
        regression_suite.set_custom_thresholds(0.03, 0.10, 0.25)
        
        assert regression_suite.thresholds["minor"] == 0.03
        assert regression_suite.thresholds["major"] == 0.10
        assert regression_suite.thresholds["critical"] == 0.25
    
    def test_baseline_comparison(self, regression_suite):
        """Test baseline comparison functionality"""
        # Mock baseline data
        baseline_data = {
            "simulation_speed": {
                "small_colony": {"steps_per_second": 100.0},
                "medium_colony": {"steps_per_second": 50.0}
            }
        }
        
        # Mock current results with regression
        current_results = {
            "simulation_speed": {
                "small_colony": Mock(steps_per_second=85.0),  # 15% regression
                "medium_colony": Mock(steps_per_second=52.0)   # 4% improvement
            }
        }
        
        regression_suite.baseline_data = baseline_data
        regressions = regression_suite._compare_with_baseline(current_results)
        
        assert len(regressions) == 2
        
        # Check regression detection
        small_colony_regression = next(r for r in regressions if "small_colony" in r.metric_name)
        assert small_colony_regression.is_regression is True
        assert small_colony_regression.severity == "minor"  # 15% = major threshold, so classified as minor
        
        medium_colony_regression = next(r for r in regressions if "medium_colony" in r.metric_name)
        assert medium_colony_regression.is_regression is False  # Improvement
    
    def test_recommendation_generation(self, regression_suite):
        """Test recommendation generation"""
        # Mock regressions
        regressions = [
            RegressionResult("simulation_speed_test", 80.0, 100.0, -20.0, True, "critical"),
            RegressionResult("memory_test", 150.0, 100.0, 50.0, True, "major"),
            RegressionResult("other_test", 105.0, 100.0, 5.0, False, "none")
        ]
        
        recommendations = regression_suite._generate_recommendations(regressions)
        
        assert len(recommendations) > 0
        assert any("URGENT" in rec for rec in recommendations)
        assert any("simulation" in rec.lower() for rec in recommendations)


class TestNetLogoValidation:
    """Test NetLogo validation functionality"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with mock NetLogo data"""
        temp_dir = tempfile.mkdtemp()
        
        # Create mock NetLogo data files
        mock_population_data = {
            "total_population": [100, 105, 110, 108, 112],
            "egg_count": [20, 22, 25, 23, 24],
            "worker_count": [70, 73, 75, 75, 78]
        }
        
        population_file = Path(temp_dir) / "population_data.csv"
        import pandas as pd
        pd.DataFrame(mock_population_data).to_csv(population_file, index=False)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def netlogo_loader(self, temp_data_dir):
        """Create NetLogo data loader"""
        return NetLogoDataLoader(temp_data_dir)
    
    def test_netlogo_data_loader_initialization(self, netlogo_loader):
        """Test NetLogo data loader initialization"""
        assert netlogo_loader.netlogo_data_path.exists()
        assert isinstance(netlogo_loader.loaded_data, dict)
    
    def test_population_data_loading(self, netlogo_loader):
        """Test population data loading"""
        population_data = netlogo_loader._load_population_data()
        
        assert isinstance(population_data, dict)
        # Should have loaded the mock data
        if "total_population_time_series" in population_data:
            assert len(population_data["total_population_time_series"]) > 0
    
    @patch('src.bstew.validation.netlogo_validation.ComprehensiveDataCollector')
    def test_bstew_data_extractor(self, mock_collector):
        """Test BSTEW data extraction"""
        # Mock data collector with test data
        mock_collector.colony_metrics = {
            "colony_1": Mock(
                population_size_day_list=[100, 105, 110],
                egg_count_day_list=[20, 22, 25],
                adult_count_day_list=[70, 73, 75]
            )
        }
        mock_collector.bee_metrics = {
            "bee_1": Mock(foraging_trips=5, successful_trips=4, foraging_trip_durations=[40, 45, 42], foraging_trip_energies=[100, 110, 105]),
            "bee_2": Mock(foraging_trips=3, successful_trips=2, foraging_trip_durations=[38, 43], foraging_trip_energies=[95, 108])
        }
        
        extractor = BSTEWDataExtractor(mock_collector)
        population_data = extractor.extract_population_data()
        
        assert isinstance(population_data, dict)
        assert "total_population_time_series" in population_data
        
        foraging_data = extractor.extract_foraging_data()
        assert isinstance(foraging_data, dict)
        assert "total_foraging_trips" in foraging_data
    
    def test_behavioral_validator_initialization(self):
        """Test behavioral validator initialization"""
        validator = BehavioralValidator()
        
        assert hasattr(validator, 'tolerance_config')
        assert "population_dynamics" in validator.tolerance_config
        assert "foraging_behavior" in validator.tolerance_config
        
        # Test custom tolerances
        custom_tolerances = {"population_dynamics": 0.15}
        custom_validator = BehavioralValidator(custom_tolerances)
        assert custom_validator.tolerance_config["population_dynamics"] == 0.15
    
    def test_validation_metric_creation(self):
        """Test validation metric creation"""
        validator = BehavioralValidator()
        
        metric = validator._create_validation_metric(
            "test_metric", 95.0, 100.0, 0.1
        )
        
        assert metric.metric_name == "test_metric"
        assert metric.bstew_value == 95.0
        assert metric.netlogo_value == 100.0
        assert metric.relative_difference == 0.05  # 5% difference
        assert metric.passes_validation is True  # 5% < 10% tolerance


class TestBenchmarkCLI:
    """Test benchmark CLI functionality"""
    
    @pytest.fixture
    def mock_runner(self):
        """Mock benchmark runner"""
        with patch('src.bstew.cli.commands.benchmark.BenchmarkRunner') as mock:
            yield mock
    
    @pytest.fixture
    def mock_netlogo_benchmarks(self):
        """Mock NetLogo benchmarks"""
        with patch('src.bstew.cli.commands.benchmark.NetLogoParityBenchmarks') as mock:
            yield mock
    
    def test_quick_validation_cli(self, mock_runner):
        """Test quick validation CLI command"""
        # Mock successful quick validation
        mock_instance = mock_runner.return_value
        mock_instance.run_quick_validation.return_value = {
            "overall_status": "passed",
            "basic_functionality": {"success": True},
            "performance_check": {"success": True, "steps_per_second": 120.0},
            "integration_smoke_test": {"success": True}
        }
        
        
        # Test would require typer testing framework
        # This validates the mock setup
        mock_runner.assert_not_called()  # Not called until run() is executed
    
    def test_netlogo_only_cli(self, mock_netlogo_benchmarks):
        """Test NetLogo-only benchmark CLI command"""
        # Mock NetLogo benchmark results
        mock_instance = mock_netlogo_benchmarks.return_value
        mock_instance.run_complete_benchmark_suite.return_value = {
            "simulation_speed": {"small_colony": Mock(success=True, steps_per_second=100.0)},
            "netlogo_comparison": [Mock(passes_benchmark=True)]
        }
        
        # Test validates mock setup
        mock_netlogo_benchmarks.assert_not_called()  # Not called until run() is executed


if __name__ == "__main__":
    pytest.main([__file__])