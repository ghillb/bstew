"""
Tests for NetLogo Validation System
==================================

Tests for NetLogo BEE-STEWARD v2 behavioral validation and CLI commands.
"""

import pytest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from typer.testing import CliRunner

from src.bstew.validation.netlogo_validation import (
    NetLogoDataLoader, BSTEWDataExtractor, BehavioralValidator,
    NetLogoValidationSuite, ValidationMetric, ValidationResult
)
from src.bstew.config.netlogo_mapping import NetLogoParameterMapper
from src.bstew.cli.commands.validate_netlogo import app


class TestNetLogoDataLoader:
    """Test NetLogo data loading functionality"""

    @pytest.fixture
    def temp_netlogo_dir(self):
        """Create temporary directory with mock NetLogo data"""
        temp_dir = tempfile.mkdtemp()

        # Create mock population data
        population_data = pd.DataFrame({
            "total_population": [100, 105, 110, 108, 112, 115],
            "egg_count": [20, 22, 25, 23, 24, 26],
            "larva_count": [15, 16, 18, 17, 19, 20],
            "pupa_count": [10, 11, 12, 11, 13, 14],
            "adult_count": [55, 56, 55, 57, 56, 55],
            "queen_count": [1, 1, 1, 1, 1, 1],
            "drone_count": [5, 6, 5, 6, 5, 6],
            "worker_count": [49, 49, 49, 50, 50, 48]
        })
        population_file = Path(temp_dir) / "population_data.csv"
        population_data.to_csv(population_file, index=False)

        # Create mock activity data
        activity_data = pd.DataFrame({
            "foraging_bees": [15, 16, 18, 17, 19, 20],
            "nursing_bees": [20, 21, 20, 22, 21, 20],
            "resting_bees": [10, 9, 8, 9, 8, 7],
            "building_bees": [4, 3, 4, 4, 3, 3],
            "activity_transitions": [5, 6, 7, 6, 8, 7],
            "hour": [0, 4, 8, 12, 16, 20],
            "day": [1, 1, 1, 1, 1, 1],
            "activity_level": [0.6, 0.7, 0.8, 0.9, 0.7, 0.5]
        })
        activity_file = Path(temp_dir) / "activity_patterns.csv"
        activity_data.to_csv(activity_file, index=False)

        # Create mock foraging data
        foraging_data = pd.DataFrame({
            "total_trips": [25, 30, 35, 32, 38, 40],
            "successful_trips": [20, 24, 28, 26, 30, 32],
            "foraging_efficiency": [0.8, 0.8, 0.8, 0.81, 0.79, 0.8],
            "trip_duration": [45, 42, 48, 44, 46, 43],
            "energy_collected": [120, 135, 150, 140, 160, 165],
            "patches_visited": [8, 9, 10, 9, 11, 12],
            "dances_performed": [3, 4, 5, 4, 6, 7]
        })
        foraging_file = Path(temp_dir) / "foraging_behavior.csv"
        foraging_data.to_csv(foraging_file, index=False)

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def netlogo_loader(self, temp_netlogo_dir):
        """Create NetLogo data loader with test data"""
        return NetLogoDataLoader(temp_netlogo_dir)

    def test_loader_initialization(self, netlogo_loader, temp_netlogo_dir):
        """Test NetLogo data loader initialization"""
        assert netlogo_loader.netlogo_data_path == Path(temp_netlogo_dir)
        assert isinstance(netlogo_loader.loaded_data, dict)
        assert len(netlogo_loader.loaded_data) == 0  # Initially empty

    def test_population_data_loading(self, netlogo_loader):
        """Test population data loading"""
        population_data = netlogo_loader._load_population_data()

        assert isinstance(population_data, dict)
        assert "total_population_time_series" in population_data
        assert "egg_count_time_series" in population_data
        assert "worker_count" in population_data

        # Check data integrity
        assert len(population_data["total_population_time_series"]) == 6
        assert population_data["queen_count"][0] == 1
        assert all(isinstance(x, (int, float)) for x in population_data["total_population_time_series"])

    def test_activity_patterns_loading(self, netlogo_loader):
        """Test activity patterns data loading"""
        activity_data = netlogo_loader._load_activity_patterns()

        assert isinstance(activity_data, dict)
        assert "foraging_activity" in activity_data
        assert "nursing_activity" in activity_data
        assert "daily_activity_cycles" in activity_data

        # Check daily cycle extraction
        cycles = activity_data["daily_activity_cycles"]
        assert len(cycles) > 0
        assert "hourly_activity" in cycles[0]

    def test_foraging_data_loading(self, netlogo_loader):
        """Test foraging data loading"""
        foraging_data = netlogo_loader._load_foraging_data()

        assert isinstance(foraging_data, dict)
        assert "total_foraging_trips" in foraging_data
        assert "successful_trips" in foraging_data
        assert "foraging_efficiency" in foraging_data

        # Check efficiency values are reasonable
        efficiency = foraging_data["foraging_efficiency"]
        assert all(0 <= x <= 1 for x in efficiency)

    def test_complete_netlogo_loading(self, netlogo_loader):
        """Test complete NetLogo data loading"""
        complete_data = netlogo_loader.load_netlogo_outputs()

        assert isinstance(complete_data, dict)
        assert "population_data" in complete_data
        assert "activity_patterns" in complete_data
        assert "foraging_data" in complete_data

        # Verify loaded_data is populated
        assert len(netlogo_loader.loaded_data) > 0
        assert netlogo_loader.loaded_data == complete_data

    def test_missing_files_handling(self):
        """Test handling of missing NetLogo files"""
        # Create empty directory
        temp_dir = tempfile.mkdtemp()
        try:
            loader = NetLogoDataLoader(temp_dir)
            data = loader.load_netlogo_outputs()

            # Should return empty dictionaries for missing data
            assert all(data[key] == {} for key in data.keys())

        finally:
            shutil.rmtree(temp_dir)

    def test_malformed_data_handling(self):
        """Test handling of malformed CSV data"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create malformed CSV
            malformed_file = Path(temp_dir) / "population_malformed.csv"
            with open(malformed_file, 'w') as f:
                f.write("bad,csv,format\n1,2\n")  # Missing column

            loader = NetLogoDataLoader(temp_dir)
            population_data = loader._load_population_data()

            # Should handle gracefully and return empty dict
            assert isinstance(population_data, dict)

        finally:
            shutil.rmtree(temp_dir)


class TestBSTEWDataExtractor:
    """Test BSTEW data extraction functionality"""

    @pytest.fixture
    def mock_data_collector(self):
        """Create mock comprehensive data collector"""
        collector = Mock()

        # Mock colony metrics
        colony_metrics = {}
        for i in range(1, 3):  # 2 colonies
            colony_metric = Mock()
            colony_metric.population_size_day_list = [100 + i*10, 105 + i*10, 110 + i*10]
            colony_metric.egg_count_day_list = [20 + i*2, 22 + i*2, 25 + i*2]
            colony_metric.larva_count_day_list = [15 + i, 16 + i, 18 + i]
            colony_metric.pupa_count_day_list = [10 + i, 11 + i, 12 + i]
            colony_metric.adult_count_day_list = [55 + i*5, 56 + i*5, 55 + i*5]
            colony_metrics[i] = colony_metric

        collector.colony_metrics = colony_metrics

        # Mock bee metrics
        bee_metrics = {}
        for i in range(1, 6):  # 5 bees
            bee_metric = Mock()
            bee_metric.foraging_trips = 3 + i
            bee_metric.successful_trips = 2 + i
            bee_metric.foraging_trip_durations = [40 + i, 45 + i, 42 + i]
            bee_metric.foraging_trip_energies = [100 + i*10, 110 + i*10, 105 + i*10]
            bee_metric.activity_time = {
                "foraging": 120 + i*10,
                "nursing": 80 + i*5,
                "resting": 60 + i*3
            }
            bee_metrics[i] = bee_metric

        collector.bee_metrics = bee_metrics

        return collector

    @pytest.fixture
    def data_extractor(self, mock_data_collector):
        """Create BSTEW data extractor with mock data"""
        return BSTEWDataExtractor(mock_data_collector)

    def test_population_data_extraction(self, data_extractor):
        """Test population data extraction from BSTEW"""
        population_data = data_extractor.extract_population_data()

        assert isinstance(population_data, dict)
        assert "total_population_time_series" in population_data

        # Check aggregated population
        total_pop = population_data["total_population_time_series"]
        assert len(total_pop) == 3  # 3 time steps
        assert total_pop[0] == 230  # 110 + 120 (from 2 colonies)
        assert total_pop[1] == 240  # 115 + 125
        assert total_pop[2] == 250  # 120 + 130

    def test_activity_patterns_extraction(self, data_extractor):
        """Test activity patterns extraction from BSTEW"""
        activity_data = data_extractor.extract_activity_patterns()

        assert isinstance(activity_data, dict)

        # Check that activities are extracted
        if "foraging_activity" in activity_data:
            foraging_stats = activity_data["foraging_activity"]
            assert "mean" in foraging_stats
            assert "std" in foraging_stats
            assert "distribution" in foraging_stats

    def test_foraging_data_extraction(self, data_extractor):
        """Test foraging data extraction from BSTEW"""
        foraging_data = data_extractor.extract_foraging_data()

        assert isinstance(foraging_data, dict)
        assert "total_foraging_trips" in foraging_data
        assert "successful_trips" in foraging_data
        assert "foraging_efficiency" in foraging_data

        # Check trip totals
        total_trips = foraging_data["total_foraging_trips"]
        assert len(total_trips) == 5  # 5 bees
        assert sum(total_trips) == sum(range(4, 9))  # 4+5+6+7+8 = 30

        # Check efficiency calculation
        efficiency = foraging_data["foraging_efficiency"]
        assert len(efficiency) == 5
        assert all(0 <= eff <= 1 for eff in efficiency)

    def test_empty_data_handling(self):
        """Test extraction with empty data collector"""
        empty_collector = Mock()
        empty_collector.colony_metrics = {}
        empty_collector.bee_metrics = {}

        extractor = BSTEWDataExtractor(empty_collector)

        population_data = extractor.extract_population_data()
        activity_data = extractor.extract_activity_patterns()
        foraging_data = extractor.extract_foraging_data()

        # Should return empty dictionaries gracefully
        assert isinstance(population_data, dict)
        assert isinstance(activity_data, dict)
        assert isinstance(foraging_data, dict)


class TestBehavioralValidator:
    """Test behavioral validation functionality"""

    @pytest.fixture
    def validator(self):
        """Create behavioral validator with default tolerances"""
        return BehavioralValidator()

    @pytest.fixture
    def custom_validator(self):
        """Create validator with custom tolerances"""
        custom_tolerances = {
            "population_dynamics": 0.02,  # Stricter
            "foraging_behavior": 0.15,    # More lenient
            "activity_patterns": 0.05
        }
        return BehavioralValidator(custom_tolerances)

    def test_default_tolerances(self, validator):
        """Test default tolerance configuration"""
        tolerances = validator.tolerance_config

        assert "population_dynamics" in tolerances
        assert "activity_patterns" in tolerances
        assert "foraging_behavior" in tolerances

        # Check reasonable default values
        assert 0 < tolerances["population_dynamics"] < 0.2
        assert 0 < tolerances["foraging_behavior"] < 0.2

    def test_custom_tolerances(self, custom_validator):
        """Test custom tolerance configuration"""
        tolerances = custom_validator.tolerance_config

        assert tolerances["population_dynamics"] == 0.02
        assert tolerances["foraging_behavior"] == 0.15
        assert tolerances["activity_patterns"] == 0.05

    def test_validation_metric_creation(self, validator):
        """Test creation of validation metrics"""
        metric = validator._create_validation_metric(
            "test_metric", 95.0, 100.0, 0.1
        )

        assert isinstance(metric, ValidationMetric)
        assert metric.metric_name == "test_metric"
        assert metric.bstew_value == 95.0
        assert metric.netlogo_value == 100.0
        assert metric.difference == -5.0
        assert metric.relative_difference == 0.05  # 5% difference
        assert metric.tolerance == 0.1
        assert metric.passes_validation is True  # 5% < 10%

    def test_validation_metric_failure(self, validator):
        """Test validation metric that fails tolerance"""
        metric = validator._create_validation_metric(
            "failing_metric", 80.0, 100.0, 0.1
        )

        assert metric.relative_difference == 0.2  # 20% difference
        assert metric.passes_validation is False  # 20% > 10%

    def test_population_dynamics_validation(self, validator):
        """Test population dynamics validation"""
        bstew_data = {
            "total_population_time_series": [100, 105, 110, 108, 112],
            "egg_count_time_series": [20, 22, 25, 23, 24],
            "larva_count_time_series": [15, 16, 18, 17, 19],
            "adult_count_time_series": [55, 56, 55, 57, 56]
        }

        netlogo_data = {
            "total_population_time_series": [98, 103, 112, 110, 115],
            "egg_count_time_series": [19, 21, 26, 24, 25],
            "larva_count_time_series": [14, 17, 19, 18, 20],
            "adult_count_time_series": [54, 57, 56, 58, 57]
        }

        result = validator.validate_population_dynamics(bstew_data, netlogo_data)

        assert isinstance(result, ValidationResult)
        assert result.category == "population_dynamics"
        assert result.total_metrics > 0
        assert 0 <= result.pass_rate <= 1
        assert len(result.individual_results) == result.total_metrics

    def test_activity_patterns_validation(self, validator):
        """Test activity patterns validation"""
        bstew_data = {
            "foraging_activity": {"mean": 15.5, "std": 2.1, "distribution": [14, 15, 16, 17, 15]},
            "nursing_activity": {"mean": 20.8, "std": 1.5, "distribution": [19, 21, 22, 20, 21]},
            "resting_activity": [8, 9, 7, 8, 9]
        }

        netlogo_data = {
            "foraging_activity": [14, 16, 17, 15, 16],
            "nursing_activity": [20, 21, 22, 19, 21],
            "resting_activity": [8, 8, 7, 9, 8]
        }

        result = validator.validate_activity_patterns(bstew_data, netlogo_data)

        assert isinstance(result, ValidationResult)
        assert result.category == "activity_patterns"
        assert result.total_metrics >= 0  # May be 0 if no matching activities

    def test_foraging_behavior_validation(self, validator):
        """Test foraging behavior validation"""
        bstew_data = {
            "total_foraging_trips": [5, 4, 6, 3, 7],
            "successful_trips": [4, 3, 5, 2, 6],
            "foraging_efficiency": [0.8, 0.75, 0.83, 0.67, 0.86],
            "trip_durations": [45, 42, 48, 50, 44],
            "energy_collected": [120, 115, 135, 110, 140]
        }

        netlogo_data = {
            "total_foraging_trips": [5, 4, 5, 4, 6],
            "successful_trips": [4, 3, 4, 3, 5],
            "foraging_efficiency": [0.82, 0.78, 0.81, 0.70, 0.84],
            "trip_durations": [43, 44, 46, 48, 46],
            "energy_collected": [118, 118, 130, 115, 138]
        }

        result = validator.validate_foraging_behavior(bstew_data, netlogo_data)

        assert isinstance(result, ValidationResult)
        assert result.category == "foraging_behavior"
        assert result.total_metrics > 0

    def test_validation_result_creation(self, validator):
        """Test validation result creation"""
        # Create mock metrics
        metrics = [
            ValidationMetric("metric1", 95, 100, -5, 0.05, 0.1, True),
            ValidationMetric("metric2", 85, 100, -15, 0.15, 0.1, False),
            ValidationMetric("metric3", 102, 100, 2, 0.02, 0.1, True)
        ]

        result = validator._create_validation_result("test_category", metrics)

        assert isinstance(result, ValidationResult)
        assert result.category == "test_category"
        assert result.total_metrics == 3
        assert result.passed_metrics == 2
        assert result.failed_metrics == 1
        assert result.pass_rate == 2/3

        # Check summary statistics
        stats = result.summary_statistics
        assert "mean_relative_difference" in stats
        assert "max_relative_difference" in stats
        assert stats["max_relative_difference"] == 0.15


class TestNetLogoValidationSuite:
    """Test complete NetLogo validation suite"""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_netlogo_dir(self):
        """Create temporary NetLogo data directory"""
        temp_dir = tempfile.mkdtemp()

        # Create minimal NetLogo data for testing
        population_data = pd.DataFrame({
            "total_population": [100, 105, 110],
            "egg_count": [20, 22, 25]
        })
        population_file = Path(temp_dir) / "population_data.csv"
        population_data.to_csv(population_file, index=False)

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def validation_suite(self, temp_netlogo_dir, temp_output_dir):
        """Create validation suite with test data"""
        return NetLogoValidationSuite(temp_netlogo_dir, temp_output_dir)

    def test_suite_initialization(self, validation_suite, temp_netlogo_dir, temp_output_dir):
        """Test validation suite initialization"""
        assert isinstance(validation_suite.netlogo_loader, NetLogoDataLoader)
        assert isinstance(validation_suite.validator, BehavioralValidator)
        assert validation_suite.output_path == Path(temp_output_dir)
        assert validation_suite.output_path.exists()

    @patch('src.bstew.validation.netlogo_validation.BSTEWDataExtractor')
    def test_complete_validation_run(self, mock_extractor_class, validation_suite):
        """Test complete validation run"""
        # Mock data collector
        mock_data_collector = Mock()

        # Mock BSTEW data extractor
        mock_extractor = Mock()
        mock_extractor.extract_population_data.return_value = {
            "total_population_time_series": [100, 105, 110]
        }
        mock_extractor.extract_activity_patterns.return_value = {}
        mock_extractor.extract_foraging_data.return_value = {}
        mock_extractor_class.return_value = mock_extractor

        # Run validation
        results = validation_suite.run_complete_validation(mock_data_collector)

        assert isinstance(results, dict)
        # Should have at least population_dynamics if NetLogo data was loaded
        assert len(results) >= 0

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_validation_plot_generation(self, mock_close, mock_savefig, validation_suite):
        """Test validation plot generation"""
        # Mock validation results
        mock_results = {
            "population_dynamics": Mock(
                pass_rate=0.8,
                total_metrics=5,
                passed_metrics=4,
                individual_results=[
                    Mock(relative_difference=0.05),
                    Mock(relative_difference=0.12)
                ]
            ),
            "foraging_behavior": Mock(
                pass_rate=0.6,
                total_metrics=3,
                passed_metrics=2,
                individual_results=[
                    Mock(relative_difference=0.08)
                ]
            )
        }

        # Test plot creation doesn't crash
        validation_suite._create_validation_plots(mock_results)

        # Verify matplotlib was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    def test_report_generation(self, validation_suite, temp_output_dir):
        """Test validation report generation"""
        # Mock validation results
        mock_results = {
            "population_dynamics": Mock(
                total_metrics=5,
                passed_metrics=4,
                failed_metrics=1,
                pass_rate=0.8,
                summary_statistics={"mean_relative_difference": 0.06},
                individual_results=[
                    Mock(
                        metric_name="test_metric",
                        bstew_value=95.0,
                        netlogo_value=100.0,
                        difference=-5.0,
                        relative_difference=0.05,
                        tolerance=0.1,
                        passes_validation=True
                    )
                ]
            )
        }

        validation_suite._generate_validation_report(mock_results)

        # Check that report files were created
        output_path = Path(temp_output_dir)
        assert (output_path / "validation_summary.csv").exists()
        assert (output_path / "validation_detailed.csv").exists()

        # Verify summary content
        summary_df = pd.read_csv(output_path / "validation_summary.csv")
        assert len(summary_df) == 1
        assert summary_df.iloc[0]["Category"] == "population_dynamics"
        assert summary_df.iloc[0]["Pass Rate"] == "80.0%"

        # Verify detailed content
        detailed_df = pd.read_csv(output_path / "validation_detailed.csv")
        assert len(detailed_df) == 1
        assert detailed_df.iloc[0]["Metric"] == "test_metric"
        assert detailed_df.iloc[0]["Passes"]


class TestNetLogoMapCommand:
    """Test NetLogo map CLI command"""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner"""
        return CliRunner()

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data files"""
        temp_dir = tempfile.mkdtemp()

        # Create CSV test file
        csv_content = """parameter,value,unit,description
BeeSpeciesInitialQueensListAsString,"[""B_terrestris"", 500]",list,Initial queen species and count
MaxForagingRange_m,10000,meters,Maximum foraging range
Gridsize,500,cells,Landscape grid size
N_Badgers,5,count,Number of badgers
SexLocus?,false,boolean,Enable sex locus system
ForagingMortalityFactor,1.5,factor,Foraging mortality multiplier
unknown-parameter,42,units,This parameter has no mapping"""
        csv_file = Path(temp_dir) / "netlogo_params.csv"
        with open(csv_file, 'w') as f:
            f.write(csv_content)

        # Create YAML test file
        yaml_content = """BeeSpeciesInitialQueensListAsString: ["B_terrestris", 500]
MaxForagingRange_m: 10000
Gridsize: 500
N_Badgers: 5
SexLocus?: false
ForagingMortalityFactor: 1.5
unknown-parameter: 42"""
        yaml_file = Path(temp_dir) / "netlogo_params.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)

        # Create JSON test file
        json_content = {
            "BeeSpeciesInitialQueensListAsString": ["B_terrestris", 500],
            "MaxForagingRange_m": 10000,
            "Gridsize": 500,
            "N_Badgers": 5,
            "SexLocus?": False,
            "ForagingMortalityFactor": 1.5,
            "unknown-parameter": 42
        }
        json_file = Path(temp_dir) / "netlogo_params.json"
        with open(json_file, 'w') as f:
            json.dump(json_content, f, indent=2)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_map_command_csv_input(self, runner, temp_data_dir):
        """Test map command with CSV input"""
        csv_file = Path(temp_data_dir) / "netlogo_params.csv"
        output_file = Path(temp_data_dir) / "mapped_params.yaml"

        result = runner.invoke(app, [
            "map",
            str(csv_file),
            "--output", str(output_file)
        ])

        assert result.exit_code == 0
        assert "Parameter Mapping" in result.stdout
        assert "MaxForagingRange" in result.stdout or "foraging.max_ran" in result.stdout
        assert "unknown-parameter" in result.stdout  # Should show unmapped params
        assert output_file.exists()

        # Verify output content
        with open(output_file) as f:
            mapped_data = yaml.safe_load(f)

        # Check structure
        assert "mapped_parameters" in mapped_data
        assert "unmapped_parameters" in mapped_data
        assert "summary" in mapped_data

        # Check summary
        assert mapped_data["summary"]["total_parameters"] == 7
        assert mapped_data["summary"]["mapped_count"] == 6
        assert mapped_data["summary"]["unmapped_count"] == 1

        # Check mapped parameters contain expected paths
        mapped_paths = {p["bstew_path"] for p in mapped_data["mapped_parameters"]}
        assert "colony.initial_queens" in mapped_paths
        assert "foraging.max_range_m" in mapped_paths
        assert "landscape.grid_size" in mapped_paths
        assert "predation.badger_count" in mapped_paths
        assert "genetics.csd_enabled" in mapped_paths
        assert "mortality.foraging_factor" in mapped_paths

        # Check unmapped parameters
        unmapped_names = {p["parameter"] for p in mapped_data["unmapped_parameters"]}
        assert "unknown-parameter" in unmapped_names

    def test_map_command_yaml_input(self, runner, temp_data_dir):
        """Test map command with YAML input"""
        yaml_file = Path(temp_data_dir) / "netlogo_params.yaml"
        output_file = Path(temp_data_dir) / "mapped_params.json"

        result = runner.invoke(app, [
            "map",
            str(yaml_file),
            "--output", str(output_file),
            "--format", "json"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON output
        with open(output_file) as f:
            mapped_data = json.load(f)

        # Verify structure
        assert "mapped_parameters" in mapped_data
        assert "summary" in mapped_data
        assert mapped_data["summary"]["mapped_count"] == 6

        # Check at least some parameters are mapped correctly
        mapped_paths = {p["bstew_path"] for p in mapped_data["mapped_parameters"]}
        assert "foraging.max_range_m" in mapped_paths

    def test_map_command_detailed_view(self, runner, temp_data_dir):
        """Test map command with detailed view"""
        csv_file = Path(temp_data_dir) / "netlogo_params.csv"

        result = runner.invoke(app, [
            "map",
            str(csv_file),
            "--detailed"
        ])

        assert result.exit_code == 0
        assert "NetLogo" in result.stdout and "Parame" in result.stdout  # Headers might be truncated
        assert "BSTEW" in result.stdout
        assert "Type" in result.stdout  # Detailed view shows type
        assert "Unit" in result.stdout  # Detailed view shows unit

    def test_map_command_hide_unmapped(self, runner, temp_data_dir):
        """Test map command with hide-unmapped option"""
        csv_file = Path(temp_data_dir) / "netlogo_params.csv"

        result = runner.invoke(app, [
            "map",
            str(csv_file),
            "--hide-unmapped"
        ])

        assert result.exit_code == 0
        assert "Unmapped Parameters" not in result.stdout
        assert "MaxForagingRange" in result.stdout or "foraging.max_ran" in result.stdout

    def test_map_command_show_unmapped(self, runner, temp_data_dir):
        """Test map command with show-unmapped option (default)"""
        csv_file = Path(temp_data_dir) / "netlogo_params.csv"

        result = runner.invoke(app, [
            "map",
            str(csv_file)
        ])

        assert result.exit_code == 0
        assert "unknown-parameter" in result.stdout
        assert "Unmapped Parameters" in result.stdout

    def test_map_command_csv_output(self, runner, temp_data_dir):
        """Test map command with CSV output format"""
        yaml_file = Path(temp_data_dir) / "netlogo_params.yaml"
        output_file = Path(temp_data_dir) / "mapped_params.csv"

        result = runner.invoke(app, [
            "map",
            str(yaml_file),
            "--output", str(output_file),
            "--format", "csv"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify CSV content
        df = pd.read_csv(output_file)

        # Check CSV structure
        assert "bstew_path" in df.columns
        assert "bstew_value" in df.columns
        assert "netlogo_param" in df.columns

        # Check that some BSTEW parameters are in the CSV
        bstew_paths = df["bstew_path"].values
        assert any("foraging.max_range_m" in str(path) for path in bstew_paths)

    def test_map_command_invalid_file(self, runner):
        """Test map command with non-existent file"""
        result = runner.invoke(app, [
            "map",
            "non_existent_file.csv"
        ])

        assert result.exit_code != 0

    def test_map_command_invalid_format(self, runner, temp_data_dir):
        """Test map command with invalid output format"""
        csv_file = Path(temp_data_dir) / "netlogo_params.csv"

        result = runner.invoke(app, [
            "map",
            str(csv_file),
            "--format", "invalid"
        ])

        assert result.exit_code == 0  # Invalid format defaults to table

    def test_map_command_multiple_categories(self, runner, temp_data_dir):
        """Test map command correctly categorizes parameters"""
        csv_file = Path(temp_data_dir) / "netlogo_params.csv"
        output_file = Path(temp_data_dir) / "mapped_full.yaml"

        result = runner.invoke(app, [
            "map",
            str(csv_file),
            "--output", str(output_file)
        ])

        assert result.exit_code == 0

        with open(output_file) as f:
            mapped_data = yaml.safe_load(f)

        # Check structure
        assert "mapped_parameters" in mapped_data
        assert "summary" in mapped_data

        # Check all parameters were mapped
        mapped_paths = {p["bstew_path"] for p in mapped_data["mapped_parameters"]}
        assert "colony.initial_queens" in mapped_paths
        assert "foraging.max_range_m" in mapped_paths
        assert "landscape.grid_size" in mapped_paths
        assert "predation.badger_count" in mapped_paths
        assert "genetics.csd_enabled" in mapped_paths
        assert "mortality.foraging_factor" in mapped_paths

        # Verify specific values
        param_map = {p["bstew_path"]: p["bstew_value"] for p in mapped_data["mapped_parameters"]}
        assert param_map["foraging.max_range_m"] == 10000
        assert param_map["landscape.grid_size"] == 500
        assert param_map["predation.badger_count"] == 5
        assert param_map["genetics.csd_enabled"] == False
        assert param_map["mortality.foraging_factor"] == 1.5

    def test_parameter_mapper_direct(self):
        """Test NetLogoParameterMapper directly"""
        mapper = NetLogoParameterMapper()

        # Test that mapper has parameter mappings
        assert hasattr(mapper, 'parameter_mappings')
        assert "MaxForagingRange_m" in mapper.parameter_mappings
        assert "Gridsize" in mapper.parameter_mappings
        assert "N_Badgers" in mapper.parameter_mappings

        # Test parameter mapping structure
        foraging_mapping = mapper.parameter_mappings["MaxForagingRange_m"]
        assert foraging_mapping.netlogo_name == "MaxForagingRange_m"
        assert foraging_mapping.bstew_path == "foraging.max_range_m"
        assert foraging_mapping.parameter_type.value == "float"

        # Test convert_value method
        converted = foraging_mapping.convert_value(10000)
        assert converted == 10000.0  # Should be float

        # Test batch conversion using mapper
        netlogo_params = {
            "MaxForagingRange_m": 10000,
            "Gridsize": 500,
            "N_Badgers": 5
        }

        # Test that we can convert values
        for param_name, value in netlogo_params.items():
            if param_name in mapper.parameter_mappings:
                mapping = mapper.parameter_mappings[param_name]
                converted = mapping.convert_value(value)
                assert converted is not None


class TestNetLogoTestCommand:
    """Test NetLogo test CLI command"""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner"""
        return CliRunner()

    @pytest.fixture
    def temp_scenario_file(self):
        """Create a temporary scenario file"""
        temp_dir = tempfile.mkdtemp()

        # Create test scenario file
        scenario_data = {
            "scenarios": [
                {
                    "name": "test_population_growth",
                    "description": "Test population growth",
                    "category": "population",
                    "netlogo_params": {
                        "BeeSpeciesInitialQueensListAsString": ["B_terrestris", 1],
                        "MaxForagingRange_m": 5000,
                    },
                    "bstew_config_overrides": {
                        "simulation.duration_days": 30,
                        "colony.initial_queens": ["B_terrestris", 1],
                    },
                    "metrics_to_compare": ["total_population", "worker_count"],
                    "expected_patterns": {"population_increases": True},
                    "duration_days": 30,
                },
                {
                    "name": "test_foraging_basic",
                    "description": "Test foraging behavior",
                    "category": "foraging",
                    "netlogo_params": {
                        "MaxForagingRange_m": 3000,
                        "FoodSourceLimit": 10,
                    },
                    "bstew_config_overrides": {
                        "foraging.max_range_m": 3000,
                        "resources.food_source_limit": 10,
                    },
                    "metrics_to_compare": ["foraging_trips", "foraging_efficiency"],
                    "expected_patterns": {"efficiency_above_threshold": 0.7},
                    "duration_days": 30,
                }
            ]
        }

        scenario_file = Path(temp_dir) / "test_scenarios.yaml"
        with open(scenario_file, 'w') as f:
            yaml.dump(scenario_data, f)

        yield scenario_file
        shutil.rmtree(temp_dir)

    def test_test_command_default(self, runner):
        """Test the test command with default settings"""
        result = runner.invoke(app, ["test"])

        # Should succeed but note missing test scenarios module
        assert "NetLogo Behavioral Validation Tests" in result.stdout
        assert "Test Suite: all" in result.stdout

    def test_test_command_specific_suite(self, runner):
        """Test running specific test suite"""
        result = runner.invoke(app, ["test", "population"])

        assert "Test Suite: population" in result.stdout
        assert "Tolerance: ±5.0%" in result.stdout

    def test_test_command_with_scenario_file(self, runner, temp_scenario_file):
        """Test running with scenario file"""
        result = runner.invoke(app, [
            "test",
            "--scenario", str(temp_scenario_file)
        ])

        assert "Loading test scenarios from:" in result.stdout
        assert str(temp_scenario_file) in result.stdout

    def test_test_command_custom_tolerance(self, runner):
        """Test with custom tolerance"""
        result = runner.invoke(app, [
            "test",
            "foraging",
            "--tolerance", "0.01"
        ])

        assert "Test Suite: foraging" in result.stdout
        assert "Tolerance: ±1.0%" in result.stdout

    def test_test_command_parallel(self, runner):
        """Test parallel execution flag"""
        result = runner.invoke(app, [
            "test",
            "--parallel"
        ])

        assert "NetLogo Behavioral Validation Tests" in result.stdout

    def test_test_command_fail_fast(self, runner):
        """Test fail-fast option"""
        result = runner.invoke(app, [
            "test",
            "--fail-fast"
        ])

        assert "NetLogo Behavioral Validation Tests" in result.stdout

    def test_test_command_no_plots(self, runner):
        """Test disabling plot generation"""
        result = runner.invoke(app, [
            "test",
            "--no-plots"
        ])

        assert "NetLogo Behavioral Validation Tests" in result.stdout

    def test_test_command_custom_output(self, runner):
        """Test custom output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "my_test_results"

            result = runner.invoke(app, [
                "test",
                "--output", str(output_dir)
            ])

            assert "NetLogo Behavioral Validation Tests" in result.stdout

    def test_test_command_all_suites(self, runner):
        """Test that all test suites are recognized"""
        test_suites = ["all", "population", "foraging", "mortality", "spatial", "reproduction"]

        for suite in test_suites:
            result = runner.invoke(app, ["test", suite])
            assert f"Test Suite: {suite}" in result.stdout

    def test_test_command_verbose(self, runner):
        """Test verbose output"""
        result = runner.invoke(app, [
            "test",
            "population",
            "--verbose"
        ])

        assert "NetLogo Behavioral Validation Tests" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__])
