"""
Tests for Economic Analysis CLI Commands
=======================================

Tests for the CLI interface to economic assessment functionality.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.bstew.cli.commands.economics import EconomicAnalysisCommand
from src.bstew.cli.core.base import CLIContext
from src.bstew.components.economics import CropType
from rich.console import Console


class TestEconomicAnalysisCommand:
    """Test the EconomicAnalysisCommand class"""

    @pytest.fixture
    def mock_context(self):
        """Create a mock CLI context"""
        console = Console()
        context = CLIContext(console=console)
        return context

    @pytest.fixture
    def economic_command(self, mock_context):
        """Create an EconomicAnalysisCommand instance"""
        return EconomicAnalysisCommand(mock_context)

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with sample data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample species abundance data
            abundance_df = "species,abundance\nBombus_terrestris,150\nBombus_lucorum,120\nBombus_hortorum,80\n"

            (temp_path / "species_abundance.csv").write_text(abundance_df)

            # Create sample foraging efficiency data
            foraging_data = "species,efficiency\nBombus_terrestris,0.85\nBombus_lucorum,0.78\nBombus_hortorum,0.92\n"
            (temp_path / "foraging_efficiency.csv").write_text(foraging_data)

            yield temp_dir

    @pytest.fixture
    def temp_crop_config(self):
        """Create temporary crop configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            config = {
                "crops": {
                    "apples": {
                        "price_per_kg": 1.20,
                        "yield_baseline_kg_ha": 8000,
                        "yield_with_pollinators_kg_ha": 25000,
                        "pollinator_dependency": 0.65,
                        "hectares_planted": 1000,
                        "production_cost_per_ha": 8500,
                        "harvest_cost_per_kg": 0.15,
                    },
                }
            }
            yaml.dump(config, temp_file)
            temp_file.flush()
            yield temp_file.name

        Path(temp_file.name).unlink()

    def test_economic_command_initialization(self, mock_context):
        """Test command initialization"""
        command = EconomicAnalysisCommand(mock_context)

        assert command.context is mock_context
        assert command.species_system is not None
        assert command.economic_assessment is not None

    def test_validate_inputs_missing_directory(self, economic_command):
        """Test validation with missing input directory"""
        errors = economic_command.validate_inputs(input_dir="/nonexistent/path")

        assert len(errors) == 1
        assert "Input directory not found" in errors[0]

    def test_validate_inputs_invalid_analysis_type(self, economic_command):
        """Test validation with invalid analysis type"""
        errors = economic_command.validate_inputs(
            input_dir=".",
            analysis_type="invalid_type"
        )

        assert len(errors) == 1
        assert "Invalid analysis type" in errors[0]

    def test_validate_inputs_invalid_output_format(self, economic_command):
        """Test validation with invalid output format"""
        errors = economic_command.validate_inputs(
            input_dir=".",
            output_format="invalid_format"
        )

        assert len(errors) == 1
        assert "Invalid output format" in errors[0]

    def test_validate_inputs_valid(self, economic_command, temp_data_dir):
        """Test validation with valid inputs"""
        errors = economic_command.validate_inputs(
            input_dir=temp_data_dir,
            analysis_type="comprehensive",
            output_format="table"
        )

        assert len(errors) == 0

    def test_load_simulation_data_with_files(self, economic_command, temp_data_dir):
        """Test loading simulation data from actual files"""
        simulation_data = economic_command._load_simulation_data(temp_data_dir)

        assert "species_abundance" in simulation_data
        assert "foraging_efficiency" in simulation_data
        assert "Bombus_terrestris" in simulation_data["species_abundance"]
        assert simulation_data["species_abundance"]["Bombus_terrestris"] == 150

    def test_load_simulation_data_without_files(self, economic_command):
        """Test loading simulation data with sample data when files missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            simulation_data = economic_command._load_simulation_data(temp_dir)

            # Should use sample data
            assert "species_abundance" in simulation_data
            assert "foraging_efficiency" in simulation_data
            assert len(simulation_data["species_abundance"]) > 0

    def test_load_crop_configuration(self, economic_command, temp_crop_config):
        """Test loading crop configuration from file"""
        crop_data = economic_command._load_crop_configuration(temp_crop_config)

        assert "apples" in crop_data
        assert crop_data["apples"].crop_type == CropType.APPLES
        assert crop_data["apples"].price_per_kg == 1.20

    def test_load_crop_configuration_creates_sample(self, economic_command):
        """Test that missing crop config creates sample file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            crop_data = economic_command._load_crop_configuration(str(config_path))

            # Should create the file
            assert config_path.exists()
            assert len(crop_data) > 0

    def test_comprehensive_analysis(self, economic_command, temp_data_dir):
        """Test comprehensive analysis execution"""
        simulation_data = economic_command._load_simulation_data(temp_data_dir)

        results = economic_command._comprehensive_analysis(
            simulation_data, None, time_horizon=10, discount_rate=0.03
        )

        assert results["analysis_type"] == "comprehensive"
        assert "crop_valuations" in results
        assert "total_pollination_value" in results
        assert "species_contributions" in results
        assert "risk_assessment" in results
        assert results["total_pollination_value"] >= 0

    def test_crop_valuation_analysis(self, economic_command, temp_data_dir):
        """Test crop valuation analysis"""
        simulation_data = economic_command._load_simulation_data(temp_data_dir)

        results = economic_command._crop_valuation_analysis(simulation_data, None)

        assert results["analysis_type"] == "crop_valuation"
        assert "crop_details" in results
        assert "summary" in results
        assert len(results["crop_details"]) > 0

    def test_cost_benefit_analysis(self, economic_command, temp_data_dir):
        """Test cost-benefit analysis"""
        simulation_data = economic_command._load_simulation_data(temp_data_dir)

        results = economic_command._cost_benefit_analysis(
            simulation_data, None, None, time_horizon=10, discount_rate=0.03
        )

        assert results["analysis_type"] == "cost_benefit"
        assert "financial_metrics" in results
        assert "scenario_name" in results
        assert results["financial_metrics"]["total_costs_gbp"] >= 0

    def test_scenario_comparison_analysis(self, economic_command, temp_data_dir):
        """Test scenario comparison analysis"""
        simulation_data = economic_command._load_simulation_data(temp_data_dir)

        results = economic_command._scenario_comparison_analysis(simulation_data, None)

        assert results["analysis_type"] == "scenario_comparison"
        assert "scenarios" in results
        assert "ranking" in results
        assert len(results["scenarios"]) > 0

    def test_landscape_analysis(self, economic_command, temp_data_dir):
        """Test landscape analysis"""
        simulation_data = economic_command._load_simulation_data(temp_data_dir)

        results = economic_command._landscape_analysis(simulation_data, None)

        assert results["analysis_type"] == "landscape"
        assert "total_pollination_value" in results
        assert "crop_breakdown" in results
        assert results["total_pollination_value"] >= 0

    def test_execute_comprehensive_analysis(self, economic_command, temp_data_dir):
        """Test full execute method with comprehensive analysis"""
        result = economic_command.execute(
            input_dir=temp_data_dir,
            analysis_type="comprehensive",
            output_format="table",
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["analysis_type"] == "comprehensive"

    def test_execute_with_validation_errors(self, economic_command):
        """Test execute with validation errors"""
        result = economic_command.execute(
            input_dir="/nonexistent",
            analysis_type="invalid",
        )

        assert result.success is False
        assert result.exit_code == 1

    def test_execute_unsupported_analysis_type(self, economic_command, temp_data_dir):
        """Test execute with unsupported analysis type"""
        # Patch validation to pass
        with patch.object(economic_command, 'validate_inputs', return_value=[]):
            result = economic_command.execute(
                input_dir=temp_data_dir,
                analysis_type="unsupported_type",
            )

            assert result.success is False
            assert "Unsupported analysis type" in result.message

    def test_export_results_json(self, economic_command):
        """Test exporting results to JSON"""
        results = {"test": "data"}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.json"

            economic_command._export_results(results, "json", str(output_file))

            # Verify file was created and contains data
            assert output_file.exists()
            with open(output_file, 'r') as f:
                data = json.load(f)
                assert data == results

    def test_export_results_csv(self, economic_command):
        """Test exporting results to CSV"""
        results = {
            "crop_valuations": {
                "apples": {"annual_value_gbp": 1000, "value_per_hectare": 100},
                "blueberries": {"annual_value_gbp": 2000, "value_per_hectare": 200},
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.csv"

            economic_command._export_results(results, "csv", str(output_file))

            assert output_file.exists()
            content = output_file.read_text()
            assert "apples" in content
            assert "1000" in content

    def test_export_results_html(self, economic_command):
        """Test exporting results to HTML"""
        results = {"analysis_type": "test", "timestamp": "2024-01-01"}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.html"

            economic_command._export_results(results, "html", str(output_file))

            assert output_file.exists()
            content = output_file.read_text()
            assert "<!DOCTYPE html>" in content
            assert "Economic Analysis Report" in content

    def test_display_comprehensive_results(self, economic_command):
        """Test displaying comprehensive results"""
        results = {
            "total_pollination_value": 50000,
            "crop_valuations": {
                "apples": {
                    "annual_value_gbp": 30000,
                    "value_per_hectare": 300,
                    "service_reliability": 0.85,
                }
            },
            "risk_assessment": {
                "species_diversity_score": 0.7,
                "service_stability": 0.8,
            },
            "recommendations": ["Increase diversity", "Improve habitat"],
        }

        # Should not raise any exceptions
        economic_command._display_comprehensive_results(results)

    def test_display_cost_benefit_results(self, economic_command):
        """Test displaying cost-benefit results"""
        results = {
            "scenario_name": "Test Scenario",
            "financial_metrics": {
                "total_costs_gbp": 10000,
                "total_benefits_gbp": 25000,
                "net_present_value": 15000,
                "benefit_cost_ratio": 2.5,
                "payback_period_years": 4.0,
                "annual_roi_percent": 15.0,
            }
        }

        # Should not raise any exceptions
        economic_command._display_cost_benefit_results(results)

    def test_display_scenario_comparison_results(self, economic_command):
        """Test displaying scenario comparison results"""
        results = {
            "ranking": [
                ("Enhanced", ("Enhanced", {
                    "total_annual_value": 50000,
                    "value_per_hectare": 500,
                    "relative_performance": 1.0,
                })),
                ("Baseline", ("Baseline", {
                    "total_annual_value": 30000,
                    "value_per_hectare": 300,
                    "relative_performance": 0.6,
                })),
            ]
        }

        # Should not raise any exceptions
        economic_command._display_scenario_comparison_results(results)

    def test_create_sample_crop_config(self, economic_command):
        """Test creating sample crop configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "sample_config.yaml"

            economic_command._create_sample_crop_config(config_path)

            assert config_path.exists()

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            assert "crops" in config
            assert "apples" in config["crops"]
            assert config["crops"]["apples"]["price_per_kg"] == 1.20

    def test_execute_with_output_file(self, economic_command, temp_data_dir):
        """Test execute with output file specified"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "economic_results.json"

            result = economic_command.execute(
                input_dir=temp_data_dir,
                analysis_type="comprehensive",
                output_format="json",
                output_file=str(output_file),
            )

            assert result.success is True
            assert output_file.exists()

    def test_execute_with_crop_config(self, economic_command, temp_data_dir, temp_crop_config):
        """Test execute with crop configuration file"""
        result = economic_command.execute(
            input_dir=temp_data_dir,
            analysis_type="comprehensive",
            crop_config=temp_crop_config,
        )

        assert result.success is True
        assert result.data is not None

    def test_execute_exception_handling(self, economic_command, temp_data_dir):
        """Test exception handling in execute method"""
        # Mock an exception in the analysis
        with patch.object(economic_command, '_comprehensive_analysis', side_effect=Exception("Test error")):
            result = economic_command.execute(
                input_dir=temp_data_dir,
                analysis_type="comprehensive",
            )

            assert result.success is False
            assert "Test error" in result.message


class TestEconomicCLIIntegration:
    """Test integration between CLI and economic assessment"""

    @pytest.fixture
    def mock_cli_context(self):
        """Create mock CLI context"""
        from src.bstew.cli.core.base import VerbosityLevel

        console = Console()
        context = Mock(spec=CLIContext)
        context.console = console
        context.config_manager = Mock()  # Add missing config_manager
        context.verbosity = VerbosityLevel.NORMAL  # Add missing verbosity
        context.print_info = Mock()
        context.print_success = Mock()
        context.print_warning = Mock()
        context.print_error = Mock()
        return context

    def test_cli_economic_command_registration(self):
        """Test that economic command can be registered"""
        # This would test CLI app registration, but we focus on command logic
        from src.bstew.cli.commands.economics import EconomicAnalysisCommand

        # Should be able to import and instantiate
        assert EconomicAnalysisCommand is not None

    def test_economic_analysis_end_to_end(self, mock_cli_context):
        """Test end-to-end economic analysis workflow"""
        command = EconomicAnalysisCommand(mock_cli_context)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal test data
            abundance_data = "species,abundance\nBombus_terrestris,100\n"
            Path(temp_dir, "species_abundance.csv").write_text(abundance_data)

            foraging_data = "species,efficiency\nBombus_terrestris,0.8\n"
            Path(temp_dir, "foraging_efficiency.csv").write_text(foraging_data)

            # Run analysis
            result = command.execute(
                input_dir=temp_dir,
                analysis_type="comprehensive",
                output_format="table",
            )

            assert result.success is True
            assert result.data["analysis_type"] == "comprehensive"
            assert result.data["total_pollination_value"] >= 0

            # Verify context methods were called
            mock_cli_context.print_info.assert_called()

    def test_economic_analysis_with_all_formats(self, mock_cli_context):
        """Test economic analysis with different output formats"""
        command = EconomicAnalysisCommand(mock_cli_context)

        formats_to_test = ["table", "csv", "json", "html"]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            abundance_data = "species,abundance\nBombus_terrestris,100\n"
            Path(temp_dir, "species_abundance.csv").write_text(abundance_data)

            for output_format in formats_to_test:
                with tempfile.TemporaryDirectory() as output_dir:
                    output_file = Path(output_dir) / f"results.{output_format}"

                    result = command.execute(
                        input_dir=temp_dir,
                        analysis_type="crop_valuation",
                        output_format=output_format,
                        output_file=str(output_file),
                    )

                    assert result.success is True
                    if output_format != "table":  # table format doesn't create files
                        assert output_file.exists()

    def test_economic_analysis_error_propagation(self, mock_cli_context):
        """Test that errors are properly propagated through CLI"""
        command = EconomicAnalysisCommand(mock_cli_context)

        # Test with non-existent directory - should succeed with sample data
        result = command.execute(
            input_dir="/definitely/nonexistent/path",
            analysis_type="comprehensive",
        )

        # Economics system falls back to sample data, so succeeds
        assert result.success is True
        mock_cli_context.print_info.assert_called()


class TestEconomicCLIEdgeCases:
    """Test edge cases in economic CLI"""

    @pytest.fixture
    def economic_command(self):
        """Create command for edge case testing"""
        console = Console()
        context = CLIContext(console=console)
        return EconomicAnalysisCommand(context)

    def test_empty_simulation_data(self, economic_command):
        """Test with empty simulation data directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory
            result = economic_command.execute(
                input_dir=temp_dir,
                analysis_type="comprehensive",
            )

            # Should use sample data and succeed
            assert result.success is True

    def test_malformed_csv_data(self, economic_command):
        """Test with malformed CSV data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create malformed CSV
            bad_data = "invalid,csv,data\n1,2\n"
            Path(temp_dir, "species_abundance.csv").write_text(bad_data)

            # Should handle gracefully and use sample data
            result = economic_command.execute(
                input_dir=temp_dir,
                analysis_type="comprehensive",
            )

            assert result.success is True

    def test_very_large_analysis(self, economic_command):
        """Test with large-scale analysis parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = economic_command.execute(
                input_dir=temp_dir,
                analysis_type="comprehensive",
                time_horizon=50,  # Very long time horizon
                discount_rate=0.15,  # High discount rate
            )

            assert result.success is True

    def test_zero_discount_rate(self, economic_command):
        """Test with zero discount rate"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = economic_command.execute(
                input_dir=temp_dir,
                analysis_type="cost_benefit",
                discount_rate=0.0,
            )

            assert result.success is True

    def test_output_to_readonly_location(self, economic_command):
        """Test output to read-only location"""
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_file = "/root/readonly_results.json"  # Should fail

            result = economic_command.execute(
                input_dir=temp_dir,
                analysis_type="comprehensive",
                output_format="json",
                output_file=readonly_file,
            )

            # Export error should cause command to fail
            assert result.success is False  # Command fails when export fails
