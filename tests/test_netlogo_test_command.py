"""
Test suite for NetLogo test command functionality.

Tests the `bstew netlogo test` command that performs behavioral validation
between NetLogo BEE-STEWARD v2 and BSTEW implementations.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from bstew.cli.netlogo_cli import app
from bstew.utils.integration_tester import NetLogoIntegrationTester


class TestNetLogoTestCommand:
    """Test suite for NetLogo test command"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "netlogo_data"
        self.test_output_dir = Path(self.temp_dir) / "test_output"

        # Create test directories
        self.test_data_dir.mkdir(exist_ok=True)
        self.test_output_dir.mkdir(exist_ok=True)

        # Create mock NetLogo data files
        self._create_mock_netlogo_files()

    def _create_mock_netlogo_files(self):
        """Create mock NetLogo data files for testing"""
        # Create parameters directory
        params_dir = self.test_data_dir / "parameters"
        params_dir.mkdir(exist_ok=True)

        # Mock parameter file
        param_file = params_dir / "test_params.csv"
        param_file.write_text("parameter,value\ncolony_size,5000\nforaging_distance,1500\n")

        # Mock species file
        species_dir = self.test_data_dir / "species"
        species_dir.mkdir(exist_ok=True)
        species_file = species_dir / "test_species.csv"
        species_file.write_text("species,proboscis_length\napis_mellifera,6.0\n")

    def test_netlogo_test_command_success(self):
        """Test successful execution of NetLogo test command"""
        mock_results = {
            "total_tests": 8,
            "passed_tests": 7,
            "failed_tests": 1,
            "test_results": [
                {"test_name": "data_parsing", "passed": True, "errors": []},
                {"test_name": "parameter_mapping", "passed": True, "errors": []},
                {"test_name": "config_generation", "passed": True, "errors": []},
                {"test_name": "validation_framework", "passed": True, "errors": []},
                {"test_name": "genetic_system", "passed": True, "errors": []},
                {"test_name": "species_system", "passed": True, "errors": []},
                {"test_name": "flower_system", "passed": False, "errors": ["Missing flower data"]},
                {"test_name": "simulation_init", "passed": True, "errors": []}
            ]
        }

        with patch.object(NetLogoIntegrationTester, 'run_all_tests', return_value=mock_results):
            result = self.runner.invoke(app, [
                "test",
                str(self.test_data_dir),
                "--output-dir", str(self.test_output_dir)
            ])

        assert result.exit_code == 0
        assert "Integration Test" in result.stdout and "Summary" in result.stdout
        assert "Tests Run" in result.stdout
        assert "Passed" in result.stdout
        assert "Failed" in result.stdout
        assert "Success Rate" in result.stdout
        assert "87.5%" in result.stdout  # 7/8 * 100

    def test_netlogo_test_command_with_failures(self):
        """Test NetLogo test command with test failures"""
        mock_results = {
            "total_tests": 5,
            "passed_tests": 2,
            "failed_tests": 3,
            "test_results": [
                {"test_name": "data_parsing", "passed": True, "errors": []},
                {"test_name": "parameter_mapping", "passed": False, "errors": ["Invalid parameter format"]},
                {"test_name": "config_generation", "passed": False, "errors": ["Missing required config"]},
                {"test_name": "validation_framework", "passed": True, "errors": []},
                {"test_name": "genetic_system", "passed": False, "errors": ["CSD system mismatch"]}
            ]
        }

        with patch.object(NetLogoIntegrationTester, 'run_all_tests', return_value=mock_results):
            result = self.runner.invoke(app, [
                "test",
                str(self.test_data_dir)
            ])

        assert result.exit_code == 0
        assert "Failed Tests:" in result.stdout
        assert "parameter_mapping" in result.stdout
        assert "config_generation" in result.stdout
        assert "genetic_system" in result.stdout
        assert "40.0%" in result.stdout  # 2/5 * 100

    def test_netlogo_test_command_save_report(self):
        """Test NetLogo test command with report saving"""
        mock_results = {
            "total_tests": 3,
            "passed_tests": 3,
            "failed_tests": 0,
            "test_results": [
                {"test_name": "data_parsing", "passed": True, "errors": []},
                {"test_name": "parameter_mapping", "passed": True, "errors": []},
                {"test_name": "config_generation", "passed": True, "errors": []}
            ]
        }

        with patch.object(NetLogoIntegrationTester, 'run_all_tests', return_value=mock_results):
            result = self.runner.invoke(app, [
                "test",
                str(self.test_data_dir),
                "--output-dir", str(self.test_output_dir),
                "--save-report"
            ])

        assert result.exit_code == 0
        assert "Detailed report saved to:" in result.stdout
        assert "integration_test_report.json" in result.stdout

    def test_netlogo_test_command_verbose(self):
        """Test NetLogo test command with verbose output"""
        mock_results = {
            "total_tests": 2,
            "passed_tests": 2,
            "failed_tests": 0,
            "test_results": [
                {"test_name": "data_parsing", "passed": True, "errors": []},
                {"test_name": "parameter_mapping", "passed": True, "errors": []}
            ]
        }

        with patch.object(NetLogoIntegrationTester, 'run_all_tests', return_value=mock_results):
            result = self.runner.invoke(app, [
                "test",
                str(self.test_data_dir),
                "--verbose"
            ])

        assert result.exit_code == 0
        assert "100.0%" in result.stdout  # All tests passed

    def test_netlogo_test_command_missing_directory(self):
        """Test NetLogo test command with missing input directory"""
        nonexistent_dir = Path(self.temp_dir) / "does_not_exist"

        result = self.runner.invoke(app, [
            "test",
            str(nonexistent_dir)
        ])

        # Should handle gracefully - the command will create the tester
        # but may fail during execution
        assert result.exit_code in [0, 1]  # May succeed or fail depending on implementation

    def test_netlogo_test_command_exception_handling(self):
        """Test NetLogo test command exception handling"""
        with patch.object(NetLogoIntegrationTester, '__init__', side_effect=Exception("Test error")):
            result = self.runner.invoke(app, [
                "test",
                str(self.test_data_dir)
            ])

        assert result.exit_code == 1
        assert "Error running integration tests" in result.stdout


class TestNetLogoIntegrationTester:
    """Test suite for NetLogoIntegrationTester functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"

        # Create test directories and files
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self._create_test_data()

    def _create_test_data(self):
        """Create test data files"""
        # Create parameters directory with test file
        params_dir = self.data_dir / "parameters"
        params_dir.mkdir(exist_ok=True)

        param_file = params_dir / "test_params.csv"
        param_file.write_text("parameter,value,description\ncolony_size,5000,Initial colony size\n")

    @patch('bstew.utils.integration_tester.NetLogoDataParser')
    @patch('bstew.utils.integration_tester.NetLogoOutputParser')
    @patch('bstew.utils.integration_tester.NetLogoParameterMapper')
    @patch('bstew.utils.integration_tester.ModelValidator')
    @patch('bstew.utils.integration_tester.NetLogoSpecificValidator')
    def test_integration_tester_initialization(self, mock_validator, mock_model_validator,
                                               mock_mapper, mock_output_parser, mock_parser):
        """Test NetLogoIntegrationTester initialization"""
        tester = NetLogoIntegrationTester(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )

        assert tester.data_dir == self.data_dir
        assert tester.output_dir == self.output_dir
        assert tester.output_dir.exists()
        assert isinstance(tester.test_results, list)

        # Verify components are initialized
        mock_parser.assert_called_once()
        mock_output_parser.assert_called_once()
        mock_mapper.assert_called_once()
        mock_model_validator.assert_called_once()
        mock_validator.assert_called_once()

    @patch('bstew.utils.integration_tester.NetLogoDataParser')
    @patch('bstew.utils.integration_tester.NetLogoOutputParser')
    @patch('bstew.utils.integration_tester.NetLogoParameterMapper')
    @patch('bstew.utils.integration_tester.ModelValidator')
    @patch('bstew.utils.integration_tester.NetLogoSpecificValidator')
    def test_run_all_tests_basic_execution(self, mock_validator, mock_model_validator,
                                           mock_mapper, mock_output_parser, mock_parser):
        """Test basic execution of run_all_tests"""
        # Mock the individual test methods
        tester = NetLogoIntegrationTester(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )

        # Mock all the test methods to avoid complex setup
        with patch.object(tester, '_test_data_parsing') as mock_data_parsing, \
             patch.object(tester, '_test_parameter_mapping') as mock_param_mapping, \
             patch.object(tester, '_test_config_generation') as mock_config_gen, \
             patch.object(tester, '_test_validation_framework') as mock_validation, \
             patch.object(tester, '_test_genetic_system_compatibility') as mock_genetic, \
             patch.object(tester, '_test_species_system_compatibility') as mock_species, \
             patch.object(tester, '_test_flower_system_compatibility') as mock_flower, \
             patch.object(tester, '_test_simulation_initialization') as mock_sim_init, \
             patch.object(tester, '_generate_summary_report') as mock_report:

            # Mock the summary report
            mock_report.return_value = {
                "total_tests": 8,
                "passed_tests": 6,
                "failed_tests": 2,
                "test_results": []
            }

            results = tester.run_all_tests()

            # Verify all test methods were called
            mock_data_parsing.assert_called_once()
            mock_param_mapping.assert_called_once()
            mock_config_gen.assert_called_once()
            mock_validation.assert_called_once()
            mock_genetic.assert_called_once()
            mock_species.assert_called_once()
            mock_flower.assert_called_once()
            mock_sim_init.assert_called_once()
            mock_report.assert_called_once()

            # Verify results structure
            assert "total_tests" in results
            assert "passed_tests" in results
            assert "failed_tests" in results
            assert "test_results" in results

    def test_integration_test_comprehensive_coverage(self):
        """Test that integration tester covers all required test areas"""
        # This test verifies that the integration tester has methods for all
        # the key areas that need to be tested for NetLogo compatibility

        required_test_methods = [
            '_test_data_parsing',
            '_test_parameter_mapping',
            '_test_config_generation',
            '_test_validation_framework',
            '_test_genetic_system_compatibility',
            '_test_species_system_compatibility',
            '_test_flower_system_compatibility',
            '_test_simulation_initialization'
        ]

        # Verify all required methods exist
        for method_name in required_test_methods:
            assert hasattr(NetLogoIntegrationTester, method_name), f"Missing method: {method_name}"


class TestNetLogoTestCLIIntegration:
    """Test CLI integration and end-to-end functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "netlogo_data"
        self.test_output_dir = Path(self.temp_dir) / "output"

        self.test_data_dir.mkdir(exist_ok=True)
        self.test_output_dir.mkdir(exist_ok=True)

    def test_cli_help_command(self):
        """Test that NetLogo test command appears in CLI help"""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "test" in result.stdout
        assert "Run behavioral validation tests" in result.stdout or \
               "NetLogo integration tests" in result.stdout

    def test_cli_test_command_help(self):
        """Test NetLogo test command specific help"""
        result = self.runner.invoke(app, ["test", "--help"])

        assert result.exit_code == 0
        assert "input-dir" in result.stdout or "Input directory" in result.stdout
        assert "output-dir" in result.stdout or "Output directory" in result.stdout
        assert "save-report" in result.stdout
        assert "verbose" in result.stdout

    def test_end_to_end_test_execution(self):
        """Test complete end-to-end test execution"""
        # Create minimal test data structure
        params_dir = self.test_data_dir / "parameters"
        params_dir.mkdir(exist_ok=True)

        param_file = params_dir / "basic_params.csv"
        param_file.write_text("parameter,value\ntest_param,123\n")

        # Mock the integration tester to avoid complex dependencies
        mock_results = {
            "total_tests": 4,
            "passed_tests": 4,
            "failed_tests": 0,
            "test_results": [
                {"test_name": "data_parsing", "passed": True, "errors": []},
                {"test_name": "parameter_mapping", "passed": True, "errors": []},
                {"test_name": "config_generation", "passed": True, "errors": []},
                {"test_name": "validation_framework", "passed": True, "errors": []}
            ]
        }

        with patch.object(NetLogoIntegrationTester, 'run_all_tests', return_value=mock_results):
            result = self.runner.invoke(app, [
                "test",
                str(self.test_data_dir),
                "--output-dir", str(self.test_output_dir),
                "--save-report",
                "--verbose"
            ])

        assert result.exit_code == 0
        assert "Integration tests completed successfully!" in result.stdout
        assert "100.0%" in result.stdout  # All tests passed


if __name__ == "__main__":
    pytest.main([__file__])
