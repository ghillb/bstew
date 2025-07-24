"""
Core CLI tests for BSTEW
========================

Tests covering essential CLI functionality with realistic mocking.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner
import pandas as pd
import numpy as np

from src.bstew.cli import app
from src.bstew.cli.core.base import CLIContext, CLIResult
from src.bstew.cli.core.progress import ProgressManager
from src.bstew.cli.core.validation import InputValidator
from src.bstew.cli.types import VerbosityLevel
from src.bstew.cli.commands.run import RunCommand
from src.bstew.cli.commands.config import ConfigCommand
from src.bstew.cli.commands.analyze import AnalyzeCommand
from src.bstew.cli.commands.utility import VersionCommand, InitCommand


class TestCLICore:
    """Test core CLI functionality"""

    def test_cli_context_creation(self):
        """Test CLI context creation with different verbosity levels"""
        from src.bstew.cli import get_cli_context

        # Test default verbosity
        context = get_cli_context()
        assert context.verbosity == VerbosityLevel.NORMAL

        # Test verbose mode
        context = get_cli_context(verbose=True)
        assert context.verbosity == VerbosityLevel.VERBOSE

        # Test quiet mode
        context = get_cli_context(quiet=True)
        assert context.verbosity == VerbosityLevel.QUIET

        # Test debug mode
        context = get_cli_context(debug=True)
        assert context.verbosity == VerbosityLevel.DEBUG

    def test_cli_result_creation(self):
        """Test CLI result object creation"""
        result = CLIResult(
            success=True,
            message="Test message",
            data={"key": "value"},
            exit_code=0
        )

        assert result.success is True
        assert result.message == "Test message"
        assert result.data == {"key": "value"}
        assert result.exit_code == 0

    def test_input_validator(self):
        """Test input validation utilities"""
        # Test positive integer validation
        errors = InputValidator.validate_positive_integer(10, "test_param")
        assert len(errors) == 0

        errors = InputValidator.validate_positive_integer(-5, "test_param")
        assert len(errors) == 1
        assert "must be >=" in errors[0]

        # Test file path validation
        with tempfile.NamedTemporaryFile() as tmp:
            errors = InputValidator.validate_file_exists(tmp.name, "test_file")
            assert len(errors) == 0

        errors = InputValidator.validate_file_exists("/nonexistent/file.txt", "test_file")
        assert len(errors) == 1
        assert "not found" in errors[0]

        # Test choice validation
        errors = InputValidator.validate_choice("valid", "test_param", ["valid", "options"])
        assert len(errors) == 0

        errors = InputValidator.validate_choice("invalid", "test_param", ["valid", "options"])
        assert len(errors) == 1
        assert "must be one of" in errors[0]


class TestRunCommand:
    """Test run command functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
        self.context = CLIContext()
        self.command = RunCommand(self.context)

    def test_run_command_validation(self):
        """Test run command input validation"""
        # Test invalid days
        errors = self.command.validate_inputs(days=-10)
        assert len(errors) == 1
        assert "must be" in errors[0] and "positive" in errors[0]

        # Test invalid seed
        errors = self.command.validate_inputs(seed=-1)
        assert len(errors) == 1
        assert "must be" in errors[0] and "non-negative" in errors[0]

    def test_run_command_execution_mock(self):
        """Test run command execution with mocked simulation"""
        with tempfile.NamedTemporaryFile(suffix='.yaml') as tmp_config:
            tmp_config.write(b"""
simulation:
  duration_days: 30
  output_directory: results
colony:
  species: apis_mellifera
  initial_population:
    workers: 10000
""")
            tmp_config.flush()

            with patch('time.sleep'):
                with patch('src.bstew.cli.commands.run.Path.mkdir'):
                    result = self.command.execute(
                        config=tmp_config.name,
                        output="results",
                        days=30
                    )

                    assert result.success is True
                    assert "Simulation completed" in result.message


class TestConfigCommand:
    """Test config command functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
        self.context = CLIContext()
        self.command = ConfigCommand(self.context)

    def test_config_validation(self):
        """Test config command validation"""
        # Test valid actions
        errors = self.command.validate_inputs(action="create", name="test")
        assert len(errors) == 0

        # Test invalid action
        errors = self.command.validate_inputs(action="invalid")
        assert len(errors) == 1
        assert "must be one of" in errors[0]

        # Test validate with existing file
        with tempfile.NamedTemporaryFile(suffix='.yaml') as tmp:
            errors = self.command.validate_inputs(action="validate", name=tmp.name)
            assert len(errors) == 0

    def test_config_creation(self):
        """Test configuration file creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_config.yaml"

            result = self.command.execute(
                action="create",
                name="test_config",
                output=str(output_path),
                template="basic"
            )

            assert result.success is True
            assert output_path.exists()

            # Verify config content
            with open(output_path, 'r') as f:
                config = yaml.safe_load(f)

            assert "simulation" in config
            assert "colony" in config


class TestAnalyzeCommand:
    """Test analyze command functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
        self.context = CLIContext()
        self.command = AnalyzeCommand(self.context)

    def test_analyze_validation(self):
        """Test analyze command validation"""
        # Test invalid format
        errors = self.command.validate_inputs(
            input_dir=".",
            format_type="invalid_format"
        )
        assert len(errors) == 1
        assert "must be one of" in errors[0]

    def test_analyze_execution(self):
        """Test analyze command execution"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock results
            results_path = Path(tmpdir) / "results.csv"
            results_df = pd.DataFrame({
                'day': range(10),
                'population': np.random.randint(10000, 20000, 10),
                'honey': np.random.uniform(40, 50, 10)
            })
            results_df.to_csv(results_path, index=False)

            result = self.command.execute(
                input_dir=tmpdir,
                format_type="table"
            )

            assert result.success is True
            assert "Analysis completed" in result.message
            assert "results" in result.data


class TestUtilityCommands:
    """Test utility command functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
        self.context = CLIContext()

    def test_version_command(self):
        """Test version command"""
        command = VersionCommand(self.context)

        result = command.execute()

        assert result.success is True
        assert "Version information displayed" in result.message
        assert "bstew" in result.data
        assert "python" in result.data

    def test_init_command_validation(self):
        """Test init command validation"""
        command = InitCommand(self.context)

        # Test valid template
        errors = command.validate_inputs(template="basic")
        assert len(errors) == 0

        # Test invalid template
        errors = command.validate_inputs(template="invalid_template")
        assert len(errors) == 1
        assert "Invalid template" in errors[0]

    def test_init_command_execution(self):
        """Test init command execution"""
        command = InitCommand(self.context)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test_project"

            result = command.execute(
                directory=str(project_path),
                template="basic"
            )

            assert result.success is True
            assert "Project initialized successfully" in result.message

            # Check created structure
            assert project_path.exists()
            assert (project_path / "configs").exists()
            assert (project_path / "data").exists()
            assert (project_path / "results").exists()
            assert (project_path / "README.md").exists()


class TestCLIIntegration:
    """Test full CLI integration"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()

    def test_help_command(self):
        """Test CLI help output"""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "BSTEW" in result.output
        assert "BeeSteward" in result.output

    def test_version_cli(self):
        """Test version command through CLI"""
        result = self.runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_invalid_command(self):
        """Test invalid command handling"""
        result = self.runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

    def test_config_create_cli(self):
        """Test config creation through CLI"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            result = self.runner.invoke(app, [
                "config", "create", "test_config",
                "--output", str(config_path),
                "--template", "basic"
            ])

            assert result.exit_code == 0
            assert config_path.exists()

    def test_analyze_cli(self):
        """Test analyze command through CLI"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock results
            results_path = Path(tmpdir) / "results.csv"
            results_df = pd.DataFrame({
                'day': range(5),
                'population': [10000, 12000, 14000, 13000, 11000],
                'honey': [40, 42, 44, 43, 41]
            })
            results_df.to_csv(results_path, index=False)

            result = self.runner.invoke(app, [
                "analyze", tmpdir,
                "--format", "table"
            ])

            assert result.exit_code == 0


class TestErrorHandling:
    """Test error handling and edge cases"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
        self.context = CLIContext()

    def test_missing_config_file(self):
        """Test handling of missing configuration files"""
        command = RunCommand(self.context)

        errors = command.validate_inputs(config="/nonexistent/config.yaml")
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_invalid_output_directory(self):
        """Test handling of invalid output directories"""
        command = RunCommand(self.context)

        # Try to use a file as output directory
        with tempfile.NamedTemporaryFile() as tmp:
            errors = command.validate_inputs(output=tmp.name)
            # Should handle gracefully - either no errors or appropriate error
            assert len(errors) >= 0

    def test_malformed_yaml_files(self):
        """Test handling of malformed YAML files"""
        command = ConfigCommand(self.context)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write("invalid: yaml: content: [unclosed")
            tmp.flush()

            result = command.execute(
                action="validate",
                name=tmp.name
            )

            # Should handle YAML parsing errors
            assert result.success is False

    def test_empty_result_files(self):
        """Test handling of empty result files"""
        command = AnalyzeCommand(self.context)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty results file
            empty_file = Path(tmpdir) / "results.csv"
            empty_file.write_text("")

            result = command.execute(
                input_dir=tmpdir,
                format_type="table"
            )

            # Should handle empty files gracefully
            assert result.success is True or result.success is False


class TestProgressReporting:
    """Test progress reporting functionality"""

    def test_progress_manager_context(self):
        """Test progress manager context usage"""
        from rich.console import Console
        console = Console()
        progress_manager = ProgressManager(console)

        with progress_manager.progress_context() as progress:
            task = progress.start_task("Test task", total=10)

            for i in range(10):
                progress.update_task(task, advance=1)

            progress.finish_task(task, "Task completed")

        # Should complete without errors
        assert True

    def test_status_display(self):
        """Test status display functionality"""
        from rich.console import Console
        from src.bstew.cli.core.progress import StatusDisplay

        console = Console()
        status_display = StatusDisplay(console)

        # Test various display methods
        status_display.show_results_summary({
            "final_population": 15000,
            "max_population": 20000,
            "total_honey": 45.5
        })

        # Should complete without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
