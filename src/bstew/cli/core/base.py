"""
Base CLI command classes and utilities
======================================

Provides common functionality for all CLI commands.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from rich.console import Console
import typer

from ..types import CLIResult, VerbosityLevel, ConfigPath, OutputPath
from ...utils.config import ConfigManager
from .protocols import (
    DataValidationCommand,
    DataAnalysisCommand,
    PlottingCommand,
    ExperimentCommand,
    BatchCommand,
    OptimizationCommand,
    CalibrationCommand,
    ParameterSweepCommand,
    ScenarioComparisonCommand,
    ConfigManagementCommand,
    GenericCommand,
)


class CLIContext:
    """Shared context for CLI operations"""

    def __init__(
        self,
        console: Optional[Console] = None,
        config_manager: Optional[ConfigManager] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
    ) -> None:
        self.console = console or Console()
        self.config_manager = config_manager or ConfigManager()
        self.verbosity = verbosity

    def print_info(self, message: str, style: str = "blue") -> None:
        """Print informational message"""
        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            self.console.print(f"ℹ️  {message}", style=style)

    def print_success(self, message: str) -> None:
        """Print success message"""
        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            self.console.print(f"✅ {message}", style="green")

    def print_warning(self, message: str) -> None:
        """Print warning message"""
        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            self.console.print(f"⚠️  {message}", style="yellow")

    def print_error(self, message: str) -> None:
        """Print error message"""
        self.console.print(f"❌ {message}", style="red")

    def print_debug(self, message: str) -> None:
        """Print debug message"""
        if self.verbosity.value >= VerbosityLevel.DEBUG.value:
            self.console.print(f"🔧 {message}", style="dim")

    def print_verbose(self, message: str) -> None:
        """Print verbose message"""
        if self.verbosity.value >= VerbosityLevel.VERBOSE.value:
            self.console.print(f"📝 {message}", style="dim blue")


# Type alias for all CLI command protocols
CLICommandProtocol = Union[
    DataValidationCommand,
    DataAnalysisCommand,
    PlottingCommand,
    ExperimentCommand,
    BatchCommand,
    OptimizationCommand,
    CalibrationCommand,
    ParameterSweepCommand,
    ScenarioComparisonCommand,
    ConfigManagementCommand,
    GenericCommand,
]


class BaseCLICommand:
    """Base class for all CLI commands

    Commands must implement one of the defined protocols from .protocols module
    to ensure type safety while maintaining interface flexibility.
    """

    def __init__(self, context: CLIContext) -> None:
        self.context = context
        self.console = context.console
        self.config_manager = context.config_manager

    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate command inputs, return list of error messages"""
        return []

    def load_config(self, config_path: ConfigPath) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            config = self.config_manager.load_config(str(config_path))
            self.context.print_success(f"Loaded configuration: {config_path}")
            return config.model_dump()
        except Exception as e:
            self.context.print_error(f"Failed to load configuration: {e}")
            raise typer.Exit(1)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration and exit on error"""
        try:
            # Basic validation - check for required fields
            if "simulation" not in config:
                self.context.print_error("Configuration missing 'simulation' section")
                raise typer.Exit(1)

            if "colony" not in config:
                self.context.print_error("Configuration missing 'colony' section")
                raise typer.Exit(1)

        except Exception as e:
            self.context.print_error(f"Configuration validation error: {e}")
            raise typer.Exit(1)

    def ensure_output_dir(self, output_path: OutputPath) -> Path:
        """Ensure output directory exists"""
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        self.context.print_debug(f"Output directory: {path.absolute()}")
        return path

    def handle_exception(self, e: Exception, operation: str) -> CLIResult:
        """Handle exceptions consistently"""
        error_msg = f"{operation} failed: {e}"
        self.context.print_error(error_msg)

        if self.context.verbosity.value >= VerbosityLevel.VERBOSE.value:
            self.console.print_exception()

        return CLIResult(success=False, message=error_msg, exit_code=1)

    def apply_config_overrides(
        self, config: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply command-line overrides to configuration"""
        updated_config = config.copy()

        for key, value in overrides.items():
            if value is not None:
                # Handle nested keys like "colony.species"
                keys = key.split(".")
                current = updated_config

                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]

                current[keys[-1]] = value
                self.context.print_debug(f"Override: {key} = {value}")

        return updated_config


class CommandRegistry:
    """Registry for CLI commands"""

    def __init__(self) -> None:
        self._commands: Dict[str, type] = {}

    def register(self, name: str, command_class: type) -> None:
        """Register a command class"""
        self._commands[name] = command_class

    def get_command(self, name: str) -> Optional[type]:
        """Get a command class by name"""
        return self._commands.get(name)

    def list_commands(self) -> List[str]:
        """List all registered commands"""
        return list(self._commands.keys())


# Global command registry
command_registry = CommandRegistry()
