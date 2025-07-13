"""
Type definitions and protocols for BSTEW CLI
============================================

Defines common types, protocols, and interfaces used throughout the CLI system.
"""

from typing import Protocol, Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum


class OutputFormat(Enum):
    """Supported output formats"""
    
    TABLE = "table"
    JSON = "json"
    CSV = "csv"
    YAML = "yaml"
    EXCEL = "excel"


class VerbosityLevel(Enum):
    """CLI verbosity levels"""
    
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class CLIResult:
    """Standard result object for CLI operations"""
    
    def __init__(
        self,
        success: bool,
        message: str = "",
        data: Optional[Dict[str, Any]] = None,
        exit_code: int = 0,
    ) -> None:
        self.success = success
        self.message = message
        self.data = data or {}
        self.exit_code = exit_code
    
    def __bool__(self) -> bool:
        return self.success


class CLICommand(Protocol):
    """Protocol for CLI command implementations"""
    
    def execute(self, **kwargs: Any) -> CLIResult:
        """Execute the command with given parameters"""
        ...
    
    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate command inputs, return list of error messages"""
        ...


class ProgressReporter(Protocol):
    """Protocol for progress reporting in commands"""
    
    def start_task(self, description: str, total: Optional[int] = None) -> str:
        """Start a new progress task, return task ID"""
        ...
    
    def update_task(self, task_id: str, advance: int = 1, description: Optional[str] = None) -> None:
        """Update progress task"""
        ...
    
    def finish_task(self, task_id: str, description: Optional[str] = None) -> None:
        """Mark task as finished"""
        ...


class ConfigProvider(Protocol):
    """Protocol for configuration providers"""
    
    def load_config(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration, return list of error messages"""
        ...
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        ...


class ResultFormatter(Protocol):
    """Protocol for result formatters"""
    
    def format_results(
        self,
        data: Dict[str, Any],
        format_type: OutputFormat,
        output_file: Optional[Path] = None,
    ) -> str:
        """Format results in specified format"""
        ...


# Type aliases for common CLI parameters
ConfigPath = Union[str, Path]
OutputPath = Union[str, Path]
ParameterDict = Dict[str, Any]
ValidationErrors = List[str]

# Command parameter types
RunParameters = Dict[str, Union[str, int, float, bool, None]]
ExperimentParameters = Dict[str, Union[str, int, float, bool, List[Any], None]]
OptimizationParameters = Dict[str, Union[str, int, float, bool, None]]