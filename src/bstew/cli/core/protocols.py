"""
CLI Command Protocols for Type Safety
====================================

Defines typed protocols for different CLI command execution patterns.
This enables proper type checking while maintaining interface flexibility.
"""

from typing import Protocol, Any, Optional
from ..types import CLIResult


class FileInputCommand(Protocol):
    """Protocol for commands that process input files/directories"""

    def execute(
        self,
        input_path: str,
        output_path: str = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute command with input and output paths"""
        ...


class DataValidationCommand(Protocol):
    """Protocol for commands that validate data against reference data"""

    def execute(
        self,
        model_results: str,
        field_data: str,
        metrics: str = ...,
        output: str = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute validation command"""
        ...


class DataAnalysisCommand(Protocol):
    """Protocol for commands that analyze data with format options"""

    def execute(
        self,
        input_dir: str,
        format_type: str = ...,
        output_file: Optional[str] = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute analysis command"""
        ...


class PlottingCommand(Protocol):
    """Protocol for commands that generate plots"""

    def execute(
        self,
        input_dir: str,
        plot_type: str = ...,
        output_dir: str = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute plotting command"""
        ...


class ExperimentCommand(Protocol):
    """Protocol for commands that run designed experiments"""

    def execute(
        self,
        design_file: str,
        output_dir: str = ...,
        resume: bool = ...,
        max_workers: Optional[int] = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute experiment command"""
        ...


class BatchCommand(Protocol):
    """Protocol for commands that run batch experiments"""

    def execute(
        self,
        experiments_file: str,
        parallel: int = ...,
        output_base: str = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute batch command"""
        ...


class OptimizationCommand(Protocol):
    """Protocol for optimization commands"""

    def execute(
        self,
        target_data: str,
        config: str = ...,
        method: str = ...,
        max_iterations: int = ...,
        population_size: int = ...,
        parallel: bool = ...,
        output: str = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute optimization command"""
        ...


class CalibrationCommand(Protocol):
    """Protocol for calibration commands"""

    def execute(
        self,
        field_data: str,
        config: str = ...,
        parameters: Optional[str] = ...,
        objective: str = ...,
        output: str = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute calibration command"""
        ...


class ParameterSweepCommand(Protocol):
    """Protocol for parameter sweep commands"""

    def execute(
        self,
        parameter: str,
        min_val: float,
        max_val: float,
        steps: int = ...,
        replicates: int = ...,
        config: Optional[str] = ...,
        output_dir: str = ...,
        simulation_days: int = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute parameter sweep command"""
        ...


class ScenarioComparisonCommand(Protocol):
    """Protocol for scenario comparison commands"""

    def execute(
        self,
        scenarios_file: str,
        output_dir: str = ...,
        simulation_days: int = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute scenario comparison command"""
        ...


class ConfigManagementCommand(Protocol):
    """Protocol for configuration management commands"""

    def execute(
        self,
        action: str,
        name: Optional[str] = ...,
        template: str = ...,
        output: Optional[str] = ...,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute configuration management command"""
        ...


class GenericCommand(Protocol):
    """Fallback protocol for commands with generic parameters"""

    def execute(self, **kwargs: Any) -> CLIResult:
        """Execute command with generic parameters"""
        ...
