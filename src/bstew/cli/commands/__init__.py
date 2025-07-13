"""
CLI command implementations for BSTEW
=====================================

Individual command modules for the BSTEW CLI interface.
"""

from .run import RunCommand
from .config import ConfigCommand
from .analyze import AnalyzeCommand
from .experiment import ExperimentCommand, BatchCommand, SweepCommand, CompareCommand
from .optimization import (
    OptimizePerformanceCommand,
    OptimizeParametersCommand,
    CalibrateCommand,
    SensitivityCommand,
    UncertaintyCommand,
)
from .validation import ValidateCommand
from .utility import VersionCommand, InitCommand

# Import new command modules for availability
from . import spatial
from . import visualization
from . import display
from . import data_analysis
from . import excel_reports
from . import runtime

__all__ = [
    "RunCommand",
    "ConfigCommand", 
    "AnalyzeCommand",
    "ExperimentCommand",
    "BatchCommand",
    "SweepCommand", 
    "CompareCommand",
    "OptimizePerformanceCommand",
    "OptimizeParametersCommand",
    "CalibrateCommand",
    "SensitivityCommand",
    "UncertaintyCommand",
    "ValidateCommand",
    "VersionCommand",
    "InitCommand",
    # New command modules
    "spatial",
    "visualization", 
    "display",
    "data_analysis",
    "excel_reports",
    "runtime",
]