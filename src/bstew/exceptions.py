"""
BSTEW Exception Hierarchy
=========================

Standardized error handling for the BSTEW simulation framework.
All custom exceptions inherit from BSTEWError for consistent error handling.
"""

from typing import Optional, Any, Dict


class BSTEWError(Exception):
    """
    Base exception class for all BSTEW-specific errors.

    This provides a common base for all custom exceptions in the BSTEW framework,
    allowing for consistent error handling and logging throughout the application.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize BSTEW exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(BSTEWError):
    """
    Raised when input validation fails.

    This includes:
    - Invalid configuration parameters
    - Malformed input data files
    - Out-of-range parameter values
    - Missing required fields
    """

    pass


class SimulationError(BSTEWError):
    """
    Raised when simulation execution fails.

    This includes:
    - Model initialization failures
    - Runtime simulation crashes
    - Memory allocation errors
    - Numerical instabilities
    """

    pass


class DataError(BSTEWError):
    """
    Raised when data processing fails.

    This includes:
    - Data loading errors
    - Data parsing failures
    - Missing data files
    - Corrupted data structures
    """

    pass


class ConfigurationError(BSTEWError):
    """
    Raised when configuration operations fail.

    This includes:
    - Invalid configuration files
    - Missing configuration sections
    - Type mismatches in config values
    - Configuration update failures
    """

    pass


class AnalysisError(BSTEWError):
    """
    Raised when analysis operations fail.

    This includes:
    - Statistical calculation errors
    - Visualization generation failures
    - Missing analysis data
    - Invalid analysis parameters
    """

    pass


class IntegrationError(BSTEWError):
    """
    Raised when external integrations fail.

    This includes:
    - API connection failures
    - External service timeouts
    - Authentication errors
    - Data format mismatches
    """

    pass


class FileSystemError(BSTEWError):
    """
    Raised when file system operations fail.

    This includes:
    - Permission denied errors
    - Disk space issues
    - Invalid file paths
    - File corruption
    """

    pass


class ParameterError(BSTEWError):
    """
    Raised when parameter operations fail.

    This includes:
    - Parameter loading failures
    - Invalid parameter ranges
    - Parameter conversion errors
    - Missing parameter definitions
    """

    pass


class ResourceError(BSTEWError):
    """
    Raised when resource constraints are violated.

    This includes:
    - Memory limit exceeded
    - CPU time limit exceeded
    - Too many open files
    - Network resource exhaustion
    """

    pass


class StateError(BSTEWError):
    """
    Raised when invalid state transitions occur.

    This includes:
    - Invalid model state
    - Incorrect operation sequence
    - State corruption
    - Concurrent modification errors
    """

    pass


# Convenience functions for raising common errors with context


def raise_validation_error(
    message: str, field: Optional[str] = None, value: Optional[Any] = None
) -> None:
    """Raise a ValidationError with context."""
    details: Dict[str, Any] = {}
    if field:
        details["field"] = field
    if value is not None:
        details["value"] = value
    raise ValidationError(message, details)


def raise_data_error(
    message: str, file_path: Optional[str] = None, line_number: Optional[int] = None
) -> None:
    """Raise a DataError with file context."""
    details: Dict[str, Any] = {}
    if file_path:
        details["file_path"] = file_path
    if line_number is not None:
        details["line_number"] = line_number
    raise DataError(message, details)


def raise_config_error(
    message: str, config_key: Optional[str] = None, expected_type: Optional[str] = None
) -> None:
    """Raise a ConfigurationError with config context."""
    details: Dict[str, Any] = {}
    if config_key:
        details["config_key"] = config_key
    if expected_type:
        details["expected_type"] = expected_type
    raise ConfigurationError(message, details)
