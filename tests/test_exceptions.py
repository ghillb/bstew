"""
Test BSTEW Exception Hierarchy
==============================

Tests for the standardized error handling system.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bstew.exceptions import (
    BSTEWError,
    ValidationError,
    SimulationError,
    DataError,
    ConfigurationError,
    AnalysisError,
    IntegrationError,
    FileSystemError,
    ParameterError,
    ResourceError,
    StateError,
    raise_validation_error,
    raise_data_error,
    raise_config_error
)


class TestBSTEWExceptions:
    """Test the BSTEW exception hierarchy"""

    def test_base_exception(self):
        """Test base BSTEWError functionality"""
        # Basic exception
        with pytest.raises(BSTEWError) as exc_info:
            raise BSTEWError("Test error")

        assert str(exc_info.value) == "Test error"
        assert exc_info.value.message == "Test error"
        assert exc_info.value.details == {}

        # Exception with details
        with pytest.raises(BSTEWError) as exc_info:
            raise BSTEWError("Test error", {"key": "value", "count": 42})

        assert exc_info.value.details == {"key": "value", "count": 42}

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from BSTEWError"""
        exception_classes = [
            ValidationError,
            SimulationError,
            DataError,
            ConfigurationError,
            AnalysisError,
            IntegrationError,
            FileSystemError,
            ParameterError,
            ResourceError,
            StateError
        ]

        for exc_class in exception_classes:
            # Check inheritance
            assert issubclass(exc_class, BSTEWError)

            # Check instance
            exc = exc_class("Test message")
            assert isinstance(exc, BSTEWError)
            assert isinstance(exc, exc_class)

    def test_validation_error(self):
        """Test ValidationError functionality"""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid parameter", {"parameter": "max_range", "value": -100})

        assert "Invalid parameter" in str(exc_info.value)
        assert exc_info.value.details["parameter"] == "max_range"
        assert exc_info.value.details["value"] == -100

    def test_simulation_error(self):
        """Test SimulationError functionality"""
        with pytest.raises(SimulationError) as exc_info:
            raise SimulationError("Model initialization failed", {"step": 0, "reason": "memory"})

        assert "Model initialization failed" in str(exc_info.value)
        assert exc_info.value.details["reason"] == "memory"

    def test_data_error(self):
        """Test DataError functionality"""
        with pytest.raises(DataError) as exc_info:
            raise DataError("Corrupted data file", {"file": "data.csv", "line": 42})

        assert "Corrupted data file" in str(exc_info.value)
        assert exc_info.value.details["line"] == 42

    def test_convenience_functions(self):
        """Test convenience functions for raising errors"""
        # Test raise_validation_error
        with pytest.raises(ValidationError) as exc_info:
            raise_validation_error("Invalid value", field="temperature", value=-300)

        assert "Invalid value" in str(exc_info.value)
        assert exc_info.value.details["field"] == "temperature"
        assert exc_info.value.details["value"] == -300

        # Test raise_data_error
        with pytest.raises(DataError) as exc_info:
            raise_data_error("File not found", file_path="/tmp/data.csv", line_number=10)

        assert "File not found" in str(exc_info.value)
        assert exc_info.value.details["file_path"] == "/tmp/data.csv"
        assert exc_info.value.details["line_number"] == 10

        # Test raise_config_error
        with pytest.raises(ConfigurationError) as exc_info:
            raise_config_error("Type mismatch", config_key="simulation.duration", expected_type="int")

        assert "Type mismatch" in str(exc_info.value)
        assert exc_info.value.details["config_key"] == "simulation.duration"
        assert exc_info.value.details["expected_type"] == "int"

    def test_exception_catching_hierarchy(self):
        """Test that specific exceptions can be caught by base class"""
        # Catch specific exception by base class
        try:
            raise ValidationError("Test validation error")
        except BSTEWError as e:
            assert isinstance(e, ValidationError)
            assert str(e) == "Test validation error"

        # Catch multiple types
        errors = [
            ValidationError("Validation failed"),
            SimulationError("Simulation crashed"),
            DataError("Data corrupted")
        ]

        for error in errors:
            try:
                raise error
            except BSTEWError as e:
                assert isinstance(e, BSTEWError)
                assert isinstance(e, type(error))

    def test_exception_context_preservation(self):
        """Test that exception context is preserved through re-raising"""
        def inner_function():
            raise DataError("Inner error", {"location": "inner"})

        def outer_function():
            try:
                inner_function()
            except DataError as e:
                # Re-raise with additional context
                raise SimulationError(f"Simulation failed: {e}", {"original_error": str(e)}) from e

        with pytest.raises(SimulationError) as exc_info:
            outer_function()

        assert "Simulation failed: Inner error" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, DataError)


def test_exception_usage_in_application():
    """Test realistic usage patterns of exceptions"""

    def validate_config(config: dict) -> None:
        """Example validation function using custom exceptions"""
        if "simulation" not in config:
            raise ConfigurationError("Missing 'simulation' section", {"required_sections": ["simulation"]})

        if config.get("simulation", {}).get("duration", 0) <= 0:
            raise_validation_error(
                "Simulation duration must be positive",
                field="simulation.duration",
                value=config.get("simulation", {}).get("duration")
            )

    # Test missing section
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config({})

    assert "Missing 'simulation' section" in str(exc_info.value)

    # Test invalid value
    with pytest.raises(ValidationError) as exc_info:
        validate_config({"simulation": {"duration": -10}})

    assert "must be positive" in str(exc_info.value)
    assert exc_info.value.details["value"] == -10
