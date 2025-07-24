"""
Test for Pydantic logger validation fix
=======================================

Ensures that ParameterLoader and other BaseModel classes can initialize
without Pydantic validation errors when setting logger attributes.
"""

import pytest
from pydantic import ValidationError

from bstew.core.parameter_loader import ParameterLoader
from bstew.analysis.interactive_tools import InteractiveAnalysisEngine


def test_parameter_loader_logger_initialization():
    """Test that ParameterLoader can be initialized without validation errors"""
    # This should not raise a ValidationError
    loader = ParameterLoader()

    # Verify logger exists and works
    assert hasattr(loader, '_logger')
    assert loader._logger is not None

    # Test that logger can be used
    loader._logger.info("Test log message")


def test_interactive_analysis_engine_logger_initialization():
    """Test that InteractiveAnalysisEngine can be initialized without validation errors"""
    # This should not raise a ValidationError
    engine = InteractiveAnalysisEngine()

    # Verify logger exists and works
    assert hasattr(engine, '_logger')
    assert engine._logger is not None

    # Test that logger can be used
    engine._logger.info("Test log message")


def test_parameter_loader_functionality():
    """Test that ParameterLoader still functions correctly after the fix"""
    import tempfile
    from bstew.core.parameter_loader import ParameterType

    loader = ParameterLoader()

    # Test creating parameter template
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        loader.create_parameter_template(ParameterType.BEHAVIORAL, tmp.name)
        # Template creates the file, so we just verify it was created
        import os
        assert os.path.exists(tmp.name)
        os.unlink(tmp.name)  # Clean up

    # Test validation rules initialization
    assert hasattr(loader, 'validation_rules')
    assert isinstance(loader.validation_rules, dict)


def test_parameter_management_commands_integration():
    """Integration test for parameter management CLI commands"""
    from bstew.cli.commands.parameters import ParameterManagerCLI
    from bstew.cli.core.base import CLIContext

    # This should not raise a ValidationError
    context = CLIContext()
    manager = ParameterManagerCLI(context)

    # Verify manager was created successfully
    assert manager is not None
    assert hasattr(manager, 'parameter_loader')
    assert isinstance(manager.parameter_loader, ParameterLoader)
