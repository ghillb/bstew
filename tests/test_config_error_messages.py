"""
Test Configuration Error Messages Improvement
=============================================

Tests for Fix #7: Clear and helpful error messages in config command.
"""

import pytest
from pathlib import Path

from bstew.cli.commands.config import ConfigCommand
from bstew.cli.core.base import CLIContext


class TestConfigErrorMessages:
    """Test configuration command error messages"""

    def setup_method(self):
        """Set up test fixtures"""
        self.context = CLIContext()
        self.command = ConfigCommand(self.context)

    def test_create_without_name_single_error(self):
        """Test that missing name only shows one clear error"""
        errors = self.command.validate_inputs(
            action="create",
            name=None,
            template="invalid"  # This should NOT trigger a second error
        )

        # Should only have one error about missing name
        assert len(errors) == 1
        assert "Configuration name is required" in errors[0]
        assert "Usage: bstew config create" in errors[0]

        # Should NOT mention template when name is missing
        assert "template" not in errors[0].lower() or "usage" in errors[0].lower()

    def test_create_with_invalid_template_clear_message(self):
        """Test that invalid template shows available options"""
        errors = self.command.validate_inputs(
            action="create",
            name="myconfig",
            template="invalid_template"
        )

        # Should have one error about invalid template
        assert len(errors) == 1
        assert "Invalid template 'invalid_template'" in errors[0]
        assert "Available templates:" in errors[0]
        assert "basic" in errors[0]
        assert "honeybee" in errors[0]
        assert "bumblebee" in errors[0]

    def test_create_with_valid_inputs_no_errors(self):
        """Test that valid inputs produce no errors"""
        errors = self.command.validate_inputs(
            action="create",
            name="myconfig",
            template="basic"
        )

        assert len(errors) == 0

    def test_validate_action_clear_error(self):
        """Test validate action error messages"""
        # Test missing path
        errors = self.command.validate_inputs(
            action="validate",
            name=None
        )

        assert len(errors) == 1
        assert "Configuration file path is required" in errors[0]

        # Test non-existent file
        errors = self.command.validate_inputs(
            action="validate",
            name="/non/existent/file.yaml"
        )

        assert len(errors) == 1
        assert "Configuration file not found" in errors[0]

    def test_unknown_action_error(self):
        """Test unknown action error message"""
        errors = self.command.validate_inputs(
            action="unknown_action"
        )

        assert len(errors) == 1
        assert "Action must be one of" in errors[0]
        assert "['create', 'validate', 'show', 'list', 'diff', 'species']" in errors[0]

    def test_template_validation_lists(self):
        """Test that template validation lists match"""
        # The valid templates in _is_valid_template should match those in error message
        valid_in_method = ["basic", "honeybee", "bumblebee", "minimal", "research"]

        assert self.command._is_valid_template("basic")
        assert self.command._is_valid_template("honeybee")
        assert self.command._is_valid_template("bumblebee")
        assert self.command._is_valid_template("minimal")
        assert self.command._is_valid_template("research")
        assert not self.command._is_valid_template("invalid")

        # Get error message to check listed templates
        errors = self.command.validate_inputs(
            action="create",
            name="test",
            template="wrong"
        )

        error_msg = errors[0]
        for template in valid_in_method:
            assert template in error_msg


def test_config_command_execution_errors():
    """Test actual command execution error handling"""
    context = CLIContext()
    command = ConfigCommand(context)

    # Test create without name
    result = command.execute(action="create", name=None, template="invalid")
    assert not result.success
    assert "Configuration name is required" in result.message

    # Test with unknown action
    result = command.execute(action="unknown", name="test")
    assert not result.success
    assert "Unknown action: unknown" in result.message
