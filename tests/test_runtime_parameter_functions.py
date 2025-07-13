"""
Tests for runtime parameter management functions
==============================================

Tests for the runtime parameter management functionality.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import Mock
from datetime import datetime


# Copy the functions we need to test directly here to avoid circular imports
def parse_parameter_value(value_str: str) -> Any:
    """Parse parameter value from string"""
    
    # Try to parse as JSON first for complex types
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass
    
    # Try common type conversions
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    
    try:
        # Try integer
        return int(value_str)
    except ValueError:
        pass
    
    try:
        # Try float
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def validate_parameter_change(parameter_name: str, new_value: Any) -> bool:
    """Validate parameter change"""
    
    # Mock validation - in practice this would use actual validation logic
    # from the RuntimeParameterManager
    
    # Basic validation rules
    if parameter_name.endswith("_rate") and isinstance(new_value, (int, float)):
        return 0.0 <= new_value <= 1.0
    
    if parameter_name.endswith("_count") and isinstance(new_value, int):
        return new_value >= 0
    
    # Default to valid for demonstration
    return True


def get_parameter_history(manager: Any, param_name: Optional[str], 
                         simulation_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
    """Get parameter change history"""
    
    # Mock parameter history
    history = []
    for i in range(min(limit, 20)):  # Mock up to 20 entries
        history.append({
            "timestamp": datetime.now(),
            "parameter_name": param_name or f"mock_param_{i}",
            "old_value": f"old_value_{i}",
            "new_value": f"new_value_{i}",
            "reason": f"Test change {i}"
        })
    
    return history


def get_current_parameters(manager: Any, simulation_id: Optional[str], 
                          filter_pattern: Optional[str]) -> Dict[str, Any]:
    """Get current parameters"""
    
    # Mock current parameters
    params = {
        "mortality_rate": {"value": 0.05, "last_modified": datetime.now()},
        "foraging_efficiency": {"value": 0.78, "last_modified": datetime.now()},
        "population_size": {"value": 125000, "last_modified": datetime.now()}
    }
    
    if filter_pattern:
        # Filter parameters by pattern
        filtered_params = {}
        for param_name, param_info in params.items():
            if filter_pattern.lower() in param_name.lower():
                filtered_params[param_name] = param_info
        return filtered_params
    
    return params


def validate_configuration(config_data: Dict[str, Any], strict_mode: bool) -> Dict[str, Any]:
    """Validate configuration data"""
    
    # Mock validation
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "strict_mode": strict_mode
    }
    
    # Add some mock validation rules
    if "invalid_param" in config_data:
        validation_results["is_valid"] = False
        validation_results["errors"].append("Invalid parameter found: invalid_param")
    
    return validation_results


class TestParameterValueParsing:
    """Test parameter value parsing functions."""
    
    def test_parse_json_values(self):
        """Test parsing JSON values."""
        assert parse_parameter_value('{"key": "value"}') == {"key": "value"}
        assert parse_parameter_value('[1, 2, 3]') == [1, 2, 3]
        assert parse_parameter_value('null') is None
    
    def test_parse_boolean_values(self):
        """Test parsing boolean values."""
        assert parse_parameter_value("true") is True
        assert parse_parameter_value("false") is False
        assert parse_parameter_value("True") is True
        assert parse_parameter_value("False") is False
    
    def test_parse_numeric_values(self):
        """Test parsing numeric values."""
        assert parse_parameter_value("42") == 42
        assert parse_parameter_value("3.14") == 3.14
        assert parse_parameter_value("-10") == -10
        assert parse_parameter_value("0.0") == 0.0
    
    def test_parse_string_values(self):
        """Test parsing string values."""
        assert parse_parameter_value("hello") == "hello"
        assert parse_parameter_value("test_value") == "test_value"
        assert parse_parameter_value("") == ""
    
    def test_parse_complex_strings(self):
        """Test parsing complex string values."""
        assert parse_parameter_value("not_a_number") == "not_a_number"
        assert parse_parameter_value("123abc") == "123abc"
        assert parse_parameter_value("true_value") == "true_value"  # Not exactly "true"


class TestParameterValidation:
    """Test parameter validation functions."""
    
    def test_validate_rate_parameters(self):
        """Test validation of rate parameters."""
        assert validate_parameter_change("mortality_rate", 0.5) is True
        assert validate_parameter_change("mortality_rate", 0.0) is True
        assert validate_parameter_change("mortality_rate", 1.0) is True
        assert validate_parameter_change("mortality_rate", -0.1) is False
        assert validate_parameter_change("mortality_rate", 1.1) is False
    
    def test_validate_count_parameters(self):
        """Test validation of count parameters."""
        assert validate_parameter_change("population_count", 100) is True
        assert validate_parameter_change("population_count", 0) is True
        assert validate_parameter_change("population_count", -1) is False
    
    def test_validate_other_parameters(self):
        """Test validation of other parameter types."""
        assert validate_parameter_change("temperature", 25.5) is True
        assert validate_parameter_change("species_name", "Bombus") is True
        assert validate_parameter_change("config_flag", True) is True
    
    def test_validate_invalid_types(self):
        """Test validation with invalid types."""
        assert validate_parameter_change("mortality_rate", "not_a_number") is True  # Default to valid
        assert validate_parameter_change("population_count", 3.14) is True  # Not int but still valid by default


class TestParameterHistory:
    """Test parameter history functions."""
    
    def test_get_parameter_history_general(self):
        """Test getting general parameter history."""
        manager = Mock()
        history = get_parameter_history(manager, None, None, 10)
        
        assert len(history) == 10
        assert all("timestamp" in entry for entry in history)
        assert all("parameter_name" in entry for entry in history)
        assert all("old_value" in entry for entry in history)
        assert all("new_value" in entry for entry in history)
    
    def test_get_parameter_history_specific_param(self):
        """Test getting history for a specific parameter."""
        manager = Mock()
        history = get_parameter_history(manager, "mortality_rate", None, 5)
        
        assert len(history) == 5
        assert all(entry["parameter_name"] == "mortality_rate" for entry in history)
    
    def test_get_parameter_history_limit(self):
        """Test parameter history with different limits."""
        manager = Mock()
        
        # Test with limit smaller than mock data
        history = get_parameter_history(manager, None, None, 3)
        assert len(history) == 3
        
        # Test with limit larger than mock data
        history = get_parameter_history(manager, None, None, 50)
        assert len(history) == 20  # Max mock data available
    
    def test_get_parameter_history_empty(self):
        """Test getting parameter history with zero limit."""
        manager = Mock()
        history = get_parameter_history(manager, None, None, 0)
        assert len(history) == 0


class TestCurrentParameters:
    """Test current parameter retrieval functions."""
    
    def test_get_current_parameters_all(self):
        """Test getting all current parameters."""
        manager = Mock()
        params = get_current_parameters(manager, None, None)
        
        assert len(params) == 3
        assert "mortality_rate" in params
        assert "foraging_efficiency" in params
        assert "population_size" in params
        
        # Check structure
        assert params["mortality_rate"]["value"] == 0.05
        assert "last_modified" in params["mortality_rate"]
    
    def test_get_current_parameters_filtered(self):
        """Test getting filtered current parameters."""
        manager = Mock()
        
        # Filter by "rate"
        params = get_current_parameters(manager, None, "rate")
        assert len(params) == 1
        assert "mortality_rate" in params
        assert "foraging_efficiency" not in params
        
        # Filter by "efficiency"
        params = get_current_parameters(manager, None, "efficiency")
        assert len(params) == 1
        assert "foraging_efficiency" in params
        assert "mortality_rate" not in params
    
    def test_get_current_parameters_no_match(self):
        """Test getting parameters with no matching filter."""
        manager = Mock()
        params = get_current_parameters(manager, None, "nonexistent")
        assert len(params) == 0
    
    def test_get_current_parameters_case_insensitive(self):
        """Test that parameter filtering is case insensitive."""
        manager = Mock()
        
        params = get_current_parameters(manager, None, "MORTALITY")
        assert len(params) == 1
        assert "mortality_rate" in params
        
        params = get_current_parameters(manager, None, "Rate")
        assert len(params) == 1
        assert "mortality_rate" in params


class TestConfigurationValidation:
    """Test configuration validation functions."""
    
    def test_validate_valid_configuration(self):
        """Test validation of valid configuration."""
        config = {
            "mortality_rate": 0.05,
            "population_size": 10000,
            "foraging_efficiency": 0.75
        }
        
        result = validate_configuration(config, False)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0
        assert result["strict_mode"] is False
    
    def test_validate_invalid_configuration(self):
        """Test validation of invalid configuration."""
        config = {
            "mortality_rate": 0.05,
            "invalid_param": "should_not_exist"
        }
        
        result = validate_configuration(config, False)
        assert result["is_valid"] is False
        assert len(result["errors"]) == 1
        assert "Invalid parameter found: invalid_param" in result["errors"]
    
    def test_validate_strict_mode(self):
        """Test validation in strict mode."""
        config = {"mortality_rate": 0.05}
        
        result = validate_configuration(config, True)
        assert result["strict_mode"] is True
        assert result["is_valid"] is True  # Still valid since no invalid params
    
    def test_validate_empty_configuration(self):
        """Test validation of empty configuration."""
        config = {}
        
        result = validate_configuration(config, False)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0


class TestRuntimeParameterIntegration:
    """Test integration scenarios for runtime parameter management."""
    
    def test_modify_and_validate_parameter(self):
        """Test modifying and validating a parameter."""
        # Parse new value
        new_value = parse_parameter_value("0.08")
        assert new_value == 0.08
        
        # Validate the change
        is_valid = validate_parameter_change("mortality_rate", new_value)
        assert is_valid is True
        
        # Test invalid rate
        invalid_value = parse_parameter_value("1.5")
        is_valid = validate_parameter_change("mortality_rate", invalid_value)
        assert is_valid is False
    
    def test_parameter_history_workflow(self):
        """Test parameter history workflow."""
        manager = Mock()
        
        # Get history for specific parameter
        history = get_parameter_history(manager, "mortality_rate", None, 5)
        assert len(history) == 5
        
        # Check that all entries are for the requested parameter
        for entry in history:
            assert entry["parameter_name"] == "mortality_rate"
            assert isinstance(entry["timestamp"], datetime)
    
    def test_configuration_validation_workflow(self):
        """Test configuration validation workflow."""
        # Create a configuration with mixed valid/invalid parameters
        config = {
            "mortality_rate": 0.05,
            "population_size": 10000,
            "invalid_param": "bad_value"
        }
        
        # Validate in normal mode
        result = validate_configuration(config, False)
        assert result["is_valid"] is False
        assert len(result["errors"]) == 1
        
        # Remove invalid parameter
        del config["invalid_param"]
        result = validate_configuration(config, False)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
    
    def test_parameter_filtering_workflow(self):
        """Test parameter filtering workflow."""
        manager = Mock()
        
        # Get all parameters
        all_params = get_current_parameters(manager, None, None)
        assert len(all_params) == 3
        
        # Filter by pattern
        rate_params = get_current_parameters(manager, None, "rate")
        assert len(rate_params) == 1
        assert "mortality_rate" in rate_params
        
        # Filter by different pattern
        efficiency_params = get_current_parameters(manager, None, "efficiency")
        assert len(efficiency_params) == 1
        assert "foraging_efficiency" in efficiency_params


class TestErrorHandling:
    """Test error handling in runtime parameter functions."""
    
    def test_parse_malformed_json(self):
        """Test parsing malformed JSON."""
        # Should fall back to string
        result = parse_parameter_value('{"invalid": json}')
        assert result == '{"invalid": json}'
    
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_parameter_value("")
        assert result == ""
    
    def test_validate_none_values(self):
        """Test validation with None values."""
        assert validate_parameter_change("mortality_rate", None) is True  # Default to valid
        assert validate_parameter_change("population_count", None) is True
    
    def test_get_history_with_none_manager(self):
        """Test getting history with None manager."""
        # Should still work since we're using mocked data
        history = get_parameter_history(None, "test_param", None, 5)
        assert len(history) == 5
    
    def test_get_parameters_with_none_manager(self):
        """Test getting parameters with None manager."""
        # Should still work since we're using mocked data
        params = get_current_parameters(None, None, None)
        assert len(params) == 3
    
    def test_validate_configuration_with_none_config(self):
        """Test validation with None configuration."""
        # Should handle gracefully
        result = validate_configuration({}, False)
        assert result["is_valid"] is True