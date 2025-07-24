"""
Test Update Configuration Implementation
========================================

Tests for Fix #11 (Audit #6): Replace mock implementation of update_configuration.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bstew.core.model import BeeModel


class TestUpdateConfiguration:
    """Test the update_configuration implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a simple model with basic config
        self.model = BeeModel(config={
            'foraging': {'max_foraging_range': 1000},
            'disease': {'enable_varroa': False},
            'environment': {'weather_variation': 0.2},
            'colony': {'colony_strength': 0.8}
        })

    def test_valid_configuration_update(self):
        """Test updating configuration with valid keys"""
        updates = {
            'foraging.max_foraging_range': 1500,
            'disease.enable_varroa': True
        }

        self.model.update_configuration(updates)

        # Check values were updated
        assert self.model.config.foraging.max_foraging_range == 1500
        assert self.model.config.disease.enable_varroa == True

    def test_invalid_configuration_keys(self):
        """Test that invalid keys raise ValueError"""
        updates = {
            'invalid.key': 'value',
            'foraging.max_foraging_range': 2000
        }

        with pytest.raises(ValueError) as exc_info:
            self.model.update_configuration(updates)

        assert "Invalid configuration keys: ['invalid.key']" in str(exc_info.value)

    def test_invalid_value_types(self):
        """Test that invalid value types raise appropriate errors"""
        # Invalid type for foraging.max_foraging_range (should be number)
        with pytest.raises(RuntimeError) as exc_info:
            self.model.update_configuration({'foraging.max_foraging_range': 'not a number'})
        assert "must be positive number" in str(exc_info.value)

        # Invalid type for disease.enable_varroa (should be bool)
        with pytest.raises(RuntimeError) as exc_info:
            self.model.update_configuration({'disease.enable_varroa': 'yes'})
        assert "must be boolean" in str(exc_info.value)

    def test_invalid_value_ranges(self):
        """Test that out-of-range values raise errors"""
        # Negative foraging range
        with pytest.raises(RuntimeError) as exc_info:
            self.model.update_configuration({'foraging.max_foraging_range': -100})
        assert "must be positive number" in str(exc_info.value)

        # Colony strength out of range
        with pytest.raises(RuntimeError) as exc_info:
            self.model.update_configuration({'colony.colony_strength': 1.5})
        assert "must be between 0 and 1" in str(exc_info.value)

    def test_configuration_rollback_on_error(self):
        """Test that configuration is rolled back on error"""
        original_range = self.model.config.foraging.max_foraging_range
        original_varroa = self.model.config.disease.enable_varroa

        # Try update with one valid and one invalid value
        updates = {
            'foraging.max_foraging_range': 2000,  # Valid
            'colony.colony_strength': 2.0  # Invalid (> 1)
        }

        with pytest.raises(RuntimeError) as exc_info:
            self.model.update_configuration(updates)

        # Check that config was rolled back
        assert self.model.config.foraging.max_foraging_range == original_range
        assert self.model.config.disease.enable_varroa == original_varroa
        assert "Configuration update failed, restored previous state" in str(exc_info.value)

    def test_valid_config_updates_multiple_sections(self):
        """Test updating config values across multiple sections"""
        updates = {
            'simulation.output_frequency': 10,
            'simulation.save_state': True,
            'foraging.dance_threshold': 0.7,
            'disease.natural_resistance': 0.3
        }

        self.model.update_configuration(updates)

        # Check all values were updated
        assert self.model.config.simulation.output_frequency == 10
        assert self.model.config.simulation.save_state == True
        assert self.model.config.foraging.dance_threshold == 0.7
        assert self.model.config.disease.natural_resistance == 0.3

    def test_get_valid_config_keys(self):
        """Test the _get_valid_config_keys method"""
        valid_keys = self.model._get_valid_config_keys()

        # Check it returns a set
        assert isinstance(valid_keys, set)

        # Check expected keys are present
        expected_keys = {
            'foraging.max_foraging_range',
            'disease.enable_varroa',
            'environment.weather_variation',
            'colony.colony_strength'
        }

        for key in expected_keys:
            assert key in valid_keys

    def test_empty_update(self):
        """Test that empty update dict doesn't cause errors"""
        original_config = self.model.config.model_copy(deep=True)

        # Should not raise any errors
        self.model.update_configuration({})

        # Config should be unchanged - compare dict representations
        assert self.model.config.model_dump() == original_config.model_dump()

    def test_multiple_valid_updates(self):
        """Test updating multiple valid configuration values"""
        updates = {
            'foraging.max_foraging_range': 3000,
            'disease.enable_varroa': True,
            'environment.weather_variation': 0.5,
            'colony.colony_strength': 0.9
        }

        self.model.update_configuration(updates)

        # Check all values were updated
        assert self.model.config.foraging.max_foraging_range == 3000
        assert self.model.config.disease.enable_varroa == True
        assert self.model.config.environment.weather_variation == 0.5
        assert self.model.config.colony.colony_strength == 0.9


def test_integration_with_simulation():
    """Integration test: updating config during simulation"""
    model = BeeModel(config={
        'foraging': {'max_foraging_range': 1000},
        'simulation': {'output_frequency': 1}
    })

    # Run a few steps
    for _ in range(5):
        model.step()

    # Update configuration mid-simulation
    model.update_configuration({
        'foraging.max_foraging_range': 2000,
        'simulation.output_frequency': 5
    })

    # Continue simulation
    for _ in range(5):
        model.step()

    # Model should still be functional
    assert model.schedule.steps == 10
    assert model.config.foraging.max_foraging_range == 2000
