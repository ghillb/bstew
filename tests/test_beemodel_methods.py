"""
Test for BeeModel missing methods fix
=====================================

Ensures that BeeModel has all required methods for verbose simulation monitoring.
"""

import pytest
from bstew.core.model import BeeModel


def test_beemodel_has_required_methods():
    """Test that BeeModel has all required methods"""
    # Create a minimal BeeModel instance
    model = BeeModel(config={'simulation': {'duration_days': 1}})

    # Verify all required methods exist
    assert hasattr(model, 'get_total_population'), "BeeModel must have get_total_population method"
    assert hasattr(model, 'get_colony_count'), "BeeModel must have get_colony_count method"
    assert hasattr(model, 'get_simulation_state'), "BeeModel must have get_simulation_state method"

    # Verify methods are callable
    assert callable(model.get_total_population)
    assert callable(model.get_colony_count)
    assert callable(model.get_simulation_state)


def test_get_total_population_returns_int():
    """Test that get_total_population returns an integer"""
    model = BeeModel(config={'simulation': {'duration_days': 1}})

    # Call the method
    population = model.get_total_population()

    # Verify it returns an integer
    assert isinstance(population, int), "get_total_population must return an integer"
    assert population >= 0, "Population cannot be negative"


def test_get_simulation_state_returns_dict():
    """Test that get_simulation_state returns a proper dictionary"""
    model = BeeModel(config={'simulation': {'duration_days': 1}})

    # Call the method
    state = model.get_simulation_state()

    # Verify it returns a dictionary
    assert isinstance(state, dict), "get_simulation_state must return a dictionary"

    # Verify required keys are present
    required_keys = [
        'step',
        'total_population',
        'colony_count',
        'active_agents',
        'current_day',
        'current_season',
        'total_nectar_stores',
        'active_colonies'
    ]

    for key in required_keys:
        assert key in state, f"get_simulation_state must include '{key}' in its output"


def test_verbose_mode_integration():
    """Test that verbose mode can use these methods without errors"""
    # We don't need to import the CLI command, just test the model methods directly

    # Create a minimal model
    model = BeeModel(config={'simulation': {'duration_days': 1}})

    # Simulate what verbose mode does
    try:
        pop = model.get_total_population()
        assert isinstance(pop, int)

        # This would be called in verbose output
        verbose_msg = f"Day 0: Colony population {pop}"
        assert "Colony population" in verbose_msg
        assert str(pop) in verbose_msg

    except AttributeError as e:
        pytest.fail(f"Verbose mode failed with AttributeError: {e}")


def test_get_total_population_matches_get_total_bee_count():
    """Test that get_total_population is an alias for get_total_bee_count"""
    model = BeeModel(config={'simulation': {'duration_days': 1}})

    # Both methods should return the same value
    population = model.get_total_population()
    bee_count = model.get_total_bee_count()

    assert population == bee_count, "get_total_population should match get_total_bee_count"
