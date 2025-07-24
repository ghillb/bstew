"""
Test Habitat Modification Features
==================================

Tests for Fix #10: Complete Habitat Modification Features (margin and pond).
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bstew.components.habitat_creation import HabitatCreationAlgorithms, ResourceDensityGrid


class TestHabitatModifications:
    """Test habitat modification implementations"""

    def setup_method(self):
        """Set up test fixtures"""
        self.designer = HabitatCreationAlgorithms()

        # Create a test grid
        self.test_grid = ResourceDensityGrid(
            bounds={"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100},
            resolution_m=1.0
        )

    def test_margin_modification(self):
        """Test field margin habitat modification"""
        # Define a margin modification
        margin_mod = {
            "type": "margin",
            "coordinates": [[50, 50]],  # Center point
            "width": 3.0,
            "resource_multiplier": 0.8
        }

        # Apply modification
        self.designer._apply_modification_to_grid(self.test_grid, margin_mod)

        # Check that resources were added in margin area
        center_x, center_y = 50, 50

        # Check center has resources
        assert self.test_grid.nectar_grid[center_y, center_x] > 0
        assert self.test_grid.pollen_grid[center_y, center_x] > 0

        # Check margin width (should affect 3 cells in each direction)
        assert self.test_grid.nectar_grid[center_y + 3, center_x] > 0
        assert self.test_grid.nectar_grid[center_y - 3, center_x] > 0
        assert self.test_grid.nectar_grid[center_y, center_x + 3] > 0
        assert self.test_grid.nectar_grid[center_y, center_x - 3] > 0

        # Check resource values are correctly multiplied
        expected_nectar = 3.0 * 0.8  # base value * multiplier
        expected_pollen = 2.0 * 0.8

        assert self.test_grid.nectar_grid[center_y, center_x] == pytest.approx(expected_nectar)
        assert self.test_grid.pollen_grid[center_y, center_x] == pytest.approx(expected_pollen)

    def test_margin_with_habitat_diversity(self):
        """Test margin modification with habitat diversity grid"""
        # Add habitat diversity grid
        self.test_grid.habitat_diversity = np.zeros((100, 100))

        margin_mod = {
            "type": "margin",
            "coordinates": [[50, 50]],
            "width": 2.0
        }

        self.designer._apply_modification_to_grid(self.test_grid, margin_mod)

        # Check habitat diversity was set
        assert self.test_grid.habitat_diversity[50, 50] == 0.7

    def test_pond_modification(self):
        """Test pond habitat modification"""
        # Add water grid
        self.test_grid.water_grid = np.zeros((100, 100))

        # Define pond modification
        pond_mod = {
            "type": "pond",
            "coordinates": [[50, 50]],  # Center point
            "radius": 5.0
        }

        # Store original resource values
        original_nectar = self.test_grid.nectar_grid.copy()
        original_pollen = self.test_grid.pollen_grid.copy()

        # Add some resources to test reduction
        self.test_grid.nectar_grid[45:55, 45:55] = 10.0
        self.test_grid.pollen_grid[45:55, 45:55] = 8.0

        # Apply modification
        self.designer._apply_modification_to_grid(self.test_grid, pond_mod)

        # Check water was added in pond area
        center_x, center_y = 50, 50
        assert self.test_grid.water_grid[center_y, center_x] == 1.0

        # Check circular area
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                distance = (dx ** 2 + dy ** 2) ** 0.5
                if distance <= 5.0:
                    assert self.test_grid.water_grid[center_y + dy, center_x + dx] == 1.0
                    # Check resources were reduced
                    assert self.test_grid.nectar_grid[center_y + dy, center_x + dx] <= 1.0
                    assert self.test_grid.pollen_grid[center_y + dy, center_x + dx] <= 0.8

    def test_pond_without_water_grid(self):
        """Test pond modification when water grid doesn't exist"""
        # Remove water grid
        if hasattr(self.test_grid, 'water_grid'):
            delattr(self.test_grid, 'water_grid')

        pond_mod = {
            "type": "pond",
            "coordinates": [[50, 50]],
            "radius": 3.0
        }

        # Add some resources
        self.test_grid.nectar_grid[47:53, 47:53] = 10.0
        self.test_grid.pollen_grid[47:53, 47:53] = 8.0

        # Should not raise error
        self.designer._apply_modification_to_grid(self.test_grid, pond_mod)

        # Resources should still be reduced
        assert self.test_grid.nectar_grid[50, 50] == pytest.approx(1.0)
        assert self.test_grid.pollen_grid[50, 50] == pytest.approx(0.8)

    def test_multiple_modifications(self):
        """Test applying multiple habitat modifications"""
        # Add water grid
        self.test_grid.water_grid = np.zeros((100, 100))

        modifications = [
            {
                "type": "margin",
                "coordinates": [[20, 20]],
                "width": 2.0
            },
            {
                "type": "pond",
                "coordinates": [[80, 80]],
                "radius": 4.0
            },
            {
                "type": "wildflower_strip",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[10, 50], [90, 50]]
                }
            }
        ]

        for mod in modifications:
            self.designer._apply_modification_to_grid(self.test_grid, mod)

        # Check margin added resources
        assert self.test_grid.nectar_grid[20, 20] > 0

        # Check pond added water
        assert self.test_grid.water_grid[80, 80] == 1.0

        # Check wildflower strip added resources along line
        assert self.test_grid.nectar_grid[50, 50] > 0

    def test_boundary_conditions(self):
        """Test modifications near grid boundaries"""
        # Margin near edge
        margin_mod = {
            "type": "margin",
            "coordinates": [[2, 2]],  # Near corner
            "width": 3.0
        }

        self.designer._apply_modification_to_grid(self.test_grid, margin_mod)

        # Should not crash and should add resources within bounds
        assert self.test_grid.nectar_grid[2, 2] > 0

        # Pond partially outside grid
        self.test_grid.water_grid = np.zeros((100, 100))
        pond_mod = {
            "type": "pond",
            "coordinates": [[95, 95]],  # Near corner
            "radius": 8.0
        }

        self.designer._apply_modification_to_grid(self.test_grid, pond_mod)

        # Should add water only within bounds
        assert self.test_grid.water_grid[95, 95] == 1.0
        # But not outside bounds (would raise IndexError if attempted)


def test_integration_habitat_modifications():
    """Integration test for complete habitat modification workflow"""
    designer = HabitatCreationAlgorithms()

    # Create resource density grid
    grid = ResourceDensityGrid(
        bounds={"min_x": 0, "max_x": 200, "min_y": 0, "max_y": 200},
        resolution_m=2.0
    )

    # Add water grid for pond
    grid.water_grid = np.zeros((grid.height, grid.width))

    # Apply modifications one by one
    modifications = [
        {
            "type": "wildflower_strip",
            "geometry": {
                "type": "LineString",
                "coordinates": [[50, 100], [150, 100]]
            }
        },
        {
            "type": "margin",
            "coordinates": [[100, 50], [100, 150]],
            "width": 5.0,
            "resource_multiplier": 0.6
        },
        {
            "type": "pond",
            "coordinates": [[100, 100]],
            "radius": 10.0
        }
    ]

    for mod in modifications:
        designer._apply_modification_to_grid(grid, mod)

    # Verify grid was created correctly
    assert isinstance(grid, ResourceDensityGrid)
    assert grid.width == 100  # 200 / 2.0
    assert grid.height == 100

    # Check modifications were applied
    # Overall grid should have resources added
    assert np.sum(grid.nectar_grid) > 0
    assert np.sum(grid.pollen_grid) > 0

    # Pond should have water added
    assert np.sum(grid.water_grid) > 0
