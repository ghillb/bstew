"""
Test cases for the masterpatch system
"""

import numpy as np
from unittest.mock import Mock

from src.bstew.components.masterpatch_system import (
    MasterPatchSystem,
    MasterPatch,
    ResourceLayer,
    LayerType,
)
from src.bstew.spatial.patches import FlowerSpecies, HabitatType


class TestMasterPatchSystem:
    """Test masterpatch system functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.landscape_bounds = (0, 0, 10000, 10000)
        self.masterpatch_system = MasterPatchSystem(self.landscape_bounds)

        # Create test species
        self.test_species = [
            FlowerSpecies(
                name="White Clover",
                bloom_start=120,
                bloom_end=270,
                nectar_production=4.0,
                pollen_production=2.0,
                flower_density=25.0,
                attractiveness=0.85,
                corolla_depth_mm=4.5,
                corolla_width_mm=1.8,
                nectar_accessibility=1.0,
            ),
            FlowerSpecies(
                name="Red Clover",
                bloom_start=140,
                bloom_end=250,
                nectar_production=6.0,
                pollen_production=3.0,
                flower_density=20.0,
                attractiveness=0.90,
                corolla_depth_mm=9.2,
                corolla_width_mm=2.1,
                nectar_accessibility=0.8,
            ),
            FlowerSpecies(
                name="Dandelion",
                bloom_start=80,
                bloom_end=300,
                nectar_production=3.0,
                pollen_production=1.5,
                flower_density=30.0,
                attractiveness=0.70,
                corolla_depth_mm=3.8,
                corolla_width_mm=1.5,
                nectar_accessibility=1.0,
            ),
        ]

        self.masterpatch_system.initialize_species_database(self.test_species)

    def test_system_initialization(self):
        """Test masterpatch system initialization"""
        assert self.masterpatch_system.landscape_bounds == (0, 0, 10000, 10000)
        assert len(self.masterpatch_system.species_database) == 3
        assert "White Clover" in self.masterpatch_system.species_database
        assert len(self.masterpatch_system.masterpatches) == 0

    def test_masterpatch_creation(self):
        """Test masterpatch creation"""
        patch = self.masterpatch_system.create_masterpatch(
            "test_patch", (5000, 5000), 10000.0, HabitatType.WILDFLOWER
        )

        assert patch.patch_id == "test_patch"
        assert patch.location == (5000, 5000)
        assert patch.area_m2 == 10000.0
        assert patch.habitat_type == HabitatType.WILDFLOWER
        assert len(patch.layers) == 0

        # Check it's stored in system
        assert "test_patch" in self.masterpatch_system.masterpatches

    def test_species_layer_addition(self):
        """Test adding species layers to patches"""
        # Create patch
        patch = self.masterpatch_system.create_masterpatch(
            "test_patch", (5000, 5000), 10000.0, HabitatType.WILDFLOWER
        )

        # Add species layer
        success = self.masterpatch_system.add_species_layer_to_patch(
            "test_patch", "White Clover", 0.5
        )

        assert success
        assert len(patch.layers) == 1

        # Check layer properties
        layer = patch.get_layer_by_species("White Clover")
        assert layer is not None
        assert layer.flower_species.name == "White Clover"
        assert layer.coverage_fraction == 0.5

    def test_habitat_based_population(self):
        """Test populating patches from habitat mapping"""
        # Create patches
        for i in range(3):
            self.masterpatch_system.create_masterpatch(
                f"patch_{i}", (i * 1000, i * 1000), 10000.0, HabitatType.WILDFLOWER
            )

        # Define habitat mapping
        habitat_mapping = {
            HabitatType.WILDFLOWER: [("White Clover", 0.4), ("Dandelion", 0.6)]
        }

        # Populate patches
        self.masterpatch_system.populate_patches_from_habitat(habitat_mapping)

        # Check all patches have species
        for patch in self.masterpatch_system.masterpatches.values():
            assert len(patch.layers) == 2
            assert patch.get_layer_by_species("White Clover") is not None
            assert patch.get_layer_by_species("Dandelion") is not None

    def test_patch_resource_updates(self):
        """Test patch resource updates"""
        # Create patch with species
        patch = self.masterpatch_system.create_masterpatch(
            "test_patch", (5000, 5000), 10000.0, HabitatType.WILDFLOWER
        )
        self.masterpatch_system.add_species_layer_to_patch(
            "test_patch", "White Clover", 1.0
        )

        # Update resources
        weather_conditions = {"temperature": 20.0, "rainfall": 0.0, "wind_speed": 5.0}

        self.masterpatch_system.update_all_patches(150, weather_conditions)  # June

        # Check resources were updated
        total_nectar, total_pollen = patch.get_total_resources()
        assert total_nectar > 0
        assert total_pollen > 0

    def test_weather_factor_calculation(self):
        """Test weather factor calculation"""
        # Optimal conditions
        optimal_weather = {"temperature": 20.0, "rainfall": 0.0, "wind_speed": 5.0}
        optimal_factor = self.masterpatch_system.calculate_weather_factor(
            optimal_weather
        )
        assert optimal_factor == 0.9

        # Poor conditions
        poor_weather = {"temperature": 0.0, "rainfall": 15.0, "wind_speed": 30.0}
        poor_factor = self.masterpatch_system.calculate_weather_factor(poor_weather)
        assert poor_factor < optimal_factor

    def test_patches_in_radius(self):
        """Test finding patches within radius"""
        # Create patches at different locations
        locations = [(1000, 1000), (2000, 2000), (8000, 8000)]

        for i, location in enumerate(locations):
            self.masterpatch_system.create_masterpatch(
                f"patch_{i}", location, 10000.0, HabitatType.WILDFLOWER
            )

        # Find patches near first location
        center = (1000, 1000)
        radius = 2000.0

        nearby_patches = self.masterpatch_system.get_patches_in_radius(center, radius)

        # Should find patches 0 and 1, but not 2
        assert len(nearby_patches) == 2
        patch_ids = [p.patch_id for p in nearby_patches]
        assert "patch_0" in patch_ids
        assert "patch_1" in patch_ids
        assert "patch_2" not in patch_ids

    def test_best_patches_for_species(self):
        """Test finding best patches for bee species"""
        # Create patches with different species
        self.masterpatch_system.create_masterpatch(
            "patch_1", (1000, 1000), 10000.0, HabitatType.WILDFLOWER
        )
        self.masterpatch_system.create_masterpatch(
            "patch_2", (2000, 2000), 10000.0, HabitatType.WILDFLOWER
        )

        # Add accessible and inaccessible species
        self.masterpatch_system.add_species_layer_to_patch(
            "patch_1", "White Clover", 1.0
        )  # Accessible
        self.masterpatch_system.add_species_layer_to_patch(
            "patch_2", "Red Clover", 1.0
        )  # Less accessible

        # Update resources
        weather_conditions = {"temperature": 20.0, "rainfall": 0.0, "wind_speed": 5.0}
        self.masterpatch_system.update_all_patches(150, weather_conditions)

        # Mock proboscis system
        mock_proboscis = Mock()
        mock_proboscis.get_species_proboscis.return_value = Mock(length_mm=8.0)

        # Mock accessibility results
        def mock_accessibility(proboscis, flower):
            accessibility_mock = Mock()
            if flower.name == "White Clover":
                accessibility_mock.accessibility_score = 0.8
            else:
                accessibility_mock.accessibility_score = 0.4
            return accessibility_mock

        mock_proboscis.calculate_accessibility.side_effect = mock_accessibility

        # Get best patches
        best_patches = self.masterpatch_system.get_best_patches_for_species(
            "Bombus_terrestris", (1000, 1000), 5000.0, mock_proboscis
        )

        # Should prefer patch with accessible species
        assert len(best_patches) == 2
        assert (
            best_patches[0][1] > best_patches[1][1]
        )  # First patch should have higher score

    def test_foraging_impact_simulation(self):
        """Test foraging impact simulation"""
        # Create patch with species
        patch = self.masterpatch_system.create_masterpatch(
            "test_patch", (5000, 5000), 10000.0, HabitatType.WILDFLOWER
        )
        self.masterpatch_system.add_species_layer_to_patch(
            "test_patch", "White Clover", 1.0
        )

        # Update resources
        weather_conditions = {"temperature": 20.0, "rainfall": 0.0, "wind_speed": 5.0}
        self.masterpatch_system.update_all_patches(150, weather_conditions)

        # Mock proboscis system
        mock_proboscis = Mock()
        mock_proboscis.get_species_proboscis.return_value = Mock(length_mm=8.0)

        # Mock accessibility result
        accessibility_mock = Mock()
        accessibility_mock.accessibility_score = 0.8
        accessibility_mock.is_accessible.return_value = True
        mock_proboscis.calculate_accessibility.return_value = accessibility_mock

        # Get initial resources
        initial_nectar, initial_pollen = patch.get_total_resources()

        # Simulate foraging
        consumption_results = self.masterpatch_system.simulate_foraging_impact(
            "test_patch", "Bombus_terrestris", 2.0, mock_proboscis
        )

        # Check consumption occurred
        assert len(consumption_results) == 1
        assert "White Clover" in consumption_results
        assert consumption_results["White Clover"]["nectar_consumed"] > 0
        assert consumption_results["White Clover"]["pollen_consumed"] > 0

        # Check resources decreased
        final_nectar, final_pollen = patch.get_total_resources()
        assert final_nectar < initial_nectar
        assert final_pollen < initial_pollen

    def test_management_scheduling(self):
        """Test management scheduling"""
        # Create patch
        patch = self.masterpatch_system.create_masterpatch(
            "test_patch", (5000, 5000), 10000.0, HabitatType.WILDFLOWER
        )
        self.masterpatch_system.add_species_layer_to_patch(
            "test_patch", "White Clover", 1.0
        )

        # Schedule management
        self.masterpatch_system.schedule_management(150, ["test_patch"], "mowing", 0.5)

        # Check management was scheduled
        assert 150 in self.masterpatch_system.management_calendar

        # Apply management
        self.masterpatch_system.apply_scheduled_management(150)

        # Check management was applied
        assert patch.management_regime == "mowing"
        assert patch.management_intensity == 0.5
        assert patch.last_management_day == 150

    def test_landscape_carrying_capacity(self):
        """Test landscape carrying capacity calculation"""
        # Create patches with species
        for i in range(3):
            self.masterpatch_system.create_masterpatch(
                f"patch_{i}", (i * 1000, i * 1000), 10000.0, HabitatType.WILDFLOWER
            )
            self.masterpatch_system.add_species_layer_to_patch(
                f"patch_{i}", "White Clover", 1.0
            )

        # Calculate capacity
        capacity = self.masterpatch_system.get_landscape_carrying_capacity()

        assert "nectar_limited" in capacity
        assert "pollen_limited" in capacity
        assert "overall" in capacity
        assert capacity["overall"] > 0

    def test_habitat_based_landscape_creation(self):
        """Test creating landscape from habitat map"""
        # Create simple habitat map
        habitat_map = np.array(
            [
                [1, 1, 2],  # Grassland, Grassland, Cropland
                [3, 4, 5],  # Woodland, Hedgerow, Wildflower
                [1, 1, 1],  # Grassland
            ]
        )

        # Create landscape
        patch_count = self.masterpatch_system.create_habitat_based_landscape(
            habitat_map, 100.0
        )

        # Should create patches for all habitat types
        assert patch_count > 0
        assert len(self.masterpatch_system.masterpatches) == patch_count

        # Check patches have appropriate species
        for patch in self.masterpatch_system.masterpatches.values():
            if patch.habitat_type == HabitatType.WILDFLOWER:
                assert len(patch.layers) > 0

    def test_export_patch_data(self):
        """Test exporting patch data"""
        # Create patch with species
        self.masterpatch_system.create_masterpatch(
            "test_patch", (5000, 5000), 10000.0, HabitatType.WILDFLOWER
        )
        self.masterpatch_system.add_species_layer_to_patch(
            "test_patch", "White Clover", 1.0
        )

        # Export data
        export_data = self.masterpatch_system.export_patch_data(150)

        assert "day_of_year" in export_data
        assert "masterpatches" in export_data
        assert len(export_data["masterpatches"]) == 1

        patch_data = export_data["masterpatches"][0]
        assert patch_data["patch_id"] == "test_patch"
        assert patch_data["species_count"] == 1
        assert len(patch_data["layers"]) == 1


class TestMasterPatch:
    """Test individual masterpatch functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.patch = MasterPatch(
            patch_id="test_patch",
            location=(5000, 5000),
            area_m2=10000.0,
            habitat_type=HabitatType.WILDFLOWER,
        )

        self.test_species = FlowerSpecies(
            name="White Clover",
            bloom_start=120,
            bloom_end=270,
            nectar_production=4.0,
            pollen_production=2.0,
            flower_density=25.0,
            attractiveness=0.85,
            corolla_depth_mm=4.5,
            corolla_width_mm=1.8,
            nectar_accessibility=1.0,
        )

    def test_patch_creation(self):
        """Test masterpatch creation"""
        assert self.patch.patch_id == "test_patch"
        assert self.patch.location == (5000, 5000)
        assert self.patch.area_m2 == 10000.0
        assert self.patch.habitat_type == HabitatType.WILDFLOWER
        assert len(self.patch.layers) == 0

    def test_layer_addition(self):
        """Test adding layers to patch"""
        layer = ResourceLayer(
            layer_id="test_layer",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
            coverage_fraction=0.5,
        )

        self.patch.add_layer(layer)

        assert len(self.patch.layers) == 1
        assert "test_layer" in self.patch.layers
        assert self.patch.layer_priorities[0] == "test_layer"

    def test_layer_removal(self):
        """Test removing layers from patch"""
        layer = ResourceLayer(
            layer_id="test_layer",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
        )

        self.patch.add_layer(layer)
        self.patch.remove_layer("test_layer")

        assert len(self.patch.layers) == 0
        assert "test_layer" not in self.patch.layers

    def test_available_species_by_day(self):
        """Test getting available species by day"""
        layer = ResourceLayer(
            layer_id="test_layer",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
        )

        self.patch.add_layer(layer)

        # During bloom period
        available = self.patch.get_available_species(150)
        assert len(available) == 1
        assert available[0].name == "White Clover"

        # Outside bloom period
        available = self.patch.get_available_species(50)
        assert len(available) == 0

    def test_resource_updates(self):
        """Test updating all layers in patch"""
        layer = ResourceLayer(
            layer_id="test_layer",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
        )

        self.patch.add_layer(layer)

        # Update resources
        self.patch.update_all_layers(150, 1.0)  # Good weather

        # Check resources were updated
        assert layer.current_nectar > 0
        assert layer.current_pollen > 0

    def test_total_resources_calculation(self):
        """Test total resources calculation"""
        # Add two layers
        layer1 = ResourceLayer(
            layer_id="layer1",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
        )
        layer1.current_nectar = 100.0
        layer1.current_pollen = 50.0

        layer2 = ResourceLayer(
            layer_id="layer2",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
        )
        layer2.current_nectar = 200.0
        layer2.current_pollen = 100.0

        self.patch.add_layer(layer1)
        self.patch.add_layer(layer2)

        total_nectar, total_pollen = self.patch.get_total_resources()

        assert total_nectar == 300.0
        assert total_pollen == 150.0

    def test_resource_consumption_from_species(self):
        """Test consuming resources from specific species"""
        layer = ResourceLayer(
            layer_id="test_layer",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
        )
        layer.current_nectar = 100.0
        layer.current_pollen = 50.0

        self.patch.add_layer(layer)

        # Consume resources
        actual_nectar, actual_pollen = self.patch.consume_resources_from_species(
            "White Clover", 30.0, 20.0
        )

        assert actual_nectar == 30.0
        assert actual_pollen == 20.0
        assert layer.current_nectar == 70.0
        assert layer.current_pollen == 30.0

    def test_management_application(self):
        """Test applying management to patch"""
        layer = ResourceLayer(
            layer_id="test_layer",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
        )

        self.patch.add_layer(layer)

        # Apply mowing
        self.patch.apply_management("mowing", 0.5, 150)

        assert self.patch.management_regime == "mowing"
        assert self.patch.management_intensity == 0.5
        assert self.patch.last_management_day == 150
        assert "mowing" in layer.management_effects


class TestResourceLayer:
    """Test resource layer functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_species = FlowerSpecies(
            name="White Clover",
            bloom_start=120,
            bloom_end=270,
            nectar_production=4.0,
            pollen_production=2.0,
            flower_density=25.0,
            attractiveness=0.85,
            corolla_depth_mm=4.5,
            corolla_width_mm=1.8,
            nectar_accessibility=1.0,
        )

        self.layer = ResourceLayer(
            layer_id="test_layer",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=self.test_species,
            coverage_fraction=0.5,
        )

    def test_layer_creation(self):
        """Test resource layer creation"""
        assert self.layer.layer_id == "test_layer"
        assert self.layer.layer_type == LayerType.FLOWER_SPECIES
        assert self.layer.flower_species.name == "White Clover"
        assert self.layer.coverage_fraction == 0.5
        assert len(self.layer.temporal_availability) == 365

    def test_availability_calculation(self):
        """Test availability calculation"""
        # During bloom period
        availability = self.layer.get_availability(150)
        assert availability == 1.0

        # Outside bloom period
        availability = self.layer.get_availability(50)
        assert availability == 0.0

    def test_resource_updates(self):
        """Test resource production updates"""
        # Update during bloom period
        self.layer.update_resources(150, 1.0, 10000.0)

        assert self.layer.current_nectar > 0
        assert self.layer.current_pollen > 0

        # Update outside bloom period
        self.layer.update_resources(50, 1.0, 10000.0)

        assert self.layer.current_nectar == 0.0
        assert self.layer.current_pollen == 0.0

    def test_resource_consumption(self):
        """Test resource consumption"""
        # Set initial resources
        self.layer.current_nectar = 100.0
        self.layer.current_pollen = 50.0
        initial_depletion = self.layer.depletion_factor

        # Consume resources
        self.layer.consume_resources(30.0, 20.0)

        assert self.layer.current_nectar == 70.0
        assert self.layer.current_pollen == 30.0
        assert self.layer.depletion_factor < initial_depletion

    def test_resource_recovery(self):
        """Test resource recovery"""
        # Set depleted state
        self.layer.depletion_factor = 0.5

        # Recover resources
        self.layer.recover_resources(0.1)

        assert self.layer.depletion_factor == 0.6
