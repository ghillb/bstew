"""
Tests for Species-Configurable Communication System
=================================================

Tests the unified communication system that adapts behavior based on
bee species configuration - supporting both honey bees and bumblebees.
"""

import pytest
from unittest.mock import Mock, patch

from bstew.core.bee_species_config import BeeSpeciesType, BeeSpeciesManager, create_multi_species_simulation
from bstew.core.bee_communication import UnifiedBeeCommunicationSystem, create_communication_system, create_multi_species_communication


class TestBeeSpeciesConfig:
    """Test bee species configuration system"""

    def setup_method(self):
        """Setup test fixtures"""
        self.species_manager = BeeSpeciesManager()

    def test_honey_bee_config(self):
        """Test honey bee species configuration"""
        config = self.species_manager.get_species_config(BeeSpeciesType.APIS_MELLIFERA)

        assert config.species_type == BeeSpeciesType.APIS_MELLIFERA
        assert config.common_name == "European Honey Bee"
        assert config.scientific_name == "Apis mellifera"
        assert config.uses_dance_communication is True
        assert config.uses_scent_communication is False
        assert config.social_information_sharing > 0.8  # High social sharing
        assert config.individual_decision_weight < 0.5  # Low individual weight
        assert config.max_colony_size > 20000  # Large colonies
        assert config.max_foraging_distance_m > 5000  # Long foraging distance

    def test_bumblebee_config(self):
        """Test bumblebee species configuration"""
        config = self.species_manager.get_species_config(BeeSpeciesType.BOMBUS_TERRESTRIS)

        assert config.species_type == BeeSpeciesType.BOMBUS_TERRESTRIS
        assert config.common_name == "Large Earth Bumblebee"
        assert config.scientific_name == "Bombus terrestris"
        assert config.uses_dance_communication is False
        assert config.uses_scent_communication is True
        assert config.social_information_sharing < 0.3  # Low social sharing
        assert config.individual_decision_weight > 0.8  # High individual weight
        assert config.max_colony_size < 500  # Small colonies
        assert config.max_foraging_distance_m < 2000  # Shorter foraging distance

    def test_species_identification(self):
        """Test species identification methods"""
        assert self.species_manager.is_honey_bee(BeeSpeciesType.APIS_MELLIFERA)
        assert not self.species_manager.is_honey_bee(BeeSpeciesType.BOMBUS_TERRESTRIS)

        assert self.species_manager.is_bumblebee(BeeSpeciesType.BOMBUS_TERRESTRIS)
        assert self.species_manager.is_bumblebee(BeeSpeciesType.BOMBUS_LAPIDARIUS)
        assert not self.species_manager.is_bumblebee(BeeSpeciesType.APIS_MELLIFERA)

    def test_available_species_lists(self):
        """Test getting lists of available species"""
        all_species = self.species_manager.get_available_species()
        honey_bees = self.species_manager.get_honey_bee_species()
        bumblebees = self.species_manager.get_bumblebee_species()

        assert len(all_species) > 0
        assert len(honey_bees) == 1  # Only Apis mellifera
        assert len(bumblebees) >= 3  # Multiple Bombus species
        assert BeeSpeciesType.APIS_MELLIFERA in honey_bees
        assert BeeSpeciesType.BOMBUS_TERRESTRIS in bumblebees

    def test_species_combination_validation(self):
        """Test validation of species combinations"""
        # Valid single species
        assert self.species_manager.validate_species_combination([BeeSpeciesType.APIS_MELLIFERA])
        assert self.species_manager.validate_species_combination([BeeSpeciesType.BOMBUS_TERRESTRIS])

        # Valid multiple bumblebees (with warning)
        valid_bumbles = [BeeSpeciesType.BOMBUS_TERRESTRIS, BeeSpeciesType.BOMBUS_LAPIDARIUS]
        assert self.species_manager.validate_species_combination(valid_bumbles)

        # Invalid multiple honey bees
        invalid_honeys = [BeeSpeciesType.APIS_MELLIFERA, BeeSpeciesType.APIS_MELLIFERA]
        assert not self.species_manager.validate_species_combination(invalid_honeys)

        # Test warning for too many bumblebee species (but still valid)
        many_bumbles = [
            BeeSpeciesType.BOMBUS_TERRESTRIS,
            BeeSpeciesType.BOMBUS_LAPIDARIUS,
            BeeSpeciesType.BOMBUS_PASCUORUM,
            BeeSpeciesType.BOMBUS_HORTORUM,
            BeeSpeciesType.BOMBUS_PRATORUM  # 5 species - should trigger warning
        ]
        # Should still return True (just warns, doesn't fail)
        assert self.species_manager.validate_species_combination(many_bumbles)


class TestUnifiedCommunicationSystem:
    """Test unified communication system"""

    def test_honey_bee_communication_system(self):
        """Test honey bee communication system creation"""
        comm_system = UnifiedBeeCommunicationSystem(BeeSpeciesType.APIS_MELLIFERA)

        assert comm_system.species_type == BeeSpeciesType.APIS_MELLIFERA
        assert comm_system.is_honey_bee is True
        assert comm_system.is_bumblebee is False

        # Test species info
        info = comm_system.get_species_info()
        assert info["uses_dance_communication"] is True
        assert info["uses_scent_communication"] is False
        assert info["communication_type"] == "dance_communication"

    def test_bumblebee_communication_system(self):
        """Test bumblebee communication system creation"""
        comm_system = UnifiedBeeCommunicationSystem(BeeSpeciesType.BOMBUS_TERRESTRIS)

        assert comm_system.species_type == BeeSpeciesType.BOMBUS_TERRESTRIS
        assert comm_system.is_honey_bee is False
        assert comm_system.is_bumblebee is True

        # Test species info
        info = comm_system.get_species_info()
        assert info["uses_dance_communication"] is False
        assert info["uses_scent_communication"] is True
        assert info["communication_type"] == "scent_communication"

    @patch('bstew.core.honey_bee_communication.HoneyBeeCommunicationSystem')
    def test_honey_bee_foraging_integration(self, mock_honey_system):
        """Test honey bee foraging integration"""
        # Setup mock
        mock_instance = Mock()
        mock_honey_system.return_value = mock_instance
        mock_instance.evaluate_dance_decision.return_value = True
        mock_instance.perform_dance.return_value = Mock(
            dance_vigor=0.8,
            followers=set([101, 102]),
            dance_id="test_dance",
            dance_type=Mock(value="waggle_dance"),
            dance_duration=60.0
        )

        # Create communication system
        comm_system = UnifiedBeeCommunicationSystem(BeeSpeciesType.APIS_MELLIFERA)
        comm_system.communication_system = mock_instance

        # Test foraging integration
        foraging_result = {
            "patch_quality": 0.9,
            "energy_efficiency": 0.8,
            "distance": 200.0
        }

        result = comm_system.integrate_foraging_success_with_communication(
            123, foraging_result, {}, {}
        )

        assert result["communication_type"] == "dance"
        assert result["should_communicate"] is True
        assert result["information_shared"] is True
        assert result["communication_vigor"] == 0.8
        assert result["expected_followers"] == 2
        assert "dance_id" in result

    @patch('bstew.core.bumblebee_communication.BumblebeeCommunicationSystem')
    def test_bumblebee_foraging_integration(self, mock_bumblebee_system):
        """Test bumblebee foraging integration"""
        # Setup mock
        mock_instance = Mock()
        mock_bumblebee_system.return_value = mock_instance
        mock_instance.should_leave_scent_mark.return_value = True
        mock_instance.leave_scent_mark.return_value = {
            "strength": 0.7,
            "duration": 300.0
        }
        mock_instance.potentially_leave_scent_mark.return_value = True

        # Create communication system
        comm_system = UnifiedBeeCommunicationSystem(BeeSpeciesType.BOMBUS_TERRESTRIS)
        comm_system.communication_system = mock_instance

        # Test foraging integration
        foraging_result = {
            "patch_quality": 0.8,
            "energy_gained": 50.0,
            "distance": 400.0,
            "patch_id": 456,
            "location": (100.0, 200.0)
        }

        result = comm_system.integrate_foraging_success_with_communication(
            123, foraging_result, {}, {}
        )

        assert result["communication_type"] == "scent"
        assert result["should_communicate"] is True
        assert result["information_shared"] is True
        assert result["scent_marked"] is True
        assert result["scent_strength"] == 0.5
        assert result["communication_vigor"] == 0.5

    @patch('bstew.core.bumblebee_communication.BumblebeeCommunicationSystem')
    def test_bumblebee_scent_marking_compatibility(self, mock_bumblebee_system):
        """Test bumblebee scent marking compatibility method"""
        # Setup mock
        mock_instance = Mock()
        mock_bumblebee_system.return_value = mock_instance
        mock_instance.should_leave_scent_mark.return_value = True
        mock_instance.leave_scent_mark.return_value = {
            "strength": 0.6
        }

        # Create communication system
        comm_system = UnifiedBeeCommunicationSystem(BeeSpeciesType.BOMBUS_TERRESTRIS)
        comm_system.communication_system = mock_instance

        # Test scent marking method
        result = comm_system.potentially_leave_scent_mark(123, 0.8, 40.0, 300.0)

        assert result["scent_marked"] is True
        assert result["scent_strength"] == 0.5
        assert result["memory_updated"] is True

    def test_honey_bee_scent_marking_compatibility(self):
        """Test honey bee response to scent marking method"""
        comm_system = UnifiedBeeCommunicationSystem(BeeSpeciesType.APIS_MELLIFERA)

        # Honey bees should return default response
        result = comm_system.potentially_leave_scent_mark(123, 0.8, 40.0, 300.0)

        assert result["scent_marked"] is False
        assert result["scent_strength"] == 0.0
        assert result["memory_updated"] is False

    @patch('bstew.core.honey_bee_communication.HoneyBeeCommunicationSystem')
    def test_honey_bee_communication_metrics(self, mock_honey_system):
        """Test honey bee communication metrics"""
        # Setup mock
        mock_instance = Mock()
        mock_honey_system.return_value = mock_instance
        mock_instance.active_dances = {"dance1": Mock(followers=set([1, 2, 3]))}
        mock_instance.recruitment_success_rates = {1: 0.8, 2: 0.6}
        mock_instance.colony_knowledge = {1: {}, 2: {}, 3: {}}
        mock_instance.communication_records = [{}, {}, {}, {}]

        comm_system = UnifiedBeeCommunicationSystem(BeeSpeciesType.APIS_MELLIFERA)
        comm_system.communication_system = mock_instance

        metrics = comm_system.get_communication_metrics()

        assert metrics["active_dances"] == 1
        assert metrics["total_recruited_bees"] == 3
        assert metrics["average_dance_success_rate"] == 0.7  # (0.8 + 0.6) / 2
        assert metrics["colony_knowledge_patches"] == 3
        assert metrics["communication_events"] == 4

    @patch('bstew.core.bumblebee_communication.BumblebeeCommunicationSystem')
    def test_bumblebee_communication_metrics(self, mock_bumblebee_system):
        """Test bumblebee communication metrics"""
        # Setup mock
        mock_instance = Mock()
        mock_bumblebee_system.return_value = mock_instance
        mock_instance.active_scents = [1, 2]
        mock_instance.recent_arousal_events = [1]
        mock_instance.scent_following_success_rate = 0.3
        mock_instance.patch_memory = {1: {}, 2: {}}
        mock_instance.communication_log = [1, 2, 3]

        comm_system = UnifiedBeeCommunicationSystem(BeeSpeciesType.BOMBUS_TERRESTRIS)
        comm_system.communication_system = mock_instance

        metrics = comm_system.get_communication_metrics()

        assert metrics["active_scents"] == 2
        assert metrics["recent_arousal_events"] == 1
        assert metrics["scent_following_rate"] == 0.3
        assert metrics["memory_patches"] == 2
        assert metrics["communication_events"] == 3


class TestMultiSpeciesSystem:
    """Test multi-species system functionality"""

    def test_create_multi_species_simulation(self):
        """Test creating multi-species simulation"""
        species_list = [
            BeeSpeciesType.APIS_MELLIFERA,
            BeeSpeciesType.BOMBUS_TERRESTRIS,
            BeeSpeciesType.BOMBUS_LAPIDARIUS
        ]

        simulation_config = create_multi_species_simulation(species_list)

        assert len(simulation_config) == 3
        assert BeeSpeciesType.APIS_MELLIFERA in simulation_config
        assert BeeSpeciesType.BOMBUS_TERRESTRIS in simulation_config
        assert BeeSpeciesType.BOMBUS_LAPIDARIUS in simulation_config

        # Check honey bee configuration
        honey_config = simulation_config[BeeSpeciesType.APIS_MELLIFERA]
        assert honey_config["is_honey_bee"] is True
        assert honey_config["is_bumblebee"] is False

        # Check bumblebee configuration
        bumble_config = simulation_config[BeeSpeciesType.BOMBUS_TERRESTRIS]
        assert bumble_config["is_honey_bee"] is False
        assert bumble_config["is_bumblebee"] is True

    def test_create_multi_species_communication(self):
        """Test creating multiple communication systems"""
        species_types = [
            BeeSpeciesType.APIS_MELLIFERA,
            BeeSpeciesType.BOMBUS_TERRESTRIS
        ]

        comm_systems = create_multi_species_communication(species_types)

        assert len(comm_systems) == 2
        assert isinstance(comm_systems[BeeSpeciesType.APIS_MELLIFERA], UnifiedBeeCommunicationSystem)
        assert isinstance(comm_systems[BeeSpeciesType.BOMBUS_TERRESTRIS], UnifiedBeeCommunicationSystem)

        # Verify correct species configuration
        honey_system = comm_systems[BeeSpeciesType.APIS_MELLIFERA]
        assert honey_system.is_honey_bee is True

        bumble_system = comm_systems[BeeSpeciesType.BOMBUS_TERRESTRIS]
        assert bumble_system.is_bumblebee is True

    def test_invalid_species_combination(self):
        """Test invalid species combination raises error"""
        invalid_species = [
            BeeSpeciesType.APIS_MELLIFERA,
            BeeSpeciesType.APIS_MELLIFERA  # Duplicate honey bee
        ]

        with pytest.raises(ValueError, match="Invalid species combination"):
            create_multi_species_simulation(invalid_species)


class TestBackwardCompatibility:
    """Test backward compatibility with existing code"""

    def test_communication_system_factory(self):
        """Test communication system factory functions"""
        honey_system = create_communication_system(BeeSpeciesType.APIS_MELLIFERA)
        bumble_system = create_communication_system(BeeSpeciesType.BOMBUS_TERRESTRIS)

        assert isinstance(honey_system, UnifiedBeeCommunicationSystem)
        assert isinstance(bumble_system, UnifiedBeeCommunicationSystem)

        assert honey_system.is_honey_bee is True
        assert bumble_system.is_bumblebee is True

    def test_species_info_compatibility(self):
        """Test species information retrieval"""
        honey_system = create_communication_system(BeeSpeciesType.APIS_MELLIFERA)
        info = honey_system.get_species_info()

        # Check expected fields exist
        required_fields = [
            "species_type", "common_name", "scientific_name",
            "communication_type", "uses_dance_communication",
            "uses_scent_communication", "social_information_sharing",
            "individual_decision_weight", "max_foraging_distance",
            "colony_size_range"
        ]

        for field in required_fields:
            assert field in info, f"Missing field: {field}"

        # Check values are reasonable
        assert "Apis" in info["scientific_name"]
        assert info["uses_dance_communication"] is True
        assert info["social_information_sharing"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
