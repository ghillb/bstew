"""
Test cases for the predation system
"""

from unittest.mock import Mock

from src.bstew.components.predation import (
    PredationSystem,
    PredatorAgent,
    PredatorType,
    PredatorParameters,
)


class TestPredationSystem:
    """Test predation system functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.landscape_bounds = (0, 0, 10000, 10000)  # 10km x 10km
        self.predation_system = PredationSystem(self.landscape_bounds)

    def test_system_initialization(self):
        """Test predation system initialization"""
        assert len(self.predation_system.predators) >= 1  # At least one badger
        assert self.predation_system.landscape_bounds == (0, 0, 10000, 10000)
        assert self.predation_system.next_predator_id > 1

    def test_badger_creation(self):
        """Test badger creation"""
        initial_count = len(self.predation_system.predators)
        badger_id = self.predation_system.create_badger()

        assert len(self.predation_system.predators) == initial_count + 1
        assert badger_id in self.predation_system.predators

        badger = self.predation_system.predators[badger_id]
        assert badger.predator_type == PredatorType.BADGER
        assert badger.territory_radius == 750.0  # Half of territory size

    def test_badger_territory_check(self):
        """Test badger territory checking"""
        badger_id = self.predation_system.create_badger()
        badger = self.predation_system.predators[badger_id]

        # Test location within territory
        assert badger.is_in_territory(badger.territory_center)

        # Test location outside territory
        far_location = (
            badger.territory_center[0] + 2000,
            badger.territory_center[1] + 2000,
        )
        assert not badger.is_in_territory(far_location)

    def test_distance_calculation(self):
        """Test distance calculation"""
        badger_id = self.predation_system.create_badger()
        badger = self.predation_system.predators[badger_id]

        # Test distance to same location
        distance = badger.calculate_distance_to(badger.location)
        assert distance == 0.0

        # Test distance to known location
        target = (badger.location[0] + 100, badger.location[1])
        distance = badger.calculate_distance_to(target)
        assert abs(distance - 100.0) < 1e-10

    def test_seasonal_activity(self):
        """Test seasonal activity patterns"""
        # Test active season
        active_day = 150  # June
        activity = self.predation_system.get_seasonal_activity(active_day)
        assert activity["activity_level"] > 0.0

        # Test inactive season
        inactive_day = 1  # January
        activity = self.predation_system.get_seasonal_activity(inactive_day)
        assert activity["activity_level"] == 0.0

    def test_day_to_month_conversion(self):
        """Test day to month conversion"""
        assert self.predation_system.day_to_month(1) == 1  # January
        assert self.predation_system.day_to_month(32) == 2  # February
        assert self.predation_system.day_to_month(60) == 3  # March
        assert self.predation_system.day_to_month(365) == 12  # December

    def test_colony_encounter_probability(self):
        """Test colony encounter mechanics"""
        # Create mock colony
        mock_colony = Mock()
        mock_colony.location = (5000, 5000)
        mock_colony.get_adult_population.return_value = 1000
        mock_colony.get_brood_count.return_value = 500
        mock_colony.get_total_population.return_value = 1500
        mock_colony.resources = Mock()
        mock_colony.resources.pollen = 1000
        mock_colony.resources.nectar = 2000
        mock_colony.resources.honey = 500

        # Add development system mock
        mock_colony.development_system = Mock()
        mock_colony.development_system.developing_bees = {}

        # Add bees list mock
        mock_colony.bees = []

        # Create badger near colony
        badger_id = self.predation_system.create_badger()
        badger = self.predation_system.predators[badger_id]
        badger.location = (5000, 5100)  # 100m from colony

        # Test encounter (may or may not occur due to probability)
        encounter_count = 0
        for _ in range(100):  # Run multiple times
            encounter = self.predation_system.check_colony_encounter(
                badger, mock_colony, 150
            )
            if encounter:
                encounter_count += 1

        # Should have some encounters given proximity
        assert encounter_count > 0

    def test_colony_destruction(self):
        """Test colony destruction mechanics"""
        # Create mock colony with development system
        mock_colony = Mock()
        mock_colony.location = (5000, 5000)
        mock_colony.get_adult_population.return_value = 1000
        mock_colony.get_brood_count.return_value = 500
        mock_colony.get_total_population.return_value = 1500
        mock_colony.resources = Mock()
        mock_colony.resources.pollen = 1000
        mock_colony.resources.nectar = 2000
        mock_colony.resources.honey = 500

        # Mock bees list
        mock_bees = []
        for i in range(10):
            mock_bee = Mock()
            mock_bee.unique_id = i
            mock_bee.role.value = "worker"
            mock_bee.age = 20
            mock_bee.genotype = None
            mock_bee.status.DEAD = "dead"
            mock_bee.die = Mock()
            mock_bees.append(mock_bee)
        mock_colony.bees = mock_bees

        # Mock development system
        mock_dev_system = Mock()
        mock_dev_system.developing_bees = {}
        mock_colony.development_system = mock_dev_system

        # Create badger
        badger_id = self.predation_system.create_badger()
        badger = self.predation_system.predators[badger_id]

        # Test destruction
        destruction_result = self.predation_system.destroy_colony(
            badger, mock_colony, 150
        )

        assert destruction_result is not None
        assert "bees_killed" in destruction_result
        assert "resources_consumed" in destruction_result
        assert len(destruction_result["bees_killed"]) == 10
        assert badger.colonies_destroyed == 1

    def test_colony_risk_assessment(self):
        """Test colony risk assessment"""
        # Test location with nearby badger
        badger_id = self.predation_system.create_badger()
        badger = self.predation_system.predators[badger_id]

        # Location within foraging range
        test_location = (badger.location[0] + 200, badger.location[1])
        risk_assessment = self.predation_system.get_colony_risk_assessment(
            test_location
        )

        assert "total_risk" in risk_assessment
        assert "nearby_predators" in risk_assessment
        assert "recommendations" in risk_assessment
        assert risk_assessment["total_risk"] > 0.0
        assert len(risk_assessment["nearby_predators"]) > 0

    def test_predator_movement(self):
        """Test predator movement mechanics"""
        badger_id = self.predation_system.create_badger()
        badger = self.predation_system.predators[badger_id]

        initial_location = badger.location

        # Move badger multiple times
        for _ in range(10):
            self.predation_system.move_badger(badger)

        # Should have moved
        assert badger.location != initial_location

        # Should still be in territory
        assert badger.is_in_territory(badger.location)

        # Should have path history
        assert len(badger.path_history) > 0

    def test_predator_pressure_adjustment(self):
        """Test predator pressure adjustment"""
        initial_count = len(self.predation_system.predators)

        # Increase pressure
        self.predation_system.add_predator_pressure(0.5)  # 50% increase
        increased_count = len(self.predation_system.predators)
        assert increased_count > initial_count

        # Decrease pressure
        self.predation_system.remove_predator_pressure(0.3)  # 30% decrease
        decreased_count = len(self.predation_system.predators)
        assert decreased_count < increased_count

    def test_predation_statistics(self):
        """Test predation statistics calculation"""
        # Create mock attack history
        self.predation_system.attack_history = [
            {
                "bees_killed": [1, 2, 3],
                "resources_consumed": {"pollen": 100, "nectar": 200, "honey": 50},
            },
            {
                "bees_killed": [4, 5],
                "resources_consumed": {"pollen": 50, "nectar": 100, "honey": 25},
            },
        ]

        self.predation_system.colonies_attacked = {1, 2}

        stats = self.predation_system.get_predation_statistics()

        assert stats["total_attacks"] == 2
        assert stats["colonies_destroyed"] == 2
        assert stats["total_bees_killed"] == 5
        assert stats["total_resources_consumed"]["pollen"] == 150
        assert stats["total_resources_consumed"]["nectar"] == 300
        assert stats["total_resources_consumed"]["honey"] == 75

    def test_find_colonies_in_range(self):
        """Test finding colonies within predator range"""
        # Create badger
        badger_id = self.predation_system.create_badger()
        badger = self.predation_system.predators[badger_id]

        # Create mock colonies
        nearby_colony = Mock()
        nearby_colony.location = (badger.location[0] + 100, badger.location[1])

        far_colony = Mock()
        far_colony.location = (badger.location[0] + 2000, badger.location[1])

        colonies = [nearby_colony, far_colony]

        # Test range finding
        nearby_colonies = self.predation_system.find_colonies_in_range(badger, colonies)

        assert len(nearby_colonies) == 1
        assert nearby_colonies[0] == nearby_colony

    def test_step_processing(self):
        """Test full step processing"""
        # Create mock colony
        mock_colony = Mock()
        mock_colony.location = (5000, 5000)
        mock_colony.get_adult_population.return_value = 1000
        mock_colony.bees = []
        mock_colony.development_system = Mock()
        mock_colony.development_system.developing_bees = {}
        mock_colony.resources = Mock()
        mock_colony.resources.pollen = 1000
        mock_colony.resources.nectar = 2000
        mock_colony.resources.honey = 500

        colonies = [mock_colony]

        # Process step during active season
        results = self.predation_system.step(150, colonies)  # June

        assert "attacks" in results
        assert "colony_destructions" in results
        assert "predator_movements" in results
        assert "seasonal_activity" in results
        assert results["seasonal_activity"]["activity_level"] > 0.0

    def test_inactive_season_processing(self):
        """Test processing during inactive season"""
        mock_colony = Mock()
        colonies = [mock_colony]

        # Process step during inactive season
        results = self.predation_system.step(1, colonies)  # January

        assert len(results["attacks"]) == 0
        assert len(results["colony_destructions"]) == 0
        assert results["seasonal_activity"]["activity_level"] == 0.0


class TestPredatorAgent:
    """Test individual predator agent"""

    def test_predator_creation(self):
        """Test predator agent creation"""
        predator = PredatorAgent(
            unique_id=1,
            predator_type=PredatorType.BADGER,
            location=(1000, 1000),
            territory_center=(1000, 1000),
            territory_radius=500,
        )

        assert predator.unique_id == 1
        assert predator.predator_type == PredatorType.BADGER
        assert predator.location == (1000, 1000)
        assert predator.territory_center == (1000, 1000)
        assert predator.territory_radius == 500
        assert predator.energy == 100.0
        assert predator.colonies_destroyed == 0

    def test_territory_checking(self):
        """Test territory boundary checking"""
        predator = PredatorAgent(
            unique_id=1,
            predator_type=PredatorType.BADGER,
            location=(1000, 1000),
            territory_center=(1000, 1000),
            territory_radius=500,
        )

        # Test center
        assert predator.is_in_territory((1000, 1000))

        # Test edge
        assert predator.is_in_territory((1500, 1000))
        assert predator.is_in_territory((1000, 1500))

        # Test outside
        assert not predator.is_in_territory((1600, 1000))
        assert not predator.is_in_territory((1000, 1600))

    def test_distance_calculation(self):
        """Test distance calculation methods"""
        predator = PredatorAgent(
            unique_id=1,
            predator_type=PredatorType.BADGER,
            location=(0, 0),
            territory_center=(0, 0),
            territory_radius=500,
        )

        # Test distance calculations
        assert predator.calculate_distance_to((0, 0)) == 0.0
        assert predator.calculate_distance_to((3, 4)) == 5.0  # 3-4-5 triangle
        assert predator.calculate_distance_to((100, 0)) == 100.0


class TestPredatorParameters:
    """Test predator parameters"""

    def test_default_parameters(self):
        """Test default parameter values"""
        params = PredatorParameters()

        assert params.foraging_range_m == 735.0
        assert params.encounter_probability == 0.19
        assert params.dig_up_probability == 0.1
        assert params.territory_size_m == 1500.0
        assert params.destruction_completeness == 1.0
        assert params.resource_consumption == 0.8
        assert 4 in params.active_months  # April
        assert 10 in params.active_months  # October
        assert 1 not in params.active_months  # January

    def test_custom_parameters(self):
        """Test custom parameter values"""
        params = PredatorParameters(
            foraging_range_m=1000.0,
            encounter_probability=0.25,
            active_months=[5, 6, 7, 8],
        )

        assert params.foraging_range_m == 1000.0
        assert params.encounter_probability == 0.25
        assert params.active_months == [5, 6, 7, 8]
        assert params.dig_up_probability == 0.1  # Default value
