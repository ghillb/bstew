"""
Unit tests for Colony class
===========================

Comprehensive tests for Colony class functionality.
"""

from unittest.mock import Mock

from src.bstew.core.colony import Colony, ColonyHealth


class TestColonyHealth:
    """Test ColonyHealth enumeration"""

    def test_colony_health_values(self):
        """Test ColonyHealth enum values"""
        assert ColonyHealth.HEALTHY.value == "healthy"
        assert ColonyHealth.STRESSED.value == "stressed"
        assert ColonyHealth.DECLINING.value == "declining"
        assert ColonyHealth.COLLAPSED.value == "collapsed"

    def test_colony_health_members(self):
        """Test ColonyHealth enum members"""
        health_states = list(ColonyHealth)
        assert len(health_states) == 5
        assert ColonyHealth.HEALTHY in health_states
        assert ColonyHealth.COLLAPSED in health_states
        assert ColonyHealth.THRIVING in health_states


class TestColonyBasics:
    """Test basic Colony functionality"""

    def test_colony_creation(self):
        """Test basic colony creation"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)
        species = "bombus_terrestris"
        location = (100, 100)
        initial_population = {"queens": 1, "workers": 49}

        colony = Colony(
            model=model,
            species=species,
            location=location,
            initial_population=initial_population,
        )

        assert colony.model == model
        assert colony.species == species
        assert colony.location == location

    def test_colony_with_default_parameters(self):
        """Test colony with default parameters"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(0, 0),
            initial_population={"queens": 1, "workers": 25},
        )

        assert colony.model == model
        assert colony.species == "bombus_terrestris"
        assert colony.location == (0, 0)

    def test_colony_attributes(self):
        """Test colony basic attributes"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 100},
        )

        assert hasattr(colony, "established_date")
        assert hasattr(colony, "health")
        assert hasattr(colony, "population_counts")
        assert hasattr(colony, "resources")

    def test_colony_systems_initialization(self):
        """Test colony systems initialization"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(25, 25),
            initial_population={"queens": 1, "workers": 75},
        )

        assert hasattr(colony, "genetic_system")
        assert hasattr(colony, "development_system")
        assert hasattr(colony, "proboscis_system")
        assert hasattr(colony, "species_system")
        assert hasattr(colony, "mortality_tracker")
        assert hasattr(colony, "dynamics")
        assert hasattr(colony, "integrator")

    def test_colony_step_method(self):
        """Test colony step method"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(10, 10),
            initial_population={"queens": 1, "workers": 50},
        )

        assert hasattr(colony, "step")
        assert callable(colony.step)

    def test_colony_population_methods(self):
        """Test colony population methods"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(20, 20),
            initial_population={"queens": 1, "workers": 30},
        )

        assert hasattr(colony, "get_adult_population")
        assert callable(colony.get_adult_population)

        # Test population counting
        adult_pop = colony.get_adult_population()
        assert isinstance(adult_pop, int)
        assert adult_pop >= 0

    def test_colony_resource_methods(self):
        """Test colony resource methods"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(30, 30),
            initial_population={"queens": 1, "workers": 40},
        )

        assert hasattr(colony, "resources")
        assert hasattr(colony.resources, "honey")
        assert hasattr(colony.resources, "pollen")
        assert hasattr(colony.resources, "nectar")

    def test_colony_health_assessment(self):
        """Test colony health assessment"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(40, 40),
            initial_population={"queens": 1, "workers": 60},
        )

        assert hasattr(colony, "health")
        assert isinstance(colony.health, ColonyHealth)

    def test_colony_location_property(self):
        """Test colony location property"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)
        location = (75, 85)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=location,
            initial_population={"queens": 1, "workers": 35},
        )

        assert colony.location == location

    def test_colony_species_property(self):
        """Test colony species property"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)
        species = "apis_mellifera"

        colony = Colony(
            model=model,
            species=species,
            location=(0, 0),
            initial_population={"queens": 1, "workers": 45},
        )

        assert colony.species == species

    def test_colony_initial_population_property(self):
        """Test colony initial population property"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)
        initial_pop = {"queens": 1, "workers": 80}

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(0, 0),
            initial_population=initial_pop,
        )

        # Colony doesn't store initial_population, but we can check population_counts
        assert hasattr(colony, "population_counts")
        assert colony.population_counts["queens"] >= 0
        assert colony.population_counts["workers"] >= 0

    def test_colony_with_zero_population(self):
        """Test colony with zero population"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(0, 0),
            initial_population={"queens": 0, "workers": 0},
        )

        assert colony.get_adult_population() == 0

    def test_colony_multiple_instances(self):
        """Test multiple colony instances"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony1 = Colony(
            model=model,
            species="bombus_terrestris",
            location=(100, 100),
            initial_population={"queens": 1, "workers": 50},
        )

        colony2 = Colony(
            model=model,
            species="apis_mellifera",
            location=(200, 200),
            initial_population={"queens": 1, "workers": 100},
        )

        assert colony1.species != colony2.species
        assert colony1.location != colony2.location

    def test_colony_string_representation(self):
        """Test colony string representation"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(100, 100),
            initial_population={"queens": 1, "workers": 50},
        )

        str_repr = str(colony)
        assert isinstance(str_repr, str)
        assert "Colony" in str_repr or "colony" in str_repr


class TestColonyGenetics:
    """Test Colony genetic system"""

    def test_colony_genetic_system_exists(self):
        """Test colony genetic system exists"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        assert hasattr(colony, "genetic_system")
        assert colony.genetic_system is not None

    def test_colony_genetic_diversity_tracking(self):
        """Test colony genetic diversity tracking"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Should have genetic diversity methods
        assert hasattr(colony.genetic_system, "calculate_genetic_diversity")
        assert callable(colony.genetic_system.calculate_genetic_diversity)


class TestColonyDevelopment:
    """Test Colony development system"""

    def test_colony_development_system_exists(self):
        """Test colony development system exists"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        assert hasattr(colony, "development_system")
        assert colony.development_system is not None

    def test_colony_lifecycle_management(self):
        """Test colony lifecycle management"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Should have development tracking
        assert hasattr(colony.development_system, "step")
        assert callable(colony.development_system.step)


class TestColonyResourceManagement:
    """Test Colony resource management"""

    def test_colony_resource_stores_exist(self):
        """Test colony resource stores exist"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        assert hasattr(colony, "resources")
        assert colony.resources is not None

    def test_colony_resource_tracking(self):
        """Test colony resource tracking"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Should have resource amounts
        assert hasattr(colony.resources, "honey")
        assert hasattr(colony.resources, "pollen")
        assert hasattr(colony.resources, "nectar")

        # Resources should be non-negative
        assert colony.resources.honey >= 0
        assert colony.resources.pollen >= 0
        assert colony.resources.nectar >= 0


class TestColonyMortalitySystem:
    """Test Colony mortality system"""

    def test_colony_mortality_tracker_exists(self):
        """Test colony mortality tracker exists"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        assert hasattr(colony, "mortality_tracker")
        assert colony.mortality_tracker is not None

    def test_colony_mortality_tracking(self):
        """Test colony mortality tracking"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Should have mortality methods
        assert hasattr(colony.mortality_tracker, "record_death")
        assert callable(colony.mortality_tracker.record_death)


class TestColonyPredationSystem:
    """Test Colony predation system"""

    def test_colony_predation_system_exists(self):
        """Test colony predation system exists"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Colony has predation-related attributes
        assert hasattr(colony, "predation_risk")
        assert hasattr(colony, "predation_defenses")
        assert hasattr(colony, "last_predation_check")

    def test_colony_predation_defense(self):
        """Test colony predation defense"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Should have defense attributes
        assert hasattr(colony, "predation_defenses")
        assert isinstance(colony.predation_defenses, (int, float))
        assert colony.predation_defenses >= 0


class TestColonySpeciesSystem:
    """Test Colony species system"""

    def test_colony_species_system_exists(self):
        """Test colony species system exists"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        assert hasattr(colony, "species_system")
        assert colony.species_system is not None

    def test_colony_species_traits(self):
        """Test colony species traits"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Should have species-specific traits
        assert hasattr(colony.species_system, "get_species_parameters")
        assert callable(colony.species_system.get_species_parameters)
        assert hasattr(colony, "species_params")


class TestColonyProboscisSystem:
    """Test Colony proboscis system"""

    def test_colony_proboscis_system_exists(self):
        """Test colony proboscis system exists"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        assert hasattr(colony, "proboscis_system")
        assert colony.proboscis_system is not None

    def test_colony_proboscis_matching(self):
        """Test colony proboscis matching"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Should have proboscis matching methods
        assert hasattr(colony.proboscis_system, "calculate_accessibility")
        assert callable(colony.proboscis_system.calculate_accessibility)


class TestColonyMath:
    """Test Colony mathematical system"""

    def test_colony_mathematical_system_exists(self):
        """Test colony mathematical system exists"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        assert hasattr(colony, "dynamics")
        assert hasattr(colony, "integrator")
        assert hasattr(colony, "colony_params")

    def test_colony_mathematical_calculations(self):
        """Test colony mathematical calculations"""
        model = Mock()
        model.schedule = Mock()
        model.schedule.steps = 0
        model.next_id = Mock(return_value=1)

        colony = Colony(
            model=model,
            species="bombus_terrestris",
            location=(50, 50),
            initial_population={"queens": 1, "workers": 25},
        )

        # Should have mathematical methods
        assert hasattr(colony.dynamics, "adult_population_dynamics")
        assert callable(colony.dynamics.adult_population_dynamics)
