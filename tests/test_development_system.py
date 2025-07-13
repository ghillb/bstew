"""
Test cases for the development system integration
"""

from src.bstew.components.development import (
    DevelopmentSystem,
    DevelopmentParameters,
    DevelopingBee,
    DevelopmentStage,
    DeathCause,
)
from src.bstew.components.genetics import GeneticSystem, Sex


class TestDevelopmentSystem:
    """Test development system functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.params = DevelopmentParameters()
        self.dev_system = DevelopmentSystem(self.params)
        self.genetic_system = GeneticSystem(initial_allele_count=50)

    def test_egg_creation(self):
        """Test egg creation and tracking"""
        # Create egg without genotype
        egg_id = self.dev_system.add_egg()
        assert egg_id == 1
        assert len(self.dev_system.developing_bees) == 1

        egg = self.dev_system.developing_bees[egg_id]
        assert egg.stage == DevelopmentStage.EGG
        assert egg.weight_mg == self.params.dev_weight_egg
        assert egg.is_alive()

    def test_egg_with_genotype(self):
        """Test egg creation with genetic system"""
        genotype = self.genetic_system.create_founder_genotype(sex=Sex.FEMALE)
        egg_id = self.dev_system.add_egg(genotype)

        egg = self.dev_system.developing_bees[egg_id]
        assert egg.genotype == genotype
        assert egg.get_target_bee_type() == "worker"

    def test_diploid_male_lethality(self):
        """Test diploid male death during development"""
        # Create genotype that would be diploid male
        genotype = self.genetic_system.create_founder_genotype(sex=Sex.FEMALE)
        # Force diploid male condition
        genotype.alleles = [genotype.alleles[0], genotype.alleles[0]]  # Same alleles
        genotype.sex = Sex.DIPLOID_MALE

        egg_id = self.dev_system.add_egg(genotype)
        egg = self.dev_system.developing_bees[egg_id]

        assert not egg.is_alive()
        assert egg.death_cause == DeathCause.DIPLOID_MALE

    def test_temperature_effects(self):
        """Test temperature effects on development"""
        # Test optimal temperature
        optimal_effect = self.dev_system.calculate_temperature_effect(35.0)
        assert optimal_effect == 1.0

        # Test sub-optimal temperature
        sub_optimal_effect = self.dev_system.calculate_temperature_effect(25.0)
        assert 0.0 < sub_optimal_effect < 1.0

        # Test extreme temperature
        extreme_effect = self.dev_system.calculate_temperature_effect(5.0)
        assert extreme_effect == 0.0

    def test_stage_transitions(self):
        """Test development stage transitions"""
        egg_id = self.dev_system.add_egg()
        egg = self.dev_system.developing_bees[egg_id]

        # Simulate egg development
        egg.cumul_time_egg = self.params.dev_age_hatching_min
        egg.cumul_incubation_received = (
            self.params.incubation_per_day_egg * self.params.dev_age_hatching_min
        )

        transition = self.dev_system.check_stage_transition(egg)
        assert transition is not None
        assert transition["to_stage"] == DevelopmentStage.LARVA
        assert egg.stage == DevelopmentStage.LARVA

    def test_care_demand_calculation(self):
        """Test care demand calculation"""
        # Add eggs and larvae
        for _ in range(5):
            self.dev_system.add_egg()

        demand = self.dev_system.calculate_care_demand()
        expected_energy = 5 * self.params.energy_per_day_egg
        expected_incubation = 5 * self.params.incubation_per_day_egg

        assert demand["energy"] == expected_energy
        assert demand["incubation"] == expected_incubation

    def test_development_step_processing(self):
        """Test full development step processing"""
        # Create egg
        egg_id = self.dev_system.add_egg()

        # Provide adequate care
        care = {"energy": 1.0, "incubation": 1.0, "feeding": 1.0}

        # Process several steps
        for _ in range(5):
            self.dev_system.step(35.0, care)  # Optimal temperature

        egg = self.dev_system.developing_bees[egg_id]
        assert egg.cumul_time_egg > 0
        assert egg.cumul_incubation_received > 0
        assert egg.cumul_energy_received > 0

    def test_insufficient_care_mortality(self):
        """Test mortality from insufficient care"""
        egg_id = self.dev_system.add_egg()

        # Provide inadequate care
        care = {"energy": 0.0, "incubation": 0.0, "feeding": 0.0}

        # Process many steps with no care
        deaths = []
        for _ in range(10):
            results = self.dev_system.step(35.0, care)
            deaths.extend(results.get("deaths", []))

        # Check that the egg died and was removed
        assert egg_id not in self.dev_system.developing_bees

        # Check that there was at least one death with the correct cause
        assert len(deaths) > 0
        dead_egg = deaths[0]  # Should be our egg
        assert dead_egg.death_cause == DeathCause.INSUFFICIENT_INCUBATION

    def test_population_tracking(self):
        """Test population tracking by stage"""
        # Add different stages
        for _ in range(3):
            self.dev_system.add_egg()

        # Manually create larva for testing
        larva_id = self.dev_system.add_egg()
        larva = self.dev_system.developing_bees[larva_id]
        larva.stage = DevelopmentStage.LARVA

        population = self.dev_system.get_population_by_stage()
        assert population[DevelopmentStage.EGG] == 3
        assert population[DevelopmentStage.LARVA] == 1
        assert population[DevelopmentStage.PUPA] == 0

    def test_mortality_statistics(self):
        """Test mortality statistics tracking"""
        # Create eggs
        for _ in range(5):
            self.dev_system.add_egg()

        # Process step with no care to cause natural deaths
        care = {"energy": 0.0, "incubation": 0.0, "feeding": 0.0}

        # Process multiple steps to cause some deaths from insufficient care
        total_deaths = 0
        for _ in range(10):
            results = self.dev_system.step(35.0, care)
            total_deaths += len(results.get("deaths", []))

        # Check that some deaths occurred
        mortality_stats = self.dev_system.get_mortality_statistics()
        assert mortality_stats["total_deaths"] >= 2
        assert (
            mortality_stats["deaths_by_cause"][DeathCause.INSUFFICIENT_INCUBATION.value]
            >= 2
        )

    def test_environmental_stress(self):
        """Test environmental stress application"""
        # Add eggs
        for _ in range(3):
            self.dev_system.add_egg()

        # Apply high stress
        self.dev_system.apply_environmental_stress("heat_wave", 1.5)

        # Check stress was applied
        for bee in self.dev_system.developing_bees.values():
            assert "heat_wave" in bee.stress_factors
            assert bee.stress_factors["heat_wave"] == 1.5

    def test_end_season_mortality(self):
        """Test end-of-season mortality"""
        # Add eggs
        for _ in range(3):
            self.dev_system.add_egg()

        # Apply end-season mortality
        self.dev_system.end_season_mortality()

        # Check all are dead
        for bee in self.dev_system.developing_bees.values():
            assert not bee.is_alive()
            assert bee.death_cause == DeathCause.SEASON_END

    def test_development_summary(self):
        """Test comprehensive development summary"""
        # Add mixed population
        for _ in range(5):
            self.dev_system.add_egg()

        summary = self.dev_system.get_development_summary()

        assert "population_by_stage" in summary
        assert "total_developing" in summary
        assert "mortality_statistics" in summary
        assert "care_demand" in summary
        assert "development_success_rate" in summary

        assert summary["total_developing"] == 5
        assert summary["population_by_stage"][DevelopmentStage.EGG] == 5


class TestDevelopmentParameters:
    """Test development parameters"""

    def test_default_parameters(self):
        """Test default parameter values"""
        params = DevelopmentParameters()

        assert params.dev_age_hatching_min == 3.0
        assert params.dev_age_pupation_min == 14.0
        assert params.dev_age_emerging_min == 21.0
        assert params.dev_weight_egg == 0.15
        assert params.temperature_optimal == 35.0

    def test_custom_parameters(self):
        """Test custom parameter values"""
        params = DevelopmentParameters(
            dev_age_hatching_min=4.0, temperature_optimal=32.0
        )

        assert params.dev_age_hatching_min == 4.0
        assert params.temperature_optimal == 32.0
        assert params.dev_age_pupation_min == 14.0  # Default value


class TestDevelopingBee:
    """Test individual developing bee"""

    def test_bee_creation(self):
        """Test developing bee creation"""
        bee = DevelopingBee(unique_id=1, genotype=None, stage=DevelopmentStage.EGG)

        assert bee.unique_id == 1
        assert bee.stage == DevelopmentStage.EGG
        assert bee.is_alive()
        assert bee.weight_mg == 0.15  # Default egg weight

    def test_stress_tracking(self):
        """Test stress factor tracking"""
        bee = DevelopingBee(unique_id=1, genotype=None, stage=DevelopmentStage.LARVA)

        bee.add_stress("temperature", 0.5)
        bee.add_stress("nutrition", 0.3)

        assert bee.stress_factors["temperature"] == 0.5
        assert bee.stress_factors["nutrition"] == 0.3
        assert bee.get_total_stress() == 0.8

    def test_target_bee_type(self):
        """Test target bee type determination"""
        genetic_system = GeneticSystem()

        # Female genotype
        female_genotype = genetic_system.create_founder_genotype(sex=Sex.FEMALE)
        bee_female = DevelopingBee(
            unique_id=1, genotype=female_genotype, stage=DevelopmentStage.EGG
        )
        assert bee_female.get_target_bee_type() == "worker"

        # Male genotype
        male_genotype = genetic_system.create_founder_genotype(sex=Sex.MALE)
        bee_male = DevelopingBee(
            unique_id=2, genotype=male_genotype, stage=DevelopmentStage.EGG
        )
        assert bee_male.get_target_bee_type() == "drone"
