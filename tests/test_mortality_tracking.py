"""
Test cases for the mortality tracking system
"""

from unittest.mock import Mock
import tempfile
import json
import os

from src.bstew.components.mortality_tracking import (
    MortalityTracker,
    MortalityEvent,
    DetailedDeathCause,
    MortalityCategory,
    MortalityStatistics,
)
from src.bstew.components.genetics import Genotype, Sex, Ploidy, Allele


class TestMortalityEvent:
    """Test mortality event functionality"""

    def test_mortality_event_creation(self):
        """Test creating mortality event"""
        event = MortalityEvent(
            bee_id=123,
            death_day=150,
            death_step=3600,
            age_at_death=25.5,
            bee_role="forager",
            bee_sex="female",
            primary_cause=DetailedDeathCause.OLD_AGE,
        )

        assert event.bee_id == 123
        assert event.death_day == 150
        assert event.age_at_death == 25.5
        assert event.primary_cause == DetailedDeathCause.OLD_AGE
        assert event.mortality_category == MortalityCategory.UNKNOWN  # Default

    def test_mortality_event_to_dict(self):
        """Test converting mortality event to dictionary"""
        event = MortalityEvent(
            bee_id=456,
            death_day=100,
            death_step=2400,
            age_at_death=15.0,
            bee_role="worker",
            bee_sex="female",
            primary_cause=DetailedDeathCause.STARVATION,
            contributing_factors=[DetailedDeathCause.COLD_EXPOSURE],
            environmental_conditions={"temperature": 5.0, "rainfall": 0.0},
        )

        event_dict = event.to_dict()

        assert event_dict["bee_id"] == 456
        assert event_dict["primary_cause"] == "starvation"
        assert event_dict["contributing_factors"] == ["cold_exposure"]
        assert event_dict["environmental_conditions"]["temperature"] == 5.0


class TestMortalityTracker:
    """Test mortality tracking system functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = MortalityTracker(max_history_days=30)

    def create_mock_bee(
        self,
        bee_id: int,
        role: str = "worker",
        age: float = 20.0,
        energy: float = 50.0,
        sex: Sex = Sex.FEMALE,
    ):
        """Create mock bee agent for testing"""
        mock_bee = Mock()
        mock_bee.unique_id = bee_id
        mock_bee.age = age
        mock_bee.energy = energy
        mock_bee.disease_load = 0.0
        mock_bee.foraging_efficiency = 1.0
        mock_bee.longevity_factor = 1.0
        mock_bee.current_load = 0.0
        mock_bee.role = Mock()
        mock_bee.role.value = role
        mock_bee.location = (100.0, 200.0)

        # Mock genotype
        if sex == Sex.FEMALE:
            alleles = [
                Allele(allele_id=1, origin="maternal"),
                Allele(allele_id=2, origin="paternal"),
            ]
            mock_bee.genotype = Genotype(
                alleles=alleles, ploidy=Ploidy.DIPLOID, sex=sex, mtDNA=1
            )
        else:
            alleles = [Allele(allele_id=1, origin="maternal")]
            mock_bee.genotype = Genotype(
                alleles=alleles, ploidy=Ploidy.HAPLOID, sex=sex, mtDNA=1
            )

        # Mock colony
        mock_bee.colony = Mock()
        mock_bee.colony.species = "Bombus_terrestris"

        # Mock development history with proper values
        mock_bee.development_history = {
            "total_development_time": 21.0,  # realistic development time
            "care_received": 18.0,
        }

        return mock_bee

    def test_tracker_initialization(self):
        """Test mortality tracker initialization"""
        assert len(self.tracker.mortality_events) == 0
        assert len(self.tracker.daily_mortality) == 0
        assert self.tracker.max_history_days == 30
        assert isinstance(self.tracker.statistics, MortalityStatistics)

    def test_record_basic_death(self):
        """Test recording basic death"""
        mock_bee = self.create_mock_bee(1, "forager", 25.0)

        event = self.tracker.record_death(
            mock_bee, DetailedDeathCause.OLD_AGE, simulation_step=100
        )

        assert len(self.tracker.mortality_events) == 1
        assert event.bee_id == 1
        assert event.primary_cause == DetailedDeathCause.OLD_AGE
        assert event.bee_role == "forager"
        assert event.age_at_death == 25.0

    def test_record_death_with_contributing_factors(self):
        """Test recording death with contributing factors"""
        mock_bee = self.create_mock_bee(2, "worker", 15.0, energy=5.0)

        event = self.tracker.record_death(
            mock_bee,
            DetailedDeathCause.STARVATION,
            contributing_factors=[
                DetailedDeathCause.COLD_EXPOSURE,
                DetailedDeathCause.EXHAUSTION,
            ],
            environmental_conditions={"temperature": 2.0, "wind_speed": 15.0},
            simulation_step=200,
        )

        assert len(event.contributing_factors) == 2
        assert DetailedDeathCause.COLD_EXPOSURE in event.contributing_factors
        assert event.environmental_conditions["temperature"] == 2.0
        assert event.bee_state["energy"] == 5.0

    def test_death_cause_categorization(self):
        """Test death cause categorization"""
        mock_bee = self.create_mock_bee(3)

        # Test environmental cause
        event1 = self.tracker.record_death(mock_bee, DetailedDeathCause.COLD_EXPOSURE)
        assert event1.mortality_category == MortalityCategory.ENVIRONMENTAL

        # Test predation cause
        event2 = self.tracker.record_death(mock_bee, DetailedDeathCause.BIRD_PREDATION)
        assert event2.mortality_category == MortalityCategory.PREDATION

        # Test biological cause
        event3 = self.tracker.record_death(
            mock_bee, DetailedDeathCause.DISEASE_INFECTION
        )
        assert event3.mortality_category == MortalityCategory.BIOLOGICAL

    def test_statistics_update(self):
        """Test statistics update with multiple deaths"""
        # Record several deaths
        for i in range(5):
            mock_bee = self.create_mock_bee(i, "worker", 20.0 + i)
            self.tracker.record_death(mock_bee, DetailedDeathCause.OLD_AGE)

        for i in range(3):
            mock_bee = self.create_mock_bee(i + 10, "forager", 30.0 + i)
            self.tracker.record_death(mock_bee, DetailedDeathCause.EXHAUSTION)

        assert self.tracker.statistics.total_deaths == 8
        assert self.tracker.statistics.deaths_by_cause["old_age"] == 5
        assert self.tracker.statistics.deaths_by_cause["exhaustion"] == 3
        assert self.tracker.statistics.deaths_by_role["worker"] == 5
        assert self.tracker.statistics.deaths_by_role["forager"] == 3

    def test_age_group_classification(self):
        """Test age group classification"""
        # Young bee
        mock_bee1 = self.create_mock_bee(1, "worker", 5.0)
        self.tracker.record_death(mock_bee1, DetailedDeathCause.DISEASE_INFECTION)

        # Adult bee
        mock_bee2 = self.create_mock_bee(2, "worker", 15.0)
        self.tracker.record_death(mock_bee2, DetailedDeathCause.EXHAUSTION)

        # Old bee
        mock_bee3 = self.create_mock_bee(3, "worker", 35.0)
        self.tracker.record_death(mock_bee3, DetailedDeathCause.OLD_AGE)

        assert self.tracker.statistics.deaths_by_age_group["young"] == 1
        assert self.tracker.statistics.deaths_by_age_group["adult"] == 1
        assert self.tracker.statistics.deaths_by_age_group["old"] == 1

    def test_seasonal_classification(self):
        """Test seasonal death classification"""
        mock_bee = self.create_mock_bee(1)

        # Winter death (day 50)
        self.tracker.record_death(
            mock_bee, DetailedDeathCause.COLD_EXPOSURE, simulation_step=50 * 24
        )

        # Spring death (day 120)
        self.tracker.record_death(
            mock_bee, DetailedDeathCause.OLD_AGE, simulation_step=120 * 24
        )

        # Summer death (day 200)
        self.tracker.record_death(
            mock_bee, DetailedDeathCause.HEAT_STRESS, simulation_step=200 * 24
        )

        # Autumn death (day 300)
        self.tracker.record_death(
            mock_bee, DetailedDeathCause.STARVATION, simulation_step=300 * 24
        )

        assert self.tracker.statistics.deaths_by_season["winter"] == 1
        assert self.tracker.statistics.deaths_by_season["spring"] == 1
        assert self.tracker.statistics.deaths_by_season["summer"] == 1
        assert self.tracker.statistics.deaths_by_season["autumn"] == 1

    def test_environmental_conditions_tracking(self):
        """Test environmental conditions tracking"""
        # Add environmental data
        self.tracker.update_environmental_conditions(
            1, {"temperature": 20.0, "rainfall": 0.0}
        )
        self.tracker.update_environmental_conditions(
            2, {"temperature": 25.0, "rainfall": 5.0}
        )
        self.tracker.update_environmental_conditions(
            3, {"temperature": 15.0, "rainfall": 0.0}
        )

        assert len(self.tracker.environmental_history) == 3
        assert self.tracker.environmental_history[2]["temperature"] == 25.0

    def test_mortality_risk_assessment(self):
        """Test mortality risk assessment"""
        # Create mortality events over several days
        for day in range(1, 8):
            for i in range(2):  # 2 deaths per day
                mock_bee = self.create_mock_bee(day * 10 + i, age=20.0)
                self.tracker.record_death(
                    mock_bee, DetailedDeathCause.OLD_AGE, simulation_step=day * 24
                )

        # Add environmental conditions
        for day in range(1, 8):
            self.tracker.update_environmental_conditions(day, {"temperature": 20.0})

        risk_assessment = self.tracker.get_mortality_risk_assessment(
            7, population_size=1000
        )

        assert "overall_risk_level" in risk_assessment
        assert "recent_mortality_rate" in risk_assessment
        assert risk_assessment["recent_deaths_7days"] == 14
        assert "age_specific_risks" in risk_assessment

    def test_environmental_stress_assessment(self):
        """Test environmental stress assessment"""
        # Test cold stress
        cold_conditions = {"temperature": 2.0, "wind_speed": 5.0, "rainfall": 0.0}
        stress = self.tracker._assess_environmental_stress(cold_conditions)
        assert stress["temperature"] == "cold_stress"

        # Test heat stress
        hot_conditions = {"temperature": 40.0, "wind_speed": 5.0, "rainfall": 0.0}
        stress = self.tracker._assess_environmental_stress(hot_conditions)
        assert stress["temperature"] == "heat_stress"

        # Test high wind
        windy_conditions = {"temperature": 20.0, "wind_speed": 25.0, "rainfall": 0.0}
        stress = self.tracker._assess_environmental_stress(windy_conditions)
        assert stress["wind"] == "high"

    def test_survival_curves_calculation(self):
        """Test survival curves calculation"""
        # Create deaths with different ages for workers
        worker_ages = [10, 15, 20, 25, 30, 35, 40]
        for i, age in enumerate(worker_ages):
            mock_bee = self.create_mock_bee(i, "worker", age)
            self.tracker.record_death(mock_bee, DetailedDeathCause.OLD_AGE)

        # Create deaths for foragers
        forager_ages = [12, 18, 22, 28, 32, 38]
        for i, age in enumerate(forager_ages):
            mock_bee = self.create_mock_bee(i + 100, "forager", age)
            self.tracker.record_death(mock_bee, DetailedDeathCause.EXHAUSTION)

        survival_curves = self.tracker.calculate_survival_curves()

        assert "worker" in survival_curves
        assert "forager" in survival_curves

        # Check that survival curves start at 1.0 and decrease
        worker_curve = survival_curves["worker"]
        assert worker_curve[0] == 1.0
        assert worker_curve[-1] < worker_curve[0]

    def test_genetic_information_tracking(self):
        """Test genetic information tracking in mortality events"""
        # Create bee with diploid male genotype
        mock_bee = self.create_mock_bee(1, sex=Sex.FEMALE)  # Start as female
        # Make it diploid male (lethal condition) by setting identical alleles
        mock_bee.genotype.alleles = [
            Allele(allele_id=1, origin="maternal"),
            Allele(allele_id=1, origin="paternal"),
        ]  # Same alleles = diploid male
        mock_bee.genotype.sex = Sex.DIPLOID_MALE  # Manually set to diploid male

        event = self.tracker.record_death(mock_bee, DetailedDeathCause.DIPLOID_MALE)

        assert event.genotype_summary is not None
        assert event.genotype_summary["sex"] == "diploid_male"
        assert event.genotype_summary["ploidy"] == 2
        assert event.genotype_summary["is_diploid_male"]

    def test_export_mortality_data(self):
        """Test exporting mortality data"""
        # Create some mortality events
        for i in range(5):
            mock_bee = self.create_mock_bee(i, "worker", 20.0 + i)
            # Don't set development_history to avoid Mock serialization issues
            if hasattr(mock_bee, "development_history"):
                delattr(mock_bee, "development_history")
            self.tracker.record_death(mock_bee, DetailedDeathCause.OLD_AGE)

        # Export to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.tracker.export_mortality_data(tmp_path, include_individual_events=True)

            # Read back and verify
            with open(tmp_path, "r") as f:
                data = json.load(f)

            assert data["metadata"]["total_events"] == 5
            assert data["statistics"]["total_deaths"] == 5
            assert len(data["individual_events"]) == 5
            assert "survival_curves" in data

        finally:
            os.unlink(tmp_path)

    def test_mortality_summary(self):
        """Test mortality summary generation"""
        # Create diverse mortality events
        causes = [
            DetailedDeathCause.OLD_AGE,
            DetailedDeathCause.STARVATION,
            DetailedDeathCause.COLD_EXPOSURE,
        ]
        roles = ["worker", "forager", "drone"]

        for i in range(9):
            mock_bee = self.create_mock_bee(i, roles[i % 3], 20.0 + i)
            self.tracker.record_death(mock_bee, causes[i % 3])

        summary = self.tracker.get_mortality_summary()

        assert summary["total_deaths"] == 9
        assert len(summary["top_causes"]) <= 5
        assert "average_lifespans" in summary
        assert "deaths_by_category" in summary

    def test_cleanup_old_data(self):
        """Test cleanup of old mortality data"""
        # Create events over many days
        for day in range(1, 50):
            mock_bee = self.create_mock_bee(day, age=20.0)
            self.tracker.record_death(
                mock_bee, DetailedDeathCause.OLD_AGE, simulation_step=day * 24
            )
            self.tracker.update_environmental_conditions(day, {"temperature": 20.0})

        # Should have 49 events
        assert len(self.tracker.mortality_events) == 49
        assert len(self.tracker.environmental_history) == 49

        # Cleanup old data (keep last 30 days)
        self.tracker.cleanup_old_data(current_day=50)

        # Should now have fewer events (only last 30 days)
        assert len(self.tracker.mortality_events) <= 30
        assert len(self.tracker.environmental_history) <= 30

    def test_species_specific_tracking(self):
        """Test species-specific mortality tracking"""
        # Create bees from different species
        species_list = ["Bombus_terrestris", "Bombus_lucorum", "Bombus_lapidarius"]

        for i, species in enumerate(species_list):
            for j in range(7):  # Need more data points for survival curves
                mock_bee = self.create_mock_bee(i * 10 + j, age=20.0 + j)
                mock_bee.colony.species = species
                self.tracker.record_death(mock_bee, DetailedDeathCause.OLD_AGE)

        assert self.tracker.statistics.deaths_by_species["Bombus_terrestris"] == 7
        assert self.tracker.statistics.deaths_by_species["Bombus_lucorum"] == 7
        assert self.tracker.statistics.deaths_by_species["Bombus_lapidarius"] == 7

        # Test species-specific survival curves
        terrestris_curves = self.tracker.calculate_survival_curves(
            species="Bombus_terrestris"
        )
        assert len(terrestris_curves) > 0

    def test_development_history_tracking(self):
        """Test tracking of development history in mortality events"""
        mock_bee = self.create_mock_bee(1, age=25.0)

        # Add development history
        mock_bee.development_history = {
            "total_development_time": 21.0,
            "care_received": 50.0,
            "stress_factors": {"temperature": 0.1},
        }

        event = self.tracker.record_death(mock_bee, DetailedDeathCause.OLD_AGE)

        assert event.development_time == 21.0
        assert event.development_quality > 0.0  # Should be calculated from care/time

    def test_mortality_correlations(self):
        """Test mortality correlation calculations"""
        # Create mortality events with environmental data
        for day in range(1, 40):
            # More deaths on colder days
            death_count = 3 if day % 10 < 5 else 1  # Pattern for correlation
            temp = 15.0 if day % 10 < 5 else 25.0  # Cold days have more deaths

            for i in range(death_count):
                mock_bee = self.create_mock_bee(day * 10 + i, age=20.0)
                self.tracker.record_death(
                    mock_bee, DetailedDeathCause.COLD_EXPOSURE, simulation_step=day * 24
                )

            self.tracker.update_environmental_conditions(day, {"temperature": temp})

        # Should detect some correlation
        assert hasattr(self.tracker.statistics, "mortality_temperature_correlation")

    def test_high_mortality_detection(self):
        """Test detection of high mortality events"""
        # Create normal mortality
        for i in range(5):
            mock_bee = self.create_mock_bee(i, age=20.0)
            self.tracker.record_death(
                mock_bee, DetailedDeathCause.OLD_AGE, simulation_step=24
            )

        # Create high mortality event
        for i in range(20):  # Many deaths in one day
            mock_bee = self.create_mock_bee(i + 100, age=20.0)
            self.tracker.record_death(
                mock_bee, DetailedDeathCause.PESTICIDE_EXPOSURE, simulation_step=48
            )

        risk_assessment = self.tracker.get_mortality_risk_assessment(
            2, population_size=200
        )

        # Should detect high mortality
        assert risk_assessment["recent_deaths_7days"] > 20
        assert "pesticide_exposure" in risk_assessment["primary_concerning_causes"]
