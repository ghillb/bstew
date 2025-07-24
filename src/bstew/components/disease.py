"""
Disease and pest modeling for BSTEW
===================================

Implements comprehensive disease dynamics including Varroa mites,
viral infections, Nosema, and other bee pathogens with treatment effects.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict
import random

from ..core.agents import BeeAgent, Forager
from ..core.colony import Colony


class DiseaseType(Enum):
    """Types of bee diseases and pests"""

    VARROA_MITES = "varroa_mites"
    DEFORMED_WING_VIRUS = "deformed_wing_virus"
    BLACK_QUEEN_CELL_VIRUS = "black_queen_cell_virus"
    ACUTE_BEE_PARALYSIS_VIRUS = "acute_bee_paralysis_virus"
    NOSEMA_APIS = "nosema_apis"
    NOSEMA_CERANAE = "nosema_ceranae"
    AMERICAN_FOULBROOD = "american_foulbrood"
    EUROPEAN_FOULBROOD = "european_foulbrood"
    CHALKBROOD = "chalkbrood"
    SMALL_HIVE_BEETLE = "small_hive_beetle"
    WAX_MOTH = "wax_moth"


class TreatmentType(Enum):
    """Types of disease treatments"""

    MITICIDE = "miticide"
    ANTIBIOTIC = "antibiotic"
    ORGANIC_ACID = "organic_acid"
    ESSENTIAL_OIL = "essential_oil"
    MECHANICAL = "mechanical"
    BIOLOGICAL = "biological"
    CULTURAL = "cultural"


class DiseaseState(BaseModel):
    """Disease state for individual bee or colony"""

    model_config = {"validate_assignment": True}

    disease_type: DiseaseType = Field(description="Type of disease or pathogen")
    infection_level: float = Field(
        ge=0.0, le=1.0, description="Infection intensity (0-1 scale)"
    )
    infection_date: int = Field(ge=0, description="Day when infection occurred")
    transmission_rate: float = Field(
        ge=0.0, le=1.0, description="Rate of disease transmission"
    )
    mortality_rate: float = Field(
        ge=0.0, le=1.0, description="Disease-induced mortality rate"
    )
    virulence: float = Field(ge=0.0, le=1.0, description="Disease virulence factor")
    resistance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Host resistance level"
    )

    def get_mortality_probability(self, bee_age: int) -> float:
        """Calculate mortality probability based on infection and age"""
        base_mortality = self.mortality_rate * self.infection_level
        age_factor = 1.0 + (bee_age / 100.0)  # Older bees more susceptible
        resistance_factor = 1.0 - self.resistance

        return min(0.9, base_mortality * age_factor * resistance_factor)

    def progress_infection(
        self, days_elapsed: int, environmental_stress: float = 0.0
    ) -> None:
        """Progress infection over time"""
        # Infection can increase under stress
        stress_factor = 1.0 + environmental_stress
        progression_rate = 0.01 * stress_factor * self.virulence

        self.infection_level = min(
            1.0, self.infection_level + progression_rate * days_elapsed
        )

        # Some diseases may naturally clear
        if self.disease_type in [DiseaseType.EUROPEAN_FOULBROOD]:
            clearance_rate = 0.005
            self.infection_level = max(
                0.0, self.infection_level - clearance_rate * days_elapsed
            )


class TreatmentRecord(BaseModel):
    """Record of disease treatment application"""

    model_config = {"validate_assignment": True}

    treatment_type: TreatmentType = Field(description="Type of treatment applied")
    target_diseases: List[DiseaseType] = Field(
        description="Diseases targeted by treatment"
    )
    application_date: int = Field(ge=0, description="Day when treatment was applied")
    duration: int = Field(ge=1, description="Treatment duration in days")
    efficacy: float = Field(
        ge=0.0, le=1.0, description="Treatment efficacy (0-1 scale)"
    )
    cost: float = Field(ge=0.0, description="Treatment cost")
    side_effects: Dict[str, float] = Field(
        default_factory=dict, description="Treatment side effects"
    )
    resistance_buildup: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Resistance development rate"
    )


class VarroaMiteModel:
    """
    Detailed Varroa mite population dynamics model.

    Based on research literature and implements:
    - Mite reproduction cycles
    - Host-parasite dynamics
    - Seasonal variation
    - Treatment resistance
    """

    def __init__(self, colony: Colony):
        self.colony = colony
        self.mite_population = 0
        self.phoretic_mites = 0  # Mites on adult bees
        self.reproductive_mites = 0  # Mites in brood cells

        # Model parameters
        self.reproduction_rate = 0.02  # Daily rate
        self.mortality_rate = 0.01
        self.brood_preference = 0.8  # Preference for brood over adults
        self.transmission_efficiency = 0.15  # Virus transmission per mite
        self.resistance_level = 0.0  # Treatment resistance

        # Seasonal factors
        self.seasonal_reproduction_factor = 1.0
        self.brood_availability_factor = 1.0

    def update_mite_population(
        self, current_time: int, brood_count: int, adult_count: int
    ) -> None:
        """Update mite population dynamics"""

        # Update seasonal factors
        self._update_seasonal_factors(current_time)

        # Calculate reproduction based on brood availability
        available_brood = brood_count * self.brood_availability_factor
        reproduction_sites = min(available_brood, self.reproductive_mites * 2)

        # Reproduction equation: dV/dt = β * V * (Bc/(Bc + Kv)) - δv * V
        reproduction = (
            self.reproduction_rate
            * self.seasonal_reproduction_factor
            * self.reproductive_mites
            * (reproduction_sites / (reproduction_sites + 100))
        )

        mortality = self.mortality_rate * self.mite_population

        # Update total population
        self.mite_population = int(self.mite_population + reproduction - mortality)
        self.mite_population = max(0, self.mite_population)

        # Distribute mites between phoretic and reproductive
        if brood_count > 0:
            brood_preference_factor = self.brood_preference
            self.reproductive_mites = int(
                min(brood_count * 0.1, self.mite_population * brood_preference_factor)
            )
        else:
            self.reproductive_mites = 0

        self.phoretic_mites = self.mite_population - self.reproductive_mites

    def _update_seasonal_factors(self, current_time: int) -> None:
        """Update seasonal reproduction factors"""
        day_of_year = current_time % 365

        # Peak reproduction in summer, reduced in winter
        if 120 <= day_of_year <= 270:  # April-September
            self.seasonal_reproduction_factor = 1.2
            self.brood_availability_factor = 1.0
        elif day_of_year < 60 or day_of_year > 330:  # Winter
            self.seasonal_reproduction_factor = 0.3
            self.brood_availability_factor = 0.2
        else:  # Spring/Fall
            self.seasonal_reproduction_factor = 0.8
            self.brood_availability_factor = 0.7

    def get_infestation_level(self, adult_count: int) -> float:
        """Calculate mites per 100 adult bees"""
        if adult_count == 0:
            return 0.0
        return (self.phoretic_mites / adult_count) * 100

    def apply_treatment(self, treatment: TreatmentRecord, current_time: int) -> None:
        """Apply mite treatment and calculate effects"""

        if TreatmentType.MITICIDE == treatment.treatment_type:
            # Miticide effectiveness varies by resistance
            effective_efficacy = treatment.efficacy * (1.0 - self.resistance_level)

            # Kill percentage of mites
            mites_killed = self.mite_population * effective_efficacy
            self.mite_population = int(self.mite_population - mites_killed)
            self.mite_population = max(0, self.mite_population)

            # Update resistance
            self.resistance_level = min(
                0.9, self.resistance_level + treatment.resistance_buildup
            )

            # Redistribute remaining mites
            if self.mite_population > 0:
                ratio = self.reproductive_mites / (self.mite_population + mites_killed)
                self.reproductive_mites = int(self.mite_population * ratio)
                self.phoretic_mites = self.mite_population - self.reproductive_mites

        elif TreatmentType.ORGANIC_ACID == treatment.treatment_type:
            # Organic acids more effective against phoretic mites
            phoretic_killed = self.phoretic_mites * treatment.efficacy * 0.8
            reproductive_killed = self.reproductive_mites * treatment.efficacy * 0.3

            self.phoretic_mites = int(max(0, self.phoretic_mites - phoretic_killed))
            self.reproductive_mites = int(
                max(0, self.reproductive_mites - reproductive_killed)
            )
            self.mite_population = self.phoretic_mites + self.reproductive_mites

    def get_virus_transmission_pressure(self) -> float:
        """Calculate virus transmission pressure from mites"""
        return min(1.0, self.mite_population * self.transmission_efficiency / 1000.0)


class VirusModel:
    """
    Models viral infections in bee colonies.

    Implements major bee viruses:
    - Deformed Wing Virus (DWV)
    - Black Queen Cell Virus (BQCV)
    - Acute Bee Paralysis Virus (ABPV)
    """

    def __init__(self, colony: Colony):
        self.colony = colony
        self.virus_loads: Dict[DiseaseType, float] = {}
        self.transmission_rates: Dict[DiseaseType, float] = {
            DiseaseType.DEFORMED_WING_VIRUS: 0.1,
            DiseaseType.BLACK_QUEEN_CELL_VIRUS: 0.05,
            DiseaseType.ACUTE_BEE_PARALYSIS_VIRUS: 0.15,
        }

        # Initialize virus loads
        for virus_type in self.transmission_rates.keys():
            self.virus_loads[virus_type] = 0.0

    def update_virus_dynamics(
        self, mite_transmission_pressure: float, colony_stress: float, current_time: int
    ) -> None:
        """Update virus population dynamics"""

        for virus_type, current_load in self.virus_loads.items():
            # Mite-mediated transmission (especially for DWV)
            if virus_type == DiseaseType.DEFORMED_WING_VIRUS:
                mite_transmission = mite_transmission_pressure * 0.8
            else:
                mite_transmission = mite_transmission_pressure * 0.2

            # Direct bee-to-bee transmission
            base_transmission = self.transmission_rates[virus_type]
            bee_transmission = base_transmission * current_load * colony_stress

            # Natural clearance
            clearance_rate = (
                0.02 if virus_type != DiseaseType.DEFORMED_WING_VIRUS else 0.01
            )
            clearance = current_load * clearance_rate

            # Update virus load
            new_load = current_load + mite_transmission + bee_transmission - clearance
            self.virus_loads[virus_type] = max(0.0, min(1.0, new_load))

    def get_bee_infection_probability(
        self, bee: BeeAgent, virus_type: DiseaseType
    ) -> float:
        """Calculate probability of bee being infected with virus"""
        base_probability = self.virus_loads[virus_type]

        # Age susceptibility
        if bee.age < 5:  # Young bees more susceptible
            age_factor = 1.5
        elif bee.age > 30:  # Old bees more susceptible
            age_factor = 1.3
        else:
            age_factor = 1.0

        # Role susceptibility
        if isinstance(bee, Forager):
            role_factor = 1.2  # Higher exposure
        else:
            role_factor = 1.0

        return min(0.9, base_probability * age_factor * role_factor)

    def infect_bee(
        self, bee: BeeAgent, virus_type: DiseaseType, current_time: int
    ) -> None:
        """Infect bee with virus"""
        infection_level = random.uniform(0.1, 0.8)

        disease_state = DiseaseState(
            disease_type=virus_type,
            infection_level=infection_level,
            infection_date=current_time,
            transmission_rate=self.transmission_rates[virus_type],
            mortality_rate=self._get_virus_mortality_rate(virus_type),
            virulence=self._get_virus_virulence(virus_type),
        )

        # Add to bee's disease states
        if not hasattr(bee, "disease_states"):
            bee.disease_states = {}
        bee.disease_states[virus_type] = disease_state

    def _get_virus_mortality_rate(self, virus_type: DiseaseType) -> float:
        """Get virus-specific mortality rate"""
        mortality_rates = {
            DiseaseType.DEFORMED_WING_VIRUS: 0.1,
            DiseaseType.BLACK_QUEEN_CELL_VIRUS: 0.15,
            DiseaseType.ACUTE_BEE_PARALYSIS_VIRUS: 0.3,
        }
        return mortality_rates.get(virus_type, 0.1)

    def _get_virus_virulence(self, virus_type: DiseaseType) -> float:
        """Get virus-specific virulence"""
        virulence_levels = {
            DiseaseType.DEFORMED_WING_VIRUS: 0.6,
            DiseaseType.BLACK_QUEEN_CELL_VIRUS: 0.4,
            DiseaseType.ACUTE_BEE_PARALYSIS_VIRUS: 0.8,
        }
        return virulence_levels.get(virus_type, 0.5)


class NosemaModel:
    """
    Models Nosema spore infections.

    Implements:
    - Nosema apis and Nosema ceranae
    - Spore dynamics
    - Environmental transmission
    - Seasonal patterns
    """

    def __init__(self, colony: Colony):
        self.colony = colony
        self.spore_counts: Dict[DiseaseType, int] = {
            DiseaseType.NOSEMA_APIS: 0,
            DiseaseType.NOSEMA_CERANAE: 0,
        }
        self.environmental_spores = 0

    def update_nosema_dynamics(
        self, current_time: int, weather_conditions: Dict[str, float]
    ) -> None:
        """Update Nosema spore dynamics"""

        # Environmental factors
        temperature = weather_conditions.get("temperature", 15.0)
        humidity = weather_conditions.get("humidity", 60.0)

        # Temperature affects spore survival
        if temperature < 5 or temperature > 35:
            spore_survival = 0.5
        else:
            spore_survival = 1.0

        # High humidity promotes transmission
        humidity_factor = 1.0 + (humidity - 60) / 100.0

        for nosema_type, spore_count in self.spore_counts.items():
            # Spore multiplication in infected bees
            infected_bee_count = self._count_infected_bees(nosema_type)
            spore_production = infected_bee_count * 1000 * spore_survival

            # Environmental transmission
            (self.environmental_spores * humidity_factor * spore_survival / 1000000.0)

            # Spore decay
            decay_rate = 0.1 if nosema_type == DiseaseType.NOSEMA_APIS else 0.05
            spore_decay = spore_count * decay_rate

            # Update spore count
            self.spore_counts[nosema_type] = int(
                self.spore_counts[nosema_type] + spore_production - spore_decay
            )
            self.spore_counts[nosema_type] = max(0, self.spore_counts[nosema_type])

        # Update environmental spores
        total_production = sum(self.spore_counts.values()) * 0.1
        environmental_decay = self.environmental_spores * 0.2
        self.environmental_spores = int(
            self.environmental_spores + total_production - environmental_decay
        )
        self.environmental_spores = max(0, self.environmental_spores)

    def _count_infected_bees(self, nosema_type: DiseaseType) -> int:
        """Count bees infected with specific Nosema type"""
        count = 0
        for bee in self.colony.get_bees():
            if hasattr(bee, "disease_states") and nosema_type in bee.disease_states:
                count += 1
        return count


class TreatmentManager:
    """
    Manages disease treatment applications and schedules.

    Implements:
    - Treatment scheduling
    - Efficacy tracking
    - Resistance monitoring
    - Cost-benefit analysis
    """

    def __init__(self, colony: Colony):
        self.colony = colony
        self.treatment_history: List[TreatmentRecord] = []
        self.scheduled_treatments: List[Tuple[int, TreatmentRecord]] = []
        self.resistance_levels: Dict[TreatmentType, float] = defaultdict(float)

    def schedule_treatment(
        self, treatment: TreatmentRecord, application_day: int
    ) -> None:
        """Schedule a treatment for future application"""
        self.scheduled_treatments.append((application_day, treatment))
        self.scheduled_treatments.sort(key=lambda x: x[0])

    def apply_scheduled_treatments(self, current_time: int) -> None:
        """Apply any scheduled treatments for current day"""

        treatments_to_apply = []
        remaining_treatments = []

        for day, treatment in self.scheduled_treatments:
            if day <= current_time:
                treatments_to_apply.append(treatment)
            else:
                remaining_treatments.append((day, treatment))

        self.scheduled_treatments = remaining_treatments

        for treatment in treatments_to_apply:
            self.apply_treatment(treatment, current_time)

    def apply_treatment(self, treatment: TreatmentRecord, current_time: int) -> None:
        """Apply treatment to colony"""

        # Apply to mite populations
        if hasattr(self.colony, "mite_model") and self.colony.mite_model is not None:
            self.colony.mite_model.apply_treatment(treatment, current_time)

        # Apply to individual bees
        for bee in self.colony.get_bees():
            self._apply_treatment_to_bee(bee, treatment, current_time)

        # Update resistance
        self.resistance_levels[treatment.treatment_type] += treatment.resistance_buildup

        # Record treatment
        self.treatment_history.append(treatment)

    def _apply_treatment_to_bee(
        self, bee: BeeAgent, treatment: TreatmentRecord, current_time: int
    ) -> None:
        """Apply treatment to individual bee"""

        if not hasattr(bee, "disease_states"):
            return

        for disease_type in treatment.target_diseases:
            if disease_type in bee.disease_states:
                disease = bee.disease_states[disease_type]

                # Reduce infection level
                reduction = treatment.efficacy * (
                    1.0 - self.resistance_levels[treatment.treatment_type]
                )
                disease.infection_level *= 1.0 - reduction

                # Apply side effects
                for effect, magnitude in treatment.side_effects.items():
                    if effect == "reduced_lifespan":
                        bee.longevity_factor *= 1.0 - magnitude
                    elif effect == "reduced_activity":
                        bee.energy *= 1.0 - magnitude

    def recommend_treatment(self, current_time: int) -> Optional[TreatmentRecord]:
        """Recommend treatment based on current disease pressure"""

        # Assess disease threats
        threats = self._assess_disease_threats()

        if not threats:
            return None

        # Find most severe threat
        primary_threat = max(threats.items(), key=lambda x: x[1])
        disease_type, severity = primary_threat

        if severity < 0.3:  # Below treatment threshold
            return None

        # Recommend appropriate treatment
        if disease_type == DiseaseType.VARROA_MITES:
            return TreatmentRecord(
                treatment_type=TreatmentType.MITICIDE,
                target_diseases=[DiseaseType.VARROA_MITES],
                application_date=current_time,
                duration=7,
                efficacy=0.85,
                cost=50.0,
                resistance_buildup=0.05,
            )
        elif disease_type in [DiseaseType.NOSEMA_APIS, DiseaseType.NOSEMA_CERANAE]:
            return TreatmentRecord(
                treatment_type=TreatmentType.ANTIBIOTIC,
                target_diseases=[disease_type],
                application_date=current_time,
                duration=14,
                efficacy=0.7,
                cost=30.0,
                resistance_buildup=0.02,
            )

        return None

    def _assess_disease_threats(self) -> Dict[DiseaseType, float]:
        """Assess current disease threat levels"""
        threats: Dict[DiseaseType, float] = {}

        # Assess mite infestation
        if hasattr(self.colony, "mite_model") and self.colony.mite_model is not None:
            infestation_level = self.colony.mite_model.get_infestation_level(
                self.colony.get_adult_population()
            )
            if infestation_level > 0:
                threats[DiseaseType.VARROA_MITES] = min(1.0, infestation_level / 10.0)

        # Assess viral loads
        if hasattr(self.colony, "virus_model") and self.colony.virus_model is not None:
            for virus_type, load in self.colony.virus_model.virus_loads.items():
                if load > 0:
                    threats[virus_type] = load

        # Assess Nosema
        if (
            hasattr(self.colony, "nosema_model")
            and self.colony.nosema_model is not None
        ):
            for (
                nosema_type,
                spore_count,
            ) in self.colony.nosema_model.spore_counts.items():
                if spore_count > 1000:
                    threats[nosema_type] = min(1.0, spore_count / 100000.0)

        return threats

    def get_treatment_effectiveness(self) -> Dict[str, Any]:
        """Analyze treatment effectiveness"""

        if not self.treatment_history:
            return {"total_treatments": 0}

        treatment_counts: Dict[str, int] = defaultdict(int)
        for treatment in self.treatment_history:
            treatment_counts[treatment.treatment_type.value] += 1

        return {
            "total_treatments": len(self.treatment_history),
            "treatments_by_type": dict(treatment_counts),
            "average_cost": sum(t.cost for t in self.treatment_history)
            / len(self.treatment_history),
            "resistance_levels": dict(self.resistance_levels),
        }


class DiseaseManager:
    """
    Comprehensive disease management system for colonies.

    Integrates all disease models and provides unified interface.
    """

    def __init__(self, colony: Colony):
        self.colony = colony
        self.mite_model = VarroaMiteModel(colony)
        self.virus_model = VirusModel(colony)
        self.nosema_model = NosemaModel(colony)
        self.treatment_manager = TreatmentManager(colony)

        # Add models to colony
        setattr(colony, "mite_model", self.mite_model)
        setattr(colony, "virus_model", self.virus_model)
        setattr(colony, "nosema_model", self.nosema_model)
        setattr(colony, "treatment_manager", self.treatment_manager)

    @property
    def varroa_model(self) -> "VarroaMiteModel":
        """Alias for mite_model for backward compatibility"""
        return self.mite_model

    def update_disease_dynamics(
        self, current_time: int, weather_conditions: Dict[str, float]
    ) -> None:
        """Update all disease dynamics"""

        # Get colony state
        brood_count = self.colony.population_counts["brood"]
        adult_count = self.colony.get_adult_population()

        # Calculate colony stress
        colony_stress = self._calculate_colony_stress()

        # Update mite population
        self.mite_model.update_mite_population(current_time, brood_count, adult_count)

        # Update virus dynamics
        mite_pressure = self.mite_model.get_virus_transmission_pressure()
        self.virus_model.update_virus_dynamics(
            mite_pressure, colony_stress, current_time
        )

        # Update Nosema
        self.nosema_model.update_nosema_dynamics(current_time, weather_conditions)

        # Process individual bee infections
        self._process_individual_infections(current_time)

        # Apply scheduled treatments
        self.treatment_manager.apply_scheduled_treatments(current_time)

        # Check for treatment recommendations
        if current_time % 7 == 0:  # Weekly assessment
            recommended_treatment = self.treatment_manager.recommend_treatment(
                current_time
            )
            if recommended_treatment:
                self.treatment_manager.schedule_treatment(
                    recommended_treatment, current_time + 1
                )

    def _calculate_colony_stress(self) -> float:
        """Calculate overall colony stress level"""
        # Resource stress
        resource_adequacy = self.colony.get_resource_adequacy()
        resource_stress = 1.0 - resource_adequacy

        # Population stress
        current_pop = self.colony.get_adult_population()
        optimal_pop = 20000
        pop_stress = abs(current_pop - optimal_pop) / optimal_pop

        # Seasonal stress
        season = self.colony.model.get_current_season()
        seasonal_stress = 0.3 if season == "winter" else 0.0

        return min(1.0, (resource_stress + pop_stress + seasonal_stress) / 3.0)

    def _process_individual_infections(self, current_time: int) -> None:
        """Process disease infections for individual bees"""

        for bee in self.colony.get_bees():
            # Check for new infections
            self._check_new_infections(bee, current_time)

            # Progress existing infections
            if hasattr(bee, "disease_states"):
                for disease_type, disease_state in list(bee.disease_states.items()):
                    # Progress infection
                    days_elapsed = 1  # Assume daily updates
                    colony_stress = self._calculate_colony_stress()
                    disease_state.progress_infection(days_elapsed, colony_stress)

                    # Check mortality
                    mortality_prob = disease_state.get_mortality_probability(bee.age)
                    if random.random() < mortality_prob:
                        bee.die()
                        break

                    # Remove cleared infections
                    if disease_state.infection_level < 0.01:
                        del bee.disease_states[disease_type]

    def _check_new_infections(self, bee: BeeAgent, current_time: int) -> None:
        """Check for new disease infections in bee"""

        # Virus infections
        for virus_type in self.virus_model.virus_loads.keys():
            infection_prob = self.virus_model.get_bee_infection_probability(
                bee, virus_type
            )

            if random.random() < infection_prob * 0.01:  # Daily infection chance
                if (
                    not hasattr(bee, "disease_states")
                    or virus_type not in bee.disease_states
                ):
                    self.virus_model.infect_bee(bee, virus_type, current_time)

        # Nosema infections
        for nosema_type, spore_count in self.nosema_model.spore_counts.items():
            if spore_count > 1000:
                infection_prob = min(0.1, spore_count / 100000.0)

                if random.random() < infection_prob * 0.01:
                    if (
                        not hasattr(bee, "disease_states")
                        or nosema_type not in bee.disease_states
                    ):
                        self._infect_bee_with_nosema(bee, nosema_type, current_time)

    def _infect_bee_with_nosema(
        self, bee: BeeAgent, nosema_type: DiseaseType, current_time: int
    ) -> None:
        """Infect bee with Nosema"""
        infection_level = random.uniform(0.1, 0.6)

        disease_state = DiseaseState(
            disease_type=nosema_type,
            infection_level=infection_level,
            infection_date=current_time,
            transmission_rate=0.05,
            mortality_rate=0.05,
            virulence=0.4,
        )

        if not hasattr(bee, "disease_states"):
            bee.disease_states = {}
        bee.disease_states[nosema_type] = disease_state

    def get_disease_summary(self) -> Dict[str, Any]:
        """Get comprehensive disease status summary"""

        # Count infected bees
        infected_counts: Dict[str, int] = defaultdict(int)
        total_bees = len(self.colony.get_bees())

        for bee in self.colony.get_bees():
            if hasattr(bee, "disease_states"):
                for disease_type in bee.disease_states.keys():
                    infected_counts[disease_type.value] += 1

        return {
            "mite_population": self.mite_model.mite_population,
            "mite_infestation_rate": self.mite_model.get_infestation_level(
                self.colony.get_adult_population()
            ),
            "virus_loads": {
                vt.value: load for vt, load in self.virus_model.virus_loads.items()
            },
            "nosema_spores": {
                nt.value: count for nt, count in self.nosema_model.spore_counts.items()
            },
            "infected_bee_counts": dict(infected_counts),
            "infection_rates": {
                disease: count / total_bees if total_bees > 0 else 0
                for disease, count in infected_counts.items()
            },
            "treatment_summary": self.treatment_manager.get_treatment_effectiveness(),
            "colony_stress_level": self._calculate_colony_stress(),
        }
