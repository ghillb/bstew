"""
Development phase system for BSTEW
==================================

Implements individual bee development tracking with realistic stage progression,
environmental dependencies, and species-specific timing parameters.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging
from collections import defaultdict

from .genetics import Genotype, Sex


class DevelopmentStage(Enum):
    """Bee development stages"""

    EGG = "egg"
    LARVA = "larva"
    PUPA = "pupa"
    ADULT = "adult"


class DeathCause(Enum):
    """Causes of death during development"""

    INSUFFICIENT_INCUBATION = "insufficient_incubation"
    INSUFFICIENT_WEIGHT = "insufficient_weight"
    ENERGY_SHORTAGE = "energy_shortage"
    NO_ADULT_CARE = "no_adult_care"
    DIPLOID_MALE = "diploid_male"
    ENVIRONMENTAL_STRESS = "environmental_stress"
    PREDATION = "predation"
    DISEASE = "disease"
    OLD_AGE = "old_age"
    SEASON_END = "season_end"


class DevelopmentParameters(BaseModel):
    """Species-specific development parameters"""

    model_config = {"validate_assignment": True}

    # Minimum development times (days)
    dev_age_hatching_min: float = Field(
        default=3.0, ge=0.0, description="Minimum hatching age in days"
    )
    dev_age_pupation_min: float = Field(
        default=14.0, ge=0.0, description="Minimum pupation age in days"
    )
    dev_age_emerging_min: float = Field(
        default=21.0, ge=0.0, description="Minimum emerging age in days"
    )

    # Weight thresholds (mg)
    dev_weight_egg: float = Field(default=0.15, ge=0.0, description="Egg weight in mg")
    dev_weight_pupation_min: float = Field(
        default=120.0, ge=0.0, description="Minimum pupation weight in mg"
    )
    dev_weight_adult_min: float = Field(
        default=100.0, ge=0.0, description="Minimum adult weight in mg"
    )

    # Energy requirements (kJ)
    energy_per_day_egg: float = Field(
        default=0.001, ge=0.0, description="Daily energy requirement for eggs"
    )
    energy_per_day_larva: float = Field(
        default=0.01, ge=0.0, description="Daily energy requirement for larvae"
    )
    energy_per_day_pupa: float = Field(
        default=0.005, ge=0.0, description="Daily energy requirement for pupae"
    )

    # Incubation requirements (kJ)
    incubation_per_day_egg: float = Field(
        default=0.002, ge=0.0, description="Daily incubation requirement for eggs"
    )
    incubation_per_day_larva: float = Field(
        default=0.02, ge=0.0, description="Daily incubation requirement for larvae"
    )
    incubation_per_day_pupa: float = Field(
        default=0.01, ge=0.0, description="Daily incubation requirement for pupae"
    )

    # Temperature thresholds (°C)
    temperature_min: float = Field(
        default=10.0, description="Minimum temperature for development"
    )
    temperature_optimal: float = Field(
        default=35.0, description="Optimal temperature for development"
    )
    temperature_max: float = Field(
        default=45.0, description="Maximum temperature for development"
    )


class DevelopingBee(BaseModel):
    """Individual developing bee with full lifecycle tracking"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    unique_id: int = Field(ge=0, description="Unique bee identifier")
    genotype: Optional[Genotype] = Field(default=None, description="Bee genotype")
    stage: DevelopmentStage = Field(description="Current development stage")
    age_days: float = Field(default=0.0, ge=0.0, description="Age in days")
    weight_mg: float = Field(default=0.0, ge=0.0, description="Weight in milligrams")

    # Stage-specific cumulative times
    cumul_time_egg: float = Field(
        default=0.0, ge=0.0, description="Cumulative time as egg"
    )
    cumul_time_larva: float = Field(
        default=0.0, ge=0.0, description="Cumulative time as larva"
    )
    cumul_time_pupa: float = Field(
        default=0.0, ge=0.0, description="Cumulative time as pupa"
    )

    # Energy and care tracking
    cumul_incubation_received: float = Field(
        default=0.0, ge=0.0, description="Cumulative incubation received"
    )
    cumul_energy_received: float = Field(
        default=0.0, ge=0.0, description="Cumulative energy received"
    )
    care_deficit: float = Field(default=0.0, ge=0.0, description="Care deficit")

    # Environmental exposure
    temperature_exposure: List[float] = Field(
        default_factory=list, description="Temperature exposure history"
    )
    stress_factors: Dict[str, float] = Field(
        default_factory=dict, description="Environmental stress factors"
    )

    # Development success indicators
    development_success: bool = Field(
        default=True, description="Whether development is successful"
    )
    death_cause: Optional[DeathCause] = Field(
        default=None, description="Cause of death if applicable"
    )
    death_day: Optional[float] = Field(
        default=None, description="Day of death if applicable"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize weight based on stage after model creation"""
        if self.stage == DevelopmentStage.EGG and self.weight_mg == 0.0:
            self.weight_mg = 0.15
        elif self.stage == DevelopmentStage.LARVA:
            self.weight_mg = 5.0
        elif self.stage == DevelopmentStage.PUPA:
            self.weight_mg = 120.0
        elif self.stage == DevelopmentStage.ADULT:
            self.weight_mg = 100.0

        # Check for diploid male lethality
        if self.genotype and self.genotype.is_diploid_male():
            self.development_success = False
            self.death_cause = DeathCause.DIPLOID_MALE
            self.death_day = self.age_days

    def is_alive(self) -> bool:
        """Check if bee is still developing successfully"""
        return self.development_success and self.death_cause is None

    def get_target_bee_type(self) -> str:
        """Get the target bee type based on genotype"""
        if not self.genotype:
            return "worker"  # Default

        if self.genotype.sex == Sex.MALE:
            return "drone"
        elif self.genotype.sex == Sex.FEMALE:
            return "worker"  # Could become forager or queen based on care
        else:  # Diploid male
            return "dead"  # Will die before emergence

    def add_stress(self, factor: str, intensity: float) -> None:
        """Add stress factor"""
        self.stress_factors[factor] = intensity

    def get_total_stress(self) -> float:
        """Get total stress level"""
        return sum(self.stress_factors.values())


class DevelopmentSystem:
    """
    Manages individual bee development through all life stages.

    Implements:
    - Individual bee tracking through egg→larva→pupa→adult
    - Temperature-dependent development rates
    - Energy and incubation requirements
    - Realistic mortality at each stage
    - Stage-specific care requirements
    """

    def __init__(self, species_params: DevelopmentParameters):
        self.params = species_params
        self.developing_bees: Dict[int, DevelopingBee] = {}
        self.next_bee_id = 1

        # Mortality tracking
        self.mortality_stats: Dict[DevelopmentStage, Dict[str, int]] = {
            stage: defaultdict(int) for stage in DevelopmentStage
        }

        self.logger = logging.getLogger(__name__)

    def add_egg(self, genotype: Optional[Genotype] = None) -> int:
        """Add new egg to development system"""
        bee_id = self.next_bee_id
        self.next_bee_id += 1

        egg = DevelopingBee(
            unique_id=bee_id,
            genotype=genotype,
            stage=DevelopmentStage.EGG,
            weight_mg=self.params.dev_weight_egg,
        )

        self.developing_bees[bee_id] = egg
        return bee_id

    def step(
        self, temperature: float, available_care: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute one development step for all developing bees"""

        results: Dict[str, Any] = {
            "emerged_bees": [],
            "deaths": [],
            "stage_transitions": [],
            "care_demand": defaultdict(float),
        }
        # Type hints for lists to help mypy
        results["emerged_bees"] = []
        results["deaths"] = []
        results["stage_transitions"] = []

        # Process each developing bee
        for bee_id, bee in list(self.developing_bees.items()):
            if not bee.is_alive():
                continue

            # Age the bee
            bee.age_days += 1.0

            # Track temperature exposure
            bee.temperature_exposure.append(temperature)

            # Calculate temperature effect on development
            temp_effect = self.calculate_temperature_effect(temperature)

            # Stage-specific processing
            if bee.stage == DevelopmentStage.EGG:
                self.process_egg_development(bee, temp_effect, available_care)
            elif bee.stage == DevelopmentStage.LARVA:
                self.process_larva_development(bee, temp_effect, available_care)
            elif bee.stage == DevelopmentStage.PUPA:
                self.process_pupa_development(bee, temp_effect, available_care)

            # Check for stage transitions
            transition = self.check_stage_transition(bee)
            if transition:
                results["stage_transitions"].append(transition)

            # Check for emergence
            if bee.stage == DevelopmentStage.ADULT:
                results["emerged_bees"].append(bee)
                del self.developing_bees[bee_id]

            # Check for death
            if not bee.is_alive():
                results["deaths"].append(bee)
                if bee.death_cause is not None:
                    self.mortality_stats[bee.stage][bee.death_cause.value] += 1
                del self.developing_bees[bee_id]

        return results

    def process_egg_development(
        self, egg: DevelopingBee, temp_effect: float, available_care: Dict[str, float]
    ) -> None:
        """Process egg development"""
        egg.cumul_time_egg += temp_effect

        # Energy requirements
        energy_needed = self.params.energy_per_day_egg * temp_effect
        energy_received = min(energy_needed, available_care.get("energy", 0.0))
        egg.cumul_energy_received += energy_received

        # Incubation requirements
        incubation_needed = self.params.incubation_per_day_egg * temp_effect
        incubation_received = min(
            incubation_needed, available_care.get("incubation", 0.0)
        )
        egg.cumul_incubation_received += incubation_received

        # Track care deficit
        if energy_received < energy_needed * 0.8:
            egg.care_deficit += energy_needed - energy_received

        # Death from insufficient incubation
        if (
            egg.cumul_incubation_received
            < egg.cumul_time_egg * self.params.incubation_per_day_egg * 0.6
        ):
            egg.development_success = False
            egg.death_cause = DeathCause.INSUFFICIENT_INCUBATION
            egg.death_day = egg.age_days

    def process_larva_development(
        self, larva: DevelopingBee, temp_effect: float, available_care: Dict[str, float]
    ) -> None:
        """Process larva development"""
        larva.cumul_time_larva += temp_effect

        # Weight gain
        base_growth = 8.0 * temp_effect  # mg per day
        growth_factor = min(1.0, available_care.get("feeding", 0.0))
        actual_growth = base_growth * growth_factor
        larva.weight_mg += actual_growth

        # Energy requirements (higher for larvae)
        energy_needed = self.params.energy_per_day_larva * temp_effect
        energy_received = min(energy_needed, available_care.get("energy", 0.0))
        larva.cumul_energy_received += energy_received

        # Incubation requirements
        incubation_needed = self.params.incubation_per_day_larva * temp_effect
        incubation_received = min(
            incubation_needed, available_care.get("incubation", 0.0)
        )
        larva.cumul_incubation_received += incubation_received

        # Track care deficit
        if energy_received < energy_needed * 0.8:
            larva.care_deficit += energy_needed - energy_received

        # Death from insufficient care
        if larva.care_deficit > energy_needed * 5.0:  # 5 days worth of deficit
            larva.development_success = False
            larva.death_cause = DeathCause.NO_ADULT_CARE
            larva.death_day = larva.age_days

        # Death from insufficient incubation
        if (
            larva.cumul_incubation_received
            < larva.cumul_time_larva * self.params.incubation_per_day_larva * 0.5
        ):
            larva.development_success = False
            larva.death_cause = DeathCause.INSUFFICIENT_INCUBATION
            larva.death_day = larva.age_days

    def process_pupa_development(
        self, pupa: DevelopingBee, temp_effect: float, available_care: Dict[str, float]
    ) -> None:
        """Process pupa development"""
        pupa.cumul_time_pupa += temp_effect

        # Pupae need less direct care but still need incubation
        incubation_needed = self.params.incubation_per_day_pupa * temp_effect
        incubation_received = min(
            incubation_needed, available_care.get("incubation", 0.0)
        )
        pupa.cumul_incubation_received += incubation_received

        # Weight loss during pupation (metamorphosis)
        weight_loss = 2.0 * temp_effect  # mg per day
        pupa.weight_mg = max(
            self.params.dev_weight_adult_min, pupa.weight_mg - weight_loss
        )

        # Death from insufficient incubation
        if (
            pupa.cumul_incubation_received
            < pupa.cumul_time_pupa * self.params.incubation_per_day_pupa * 0.4
        ):
            pupa.development_success = False
            pupa.death_cause = DeathCause.INSUFFICIENT_INCUBATION
            pupa.death_day = pupa.age_days

    def calculate_temperature_effect(self, temperature: float) -> float:
        """Calculate temperature effect on development rate"""
        if temperature < self.params.temperature_min:
            return 0.0  # No development below minimum
        elif temperature > self.params.temperature_max:
            return 0.0  # No development above maximum
        elif temperature == self.params.temperature_optimal:
            return 1.0  # Optimal rate
        else:
            # Linear interpolation between min/max and optimal
            if temperature < self.params.temperature_optimal:
                return (temperature - self.params.temperature_min) / (
                    self.params.temperature_optimal - self.params.temperature_min
                )
            else:
                return 1.0 - (temperature - self.params.temperature_optimal) / (
                    self.params.temperature_max - self.params.temperature_optimal
                )

    def check_stage_transition(self, bee: DevelopingBee) -> Optional[Dict[str, Any]]:
        """Check if bee should transition to next stage"""
        transition = None

        if bee.stage == DevelopmentStage.EGG:
            if (
                bee.cumul_time_egg >= self.params.dev_age_hatching_min
                and bee.cumul_incubation_received
                >= self.params.incubation_per_day_egg
                * self.params.dev_age_hatching_min
                * 0.8
            ):
                bee.stage = DevelopmentStage.LARVA
                transition = {
                    "bee_id": bee.unique_id,
                    "from_stage": DevelopmentStage.EGG,
                    "to_stage": DevelopmentStage.LARVA,
                    "day": bee.age_days,
                }

        elif bee.stage == DevelopmentStage.LARVA:
            if (
                bee.cumul_time_larva >= self.params.dev_age_pupation_min
                and bee.weight_mg >= self.params.dev_weight_pupation_min
            ):
                bee.stage = DevelopmentStage.PUPA
                transition = {
                    "bee_id": bee.unique_id,
                    "from_stage": DevelopmentStage.LARVA,
                    "to_stage": DevelopmentStage.PUPA,
                    "day": bee.age_days,
                }

        elif bee.stage == DevelopmentStage.PUPA:
            if (
                bee.cumul_time_pupa >= self.params.dev_age_emerging_min
                and bee.weight_mg >= self.params.dev_weight_adult_min
            ):
                bee.stage = DevelopmentStage.ADULT
                transition = {
                    "bee_id": bee.unique_id,
                    "from_stage": DevelopmentStage.PUPA,
                    "to_stage": DevelopmentStage.ADULT,
                    "day": bee.age_days,
                }

        return transition

    def calculate_care_demand(self) -> Dict[str, float]:
        """Calculate total care demand from all developing bees"""
        demand = {"energy": 0.0, "incubation": 0.0, "feeding": 0.0}

        for bee in self.developing_bees.values():
            if not bee.is_alive():
                continue

            if bee.stage == DevelopmentStage.EGG:
                demand["energy"] += self.params.energy_per_day_egg
                demand["incubation"] += self.params.incubation_per_day_egg
            elif bee.stage == DevelopmentStage.LARVA:
                demand["energy"] += self.params.energy_per_day_larva
                demand["incubation"] += self.params.incubation_per_day_larva
                demand["feeding"] += 1.0  # Feeding demand
            elif bee.stage == DevelopmentStage.PUPA:
                demand["incubation"] += self.params.incubation_per_day_pupa

        return demand

    def get_population_by_stage(self) -> Dict[DevelopmentStage, int]:
        """Get population count by development stage"""
        counts = {stage: 0 for stage in DevelopmentStage}

        for bee in self.developing_bees.values():
            if bee.is_alive():
                counts[bee.stage] += 1

        return counts

    def get_mortality_statistics(self) -> Dict[str, Any]:
        """Get detailed mortality statistics"""
        stats: Dict[str, Any] = {
            "total_deaths": 0,
            "deaths_by_stage": dict(self.mortality_stats),
            "deaths_by_cause": defaultdict(int),
        }

        for stage_deaths in self.mortality_stats.values():
            for cause, count in stage_deaths.items():
                stats["total_deaths"] += count
                stats["deaths_by_cause"][cause] += count

        return stats

    def get_development_summary(self) -> Dict[str, Any]:
        """Get comprehensive development summary"""
        population = self.get_population_by_stage()
        mortality = self.get_mortality_statistics()
        care_demand = self.calculate_care_demand()

        return {
            "population_by_stage": population,
            "total_developing": sum(population.values()),
            "mortality_statistics": mortality,
            "care_demand": care_demand,
            "development_success_rate": self.calculate_success_rate(),
        }

    def calculate_success_rate(self) -> float:
        """Calculate overall development success rate"""
        total_deaths = sum(
            sum(stage_deaths.values()) for stage_deaths in self.mortality_stats.values()
        )
        total_bees = len(self.developing_bees) + total_deaths

        if total_bees == 0:
            return 1.0

        return 1.0 - (total_deaths / total_bees)

    def apply_environmental_stress(self, stress_type: str, intensity: float) -> None:
        """Apply environmental stress to all developing bees"""
        for bee in self.developing_bees.values():
            if bee.is_alive():
                bee.add_stress(stress_type, intensity)

                # Environmental stress can cause death
                if bee.get_total_stress() > 2.0:  # High stress threshold
                    bee.development_success = False
                    bee.death_cause = DeathCause.ENVIRONMENTAL_STRESS
                    bee.death_day = bee.age_days

    def end_season_mortality(self) -> None:
        """Apply end-of-season mortality to all developing bees"""
        for bee in self.developing_bees.values():
            if bee.is_alive():
                bee.development_success = False
                bee.death_cause = DeathCause.SEASON_END
                bee.death_day = bee.age_days

    def get_genetic_diversity(self) -> Dict[str, Any]:
        """Get genetic diversity of developing population"""
        genotypes = [
            bee.genotype
            for bee in self.developing_bees.values()
            if bee.is_alive() and bee.genotype is not None
        ]

        if not genotypes:
            return {"diversity": 0.0, "diploid_male_frequency": 0.0}

        diploid_males = sum(1 for g in genotypes if g.is_diploid_male())

        return {
            "diversity": len(set(tuple(g.get_allele_ids()) for g in genotypes))
            / len(genotypes),
            "diploid_male_frequency": diploid_males / len(genotypes),
            "total_genotyped": len(genotypes),
        }
