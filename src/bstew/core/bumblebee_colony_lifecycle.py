"""
Bumblebee Annual Colony Lifecycle - Biologically Accurate Implementation
=======================================================================

CRITICAL: This implements the annual colony cycle of bumblebees, which is
FUNDAMENTALLY DIFFERENT from honey bees' perennial colonies.

Based on:
- Goulson (2010): Bumblebees: behaviour, ecology, and conservation
- Plowright & Jay (1977): On the relative importance of different castes
- Duchateau & Velthuis (1988): Development and reproductive strategies

Key differences from honey bees:
- Annual cycle (not perennial)
- Only mated queens survive winter
- Colony reaches peak ~50-400 individuals (not 20,000-80,000)
- Distinct seasonal phases with different caste production
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import random


class BumblebeeCaste(Enum):
    """Biologically accurate bumblebee castes"""

    FOUNDRESS_QUEEN = "foundress_queen"  # Overwintered queen founding colony
    WORKER = "worker"  # Non-reproductive female
    MALE = "male"  # Reproductive male (produced mid-season)
    GYNE = "gyne"  # New queen (produced late season)


class ColonyPhase(Enum):
    """Annual colony cycle phases - critical for biological accuracy"""

    HIBERNATION = "hibernation"  # Winter: Only mated gynes survive
    FOUNDING = "founding"  # Spring: Single queen establishes nest
    GROWTH = "growth"  # Early summer: Worker production
    REPRODUCTION = "reproduction"  # Mid-summer: Male/gyne production
    DECLINE = "decline"  # Late summer: Colony senescence
    DEATH = "death"  # Colony dies (except overwintering gynes)


@dataclass
class ColonyDemographics:
    """Track colony population by caste"""

    foundress_queens: int = 0
    workers: int = 0
    males: int = 0
    gynes: int = 0

    @property
    def total_population(self) -> int:
        return self.foundress_queens + self.workers + self.males + self.gynes

    @property
    def foraging_population(self) -> int:
        """Only workers and foundress queens forage"""
        return self.foundress_queens + self.workers


class BumblebeeColonyLifecycleModel(BaseModel):
    """Parameters for bumblebee annual colony lifecycle"""

    model_config = {"validate_assignment": True}

    # Timing parameters (day of year)
    hibernation_end_day: int = Field(
        default=75,
        ge=60,
        le=120,
        description="Day when queens emerge from hibernation (mid-March)",
    )
    colony_founding_duration: int = Field(
        default=30, ge=14, le=60, description="Days for colony establishment"
    )
    worker_production_peak_day: int = Field(
        default=150, ge=120, le=180, description="Peak worker production (late May)"
    )
    reproduction_start_day: int = Field(
        default=180,
        ge=150,
        le=210,
        description="Start of male/gyne production (early July)",
    )
    colony_decline_start_day: int = Field(
        default=240,
        ge=210,
        le=270,
        description="Colony senescence begins (late August)",
    )
    hibernation_start_day: int = Field(
        default=300,
        ge=270,
        le=330,
        description="New queens enter hibernation (late October)",
    )

    # Population parameters
    max_colony_size: int = Field(
        default=200,
        ge=50,
        le=400,
        description="Maximum colony size (literature: 50-400 for most species)",
    )
    foundress_mortality_rate: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Proportion of foundress queens that fail to establish colonies",
    )
    worker_development_time: int = Field(
        default=21, ge=18, le=28, description="Days from egg to adult worker"
    )
    male_development_time: int = Field(
        default=24, ge=21, le=30, description="Days from egg to adult male"
    )
    gyne_development_time: int = Field(
        default=25, ge=22, le=32, description="Days from egg to adult gyne"
    )

    # Reproductive parameters
    foundress_egg_laying_rate: float = Field(
        default=8.0,
        ge=3.0,
        le=15.0,
        description="Eggs per day during colony establishment",
    )
    peak_egg_laying_rate: float = Field(
        default=25.0,
        ge=15.0,
        le=40.0,
        description="Maximum eggs per day during peak season",
    )
    switch_point_trigger: float = Field(
        default=0.7,
        ge=0.5,
        le=0.9,
        description="Colony size proportion triggering switch to reproductive castes",
    )

    # Survival parameters
    worker_daily_mortality: float = Field(
        default=0.02, ge=0.01, le=0.05, description="Daily mortality rate for workers"
    )
    winter_queen_survival: float = Field(
        default=0.1,
        ge=0.05,
        le=0.3,
        description="Proportion of gynes surviving hibernation",
    )


@dataclass
class BumblebeeColony:
    """Represents a single bumblebee colony with annual lifecycle"""

    colony_id: str
    species: str = "bombus_terrestris"  # Default species
    founded_day: int = 0
    current_phase: ColonyPhase = ColonyPhase.HIBERNATION

    # Demographics
    demographics: ColonyDemographics = field(default_factory=ColonyDemographics)

    # Lifecycle tracking
    age_days: int = 0
    days_in_current_phase: int = 0
    total_eggs_laid: int = 0
    total_individuals_produced: int = 0

    # Energy and resources
    stored_honey: float = 0.0  # Limited storage compared to honey bees
    stored_pollen: float = 0.0
    nest_temperature: float = 25.0  # Bumblebees maintain lower nest temps

    # Lifecycle model
    lifecycle_model: BumblebeeColonyLifecycleModel = field(
        default_factory=BumblebeeColonyLifecycleModel
    )

    # Status flags
    is_active: bool = True
    hibernating_gynes: int = 0  # Number of successfully mated gynes

    def update_lifecycle_phase(self, day_of_year: int) -> None:
        """Update colony phase based on time of year and colony state"""
        old_phase = self.current_phase

        # Determine phase based on day of year and colony state
        if day_of_year < self.lifecycle_model.hibernation_end_day:
            self.current_phase = ColonyPhase.HIBERNATION

        elif day_of_year < (
            self.lifecycle_model.hibernation_end_day
            + self.lifecycle_model.colony_founding_duration
        ):
            if self.demographics.foundress_queens > 0:
                self.current_phase = ColonyPhase.FOUNDING
            else:
                self.current_phase = ColonyPhase.HIBERNATION

        elif day_of_year < self.lifecycle_model.reproduction_start_day:
            if self.demographics.total_population > 10:
                self.current_phase = ColonyPhase.GROWTH
            elif self.demographics.foundress_queens > 0:
                self.current_phase = ColonyPhase.FOUNDING
            else:
                self.current_phase = ColonyPhase.DEATH

        elif day_of_year < self.lifecycle_model.colony_decline_start_day:
            if self.demographics.total_population > 5:
                self.current_phase = ColonyPhase.REPRODUCTION
            else:
                self.current_phase = ColonyPhase.DECLINE

        elif day_of_year < self.lifecycle_model.hibernation_start_day:
            self.current_phase = ColonyPhase.DECLINE

        else:
            self.current_phase = ColonyPhase.HIBERNATION

        # Reset phase counter if phase changed
        if old_phase != self.current_phase:
            self.days_in_current_phase = 0
        else:
            self.days_in_current_phase += 1

    def get_egg_laying_rate(self) -> float:
        """Calculate current egg laying rate based on phase and colony state"""
        if self.current_phase == ColonyPhase.HIBERNATION:
            return 0.0
        elif self.current_phase == ColonyPhase.FOUNDING:
            return self.lifecycle_model.foundress_egg_laying_rate
        elif self.current_phase == ColonyPhase.GROWTH:
            # Ramp up to peak laying rate
            progress = min(1.0, self.days_in_current_phase / 30)
            return self.lifecycle_model.foundress_egg_laying_rate + progress * (
                self.lifecycle_model.peak_egg_laying_rate
                - self.lifecycle_model.foundress_egg_laying_rate
            )
        elif self.current_phase == ColonyPhase.REPRODUCTION:
            return self.lifecycle_model.peak_egg_laying_rate
        elif self.current_phase == ColonyPhase.DECLINE:
            # Decline in egg laying
            decline_progress = min(1.0, self.days_in_current_phase / 60)
            return self.lifecycle_model.peak_egg_laying_rate * (1 - decline_progress)
        else:
            return 0.0

    def determine_new_caste(self) -> BumblebeeCaste:
        """Determine caste of new individuals based on colony phase and size"""
        if self.current_phase in [ColonyPhase.FOUNDING, ColonyPhase.GROWTH]:
            return BumblebeeCaste.WORKER

        elif self.current_phase == ColonyPhase.REPRODUCTION:
            # Switch point logic: produce reproductives when colony reaches threshold
            size_ratio = (
                self.demographics.total_population
                / self.lifecycle_model.max_colony_size
            )

            if size_ratio >= self.lifecycle_model.switch_point_trigger:
                # Produce both males and gynes
                if (
                    random.random() < 0.6
                ):  # Slight bias toward males early in reproductive phase
                    return BumblebeeCaste.MALE
                else:
                    return BumblebeeCaste.GYNE
            else:
                # Still producing workers
                return BumblebeeCaste.WORKER

        elif self.current_phase == ColonyPhase.DECLINE:
            # Mostly reproductives, few workers
            if random.random() < 0.8:
                return (
                    BumblebeeCaste.GYNE
                    if random.random() < 0.4
                    else BumblebeeCaste.MALE
                )
            else:
                return BumblebeeCaste.WORKER

        else:
            return BumblebeeCaste.WORKER  # Default

    def apply_daily_mortality(self) -> None:
        """Apply daily mortality to different castes"""
        # Worker mortality
        worker_deaths = 0
        for _ in range(self.demographics.workers):
            if random.random() < self.lifecycle_model.worker_daily_mortality:
                worker_deaths += 1
        self.demographics.workers = max(0, self.demographics.workers - worker_deaths)

        # Foundress queen mortality (lower rate)
        if (
            self.demographics.foundress_queens > 0
            and random.random() < self.lifecycle_model.worker_daily_mortality * 0.3
        ):
            self.demographics.foundress_queens = 0
            # Colony collapse if foundress dies early
            if self.current_phase in [ColonyPhase.FOUNDING, ColonyPhase.GROWTH]:
                self.current_phase = ColonyPhase.DEATH
                self.is_active = False

        # Male mortality (higher rate - they don't live long)
        male_deaths = 0
        for _ in range(self.demographics.males):
            if random.random() < self.lifecycle_model.worker_daily_mortality * 2:
                male_deaths += 1
        self.demographics.males = max(0, self.demographics.males - male_deaths)

        # Gyne mortality (low rate - they need to survive)
        gyne_deaths = 0
        for _ in range(self.demographics.gynes):
            if random.random() < self.lifecycle_model.worker_daily_mortality * 0.5:
                gyne_deaths += 1
        self.demographics.gynes = max(0, self.demographics.gynes - gyne_deaths)

    def produce_new_individuals(self, day_of_year: int) -> None:
        """Produce new individuals based on lifecycle phase"""
        if not self.is_active:
            return

        eggs_laid = self.get_egg_laying_rate()

        # Simple development model - individuals emerge after development time
        for _ in range(int(eggs_laid)):
            new_caste = self.determine_new_caste()

            # Add to appropriate caste (simplified - immediate emergence)
            if new_caste == BumblebeeCaste.WORKER:
                self.demographics.workers += 1
            elif new_caste == BumblebeeCaste.MALE:
                self.demographics.males += 1
            elif new_caste == BumblebeeCaste.GYNE:
                self.demographics.gynes += 1

            self.total_individuals_produced += 1
            self.total_eggs_laid += 1

    def prepare_for_hibernation(self) -> None:
        """Prepare gynes for hibernation - most colony members die"""
        if self.current_phase == ColonyPhase.HIBERNATION:
            # Only mated gynes survive winter
            mated_gynes = int(
                self.demographics.gynes * 0.8
            )  # Assume 80% mating success
            surviving_gynes = 0

            for _ in range(mated_gynes):
                if random.random() < self.lifecycle_model.winter_queen_survival:
                    surviving_gynes += 1

            self.hibernating_gynes = surviving_gynes

            # Rest of colony dies
            self.demographics.foundress_queens = 0
            self.demographics.workers = 0
            self.demographics.males = 0
            self.demographics.gynes = 0

            self.is_active = False

    def attempt_colony_founding(self, day_of_year: int) -> bool:
        """Attempt to found new colony from hibernating gyne"""
        if (
            self.hibernating_gynes > 0
            and day_of_year >= self.lifecycle_model.hibernation_end_day
        ):
            # Foundress mortality - many gynes fail to establish colonies
            if random.random() > self.lifecycle_model.foundress_mortality_rate:
                self.demographics.foundress_queens = 1
                self.hibernating_gynes -= 1
                self.is_active = True
                self.current_phase = ColonyPhase.FOUNDING
                self.founded_day = day_of_year
                self.age_days = 0
                return True

        return False

    def get_foraging_capacity(self) -> int:
        """Get number of individuals available for foraging"""
        return self.demographics.foraging_population

    def get_colony_summary(self) -> Dict[str, Any]:
        """Get summary of colony state"""
        return {
            "colony_id": self.colony_id,
            "species": self.species,
            "phase": self.current_phase.value,
            "age_days": self.age_days,
            "days_in_phase": self.days_in_current_phase,
            "is_active": self.is_active,
            "demographics": {
                "foundress_queens": self.demographics.foundress_queens,
                "workers": self.demographics.workers,
                "males": self.demographics.males,
                "gynes": self.demographics.gynes,
                "total": self.demographics.total_population,
                "foragers": self.demographics.foraging_population,
            },
            "hibernating_gynes": self.hibernating_gynes,
            "total_produced": self.total_individuals_produced,
            "stored_resources": {
                "honey": self.stored_honey,
                "pollen": self.stored_pollen,
            },
        }


class BumblebeeColonyManager:
    """Manages multiple bumblebee colonies with proper annual lifecycles"""

    def __init__(self, species: str = "bombus_terrestris"):
        self.species = species
        self.colonies: Dict[str, BumblebeeColony] = {}
        self.next_colony_id = 1
        self.lifecycle_model = BumblebeeColonyLifecycleModel()

    def create_hibernating_gynes(self, count: int) -> List[str]:
        """Create hibernating gynes that will attempt colony founding"""
        new_colony_ids = []

        for _ in range(count):
            colony_id = f"colony_{self.next_colony_id}"
            self.next_colony_id += 1

            colony = BumblebeeColony(
                colony_id=colony_id,
                species=self.species,
                current_phase=ColonyPhase.HIBERNATION,
                lifecycle_model=self.lifecycle_model,
            )
            colony.hibernating_gynes = 1
            colony.is_active = False

            self.colonies[colony_id] = colony
            new_colony_ids.append(colony_id)

        return new_colony_ids

    def update_all_colonies(self, day_of_year: int) -> None:
        """Update all colonies for one day"""
        colonies_to_remove = []

        for colony_id, colony in self.colonies.items():
            colony.age_days += 1
            colony.update_lifecycle_phase(day_of_year)

            if colony.is_active:
                colony.apply_daily_mortality()
                colony.produce_new_individuals(day_of_year)

                # Check if colony should die
                if (
                    colony.demographics.total_population == 0
                    or colony.current_phase == ColonyPhase.DEATH
                ):
                    colony.is_active = False

            # Handle hibernation transition
            if colony.current_phase == ColonyPhase.HIBERNATION and colony.is_active:
                colony.prepare_for_hibernation()

            # Attempt colony founding from hibernating gynes
            if not colony.is_active and colony.hibernating_gynes > 0:
                colony.attempt_colony_founding(day_of_year)

            # Remove colonies with no hibernating gynes and inactive
            if not colony.is_active and colony.hibernating_gynes == 0:
                colonies_to_remove.append(colony_id)

        # Clean up dead colonies
        for colony_id in colonies_to_remove:
            del self.colonies[colony_id]

    def get_active_colonies(self) -> List[BumblebeeColony]:
        """Get all actively foraging colonies"""
        return [colony for colony in self.colonies.values() if colony.is_active]

    def get_total_foraging_population(self) -> int:
        """Get total foraging population across all active colonies"""
        return sum(
            colony.get_foraging_capacity() for colony in self.get_active_colonies()
        )

    def get_population_summary(self) -> Dict[str, Any]:
        """Get summary of all colony populations"""
        active_colonies = self.get_active_colonies()
        hibernating_colonies = [
            c
            for c in self.colonies.values()
            if not c.is_active and c.hibernating_gynes > 0
        ]

        total_demographics = ColonyDemographics()
        for colony in active_colonies:
            total_demographics.foundress_queens += colony.demographics.foundress_queens
            total_demographics.workers += colony.demographics.workers
            total_demographics.males += colony.demographics.males
            total_demographics.gynes += colony.demographics.gynes

        return {
            "active_colonies": len(active_colonies),
            "hibernating_colonies": len(hibernating_colonies),
            "total_hibernating_gynes": sum(
                c.hibernating_gynes for c in hibernating_colonies
            ),
            "total_demographics": {
                "foundress_queens": total_demographics.foundress_queens,
                "workers": total_demographics.workers,
                "males": total_demographics.males,
                "gynes": total_demographics.gynes,
                "total": total_demographics.total_population,
                "foragers": total_demographics.foraging_population,
            },
            "lifecycle_phases": {
                phase.value: len(
                    [c for c in active_colonies if c.current_phase == phase]
                )
                for phase in ColonyPhase
            },
        }


# CRITICAL: This implements the annual lifecycle that is fundamental to bumblebee biology
# Honey bees have perennial colonies - this difference affects all aspects of modeling
