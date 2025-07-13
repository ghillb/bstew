"""
Colony management system for BSTEW
==================================

Manages bee populations, resources, and colony-level behaviors.
Integrates with the mathematical foundations for population dynamics.
"""

from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field, computed_field
from enum import Enum
import logging
import random

from .agents import BeeAgent, Queen, Worker, Forager, Drone, BeeRole, BeeStatus
from .mathematics import ColonyDynamics, ColonyParameters, ODEIntegrator
from ..components.genetics import GeneticSystem, Genotype, Sex
from ..components.development import (
    DevelopmentSystem,
    DevelopmentParameters,
    DevelopingBee,
    DeathCause,
)
from ..components.predation import PredationSystem
from ..components.proboscis_matching import ProboscisCorollaSystem
from ..components.species_system import SpeciesSystem
from ..components.mortality_tracking import MortalityTracker


class ColonyHealth(Enum):
    """Colony health status enumeration"""

    THRIVING = "thriving"
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DECLINING = "declining"
    COLLAPSED = "collapsed"


class ColonyNeeds(BaseModel):
    """Assessment of colony resource and population needs"""

    model_config = {"validate_assignment": True}

    foragers: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Forager need level (0-1)"
    )
    nurses: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Nurse need level (0-1)"
    )
    guards: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Guard need level (0-1)"
    )
    builders: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Builder need level (0-1)"
    )
    food: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Food need level (0-1)"
    )
    space: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Space need level (0-1)"
    )


class ResourceStores(BaseModel):
    """Colony resource storage"""

    model_config = {"validate_assignment": True}

    pollen: float = Field(default=0.0, ge=0.0, description="Pollen stores in mg")
    nectar: float = Field(default=0.0, ge=0.0, description="Nectar stores in mg")
    honey: float = Field(default=0.0, ge=0.0, description="Honey stores in mg")
    wax: float = Field(default=0.0, ge=0.0, description="Wax stores in mg")
    propolis: float = Field(default=0.0, ge=0.0, description="Propolis stores in mg")

    @computed_field
    def total_food(self) -> float:
        """Total food resources available"""
        return self.pollen + self.nectar + self.honey


class DiseaseStatus(BaseModel):
    """Colony disease and pest status"""

    model_config = {"validate_assignment": True}

    varroa_mites: int = Field(default=0, ge=0, description="Number of Varroa mites")
    virus_load: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Viral infection level (0-1)"
    )
    nosema_spores: int = Field(default=0, ge=0, description="Number of Nosema spores")
    bacterial_infection: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Bacterial infection level (0-1)"
    )
    fungal_infection: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fungal infection level (0-1)"
    )

    @computed_field
    def infestation_level(self) -> float:
        """Overall infestation level (0-1)"""
        return min(
            1.0,
            (
                self.varroa_mites / 1000.0
                + self.virus_load
                + self.nosema_spores / 10000.0
                + self.bacterial_infection
                + self.fungal_infection
            )
            / 5.0,
        )


class Colony:
    """
    Colony management system integrating agent-based and mathematical models.

    Manages:
    - Bee populations by role and life stage
    - Resource stores and consumption
    - Disease and pest dynamics
    - Colony-level behaviors and decisions
    """

    def __init__(
        self,
        model: Any,
        species: str,
        location: Tuple[float, float],
        initial_population: Any,
        unique_id: int = None,  # type: ignore[assignment]
    ):
        self.model = model
        self.species = species
        self.location = location
        self.established_date = model.schedule.steps
        self.unique_id = unique_id or model.next_id()

        # Population tracking
        self.bees: List[BeeAgent] = []
        self.population_counts: Dict[str, int] = {
            "queens": 0,
            "workers": 0,
            "foragers": 0,
            "drones": 0,
            "brood": 0,
            "eggs": 0,
        }

        # Genetic system integration
        self.genetic_system = GeneticSystem(initial_allele_count=100)
        self.brood_genotypes: List[
            Genotype
        ] = []  # Store genotypes for developing brood
        self.stress_factors: Dict[
            str, float
        ] = {}  # Track colony stress from various sources

        # Species system integration
        self.species_system = SpeciesSystem()
        self.species_params = self.species_system.get_species_parameters(species)

        # Development system integration (species-specific if available)
        if self.species_params:
            self.development_params = self.species_params.development_parameters
        else:
            self.development_params = DevelopmentParameters()
        self.development_system = DevelopmentSystem(self.development_params)

        # Colony care capacity
        self.care_capacity = {"energy": 0.0, "incubation": 0.0, "feeding": 0.0}

        # Proboscis-corolla matching system
        self.proboscis_system = ProboscisCorollaSystem()

        # Mortality tracking system
        self.mortality_tracker = MortalityTracker(max_history_days=365)

        # Mathematical model integration
        self.colony_params = ColonyParameters()
        self.dynamics = ColonyDynamics(self.colony_params)
        self.integrator = ODEIntegrator(self.dynamics)

        # Colony state
        self.resources = ResourceStores()
        self.disease_status = DiseaseStatus()
        self.health = ColonyHealth.HEALTHY
        self.pheromone_level = 100.0
        self.comb_cells = 10000  # Available comb cells
        self.occupied_cells = 0

        # Colony behavior parameters (species-specific if available)
        if self.species_params:
            self.foraging_range = self.species_params.foraging_range_m
            self.max_colony_size = self.species_params.max_colony_size
            self.typical_colony_size = self.species_params.typical_colony_size
            self.swarm_threshold = self.species_params.max_colony_size * 0.8
            self.collapse_threshold = self.species_params.typical_colony_size * 0.1
        else:
            self.foraging_range = 2000.0  # meters
            self.max_colony_size = 50000
            self.typical_colony_size = 25000
            self.swarm_threshold = 40000  # bee population
            self.collapse_threshold = 2500  # minimum viable population

        self.territorial_radius = 100.0  # meters

        # Predation vulnerability
        self.predation_risk = 0.0
        self.last_predation_check = 0
        self.predation_defenses = 0.0  # Colony defense capability

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize population
        self.initialize_population(initial_population)

    def initialize_population(self, initial_pop: Any) -> None:
        """Initialize colony with starting population"""
        # Handle both integer and dictionary inputs for backward compatibility
        if isinstance(initial_pop, int):
            # Convert integer to dictionary format
            total_pop = initial_pop
            initial_pop = {
                "queens": 1,
                "workers": max(0, total_pop - 1),
                "foragers": 0,
                "drones": 0,
            }
        elif not isinstance(initial_pop, dict):
            # Default population if invalid input
            initial_pop = {"queens": 1, "workers": 100, "foragers": 20, "drones": 0}

        # Create queen with founder genotype
        if initial_pop.get("queens", 0) > 0:
            queen_genotype = self.genetic_system.create_founder_genotype(sex=Sex.FEMALE)
            queen = Queen(
                self.model.next_id(), self.model, self, genotype=queen_genotype
            )
            queen.genetic_system = self.genetic_system  # Connect genetic system
            self.add_bee(queen)

        # Create workers with founder genotypes
        for _ in range(initial_pop.get("workers", 0)):
            worker_genotype = self.genetic_system.create_founder_genotype(
                sex=Sex.FEMALE
            )
            worker = Worker(
                self.model.next_id(), self.model, self, genotype=worker_genotype
            )
            self.add_bee(worker)

        # Create foragers with founder genotypes
        for _ in range(initial_pop.get("foragers", 0)):
            forager_genotype = self.genetic_system.create_founder_genotype(
                sex=Sex.FEMALE
            )
            forager = Forager(
                self.model.next_id(), self.model, self, genotype=forager_genotype
            )
            self.add_bee(forager)

        # Create drones with founder genotypes
        for _ in range(initial_pop.get("drones", 0)):
            drone_genotype = self.genetic_system.create_founder_genotype(sex=Sex.MALE)
            drone = Drone(
                self.model.next_id(), self.model, self, genotype=drone_genotype
            )
            self.add_bee(drone)

        # Set initial brood
        self.population_counts["brood"] = initial_pop.get("brood", 1000)
        self.population_counts["eggs"] = initial_pop.get("eggs", 500)

        self.update_population_counts()

    def add_bee(self, bee: BeeAgent) -> None:
        """Add a bee to the colony"""
        bee.colony = self
        bee.location = self.location

        # Set proboscis characteristics based on species
        if hasattr(self, "proboscis_system") and self.proboscis_system:
            bee.proboscis_characteristics = self.proboscis_system.get_species_proboscis(
                self.species
            )

        self.bees.append(bee)
        self.model.schedule.add(bee)
        self.update_population_counts()

    def remove_bee(self, bee: BeeAgent) -> None:
        """Remove a bee from the colony"""
        if bee in self.bees:
            self.bees.remove(bee)
        self.update_population_counts()

    def update_population_counts(self) -> None:
        """Update population counts by role"""
        self.population_counts = {
            "queens": len([b for b in self.bees if isinstance(b, Queen)]),
            "workers": len(
                [
                    b
                    for b in self.bees
                    if isinstance(b, Worker) and not isinstance(b, Forager)
                ]
            ),
            "foragers": len([b for b in self.bees if isinstance(b, Forager)]),
            "drones": len([b for b in self.bees if isinstance(b, Drone)]),
            "brood": self.population_counts.get("brood", 0),
            "eggs": self.population_counts.get("eggs", 0),
        }

    def get_total_population(self) -> int:
        """Get total bee population including brood"""
        return sum(self.population_counts.values())

    def get_adult_population(self) -> int:
        """Get adult bee population only"""
        return len(self.bees)

    @property
    def agents(self) -> List[BeeAgent]:
        """Get all bees in colony (alias for bees attribute for compatibility)"""
        return self.bees

    def get_bees(self) -> List[BeeAgent]:
        """Get all bees in colony"""
        return [bee for bee in self.bees if bee.status != BeeStatus.DEAD]

    def get_bees_by_role(self, role: BeeRole) -> List[BeeAgent]:
        """Get bees of specific role"""
        return [
            bee
            for bee in self.bees
            if bee.role == role and bee.status != BeeStatus.DEAD
        ]

    def step(self) -> None:
        """Execute one simulation step for the colony"""
        self.update_population_counts()
        self.calculate_care_capacity()
        self.process_individual_development()
        self.process_brood_development()
        self.update_resource_dynamics()
        self.update_disease_dynamics()
        self.assess_colony_health()
        self.make_colony_decisions()

    def process_brood_development(self) -> None:
        """Process brood development using mathematical model"""
        # Get current state
        Bo = self.population_counts["brood"]
        H = self.population_counts["workers"] + self.population_counts["foragers"]
        fp = self.resources.pollen
        fn = self.resources.nectar

        # Calculate egg laying rate
        queens = self.get_bees_by_role(BeeRole.QUEEN)
        if queens:
            queen = queens[0]
            L = self.dynamics.egg_laying_rate(H, fp, fn, queen.age)
        else:
            L = 0.0

        # Calculate brood development
        dBo_dt = self.dynamics.brood_development(Bo, L, H, fp, fn)

        # Update brood count
        self.population_counts["brood"] += int(dBo_dt)
        self.population_counts["brood"] = max(0, self.population_counts["brood"])

        # Process brood emergence
        emergence_rate = (
            Bo
            / self.colony_params.development_time
            * self.dynamics.survival_function(H, fp, fn)
        )

        emerging_bees = int(self.model.random.poisson(emergence_rate))

        for _ in range(emerging_bees):
            if self.population_counts["brood"] > 0:
                self.population_counts["brood"] -= 1

                # Get genotype for this brood if available
                genotype = None
                if self.brood_genotypes:
                    genotype = self.brood_genotypes.pop(0)

                # Create bee based on genotype or default logic
                if genotype:
                    if genotype.sex == Sex.MALE:
                        drone = Drone(
                            self.model.next_id(), self.model, self, genotype=genotype
                        )
                        self.add_bee(drone)
                    elif genotype.sex == Sex.FEMALE:
                        worker = Worker(
                            self.model.next_id(), self.model, self, genotype=genotype
                        )
                        self.add_bee(worker)
                    # Diploid males die immediately and are handled in agent constructor
                else:
                    # Fallback to original logic
                    if self.model.random.random() < 0.95:
                        worker = Worker(self.model.next_id(), self.model, self)
                        self.add_bee(worker)
                    else:
                        drone = Drone(self.model.next_id(), self.model, self)
                        self.add_bee(drone)

    def add_brood(self) -> None:
        """Add new brood (called by queen egg laying)"""
        if self.occupied_cells < self.comb_cells:
            self.population_counts["brood"] += 1
            self.occupied_cells += 1

    def add_brood_with_genotype(self, genotype: Genotype) -> None:
        """Add new brood with specific genotype (genetic system integration)"""
        if self.occupied_cells < self.comb_cells:
            # Add to development system instead of simple brood count
            self.development_system.add_egg(genotype)
            self.occupied_cells += 1

            # Update population counts
            self.population_counts["brood"] += 1
            self.population_counts["eggs"] += 1

    def add_stress(self, cause: str, intensity: float) -> None:
        """Add stress factor to colony"""
        self.stress_factors[cause] = intensity

        # Apply immediate effects based on stress type
        if cause == "diploid_male_production":
            # Reduce egg laying efficiency
            # Reduce egg laying efficiency - access through hasattr check
            if hasattr(self.colony_params, "max_egg_laying_rate"):
                self.colony_params.max_egg_laying_rate *= 1.0 - intensity
            # Increase worker mortality
            for bee in self.bees:
                if isinstance(bee, Worker):
                    bee.disease_load += intensity * 0.1

    def update_resource_dynamics(self) -> None:
        """Update resource stores using mathematical model"""
        total_bees = self.get_adult_population()
        foragers = len(self.get_bees_by_role(BeeRole.FORAGER))

        # Assume half foragers collect pollen, half nectar
        pollen_foragers = foragers / 2.0
        nectar_foragers = foragers / 2.0

        # Update pollen
        dfp_dt = self.dynamics.pollen_dynamics(
            self.resources.pollen, pollen_foragers, total_bees
        )
        self.resources.pollen += dfp_dt
        self.resources.pollen = max(0.0, self.resources.pollen)

        # Update nectar
        dfn_dt = self.dynamics.nectar_dynamics(
            self.resources.nectar, nectar_foragers, total_bees
        )
        self.resources.nectar += dfn_dt
        self.resources.nectar = max(0.0, self.resources.nectar)

        # Convert nectar to honey (simplified)
        honey_conversion_rate = 0.1  # fraction of nectar converted to honey per day
        converted = self.resources.nectar * honey_conversion_rate
        self.resources.nectar -= converted
        self.resources.honey += converted * 0.8  # Concentration factor

    def update_disease_dynamics(self) -> None:
        """Update disease and pest populations"""
        brood_count = self.population_counts["brood"]

        # Varroa mite dynamics
        dV_dt = self.dynamics.varroa_dynamics(
            self.disease_status.varroa_mites, brood_count
        )
        self.disease_status.varroa_mites += int(dV_dt)
        self.disease_status.varroa_mites = max(0, self.disease_status.varroa_mites)

        # Virus transmission from mites
        if self.disease_status.varroa_mites > 0:
            total_bees = self.get_adult_population()
            if total_bees > 0:
                transmission_rate = min(
                    0.1, self.disease_status.varroa_mites / total_bees
                )
                self.disease_status.virus_load += transmission_rate * 0.01
                self.disease_status.virus_load = min(
                    1.0, self.disease_status.virus_load
                )

        # Natural virus recovery
        self.disease_status.virus_load *= 0.99

    def assess_colony_health(self) -> None:
        """Assess overall colony health status"""
        self.get_total_population()
        adult_pop = self.get_adult_population()

        # Population health
        pop_score = min(1.0, adult_pop / 20000.0)  # Optimal around 20k bees

        # Resource health
        total_food = (
            self.resources.total_food()
            if callable(self.resources.total_food)
            else self.resources.total_food
        )
        resource_score = min(1.0, total_food / 5000.0)  # 5kg optimal

        # Disease health
        infestation_level: float = self.disease_status.infestation_level  # type: ignore[assignment]
        disease_score = 1.0 - infestation_level

        # Overall health score
        health_score = (pop_score + resource_score + disease_score) / 3.0

        if health_score > 0.8:
            self.health = ColonyHealth.THRIVING
        elif health_score > 0.6:
            self.health = ColonyHealth.HEALTHY
        elif health_score > 0.4:
            self.health = ColonyHealth.STRESSED
        elif health_score > 0.2:
            self.health = ColonyHealth.DECLINING
        else:
            self.health = ColonyHealth.COLLAPSED

    def make_colony_decisions(self) -> None:
        """Make colony-level behavioral decisions"""
        # Swarm decision
        if self.get_adult_population() > self.swarm_threshold and self.health in [
            ColonyHealth.THRIVING,
            ColonyHealth.HEALTHY,
        ]:
            self.prepare_swarm()

        # Collapse check
        if self.get_adult_population() < self.collapse_threshold:
            self.trigger_collapse()

        # Seasonal preparation
        self.prepare_for_season()

    def prepare_swarm(self) -> None:
        """Prepare for swarming behavior"""
        # Create new queen cells
        # Reduce foraging
        # Prepare swarm cluster
        pass

    def trigger_collapse(self) -> None:
        """Handle colony collapse"""
        self.health = ColonyHealth.COLLAPSED
        # Remove remaining bees
        for bee in self.bees[:]:
            bee.die()

    def prepare_for_season(self) -> None:
        """Seasonal behavioral adjustments"""
        season = self.model.get_current_season()

        if season == "winter":
            # Reduce activity, form cluster
            self.colony_params.max_egg_laying_rate *= 0.1
        elif season == "spring":
            # Increase activity, build up population
            self.colony_params.max_egg_laying_rate *= 1.5
        elif season == "summer":
            # Peak activity
            self.colony_params.max_egg_laying_rate *= 1.0
        elif season == "autumn":
            # Prepare for winter
            self.colony_params.max_egg_laying_rate *= 0.7

    def get_resource_adequacy(self) -> float:
        """Calculate resource adequacy for egg laying"""
        total_food = (
            self.resources.total_food()
            if callable(self.resources.total_food)
            else self.resources.total_food
        )
        total_bees = self.get_adult_population()

        if total_bees == 0:
            return 0.0

        food_per_bee = total_food / total_bees
        optimal_food_per_bee = 1.0  # mg per bee

        return min(1.0, food_per_bee / optimal_food_per_bee)

    def get_population_pressure(self) -> float:
        """Calculate population pressure for egg laying"""
        current_pop = self.get_adult_population()
        max_capacity = self.comb_cells * 0.5  # Assume 50% for adults

        if current_pop >= max_capacity:
            return 0.1  # Severely limited
        else:
            return 1.0 - (current_pop / max_capacity)

    def assess_needs(self) -> ColonyNeeds:
        """Assess colony needs for different bee roles"""
        needs = ColonyNeeds()

        total_pop = self.get_adult_population()
        if total_pop == 0:
            return needs

        # Forager needs based on resource levels
        if self.resources.total_food < total_pop * 2.0:  # type: ignore
            needs.foragers = 0.8

        # Nurse needs based on brood ratio
        brood_ratio = self.population_counts["brood"] / total_pop
        if brood_ratio > 0.3:
            needs.nurses = 0.7

        # Food needs
        if self.resources.total_food < total_pop * 1.0:  # type: ignore
            needs.food = 0.9

        return needs

    def add_resources(self, resource_type: str, amount: float) -> None:
        """Add resources to colony stores"""
        if resource_type == "pollen":
            self.resources.pollen += amount
        elif resource_type == "nectar":
            self.resources.nectar += amount
        elif resource_type == "honey":
            self.resources.honey += amount

    def consume_resources(self, amount: float) -> None:
        """Consume resources for bee maintenance"""
        # Prefer honey, then nectar, then pollen
        if self.resources.honey >= amount:
            self.resources.honey -= amount
        elif self.resources.nectar >= amount:
            self.resources.nectar -= amount
        elif self.resources.pollen >= amount:
            self.resources.pollen -= amount

    def tend_brood(self, nurse_bee: BeeAgent) -> None:
        """Handle brood tending by nurse bee"""
        # Improve brood survival
        # Consume resources for brood care
        self.consume_resources(1.0)

    def build_comb(self, builder_bee: BeeAgent) -> None:
        """Handle comb construction"""
        if self.resources.honey > 10.0:  # Need honey to make wax
            self.resources.honey -= 10.0
            self.resources.wax += 1.0
            self.comb_cells += 10

    def get_brood_count(self) -> int:
        """Get current brood count"""
        return self.population_counts["brood"]

    def get_genetic_summary(self) -> Dict[str, Any]:
        """Get genetic summary for colony"""
        all_genotypes = []

        # Collect genotypes from all bees
        for bee in self.bees:
            if bee.genotype:
                all_genotypes.append(bee.genotype)

        # Add brood genotypes
        all_genotypes.extend(self.brood_genotypes)

        if not all_genotypes:
            return {}

        return self.genetic_system.get_genetic_summary(all_genotypes)

    def get_stress_level(self) -> float:
        """Get overall colony stress level"""
        if not self.stress_factors:
            return 0.0

        # Calculate weighted average of stress factors
        total_stress = sum(self.stress_factors.values())
        return min(1.0, total_stress / len(self.stress_factors))

    def calculate_care_capacity(self) -> None:
        """Calculate available care capacity from adult bees"""
        nurses = self.get_bees_by_role(BeeRole.NURSE)
        workers = [
            bee
            for bee in self.get_bees_by_role(BeeRole.WORKER)
            if bee.role == BeeRole.NURSE
        ]

        total_nurses = len(nurses) + len(workers)

        # Each nurse can provide care
        self.care_capacity = {
            "energy": total_nurses * 0.5,  # Energy units per nurse per day
            "incubation": total_nurses * 1.0,  # Incubation units per nurse per day
            "feeding": total_nurses * 2.0,  # Feeding units per nurse per day
        }

        # Resource limitations
        available_food = (
            self.resources.total_food()
            if callable(self.resources.total_food)
            else self.resources.total_food
        )
        if available_food < total_nurses * 2.0:  # Minimum food per nurse
            food_factor = (
                available_food / (total_nurses * 2.0) if total_nurses > 0 else 0.0
            )
            self.care_capacity["energy"] *= food_factor
            self.care_capacity["feeding"] *= food_factor

    def process_individual_development(self) -> None:
        """Process individual bee development using development system"""
        # Get current temperature (simplified - could be from weather system)
        current_temp = 32.0  # Assume nest temperature regulation

        # Process development step
        dev_results = self.development_system.step(current_temp, self.care_capacity)

        # Handle emerged bees
        for emerged_bee in dev_results["emerged_bees"]:
            self.create_adult_from_development(emerged_bee)

        # Handle deaths
        for dead_bee in dev_results["deaths"]:
            self.record_development_death(dead_bee)

        # Update population counts based on development system
        dev_population = self.development_system.get_population_by_stage()
        from ..components.development import DevelopmentStage

        self.population_counts["eggs"] = dev_population.get(DevelopmentStage.EGG, 0)
        # Note: 'brood' now includes larvae and pupae
        self.population_counts["brood"] = dev_population.get(
            DevelopmentStage.LARVA, 0
        ) + dev_population.get(DevelopmentStage.PUPA, 0)

    def create_adult_from_development(self, developed_bee: DevelopingBee) -> None:
        """Create adult bee from successfully developed bee"""
        bee_type = developed_bee.get_target_bee_type()

        if bee_type == "drone":
            adult_bee = Drone(
                self.model.next_id(), self.model, self, genotype=developed_bee.genotype
            )
        elif bee_type == "worker":
            adult_bee = Worker(
                self.model.next_id(), self.model, self, genotype=developed_bee.genotype
            )
        else:
            return  # Dead or unknown type

        # Transfer development history
        adult_bee.development_history = {  # type: ignore[assignment]
            "total_development_time": developed_bee.age_days,
            "final_weight": developed_bee.weight_mg,
            "care_received": developed_bee.cumul_energy_received,
            "incubation_received": developed_bee.cumul_incubation_received,
            "stress_factors": developed_bee.stress_factors.copy(),
        }

        # Add to colony
        self.add_bee(adult_bee)

    def record_development_death(self, dead_bee: DevelopingBee) -> None:
        """Record death during development"""
        # Update colony stress based on death cause
        if dead_bee.death_cause is not None:
            if dead_bee.death_cause.value == "diploid_male":
                self.add_stress("diploid_male_death", 0.1)
            elif dead_bee.death_cause.value == "no_adult_care":
                self.add_stress("insufficient_care", 0.2)
            elif dead_bee.death_cause.value == "insufficient_incubation":
                self.add_stress("temperature_regulation", 0.15)

            # Log death for analysis
            self.logger.warning(
                f"Development death: {dead_bee.death_cause.value} at stage {dead_bee.stage.value}"
            )

    def get_development_summary(self) -> Dict[str, Any]:
        """Get comprehensive development summary"""
        return self.development_system.get_development_summary()

    def apply_environmental_stress_to_brood(
        self, stress_type: str, intensity: float
    ) -> None:
        """Apply environmental stress to developing brood"""
        self.development_system.apply_environmental_stress(stress_type, intensity)

    def end_season_brood_mortality(self) -> None:
        """Apply end-of-season mortality to all developing brood"""
        self.development_system.end_season_mortality()

    def update_predation_risk(self, predation_system: "PredationSystem") -> None:
        """Update predation risk assessment for colony"""
        risk_assessment = predation_system.get_colony_risk_assessment(self.location)
        self.predation_risk = risk_assessment["total_risk"]

        # Update colony stress based on predation risk
        if self.predation_risk > 0.2:
            self.add_stress("predation_pressure", self.predation_risk)

        # Adjust colony behavior based on risk
        if self.predation_risk > 0.1:
            # Reduce foraging activity
            self.foraging_range *= 0.8
            # Increase defensive behaviors
            self.predation_defenses = min(1.0, self.predation_defenses + 0.1)

    def handle_predation_attack(self, attack_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle predation attack on colony"""

        # Record attack
        attack_result = {
            "attack_day": attack_details["day"],
            "predator_type": attack_details.get("predator_type", "unknown"),
            "colony_survived": True,
            "casualties": [],
            "resource_loss": {},
        }

        # Apply colony defenses
        defense_effectiveness = self.predation_defenses * 0.3  # 30% max defense
        attack_success_probability = attack_details.get("success_probability", 0.1)

        # Adjust success probability based on defenses
        adjusted_success = attack_success_probability * (1.0 - defense_effectiveness)

        # Check if attack succeeds
        if random.random() < adjusted_success:
            # Colony is destroyed
            attack_result["colony_survived"] = False
            attack_result = self.handle_colony_destruction(attack_result)
        else:
            # Colony survives but may have casualties
            attack_result = self.handle_partial_predation(attack_result)

        # Increase stress and defensive behaviors
        self.add_stress("predation_attack", 0.5)
        self.predation_defenses = min(1.0, self.predation_defenses + 0.2)

        return attack_result

    def handle_colony_destruction(
        self, attack_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle complete colony destruction"""

        # Record pre-destruction state
        attack_result["pre_destruction_population"] = self.get_total_population()
        attack_result["pre_destruction_resources"] = {
            "pollen": self.resources.pollen,
            "nectar": self.resources.nectar,
            "honey": self.resources.honey,
        }

        # Kill all adult bees
        casualties = []
        for bee in self.bees[:]:
            casualties.append(
                {
                    "id": bee.unique_id,
                    "role": bee.role.value,
                    "age": bee.age,
                    "genotype_summary": (
                        bee.genotype.get_allele_ids() if bee.genotype else None
                    ),
                }
            )
            bee.die()

        # Destroy all developing brood
        if hasattr(self, "development_system"):
            for bee_id, developing_bee in list(
                self.development_system.developing_bees.items()
            ):
                if developing_bee.is_alive():
                    casualties.append(
                        {
                            "id": bee_id,
                            "stage": developing_bee.stage.value,
                            "age": developing_bee.age_days,
                        }
                    )
                    developing_bee.development_success = False
                    developing_bee.death_cause = DeathCause.PREDATION

        # Consume resources
        resource_consumption = 0.8  # 80% of resources consumed
        attack_result["resource_loss"] = {
            "pollen": self.resources.pollen * resource_consumption,
            "nectar": self.resources.nectar * resource_consumption,
            "honey": self.resources.honey * resource_consumption,
        }

        self.resources.pollen *= 1.0 - resource_consumption
        self.resources.nectar *= 1.0 - resource_consumption
        self.resources.honey *= 1.0 - resource_consumption

        # Set colony as collapsed
        self.trigger_collapse()

        attack_result["casualties"] = casualties

        return attack_result

    def handle_partial_predation(self, attack_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle partial predation (colony survives but has casualties)"""

        # Kill some foragers (most vulnerable)
        foragers = self.get_bees_by_role(BeeRole.FORAGER)
        casualties_count = min(
            len(foragers), max(1, int(len(foragers) * 0.1))
        )  # 10% of foragers

        casualties = []
        for i in range(casualties_count):
            if foragers:
                victim = random.choice(foragers)
                casualties.append(
                    {
                        "id": victim.unique_id,
                        "role": victim.role.value,
                        "age": victim.age,
                    }
                )
                victim.die()
                foragers.remove(victim)

        # Small resource loss
        resource_loss = 0.1  # 10% of resources
        attack_result["resource_loss"] = {
            "pollen": self.resources.pollen * resource_loss,
            "nectar": self.resources.nectar * resource_loss,
            "honey": self.resources.honey * resource_loss,
        }

        self.resources.pollen *= 1.0 - resource_loss
        self.resources.nectar *= 1.0 - resource_loss
        self.resources.honey *= 1.0 - resource_loss

        attack_result["casualties"] = casualties

        return attack_result

    def get_predation_vulnerability(self) -> Dict[str, Any]:
        """Calculate colony vulnerability to predation"""

        # Base vulnerability factors
        population_factor = min(
            1.0, self.get_total_population() / 10000.0
        )  # Larger colonies more attractive
        resource_factor = min(
            1.0,
            self.resources.total_food / 5000.0,  # type: ignore
        )  # More resources = more attractive
        activity_factor = len(self.get_bees_by_role(BeeRole.FORAGER)) / max(
            1, self.get_adult_population()
        )  # Activity level

        # Defense factors
        defense_factor = 1.0 - self.predation_defenses
        health_factor = (
            1.0 if self.health.value in ["thriving", "healthy"] else 1.5
        )  # Weak colonies more vulnerable

        vulnerability = (
            population_factor
            * resource_factor
            * activity_factor
            * defense_factor
            * health_factor
        ) / 5.0

        return {
            "overall_vulnerability": min(1.0, vulnerability),
            "population_factor": population_factor,
            "resource_factor": resource_factor,
            "activity_factor": activity_factor,
            "defense_factor": defense_factor,
            "health_factor": health_factor,
            "current_risk": self.predation_risk,
            "defenses": self.predation_defenses,
        }

    def get_species_characteristics(self) -> Dict[str, Any]:
        """Get species-specific characteristics"""
        if not self.species_params:
            return {}

        return {
            "species_name": self.species_params.species_name,
            "species_type": self.species_params.species_type.value,
            "proboscis_length": self.species_params.proboscis_characteristics.length_mm,
            "body_size": self.species_params.body_size_mm,
            "foraging_range": self.species_params.foraging_range_m,
            "max_colony_size": self.species_params.max_colony_size,
            "typical_colony_size": self.species_params.typical_colony_size,
            "competition_strength": self.species_params.competition_strength,
            "social_dominance": self.species_params.social_dominance,
            "cold_tolerance": self.species_params.cold_tolerance,
            "drought_tolerance": self.species_params.drought_tolerance,
            "habitat_preferences": {
                "nest": self.species_params.nest_habitat_preferences,
                "foraging": self.species_params.foraging_habitat_preferences,
            },
            "phenology": {
                "emerging_day_mean": self.species_params.emerging_day_mean,
                "active_season_start": self.species_params.active_season_start,
                "active_season_end": self.species_params.active_season_end,
            },
        }

    def get_species_foraging_efficiency(
        self, environmental_conditions: Dict[str, Any]
    ) -> float:
        """Calculate species-specific foraging efficiency"""
        if not self.species_params:
            return 1.0

        base_efficiency = self.species_params.foraging_aggressiveness

        # Environmental adjustments
        temperature = environmental_conditions.get("temperature", 20.0)
        day_of_year = environmental_conditions.get("day_of_year", 150)

        # Temperature tolerance adjustment
        if temperature < 10.0:
            temp_efficiency = self.species_params.cold_tolerance
        elif temperature > 30.0:
            temp_efficiency = self.species_params.drought_tolerance
        else:
            temp_efficiency = 1.0

        # Seasonal activity adjustment
        if (
            self.species_params.active_season_start
            <= day_of_year
            <= self.species_params.active_season_end
        ):
            seasonal_efficiency = 1.0
        else:
            seasonal_efficiency = 0.1  # Reduced activity outside season

        return base_efficiency * temp_efficiency * seasonal_efficiency

    def calculate_interspecies_competition(
        self, competing_colonies: List["Colony"], resource_overlap: float = 0.5
    ) -> float:
        """Calculate competition effects from other species"""
        if not self.species_params or not competing_colonies:
            return 0.0

        total_competition = 0.0

        for competitor_colony in competing_colonies:
            if competitor_colony.species != self.species:
                competition_effect = self.species_system.calculate_competition_effect(
                    self.species, competitor_colony.species, resource_overlap
                )

                # Weight by competitor colony size and proximity
                competitor_strength = competitor_colony.get_adult_population() / 1000.0
                total_competition += competition_effect * competitor_strength

        return min(1.0, total_competition)

    def is_species_active(self, day_of_year: int) -> bool:
        """Check if species is active on given day"""
        if not self.species_params:
            return True

        return (
            self.species_params.active_season_start
            <= day_of_year
            <= self.species_params.active_season_end
        )

    def get_habitat_suitability(self, habitat_type: str) -> Tuple[float, float]:
        """Get habitat suitability for nesting and foraging"""
        if not self.species_params:
            return 0.5, 0.5

        return self.species_system.get_optimal_habitats(self.species, habitat_type)

    def get_honey_stores(self) -> float:
        """Get colony honey stores"""
        return self.resources.honey

    def get_health_score(self) -> float:
        """Get colony health score (0-1)"""
        # Simple health score based on various factors
        if self.health == ColonyHealth.THRIVING:
            return 1.0
        elif self.health == ColonyHealth.HEALTHY:
            return 0.8
        elif self.health == ColonyHealth.STRESSED:
            return 0.6
        elif self.health == ColonyHealth.DECLINING:
            return 0.4
        else:  # COLLAPSED
            return 0.0

    def get_forager_count(self) -> int:
        """Get number of forager bees"""
        return len(self.get_bees_by_role(BeeRole.FORAGER))
