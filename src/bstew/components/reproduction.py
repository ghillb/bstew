"""
Colony reproduction and swarming for BSTEW
==========================================

Implements queen supersedure, swarming behavior, and new colony establishment
based on colony health, population dynamics, and seasonal patterns.
"""

import random
from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field
from enum import Enum

from ..core.agents import Queen, Worker, Forager, BeeAgent
from ..core.colony import Colony


class SwarmingStage(Enum):
    """Stages of swarming process"""

    NOT_SWARMING = "not_swarming"
    PREPARATION = "preparation"
    QUEEN_CELLS = "queen_cells"
    SCOUT_SELECTION = "scout_selection"
    SITE_SELECTION = "site_selection"
    DEPARTURE = "departure"
    ESTABLISHMENT = "establishment"


class QueenState(Enum):
    """Queen reproductive states"""

    VIRGIN = "virgin"
    MATED = "mated"
    LAYING = "laying"
    DECLINING = "declining"
    SUPERSEDED = "superseded"
    DEAD = "dead"


class QueenCell(BaseModel):
    """Queen cell development tracking"""

    model_config = {"validate_assignment": True}

    cell_id: int = Field(ge=0, description="Unique cell identifier")
    age: int = Field(ge=0, description="Days since started")
    development_stage: str = Field(
        description="Development stage: egg, larva, pupa, emerged"
    )
    location: Tuple[float, float] = Field(description="Cell location coordinates")
    quality_score: float = Field(ge=0.0, le=1.0, description="Nursing quality score")
    emergency_cell: bool = Field(
        default=False, description="Emergency vs supersedure cell"
    )

    def get_emergence_probability(self) -> float:
        """Calculate probability queen will emerge successfully"""
        if self.age < 16:  # Minimum development time
            return 0.0
        elif self.age > 20:  # Too old
            return 0.1
        else:
            base_prob = 0.8
            quality_factor = self.quality_score
            return min(0.95, base_prob * quality_factor)


class SwarmingDecision(BaseModel):
    """Decision factors for swarming"""

    model_config = {"validate_assignment": True}

    population_pressure: float = Field(
        ge=0.0, le=1.0, description="Population pressure factor"
    )
    resource_adequacy: float = Field(
        ge=0.0, le=1.0, description="Resource adequacy factor"
    )
    queen_quality: float = Field(ge=0.0, le=1.0, description="Queen quality factor")
    seasonal_timing: float = Field(ge=0.0, le=1.0, description="Seasonal timing factor")
    congestion_level: float = Field(
        ge=0.0, le=1.0, description="Colony congestion level"
    )
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall swarming score")

    def should_swarm(self, threshold: float = 0.6) -> bool:
        """Determine if colony should initiate swarming"""
        return self.overall_score > threshold


class NestSite(BaseModel):
    """Potential nest site for new colony"""

    model_config = {"validate_assignment": True}

    site_id: int = Field(ge=0, description="Unique site identifier")
    location: Tuple[float, float] = Field(description="Site location coordinates")
    cavity_volume: float = Field(ge=0.0, description="Cavity volume in liters")
    entrance_size: float = Field(ge=0.0, description="Entrance size in cm²")
    protection_level: float = Field(
        ge=0.0, le=1.0, description="Protection level (0-1 scale)"
    )
    resource_proximity: float = Field(ge=0.0, description="Distance to resources")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    discovered_time: int = Field(ge=0, description="Discovery time")
    scout_count: int = Field(default=0, ge=0, description="Number of scouts visited")


class QueenSupersedure:
    """
    Manages queen supersedure process.

    Implements:
    - Queen quality assessment
    - Emergency queen rearing
    - Supersedure decision making
    - Queen replacement
    """

    def __init__(self, colony: Colony):
        self.colony = colony
        self.supersedure_cells: List[QueenCell] = []
        self.emergency_cells: List[QueenCell] = []
        self.cell_counter = 0

        # Supersedure parameters
        self.min_queen_age_for_supersedure = 180  # Days
        self.supersedure_probability_base = (
            0.02  # Daily probability when conditions met
        )
        self.emergency_threshold = 0.3  # Queen quality threshold for emergency

    def assess_queen_quality(self, queen: Queen, current_time: int) -> float:
        """Assess current queen quality"""
        if not queen:
            return 0.0

        quality_factors = {}

        # Age factor
        if queen.age < 180:  # Young queen
            quality_factors["age"] = 1.0
        elif queen.age < 365:  # Mature queen
            quality_factors["age"] = 0.8
        elif queen.age < 730:  # Aging queen
            quality_factors["age"] = 0.5
        else:  # Old queen
            quality_factors["age"] = 0.2

        # Egg laying rate
        recent_laying = getattr(queen, "daily_egg_production", 1500)
        max_laying = 2000
        quality_factors["laying"] = min(1.0, recent_laying / max_laying)

        # Pheromone production
        pheromone_strength = getattr(queen, "pheromone_strength", 0.8)
        quality_factors["pheromone"] = pheromone_strength

        # Health status
        health = 1.0
        if hasattr(queen, "disease_states") and queen.disease_states:
            health = 1.0 - sum(
                ds.infection_level for ds in queen.disease_states.values()
            ) / len(queen.disease_states)
        quality_factors["health"] = health

        # Weighted average
        weights = {"age": 0.3, "laying": 0.4, "pheromone": 0.2, "health": 0.1}
        overall_quality = sum(
            quality_factors[factor] * weights[factor] for factor in weights.keys()
        )

        return overall_quality

    def check_supersedure_conditions(self, current_time: int) -> bool:
        """Check if supersedure should be initiated"""
        queen = self.colony.get_queen()  # type: ignore[attr-defined]
        if not queen:
            return True  # Emergency replacement needed

        queen_quality = self.assess_queen_quality(queen, current_time)

        # Emergency supersedure
        if queen_quality < self.emergency_threshold:
            self._initiate_emergency_supersedure(current_time)
            return True

        # Normal supersedure conditions
        if queen.age > self.min_queen_age_for_supersedure:
            # Calculate supersedure probability
            age_factor = (queen.age - self.min_queen_age_for_supersedure) / 365.0
            quality_factor = 1.0 - queen_quality
            seasonal_factor = self._get_seasonal_supersedure_factor(current_time)

            probability = (
                self.supersedure_probability_base
                * age_factor
                * quality_factor
                * seasonal_factor
            )

            if random.random() < probability:
                self._initiate_supersedure(current_time)
                return True

        return False

    def _initiate_supersedure(self, current_time: int) -> None:
        """Initiate normal supersedure process"""
        # Create 3-5 supersedure cells
        num_cells = random.randint(3, 5)

        for i in range(num_cells):
            cell = QueenCell(
                cell_id=self._get_next_cell_id(),
                age=0,
                development_stage="egg",
                location=(random.uniform(-50, 50), random.uniform(-50, 50)),
                quality_score=random.uniform(0.7, 1.0),
                emergency_cell=False,
            )
            self.supersedure_cells.append(cell)

    def _initiate_emergency_supersedure(self, current_time: int) -> None:
        """Initiate emergency queen replacement"""
        # Create more emergency cells with faster development
        num_cells = random.randint(5, 8)

        for i in range(num_cells):
            cell = QueenCell(
                cell_id=self._get_next_cell_id(),
                age=0,
                development_stage="egg",
                location=(random.uniform(-50, 50), random.uniform(-50, 50)),
                quality_score=random.uniform(0.6, 0.9),
                emergency_cell=True,
            )
            self.emergency_cells.append(cell)

    def update_queen_cells(self, current_time: int) -> None:
        """Update development of queen cells"""
        all_cells = self.supersedure_cells + self.emergency_cells

        for cell in all_cells[:]:
            cell.age += 1

            # Update development stage
            if cell.age >= 16:
                cell.development_stage = "pupa"
            elif cell.age >= 6:
                cell.development_stage = "larva"

            # Check for emergence
            if cell.age >= 16:
                emergence_prob = cell.get_emergence_probability()

                if random.random() < emergence_prob:
                    self._emerge_new_queen(cell, current_time)

                    # Remove cell from tracking
                    if cell in self.supersedure_cells:
                        self.supersedure_cells.remove(cell)
                    if cell in self.emergency_cells:
                        self.emergency_cells.remove(cell)

    def _emerge_new_queen(self, cell: QueenCell, current_time: int) -> None:
        """Handle emergence of new queen"""
        # Create new queen
        new_queen = Queen(
            unique_id=self.colony.model.next_id(),
            model=self.colony.model,
            colony=self.colony,
        )
        new_queen.age = 0
        new_queen.state = QueenState.VIRGIN
        new_queen.quality_score = cell.quality_score

        # Add to colony
        self.colony.add_agent(new_queen)  # type: ignore[attr-defined]

        # Handle queen conflict if existing queen present
        existing_queen = self.colony.get_queen()  # type: ignore[attr-defined]
        if existing_queen and existing_queen != new_queen:
            self._resolve_queen_conflict(existing_queen, new_queen)

    def _resolve_queen_conflict(self, old_queen: Queen, new_queen: Queen) -> None:
        """Resolve conflict between old and new queens"""
        # Determine winner based on quality and age
        old_score = (
            self.assess_queen_quality(old_queen, 0) * 0.8
        )  # Existing queen disadvantage
        new_score = new_queen.quality_score

        if new_score > old_score:
            # New queen wins
            old_queen.state = QueenState.SUPERSEDED
            old_queen.die()
            new_queen.state = QueenState.VIRGIN
        else:
            # Old queen wins
            new_queen.die()

    def _get_seasonal_supersedure_factor(self, current_time: int) -> float:
        """Get seasonal factor for supersedure timing"""
        day_of_year = current_time % 365

        # Peak supersedure in spring/early summer
        if 90 <= day_of_year <= 180:  # April-June
            return 1.5
        elif 60 <= day_of_year <= 240:  # March-August
            return 1.0
        else:
            return 0.3

    def _get_next_cell_id(self) -> int:
        """Get next available cell ID"""
        self.cell_counter += 1
        return self.cell_counter

    def get_supersedure_status(self) -> Dict[str, Any]:
        """Get current supersedure status"""
        return {
            "active_supersedure_cells": len(self.supersedure_cells),
            "active_emergency_cells": len(self.emergency_cells),
            "total_cells_produced": self.cell_counter,
            "cell_details": [
                {
                    "cell_id": cell.cell_id,
                    "age": cell.age,
                    "stage": cell.development_stage,
                    "quality": cell.quality_score,
                    "emergency": cell.emergency_cell,
                }
                for cell in self.supersedure_cells + self.emergency_cells
            ],
        }


class SwarmingBehavior:
    """
    Manages swarming behavior and decision making.

    Implements:
    - Swarming triggers and timing
    - Scout bee selection and site evaluation
    - Swarm departure coordination
    - New colony establishment
    """

    def __init__(self, colony: Colony):
        self.colony = colony
        self.swarming_stage = SwarmingStage.NOT_SWARMING
        self.swarm_cells: List[QueenCell] = []
        self.potential_sites: List[NestSite] = []
        self.scout_bees: Set[int] = set()  # Agent IDs
        self.selected_site: Optional[NestSite] = None
        self.site_counter = 0

        # Swarming parameters
        self.swarming_season_start = 90  # Day of year (April)
        self.swarming_season_end = 180  # Day of year (June)
        self.min_population_for_swarming = 15000
        self.optimal_swarm_size = 0.6  # Fraction of population
        self.site_evaluation_days = 7  # Days for site evaluation

    def assess_swarming_conditions(self, current_time: int) -> SwarmingDecision:
        """Assess conditions for swarming initiation"""

        # Population pressure
        current_pop = self.colony.get_adult_population()
        optimal_pop = 20000
        population_pressure = min(
            1.0, max(0.0, (current_pop - optimal_pop) / optimal_pop)
        )

        # Resource adequacy
        resource_adequacy = self.colony.get_resource_adequacy()

        # Queen quality
        queen = self.colony.get_queen()  # type: ignore[attr-defined]
        queen_quality = 0.8 if queen and queen.age < 365 else 0.4

        # Seasonal timing
        day_of_year = current_time % 365
        if self.swarming_season_start <= day_of_year <= self.swarming_season_end:
            seasonal_timing = 1.0
        else:
            seasonal_timing = 0.1

        # Congestion level
        brood_count = self.colony.population_counts.get("brood", 0)
        congestion_level = min(1.0, brood_count / 15000)

        # Calculate overall score
        weights = {
            "population_pressure": 0.3,
            "resource_adequacy": 0.2,
            "queen_quality": 0.2,
            "seasonal_timing": 0.2,
            "congestion_level": 0.1,
        }

        overall_score = (
            weights["population_pressure"] * population_pressure
            + weights["resource_adequacy"] * resource_adequacy
            + weights["queen_quality"] * queen_quality
            + weights["seasonal_timing"] * seasonal_timing
            + weights["congestion_level"] * congestion_level
        )

        return SwarmingDecision(
            population_pressure=population_pressure,
            resource_adequacy=resource_adequacy,
            queen_quality=queen_quality,
            seasonal_timing=seasonal_timing,
            congestion_level=congestion_level,
            overall_score=overall_score,
        )

    def update_swarming_process(self, current_time: int) -> None:
        """Update swarming process based on current stage"""

        if self.swarming_stage == SwarmingStage.NOT_SWARMING:
            decision = self.assess_swarming_conditions(current_time)
            if (
                decision.should_swarm()
                and self.colony.get_adult_population()
                > self.min_population_for_swarming
            ):
                self._initiate_swarming(current_time)

        elif self.swarming_stage == SwarmingStage.PREPARATION:
            self._update_preparation_stage(current_time)

        elif self.swarming_stage == SwarmingStage.QUEEN_CELLS:
            self._update_queen_cell_stage(current_time)

        elif self.swarming_stage == SwarmingStage.SCOUT_SELECTION:
            self._update_scout_selection_stage(current_time)

        elif self.swarming_stage == SwarmingStage.SITE_SELECTION:
            self._update_site_selection_stage(current_time)

        elif self.swarming_stage == SwarmingStage.DEPARTURE:
            self._update_departure_stage(current_time)

        elif self.swarming_stage == SwarmingStage.ESTABLISHMENT:
            self._update_establishment_stage(current_time)

    def _initiate_swarming(self, current_time: int) -> None:
        """Initiate swarming process"""
        self.swarming_stage = SwarmingStage.PREPARATION

        # Create swarm cells
        num_cells = random.randint(8, 12)
        for i in range(num_cells):
            cell = QueenCell(
                cell_id=i,
                age=0,
                development_stage="egg",
                location=(random.uniform(-100, 100), random.uniform(-100, 100)),
                quality_score=random.uniform(0.8, 1.0),
                emergency_cell=False,
            )
            self.swarm_cells.append(cell)

    def _update_preparation_stage(self, current_time: int) -> None:
        """Update preparation stage"""
        # Advance to queen cell development after preparation
        if len(self.swarm_cells) > 0:
            self.swarming_stage = SwarmingStage.QUEEN_CELLS

    def _update_queen_cell_stage(self, current_time: int) -> None:
        """Update queen cell development stage"""
        # Update cell development
        for cell in self.swarm_cells:
            cell.age += 1

            if cell.age >= 16:
                cell.development_stage = "pupa"
            elif cell.age >= 6:
                cell.development_stage = "larva"

        # When cells are near emergence, start scout selection
        if any(cell.age >= 10 for cell in self.swarm_cells):
            self.swarming_stage = SwarmingStage.SCOUT_SELECTION

    def _update_scout_selection_stage(self, current_time: int) -> None:
        """Update scout selection stage"""
        # Select scout bees from experienced foragers
        if len(self.scout_bees) < 50:  # Target 50 scout bees
            foragers = [
                bee
                for bee in self.colony.get_bees()
                if isinstance(bee, Forager) and bee.age > 20
            ]

            available_scouts = [
                f for f in foragers if f.unique_id not in self.scout_bees
            ]

            if available_scouts:
                new_scouts = random.sample(
                    available_scouts, min(10, len(available_scouts))
                )
                self.scout_bees.update(scout.unique_id for scout in new_scouts)

        # Generate potential nest sites
        self._generate_nest_sites()

        # Transition to site selection
        if len(self.scout_bees) >= 20 and len(self.potential_sites) >= 3:
            self.swarming_stage = SwarmingStage.SITE_SELECTION

    def _update_site_selection_stage(self, current_time: int) -> None:
        """Update site selection stage"""
        # Scouts evaluate sites
        for site in self.potential_sites:
            # Add scouts to sites based on quality
            recruitment_prob = site.quality_score * 0.1
            new_scouts = sum(
                1
                for _ in range(len(self.scout_bees))
                if random.random() < recruitment_prob
            )
            site.scout_count += new_scouts

        # Check for consensus (site with >50% of scouts)
        total_scouts = len(self.scout_bees)
        for site in self.potential_sites:
            if site.scout_count > total_scouts * 0.5:
                self.selected_site = site
                self.swarming_stage = SwarmingStage.DEPARTURE
                break

    def _update_departure_stage(self, current_time: int) -> None:
        """Update departure stage"""
        # Check if first queen has emerged
        emerged_queens = [cell for cell in self.swarm_cells if cell.age >= 16]

        if emerged_queens and self.selected_site:
            # Execute swarm departure
            self._execute_swarm_departure(current_time)
            self.swarming_stage = SwarmingStage.ESTABLISHMENT

    def _update_establishment_stage(self, current_time: int) -> None:
        """Update establishment stage"""
        # New colony establishment is handled externally
        # Reset swarming state
        self.swarming_stage = SwarmingStage.NOT_SWARMING
        self.swarm_cells.clear()
        self.potential_sites.clear()
        self.scout_bees.clear()
        self.selected_site = None

    def _generate_nest_sites(self) -> None:
        """Generate potential nest sites"""
        if len(self.potential_sites) < 10:  # Maintain pool of sites
            for _ in range(3):
                site = NestSite(
                    site_id=self._get_next_site_id(),
                    location=(random.uniform(-2000, 2000), random.uniform(-2000, 2000)),
                    cavity_volume=random.uniform(20, 80),  # Liters
                    entrance_size=random.uniform(5, 25),  # cm²
                    protection_level=random.uniform(0.3, 1.0),
                    resource_proximity=random.uniform(100, 1000),  # meters
                    quality_score=random.uniform(0.4, 1.0),
                    discovered_time=0,
                )
                self.potential_sites.append(site)

    def _execute_swarm_departure(self, current_time: int) -> None:
        """Execute swarm departure"""
        # Calculate swarm size
        total_population = self.colony.get_adult_population()
        swarm_size = int(total_population * self.optimal_swarm_size)

        # Select bees for swarm (workers and foragers, exclude queen)
        workers = [
            bee
            for bee in self.colony.get_bees()
            if isinstance(bee, (Worker, Forager)) and not isinstance(bee, Queen)
        ]

        if len(workers) >= swarm_size:
            swarm_bees = random.sample(workers, swarm_size)

            # Remove swarm bees from colony
            for bee in swarm_bees:
                self.colony.remove_agent(bee)  # type: ignore[attr-defined]

            # Create new colony at selected site
            self._create_new_colony(swarm_bees, current_time)

    def _create_new_colony(self, swarm_bees: List[BeeAgent], current_time: int) -> None:
        """Create new colony from swarm"""
        if not self.selected_site:
            return

        # Create new colony
        new_colony = Colony(  # type: ignore[call-arg]
            unique_id=self.colony.model.next_id(),
            model=self.colony.model,
        )

        # Add swarm bees to new colony
        for bee in swarm_bees:
            new_colony.add_agent(bee)  # type: ignore[attr-defined]
            bee.model = new_colony.model
            bee.pos = self.selected_site.location

        # Create new queen for swarm (would emerge from swarm cells)
        new_queen = Queen(
            unique_id=self.colony.model.next_id(),
            model=new_colony.model,
            colony=new_colony,
        )
        new_queen.state = QueenState.VIRGIN
        new_colony.add_agent(new_queen)  # type: ignore[attr-defined]

        # Add new colony to model
        self.colony.model.schedule.add(new_colony)

    def _get_next_site_id(self) -> int:
        """Get next available site ID"""
        self.site_counter += 1
        return self.site_counter

    def get_swarming_status(self) -> Dict[str, Any]:
        """Get current swarming status"""
        return {
            "swarming_stage": self.swarming_stage.value,
            "swarm_cells": len(self.swarm_cells),
            "scout_bees": len(self.scout_bees),
            "potential_sites": len(self.potential_sites),
            "selected_site": self.selected_site.site_id if self.selected_site else None,
            "site_details": [
                {
                    "site_id": site.site_id,
                    "quality": site.quality_score,
                    "scout_count": site.scout_count,
                    "volume": site.cavity_volume,
                    "distance": site.resource_proximity,
                }
                for site in self.potential_sites
            ],
        }


class ReproductionManager:
    """
    Comprehensive reproduction management system.

    Integrates supersedure and swarming behaviors with colony lifecycle.
    """

    def __init__(self, colony: Colony):
        self.colony = colony
        self.supersedure = QueenSupersedure(colony)
        self.swarming = SwarmingBehavior(colony)

        # Add components to colony
        colony.supersedure = self.supersedure  # type: ignore[attr-defined]
        colony.swarming = self.swarming  # type: ignore[attr-defined]

    def update_reproduction_dynamics(self, current_time: int) -> None:
        """Update all reproduction dynamics"""

        # Update queen supersedure
        self.supersedure.check_supersedure_conditions(current_time)
        self.supersedure.update_queen_cells(current_time)

        # Update swarming behavior
        self.swarming.update_swarming_process(current_time)

        # Handle queen mating flights
        self._update_queen_mating(current_time)

        # Update drone production
        self._update_drone_production(current_time)

    def _update_queen_mating(self, current_time: int) -> None:
        """Update queen mating behavior"""
        queen = self.colony.get_queen()  # type: ignore[attr-defined]

        if queen and queen.state == QueenState.VIRGIN and queen.age >= 5:
            # Weather-dependent mating flight
            weather_suitable = random.random() < 0.8  # 80% suitable weather

            if weather_suitable:
                mating_success = random.random() < 0.9  # 90% mating success

                if mating_success:
                    queen.state = QueenState.MATED
                    # Start laying eggs after 2-3 days
                    if queen.age >= 8:
                        queen.state = QueenState.LAYING

    def _update_drone_production(self, current_time: int) -> None:
        """Update drone production for mating"""
        day_of_year = current_time % 365

        # Drone production peak in spring/early summer
        if 60 <= day_of_year <= 210:  # March-July
            drone_production_rate = 0.15  # 15% of new bees are drones
        else:
            drone_production_rate = 0.02  # Minimal drone production

        # Apply drone production rate to colony
        self.colony.drone_production_rate = drone_production_rate  # type: ignore[attr-defined]

    def get_reproduction_summary(self) -> Dict[str, Any]:
        """Get comprehensive reproduction status"""

        queen = self.colony.get_queen()  # type: ignore[attr-defined]
        queen_status = {
            "present": queen is not None,
            "age": queen.age if queen else 0,
            "state": queen.state.value if queen else "none",
            "quality": self.supersedure.assess_queen_quality(queen, 0) if queen else 0,
        }

        return {
            "queen_status": queen_status,
            "supersedure_status": self.supersedure.get_supersedure_status(),
            "swarming_status": self.swarming.get_swarming_status(),
            "drone_production_rate": getattr(
                self.colony, "drone_production_rate", 0.02
            ),
        }
