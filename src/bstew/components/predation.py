"""
Predation system for BSTEW
=========================

Implements badger predation with colony destruction mechanics,
territorial behavior, and realistic encounter probabilities.
"""

import random
from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
import logging
import math

from .development import DeathCause


class PredatorType(Enum):
    """Types of predators"""

    BADGER = "badger"
    BIRD = "bird"
    WASP = "wasp"
    MOUSE = "mouse"


class PredatorParameters(BaseModel):
    """Parameters for predator behavior"""

    model_config = {"validate_assignment": True}

    # Badger-specific parameters (from NetLogo)
    foraging_range_m: float = Field(
        default=735.0, ge=0.0, description="Badger foraging range in meters"
    )
    encounter_probability: float = Field(
        default=0.19, ge=0.0, le=1.0, description="Probability of finding colony"
    )
    dig_up_probability: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Probability of successful attack"
    )
    territory_size_m: float = Field(
        default=1500.0, ge=0.0, description="Territory diameter"
    )

    # Movement parameters
    movement_speed_m_per_hour: float = Field(
        default=500.0, ge=0.0, description="Movement speed"
    )
    foraging_hours_per_day: float = Field(
        default=8.0, ge=0.0, le=24.0, description="Active foraging hours"
    )

    # Seasonal activity
    active_months: List[int] = Field(
        default_factory=lambda: [4, 5, 6, 7, 8, 9, 10],
        description="Active months (Apr-Oct)",
    )

    # Colony destruction parameters
    destruction_completeness: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Fraction of colony destroyed"
    )
    resource_consumption: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Fraction of resources consumed"
    )


class PredatorAgent(BaseModel):
    """Individual predator agent"""

    model_config = {"validate_assignment": True}

    unique_id: int = Field(ge=0, description="Unique predator identifier")
    predator_type: PredatorType = Field(description="Type of predator")
    location: Tuple[float, float] = Field(description="Current location coordinates")
    territory_center: Tuple[float, float] = Field(
        description="Territory center coordinates"
    )
    territory_radius: float = Field(ge=0.0, description="Territory radius")

    # Behavior state
    energy: float = Field(default=100.0, ge=0.0, description="Current energy level")
    foraging_success: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Foraging success rate"
    )
    last_colony_attack: int = Field(default=-1, description="Last colony attack time")
    colonies_destroyed: int = Field(
        default=0, ge=0, description="Number of colonies destroyed"
    )

    # Movement tracking
    current_target: Optional[Tuple[float, float]] = Field(
        default=None, description="Current movement target"
    )
    path_history: List[Tuple[float, float]] = Field(
        default_factory=list, description="Movement path history"
    )

    def is_in_territory(self, location: Tuple[float, float]) -> bool:
        """Check if location is within predator's territory"""
        distance = math.sqrt(
            (location[0] - self.territory_center[0]) ** 2
            + (location[1] - self.territory_center[1]) ** 2
        )
        return distance <= self.territory_radius

    def calculate_distance_to(self, location: Tuple[float, float]) -> float:
        """Calculate distance to target location"""
        return math.sqrt(
            (location[0] - self.location[0]) ** 2
            + (location[1] - self.location[1]) ** 2
        )


class PredationSystem:
    """
    Manages predator populations and colony attacks.

    Implements:
    - Badger territorial behavior and foraging
    - Colony encounter and destruction mechanics
    - Seasonal predator activity patterns
    - Multiple predator types with different behaviors
    """

    def __init__(self, landscape_bounds: Tuple[float, float, float, float]):
        """
        Initialize predation system.

        Args:
            landscape_bounds: (min_x, min_y, max_x, max_y) of landscape
        """
        self.landscape_bounds = landscape_bounds
        self.predators: Dict[int, PredatorAgent] = {}
        self.next_predator_id = 1

        # Predator parameters
        self.badger_params = PredatorParameters()

        # Attack statistics
        self.attack_history: List[Dict[str, Any]] = []
        self.colonies_attacked: Set[int] = set()
        self.seasonal_activity: Dict[str, float] = {}

        # Initialize badger population
        self.initialize_badger_population()

        self.logger = logging.getLogger(__name__)

    def initialize_badger_population(self) -> None:
        """Initialize badger population based on landscape size"""
        # Calculate landscape area
        width = self.landscape_bounds[2] - self.landscape_bounds[0]
        height = self.landscape_bounds[3] - self.landscape_bounds[1]
        area_km2 = (width * height) / 1000000  # Convert to km²

        # Badger density: approximately 1 badger per 0.5-2 km²
        badger_density_per_km2 = 1.0
        num_badgers = max(1, int(area_km2 * badger_density_per_km2))

        for _ in range(num_badgers):
            self.create_badger()

    def create_badger(self) -> int:
        """Create a new badger agent"""
        badger_id = self.next_predator_id
        self.next_predator_id += 1

        # Random location within landscape
        x = random.uniform(self.landscape_bounds[0], self.landscape_bounds[2])
        y = random.uniform(self.landscape_bounds[1], self.landscape_bounds[3])
        location = (x, y)

        # Territory around location
        territory_radius = self.badger_params.territory_size_m / 2

        badger = PredatorAgent(
            unique_id=badger_id,
            predator_type=PredatorType.BADGER,
            location=location,
            territory_center=location,
            territory_radius=territory_radius,
        )

        self.predators[badger_id] = badger
        return badger_id

    def step(self, day_of_year: int, colonies: List[Any]) -> Dict[str, Any]:
        """Execute one step of predation system"""

        results: Dict[str, Any] = {
            "attacks": [],
            "colony_destructions": [],
            "predator_movements": [],
            "seasonal_activity": self.get_seasonal_activity(day_of_year),
        }

        # Check if predators are active this time of year
        month = self.day_to_month(day_of_year)
        if month not in self.badger_params.active_months:
            return results

        # Process each predator
        for predator in self.predators.values():
            if predator.predator_type == PredatorType.BADGER:
                attack_result = self.process_badger_foraging(
                    predator, colonies, day_of_year
                )
                if attack_result:
                    results["attacks"].append(attack_result)

                    # Check if attack was successful
                    if attack_result["successful"]:
                        results["colony_destructions"].append(attack_result)

        return results

    def process_badger_foraging(
        self, badger: PredatorAgent, colonies: List[Any], day: int
    ) -> Optional[Dict[str, Any]]:
        """Process badger foraging behavior"""

        # Update badger location (simplified random walk within territory)
        self.move_badger(badger)

        # Find colonies within foraging range
        nearby_colonies = self.find_colonies_in_range(badger, colonies)

        if not nearby_colonies:
            return None

        # Check for colony encounters
        for colony in nearby_colonies:
            encounter_result = self.check_colony_encounter(badger, colony, day)
            if encounter_result:
                return encounter_result

        return None

    def move_badger(self, badger: PredatorAgent) -> None:
        """Move badger within its territory"""
        # Calculate movement distance per step
        daily_movement = (
            self.badger_params.movement_speed_m_per_hour
            * self.badger_params.foraging_hours_per_day
        )

        # Random walk within territory
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(
            0, daily_movement * 0.1
        )  # 10% of daily movement per step

        new_x = badger.location[0] + distance * math.cos(angle)
        new_y = badger.location[1] + distance * math.sin(angle)
        new_location = (new_x, new_y)

        # Keep within territory bounds
        if badger.is_in_territory(new_location):
            badger.location = new_location
            badger.path_history.append(new_location)

            # Limit history length
            if len(badger.path_history) > 100:
                badger.path_history.pop(0)
        else:
            # Move back towards territory center
            center_x, center_y = badger.territory_center
            current_x, current_y = badger.location

            # Calculate direction to center
            dx = center_x - current_x
            dy = center_y - current_y
            distance_to_center = math.sqrt(dx**2 + dy**2)

            if distance_to_center > 0:
                # Move towards center
                move_distance = min(distance, distance_to_center)
                new_x = current_x + (dx / distance_to_center) * move_distance
                new_y = current_y + (dy / distance_to_center) * move_distance
                badger.location = (new_x, new_y)

    def find_colonies_in_range(
        self, predator: PredatorAgent, colonies: List[Any]
    ) -> List[Any]:
        """Find colonies within predator's foraging range"""
        nearby_colonies = []

        for colony in colonies:
            distance = predator.calculate_distance_to(colony.location)
            if distance <= self.badger_params.foraging_range_m:
                nearby_colonies.append(colony)

        return nearby_colonies

    def check_colony_encounter(
        self, badger: PredatorAgent, colony: Any, day: int
    ) -> Optional[Dict[str, Any]]:
        """Check if badger encounters and attacks colony"""

        # Check encounter probability
        if random.random() > self.badger_params.encounter_probability:
            return None

        # Record encounter
        encounter_result = {
            "predator_id": badger.unique_id,
            "colony_id": (
                colony.model.unique_id if hasattr(colony, "model") else id(colony)
            ),
            "day": day,
            "encounter_distance": badger.calculate_distance_to(colony.location),
            "successful": False,
            "destruction_details": None,
        }

        # Check if attack is successful
        if random.random() <= self.badger_params.dig_up_probability:
            # Successful attack - destroy colony
            destruction_result = self.destroy_colony(badger, colony, day)
            encounter_result["successful"] = True
            encounter_result["destruction_details"] = destruction_result

        return encounter_result

    def destroy_colony(
        self, predator: PredatorAgent, colony: Any, day: int
    ) -> Dict[str, Any]:
        """Destroy colony and record details"""

        # Record pre-destruction state
        pre_destruction_state = {
            "adult_population": colony.get_adult_population(),
            "brood_count": colony.get_brood_count(),
            "total_population": colony.get_total_population(),
            "resources": {
                "pollen": colony.resources.pollen,
                "nectar": colony.resources.nectar,
                "honey": colony.resources.honey,
            },
        }

        # Destroy brood in development system
        if hasattr(colony, "development_system"):
            self.destroy_developing_brood(colony.development_system)

        # Kill all adult bees
        bees_killed = []
        for bee in colony.bees[:]:  # Copy list to avoid modification during iteration
            bee_info = {
                "id": bee.unique_id,
                "role": bee.role.value,
                "age": bee.age,
                "genotype": bee.genotype.get_allele_ids() if bee.genotype else None,
            }
            bees_killed.append(bee_info)
            bee.status = bee.status.DEAD if hasattr(bee.status, "DEAD") else "dead"
            bee.die()

        # Consume resources
        resources_consumed = {
            "pollen": colony.resources.pollen * self.badger_params.resource_consumption,
            "nectar": colony.resources.nectar * self.badger_params.resource_consumption,
            "honey": colony.resources.honey * self.badger_params.resource_consumption,
        }

        colony.resources.pollen -= resources_consumed["pollen"]
        colony.resources.nectar -= resources_consumed["nectar"]
        colony.resources.honey -= resources_consumed["honey"]

        # Set colony as collapsed
        colony.health = (
            colony.health.COLLAPSED
            if hasattr(colony.health, "COLLAPSED")
            else "collapsed"
        )

        # Update predator stats
        predator.colonies_destroyed += 1
        predator.last_colony_attack = day
        # Update foraging success as a rate (successful attacks / total attempts)
        # Increment success rate with exponential moving average to keep it in [0,1] range
        predator.foraging_success = min(1.0, predator.foraging_success + 0.1)

        # Record in attack history
        attack_record = {
            "predator_id": predator.unique_id,
            "colony_id": id(colony),
            "day": day,
            "pre_destruction_state": pre_destruction_state,
            "bees_killed": bees_killed,
            "resources_consumed": resources_consumed,
        }

        self.attack_history.append(attack_record)
        self.colonies_attacked.add(id(colony))

        self.logger.warning(
            f"Colony {id(colony)} destroyed by badger {predator.unique_id} on day {day}"
        )

        return attack_record

    def destroy_developing_brood(self, development_system: Any) -> List[Dict[str, Any]]:
        """Destroy all developing brood in development system"""
        brood_killed = []

        for bee_id, developing_bee in list(development_system.developing_bees.items()):
            if developing_bee.is_alive():
                brood_info = {
                    "id": bee_id,
                    "stage": developing_bee.stage.value,
                    "age": developing_bee.age_days,
                    "genotype": (
                        developing_bee.genotype.get_allele_ids()
                        if developing_bee.genotype
                        else None
                    ),
                }
                brood_killed.append(brood_info)

                # Kill the developing bee
                developing_bee.development_success = False
                developing_bee.death_cause = DeathCause.PREDATION
                developing_bee.death_day = developing_bee.age_days

        return brood_killed

    def get_seasonal_activity(self, day_of_year: int) -> Dict[str, Any]:
        """Get seasonal activity level"""
        month = self.day_to_month(day_of_year)

        if month in self.badger_params.active_months:
            # Peak activity in late spring/early summer
            if month in [5, 6, 7]:  # May, June, July
                activity_level = 1.0
            else:
                activity_level = 0.7
        else:
            activity_level = 0.0

        return {
            "month": month,
            "activity_level": activity_level,
            "active_predators": len(
                [
                    p
                    for p in self.predators.values()
                    if p.predator_type == PredatorType.BADGER
                ]
            ),
        }

    def day_to_month(self, day_of_year: int) -> int:
        """Convert day of year to month"""
        # Simplified month calculation
        cumulative_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

        for month in range(1, 13):
            if day_of_year <= cumulative_days[month]:
                return month

        return 12

    def get_predation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive predation statistics"""

        total_attacks = len(self.attack_history)
        successful_attacks = len([a for a in self.attack_history if a])

        # Calculate mortality by predator type
        mortality_by_predator = {}
        for predator in self.predators.values():
            predator_type = predator.predator_type.value
            if predator_type not in mortality_by_predator:
                mortality_by_predator[predator_type] = {
                    "colonies_destroyed": 0,
                    "total_bees_killed": 0,
                    "resources_consumed": {"pollen": 0, "nectar": 0, "honey": 0},
                }

            current_count = mortality_by_predator[predator_type]["colonies_destroyed"]
            mortality_by_predator[predator_type]["colonies_destroyed"] = (
                int(current_count) if isinstance(current_count, (int, float)) else 0
            ) + predator.colonies_destroyed

        # Calculate total mortality from attack history
        total_bees_killed = 0
        total_resources_consumed = {"pollen": 0, "nectar": 0, "honey": 0}

        for attack in self.attack_history:
            if attack.get("bees_killed"):
                total_bees_killed += len(attack["bees_killed"])
            if attack.get("resources_consumed"):
                for resource, amount in attack["resources_consumed"].items():
                    total_resources_consumed[resource] += amount

        return {
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "success_rate": (
                successful_attacks / total_attacks if total_attacks > 0 else 0.0
            ),
            "colonies_destroyed": len(self.colonies_attacked),
            "total_bees_killed": total_bees_killed,
            "total_resources_consumed": total_resources_consumed,
            "mortality_by_predator": mortality_by_predator,
            "active_predators": len(self.predators),
            "attack_frequency": total_attacks / 365.0 if total_attacks > 0 else 0.0,
        }

    def get_predator_locations(self) -> List[Dict[str, Any]]:
        """Get current predator locations for visualization"""
        locations = []

        for predator in self.predators.values():
            locations.append(
                {
                    "id": predator.unique_id,
                    "type": predator.predator_type.value,
                    "location": predator.location,
                    "territory_center": predator.territory_center,
                    "territory_radius": predator.territory_radius,
                    "colonies_destroyed": predator.colonies_destroyed,
                    "foraging_success": predator.foraging_success,
                }
            )

        return locations

    def add_predator_pressure(self, intensity: float) -> None:
        """Increase predator pressure by adding more predators"""
        current_count = len(self.predators)
        additional_predators = int(current_count * intensity)

        for _ in range(additional_predators):
            self.create_badger()

        self.logger.info(
            f"Added {additional_predators} predators due to increased pressure"
        )

    def remove_predator_pressure(self, intensity: float) -> None:
        """Decrease predator pressure by removing predators"""
        current_count = len(self.predators)
        predators_to_remove = int(current_count * intensity)

        predator_ids = list(self.predators.keys())
        for _ in range(min(predators_to_remove, len(predator_ids))):
            predator_id = random.choice(predator_ids)
            del self.predators[predator_id]
            predator_ids.remove(predator_id)

        self.logger.info(
            f"Removed {predators_to_remove} predators due to decreased pressure"
        )

    def get_colony_risk_assessment(
        self, colony_location: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Assess predation risk for a colony at given location"""

        nearby_predators = []
        total_risk = 0.0

        for predator in self.predators.values():
            distance = predator.calculate_distance_to(colony_location)

            if distance <= self.badger_params.foraging_range_m:
                risk_factor = 1.0 - (distance / self.badger_params.foraging_range_m)
                risk_factor *= (
                    self.badger_params.encounter_probability
                    * self.badger_params.dig_up_probability
                )

                nearby_predators.append(
                    {
                        "predator_id": predator.unique_id,
                        "type": predator.predator_type.value,
                        "distance": distance,
                        "risk_factor": risk_factor,
                    }
                )

                total_risk += risk_factor

        return {
            "total_risk": min(1.0, total_risk),  # Cap at 100%
            "nearby_predators": nearby_predators,
            "high_risk": total_risk > 0.1,  # Above 10% risk
            "recommendations": self.get_risk_recommendations(total_risk),
        }

    def get_risk_recommendations(self, risk_level: float) -> List[str]:
        """Get recommendations based on risk level"""
        recommendations = []

        if risk_level > 0.3:
            recommendations.append("Very high risk - consider alternative location")
            recommendations.append("Implement maximum colony defenses")
        elif risk_level > 0.1:
            recommendations.append("High risk - monitor closely")
            recommendations.append("Consider colony size management")
        elif risk_level > 0.05:
            recommendations.append("Moderate risk - standard precautions")
        else:
            recommendations.append("Low risk - minimal precautions needed")

        return recommendations
