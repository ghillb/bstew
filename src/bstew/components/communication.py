"""
Bee communication systems for BSTEW
===================================

Implements waggle dance communication, pheromone trails,
and other forms of bee information sharing.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from collections import deque
import random

from ..core.agents import Forager, Worker, BeeAgent, BeeRole
from ..spatial.patches import ResourcePatch


class CommunicationType(Enum):
    """Types of bee communication"""

    WAGGLE_DANCE = "waggle_dance"
    TREMBLE_DANCE = "tremble_dance"
    STOP_SIGNAL = "stop_signal"
    PHEROMONE_TRAIL = "pheromone_trail"
    ALARM_PHEROMONE = "alarm_pheromone"
    QUEEN_PHEROMONE = "queen_pheromone"


class WaggleDanceInfo(BaseModel):
    """Information encoded in waggle dance"""

    model_config = {"validate_assignment": True}

    patch_id: int = Field(ge=0, description="Target patch identifier")
    distance: float = Field(ge=0.0, description="Distance to patch in meters")
    direction: float = Field(description="Direction in radians from sun")
    quality: float = Field(ge=0.0, le=1.0, description="Resource quality (0-1 scale)")
    resource_type: str = Field(description="Resource type: nectar, pollen, water")
    dance_duration: int = Field(ge=0, description="Dance duration in time steps")
    follower_count: int = Field(default=0, ge=0, description="Number of followers")
    created_time: int = Field(default=0, ge=0, description="Dance creation time")
    dancer_id: int = Field(default=0, ge=0, description="Dancer bee identifier")

    def get_decoded_location(
        self, hive_location: Tuple[float, float], sun_direction: float
    ) -> Tuple[float, float]:
        """Decode dance to get patch location"""
        # Convert dance direction to absolute direction
        absolute_direction = self.direction + sun_direction

        # Calculate patch location
        patch_x = hive_location[0] + self.distance * math.cos(absolute_direction)
        patch_y = hive_location[1] + self.distance * math.sin(absolute_direction)

        return (patch_x, patch_y)

    def get_dance_vigor(self) -> float:
        """Calculate dance vigor based on quality and distance"""
        # Higher quality and closer patches get more vigorous dances
        distance_factor = max(0.1, 1.0 - self.distance / 2000.0)  # Decay over 2km
        return self.quality * distance_factor


class PheromoneTrail(BaseModel):
    """Pheromone trail information"""

    model_config = {"validate_assignment": True}

    trail_id: int = Field(ge=0, description="Unique trail identifier")
    start_location: Tuple[float, float] = Field(description="Trail start coordinates")
    end_location: Tuple[float, float] = Field(description="Trail end coordinates")
    pheromone_type: str = Field(description="Type of pheromone")
    strength: float = Field(ge=0.0, le=1.0, description="Pheromone strength")
    decay_rate: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Daily decay rate"
    )
    created_time: int = Field(default=0, ge=0, description="Trail creation time")
    last_reinforced: int = Field(default=0, ge=0, description="Last reinforcement time")

    def get_strength_at_time(self, current_time: int) -> float:
        """Get current pheromone strength"""
        time_elapsed = current_time - self.last_reinforced
        return self.strength * (self.decay_rate**time_elapsed)

    def reinforce(self, additional_strength: float, current_time: int) -> None:
        """Reinforce pheromone trail"""
        self.strength += additional_strength
        self.last_reinforced = current_time
        self.strength = min(self.strength, 1.0)  # Cap at maximum


class WaggleDanceDecoder:
    """
    Decodes and interprets waggle dance information.

    Implements:
    - Distance encoding/decoding
    - Direction calculation
    - Quality assessment
    - Error modeling in communication
    """

    def __init__(self) -> None:
        # Dance parameters (based on real bee research)
        self.duration_to_distance_factor = 0.75  # seconds per 100m
        self.waggle_run_distance_factor = 1.0  # cm per 100m
        self.direction_error_std = 0.1  # radians
        self.distance_error_factor = 0.1  # 10% error

    def encode_dance(
        self,
        patch: ResourcePatch,
        hive_location: Tuple[float, float],
        sun_direction: float,
        quality: float,
        current_time: int,
        dancer_id: int,
    ) -> WaggleDanceInfo:
        """Encode patch information into dance"""

        # Calculate distance and direction
        dx = patch.x - hive_location[0]
        dy = patch.y - hive_location[1]
        distance = math.sqrt(dx**2 + dy**2)

        # Direction relative to sun
        absolute_direction = math.atan2(dy, dx)
        relative_direction = absolute_direction - sun_direction

        # Normalize direction to [0, 2π]
        while relative_direction < 0:
            relative_direction += 2 * math.pi
        while relative_direction >= 2 * math.pi:
            relative_direction -= 2 * math.pi

        # Calculate dance duration based on quality and distance
        base_duration = max(5, int(distance / 100))  # Minimum 5 steps
        quality_factor = 0.5 + quality  # Range 0.5-1.5
        dance_duration = int(base_duration * quality_factor)

        return WaggleDanceInfo(
            patch_id=patch.id,
            distance=distance,
            direction=relative_direction,
            quality=quality,
            resource_type=patch.primary_resource_type,
            dance_duration=dance_duration,
            created_time=current_time,
            dancer_id=dancer_id,
        )

    def decode_dance(
        self,
        dance: WaggleDanceInfo,
        hive_location: Tuple[float, float],
        sun_direction: float,
        add_noise: bool = True,
    ) -> Tuple[float, float, float]:
        """Decode dance to get patch information with realistic errors"""

        # Add communication errors
        if add_noise:
            direction_error = random.gauss(0, self.direction_error_std)
            distance_error = random.gauss(1.0, self.distance_error_factor)

            decoded_direction = dance.direction + direction_error
            decoded_distance = dance.distance * distance_error
        else:
            decoded_direction = dance.direction
            decoded_distance = dance.distance

        # Convert to absolute coordinates
        absolute_direction = decoded_direction + sun_direction
        patch_x = hive_location[0] + decoded_distance * math.cos(absolute_direction)
        patch_y = hive_location[1] + decoded_distance * math.sin(absolute_direction)

        return (patch_x, patch_y, dance.quality)

    def calculate_dance_attractiveness(
        self, dance: WaggleDanceInfo, current_time: int
    ) -> float:
        """Calculate how attractive a dance is to followers"""

        # Dance vigor (based on quality and distance)
        vigor = dance.get_dance_vigor()

        # Recency factor (newer dances more attractive)
        age = current_time - dance.created_time
        recency_factor = math.exp(-age / 10.0)  # Decay over 10 time steps

        # Social proof (more followers = more attractive)
        social_factor = 1.0 + math.log(1 + dance.follower_count) * 0.1

        return vigor * recency_factor * social_factor


class CommunicationNetwork:
    """
    Manages colony-wide communication network.

    Handles:
    - Dance floor management
    - Information propagation
    - Communication efficiency
    - Social learning dynamics
    """

    def __init__(self, colony: Any) -> None:
        self.colony = colony
        self.active_dances: List[WaggleDanceInfo] = []
        self.pheromone_trails: List[PheromoneTrail] = []
        self.dance_decoder = WaggleDanceDecoder()

        # Communication parameters
        self.max_active_dances = 10
        self.dance_attendance_rate = 0.3  # Probability of attending dance area
        self.following_probability = 0.4  # Probability of following attractive dance
        self.pheromone_detection_range = 50.0  # meters

        # Statistics
        self.total_dances_performed = 0
        self.total_followers_recruited = 0
        self.communication_efficiency_history: deque[float] = deque(maxlen=100)

    def add_waggle_dance(
        self, dancer: Any, patch: ResourcePatch, quality: float, current_time: int
    ) -> None:
        """Add new waggle dance to communication network"""

        if len(self.active_dances) >= self.max_active_dances:
            # Remove oldest dance
            self.active_dances.pop(0)

        # Create dance information
        sun_direction = self._get_sun_direction(current_time)
        dance = self.dance_decoder.encode_dance(
            patch,
            self.colony.location,
            sun_direction,
            quality,
            current_time,
            dancer.unique_id,
        )

        self.active_dances.append(dance)
        self.total_dances_performed += 1

        # Recruit followers
        self._recruit_dance_followers(dance, current_time)

    def _recruit_dance_followers(
        self, dance: WaggleDanceInfo, current_time: int
    ) -> None:
        """Recruit bees to follow dance"""

        # Get potential followers (unemployed foragers and house bees)
        potential_followers = []

        for bee in self.colony.get_bees():
            if (
                isinstance(bee, Forager)
                and bee.target_patch is None
                or isinstance(bee, Worker)
                and bee.role == BeeRole.NURSE
            ):
                # Check if bee attends dance area
                if random.random() < self.dance_attendance_rate:
                    potential_followers.append(bee)

        # Determine dance attractiveness
        attractiveness = self.dance_decoder.calculate_dance_attractiveness(
            dance, current_time
        )

        # Recruit followers based on attractiveness
        followers_recruited = 0
        for bee in potential_followers:
            if random.random() < attractiveness * self.following_probability:
                self._assign_follower_to_dance(bee, dance)
                followers_recruited += 1

        dance.follower_count = followers_recruited
        self.total_followers_recruited += followers_recruited

    def _assign_follower_to_dance(self, follower: Any, dance: WaggleDanceInfo) -> None:
        """Assign bee to follow dance information"""

        if isinstance(follower, Forager):
            # Decode dance to get patch location
            sun_direction = self._get_sun_direction(dance.created_time)
            patch_x, patch_y, quality = self.dance_decoder.decode_dance(
                dance, self.colony.location, sun_direction, add_noise=True
            )
            patch_location = (patch_x, patch_y)

            # Find nearest patch to decoded location
            nearest_patch = self._find_nearest_patch(patch_location)
            if nearest_patch:
                follower.target_patch = nearest_patch.id
                follower.following_dance = dance

        elif isinstance(follower, Worker):
            # Convert worker to forager if appropriate
            if follower.age > follower.role_transition_age:
                follower.transition_to_forager()
                self._assign_follower_to_dance(follower, dance)

    def _find_nearest_patch(
        self, location: Tuple[float, float]
    ) -> Optional[ResourcePatch]:
        """Find nearest resource patch to given location"""
        min_distance = float("inf")
        nearest_patch = None

        for patch in self.colony.model.landscape.patches.values():
            distance = math.sqrt(
                (patch.x - location[0]) ** 2 + (patch.y - location[1]) ** 2
            )
            if distance < min_distance and patch.has_resources():
                min_distance = distance
                nearest_patch = patch

        return nearest_patch

    def _get_sun_direction(self, current_time: int) -> float:
        """Calculate sun direction based on time of day"""
        # Simplified sun movement (12-hour cycle)
        hour_of_day = (current_time % 12) / 12.0
        sun_angle = hour_of_day * 2 * math.pi  # Full rotation in 12 hours
        return sun_angle

    def update_communication_network(self, current_time: int) -> None:
        """Update communication network state"""

        # Remove expired dances
        self.active_dances = [
            dance
            for dance in self.active_dances
            if current_time - dance.created_time < dance.dance_duration
        ]

        # Update pheromone trails
        self._update_pheromone_trails(current_time)

        # Calculate communication efficiency
        self._calculate_communication_efficiency()

    def _update_pheromone_trails(self, current_time: int) -> None:
        """Update pheromone trail strengths"""

        # Decay existing trails
        for trail in self.pheromone_trails[:]:
            if trail.get_strength_at_time(current_time) < 0.01:
                self.pheromone_trails.remove(trail)

    def _calculate_communication_efficiency(self) -> None:
        """Calculate current communication efficiency"""

        if self.total_dances_performed == 0:
            efficiency = 0.0
        else:
            # Efficiency based on follower recruitment rate
            efficiency = self.total_followers_recruited / self.total_dances_performed

        self.communication_efficiency_history.append(efficiency)

    def get_active_dances_for_bee(self, bee: BeeAgent) -> List[WaggleDanceInfo]:
        """Get relevant active dances for a specific bee"""

        # Filter dances based on bee characteristics and needs
        relevant_dances = []

        for dance in self.active_dances:
            # Don't follow own dance
            if isinstance(bee, Forager) and dance.dancer_id == bee.unique_id:
                continue

            # Check if dance is still attractive
            attractiveness = self.dance_decoder.calculate_dance_attractiveness(
                dance, self.colony.model.schedule.steps
            )

            if attractiveness > 0.3:  # Minimum attractiveness threshold
                relevant_dances.append(dance)

        return relevant_dances

    def add_pheromone_trail(
        self,
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        pheromone_type: str,
        strength: float,
        current_time: int,
    ) -> None:
        """Add pheromone trail to network"""

        trail = PheromoneTrail(
            trail_id=len(self.pheromone_trails),
            start_location=start_location,
            end_location=end_location,
            pheromone_type=pheromone_type,
            strength=strength,
            created_time=current_time,
            last_reinforced=current_time,
        )

        self.pheromone_trails.append(trail)

    def get_pheromone_gradient(
        self, location: Tuple[float, float], pheromone_type: str, current_time: int
    ) -> Tuple[float, float]:
        """Get pheromone gradient at location for guidance"""

        gradient_x = 0.0
        gradient_y = 0.0

        for trail in self.pheromone_trails:
            if trail.pheromone_type != pheromone_type:
                continue

            # Calculate distance to trail
            dist_to_start = math.sqrt(
                (location[0] - trail.start_location[0]) ** 2
                + (location[1] - trail.start_location[1]) ** 2
            )
            dist_to_end = math.sqrt(
                (location[0] - trail.end_location[0]) ** 2
                + (location[1] - trail.end_location[1]) ** 2
            )

            if min(dist_to_start, dist_to_end) < self.pheromone_detection_range:
                # Calculate gradient direction
                strength = trail.get_strength_at_time(current_time)

                # Direction towards end of trail
                dx = trail.end_location[0] - location[0]
                dy = trail.end_location[1] - location[1]
                distance = math.sqrt(dx**2 + dy**2)

                if distance > 0:
                    gradient_x += (dx / distance) * strength
                    gradient_y += (dy / distance) * strength

        return (gradient_x, gradient_y)

    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication network statistics"""

        avg_efficiency = (
            sum(self.communication_efficiency_history)
            / len(self.communication_efficiency_history)
            if self.communication_efficiency_history
            else 0.0
        )

        return {
            "active_dances": len(self.active_dances),
            "total_dances_performed": self.total_dances_performed,
            "total_followers_recruited": self.total_followers_recruited,
            "current_efficiency": (
                self.total_followers_recruited / max(1, self.total_dances_performed)
            ),
            "average_efficiency": avg_efficiency,
            "active_pheromone_trails": len(self.pheromone_trails),
            "dance_info": [
                {
                    "patch_id": dance.patch_id,
                    "quality": dance.quality,
                    "distance": dance.distance,
                    "followers": dance.follower_count,
                    "age": self.colony.model.schedule.steps - dance.created_time,
                }
                for dance in self.active_dances
            ],
        }


class SocialLearning:
    """
    Implements social learning mechanisms in bee colonies.

    Handles:
    - Information quality assessment
    - Learning from unsuccessful foraging
    - Collective decision making
    - Cultural transmission of knowledge
    """

    def __init__(self, colony: Any) -> None:
        self.colony = colony
        self.collective_knowledge: Dict[int, Dict[str, Any]] = {}
        self.learning_rate = 0.1
        self.forgetting_rate = 0.95

    def update_collective_knowledge(
        self, patch_id: int, experience_data: Dict[str, Any], current_time: int
    ) -> None:
        """Update collective knowledge about patch"""

        if patch_id not in self.collective_knowledge:
            self.collective_knowledge[patch_id] = {
                "quality_estimates": [],
                "reliability_score": 0.5,
                "last_updated": current_time,
                "total_visits": 0,
                "successful_visits": 0,
            }

        knowledge = self.collective_knowledge[patch_id]

        # Update with new experience
        knowledge["quality_estimates"].append(experience_data["quality"])
        knowledge["total_visits"] += 1
        knowledge["last_updated"] = current_time

        if experience_data["successful"]:
            knowledge["successful_visits"] += 1

        # Update reliability
        if knowledge["total_visits"] > 0:
            success_rate = knowledge["successful_visits"] / knowledge["total_visits"]
            knowledge["reliability_score"] = success_rate

        # Limit memory size
        if len(knowledge["quality_estimates"]) > 20:
            knowledge["quality_estimates"] = knowledge["quality_estimates"][-20:]

    def get_collective_patch_assessment(
        self, patch_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get collective assessment of patch quality"""

        if patch_id not in self.collective_knowledge:
            return None

        knowledge = self.collective_knowledge[patch_id]

        if not knowledge["quality_estimates"]:
            return None

        return {
            "average_quality": np.mean(knowledge["quality_estimates"]),
            "quality_variance": np.var(knowledge["quality_estimates"]),
            "reliability_score": knowledge["reliability_score"],
            "sample_size": len(knowledge["quality_estimates"]),
            "last_updated": knowledge["last_updated"],
        }

    def decay_knowledge(self, current_time: int) -> None:
        """Decay old knowledge to simulate forgetting"""

        for patch_id, knowledge in list(self.collective_knowledge.items()):
            days_since_update = current_time - knowledge["last_updated"]

            if days_since_update > 0:
                decay_factor = self.forgetting_rate**days_since_update

                # Decay quality estimates
                knowledge["quality_estimates"] = [
                    q * decay_factor for q in knowledge["quality_estimates"]
                ]

                # Remove very old or unreliable knowledge
                if days_since_update > 30 or knowledge["reliability_score"] < 0.1:
                    del self.collective_knowledge[patch_id]
