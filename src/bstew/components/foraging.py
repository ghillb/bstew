"""
Advanced foraging behaviors for BSTEW
=====================================

Implements complex foraging decision-making, route optimization,
and resource assessment behaviors for bee agents.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict
import random

from ..spatial.patches import ResourcePatch
from ..core.agents import Forager


class ForagingState(Enum):
    """Foraging behavior states"""

    SEARCHING = "searching"
    TRAVELING = "traveling"
    COLLECTING = "collecting"
    RETURNING = "returning"
    DANCING = "dancing"
    FOLLOWING_DANCE = "following_dance"
    RESTING = "resting"


class ForagingTrip(BaseModel):
    """Record of a foraging trip"""

    model_config = {"validate_assignment": True}

    trip_id: int = Field(ge=0, description="Unique trip identifier")
    start_time: int = Field(ge=0, description="Trip start time")
    end_time: Optional[int] = Field(default=None, description="Trip end time")
    patch_visited: Optional[int] = Field(default=None, description="Patch ID visited")
    distance_traveled: float = Field(
        default=0.0, ge=0.0, description="Total distance traveled"
    )
    energy_spent: float = Field(default=0.0, ge=0.0, description="Energy expenditure")
    resources_collected: float = Field(
        default=0.0, ge=0.0, description="Resources collected"
    )
    resource_type: str = Field(
        default="mixed", description="Type of resources collected"
    )
    successful: bool = Field(default=False, description="Whether trip was successful")


class PatchMemory(BaseModel):
    """Memory of a resource patch"""

    model_config = {"validate_assignment": True}

    patch_id: int = Field(ge=0, description="Patch identifier")
    location: Tuple[float, float] = Field(description="Patch location coordinates")
    last_visit: int = Field(ge=0, description="Last visit time")
    visit_count: int = Field(default=0, ge=0, description="Number of visits")
    average_quality: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average patch quality"
    )
    average_distance: float = Field(
        default=0.0, ge=0.0, description="Average distance to patch"
    )
    last_resources: float = Field(
        default=0.0, ge=0.0, description="Resources from last visit"
    )
    reliability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Patch reliability score"
    )
    seasonal_pattern: List[float] = Field(
        default_factory=list, description="Seasonal resource pattern"
    )

    def update_visit(
        self, quality: float, distance: float, resources: float, current_time: int
    ) -> None:
        """Update memory based on visit"""
        self.visit_count += 1
        self.last_visit = current_time

        # Update averages with exponential smoothing
        alpha = 0.3  # Learning rate
        self.average_quality = alpha * quality + (1 - alpha) * self.average_quality
        self.average_distance = alpha * distance + (1 - alpha) * self.average_distance
        self.last_resources = resources

        # Update reliability based on resource availability
        if resources > 0:
            self.reliability = min(1.0, self.reliability + 0.1)
        else:
            self.reliability = max(0.0, self.reliability - 0.2)


class ForagingDecisionMaker:
    """
    Advanced foraging decision-making system.

    Implements:
    - Patch selection optimization
    - Risk assessment
    - Energy budget management
    - Memory-based decisions
    - Social information integration
    """

    def __init__(self, forager: Forager):
        self.forager = forager
        self.patch_memories: Dict[int, PatchMemory] = {}
        self.foraging_state = ForagingState.SEARCHING
        self.current_trip: Optional[ForagingTrip] = None
        self.trip_history: List[ForagingTrip] = []
        self.trip_counter = 0

        # Decision parameters
        self.exploration_rate = 0.2  # Probability of exploring new patches
        self.memory_decay_rate = 0.95  # Daily decay of patch memory
        self.risk_tolerance = 0.7  # Higher = more willing to take risks
        self.energy_conservation_threshold = 0.3  # When to conserve energy

        # Patch selection weights
        self.quality_weight = 0.4
        self.distance_weight = 0.3
        self.reliability_weight = 0.2
        self.novelty_weight = 0.1

    def decide_next_action(
        self, current_time: int, available_patches: List[ResourcePatch]
    ) -> str:
        """Decide on next foraging action"""

        # Update patch memories
        self._decay_memories(current_time)

        # State-based decision making
        if self.foraging_state == ForagingState.SEARCHING:
            return self._decide_search_action(available_patches, current_time)
        elif self.foraging_state == ForagingState.TRAVELING:
            return self._decide_travel_action()
        elif self.foraging_state == ForagingState.COLLECTING:
            return self._decide_collection_action()
        elif self.foraging_state == ForagingState.RETURNING:
            return self._decide_return_action()
        elif self.foraging_state == ForagingState.DANCING:
            return self._decide_dance_action()
        elif self.foraging_state == ForagingState.FOLLOWING_DANCE:
            return self._decide_follow_action()
        else:  # RESTING
            return self._decide_rest_action()

    def _decide_search_action(
        self, available_patches: List[ResourcePatch], current_time: int
    ) -> str:
        """Decide what to do when searching for patches"""

        # Check energy levels
        if (
            self.forager.energy
            < self.energy_conservation_threshold * self.forager.max_energy
        ):
            self.foraging_state = ForagingState.RESTING
            return "rest"

        # Check for social information (dance following)
        if self._should_follow_dance():
            self.foraging_state = ForagingState.FOLLOWING_DANCE
            return "follow_dance"

        # Select target patch
        target_patch = self._select_target_patch(available_patches, current_time)

        if target_patch:
            self._start_trip(target_patch.id, current_time)
            self.foraging_state = ForagingState.TRAVELING
            return f"travel_to_patch_{target_patch.id}"
        else:
            # No suitable patch found - explore or rest
            if random.random() < self.exploration_rate:
                return "explore_randomly"
            else:
                self.foraging_state = ForagingState.RESTING
                return "rest"

    def _select_target_patch(
        self, available_patches: List[ResourcePatch], current_time: int
    ) -> Optional[ResourcePatch]:
        """Select optimal target patch using multi-criteria decision making"""

        if not available_patches:
            return None

        patch_scores = []

        for patch in available_patches:
            score = self._calculate_patch_score(patch, current_time)
            patch_scores.append((score, patch))

        # Sort by score and select best patch (with some randomness)
        patch_scores.sort(reverse=True, key=lambda x: x[0])

        # Probabilistic selection favoring better patches
        if patch_scores:
            weights = [
                math.exp(score * 3) for score, _ in patch_scores[:5]
            ]  # Top 5 patches
            selected_idx = random.choices(range(len(weights)), weights=weights)[0]
            return patch_scores[selected_idx][1]

        return None

    def _calculate_patch_score(self, patch: ResourcePatch, current_time: int) -> float:
        """Calculate attractiveness score for a patch"""

        # Base quality score
        quality_score = patch.get_resource_quality()

        # Distance penalty
        distance = self.forager.get_distance_to(patch.location)
        distance_score = 1.0 / (1.0 + distance / 1000.0)  # Normalize by 1km

        # Memory-based reliability
        reliability_score = 0.5  # Default for unknown patches
        if patch.id in self.patch_memories:
            memory = self.patch_memories[patch.id]
            reliability_score = memory.reliability

            # Adjust quality based on memory
            quality_score = 0.7 * quality_score + 0.3 * memory.average_quality

        # Novelty bonus for exploration
        novelty_score = 1.0
        if patch.id in self.patch_memories:
            days_since_visit = current_time - self.patch_memories[patch.id].last_visit
            novelty_score = min(1.0, days_since_visit / 7.0)  # Peaks at 1 week

        # Energy cost consideration
        energy_cost = self._estimate_energy_cost(distance)
        energy_score = 1.0 if self.forager.energy > energy_cost * 2 else 0.5

        # Combine scores
        total_score = (
            self.quality_weight * quality_score
            + self.distance_weight * distance_score
            + self.reliability_weight * reliability_score
            + self.novelty_weight * novelty_score
        ) * energy_score

        return total_score

    def _estimate_energy_cost(self, distance: float) -> float:
        """Estimate energy cost for round trip to patch"""
        cost = distance * 2 * self.forager.model.config.foraging.energy_cost_per_meter
        return float(cost)

    def _start_trip(self, patch_id: int, current_time: int) -> None:
        """Start a new foraging trip"""
        self.trip_counter += 1
        self.current_trip = ForagingTrip(
            trip_id=self.trip_counter, start_time=current_time, patch_visited=patch_id
        )

    def _decide_travel_action(self) -> str:
        """Decide action while traveling to patch"""
        # Simulate travel time/energy cost
        if self.current_trip:
            # Check if arrived at patch
            if random.random() < 0.8:  # 80% chance to arrive each step
                self.foraging_state = ForagingState.COLLECTING
                return "start_collecting"

        return "continue_traveling"

    def _decide_collection_action(self) -> str:
        """Decide action while collecting resources"""
        if not self.current_trip:
            self.foraging_state = ForagingState.SEARCHING
            return "search"

        # Simulate resource collection
        collection_success = random.random() < 0.9  # 90% success rate

        if collection_success:
            # Update trip record
            self.current_trip.resources_collected = random.uniform(20, 50)  # mg
            self.current_trip.successful = True

            # Update patch memory
            if self.current_trip.patch_visited:
                self._update_patch_memory(
                    self.current_trip.patch_visited, self.current_trip.start_time
                )

        # Check if carrying capacity reached or patch depleted
        if (
            self.forager.current_load >= self.forager.carrying_capacity * 0.8
            or not collection_success
        ):
            self.foraging_state = ForagingState.RETURNING
            return "return_to_hive"

        return "continue_collecting"

    def _decide_return_action(self) -> str:
        """Decide action while returning to hive"""
        # Simulate return travel
        if random.random() < 0.7:  # 70% chance to arrive each step
            self._complete_trip()

            # Decide whether to dance based on trip success
            if (
                self.current_trip
                and self.current_trip.successful
                and self.current_trip.resources_collected > self.forager.dance_threshold
            ):
                self.foraging_state = ForagingState.DANCING
                return "start_dancing"
            else:
                self.foraging_state = ForagingState.RESTING
                return "rest"

        return "continue_returning"

    def _decide_dance_action(self) -> str:
        """Decide dancing behavior"""
        # Dance for a few time steps
        if random.random() < 0.3:  # 30% chance to stop dancing each step
            self.foraging_state = ForagingState.SEARCHING
            return "stop_dancing"
        return "continue_dancing"

    def _decide_follow_action(self) -> str:
        """Decide action when following dance"""
        # Follow dance to learn about patch
        self.foraging_state = ForagingState.TRAVELING
        return "travel_to_danced_patch"

    def _decide_rest_action(self) -> str:
        """Decide action while resting"""
        # Rest until energy recovers
        if self.forager.energy > 0.8 * self.forager.max_energy:
            self.foraging_state = ForagingState.SEARCHING
            return "start_searching"
        return "continue_resting"

    def _complete_trip(self) -> None:
        """Complete current foraging trip"""
        if self.current_trip:
            self.current_trip.end_time = self.forager.model.schedule.steps
            self.trip_history.append(self.current_trip)
            self.current_trip = None

    def _update_patch_memory(self, patch_id: int, current_time: int) -> None:
        """Update memory of visited patch"""
        # Get patch from landscape
        patch = self.forager.model.landscape.get_patch(patch_id)
        if not patch:
            return

        # Create or update memory
        if patch_id not in self.patch_memories:
            self.patch_memories[patch_id] = PatchMemory(
                patch_id=patch_id, location=patch.location, last_visit=current_time
            )

        memory = self.patch_memories[patch_id]
        distance = self.forager.get_distance_to(patch.location)
        quality = patch.get_resource_quality()
        resources = patch.get_available_resources()

        memory.update_visit(quality, distance, resources, current_time)

    def _decay_memories(self, current_time: int) -> None:
        """Decay patch memories over time"""
        for memory in self.patch_memories.values():
            days_since_visit = current_time - memory.last_visit
            if days_since_visit > 0:
                decay_factor = self.memory_decay_rate**days_since_visit
                memory.average_quality *= decay_factor
                memory.reliability *= decay_factor

    def _should_follow_dance(self) -> bool:
        """Decide whether to follow a dance"""
        # Check if there are dancing foragers nearby
        # For now, simple probability
        return random.random() < 0.1  # 10% chance to follow dance

    def get_foraging_efficiency(self) -> float:
        """Calculate foraging efficiency based on trip history"""
        if not self.trip_history:
            return 0.0

        recent_trips = self.trip_history[-10:]  # Last 10 trips
        successful_trips = [t for t in recent_trips if t.successful]

        if not recent_trips:
            return 0.0

        success_rate = len(successful_trips) / len(recent_trips)

        if successful_trips:
            avg_resources = sum(t.resources_collected for t in successful_trips) / len(
                successful_trips
            )
            avg_energy = sum(t.energy_spent for t in successful_trips) / len(
                successful_trips
            )

            efficiency = (avg_resources / max(avg_energy, 1.0)) * success_rate
        else:
            efficiency = 0.0

        return efficiency

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of patch memories"""
        if not self.patch_memories:
            return {"total_patches": 0}

        return {
            "total_patches": len(self.patch_memories),
            "avg_reliability": sum(m.reliability for m in self.patch_memories.values())
            / len(self.patch_memories),
            "total_visits": sum(m.visit_count for m in self.patch_memories.values()),
            "best_patch_id": max(
                self.patch_memories.keys(),
                key=lambda pid: self.patch_memories[pid].average_quality,
            ),
            "most_visited_patch": max(
                self.patch_memories.keys(),
                key=lambda pid: self.patch_memories[pid].visit_count,
            ),
        }


class RouteOptimizer:
    """
    Optimizes foraging routes for multiple patch visits.

    Uses simplified algorithms for:
    - Traveling Salesman Problem (TSP)
    - Energy-constrained routing
    - Multi-objective optimization
    """

    def __init__(self, forager: Forager):
        self.forager = forager

    def optimize_multi_patch_route(
        self, patches: List[ResourcePatch], max_energy: float
    ) -> List[ResourcePatch]:
        """Optimize route for visiting multiple patches"""

        if len(patches) <= 1:
            return patches

        # Calculate distance matrix
        n = len(patches)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = self._calculate_distance(patches[i], patches[j])

        # Simple greedy TSP approximation
        current_pos = 0  # Start from first patch
        visited = {0}
        route = [patches[0]]
        total_energy = 0

        while len(visited) < n:
            min_dist = float("inf")
            next_patch = -1

            for j in range(n):
                if j not in visited and distances[current_pos, j] < min_dist:
                    # Check energy constraint
                    trip_energy = (
                        distances[current_pos, j]
                        * self.forager.model.config.foraging.energy_cost_per_meter
                    )
                    if total_energy + trip_energy <= max_energy:
                        min_dist = distances[current_pos, j]
                        next_patch = j

            if next_patch == -1:
                break  # No more reachable patches

            visited.add(next_patch)
            route.append(patches[next_patch])
            total_energy += (
                min_dist * self.forager.model.config.foraging.energy_cost_per_meter
            )
            current_pos = next_patch

        return route

    def _calculate_distance(
        self, patch1: ResourcePatch, patch2: ResourcePatch
    ) -> float:
        """Calculate distance between two patches"""
        dx = patch1.x - patch2.x
        dy = patch1.y - patch2.y
        return math.sqrt(dx**2 + dy**2)

    def estimate_route_value(self, route: List[ResourcePatch]) -> float:
        """Estimate total value of a route"""
        total_value: float = 0
        total_distance: float = 0

        for i, patch in enumerate(route):
            # Add patch value
            total_value += (
                patch.get_resource_quality() * patch.get_available_resources()
            )

            # Add travel cost
            if i > 0:
                distance = self._calculate_distance(route[i - 1], patch)
                total_distance += distance

        # Return value minus cost
        energy_cost = (
            total_distance * self.forager.model.config.foraging.energy_cost_per_meter
        )
        result = total_value - energy_cost * 0.1  # Weight energy cost
        return float(result)


class CollectiveForagingIntelligence:
    """
    Manages collective foraging intelligence and information sharing.

    Implements:
    - Swarm intelligence algorithms
    - Information pooling
    - Collective patch assessment
    - Resource allocation optimization
    """

    def __init__(self, colony: Any) -> None:
        self.colony = colony
        self.shared_patch_info: Dict[int, Dict[str, Any]] = {}
        self.patch_visit_counts: Dict[int, int] = defaultdict(int)
        self.global_patch_rankings: List[Tuple[int, float]] = []

    def update_shared_information(
        self,
        forager: Any,
        patch_id: int,
        quality: float,
        resources: float,
        distance: float,
    ) -> None:
        """Update shared patch information from forager experience"""

        if patch_id not in self.shared_patch_info:
            self.shared_patch_info[patch_id] = {
                "quality_samples": [],
                "resource_samples": [],
                "distance_samples": [],
                "last_updated": 0,
                "reliability": 0.5,
            }

        info = self.shared_patch_info[patch_id]

        # Add new samples
        info["quality_samples"].append(quality)
        info["resource_samples"].append(resources)
        info["distance_samples"].append(distance)
        info["last_updated"] = self.colony.model.schedule.steps

        # Keep only recent samples
        max_samples = 20
        if len(info["quality_samples"]) > max_samples:
            info["quality_samples"] = info["quality_samples"][-max_samples:]
            info["resource_samples"] = info["resource_samples"][-max_samples:]
            info["distance_samples"] = info["distance_samples"][-max_samples:]

        # Update visit count
        self.patch_visit_counts[patch_id] += 1

        # Update reliability based on consistency
        if len(info["quality_samples"]) >= 3:
            quality_var = np.var(info["quality_samples"])
            info["reliability"] = max(0.1, 1.0 - quality_var)

    def get_patch_recommendation(
        self, forager: Forager, available_patches: List[ResourcePatch]
    ) -> Optional[ResourcePatch]:
        """Get patch recommendation based on collective intelligence"""

        patch_scores = []

        for patch in available_patches:
            score = self._calculate_collective_score(patch, forager)
            patch_scores.append((score, patch))

        if patch_scores:
            # Sort by score and add some randomness
            patch_scores.sort(reverse=True, key=lambda x: x[0])

            # Weighted random selection from top patches
            top_patches = patch_scores[: min(3, len(patch_scores))]
            weights = [score for score, _ in top_patches]

            if sum(weights) > 0:
                selected_idx = random.choices(range(len(weights)), weights=weights)[0]
                return top_patches[selected_idx][1]

        return None

    def _calculate_collective_score(
        self, patch: ResourcePatch, forager: Forager
    ) -> float:
        """Calculate patch score based on collective information"""

        base_score = patch.get_resource_quality()

        # Incorporate shared information
        if patch.id in self.shared_patch_info:
            info = self.shared_patch_info[patch.id]

            if info["quality_samples"]:
                shared_quality = np.mean(info["quality_samples"])
                reliability = info["reliability"]

                # Weighted average of individual and shared assessment
                base_score = (
                    1 - reliability
                ) * base_score + reliability * shared_quality

        # Distance penalty
        distance = forager.get_distance_to(patch.location)
        distance_penalty = 1.0 / (1.0 + distance / 1000.0)

        # Crowding penalty (avoid over-exploitation)
        visit_count = self.patch_visit_counts.get(patch.id, 0)
        crowding_penalty = 1.0 / (1.0 + visit_count / 10.0)

        return base_score * distance_penalty * crowding_penalty

    def update_global_rankings(self) -> None:
        """Update global patch rankings based on collective information"""

        patch_rankings = []

        for patch_id, info in self.shared_patch_info.items():
            if info["quality_samples"]:
                avg_quality = np.mean(info["quality_samples"])
                reliability = info["reliability"]
                visit_count = self.patch_visit_counts[patch_id]

                # Composite score
                score = avg_quality * reliability * math.log(1 + visit_count)
                patch_rankings.append((patch_id, score))

        # Sort by score
        patch_rankings.sort(reverse=True, key=lambda x: x[1])
        self.global_patch_rankings = patch_rankings

    def get_foraging_statistics(self) -> Dict[str, Any]:
        """Get collective foraging statistics"""

        total_patches = len(self.shared_patch_info)
        total_visits = sum(self.patch_visit_counts.values())

        if self.shared_patch_info:
            avg_quality = np.mean(
                [
                    np.mean(info["quality_samples"]) if info["quality_samples"] else 0
                    for info in self.shared_patch_info.values()
                ]
            )

            avg_reliability = np.mean(
                [info["reliability"] for info in self.shared_patch_info.values()]
            )
        else:
            avg_quality = 0.0
            avg_reliability = 0.0

        return {
            "total_patches_known": total_patches,
            "total_visits": total_visits,
            "average_patch_quality": avg_quality,
            "average_reliability": avg_reliability,
            "most_visited_patch": (
                max(self.patch_visit_counts.items(), key=lambda x: x[1])[0]
                if self.patch_visit_counts
                else None
            ),
            "top_ranked_patches": self.global_patch_rankings[:5],
        }
