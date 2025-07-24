"""
Foraging Trip Management System for NetLogo BEE-STEWARD v2 Parity
================================================================

Complete foraging trip lifecycle tracking with success/failure criteria,
return trip optimization, and comprehensive energy cost calculations.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import math
import time


class TripPhase(Enum):
    """Phases of a foraging trip"""

    OUTBOUND = "outbound"
    FORAGING = "foraging"
    RETURN = "return"
    UNLOADING = "unloading"
    COMPLETED = "completed"
    ABORTED = "aborted"


class TripResult(Enum):
    """Overall trip success/failure results"""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ABORTED = "aborted"
    ENERGY_EXHAUSTED = "energy_exhausted"
    WEATHER_ABORT = "weather_abort"
    CAPACITY_EXCEEDED = "capacity_exceeded"


@dataclass
class TripWaypoint:
    """Individual waypoint in a foraging trip"""

    patch_id: int
    location: Tuple[float, float]
    arrival_time: float
    departure_time: Optional[float] = None
    collection_attempts: int = 0
    successful_collections: int = 0
    nectar_collected: float = 0.0
    pollen_collected: float = 0.0
    energy_spent: float = 0.0
    handling_time: float = 0.0
    collection_efficiency: float = 0.0


@dataclass
class TripMetrics:
    """Comprehensive trip performance metrics"""

    trip_id: str
    bee_id: int
    start_time: float
    end_time: Optional[float] = None
    total_duration: float = 0.0

    # Distance and movement
    total_distance: float = 0.0
    outbound_distance: float = 0.0
    return_distance: float = 0.0
    flight_efficiency: float = 1.0

    # Energy management
    initial_energy: float = 100.0
    final_energy: float = 100.0
    energy_consumed: float = 0.0
    energy_efficiency: float = 1.0

    # Resource collection
    patches_visited: int = 0
    collection_attempts: int = 0
    successful_collections: int = 0
    total_nectar: float = 0.0
    total_pollen: float = 0.0

    # Performance indicators
    success_rate: float = 0.0
    resource_per_energy: float = 0.0
    resource_per_time: float = 0.0
    trip_result: TripResult = TripResult.SUCCESS

    # Route optimization
    route_optimality: float = 1.0
    return_optimization: float = 1.0

    # Environmental factors
    weather_conditions: Dict[str, float] = field(default_factory=dict)
    environmental_penalties: float = 0.0


class EnergyCalculator(BaseModel):
    """Energy cost calculations for foraging trips"""

    model_config = {"validate_assignment": True}

    # Base energy costs
    base_flight_cost: float = Field(
        default=0.1, ge=0.0, description="Base energy cost per meter"
    )
    base_foraging_cost: float = Field(
        default=0.5, ge=0.0, description="Base energy cost per foraging attempt"
    )
    base_handling_cost: float = Field(
        default=0.2, ge=0.0, description="Base energy cost per second handling"
    )

    # Load-based multipliers
    empty_flight_multiplier: float = Field(
        default=1.0, ge=0.0, description="Empty flight energy multiplier"
    )
    loaded_flight_multiplier: float = Field(
        default=1.5, ge=1.0, description="Loaded flight energy multiplier"
    )
    max_load_penalty: float = Field(
        default=2.0, ge=1.0, description="Maximum load penalty multiplier"
    )

    # Environmental factors
    wind_cost_factor: float = Field(
        default=0.1, ge=0.0, description="Wind resistance energy factor"
    )
    temperature_cost_factor: float = Field(
        default=0.05, ge=0.0, description="Temperature regulation cost"
    )
    altitude_cost_factor: float = Field(
        default=0.02, ge=0.0, description="Altitude flight cost"
    )

    def calculate_flight_energy(
        self,
        distance: float,
        current_load: float,
        max_load: float,
        environmental_conditions: Dict[str, float],
    ) -> float:
        """Calculate energy cost for flight"""

        # Base flight cost
        base_cost = distance * self.base_flight_cost

        # Load-based adjustment
        load_factor = current_load / max_load if max_load > 0 else 0.0
        load_multiplier = self.empty_flight_multiplier + (
            (self.loaded_flight_multiplier - self.empty_flight_multiplier) * load_factor
        )

        # Maximum load penalty
        if load_factor > 0.9:
            excess_load = (load_factor - 0.9) / 0.1
            load_multiplier *= 1.0 + excess_load * (self.max_load_penalty - 1.0)

        # Environmental adjustments
        env_multiplier = 1.0

        # Wind resistance
        wind_speed = environmental_conditions.get("wind_speed", 0.0)
        if wind_speed > 2.0:
            env_multiplier += (wind_speed - 2.0) * self.wind_cost_factor

        # Temperature regulation
        temperature = environmental_conditions.get("temperature", 22.0)
        if temperature < 10.0 or temperature > 30.0:
            temp_stress = abs(temperature - 22.0) / 20.0
            env_multiplier += temp_stress * self.temperature_cost_factor

        # Altitude (if available)
        altitude = environmental_conditions.get("altitude", 0.0)
        if altitude > 100.0:
            env_multiplier += (altitude / 1000.0) * self.altitude_cost_factor

        return base_cost * load_multiplier * env_multiplier

    def calculate_foraging_energy(
        self,
        handling_time: float,
        collection_attempts: int,
        environmental_conditions: Dict[str, float],
    ) -> float:
        """Calculate energy cost for foraging activities"""

        # Base foraging cost
        base_cost = collection_attempts * self.base_foraging_cost

        # Handling time cost
        handling_cost = handling_time * self.base_handling_cost

        # Environmental factors
        env_multiplier = 1.0

        # Weather stress
        weather_condition = environmental_conditions.get("weather", "clear")
        if weather_condition in ["rain", "wind"]:
            env_multiplier += 0.3
        elif weather_condition == "thunderstorm":
            env_multiplier += 0.7

        return (base_cost + handling_cost) * env_multiplier


class RouteOptimizer(BaseModel):
    """Route optimization for foraging trips"""

    model_config = {"validate_assignment": True}

    # Optimization parameters
    max_patches_per_trip: int = Field(
        default=10, ge=1, description="Maximum patches per trip"
    )
    optimization_algorithm: str = Field(
        default="nearest_neighbor", description="Optimization algorithm"
    )
    return_optimization: bool = Field(default=True, description="Optimize return route")

    # Heuristic weights
    distance_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Distance importance"
    )
    quality_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Quality importance"
    )
    energy_weight: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Energy importance"
    )
    time_weight: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Time importance"
    )

    def optimize_route(
        self,
        hive_location: Tuple[float, float],
        available_patches: List[Dict[str, Any]],
        current_energy: float,
        max_energy: float,
    ) -> List[int]:
        """Optimize foraging route using selected algorithm"""

        if not available_patches:
            return []

        if self.optimization_algorithm == "nearest_neighbor":
            return self._nearest_neighbor_route(hive_location, available_patches)
        elif self.optimization_algorithm == "quality_first":
            return self._quality_first_route(available_patches)
        elif self.optimization_algorithm == "energy_optimal":
            return self._energy_optimal_route(
                hive_location, available_patches, current_energy
            )
        elif self.optimization_algorithm == "balanced":
            return self._balanced_route(
                hive_location, available_patches, current_energy
            )
        else:
            return self._nearest_neighbor_route(hive_location, available_patches)

    def _nearest_neighbor_route(
        self, start_location: Tuple[float, float], patches: List[Dict[str, Any]]
    ) -> List[int]:
        """Nearest neighbor route optimization"""

        if not patches:
            return []

        route: List[int] = []
        current_location = start_location
        remaining_patches = patches.copy()

        while remaining_patches and len(route) < self.max_patches_per_trip:
            # Find nearest patch
            nearest_patch = min(
                remaining_patches,
                key=lambda p: self._calculate_distance(current_location, p["location"]),
            )

            route.append(nearest_patch["patch_id"])
            current_location = nearest_patch["location"]
            remaining_patches.remove(nearest_patch)

        return route

    def _quality_first_route(self, patches: List[Dict[str, Any]]) -> List[int]:
        """Quality-first route optimization"""

        # Sort by quality descending
        sorted_patches = sorted(
            patches, key=lambda p: p.get("quality", 0.0), reverse=True
        )

        return [p["patch_id"] for p in sorted_patches[: self.max_patches_per_trip]]

    def _energy_optimal_route(
        self,
        start_location: Tuple[float, float],
        patches: List[Dict[str, Any]],
        current_energy: float,
    ) -> List[int]:
        """Energy-optimal route planning"""

        route: List[int] = []
        current_location = start_location
        remaining_energy = current_energy
        remaining_patches = patches.copy()

        while remaining_patches and len(route) < self.max_patches_per_trip:
            best_patch = None
            best_score = -1.0

            for patch in remaining_patches:
                # Calculate energy requirement
                distance = self._calculate_distance(current_location, patch["location"])
                energy_required = distance * 0.2  # Simplified energy cost

                # Return trip energy
                return_energy = (
                    self._calculate_distance(patch["location"], start_location) * 0.2
                )

                # Check if feasible
                if energy_required + return_energy < remaining_energy:
                    # Calculate efficiency score
                    quality = patch.get("quality", 0.5)
                    score = quality / (energy_required + 0.1)  # Avoid division by zero

                    if score > best_score:
                        best_score = score
                        best_patch = patch

            if best_patch:
                route.append(best_patch["patch_id"])
                current_location = best_patch["location"]
                distance = self._calculate_distance(
                    current_location, best_patch["location"]
                )
                remaining_energy -= distance * 0.2
                remaining_patches.remove(best_patch)
            else:
                break  # No more feasible patches

        return route

    def _balanced_route(
        self,
        start_location: Tuple[float, float],
        patches: List[Dict[str, Any]],
        current_energy: float,
    ) -> List[int]:
        """Balanced multi-criteria route optimization"""

        route: List[int] = []
        current_location = start_location
        remaining_patches = patches.copy()

        while remaining_patches and len(route) < self.max_patches_per_trip:
            best_patch = None
            best_score = -1.0

            for patch in remaining_patches:
                # Calculate individual scores
                distance = self._calculate_distance(current_location, patch["location"])
                distance_score = 1.0 / (1.0 + distance / 1000.0)  # Normalize by 1km

                quality_score = patch.get("quality", 0.5)

                energy_cost = distance * 0.2
                energy_score = 1.0 - (energy_cost / current_energy)

                handling_time = patch.get("handling_time", 3.0)
                time_score = 1.0 / (1.0 + handling_time / 10.0)  # Normalize by 10s

                # Combined score
                combined_score = (
                    distance_score * self.distance_weight
                    + quality_score * self.quality_weight
                    + energy_score * self.energy_weight
                    + time_score * self.time_weight
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_patch = patch

            if best_patch:
                route.append(best_patch["patch_id"])
                current_location = best_patch["location"]
                remaining_patches.remove(best_patch)
            else:
                break

        return route

    def _calculate_distance(
        self, loc1: Tuple[float, float], loc2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two locations"""
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    def optimize_return_route(
        self,
        current_location: Tuple[float, float],
        hive_location: Tuple[float, float],
        current_load: float,
        energy_remaining: float,
    ) -> Dict[str, Any]:
        """Optimize return route to hive"""

        # Direct distance to hive
        direct_distance = self._calculate_distance(current_location, hive_location)

        # Calculate load factor
        load_factor = min(1.0, current_load / 100.0)  # Normalize load

        # Energy factor
        energy_factor = energy_remaining / 100.0  # Normalize energy

        # Optimization decisions
        if energy_factor < 0.3:  # Low energy - direct route
            return {
                "route_type": "direct",
                "waypoints": [hive_location],
                "distance": direct_distance,
                "optimization_reason": "energy_conservation",
            }
        elif load_factor > 0.8:  # Heavy load - direct route
            return {
                "route_type": "direct",
                "waypoints": [hive_location],
                "distance": direct_distance,
                "optimization_reason": "heavy_load",
            }
        else:
            # Check for intermediate stops (simplified)
            return {
                "route_type": "direct",
                "waypoints": [hive_location],
                "distance": direct_distance,
                "optimization_reason": "optimal_direct",
            }


class ForagingTripManager(BaseModel):
    """Complete foraging trip management system"""

    model_config = {"validate_assignment": True}

    # Component systems
    energy_calculator: EnergyCalculator = Field(default_factory=EnergyCalculator)
    route_optimizer: RouteOptimizer = Field(default_factory=RouteOptimizer)

    # Trip management
    active_trips: Dict[str, TripMetrics] = Field(
        default_factory=dict, description="Active trips"
    )
    completed_trips: List[TripMetrics] = Field(
        default_factory=list, description="Completed trips"
    )

    # Performance tracking
    trip_counter: int = Field(default=0, description="Trip counter")
    success_rate_window: int = Field(
        default=50, ge=1, description="Success rate calculation window"
    )

    def start_foraging_trip(
        self,
        bee_id: int,
        hive_location: Tuple[float, float],
        available_patches: List[Dict[str, Any]],
        initial_energy: float,
        environmental_conditions: Dict[str, float],
    ) -> str:
        """Start a new foraging trip"""

        self.trip_counter += 1
        trip_id = f"trip_{bee_id}_{self.trip_counter}"

        # Optimize route
        _route = self.route_optimizer.optimize_route(
            hive_location, available_patches, initial_energy, 100.0
        )

        # Create trip metrics
        trip_metrics = TripMetrics(
            trip_id=trip_id,
            bee_id=bee_id,
            start_time=time.time(),
            initial_energy=initial_energy,
            final_energy=initial_energy,
            weather_conditions=environmental_conditions.copy(),
        )

        self.active_trips[trip_id] = trip_metrics

        return trip_id

    def update_trip_waypoint(self, trip_id: str, waypoint: TripWaypoint) -> None:
        """Update trip with new waypoint information"""

        if trip_id not in self.active_trips:
            return

        trip = self.active_trips[trip_id]

        # Update trip metrics
        trip.patches_visited += 1
        trip.collection_attempts += waypoint.collection_attempts
        trip.successful_collections += waypoint.successful_collections
        trip.total_nectar += waypoint.nectar_collected
        trip.total_pollen += waypoint.pollen_collected

        # Update distances (simplified)
        if waypoint.departure_time and waypoint.arrival_time:
            trip.total_duration += waypoint.departure_time - waypoint.arrival_time

        # Update energy
        trip.energy_consumed += waypoint.energy_spent
        trip.final_energy = trip.initial_energy - trip.energy_consumed

    def complete_trip(
        self,
        trip_id: str,
        final_location: Tuple[float, float],
        final_energy: float,
        trip_result: TripResult,
    ) -> TripMetrics:
        """Complete a foraging trip and calculate final metrics"""

        if trip_id not in self.active_trips:
            raise ValueError(f"Trip {trip_id} not found in active trips")

        trip = self.active_trips[trip_id]

        # Finalize trip metrics
        trip.end_time = time.time()
        trip.total_duration = trip.end_time - trip.start_time
        trip.final_energy = final_energy
        trip.energy_consumed = trip.initial_energy - final_energy
        trip.trip_result = trip_result

        # Calculate performance metrics
        self._calculate_trip_performance(trip)

        # Move to completed trips
        self.completed_trips.append(trip)
        if len(self.completed_trips) > 1000:  # Limit memory usage
            self.completed_trips.pop(0)

        del self.active_trips[trip_id]

        return trip

    def _calculate_trip_performance(self, trip: TripMetrics) -> None:
        """Calculate comprehensive trip performance metrics"""

        # Success rate
        if trip.collection_attempts > 0:
            trip.success_rate = trip.successful_collections / trip.collection_attempts

        # Resource efficiency
        total_resources = trip.total_nectar + trip.total_pollen
        if trip.energy_consumed > 0:
            trip.resource_per_energy = total_resources / trip.energy_consumed

        if trip.total_duration > 0:
            trip.resource_per_time = total_resources / trip.total_duration

        # Energy efficiency
        if trip.initial_energy > 0:
            trip.energy_efficiency = 1.0 - (trip.energy_consumed / trip.initial_energy)

        # Environmental penalties
        weather_condition = trip.weather_conditions.get("weather", "clear")
        if weather_condition in ["rain", "wind"]:
            trip.environmental_penalties += 0.2
        elif weather_condition == "thunderstorm":
            trip.environmental_penalties += 0.5

    def abort_trip(self, trip_id: str, reason: str) -> Optional[TripMetrics]:
        """Abort an active trip"""

        if trip_id not in self.active_trips:
            return None

        trip = self.active_trips[trip_id]
        trip.end_time = time.time()
        trip.total_duration = trip.end_time - trip.start_time
        trip.trip_result = TripResult.ABORTED

        # Record abort reason (store as numeric code for type safety)
        trip.weather_conditions["abort_reason"] = 1.0  # Numeric code for abort

        # Move to completed trips
        self.completed_trips.append(trip)
        del self.active_trips[trip_id]

        return trip

    def get_trip_recommendations(
        self, bee_id: int, current_energy: float, weather_conditions: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get trip recommendations based on current conditions"""

        recommendations = []

        # Energy-based recommendations
        if current_energy < 30:
            recommendations.append(
                {
                    "type": "energy_warning",
                    "message": "Energy too low for extended foraging",
                    "priority": "high",
                }
            )
        elif current_energy < 50:
            recommendations.append(
                {
                    "type": "energy_caution",
                    "message": "Consider shorter foraging trips",
                    "priority": "medium",
                }
            )

        # Weather-based recommendations
        weather = weather_conditions.get("weather", "clear")
        if weather in ["rain", "thunderstorm"]:
            recommendations.append(
                {
                    "type": "weather_warning",
                    "message": "Avoid foraging in current weather",
                    "priority": "high",
                }
            )

        wind_speed = weather_conditions.get("wind_speed", 0.0)
        if wind_speed > 5.0:
            recommendations.append(
                {
                    "type": "wind_warning",
                    "message": "High wind conditions may increase energy costs",
                    "priority": "medium",
                }
            )

        # Historical performance recommendations
        recent_trips = self.get_recent_trip_performance(bee_id)
        if recent_trips["success_rate"] < 0.5:
            recommendations.append(
                {
                    "type": "performance_advice",
                    "message": "Consider visiting familiar patches to improve success rate",
                    "priority": "medium",
                }
            )

        return {
            "recommendations": recommendations,
            "optimal_trip_length": self._calculate_optimal_trip_length(
                current_energy, weather_conditions
            ),
            "energy_budget": self._calculate_energy_budget(
                current_energy, weather_conditions
            ),
        }

    def get_recent_trip_performance(self, bee_id: int) -> Dict[str, float]:
        """Get recent trip performance for a bee"""

        recent_trips = [
            trip
            for trip in self.completed_trips[-self.success_rate_window :]
            if trip.bee_id == bee_id
        ]

        if not recent_trips:
            return {
                "success_rate": 0.5,
                "average_energy_efficiency": 0.5,
                "average_resource_per_time": 0.0,
                "trip_count": 0,
            }

        return {
            "success_rate": sum(trip.success_rate for trip in recent_trips)
            / len(recent_trips),
            "average_energy_efficiency": sum(
                trip.energy_efficiency for trip in recent_trips
            )
            / len(recent_trips),
            "average_resource_per_time": sum(
                trip.resource_per_time for trip in recent_trips
            )
            / len(recent_trips),
            "trip_count": len(recent_trips),
        }

    def _calculate_optimal_trip_length(
        self, current_energy: float, weather_conditions: Dict[str, float]
    ) -> int:
        """Calculate optimal number of patches for trip"""

        # Base calculation
        energy_factor = current_energy / 100.0
        base_patches = int(energy_factor * 10)  # 0-10 patches based on energy

        # Weather adjustment
        weather = weather_conditions.get("weather", "clear")
        if weather in ["rain", "thunderstorm"]:
            base_patches = max(1, base_patches // 2)

        wind_speed = weather_conditions.get("wind_speed", 0.0)
        if wind_speed > 5.0:
            base_patches = max(1, int(base_patches * 0.7))

        return max(1, min(base_patches, self.route_optimizer.max_patches_per_trip))

    def _calculate_energy_budget(
        self, current_energy: float, weather_conditions: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate energy budget for trip"""

        # Reserve energy for return trip
        reserve_energy = current_energy * 0.3

        # Available energy for foraging
        available_energy = current_energy - reserve_energy

        # Weather adjustments
        weather_multiplier = 1.0
        weather = weather_conditions.get("weather", "clear")
        if weather in ["rain", "wind"]:
            weather_multiplier = 1.3
        elif weather == "thunderstorm":
            weather_multiplier = 1.6

        # Adjusted available energy
        adjusted_available = available_energy / weather_multiplier

        return {
            "total_energy": current_energy,
            "reserve_energy": reserve_energy,
            "available_energy": available_energy,
            "weather_adjusted_available": adjusted_available,
            "weather_multiplier": weather_multiplier,
        }

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""

        if not self.completed_trips:
            return {"total_trips": 0}

        # Overall statistics
        total_trips = len(self.completed_trips)
        successful_trips = sum(
            1 for trip in self.completed_trips if trip.trip_result == TripResult.SUCCESS
        )

        # Performance metrics
        avg_success_rate = (
            sum(trip.success_rate for trip in self.completed_trips) / total_trips
        )
        avg_energy_efficiency = (
            sum(trip.energy_efficiency for trip in self.completed_trips) / total_trips
        )
        avg_resource_per_time = (
            sum(trip.resource_per_time for trip in self.completed_trips) / total_trips
        )

        # Resource collection
        total_nectar = sum(trip.total_nectar for trip in self.completed_trips)
        total_pollen = sum(trip.total_pollen for trip in self.completed_trips)

        return {
            "total_trips": total_trips,
            "successful_trips": successful_trips,
            "success_percentage": (successful_trips / total_trips) * 100,
            "average_success_rate": avg_success_rate,
            "average_energy_efficiency": avg_energy_efficiency,
            "average_resource_per_time": avg_resource_per_time,
            "total_nectar_collected": total_nectar,
            "total_pollen_collected": total_pollen,
            "active_trips": len(self.active_trips),
        }
