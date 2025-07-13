"""
Enhanced Foraging Algorithms for NetLogo BEE-STEWARD v2 Parity
============================================================

Implements sophisticated foraging decision-making algorithms matching
NetLogo's advanced behavioral patterns and memory systems.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
import random
from dataclasses import dataclass

from .agents import ForagingMemory
from .patch_selection import AdvancedPatchSelector, PatchInfo, PatchSelectionStrategy


class ForagingDecisionType(Enum):
    """Types of foraging decisions"""
    EXPLORE_NEW = "explore_new"
    EXPLOIT_KNOWN = "exploit_known"
    FOLLOW_DANCE = "follow_dance"
    RETURN_HOME = "return_home"
    SWITCH_RESOURCE = "switch_resource"
    INVESTIGATE_SCENT = "investigate_scent"


class ForagingStrategy(Enum):
    """Foraging strategy types"""
    CONSERVATIVE = "conservative"  # Prefer known patches
    EXPLORATORY = "exploratory"   # Prefer new patches
    BALANCED = "balanced"         # Mix of both
    OPPORTUNISTIC = "opportunistic"  # Follow best opportunities


@dataclass
class ForagingTripLifecycle:
    """Complete foraging trip lifecycle timing"""
    travel_time_to_patch: float  # Time to fly to patch (seconds)
    handling_time_per_flower: float  # Time to handle each flower (seconds)
    inter_flower_time: float  # Time to move between flowers (seconds)
    flowers_visited: int  # Number of flowers visited
    travel_time_to_hive: float  # Time to return to hive (seconds)
    total_trip_duration: float  # Total trip time (seconds)
    energy_consumed_travel: float  # Energy consumed during travel
    energy_consumed_foraging: float  # Energy consumed during foraging
    energy_gained: float  # Energy gained from resources
    net_energy_gain: float  # Net energy gain/loss

@dataclass
class ForagingContext:
    """Context information for foraging decisions"""
    current_season: str
    current_hour: int
    current_weather: str
    colony_energy_level: float
    patch_competition: Dict[int, float]
    dance_information: List[Any]
    scent_trails: List[Any]
    energy_threshold: float = 50.0
    # Enhanced trip lifecycle context
    wind_speed: float = 0.0  # Wind speed affecting flight time
    temperature: float = 20.0  # Temperature affecting activity
    flower_density: Dict[int, float] = None  # Flower density per patch
    patch_distances: Dict[int, float] = None  # Distance to each patch


class ForagingDecisionEngine(BaseModel):
    """Advanced foraging decision engine matching NetLogo complexity"""
    
    model_config = {"validate_assignment": True}
    
    # Memory management
    memory_capacity: int = Field(default=20, ge=1, le=100, description="Maximum patches remembered")
    memory_decay_rate: float = Field(default=0.95, ge=0.0, le=1.0, description="Daily memory decay")
    
    # Decision weights
    quality_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Quality importance")
    distance_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Distance importance")
    memory_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Memory strength importance")
    novelty_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Novelty preference")
    
    # Behavioral parameters
    exploration_probability: float = Field(default=0.2, ge=0.0, le=1.0, description="Base exploration rate")
    dance_following_probability: float = Field(default=0.3, ge=0.0, le=1.0, description="Dance following rate")
    energy_conservation_threshold: float = Field(default=30.0, ge=0.0, le=100.0, description="Energy conservation threshold")
    
    # Internal state
    foraging_memories: Dict[int, ForagingMemory] = Field(default_factory=dict, description="Foraging memories by patch ID")
    current_strategy: ForagingStrategy = Field(default=ForagingStrategy.BALANCED, description="Current foraging strategy")
    decision_history: List[ForagingDecisionType] = Field(default_factory=list, description="Decision history")
    success_rates: Dict[ForagingDecisionType, float] = Field(default_factory=dict, description="Success rates by decision type")
    patch_selector: AdvancedPatchSelector = Field(default_factory=AdvancedPatchSelector, description="Advanced patch selector")
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Initialize success rates
        if not self.success_rates:
            for decision_type in ForagingDecisionType:
                self.success_rates[decision_type] = 0.5  # Start neutral
    
    def make_foraging_decision(self, bee_id: int, context: ForagingContext, 
                             current_energy: float, available_patches: List[Any]) -> Tuple[ForagingDecisionType, Optional[int]]:
        """Make sophisticated foraging decision based on context and memory"""
        
        # Update memory decay
        self.update_memory_decay(context)
        
        # Determine current strategy based on context
        self.update_foraging_strategy(context, current_energy)
        
        # Get decision probabilities
        decision_probs = self.calculate_decision_probabilities(context, current_energy)
        
        # Make weighted decision
        decision_type = self.weighted_random_choice(decision_probs)
        
        # Execute decision logic
        target_patch = self.execute_decision(decision_type, context, available_patches)
        
        # Update decision history
        self.decision_history.append(decision_type)
        if len(self.decision_history) > 50:  # Keep last 50 decisions
            self.decision_history.pop(0)
        
        return decision_type, target_patch
    
    def update_memory_decay(self, context: ForagingContext) -> None:
        """Update memory strength for all remembered patches"""
        for patch_id, memory in self.foraging_memories.items():
            days_since_visit = max(0, context.current_hour - memory.last_visit) // 24
            memory.update_memory_strength(days_since_visit)
    
    def update_foraging_strategy(self, context: ForagingContext, current_energy: float) -> None:
        """Update foraging strategy based on context"""
        
        # Energy-based strategy adjustment
        if current_energy < self.energy_conservation_threshold:
            self.current_strategy = ForagingStrategy.CONSERVATIVE
        elif context.colony_energy_level < 0.3:  # Colony in need
            self.current_strategy = ForagingStrategy.OPPORTUNISTIC
        elif context.current_season in ["spring", "summer"]:
            self.current_strategy = ForagingStrategy.EXPLORATORY
        else:
            self.current_strategy = ForagingStrategy.BALANCED
    
    def calculate_decision_probabilities(self, context: ForagingContext, 
                                       current_energy: float) -> Dict[ForagingDecisionType, float]:
        """Calculate probability weights for each decision type"""
        
        probs = {}
        
        # Base probabilities by strategy
        if self.current_strategy == ForagingStrategy.CONSERVATIVE:
            probs = {
                ForagingDecisionType.EXPLOIT_KNOWN: 0.6,
                ForagingDecisionType.EXPLORE_NEW: 0.1,
                ForagingDecisionType.FOLLOW_DANCE: 0.2,
                ForagingDecisionType.RETURN_HOME: 0.1,
                ForagingDecisionType.SWITCH_RESOURCE: 0.0,
                ForagingDecisionType.INVESTIGATE_SCENT: 0.0
            }
        elif self.current_strategy == ForagingStrategy.EXPLORATORY:
            probs = {
                ForagingDecisionType.EXPLOIT_KNOWN: 0.3,
                ForagingDecisionType.EXPLORE_NEW: 0.4,
                ForagingDecisionType.FOLLOW_DANCE: 0.1,
                ForagingDecisionType.RETURN_HOME: 0.05,
                ForagingDecisionType.SWITCH_RESOURCE: 0.1,
                ForagingDecisionType.INVESTIGATE_SCENT: 0.05
            }
        elif self.current_strategy == ForagingStrategy.OPPORTUNISTIC:
            probs = {
                ForagingDecisionType.EXPLOIT_KNOWN: 0.2,
                ForagingDecisionType.EXPLORE_NEW: 0.2,
                ForagingDecisionType.FOLLOW_DANCE: 0.4,
                ForagingDecisionType.RETURN_HOME: 0.05,
                ForagingDecisionType.SWITCH_RESOURCE: 0.1,
                ForagingDecisionType.INVESTIGATE_SCENT: 0.05
            }
        else:  # BALANCED
            probs = {
                ForagingDecisionType.EXPLOIT_KNOWN: 0.4,
                ForagingDecisionType.EXPLORE_NEW: 0.25,
                ForagingDecisionType.FOLLOW_DANCE: 0.2,
                ForagingDecisionType.RETURN_HOME: 0.05,
                ForagingDecisionType.SWITCH_RESOURCE: 0.05,
                ForagingDecisionType.INVESTIGATE_SCENT: 0.05
            }
        
        # Adjust based on success rates
        for decision_type, prob in probs.items():
            success_rate = self.success_rates.get(decision_type, 0.5)
            adjustment = (success_rate - 0.5) * 0.3  # ±15% adjustment
            probs[decision_type] = max(0.01, prob + adjustment)
        
        # Energy-based adjustments
        if current_energy < 30:
            probs[ForagingDecisionType.RETURN_HOME] *= 3.0
            probs[ForagingDecisionType.EXPLORE_NEW] *= 0.3
        
        # Weather adjustments
        if context.current_weather in ["heavy_rain", "thunderstorm"]:
            probs[ForagingDecisionType.RETURN_HOME] *= 2.0
            probs[ForagingDecisionType.EXPLORE_NEW] *= 0.5
        
        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            for decision_type in probs:
                probs[decision_type] /= total
        
        return probs
    
    def weighted_random_choice(self, probabilities: Dict[ForagingDecisionType, float]) -> ForagingDecisionType:
        """Make weighted random choice from probabilities"""
        rand_val = random.random()
        cumulative = 0.0
        
        for decision_type, prob in probabilities.items():
            cumulative += prob
            if rand_val <= cumulative:
                return decision_type
        
        # Fallback
        return ForagingDecisionType.EXPLOIT_KNOWN
    
    def execute_decision(self, decision_type: ForagingDecisionType, context: ForagingContext, 
                        available_patches: List[Any]) -> Optional[int]:
        """Execute specific foraging decision"""
        
        if decision_type == ForagingDecisionType.EXPLOIT_KNOWN:
            return self.select_best_known_patch(context)
        
        elif decision_type == ForagingDecisionType.EXPLORE_NEW:
            return self.select_exploration_patch(available_patches, context)
        
        elif decision_type == ForagingDecisionType.FOLLOW_DANCE:
            return self.follow_dance_information(context)
        
        elif decision_type == ForagingDecisionType.RETURN_HOME:
            return None  # Signal to return home
        
        elif decision_type == ForagingDecisionType.SWITCH_RESOURCE:
            return self.select_alternative_resource_patch(context)
        
        elif decision_type == ForagingDecisionType.INVESTIGATE_SCENT:
            return self.follow_scent_trail(context)
        
        return None
    
    def select_best_known_patch(self, context: ForagingContext) -> Optional[int]:
        """Select best known patch based on contextual memory"""
        
        if not self.foraging_memories:
            return None
        
        best_patch = None
        best_score = -1.0
        
        for patch_id, memory in self.foraging_memories.items():
            # Calculate contextual quality
            quality = memory.get_contextual_quality(
                context.current_season, context.current_hour, context.current_weather
            )
            
            # Calculate utility score
            score = self.calculate_patch_utility(memory, quality, context)
            
            if score > best_score:
                best_score = score
                best_patch = patch_id
        
        return best_patch
    
    def calculate_patch_utility(self, memory: ForagingMemory, contextual_quality: float, 
                              context: ForagingContext) -> float:
        """Calculate utility score for a patch"""
        
        # Base utility from quality
        utility = contextual_quality * self.quality_weight
        
        # Distance penalty
        distance_penalty = (memory.distance / 1000.0) * self.distance_weight  # Normalize by 1km
        utility -= distance_penalty
        
        # Memory strength bonus
        memory_bonus = memory.memory_strength * self.memory_weight
        utility += memory_bonus
        
        # Success rate bonus
        success_bonus = memory.success_rate * 0.2
        utility += success_bonus
        
        # Competition penalty
        competition_penalty = context.patch_competition.get(memory.patch_id, 0.0) * 0.1
        utility -= competition_penalty
        
        # Energy yield bonus
        energy_bonus = min(1.0, memory.energy_yield / 50.0) * 0.1  # Normalize by 50 units
        utility += energy_bonus
        
        return max(0.0, utility)
    
    def select_exploration_patch(self, available_patches: List[Any], context: ForagingContext) -> Optional[int]:
        """Select new patch for exploration using advanced patch selection"""
        
        if not available_patches:
            return None
        
        # Convert to PatchInfo objects for advanced selection
        patch_infos = self._convert_to_patch_info(available_patches)
        
        # Filter out known patches
        unknown_patches = [
            patch for patch in patch_infos 
            if patch.patch_id not in self.foraging_memories
        ]
        
        if not unknown_patches:
            # All patches are known, select least visited
            return self.select_least_visited_patch(available_patches)
        
        # Use advanced patch selector for exploration
        current_conditions = {
            "season": context.current_season,
            "hour": context.current_hour,
            "weather": context.current_weather
        }
        
        # Use exploratory strategy for unknown patches
        selected_patches = self.patch_selector.select_optimal_patches(
            unknown_patches,
            bee_species="apis_mellifera",  # Default species
            bee_energy=100.0,  # Default energy
            bee_memory=self.foraging_memories,
            current_conditions=current_conditions,
            strategy=PatchSelectionStrategy.MULTI_CRITERIA
        )
        
        return selected_patches[0].patch_id if selected_patches else None
    
    def select_least_visited_patch(self, available_patches: List[Any]) -> Optional[int]:
        """Select least visited patch among known patches"""
        
        least_visited = None
        min_visits = float('inf')
        
        for patch in available_patches:
            if patch.id in self.foraging_memories:
                visits = self.foraging_memories[patch.id].visits
                if visits < min_visits:
                    min_visits = visits
                    least_visited = patch.id
        
        return least_visited
    
    def follow_dance_information(self, context: ForagingContext) -> Optional[int]:
        """Select patch based on dance information"""
        
        if not context.dance_information:
            return None
        
        # Score dance information by quality and recency
        best_patch = None
        best_score = -1.0
        
        for dance_info in context.dance_information:
            # Extract dance information
            patch_id = getattr(dance_info, 'patch_id', None)
            quality = getattr(dance_info, 'quality', 0.5)
            urgency = getattr(dance_info, 'urgency', 1.0)
            
            if patch_id is None:
                continue
            
            # Score based on quality and urgency
            score = quality * urgency
            
            if score > best_score:
                best_score = score
                best_patch = patch_id
        
        return best_patch
    
    def select_alternative_resource_patch(self, context: ForagingContext) -> Optional[int]:
        """Select patch with different resource type"""
        
        # Get current resource preference
        current_resource = self.get_current_resource_preference()
        alternative_resource = "pollen" if current_resource == "nectar" else "nectar"
        
        # Find best patch with alternative resource
        best_patch = None
        best_score = -1.0
        
        for patch_id, memory in self.foraging_memories.items():
            if memory.resource_type == alternative_resource:
                quality = memory.get_contextual_quality(
                    context.current_season, context.current_hour, context.current_weather
                )
                score = self.calculate_patch_utility(memory, quality, context)
                
                if score > best_score:
                    best_score = score
                    best_patch = patch_id
        
        return best_patch
    
    def follow_scent_trail(self, context: ForagingContext) -> Optional[int]:
        """Follow scent trail to patch"""
        
        if not context.scent_trails:
            return None
        
        # Select strongest scent trail
        strongest_trail = max(context.scent_trails, key=lambda x: getattr(x, 'strength', 0.0))
        return getattr(strongest_trail, 'patch_id', None)
    
    def get_current_resource_preference(self) -> str:
        """Get current resource preference based on recent foraging"""
        
        recent_decisions = self.decision_history[-10:]  # Last 10 decisions
        nectar_count = sum(1 for d in recent_decisions if 'nectar' in str(d))
        pollen_count = sum(1 for d in recent_decisions if 'pollen' in str(d))
        
        return "nectar" if nectar_count >= pollen_count else "pollen"
    
    def update_foraging_memory(self, patch_id: int, success: bool, energy_gained: float, 
                             context: ForagingContext) -> None:
        """Update foraging memory after visit"""
        
        if patch_id not in self.foraging_memories:
            # Create new memory entry
            self.foraging_memories[patch_id] = ForagingMemory(
                patch_id=patch_id,
                quality=0.5,  # Default quality
                distance=100.0,  # Default distance
                resource_type="nectar",  # Default type
                visits=0,
                last_visit=context.current_hour,
                success_rate=0.0
            )
        
        memory = self.foraging_memories[patch_id]
        
        # Update visit statistics
        memory.visits += 1
        memory.last_visit = context.current_hour
        
        # Update success rate
        old_success = memory.success_rate * (memory.visits - 1)
        new_success = old_success + (1.0 if success else 0.0)
        memory.success_rate = new_success / memory.visits
        
        # Update energy yield
        old_yield = memory.energy_yield * (memory.visits - 1)
        new_yield = old_yield + energy_gained
        memory.energy_yield = new_yield / memory.visits
        
        # Update contextual quality
        self.update_contextual_quality(memory, context, success)
        
        # Manage memory capacity
        if len(self.foraging_memories) > self.memory_capacity:
            self.prune_weak_memories()
    
    def update_contextual_quality(self, memory: ForagingMemory, context: ForagingContext, 
                                success: bool) -> None:
        """Update contextual quality based on experience"""
        
        # Update seasonal quality
        if context.current_season not in memory.seasonal_quality:
            memory.seasonal_quality[context.current_season] = 1.0
        
        adjustment = 0.1 if success else -0.1
        memory.seasonal_quality[context.current_season] += adjustment
        memory.seasonal_quality[context.current_season] = max(0.1, min(2.0, 
            memory.seasonal_quality[context.current_season]))
        
        # Update time of day quality
        if context.current_hour not in memory.time_of_day_quality:
            memory.time_of_day_quality[context.current_hour] = 1.0
        
        memory.time_of_day_quality[context.current_hour] += adjustment
        memory.time_of_day_quality[context.current_hour] = max(0.1, min(2.0,
            memory.time_of_day_quality[context.current_hour]))
        
        # Update weather quality
        if context.current_weather not in memory.weather_quality:
            memory.weather_quality[context.current_weather] = 1.0
        
        memory.weather_quality[context.current_weather] += adjustment
        memory.weather_quality[context.current_weather] = max(0.1, min(2.0,
            memory.weather_quality[context.current_weather]))
    
    def prune_weak_memories(self) -> None:
        """Remove weakest memories to maintain capacity"""
        
        # Sort by memory strength and remove weakest
        sorted_memories = sorted(
            self.foraging_memories.items(),
            key=lambda x: x[1].memory_strength
        )
        
        # Remove bottom 10% or at least 1
        num_to_remove = max(1, len(sorted_memories) // 10)
        
        for i in range(num_to_remove):
            patch_id, _ = sorted_memories[i]
            del self.foraging_memories[patch_id]
    
    def update_decision_success_rate(self, decision_type: ForagingDecisionType, success: bool) -> None:
        """Update success rate for decision type"""
        
        if decision_type not in self.success_rates:
            self.success_rates[decision_type] = 0.5
        
        # Running average with decay
        alpha = 0.1  # Learning rate
        new_value = 1.0 if success else 0.0
        self.success_rates[decision_type] = (
            (1 - alpha) * self.success_rates[decision_type] + alpha * new_value
        )
    
    def get_foraging_summary(self) -> Dict[str, Any]:
        """Get comprehensive foraging behavior summary"""
        
        return {
            "strategy": self.current_strategy.value,
            "memory_count": len(self.foraging_memories),
            "decision_history": [d.value for d in self.decision_history[-10:]],
            "success_rates": {k.value: v for k, v in self.success_rates.items()},
            "memory_quality": {
                patch_id: memory.memory_strength 
                for patch_id, memory in self.foraging_memories.items()
            },
            "exploration_rate": self.exploration_probability,
            "average_success_rate": sum(self.success_rates.values()) / len(self.success_rates)
        }
    
    def _convert_to_patch_info(self, patches: List[Any]) -> List[PatchInfo]:
        """Convert generic patch objects to PatchInfo objects"""
        from .patch_selection import PatchInfo, ResourceType, PatchQualityMetric
        
        patch_infos = []
        
        for patch in patches:
            # Extract patch information with safe defaults
            patch_id = getattr(patch, 'id', getattr(patch, 'patch_id', 0))
            location = getattr(patch, 'location', (0.0, 0.0))
            distance = getattr(patch, 'distance', getattr(patch, 'distance_from_hive', 100.0))
            
            # Create quality metrics
            quality_metrics = {
                PatchQualityMetric.RESOURCE_DENSITY: getattr(patch, 'resource_density', 0.5),
                PatchQualityMetric.SUGAR_CONCENTRATION: getattr(patch, 'sugar_concentration', 0.3),
                PatchQualityMetric.ACCESSIBILITY: getattr(patch, 'accessibility', 0.8),
                PatchQualityMetric.COMPETITION_LEVEL: getattr(patch, 'competition_level', 0.2),
                PatchQualityMetric.HANDLING_TIME: getattr(patch, 'handling_time', 0.5)
            }
            
            # Create PatchInfo object
            patch_info = PatchInfo(
                patch_id=patch_id,
                location=location,
                resource_type=ResourceType.NECTAR,  # Default
                quality_metrics=quality_metrics,
                species_compatibility={"apis_mellifera": 0.8},  # Default compatibility
                distance_from_hive=distance,
                current_foragers=getattr(patch, 'current_foragers', 0),
                max_capacity=getattr(patch, 'max_capacity', 10),
                depletion_rate=getattr(patch, 'depletion_rate', 0.1),
                regeneration_rate=getattr(patch, 'regeneration_rate', 0.05),
                seasonal_availability={"spring": 1.0, "summer": 1.2, "autumn": 0.8, "winter": 0.2}
            )
            
            patch_infos.append(patch_info)
        
        return patch_infos


class ForagingTripManager(BaseModel):
    """Complete foraging trip lifecycle manager with detailed timing"""
    
    model_config = {"validate_assignment": True}
    
    # Flight parameters
    base_flight_speed: float = Field(default=6.5, ge=0.1, le=20.0, description="Base flight speed m/s")
    wind_effect_factor: float = Field(default=0.3, ge=0.0, le=1.0, description="Wind effect on flight speed")
    energy_per_meter_flight: float = Field(default=0.001, ge=0.0, le=0.01, description="Energy per meter of flight")
    
    # Foraging parameters
    base_handling_time: float = Field(default=8.0, ge=1.0, le=30.0, description="Base flower handling time (seconds)")
    inter_flower_time: float = Field(default=2.0, ge=0.1, le=10.0, description="Time between flowers (seconds)")
    energy_per_second_foraging: float = Field(default=0.05, ge=0.0, le=0.2, description="Energy per second foraging")
    
    # Resource parameters
    base_nectar_per_flower: float = Field(default=0.5, ge=0.1, le=2.0, description="Base nectar per flower (mg)")
    base_pollen_per_flower: float = Field(default=0.3, ge=0.1, le=1.0, description="Base pollen per flower (mg)")
    crop_capacity: float = Field(default=70.0, ge=10.0, le=150.0, description="Bee crop capacity (mg)")
    pollen_capacity: float = Field(default=20.0, ge=5.0, le=50.0, description="Pollen basket capacity (mg)")
    
    # Trip history
    completed_trips: List[ForagingTripLifecycle] = Field(default_factory=list, description="Completed trip history")
    
    def simulate_complete_foraging_trip(self, patch_id: int, context: ForagingContext,
                                      bee_energy: float, bee_physiology: Any = None) -> ForagingTripLifecycle:
        """Simulate complete foraging trip with detailed lifecycle"""
        
        # Get patch information
        patch_distance = context.patch_distances.get(patch_id, 100.0) if context.patch_distances else 100.0
        flower_density = context.flower_density.get(patch_id, 1.0) if context.flower_density else 1.0
        
        # Calculate travel time to patch
        travel_time_to_patch = self.calculate_travel_time(patch_distance, context.wind_speed, "to_patch")
        
        # Calculate foraging parameters
        flowers_available = max(1, int(flower_density * 50))  # Density affects available flowers
        handling_time = self.calculate_handling_time(context, bee_physiology)
        inter_flower_time = self.calculate_inter_flower_time(context, flower_density)
        
        # Simulate flower visitation
        trip_result = self.simulate_flower_visitation(
            flowers_available, handling_time, inter_flower_time, 
            context, bee_energy, patch_id, bee_physiology
        )
        
        # Calculate return travel time (may be different due to load)
        travel_time_to_hive = self.calculate_travel_time(
            patch_distance, context.wind_speed, "to_hive", 
            load_factor=trip_result['load_factor']
        )
        
        # Calculate energy consumption
        energy_consumed_travel = self.calculate_travel_energy(patch_distance, context.wind_speed, 
                                                           trip_result['load_factor'])
        energy_consumed_foraging = self.calculate_foraging_energy(trip_result['flowers_visited'], 
                                                                handling_time, inter_flower_time)
        
        # Create complete trip lifecycle
        total_duration = travel_time_to_patch + trip_result['foraging_time'] + travel_time_to_hive
        net_energy_gain = trip_result['energy_gained'] - energy_consumed_travel - energy_consumed_foraging
        
        trip_lifecycle = ForagingTripLifecycle(
            travel_time_to_patch=travel_time_to_patch,
            handling_time_per_flower=handling_time,
            inter_flower_time=inter_flower_time,
            flowers_visited=trip_result['flowers_visited'],
            travel_time_to_hive=travel_time_to_hive,
            total_trip_duration=total_duration,
            energy_consumed_travel=energy_consumed_travel,
            energy_consumed_foraging=energy_consumed_foraging,
            energy_gained=trip_result['energy_gained'],
            net_energy_gain=net_energy_gain
        )
        
        # Store trip for learning
        self.completed_trips.append(trip_lifecycle)
        if len(self.completed_trips) > 100:  # Keep last 100 trips
            self.completed_trips.pop(0)
        
        return trip_lifecycle
    
    def calculate_travel_time(self, distance: float, wind_speed: float, direction: str, 
                            load_factor: float = 0.0) -> float:
        """Calculate travel time considering wind and load"""
        
        # Base flight speed
        effective_speed = self.base_flight_speed
        
        # Wind effect (headwind slows down, tailwind speeds up)
        wind_effect = wind_speed * self.wind_effect_factor
        if direction == "to_patch":
            # Assume random wind direction, average effect
            effective_speed -= wind_effect * 0.5
        else:  # to_hive
            # Returning bees may have different wind conditions
            effective_speed -= wind_effect * 0.3
        
        # Load effect (carrying nectar/pollen slows flight)
        load_penalty = load_factor * 0.2  # 20% slower at full load
        effective_speed *= (1.0 - load_penalty)
        
        # Ensure minimum speed
        effective_speed = max(1.0, effective_speed)
        
        # Calculate time (distance in meters, speed in m/s)
        travel_time = distance / effective_speed
        
        return travel_time
    
    def calculate_handling_time(self, context: ForagingContext, bee_physiology: Any = None) -> float:
        """Calculate flower handling time based on conditions and physiology"""
        
        handling_time = self.base_handling_time
        
        # Temperature effect
        if context.temperature < 15:
            handling_time *= 1.3  # Slower in cold
        elif context.temperature > 30:
            handling_time *= 1.1  # Slightly slower in heat
        
        # Weather effect
        if context.current_weather in ["light_rain", "cloudy"]:
            handling_time *= 1.2
        elif context.current_weather in ["heavy_rain", "storm"]:
            handling_time *= 1.5
        
        # Enhanced bee physiology effects
        if bee_physiology:
            # Proboscis-corolla matching efficiency
            proboscis_length = getattr(bee_physiology, 'proboscis_length_mm', 6.5)
            corolla_depth = context.metadata.get('average_corolla_depth', 6.0) if hasattr(context, 'metadata') else 6.0
            
            # Calculate proboscis-corolla match efficiency
            match_efficiency = self.calculate_proboscis_corolla_match(proboscis_length, corolla_depth)
            handling_time *= (2.0 - match_efficiency)  # Better match = faster handling
            
            # Glossa length affects nectar extraction speed
            glossa_length = getattr(bee_physiology, 'glossaLength_mm', 4.0)
            if glossa_length > 4.5:
                handling_time *= 0.9  # Longer glossa = faster extraction
            elif glossa_length < 3.5:
                handling_time *= 1.1  # Shorter glossa = slower extraction
            
            # Body weight affects handling agility
            weight_mg = getattr(bee_physiology, 'weight_mg', 100.0)
            if weight_mg > 120:
                handling_time *= 1.05  # Heavier bees slightly slower
            elif weight_mg < 80:
                handling_time *= 1.02  # Very light bees may struggle
        
        return handling_time
    
    def calculate_proboscis_corolla_match(self, proboscis_length: float, corolla_depth: float) -> float:
        """Calculate proboscis-corolla matching efficiency (0-1)"""
        
        # Optimal match when proboscis is slightly longer than corolla
        optimal_ratio = 1.1  # 10% longer than corolla depth
        actual_ratio = proboscis_length / max(0.1, corolla_depth)
        
        if actual_ratio < 1.0:
            # Proboscis too short - cannot reach nectar efficiently
            efficiency = actual_ratio * 0.8  # Maximum 80% efficiency if too short
        elif actual_ratio <= optimal_ratio:
            # Good match - linear efficiency
            efficiency = 0.8 + 0.2 * (actual_ratio - 1.0) / (optimal_ratio - 1.0)
        else:
            # Proboscis too long - diminishing returns
            excess = actual_ratio - optimal_ratio
            efficiency = max(0.6, 1.0 - excess * 0.1)  # Decrease efficiency with excess length
        
        return min(1.0, max(0.1, efficiency))
    
    def calculate_resource_extraction_efficiency(self, bee_physiology: Any, context: ForagingContext) -> float:
        """Calculate resource extraction efficiency based on bee physiology"""
        
        # Base efficiency
        efficiency = 1.0
        
        # Proboscis-corolla matching efficiency
        proboscis_length = getattr(bee_physiology, 'proboscis_length_mm', 6.5)
        corolla_depth = context.metadata.get('average_corolla_depth', 6.0) if hasattr(context, 'metadata') else 6.0
        match_efficiency = self.calculate_proboscis_corolla_match(proboscis_length, corolla_depth)
        efficiency *= match_efficiency
        
        # Glossa length affects nectar extraction capacity
        glossa_length = getattr(bee_physiology, 'glossaLength_mm', 4.0)
        if glossa_length > 4.5:
            efficiency *= 1.1  # Longer glossa extracts more nectar
        elif glossa_length < 3.5:
            efficiency *= 0.9  # Shorter glossa extracts less
        
        # Mandible width affects pollen collection
        mandible_width = getattr(bee_physiology, 'mandible_width_mm', 1.0)
        if mandible_width > 1.2:
            efficiency *= 1.05  # Wider mandibles better for pollen
        elif mandible_width < 0.8:
            efficiency *= 0.95  # Narrower mandibles less effective
        
        # Body weight affects stability and reach
        weight_mg = getattr(bee_physiology, 'weight_mg', 100.0)
        if 90 <= weight_mg <= 110:
            efficiency *= 1.02  # Optimal weight range
        elif weight_mg > 130:
            efficiency *= 0.95  # Too heavy - less agile
        elif weight_mg < 70:
            efficiency *= 0.93  # Too light - less effective manipulation
        
        return min(1.3, max(0.5, efficiency))  # Cap efficiency between 50% and 130%
    
    def can_access_flower(self, bee_physiology: Any, flower_morphology: Dict[str, float]) -> bool:
        """Validate if bee can physically access flower resources"""
        
        if not bee_physiology:
            return True  # Default access
        
        # Proboscis length vs corolla depth
        proboscis_length = getattr(bee_physiology, 'proboscis_length_mm', 6.5)
        corolla_depth = flower_morphology.get('corolla_depth_mm', 6.0)
        
        # Bee cannot access if proboscis is significantly shorter than corolla
        if proboscis_length < corolla_depth * 0.8:
            return False
        
        # Mandible width vs flower opening
        mandible_width = getattr(bee_physiology, 'mandible_width_mm', 1.0)
        flower_opening = flower_morphology.get('opening_diameter_mm', 3.0)
        
        # Bee cannot access if mandible is too wide for flower opening
        if mandible_width > flower_opening * 1.2:
            return False
        
        # Body size constraints
        weight_mg = getattr(bee_physiology, 'weight_mg', 100.0)
        min_flower_size = flower_morphology.get('minimum_bee_size_mg', 50.0)
        max_flower_size = flower_morphology.get('maximum_bee_size_mg', 200.0)
        
        # Check if bee size is within flower tolerance
        if weight_mg < min_flower_size or weight_mg > max_flower_size:
            return False
        
        return True
    
    def calculate_inter_flower_time(self, context: ForagingContext, flower_density: float) -> float:
        """Calculate time to move between flowers"""
        
        base_time = self.inter_flower_time
        
        # Density effect - more flowers means less travel between them
        density_factor = max(0.5, min(2.0, 1.0 / flower_density))
        base_time *= density_factor
        
        # Weather effect on movement
        if context.current_weather in ["windy", "light_rain"]:
            base_time *= 1.2
        elif context.current_weather in ["heavy_rain", "storm"]:
            base_time *= 1.8
        
        return base_time
    
    def simulate_flower_visitation(self, flowers_available: int, handling_time: float, 
                                 inter_flower_time: float, context: ForagingContext,
                                 bee_energy: float, patch_id: int, bee_physiology: Any = None) -> Dict[str, Any]:
        """Simulate the flower visitation process"""
        
        flowers_visited = 0
        total_foraging_time = 0.0
        nectar_collected = 0.0
        pollen_collected = 0.0
        energy_gained = 0.0
        
        # Get physiological capacity limits
        crop_capacity = self.crop_capacity
        pollen_capacity = self.pollen_capacity
        
        if bee_physiology:
            # Use actual crop volume from physiology data
            crop_capacity = getattr(bee_physiology, 'cropvolume_myl', 70.0)
            # Adjust pollen capacity based on body size
            weight_mg = getattr(bee_physiology, 'weight_mg', 100.0)
            pollen_capacity = self.pollen_capacity * (weight_mg / 100.0)  # Scale with weight
        
        # Determine resource collection efficiency
        nectar_per_flower = self.base_nectar_per_flower
        pollen_per_flower = self.base_pollen_per_flower
        
        # Physiological extraction efficiency
        if bee_physiology:
            extraction_efficiency = self.calculate_resource_extraction_efficiency(bee_physiology, context)
            nectar_per_flower *= extraction_efficiency
            pollen_per_flower *= extraction_efficiency
        
        # Seasonal adjustments
        if context.current_season == "spring":
            nectar_per_flower *= 1.2
            pollen_per_flower *= 1.4
        elif context.current_season == "summer":
            nectar_per_flower *= 1.0
            pollen_per_flower *= 1.0
        elif context.current_season == "autumn":
            nectar_per_flower *= 0.8
            pollen_per_flower *= 0.6
        else:  # winter
            nectar_per_flower *= 0.3
            pollen_per_flower *= 0.2
        
        # Time of day adjustments
        if 6 <= context.current_hour <= 9:  # Morning peak
            nectar_per_flower *= 1.1
        elif 17 <= context.current_hour <= 19:  # Evening peak
            nectar_per_flower *= 1.05
        elif context.current_hour < 6 or context.current_hour > 20:  # Night
            nectar_per_flower *= 0.3
        
        # Visit flowers until capacity full or flowers exhausted
        for flower_num in range(flowers_available):
            # Check if bee has capacity using physiological limits
            if nectar_collected >= crop_capacity and pollen_collected >= pollen_capacity:
                break
            
            # Check energy levels
            energy_cost = self.energy_per_second_foraging * (handling_time + inter_flower_time)
            if bee_energy - energy_cost < 10:  # Reserve energy for return
                break
            
            # Visit flower
            flowers_visited += 1
            total_foraging_time += handling_time
            
            # Collect resources (probabilistic) using physiological capacity limits
            if nectar_collected < crop_capacity and random.random() < 0.8:  # 80% nectar success
                nectar_this_flower = min(nectar_per_flower, crop_capacity - nectar_collected)
                nectar_collected += nectar_this_flower
                energy_gained += nectar_this_flower * 0.5  # Energy content
            
            if pollen_collected < pollen_capacity and random.random() < 0.6:  # 60% pollen success
                pollen_this_flower = min(pollen_per_flower, pollen_capacity - pollen_collected)
                pollen_collected += pollen_this_flower
                energy_gained += pollen_this_flower * 0.3  # Energy content
            
            # Add inter-flower travel time (except for last flower)
            if flower_num < flowers_available - 1:
                total_foraging_time += inter_flower_time
        
        # Calculate load factor for return flight using physiological capacities
        nectar_load_factor = nectar_collected / crop_capacity
        pollen_load_factor = pollen_collected / pollen_capacity
        load_factor = (nectar_load_factor + pollen_load_factor) / 2
        
        return {
            'flowers_visited': flowers_visited,
            'foraging_time': total_foraging_time,
            'nectar_collected': nectar_collected,
            'pollen_collected': pollen_collected,
            'energy_gained': energy_gained,
            'load_factor': load_factor
        }
    
    def calculate_travel_energy(self, distance: float, wind_speed: float, load_factor: float) -> float:
        """Calculate energy consumed during travel"""
        
        # Base energy consumption
        base_energy = distance * self.energy_per_meter_flight
        
        # Wind resistance effect
        wind_resistance = 1.0 + (wind_speed * 0.1)  # Higher wind = more energy
        
        # Load effect
        load_effect = 1.0 + (load_factor * 0.3)  # Carrying load costs more energy
        
        total_energy = base_energy * wind_resistance * load_effect * 2  # Round trip
        
        return total_energy
    
    def calculate_foraging_energy(self, flowers_visited: int, handling_time: float, 
                                inter_flower_time: float) -> float:
        """Calculate energy consumed during foraging"""
        
        # Energy for flower handling
        handling_energy = flowers_visited * handling_time * self.energy_per_second_foraging
        
        # Energy for inter-flower movement
        movement_energy = (flowers_visited - 1) * inter_flower_time * self.energy_per_second_foraging
        
        return handling_energy + movement_energy
    
    def get_trip_efficiency_metrics(self) -> Dict[str, float]:
        """Get efficiency metrics from completed trips"""
        
        if not self.completed_trips:
            return {}
        
        # Calculate averages
        avg_trip_duration = sum(trip.total_trip_duration for trip in self.completed_trips) / len(self.completed_trips)
        avg_flowers_visited = sum(trip.flowers_visited for trip in self.completed_trips) / len(self.completed_trips)
        avg_net_energy_gain = sum(trip.net_energy_gain for trip in self.completed_trips) / len(self.completed_trips)
        avg_energy_efficiency = sum(trip.energy_gained / max(1, trip.energy_consumed_travel + trip.energy_consumed_foraging) 
                                  for trip in self.completed_trips) / len(self.completed_trips)
        
        # Calculate success rate (positive energy gain)
        successful_trips = sum(1 for trip in self.completed_trips if trip.net_energy_gain > 0)
        success_rate = successful_trips / len(self.completed_trips)
        
        return {
            "average_trip_duration": avg_trip_duration,
            "average_flowers_visited": avg_flowers_visited,
            "average_net_energy_gain": avg_net_energy_gain,
            "average_energy_efficiency": avg_energy_efficiency,
            "success_rate": success_rate,
            "total_trips": len(self.completed_trips)
        }
    
    def optimize_foraging_parameters(self) -> None:
        """Optimize foraging parameters based on trip history"""
        
        if len(self.completed_trips) < 10:
            return  # Need enough data
        
        # Analyze successful vs unsuccessful trips
        successful_trips = [trip for trip in self.completed_trips if trip.net_energy_gain > 0]
        unsuccessful_trips = [trip for trip in self.completed_trips if trip.net_energy_gain <= 0]
        
        if not successful_trips:
            return
        
        # Learn optimal flower visit patterns
        avg_successful_flowers = sum(trip.flowers_visited for trip in successful_trips) / len(successful_trips)
        avg_unsuccessful_flowers = sum(trip.flowers_visited for trip in unsuccessful_trips) / len(unsuccessful_trips) if unsuccessful_trips else 0
        
        # Adjust handling time based on efficiency
        if avg_successful_flowers > avg_unsuccessful_flowers:
            # More flowers visited in successful trips, maybe we can handle faster
            self.base_handling_time *= 0.98
        else:
            # Fewer flowers but more successful, maybe need more careful handling
            self.base_handling_time *= 1.02
        
        # Keep handling time within reasonable bounds
        self.base_handling_time = max(3.0, min(15.0, self.base_handling_time))
    
    def predict_trip_outcome(self, patch_id: int, context: ForagingContext, 
                           bee_energy: float) -> Dict[str, float]:
        """Predict trip outcome without executing it"""
        
        patch_distance = context.patch_distances.get(patch_id, 100.0) if context.patch_distances else 100.0
        flower_density = context.flower_density.get(patch_id, 1.0) if context.flower_density else 1.0
        
        # Quick estimates
        travel_time = self.calculate_travel_time(patch_distance, context.wind_speed, "to_patch") * 2
        estimated_flowers = min(20, int(flower_density * 10))  # Conservative estimate
        foraging_time = estimated_flowers * (self.base_handling_time + self.inter_flower_time)
        
        # Energy estimates
        travel_energy = self.calculate_travel_energy(patch_distance, context.wind_speed, 0.5)
        foraging_energy = self.calculate_foraging_energy(estimated_flowers, self.base_handling_time, self.inter_flower_time)
        estimated_energy_gain = estimated_flowers * self.base_nectar_per_flower * 0.5
        
        return {
            "predicted_duration": travel_time + foraging_time,
            "predicted_flowers": estimated_flowers,
            "predicted_energy_gain": estimated_energy_gain - travel_energy - foraging_energy,
            "predicted_efficiency": estimated_energy_gain / max(1, travel_energy + foraging_energy),
            "risk_level": self.calculate_trip_risk(patch_distance, context, bee_energy)
        }
    
    def calculate_trip_risk(self, distance: float, context: ForagingContext, bee_energy: float) -> float:
        """Calculate risk level of a foraging trip"""
        
        risk = 0.0
        
        # Distance risk
        risk += min(0.3, distance / 1000.0)  # Higher risk for distant patches
        
        # Energy risk
        if bee_energy < 40:
            risk += 0.3
        elif bee_energy < 60:
            risk += 0.1
        
        # Weather risk
        if context.current_weather in ["heavy_rain", "storm"]:
            risk += 0.4
        elif context.current_weather in ["light_rain", "windy"]:
            risk += 0.2
        
        # Time of day risk
        if context.current_hour < 6 or context.current_hour > 20:
            risk += 0.3
        
        # Wind risk
        if context.wind_speed > 5.0:
            risk += min(0.2, context.wind_speed / 25.0)
        
        return min(1.0, risk)