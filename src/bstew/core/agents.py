"""
Agent-based modeling framework for BSTEW
========================================

Mesa-based agent classes representing bees in the colony simulation.
Implements the NetLogo breed system using Python class hierarchy.
"""

import mesa
from typing import List, Optional, Tuple, Any, Dict
from .enums import BeeStatus, BeeRole
from pydantic import BaseModel, Field
import math

from ..components.genetics import Genotype, SpermathecaManager, Sex, Ploidy
from ..components.mortality_tracking import DetailedDeathCause
from .activity_state_machine import ActivityStateMachine, PersonalTimeTracker


# BeeRole and BeeStatus are now imported from enums module


class ForagingMemory(BaseModel):
    """Memory structure for foraging experiences - NetLogo enhanced"""

    model_config = {"validate_assignment": True}

    patch_id: int = Field(description="Unique identifier for the patch")
    quality: float = Field(ge=0.0, le=1.0, description="Resource quality (0-1)")
    distance: float = Field(ge=0.0, description="Distance to patch in meters")
    resource_type: str = Field(description="Type of resource (nectar/pollen)")
    visits: int = Field(default=0, ge=0, description="Number of visits to this patch")
    last_visit: int = Field(default=0, ge=0, description="Time step of last visit")
    success_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Success rate of foraging (0-1)"
    )

    # Enhanced memory attributes
    patch_color: str = Field(default="green", description="Visual landmark color")
    seasonal_quality: Dict[str, float] = Field(
        default_factory=dict, description="Quality by season"
    )
    time_of_day_quality: Dict[int, float] = Field(
        default_factory=dict, description="Quality by hour"
    )
    weather_quality: Dict[str, float] = Field(
        default_factory=dict, description="Quality by weather"
    )
    energy_yield: float = Field(
        default=0.0, ge=0.0, description="Average energy yield per visit"
    )
    travel_time: float = Field(default=0.0, ge=0.0, description="Travel time to patch")
    competition_level: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Competition intensity"
    )
    memory_strength: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Memory decay factor"
    )

    def update_memory_strength(self, days_since_visit: int) -> None:
        """Update memory strength based on time decay"""
        decay_rate = 0.95  # 5% decay per day
        self.memory_strength = max(
            0.1, self.memory_strength * (decay_rate**days_since_visit)
        )

    def get_contextual_quality(self, season: str, hour: int, weather: str) -> float:
        """Get quality adjusted for current context"""
        base_quality = self.quality

        # Seasonal adjustment
        if season in self.seasonal_quality:
            base_quality *= self.seasonal_quality[season]

        # Time of day adjustment
        if hour in self.time_of_day_quality:
            base_quality *= self.time_of_day_quality[hour]

        # Weather adjustment
        if weather in self.weather_quality:
            base_quality *= self.weather_quality[weather]

        # Memory strength adjustment
        base_quality *= self.memory_strength

        return min(1.0, base_quality)


class DanceInfo(BaseModel):
    """Information communicated through waggle dance"""

    model_config = {"validate_assignment": True}

    patch_id: int = Field(description="Unique identifier for the patch")
    distance: float = Field(ge=0.0, description="Distance to patch in meters")
    direction: float = Field(
        ge=0.0, le=2 * 3.14159, description="Direction in radians (0-2π)"
    )
    quality: float = Field(ge=0.0, le=1.0, description="Resource quality (0-1)")
    resource_type: str = Field(description="Type of resource (nectar/pollen)")
    urgency: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Dance urgency factor (0-1)"
    )


class BeeAgent(mesa.Agent):
    """
    Base class for all bee agents in the simulation.

    Implements common bee behaviors and state management.
    Maps to NetLogo turtle/breed system.
    """

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        age: int = 0,
        genotype: Optional[Genotype] = None,
    ):
        super().__init__(model)
        self.unique_id = unique_id
        self.age = age
        self.energy = 100.0
        self.max_energy = 100.0
        self.location: Optional[Tuple[float, float]] = None
        self.status = BeeStatus.ALIVE
        self.role = BeeRole.NURSE
        self.colony: Optional[Any] = None
        self.disease_load = 0.0
        self.last_active_step = 0

        # Genetic system integration
        self.genotype = genotype
        if self.genotype:
            # Set role based on genetic sex
            if self.genotype.sex == Sex.MALE:
                self.role = BeeRole.DRONE
            elif self.genotype.sex == Sex.FEMALE:
                self.role = BeeRole.NURSE  # Default, can change to forager
            elif self.genotype.sex == Sex.DIPLOID_MALE:
                self.role = BeeRole.DRONE
                self.status = BeeStatus.DEAD  # Diploid males die

        # Genetic/individual traits
        self.foraging_efficiency = self.model.random.gauss(1.0, 0.1)
        self.longevity_factor = self.model.random.gauss(1.0, 0.05)
        self.disease_resistance = self.model.random.gauss(1.0, 0.1)

        # Genetic influence on traits
        if self.genotype:
            self._apply_genetic_effects()

        # Development history (set by colony when bee emerges)
        self.development_history: Optional[Dict[str, Any]] = None

        # Physiological parameters
        self.cropvolume_myl: float = 0.0  # Crop volume in microliters
        self.glossaLength_mm: float = 0.0  # Glossa/tongue length in millimeters
        self.weight_mg: float = 0.0  # Body weight in milligrams

        # Proboscis characteristics for corolla matching
        self.proboscis_length_mm: float = 0.0  # Total proboscis length
        self.proboscis_diameter_mm: float = 0.0  # Proboscis diameter
        self.mandible_width_mm: float = 0.0  # Mandible width

        # Initialize physiological parameters based on role and species
        self._initialize_physiological_parameters()

        # Activity state machine integration
        self.activity_state_machine: Optional[ActivityStateMachine] = None
        self.personal_time_tracker: Optional[PersonalTimeTracker] = None

        # Enhanced foraging system integration
        self.foraging_decision_engine: Optional[Any] = None
        self.foraging_memories: Dict[int, ForagingMemory] = {}
        self.current_foraging_target: Optional[int] = None

        # Resource collection integration
        self.resource_collector: Optional[Any] = None
        self.bee_physiology: Optional[Any] = None

        # Data collection integration
        self.data_collector: Optional[Any] = None

        # Enhanced personal time tracking integration
        self.recent_dance_followers: int = 0

        # Initialize personal time tracking if activity state machine available
        if hasattr(model, "activity_state_machine") and model.activity_state_machine:
            self.activity_state_machine = model.activity_state_machine
            self.initialize_personal_time_tracking()
        else:
            self.activity_state_machine = None

    def _apply_genetic_effects(self) -> None:
        """Apply genetic effects to individual traits"""
        if not self.genotype:
            return

        # Diploid males have reduced viability
        if self.genotype.is_diploid_male():
            self.longevity_factor *= 0.1  # Severely reduced lifespan
            self.energy *= 0.5  # Lower energy
            self.disease_resistance *= 0.3  # Poor disease resistance

        # Inbreeding effects (simplified)
        if self.genotype.ploidy == Ploidy.DIPLOID:
            allele_ids = self.genotype.get_allele_ids()
            if len(set(allele_ids)) < len(allele_ids):  # Some alleles are identical
                inbreeding_depression = 0.9  # 10% reduction in fitness
                self.foraging_efficiency *= inbreeding_depression
                self.longevity_factor *= inbreeding_depression
                self.disease_resistance *= inbreeding_depression

    def _initialize_physiological_parameters(self) -> None:
        """Initialize physiological parameters based on bee role and species"""

        # Helper function to safely get random values
        def safe_gauss(mean: float, std: float) -> float:
            try:
                result = self.model.random.gauss(mean, std)
                # Check if result is a Mock object or not a number
                if hasattr(result, "_mock_name") or not isinstance(
                    result, (int, float)
                ):
                    return mean  # Fallback to mean for testing
                return float(result)
            except (AttributeError, TypeError):
                return mean  # Fallback to mean if random method doesn't work

        # Base values for Bombus terrestris (most common species in BEE-STEWARD)
        if self.role == BeeRole.QUEEN:
            # Queen parameters - larger and more capable
            self.weight_mg = safe_gauss(800.0, 50.0)  # 700-900mg
            self.glossaLength_mm = safe_gauss(18.0, 1.0)  # 16-20mm
            self.cropvolume_myl = safe_gauss(200.0, 20.0)  # 160-240μl
            self.proboscis_length_mm = (
                self.glossaLength_mm + 2.0
            )  # Include other mouthparts
            self.proboscis_diameter_mm = safe_gauss(0.8, 0.1)  # 0.6-1.0mm
            self.mandible_width_mm = safe_gauss(3.5, 0.2)  # 3.1-3.9mm

        elif self.role == BeeRole.NURSE:
            # Worker nurse parameters - medium sized
            self.weight_mg = safe_gauss(180.0, 20.0)  # 140-220mg
            self.glossaLength_mm = safe_gauss(12.0, 1.0)  # 10-14mm
            self.cropvolume_myl = safe_gauss(40.0, 5.0)  # 30-50μl
            self.proboscis_length_mm = self.glossaLength_mm + 1.5
            self.proboscis_diameter_mm = safe_gauss(0.5, 0.05)  # 0.4-0.6mm
            self.mandible_width_mm = safe_gauss(2.2, 0.1)  # 2.0-2.4mm

        elif self.role == BeeRole.FORAGER:
            # Forager parameters - optimized for foraging
            self.weight_mg = safe_gauss(160.0, 15.0)  # 130-190mg
            self.glossaLength_mm = safe_gauss(
                13.5, 1.2
            )  # 11-16mm (longer for better access)
            self.cropvolume_myl = safe_gauss(45.0, 8.0)  # 29-61μl (larger for carrying)
            self.proboscis_length_mm = self.glossaLength_mm + 1.8
            self.proboscis_diameter_mm = safe_gauss(0.52, 0.08)  # 0.36-0.68mm
            self.mandible_width_mm = safe_gauss(2.1, 0.15)  # 1.8-2.4mm

        elif self.role == BeeRole.DRONE:
            # Drone parameters - males, different morphology
            self.weight_mg = safe_gauss(220.0, 25.0)  # 170-270mg
            self.glossaLength_mm = safe_gauss(
                8.0, 0.8
            )  # 6.4-9.6mm (shorter, less foraging)
            self.cropvolume_myl = safe_gauss(
                25.0, 5.0
            )  # 15-35μl (smaller, less storage needed)
            self.proboscis_length_mm = self.glossaLength_mm + 1.0
            self.proboscis_diameter_mm = safe_gauss(0.45, 0.05)  # 0.35-0.55mm
            self.mandible_width_mm = safe_gauss(2.8, 0.2)  # 2.4-3.2mm (larger head)

        else:
            # Default/unknown role parameters
            self.weight_mg = safe_gauss(170.0, 20.0)
            self.glossaLength_mm = safe_gauss(11.0, 1.0)
            self.cropvolume_myl = safe_gauss(35.0, 5.0)
            self.proboscis_length_mm = self.glossaLength_mm + 1.5
            self.proboscis_diameter_mm = safe_gauss(0.5, 0.05)
            self.mandible_width_mm = safe_gauss(2.2, 0.2)

        # Apply genetic variation if genotype is available
        if self.genotype:
            self._apply_genetic_physiological_effects()

        # Ensure all values are positive and within reasonable bounds
        self.weight_mg = max(10.0, self.weight_mg)  # Minimum 10mg
        self.glossaLength_mm = max(2.0, self.glossaLength_mm)  # Minimum 2mm
        self.cropvolume_myl = max(5.0, self.cropvolume_myl)  # Minimum 5μl
        self.proboscis_length_mm = max(3.0, self.proboscis_length_mm)  # Minimum 3mm
        self.proboscis_diameter_mm = max(
            0.1, self.proboscis_diameter_mm
        )  # Minimum 0.1mm
        self.mandible_width_mm = max(0.5, self.mandible_width_mm)  # Minimum 0.5mm

    def _apply_genetic_physiological_effects(self) -> None:
        """Apply genetic effects to physiological parameters"""
        if not self.genotype:
            return

        # Helper function to safely get random values
        def safe_gauss(mean: float, std: float) -> float:
            try:
                result = self.model.random.gauss(mean, std)
                if hasattr(result, "_mock_name") or not isinstance(
                    result, (int, float)
                ):
                    return mean
                return float(result)
            except (AttributeError, TypeError):
                return mean

        # Generate genetic variation factors (±10% for most traits)
        size_factor = safe_gauss(1.0, 0.1)  # Overall size variation
        tongue_factor = safe_gauss(1.0, 0.15)  # Tongue length has higher variation

        # Apply size effects
        self.weight_mg *= size_factor
        self.mandible_width_mm *= size_factor
        self.proboscis_diameter_mm *= size_factor

        # Apply specialized tongue length genetics
        self.glossaLength_mm *= tongue_factor
        self.proboscis_length_mm *= tongue_factor

        # Crop volume scales with size but has independent variation
        crop_factor = safe_gauss(1.0, 0.12)
        self.cropvolume_myl *= (
            size_factor * 0.7 + crop_factor * 0.3
        )  # Weighted combination

        # Diploid males have altered physiology
        if self.genotype.is_diploid_male():
            self.weight_mg *= 0.8  # Reduced size
            self.glossaLength_mm *= 0.9  # Slightly shorter tongue
            self.cropvolume_myl *= 0.7  # Reduced crop capacity

    def calculate_state_specific_energy_consumption(self) -> float:
        """Calculate energy consumption based on current activity state and physiology"""
        base_metabolic_rate = (
            float(self.weight_mg) * 0.01
        )  # Base rate proportional to weight

        # Get current activity state
        current_state = getattr(self, "status", BeeStatus.ALIVE)

        # State-specific energy multipliers
        state_multipliers = {
            BeeStatus.HIBERNATING: 0.1,  # Very low metabolism
            BeeStatus.RESTING: 0.5,  # Resting metabolism
            BeeStatus.NURSING: 0.8,  # Active nursing work
            BeeStatus.NEST_CONSTRUCTION: 0.9,  # Physical work
            BeeStatus.FORAGING: 1.5,  # High energy flying and searching
            BeeStatus.SEARCHING: 1.3,  # Active searching
            BeeStatus.NECTAR_FORAGING: 1.8,  # Intensive foraging
            BeeStatus.POLLEN_FORAGING: 2.0,  # Most energy intensive
            BeeStatus.COLLECT_NECTAR: 1.6,  # Nectar collection
            BeeStatus.COLLECT_POLLEN: 1.8,  # Pollen collection
            BeeStatus.BRINGING_NECTAR: 1.4,  # Flying with nectar load
            BeeStatus.BRINGING_POLLEN: 1.5,  # Flying with pollen load
            BeeStatus.RETURNING_EMPTY: 1.2,  # Flying back empty
            BeeStatus.RETURNING_UNHAPPY_NECTAR: 1.3,  # Unsuccessful return
            BeeStatus.RETURNING_UNHAPPY_POLLEN: 1.3,  # Unsuccessful return
            BeeStatus.DANCING: 1.2,  # Communication behavior
            BeeStatus.EGG_LAYING: 0.9,  # Queen egg laying activity
            BeeStatus.EXPERIMENTAL_FORAGING_NECTAR: 1.7,  # Experimental foraging
            BeeStatus.EXPERIMENTAL_FORAGING_POLLEN: 1.9,  # Experimental foraging
        }

        multiplier = float(
            state_multipliers.get(current_state, 0.6)
        )  # Default moderate activity

        # Additional factors based on physiology
        flight_cost_factor = (
            float(self.weight_mg) / 160.0
        ) ** 0.75  # Allometric scaling for flight
        foraging_efficiency_factor = 1.0 / max(
            0.1, float(self.foraging_efficiency)
        )  # Less efficient = more energy

        # Calculate total energy consumption
        energy_consumption = (
            base_metabolic_rate
            * multiplier
            * flight_cost_factor
            * foraging_efficiency_factor
        )

        return float(energy_consumption)

    def calculate_foraging_capacity(self) -> Dict[str, float]:
        """Calculate foraging capacity based on physiological parameters"""
        return {
            "max_nectar_load_myl": self.cropvolume_myl
            * 0.8,  # 80% of crop volume usable
            "max_pollen_load_mg": self.weight_mg * 0.15,  # 15% of body weight in pollen
            "flight_range_m": (self.weight_mg / 200.0) ** 0.5
            * 1000,  # Flight range based on size
            "handling_time_factor": 1.0
            / (self.glossaLength_mm / 12.0),  # Longer tongue = faster handling
        }

    def can_access_flower(
        self, corolla_depth_mm: float, corolla_width_mm: float
    ) -> bool:
        """Check if bee can access a flower based on proboscis-corolla matching"""
        # Check if proboscis is long enough to reach nectar
        can_reach = (
            self.proboscis_length_mm >= corolla_depth_mm * 0.9
        )  # 90% reach required

        # Check if proboscis fits in corolla opening
        can_fit = (
            self.proboscis_diameter_mm <= corolla_width_mm * 0.8
        )  # 80% fit tolerance

        # Check if mandibles can handle flower (for pollen collection)
        can_handle = (
            self.mandible_width_mm <= corolla_width_mm + 2.0
        )  # Some overhang allowed

        return can_reach and can_fit and can_handle

    def calculate_resource_extraction_efficiency(
        self, flower_type: Dict[str, float]
    ) -> float:
        """Calculate efficiency of resource extraction from specific flower type"""
        corolla_depth = flower_type.get("corolla_depth_mm", 10.0)
        corolla_width = flower_type.get("corolla_width_mm", 5.0)
        nectar_concentration = flower_type.get("nectar_concentration", 0.3)

        # Base efficiency from proboscis-corolla match
        depth_match = min(1.0, self.proboscis_length_mm / corolla_depth)
        width_match = min(1.0, corolla_width / self.proboscis_diameter_mm)

        # Efficiency decreases if mismatch is severe
        if depth_match < 0.5:  # Can't reach deep enough
            return 0.1
        if width_match < 1.5:  # Proboscis too thick
            return 0.1

        # Calculate extraction efficiency
        base_efficiency = (depth_match * 0.6 + width_match * 0.4) * 0.8

        # Tongue length affects extraction speed and completeness
        tongue_efficiency = min(1.2, self.glossaLength_mm / corolla_depth)

        # Nectar concentration affects extraction difficulty
        concentration_factor = 0.5 + (
            nectar_concentration * 1.5
        )  # Easier to extract concentrated nectar

        total_efficiency = base_efficiency * tongue_efficiency * concentration_factor

        return min(1.0, max(0.0, total_efficiency))

    def step(self) -> None:
        """Execute one simulation step for this agent"""
        self.age += 1
        self.update_energy()
        self.check_mortality()

        if self.status == BeeStatus.ALIVE:
            self.update_activity_state()
            self.execute_role_behavior()

            # Collect individual bee data
            if self.data_collector:
                self.data_collector.collect_bee_data(self, self.model.schedule.steps)

    def update_activity_state(self) -> None:
        """Update activity state using enhanced state machine with personal time tracking"""
        if self.activity_state_machine and hasattr(self.model, "schedule"):
            current_step = self.model.schedule.steps

            # Get personal time tracker
            tracker = self.activity_state_machine.get_personal_tracker(self.unique_id)

            # Update energy and success history in tracker
            tracker.record_energy_level(self.energy)

            # Update environmental context
            env_context = self.get_environmental_context()
            tracker.update_environmental_context(env_context)

            # Update personal time tracking
            self.activity_state_machine.update_bee_activity(
                self.unique_id, current_step
            )

            # Get enhanced current conditions including tracker data
            current_conditions = self.get_enhanced_current_conditions(tracker)

            # Check for state transitions using enhanced logic
            new_state = self.activity_state_machine.should_transition_state(
                self.unique_id, current_conditions
            )

            if new_state and new_state != self.status:
                self.transition_to_state(new_state, current_step)

    def get_current_conditions(self) -> Dict[str, Any]:
        """Get current conditions for state transition decisions"""
        conditions = {
            "role": self.role.value,
            "energy": "sufficient" if self.energy > 50 else "low",
            "age": self.age,
            "at_hive": self.location == self.colony.location if self.colony else False,
        }

        # Add role-specific conditions
        if self.role == BeeRole.FORAGER:
            conditions.update(
                {
                    "patch_found": self.target_patch is not None
                    if hasattr(self, "target_patch")
                    else False,
                    "resources_available": True,  # Simplified
                    "collection_successful": True,  # Simplified
                    "resource_quality": "high"
                    if self.model.random.random() > 0.3
                    else "low",
                }
            )
        elif self.role == BeeRole.NURSE:
            conditions.update(
                {
                    "brood_present": self.colony.get_brood_count() > 0
                    if self.colony
                    else False
                }
            )
        elif self.role == BeeRole.BUILDER:
            conditions.update(
                {
                    "construction_needed": True  # Simplified
                }
            )

        return conditions

    def transition_to_state(self, new_state: BeeStatus, current_step: int) -> None:
        """Transition to new activity state"""
        if self.activity_state_machine:
            self.activity_state_machine.transition_bee_state(
                self.unique_id, new_state, current_step
            )

        # Update bee's status
        old_status = self.status
        self.status = new_state

        # Handle state-specific initialization
        self.on_state_transition(old_status, new_state)

    def get_environmental_context(self) -> Dict[str, Any]:
        """Get environmental context for personal time tracking"""
        context = {}

        # Get environmental data from model if available
        if hasattr(self.model, "get_environmental_conditions"):
            context.update(self.model.get_environmental_conditions())

        # Default environmental values
        context.setdefault("weather", "clear")
        context.setdefault("temperature", 20)
        context.setdefault("time_of_day", 12)
        context.setdefault("season", "spring")

        return context

    def get_enhanced_current_conditions(self, tracker: Any) -> Dict[str, Any]:
        """Get enhanced current conditions including personal tracker data"""
        # Start with basic conditions
        conditions = self.get_current_conditions()

        # Add enhanced energy information
        if self.energy <= 10:
            conditions["energy"] = "critical"
        elif self.energy <= 30:
            conditions["energy"] = "low"
        elif self.energy >= 80:
            conditions["energy"] = "high"
        else:
            conditions["energy"] = "sufficient"

        conditions["energy_value"] = self.energy
        conditions["energy_trend"] = tracker.get_energy_trend()

        # Add success rate information
        conditions["recent_success_rate"] = tracker.get_recent_success_rate()

        # Add environmental context
        env_context = self.get_environmental_context()
        conditions.update(env_context)

        # Add colony-level information
        if self.colony:
            conditions["colony_energy_level"] = self.get_colony_energy_level()
            conditions["colony_needs_eggs"] = getattr(self.colony, "needs_eggs", True)
            conditions["brood_needs_care"] = getattr(
                self.colony, "brood_needs_care", False
            )
            conditions["construction_needed"] = getattr(
                self.colony, "needs_construction", False
            )

        # Add patch availability
        conditions["patches_available"] = hasattr(self.model, "get_available_patches")
        if conditions["patches_available"]:
            try:
                patches = self.model.get_available_patches()
                conditions["patches_available"] = len(patches) > 0 if patches else False
            except Exception:
                conditions["patches_available"] = True

        # Add location-based conditions
        if self.colony and hasattr(self.colony, "location") and self.location:
            hive_distance = self.get_distance_to(self.colony.location)
            conditions["at_hive"] = hive_distance < 10
            conditions["hive_distance"] = hive_distance

        # Add foraging target information
        conditions["has_target"] = self.current_foraging_target is not None
        if hasattr(self, "target_patch") and self.target_patch:
            conditions["at_patch"] = True
            conditions["patch_found"] = True
        else:
            conditions["at_patch"] = False
            conditions["patch_found"] = False

        # Add resource-related conditions
        conditions["collection_successful"] = (
            True  # Will be updated by specific behaviors
        )
        conditions["resource_quality"] = "medium"  # Default, updated during collection
        conditions["resource_type"] = "nectar"  # Default preference

        # Add dance-related information
        if hasattr(self, "recent_dance_followers"):
            conditions["recent_dance_followers"] = self.recent_dance_followers
        else:
            conditions["recent_dance_followers"] = 0

        return conditions

    def get_colony_energy_level(self) -> str:
        """Get colony energy level for decision making"""
        if not self.colony:
            return "sufficient"

        # Simple heuristic based on colony size and resources
        if hasattr(self.colony, "get_total_resources"):
            resources = self.colony.get_total_resources()
            bee_count = getattr(self.colony, "bee_count", 100)

            if resources < bee_count * 10:
                return "low"
            elif resources > bee_count * 50:
                return "high"
            else:
                return "sufficient"

        return "sufficient"

    def on_state_transition(self, old_state: BeeStatus, new_state: BeeStatus) -> None:
        """Handle state transition events with personal time tracking"""
        # Record transition in personal tracker if available
        if self.activity_state_machine:
            tracker = self.activity_state_machine.get_personal_tracker(self.unique_id)

            # Determine if the previous activity was successful
            success = self.evaluate_activity_success(old_state)
            tracker.record_activity_success(success)

            # Adapt behavioral preferences based on the transition
            env_context = self.get_environmental_context()
            tracker.adapt_behavior_preferences(old_state, success, env_context)

        # Handle role-specific transition behaviors
        self.handle_role_specific_transition(old_state, new_state)

    def evaluate_activity_success(self, completed_state: BeeStatus) -> bool:
        """Evaluate if the completed activity was successful"""
        # Foraging activities
        if completed_state in [BeeStatus.COLLECT_NECTAR, BeeStatus.COLLECT_POLLEN]:
            # Success if bee collected resources
            return (
                self.is_carrying_capacity_full()
                if hasattr(self, "is_carrying_capacity_full")
                else True
            )

        elif completed_state in [BeeStatus.BRINGING_NECTAR, BeeStatus.BRINGING_POLLEN]:
            # Success if bee reached hive
            if self.colony and hasattr(self.colony, "location") and self.location:
                return self.get_distance_to(self.colony.location) < 10
            return True

        elif completed_state == BeeStatus.SEARCHING:
            # Success if bee found a patch
            return self.current_foraging_target is not None

        elif completed_state == BeeStatus.DANCING:
            # Success if bee had followers (stored in recent_dance_followers)
            return getattr(self, "recent_dance_followers", 0) > 0

        elif completed_state == BeeStatus.NURSING:
            # Success if there was brood to care for
            return (
                self.colony.get_brood_count() > 0
                if self.colony and hasattr(self.colony, "get_brood_count")
                else True
            )

        elif completed_state == BeeStatus.EGG_LAYING:
            # Success if queen had energy and space
            return self.energy > 20 and self.role == BeeRole.QUEEN

        elif completed_state == BeeStatus.NEST_CONSTRUCTION:
            # Success if construction was needed
            return (
                getattr(self.colony, "needs_construction", True)
                if self.colony
                else True
            )

        # Default to success for other activities
        return True

    def handle_role_specific_transition(
        self, old_state: BeeStatus, new_state: BeeStatus
    ) -> None:
        """Handle role-specific behaviors during state transitions"""
        # Queen-specific behaviors
        if self.role == BeeRole.QUEEN:
            if new_state == BeeStatus.EGG_LAYING:
                # Initialize egg laying parameters
                pass

        # Forager-specific behaviors
        elif self.role == BeeRole.FORAGER:
            if new_state == BeeStatus.DANCING:
                # Initialize dance parameters
                self.recent_dance_followers = 0
            elif new_state in [BeeStatus.BRINGING_NECTAR, BeeStatus.BRINGING_POLLEN]:
                # Record successful collection
                if (
                    hasattr(self, "current_foraging_target")
                    and self.current_foraging_target
                ):
                    patch_id = self.current_foraging_target
                    resources = getattr(
                        self, "unload_resources", lambda: {"nectar": 0, "pollen": 0}
                    )()
                    total_resources = sum(resources.values())
                    self.record_foraging_success(patch_id, True, total_resources)

        # Nurse-specific behaviors
        elif self.role == BeeRole.NURSE:
            if new_state == BeeStatus.NURSING:
                # Reset nursing efficiency metrics
                pass

    def update_energy(self) -> None:
        """Update energy levels based on activity, physiology, and age"""
        # Use state-specific energy consumption based on physiology
        state_consumption = self.calculate_state_specific_energy_consumption()

        # Age factor - energy consumption increases with age
        age_factor = 1.0 + (self.age / 100.0)

        # Apply energy consumption
        total_consumption = state_consumption * age_factor
        self.energy -= total_consumption

        # Ensure energy stays within bounds
        self.energy = max(0.0, min(self.max_energy, self.energy))

    def get_activity_factor(self) -> float:
        """Get energy consumption multiplier based on current activity"""
        if (
            self.activity_state_machine
            and self.status in self.activity_state_machine.state_configs
        ):
            return self.activity_state_machine.get_state_energy_consumption(self.status)

        # Fallback to basic factors
        if self.status == BeeStatus.FORAGING:
            return 3.0
        elif self.status == BeeStatus.NURSING:
            return 1.5
        elif self.status == BeeStatus.DANCING:
            return 2.0
        else:
            return 1.0

    def make_foraging_decision(
        self, available_patches: List[Any]
    ) -> Tuple[str, Optional[int]]:
        """Make sophisticated foraging decision using enhanced algorithms"""

        if not self.foraging_decision_engine:
            return "exploit_known", None

        # Create foraging context
        context = self.create_foraging_context()

        # Make decision using enhanced algorithm
        decision_type, target_patch = (
            self.foraging_decision_engine.make_foraging_decision(
                self.unique_id, context, self.energy, available_patches
            )
        )

        return decision_type.value, target_patch

    def create_foraging_context(self) -> Any:
        """Create foraging context for decision making"""
        # Import here to avoid circular imports
        from .foraging_algorithms import ForagingContext

        # Get environmental information
        current_season = "spring"  # Default, should be from model
        current_hour = 12  # Default, should be from model
        current_weather = "clear"  # Default, should be from model

        if hasattr(self.model, "get_environmental_info"):
            env_info = self.model.get_environmental_info()
            current_season = env_info.get("season", "spring")
            current_hour = env_info.get("hour", 12)
            current_weather = env_info.get("weather", "clear")

        # Get colony information
        colony_energy_level = 0.5  # Default
        if self.colony and hasattr(self.colony, "get_energy_level"):
            colony_energy_level = self.colony.get_energy_level()

        # Get patch competition info
        patch_competition = {}
        if hasattr(self.model, "get_patch_competition"):
            patch_competition = self.model.get_patch_competition()

        # Get dance information
        dance_information = []
        if hasattr(self.model, "get_dance_information"):
            dance_information = self.model.get_dance_information()

        # Get scent trails
        scent_trails = []
        if hasattr(self.model, "get_scent_trails"):
            scent_trails = self.model.get_scent_trails()

        return ForagingContext(
            current_season=current_season,
            current_hour=current_hour,
            current_weather=current_weather,
            colony_energy_level=colony_energy_level,
            patch_competition=patch_competition,
            dance_information=dance_information,
            scent_trails=scent_trails,
        )

    def update_foraging_memory(
        self, patch_id: int, success: bool, energy_gained: float
    ) -> None:
        """Update foraging memory after patch visit"""

        if self.foraging_decision_engine:
            context = self.create_foraging_context()
            self.foraging_decision_engine.update_foraging_memory(
                patch_id, success, energy_gained, context
            )

    def get_foraging_summary(self) -> Dict[str, Any]:
        """Get comprehensive foraging behavior summary"""

        if self.foraging_decision_engine:
            return dict(self.foraging_decision_engine.get_foraging_summary())

        return {
            "strategy": "basic",
            "memory_count": len(self.foraging_memories),
            "current_target": self.current_foraging_target,
        }

    def record_foraging_success(
        self, patch_id: int, success: bool, energy_gained: float
    ) -> None:
        """Record foraging attempt for data collection"""

        # Update foraging memory
        self.update_foraging_memory(patch_id, success, energy_gained)

        # Record metrics if data collector available
        if self.data_collector and hasattr(self.data_collector, "bee_metrics"):
            if self.unique_id in self.data_collector.bee_metrics:
                metrics = self.data_collector.bee_metrics[self.unique_id]
                metrics.foraging_trips += 1
                if success:
                    metrics.successful_foraging += 1
                    metrics.energy_collected += energy_gained

    def record_dance_performance(
        self, dance_type: str, duration: int, followers: int
    ) -> None:
        """Record dance performance for data collection"""

        if self.data_collector and hasattr(self.data_collector, "bee_metrics"):
            if self.unique_id in self.data_collector.bee_metrics:
                metrics = self.data_collector.bee_metrics[self.unique_id]
                metrics.dances_performed += 1
                metrics.communication_events += followers

    def record_death(self, cause: str) -> None:
        """Record death information for data collection"""

        if self.data_collector and hasattr(self.data_collector, "bee_metrics"):
            if self.unique_id in self.data_collector.bee_metrics:
                metrics = self.data_collector.bee_metrics[self.unique_id]
                metrics.death_cause = cause
                metrics.lifespan = self.age

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for this bee"""

        base_metrics = {
            "bee_id": self.unique_id,
            "age": self.age,
            "role": self.role.value if hasattr(self.role, "value") else str(self.role),
            "status": self.status.value
            if hasattr(self.status, "value")
            else str(self.status),
            "energy": self.energy,
            "foraging_efficiency": self.foraging_efficiency,
            "disease_resistance": self.disease_resistance,
            "longevity_factor": self.longevity_factor,
        }

        # Add activity state machine metrics
        if self.activity_state_machine:
            activity_summary = self.activity_state_machine.get_activity_summary(
                self.unique_id
            )
            base_metrics.update(activity_summary)

        # Add foraging metrics
        foraging_summary = self.get_foraging_summary()
        base_metrics.update(foraging_summary)

        # Add genetic information
        if self.genotype:
            base_metrics.update(
                {
                    "sex": self.genotype.sex.value
                    if hasattr(self.genotype.sex, "value")
                    else str(self.genotype.sex),
                    "ploidy": self.genotype.ploidy.value
                    if hasattr(self.genotype.ploidy, "value")
                    else str(self.genotype.ploidy),
                    "is_diploid_male": self.genotype.is_diploid_male(),
                }
            )

        return base_metrics

    def initialize_resource_collection(self) -> None:
        """Initialize resource collection system"""

        # Import here to avoid circular imports
        from .resource_collection import DetailedResourceCollector, BeePhysiology

        # Initialize resource collector
        self.resource_collector = DetailedResourceCollector()

        # Initialize bee physiology based on role and genetics
        proboscis_length = 6.5  # Default honey bee proboscis length
        crop_capacity = 70.0  # Default crop capacity
        pollen_capacity = 20.0  # Default pollen basket capacity

        # Adjust based on role
        if self.role == BeeRole.FORAGER:
            proboscis_length *= 1.1  # Foragers have slightly longer proboscis
            crop_capacity *= 1.2  # Larger crop capacity
            pollen_capacity *= 1.3  # Larger pollen baskets

        # Genetic influences
        if self.genotype:
            # Add genetic variation (simplified)
            proboscis_length *= self.model.random.gauss(1.0, 0.1)
            crop_capacity *= self.model.random.gauss(1.0, 0.15)
            pollen_capacity *= self.model.random.gauss(1.0, 0.12)

        # Create bee physiology
        self.bee_physiology = BeePhysiology(
            proboscis_length=max(4.0, proboscis_length),
            proboscis_diameter=0.3,
            crop_capacity=max(40.0, crop_capacity),
            current_crop_volume=0.0,
            pollen_basket_capacity=max(10.0, pollen_capacity),
            current_pollen_load=0.0,
            pumping_rate=2.0,
            energy_level=self.energy,
            collection_efficiency=self.foraging_efficiency,
        )

    def attempt_resource_collection(
        self, patch: Any, resource_type: str = "nectar"
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Attempt to collect resources from a patch"""

        if not self.resource_collector or not self.bee_physiology:
            self.initialize_resource_collection()

        # After initialization, these should not be None
        assert self.resource_collector is not None
        assert self.bee_physiology is not None

        # Import resource types
        from .resource_collection import ResourceType

        # Convert resource type
        if resource_type == "nectar":
            res_type = ResourceType.NECTAR
        elif resource_type == "pollen":
            res_type = ResourceType.POLLEN
        else:
            res_type = ResourceType.NECTAR

        # Create flower characteristics from patch
        flower = self._create_flower_from_patch(patch)

        # Get environmental conditions
        env_conditions = self._get_environmental_conditions()

        # Calculate experience level
        experience_level = self._calculate_experience_level(patch)

        # Update bee physiology energy
        self.bee_physiology.energy_level = self.energy
        self.bee_physiology.collection_efficiency = self.foraging_efficiency

        # Attempt collection
        result, amount, details = self.resource_collector.attempt_resource_collection(
            self.bee_physiology, flower, res_type, experience_level, env_conditions
        )

        # Update bee state
        self.resource_collector.update_bee_physiology(
            self.bee_physiology,
            result,
            amount,
            res_type,
            details.get("handling_time", 3.0),
        )

        # Update agent energy
        self.energy = self.bee_physiology.energy_level

        return result.value, amount, details

    def _create_flower_from_patch(self, patch: Any) -> Any:
        """Create flower characteristics from patch data"""

        # Import here to avoid circular imports
        from .resource_collection import FlowerCharacteristics

        # Extract or estimate flower characteristics
        corolla_depth = getattr(patch, "corolla_depth", 5.0)
        corolla_diameter = getattr(patch, "corolla_diameter", 3.0)
        nectar_volume = getattr(patch, "nectar_volume", 2.5)
        sugar_concentration = getattr(patch, "sugar_concentration", 0.3)
        pollen_availability = getattr(patch, "pollen_availability", 4.0)
        accessibility = getattr(patch, "accessibility_index", 0.8)
        handling_time = getattr(patch, "handling_time", 3.0)
        quality = getattr(patch, "quality", 0.75)

        return FlowerCharacteristics(
            corolla_depth=corolla_depth,
            corolla_diameter=corolla_diameter,
            nectar_volume=nectar_volume,
            sugar_concentration=sugar_concentration,
            pollen_availability=pollen_availability,
            accessibility_index=accessibility,
            handling_time=handling_time,
            reward_quality=quality,
        )

    def _get_environmental_conditions(self) -> Dict[str, float]:
        """Get current environmental conditions"""

        # Get from model if available
        if hasattr(self.model, "get_environmental_conditions"):
            return dict(self.model.get_environmental_conditions())

        # Default conditions
        return {
            "temperature": 22.0,
            "humidity": 60.0,
            "wind_speed": 2.0,
            "precipitation": 0.0,
        }

    def _calculate_experience_level(self, patch: Any) -> float:
        """Calculate experience level for this patch"""

        patch_id = getattr(patch, "id", getattr(patch, "patch_id", 0))

        # Check foraging memory
        if patch_id in self.foraging_memories:
            memory = self.foraging_memories[patch_id]
            visits = memory.visits
            success_rate = memory.success_rate

            # Experience increases with visits and success
            experience = min(1.0, (visits / 10.0) * success_rate)
            return experience

        return 0.1  # Minimal experience for new patches

    def get_crop_load_percentage(self) -> float:
        """Get current crop load as percentage of capacity"""

        if not self.bee_physiology:
            return 0.0

        return (
            float(self.bee_physiology.current_crop_volume)
            / float(self.bee_physiology.crop_capacity)
        ) * 100.0

    def get_pollen_load_percentage(self) -> float:
        """Get current pollen load as percentage of capacity"""

        if not self.bee_physiology:
            return 0.0

        return (
            float(self.bee_physiology.current_pollen_load)
            / float(self.bee_physiology.pollen_basket_capacity)
        ) * 100.0

    def is_carrying_capacity_full(self) -> bool:
        """Check if bee is at or near carrying capacity"""

        if not self.bee_physiology:
            return False

        crop_full = self.get_crop_load_percentage() >= 85.0
        pollen_full = self.get_pollen_load_percentage() >= 85.0

        return crop_full or pollen_full

    def unload_resources(self) -> Dict[str, float]:
        """Unload collected resources at hive"""

        if not self.bee_physiology:
            return {"nectar": 0.0, "pollen": 0.0}

        nectar_amount = self.bee_physiology.current_crop_volume
        pollen_amount = self.bee_physiology.current_pollen_load

        # Reset loads
        self.bee_physiology.current_crop_volume = 0.0
        self.bee_physiology.current_pollen_load = 0.0

        return {"nectar": nectar_amount, "pollen": pollen_amount}

    def get_resource_collection_stats(self) -> Dict[str, Any]:
        """Get resource collection statistics"""

        if not self.resource_collector:
            return {}

        stats = dict(self.resource_collector.get_collection_efficiency_stats())

        # Add bee-specific stats
        if self.bee_physiology:
            stats.update(
                {
                    "proboscis_length": self.bee_physiology.proboscis_length,
                    "crop_capacity": self.bee_physiology.crop_capacity,
                    "current_crop_load": self.bee_physiology.current_crop_volume,
                    "pollen_capacity": self.bee_physiology.pollen_basket_capacity,
                    "current_pollen_load": self.bee_physiology.current_pollen_load,
                    "collection_efficiency": self.bee_physiology.collection_efficiency,
                }
            )

        return stats

    def check_mortality(self) -> None:
        """Check if bee dies from age, energy, or disease"""
        # Age-based mortality
        age_mortality = self.get_age_mortality_probability()

        # Energy-based mortality
        energy_mortality = 0.0
        if self.energy <= 0:
            energy_mortality = 0.5
        elif self.energy < 20:
            energy_mortality = 0.1

        # Disease-based mortality
        disease_mortality = self.disease_load * 0.1

        # Combined mortality probability
        total_mortality = min(0.9, age_mortality + energy_mortality + disease_mortality)
        total_mortality /= self.longevity_factor

        if self.model.random.random() < total_mortality:
            # Determine primary death cause before changing status
            death_cause = self.determine_death_cause(
                age_mortality, energy_mortality, disease_mortality
            )

            # Record death information
            self.record_death(death_cause.value)

            # Update status
            self.status = BeeStatus.DEAD

            contributing_factors = self.get_contributing_factors(
                age_mortality, energy_mortality, disease_mortality
            )

            self.die(death_cause, contributing_factors)

    def get_age_mortality_probability(self) -> float:
        """Calculate age-based mortality probability"""
        if self.role == BeeRole.QUEEN:
            return 0.0001  # Queens live much longer
        elif self.role == BeeRole.FORAGER:
            return 0.02 + (self.age / 1000.0)  # Foragers have high mortality
        else:
            return 0.005 + (self.age / 2000.0)  # House bees

    def die(
        self,
        death_cause: DetailedDeathCause = DetailedDeathCause.UNKNOWN,
        contributing_factors: Optional[List[DetailedDeathCause]] = None,
    ) -> None:
        """Handle bee death with detailed mortality tracking"""
        if contributing_factors is None:
            contributing_factors = []

        # Record mortality if tracker is available
        if self.colony and hasattr(self.colony, "mortality_tracker"):
            environmental_conditions = (
                self.model.get_environmental_conditions()
                if hasattr(self.model, "get_environmental_conditions")
                else {}
            )
            simulation_step = (
                self.model.schedule.steps if hasattr(self.model, "schedule") else 0
            )

            self.colony.mortality_tracker.record_death(
                self,
                death_cause,
                contributing_factors,
                environmental_conditions,
                simulation_step,
            )

        # Cleanup personal time tracking
        self.cleanup_personal_tracking()

        # Remove from colony and model
        if self.colony:
            self.colony.remove_bee(self)
        self.model.schedule.remove(self)

    def determine_death_cause(
        self, age_mortality: float, energy_mortality: float, disease_mortality: float
    ) -> DetailedDeathCause:
        """Determine primary death cause based on mortality factors"""
        # Find the highest mortality factor
        max_factor = max(age_mortality, energy_mortality, disease_mortality)

        if max_factor == age_mortality:
            return DetailedDeathCause.OLD_AGE
        elif max_factor == energy_mortality:
            if self.energy <= 0:
                return DetailedDeathCause.STARVATION
            else:
                return DetailedDeathCause.ENERGY_DEPLETION
        elif max_factor == disease_mortality:
            return DetailedDeathCause.DISEASE_INFECTION
        else:
            return DetailedDeathCause.UNKNOWN

    def get_contributing_factors(
        self, age_mortality: float, energy_mortality: float, disease_mortality: float
    ) -> List[DetailedDeathCause]:
        """Get contributing factors for death"""
        factors = []

        # Include significant contributing factors
        if age_mortality > 0.01:
            factors.append(DetailedDeathCause.OLD_AGE)
        if energy_mortality > 0.01:
            if self.energy <= 0:
                factors.append(DetailedDeathCause.STARVATION)
            else:
                factors.append(DetailedDeathCause.ENERGY_DEPLETION)
        if disease_mortality > 0.01:
            factors.append(DetailedDeathCause.DISEASE_INFECTION)

        return factors

    def execute_role_behavior(self) -> None:
        """Execute behavior based on current activity state"""
        # Execute state-specific behavior based on current status
        if self.status == BeeStatus.HIBERNATING:
            self.execute_hibernating_behavior()
        elif self.status == BeeStatus.NEST_CONSTRUCTION:
            self.execute_nest_construction_behavior()
        elif self.status == BeeStatus.SEARCHING:
            self.execute_searching_behavior()
        elif self.status == BeeStatus.RETURNING_EMPTY:
            self.execute_returning_empty_behavior()
        elif self.status == BeeStatus.RETURNING_UNHAPPY_NECTAR:
            self.execute_returning_unhappy_nectar_behavior()
        elif self.status == BeeStatus.RETURNING_UNHAPPY_POLLEN:
            self.execute_returning_unhappy_pollen_behavior()
        elif self.status == BeeStatus.NECTAR_FORAGING:
            self.execute_nectar_foraging_behavior()
        elif self.status == BeeStatus.COLLECT_NECTAR:
            self.execute_collect_nectar_behavior()
        elif self.status == BeeStatus.BRINGING_NECTAR:
            self.execute_bringing_nectar_behavior()
        elif self.status == BeeStatus.EXPERIMENTAL_FORAGING_NECTAR:
            self.execute_experimental_foraging_nectar_behavior()
        elif self.status == BeeStatus.POLLEN_FORAGING:
            self.execute_pollen_foraging_behavior()
        elif self.status == BeeStatus.COLLECT_POLLEN:
            self.execute_collect_pollen_behavior()
        elif self.status == BeeStatus.BRINGING_POLLEN:
            self.execute_bringing_pollen_behavior()
        elif self.status == BeeStatus.EXPERIMENTAL_FORAGING_POLLEN:
            self.execute_experimental_foraging_pollen_behavior()
        elif self.status == BeeStatus.EGG_LAYING:
            self.execute_egg_laying_behavior()
        elif self.status == BeeStatus.NURSING:
            self.execute_nursing_behavior()
        elif self.status == BeeStatus.DANCING:
            self.execute_dancing_behavior()
        elif self.status == BeeStatus.RESTING:
            self.execute_resting_behavior()
        elif self.status == BeeStatus.FORAGING:
            self.execute_foraging_behavior()
        elif self.status == BeeStatus.ALIVE:
            self.execute_alive_behavior()
        elif self.status == BeeStatus.DEAD:
            self.execute_dead_behavior()
        else:
            # Default behavior for unknown states
            self.execute_default_behavior()

    # ============================================
    # STATE-SPECIFIC BEHAVIOR IMPLEMENTATIONS
    # ============================================

    def execute_hibernating_behavior(self) -> None:
        """Behavior during hibernation state"""
        # Extremely low energy consumption during hibernation
        self.energy -= 0.1

        # Check if hibernation should end based on environmental conditions
        if hasattr(self.model, "get_temperature") and self.model.get_temperature() > 15:
            # Wake up if temperature rises above threshold
            if self.activity_state_machine:
                current_conditions = self.get_current_conditions()
                current_conditions["temperature"] = "warm"
                new_state = self.activity_state_machine.should_transition_state(
                    self.unique_id, current_conditions
                )
                if new_state and new_state != self.status:
                    self.transition_to_state(new_state, self.model.schedule.steps)

    def execute_nest_construction_behavior(self) -> None:
        """Behavior during nest construction"""
        # Higher energy consumption for construction work
        self.energy -= 2.5

        # Check if bee has materials and space for construction
        if self.colony and hasattr(self.colony, "needs_construction"):
            construction_success = self.model.random.random() < 0.7  # 70% success rate

            if construction_success:
                # Contribute to colony construction
                if hasattr(self.colony, "add_construction_progress"):
                    self.colony.add_construction_progress(1.0)

                # Check if construction is complete
                if not self.colony.needs_construction():
                    # Transition to different behavior when construction done
                    self.transition_to_state(
                        BeeStatus.RESTING, self.model.schedule.steps
                    )

    def execute_searching_behavior(self) -> None:
        """Behavior during patch searching"""
        # High energy consumption while searching
        self.energy -= 3.0

        # Search for new foraging patches
        if hasattr(self.model, "get_available_patches"):
            available_patches = self.model.get_available_patches()

            if available_patches:
                # Use foraging decision engine if available
                if self.foraging_decision_engine:
                    context = self.create_foraging_context()
                    decision_type, target_patch = (
                        self.foraging_decision_engine.make_foraging_decision(
                            self.unique_id, context, self.energy, available_patches
                        )
                    )

                    if target_patch:
                        self.current_foraging_target = target_patch
                        # Transition to appropriate collection state
                        if self.model.random.random() < 0.6:  # Prefer nectar
                            self.transition_to_state(
                                BeeStatus.NECTAR_FORAGING, self.model.schedule.steps
                            )
                        else:
                            self.transition_to_state(
                                BeeStatus.POLLEN_FORAGING, self.model.schedule.steps
                            )
                    else:
                        # No suitable patch found, continue searching or return
                        if self.energy < 30:  # Low energy, return empty
                            self.transition_to_state(
                                BeeStatus.RETURNING_EMPTY, self.model.schedule.steps
                            )
                else:
                    # Basic patch selection without advanced decision engine
                    selected_patch = self.model.random.choice(available_patches)
                    self.current_foraging_target = (
                        selected_patch.patch_id
                        if hasattr(selected_patch, "patch_id")
                        else 0
                    )
                    self.transition_to_state(
                        BeeStatus.NECTAR_FORAGING, self.model.schedule.steps
                    )
            else:
                # No patches available, return empty
                self.transition_to_state(
                    BeeStatus.RETURNING_EMPTY, self.model.schedule.steps
                )

    def execute_returning_empty_behavior(self) -> None:
        """Behavior when returning to hive without resources"""
        # Moderate energy consumption for return trip
        self.energy -= 2.0

        # Move towards hive
        if self.colony and hasattr(self.colony, "location"):
            hive_distance = (
                self.get_distance_to(self.colony.location) if self.location else 0
            )

            if hive_distance < 10:  # Close to hive
                # Arrived at hive, rest and potentially start new foraging trip
                self.transition_to_state(BeeStatus.RESTING, self.model.schedule.steps)

                # Record unsuccessful foraging trip
                self.record_foraging_success(0, False, 0.0)
            else:
                # Still traveling, continue movement
                self.move_towards_hive()

    def execute_returning_unhappy_nectar_behavior(self) -> None:
        """Behavior when returning with poor quality nectar"""
        # Energy consumption during return
        self.energy -= 2.0

        if self.colony and hasattr(self.colony, "location"):
            hive_distance = (
                self.get_distance_to(self.colony.location) if self.location else 0
            )

            if hive_distance < 10:  # At hive
                # Unload poor quality nectar without dancing
                resources = self.unload_resources()
                if self.colony and hasattr(self.colony, "add_resources"):
                    self.colony.add_resources(resources)

                # Record moderate success
                self.record_foraging_success(
                    self.current_foraging_target or 0, True, resources.get("nectar", 0)
                )

                # Don't dance for poor quality resources
                self.transition_to_state(BeeStatus.RESTING, self.model.schedule.steps)
            else:
                self.move_towards_hive()

    def execute_returning_unhappy_pollen_behavior(self) -> None:
        """Behavior when returning with poor quality pollen"""
        # Similar to unhappy nectar but for pollen
        self.energy -= 2.0

        if self.colony and hasattr(self.colony, "location"):
            hive_distance = (
                self.get_distance_to(self.colony.location) if self.location else 0
            )

            if hive_distance < 10:  # At hive
                # Unload poor quality pollen without dancing
                resources = self.unload_resources()
                if self.colony and hasattr(self.colony, "add_resources"):
                    self.colony.add_resources(resources)

                # Record moderate success
                self.record_foraging_success(
                    self.current_foraging_target or 0, True, resources.get("pollen", 0)
                )

                self.transition_to_state(BeeStatus.RESTING, self.model.schedule.steps)
            else:
                self.move_towards_hive()

    def execute_nectar_foraging_behavior(self) -> None:
        """Behavior during nectar foraging trip"""
        # High energy consumption while foraging
        self.energy -= 3.5

        # Move to target patch and attempt collection
        if self.current_foraging_target and hasattr(self.model, "get_patch_by_id"):
            target_patch = self.model.get_patch_by_id(self.current_foraging_target)

            if target_patch:
                patch_distance = self.get_distance_to_patch(target_patch)

                if patch_distance < 5:  # At patch
                    # Transition to collection state
                    self.transition_to_state(
                        BeeStatus.COLLECT_NECTAR, self.model.schedule.steps
                    )
                else:
                    # Still traveling to patch
                    self.move_towards_patch(target_patch)
            else:
                # Patch no longer available
                self.transition_to_state(BeeStatus.SEARCHING, self.model.schedule.steps)

    def execute_collect_nectar_behavior(self) -> None:
        """Behavior during nectar collection at patch"""
        # Moderate energy consumption during collection
        self.energy -= 1.5

        if self.current_foraging_target and hasattr(self.model, "get_patch_by_id"):
            target_patch = self.model.get_patch_by_id(self.current_foraging_target)

            if target_patch:
                # Attempt resource collection
                collection_result, energy_gained, details = (
                    self.attempt_resource_collection(target_patch, "nectar")
                )

                if collection_result == "success":
                    self.energy += energy_gained * 0.1  # Small energy boost from nectar

                    # Check if carrying capacity is full
                    if self.is_carrying_capacity_full():
                        # Determine quality and transition to appropriate returning state
                        resource_quality = details.get("quality", 0.5)
                        if resource_quality > 0.7:
                            self.transition_to_state(
                                BeeStatus.BRINGING_NECTAR, self.model.schedule.steps
                            )
                        else:
                            self.transition_to_state(
                                BeeStatus.RETURNING_UNHAPPY_NECTAR,
                                self.model.schedule.steps,
                            )
                    # else: continue collecting
                elif collection_result == "depleted":
                    # Patch is depleted, search for new patch
                    self.transition_to_state(
                        BeeStatus.SEARCHING, self.model.schedule.steps
                    )
                elif collection_result == "incompatible":
                    # Flower incompatible with bee physiology
                    self.transition_to_state(
                        BeeStatus.SEARCHING, self.model.schedule.steps
                    )

    def execute_bringing_nectar_behavior(self) -> None:
        """Behavior when returning with high-quality nectar"""
        # Energy consumption during return with load
        self.energy -= 2.5

        if self.colony and hasattr(self.colony, "location"):
            hive_distance = (
                self.get_distance_to(self.colony.location) if self.location else 0
            )

            if hive_distance < 10:  # At hive
                # Unload high-quality nectar
                resources = self.unload_resources()
                if self.colony and hasattr(self.colony, "add_resources"):
                    self.colony.add_resources(resources)

                # Record successful foraging
                self.record_foraging_success(
                    self.current_foraging_target or 0, True, resources.get("nectar", 0)
                )

                # High-quality resources trigger dancing
                self.transition_to_state(BeeStatus.DANCING, self.model.schedule.steps)
            else:
                self.move_towards_hive()

    def execute_experimental_foraging_nectar_behavior(self) -> None:
        """Behavior during experimental nectar foraging (exploring new areas)"""
        # Very high energy consumption for exploration
        self.energy -= 4.0

        # Explore new areas for potential patches
        if hasattr(self.model, "explore_new_patches"):
            new_patches = self.model.explore_new_patches(
                self.location if self.location else (0, 0)
            )

            if new_patches:
                # Found new patch, update memory
                best_patch = max(new_patches, key=lambda p: getattr(p, "quality", 0.5))
                self.current_foraging_target = getattr(best_patch, "patch_id", 0)

                # Add to foraging memory
                if hasattr(best_patch, "patch_id"):
                    memory = ForagingMemory(
                        patch_id=best_patch.patch_id,
                        quality=getattr(best_patch, "quality", 0.5),
                        distance=self.get_distance_to_patch(best_patch),
                        resource_type="nectar",
                    )
                    self.foraging_memories[best_patch.patch_id] = memory

                # Transition to collection
                self.transition_to_state(
                    BeeStatus.COLLECT_NECTAR, self.model.schedule.steps
                )
            else:
                # No new patches found, return to normal searching
                if self.energy < 40:  # Low energy from exploration
                    self.transition_to_state(
                        BeeStatus.RETURNING_EMPTY, self.model.schedule.steps
                    )
                else:
                    self.transition_to_state(
                        BeeStatus.SEARCHING, self.model.schedule.steps
                    )

    def execute_pollen_foraging_behavior(self) -> None:
        """Behavior during pollen foraging trip"""
        # High energy consumption for pollen foraging
        self.energy -= 3.0

        if self.current_foraging_target and hasattr(self.model, "get_patch_by_id"):
            target_patch = self.model.get_patch_by_id(self.current_foraging_target)

            if target_patch:
                patch_distance = self.get_distance_to_patch(target_patch)

                if patch_distance < 5:  # At patch
                    self.transition_to_state(
                        BeeStatus.COLLECT_POLLEN, self.model.schedule.steps
                    )
                else:
                    self.move_towards_patch(target_patch)
            else:
                self.transition_to_state(BeeStatus.SEARCHING, self.model.schedule.steps)

    def execute_collect_pollen_behavior(self) -> None:
        """Behavior during pollen collection at patch"""
        # Higher energy consumption for pollen collection (more work than nectar)
        self.energy -= 2.0

        if self.current_foraging_target and hasattr(self.model, "get_patch_by_id"):
            target_patch = self.model.get_patch_by_id(self.current_foraging_target)

            if target_patch:
                collection_result, energy_gained, details = (
                    self.attempt_resource_collection(target_patch, "pollen")
                )

                if collection_result == "success":
                    # Pollen provides less immediate energy than nectar
                    self.energy += energy_gained * 0.05

                    if self.is_carrying_capacity_full():
                        resource_quality = details.get("quality", 0.5)
                        if resource_quality > 0.7:
                            self.transition_to_state(
                                BeeStatus.BRINGING_POLLEN, self.model.schedule.steps
                            )
                        else:
                            self.transition_to_state(
                                BeeStatus.RETURNING_UNHAPPY_POLLEN,
                                self.model.schedule.steps,
                            )
                elif collection_result == "depleted":
                    self.transition_to_state(
                        BeeStatus.SEARCHING, self.model.schedule.steps
                    )
                elif collection_result == "incompatible":
                    self.transition_to_state(
                        BeeStatus.SEARCHING, self.model.schedule.steps
                    )

    def execute_bringing_pollen_behavior(self) -> None:
        """Behavior when returning with high-quality pollen"""
        # Energy consumption with pollen load
        self.energy -= 3.0  # Pollen is heavier than nectar

        if self.colony and hasattr(self.colony, "location"):
            hive_distance = (
                self.get_distance_to(self.colony.location) if self.location else 0
            )

            if hive_distance < 10:  # At hive
                resources = self.unload_resources()
                if self.colony and hasattr(self.colony, "add_resources"):
                    self.colony.add_resources(resources)

                self.record_foraging_success(
                    self.current_foraging_target or 0, True, resources.get("pollen", 0)
                )

                # Good pollen sources also trigger dancing
                self.transition_to_state(BeeStatus.DANCING, self.model.schedule.steps)
            else:
                self.move_towards_hive()

    def execute_experimental_foraging_pollen_behavior(self) -> None:
        """Behavior during experimental pollen foraging"""
        # Very high energy consumption for pollen exploration
        self.energy -= 4.5

        if hasattr(self.model, "explore_new_patches"):
            new_patches = self.model.explore_new_patches(
                self.location if self.location else (0, 0)
            )

            if new_patches:
                # Prefer patches with high pollen availability
                best_patch = max(
                    new_patches, key=lambda p: getattr(p, "pollen_quality", 0.5)
                )
                self.current_foraging_target = getattr(best_patch, "patch_id", 0)

                if hasattr(best_patch, "patch_id"):
                    memory = ForagingMemory(
                        patch_id=best_patch.patch_id,
                        quality=getattr(best_patch, "pollen_quality", 0.5),
                        distance=self.get_distance_to_patch(best_patch),
                        resource_type="pollen",
                    )
                    self.foraging_memories[best_patch.patch_id] = memory

                self.transition_to_state(
                    BeeStatus.COLLECT_POLLEN, self.model.schedule.steps
                )
            else:
                if self.energy < 40:
                    self.transition_to_state(
                        BeeStatus.RETURNING_EMPTY, self.model.schedule.steps
                    )
                else:
                    self.transition_to_state(
                        BeeStatus.SEARCHING, self.model.schedule.steps
                    )

    def execute_egg_laying_behavior(self) -> None:
        """Behavior during egg laying (Queen-specific)"""
        # Only queens can lay eggs
        if self.role != BeeRole.QUEEN:
            self.transition_to_state(BeeStatus.RESTING, self.model.schedule.steps)
            return

        # Moderate energy consumption for egg laying
        self.energy -= 2.0

        # Lay eggs if colony has capacity
        if (
            self.colony
            and hasattr(self.colony, "can_lay_eggs")
            and self.colony.can_lay_eggs()
        ):
            eggs_laid = self.model.random.randint(1, 5)  # 1-5 eggs per step

            for _ in range(eggs_laid):
                if hasattr(self.colony, "add_egg"):
                    self.colony.add_egg()

            # Record reproductive success
            if self.data_collector and hasattr(self.data_collector, "bee_metrics"):
                if self.unique_id in self.data_collector.bee_metrics:
                    # Track reproductive output
                    pass  # Could add egg-laying metrics

        # Queens may rest after laying or continue based on colony needs
        if self.energy < 50 or self.model.random.random() < 0.3:
            self.transition_to_state(BeeStatus.RESTING, self.model.schedule.steps)

    def execute_nursing_behavior(self) -> None:
        """Behavior during nursing/brood care"""
        # Moderate energy consumption for nursing
        self.energy -= 1.8

        # Care for brood if available
        if self.colony and hasattr(self.colony, "get_brood_needing_care"):
            brood_needing_care = self.colony.get_brood_needing_care()

            if brood_needing_care:
                # Provide care to brood
                care_provided = min(
                    3, len(brood_needing_care)
                )  # Care for up to 3 brood per step

                for i in range(care_provided):
                    if hasattr(self.colony, "provide_brood_care"):
                        self.colony.provide_brood_care(brood_needing_care[i])

                # Record nursing activity
                if self.data_collector and hasattr(self.data_collector, "bee_metrics"):
                    if self.unique_id in self.data_collector.bee_metrics:
                        metrics = self.data_collector.bee_metrics[self.unique_id]
                        metrics.communication_events += (
                            care_provided  # Track care events
                        )
            else:
                # No brood to care for, may transition to other activities
                if self.model.random.random() < 0.4:  # 40% chance to switch activity
                    if self.age > 15 and self.energy > 60:  # Mature with good energy
                        self.transition_to_state(
                            BeeStatus.SEARCHING, self.model.schedule.steps
                        )
                    else:
                        self.transition_to_state(
                            BeeStatus.RESTING, self.model.schedule.steps
                        )

    def execute_dancing_behavior(self) -> None:
        """Behavior during waggle dance communication"""
        # High energy consumption for dancing
        self.energy -= 3.0

        # Perform waggle dance to communicate patch information
        if (
            self.current_foraging_target
            and self.current_foraging_target in self.foraging_memories
        ):
            memory = self.foraging_memories[self.current_foraging_target]

            # Create dance info
            if hasattr(self.colony, "location") and self.location:
                direction = self.calculate_direction_to_patch(memory)

                dance_info = DanceInfo(
                    patch_id=memory.patch_id,
                    distance=memory.distance,
                    direction=direction,
                    quality=memory.quality,
                    resource_type=memory.resource_type,
                )

                # Communicate to other bees
                followers = self.communicate_patch_info(dance_info)

                # Record dance performance
                self.record_dance_performance("waggle", 5, followers)

        # Dance for limited time, then rest
        if self.model.random.random() < 0.6:  # 60% chance to stop dancing
            self.transition_to_state(BeeStatus.RESTING, self.model.schedule.steps)

    def execute_resting_behavior(self) -> None:
        """Behavior during resting state"""
        # Very low energy consumption while resting
        self.energy -= 0.5

        # Slowly recover energy while resting
        if self.energy < self.max_energy:
            self.energy += 0.8
            self.energy = min(self.max_energy, self.energy)

        # Decide on next activity based on colony needs and individual state
        if self.energy > 60:  # Sufficient energy to work
            if self.model.random.random() < 0.4:  # 40% chance to become active
                # Choose next activity based on role and colony needs
                next_activity = self.choose_next_activity()
                if next_activity and next_activity != self.status:
                    self.transition_to_state(next_activity, self.model.schedule.steps)

    def execute_foraging_behavior(self) -> None:
        """Behavior during general foraging (backward compatibility)"""
        # This is the old generic foraging state, transition to specific state
        if self.model.random.random() < 0.7:  # Prefer nectar
            self.transition_to_state(BeeStatus.SEARCHING, self.model.schedule.steps)
        else:
            # Go directly to specific foraging if target known
            if self.current_foraging_target:
                if self.model.random.random() < 0.6:
                    self.transition_to_state(
                        BeeStatus.NECTAR_FORAGING, self.model.schedule.steps
                    )
                else:
                    self.transition_to_state(
                        BeeStatus.POLLEN_FORAGING, self.model.schedule.steps
                    )
            else:
                self.transition_to_state(BeeStatus.SEARCHING, self.model.schedule.steps)

    def execute_alive_behavior(self) -> None:
        """Behavior for basic alive state (choose appropriate activity)"""
        # Basic alive state - choose appropriate activity based on role and conditions
        next_activity = self.choose_next_activity()
        if next_activity and next_activity != BeeStatus.ALIVE:
            self.transition_to_state(next_activity, self.model.schedule.steps)
        else:
            # Default to resting if no specific activity chosen
            self.transition_to_state(BeeStatus.RESTING, self.model.schedule.steps)

    def execute_dead_behavior(self) -> None:
        """Behavior for dead bees (no behavior)"""
        # Dead bees don't perform any actions
        pass

    def execute_default_behavior(self) -> None:
        """Default behavior for unknown states"""
        # Unknown state, transition to resting
        self.transition_to_state(BeeStatus.RESTING, self.model.schedule.steps)

    # ============================================
    # HELPER METHODS FOR STATE BEHAVIORS
    # ============================================

    def choose_next_activity(self) -> Optional[BeeStatus]:
        """Choose next activity based on role, energy, and colony needs"""
        if self.role == BeeRole.QUEEN:
            return BeeStatus.EGG_LAYING if self.energy > 40 else BeeStatus.RESTING
        elif self.role == BeeRole.NURSE:
            if (
                self.colony
                and hasattr(self.colony, "get_brood_count")
                and self.colony.get_brood_count() > 0
            ):
                return BeeStatus.NURSING
            elif self.age > 15 and self.energy > 50:  # Mature enough to forage
                return BeeStatus.SEARCHING
            else:
                return BeeStatus.RESTING
        elif self.role == BeeRole.FORAGER:
            if self.energy > 40:
                return BeeStatus.SEARCHING
            else:
                return BeeStatus.RESTING
        elif self.role == BeeRole.BUILDER:
            if (
                self.colony
                and hasattr(self.colony, "needs_construction")
                and self.colony.needs_construction()
            ):
                return BeeStatus.NEST_CONSTRUCTION
            else:
                return BeeStatus.RESTING
        else:
            return BeeStatus.RESTING

    def move_towards_hive(self) -> None:
        """Move bee towards hive location"""
        if self.colony and hasattr(self.colony, "location") and self.location:
            hive_x, hive_y = self.colony.location
            curr_x, curr_y = self.location

            # Simple movement towards hive
            dx = hive_x - curr_x
            dy = hive_y - curr_y
            distance = self.get_distance_to((hive_x, hive_y))

            if distance > 0:
                move_speed = 5.0  # units per step
                move_x = (dx / distance) * move_speed
                move_y = (dy / distance) * move_speed

                self.location = (curr_x + move_x, curr_y + move_y)

    def move_towards_patch(self, patch: Any) -> None:
        """Move bee towards target patch"""
        if hasattr(patch, "location") and self.location:
            patch_x, patch_y = patch.location
            curr_x, curr_y = self.location

            dx = patch_x - curr_x
            dy = patch_y - curr_y
            distance = self.get_distance_to((patch_x, patch_y))

            if distance > 0:
                move_speed = 4.0  # Slightly slower when carrying resources
                move_x = (dx / distance) * move_speed
                move_y = (dy / distance) * move_speed

                self.location = (curr_x + move_x, curr_y + move_y)

    def get_distance_to_patch(self, patch: Any) -> float:
        """Calculate distance to a patch"""
        if hasattr(patch, "location") and self.location:
            patch_x, patch_y = patch.location
            curr_x, curr_y = self.location
            return math.sqrt((patch_x - curr_x) ** 2 + (patch_y - curr_y) ** 2)
        return 0.0

    def calculate_direction_to_patch(self, memory: ForagingMemory) -> float:
        """Calculate direction to patch for dance communication"""
        # Simplified direction calculation (in radians)
        if self.location and hasattr(self.model, "get_patch_by_id"):
            patch = self.model.get_patch_by_id(memory.patch_id)
            if patch and hasattr(patch, "location"):
                patch_x, patch_y = patch.location
                curr_x, curr_y = self.location

                dx = patch_x - curr_x
                dy = patch_y - curr_y

                return math.atan2(dy, dx)  # Returns angle in radians

        return 0.0  # Default direction

    def communicate_patch_info(self, dance_info: DanceInfo) -> int:
        """Communicate patch information to other bees, return number of followers"""
        followers = 0

        # Get nearby bees who might follow the dance
        if self.colony and hasattr(self.colony, "get_nearby_bees"):
            nearby_bees = self.colony.get_nearby_bees(
                self.location if self.location else (0, 0), radius=10
            )

            for bee in nearby_bees:
                if (
                    bee != self
                    and hasattr(bee, "role")
                    and bee.role in [BeeRole.FORAGER, BeeRole.NURSE]
                    and bee.energy > 40
                ):
                    # Probability of following dance based on patch quality
                    follow_probability = dance_info.quality * 0.7

                    if self.model.random.random() < follow_probability:
                        # Bee follows dance, learns patch location
                        if hasattr(bee, "foraging_memories"):
                            bee.foraging_memories[dance_info.patch_id] = ForagingMemory(
                                patch_id=dance_info.patch_id,
                                quality=dance_info.quality
                                * 0.9,  # Slightly reduced quality for followers
                                distance=dance_info.distance,
                                resource_type=dance_info.resource_type,
                            )
                            bee.current_foraging_target = dance_info.patch_id

                        followers += 1

        return followers

    # ============================================
    # ENHANCED PERSONAL TIME TRACKING INTEGRATION
    # ============================================

    def get_personal_activity_summary(self) -> Dict[str, Any]:
        """Get comprehensive activity summary for this bee"""
        if not self.activity_state_machine:
            return {}

        tracker = self.activity_state_machine.get_personal_tracker(self.unique_id)
        summary = self.activity_state_machine.get_activity_summary(self.unique_id)

        # Add bee-specific information
        summary.update(
            {
                "bee_id": self.unique_id,
                "role": self.role.value
                if hasattr(self.role, "value")
                else str(self.role),
                "age": self.age,
                "energy": self.energy,
                "location": self.location,
                "colony_id": getattr(self.colony, "unique_id", None)
                if self.colony
                else None,
            }
        )

        # Add behavioral analytics
        if hasattr(tracker, "get_recent_success_rate"):
            summary["recent_success_rate"] = tracker.get_recent_success_rate()
            summary["energy_trend"] = tracker.get_energy_trend()
            summary["behavioral_preferences"] = tracker.behavioral_preferences.copy()
            summary["preferred_sequences"] = tracker.get_preferred_state_sequence()

        return summary

    def get_behavioral_recommendations(self) -> List[Dict[str, Any]]:
        """Get behavioral recommendations for this bee"""
        if not self.activity_state_machine:
            return []

        tracker = self.activity_state_machine.get_personal_tracker(self.unique_id)
        current_conditions = self.get_enhanced_current_conditions(tracker)

        return self.activity_state_machine.get_context_aware_recommendations(
            self.unique_id, current_conditions
        )

    def predict_future_behavior(self, steps: int = 5) -> List[Tuple[BeeStatus, float]]:
        """Predict future behavioral states"""
        if not self.activity_state_machine:
            return []

        tracker = self.activity_state_machine.get_personal_tracker(self.unique_id)
        current_conditions = self.get_enhanced_current_conditions(tracker)

        return self.activity_state_machine.predict_next_state(
            self.unique_id, current_conditions, steps
        )

    def analyze_behavioral_patterns(self) -> Dict[str, Any]:
        """Analyze this bee's behavioral patterns"""
        if not self.activity_state_machine:
            return {}

        return self.activity_state_machine.analyze_state_patterns(self.unique_id)

    def update_dance_followers(self, followers: int) -> None:
        """Update the number of dance followers for tracking"""
        self.recent_dance_followers = followers

        # Record in personal tracker
        if self.activity_state_machine:
            tracker = self.activity_state_machine.get_personal_tracker(self.unique_id)
            # Success is determined by having followers
            success = followers > 0
            tracker.record_activity_success(success)

    def initialize_personal_time_tracking(self) -> None:
        """Initialize enhanced personal time tracking for this bee"""
        if self.activity_state_machine:
            # Ensure tracker exists
            tracker = self.activity_state_machine.get_personal_tracker(self.unique_id)

            # Initialize with current state
            if hasattr(self.model, "schedule"):
                current_step = self.model.schedule.steps
                tracker.transition_to_state(self.status, current_step)

            # Set initial preferences based on role
            self.set_initial_behavioral_preferences(tracker)

    def set_initial_behavioral_preferences(self, tracker: Any) -> None:
        """Set initial behavioral preferences based on bee role"""
        if self.role == BeeRole.QUEEN:
            tracker.behavioral_preferences.update(
                {
                    BeeStatus.EGG_LAYING.value: 0.8,
                    BeeStatus.RESTING.value: 0.6,
                    BeeStatus.NURSING.value: 0.3,
                }
            )
        elif self.role == BeeRole.FORAGER:
            tracker.behavioral_preferences.update(
                {
                    BeeStatus.SEARCHING.value: 0.8,
                    BeeStatus.NECTAR_FORAGING.value: 0.7,
                    BeeStatus.POLLEN_FORAGING.value: 0.6,
                    BeeStatus.DANCING.value: 0.5,
                }
            )
        elif self.role == BeeRole.NURSE:
            tracker.behavioral_preferences.update(
                {
                    BeeStatus.NURSING.value: 0.8,
                    BeeStatus.RESTING.value: 0.6,
                    BeeStatus.SEARCHING.value: 0.4,
                }
            )
        elif self.role == BeeRole.BUILDER:
            tracker.behavioral_preferences.update(
                {BeeStatus.NEST_CONSTRUCTION.value: 0.8, BeeStatus.RESTING.value: 0.5}
            )

    def get_contextual_state_preference(self, state: BeeStatus) -> float:
        """Get contextual preference for a specific state"""
        if not self.activity_state_machine:
            return 0.5

        tracker = self.activity_state_machine.get_personal_tracker(self.unique_id)
        current_conditions = self.get_enhanced_current_conditions(tracker)

        return tracker.get_contextual_preference(state, current_conditions)

    def cleanup_personal_tracking(self) -> None:
        """Cleanup personal tracking when bee dies"""
        if self.activity_state_machine:
            # Record final state and cleanup
            self.activity_state_machine.remove_bee_tracker(self.unique_id)

    def consume_resources(self, amount: float) -> None:
        """Consume resources from colony stores"""
        if self.colony:
            self.colony.consume_resources(amount)

    def get_distance_to(self, target_pos: Tuple[float, float]) -> float:
        """Calculate distance to target position"""
        if self.location is None:
            return float("inf")
        dx = target_pos[0] - self.location[0]
        dy = target_pos[1] - self.location[1]
        return math.sqrt(dx**2 + dy**2)

    # =======================================================================
    # DOCUMENTED API METHODS - Required per spec lines 187-191
    # =======================================================================

    def forage(self, patches: List[Any]) -> Dict[str, Any]:
        """
        Core foraging method for documented API compliance.

        Args:
            patches: List of resource patches available for foraging

        Returns:
            Dictionary containing foraging results including:
            - success: Whether foraging was successful
            - energy_gained: Amount of energy gained
            - resource_collected: Type and amount of resource collected
            - patch_visited: ID of patch visited
            - time_spent: Time spent foraging
        """
        if self.status != BeeStatus.ALIVE:
            return {
                "success": False,
                "energy_gained": 0.0,
                "resource_collected": {"nectar": 0.0, "pollen": 0.0},
                "patch_visited": None,
                "time_spent": 0.0,
                "reason": "bee_not_alive",
            }

        # Check if bee can forage
        if self.role not in [BeeRole.FORAGER, BeeRole.WORKER]:
            return {
                "success": False,
                "energy_gained": 0.0,
                "resource_collected": {"nectar": 0.0, "pollen": 0.0},
                "patch_visited": None,
                "time_spent": 0.0,
                "reason": "not_forager_role",
            }

        # Check energy level
        if self.energy < 20.0:  # Minimum energy to forage
            return {
                "success": False,
                "energy_gained": 0.0,
                "resource_collected": {"nectar": 0.0, "pollen": 0.0},
                "patch_visited": None,
                "time_spent": 0.0,
                "reason": "insufficient_energy",
            }

        # Check if already at carrying capacity
        if self.is_carrying_capacity_full():
            return {
                "success": False,
                "energy_gained": 0.0,
                "resource_collected": {"nectar": 0.0, "pollen": 0.0},
                "patch_visited": None,
                "time_spent": 0.0,
                "reason": "carrying_capacity_full",
            }

        # No patches available
        if not patches:
            return {
                "success": False,
                "energy_gained": 0.0,
                "resource_collected": {"nectar": 0.0, "pollen": 0.0},
                "patch_visited": None,
                "time_spent": 0.0,
                "reason": "no_patches_available",
            }

        # Select best patch based on memory and decision making
        target_patch = self._select_best_patch(patches)
        if not target_patch:
            return {
                "success": False,
                "energy_gained": 0.0,
                "resource_collected": {"nectar": 0.0, "pollen": 0.0},
                "patch_visited": None,
                "time_spent": 0.0,
                "reason": "no_suitable_patch",
            }

        # Calculate travel time and energy cost
        travel_distance = self._calculate_patch_distance(target_patch)
        travel_time = travel_distance / 5.0  # 5 meters per time unit
        travel_energy_cost = travel_distance * 0.1  # 0.1 energy per meter

        # Attempt resource collection at the patch
        collection_result = self.attempt_resource_collection(target_patch)
        success = collection_result[0] == "success"
        energy_gained = collection_result[1]
        collection_details = collection_result[2]

        # Calculate net energy (gained - travel cost)
        net_energy = energy_gained - travel_energy_cost
        self.update_energy_direct(net_energy)

        # Update foraging memory
        patch_id = getattr(target_patch, "unique_id", id(target_patch))
        self.update_foraging_memory(patch_id, success, energy_gained)

        # Record foraging metrics
        total_time = travel_time * 2 + (
            5.0 if success else 1.0
        )  # Round trip + foraging time

        return {
            "success": success,
            "energy_gained": net_energy,
            "resource_collected": {
                "nectar": collection_details.get("nectar_collected", 0.0),
                "pollen": collection_details.get("pollen_collected", 0.0),
            },
            "patch_visited": patch_id,
            "time_spent": total_time,
            "travel_distance": travel_distance,
            "collection_efficiency": collection_details.get("efficiency", 0.0),
            "reason": "successful_foraging" if success else "collection_failed",
        }

    def update_energy_with_change(self, energy_change: float) -> float:
        """
        Update bee's energy level with the specified change.

        Args:
            energy_change: Amount to change energy (positive or negative)

        Returns:
            New energy level after update
        """
        self.energy = max(0.0, min(self.max_energy, self.energy + energy_change))

        # Check for death due to energy depletion
        if self.energy <= 0.0 and self.status == BeeStatus.ALIVE:
            self.die(
                death_cause=DetailedDeathCause.ENERGY_DEPLETION,
                contributing_factors=[DetailedDeathCause.ENERGY_DEPLETION],
            )

        return self.energy

    def update_energy_direct(self, energy_change: float) -> None:
        """Direct energy update without return value (for internal use)"""
        self.update_energy_with_change(energy_change)

    def _select_best_patch(self, patches: List[Any]) -> Optional[Any]:
        """Select the best patch for foraging based on memory and quality"""
        if not patches:
            return None

        best_patch = None
        best_score = -1.0

        for patch in patches:
            # Calculate patch attractiveness score
            patch_id = getattr(patch, "unique_id", id(patch))

            # Use memory if available
            if patch_id in self.foraging_memories:
                memory = self.foraging_memories[patch_id]
                quality_score = memory.quality * memory.success_rate
                distance_penalty = memory.distance / 1000.0  # Normalize distance
                score = quality_score - distance_penalty + memory.memory_strength
            else:
                # New patch - estimate based on patch properties
                quality = getattr(patch, "quality", 0.5)
                distance = self._calculate_patch_distance(patch)
                distance_penalty = distance / 1000.0
                score = quality - distance_penalty + 0.1  # Small bonus for exploration

            if score > best_score:
                best_score = score
                best_patch = patch

        return best_patch

    def _calculate_patch_distance(self, patch: Any) -> float:
        """Calculate distance to a patch"""
        if not hasattr(patch, "location") or self.location is None:
            return 100.0  # Default distance if location unavailable

        patch_location = getattr(patch, "location", (0.0, 0.0))
        return self.get_distance_to(patch_location)


class Queen(BeeAgent):
    """
    Queen bee agent - responsible for egg laying and pheromone production.

    Maps to NetLogo queen breed with egg-laying procedures.
    """

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        colony: Any,
        genotype: Optional[Genotype] = None,
    ):
        super().__init__(unique_id, model, age=0, genotype=genotype)
        self.role = BeeRole.QUEEN
        self.colony = colony
        self.egg_laying_rate = 0.0
        self.max_egg_laying_rate = 2000.0
        self.pheromone_level = 100.0
        self.mated = False
        self.sperm_count = 0

        # Genetic system integration
        self.spermatheca = SpermathecaManager()
        self.genetic_system: Optional[Any] = None  # Will be set by colony

        # Ensure queen is female (or diploid male that dies)
        if self.genotype and self.genotype.sex != Sex.FEMALE:
            if self.genotype.is_diploid_male():
                self.status = BeeStatus.DEAD  # Diploid male queens die
            else:
                # Male cannot be queen - this should not happen
                raise ValueError("Male genotype cannot be queen")

    def step(self) -> None:
        """Queen-specific step behavior"""
        super().step()
        if self.status == BeeStatus.ALIVE:
            self.lay_eggs()
            self.produce_pheromones()

    def lay_eggs(self) -> None:
        """Lay eggs based on colony conditions using genetic system"""
        if not self.mated or not self.genotype:
            return

        # Calculate egg laying rate based on colony conditions
        if self.colony is None:
            return
        resource_factor = self.colony.get_resource_adequacy()
        population_factor = self.colony.get_population_pressure()
        seasonal_factor = self.model.get_seasonal_factor()

        self.egg_laying_rate = (
            self.max_egg_laying_rate
            * resource_factor
            * population_factor
            * seasonal_factor
        )

        # Lay eggs using genetic system
        eggs_to_lay = int(self.model.random.poisson(self.egg_laying_rate / 365.0))
        for _ in range(eggs_to_lay):
            # Decide whether to fertilize egg (70% fertilized for workers/queens, 30% unfertilized for drones)
            if self.model.random.random() < 0.7:
                # Fertilized egg
                sperm_cell = self.spermatheca.get_sperm_for_fertilization()
                if sperm_cell and self.genetic_system:
                    offspring_genotype = self.genetic_system.fertilize_egg(
                        self.genotype, sperm_cell
                    )
                    # Check for diploid male
                    if offspring_genotype.is_diploid_male():
                        # Colony stress from diploid male production
                        self.colony.add_stress("diploid_male_production", 0.2)
                    self.colony.add_brood_with_genotype(offspring_genotype)
            else:
                # Unfertilized egg (drone)
                if self.genetic_system:
                    drone_genotype = self.genetic_system.create_unfertilized_egg(
                        self.genotype
                    )
                    self.colony.add_brood_with_genotype(drone_genotype)

    def produce_pheromones(self) -> None:
        """Produce queen pheromones to maintain colony cohesion"""
        self.pheromone_level = min(100.0, self.pheromone_level + 5.0)
        if self.colony:
            self.colony.pheromone_level = self.pheromone_level

    def mate_with_drones(self, drone_genotypes: List[Genotype]) -> Optional[int]:
        """Mate with multiple drones and store sperm"""
        if not self.genotype or not self.genetic_system:
            return None

        # Simulate mating flight
        sperm_cells = self.genetic_system.mate_individuals(
            self.genotype, drone_genotypes
        )

        # Store sperm in spermatheca
        stored_count = self.spermatheca.store_sperm(sperm_cells)

        if stored_count > 0:
            self.mated = True
            self.sperm_count = self.spermatheca.get_sperm_count()

        return stored_count

    def age_sperm(self) -> None:
        """Age sperm in spermatheca"""
        self.spermatheca.age_sperm()
        self.sperm_count = self.spermatheca.get_sperm_count()


class Worker(BeeAgent):
    """
    Worker bee agent - can be nurse, forager, guard, or builder.

    Maps to NetLogo worker breed with role-specific behaviors.
    """

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        colony: Any,
        genotype: Optional[Genotype] = None,
    ):
        super().__init__(unique_id, model, age=0, genotype=genotype)
        self.colony = colony
        self.role = BeeRole.NURSE
        self.foraging_history: List[ForagingMemory] = []
        self.dance_threshold = 0.5
        self.role_transition_age = 21  # Days to transition from nurse to forager

        # Genetic system integration - workers must be female
        if self.genotype and self.genotype.sex != Sex.FEMALE:
            if self.genotype.is_diploid_male():
                self.status = BeeStatus.DEAD  # Diploid male workers die
            else:
                # Male cannot be worker - should be drone
                raise ValueError("Male genotype cannot be worker")

    def step(self) -> None:
        """Worker-specific step behavior"""
        super().step()
        if self.status == BeeStatus.ALIVE:
            self.consider_role_transition()

    def consider_role_transition(self) -> None:
        """Consider transitioning between roles based on age and colony needs"""
        if self.age > self.role_transition_age and self.role == BeeRole.NURSE:
            if self.colony is not None:
                colony_needs = self.colony.assess_needs()
                if colony_needs is not None and colony_needs.get(
                    "foragers", 0
                ) > colony_needs.get("nurses", 0):
                    self.transition_to_forager()

    def transition_to_forager(self) -> None:
        """Transition from nurse to forager role"""
        self.role = BeeRole.FORAGER
        self.status = BeeStatus.FORAGING

    def execute_role_behavior(self) -> None:
        """Execute behavior based on current role"""
        if self.role == BeeRole.NURSE:
            self.nurse_behavior()
        elif self.role == BeeRole.FORAGER:
            self.forager_behavior()
        elif self.role == BeeRole.GUARD:
            self.guard_behavior()
        elif self.role == BeeRole.BUILDER:
            self.builder_behavior()

    def nurse_behavior(self) -> None:
        """Nursing behavior - tend to brood"""
        if self.colony:
            brood_count = self.colony.get_brood_count()
            if brood_count > 0:
                self.status = BeeStatus.NURSING
                self.energy -= 2.0
                self.colony.tend_brood(self)
            else:
                self.status = BeeStatus.RESTING

    def forager_behavior(self) -> None:
        """Foraging behavior - implemented in Forager subclass"""
        pass

    def guard_behavior(self) -> None:
        """Guard behavior - protect colony entrance"""
        self.status = BeeStatus.RESTING
        self.energy -= 1.5

    def builder_behavior(self) -> None:
        """Builder behavior - construct and maintain comb"""
        if self.colony:
            self.colony.build_comb(self)
            self.energy -= 2.0


class Forager(Worker):
    """
    Forager bee agent - specialized for resource collection.

    Maps to NetLogo forager breed with complex foraging behaviors.
    """

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        colony: Any,
        genotype: Optional[Genotype] = None,
    ):
        super().__init__(unique_id, model, colony, genotype=genotype)
        self.role = BeeRole.FORAGER
        self.target_patch: Optional[int] = None
        self.carrying_capacity = 50.0  # mg
        self.current_load = 0.0
        self.resource_type: Optional[str] = None
        self.foraging_range = 2000.0  # meters
        self.dance_followers: List[int] = []
        self.following_dance: Optional[Any] = None

    def forager_behavior(self) -> None:
        """Main foraging behavior decision tree"""
        if self.current_load >= self.carrying_capacity:
            self.return_to_hive()
        elif self.target_patch is None:
            self.find_foraging_patch()
        else:
            self.forage_at_patch()

    def find_foraging_patch(self) -> None:
        """Find a suitable foraging patch"""
        # Check if following a dance
        if self.following_dance:
            self.target_patch = self.following_dance.patch_id
            self.following_dance = None
            return

        # Use memory of previous successful patches
        if self.foraging_history:
            best_memory = max(
                self.foraging_history, key=lambda m: m.quality / (m.distance + 1)
            )
            if best_memory.quality > 0.5:
                self.target_patch = best_memory.patch_id
                return

        # Scout for new patches
        self.scout_new_patches()

    def scout_new_patches(self) -> None:
        """Scout for new resource patches with masterpatch and proboscis-corolla matching"""
        landscape = self.model.landscape
        current_pos = (
            self.location
            if self.location is not None
            else (
                self.colony.location
                if self.colony is not None and self.colony.location is not None
                else (0.0, 0.0)
            )
        )

        # Get proboscis-corolla matching system
        proboscis_system = getattr(self.model, "proboscis_system", None)
        species_name = getattr(self.colony, "species", "Bombus_terrestris")

        # Use masterpatch system if available
        if landscape.use_masterpatch_system and proboscis_system:
            # Get best patches for this species
            patch_scores = landscape.get_best_foraging_patches(
                species_name, current_pos, self.foraging_range, proboscis_system
            )

            if patch_scores:
                # Select patch probabilistically based on scores
                patches, scores = zip(*patch_scores)

                # Filter out patches with zero score
                valid_patches = [(p, s) for p, s in zip(patches, scores) if s > 0]

                if valid_patches:
                    patches, scores = zip(*valid_patches)
                    selected_patch = self.model.random.choices(patches, weights=scores)[
                        0
                    ]
                    self.target_patch = selected_patch.patch_id

        else:
            # Fallback to regular patch selection
            nearby_patches = landscape.get_patches_in_radius(
                current_pos, self.foraging_range
            )
            resource_patches = [p for p in nearby_patches if p.has_resources()]

            if resource_patches:
                # Select patch based on quality, distance, and flower accessibility
                weights = []
                for patch in resource_patches:
                    distance = self.get_distance_to(patch.location)
                    base_quality = patch.get_resource_quality()

                    # Apply proboscis-corolla matching if available
                    if proboscis_system and self.proboscis_characteristics:
                        # Calculate accessibility for flowers in this patch
                        accessibility_score = (
                            proboscis_system.calculate_patch_accessibility_score(
                                species_name, patch.flower_species
                            )
                        )

                        # Adjust quality based on accessibility
                        adjusted_quality = base_quality * accessibility_score
                    else:
                        adjusted_quality = base_quality

                    weight = adjusted_quality / (distance + 1)
                    weights.append(weight)

                # Probabilistic selection
                if sum(weights) > 0:
                    selected_patch = self.model.random.choices(
                        resource_patches, weights=weights
                    )[0]
                    self.target_patch = selected_patch.id

    def forage_at_patch(self) -> None:
        """Forage at the current target patch with masterpatch support"""
        if self.target_patch is None:
            return

        landscape = self.model.landscape

        # Handle both masterpatch and regular patch systems
        if landscape.use_masterpatch_system:
            # Get masterpatch
            if self.target_patch in landscape.masterpatch_system.masterpatches:
                patch = landscape.masterpatch_system.masterpatches[self.target_patch]

                # Check if patch has resources
                total_nectar, total_pollen = patch.get_total_resources()
                if total_nectar + total_pollen <= 0:
                    self.target_patch = None
                    return
            else:
                self.target_patch = None
                return
        else:
            # Regular patch system
            patch = landscape.get_patch(self.target_patch)

            if patch is None or not patch.has_resources():
                self.target_patch = None
                return

        # Move to patch (simplified - in full implementation would be gradual)
        self.location = patch.location
        self.status = BeeStatus.FORAGING

        # Collect resources with masterpatch and proboscis-corolla matching
        base_collection_rate = self.foraging_efficiency * 10.0  # mg per step

        if landscape.use_masterpatch_system:
            # Use masterpatch system for resource collection
            proboscis_system = getattr(self.model, "proboscis_system", None)
            species_name = getattr(self.colony, "species", "Bombus_terrestris")

            # Simulate foraging on masterpatch
            foraging_duration = 1.0  # 1 hour foraging per step
            consumption_results = landscape.simulate_foraging_on_masterpatch(
                self.target_patch, species_name, foraging_duration, proboscis_system
            )

            if consumption_results:
                # Calculate total resources collected
                total_nectar = sum(
                    result["nectar_consumed"] for result in consumption_results.values()
                )
                total_pollen = sum(
                    result["pollen_consumed"] for result in consumption_results.values()
                )
                total_collected = total_nectar + total_pollen

                # Limit by carrying capacity
                if total_collected > 0:
                    capacity_factor = min(
                        1.0,
                        (self.carrying_capacity - self.current_load) / total_collected,
                    )
                    actual_collected = total_collected * capacity_factor

                    self.current_load += actual_collected
                    self.resource_type = (
                        "nectar" if total_nectar > total_pollen else "pollen"
                    )

                    # Calculate average accessibility for energy cost
                    total_accessibility = sum(
                        result["accessibility"]
                        for result in consumption_results.values()
                    )
                    avg_accessibility = total_accessibility / len(consumption_results)

                    # Apply energy cost based on accessibility
                    energy_cost_multiplier = 1.0 + (1.0 - avg_accessibility) * 2.0
                    additional_energy_cost = (energy_cost_multiplier - 1.0) * 2.0
                    self.energy -= additional_energy_cost

        else:
            # Regular patch system
            available = patch.get_available_resources()

            if available > 0:
                # Apply proboscis-corolla matching efficiency
                collection_efficiency = 1.0
                energy_cost_multiplier = 1.0

                proboscis_system = getattr(self.model, "proboscis_system", None)
                if (
                    proboscis_system
                    and self.proboscis_characteristics
                    and hasattr(patch, "flower_species")
                ):
                    species_name = getattr(self.colony, "species", "Bombus_terrestris")

                    # Calculate weighted efficiency based on flower accessibility
                    total_efficiency = 0.0
                    total_weight = 0.0

                    for flower in patch.flower_species:
                        if (
                            flower.bloom_start
                            <= self.model.schedule.steps
                            <= flower.bloom_end
                        ):
                            accessibility_result = (
                                proboscis_system.calculate_accessibility(
                                    proboscis_system.get_species_proboscis(
                                        species_name
                                    ),
                                    flower,
                                )
                            )

                            if accessibility_result.is_accessible():
                                weight = (
                                    flower.flower_density * flower.nectar_production
                                )
                                total_efficiency += (
                                    accessibility_result.nectar_extraction_efficiency
                                    * weight
                                )
                                total_weight += weight

                    if total_weight > 0:
                        collection_efficiency = total_efficiency / total_weight
                        # Average energy cost for accessible flowers
                        energy_cost_multiplier = sum(
                            proboscis_system.calculate_accessibility(
                                proboscis_system.get_species_proboscis(species_name),
                                flower,
                            ).energy_cost_multiplier
                            for flower in patch.flower_species
                            if flower.bloom_start
                            <= self.model.schedule.steps
                            <= flower.bloom_end
                        ) / len(patch.flower_species)

                # Apply efficiency to collection rate
                effective_collection_rate = base_collection_rate * collection_efficiency

                collected = min(
                    effective_collection_rate,
                    available,
                    self.carrying_capacity - self.current_load,
                )

                self.current_load += collected
                self.resource_type = patch.primary_resource_type
                patch.consume_resources(collected)

                # Apply energy cost from proboscis-corolla matching
                additional_energy_cost = (energy_cost_multiplier - 1.0) * 2.0
                self.energy -= additional_energy_cost

        # Update foraging memory
        # Determine success based on whether we collected resources
        success = self.current_load > 0
        energy_gained = (
            self.current_load * 0.1
        )  # Rough estimate of energy from collected resources
        patch_id = getattr(patch, "patch_id", self.target_patch) or 0
        self.update_foraging_memory(patch_id, success, energy_gained)

        # Energy cost for foraging
        if self.colony is not None and self.colony.location is not None:
            distance = self.get_distance_to(self.colony.location)
            flight_cost = distance * 0.01
            self.energy -= flight_cost

    def update_foraging_memory(
        self, patch_id: int, success: bool, energy_gained: float
    ) -> None:
        """Update memory of foraging experience"""
        memory = next(
            (m for m in self.foraging_history if m.patch_id == patch_id), None
        )

        if memory is None:
            memory = ForagingMemory(
                patch_id=patch_id,
                quality=0.5,  # Default quality
                distance=0.0,  # Will be updated if needed
                resource_type="nectar",  # Default type
            )
            self.foraging_history.append(memory)

        memory.visits += 1
        memory.last_visit = self.model.schedule.steps
        if success:
            memory.quality = 0.9 * memory.quality + 0.1 * (
                energy_gained / 10.0
            )  # Normalize energy to quality

    def return_to_hive(self) -> None:
        """Return to hive and unload resources"""
        if self.colony and self.current_load > 0:
            self.location = self.colony.location
            self.colony.add_resources(self.resource_type, self.current_load)

            # Decide whether to dance
            if self.should_dance():
                self.perform_waggle_dance()

            self.current_load = 0.0
            self.resource_type = None

    def should_dance(self) -> bool:
        """Determine if forager should perform waggle dance"""
        if self.target_patch is None:
            return False

        # Dance if patch quality is above threshold
        patch = self.model.landscape.get_patch(self.target_patch)
        if patch is None:
            return False

        quality = patch.get_resource_quality()
        return bool(quality > self.dance_threshold)

    def perform_waggle_dance(self) -> None:
        """Perform waggle dance to communicate patch location"""
        if self.target_patch is None:
            return

        patch = self.model.landscape.get_patch(self.target_patch)
        if patch is None:
            return

        # Create dance information
        dance_info = DanceInfo(
            patch_id=self.target_patch,
            distance=self.get_distance_to(patch.location),
            direction=self.calculate_direction_to(patch.location),
            quality=patch.get_resource_quality(),
            resource_type=patch.primary_resource_type,
        )

        self.status = BeeStatus.DANCING

        # Recruit followers
        self.recruit_followers(dance_info)

    def calculate_direction_to(self, target_pos: Tuple[float, float]) -> float:
        """Calculate direction to target position in radians"""
        if self.location is None:
            return 0.0

        dx = target_pos[0] - self.location[0]
        dy = target_pos[1] - self.location[1]
        return math.atan2(dy, dx)

    def recruit_followers(self, dance_info: DanceInfo) -> None:
        """Recruit other bees to follow dance"""
        if self.colony is None:
            return

        colony_foragers = [
            agent
            for agent in self.colony.get_bees()
            if isinstance(agent, Forager) and agent != self
        ]

        # Recruitment probability based on dance quality
        recruitment_prob = min(0.5, dance_info.quality)

        for forager in colony_foragers:
            if (
                forager.target_patch is None
                and self.model.random.random() < recruitment_prob
            ):
                forager.following_dance = dance_info
                self.dance_followers.append(forager)


class Drone(BeeAgent):
    """
    Drone bee agent - male bees for reproduction.

    Maps to NetLogo drone breed with reproductive behaviors.
    """

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        colony: Any,
        genotype: Optional[Genotype] = None,
    ):
        super().__init__(unique_id, model, age=0, genotype=genotype)
        self.role = BeeRole.DRONE
        self.colony = colony
        self.reproductive_maturity = 14  # days
        self.can_mate = False

        # Genetic system integration - drones must be male
        if self.genotype and self.genotype.sex != Sex.MALE:
            raise ValueError("Drone genotype must be male (haploid)")

        # Ensure drone is male
        if self.genotype and self.genotype.ploidy != Ploidy.HAPLOID:
            raise ValueError("Drones must be haploid")

    def step(self) -> None:
        """Drone-specific step behavior"""
        super().step()
        if self.status == BeeStatus.ALIVE:
            self.check_maturity()
            self.drone_behavior()

    def check_maturity(self) -> None:
        """Check if drone has reached reproductive maturity"""
        if self.age >= self.reproductive_maturity:
            self.can_mate = True

    def drone_behavior(self) -> None:
        """Drone behavior - mostly resting except during mating flights"""
        if self.can_mate and self.model.is_mating_season():
            self.status = BeeStatus.FORAGING  # Mating flight
            self.energy -= 5.0  # High energy cost
        else:
            self.status = BeeStatus.RESTING

    def execute_role_behavior(self) -> None:
        """Execute drone-specific behavior"""
        self.consume_resources(2.0)  # Drones consume more resources
