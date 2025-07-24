"""
Advanced Patch Selection Algorithms for NetLogo BEE-STEWARD v2 Parity
====================================================================

Sophisticated patch selection algorithms matching NetLogo's complex foraging
decision-making including threshold-based selection, species-specific preferences,
and distance-quality tradeoffs.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from dataclasses import dataclass
import math
import numpy as np
from collections import defaultdict


class PatchSelectionStrategy(Enum):
    """Patch selection strategy types"""

    THRESHOLD_BASED = "threshold_based"
    DISTANCE_WEIGHTED = "distance_weighted"
    QUALITY_MAXIMIZING = "quality_maximizing"
    ENERGY_OPTIMIZING = "energy_optimizing"
    PROBABILISTIC = "probabilistic"
    MULTI_CRITERIA = "multi_criteria"


class ResourceType(Enum):
    """Resource types available in patches"""

    NECTAR = "nectar"
    POLLEN = "pollen"
    MIXED = "mixed"
    WATER = "water"
    PROPOLIS = "propolis"


class PatchQualityMetric(Enum):
    """Metrics for assessing patch quality"""

    RESOURCE_DENSITY = "resource_density"
    SUGAR_CONCENTRATION = "sugar_concentration"
    FLOWER_COUNT = "flower_count"
    ACCESSIBILITY = "accessibility"
    COMPETITION_LEVEL = "competition_level"
    HANDLING_TIME = "handling_time"
    COROLLA_DEPTH = "corolla_depth"
    COROLLA_WIDTH = "corolla_width"


@dataclass
class PatchInfo:
    """Information about a foraging patch"""

    patch_id: int
    location: Tuple[float, float]
    resource_type: ResourceType
    quality_metrics: Dict[PatchQualityMetric, float]
    species_compatibility: Dict[str, float]
    distance_from_hive: float
    current_foragers: int
    max_capacity: int
    depletion_rate: float
    regeneration_rate: float
    seasonal_availability: Dict[str, float]
    last_visited: Optional[int] = None
    total_visits: int = 0
    success_rate: float = 1.0


class SpeciesPreferences(BaseModel):
    """Species-specific foraging preferences"""

    model_config = {"validate_assignment": True}

    species_name: str = Field(description="Species name")

    # Resource preferences
    nectar_preference: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Nectar preference weight"
    )
    pollen_preference: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Pollen preference weight"
    )

    # Quality thresholds
    minimum_sugar_concentration: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Minimum sugar concentration"
    )
    quality_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum patch quality threshold"
    )

    # Distance preferences
    max_foraging_distance: float = Field(
        default=1000.0, ge=0.0, description="Maximum foraging distance (m)"
    )
    distance_penalty_factor: float = Field(
        default=0.1, ge=0.0, description="Distance penalty factor"
    )

    # Competition tolerance
    competition_tolerance: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Competition tolerance level"
    )
    crowding_threshold: int = Field(
        default=5, ge=0, description="Crowding threshold (foragers per patch)"
    )

    # Temporal preferences
    preferred_foraging_hours: List[int] = Field(
        default_factory=lambda: list(range(8, 18)),
        description="Preferred foraging hours",
    )
    weather_sensitivity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weather sensitivity factor"
    )

    # Flower morphology preferences
    preferred_corolla_depth: Tuple[float, float] = Field(
        default=(2.0, 8.0), description="Preferred corolla depth range (mm)"
    )
    proboscis_length: float = Field(
        default=6.0, ge=0.0, description="Proboscis length (mm)"
    )

    @field_validator("pollen_preference")
    @classmethod
    def validate_resource_preferences(cls, v: float, info: Any) -> float:
        # Ensure nectar + pollen preferences sum to 1.0
        if hasattr(info, "data"):
            nectar_pref = info.data.get("nectar_preference", 0.7)
            total = v + nectar_pref
            if abs(total - 1.0) > 0.1:  # Allow reasonable floating point errors
                raise ValueError(
                    f"Nectar and pollen preferences must sum to approximately 1.0, got {total}"
                )
        return v


class PatchQualityAssessment(BaseModel):
    """Comprehensive patch quality assessment"""

    model_config = {"validate_assignment": True}

    # Quality weights
    resource_density_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Resource density weight"
    )
    sugar_concentration_weight: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Sugar concentration weight"
    )
    accessibility_weight: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Accessibility weight"
    )
    competition_weight: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Competition weight"
    )
    handling_time_weight: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Handling time weight"
    )

    # Environmental factors
    weather_impact_factor: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Weather impact factor"
    )
    time_of_day_factor: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Time of day factor"
    )
    seasonal_factor: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Seasonal availability factor"
    )

    def calculate_overall_quality(
        self, patch: PatchInfo, current_conditions: Dict[str, Any]
    ) -> float:
        """Calculate overall patch quality score"""

        base_quality = 0.0

        # Resource density component
        resource_density = patch.quality_metrics.get(
            PatchQualityMetric.RESOURCE_DENSITY, 0.5
        )
        base_quality += resource_density * self.resource_density_weight

        # Sugar concentration component
        sugar_concentration = patch.quality_metrics.get(
            PatchQualityMetric.SUGAR_CONCENTRATION, 0.5
        )
        base_quality += sugar_concentration * self.sugar_concentration_weight

        # Accessibility component
        accessibility = patch.quality_metrics.get(PatchQualityMetric.ACCESSIBILITY, 0.5)
        base_quality += accessibility * self.accessibility_weight

        # Competition penalty
        competition_level = patch.quality_metrics.get(
            PatchQualityMetric.COMPETITION_LEVEL, 0.0
        )
        competition_penalty = competition_level * self.competition_weight
        base_quality -= competition_penalty

        # Handling time efficiency
        handling_time = patch.quality_metrics.get(PatchQualityMetric.HANDLING_TIME, 1.0)
        handling_efficiency = max(0.0, 1.0 - handling_time) * self.handling_time_weight
        base_quality += handling_efficiency

        # Apply environmental factors
        weather_condition = current_conditions.get("weather", "clear")
        if weather_condition in ["rain", "thunderstorm"]:
            base_quality *= self.weather_impact_factor

        # Time of day adjustment
        current_hour = current_conditions.get("hour", 12)
        if current_hour < 8 or current_hour > 18:  # Early morning or late evening
            base_quality *= self.time_of_day_factor

        # Seasonal adjustment
        current_season = current_conditions.get("season", "spring")
        seasonal_multiplier = patch.seasonal_availability.get(current_season, 1.0)
        base_quality *= seasonal_multiplier * self.seasonal_factor

        return max(0.0, min(1.0, base_quality))


class DistanceQualityTradeoff(BaseModel):
    """Distance-quality tradeoff calculations"""

    model_config = {"validate_assignment": True}

    # Tradeoff parameters
    distance_decay_rate: float = Field(
        default=0.001, ge=0.0, description="Distance decay rate"
    )
    energy_cost_per_meter: float = Field(
        default=0.01, ge=0.0, description="Energy cost per meter"
    )
    quality_sensitivity: float = Field(
        default=2.0, ge=0.0, description="Quality sensitivity factor"
    )

    # Optimization parameters
    min_acceptable_quality: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum acceptable quality"
    )
    max_travel_distance: float = Field(
        default=2000.0, ge=0.0, description="Maximum travel distance (m)"
    )

    def calculate_patch_utility(
        self, patch: PatchInfo, quality: float, bee_energy: float
    ) -> float:
        """Calculate patch utility considering distance-quality tradeoff"""

        # Base utility from quality
        quality_utility = quality**self.quality_sensitivity

        # Distance penalty
        distance_penalty = math.exp(
            -self.distance_decay_rate * patch.distance_from_hive
        )

        # Energy cost consideration
        round_trip_cost = 2 * patch.distance_from_hive * self.energy_cost_per_meter
        energy_feasibility = max(0.0, 1.0 - (round_trip_cost / bee_energy))

        # Combined utility
        utility = quality_utility * distance_penalty * energy_feasibility

        # Apply minimum quality filter
        if quality < self.min_acceptable_quality:
            utility *= 0.1  # Severe penalty for low quality

        # Apply maximum distance filter
        if patch.distance_from_hive > self.max_travel_distance:
            utility *= 0.1  # Severe penalty for excessive distance

        return float(utility)


class ThresholdBasedSelection(BaseModel):
    """Threshold-based patch selection algorithm"""

    model_config = {"validate_assignment": True}

    # Selection thresholds
    quality_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Quality threshold"
    )
    distance_threshold: float = Field(
        default=500.0, ge=0.0, description="Distance threshold (m)"
    )
    competition_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Competition threshold"
    )

    # Adaptive thresholds
    adaptive_thresholds: bool = Field(
        default=True, description="Use adaptive thresholds"
    )
    success_rate_influence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Success rate influence on thresholds"
    )

    def select_patches(
        self,
        available_patches: List[PatchInfo],
        patch_qualities: Dict[int, float],
        bee_memory: Dict[int, Any],
    ) -> List[PatchInfo]:
        """Select patches based on threshold criteria"""

        qualified_patches = []

        # Adapt thresholds based on past success
        current_quality_threshold = self.quality_threshold
        current_distance_threshold = self.distance_threshold

        if self.adaptive_thresholds and bee_memory:
            avg_success_rate = self._calculate_average_success_rate(bee_memory)

            # Lower thresholds if success rate is low
            if avg_success_rate < 0.5:
                current_quality_threshold *= 0.8
                current_distance_threshold *= 1.2
            # Raise thresholds if success rate is high
            elif avg_success_rate > 0.8:
                current_quality_threshold *= 1.1
                current_distance_threshold *= 0.9

        # Apply threshold filters
        for patch in available_patches:
            quality = patch_qualities.get(patch.patch_id, 0.0)

            # Quality threshold
            if quality < current_quality_threshold:
                continue

            # Distance threshold
            if patch.distance_from_hive > current_distance_threshold:
                continue

            # Competition threshold
            competition_level = patch.quality_metrics.get(
                PatchQualityMetric.COMPETITION_LEVEL, 0.0
            )
            if competition_level > self.competition_threshold:
                continue

            # Capacity check
            if patch.current_foragers >= patch.max_capacity:
                continue

            qualified_patches.append(patch)

        return qualified_patches

    def _calculate_average_success_rate(self, bee_memory: Dict[int, Any]) -> float:
        """Calculate average success rate from bee memory"""

        if not bee_memory:
            return 0.5  # Default neutral success rate

        success_rates = []
        for patch_id, memory in bee_memory.items():
            if hasattr(memory, "success_rate"):
                success_rates.append(memory.success_rate)

        return sum(success_rates) / len(success_rates) if success_rates else 0.5


class AdvancedPatchSelector(BaseModel):
    """Advanced patch selection system matching NetLogo complexity"""

    model_config = {"validate_assignment": True}

    # Component systems
    quality_assessment: PatchQualityAssessment = Field(
        default_factory=PatchQualityAssessment
    )
    distance_tradeoff: DistanceQualityTradeoff = Field(
        default_factory=DistanceQualityTradeoff
    )
    threshold_selection: ThresholdBasedSelection = Field(
        default_factory=ThresholdBasedSelection
    )

    # Selection parameters
    default_strategy: PatchSelectionStrategy = Field(
        default=PatchSelectionStrategy.MULTI_CRITERIA
    )
    max_patches_to_consider: int = Field(
        default=20, ge=1, description="Maximum patches to consider"
    )

    # Internal state
    species_preferences: Dict[str, SpeciesPreferences] = Field(
        default_factory=dict, description="Species preferences"
    )
    selection_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Selection history"
    )
    selection_performance: Dict[str, float] = Field(
        default_factory=dict, description="Selection performance metrics"
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Initialize default species preferences if not provided
        if not self.species_preferences:
            self._initialize_default_species()

    def _initialize_default_species(self) -> None:
        """Initialize default species preferences"""

        # Generic bee species preferences (compatibility layer)
        self.species_preferences["apis_mellifera"] = SpeciesPreferences(
            species_name="apis_mellifera",
            nectar_preference=0.7,
            pollen_preference=0.3,
            minimum_sugar_concentration=0.2,
            quality_threshold=0.5,
            max_foraging_distance=1500.0,
            proboscis_length=6.5,
        )

        # Bumblebee preferences (example)
        self.species_preferences["bombus_terrestris"] = SpeciesPreferences(
            species_name="bombus_terrestris",
            nectar_preference=0.5,
            pollen_preference=0.5,
            minimum_sugar_concentration=0.15,
            quality_threshold=0.4,
            max_foraging_distance=800.0,
            proboscis_length=8.0,
            competition_tolerance=0.5,
        )

    def select_optimal_patches(
        self,
        available_patches: List[PatchInfo],
        bee_species: str,
        bee_energy: float,
        bee_memory: Dict[int, Any],
        current_conditions: Dict[str, Any],
        strategy: Optional[PatchSelectionStrategy] = None,
    ) -> List[PatchInfo]:
        """Select optimal patches using advanced algorithms"""

        if not available_patches:
            return []

        # Get species preferences
        species_prefs = self.species_preferences.get(
            bee_species, self.species_preferences["apis_mellifera"]
        )

        # Use specified strategy or default
        selection_strategy = strategy or self.default_strategy

        # Filter patches by basic criteria
        feasible_patches = self._filter_feasible_patches(
            available_patches, species_prefs, current_conditions
        )

        if not feasible_patches:
            return []

        # Limit consideration set
        if len(feasible_patches) > self.max_patches_to_consider:
            feasible_patches = feasible_patches[: self.max_patches_to_consider]

        # Calculate patch qualities
        patch_qualities = {}
        for patch in feasible_patches:
            quality = self.quality_assessment.calculate_overall_quality(
                patch, current_conditions
            )
            patch_qualities[patch.patch_id] = quality

        # Apply selection strategy
        selected_patches = self._apply_selection_strategy(
            feasible_patches,
            patch_qualities,
            species_prefs,
            bee_energy,
            bee_memory,
            selection_strategy,
        )

        # Record selection history
        self._record_selection(selected_patches, patch_qualities, selection_strategy)

        return selected_patches

    def _filter_feasible_patches(
        self,
        patches: List[PatchInfo],
        species_prefs: SpeciesPreferences,
        current_conditions: Dict[str, Any],
    ) -> List[PatchInfo]:
        """Filter patches by basic feasibility criteria"""

        feasible = []

        for patch in patches:
            # Distance filter
            if patch.distance_from_hive > species_prefs.max_foraging_distance:
                continue

            # Resource type compatibility
            if (
                patch.resource_type == ResourceType.NECTAR
                and species_prefs.nectar_preference < 0.1
            ):
                continue
            if (
                patch.resource_type == ResourceType.POLLEN
                and species_prefs.pollen_preference < 0.1
            ):
                continue

            # Sugar concentration filter
            sugar_conc = patch.quality_metrics.get(
                PatchQualityMetric.SUGAR_CONCENTRATION, 0.0
            )
            if sugar_conc < species_prefs.minimum_sugar_concentration:
                continue

            # Capacity filter
            if patch.current_foragers >= patch.max_capacity:
                continue

            # Flower morphology compatibility
            if self._check_flower_compatibility(patch, species_prefs):
                feasible.append(patch)

        return feasible

    def _check_flower_compatibility(
        self, patch: PatchInfo, species_prefs: SpeciesPreferences
    ) -> bool:
        """Check if bee can physically access flowers in patch"""

        # Get species compatibility score
        species_compatibility = patch.species_compatibility.get(
            species_prefs.species_name, 0.5
        )

        # Check proboscis-corolla compatibility
        if hasattr(patch, "corolla_depth"):
            corolla_depth = patch.corolla_depth
            min_depth, max_depth = species_prefs.preferred_corolla_depth

            # Check if proboscis can reach nectar
            if corolla_depth > species_prefs.proboscis_length:
                return False

            # Check if corolla depth is in preferred range
            if not (min_depth <= corolla_depth <= max_depth):
                species_compatibility *= 0.7  # Reduced compatibility

        return species_compatibility > 0.2  # Minimum compatibility threshold

    def _apply_selection_strategy(
        self,
        patches: List[PatchInfo],
        patch_qualities: Dict[int, float],
        species_prefs: SpeciesPreferences,
        bee_energy: float,
        bee_memory: Dict[int, Any],
        strategy: PatchSelectionStrategy,
    ) -> List[PatchInfo]:
        """Apply specific selection strategy"""

        if strategy == PatchSelectionStrategy.THRESHOLD_BASED:
            return self.threshold_selection.select_patches(
                patches, patch_qualities, bee_memory
            )

        elif strategy == PatchSelectionStrategy.DISTANCE_WEIGHTED:
            return self._distance_weighted_selection(
                patches, patch_qualities, bee_energy
            )

        elif strategy == PatchSelectionStrategy.QUALITY_MAXIMIZING:
            return self._quality_maximizing_selection(patches, patch_qualities)

        elif strategy == PatchSelectionStrategy.ENERGY_OPTIMIZING:
            return self._energy_optimizing_selection(
                patches, patch_qualities, bee_energy
            )

        elif strategy == PatchSelectionStrategy.PROBABILISTIC:
            return self._probabilistic_selection(
                patches, patch_qualities, species_prefs
            )

        elif strategy == PatchSelectionStrategy.MULTI_CRITERIA:
            return self._multi_criteria_selection(
                patches, patch_qualities, species_prefs, bee_energy
            )

    def _distance_weighted_selection(
        self,
        patches: List[PatchInfo],
        patch_qualities: Dict[int, float],
        bee_energy: float,
    ) -> List[PatchInfo]:
        """Select patches with distance-weighted quality"""

        scored_patches = []

        for patch in patches:
            quality = patch_qualities.get(patch.patch_id, 0.0)
            utility = self.distance_tradeoff.calculate_patch_utility(
                patch, quality, bee_energy
            )
            scored_patches.append((patch, utility))

        # Sort by utility score
        scored_patches.sort(key=lambda x: x[1], reverse=True)

        # Return top patches
        return [patch for patch, _ in scored_patches[:3]]

    def _quality_maximizing_selection(
        self, patches: List[PatchInfo], patch_qualities: Dict[int, float]
    ) -> List[PatchInfo]:
        """Select highest quality patches"""

        # Sort by quality
        sorted_patches = sorted(
            patches, key=lambda p: patch_qualities.get(p.patch_id, 0.0), reverse=True
        )

        # Return top 3 patches
        return sorted_patches[:3]

    def _energy_optimizing_selection(
        self,
        patches: List[PatchInfo],
        patch_qualities: Dict[int, float],
        bee_energy: float,
    ) -> List[PatchInfo]:
        """Select patches optimizing energy efficiency"""

        energy_efficient_patches = []

        for patch in patches:
            quality = patch_qualities.get(patch.patch_id, 0.0)

            # Calculate energy efficiency
            travel_cost = (
                2
                * patch.distance_from_hive
                * self.distance_tradeoff.energy_cost_per_meter
            )
            expected_gain = quality * 50.0  # Assume quality correlates with energy gain

            if travel_cost < bee_energy and expected_gain > travel_cost:
                efficiency = expected_gain / travel_cost
                energy_efficient_patches.append((patch, efficiency))

        # Sort by efficiency
        energy_efficient_patches.sort(key=lambda x: x[1], reverse=True)

        return [patch for patch, _ in energy_efficient_patches[:3]]

    def _probabilistic_selection(
        self,
        patches: List[PatchInfo],
        patch_qualities: Dict[int, float],
        species_prefs: SpeciesPreferences,
    ) -> List[PatchInfo]:
        """Select patches probabilistically based on quality"""

        # Calculate selection probabilities
        qualities = [patch_qualities.get(p.patch_id, 0.0) for p in patches]

        if not qualities or sum(qualities) == 0:
            return patches[:1] if patches else []

        # Normalize probabilities
        total_quality = sum(qualities)
        probabilities = [q / total_quality for q in qualities]

        # Select patches probabilistically
        selected_patches = []
        for _ in range(min(3, len(patches))):
            if patches:
                # Use indices for numpy choice
                indices = np.arange(len(patches))
                selected_index = np.random.choice(indices, p=probabilities)
                selected_patch = patches[selected_index]
                selected_patches.append(selected_patch)

                # Remove selected patch from further selection
                patches.pop(selected_index)
                probabilities.pop(selected_index)

                # Renormalize probabilities
                if probabilities:
                    total_prob = sum(probabilities)
                    probabilities = [p / total_prob for p in probabilities]

        return selected_patches

    def _multi_criteria_selection(
        self,
        patches: List[PatchInfo],
        patch_qualities: Dict[int, float],
        species_prefs: SpeciesPreferences,
        bee_energy: float,
    ) -> List[PatchInfo]:
        """Select patches using multi-criteria decision analysis"""

        scored_patches = []

        for patch in patches:
            quality = patch_qualities.get(patch.patch_id, 0.0)

            # Multi-criteria scoring
            quality_score = quality * 0.4

            # Distance score (closer is better)
            distance_score = (
                max(
                    0,
                    1.0
                    - (patch.distance_from_hive / species_prefs.max_foraging_distance),
                )
                * 0.3
            )

            # Energy efficiency score
            travel_cost = (
                2
                * patch.distance_from_hive
                * self.distance_tradeoff.energy_cost_per_meter
            )
            energy_score = max(0, 1.0 - (travel_cost / bee_energy)) * 0.2

            # Competition score (less competition is better)
            competition_level = patch.quality_metrics.get(
                PatchQualityMetric.COMPETITION_LEVEL, 0.0
            )
            competition_score = (1.0 - competition_level) * 0.1

            # Combined score
            total_score = (
                quality_score + distance_score + energy_score + competition_score
            )

            scored_patches.append((patch, total_score))

        # Sort by total score
        scored_patches.sort(key=lambda x: x[1], reverse=True)

        # Return top patches
        return [patch for patch, _ in scored_patches[:3]]

    def _record_selection(
        self,
        selected_patches: List[PatchInfo],
        patch_qualities: Dict[int, float],
        strategy: PatchSelectionStrategy,
    ) -> None:
        """Record selection for performance tracking"""

        selection_record = {
            "strategy": strategy.value,
            "patch_count": len(selected_patches),
            "average_quality": sum(
                patch_qualities.get(p.patch_id, 0.0) for p in selected_patches
            )
            / len(selected_patches)
            if selected_patches
            else 0.0,
            "average_distance": sum(p.distance_from_hive for p in selected_patches)
            / len(selected_patches)
            if selected_patches
            else 0.0,
            "patch_ids": [p.patch_id for p in selected_patches],
        }

        self.selection_history.append(selection_record)

        # Maintain history size
        if len(self.selection_history) > 1000:
            self.selection_history.pop(0)

    def get_selection_performance(self) -> Dict[str, Any]:
        """Get performance metrics for patch selection"""

        if not self.selection_history:
            return {}

        # Calculate performance metrics
        strategies = defaultdict(list)
        for record in self.selection_history:
            strategies[record["strategy"]].append(record)

        performance = {}
        for strategy, records in strategies.items():
            if records:
                avg_quality = sum(r["average_quality"] for r in records) / len(records)
                avg_distance = sum(r["average_distance"] for r in records) / len(
                    records
                )
                avg_patch_count = sum(r["patch_count"] for r in records) / len(records)

                performance[strategy] = {
                    "average_quality": avg_quality,
                    "average_distance": avg_distance,
                    "average_patch_count": avg_patch_count,
                    "selection_count": len(records),
                }

        return performance

    def add_species_preferences(
        self, species_name: str, preferences: SpeciesPreferences
    ) -> None:
        """Add species-specific preferences"""
        self.species_preferences[species_name] = preferences

    def update_patch_info(self, patch_id: int, updated_info: Dict[str, Any]) -> None:
        """Update patch information based on recent observations"""
        # This would be called when bees return from foraging with updated patch info
        pass

    def get_patch_recommendations(
        self, bee_species: str, current_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get patch recommendations for a species"""

        species_prefs = self.species_preferences.get(
            bee_species, self.species_preferences["apis_mellifera"]
        )

        recommendations = []

        # Add general recommendations based on species preferences
        recommendations.append(
            {
                "type": "resource_preference",
                "message": f"Focus on {'nectar' if species_prefs.nectar_preference > 0.5 else 'pollen'} sources",
                "priority": "high",
            }
        )

        recommendations.append(
            {
                "type": "distance_guidance",
                "message": f"Stay within {species_prefs.max_foraging_distance}m of hive",
                "priority": "medium",
            }
        )

        # Weather-based recommendations
        weather = current_conditions.get("weather", "clear")
        if weather in ["rain", "thunderstorm"]:
            recommendations.append(
                {
                    "type": "weather_warning",
                    "message": "Avoid foraging in current weather conditions",
                    "priority": "high",
                }
            )

        return recommendations
