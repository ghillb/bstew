"""
Masterpatch/Layer Resource System for BSTEW
==========================================

Implements the NetLogo masterpatch system where geographic locations
can contain multiple species-specific resource layers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging
import math
from datetime import datetime

from ..spatial.patches import FlowerSpecies, HabitatType
from .flower_community_layers import (
    FlowerCommunity,
    VerticalLayer,
    CommunityLayerSystem,
)


class LayerType(Enum):
    """Types of resource layers"""

    FLOWER_SPECIES = "flower_species"
    HABITAT_BASE = "habitat_base"
    MANAGEMENT = "management"
    SEASONAL = "seasonal"


class ResourceLayer(BaseModel):
    """Individual resource layer within a masterpatch"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    layer_id: str = Field(description="Unique layer identifier")
    layer_type: LayerType = Field(description="Type of resource layer")
    flower_species: FlowerSpecies = Field(description="Flower species in this layer")
    coverage_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of patch covered by this layer",
    )
    temporal_availability: List[float] = Field(
        default_factory=list, description="Daily availability (365 values)"
    )
    management_effects: Dict[str, float] = Field(
        default_factory=dict, description="Management effect factors"
    )

    # Current state
    current_nectar: float = Field(
        default=0.0, ge=0.0, description="Current nectar availability"
    )
    current_pollen: float = Field(
        default=0.0, ge=0.0, description="Current pollen availability"
    )
    depletion_factor: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Resource depletion from foraging"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize temporal availability after model creation"""
        if not self.temporal_availability:
            # Initialize with bloom period availability
            self.temporal_availability = [0.0] * 365
            for day in range(
                self.flower_species.bloom_start, self.flower_species.bloom_end + 1
            ):
                if day < 365:
                    self.temporal_availability[day] = 1.0

    def get_availability(self, day_of_year: int) -> float:
        """Get layer availability for specific day"""
        if 0 <= day_of_year < len(self.temporal_availability):
            return self.temporal_availability[day_of_year]
        return 0.0

    def update_resources(
        self, day_of_year: int, weather_factor: float, area_m2: float
    ) -> None:
        """Update resource production for this layer"""
        availability = self.get_availability(day_of_year)

        if availability > 0:
            # Calculate base production
            flowers_per_patch = (
                self.flower_species.flower_density * area_m2 * self.coverage_fraction
            )

            base_nectar = (
                flowers_per_patch
                * self.flower_species.nectar_production
                * availability
                * weather_factor
                * self.depletion_factor
            )
            base_pollen = (
                flowers_per_patch
                * self.flower_species.pollen_production
                * availability
                * weather_factor
                * self.depletion_factor
            )

            # Apply management effects
            management_factor = 1.0
            for effect, value in self.management_effects.items():
                if effect == "fertilizer":
                    management_factor *= 1.0 + value
                elif effect == "mowing":
                    management_factor *= 1.0 - value
                elif effect == "pesticide":
                    management_factor *= 1.0 - value * 0.8

            self.current_nectar = base_nectar * management_factor
            self.current_pollen = base_pollen * management_factor
        else:
            self.current_nectar = 0.0
            self.current_pollen = 0.0

    def consume_resources(self, nectar_consumed: float, pollen_consumed: float) -> None:
        """Consume resources and update depletion"""
        self.current_nectar = max(0.0, self.current_nectar - nectar_consumed)
        self.current_pollen = max(0.0, self.current_pollen - pollen_consumed)

        # Update depletion factor based on consumption
        total_available = self.current_nectar + self.current_pollen
        total_consumed = nectar_consumed + pollen_consumed

        if total_available > 0:
            consumption_rate = total_consumed / total_available
            self.depletion_factor = max(
                0.1, self.depletion_factor - consumption_rate * 0.1
            )

    def recover_resources(self, recovery_rate: float = 0.05) -> None:
        """Recover from depletion over time"""
        self.depletion_factor = min(1.0, self.depletion_factor + recovery_rate)


class MasterPatch(BaseModel):
    """Master patch containing multiple resource layers"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    patch_id: str = Field(description="Unique patch identifier")
    location: Tuple[float, float] = Field(description="Patch location coordinates")
    area_m2: float = Field(ge=0.0, description="Patch area in square meters")
    habitat_type: HabitatType = Field(description="Primary habitat type")

    # Layer management
    layers: Dict[str, ResourceLayer] = Field(
        default_factory=dict, description="Resource layers in patch"
    )
    layer_priorities: List[str] = Field(
        default_factory=list, description="Layer priority order"
    )

    # Patch properties
    elevation_m: float = Field(default=0.0, description="Elevation in meters")
    slope_degrees: float = Field(
        default=0.0, ge=0.0, le=90.0, description="Slope in degrees"
    )
    aspect_degrees: float = Field(
        default=0.0,
        ge=0.0,
        lt=360.0,
        description="Aspect in degrees (0=north, 90=east)",
    )
    soil_quality: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Soil quality factor"
    )
    water_availability: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Water availability factor"
    )

    # Management status
    management_regime: str = Field(
        default="none", description="Current management regime"
    )
    last_management_day: int = Field(default=-1, description="Last management day")
    management_intensity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Management intensity"
    )

    # Foraging activity
    foraging_pressure: float = Field(
        default=0.0, ge=0.0, description="Current foraging pressure"
    )
    last_visited_day: int = Field(default=-1, description="Last visited day")
    visitor_count: int = Field(default=0, ge=0, description="Total visitor count")

    # Hierarchical community integration
    flower_community: Optional[FlowerCommunity] = Field(
        default=None, description="Associated hierarchical flower community"
    )
    community_integration_mode: str = Field(
        default="legacy",
        description="Integration mode: 'legacy', 'community', 'hybrid'",
    )
    vertical_layer_mapping: Dict[str, VerticalLayer] = Field(
        default_factory=dict, description="Mapping of legacy layers to vertical layers"
    )

    def add_layer(self, layer: ResourceLayer, priority: int = 0) -> None:
        """Add resource layer to patch"""
        self.layers[layer.layer_id] = layer

        # Insert at priority position
        if priority < len(self.layer_priorities):
            self.layer_priorities.insert(priority, layer.layer_id)
        else:
            self.layer_priorities.append(layer.layer_id)

    def remove_layer(self, layer_id: str) -> None:
        """Remove resource layer from patch"""
        if layer_id in self.layers:
            del self.layers[layer_id]
            if layer_id in self.layer_priorities:
                self.layer_priorities.remove(layer_id)

    def get_layer_by_species(self, species_name: str) -> Optional[ResourceLayer]:
        """Get layer by flower species name"""
        for layer in self.layers.values():
            if layer.flower_species.name == species_name:
                return layer
        return None

    def get_available_species(self, day_of_year: int) -> List[FlowerSpecies]:
        """Get all flower species available on given day"""
        available_species = []

        for layer in self.layers.values():
            if layer.get_availability(day_of_year) > 0:
                available_species.append(layer.flower_species)

        return available_species

    def update_all_layers(self, day_of_year: int, weather_factor: float) -> None:
        """Update all resource layers"""
        for layer in self.layers.values():
            layer.update_resources(day_of_year, weather_factor, self.area_m2)
            layer.recover_resources()

    def get_total_resources(self) -> Tuple[float, float]:
        """Get total nectar and pollen across all layers"""
        total_nectar = sum(layer.current_nectar for layer in self.layers.values())
        total_pollen = sum(layer.current_pollen for layer in self.layers.values())
        return total_nectar, total_pollen

    def get_resource_quality(self) -> float:
        """Calculate overall resource quality"""
        total_nectar, total_pollen = self.get_total_resources()
        total_resources = total_nectar + total_pollen

        if total_resources == 0:
            return 0.0

        # Weight by species attractiveness and accessibility
        weighted_quality = 0.0
        total_weight = 0.0

        for layer in self.layers.values():
            layer_resources = layer.current_nectar + layer.current_pollen
            if layer_resources > 0:
                weight = layer_resources
                quality = (
                    layer.flower_species.attractiveness
                    * layer.flower_species.nectar_accessibility
                )
                weighted_quality += quality * weight
                total_weight += weight

        return weighted_quality / total_weight if total_weight > 0 else 0.0

    def consume_resources_from_species(
        self, species_name: str, nectar_amount: float, pollen_amount: float
    ) -> Tuple[float, float]:
        """Consume resources from specific species layer"""
        layer = self.get_layer_by_species(species_name)
        if layer:
            actual_nectar = min(nectar_amount, layer.current_nectar)
            actual_pollen = min(pollen_amount, layer.current_pollen)

            layer.consume_resources(actual_nectar, actual_pollen)

            # Update foraging pressure
            self.foraging_pressure += (actual_nectar + actual_pollen) / 1000.0

            return actual_nectar, actual_pollen

        return 0.0, 0.0

    def apply_management(
        self, management_type: str, intensity: float, day_of_year: int
    ) -> None:
        """Apply management practices to patch"""
        self.management_regime = management_type
        self.management_intensity = intensity
        self.last_management_day = day_of_year

        # Apply to all layers
        for layer in self.layers.values():
            if management_type == "mowing":
                layer.management_effects["mowing"] = intensity
                # Reset temporal availability for affected period
                for day in range(day_of_year, min(day_of_year + 30, 365)):
                    layer.temporal_availability[day] *= 1.0 - intensity

            elif management_type == "fertilizer":
                layer.management_effects["fertilizer"] = intensity

            elif management_type == "pesticide":
                layer.management_effects["pesticide"] = intensity

            elif management_type == "sowing":
                # Add new species layers for sown species
                if layer.flower_species.name in ["Phacelia", "Borage", "White Clover"]:
                    layer.coverage_fraction = min(
                        1.0, layer.coverage_fraction + intensity
                    )

    def get_species_accessibility_for_bee(
        self, bee_species: str, proboscis_system: Any
    ) -> Dict[str, float]:
        """Get accessibility scores for all species in patch"""
        accessibility_scores = {}

        for layer in self.layers.values():
            if layer.current_nectar > 0 or layer.current_pollen > 0:
                proboscis = proboscis_system.get_species_proboscis(bee_species)
                result = proboscis_system.calculate_accessibility(
                    proboscis, layer.flower_species
                )
                accessibility_scores[layer.flower_species.name] = (
                    result.accessibility_score
                )

        return accessibility_scores

    def initialize_flower_community(
        self,
        community_type: str = "mixed_grassland",
        community_system: Optional["CommunityLayerSystem"] = None,
    ) -> None:
        """Initialize hierarchical flower community for this patch"""

        if community_system is None:
            community_system = CommunityLayerSystem()

        # Create community template based on habitat type
        template_mapping = {
            HabitatType.GRASSLAND: "uk_chalk_grassland",
            HabitatType.WOODLAND: "woodland_edge",
            HabitatType.HEDGEROW: "woodland_edge",
            HabitatType.WILDFLOWER: "uk_chalk_grassland",
            HabitatType.CROPLAND: "uk_chalk_grassland",
        }

        template_name = template_mapping.get(self.habitat_type, "uk_chalk_grassland")

        try:
            # Create flower community
            self.flower_community = community_system.create_community_from_template(
                community_id=f"community_{self.patch_id}",
                template_name=template_name,
                location=self.location,
                area_m2=self.area_m2,
                customizations={
                    "microclimate": {
                        "temperature_buffer": self.elevation_m
                        / 1000.0,  # Higher = cooler
                        "wind_protection": min(
                            1.0, self.slope_degrees / 45.0
                        ),  # Steeper = more protected
                    }
                },
            )

            # Switch to community integration mode
            self.community_integration_mode = "community"

            # Create mapping between legacy layers and vertical layers
            self._create_vertical_layer_mapping()

        except Exception as e:
            # Fall back to legacy mode if community creation fails
            self.community_integration_mode = "legacy"
            logging.getLogger(__name__).warning(
                f"Failed to initialize community for patch {self.patch_id}: {e}"
            )

    def _create_vertical_layer_mapping(self) -> None:
        """Create mapping between legacy resource layers and vertical layers"""

        if not self.flower_community:
            return

        # Map existing resource layers to vertical layers based on species characteristics
        for layer_id, resource_layer in self.layers.items():
            species = resource_layer.flower_species

            # Determine vertical layer based on species characteristics
            if hasattr(species, "height_m"):
                height = species.height_m
                if height >= 3.0:
                    vertical_layer = VerticalLayer.CANOPY
                elif height >= 1.0:
                    vertical_layer = VerticalLayer.UNDERSTORY
                else:
                    vertical_layer = VerticalLayer.GROUND
            else:
                # Default mapping based on species name/type
                if species.name in ["Hawthorn", "Blackthorn", "Lime Tree"]:
                    vertical_layer = VerticalLayer.CANOPY
                elif species.name in ["Rose", "Bramble", "Gorse"]:
                    vertical_layer = VerticalLayer.UNDERSTORY
                else:
                    vertical_layer = VerticalLayer.GROUND

            self.vertical_layer_mapping[layer_id] = vertical_layer

    def enable_hybrid_integration(self) -> None:
        """Enable hybrid mode combining legacy layers with community structure"""

        if not self.flower_community:
            self.initialize_flower_community()

        if self.flower_community:
            self.community_integration_mode = "hybrid"

            # Synchronize legacy layers with community layers
            self._synchronize_with_community()

    def _synchronize_with_community(self) -> None:
        """Synchronize legacy resource layers with community structure"""

        if not self.flower_community or self.community_integration_mode != "hybrid":
            return

        # Update legacy layers based on community resource distribution
        community_resources = self.flower_community.update_community_resources(
            day_of_year=datetime.now().timetuple().tm_yday
        )

        # Distribute community resources to legacy layers
        total_legacy_resources = sum(
            layer.current_nectar + layer.current_pollen
            for layer in self.layers.values()
        )

        if total_legacy_resources > 0:
            nectar_multiplier = (
                community_resources["nectar_production"] / total_legacy_resources
            )
            pollen_multiplier = (
                community_resources["pollen_production"] / total_legacy_resources
            )

            for layer in self.layers.values():
                layer.current_nectar *= nectar_multiplier
                layer.current_pollen *= pollen_multiplier

    def get_hierarchical_resources(self, day_of_year: int) -> Dict[str, Any]:
        """Get resources using hierarchical community model"""

        if self.community_integration_mode == "legacy" or not self.flower_community:
            # Fall back to legacy resource calculation
            return self._get_legacy_resources(day_of_year)

        elif self.community_integration_mode == "community":
            # Use pure community model
            return self.flower_community.update_community_resources(day_of_year)

        elif self.community_integration_mode == "hybrid":
            # Combine community and legacy models
            community_resources = self.flower_community.update_community_resources(
                day_of_year
            )
            legacy_resources = self._get_legacy_resources(day_of_year)

            # Weighted combination (70% community, 30% legacy for smooth transition)
            return {
                "nectar_production": (
                    community_resources["nectar_production"] * 0.7
                    + legacy_resources["nectar_production"] * 0.3
                ),
                "pollen_production": (
                    community_resources["pollen_production"] * 0.7
                    + legacy_resources["pollen_production"] * 0.3
                ),
                "integration_mode": "hybrid",
                "community_efficiency": community_resources.get(
                    "community_efficiency", 1.0
                ),
                "legacy_contribution": 0.3,
                "community_contribution": 0.7,
            }

        return {}

    def _get_legacy_resources(self, day_of_year: int) -> Dict[str, float]:
        """Get resources using legacy layer model"""

        total_nectar = sum(layer.current_nectar for layer in self.layers.values())
        total_pollen = sum(layer.current_pollen for layer in self.layers.values())

        return {
            "nectar_production": total_nectar,
            "pollen_production": total_pollen,
            "legacy_mode": True,
        }

    def get_vertical_layer_resources(
        self, vertical_layer: VerticalLayer, day_of_year: int
    ) -> Dict[str, float]:
        """Get resources for specific vertical layer"""

        if not self.flower_community:
            return {"nectar_production": 0.0, "pollen_production": 0.0}

        if vertical_layer not in self.flower_community.layers:
            return {"nectar_production": 0.0, "pollen_production": 0.0}

        layer = self.flower_community.layers[vertical_layer]
        competition_effects = self.flower_community.calculate_layer_interactions()
        layer_competition = competition_effects.get(vertical_layer, {})

        return layer.get_resource_production(
            day_of_year, self.flower_community.base_conditions, layer_competition
        )

    def simulate_community_succession(self, years: int = 5) -> Dict[str, Any]:
        """Simulate succession in hierarchical community"""

        if not self.flower_community:
            return {"error": "No flower community initialized"}

        succession_history = self.flower_community.simulate_succession(years)
        return {"succession_history": succession_history, "years": years}

    def get_biodiversity_metrics(self) -> Dict[str, Any]:
        """Get biodiversity metrics from hierarchical community"""

        if not self.flower_community:
            # Calculate legacy biodiversity
            species_count = len(
                set(layer.flower_species.name for layer in self.layers.values())
            )
            return {
                "species_count": species_count,
                "diversity_index": min(1.0, species_count / 10.0),  # Simple diversity
                "mode": "legacy",
            }

        return {
            "diversity_index": self.flower_community.current_diversity_index,
            "stability_index": self.flower_community.stability_index,
            "resilience_score": self.flower_community.resilience_score,
            "layer_count": len(self.flower_community.layers),
            "species_count": sum(
                len(layer.dominant_species)
                + len(layer.subordinate_species)
                + len(layer.rare_species)
                for layer in self.flower_community.layers.values()
            ),
            "mode": "community",
        }

    def apply_disturbance(
        self, disturbance_type: str, intensity: float
    ) -> Dict[str, Any]:
        """Apply disturbance to both legacy and community layers"""

        results: Dict[str, Any] = {"legacy_effects": {}, "community_effects": {}}

        # Apply to legacy layers
        for layer_id, layer in self.layers.items():
            if disturbance_type == "mowing":
                # Reduce current resources
                layer.current_nectar *= 1.0 - intensity
                layer.current_pollen *= 1.0 - intensity
                results["legacy_effects"][layer_id] = {"resource_reduction": intensity}

            elif disturbance_type == "drought":
                # Reduce water-dependent production
                layer.depletion_factor *= 1.0 - intensity * 0.5
                results["legacy_effects"][layer_id] = {
                    "depletion_increase": intensity * 0.5
                }

        # Apply to community if present
        if self.flower_community:
            community_disturbance = self.flower_community._apply_random_disturbance()
            results["community_effects"] = community_disturbance

        return results


class MasterPatchSystem:
    """
    Manages the masterpatch/layer resource system.

    Implements:
    - Geographic locations (masterpatches) with multiple species layers
    - Species-specific resource availability and management
    - Temporal resource dynamics with complex interactions
    - Management effects on different species layers
    - Foraging pressure and resource depletion
    """

    def __init__(self, landscape_bounds: Tuple[float, float, float, float]):
        self.landscape_bounds = landscape_bounds
        self.masterpatches: Dict[str, MasterPatch] = {}
        self.species_database: Dict[str, FlowerSpecies] = {}

        # Management schedules
        self.management_calendar: Dict[int, List[Dict[str, Any]]] = {}

        # System parameters
        self.patch_size_m2 = 100.0  # Default patch size
        self.max_layers_per_patch = 10
        self.resource_recovery_rate = 0.02  # Daily recovery rate

        self.logger = logging.getLogger(__name__)

    def initialize_species_database(self, species_list: List[FlowerSpecies]) -> None:
        """Initialize species database"""
        for species in species_list:
            self.species_database[species.name] = species

    def create_masterpatch(
        self,
        patch_id: str,
        location: Tuple[float, float],
        area_m2: float,
        habitat_type: HabitatType,
    ) -> MasterPatch:
        """Create new master patch"""

        patch = MasterPatch(
            patch_id=patch_id,
            location=location,
            area_m2=area_m2,
            habitat_type=habitat_type,
        )

        self.masterpatches[patch_id] = patch
        return patch

    def add_species_layer_to_patch(
        self, patch_id: str, species_name: str, coverage_fraction: float = 1.0
    ) -> bool:
        """Add species layer to existing patch"""

        if patch_id not in self.masterpatches:
            return False

        if species_name not in self.species_database:
            return False

        patch = self.masterpatches[patch_id]

        # Check if layer already exists
        existing_layer = patch.get_layer_by_species(species_name)
        if existing_layer:
            # Update coverage
            existing_layer.coverage_fraction = coverage_fraction
            return True

        # Create new layer
        species = self.species_database[species_name]
        layer = ResourceLayer(
            layer_id=f"{patch_id}_{species_name}",
            layer_type=LayerType.FLOWER_SPECIES,
            flower_species=species,
            coverage_fraction=coverage_fraction,
        )

        patch.add_layer(layer)
        return True

    def populate_patches_from_habitat(
        self, habitat_species_mapping: Dict[HabitatType, List[Tuple[str, float]]]
    ) -> None:
        """Populate patches with species based on habitat type"""

        for patch in self.masterpatches.values():
            if patch.habitat_type in habitat_species_mapping:
                species_list = habitat_species_mapping[patch.habitat_type]

                for species_name, coverage in species_list:
                    self.add_species_layer_to_patch(
                        patch.patch_id, species_name, coverage
                    )

    def update_all_patches(
        self, day_of_year: int, weather_conditions: Dict[str, float]
    ) -> None:
        """Update all patches for given day"""

        # Calculate weather factor
        weather_factor = self.calculate_weather_factor(weather_conditions)

        # Update each patch
        for patch in self.masterpatches.values():
            patch.update_all_layers(day_of_year, weather_factor)

            # Decay foraging pressure
            patch.foraging_pressure *= 0.95

        # Apply scheduled management
        self.apply_scheduled_management(day_of_year)

    def calculate_weather_factor(self, weather_conditions: Dict[str, float]) -> float:
        """Calculate weather impact on resource production"""

        temperature = weather_conditions.get("temperature", 15.0)
        rainfall = weather_conditions.get("rainfall", 0.0)
        wind_speed = weather_conditions.get("wind_speed", 0.0)

        # Temperature effects
        if temperature < 5.0:
            temp_factor = 0.1
        elif temperature < 10.0:
            temp_factor = 0.5
        elif temperature < 30.0:
            temp_factor = 1.0
        else:
            temp_factor = max(0.2, 1.0 - (temperature - 30.0) / 20.0)

        # Rainfall effects
        if rainfall > 10.0:
            rain_factor = 0.3  # Heavy rain reduces production
        elif rainfall > 2.0:
            rain_factor = 0.7
        else:
            rain_factor = 1.0

        # Wind effects
        wind_factor = max(0.5, 1.0 - wind_speed / 50.0)

        return temp_factor * rain_factor * wind_factor

    def schedule_management(
        self,
        day_of_year: int,
        patch_ids: List[str],
        management_type: str,
        intensity: float,
    ) -> None:
        """Schedule management for specific patches"""

        if day_of_year not in self.management_calendar:
            self.management_calendar[day_of_year] = []

        self.management_calendar[day_of_year].append(
            {
                "patch_ids": patch_ids,
                "management_type": management_type,
                "intensity": intensity,
            }
        )

    def apply_scheduled_management(self, day_of_year: int) -> None:
        """Apply scheduled management actions"""

        if day_of_year in self.management_calendar:
            for management_action in self.management_calendar[day_of_year]:
                patch_ids = management_action["patch_ids"]
                management_type = management_action["management_type"]
                intensity = management_action["intensity"]

                for patch_id in patch_ids:
                    if patch_id in self.masterpatches:
                        patch = self.masterpatches[patch_id]
                        patch.apply_management(management_type, intensity, day_of_year)

    def get_patches_in_radius(
        self, center: Tuple[float, float], radius: float
    ) -> List[MasterPatch]:
        """Get patches within radius of center point"""

        nearby_patches = []

        for patch in self.masterpatches.values():
            distance = math.sqrt(
                (patch.location[0] - center[0]) ** 2
                + (patch.location[1] - center[1]) ** 2
            )

            if distance <= radius:
                nearby_patches.append(patch)

        return nearby_patches

    def get_best_patches_for_species(
        self,
        bee_species: str,
        center: Tuple[float, float],
        radius: float,
        proboscis_system: Any,
    ) -> List[Tuple[MasterPatch, float]]:
        """Get best patches for bee species within radius"""

        nearby_patches = self.get_patches_in_radius(center, radius)
        patch_scores = []

        for patch in nearby_patches:
            # Calculate patch score based on accessibility and resources
            accessibility_scores = patch.get_species_accessibility_for_bee(
                bee_species, proboscis_system
            )

            if accessibility_scores:
                total_nectar, total_pollen = patch.get_total_resources()
                total_resources = total_nectar + total_pollen

                # Weight by accessibility
                weighted_accessibility = 0.0
                total_weight = 0.0

                for species_name, accessibility in accessibility_scores.items():
                    layer = patch.get_layer_by_species(species_name)
                    if layer:
                        weight = layer.current_nectar + layer.current_pollen
                        weighted_accessibility += accessibility * weight
                        total_weight += weight

                if total_weight > 0:
                    avg_accessibility = weighted_accessibility / total_weight
                    patch_score = total_resources * avg_accessibility
                    patch_scores.append((patch, patch_score))

        # Sort by score (highest first)
        patch_scores.sort(key=lambda x: x[1], reverse=True)
        return patch_scores

    def simulate_foraging_impact(
        self,
        patch_id: str,
        bee_species: str,
        duration_hours: float,
        proboscis_system: Any,
    ) -> Dict[str, Dict[str, float]]:
        """Simulate foraging impact on patch"""

        if patch_id not in self.masterpatches:
            return {}

        patch = self.masterpatches[patch_id]

        # Get accessible species
        accessibility_scores = patch.get_species_accessibility_for_bee(
            bee_species, proboscis_system
        )

        # Calculate consumption based on accessibility and preference
        consumption_results = {}

        for species_name, accessibility in accessibility_scores.items():
            if accessibility > 0.2:  # Only forage on reasonably accessible flowers
                layer = patch.get_layer_by_species(species_name)
                if layer:
                    # Calculate consumption rate
                    base_consumption_rate = 0.5  # mg per hour
                    consumption_rate = (
                        base_consumption_rate * accessibility * duration_hours
                    )

                    # Prefer nectar over pollen (70:30 ratio)
                    nectar_consumed = consumption_rate * 0.7
                    pollen_consumed = consumption_rate * 0.3

                    # Consume resources
                    actual_nectar, actual_pollen = patch.consume_resources_from_species(
                        species_name, nectar_consumed, pollen_consumed
                    )

                    consumption_results[species_name] = {
                        "nectar_consumed": actual_nectar,
                        "pollen_consumed": actual_pollen,
                        "accessibility": accessibility,
                    }

        return consumption_results

    def get_landscape_carrying_capacity(self) -> Dict[str, float]:
        """Calculate landscape carrying capacity"""

        total_annual_nectar = 0.0
        total_annual_pollen = 0.0

        # Simulate full year
        for day in range(1, 366):
            daily_nectar = 0.0
            daily_pollen = 0.0

            for patch in self.masterpatches.values():
                for layer in patch.layers.values():
                    availability = layer.get_availability(day)
                    if availability > 0:
                        flowers = (
                            layer.flower_species.flower_density
                            * patch.area_m2
                            * layer.coverage_fraction
                        )
                        daily_nectar += (
                            flowers
                            * layer.flower_species.nectar_production
                            * availability
                        )
                        daily_pollen += (
                            flowers
                            * layer.flower_species.pollen_production
                            * availability
                        )

            total_annual_nectar += daily_nectar
            total_annual_pollen += daily_pollen

        # Convert to colony carrying capacity
        nectar_colonies = total_annual_nectar / 50000  # 50g nectar per colony per year
        pollen_colonies = total_annual_pollen / 20000  # 20g pollen per colony per year

        return {
            "nectar_limited": nectar_colonies,
            "pollen_limited": pollen_colonies,
            "overall": min(nectar_colonies, pollen_colonies),
            "total_patches": len(self.masterpatches),
            "average_species_per_patch": float(
                np.mean([len(p.layers) for p in self.masterpatches.values()])
            ),
            "total_species_layers": sum(
                len(p.layers) for p in self.masterpatches.values()
            ),
        }

    def export_patch_data(self, day_of_year: int) -> Dict[str, Any]:
        """Export current patch data for analysis"""

        patch_data: Dict[str, Any] = {"day_of_year": day_of_year, "masterpatches": []}

        for patch in self.masterpatches.values():
            patch_info: Dict[str, Any] = {
                "patch_id": patch.patch_id,
                "location": patch.location,
                "area_m2": patch.area_m2,
                "habitat_type": patch.habitat_type.value,
                "total_nectar": sum(
                    layer.current_nectar for layer in patch.layers.values()
                ),
                "total_pollen": sum(
                    layer.current_pollen for layer in patch.layers.values()
                ),
                "species_count": len(patch.layers),
                "foraging_pressure": patch.foraging_pressure,
                "resource_quality": patch.get_resource_quality(),
                "layers": [],
            }

            for layer in patch.layers.values():
                layer_info = {
                    "species_name": layer.flower_species.name,
                    "coverage_fraction": layer.coverage_fraction,
                    "current_nectar": layer.current_nectar,
                    "current_pollen": layer.current_pollen,
                    "availability": layer.get_availability(day_of_year),
                    "depletion_factor": layer.depletion_factor,
                }
                patch_info["layers"].append(layer_info)

            patch_data["masterpatches"].append(patch_info)

        return patch_data

    def create_habitat_based_landscape(
        self, habitat_map: np.ndarray, cell_size_m: float
    ) -> int:
        """Create masterpatches from habitat map"""

        # Define habitat-species associations
        habitat_species = {
            HabitatType.WILDFLOWER: [
                ("White Clover", 0.4),
                ("Red Clover", 0.3),
                ("Dandelion", 0.6),
                ("Phacelia", 0.2),
                ("Borage", 0.1),
            ],
            HabitatType.HEDGEROW: [
                ("Hawthorn", 0.5),
                ("Blackthorn", 0.3),
                ("Bramble", 0.4),
                ("Gorse", 0.2),
            ],
            HabitatType.CROPLAND: [
                ("Oilseed Rape", 0.8),
                ("Field Bean", 0.6),
                ("White Clover", 0.1),
            ],
            HabitatType.GRASSLAND: [
                ("White Clover", 0.3),
                ("Dandelion", 0.5),
                ("Heather", 0.1),
            ],
            HabitatType.WOODLAND: [
                ("Lime Tree", 0.2),
                ("Hawthorn", 0.1),
                ("Bramble", 0.3),
            ],
        }

        patch_count = 0
        height, width = habitat_map.shape

        for y in range(height):
            for x in range(width):
                habitat_value = habitat_map[y, x]

                # Map habitat value to HabitatType
                habitat_type = self.map_habitat_value_to_type(habitat_value)

                if habitat_type in habitat_species:
                    # Create patch
                    patch_id = f"patch_{x}_{y}"
                    location = (x * cell_size_m, y * cell_size_m)
                    area = cell_size_m * cell_size_m

                    self.create_masterpatch(patch_id, location, area, habitat_type)

                    # Add species layers
                    for species_name, coverage in habitat_species[habitat_type]:
                        self.add_species_layer_to_patch(
                            patch_id, species_name, coverage
                        )

                    patch_count += 1

        return patch_count

    def map_habitat_value_to_type(self, habitat_value: int) -> HabitatType:
        """Map habitat map value to HabitatType"""

        # Standard habitat mapping
        habitat_mapping = {
            0: HabitatType.BARE_SOIL,
            1: HabitatType.GRASSLAND,
            2: HabitatType.CROPLAND,
            3: HabitatType.WOODLAND,
            4: HabitatType.HEDGEROW,
            5: HabitatType.WILDFLOWER,
            6: HabitatType.URBAN,
            7: HabitatType.WATER,
            8: HabitatType.ROAD,
            9: HabitatType.BUILDING,
        }

        return habitat_mapping.get(habitat_value, HabitatType.BARE_SOIL)
