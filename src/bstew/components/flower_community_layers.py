"""
Flower Community Layers System for BSTEW
========================================

Implements hierarchical layering system for complex flower communities
with canopy/understory/ground layer modeling, vertical resource distribution,
and advanced succession dynamics.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, date
import logging
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from ..spatial.patches import FlowerSpecies


class VerticalLayer(Enum):
    """Vertical layers in flower community structure"""

    CANOPY = "canopy"  # 3-15m: Trees, large shrubs
    UNDERSTORY = "understory"  # 1-3m: Small shrubs, tall herbs
    GROUND = "ground"  # 0-1m: Grasses, low herbs, forbs
    ROOT = "root"  # Below ground: Root systems, soil interactions


class SuccessionStage(Enum):
    """Succession stages of flower communities"""

    PIONEER = "pioneer"  # Early colonizers, annual species
    EARLY = "early"  # Fast-growing perennials
    MID = "mid"  # Competitive perennials, shrubs
    LATE = "late"  # Climax species, trees
    CLIMAX = "climax"  # Mature stable community


class CompetitionType(Enum):
    """Types of inter-layer competition"""

    LIGHT = "light"  # Competition for sunlight
    NUTRIENTS = "nutrients"  # Soil nutrient competition
    WATER = "water"  # Water resource competition
    SPACE = "space"  # Physical space competition
    POLLINATORS = "pollinators"  # Pollinator resource competition


@dataclass
class LayerInteraction:
    """Defines interaction between community layers"""

    source_layer: VerticalLayer
    target_layer: VerticalLayer
    interaction_type: CompetitionType
    strength: float = Field(ge=0.0, le=1.0)  # Interaction strength
    seasonal_modifier: Dict[int, float] = field(
        default_factory=dict
    )  # Day-specific modifiers


@dataclass
class VerticalResourceGradient:
    """Vertical distribution of resources in community"""

    light_profile: Dict[VerticalLayer, float] = field(default_factory=dict)
    nutrient_profile: Dict[VerticalLayer, float] = field(default_factory=dict)
    water_profile: Dict[VerticalLayer, float] = field(default_factory=dict)
    pollinator_access: Dict[VerticalLayer, float] = field(default_factory=dict)


class CommunityLayer(BaseModel):
    """Individual layer within flower community hierarchy"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    layer_id: str = Field(description="Unique layer identifier")
    vertical_layer: VerticalLayer = Field(description="Vertical position in community")
    succession_stage: SuccessionStage = Field(description="Current succession stage")

    # Species composition
    dominant_species: List[FlowerSpecies] = Field(default_factory=list)
    subordinate_species: List[FlowerSpecies] = Field(default_factory=list)
    rare_species: List[FlowerSpecies] = Field(default_factory=list)

    # Layer characteristics
    coverage_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    height_range: Tuple[float, float] = Field(description="Min/max height in meters")
    density_factor: float = Field(default=1.0, ge=0.0, description="Relative density")

    # Resource dynamics
    resource_efficiency: Dict[str, float] = Field(
        default_factory=lambda: {
            "light_capture": 1.0,
            "nutrient_uptake": 1.0,
            "water_uptake": 1.0,
            "pollinator_attraction": 1.0,
        }
    )

    # Competition effects
    competitive_ability: Dict[CompetitionType, float] = Field(default_factory=dict)
    stress_tolerance: Dict[str, float] = Field(default_factory=dict)

    # Temporal dynamics
    phenology_phases: Dict[str, Tuple[int, int]] = Field(
        default_factory=dict, description="Phase start/end days"
    )
    succession_trajectory: Optional[SuccessionStage] = None
    succession_rate: float = Field(
        default=0.01, ge=0.0, description="Annual succession rate"
    )

    def get_resource_production(
        self,
        day_of_year: int,
        base_conditions: VerticalResourceGradient,
        competition_effects: Dict[CompetitionType, float],
    ) -> Dict[str, float]:
        """Calculate resource production considering competition"""

        # Get base resource availability for this layer
        base_light = base_conditions.light_profile.get(self.vertical_layer, 1.0)
        base_nutrients = base_conditions.nutrient_profile.get(self.vertical_layer, 1.0)
        base_water = base_conditions.water_profile.get(self.vertical_layer, 1.0)
        base_pollinator_access = base_conditions.pollinator_access.get(
            self.vertical_layer, 1.0
        )

        # Apply competition effects
        light_available = base_light * (
            1.0 - competition_effects.get(CompetitionType.LIGHT, 0.0)
        )
        nutrients_available = base_nutrients * (
            1.0 - competition_effects.get(CompetitionType.NUTRIENTS, 0.0)
        )
        water_available = base_water * (
            1.0 - competition_effects.get(CompetitionType.WATER, 0.0)
        )
        pollinator_available = base_pollinator_access * (
            1.0 - competition_effects.get(CompetitionType.POLLINATORS, 0.0)
        )

        # Calculate resource capture efficiency
        light_captured = light_available * self.resource_efficiency["light_capture"]
        nutrients_captured = (
            nutrients_available * self.resource_efficiency["nutrient_uptake"]
        )
        water_captured = water_available * self.resource_efficiency["water_uptake"]
        pollinator_captured = (
            pollinator_available * self.resource_efficiency["pollinator_attraction"]
        )

        # Calculate nectar and pollen production from all species
        total_nectar = 0.0
        total_pollen = 0.0

        for species_list, weight in [
            (self.dominant_species, 1.0),
            (self.subordinate_species, 0.5),
            (self.rare_species, 0.1),
        ]:
            for species in species_list:
                if species.bloom_start <= day_of_year <= species.bloom_end:
                    # Resource limitation is minimum of all resources
                    resource_limit = min(
                        light_captured, nutrients_captured, water_captured
                    )

                    species_nectar = (
                        species.nectar_production
                        * species.flower_density
                        * self.coverage_fraction
                        * self.density_factor
                        * weight
                        * resource_limit
                        * pollinator_captured
                    )

                    species_pollen = (
                        species.pollen_production
                        * species.flower_density
                        * self.coverage_fraction
                        * self.density_factor
                        * weight
                        * resource_limit
                        * pollinator_captured
                    )

                    total_nectar += species_nectar
                    total_pollen += species_pollen

        return {
            "nectar_production": total_nectar,
            "pollen_production": total_pollen,
            "light_utilization": light_captured,
            "nutrient_utilization": nutrients_captured,
            "water_utilization": water_captured,
            "resource_limitation": min(
                light_captured, nutrients_captured, water_captured
            ),
        }

    def advance_succession(self, environmental_pressure: float = 0.0) -> bool:
        """Advance succession stage based on conditions"""
        if not self.succession_trajectory:
            return False

        # Calculate succession probability
        base_rate = self.succession_rate
        environmental_modifier = 1.0 + environmental_pressure

        succession_probability = base_rate * environmental_modifier

        if np.random.random() < succession_probability:
            # Advance to next stage
            current_stages = list(SuccessionStage)
            current_index = current_stages.index(self.succession_stage)

            if current_index < len(current_stages) - 1:
                self.succession_stage = current_stages[current_index + 1]
                return True

        return False

    def calculate_biodiversity_index(self) -> float:
        """Calculate Simpson's diversity index for layer"""
        all_species = (
            self.dominant_species + self.subordinate_species + self.rare_species
        )

        if not all_species:
            return 0.0

        # Assign relative abundances
        abundances = []
        abundances.extend([1.0] * len(self.dominant_species))
        abundances.extend([0.5] * len(self.subordinate_species))
        abundances.extend([0.1] * len(self.rare_species))

        total_abundance = sum(abundances)

        if total_abundance == 0:
            return 0.0

        # Calculate Simpson's index
        simpson_index = 0.0
        for abundance in abundances:
            proportion = abundance / total_abundance
            simpson_index += proportion * proportion

        # Return Simpson's diversity (1 - D)
        return 1.0 - simpson_index


class FlowerCommunity(BaseModel):
    """Complete hierarchical flower community"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    community_id: str = Field(description="Unique community identifier")
    community_type: str = Field(description="Community type classification")
    location: Tuple[float, float] = Field(description="Geographic coordinates")
    area_m2: float = Field(ge=0.0, description="Community area")

    # Layer structure
    layers: Dict[VerticalLayer, CommunityLayer] = Field(default_factory=dict)
    layer_interactions: List[LayerInteraction] = Field(default_factory=list)

    # Environmental conditions
    base_conditions: VerticalResourceGradient = Field(
        default_factory=VerticalResourceGradient
    )
    microclimate_modifiers: Dict[str, float] = Field(default_factory=dict)

    # Community dynamics
    disturbance_history: List[Dict[str, Any]] = Field(default_factory=list)
    management_regime: Optional[str] = None
    establishment_date: Optional[date] = None

    # State tracking
    current_diversity_index: float = Field(default=0.0, ge=0.0)
    stability_index: float = Field(default=1.0, ge=0.0)
    resilience_score: float = Field(default=1.0, ge=0.0)

    def add_layer(self, layer: CommunityLayer) -> None:
        """Add layer to community structure"""
        self.layers[layer.vertical_layer] = layer
        self._update_community_metrics()

    def remove_layer(self, vertical_layer: VerticalLayer) -> None:
        """Remove layer from community"""
        if vertical_layer in self.layers:
            del self.layers[vertical_layer]
            self._update_community_metrics()

    def calculate_layer_interactions(
        self,
    ) -> Dict[VerticalLayer, Dict[CompetitionType, float]]:
        """Calculate competition effects between layers"""
        competition_matrix: Dict[VerticalLayer, Dict[CompetitionType, float]] = (
            defaultdict(lambda: defaultdict(float))
        )

        for interaction in self.layer_interactions:
            source = interaction.source_layer
            target = interaction.target_layer
            comp_type = interaction.interaction_type

            # Calculate competition strength based on layer presence and characteristics
            if source in self.layers and target in self.layers:
                source_layer = self.layers[source]
                target_layer = self.layers[target]

                # Competition strength based on coverage and competitive ability
                source_strength = (
                    source_layer.coverage_fraction
                    * source_layer.density_factor
                    * source_layer.competitive_ability.get(comp_type, 0.5)
                )

                target_vulnerability = 1.0 - target_layer.stress_tolerance.get(
                    comp_type.value, 0.5
                )

                competition_effect = (
                    interaction.strength * source_strength * target_vulnerability
                )
                competition_matrix[target][comp_type] += competition_effect

        return dict(competition_matrix)

    def update_community_resources(
        self, day_of_year: int, weather_factor: float = 1.0
    ) -> Dict[str, float]:
        """Update resources across all layers"""
        competition_effects = self.calculate_layer_interactions()

        total_resources = {
            "nectar_production": 0.0,
            "pollen_production": 0.0,
            "total_biomass": 0.0,
            "community_efficiency": 0.0,
        }

        layer_efficiencies = []

        for vertical_layer, layer in self.layers.items():
            layer_competition = competition_effects.get(vertical_layer, {})

            layer_resources = layer.get_resource_production(
                day_of_year, self.base_conditions, layer_competition
            )

            # Apply weather and microclimate effects
            weather_modifier = weather_factor
            for modifier_type, modifier_value in self.microclimate_modifiers.items():
                if modifier_type == "temperature_buffer":
                    weather_modifier *= 1.0 + modifier_value * 0.1
                elif modifier_type == "humidity_retention":
                    weather_modifier *= 1.0 + modifier_value * 0.05
                elif modifier_type == "wind_protection":
                    weather_modifier *= 1.0 + modifier_value * 0.03

            # Scale by weather and area
            scaled_nectar = (
                layer_resources["nectar_production"]
                * weather_modifier
                * self.area_m2
                / 10000
            )
            scaled_pollen = (
                layer_resources["pollen_production"]
                * weather_modifier
                * self.area_m2
                / 10000
            )

            total_resources["nectar_production"] += scaled_nectar
            total_resources["pollen_production"] += scaled_pollen
            total_resources["total_biomass"] += (
                layer.coverage_fraction * layer.density_factor
            )

            layer_efficiencies.append(layer_resources["resource_limitation"])

        # Calculate community-level efficiency
        if layer_efficiencies:
            total_resources["community_efficiency"] = float(np.mean(layer_efficiencies))

        return total_resources

    def simulate_succession(
        self, years: int = 1, disturbance_probability: float = 0.02
    ) -> List[Dict[str, Any]]:
        """Simulate succession dynamics over time"""
        succession_history: List[Dict[str, Any]] = []

        for year in range(years):
            year_data: Dict[str, Any] = {
                "year": year,
                "succession_events": [],
                "disturbances": [],
                "diversity_changes": [],
            }

            # Check for disturbances
            if np.random.random() < disturbance_probability:
                disturbance = self._apply_random_disturbance()
                year_data["disturbances"].append(disturbance)

            # Advance succession in each layer
            for layer_id, layer in self.layers.items():
                initial_stage = layer.succession_stage

                # Environmental pressure based on competition
                competition_effects = self.calculate_layer_interactions()
                layer_competition = competition_effects.get(layer_id, {})
                pressure = sum(layer_competition.values()) * 0.1

                if layer.advance_succession(pressure):
                    year_data["succession_events"].append(
                        {
                            "layer": layer_id.value,
                            "from_stage": initial_stage.value,
                            "to_stage": layer.succession_stage.value,
                            "pressure": pressure,
                        }
                    )

            # Update diversity metrics
            initial_diversity = self.current_diversity_index
            self._update_community_metrics()
            diversity_change = self.current_diversity_index - initial_diversity

            if abs(diversity_change) > 0.01:
                year_data["diversity_changes"].append(
                    {
                        "change": diversity_change,
                        "new_diversity": self.current_diversity_index,
                    }
                )

            succession_history.append(year_data)

        return succession_history

    def _apply_random_disturbance(self) -> Dict[str, Any]:
        """Apply random disturbance to community"""
        disturbance_types: List[Dict[str, Any]] = [
            {"type": "drought", "intensity": 0.3, "affected_layers": ["ground"]},
            {
                "type": "frost",
                "intensity": 0.4,
                "affected_layers": ["canopy", "understory"],
            },
            {
                "type": "herbivory",
                "intensity": 0.2,
                "affected_layers": ["ground", "understory"],
            },
            {
                "type": "fire",
                "intensity": 0.8,
                "affected_layers": ["canopy", "understory", "ground"],
            },
            {"type": "wind", "intensity": 0.3, "affected_layers": ["canopy"]},
            {
                "type": "flooding",
                "intensity": 0.5,
                "affected_layers": ["ground", "root"],
            },
        ]

        disturbance: Dict[str, Any] = disturbance_types[
            np.random.randint(len(disturbance_types))
        ]

        # Apply disturbance effects
        for layer_name in disturbance["affected_layers"]:
            try:
                layer_enum = VerticalLayer(layer_name)
                if layer_enum in self.layers:
                    layer = self.layers[layer_enum]
                    # Reduce coverage and density
                    layer.coverage_fraction *= 1.0 - disturbance["intensity"] * 0.5
                    layer.density_factor *= 1.0 - disturbance["intensity"] * 0.3

                    # Reset succession if severe
                    if disturbance["intensity"] > 0.6:
                        layer.succession_stage = SuccessionStage.PIONEER
            except ValueError:
                continue

        disturbance_record = {
            "date": datetime.now().isoformat(),
            "type": disturbance["type"],
            "intensity": disturbance["intensity"],
            "affected_layers": disturbance["affected_layers"],
        }

        self.disturbance_history.append(disturbance_record)
        return disturbance_record

    def _update_community_metrics(self) -> None:
        """Update community-level diversity and stability metrics"""
        if not self.layers:
            self.current_diversity_index = 0.0
            self.stability_index = 0.0
            return

        # Calculate community diversity (average of layer diversities)
        layer_diversities = []
        for layer in self.layers.values():
            layer_diversity = layer.calculate_biodiversity_index()
            layer_diversities.append(layer_diversity)

        self.current_diversity_index = (
            float(np.mean(layer_diversities)) if layer_diversities else 0.0
        )

        # Calculate stability index based on layer balance
        layer_coverages = [layer.coverage_fraction for layer in self.layers.values()]
        coverage_variance = (
            float(np.var(layer_coverages)) if len(layer_coverages) > 1 else 0.0
        )
        self.stability_index = float(max(0.0, 1.0 - coverage_variance))

        # Calculate resilience based on succession stage diversity
        stage_counts: Dict[SuccessionStage, int] = defaultdict(int)
        for layer in self.layers.values():
            stage_counts[layer.succession_stage] += 1

        stage_diversity = len(stage_counts) / len(SuccessionStage)
        self.resilience_score = min(1.0, stage_diversity * 1.2)

    def get_species_accessibility_profile(
        self, proboscis_system: Any
    ) -> Dict[str, Dict[str, float]]:
        """Get accessibility profiles for all species across layers"""
        accessibility_profiles = {}

        for layer_id, layer in self.layers.items():
            layer_profile = {}

            all_species = (
                layer.dominant_species + layer.subordinate_species + layer.rare_species
            )

            for species in all_species:
                # Get pollinator access for this layer
                layer_access = self.base_conditions.pollinator_access.get(layer_id, 1.0)

                # Modify by layer height (higher layers may be less accessible)
                height_modifier = 1.0
                if layer_id == VerticalLayer.CANOPY:
                    height_modifier = 0.7
                elif layer_id == VerticalLayer.UNDERSTORY:
                    height_modifier = 0.9

                base_accessibility = layer_access * height_modifier
                layer_profile[species.name] = base_accessibility

            if layer_profile:
                accessibility_profiles[layer_id.value] = layer_profile

        return accessibility_profiles

    def export_community_structure(self) -> Dict[str, Any]:
        """Export community structure for analysis"""
        return {
            "community_id": self.community_id,
            "community_type": self.community_type,
            "location": self.location,
            "area_m2": self.area_m2,
            "establishment_date": self.establishment_date.isoformat()
            if self.establishment_date
            else None,
            "layer_count": len(self.layers),
            "layers": {
                layer_id.value: {
                    "succession_stage": layer.succession_stage.value,
                    "height_range": layer.height_range,
                    "coverage_fraction": layer.coverage_fraction,
                    "density_factor": layer.density_factor,
                    "species_count": {
                        "dominant": len(layer.dominant_species),
                        "subordinate": len(layer.subordinate_species),
                        "rare": len(layer.rare_species),
                    },
                    "biodiversity_index": layer.calculate_biodiversity_index(),
                }
                for layer_id, layer in self.layers.items()
            },
            "community_metrics": {
                "diversity_index": self.current_diversity_index,
                "stability_index": self.stability_index,
                "resilience_score": self.resilience_score,
            },
            "disturbance_count": len(self.disturbance_history),
            "management_regime": self.management_regime,
        }


class CommunityLayerSystem:
    """
    Advanced Community Layer System for BSTEW

    Manages hierarchical flower communities with vertical layering,
    complex succession dynamics, and multi-layer resource competition.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Community management
        self.communities: Dict[str, FlowerCommunity] = {}
        self.community_templates: Dict[str, Dict[str, Any]] = {}

        # System parameters
        self.default_gradient = self._create_default_resource_gradient()
        self.succession_rates = self._initialize_succession_rates()

    def _create_default_resource_gradient(self) -> VerticalResourceGradient:
        """Create default vertical resource gradient"""
        return VerticalResourceGradient(
            light_profile={
                VerticalLayer.CANOPY: 1.0,  # Full light
                VerticalLayer.UNDERSTORY: 0.6,  # Partially shaded
                VerticalLayer.GROUND: 0.3,  # Heavily shaded
                VerticalLayer.ROOT: 0.0,  # No light
            },
            nutrient_profile={
                VerticalLayer.CANOPY: 0.7,  # Less soil contact
                VerticalLayer.UNDERSTORY: 0.8,  # Good soil access
                VerticalLayer.GROUND: 1.0,  # Maximum soil contact
                VerticalLayer.ROOT: 1.0,  # Direct soil access
            },
            water_profile={
                VerticalLayer.CANOPY: 0.8,  # Atmospheric moisture
                VerticalLayer.UNDERSTORY: 0.9,  # Protected from wind
                VerticalLayer.GROUND: 1.0,  # Soil moisture
                VerticalLayer.ROOT: 1.0,  # Direct soil water
            },
            pollinator_access={
                VerticalLayer.CANOPY: 0.7,  # High but wind exposed
                VerticalLayer.UNDERSTORY: 1.0,  # Optimal height and protection
                VerticalLayer.GROUND: 0.9,  # Accessible but low
                VerticalLayer.ROOT: 0.0,  # No flowers
            },
        )

    def _initialize_succession_rates(self) -> Dict[str, float]:
        """Initialize succession rates by community type"""
        return {
            "grassland": 0.05,  # Slow succession
            "meadow": 0.03,  # Very slow
            "scrubland": 0.08,  # Moderate succession
            "woodland_edge": 0.10,  # Fast succession
            "disturbed": 0.15,  # Very fast succession
            "wetland": 0.02,  # Very slow succession
        }

    def create_community_template(
        self, template_name: str, template_data: Dict[str, Any]
    ) -> None:
        """Create reusable community template"""
        self.community_templates[template_name] = template_data
        self.logger.info(f"Created community template: {template_name}")

    def create_community_from_template(
        self,
        community_id: str,
        template_name: str,
        location: Tuple[float, float],
        area_m2: float,
        customizations: Optional[Dict[str, Any]] = None,
    ) -> FlowerCommunity:
        """Create flower community from template"""

        if template_name not in self.community_templates:
            raise ValueError(f"Template {template_name} not found")

        template = self.community_templates[template_name]

        # Create base community
        community = FlowerCommunity(
            community_id=community_id,
            community_type=template.get("community_type", template_name),
            location=location,
            area_m2=area_m2,
            base_conditions=self.default_gradient,
            establishment_date=date.today(),
        )

        # Apply template layers
        for layer_data in template.get("layers", []):
            layer = self._create_layer_from_template(layer_data)
            community.add_layer(layer)

        # Apply template interactions
        for interaction_data in template.get("interactions", []):
            interaction = LayerInteraction(
                source_layer=VerticalLayer(interaction_data["source"]),
                target_layer=VerticalLayer(interaction_data["target"]),
                interaction_type=CompetitionType(interaction_data["type"]),
                strength=interaction_data["strength"],
            )
            community.layer_interactions.append(interaction)

        # Apply customizations
        if customizations:
            self._apply_community_customizations(community, customizations)

        self.communities[community_id] = community
        self.logger.info(
            f"Created community {community_id} from template {template_name}"
        )

        return community

    def _create_layer_from_template(self, layer_data: Dict[str, Any]) -> CommunityLayer:
        """Create community layer from template data"""

        # Create basic layer
        layer = CommunityLayer(
            layer_id=layer_data["layer_id"],
            vertical_layer=VerticalLayer(layer_data["vertical_layer"]),
            succession_stage=SuccessionStage(
                layer_data.get("succession_stage", "early")
            ),
            coverage_fraction=layer_data.get("coverage_fraction", 1.0),
            height_range=tuple(layer_data.get("height_range", [0.0, 1.0])),
            density_factor=layer_data.get("density_factor", 1.0),
        )

        # Set species composition (would load from species database)
        # For now, using placeholder logic
        # layer.dominant_species = self._load_species_by_names(layer_data.get("species", []))

        # Set resource efficiency
        if "resource_efficiency" in layer_data:
            layer.resource_efficiency.update(layer_data["resource_efficiency"])

        # Set competitive abilities
        if "competitive_ability" in layer_data:
            for comp_type, value in layer_data["competitive_ability"].items():
                layer.competitive_ability[CompetitionType(comp_type)] = value

        return layer

    def _apply_community_customizations(
        self, community: FlowerCommunity, customizations: Dict[str, Any]
    ) -> None:
        """Apply customizations to community"""

        if "microclimate" in customizations:
            community.microclimate_modifiers.update(customizations["microclimate"])

        if "management_regime" in customizations:
            community.management_regime = customizations["management_regime"]

        if "resource_gradients" in customizations:
            # Modify resource gradients
            gradients = customizations["resource_gradients"]
            if "light_modifier" in gradients:
                for layer, modifier in gradients["light_modifier"].items():
                    layer_enum = VerticalLayer(layer)
                    current = community.base_conditions.light_profile.get(
                        layer_enum, 1.0
                    )
                    community.base_conditions.light_profile[layer_enum] = (
                        current * modifier
                    )

    def simulate_landscape_succession(
        self, years: int = 10, climate_scenario: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Simulate succession across all communities"""

        landscape_results: Dict[str, Any] = {
            "total_years": years,
            "community_count": len(self.communities),
            "climate_scenario": climate_scenario or {},
            "results": {},
        }

        for community_id, community in self.communities.items():
            # Apply climate effects
            disturbance_prob = 0.02
            if climate_scenario:
                temp_change = climate_scenario.get("temperature_change", 0.0)
                precip_change = climate_scenario.get("precipitation_change", 0.0)

                # Increase disturbance with climate change
                disturbance_prob *= (
                    1.0 + abs(temp_change) * 0.1 + abs(precip_change) * 0.05
                )

            # Run succession simulation
            succession_results = community.simulate_succession(years, disturbance_prob)

            landscape_results["results"][community_id] = {
                "succession_history": succession_results,
                "final_diversity": community.current_diversity_index,
                "final_stability": community.stability_index,
                "final_resilience": community.resilience_score,
                "total_disturbances": len(community.disturbance_history),
            }

        return landscape_results

    def analyze_community_connectivity(self) -> Dict[str, Any]:
        """Analyze connectivity between flower communities"""

        if len(self.communities) < 2:
            return {
                "connectivity_score": 0.0,
                "connections": [],
                "analysis": "Insufficient communities",
            }

        connections = []

        community_list = list(self.communities.values())

        for i, comm1 in enumerate(community_list):
            for j, comm2 in enumerate(community_list[i + 1 :], i + 1):
                # Calculate distance
                distance = np.sqrt(
                    (comm1.location[0] - comm2.location[0]) ** 2
                    + (comm1.location[1] - comm2.location[1]) ** 2
                )

                # Calculate connectivity strength based on distance and community characteristics
                max_distance = 2000  # Maximum connection distance (2km)

                if distance <= max_distance:
                    # Distance factor (closer = stronger connection)
                    distance_factor = 1.0 - (distance / max_distance)

                    # Similarity factor (similar communities connect better)
                    type_similarity = (
                        1.0 if comm1.community_type == comm2.community_type else 0.7
                    )

                    # Diversity factor (diverse communities provide better connectivity)
                    diversity_factor = (
                        comm1.current_diversity_index + comm2.current_diversity_index
                    ) / 2

                    connectivity_strength = (
                        distance_factor * type_similarity * diversity_factor
                    )

                    connections.append(
                        {
                            "from_community": comm1.community_id,
                            "to_community": comm2.community_id,
                            "distance_m": distance,
                            "strength": connectivity_strength,
                            "type_similarity": type_similarity,
                            "diversity_factor": diversity_factor,
                        }
                    )

        # Calculate overall connectivity score
        if connections:
            total_strength = sum(conn["strength"] for conn in connections)
            max_possible_connections = (
                len(self.communities) * (len(self.communities) - 1) / 2
            )
            connectivity_score = total_strength / max_possible_connections
        else:
            connectivity_score = 0.0

        return {
            "connectivity_score": connectivity_score,
            "connections": connections,
            "total_communities": len(self.communities),
            "connected_pairs": len(connections),
            "average_connection_strength": np.mean([c["strength"] for c in connections])
            if connections
            else 0.0,
        }

    def export_landscape_summary(self) -> Dict[str, Any]:
        """Export comprehensive landscape summary"""

        if not self.communities:
            return {"error": "No communities to analyze"}

        # Aggregate statistics
        total_area = sum(comm.area_m2 for comm in self.communities.values())
        avg_diversity = np.mean(
            [comm.current_diversity_index for comm in self.communities.values()]
        )
        avg_stability = np.mean(
            [comm.stability_index for comm in self.communities.values()]
        )
        avg_resilience = np.mean(
            [comm.resilience_score for comm in self.communities.values()]
        )

        # Community type distribution
        type_counts: Dict[str, int] = defaultdict(int)
        for community in self.communities.values():
            type_counts[community.community_type] += 1

        # Succession stage distribution
        stage_counts: Dict[str, int] = defaultdict(int)
        for community in self.communities.values():
            for layer in community.layers.values():
                stage_counts[layer.succession_stage.value] += 1

        # Layer distribution
        layer_counts: Dict[str, int] = defaultdict(int)
        for community in self.communities.values():
            for layer_type in community.layers.keys():
                layer_counts[layer_type.value] += 1

        return {
            "landscape_summary": {
                "total_communities": len(self.communities),
                "total_area_m2": total_area,
                "total_area_ha": total_area / 10000,
                "average_diversity": avg_diversity,
                "average_stability": avg_stability,
                "average_resilience": avg_resilience,
            },
            "community_types": dict(type_counts),
            "succession_stages": dict(stage_counts),
            "layer_distribution": dict(layer_counts),
            "community_details": {
                comm_id: community.export_community_structure()
                for comm_id, community in self.communities.items()
            },
        }


# Initialize standard community templates
def initialize_standard_templates() -> Dict[str, Dict[str, Any]]:
    """Initialize standard flower community templates"""

    templates = {
        "uk_chalk_grassland": {
            "community_type": "chalk_grassland",
            "layers": [
                {
                    "layer_id": "ground_herbs",
                    "vertical_layer": "ground",
                    "succession_stage": "mid",
                    "coverage_fraction": 0.8,
                    "height_range": [0.0, 0.5],
                    "species": [
                        "Lotus corniculatus",
                        "Thymus serpyllum",
                        "Centaurea nigra",
                    ],
                    "resource_efficiency": {
                        "light_capture": 0.9,
                        "nutrient_uptake": 1.0,
                        "water_uptake": 0.8,
                        "pollinator_attraction": 1.0,
                    },
                },
                {
                    "layer_id": "tall_herbs",
                    "vertical_layer": "understory",
                    "succession_stage": "mid",
                    "coverage_fraction": 0.3,
                    "height_range": [0.5, 1.5],
                    "species": ["Centaurea scabiosa", "Leucanthemum vulgare"],
                    "resource_efficiency": {
                        "light_capture": 0.8,
                        "nutrient_uptake": 0.9,
                        "water_uptake": 0.9,
                        "pollinator_attraction": 0.9,
                    },
                },
            ],
            "interactions": [
                {
                    "source": "understory",
                    "target": "ground",
                    "type": "light",
                    "strength": 0.3,
                }
            ],
        },
        "woodland_edge": {
            "community_type": "woodland_edge",
            "layers": [
                {
                    "layer_id": "canopy_edge",
                    "vertical_layer": "canopy",
                    "succession_stage": "late",
                    "coverage_fraction": 0.4,
                    "height_range": [3.0, 10.0],
                    "species": ["Crataegus monogyna", "Prunus spinosa"],
                    "competitive_ability": {"light": 0.9, "space": 0.8},
                },
                {
                    "layer_id": "shrub_layer",
                    "vertical_layer": "understory",
                    "succession_stage": "mid",
                    "coverage_fraction": 0.6,
                    "height_range": [1.0, 3.0],
                    "species": ["Rosa canina", "Rubus fruticosus"],
                },
                {
                    "layer_id": "woodland_herbs",
                    "vertical_layer": "ground",
                    "succession_stage": "mid",
                    "coverage_fraction": 0.5,
                    "height_range": [0.0, 1.0],
                    "species": ["Geum urbanum", "Ajuga reptans"],
                },
            ],
            "interactions": [
                {
                    "source": "canopy",
                    "target": "understory",
                    "type": "light",
                    "strength": 0.4,
                },
                {
                    "source": "canopy",
                    "target": "ground",
                    "type": "light",
                    "strength": 0.6,
                },
                {
                    "source": "understory",
                    "target": "ground",
                    "type": "light",
                    "strength": 0.3,
                },
            ],
        },
    }

    return templates
