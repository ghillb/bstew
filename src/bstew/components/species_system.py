"""
Multi-Species System for BSTEW
==============================

Implements support for multiple bumblebee species with species-specific
parameters, behaviors, and ecological interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging

from .proboscis_matching import ProboscisCharacteristics
from .development import DevelopmentParameters


class SpeciesType(Enum):
    """Bumblebee species types"""

    BOMBUS_TERRESTRIS = "Bombus_terrestris"
    BOMBUS_LUCORUM = "Bombus_lucorum"
    BOMBUS_LAPIDARIUS = "Bombus_lapidarius"
    BOMBUS_PRATORUM = "Bombus_pratorum"
    BOMBUS_PASCUORUM = "Bombus_pascuorum"
    BOMBUS_HORTORUM = "Bombus_hortorum"
    BOMBUS_RUDERATUS = "Bombus_ruderatus"


class SpeciesParameters(BaseModel):
    """Complete species-specific parameters"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    species_name: str = Field(description="Species common name")
    species_type: SpeciesType = Field(description="Species taxonomic type")

    # Morphology
    proboscis_characteristics: ProboscisCharacteristics = Field(
        description="Proboscis characteristics"
    )
    body_size_mm: float = Field(ge=0.0, description="Body size in millimeters")
    wing_length_mm: float = Field(ge=0.0, description="Wing length in millimeters")
    weight_mg: float = Field(ge=0.0, description="Body weight in milligrams")

    # Life history
    development_parameters: DevelopmentParameters = Field(
        description="Development stage parameters"
    )
    max_lifespan_workers: int = Field(
        ge=1, description="Maximum worker lifespan in days"
    )
    max_lifespan_queens: int = Field(ge=1, description="Maximum queen lifespan in days")
    max_lifespan_drones: int = Field(ge=1, description="Maximum drone lifespan in days")

    # Phenology
    emerging_day_mean: int = Field(
        ge=1, le=365, description="Mean emergence day of year"
    )
    emerging_day_sd: float = Field(
        ge=0.0, description="Emergence day standard deviation"
    )
    active_season_start: int = Field(
        ge=1, le=365, description="Active season start day"
    )
    active_season_end: int = Field(ge=1, le=365, description="Active season end day")

    # Foraging behavior
    flight_velocity_ms: float = Field(ge=0.0, description="Flight velocity in m/s")
    foraging_range_m: float = Field(ge=0.0, description="Foraging range in meters")
    search_length_m: float = Field(ge=0.0, description="Foraging search distance")
    nectar_load_capacity_mg: float = Field(
        ge=0.0, description="Nectar carrying capacity"
    )
    pollen_load_capacity_mg: float = Field(
        ge=0.0, description="Pollen carrying capacity"
    )

    # Colony characteristics
    max_colony_size: int = Field(ge=1, description="Maximum colony size")
    typical_colony_size: int = Field(ge=1, description="Typical colony size")
    brood_development_time: float = Field(
        ge=0.0, description="Brood development time in days"
    )

    # Habitat preferences
    nest_habitat_preferences: Dict[str, float] = Field(
        description="Nest habitat preference scores"
    )
    foraging_habitat_preferences: Dict[str, float] = Field(
        description="Foraging habitat preference scores"
    )

    # Ecological traits
    cold_tolerance: float = Field(
        ge=0.0, le=1.0, description="Cold tolerance (0-1 scale)"
    )
    drought_tolerance: float = Field(
        ge=0.0, le=1.0, description="Drought tolerance (0-1 scale)"
    )
    competition_strength: float = Field(
        ge=0.0, le=1.0, description="Competition strength (0-1 scale)"
    )

    # Behavioral traits
    foraging_aggressiveness: float = Field(
        ge=0.0, le=1.0, description="Foraging aggressiveness (0-1 scale)"
    )
    site_fidelity: float = Field(
        ge=0.0, le=1.0, description="Site fidelity (0-1 scale)"
    )
    social_dominance: float = Field(
        ge=0.0, le=1.0, description="Social dominance (0-1 scale)"
    )


class SpeciesSystem:
    """
    Manages multi-species bumblebee communities.

    Implements:
    - Species-specific parameters and behaviors
    - Interspecies competition and resource partitioning
    - Phenological differences and niche separation
    - Community composition and dynamics
    """

    def __init__(self) -> None:
        self.species_parameters: Dict[str, SpeciesParameters] = {}
        self.species_interactions: Dict[Tuple[str, str], float] = {}
        self.community_composition: Dict[str, int] = {}

        # Initialize UK bumblebee species
        self.initialize_uk_species()

        # Initialize species interactions
        self.initialize_species_interactions()

        self.logger = logging.getLogger(__name__)

    def initialize_uk_species(self) -> None:
        """Initialize UK bumblebee species with realistic parameters"""

        # Bombus terrestris - Short-tongued, dominant, early emerging
        self.species_parameters["Bombus_terrestris"] = SpeciesParameters(
            species_name="Bombus_terrestris",
            species_type=SpeciesType.BOMBUS_TERRESTRIS,
            proboscis_characteristics=ProboscisCharacteristics(
                length_mm=7.2, width_mm=0.25, flexibility=0.85, extension_efficiency=0.9
            ),
            body_size_mm=22.0,
            wing_length_mm=18.0,
            weight_mg=850.0,
            development_parameters=DevelopmentParameters(
                dev_age_hatching_min=3.0,
                dev_age_pupation_min=12.0,
                dev_age_emerging_min=18.0,
                dev_weight_egg=0.15,
                dev_weight_pupation_min=110.0,
                temperature_optimal=32.0,
            ),
            max_lifespan_workers=35,
            max_lifespan_queens=365,
            max_lifespan_drones=28,
            emerging_day_mean=60,  # Early March
            emerging_day_sd=10.0,
            active_season_start=50,
            active_season_end=280,
            flight_velocity_ms=3.2,
            foraging_range_m=1500.0,
            search_length_m=800.0,
            nectar_load_capacity_mg=45.0,
            pollen_load_capacity_mg=12.0,
            max_colony_size=400,
            typical_colony_size=200,
            brood_development_time=21.0,
            nest_habitat_preferences={
                "woodland": 0.3,
                "hedgerow": 0.7,
                "grassland": 0.5,
                "urban": 0.8,
            },
            foraging_habitat_preferences={
                "cropland": 0.9,
                "wildflower": 0.8,
                "hedgerow": 0.6,
                "grassland": 0.7,
            },
            cold_tolerance=0.8,
            drought_tolerance=0.7,
            competition_strength=0.9,
            foraging_aggressiveness=0.8,
            site_fidelity=0.6,
            social_dominance=0.9,
        )

        # Bombus lucorum - Short-tongued, early emerging, cold tolerant
        self.species_parameters["Bombus_lucorum"] = SpeciesParameters(
            species_name="Bombus_lucorum",
            species_type=SpeciesType.BOMBUS_LUCORUM,
            proboscis_characteristics=ProboscisCharacteristics(
                length_mm=7.8,
                width_mm=0.23,
                flexibility=0.82,
                extension_efficiency=0.88,
            ),
            body_size_mm=20.0,
            wing_length_mm=17.0,
            weight_mg=750.0,
            development_parameters=DevelopmentParameters(
                dev_age_hatching_min=3.5,
                dev_age_pupation_min=13.0,
                dev_age_emerging_min=19.0,
                dev_weight_egg=0.14,
                dev_weight_pupation_min=105.0,
                temperature_optimal=30.0,
            ),
            max_lifespan_workers=32,
            max_lifespan_queens=350,
            max_lifespan_drones=25,
            emerging_day_mean=55,  # Mid March
            emerging_day_sd=12.0,
            active_season_start=45,
            active_season_end=270,
            flight_velocity_ms=3.0,
            foraging_range_m=1200.0,
            search_length_m=700.0,
            nectar_load_capacity_mg=40.0,
            pollen_load_capacity_mg=11.0,
            max_colony_size=350,
            typical_colony_size=180,
            brood_development_time=22.0,
            nest_habitat_preferences={
                "woodland": 0.6,
                "hedgerow": 0.8,
                "grassland": 0.7,
                "urban": 0.4,
            },
            foraging_habitat_preferences={
                "wildflower": 0.9,
                "hedgerow": 0.8,
                "grassland": 0.8,
                "cropland": 0.6,
            },
            cold_tolerance=0.9,
            drought_tolerance=0.6,
            competition_strength=0.7,
            foraging_aggressiveness=0.6,
            site_fidelity=0.8,
            social_dominance=0.7,
        )

        # Bombus lapidarius - Short-tongued, heat tolerant
        self.species_parameters["Bombus_lapidarius"] = SpeciesParameters(
            species_name="Bombus_lapidarius",
            species_type=SpeciesType.BOMBUS_LAPIDARIUS,
            proboscis_characteristics=ProboscisCharacteristics(
                length_mm=8.1,
                width_mm=0.24,
                flexibility=0.83,
                extension_efficiency=0.87,
            ),
            body_size_mm=18.0,
            wing_length_mm=16.0,
            weight_mg=680.0,
            development_parameters=DevelopmentParameters(
                dev_age_hatching_min=3.2,
                dev_age_pupation_min=12.5,
                dev_age_emerging_min=18.5,
                dev_weight_egg=0.13,
                dev_weight_pupation_min=100.0,
                temperature_optimal=33.0,
            ),
            max_lifespan_workers=38,
            max_lifespan_queens=340,
            max_lifespan_drones=30,
            emerging_day_mean=80,  # Late March
            emerging_day_sd=15.0,
            active_season_start=70,
            active_season_end=290,
            flight_velocity_ms=3.5,
            foraging_range_m=1800.0,
            search_length_m=900.0,
            nectar_load_capacity_mg=38.0,
            pollen_load_capacity_mg=10.0,
            max_colony_size=300,
            typical_colony_size=150,
            brood_development_time=20.0,
            nest_habitat_preferences={
                "woodland": 0.4,
                "hedgerow": 0.6,
                "grassland": 0.8,
                "urban": 0.7,
            },
            foraging_habitat_preferences={
                "wildflower": 0.8,
                "grassland": 0.9,
                "hedgerow": 0.7,
                "cropland": 0.5,
            },
            cold_tolerance=0.6,
            drought_tolerance=0.8,
            competition_strength=0.6,
            foraging_aggressiveness=0.7,
            site_fidelity=0.7,
            social_dominance=0.6,
        )

        # Bombus pratorum - Small, early emerging, short-lived
        self.species_parameters["Bombus_pratorum"] = SpeciesParameters(
            species_name="Bombus_pratorum",
            species_type=SpeciesType.BOMBUS_PRATORUM,
            proboscis_characteristics=ProboscisCharacteristics(
                length_mm=9.4,
                width_mm=0.22,
                flexibility=0.88,
                extension_efficiency=0.91,
            ),
            body_size_mm=14.0,
            wing_length_mm=13.0,
            weight_mg=420.0,
            development_parameters=DevelopmentParameters(
                dev_age_hatching_min=2.8,
                dev_age_pupation_min=11.0,
                dev_age_emerging_min=16.0,
                dev_weight_egg=0.10,
                dev_weight_pupation_min=85.0,
                temperature_optimal=31.0,
            ),
            max_lifespan_workers=25,
            max_lifespan_queens=280,
            max_lifespan_drones=20,
            emerging_day_mean=65,  # Early March
            emerging_day_sd=8.0,
            active_season_start=55,
            active_season_end=220,  # Short season
            flight_velocity_ms=2.8,
            foraging_range_m=800.0,
            search_length_m=400.0,
            nectar_load_capacity_mg=25.0,
            pollen_load_capacity_mg=7.0,
            max_colony_size=100,
            typical_colony_size=60,
            brood_development_time=18.0,
            nest_habitat_preferences={
                "woodland": 0.8,
                "hedgerow": 0.9,
                "grassland": 0.6,
                "urban": 0.5,
            },
            foraging_habitat_preferences={
                "wildflower": 0.9,
                "woodland": 0.8,
                "hedgerow": 0.8,
                "grassland": 0.7,
            },
            cold_tolerance=0.8,
            drought_tolerance=0.5,
            competition_strength=0.4,
            foraging_aggressiveness=0.5,
            site_fidelity=0.9,
            social_dominance=0.4,
        )

        # Bombus pascuorum - Medium-tongued, late emerging, long season
        self.species_parameters["Bombus_pascuorum"] = SpeciesParameters(
            species_name="Bombus_pascuorum",
            species_type=SpeciesType.BOMBUS_PASCUORUM,
            proboscis_characteristics=ProboscisCharacteristics(
                length_mm=11.2,
                width_mm=0.21,
                flexibility=0.92,
                extension_efficiency=0.93,
            ),
            body_size_mm=16.0,
            wing_length_mm=15.0,
            weight_mg=520.0,
            development_parameters=DevelopmentParameters(
                dev_age_hatching_min=3.5,
                dev_age_pupation_min=14.0,
                dev_age_emerging_min=20.0,
                dev_weight_egg=0.12,
                dev_weight_pupation_min=95.0,
                temperature_optimal=32.0,
            ),
            max_lifespan_workers=40,
            max_lifespan_queens=330,
            max_lifespan_drones=32,
            emerging_day_mean=100,  # Mid April
            emerging_day_sd=18.0,
            active_season_start=85,
            active_season_end=310,  # Long season
            flight_velocity_ms=3.1,
            foraging_range_m=1400.0,
            search_length_m=750.0,
            nectar_load_capacity_mg=35.0,
            pollen_load_capacity_mg=9.0,
            max_colony_size=200,
            typical_colony_size=120,
            brood_development_time=22.0,
            nest_habitat_preferences={
                "woodland": 0.5,
                "hedgerow": 0.7,
                "grassland": 0.9,
                "urban": 0.6,
            },
            foraging_habitat_preferences={
                "wildflower": 0.9,
                "grassland": 0.8,
                "hedgerow": 0.8,
                "cropland": 0.7,
            },
            cold_tolerance=0.7,
            drought_tolerance=0.7,
            competition_strength=0.6,
            foraging_aggressiveness=0.6,
            site_fidelity=0.8,
            social_dominance=0.5,
        )

        # Bombus hortorum - Long-tongued, specialist
        self.species_parameters["Bombus_hortorum"] = SpeciesParameters(
            species_name="Bombus_hortorum",
            species_type=SpeciesType.BOMBUS_HORTORUM,
            proboscis_characteristics=ProboscisCharacteristics(
                length_mm=15.3,
                width_mm=0.19,
                flexibility=0.95,
                extension_efficiency=0.96,
            ),
            body_size_mm=20.0,
            wing_length_mm=17.0,
            weight_mg=720.0,
            development_parameters=DevelopmentParameters(
                dev_age_hatching_min=4.0,
                dev_age_pupation_min=15.0,
                dev_age_emerging_min=22.0,
                dev_weight_egg=0.16,
                dev_weight_pupation_min=115.0,
                temperature_optimal=31.0,
            ),
            max_lifespan_workers=45,
            max_lifespan_queens=320,
            max_lifespan_drones=35,
            emerging_day_mean=85,  # Late March
            emerging_day_sd=20.0,
            active_season_start=75,
            active_season_end=285,
            flight_velocity_ms=2.9,
            foraging_range_m=2000.0,  # Longer range for deep flowers
            search_length_m=1200.0,
            nectar_load_capacity_mg=50.0,
            pollen_load_capacity_mg=13.0,
            max_colony_size=250,
            typical_colony_size=130,
            brood_development_time=24.0,
            nest_habitat_preferences={
                "woodland": 0.7,
                "hedgerow": 0.8,
                "grassland": 0.6,
                "urban": 0.5,
            },
            foraging_habitat_preferences={
                "wildflower": 0.9,
                "woodland": 0.8,
                "hedgerow": 0.9,
                "grassland": 0.6,
            },
            cold_tolerance=0.7,
            drought_tolerance=0.6,
            competition_strength=0.5,
            foraging_aggressiveness=0.4,
            site_fidelity=0.9,
            social_dominance=0.5,
        )

        # Bombus ruderatus - Very long-tongued, rare specialist
        self.species_parameters["Bombus_ruderatus"] = SpeciesParameters(
            species_name="Bombus_ruderatus",
            species_type=SpeciesType.BOMBUS_RUDERATUS,
            proboscis_characteristics=ProboscisCharacteristics(
                length_mm=17.1,
                width_mm=0.18,
                flexibility=0.97,
                extension_efficiency=0.98,
            ),
            body_size_mm=24.0,
            wing_length_mm=20.0,
            weight_mg=920.0,
            development_parameters=DevelopmentParameters(
                dev_age_hatching_min=4.5,
                dev_age_pupation_min=16.0,
                dev_age_emerging_min=24.0,
                dev_weight_egg=0.18,
                dev_weight_pupation_min=125.0,
                temperature_optimal=30.0,
            ),
            max_lifespan_workers=50,
            max_lifespan_queens=300,
            max_lifespan_drones=40,
            emerging_day_mean=95,  # Early April
            emerging_day_sd=25.0,
            active_season_start=85,
            active_season_end=275,
            flight_velocity_ms=2.7,
            foraging_range_m=2500.0,  # Longest range
            search_length_m=1500.0,
            nectar_load_capacity_mg=55.0,
            pollen_load_capacity_mg=14.0,
            max_colony_size=180,
            typical_colony_size=90,
            brood_development_time=26.0,
            nest_habitat_preferences={
                "woodland": 0.8,
                "hedgerow": 0.9,
                "grassland": 0.5,
                "urban": 0.3,
            },
            foraging_habitat_preferences={
                "wildflower": 0.9,
                "woodland": 0.9,
                "hedgerow": 0.8,
                "grassland": 0.4,
            },
            cold_tolerance=0.6,
            drought_tolerance=0.5,
            competition_strength=0.4,
            foraging_aggressiveness=0.3,
            site_fidelity=0.9,
            social_dominance=0.3,
        )

    def initialize_species_interactions(self) -> None:
        """Initialize competitive interactions between species"""

        # Define competitive strength matrix
        # Higher values = stronger competition

        species_list = list(self.species_parameters.keys())

        for i, species1 in enumerate(species_list):
            for j, species2 in enumerate(species_list):
                if i != j:
                    # Base interaction strength
                    params1 = self.species_parameters[species1]
                    params2 = self.species_parameters[species2]

                    # Size-based competition
                    size_ratio = params1.body_size_mm / params2.body_size_mm

                    # Dominance-based competition
                    dominance_diff = params1.social_dominance - params2.social_dominance

                    # Temporal overlap
                    overlap = self.calculate_temporal_overlap(species1, species2)

                    # Resource overlap (based on proboscis length similarity)
                    resource_overlap = self.calculate_resource_overlap(
                        species1, species2
                    )

                    # Combined interaction strength
                    interaction_strength = (
                        size_ratio * 0.3
                        + dominance_diff * 0.4
                        + overlap * 0.2
                        + resource_overlap * 0.1
                    )

                    self.species_interactions[(species1, species2)] = max(
                        0.0, interaction_strength
                    )

    def calculate_temporal_overlap(self, species1: str, species2: str) -> float:
        """Calculate temporal niche overlap between species"""

        params1 = self.species_parameters[species1]
        params2 = self.species_parameters[species2]

        # Calculate overlap in active seasons
        overlap_start = max(params1.active_season_start, params2.active_season_start)
        overlap_end = min(params1.active_season_end, params2.active_season_end)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_days = overlap_end - overlap_start

        # Normalize by shorter season length
        season1_length = params1.active_season_end - params1.active_season_start
        season2_length = params2.active_season_end - params2.active_season_start
        min_season_length = min(season1_length, season2_length)

        return overlap_days / min_season_length if min_season_length > 0 else 0.0

    def calculate_resource_overlap(self, species1: str, species2: str) -> float:
        """Calculate resource niche overlap between species"""

        params1 = self.species_parameters[species1]
        params2 = self.species_parameters[species2]

        # Proboscis length similarity
        proboscis1 = params1.proboscis_characteristics.length_mm
        proboscis2 = params2.proboscis_characteristics.length_mm

        length_diff = abs(proboscis1 - proboscis2)
        max_length = max(proboscis1, proboscis2)

        length_similarity = 1.0 - (length_diff / max_length) if max_length > 0 else 1.0

        # Foraging range overlap
        range1 = params1.foraging_range_m
        range2 = params2.foraging_range_m

        range_diff = abs(range1 - range2)
        max_range = max(range1, range2)

        range_similarity = 1.0 - (range_diff / max_range) if max_range > 0 else 1.0

        return (length_similarity + range_similarity) / 2.0

    def get_species_parameters(self, species_name: str) -> Optional[SpeciesParameters]:
        """Get parameters for specific species"""
        return self.species_parameters.get(species_name)

    def get_available_species(self, day_of_year: int) -> List[str]:
        """Get species that are active on given day"""
        available_species = []

        for species_name, params in self.species_parameters.items():
            if params.active_season_start <= day_of_year <= params.active_season_end:
                available_species.append(species_name)

        return available_species

    def get_emerging_species(self, day_of_year: int) -> List[str]:
        """Get species that are emerging on given day"""
        emerging_species = []

        for species_name, params in self.species_parameters.items():
            # Check if day is within emergence window
            emergence_start = params.emerging_day_mean - 2 * params.emerging_day_sd
            emergence_end = params.emerging_day_mean + 2 * params.emerging_day_sd

            if emergence_start <= day_of_year <= emergence_end:
                emerging_species.append(species_name)

        return emerging_species

    def calculate_competition_effect(
        self, focal_species: str, competitor_species: str, resource_overlap: float
    ) -> float:
        """Calculate competitive effect of competitor on focal species"""

        if focal_species == competitor_species:
            return 0.0

        interaction_key = (competitor_species, focal_species)
        base_competition = self.species_interactions.get(interaction_key, 0.0)

        # Scale by resource overlap
        competition_effect = base_competition * resource_overlap

        return min(1.0, competition_effect)

    def get_optimal_habitats(
        self, species_name: str, habitat_type: str
    ) -> Tuple[float, float]:
        """Get habitat preferences for species"""

        params = self.species_parameters.get(species_name)
        if not params:
            return 0.5, 0.5

        nest_preference = params.nest_habitat_preferences.get(habitat_type, 0.5)
        foraging_preference = params.foraging_habitat_preferences.get(habitat_type, 0.5)

        return nest_preference, foraging_preference

    def simulate_community_assembly(
        self,
        landscape_capacity: Dict[str, float],
        environmental_conditions: Dict[str, float],
    ) -> Dict[str, int]:
        """Simulate multi-species community assembly"""

        community: Dict[str, int] = {}

        # Calculate species suitability
        species_suitability = {}

        for species_name, params in self.species_parameters.items():
            # Environmental suitability
            temp_suitability = 1.0  # Could be based on temperature tolerance

            # Habitat suitability (simplified)
            habitat_suitability = np.mean(
                list(params.foraging_habitat_preferences.values())
            )

            # Phenological suitability
            day_of_year = environmental_conditions.get("day_of_year", 150)
            phenological_suitability = (
                1.0
                if (
                    params.active_season_start
                    <= day_of_year
                    <= params.active_season_end
                )
                else 0.0
            )

            # Combined suitability
            suitability = (
                temp_suitability * habitat_suitability * phenological_suitability
            )
            species_suitability[species_name] = suitability

        # Assign colonies based on suitability and competition
        total_capacity = landscape_capacity.get("overall", 0)

        # Sort species by suitability
        sorted_species = sorted(
            species_suitability.items(), key=lambda x: float(x[1]), reverse=True
        )

        remaining_capacity = total_capacity

        for species_name, suitability in sorted_species:
            if remaining_capacity <= 0:
                break

            if suitability > 0:
                # Calculate potential colony count
                params = self.species_parameters[species_name]

                # Base colony count from suitability
                base_colonies = int(
                    suitability * total_capacity * 0.2
                )  # Max 20% per species

                # Adjust for competition
                competition_factor = 1.0
                for competitor in community.keys():
                    if competitor != species_name:
                        competition_effect = self.calculate_competition_effect(
                            species_name,
                            competitor,
                            0.5,  # Assume moderate resource overlap
                        )
                        competition_factor *= 1.0 - competition_effect * 0.5

                # Final colony count
                colony_count = max(0, int(base_colonies * competition_factor))
                colony_count = min(colony_count, int(remaining_capacity))

                if colony_count > 0:
                    community[species_name] = colony_count
                    remaining_capacity -= colony_count

        return community

    def get_species_phenology(self, species_name: str) -> Dict[str, Any]:
        """Get species phenology information"""

        params = self.species_parameters.get(species_name)
        if not params:
            return {}

        return {
            "emerging_day_mean": params.emerging_day_mean,
            "emerging_day_sd": params.emerging_day_sd,
            "active_season_start": params.active_season_start,
            "active_season_end": params.active_season_end,
            "season_length": params.active_season_end - params.active_season_start,
        }

    def get_community_diversity_metrics(
        self, community: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculate community diversity metrics"""

        if not community:
            return {"richness": 0.0, "shannon": 0.0, "evenness": 0.0}

        # Species richness
        richness = len(community)

        # Shannon diversity
        total_colonies = sum(community.values())
        shannon = 0.0

        if total_colonies > 0:
            for count in community.values():
                if count > 0:
                    proportion = count / total_colonies
                    shannon -= proportion * np.log(proportion)

        # Pielou's evenness
        evenness = shannon / np.log(richness) if richness > 1 else 0.0

        return {
            "richness": richness,
            "shannon": shannon,
            "evenness": evenness,
            "total_colonies": total_colonies,
        }

    def get_species_trait_summary(self) -> Dict[str, Any]:
        """Get summary of species traits for analysis"""

        trait_summary: Dict[str, Any] = {
            "species_count": len(self.species_parameters),
            "proboscis_lengths": {},
            "body_sizes": {},
            "foraging_ranges": {},
            "colony_sizes": {},
            "phenology": {},
        }

        for species_name, params in self.species_parameters.items():
            trait_summary["proboscis_lengths"][species_name] = (
                params.proboscis_characteristics.length_mm
            )
            trait_summary["body_sizes"][species_name] = params.body_size_mm
            trait_summary["foraging_ranges"][species_name] = params.foraging_range_m
            trait_summary["colony_sizes"][species_name] = params.typical_colony_size
            trait_summary["phenology"][species_name] = {
                "emergence": params.emerging_day_mean,
                "season_start": params.active_season_start,
                "season_end": params.active_season_end,
            }

        return trait_summary

    def predict_species_response_to_management(
        self, management_type: str, intensity: float
    ) -> Dict[str, float]:
        """Predict species response to management practices"""

        responses = {}

        for species_name, params in self.species_parameters.items():
            if management_type == "wildflower_strips":
                # Positive response, stronger for long-tongued species
                base_response = 0.3
                proboscis_bonus = (
                    params.proboscis_characteristics.length_mm - 7.0
                ) / 10.0
                response = base_response + proboscis_bonus * 0.2

            elif management_type == "mowing":
                # Negative response, stronger for site-faithful species
                base_response = -0.4
                fidelity_penalty = params.site_fidelity * -0.3
                response = base_response + fidelity_penalty

            elif management_type == "pesticide":
                # Negative response, stronger for less competitive species
                base_response = -0.6
                competition_effect = (1.0 - params.competition_strength) * -0.4
                response = base_response + competition_effect

            else:
                response = 0.0

            responses[species_name] = max(-1.0, min(1.0, response * intensity))

        return responses
