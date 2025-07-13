"""
Resource distribution system for BSTEW
======================================

Manages resource distribution patterns and temporal dynamics.
Handles seasonal flowering, weather effects, and resource management.
"""

from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field
import math
import json

from .patches import ResourcePatch, HabitatType, FlowerSpecies


class SeasonalPattern(BaseModel):
    """Seasonal resource production pattern"""

    model_config = {"validate_assignment": True}

    peak_start: int = Field(ge=1, le=365, description="Peak start day of year")
    peak_end: int = Field(ge=1, le=365, description="Peak end day of year")
    peak_intensity: float = Field(ge=0.0, description="Multiplier at peak")
    base_intensity: float = Field(ge=0.0, description="Multiplier at minimum")

    def get_intensity(self, day_of_year: int) -> float:
        """Get intensity for given day of year"""
        if self.peak_start <= day_of_year <= self.peak_end:
            # Within peak period
            peak_center = (self.peak_start + self.peak_end) / 2
            distance_from_center = abs(day_of_year - peak_center)
            peak_half_width = (self.peak_end - self.peak_start) / 2

            # Cosine interpolation within peak
            if peak_half_width > 0:
                intensity_factor = math.cos(
                    math.pi * distance_from_center / peak_half_width
                )
                return self.base_intensity + (
                    self.peak_intensity - self.base_intensity
                ) * max(0, intensity_factor)
            else:
                return self.peak_intensity
        else:
            return self.base_intensity


class ResourceDistribution:
    """
    Manages resource distribution across landscape.

    Handles:
    - Seasonal flowering patterns
    - Weather-dependent resource availability
    - Resource quality gradients
    - Habitat-specific resource production
    """

    def __init__(self, landscape_grid: Any) -> None:
        self.landscape = landscape_grid

        # Seasonal patterns by habitat type
        self.seasonal_patterns = self.get_default_seasonal_patterns()

        # Weather impact parameters
        self.weather_sensitivity = {
            HabitatType.WILDFLOWER: 0.8,
            HabitatType.HEDGEROW: 0.6,
            HabitatType.CROPLAND: 0.9,
            HabitatType.GRASSLAND: 0.5,
            HabitatType.WOODLAND: 0.3,
        }

        # Resource quality modifiers
        self.quality_modifiers: Dict[str, float] = {}

        # Flowering species database
        self.species_database = self.load_species_database()

    def get_default_seasonal_patterns(self) -> Dict[HabitatType, SeasonalPattern]:
        """Get default seasonal patterns for each habitat type"""
        return {
            HabitatType.WILDFLOWER: SeasonalPattern(
                peak_start=120, peak_end=240, peak_intensity=2.0, base_intensity=0.1
            ),  # May-August peak
            HabitatType.HEDGEROW: SeasonalPattern(
                peak_start=100, peak_end=200, peak_intensity=1.8, base_intensity=0.2
            ),  # April-July peak
            HabitatType.CROPLAND: SeasonalPattern(
                peak_start=110, peak_end=180, peak_intensity=3.0, base_intensity=0.0
            ),  # April-June peak
            HabitatType.GRASSLAND: SeasonalPattern(
                peak_start=130, peak_end=220, peak_intensity=1.5, base_intensity=0.3
            ),  # May-July peak
            HabitatType.WOODLAND: SeasonalPattern(
                peak_start=90, peak_end=150, peak_intensity=1.2, base_intensity=0.1
            ),  # March-May peak
        }

    def load_species_database(self) -> Dict[str, FlowerSpecies]:
        """Load flowering species database"""
        # In a full implementation, this would load from external files
        species_db = {}

        # Common UK flowering species with corolla depth data
        species_data = [
            # (name, bloom_start, bloom_end, nectar_prod, pollen_prod, density, attract, corolla_depth, corolla_width)
            ("White Clover", 120, 270, 4.0, 2.0, 25.0, 0.85, 4.5, 1.8),
            ("Red Clover", 140, 250, 6.0, 3.0, 20.0, 0.90, 9.2, 2.1),
            ("Dandelion", 80, 300, 3.0, 1.5, 30.0, 0.70, 3.8, 1.5),
            ("Bramble", 150, 210, 8.0, 4.0, 15.0, 0.95, 5.1, 2.0),
            ("Hawthorn", 120, 150, 6.0, 4.0, 8.0, 0.80, 2.9, 1.2),
            ("Blackthorn", 90, 120, 4.0, 3.0, 10.0, 0.75, 3.2, 1.3),
            ("Oilseed Rape", 100, 140, 15.0, 8.0, 150.0, 0.95, 2.1, 1.0),
            ("Field Bean", 140, 180, 3.0, 8.0, 60.0, 0.75, 8.7, 2.2),
            ("Heather", 210, 270, 5.0, 2.0, 40.0, 0.85, 4.6, 1.8),
            ("Gorse", 60, 300, 4.0, 2.5, 20.0, 0.80, 7.3, 2.0),
            ("Lime Tree", 150, 180, 10.0, 2.0, 5.0, 0.90, 4.8, 1.9),
            ("Phacelia", 160, 220, 12.0, 6.0, 80.0, 0.95, 3.9, 1.6),
            ("Borage", 150, 240, 8.0, 3.0, 50.0, 0.85, 5.4, 2.1),
            ("Lavender", 160, 220, 6.0, 2.0, 30.0, 0.90, 6.8, 2.0),
            ("Rosemary", 100, 200, 3.0, 1.0, 20.0, 0.70, 8.1, 1.9),
            ("Foxglove", 160, 200, 2.0, 1.0, 5.0, 0.60, 25.0, 3.0),
            ("Comfrey", 140, 220, 7.0, 3.0, 10.0, 0.75, 12.5, 2.8),
            ("Viper's Bugloss", 150, 240, 5.0, 2.0, 15.0, 0.80, 7.9, 2.2),
            ("Honeysuckle", 160, 220, 8.0, 1.0, 8.0, 0.85, 18.7, 2.5),
        ]

        for (
            name,
            start,
            end,
            nectar,
            pollen,
            density,
            attract,
            corolla_depth,
            corolla_width,
        ) in species_data:
            # Calculate nectar accessibility based on corolla depth
            if corolla_depth > 15.0:
                nectar_accessibility = 0.6  # Deep flowers less accessible
            elif corolla_depth > 10.0:
                nectar_accessibility = 0.8
            else:
                nectar_accessibility = 1.0  # Shallow flowers fully accessible

            species_db[name] = FlowerSpecies(
                name=name,
                bloom_start=start,
                bloom_end=end,
                nectar_production=nectar,
                pollen_production=pollen,
                flower_density=density,
                attractiveness=attract,
                corolla_depth_mm=corolla_depth,
                corolla_width_mm=corolla_width,
                nectar_accessibility=nectar_accessibility,
            )

        return species_db

    def update_landscape_resources(
        self, day_of_year: int, weather: Dict[str, float]
    ) -> None:
        """Update all landscape resources for current day"""
        for patch in self.landscape.patches.values():
            self.update_patch_resources(patch, day_of_year, weather)

    def update_patch_resources(
        self, patch: ResourcePatch, day_of_year: int, weather: Dict[str, float]
    ) -> None:
        """Update resources for individual patch"""
        # Get seasonal pattern for habitat
        seasonal_pattern = self.seasonal_patterns.get(patch.habitat_type)
        if seasonal_pattern is None:
            return

        # Calculate seasonal intensity
        seasonal_intensity = seasonal_pattern.get_intensity(day_of_year)

        # Calculate weather impact
        weather_impact = self.calculate_weather_impact(patch, weather)

        # Update patch resources
        patch.update_resources(day_of_year, weather)

        # Apply seasonal and weather modifiers
        patch.current_nectar *= seasonal_intensity * weather_impact
        patch.current_pollen *= seasonal_intensity * weather_impact

        # Apply habitat-specific limits
        max_nectar, max_pollen = self.get_habitat_limits(patch.habitat_type)
        patch.current_nectar = min(patch.current_nectar, max_nectar)
        patch.current_pollen = min(patch.current_pollen, max_pollen)

    def calculate_weather_impact(
        self, patch: ResourcePatch, weather: Dict[str, float]
    ) -> float:
        """Calculate weather impact on resource production"""
        sensitivity = self.weather_sensitivity.get(patch.habitat_type, 0.5)
        base_impact = patch.calculate_weather_factor(weather)

        # Modify impact based on habitat sensitivity
        return 1.0 + sensitivity * (base_impact - 1.0)

    def get_habitat_limits(self, habitat_type: HabitatType) -> Tuple[float, float]:
        """Get maximum resource limits for habitat type"""
        limits = {
            HabitatType.WILDFLOWER: (1000.0, 500.0),
            HabitatType.HEDGEROW: (800.0, 400.0),
            HabitatType.CROPLAND: (1500.0, 750.0),
            HabitatType.GRASSLAND: (400.0, 200.0),
            HabitatType.WOODLAND: (300.0, 150.0),
        }
        return limits.get(habitat_type, (100.0, 50.0))

    def create_resource_gradient(
        self, center: Tuple[float, float], radius: float, max_intensity: float
    ) -> None:
        """Create a resource quality gradient around a center point"""
        for patch in self.landscape.patches.values():
            distance = self.landscape.calculate_distance(patch.location, center)

            if distance <= radius:
                # Gaussian gradient
                intensity = max_intensity * math.exp(-((distance / radius) ** 2))

                # Apply to patch
                patch.base_nectar_production *= 1.0 + intensity
                patch.base_pollen_production *= 1.0 + intensity

    def add_resource_hotspot(
        self, center: Tuple[float, float], radius: float, species_list: List[str]
    ) -> None:
        """Add a resource hotspot with specific flowering species"""
        patches = self.landscape.get_patches_in_radius(center, radius)

        for patch in patches:
            # Add species to patch
            for species_name in species_list:
                if species_name in self.species_database:
                    species = self.species_database[species_name]

                    # Create copy with adjusted density based on distance
                    distance = self.landscape.calculate_distance(patch.location, center)
                    density_factor = max(0.1, 1.0 - (distance / radius))

                    adjusted_species = FlowerSpecies(
                        name=species.name,
                        bloom_start=species.bloom_start,
                        bloom_end=species.bloom_end,
                        nectar_production=species.nectar_production,
                        pollen_production=species.pollen_production,
                        flower_density=species.flower_density * density_factor,
                        attractiveness=species.attractiveness,
                    )

                    patch.flower_species.append(adjusted_species)

    def simulate_agricultural_management(self, day_of_year: int) -> None:
        """Simulate agricultural management effects"""
        # Harvesting effects
        if 200 <= day_of_year <= 220:  # August harvesting
            cropland_patches = self.landscape.get_patches_by_habitat(
                HabitatType.CROPLAND
            )

            for patch in cropland_patches:
                # Remove flowering crops
                patch.flower_species = [
                    s for s in patch.flower_species if s.bloom_end < day_of_year
                ]
                patch.current_nectar *= 0.1
                patch.current_pollen *= 0.1

        # Planting effects
        if 80 <= day_of_year <= 120:  # Spring planting
            cropland_patches = self.landscape.get_patches_by_habitat(
                HabitatType.CROPLAND
            )

            for patch in cropland_patches:
                # Add new crops
                if len(patch.flower_species) == 0:
                    # Plant oilseed rape
                    if "Oilseed Rape" in self.species_database:
                        patch.flower_species.append(
                            self.species_database["Oilseed Rape"]
                        )

    def apply_conservation_measures(
        self, measure_type: str, target_patches: List[ResourcePatch]
    ) -> None:
        """Apply conservation measures to target patches"""
        if measure_type == "wildflower_strips":
            for patch in target_patches:
                # Convert to wildflower habitat
                patch.habitat_type = HabitatType.WILDFLOWER
                patch.initialize_habitat_properties()

                # Add diverse wildflower species
                wildflower_species = [
                    "White Clover",
                    "Red Clover",
                    "Phacelia",
                    "Borage",
                ]
                for species_name in wildflower_species:
                    if species_name in self.species_database:
                        patch.flower_species.append(self.species_database[species_name])

        elif measure_type == "hedgerow_creation":
            for patch in target_patches:
                patch.habitat_type = HabitatType.HEDGEROW
                patch.initialize_habitat_properties()

                # Add hedgerow species
                hedgerow_species = ["Hawthorn", "Blackthorn", "Bramble"]
                for species_name in hedgerow_species:
                    if species_name in self.species_database:
                        patch.flower_species.append(self.species_database[species_name])

        elif measure_type == "reduced_mowing":
            for patch in target_patches:
                # Increase flowering period and density
                for species in patch.flower_species:
                    species.bloom_end = min(300, species.bloom_end + 30)
                    species.flower_density *= 1.5

    def calculate_landscape_carrying_capacity(self) -> Dict[str, float]:
        """Calculate landscape carrying capacity for bee colonies"""
        total_nectar_production = 0.0
        total_pollen_production = 0.0

        # Calculate annual production
        for day in range(1, 366):
            daily_nectar = 0.0
            daily_pollen = 0.0

            for patch in self.landscape.patches.values():
                seasonal_pattern = self.seasonal_patterns.get(patch.habitat_type)
                if seasonal_pattern:
                    intensity = seasonal_pattern.get_intensity(day)
                    daily_nectar += patch.base_nectar_production * intensity
                    daily_pollen += patch.base_pollen_production * intensity

            total_nectar_production += daily_nectar
            total_pollen_production += daily_pollen

        # Estimate colony carrying capacity
        # Assume a colony needs ~50kg nectar and ~20kg pollen per year
        nectar_colonies = total_nectar_production / 50000  # mg to kg conversion
        pollen_colonies = total_pollen_production / 20000

        return {
            "nectar_limited": nectar_colonies,
            "pollen_limited": pollen_colonies,
            "overall": min(nectar_colonies, pollen_colonies),
            "total_nectar_kg": total_nectar_production / 1000,
            "total_pollen_kg": total_pollen_production / 1000,
        }

    def export_resource_map(self, day_of_year: int) -> Dict[str, Any]:
        """Export current resource distribution as map"""
        resource_map: Dict[str, Any] = {"day_of_year": day_of_year, "patches": []}

        for patch in self.landscape.patches.values():
            patch_data = {
                "id": patch.id,
                "x": patch.x,
                "y": patch.y,
                "habitat": patch.habitat_type.value,
                "nectar": patch.current_nectar,
                "pollen": patch.current_pollen,
                "quality": patch.get_resource_quality(),
                "flowering_species": len(
                    [
                        s
                        for s in patch.flower_species
                        if s.bloom_start <= day_of_year <= s.bloom_end
                    ]
                ),
            }
            resource_map["patches"].append(patch_data)

        return resource_map

    def save_species_database(self, filepath: str) -> None:
        """Save species database to file"""
        species_data = {}

        for name, species in self.species_database.items():
            species_data[name] = {
                "bloom_start": species.bloom_start,
                "bloom_end": species.bloom_end,
                "nectar_production": species.nectar_production,
                "pollen_production": species.pollen_production,
                "flower_density": species.flower_density,
                "attractiveness": species.attractiveness,
            }

        with open(filepath, "w") as f:
            json.dump(species_data, f, indent=2)

    def load_species_database_from_file(self, filepath: str) -> None:
        """Load species database from file"""
        with open(filepath, "r") as f:
            species_data = json.load(f)

        self.species_database = {}

        for name, data in species_data.items():
            self.species_database[name] = FlowerSpecies(
                name=name,
                bloom_start=data["bloom_start"],
                bloom_end=data["bloom_end"],
                nectar_production=data["nectar_production"],
                pollen_production=data["pollen_production"],
                flower_density=data["flower_density"],
                attractiveness=data["attractiveness"],
            )
