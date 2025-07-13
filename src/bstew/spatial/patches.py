"""
Patch system for BSTEW spatial modeling
=======================================

Individual patches with resource and habitat information.
Maps to NetLogo patches with enhanced functionality.
"""

from typing import Dict, List, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field
import math


class HabitatType(Enum):
    """Enumeration of habitat types"""

    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WOODLAND = "woodland"
    HEDGEROW = "hedgerow"
    WILDFLOWER = "wildflower"
    URBAN = "urban"
    WATER = "water"
    BARE_SOIL = "bare_soil"
    ROAD = "road"
    BUILDING = "building"


class FlowerSpecies(BaseModel):
    """Individual flower species information"""

    model_config = {"validate_assignment": True}

    name: str = Field(description="Flower species name")
    bloom_start: int = Field(ge=1, le=365, description="Bloom start day of year")
    bloom_end: int = Field(ge=1, le=365, description="Bloom end day of year")
    nectar_production: float = Field(
        ge=0.0, description="Nectar production mg per flower per day"
    )
    pollen_production: float = Field(
        ge=0.0, description="Pollen production mg per flower per day"
    )
    flower_density: float = Field(ge=0.0, description="Flowers per square meter")
    attractiveness: float = Field(
        ge=0.0, le=1.0, description="Attractiveness to bees (0-1 scale)"
    )
    corolla_depth_mm: float = Field(
        default=5.0, ge=0.0, description="Corolla depth in millimeters"
    )
    corolla_width_mm: float = Field(
        default=3.0, ge=0.0, description="Corolla width in millimeters"
    )
    nectar_accessibility: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Base accessibility without proboscis constraint",
    )


class ResourceQuality(BaseModel):
    """Resource quality assessment for a patch"""

    model_config = {"validate_assignment": True}

    nectar_quality: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Nectar quality (0-1 scale)"
    )
    pollen_quality: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Pollen quality (0-1 scale)"
    )
    accessibility: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Resource accessibility (0-1 scale)"
    )
    competition: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Competition level (0-1 scale)"
    )
    weather_impact: float = Field(
        default=1.0, ge=0.0, description="Weather impact factor"
    )

    def overall_quality(self) -> float:
        """Calculate overall resource quality"""
        return (
            (self.nectar_quality + self.pollen_quality)
            / 2.0
            * self.accessibility
            * (1.0 - self.competition)
            * self.weather_impact
        )


class ResourcePatch:
    """
    Individual patch with resource and habitat information.

    Represents a spatial cell in the landscape grid with:
    - Habitat type and characteristics
    - Resource production and availability
    - Temporal dynamics (seasonal, weather)
    - Visitation tracking
    """

    def __init__(self, patch_id: int, x: float, y: float, habitat_type: HabitatType):
        self.id = patch_id
        self.x = x
        self.y = y
        self.location = (x, y)
        self.habitat_type = habitat_type

        # Resource production
        self.base_nectar_production = 0.0  # mg per day
        self.base_pollen_production = 0.0  # mg per day
        self.current_nectar = 0.0
        self.current_pollen = 0.0
        self.max_nectar_capacity = 1000.0
        self.max_pollen_capacity = 500.0

        # Flower species composition
        self.flower_species: List[FlowerSpecies] = []
        self.flower_density = 0.0  # Total flowers per m²

        # Temporal tracking
        self.last_updated = 0
        self.visited_recently = False
        self.visit_count = 0
        self.last_visit_time = 0
        self.forager_pressure = 0.0

        # Environmental factors
        self.elevation = 0.0
        self.slope = 0.0
        self.aspect = 0.0  # degrees from north
        self.soil_quality = 1.0
        self.water_availability = 1.0

        # Derived properties
        self.primary_resource_type = "nectar"
        self.quality = ResourceQuality()

        # Initialize based on habitat type
        self.initialize_habitat_properties()

    def initialize_habitat_properties(self) -> None:
        """Initialize patch properties based on habitat type"""
        if self.habitat_type == HabitatType.WILDFLOWER:
            self.base_nectar_production = 50.0
            self.base_pollen_production = 25.0
            self.flower_density = 100.0
            self.add_wildflower_species()

        elif self.habitat_type == HabitatType.HEDGEROW:
            self.base_nectar_production = 30.0
            self.base_pollen_production = 20.0
            self.flower_density = 50.0
            self.add_hedgerow_species()

        elif self.habitat_type == HabitatType.CROPLAND:
            self.base_nectar_production = 20.0
            self.base_pollen_production = 15.0
            self.flower_density = 30.0
            self.add_crop_species()

        elif self.habitat_type == HabitatType.GRASSLAND:
            self.base_nectar_production = 15.0
            self.base_pollen_production = 10.0
            self.flower_density = 25.0
            self.add_grassland_species()

        elif self.habitat_type == HabitatType.WOODLAND:
            self.base_nectar_production = 10.0
            self.base_pollen_production = 5.0
            self.flower_density = 10.0
            self.add_woodland_species()

        else:
            # Non-flowering habitats
            self.base_nectar_production = 0.0
            self.base_pollen_production = 0.0
            self.flower_density = 0.0

    def add_wildflower_species(self) -> None:
        """Add typical wildflower species"""
        species = [
            FlowerSpecies(
                name="Clover",
                bloom_start=120,
                bloom_end=240,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=20.0,
                attractiveness=0.8,
            ),
            FlowerSpecies(
                name="Dandelion",
                bloom_start=80,
                bloom_end=300,
                nectar_production=3.0,
                pollen_production=1.5,
                flower_density=15.0,
                attractiveness=0.7,
            ),
            FlowerSpecies(
                name="Bramble",
                bloom_start=150,
                bloom_end=210,
                nectar_production=8.0,
                pollen_production=3.0,
                flower_density=10.0,
                attractiveness=0.9,
            ),
            FlowerSpecies(
                name="Wild Carrot",
                bloom_start=180,
                bloom_end=270,
                nectar_production=4.0,
                pollen_production=2.5,
                flower_density=8.0,
                attractiveness=0.6,
            ),
        ]
        self.flower_species.extend(species)

    def add_hedgerow_species(self) -> None:
        """Add typical hedgerow species"""
        species = [
            FlowerSpecies(
                name="Hawthorn",
                bloom_start=120,
                bloom_end=150,
                nectar_production=6.0,
                pollen_production=4.0,
                flower_density=5.0,
                attractiveness=0.8,
            ),
            FlowerSpecies(
                name="Blackthorn",
                bloom_start=90,
                bloom_end=120,
                nectar_production=4.0,
                pollen_production=3.0,
                flower_density=8.0,
                attractiveness=0.7,
            ),
            FlowerSpecies(
                name="Elder",
                bloom_start=150,
                bloom_end=180,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=3.0,
                attractiveness=0.6,
            ),
        ]
        self.flower_species.extend(species)

    def add_crop_species(self) -> None:
        """Add typical crop species"""
        species = [
            FlowerSpecies(
                name="Oilseed Rape",
                bloom_start=100,
                bloom_end=140,
                nectar_production=15.0,
                pollen_production=8.0,
                flower_density=100.0,
                attractiveness=0.9,
            ),
            FlowerSpecies(
                name="Sunflower",
                bloom_start=200,
                bloom_end=250,
                nectar_production=20.0,
                pollen_production=12.0,
                flower_density=1.0,
                attractiveness=0.95,
            ),
            FlowerSpecies(
                name="Field Bean",
                bloom_start=140,
                bloom_end=180,
                nectar_production=3.0,
                pollen_production=8.0,
                flower_density=50.0,
                attractiveness=0.7,
            ),
        ]
        self.flower_species.extend(species)

    def add_grassland_species(self) -> None:
        """Add typical grassland species"""
        species = [
            FlowerSpecies(
                name="Buttercup",
                bloom_start=120,
                bloom_end=180,
                nectar_production=2.0,
                pollen_production=1.0,
                flower_density=10.0,
                attractiveness=0.5,
            ),
            FlowerSpecies(
                name="Plantain",
                bloom_start=100,
                bloom_end=250,
                nectar_production=1.0,
                pollen_production=2.0,
                flower_density=15.0,
                attractiveness=0.4,
            ),
            FlowerSpecies(
                name="Daisy",
                bloom_start=80,
                bloom_end=300,
                nectar_production=1.5,
                pollen_production=0.5,
                flower_density=20.0,
                attractiveness=0.3,
            ),
        ]
        self.flower_species.extend(species)

    def add_woodland_species(self) -> None:
        """Add typical woodland species"""
        species = [
            FlowerSpecies(
                name="Bluebell",
                bloom_start=110,
                bloom_end=140,
                nectar_production=2.0,
                pollen_production=1.0,
                flower_density=5.0,
                attractiveness=0.6,
            ),
            FlowerSpecies(
                name="Wood Anemone",
                bloom_start=80,
                bloom_end=120,
                nectar_production=1.0,
                pollen_production=0.5,
                flower_density=8.0,
                attractiveness=0.4,
            ),
            FlowerSpecies(
                name="Wild Garlic",
                bloom_start=100,
                bloom_end=130,
                nectar_production=1.5,
                pollen_production=1.0,
                flower_density=12.0,
                attractiveness=0.3,
            ),
        ]
        self.flower_species.extend(species)

    def update_resources(self, day_of_year: int, weather: Dict[str, float]) -> None:
        """Update resource production based on season and weather"""
        if day_of_year == self.last_updated:
            return

        self.last_updated = day_of_year

        # Calculate total production from all flowering species
        total_nectar = 0.0
        total_pollen = 0.0

        for species in self.flower_species:
            if species.bloom_start <= day_of_year <= species.bloom_end:
                # Species is flowering
                bloom_intensity = self.calculate_bloom_intensity(species, day_of_year)
                weather_factor = self.calculate_weather_factor(weather)

                nectar_contribution = (
                    species.nectar_production
                    * species.flower_density
                    * bloom_intensity
                    * weather_factor
                )

                pollen_contribution = (
                    species.pollen_production
                    * species.flower_density
                    * bloom_intensity
                    * weather_factor
                )

                total_nectar += nectar_contribution
                total_pollen += pollen_contribution

        # Update current resources (accumulate with decay)
        decay_rate = 0.1  # Daily decay rate
        self.current_nectar = min(
            self.max_nectar_capacity,
            self.current_nectar * (1 - decay_rate) + total_nectar,
        )
        self.current_pollen = min(
            self.max_pollen_capacity,
            self.current_pollen * (1 - decay_rate) + total_pollen,
        )

        # Update resource quality
        self.update_resource_quality(weather)

        # Reset recent visitation
        self.visited_recently = False
        self.forager_pressure *= 0.9  # Decay pressure

    def calculate_bloom_intensity(
        self, species: FlowerSpecies, day_of_year: int
    ) -> float:
        """Calculate bloom intensity for a species on given day"""
        bloom_duration = species.bloom_end - species.bloom_start
        if bloom_duration <= 0:
            return 0.0

        # Peak bloom at middle of flowering period
        bloom_center = (species.bloom_start + species.bloom_end) / 2
        distance_from_peak = abs(day_of_year - bloom_center)

        # Gaussian-like bloom curve
        intensity = math.exp(-((distance_from_peak / (bloom_duration / 4)) ** 2))
        return intensity

    def calculate_weather_factor(self, weather: Dict[str, float]) -> float:
        """Calculate weather impact on resource production"""
        temperature = weather.get("temperature", 15.0)  # °C
        rainfall = weather.get("rainfall", 0.0)  # mm
        wind_speed = weather.get("wind_speed", 5.0)  # mph
        humidity = weather.get("humidity", 60.0)  # %

        # Temperature factor (optimal around 15-25°C)
        if 15 <= temperature <= 25:
            temp_factor = 1.0
        elif temperature < 10 or temperature > 30:
            temp_factor = 0.3
        else:
            temp_factor = 0.7

        # Rainfall factor (moderate rain helps, heavy rain hinders)
        if rainfall < 1:
            rain_factor = 0.8  # Drought stress
        elif rainfall < 5:
            rain_factor = 1.2  # Beneficial
        elif rainfall < 15:
            rain_factor = 1.0  # Normal
        else:
            rain_factor = 0.5  # Heavy rain

        # Wind factor (high wind reduces nectar availability)
        wind_factor = max(0.2, 1.0 - (wind_speed / 30.0))

        # Humidity factor (moderate humidity optimal)
        if 40 <= humidity <= 70:
            humidity_factor = 1.0
        else:
            humidity_factor = 0.8

        return temp_factor * rain_factor * wind_factor * humidity_factor

    def update_resource_quality(self, weather: Dict[str, float]) -> None:
        """Update resource quality assessment"""
        # Base quality from resource availability
        nectar_ratio = self.current_nectar / max(1.0, self.max_nectar_capacity)
        pollen_ratio = self.current_pollen / max(1.0, self.max_pollen_capacity)

        self.quality.nectar_quality = nectar_ratio
        self.quality.pollen_quality = pollen_ratio

        # Weather impact
        self.quality.weather_impact = self.calculate_weather_factor(weather)

        # Competition from forager pressure
        self.quality.competition = min(0.8, self.forager_pressure / 10.0)

        # Accessibility (could be modified by distance, terrain, etc.)
        self.quality.accessibility = 1.0

    def has_resources(self) -> bool:
        """Check if patch has available resources"""
        return self.current_nectar > 0.1 or self.current_pollen > 0.1

    def get_available_resources(self) -> float:
        """Get total available resources"""
        return self.current_nectar + self.current_pollen

    def get_resource_quality(self) -> float:
        """Get overall resource quality (0-1)"""
        return self.quality.overall_quality()

    def consume_resources(self, amount: float, resource_type: str = "nectar") -> float:
        """Consume resources from patch"""
        if resource_type == "nectar":
            consumed = min(amount, self.current_nectar)
            self.current_nectar -= consumed
        elif resource_type == "pollen":
            consumed = min(amount, self.current_pollen)
            self.current_pollen -= consumed
        else:
            # Mixed consumption
            nectar_consumed = min(amount / 2, self.current_nectar)
            pollen_consumed = min(amount / 2, self.current_pollen)
            self.current_nectar -= nectar_consumed
            self.current_pollen -= pollen_consumed
            consumed = nectar_consumed + pollen_consumed

        # Track visitation
        self.visit_count += 1
        self.visited_recently = True
        self.forager_pressure += 1.0

        return consumed

    def get_neighbors(
        self, landscape_grid: Any, radius: float
    ) -> List["ResourcePatch"]:
        """Get neighboring patches within radius"""
        neighbors = []

        for patch in landscape_grid.get_patches_in_radius(self.location, radius):
            if patch.id != self.id:
                neighbors.append(patch)

        return neighbors

    def calculate_distance_to(self, target_pos: Tuple[float, float]) -> float:
        """Calculate distance to target position"""
        dx = target_pos[0] - self.x
        dy = target_pos[1] - self.y
        return math.sqrt(dx**2 + dy**2)

    def get_seasonal_attractiveness(self, day_of_year: int) -> float:
        """Get seasonal attractiveness based on flowering species"""
        total_attractiveness = 0.0
        flowering_count = 0

        for species in self.flower_species:
            if species.bloom_start <= day_of_year <= species.bloom_end:
                bloom_intensity = self.calculate_bloom_intensity(species, day_of_year)
                total_attractiveness += species.attractiveness * bloom_intensity
                flowering_count += 1

        if flowering_count > 0:
            return total_attractiveness / flowering_count
        else:
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert patch to dictionary for serialization"""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "habitat_type": self.habitat_type.value,
            "current_nectar": self.current_nectar,
            "current_pollen": self.current_pollen,
            "visit_count": self.visit_count,
            "quality": self.quality.overall_quality(),
            "flower_species_count": len(self.flower_species),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourcePatch":
        """Create patch from dictionary"""
        patch = cls(data["id"], data["x"], data["y"], HabitatType(data["habitat_type"]))
        patch.current_nectar = data.get("current_nectar", 0.0)
        patch.current_pollen = data.get("current_pollen", 0.0)
        patch.visit_count = data.get("visit_count", 0)
        return patch
