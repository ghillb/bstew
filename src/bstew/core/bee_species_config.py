"""
Configurable Bee Species System for BSTEW
=========================================

Supports both honey bees (Apis mellifera) and bumblebees (Bombus spp.)
with species-specific communication patterns, foraging behaviors, and
colony structures.
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging

# Import will be done dynamically to avoid circular imports


class BeeSpeciesType(Enum):
    """All supported bee species types"""

    # Honey bees
    APIS_MELLIFERA = "Apis_mellifera"

    # Bumblebees
    BOMBUS_TERRESTRIS = "Bombus_terrestris"
    BOMBUS_LUCORUM = "Bombus_lucorum"
    BOMBUS_LAPIDARIUS = "Bombus_lapidarius"
    BOMBUS_PRATORUM = "Bombus_pratorum"
    BOMBUS_PASCUORUM = "Bombus_pascuorum"
    BOMBUS_HORTORUM = "Bombus_hortorum"


class CommunicationType(Enum):
    """Communication system types"""

    DANCE_COMMUNICATION = "dance_communication"  # Honey bees
    SCENT_COMMUNICATION = "scent_communication"  # Bumblebees


class BeeSpeciesConfig(BaseModel):
    """Configuration for a bee species"""

    model_config = {"validate_assignment": True}

    # Species identification
    species_type: BeeSpeciesType = Field(description="Species type")
    common_name: str = Field(description="Common species name")
    scientific_name: str = Field(description="Scientific species name")

    # Communication system
    communication_type: CommunicationType = Field(
        description="Communication system type"
    )

    # Colony parameters
    max_colony_size: int = Field(ge=1, description="Maximum colony size")
    min_colony_size: int = Field(ge=1, description="Minimum colony size")
    typical_forager_count: int = Field(ge=1, description="Typical number of foragers")

    # Foraging parameters
    max_foraging_distance_m: float = Field(
        ge=0.0, description="Maximum foraging distance"
    )
    typical_foraging_distance_m: float = Field(
        ge=0.0, description="Typical foraging distance"
    )
    min_temperature_c: float = Field(description="Minimum foraging temperature")

    # Communication parameters
    uses_dance_communication: bool = Field(
        description="Uses waggle dance communication"
    )
    uses_scent_communication: bool = Field(description="Uses scent-based communication")
    social_information_sharing: float = Field(
        ge=0.0, le=1.0, description="Social information sharing level"
    )
    individual_decision_weight: float = Field(
        ge=0.0, le=1.0, description="Individual vs social decision weight"
    )

    # Memory and learning
    patch_memory_capacity: int = Field(ge=1, description="Number of patches remembered")
    learning_rate: float = Field(ge=0.0, le=1.0, description="Learning rate")

    # Physical characteristics
    proboscis_length_mm: float = Field(ge=0.0, description="Proboscis length")
    body_size_mm: float = Field(ge=0.0, description="Body size")
    flight_speed_ms: float = Field(ge=0.0, description="Flight speed m/s")


class BeeSpeciesManager:
    """Manages multiple bee species configurations"""

    def __init__(self):
        self.species_configs = self._initialize_default_species()
        self.logger = logging.getLogger(__name__)

    def _initialize_default_species(self) -> Dict[BeeSpeciesType, BeeSpeciesConfig]:
        """Initialize default species configurations"""

        species = {}

        # Honey bee (Apis mellifera)
        species[BeeSpeciesType.APIS_MELLIFERA] = BeeSpeciesConfig(
            species_type=BeeSpeciesType.APIS_MELLIFERA,
            common_name="European Honey Bee",
            scientific_name="Apis mellifera",
            communication_type=CommunicationType.DANCE_COMMUNICATION,
            max_colony_size=80000,
            min_colony_size=20000,
            typical_forager_count=15000,
            max_foraging_distance_m=6000,
            typical_foraging_distance_m=2000,
            min_temperature_c=12.0,
            uses_dance_communication=True,
            uses_scent_communication=False,
            social_information_sharing=0.9,  # High social information sharing
            individual_decision_weight=0.3,  # Low individual decision weight
            patch_memory_capacity=100,  # High memory capacity due to dance sharing
            learning_rate=0.2,
            proboscis_length_mm=6.5,
            body_size_mm=12.0,
            flight_speed_ms=6.5,
        )

        # Bumblebee - Bombus terrestris
        species[BeeSpeciesType.BOMBUS_TERRESTRIS] = BeeSpeciesConfig(
            species_type=BeeSpeciesType.BOMBUS_TERRESTRIS,
            common_name="Large Earth Bumblebee",
            scientific_name="Bombus terrestris",
            communication_type=CommunicationType.SCENT_COMMUNICATION,
            max_colony_size=400,
            min_colony_size=80,
            typical_forager_count=50,
            max_foraging_distance_m=1500,
            typical_foraging_distance_m=500,
            min_temperature_c=8.0,
            uses_dance_communication=False,
            uses_scent_communication=True,
            social_information_sharing=0.2,  # Low social information sharing
            individual_decision_weight=0.9,  # High individual decision weight
            patch_memory_capacity=12,  # Limited memory capacity
            learning_rate=0.1,
            proboscis_length_mm=7.2,
            body_size_mm=18.0,
            flight_speed_ms=4.5,
        )

        # Bumblebee - Bombus lapidarius
        species[BeeSpeciesType.BOMBUS_LAPIDARIUS] = BeeSpeciesConfig(
            species_type=BeeSpeciesType.BOMBUS_LAPIDARIUS,
            common_name="Red-tailed Bumblebee",
            scientific_name="Bombus lapidarius",
            communication_type=CommunicationType.SCENT_COMMUNICATION,
            max_colony_size=300,
            min_colony_size=60,
            typical_forager_count=40,
            max_foraging_distance_m=1200,
            typical_foraging_distance_m=400,
            min_temperature_c=8.0,
            uses_dance_communication=False,
            uses_scent_communication=True,
            social_information_sharing=0.15,
            individual_decision_weight=0.9,
            patch_memory_capacity=10,
            learning_rate=0.08,
            proboscis_length_mm=8.1,
            body_size_mm=16.0,
            flight_speed_ms=4.2,
        )

        # Bumblebee - Bombus pascuorum
        species[BeeSpeciesType.BOMBUS_PASCUORUM] = BeeSpeciesConfig(
            species_type=BeeSpeciesType.BOMBUS_PASCUORUM,
            common_name="Common Carder Bee",
            scientific_name="Bombus pascuorum",
            communication_type=CommunicationType.SCENT_COMMUNICATION,
            max_colony_size=250,
            min_colony_size=50,
            typical_forager_count=35,
            max_foraging_distance_m=1000,
            typical_foraging_distance_m=350,
            min_temperature_c=8.0,
            uses_dance_communication=False,
            uses_scent_communication=True,
            social_information_sharing=0.18,
            individual_decision_weight=0.85,
            patch_memory_capacity=8,
            learning_rate=0.12,
            proboscis_length_mm=11.2,  # Long-tongued
            body_size_mm=14.0,
            flight_speed_ms=4.0,
        )

        return species

    def get_species_config(self, species_type: BeeSpeciesType) -> BeeSpeciesConfig:
        """Get configuration for a species"""
        return self.species_configs[species_type]

    def get_communication_system(self, species_type: BeeSpeciesType):
        """Get appropriate communication system for species"""
        config = self.get_species_config(species_type)

        if config.communication_type == CommunicationType.DANCE_COMMUNICATION:
            # Dynamic import to avoid circular dependency
            from .honey_bee_communication import HoneyBeeCommunicationSystem

            return HoneyBeeCommunicationSystem()
        else:
            # Dynamic import to avoid circular dependency
            from .bumblebee_communication import BumblebeeCommunicationSystem

            return BumblebeeCommunicationSystem()

    def is_honey_bee(self, species_type: BeeSpeciesType) -> bool:
        """Check if species is a honey bee"""
        return species_type == BeeSpeciesType.APIS_MELLIFERA

    def is_bumblebee(self, species_type: BeeSpeciesType) -> bool:
        """Check if species is a bumblebee"""
        return species_type.value.startswith("Bombus_")

    def get_available_species(self) -> List[BeeSpeciesType]:
        """Get list of all available species"""
        return list(self.species_configs.keys())

    def get_honey_bee_species(self) -> List[BeeSpeciesType]:
        """Get list of honey bee species"""
        return [s for s in self.species_configs.keys() if self.is_honey_bee(s)]

    def get_bumblebee_species(self) -> List[BeeSpeciesType]:
        """Get list of bumblebee species"""
        return [s for s in self.species_configs.keys() if self.is_bumblebee(s)]

    def add_custom_species(self, config: BeeSpeciesConfig) -> None:
        """Add a custom species configuration"""
        self.species_configs[config.species_type] = config
        self.logger.info(f"Added custom species: {config.common_name}")

    def validate_species_combination(self, species_list: List[BeeSpeciesType]) -> bool:
        """Validate that species combination is realistic"""
        honey_bees = [s for s in species_list if self.is_honey_bee(s)]
        bumblebees = [s for s in species_list if self.is_bumblebee(s)]

        # Multiple honey bee species in same location is unrealistic
        if len(honey_bees) > 1:
            self.logger.warning(
                "Multiple honey bee species in same simulation is unrealistic"
            )
            return False

        # Check for unrealistic number of bumblebee species
        if len(bumblebees) > 4:
            self.logger.warning(
                f"Having {len(bumblebees)} bumblebee species in one location "
                "may be ecologically unrealistic. Consider limiting to 3-4 species "
                "that naturally coexist through resource partitioning."
            )
            # Don't return False, just warn - let researchers decide

        # Check for competing species
        if (
            BeeSpeciesType.BOMBUS_TERRESTRIS in species_list
            and BeeSpeciesType.BOMBUS_LUCORUM in species_list
        ):
            self.logger.warning("B. terrestris and B. lucorum may compete heavily")

        return True


# Global species manager instance
species_manager = BeeSpeciesManager()


def get_species_config(species_type: BeeSpeciesType) -> BeeSpeciesConfig:
    """Get species configuration"""
    return species_manager.get_species_config(species_type)


def get_communication_system(species_type: BeeSpeciesType):
    """Get communication system for species"""
    return species_manager.get_communication_system(species_type)


def create_multi_species_simulation(
    species_list: List[BeeSpeciesType],
) -> Dict[BeeSpeciesType, Dict[str, Any]]:
    """Create a multi-species simulation configuration"""

    if not species_manager.validate_species_combination(species_list):
        raise ValueError("Invalid species combination")

    simulation_config = {}

    for species_type in species_list:
        config = species_manager.get_species_config(species_type)
        comm_system = species_manager.get_communication_system(species_type)

        simulation_config[species_type] = {
            "config": config,
            "communication_system": comm_system,
            "is_honey_bee": species_manager.is_honey_bee(species_type),
            "is_bumblebee": species_manager.is_bumblebee(species_type),
        }

    return simulation_config
