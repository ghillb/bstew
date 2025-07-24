"""
NetLogo Parameter Mapping System for BSTEW
==========================================

Maps NetLogo BEE-STEWARD parameters to BSTEW configuration format.
Handles parameter conversion, unit transformation, and validation.
"""

import json
import yaml
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import logging
from enum import Enum
from datetime import datetime


class ParameterType(Enum):
    """Parameter data types"""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    LIST = "list"
    DICT = "dict"


class UnitType(Enum):
    """Unit conversion types"""

    NONE = "none"
    LENGTH = "length"
    MASS = "mass"
    TIME = "time"
    ENERGY = "energy"
    CONCENTRATION = "concentration"
    PROBABILITY = "probability"
    PERCENTAGE = "percentage"


class ParameterMapping(BaseModel):
    """Single parameter mapping definition"""

    model_config = {"validate_assignment": True}

    netlogo_name: str = Field(description="NetLogo parameter name")
    bstew_path: str = Field(description="BSTEW configuration path")
    parameter_type: ParameterType = Field(description="Parameter data type")
    unit_type: UnitType = Field(description="Unit conversion type")
    conversion_factor: float = Field(default=1.0, description="Unit conversion factor")
    default_value: Any = Field(default=None, description="Default value if missing")
    validation_min: Optional[float] = Field(
        default=None, description="Minimum valid value"
    )
    validation_max: Optional[float] = Field(
        default=None, description="Maximum valid value"
    )
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")

    def convert_value(self, netlogo_value: Any) -> Any:
        """Convert NetLogo value to BSTEW format"""
        if netlogo_value is None:
            return self.default_value

        # Extract value from NetLogoParameter if needed
        if hasattr(netlogo_value, "value"):
            netlogo_value = netlogo_value.value

        try:
            # Apply type conversion
            converted: Any
            if self.parameter_type == ParameterType.INTEGER:
                converted = int(float(netlogo_value))
            elif self.parameter_type == ParameterType.FLOAT:
                converted = float(netlogo_value)
            elif self.parameter_type == ParameterType.BOOLEAN:
                if isinstance(netlogo_value, str):
                    converted = netlogo_value.lower() in ["true", "1", "yes", "on"]
                else:
                    converted = bool(netlogo_value)
            elif self.parameter_type == ParameterType.STRING:
                converted = str(netlogo_value)
            elif self.parameter_type == ParameterType.LIST:
                if isinstance(netlogo_value, str):
                    # Parse NetLogo list format
                    converted = self._parse_netlogo_list(netlogo_value)
                else:
                    converted = list(netlogo_value)
            else:
                converted = netlogo_value

            # Apply unit conversion
            if self.unit_type != UnitType.NONE and isinstance(converted, (int, float)):
                converted = converted * self.conversion_factor

            # Validate range
            if (
                self.validation_min is not None
                and isinstance(converted, (int, float))
                and converted < self.validation_min
            ):
                logging.warning(
                    f"Parameter {self.netlogo_name} value {converted} below minimum {self.validation_min}"
                )
                converted = self.validation_min

            if (
                self.validation_max is not None
                and isinstance(converted, (int, float))
                and converted > self.validation_max
            ):
                logging.warning(
                    f"Parameter {self.netlogo_name} value {converted} above maximum {self.validation_max}"
                )
                converted = self.validation_max

            return converted

        except (ValueError, TypeError) as e:
            logging.error(f"Error converting parameter {self.netlogo_name}: {e}")
            return self.default_value

    def _parse_netlogo_list(self, value_str: str) -> List[Any]:
        """Parse NetLogo list format"""
        if not value_str:
            return []

        value_str = value_str.strip()

        # Handle triple-quoted strings
        if value_str.startswith('"""') and value_str.endswith('"""'):
            value_str = value_str[3:-3]

        # Split by spaces and parse each item
        if " " in value_str:
            items = value_str.split()
        else:
            items = [value_str]

        parsed_items: List[Any] = []
        for item in items:
            try:
                if "." in item:
                    parsed_items.append(float(item))
                else:
                    parsed_items.append(int(item))
            except ValueError:
                parsed_items.append(item.strip("\"'"))

        return parsed_items


class SpeciesMapping(BaseModel):
    """Species-specific parameter mapping"""

    model_config = {"validate_assignment": True}

    species_mappings: Dict[str, ParameterMapping] = Field(
        default_factory=dict, description="Species parameter mappings"
    )

    def add_mapping(self, mapping: ParameterMapping) -> None:
        """Add species parameter mapping"""
        self.species_mappings[mapping.netlogo_name] = mapping

    def convert_species_data(self, netlogo_species: Dict[str, Any]) -> Dict[str, Any]:
        """Convert NetLogo species data to BSTEW format"""
        bstew_species = {}

        # Handle NetLogoSpeciesData objects
        if hasattr(netlogo_species, "parameters"):
            species_params = netlogo_species.parameters
        else:
            species_params = netlogo_species

        for netlogo_name, netlogo_value in species_params.items():
            if netlogo_name in self.species_mappings:
                mapping = self.species_mappings[netlogo_name]
                converted_value = mapping.convert_value(netlogo_value)
                bstew_species[mapping.bstew_path] = converted_value
            else:
                # Pass through unmapped parameters
                bstew_species[netlogo_name] = netlogo_value

        return bstew_species


class NetLogoParameterMapper:
    """
    Main NetLogo parameter mapping system.

    Handles conversion of NetLogo parameters to BSTEW configuration format.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.parameter_mappings: Dict[str, ParameterMapping] = {}
        self.species_mappings: Dict[str, SpeciesMapping] = {}
        self.flower_mappings: Dict[str, ParameterMapping] = {}

        # Initialize default mappings
        self._initialize_system_mappings()
        self._initialize_species_mappings()
        self._initialize_flower_mappings()

    def _initialize_system_mappings(self) -> None:
        """Initialize system parameter mappings"""

        # Population parameters
        self.parameter_mappings["BeeSpeciesInitialQueensListAsString"] = (
            ParameterMapping(
                netlogo_name="BeeSpeciesInitialQueensListAsString",
                bstew_path="colony.initial_queens",
                parameter_type=ParameterType.LIST,
                unit_type=UnitType.NONE,
                default_value=["B_terrestris", 500],
                description="Initial queen species and count",
            )
        )

        # Spatial parameters
        self.parameter_mappings["MaxForagingRange_m"] = ParameterMapping(
            netlogo_name="MaxForagingRange_m",
            bstew_path="foraging.max_range_m",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.LENGTH,
            default_value=10000.0,
            validation_min=100.0,
            validation_max=50000.0,
            description="Maximum foraging range in meters",
        )

        self.parameter_mappings["MaxPatchRadius_m"] = ParameterMapping(
            netlogo_name="MaxPatchRadius_m",
            bstew_path="landscape.max_patch_radius_m",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.LENGTH,
            default_value=500.0,
            validation_min=10.0,
            validation_max=2000.0,
            description="Maximum patch radius in meters",
        )

        self.parameter_mappings["Gridsize"] = ParameterMapping(
            netlogo_name="Gridsize",
            bstew_path="landscape.grid_size",
            parameter_type=ParameterType.INTEGER,
            unit_type=UnitType.LENGTH,
            default_value=500,
            validation_min=100,
            validation_max=2000,
            description="Landscape grid size",
        )

        # Badger parameters
        self.parameter_mappings["N_Badgers"] = ParameterMapping(
            netlogo_name="N_Badgers",
            bstew_path="predation.badger_count",
            parameter_type=ParameterType.INTEGER,
            unit_type=UnitType.NONE,
            default_value=0,
            validation_min=0,
            validation_max=100,
            description="Number of badgers in simulation",
        )

        # Genetic parameters
        self.parameter_mappings["SexLocus?"] = ParameterMapping(
            netlogo_name="SexLocus?",
            bstew_path="genetics.csd_enabled",
            parameter_type=ParameterType.BOOLEAN,
            unit_type=UnitType.NONE,
            default_value=False,
            description="Enable sex locus (CSD) system",
        )

        self.parameter_mappings["UnlimitedMales?"] = ParameterMapping(
            netlogo_name="UnlimitedMales?",
            bstew_path="genetics.unlimited_males",
            parameter_type=ParameterType.BOOLEAN,
            unit_type=UnitType.NONE,
            default_value=True,
            description="Unlimited male production",
        )

        # Mortality parameters
        self.parameter_mappings["ForagingMortalityFactor"] = ParameterMapping(
            netlogo_name="ForagingMortalityFactor",
            bstew_path="mortality.foraging_factor",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.NONE,
            default_value=1.0,
            validation_min=0.0,
            validation_max=10.0,
            description="Foraging mortality multiplier",
        )

        self.parameter_mappings["WinterMortality?"] = ParameterMapping(
            netlogo_name="WinterMortality?",
            bstew_path="mortality.winter_mortality_enabled",
            parameter_type=ParameterType.BOOLEAN,
            unit_type=UnitType.NONE,
            default_value=True,
            description="Enable winter mortality",
        )

        # Food source parameters
        self.parameter_mappings["FoodSourceLimit"] = ParameterMapping(
            netlogo_name="FoodSourceLimit",
            bstew_path="resources.food_source_limit",
            parameter_type=ParameterType.INTEGER,
            unit_type=UnitType.NONE,
            default_value=25,
            validation_min=1,
            validation_max=1000,
            description="Maximum number of food sources",
        )

        self.parameter_mappings["MinSizeFoodSources?"] = ParameterMapping(
            netlogo_name="MinSizeFoodSources?",
            bstew_path="resources.min_size_enabled",
            parameter_type=ParameterType.BOOLEAN,
            unit_type=UnitType.NONE,
            default_value=True,
            description="Enable minimum food source size",
        )

        self.parameter_mappings["RemoveEmptyFoodSources?"] = ParameterMapping(
            netlogo_name="RemoveEmptyFoodSources?",
            bstew_path="resources.remove_empty_sources",
            parameter_type=ParameterType.BOOLEAN,
            unit_type=UnitType.NONE,
            default_value=True,
            description="Remove depleted food sources",
        )

        # Visualization parameters (optional)
        self.parameter_mappings["ShowQueens?"] = ParameterMapping(
            netlogo_name="ShowQueens?",
            bstew_path="visualization.show_queens",
            parameter_type=ParameterType.BOOLEAN,
            unit_type=UnitType.NONE,
            default_value=True,
            required=False,
            description="Display queens in visualization",
        )

        self.parameter_mappings["ShowPlots?"] = ParameterMapping(
            netlogo_name="ShowPlots?",
            bstew_path="visualization.show_plots",
            parameter_type=ParameterType.BOOLEAN,
            unit_type=UnitType.NONE,
            default_value=True,
            required=False,
            description="Display plots in visualization",
        )

        # Environmental parameters
        self.parameter_mappings["Weather"] = ParameterMapping(
            netlogo_name="Weather",
            bstew_path="environment.weather_type",
            parameter_type=ParameterType.STRING,
            unit_type=UnitType.NONE,
            default_value="Constant 8 hrs",
            description="Weather pattern type",
        )

        # Advanced parameters
        self.parameter_mappings["Lambda_detectProb"] = ParameterMapping(
            netlogo_name="Lambda_detectProb",
            bstew_path="detection.lambda_detect_prob",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.NONE,
            default_value=-0.005,
            validation_min=-1.0,
            validation_max=1.0,
            description="Lambda detection probability",
        )

        self.parameter_mappings["MasterSizeFactor"] = ParameterMapping(
            netlogo_name="MasterSizeFactor",
            bstew_path="landscape.master_size_factor",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.NONE,
            default_value=1.0,
            validation_min=0.1,
            validation_max=10.0,
            description="Master patch size multiplier",
        )

        self.parameter_mappings["MaxHibernatingQueens"] = ParameterMapping(
            netlogo_name="MaxHibernatingQueens",
            bstew_path="hibernation.max_hibernating_queens",
            parameter_type=ParameterType.INTEGER,
            unit_type=UnitType.NONE,
            default_value=10000,
            validation_min=0,
            validation_max=100000,
            description="Maximum hibernating queens",
        )

        # Stewardship parameters
        self.parameter_mappings["CropRotationListAsString"] = ParameterMapping(
            netlogo_name="CropRotationListAsString",
            bstew_path="stewardship.crop_rotation_plan",
            parameter_type=ParameterType.STRING,
            unit_type=UnitType.NONE,
            default_value="",
            required=False,
            description="Crop rotation plan as string format",
        )

    def _initialize_species_mappings(self) -> None:
        """Initialize species parameter mappings"""

        # Common species mapping
        species_mapping = SpeciesMapping()

        # Basic species parameters
        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="emergingDay_mean",
                bstew_path="emergence_day_mean",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.TIME,
                default_value=120.0,
                validation_min=1.0,
                validation_max=365.0,
                description="Mean emergence day",
            )
        )

        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="emergingDay_sd",
                bstew_path="emergence_day_sd",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.TIME,
                default_value=20.0,
                validation_min=0.0,
                validation_max=100.0,
                description="Emergence day standard deviation",
            )
        )

        # Morphological parameters
        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="proboscis_min_mm",
                bstew_path="proboscis_length_min",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.LENGTH,
                conversion_factor=0.001,  # mm to m
                default_value=0.007,  # 7mm
                validation_min=0.003,
                validation_max=0.025,
                description="Minimum proboscis length",
            )
        )

        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="proboscis_max_mm",
                bstew_path="proboscis_length_max",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.LENGTH,
                conversion_factor=0.001,  # mm to m
                default_value=0.012,  # 12mm
                validation_min=0.005,
                validation_max=0.030,
                description="Maximum proboscis length",
            )
        )

        # Life history parameters
        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="maxLifespanWorkers",
                bstew_path="max_lifespan_workers",
                parameter_type=ParameterType.INTEGER,
                unit_type=UnitType.TIME,
                default_value=60,
                validation_min=10,
                validation_max=200,
                description="Maximum worker lifespan in days",
            )
        )

        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="seasonStop",
                bstew_path="season_end_day",
                parameter_type=ParameterType.INTEGER,
                unit_type=UnitType.TIME,
                default_value=305,
                validation_min=200,
                validation_max=365,
                description="Season end day",
            )
        )

        # Foraging parameters
        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="flightVelocity_m/s",
                bstew_path="flight_velocity_ms",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.NONE,
                default_value=5.0,
                validation_min=1.0,
                validation_max=20.0,
                description="Flight velocity in m/s",
            )
        )

        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="searchLength_m",
                bstew_path="search_length_m",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.LENGTH,
                default_value=2500.0,
                validation_min=100.0,
                validation_max=10000.0,
                description="Search length in meters",
            )
        )

        # Development parameters
        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="devAgeHatchingMin_d",
                bstew_path="dev_age_hatching_min_d",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.TIME,
                default_value=5.0,
                validation_min=1.0,
                validation_max=20.0,
                description="Minimum hatching age in days",
            )
        )

        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="devAgePupationMin_d",
                bstew_path="dev_age_pupation_min_d",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.TIME,
                default_value=12.9,
                validation_min=5.0,
                validation_max=30.0,
                description="Minimum pupation age in days",
            )
        )

        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="devAgeEmergingMin_d",
                bstew_path="dev_age_emerging_min_d",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.TIME,
                default_value=23.8,
                validation_min=10.0,
                validation_max=50.0,
                description="Minimum emerging age in days",
            )
        )

        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="devWeightEgg_mg",
                bstew_path="dev_weight_egg_mg",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.MASS,
                default_value=1.5,
                validation_min=0.1,
                validation_max=10.0,
                description="Egg weight in mg",
            )
        )

        # Capacity parameters
        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="specMax_cropVolume_myl",
                bstew_path="max_crop_volume_mul",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.NONE,
                default_value=173.0,
                validation_min=50.0,
                validation_max=500.0,
                description="Maximum crop volume in microliters",
            )
        )

        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="specMax_pollenPellets_g",
                bstew_path="max_pollen_pellets_g",
                parameter_type=ParameterType.FLOAT,
                unit_type=UnitType.MASS,
                default_value=0.15,
                validation_min=0.05,
                validation_max=1.0,
                description="Maximum pollen pellets in grams",
            )
        )

        # Habitat parameters
        species_mapping.add_mapping(
            ParameterMapping(
                netlogo_name="nestHabitatsList",
                bstew_path="nest_habitats",
                parameter_type=ParameterType.LIST,
                unit_type=UnitType.NONE,
                default_value=["Grassland", "Garden", "Hedgerow"],
                description="List of suitable nesting habitats",
            )
        )

        # Apply to all species
        for species_id in [
            "B_terrestris",
            "B_pascuorum",
            "B_lapidarius",
            "B_hortorum",
            "B_pratorum",
            "B_hypnorum",
            "Psithyrus",
        ]:
            self.species_mappings[species_id] = species_mapping

    def _initialize_flower_mappings(self) -> None:
        """Initialize flower parameter mappings"""

        self.flower_mappings["pollen_g/flower"] = ParameterMapping(
            netlogo_name="pollen_g/flower",
            bstew_path="pollen_production",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.MASS,
            conversion_factor=1000.0,  # g to mg
            default_value=0.001,  # 1mg
            validation_min=0.0,
            validation_max=1000.0,
            description="Pollen production per flower in mg",
        )

        self.flower_mappings["nectar_ml/flower"] = ParameterMapping(
            netlogo_name="nectar_ml/flower",
            bstew_path="nectar_production",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.MASS,
            conversion_factor=1000.0,  # ml to mg (assuming density ~1)
            default_value=0.001,  # 1mg
            validation_min=0.0,
            validation_max=100.0,
            description="Nectar production per flower in mg",
        )

        self.flower_mappings["startDay"] = ParameterMapping(
            netlogo_name="startDay",
            bstew_path="bloom_start",
            parameter_type=ParameterType.INTEGER,
            unit_type=UnitType.TIME,
            default_value=120,
            validation_min=1,
            validation_max=365,
            description="Bloom start day",
        )

        self.flower_mappings["stopDay"] = ParameterMapping(
            netlogo_name="stopDay",
            bstew_path="bloom_end",
            parameter_type=ParameterType.INTEGER,
            unit_type=UnitType.TIME,
            default_value=240,
            validation_min=1,
            validation_max=365,
            description="Bloom end day",
        )

        self.flower_mappings["corollaDepth_mm"] = ParameterMapping(
            netlogo_name="corollaDepth_mm",
            bstew_path="corolla_depth",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.LENGTH,
            conversion_factor=0.001,  # mm to m
            default_value=0.005,  # 5mm
            validation_min=0.0,
            validation_max=0.030,
            description="Corolla depth in meters",
        )

        self.flower_mappings["concentration_mol/l"] = ParameterMapping(
            netlogo_name="concentration_mol/l",
            bstew_path="nectar_concentration",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.CONCENTRATION,
            default_value=1.0,
            validation_min=0.0,
            validation_max=10.0,
            description="Nectar concentration in mol/L",
        )

        self.flower_mappings["proteinPollenProp"] = ParameterMapping(
            netlogo_name="proteinPollenProp",
            bstew_path="protein_content",
            parameter_type=ParameterType.FLOAT,
            unit_type=UnitType.PERCENTAGE,
            default_value=0.2,
            validation_min=0.0,
            validation_max=1.0,
            description="Protein content of pollen (proportion)",
        )

    def convert_system_parameters(
        self, netlogo_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert NetLogo system parameters to BSTEW configuration"""

        bstew_config: Dict[str, Any] = {
            "colony": {},
            "foraging": {},
            "landscape": {},
            "predation": {},
            "genetics": {},
            "mortality": {},
            "resources": {},
            "visualization": {},
            "environment": {},
            "detection": {},
            "hibernation": {},
            "stewardship": {},
        }

        # Handle NetLogoParameter objects in dictionary
        processed_params = {}
        for netlogo_name, netlogo_value in netlogo_params.items():
            if hasattr(netlogo_value, "value"):
                processed_params[netlogo_name] = netlogo_value.value
            else:
                processed_params[netlogo_name] = netlogo_value

        for netlogo_name, netlogo_value in processed_params.items():
            if netlogo_name in self.parameter_mappings:
                mapping = self.parameter_mappings[netlogo_name]
                converted_value = mapping.convert_value(netlogo_value)

                # Set value in nested dictionary
                self._set_nested_value(
                    bstew_config, mapping.bstew_path, converted_value
                )

                self.logger.debug(
                    f"Mapped {netlogo_name} -> {mapping.bstew_path}: {converted_value}"
                )
            else:
                self.logger.warning(f"No mapping found for parameter: {netlogo_name}")

        return bstew_config

    def convert_species_parameters(
        self, netlogo_species: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert NetLogo species parameters to BSTEW format"""

        bstew_species = {}

        for species_id, species_data in netlogo_species.items():
            if species_id in self.species_mappings:
                species_mapping = self.species_mappings[species_id]
                bstew_species[species_id] = species_mapping.convert_species_data(
                    species_data
                )

                self.logger.debug(
                    f"Mapped species {species_id} with {len(bstew_species[species_id])} parameters"
                )
            else:
                self.logger.warning(f"No species mapping found for: {species_id}")
                bstew_species[species_id] = species_data

        return bstew_species

    def convert_flower_parameters(
        self, netlogo_flowers: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert NetLogo flower parameters to BSTEW format"""

        bstew_flowers = {}

        for flower_name, flower_data in netlogo_flowers.items():
            bstew_flower = {"name": flower_name}

            # Handle NetLogoFlowerData objects
            if hasattr(flower_data, "species_name"):
                # Convert NetLogoFlowerData to dictionary
                flower_dict = {
                    "pollen_g/flower": flower_data.pollen_per_flower,
                    "nectar_ml/flower": flower_data.nectar_per_flower,
                    "proteinPollenProp": flower_data.protein_pollen_prop,
                    "concentration_mol/l": flower_data.concentration_mol_per_l,
                    "startDay": flower_data.start_day,
                    "stopDay": flower_data.stop_day,
                    "corollaDepth_mm": flower_data.corolla_depth_mm,
                    "inter_flower_time_s": flower_data.inter_flower_time_s,
                }
            else:
                flower_dict = flower_data

            for netlogo_name, netlogo_value in flower_dict.items():
                if netlogo_name in self.flower_mappings:
                    mapping = self.flower_mappings[netlogo_name]
                    converted_value = mapping.convert_value(netlogo_value)
                    bstew_flower[mapping.bstew_path] = converted_value
                else:
                    # Pass through unmapped parameters
                    bstew_flower[netlogo_name] = netlogo_value

            bstew_flowers[flower_name] = bstew_flower

        self.logger.debug(f"Mapped {len(bstew_flowers)} flower species")
        return bstew_flowers

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation"""
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get summary of all parameter mappings"""

        return {
            "system_parameters": {
                "count": len(self.parameter_mappings),
                "required": sum(
                    1 for m in self.parameter_mappings.values() if m.required
                ),
                "optional": sum(
                    1 for m in self.parameter_mappings.values() if not m.required
                ),
                "by_type": {
                    ptype.value: sum(
                        1
                        for m in self.parameter_mappings.values()
                        if m.parameter_type == ptype
                    )
                    for ptype in ParameterType
                },
            },
            "species_parameters": {
                "species_count": len(self.species_mappings),
                "parameters_per_species": (
                    len(next(iter(self.species_mappings.values())).species_mappings)
                    if self.species_mappings
                    else 0
                ),
            },
            "flower_parameters": {
                "count": len(self.flower_mappings),
                "parameters": list(self.flower_mappings.keys()),
            },
        }

    def validate_conversion(
        self, netlogo_data: Dict[str, Any], bstew_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate parameter conversion"""

        validation_results: Dict[str, Any] = {
            "total_parameters": 0,
            "mapped_parameters": 0,
            "missing_parameters": [],
            "type_mismatches": [],
            "range_violations": [],
            "conversion_warnings": [],
        }

        # Check system parameters
        if "parameters" in netlogo_data:
            for param_name, param_value in netlogo_data["parameters"].items():
                validation_results["total_parameters"] += 1

                if param_name in self.parameter_mappings:
                    validation_results["mapped_parameters"] += 1

                    mapping = self.parameter_mappings[param_name]
                    converted_value = mapping.convert_value(param_value)

                    # Check if conversion changed type unexpectedly
                    if (
                        mapping.parameter_type == ParameterType.INTEGER
                        and not isinstance(converted_value, int)
                    ):
                        validation_results["type_mismatches"].append(
                            {
                                "parameter": param_name,
                                "expected_type": "integer",
                                "actual_type": type(converted_value).__name__,
                            }
                        )

                    # Check range violations
                    if (
                        mapping.validation_min is not None
                        and isinstance(converted_value, (int, float))
                        and converted_value < mapping.validation_min
                    ):
                        validation_results["range_violations"].append(
                            {
                                "parameter": param_name,
                                "value": converted_value,
                                "min_allowed": mapping.validation_min,
                            }
                        )

                else:
                    validation_results["missing_parameters"].append(param_name)

        # Avoid division by zero
        if validation_results["total_parameters"] == 0:
            validation_results["coverage"] = 0.0
        else:
            validation_results["coverage"] = (
                validation_results["mapped_parameters"]
                / validation_results["total_parameters"]
                * 100
            )

        return validation_results

    def export_mapping_config(self, output_file: str) -> None:
        """Export mapping configuration to file"""

        mapping_config = {
            "system_parameters": {
                name: {
                    "netlogo_name": mapping.netlogo_name,
                    "bstew_path": mapping.bstew_path,
                    "parameter_type": mapping.parameter_type.value,
                    "unit_type": mapping.unit_type.value,
                    "conversion_factor": mapping.conversion_factor,
                    "default_value": mapping.default_value,
                    "validation_min": mapping.validation_min,
                    "validation_max": mapping.validation_max,
                    "description": mapping.description,
                    "required": mapping.required,
                }
                for name, mapping in self.parameter_mappings.items()
            },
            "flower_parameters": {
                name: {
                    "netlogo_name": mapping.netlogo_name,
                    "bstew_path": mapping.bstew_path,
                    "parameter_type": mapping.parameter_type.value,
                    "unit_type": mapping.unit_type.value,
                    "conversion_factor": mapping.conversion_factor,
                    "default_value": mapping.default_value,
                    "description": mapping.description,
                }
                for name, mapping in self.flower_mappings.items()
            },
        }

        output_path = Path(output_file)

        if output_path.suffix.lower() == ".json":
            with open(output_path, "w") as f:
                json.dump(mapping_config, f, indent=2)
        elif output_path.suffix.lower() in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.dump(mapping_config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        self.logger.info(f"Exported mapping configuration to {output_path}")


def convert_netlogo_to_bstew(
    netlogo_data: Dict[str, Any], output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to convert NetLogo data to BSTEW format.

    Args:
        netlogo_data: Parsed NetLogo data dictionary
        output_file: Optional output file for BSTEW configuration

    Returns:
        BSTEW configuration dictionary
    """

    mapper = NetLogoParameterMapper()

    # Convert system parameters
    bstew_config = {}
    if "parameters" in netlogo_data:
        system_config = mapper.convert_system_parameters(netlogo_data["parameters"])
        bstew_config.update(system_config)

    # Convert species parameters
    if "species" in netlogo_data:
        species_config = mapper.convert_species_parameters(netlogo_data["species"])
        bstew_config["species"] = species_config

    # Convert flower parameters
    if "flowers" in netlogo_data:
        flower_config = mapper.convert_flower_parameters(netlogo_data["flowers"])
        bstew_config["flowers"] = flower_config

    # Add metadata
    bstew_config["metadata"] = {
        "converted_from": "NetLogo BEE-STEWARD",
        "conversion_timestamp": datetime.now().isoformat(),
        "total_parameters": len(netlogo_data.get("parameters", {})),
        "total_species": len(netlogo_data.get("species", {})),
        "total_flowers": len(netlogo_data.get("flowers", {})),
    }

    # Export if requested
    if output_file:
        output_path = Path(output_file)

        if output_path.suffix.lower() == ".json":
            with open(output_path, "w") as f:
                json.dump(bstew_config, f, indent=2, default=str)
        elif output_path.suffix.lower() in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.dump(bstew_config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        logging.getLogger(__name__).info(
            f"BSTEW configuration exported to {output_path}"
        )

    return bstew_config
