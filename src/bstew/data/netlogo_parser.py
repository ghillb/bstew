"""
NetLogo Data Parser for BSTEW
=============================

Parses NetLogo BEE-STEWARD data files and converts them to BSTEW-compatible formats.
Handles NetLogo-specific formats, parameter files, and data structures.
"""

import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import logging

# from ..spatial.patches import FlowerSpecies, HabitatType
# from ..components.species_system import BumblebeeSpecies


class NetLogoParameter(BaseModel):
    """NetLogo parameter with metadata"""

    model_config = {"validate_assignment": True}

    name: str = Field(description="Parameter name")
    value: Any = Field(description="Parameter value")
    data_type: str = Field(description="Data type of parameter")
    description: str = Field(default="", description="Parameter description")
    unit: str = Field(default="", description="Parameter unit")
    source_file: str = Field(default="", description="Source file path")


class NetLogoSpeciesData(BaseModel):
    """NetLogo species data structure"""

    model_config = {"validate_assignment": True}

    species_id: str = Field(description="Species identifier")
    name: str = Field(description="Species name")
    parameters: Dict[str, Any] = Field(description="Species parameters")
    source_file: str = Field(description="Source file path")


class NetLogoFlowerData(BaseModel):
    """NetLogo flower species data structure"""

    model_config = {"validate_assignment": True}

    species_name: str = Field(description="Flower species name")
    pollen_per_flower: float = Field(ge=0.0, description="Pollen per flower in grams")
    nectar_per_flower: float = Field(ge=0.0, description="Nectar per flower in ml")
    protein_pollen_prop: float = Field(
        ge=0.0, le=1.0, description="Protein proportion in pollen"
    )
    concentration_mol_per_l: float = Field(
        ge=0.0, description="Nectar concentration mol/l"
    )
    start_day: int = Field(
        ge=0, le=365, description="Bloom start day (0 = not applicable)"
    )
    stop_day: int = Field(
        ge=0, le=365, description="Bloom end day (0 = not applicable)"
    )
    corolla_depth_mm: float = Field(ge=0.0, description="Corolla depth in mm")
    inter_flower_time_s: float = Field(
        ge=0.0, description="Inter-flower time in seconds"
    )
    source_file: str = Field(description="Source file path")


class NetLogoFoodSource(BaseModel):
    """NetLogo food source location data"""

    model_config = {"validate_assignment": True}

    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    patch_id: str = Field(description="Patch identifier")
    area_m2: float = Field(ge=0.0, description="Area in square meters")
    habitat_type: str = Field(description="Habitat type")
    source_file: str = Field(description="Source file path")


class NetLogoStringParser:
    """
    Parses NetLogo-specific string formats and data structures.

    Handles:
    - NetLogo string arrays: ["item1" "item2"]
    - NetLogo quoted strings: triple-quoted strings
    - NetLogo boolean values: TRUE/FALSE
    - NetLogo lists and arrays
    """

    @staticmethod
    def parse_netlogo_string(value: str) -> Any:
        """Parse NetLogo string format to Python value"""
        if pd.isna(value) or value == "":
            return None

        value = str(value).strip()

        # Handle NetLogo triple-quoted strings
        if value.startswith('"""') and value.endswith('"""'):
            return value[3:-3]

        # Handle NetLogo boolean values
        if value.upper() == "TRUE":
            return True
        elif value.upper() == "FALSE":
            return False

        # Handle NetLogo string arrays like ["item1" "item2"]
        if value.startswith("[") and value.endswith("]"):
            return NetLogoStringParser._parse_netlogo_array(value)

        # Handle regular quoted strings
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]

        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    @staticmethod
    def _parse_netlogo_array(array_str: str) -> List[str]:
        """Parse NetLogo array format like ["item1" "item2"]"""
        # Remove outer brackets
        inner = array_str[1:-1].strip()

        if not inner:
            return []

        # Split by quotes and extract items
        items = []
        current_item = ""
        in_quotes = False

        i = 0
        while i < len(inner):
            char = inner[i]

            if char == '"':
                if in_quotes:
                    # End of quoted string
                    items.append(current_item)
                    current_item = ""
                    in_quotes = False
                else:
                    # Start of quoted string
                    in_quotes = True
            elif in_quotes:
                current_item += char
            # Skip whitespace between items

            i += 1

        return items

    @staticmethod
    def parse_netlogo_list_string(value: str) -> List[Any]:
        """Parse NetLogo list strings like 'B_terrestris 500'"""
        if pd.isna(value) or value == "":
            return []

        value = str(value).strip()

        # Handle triple-quoted strings
        if value.startswith('"""') and value.endswith('"""'):
            value = value[3:-3]

        # Split by spaces and parse each item
        items: List[Any] = []
        parts = value.split()

        for part in parts:
            # Try to parse as number
            try:
                if "." in part:
                    items.append(float(part))
                else:
                    items.append(int(part))
            except ValueError:
                items.append(part)

        return items


class NetLogoParameterParser:
    """
    Parses NetLogo parameter files like _SYSTEM_Parameters.csv.

    Handles the specific format used in NetLogo BEE-STEWARD model.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.string_parser = NetLogoStringParser()

    def parse_parameters_file(self, filepath: str) -> Dict[str, NetLogoParameter]:
        """Parse NetLogo parameters CSV file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"NetLogo parameters file not found: {filepath}")

        try:
            # Read the CSV file
            df = pd.read_csv(filepath)

            if len(df) < 1:
                raise ValueError("Parameters file must have at least 1 row of data")

            # First row is headers, second row is values (or first data row)
            headers = df.columns.tolist()
            if len(df) > 1:
                values = df.iloc[1].tolist()  # Use second row if available
            else:
                values = df.iloc[0].tolist()  # Use first row if only one row

            parameters = {}

            for header, value in zip(headers, values):
                # Clean up header name
                clean_header = header.strip()

                # Parse value using NetLogo string parser
                parsed_value = self.string_parser.parse_netlogo_string(value)

                # Determine data type
                data_type = self._determine_data_type(parsed_value)

                parameters[clean_header] = NetLogoParameter(
                    name=clean_header,
                    value=parsed_value,
                    data_type=data_type,
                    source_file=str(file_path),
                )

            self.logger.info(f"Parsed {len(parameters)} parameters from {filepath}")
            return parameters

        except Exception as e:
            raise ValueError(f"Error parsing NetLogo parameters file {filepath}: {e}")

    def _determine_data_type(self, value: Any) -> str:
        """Determine data type of parsed value"""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, str):
            return "string"
        else:
            return "unknown"


class NetLogoSpeciesParser:
    """
    Parses NetLogo species files like _SYSTEM_BumbleSpecies_UK_01.csv.

    Handles species-specific parameters and data structures.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.string_parser = NetLogoStringParser()

    def parse_species_file(self, filepath: str) -> Dict[str, NetLogoSpeciesData]:
        """Parse NetLogo species CSV file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"NetLogo species file not found: {filepath}")

        try:
            df = pd.read_csv(filepath)

            species_data = {}

            for _, row in df.iterrows():
                species_id = str(row["species"])
                species_name = str(row["name"])

                # Parse all parameters
                parameters = {}
                for col in df.columns:
                    if col not in ["species", "name"]:
                        raw_value = row[col]
                        parsed_value = self.string_parser.parse_netlogo_string(
                            raw_value
                        )
                        parameters[col] = parsed_value

                species_data[species_id] = NetLogoSpeciesData(
                    species_id=species_id,
                    name=species_name,
                    parameters=parameters,
                    source_file=str(file_path),
                )

            self.logger.info(f"Parsed {len(species_data)} species from {filepath}")
            return species_data

        except Exception as e:
            raise ValueError(f"Error parsing NetLogo species file {filepath}: {e}")

    def convert_to_bstew_species(
        self, netlogo_species: Dict[str, NetLogoSpeciesData]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert NetLogo species data to BSTEW-compatible format"""
        bstew_species = {}

        for species_id, species_data in netlogo_species.items():
            params = species_data.parameters

            # Extract habitat list
            habitat_list = params.get("nestHabitatsList", [])
            if isinstance(habitat_list, str):
                habitat_list = self.string_parser._parse_netlogo_array(habitat_list)

            # Map NetLogo parameters to BSTEW parameters
            bstew_params = {
                "species_id": species_id,
                "common_name": species_data.name,
                "scientific_name": species_data.name,
                "emergence_day_mean": params.get("emergingDay_mean", 120),
                "emergence_day_sd": params.get("emergingDay_sd", 20),
                "nest_habitats": habitat_list,
                "proboscis_length_min": params.get("proboscis_min_mm", 7.0),
                "proboscis_length_max": params.get("proboscis_max_mm", 12.0),
                "growth_factor": params.get("growthFactor", 1.88),
                "season_end_day": params.get("seasonStop", 305),
                "max_lifespan_workers": params.get("maxLifespanWorkers", 60),
                "batch_size": params.get("batchsize", 10),
                "flight_velocity_ms": params.get("flightVelocity_m/s", 5.0),
                "search_length_m": params.get("searchLength_m", 2500),
                "time_unloading_s": params.get("timeUnloading", 165),
                "max_crop_volume_mul": params.get("specMax_cropVolume_myl", 173),
                "max_pollen_pellets_g": params.get("specMax_pollenPellets_g", 0.15),
                "min_to_max_factor": params.get("minToMaxFactor", 2.0),
                "dev_age_hatching_min_d": params.get("devAgeHatchingMin_d", 5),
                "dev_age_pupation_min_d": params.get("devAgePupationMin_d", 12.9),
                "dev_age_emerging_min_d": params.get("devAgeEmergingMin_d", 23.8),
                "dev_weight_egg_mg": params.get("devWeightEgg_mg", 1.5),
                "dev_weight_pupation_min_mg": params.get(
                    "devWeightPupationMin_mg", 50.0
                ),
                "dev_weight_pupation_max_mg": params.get(
                    "devWeightPupationMax_mg", 200.0
                ),
                "pollen_to_biomass_factor": params.get("pollenToBodymassFactor", 1.0),
                "daily_nest_site_chance": params.get("dailyNestSiteChance", 0.2),
            }

            bstew_species[species_id] = bstew_params

        return bstew_species


class NetLogoFlowerParser:
    """
    Parses NetLogo flower species files like _SYSTEM_Flowerspecies.csv.

    Handles flower-specific parameters and phenology data.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.string_parser = NetLogoStringParser()

    def parse_flower_file(self, filepath: str) -> Dict[str, NetLogoFlowerData]:
        """Parse NetLogo flower species CSV file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"NetLogo flower file not found: {filepath}")

        try:
            df = pd.read_csv(filepath)

            flower_data = {}

            for _, row in df.iterrows():
                species_name = str(row["Flowerspecies"])

                # Clean up species name (remove quotes)
                if species_name.startswith('"""') and species_name.endswith('"""'):
                    species_name = species_name[3:-3]

                flower_data[species_name] = NetLogoFlowerData(
                    species_name=species_name,
                    pollen_per_flower=float(row["pollen_g/flower"]),
                    nectar_per_flower=float(row["nectar_ml/flower"]),
                    protein_pollen_prop=float(row["proteinPollenProp"]),
                    concentration_mol_per_l=float(row["concentration_mol/l"]),
                    start_day=int(row["startDay"]),
                    stop_day=int(row["stopDay"]),
                    corolla_depth_mm=float(row["corollaDepth_mm"]),
                    inter_flower_time_s=float(row["intFlowerTime_s"]),
                    source_file=str(file_path),
                )

            self.logger.info(
                f"Parsed {len(flower_data)} flower species from {filepath}"
            )
            return flower_data

        except Exception as e:
            raise ValueError(f"Error parsing NetLogo flower file {filepath}: {e}")

    def convert_to_bstew_flowers(
        self, netlogo_flowers: Dict[str, NetLogoFlowerData]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert NetLogo flower data to BSTEW-compatible format"""
        bstew_flowers = {}

        for species_name, flower_data in netlogo_flowers.items():
            # Convert NetLogo parameters to BSTEW parameters
            bstew_flowers[species_name] = {
                "name": species_name,
                "bloom_start": flower_data.start_day,
                "bloom_end": flower_data.stop_day,
                "nectar_production": flower_data.nectar_per_flower
                * 1000,  # Convert ml to mg
                "pollen_production": flower_data.pollen_per_flower
                * 1000,  # Convert g to mg
                "flower_density": 100.0,  # Default density per m²
                "attractiveness": 0.8,  # Default attractiveness
                "nectar_concentration": flower_data.concentration_mol_per_l,
                "corolla_depth": flower_data.corolla_depth_mm,
                "nectar_accessibility": 1.0,  # Will be calculated based on corolla depth
                "pollen_accessibility": 1.0,
                "protein_content": flower_data.protein_pollen_prop,
                "inter_flower_time": flower_data.inter_flower_time_s,
            }

        return bstew_flowers


class NetLogoFoodSourceParser:
    """
    Parses NetLogo food source files like _SYSTEM_Example_Farm_Foodsources.txt.

    Handles spatial food source location data.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def parse_food_source_file(self, filepath: str) -> List[NetLogoFoodSource]:
        """Parse NetLogo food source location file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"NetLogo food source file not found: {filepath}")

        try:
            food_sources = []

            with open(filepath, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    if not line or line.startswith("#"):
                        continue

                    # Parse line format: x y patch_id area habitat_type
                    parts = line.split()

                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            patch_id = (
                                parts[2] if len(parts) > 2 else f"patch_{line_num}"
                            )
                            area_m2 = float(parts[3]) if len(parts) > 3 else 100.0
                            habitat_type = parts[4] if len(parts) > 4 else "grassland"

                            food_sources.append(
                                NetLogoFoodSource(
                                    x=x,
                                    y=y,
                                    patch_id=patch_id,
                                    area_m2=area_m2,
                                    habitat_type=habitat_type,
                                    source_file=str(file_path),
                                )
                            )

                        except ValueError as e:
                            self.logger.warning(
                                f"Error parsing line {line_num} in {filepath}: {e}"
                            )
                            continue

            self.logger.info(f"Parsed {len(food_sources)} food sources from {filepath}")
            return food_sources

        except Exception as e:
            raise ValueError(f"Error parsing NetLogo food source file {filepath}: {e}")


class NetLogoHabitatParser:
    """
    Parses NetLogo habitat files like _SYSTEM_Habitats.csv.

    Handles habitat-species mapping matrices.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def parse_habitat_file(self, filepath: str) -> Dict[str, Dict[str, float]]:
        """Parse NetLogo habitat-species mapping file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"NetLogo habitat file not found: {filepath}")

        try:
            df = pd.read_csv(filepath)

            habitat_mapping = {}

            # First column should be habitat names
            if len(df.columns) == 0:
                raise ValueError("Habitat file has no columns")

            habitat_col = df.columns[0]

            for _, row in df.iterrows():
                habitat_name = str(row[habitat_col])

                # Get species abundances for this habitat
                species_abundances = {}
                for col in df.columns[1:]:
                    species_name = str(col)
                    abundance = float(row[col]) if pd.notna(row[col]) else 0.0
                    species_abundances[species_name] = abundance

                habitat_mapping[habitat_name] = species_abundances

            self.logger.info(f"Parsed {len(habitat_mapping)} habitats from {filepath}")
            return habitat_mapping

        except Exception as e:
            raise ValueError(f"Error parsing NetLogo habitat file {filepath}: {e}")


class NetLogoDataParser:
    """
    Main NetLogo data parser that coordinates all sub-parsers.

    Provides a unified interface for parsing all NetLogo BEE-STEWARD data files.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Initialize sub-parsers
        self.parameter_parser = NetLogoParameterParser()
        self.species_parser = NetLogoSpeciesParser()
        self.flower_parser = NetLogoFlowerParser()
        self.food_source_parser = NetLogoFoodSourceParser()
        self.habitat_parser = NetLogoHabitatParser()

    def parse_all_data_files(self, data_dir: str) -> Dict[str, Any]:
        """Parse all NetLogo data files in directory"""
        data_path = Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        results: Dict[str, Any] = {
            "parameters": {},
            "species": {},
            "flowers": {},
            "food_sources": {},
            "habitats": {},
            "landscapes": {},
            "metadata": {
                "source_directory": str(data_path),
                "parser_version": "1.0.0",
                "files_processed": [],
            },
        }

        try:
            # Parse parameter files
            param_files = list(data_path.glob("**/*Parameters.csv"))
            for param_file in param_files:
                try:
                    params = self.parameter_parser.parse_parameters_file(
                        str(param_file)
                    )
                    results["parameters"][param_file.stem] = params
                    results["metadata"]["files_processed"].append(str(param_file))
                except Exception as e:
                    self.logger.error(f"Error parsing parameter file {param_file}: {e}")

            # Parse species files
            species_files = list(data_path.glob("**/*BumbleSpecies*.csv"))
            for species_file in species_files:
                try:
                    species = self.species_parser.parse_species_file(str(species_file))
                    results["species"][species_file.stem] = species
                    results["metadata"]["files_processed"].append(str(species_file))
                except Exception as e:
                    self.logger.error(f"Error parsing species file {species_file}: {e}")

            # Parse flower files
            flower_files = list(data_path.glob("**/*Flowerspecies*.csv"))
            for flower_file in flower_files:
                try:
                    flowers = self.flower_parser.parse_flower_file(str(flower_file))
                    results["flowers"][flower_file.stem] = flowers
                    results["metadata"]["files_processed"].append(str(flower_file))
                except Exception as e:
                    self.logger.error(f"Error parsing flower file {flower_file}: {e}")

            # Parse food source files
            foodsource_files = list(data_path.glob("**/*Foodsources*.txt"))
            for foodsource_file in foodsource_files:
                try:
                    food_sources = self.food_source_parser.parse_food_source_file(
                        str(foodsource_file)
                    )
                    results["food_sources"][foodsource_file.stem] = food_sources
                    results["metadata"]["files_processed"].append(str(foodsource_file))
                except Exception as e:
                    self.logger.error(
                        f"Error parsing food source file {foodsource_file}: {e}"
                    )

            # Parse habitat files
            habitat_files = list(data_path.glob("**/*Habitats*.csv"))
            for habitat_file in habitat_files:
                try:
                    habitats = self.habitat_parser.parse_habitat_file(str(habitat_file))
                    results["habitats"][habitat_file.stem] = habitats
                    results["metadata"]["files_processed"].append(str(habitat_file))
                except Exception as e:
                    self.logger.error(f"Error parsing habitat file {habitat_file}: {e}")

            # Parse landscape images
            landscape_files = list(data_path.glob("**/*/SYSTEM_Example*.png"))
            for landscape_file in landscape_files:
                results["landscapes"][landscape_file.stem] = str(landscape_file)
                results["metadata"]["files_processed"].append(str(landscape_file))

            self.logger.info(
                f"Parsed {len(results['metadata']['files_processed'])} files from {data_dir}"
            )

        except Exception as e:
            self.logger.error(f"Error parsing data directory {data_dir}: {e}")
            raise

        return results

    def export_parsed_data(self, parsed_data: Dict[str, Any], output_dir: str) -> str:
        """Export parsed data to JSON format"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert complex objects to serializable format
        serializable_data = self._make_serializable(parsed_data)

        output_file = output_path / "netlogo_parsed_data.json"

        try:
            with open(output_file, "w") as f:
                json.dump(serializable_data, f, indent=2, default=str)

            self.logger.info(f"Exported parsed data to {output_file}")
            return str(output_file)

        except Exception as e:
            raise ValueError(f"Error exporting parsed data: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert complex objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


def parse_netlogo_data(
    data_dir: str, output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to parse NetLogo data directory.

    Args:
        data_dir: Directory containing NetLogo data files
        output_dir: Optional directory to export parsed data

    Returns:
        Dictionary containing all parsed data
    """
    parser = NetLogoDataParser()

    # Parse all data files
    parsed_data = parser.parse_all_data_files(data_dir)

    # Export if output directory specified
    if output_dir:
        parser.export_parsed_data(parsed_data, output_dir)

    return parsed_data
