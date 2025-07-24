"""
Species Configuration Loader for BSTEW
======================================

Loads and validates species configurations from YAML/JSON files
and integrates them with the multi-species simulation system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError
import logging

from ..core.bee_species_config import (
    BeeSpeciesType,
    BeeSpeciesConfig,
    BeeSpeciesManager,
    create_multi_species_simulation,
)
from ..core.bee_communication import create_multi_species_communication


class SimulationEnvironmentConfig(BaseModel):
    """Environment configuration for simulation"""

    landscape_size: List[float] = Field(
        description="Landscape dimensions [width, height] in meters"
    )
    temperature_range: List[float] = Field(
        description="Temperature range [min, max] in Celsius"
    )
    flower_patch_count: int = Field(ge=1, description="Number of flower patches")
    resource_regeneration_rate: float = Field(
        ge=0.0, le=1.0, description="Resource regeneration rate"
    )


class SpeciesInstanceConfig(BaseModel):
    """Configuration for a species instance in simulation"""

    type: str = Field(description="Species type (e.g., APIS_MELLIFERA)")
    colonies: int = Field(ge=1, description="Number of colonies")
    configuration: Dict[str, Any] = Field(description="Species-specific configuration")


class SimulationAnalysisConfig(BaseModel):
    """Analysis configuration"""

    metrics: List[str] = Field(description="Metrics to collect")
    output_frequency: str = Field(description="Output frequency")
    save_individual_tracks: bool = False
    save_communication_events: bool = True


class SimulationOutputConfig(BaseModel):
    """Output configuration"""

    directory: str = Field(description="Output directory path")
    formats: List[str] = Field(description="Output formats")
    include_visualizations: bool = False


class SimulationConfig(BaseModel):
    """Complete simulation configuration"""

    name: str = Field(description="Simulation name")
    description: Optional[str] = None
    duration_days: int = Field(ge=1, description="Simulation duration in days")
    time_step_minutes: int = Field(ge=1, description="Time step in minutes")

    environment: SimulationEnvironmentConfig
    species: List[SpeciesInstanceConfig]
    analysis: SimulationAnalysisConfig
    output: SimulationOutputConfig


class SpeciesConfigLoader:
    """Loads and validates species configurations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.species_manager = BeeSpeciesManager()

    def load_from_yaml(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file"""

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        return self._process_config(raw_config, config_path)

    def load_from_json(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file"""

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            raw_config = json.load(f)

        return self._process_config(raw_config, config_path)

    def _process_config(
        self, raw_config: Dict[str, Any], config_path: Path
    ) -> Dict[str, Any]:
        """Process and validate configuration"""

        try:
            # Validate main simulation configuration
            simulation_config = SimulationConfig(**raw_config["simulation"])

            # Extract species types
            species_types = []
            for species_instance in simulation_config.species:
                try:
                    species_type = BeeSpeciesType[species_instance.type]
                    species_types.append(species_type)
                except KeyError:
                    raise ValueError(f"Unknown species type: {species_instance.type}")

            # Validate species combination
            if not self.species_manager.validate_species_combination(species_types):
                self.logger.warning("Species combination may have ecological issues")

            # Apply any species overrides
            species_overrides = raw_config.get("species_overrides", {})
            self._apply_species_overrides(species_overrides)

            # Create multi-species configuration
            multi_species_config = create_multi_species_simulation(species_types)

            # Create communication systems
            communication_systems = create_multi_species_communication(species_types)

            return {
                "simulation_config": simulation_config,
                "species_types": species_types,
                "multi_species_config": multi_species_config,
                "communication_systems": communication_systems,
                "config_path": str(config_path),
            }

        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to process configuration: {e}")

    def _apply_species_overrides(self, overrides: Dict[str, Dict[str, Any]]) -> None:
        """Apply custom species parameter overrides"""

        for species_name, override_params in overrides.items():
            try:
                species_type = BeeSpeciesType[species_name]
                current_config = self.species_manager.get_species_config(species_type)

                # Create modified configuration
                modified_params = current_config.model_dump()
                modified_params.update(override_params)

                # Validate and update
                modified_config = BeeSpeciesConfig(**modified_params)
                self.species_manager.add_custom_species(modified_config)

                self.logger.info(
                    f"Applied overrides to {species_name}: {list(override_params.keys())}"
                )

            except KeyError:
                self.logger.warning(f"Unknown species in overrides: {species_name}")
            except Exception as e:
                self.logger.error(f"Failed to apply overrides to {species_name}: {e}")

    def create_simulation_from_config(
        self, config_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Create complete simulation setup from configuration file"""

        # Determine file format and load
        config_path = Path(config_path)
        if (
            config_path.suffix.lower() == ".yaml"
            or config_path.suffix.lower() == ".yml"
        ):
            config = self.load_from_yaml(config_path)
        elif config_path.suffix.lower() == ".json":
            config = self.load_from_json(config_path)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")

        # Add simulation setup helpers
        config["setup_helpers"] = self._create_setup_helpers(config)

        return config

    def _create_setup_helpers(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create helper functions for simulation setup"""

        simulation_config = config["simulation_config"]
        species_types = config["species_types"]

        helpers = {
            "get_total_colonies": lambda: sum(
                s.colonies for s in simulation_config.species
            ),
            "get_species_count": lambda: len(species_types),
            "get_honey_bee_colonies": lambda: sum(
                s.colonies
                for s in simulation_config.species
                if BeeSpeciesType[s.type] == BeeSpeciesType.APIS_MELLIFERA
            ),
            "get_bumblebee_colonies": lambda: sum(
                s.colonies
                for s in simulation_config.species
                if BeeSpeciesType[s.type].value.startswith("Bombus_")
            ),
            "has_multi_species": lambda: len(species_types) > 1,
            "has_honey_bees": lambda: BeeSpeciesType.APIS_MELLIFERA in species_types,
            "has_bumblebees": lambda: any(
                species_type.value.startswith("Bombus_")
                for species_type in species_types
            ),
        }

        return helpers

    def validate_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate configuration file and return validation report"""

        try:
            config = self.create_simulation_from_config(config_path)

            report = {
                "valid": True,
                "config_path": str(config_path),
                "simulation_name": config["simulation_config"].name,
                "species_count": len(config["species_types"]),
                "total_colonies": config["setup_helpers"]["get_total_colonies"](),
                "duration_days": config["simulation_config"].duration_days,
                "has_multi_species": config["setup_helpers"]["has_multi_species"](),
                "species_breakdown": {
                    "honey_bees": config["setup_helpers"]["get_honey_bee_colonies"](),
                    "bumblebees": config["setup_helpers"]["get_bumblebee_colonies"](),
                },
                "warnings": [],
                "errors": [],
            }

            # Check for potential issues
            if report["total_colonies"] > 20:
                report["warnings"].append(
                    "Large number of colonies may impact performance"
                )

            if report["species_breakdown"]["honey_bees"] > 1:
                report["warnings"].append(
                    "Multiple honey bee colonies in same area is unrealistic"
                )

            return report

        except Exception as e:
            return {"valid": False, "config_path": str(config_path), "errors": [str(e)]}


def load_simulation_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to load simulation configuration"""
    loader = SpeciesConfigLoader()
    return loader.create_simulation_from_config(config_path)


def validate_simulation_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to validate simulation configuration"""
    loader = SpeciesConfigLoader()
    return loader.validate_config_file(config_path)
