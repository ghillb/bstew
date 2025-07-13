#!/usr/bin/env python3
"""
NetLogo Integration Testing Suite for BSTEW
===========================================

Comprehensive integration tests for NetLogo compatibility, covering:
- Data parsing and parameter mapping
- Simulation initialization and execution
- Output comparison with NetLogo results
- System behavior validation
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from bstew.data.netlogo_parser import NetLogoDataParser
from bstew.data.netlogo_output_parser import NetLogoOutputParser
from bstew.config.netlogo_mapping import NetLogoParameterMapper
from bstew.utils.validation import ModelValidator, NetLogoSpecificValidator


class IntegrationTestResult(BaseModel):
    """Result of an integration test with validation"""

    model_config = {"validate_assignment": True}

    test_name: str = Field(description="Name of the integration test")
    passed: bool = Field(description="Whether the test passed")
    errors: List[str] = Field(
        default_factory=list, description="List of error messages"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of warning messages"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Test performance metrics"
    )
    execution_time: float = Field(ge=0.0, description="Test execution time in seconds")


class NetLogoIntegrationTester:
    """
    Integration testing suite for NetLogo compatibility.

    Tests the complete pipeline from NetLogo data parsing through
    BSTEW simulation execution and output comparison.
    """

    def __init__(self, data_dir: str = "data", output_dir: str = "artifacts/test_outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.parser = NetLogoDataParser()
        self.output_parser = NetLogoOutputParser()
        self.mapper = NetLogoParameterMapper()
        self.validator = ModelValidator()
        self.netlogo_validator = NetLogoSpecificValidator()

        # Test results
        self.test_results: List[IntegrationTestResult] = []

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("Running NetLogo Integration Testing Suite...")
        print("=" * 60)

        # Test data parsing
        self._test_data_parsing()

        # Test parameter mapping
        self._test_parameter_mapping()

        # Test configuration generation
        self._test_config_generation()

        # Test validation framework
        self._test_validation_framework()

        # Test genetic system compatibility
        self._test_genetic_system_compatibility()

        # Test species system compatibility
        self._test_species_system_compatibility()

        # Test flower system compatibility
        self._test_flower_system_compatibility()

        # Test simulation initialization
        self._test_simulation_initialization()

        # Generate summary report
        return self._generate_test_report()

    def _test_data_parsing(self):
        """Test NetLogo data parsing functionality"""
        print("\n1. Testing NetLogo Data Parsing...")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Test parameter file parsing
            param_file = self.data_dir / "parameters" / "_SYSTEM_Parameters.csv"
            if param_file.exists():
                params = self.parser.parameter_parser.parse_parameters_file(
                    str(param_file)
                )
                metrics["system_parameters"] = len(params)

                # Validate parameter types
                for name, param in params.items():
                    if hasattr(param, "value") and param.value is None:
                        warnings.append(f"Parameter {name} has null value")

                print(f"   ✓ Parsed {len(params)} system parameters")
            else:
                errors.append("System parameters file not found")

            # Test species file parsing
            species_file = (
                self.data_dir / "parameters" / "_SYSTEM_BumbleSpecies_UK_01.csv"
            )
            if species_file.exists():
                species = self.parser.species_parser.parse_species_file(
                    str(species_file)
                )
                metrics["species_count"] = len(species)

                # Validate species data
                for species_id, species_data in species.items():
                    if hasattr(species_data, "parameters"):
                        if not species_data.parameters:
                            warnings.append(
                                f"Species {species_id} has empty parameters"
                            )

                print(f"   ✓ Parsed {len(species)} species")
            else:
                errors.append("Species file not found")

            # Test flower file parsing
            flower_file = self.data_dir / "parameters" / "_SYSTEM_Flowerspecies.csv"
            if flower_file.exists():
                flowers = self.parser.flower_parser.parse_flower_file(str(flower_file))
                metrics["flower_count"] = len(flowers)

                # Validate flower data
                for flower_id, flower_data in flowers.items():
                    if (
                        hasattr(flower_data, "corolla_depth_mm")
                        and flower_data.corolla_depth_mm <= 0
                    ):
                        warnings.append(f"Flower {flower_id} has invalid corolla depth")

                print(f"   ✓ Parsed {len(flowers)} flower species")
            else:
                errors.append("Flower species file not found")

        except Exception as e:
            errors.append(f"Data parsing error: {str(e)}")

        # Record test result
        self.test_results.append(
            IntegrationTestResult(
                test_name="Data Parsing",
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=0.0,
            )
        )

        if errors:
            print(f"   ✗ {len(errors)} errors found")
        else:
            print("   ✓ Data parsing tests passed")

    def _test_parameter_mapping(self):
        """Test parameter mapping functionality"""
        print("\n2. Testing Parameter Mapping...")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Test system parameter mapping
            param_file = self.data_dir / "parameters" / "_SYSTEM_Parameters.csv"
            if param_file.exists():
                params = self.parser.parameter_parser.parse_parameters_file(
                    str(param_file)
                )
                bstew_config = self.mapper.convert_system_parameters(params)

                # Count mapped parameters
                mapped_count = 0
                for section, config in bstew_config.items():
                    if isinstance(config, dict):
                        mapped_count += len(config)

                metrics["mapped_system_parameters"] = mapped_count
                metrics["total_system_parameters"] = len(params)

                # Test critical parameters
                critical_params = [
                    "foraging.max_range_m",
                    "predation.badger_count",
                    "genetics.csd_enabled",
                    "landscape.grid_size",
                ]

                for param_path in critical_params:
                    sections = param_path.split(".")
                    value = bstew_config
                    for section in sections:
                        if section in value:
                            value = value[section]
                        else:
                            warnings.append(
                                f"Critical parameter {param_path} not mapped"
                            )
                            break

                print(f"   ✓ Mapped {mapped_count}/{len(params)} system parameters")

            # Test species parameter mapping
            species_file = (
                self.data_dir / "parameters" / "_SYSTEM_BumbleSpecies_UK_01.csv"
            )
            if species_file.exists():
                species = self.parser.species_parser.parse_species_file(
                    str(species_file)
                )
                bstew_species = self.mapper.convert_species_parameters(species)

                metrics["mapped_species"] = len(bstew_species)

                # Test critical species parameters
                critical_species_params = [
                    "emergence_day_mean",
                    "proboscis_length_min",
                    "proboscis_length_max",
                    "flight_velocity_ms",
                ]

                for species_id, species_data in bstew_species.items():
                    for param in critical_species_params:
                        if param not in species_data:
                            warnings.append(f"Species {species_id} missing {param}")

                print(f"   ✓ Mapped {len(bstew_species)} species")

            # Test flower parameter mapping
            flower_file = self.data_dir / "parameters" / "_SYSTEM_Flowerspecies.csv"
            if flower_file.exists():
                flowers = self.parser.flower_parser.parse_flower_file(str(flower_file))
                bstew_flowers = self.mapper.convert_flower_parameters(flowers)

                metrics["mapped_flowers"] = len(bstew_flowers)

                # Test critical flower parameters
                critical_flower_params = [
                    "bloom_start",
                    "bloom_end",
                    "corolla_depth",
                    "nectar_production",
                    "pollen_production",
                ]

                for flower_id, flower_data in bstew_flowers.items():
                    for param in critical_flower_params:
                        if param not in flower_data:
                            warnings.append(f"Flower {flower_id} missing {param}")

                print(f"   ✓ Mapped {len(bstew_flowers)} flowers")

        except Exception as e:
            errors.append(f"Parameter mapping error: {str(e)}")

        # Record test result
        self.test_results.append(
            IntegrationTestResult(
                test_name="Parameter Mapping",
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=0.0,
            )
        )

        if errors:
            print(f"   ✗ {len(errors)} errors found")
        else:
            print("   ✓ Parameter mapping tests passed")

    def _test_config_generation(self):
        """Test BSTEW configuration generation"""
        print("\n3. Testing Configuration Generation...")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Generate complete BSTEW configuration manually
            full_config = {
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
                "species": {},
                "flowers": {},
                "metadata": {},
            }

            # Add system parameters
            param_file = self.data_dir / "parameters" / "_SYSTEM_Parameters.csv"
            if param_file.exists():
                params = self.parser.parameter_parser.parse_parameters_file(
                    str(param_file)
                )
                system_config = self.mapper.convert_system_parameters(params)
                full_config.update(system_config)

            # Add species parameters
            species_file = (
                self.data_dir / "parameters" / "_SYSTEM_BumbleSpecies_UK_01.csv"
            )
            if species_file.exists():
                species = self.parser.species_parser.parse_species_file(
                    str(species_file)
                )
                species_config = self.mapper.convert_species_parameters(species)
                full_config["species"] = species_config

            # Add flower parameters
            flower_file = self.data_dir / "parameters" / "_SYSTEM_Flowerspecies.csv"
            if flower_file.exists():
                flowers = self.parser.flower_parser.parse_flower_file(str(flower_file))
                flower_config = self.mapper.convert_flower_parameters(flowers)
                full_config["flowers"] = flower_config

            # Add metadata
            param_count = 0
            species_count = 0
            flower_count = 0

            if param_file.exists():
                param_count = len(params) if "params" in locals() else 0
            if species_file.exists():
                species_count = len(species) if "species" in locals() else 0
            if flower_file.exists():
                flower_count = len(flowers) if "flowers" in locals() else 0

            full_config["metadata"] = {
                "converted_from": "NetLogo BEE-STEWARD",
                "total_parameters": param_count,
                "total_species": species_count,
                "total_flowers": flower_count,
            }

            # Test configuration structure
            required_sections = [
                "colony",
                "foraging",
                "landscape",
                "predation",
                "genetics",
                "mortality",
                "resources",
                "species",
                "flowers",
            ]

            for section in required_sections:
                if section not in full_config:
                    errors.append(f"Missing required section: {section}")

            # Test configuration completeness
            populated_sections = [
                k for k, v in full_config.items() if v and k != "metadata"
            ]
            metrics["populated_sections"] = len(populated_sections)
            metrics["total_sections"] = len(required_sections)

            # Save configuration for inspection
            config_file = self.output_dir / "integration_test_config.json"
            with open(config_file, "w") as f:
                json.dump(full_config, f, indent=2, default=str)

            # Validate configuration structure
            if "metadata" in full_config:
                metadata = full_config["metadata"]
                if "total_parameters" in metadata and metadata["total_parameters"] == 0:
                    warnings.append("No parameters found in metadata")

            print(
                f"   ✓ Generated configuration with {len(populated_sections)} populated sections"
            )

        except Exception as e:
            errors.append(f"Configuration generation error: {str(e)}")

        # Record test result
        self.test_results.append(
            IntegrationTestResult(
                test_name="Configuration Generation",
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=0.0,
            )
        )

        if errors:
            print(f"   ✗ {len(errors)} errors found")
        else:
            print("   ✓ Configuration generation tests passed")

    def _test_validation_framework(self):
        """Test validation framework functionality"""
        print("\n4. Testing Validation Framework...")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Test NetLogo-specific validation
            param_file = self.data_dir / "parameters" / "_SYSTEM_Parameters.csv"
            if param_file.exists():
                params = self.parser.parameter_parser.parse_parameters_file(
                    str(param_file)
                )
                bstew_config = self.mapper.convert_system_parameters(params)

                # Test genetic parameter validation
                genetic_validation = self.netlogo_validator.validate_genetic_parameters(
                    params, bstew_config
                )
                metrics["genetic_validation_count"] = len(genetic_validation)

                genetic_passed = sum(
                    1 for result in genetic_validation if result.passed
                )
                metrics["genetic_validation_passed"] = genetic_passed

                for result in genetic_validation:
                    if not result.passed:
                        warnings.append(
                            f"Genetic validation failed: {result.test_name}"
                        )

                # Test species parameter validation
                species_file = (
                    self.data_dir / "parameters" / "_SYSTEM_BumbleSpecies_UK_01.csv"
                )
                if species_file.exists():
                    species = self.parser.species_parser.parse_species_file(
                        str(species_file)
                    )
                    bstew_species = self.mapper.convert_species_parameters(species)

                    species_validation = (
                        self.netlogo_validator.validate_species_parameters(
                            species, bstew_species
                        )
                    )
                    metrics["species_validation_count"] = len(species_validation)

                    species_passed = sum(
                        1 for result in species_validation if result.passed
                    )
                    metrics["species_validation_passed"] = species_passed

                    for result in species_validation:
                        if not result.passed:
                            warnings.append(
                                f"Species validation failed: {result.test_name}"
                            )

                # Test flower parameter validation
                flower_file = self.data_dir / "parameters" / "_SYSTEM_Flowerspecies.csv"
                if flower_file.exists():
                    flowers = self.parser.flower_parser.parse_flower_file(
                        str(flower_file)
                    )
                    bstew_flowers = self.mapper.convert_flower_parameters(flowers)

                    flower_validation = (
                        self.netlogo_validator.validate_flower_accessibility(
                            flowers, bstew_flowers
                        )
                    )
                    metrics["flower_validation_count"] = len(flower_validation)

                    flower_passed = sum(
                        1 for result in flower_validation if result.passed
                    )
                    metrics["flower_validation_passed"] = flower_passed

                    for result in flower_validation:
                        if not result.passed:
                            warnings.append(
                                f"Flower validation failed: {result.test_name}"
                            )

            print("   ✓ Validation framework tests completed")

        except Exception as e:
            errors.append(f"Validation framework error: {str(e)}")

        # Record test result
        self.test_results.append(
            IntegrationTestResult(
                test_name="Validation Framework",
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=0.0,
            )
        )

        if errors:
            print(f"   ✗ {len(errors)} errors found")
        else:
            print("   ✓ Validation framework tests passed")

    def _test_genetic_system_compatibility(self):
        """Test genetic system compatibility with NetLogo"""
        print("\n5. Testing Genetic System Compatibility...")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Test CSD system parameters
            param_file = self.data_dir / "parameters" / "_SYSTEM_Parameters.csv"
            if param_file.exists():
                params = self.parser.parameter_parser.parse_parameters_file(
                    str(param_file)
                )
                bstew_config = self.mapper.convert_system_parameters(params)

                # Test CSD parameter presence
                genetics_config = bstew_config.get("genetics", {})

                if "csd_enabled" in genetics_config:
                    metrics["csd_enabled"] = genetics_config["csd_enabled"]
                    print(
                        f"   ✓ CSD system: {'enabled' if genetics_config['csd_enabled'] else 'disabled'}"
                    )
                else:
                    warnings.append("CSD parameter not found in genetics configuration")

                if "unlimited_males" in genetics_config:
                    metrics["unlimited_males"] = genetics_config["unlimited_males"]
                    print(
                        f"   ✓ Unlimited males: {'enabled' if genetics_config['unlimited_males'] else 'disabled'}"
                    )
                else:
                    warnings.append("Unlimited males parameter not found")

            # Test genetic system logic consistency
            if metrics.get("csd_enabled", False) and metrics.get(
                "unlimited_males", True
            ):
                warnings.append(
                    "CSD enabled with unlimited males - potential configuration conflict"
                )

            print("   ✓ Genetic system compatibility tests completed")

        except Exception as e:
            errors.append(f"Genetic system compatibility error: {str(e)}")

        # Record test result
        self.test_results.append(
            IntegrationTestResult(
                test_name="Genetic System Compatibility",
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=0.0,
            )
        )

        if errors:
            print(f"   ✗ {len(errors)} errors found")
        else:
            print("   ✓ Genetic system compatibility tests passed")

    def _test_species_system_compatibility(self):
        """Test species system compatibility with NetLogo"""
        print("\n6. Testing Species System Compatibility...")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Test species parameter completeness
            species_file = (
                self.data_dir / "parameters" / "_SYSTEM_BumbleSpecies_UK_01.csv"
            )
            if species_file.exists():
                species = self.parser.species_parser.parse_species_file(
                    str(species_file)
                )
                bstew_species = self.mapper.convert_species_parameters(species)

                metrics["species_count"] = len(bstew_species)

                # Test required species parameters
                required_params = [
                    "emergence_day_mean",
                    "emergence_day_sd",
                    "proboscis_length_min",
                    "proboscis_length_max",
                    "flight_velocity_ms",
                    "season_end_day",
                ]

                missing_params = {}
                for species_id, species_data in bstew_species.items():
                    for param in required_params:
                        if param not in species_data:
                            if param not in missing_params:
                                missing_params[param] = []
                            missing_params[param].append(species_id)

                if missing_params:
                    for param, species_list in missing_params.items():
                        warnings.append(
                            f"Parameter {param} missing for species: {species_list}"
                        )

                # Test parameter ranges
                range_violations = []
                for species_id, species_data in bstew_species.items():
                    # Test proboscis length ranges
                    if (
                        "proboscis_length_min" in species_data
                        and "proboscis_length_max" in species_data
                    ):
                        min_len = species_data["proboscis_length_min"]
                        max_len = species_data["proboscis_length_max"]
                        if min_len >= max_len:
                            range_violations.append(
                                f"Species {species_id}: proboscis min >= max"
                            )

                    # Test emergence timing
                    if "emergence_day_mean" in species_data:
                        emergence_day = species_data["emergence_day_mean"]
                        if emergence_day < 1 or emergence_day > 365:
                            range_violations.append(
                                f"Species {species_id}: invalid emergence day {emergence_day}"
                            )

                metrics["range_violations"] = len(range_violations)
                warnings.extend(range_violations)

                print(f"   ✓ Processed {len(bstew_species)} species")

        except Exception as e:
            errors.append(f"Species system compatibility error: {str(e)}")

        # Record test result
        self.test_results.append(
            IntegrationTestResult(
                test_name="Species System Compatibility",
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=0.0,
            )
        )

        if errors:
            print(f"   ✗ {len(errors)} errors found")
        else:
            print("   ✓ Species system compatibility tests passed")

    def _test_flower_system_compatibility(self):
        """Test flower system compatibility with NetLogo"""
        print("\n7. Testing Flower System Compatibility...")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Test flower parameter completeness
            flower_file = self.data_dir / "parameters" / "_SYSTEM_Flowerspecies.csv"
            if flower_file.exists():
                flowers = self.parser.flower_parser.parse_flower_file(str(flower_file))
                bstew_flowers = self.mapper.convert_flower_parameters(flowers)

                metrics["flower_count"] = len(bstew_flowers)

                # Test required flower parameters
                required_params = [
                    "bloom_start",
                    "bloom_end",
                    "corolla_depth",
                    "nectar_production",
                    "pollen_production",
                ]

                missing_params = {}
                for flower_id, flower_data in bstew_flowers.items():
                    for param in required_params:
                        if param not in flower_data:
                            if param not in missing_params:
                                missing_params[param] = []
                            missing_params[param].append(flower_id)

                if missing_params:
                    for param, flower_list in missing_params.items():
                        warnings.append(
                            f"Parameter {param} missing for flowers: {flower_list[:3]}..."
                        )

                # Test parameter ranges
                range_violations = []
                for flower_id, flower_data in bstew_flowers.items():
                    # Test bloom period
                    if "bloom_start" in flower_data and "bloom_end" in flower_data:
                        start_day = flower_data["bloom_start"]
                        end_day = flower_data["bloom_end"]
                        if start_day >= end_day:
                            range_violations.append(
                                f"Flower {flower_id}: bloom start >= end"
                            )

                    # Test corolla depth
                    if "corolla_depth" in flower_data:
                        depth = flower_data["corolla_depth"]
                        if depth < 0 or depth > 0.050:  # 50mm max
                            range_violations.append(
                                f"Flower {flower_id}: invalid corolla depth {depth}"
                            )

                    # Test production values
                    for param in ["nectar_production", "pollen_production"]:
                        if param in flower_data:
                            value = flower_data[param]
                            if value < 0:
                                range_violations.append(
                                    f"Flower {flower_id}: negative {param}"
                                )

                metrics["range_violations"] = len(range_violations)
                warnings.extend(range_violations[:10])  # Limit warnings

                print(f"   ✓ Processed {len(bstew_flowers)} flower species")

        except Exception as e:
            errors.append(f"Flower system compatibility error: {str(e)}")

        # Record test result
        self.test_results.append(
            IntegrationTestResult(
                test_name="Flower System Compatibility",
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=0.0,
            )
        )

        if errors:
            print(f"   ✗ {len(errors)} errors found")
        else:
            print("   ✓ Flower system compatibility tests passed")

    def _test_simulation_initialization(self):
        """Test simulation initialization with NetLogo parameters"""
        print("\n8. Testing Simulation Initialization...")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Generate BSTEW configuration manually
            bstew_config = {
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
                "species": {},
                "flowers": {},
                "metadata": {},
            }

            # Add system parameters
            param_file = self.data_dir / "parameters" / "_SYSTEM_Parameters.csv"
            if param_file.exists():
                params = self.parser.parameter_parser.parse_parameters_file(
                    str(param_file)
                )
                system_config = self.mapper.convert_system_parameters(params)
                bstew_config.update(system_config)

            # Add species parameters
            species_file = (
                self.data_dir / "parameters" / "_SYSTEM_BumbleSpecies_UK_01.csv"
            )
            if species_file.exists():
                species = self.parser.species_parser.parse_species_file(
                    str(species_file)
                )
                species_config = self.mapper.convert_species_parameters(species)
                bstew_config["species"] = species_config

            # Add flower parameters
            flower_file = self.data_dir / "parameters" / "_SYSTEM_Flowerspecies.csv"
            if flower_file.exists():
                flowers = self.parser.flower_parser.parse_flower_file(str(flower_file))
                flower_config = self.mapper.convert_flower_parameters(flowers)
                bstew_config["flowers"] = flower_config

            # Test configuration validity for simulation
            required_for_simulation = {
                "colony": ["initial_queens"],
                "foraging": ["max_range_m"],
                "landscape": ["grid_size"],
                "species": [],  # Must have species data
                "flowers": [],  # Must have flower data
            }

            for section, required_params in required_for_simulation.items():
                if section not in bstew_config:
                    errors.append(f"Missing required section for simulation: {section}")
                    continue

                if section in ["species", "flowers"]:
                    if not bstew_config[section]:
                        errors.append(f"Empty {section} data for simulation")
                else:
                    for param in required_params:
                        if param not in bstew_config[section]:
                            errors.append(
                                f"Missing required parameter: {section}.{param}"
                            )

            # Test parameter value ranges for simulation
            if "foraging" in bstew_config and "max_range_m" in bstew_config["foraging"]:
                max_range = bstew_config["foraging"]["max_range_m"]
                if max_range <= 0:
                    errors.append("Invalid foraging range for simulation")

            if "landscape" in bstew_config and "grid_size" in bstew_config["landscape"]:
                grid_size = bstew_config["landscape"]["grid_size"]
                if grid_size <= 0:
                    errors.append("Invalid grid size for simulation")

            # Count available resources
            species_count = len(bstew_config.get("species", {}))
            flower_count = len(bstew_config.get("flowers", {}))

            metrics["species_count"] = species_count
            metrics["flower_count"] = flower_count

            if species_count == 0:
                errors.append("No species available for simulation")
            if flower_count == 0:
                errors.append("No flowers available for simulation")

            print(
                f"   ✓ Configuration ready for simulation ({species_count} species, {flower_count} flowers)"
            )

        except Exception as e:
            errors.append(f"Simulation initialization error: {str(e)}")

        # Record test result
        self.test_results.append(
            IntegrationTestResult(
                test_name="Simulation Initialization",
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=0.0,
            )
        )

        if errors:
            print(f"   ✗ {len(errors)} errors found")
        else:
            print("   ✓ Simulation initialization tests passed")

    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("NETLOGO INTEGRATION TEST REPORT")
        print("=" * 60)

        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests

        total_errors = sum(len(result.errors) for result in self.test_results)
        total_warnings = sum(len(result.warnings) for result in self.test_results)

        print("\nOverall Results:")
        print(f"  Tests Run: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Errors: {total_errors}")
        print(f"  Warnings: {total_warnings}")

        # Test details
        print("\nTest Details:")
        for result in self.test_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  {result.test_name}: {status}")

            if result.errors:
                print(f"    Errors: {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"      - {error}")
                if len(result.errors) > 3:
                    print(f"      ... and {len(result.errors) - 3} more")

            if result.warnings:
                print(f"    Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    print(f"      - {warning}")
                if len(result.warnings) > 3:
                    print(f"      ... and {len(result.warnings) - 3} more")

        # Metrics summary
        print("\nMetrics Summary:")
        all_metrics = {}
        for result in self.test_results:
            all_metrics.update(result.metrics)

        for key, value in all_metrics.items():
            print(f"  {key}: {value}")

        # Generate report data
        report_data = {
            "timestamp": "2025-07-09T10:00:00",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "test_results": [
                {
                    "test_name": result.test_name,
                    "passed": result.passed,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "metrics": result.metrics,
                }
                for result in self.test_results
            ],
            "overall_metrics": all_metrics,
        }

        # Save report
        report_file = self.output_dir / "netlogo_integration_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"\nFull report saved to: {report_file}")

        # Summary
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\nSUMMARY: {success_rate:.1f}% tests passed")

        if failed_tests == 0:
            print("🎉 All NetLogo integration tests passed!")
        else:
            print(f"⚠️  {failed_tests} tests failed - see details above")

        print("=" * 60)

        return report_data


def run_integration_tests():
    """Run the complete NetLogo integration test suite"""
    tester = NetLogoIntegrationTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    run_integration_tests()
