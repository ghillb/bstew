#!/usr/bin/env python3
"""
Complete test of NetLogo parameter mapping system
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bstew.data.netlogo_parser import NetLogoDataParser
from bstew.config.netlogo_mapping import NetLogoParameterMapper
import json


def test_complete_mapping():
    """Test complete NetLogo to BSTEW parameter mapping with real data"""
    print("Testing Complete NetLogo Parameter Mapping System...")

    # Initialize parser and mapper
    parser = NetLogoDataParser()
    mapper = NetLogoParameterMapper()

    # Parse actual NetLogo data files

    # Test individual file parsing
    print("\n1. Testing individual file parsing...")

    # Test system parameters
    param_file = "data/parameters/_SYSTEM_Parameters.csv"
    if os.path.exists(param_file):
        print(f"   - Parsing system parameters from {param_file}")
        system_params = parser.parameter_parser.parse_parameters_file(param_file)
        print(f"   - Found {len(system_params)} system parameters")

        # Show sample parameters
        sample_params = list(system_params.keys())[:5]
        print(f"   - Sample parameters: {sample_params}")

        # Convert to BSTEW format
        bstew_config = mapper.convert_system_parameters(system_params)
        print(f"   - Generated BSTEW config sections: {list(bstew_config.keys())}")

        # Show populated sections
        populated_sections = [k for k, v in bstew_config.items() if v]
        print(f"   - Populated sections: {populated_sections}")

        # Show sample conversions
        if bstew_config["foraging"]:
            print(f"   - Foraging config: {bstew_config['foraging']}")
        if bstew_config["predation"]:
            print(f"   - Predation config: {bstew_config['predation']}")
        if bstew_config["genetics"]:
            print(f"   - Genetics config: {bstew_config['genetics']}")

    # Test species parameters
    species_file = "data/parameters/_SYSTEM_BumbleSpecies_UK_01.csv"
    if os.path.exists(species_file):
        print(f"\n2. Testing species parameters from {species_file}")
        species_data = parser.species_parser.parse_species_file(species_file)
        print(f"   - Found {len(species_data)} species")

        # Show sample species
        sample_species = list(species_data.keys())[:3]
        print(f"   - Sample species: {sample_species}")

        # Convert to BSTEW format
        bstew_species = mapper.convert_species_parameters(species_data)

        # Show first species conversion
        first_species_id = next(iter(bstew_species.keys()))
        first_species = bstew_species[first_species_id]
        print(
            f"   - First species ({first_species_id}): {len(first_species)} parameters"
        )

        # Show key parameters
        key_params = [
            "emergence_day_mean",
            "proboscis_length_min",
            "proboscis_length_max",
            "flight_velocity_ms",
            "season_end_day",
        ]
        for param in key_params:
            if param in first_species:
                print(f"     - {param}: {first_species[param]}")

    # Test flower parameters
    flower_file = "data/parameters/_SYSTEM_Flowerspecies.csv"
    if os.path.exists(flower_file):
        print(f"\n3. Testing flower parameters from {flower_file}")
        flower_data = parser.flower_parser.parse_flower_file(flower_file)
        print(f"   - Found {len(flower_data)} flower species")

        # Show sample flowers
        sample_flowers = list(flower_data.keys())[:3]
        print(f"   - Sample flowers: {sample_flowers}")

        # Convert to BSTEW format
        bstew_flowers = mapper.convert_flower_parameters(flower_data)

        # Show first flower conversion
        first_flower_id = next(iter(bstew_flowers.keys()))
        first_flower = bstew_flowers[first_flower_id]
        print(f"   - First flower ({first_flower_id}): {len(first_flower)} parameters")

        # Show key parameters
        key_params = [
            "bloom_start",
            "bloom_end",
            "corolla_depth",
            "nectar_production",
            "pollen_production",
            "protein_content",
        ]
        for param in key_params:
            if param in first_flower:
                print(f"     - {param}: {first_flower[param]}")

    # Test full integration
    print("\n4. Testing full integration...")

    # Create a comprehensive BSTEW config
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
    if os.path.exists(param_file):
        system_params = parser.parameter_parser.parse_parameters_file(param_file)
        system_config = mapper.convert_system_parameters(system_params)
        full_config.update(system_config)

    # Add species parameters
    if os.path.exists(species_file):
        species_data = parser.species_parser.parse_species_file(species_file)
        species_config = mapper.convert_species_parameters(species_data)
        full_config["species"] = species_config

    # Add flower parameters
    if os.path.exists(flower_file):
        flower_data = parser.flower_parser.parse_flower_file(flower_file)
        flower_config = mapper.convert_flower_parameters(flower_data)
        full_config["flowers"] = flower_config

    # Add metadata
    full_config["metadata"] = {
        "converted_from": "NetLogo BEE-STEWARD",
        "conversion_timestamp": "2025-07-09T10:00:00",
        "total_parameters": len(system_params) if os.path.exists(param_file) else 0,
        "total_species": len(species_data) if os.path.exists(species_file) else 0,
        "total_flowers": len(flower_data) if os.path.exists(flower_file) else 0,
    }

    # Save complete configuration
    output_file = "artifacts/test_outputs/complete_bstew_config.json"
    with open(output_file, "w") as f:
        json.dump(full_config, f, indent=2, default=str)

    print(f"   - Complete BSTEW configuration saved to {output_file}")
    print(f"   - Configuration sections: {list(full_config.keys())}")

    # Show summary statistics
    populated_sections = [k for k, v in full_config.items() if v and k != "metadata"]
    print(f"   - Populated sections: {populated_sections}")

    if full_config["metadata"]:
        meta = full_config["metadata"]
        print(f"   - Total parameters converted: {meta['total_parameters']}")
        print(f"   - Total species converted: {meta['total_species']}")
        print(f"   - Total flowers converted: {meta['total_flowers']}")

    print("\n✓ Complete NetLogo parameter mapping test completed successfully!")


if __name__ == "__main__":
    # Create test output directory
    os.makedirs("artifacts/test_outputs", exist_ok=True)

    # Run complete test
    test_complete_mapping()

    print("\n" + "=" * 60)
    print("Complete NetLogo Parameter Mapping Test Finished!")
    print("=" * 60)
