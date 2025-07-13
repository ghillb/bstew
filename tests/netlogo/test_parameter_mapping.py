#!/usr/bin/env python3
"""
Test script for NetLogo parameter mapping system
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bstew.data.netlogo_parser import NetLogoDataParser
from bstew.config.netlogo_mapping import (
    NetLogoParameterMapper,
    convert_netlogo_to_bstew,
)


def test_parameter_mapping():
    """Test NetLogo parameter mapping system"""
    print("Testing NetLogo Parameter Mapping System...")

    # Initialize parser and mapper
    parser = NetLogoDataParser()
    mapper = NetLogoParameterMapper()

    print("\n1. Mapping system initialized with:")
    summary = mapper.get_mapping_summary()
    print(f"   - System parameters: {summary['system_parameters']['count']}")
    print(
        f"   - Species parameters: {summary['species_parameters']['parameters_per_species']}"
    )
    print(f"   - Flower parameters: {summary['flower_parameters']['count']}")

    # Parse NetLogo data
    data_dir = "data"
    try:
        print(f"\n2. Parsing NetLogo data from {data_dir}...")
        netlogo_data = parser.parse_all_data_files(data_dir)

        print(f"   - Found {len(netlogo_data['parameters'])} parameter files")
        print(f"   - Found {len(netlogo_data['species'])} species files")
        print(f"   - Found {len(netlogo_data['flowers'])} flower files")

        # Test system parameter conversion with actual data
        print("\n3. Testing system parameter conversion...")
        param_file = "data/parameters/_SYSTEM_Parameters.csv"
        if os.path.exists(param_file):
            params = parser.parameter_parser.parse_parameters_file(param_file)

            # Convert to BSTEW format
            bstew_config = mapper.convert_system_parameters(params)

            print(f"   - Converted {len(params)} NetLogo parameters")
            print(f"   - Generated BSTEW config sections: {list(bstew_config.keys())}")

            # Show some examples
            if "foraging" in bstew_config:
                print(f"   - Foraging config: {bstew_config['foraging']}")
            if "predation" in bstew_config:
                print(f"   - Predation config: {bstew_config['predation']}")

        # Test species parameter conversion with actual data
        print("\n4. Testing species parameter conversion...")
        species_file = "data/parameters/_SYSTEM_BumbleSpecies_UK_01.csv"
        if os.path.exists(species_file):
            species_data = parser.species_parser.parse_species_file(species_file)

            # Convert to BSTEW format
            bstew_species = mapper.convert_species_parameters(species_data)

            print(f"   - Converted {len(species_data)} species")

            # Show first species example
            first_species = next(iter(bstew_species.values()))
            print(f"   - First species parameters: {len(first_species)}")
            print(
                f"   - Proboscis range: {first_species.get('proboscis_length_min', 'N/A')}-{first_species.get('proboscis_length_max', 'N/A')}"
            )
            print(
                f"   - Emergence day: {first_species.get('emergence_day_mean', 'N/A')} ± {first_species.get('emergence_day_sd', 'N/A')}"
            )

        # Test flower parameter conversion with actual data
        print("\n5. Testing flower parameter conversion...")
        flower_file = "data/parameters/_SYSTEM_Flowerspecies.csv"
        if os.path.exists(flower_file):
            flower_data = parser.flower_parser.parse_flower_file(flower_file)

            # Convert to BSTEW format
            bstew_flowers = mapper.convert_flower_parameters(flower_data)

            print(f"   - Converted {len(flower_data)} flower species")

            # Show first flower example
            first_flower = next(iter(bstew_flowers.values()))
            print(f"   - First flower parameters: {len(first_flower)}")
            print(
                f"   - Bloom period: {first_flower.get('bloom_start', 'N/A')}-{first_flower.get('bloom_end', 'N/A')}"
            )
            print(f"   - Corolla depth: {first_flower.get('corolla_depth', 'N/A')}")
            print(
                f"   - Nectar production: {first_flower.get('nectar_production', 'N/A')}"
            )

        print("\n✓ Parameter mapping tests completed successfully!")

    except Exception as e:
        print(f"✗ Error testing parameter mapping: {e}")
        import traceback

        traceback.print_exc()


def test_full_conversion():
    """Test full NetLogo to BSTEW conversion"""
    print("\nTesting full NetLogo to BSTEW conversion...")

    try:
        # Parse NetLogo data
        parser = NetLogoDataParser()
        netlogo_data = parser.parse_all_data_files("data")

        # Convert to BSTEW format
        bstew_config = convert_netlogo_to_bstew(
            netlogo_data, "artifacts/test_outputs/bstew_config.json"
        )

        print(f"   - Generated BSTEW configuration with {len(bstew_config)} sections")
        print(f"   - Metadata: {bstew_config.get('metadata', {})}")

        # Validate some key parameters
        if "foraging" in bstew_config:
            max_range = bstew_config["foraging"].get("max_range_m", "N/A")
            print(f"   - Max foraging range: {max_range} m")

        if "predation" in bstew_config:
            badger_count = bstew_config["predation"].get("badger_count", "N/A")
            print(f"   - Badger count: {badger_count}")

        if "genetics" in bstew_config:
            csd_enabled = bstew_config["genetics"].get("csd_enabled", "N/A")
            print(f"   - CSD enabled: {csd_enabled}")

        if "species" in bstew_config:
            species_count = len(bstew_config["species"])
            print(f"   - Species converted: {species_count}")

        if "flowers" in bstew_config:
            flower_count = len(bstew_config["flowers"])
            print(f"   - Flowers converted: {flower_count}")

        print("\n✓ Full conversion test completed successfully!")

    except Exception as e:
        print(f"✗ Error testing full conversion: {e}")
        import traceback

        traceback.print_exc()


def test_validation():
    """Test parameter conversion validation"""
    print("\nTesting parameter conversion validation...")

    try:
        # Parse NetLogo data
        parser = NetLogoDataParser()
        netlogo_data = parser.parse_all_data_files("data")

        # Convert to BSTEW format
        bstew_config = convert_netlogo_to_bstew(netlogo_data)

        # Validate conversion
        mapper = NetLogoParameterMapper()
        validation_results = mapper.validate_conversion(netlogo_data, bstew_config)

        print(f"   - Total parameters: {validation_results['total_parameters']}")
        print(f"   - Mapped parameters: {validation_results['mapped_parameters']}")
        print(
            f"   - Missing parameters: {len(validation_results['missing_parameters'])}"
        )
        print(f"   - Type mismatches: {len(validation_results['type_mismatches'])}")
        print(f"   - Range violations: {len(validation_results['range_violations'])}")

        if validation_results["missing_parameters"]:
            print(f"   - Missing: {validation_results['missing_parameters'][:5]}...")

        if validation_results["type_mismatches"]:
            print(
                f"   - Type mismatches: {validation_results['type_mismatches'][:3]}..."
            )

        if validation_results["range_violations"]:
            print(
                f"   - Range violations: {validation_results['range_violations'][:3]}..."
            )

        # Calculate mapping coverage
        if validation_results["total_parameters"] > 0:
            coverage = (
                validation_results["mapped_parameters"]
                / validation_results["total_parameters"]
                * 100
            )
            print(f"   - Parameter mapping coverage: {coverage:.1f}%")
        else:
            print("   - Parameter mapping coverage: 0.0% (no parameters found)")

        print("\n✓ Validation test completed successfully!")

    except Exception as e:
        print(f"✗ Error testing validation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Create test output directory
    os.makedirs("artifacts/test_outputs", exist_ok=True)

    # Run tests
    test_parameter_mapping()
    test_full_conversion()
    test_validation()

    print("\n" + "=" * 60)
    print("NetLogo Parameter Mapping Testing Complete!")
    print("=" * 60)
