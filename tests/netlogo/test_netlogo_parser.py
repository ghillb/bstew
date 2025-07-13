#!/usr/bin/env python3
"""
Test script for NetLogo data parsers
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bstew.data.netlogo_parser import NetLogoDataParser


def test_netlogo_parser():
    """Test NetLogo data parser with existing data files"""
    print("Testing NetLogo Data Parser...")

    # Initialize parser
    parser = NetLogoDataParser()

    # Test specific files

    try:
        # Test parameters file
        print("\n1. Testing parameters file...")
        param_file = "data/parameters/_SYSTEM_Parameters.csv"
        if os.path.exists(param_file):
            params = parser.parameter_parser.parse_parameters_file(param_file)
            print(f"   Parsed {len(params)} parameters")
            for name, param in list(params.items())[:3]:  # Show first 3
                print(f"   - {name}: {param.value} ({param.data_type})")

        # Test species file
        print("\n2. Testing species file...")
        species_file = "data/parameters/_SYSTEM_BumbleSpecies_UK_01.csv"
        if os.path.exists(species_file):
            species = parser.species_parser.parse_species_file(species_file)
            print(f"   Parsed {len(species)} species")
            for name, sp in list(species.items())[:3]:  # Show first 3
                print(f"   - {name}: {sp.name}")
                print(
                    f"     Proboscis: {sp.parameters.get('proboscis_min_mm', 'N/A')}-{sp.parameters.get('proboscis_max_mm', 'N/A')} mm"
                )

        # Test flower file
        print("\n3. Testing flower file...")
        flower_file = "data/parameters/_SYSTEM_Flowerspecies.csv"
        if os.path.exists(flower_file):
            flowers = parser.flower_parser.parse_flower_file(flower_file)
            print(f"   Parsed {len(flowers)} flower species")
            for name, flower in list(flowers.items())[:3]:  # Show first 3
                print(f"   - {name}: {flower.start_day}-{flower.stop_day} days")
                print(f"     Corolla depth: {flower.corolla_depth_mm} mm")

        # Test food source file
        print("\n4. Testing food source file...")
        foodsource_file = "data/landscapes/_SYSTEM_Example_Farm_Foodsources.txt"
        if os.path.exists(foodsource_file):
            food_sources = parser.food_source_parser.parse_food_source_file(
                foodsource_file
            )
            print(f"   Parsed {len(food_sources)} food sources")
            for fs in food_sources[:3]:  # Show first 3
                print(f"   - {fs.patch_id}: ({fs.x}, {fs.y}), {fs.area_m2} m²")

        # Test habitat file
        print("\n5. Testing habitat file...")
        habitat_file = "data/parameters/_SYSTEM_Habitats.csv"
        if os.path.exists(habitat_file):
            habitats = parser.habitat_parser.parse_habitat_file(habitat_file)
            print(f"   Parsed {len(habitats)} habitats")
            for name, habitat in list(habitats.items())[:2]:  # Show first 2
                print(f"   - {name}: {len(habitat)} species")

        print("\n✓ NetLogo data parser tests completed successfully!")

    except Exception as e:
        print(f"✗ Error testing NetLogo parser: {e}")
        import traceback

        traceback.print_exc()


def test_conversion():
    """Test conversion to BSTEW format"""
    print("\nTesting BSTEW conversion...")

    try:
        parser = NetLogoDataParser()

        # Test species conversion
        species_file = "data/parameters/_SYSTEM_BumbleSpecies_UK_01.csv"
        if os.path.exists(species_file):
            netlogo_species = parser.species_parser.parse_species_file(species_file)
            bstew_species = parser.species_parser.convert_to_bstew_species(
                netlogo_species
            )

            print(f"   Converted {len(bstew_species)} species to BSTEW format")
            for name, species in list(bstew_species.items())[:2]:
                print(f"   - {name}: {species['common_name']}")
                print(
                    f"     Proboscis: {species['proboscis_length_min']}-{species['proboscis_length_max']} mm"
                )
                print(
                    f"     Emergence: {species['emergence_day_mean']} ± {species['emergence_day_sd']} days"
                )

        # Test flower conversion
        flower_file = "data/parameters/_SYSTEM_Flowerspecies.csv"
        if os.path.exists(flower_file):
            netlogo_flowers = parser.flower_parser.parse_flower_file(flower_file)
            bstew_flowers = parser.flower_parser.convert_to_bstew_flowers(
                netlogo_flowers
            )

            print(f"   Converted {len(bstew_flowers)} flowers to BSTEW format")
            for name, flower in list(bstew_flowers.items())[:2]:
                print(
                    f"   - {name}: {flower['bloom_start']}-{flower['bloom_end']} days"
                )
                print(f"     Corolla depth: {flower['corolla_depth']} mm")
                print(f"     Nectar: {flower['nectar_production']} mg/flower")

        print("\n✓ BSTEW conversion tests completed successfully!")

    except Exception as e:
        print(f"✗ Error testing BSTEW conversion: {e}")
        import traceback

        traceback.print_exc()


def test_full_parse():
    """Test full data directory parsing"""
    print("\nTesting full data directory parsing...")

    try:
        parser = NetLogoDataParser()

        # Parse entire data directory
        data_dir = "data"
        parsed_data = parser.parse_all_data_files(data_dir)

        print(
            f"   Parsed data from {len(parsed_data['metadata']['files_processed'])} files"
        )
        print(f"   Parameters: {len(parsed_data['parameters'])} files")
        print(f"   Species: {len(parsed_data['species'])} files")
        print(f"   Flowers: {len(parsed_data['flowers'])} files")
        print(f"   Food sources: {len(parsed_data['food_sources'])} files")
        print(f"   Habitats: {len(parsed_data['habitats'])} files")
        print(f"   Landscapes: {len(parsed_data['landscapes'])} files")

        # Export to JSON
        output_file = parser.export_parsed_data(parsed_data, "artifacts/test_outputs")
        print(f"   Exported to: {output_file}")

        print("\n✓ Full data parsing tests completed successfully!")

    except Exception as e:
        print(f"✗ Error testing full data parsing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Create test output directory
    os.makedirs("artifacts/test_outputs", exist_ok=True)

    # Run tests
    test_netlogo_parser()
    test_conversion()
    test_full_parse()

    print("\n" + "=" * 60)
    print("NetLogo Parser Testing Complete!")
    print("=" * 60)
