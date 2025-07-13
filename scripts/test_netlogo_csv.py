#!/usr/bin/env python3
"""
Test Script for NetLogo CSV Compatibility System
==============================================

Demonstrates NetLogo CSV reading, writing, validation, and data conversion
capabilities for seamless interoperability with NetLogo models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import tempfile
import shutil
from src.bstew.core.netlogo_compatibility import (
    NetLogoCSVReader, NetLogoCSVWriter, NetLogoDataConverter,
    create_netlogo_export_package, CSVValidationLevel
)

def test_netlogo_csv_compatibility():
    """Test the NetLogo CSV compatibility system"""
    
    print("=== NetLogo CSV Compatibility System Test ===\n")
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}\n")
    
    try:
        # Test 1: Create and validate sample NetLogo agent data
        print("1. Creating sample NetLogo agent data...")
        
        agent_data = {
            'who': [0, 1, 2, 3, 4],
            'breed': ['bumblebees', 'bumblebees', 'honeybees', 'honeybees', 'mason-bees'],
            'xcor': [10.5, -5.2, 0.0, 15.8, -10.1],
            'ycor': [8.3, 12.7, -3.4, 6.9, -8.5],
            'heading': [45, 180, 270, 90, 0],
            'color': [15, 15, 25, 25, 35],
            'size': [1.0, 1.2, 0.8, 1.1, 0.9],
            'energy': [95.5, 78.2, 102.1, 85.6, 92.3],
            'age': [15, 23, 8, 31, 12],
            'status': ['foraging', 'resting', 'nursing', 'foraging', 'building']
        }
        
        agents_df = pd.DataFrame(agent_data)
        agents_file = os.path.join(temp_dir, "test_agents.csv")
        agents_df.to_csv(agents_file, index=False)
        
        print(f"   - Created agent data: {len(agents_df)} agents")
        print(f"   - Columns: {list(agents_df.columns)}")
        print()
        
        # Test 2: Read and validate NetLogo CSV
        print("2. Reading and validating NetLogo CSV...")
        
        reader = NetLogoCSVReader(validation_level=CSVValidationLevel.MODERATE)
        df_read, validation_result = reader.read_csv(agents_file, "agents")
        
        print(f"   - Validation result: {'✓ VALID' if validation_result.is_valid else '✗ INVALID'}")
        print(f"   - NetLogo compatible: {'Yes' if validation_result.netlogo_compatible else 'No'}")
        print(f"   - Errors: {len(validation_result.errors)}")
        print(f"   - Warnings: {len(validation_result.warnings)}")
        
        if validation_result.errors:
            for error in validation_result.errors:
                print(f"     Error: {error}")
        
        if validation_result.warnings:
            for warning in validation_result.warnings[:3]:  # Show first 3 warnings
                print(f"     Warning: {warning}")
        
        print()
        
        # Test 3: Test with invalid data
        print("3. Testing validation with invalid data...")
        
        invalid_data = {
            'who': [0, 1, 2, 1],  # Duplicate ID
            'breed': ['bumblebees', 'invalid-breed', 'honeybees', 'bumblebees'],
            'xcor': [10.5, 'invalid', 0.0, 15.8],  # Non-numeric coordinate
            'ycor': [8.3, 12.7, -3.4, 6.9],
            'heading': [45, 180, 270, 450],  # Out of range heading
            'energy': [95.5, -10.0, 102.1, 85.6],  # Negative energy
            'age': [15, 23, 400, 31]  # Unrealistic age
        }
        
        invalid_df = pd.DataFrame(invalid_data)
        invalid_file = os.path.join(temp_dir, "invalid_agents.csv")
        invalid_df.to_csv(invalid_file, index=False)
        
        df_invalid, invalid_validation = reader.read_csv(invalid_file, "agents")
        
        print(f"   - Validation result: {'✓ VALID' if invalid_validation.is_valid else '✗ INVALID'}")
        print(f"   - Errors found: {len(invalid_validation.errors)}")
        print(f"   - Warnings found: {len(invalid_validation.warnings)}")
        
        if invalid_validation.suggested_fixes:
            print("   - Suggested fixes:")
            for fix in invalid_validation.suggested_fixes[:3]:
                print(f"     • {fix}")
        
        print()
        
        # Test 4: Test NetLogo CSV writer
        print("4. Testing NetLogo CSV writer...")
        
        writer = NetLogoCSVWriter(
            netlogo_precision=3,
            coordinate_precision=2,
            boolean_format="true/false"
        )
        
        output_file = os.path.join(temp_dir, "output_agents.csv")
        write_result = writer.write_csv(agents_df, output_file, "agents")
        
        print(f"   - Write result: {'✓ SUCCESS' if write_result.is_valid else '✗ FAILED'}")
        print(f"   - Output file: {output_file}")
        
        # Verify written file
        if os.path.exists(output_file):
            written_df = pd.read_csv(output_file)
            print(f"   - Written data: {len(written_df)} rows, {len(written_df.columns)} columns")
        
        print()
        
        # Test 5: Test data conversion between BSTEW and NetLogo formats
        print("5. Testing data format conversion...")
        
        converter = NetLogoDataConverter(
            coordinate_scale=10.0,  # Scale coordinates by 10x
            coordinate_offset=(50.0, 50.0)  # Offset by (50, 50)
        )
        
        # Sample BSTEW agent data
        bstew_agent = {
            'unique_id': 42,
            'species': 'bombus_terrestris',
            'pos': (5.5, 3.2),
            'heading': 1.57,  # π/2 radians (90 degrees)
            'energy': 85.5,
            'age': 25,
            'status': 'foraging',
            'foraging_experience': 12,
            'nectar_load': 2.5,
            'colony_id': 1
        }
        
        # Convert to NetLogo format
        netlogo_agent = converter.convert_bstew_to_netlogo(bstew_agent, "agent")
        print("   - BSTEW → NetLogo conversion:")
        print(f"     Position: {bstew_agent['pos']} → ({netlogo_agent['xcor']}, {netlogo_agent['ycor']})")
        print(f"     Heading: {bstew_agent['heading']} rad → {netlogo_agent['heading']}°")
        print(f"     Species: {bstew_agent['species']} → {netlogo_agent['breed']}")
        print(f"     Status: {bstew_agent['status']} → {netlogo_agent['status']}")
        
        # Convert back to BSTEW format
        bstew_converted = converter.convert_netlogo_to_bstew(netlogo_agent, "agent")
        print("   - NetLogo → BSTEW conversion:")
        print(f"     Position: ({netlogo_agent['xcor']}, {netlogo_agent['ycor']}) → {bstew_converted['pos']}")
        print(f"     Heading: {netlogo_agent['heading']}° → {bstew_converted['heading']:.3f} rad")
        print(f"     Species: {netlogo_agent['breed']} → {bstew_converted['species']}")
        
        print()
        
        # Test 6: Test patch data
        print("6. Testing patch data handling...")
        
        patch_data = {
            'pxcor': [0, 1, 2, 0, 1],
            'pycor': [0, 0, 0, 1, 1],
            'pcolor': [0, 0, 0, 55, 55],
            'flower-density': [10.5, 8.2, 15.7, 3.1, 12.4],
            'nectar-amount': [45.2, 38.9, 67.3, 12.1, 52.8],
            'pollen-amount': [23.4, 19.6, 31.2, 8.7, 28.1],
            'resource-quality': [0.8, 0.6, 0.9, 0.3, 0.7]
        }
        
        patches_df = pd.DataFrame(patch_data)
        patches_file = os.path.join(temp_dir, "test_patches.csv")
        
        # Write and validate patches
        patch_write_result = writer.write_csv(patches_df, patches_file, "patches")
        print(f"   - Patch write result: {'✓ SUCCESS' if patch_write_result.is_valid else '✗ FAILED'}")
        
        # Read and validate patches
        patch_df, patch_validation = reader.read_csv(patches_file, "patches")
        print(f"   - Patch validation: {'✓ VALID' if patch_validation.is_valid else '✗ INVALID'}")
        print(f"   - Patches read: {len(patch_df)} patches")
        
        print()
        
        # Test 7: Create complete NetLogo export package
        print("7. Creating complete NetLogo export package...")
        
        export_dir = os.path.join(temp_dir, "netlogo_export")
        
        # Simulate complete model data
        model_data = {
            "agents": [
                {
                    'unique_id': i,
                    'species': 'bombus_terrestris' if i % 2 == 0 else 'apis_mellifera',
                    'pos': (np.random.uniform(-10, 10), np.random.uniform(-10, 10)),
                    'heading': np.random.uniform(0, 2 * np.pi),
                    'energy': np.random.uniform(50, 150),
                    'age': np.random.randint(1, 100),
                    'status': np.random.choice(['foraging', 'at_hive', 'nursing']),
                    'foraging_experience': np.random.randint(0, 50),
                    'colony_id': i // 10
                }
                for i in range(25)
            ],
            "patches": [
                {
                    'pos': (x, y),
                    'flower_density': np.random.uniform(0, 20),
                    'nectar_amount': np.random.uniform(0, 100),
                    'pollen_amount': np.random.uniform(0, 80),
                    'resource_quality': np.random.uniform(0.2, 1.0)
                }
                for x in range(-2, 3) for y in range(-2, 3)
            ],
            "globals": {
                'current_step': 150,
                'total_population': 25,
                'total_energy': 2500.0,
                'active_colonies': 3,
                'temperature': 22.5,
                'season': 'summer',
                'day_of_year': 180,
                'foraging_efficiency': 0.85,
                'resource_scarcity': 0.3
            }
        }
        
        export_result = create_netlogo_export_package(model_data, export_dir)
        
        print(f"   - Export success: {'✓ YES' if export_result['success'] else '✗ NO'}")
        print(f"   - Files created: {len(export_result['files_created'])}")
        
        for file_path in export_result['files_created']:
            filename = os.path.basename(file_path)
            print(f"     • {filename}")
        
        if export_result['errors']:
            print("   - Errors encountered:")
            for error in export_result['errors'][:3]:
                print(f"     • {error}")
        
        # Show validation summary
        print("   - Validation summary:")
        for filename, result in export_result['validation_results'].items():
            status = "✓ Valid" if result.is_valid else "✗ Invalid"
            print(f"     • {filename}: {status}")
        
        print()
        
        # Test 8: Performance test with larger dataset
        print("8. Performance test with larger dataset...")
        
        large_agent_data = {
            'who': list(range(1000)),
            'breed': ['bumblebees'] * 500 + ['honeybees'] * 500,
            'xcor': np.random.uniform(-50, 50, 1000),
            'ycor': np.random.uniform(-50, 50, 1000),
            'heading': np.random.uniform(0, 360, 1000),
            'color': np.random.choice([15, 25, 35], 1000),
            'size': np.random.uniform(0.5, 1.5, 1000),
            'energy': np.random.uniform(20, 150, 1000),
            'age': np.random.randint(1, 365, 1000),
            'status': np.random.choice(['foraging', 'resting', 'nursing'], 1000)
        }
        
        large_df = pd.DataFrame(large_agent_data)
        large_file = os.path.join(temp_dir, "large_agents.csv")
        
        import time
        
        # Test write performance
        start_time = time.time()
        large_write_result = writer.write_csv(large_df, large_file, "agents")
        write_time = time.time() - start_time
        
        # Test read performance
        start_time = time.time()
        large_read_df, large_read_result = reader.read_csv(large_file, "agents")
        read_time = time.time() - start_time
        
        print(f"   - Large dataset: {len(large_df)} agents")
        print(f"   - Write time: {write_time:.3f} seconds")
        print(f"   - Read time: {read_time:.3f} seconds")
        print(f"   - Write success: {'✓' if large_write_result.is_valid else '✗'}")
        print(f"   - Read success: {'✓' if large_read_result.is_valid else '✗'}")
        
        print()
        
        # Summary
        print("=== Test Summary ===")
        
        test_results = {
            "Basic CSV validation": validation_result.is_valid,
            "Invalid data detection": not invalid_validation.is_valid,
            "CSV writing": write_result.is_valid,
            "Data conversion": len(netlogo_agent) > 0,
            "Patch handling": patch_validation.is_valid,
            "Export package creation": export_result['success'],
            "Large dataset handling": large_read_result.is_valid
        }
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All NetLogo CSV compatibility tests passed!")
        else:
            print("⚠️  Some tests failed - check implementation")
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    test_netlogo_csv_compatibility()