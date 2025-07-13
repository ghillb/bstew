#!/usr/bin/env python3
"""
NetLogo CSV Command Line Interface
=================================

Command-line utility for NetLogo CSV operations including validation,
conversion, and export package creation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import pandas as pd
from pathlib import Path

from src.bstew.core.netlogo_compatibility import (
    NetLogoCSVReader, NetLogoCSVWriter, NetLogoDataConverter,
    create_netlogo_export_package, CSVValidationLevel
)

def validate_csv(args):
    """Validate a CSV file for NetLogo compatibility"""
    
    print(f"Validating CSV file: {args.input}")
    
    # Initialize reader with specified validation level
    validation_level = CSVValidationLevel(args.validation_level)
    reader = NetLogoCSVReader(validation_level=validation_level)
    
    try:
        # Read and validate CSV
        df, validation_result = reader.read_csv(args.input, args.type)
        
        # Print validation results
        print(f"\n{'='*50}")
        print("VALIDATION RESULTS")
        print(f"{'='*50}")
        
        print(f"File: {args.input}")
        print(f"Type: {args.type}")
        print(f"Rows: {validation_result.row_count}")
        print(f"Columns: {validation_result.column_count}")
        print(f"Valid: {'✓ YES' if validation_result.is_valid else '✗ NO'}")
        print(f"NetLogo Compatible: {'✓ YES' if validation_result.netlogo_compatible else '✗ NO'}")
        
        if validation_result.errors:
            print(f"\nERRORS ({len(validation_result.errors)}):")
            for i, error in enumerate(validation_result.errors, 1):
                print(f"  {i}. {error}")
        
        if validation_result.warnings:
            print(f"\nWARNINGS ({len(validation_result.warnings)}):")
            for i, warning in enumerate(validation_result.warnings, 1):
                print(f"  {i}. {warning}")
        
        if validation_result.suggested_fixes:
            print("\nSUGGESTED FIXES:")
            for i, fix in enumerate(validation_result.suggested_fixes, 1):
                print(f"  {i}. {fix}")
        
        # Save detailed results if requested
        if args.output:
            result_data = {
                "file_path": str(args.input),
                "csv_type": args.type,
                "validation_level": args.validation_level,
                "is_valid": validation_result.is_valid,
                "netlogo_compatible": validation_result.netlogo_compatible,
                "row_count": validation_result.row_count,
                "column_count": validation_result.column_count,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "suggested_fixes": validation_result.suggested_fixes
            }
            
            with open(args.output, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"\nDetailed results saved to: {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if validation_result.is_valid else 1)
        
    except Exception as e:
        print(f"ERROR: Failed to validate CSV file: {e}")
        sys.exit(1)

def convert_csv(args):
    """Convert CSV between BSTEW and NetLogo formats"""
    
    print(f"Converting CSV file: {args.input}")
    print(f"Direction: {args.direction}")
    
    try:
        # Read input CSV
        df = pd.read_csv(args.input)
        print(f"Read {len(df)} rows, {len(df.columns)} columns")
        
        # Initialize converter
        converter = NetLogoDataConverter(
            coordinate_scale=args.coord_scale,
            coordinate_offset=(args.coord_offset_x, args.coord_offset_y)
        )
        
        if args.direction == "bstew-to-netlogo":
            # Convert BSTEW data to NetLogo format
            converted_data = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                netlogo_row = converter.convert_bstew_to_netlogo(row_dict, args.type)
                converted_data.append(netlogo_row)
            
            converted_df = pd.DataFrame(converted_data)
            
            # Write using NetLogo writer
            writer = NetLogoCSVWriter()
            validation_result = writer.write_csv(converted_df, args.output, args.type)
            
        else:  # netlogo-to-bstew
            # Convert NetLogo data to BSTEW format
            converted_data = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                bstew_row = converter.convert_netlogo_to_bstew(row_dict, args.type)
                converted_data.append(bstew_row)
            
            converted_df = pd.DataFrame(converted_data)
            converted_df.to_csv(args.output, index=False)
            
            # Create mock validation result
            from src.bstew.core.netlogo_compatibility import ValidationResult
            validation_result = ValidationResult(is_valid=True)
        
        print("\nConversion completed:")
        print(f"Output file: {args.output}")
        print(f"Converted rows: {len(converted_df)}")
        print(f"Success: {'✓ YES' if validation_result.is_valid else '✗ NO'}")
        
        if hasattr(validation_result, 'errors') and validation_result.errors:
            print("Errors:")
            for error in validation_result.errors:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"ERROR: Failed to convert CSV file: {e}")
        sys.exit(1)

def create_export(args):
    """Create NetLogo export package from model data"""
    
    print("Creating NetLogo export package")
    print(f"Input data: {args.input}")
    print(f"Output directory: {args.output}")
    
    try:
        # Load model data
        if args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                model_data = json.load(f)
        else:
            print("ERROR: Input file must be JSON format")
            sys.exit(1)
        
        # Create export package
        export_result = create_netlogo_export_package(model_data, args.output)
        
        print("\nExport package creation:")
        print(f"Success: {'✓ YES' if export_result['success'] else '✗ NO'}")
        print(f"Files created: {len(export_result['files_created'])}")
        
        if export_result['files_created']:
            print("Files:")
            for file_path in export_result['files_created']:
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                print(f"  - {filename} ({file_size} bytes)")
        
        if export_result['validation_results']:
            print("\nValidation results:")
            for csv_type, result in export_result['validation_results'].items():
                status = "✓ Valid" if result.is_valid else "✗ Invalid"
                print(f"  - {csv_type}: {status}")
        
        if export_result['errors']:
            print(f"\nErrors ({len(export_result['errors'])}):")
            for error in export_result['errors']:
                print(f"  - {error}")
        
        sys.exit(0 if export_result['success'] else 1)
        
    except Exception as e:
        print(f"ERROR: Failed to create export package: {e}")
        sys.exit(1)

def list_schemas(args):
    """List available CSV schemas"""
    
    print("Available NetLogo CSV Schemas")
    print("=" * 40)
    
    reader = NetLogoCSVReader()
    
    # Agent schema
    print("\nAGENT SCHEMA:")
    for col_name, schema in reader.agent_schema.items():
        required = " (required)" if schema.required else " (optional)"
        print(f"  {col_name}: {schema.data_type.value}{required}")
        if schema.description:
            print(f"    {schema.description}")
        if schema.valid_range:
            print(f"    Range: {schema.valid_range}")
        if schema.valid_values:
            print(f"    Values: {schema.valid_values}")
    
    # Patch schema
    print("\nPATCH SCHEMA:")
    for col_name, schema in reader.patch_schema.items():
        required = " (required)" if schema.required else " (optional)"
        print(f"  {col_name}: {schema.data_type.value}{required}")
        if schema.description:
            print(f"    {schema.description}")
        if schema.valid_range:
            print(f"    Range: {schema.valid_range}")
    
    # Global schema
    print("\nGLOBAL SCHEMA:")
    for col_name, schema in reader.global_schema.items():
        required = " (required)" if schema.required else " (optional)"
        print(f"  {col_name}: {schema.data_type.value}{required}")
        if schema.description:
            print(f"    {schema.description}")
        if schema.valid_range:
            print(f"    Range: {schema.valid_range}")
        if schema.valid_values:
            print(f"    Values: {schema.valid_values}")

def create_sample(args):
    """Create sample CSV files for testing"""
    
    print(f"Creating sample {args.type} CSV file: {args.output}")
    
    try:
        if args.type == "agents":
            sample_data = {
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
        
        elif args.type == "patches":
            sample_data = {
                'pxcor': [0, 1, 2, 0, 1],
                'pycor': [0, 0, 0, 1, 1],
                'pcolor': [0, 0, 0, 55, 55],
                'flower-density': [10.5, 8.2, 15.7, 3.1, 12.4],
                'nectar-amount': [45.2, 38.9, 67.3, 12.1, 52.8],
                'pollen-amount': [23.4, 19.6, 31.2, 8.7, 28.1],
                'resource-quality': [0.8, 0.6, 0.9, 0.3, 0.7]
            }
        
        elif args.type == "globals":
            sample_data = {
                'ticks': [150],
                'total-bees': [25],
                'total-energy': [2500.0],
                'temperature': [22.5],
                'season': ['summer']
            }
        
        else:
            print(f"ERROR: Unknown sample type: {args.type}")
            sys.exit(1)
        
        # Create DataFrame and save
        df = pd.DataFrame(sample_data)
        df.to_csv(args.output, index=False)
        
        print("Sample CSV created successfully:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  File: {args.output}")
        
    except Exception as e:
        print(f"ERROR: Failed to create sample CSV: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="NetLogo CSV Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a CSV file
  python netlogo_csv_cli.py validate agents.csv --type agents
  
  # Convert BSTEW data to NetLogo format
  python netlogo_csv_cli.py convert bstew_agents.csv netlogo_agents.csv \\
    --direction bstew-to-netlogo --type agents
  
  # Create NetLogo export package
  python netlogo_csv_cli.py export model_data.json export_dir/
  
  # List available schemas
  python netlogo_csv_cli.py schemas
  
  # Create sample CSV files
  python netlogo_csv_cli.py sample agents sample_agents.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate CSV file')
    validate_parser.add_argument('input', type=Path, help='Input CSV file')
    validate_parser.add_argument('--type', choices=['agents', 'patches', 'globals'], 
                                default='agents', help='CSV data type')
    validate_parser.add_argument('--validation-level', choices=['strict', 'moderate', 'lenient'],
                                default='moderate', help='Validation strictness')
    validate_parser.add_argument('--output', type=Path, help='Save detailed results to JSON file')
    validate_parser.set_defaults(func=validate_csv)
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert CSV between formats')
    convert_parser.add_argument('input', type=Path, help='Input CSV file')
    convert_parser.add_argument('output', type=Path, help='Output CSV file')
    convert_parser.add_argument('--direction', choices=['bstew-to-netlogo', 'netlogo-to-bstew'],
                               default='bstew-to-netlogo', help='Conversion direction')
    convert_parser.add_argument('--type', choices=['agent', 'patch', 'global'],
                               default='agent', help='Data type')
    convert_parser.add_argument('--coord-scale', type=float, default=1.0,
                               help='Coordinate scaling factor')
    convert_parser.add_argument('--coord-offset-x', type=float, default=0.0,
                               help='X coordinate offset')
    convert_parser.add_argument('--coord-offset-y', type=float, default=0.0,
                               help='Y coordinate offset')
    convert_parser.set_defaults(func=convert_csv)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Create NetLogo export package')
    export_parser.add_argument('input', type=Path, help='Input JSON file with model data')
    export_parser.add_argument('output', type=Path, help='Output directory for export package')
    export_parser.set_defaults(func=create_export)
    
    # Schemas command
    schemas_parser = subparsers.add_parser('schemas', help='List available CSV schemas')
    schemas_parser.set_defaults(func=list_schemas)
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Create sample CSV files')
    sample_parser.add_argument('type', choices=['agents', 'patches', 'globals'],
                              help='Type of sample CSV to create')
    sample_parser.add_argument('output', type=Path, help='Output CSV file')
    sample_parser.set_defaults(func=create_sample)
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main()