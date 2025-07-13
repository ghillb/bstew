"""
Parameter Management CLI Commands
===============================

CLI commands for managing parameters, including NetLogo integration,
runtime modification, and parameter validation.
"""

import json
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

from ...core.parameter_loader import ParameterLoader, ParameterType, ParameterFormat
from ...core.netlogo_integration import NetLogoDataIntegrator
from ..core.base import CLIContext

console = Console()


class ParameterManagerCLI:
    """CLI for parameter management operations"""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.parameter_loader = ParameterLoader()
        self.netlogo_integrator = NetLogoDataIntegrator()
        
    def load_parameters(
        self,
        file_path: str,
        param_type: str = "behavioral",
        format_type: str = "csv",
        validate: bool = True,
        output_dir: str = "artifacts/parameters"
    ) -> None:
        """Load parameters from file"""
        
        try:
            # Parse parameter type and format
            parameter_type = ParameterType(param_type.lower())
            parameter_format = ParameterFormat(format_type.lower())
            
            self.context.print_info(f"Loading {parameter_type.value} parameters from {file_path}")
            
            # Load parameters
            if parameter_type == ParameterType.BEHAVIORAL:
                params = self.parameter_loader.load_behavioral_parameters(file_path, parameter_format)
            elif parameter_type == ParameterType.ENVIRONMENTAL:
                params = self.parameter_loader.load_environmental_parameters(file_path, parameter_format)
            elif parameter_type == ParameterType.COLONY:
                params = self.parameter_loader.load_colony_parameters(file_path, parameter_format)
            else:
                self.context.print_error(f"Unsupported parameter type: {parameter_type}")
                return
                
            # Validate if requested
            if validate:
                validation_errors = self.parameter_loader.validate_all_parameters()
                if validation_errors:
                    self.context.print_warning("Parameter validation warnings:")
                    for param_type, errors in validation_errors.items():
                        for error in errors:
                            console.print(f"  - {param_type}: {error}")
                            
            # Save validated parameters
            if self.parameter_loader.auto_save_enabled:
                self.parameter_loader.save_validated_parameters(parameter_type.value, output_dir)
                
            self.context.print_success(f"Parameters loaded successfully: {params}")
            
        except Exception as e:
            self.context.print_error(f"Error loading parameters: {e}")
            
    def inspect_parameters(
        self,
        file_path: str,
        show_details: bool = False,
        export_schema: bool = False
    ) -> None:
        """Inspect parameter file contents"""
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.context.print_error(f"File not found: {file_path}")
                return
                
            self.context.print_info(f"Inspecting parameter file: {file_path}")
            
            # Determine file type
            if file_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                
                # Basic info
                info_table = Table(title="File Information")
                info_table.add_column("Property", style="cyan")
                info_table.add_column("Value", style="white")
                
                info_table.add_row("File Path", str(file_path))
                info_table.add_row("File Size", f"{file_path.stat().st_size} bytes")
                info_table.add_row("Rows", str(len(df)))
                info_table.add_row("Columns", str(len(df.columns)))
                
                console.print(info_table)
                
                # Column info
                column_table = Table(title="Column Information")
                column_table.add_column("Column", style="cyan")
                column_table.add_column("Type", style="yellow")
                column_table.add_column("Non-Null", style="green")
                column_table.add_column("Sample Values", style="white")
                
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null = str(df[col].notna().sum())
                    sample_values = ", ".join(map(str, df[col].dropna().head(3).tolist()))
                    column_table.add_row(col, dtype, non_null, sample_values)
                    
                console.print(column_table)
                
                # Show detailed data if requested
                if show_details:
                    console.print("\n[bold]Sample Data:[/bold]")
                    console.print(df.head(10).to_string())
                    
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                console.print(Panel(json.dumps(data, indent=2), title="JSON Content"))
                
            # Export schema if requested
            if export_schema:
                schema_path = file_path.with_suffix('.schema.json')
                # Generate schema based on file content
                # This is a simplified schema generation
                schema = {
                    "file_path": str(file_path),
                    "format": file_path.suffix.lower(),
                    "analysis_date": pd.Timestamp.now().isoformat()
                }
                
                with open(schema_path, 'w') as f:
                    json.dump(schema, f, indent=2)
                    
                self.context.print_success(f"Schema exported to: {schema_path}")
                
        except Exception as e:
            self.context.print_error(f"Error inspecting parameters: {e}")
            
    def validate_parameters(
        self,
        file_path: str,
        param_type: str = "behavioral",
        strict: bool = False
    ) -> None:
        """Validate parameter file"""
        
        try:
            parameter_type = ParameterType(param_type.lower())
            
            self.context.print_info(f"Validating {parameter_type.value} parameters from {file_path}")
            
            # Load and validate parameters
            if parameter_type == ParameterType.BEHAVIORAL:
                self.parameter_loader.load_behavioral_parameters(file_path, ParameterFormat.CSV)
            elif parameter_type == ParameterType.ENVIRONMENTAL:
                self.parameter_loader.load_environmental_parameters(file_path, ParameterFormat.CSV)
            elif parameter_type == ParameterType.COLONY:
                self.parameter_loader.load_colony_parameters(file_path, ParameterFormat.CSV)
                
            # Get validation results
            validation_errors = self.parameter_loader.validate_all_parameters()
            
            if not validation_errors:
                self.context.print_success("✅ All parameters are valid!")
            else:
                self.context.print_warning("Parameter validation issues found:")
                
                for param_type, errors in validation_errors.items():
                    error_table = Table(title=f"{param_type.title()} Parameter Errors")
                    error_table.add_column("Error", style="red")
                    
                    for error in errors:
                        error_table.add_row(error)
                        
                    console.print(error_table)
                    
                if strict:
                    self.context.print_error("Validation failed in strict mode")
                    
        except Exception as e:
            self.context.print_error(f"Error validating parameters: {e}")
            
    def convert_parameters(
        self,
        input_file: str,
        output_file: str,
        input_format: str = "csv",
        output_format: str = "json",
        param_type: str = "behavioral"
    ) -> None:
        """Convert parameters between formats"""
        
        try:
            input_format_enum = ParameterFormat(input_format.lower())
            output_format_enum = ParameterFormat(output_format.lower())
            parameter_type = ParameterType(param_type.lower())
            
            self.context.print_info(f"Converting {input_file} from {input_format} to {output_format}")
            
            # Load parameters
            if parameter_type == ParameterType.BEHAVIORAL:
                params = self.parameter_loader.load_behavioral_parameters(input_file, input_format_enum)
            elif parameter_type == ParameterType.ENVIRONMENTAL:
                params = self.parameter_loader.load_environmental_parameters(input_file, input_format_enum)
            elif parameter_type == ParameterType.COLONY:
                params = self.parameter_loader.load_colony_parameters(input_file, input_format_enum)
            else:
                self.context.print_error(f"Unsupported parameter type: {parameter_type}")
                return
                
            # Save in new format
            if output_format_enum == ParameterFormat.JSON:
                with open(output_file, 'w') as f:
                    json.dump(params.model_dump(), f, indent=2)
            elif output_format_enum == ParameterFormat.CSV:
                # Convert to CSV format
                import pandas as pd
                df = pd.DataFrame([params.model_dump()]).T
                df.columns = ['value']
                df.index.name = 'parameter'
                df.to_csv(output_file)
            else:
                self.context.print_error(f"Unsupported output format: {output_format}")
                return
                
            self.context.print_success(f"Parameters converted successfully: {output_file}")
            
        except Exception as e:
            self.context.print_error(f"Error converting parameters: {e}")
            
    def discover_netlogo_files(
        self,
        directory: str,
        export_report: bool = False,
        output_dir: str = "artifacts/netlogo_discovery"
    ) -> None:
        """Discover NetLogo files in directory"""
        
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                self.context.print_error(f"Directory not found: {directory}")
                return
                
            self.context.print_info(f"Discovering NetLogo files in: {directory}")
            
            # Discover files
            files_by_type = self.netlogo_integrator.discover_netlogo_files(dir_path)
            
            # Display results
            summary_table = Table(title="NetLogo Files Discovery Summary")
            summary_table.add_column("File Type", style="cyan")
            summary_table.add_column("Count", style="yellow")
            summary_table.add_column("Total Rows", style="green")
            
            total_files = 0
            total_rows = 0
            
            for file_type, files in files_by_type.items():
                file_count = len(files)
                row_count = sum(f.row_count for f in files)
                total_files += file_count
                total_rows += row_count
                
                summary_table.add_row(file_type.value, str(file_count), str(row_count))
                
            summary_table.add_row("TOTAL", str(total_files), str(total_rows), style="bold")
            console.print(summary_table)
            
            # Detailed file information
            for file_type, files in files_by_type.items():
                if files:
                    detail_table = Table(title=f"{file_type.value.title()} Files")
                    detail_table.add_column("File", style="cyan")
                    detail_table.add_column("Rows", style="yellow")
                    detail_table.add_column("Columns", style="green")
                    detail_table.add_column("Status", style="white")
                    
                    for file_info in files:
                        status = "✅ Valid" if not file_info.validation_errors else "❌ Issues"
                        detail_table.add_row(
                            file_info.file_path.name,
                            str(file_info.row_count),
                            str(file_info.column_count),
                            status
                        )
                        
                    console.print(detail_table)
                    
            # Export report if requested
            if export_report:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Convert to serializable format
                report_data = {}
                for file_type, files in files_by_type.items():
                    report_data[file_type.value] = []
                    for file_info in files:
                        report_data[file_type.value].append({
                            "file_path": str(file_info.file_path),
                            "row_count": file_info.row_count,
                            "column_count": file_info.column_count,
                            "columns": file_info.columns,
                            "validation_errors": file_info.validation_errors,
                            "contains_data": file_info.contains_data
                        })
                        
                report_file = output_path / "netlogo_discovery_report.json"
                with open(report_file, 'w') as f:
                    json.dump(report_data, f, indent=2)
                    
                self.context.print_success(f"Discovery report exported to: {report_file}")
                
        except Exception as e:
            self.context.print_error(f"Error discovering NetLogo files: {e}")
            
    def validate_netlogo_compatibility(
        self,
        directory: str,
        fix_issues: bool = False
    ) -> None:
        """Validate NetLogo data compatibility"""
        
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                self.context.print_error(f"Directory not found: {directory}")
                return
                
            self.context.print_info(f"Validating NetLogo compatibility for: {directory}")
            
            # Run compatibility validation
            validation_report = self.netlogo_integrator.validate_netlogo_compatibility(dir_path)
            
            # Display results
            status_color = "green" if validation_report["compatible"] else "red"
            status_text = "✅ Compatible" if validation_report["compatible"] else "❌ Incompatible"
            
            console.print(f"\n[{status_color}]{status_text}[/{status_color}]")
            
            # Files found
            files_table = Table(title="Files Found")
            files_table.add_column("File Type", style="cyan")
            files_table.add_column("Count", style="yellow")
            
            for file_type, count in validation_report["files_found"].items():
                files_table.add_row(file_type, str(count))
                
            console.print(files_table)
            
            # Errors
            if validation_report["errors"]:
                error_table = Table(title="Compatibility Errors")
                error_table.add_column("Error", style="red")
                
                for error in validation_report["errors"]:
                    error_table.add_row(error)
                    
                console.print(error_table)
                
            # Warnings
            if validation_report["warnings"]:
                warning_table = Table(title="Warnings")
                warning_table.add_column("Warning", style="yellow")
                
                for warning in validation_report["warnings"]:
                    warning_table.add_row(warning)
                    
                console.print(warning_table)
                
            # Recommendations
            if validation_report["recommendations"]:
                rec_table = Table(title="Recommendations")
                rec_table.add_column("Recommendation", style="blue")
                
                for rec in validation_report["recommendations"]:
                    rec_table.add_row(rec)
                    
                console.print(rec_table)
                
            # Offer to fix issues
            if not validation_report["compatible"] and fix_issues:
                if Confirm.ask("Would you like to attempt automatic issue resolution?"):
                    self.context.print_info("Attempting to fix compatibility issues...")
                    # Add automatic fixing logic here
                    self.context.print_info("Automatic fixing not yet implemented")
                    
        except Exception as e:
            self.context.print_error(f"Error validating NetLogo compatibility: {e}")
            
    def convert_netlogo_data(
        self,
        netlogo_directory: str,
        output_directory: str = "artifacts/netlogo_converted",
        validate_first: bool = True
    ) -> None:
        """Convert NetLogo data to BSTEW format"""
        
        try:
            netlogo_dir = Path(netlogo_directory)
            output_dir = Path(output_directory)
            
            if not netlogo_dir.exists():
                self.context.print_error(f"NetLogo directory not found: {netlogo_directory}")
                return
                
            # Validate first if requested
            if validate_first:
                self.context.print_info("Validating NetLogo data before conversion...")
                validation_report = self.netlogo_integrator.validate_netlogo_compatibility(netlogo_dir)
                
                if not validation_report["compatible"]:
                    self.context.print_error("NetLogo data is not compatible. Please fix issues first.")
                    return
                    
            self.context.print_info(f"Converting NetLogo data from {netlogo_directory} to {output_directory}")
            
            # Perform conversion
            self.netlogo_integrator.convert_netlogo_to_bstew(netlogo_dir, output_dir)
            
            self.context.print_success(f"NetLogo data converted successfully to: {output_directory}")
            
            # Display converted files
            converted_files = list(output_dir.glob("**/*"))
            if converted_files:
                files_table = Table(title="Converted Files")
                files_table.add_column("File", style="cyan")
                files_table.add_column("Type", style="yellow")
                files_table.add_column("Size", style="green")
                
                for file_path in converted_files:
                    if file_path.is_file():
                        file_type = file_path.suffix.upper()
                        file_size = f"{file_path.stat().st_size} bytes"
                        files_table.add_row(file_path.name, file_type, file_size)
                        
                console.print(files_table)
                
        except Exception as e:
            self.context.print_error(f"Error converting NetLogo data: {e}")
            
    def create_parameter_template(
        self,
        param_type: str = "behavioral",
        output_file: str = "artifacts/parameters/template.csv",
        format_type: str = "csv"
    ) -> None:
        """Create parameter template file"""
        
        try:
            parameter_type = ParameterType(param_type.lower())
            
            self.context.print_info(f"Creating {parameter_type.value} parameter template")
            
            # Create template
            self.parameter_loader.create_parameter_template(parameter_type, output_file)
            
            self.context.print_success(f"Parameter template created: {output_file}")
            
        except Exception as e:
            self.context.print_error(f"Error creating parameter template: {e}")


# CLI Command Functions
def load_parameters_cmd(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    param_type: str = typer.Option("behavioral", help="Parameter type (behavioral/environmental/colony)"),
    format_type: str = typer.Option("csv", help="File format (csv/json/xlsx)"),
    validate: bool = typer.Option(True, help="Validate parameters after loading"),
    output_dir: str = typer.Option("artifacts/parameters", help="Output directory for validated parameters")
):
    """Load parameters from file"""
    context = CLIContext()
    manager = ParameterManagerCLI(context)
    manager.load_parameters(file_path, param_type, format_type, validate, output_dir)


def inspect_parameters_cmd(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    details: bool = typer.Option(False, "--details", help="Show detailed data"),
    schema: bool = typer.Option(False, "--schema", help="Export schema")
):
    """Inspect parameter file contents"""
    context = CLIContext()
    manager = ParameterManagerCLI(context)
    manager.inspect_parameters(file_path, details, schema)


def validate_parameters_cmd(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    param_type: str = typer.Option("behavioral", help="Parameter type"),
    strict: bool = typer.Option(False, "--strict", help="Strict validation mode")
):
    """Validate parameter file"""
    context = CLIContext()
    manager = ParameterManagerCLI(context)
    manager.validate_parameters(file_path, param_type, strict)


def convert_parameters_cmd(
    input_file: str = typer.Argument(..., help="Input parameter file"),
    output_file: str = typer.Argument(..., help="Output parameter file"),
    input_format: str = typer.Option("csv", help="Input format"),
    output_format: str = typer.Option("json", help="Output format"),
    param_type: str = typer.Option("behavioral", help="Parameter type")
):
    """Convert parameters between formats"""
    context = CLIContext()
    manager = ParameterManagerCLI(context)
    manager.convert_parameters(input_file, output_file, input_format, output_format, param_type)


def discover_netlogo_cmd(
    directory: str = typer.Argument(..., help="Directory to search for NetLogo files"),
    export: bool = typer.Option(False, "--export", help="Export discovery report"),
    output_dir: str = typer.Option("artifacts/netlogo_discovery", help="Output directory for reports")
):
    """Discover NetLogo files in directory"""
    context = CLIContext()
    manager = ParameterManagerCLI(context)
    manager.discover_netlogo_files(directory, export, output_dir)


def validate_netlogo_cmd(
    directory: str = typer.Argument(..., help="Directory containing NetLogo files"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix compatibility issues")
):
    """Validate NetLogo data compatibility"""
    context = CLIContext()
    manager = ParameterManagerCLI(context)
    manager.validate_netlogo_compatibility(directory, fix)


def convert_netlogo_cmd(
    netlogo_directory: str = typer.Argument(..., help="NetLogo data directory"),
    output_directory: str = typer.Option("artifacts/netlogo_converted", help="Output directory"),
    validate_first: bool = typer.Option(True, "--validate/--no-validate", help="Validate before conversion")
):
    """Convert NetLogo data to BSTEW format"""
    context = CLIContext()
    manager = ParameterManagerCLI(context)
    manager.convert_netlogo_data(netlogo_directory, output_directory, validate_first)


def create_template_cmd(
    param_type: str = typer.Option("behavioral", help="Parameter type"),
    output_file: str = typer.Option("artifacts/parameters/template.csv", help="Output file"),
    format_type: str = typer.Option("csv", help="File format")
):
    """Create parameter template file"""
    context = CLIContext()
    manager = ParameterManagerCLI(context)
    manager.create_parameter_template(param_type, output_file, format_type)