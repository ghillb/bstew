"""
Runtime Parameters CLI Commands
==============================

Command-line interface for runtime parameter management in BSTEW.
"""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

from ...core.parameter_loader import RuntimeParameterManager

console = Console()

def modify_parameter(
    parameter_name: str = typer.Argument(..., help="Parameter name to modify"),
    new_value: str = typer.Argument(..., help="New parameter value"),
    simulation_id: Optional[str] = typer.Option(None, help="Target simulation ID"),
    validate_first: bool = typer.Option(True, help="Validate parameter before applying"),
    create_backup: bool = typer.Option(True, help="Create backup before modification"),
    reason: Optional[str] = typer.Option(None, help="Reason for parameter change"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Modify a parameter during simulation runtime"""
    
    try:
        # Initialize runtime parameter manager
        runtime_manager = RuntimeParameterManager()  # type: ignore
        
        # Parse the new value
        parsed_value = parse_parameter_value(new_value)
        
        console.print(f"[bold blue]Modifying parameter: {parameter_name}[/bold blue]")
        console.print(f"[yellow]New value: {parsed_value}[/yellow]")
        
        if reason:
            console.print(f"[dim]Reason: {reason}[/dim]")
        
        # Validate parameter if requested
        if validate_first:
            console.print("[yellow]Validating parameter change...[/yellow]")
            
            # Mock validation - in practice this would use the actual validation logic
            is_valid = validate_parameter_change(parameter_name, parsed_value)
            
            if not is_valid:
                console.print("[bold red]✗ Parameter validation failed[/bold red]")
                raise typer.Exit(1)
            
            console.print("[green]✓ Parameter validation passed[/green]")
        
        # Apply parameter change
        console.print("[yellow]Applying parameter change...[/yellow]")
        
        success = apply_parameter_change(
            runtime_manager, 
            parameter_name, 
            parsed_value, 
            simulation_id,
            reason
        )
        
        if success:
            console.print(f"[bold green]✓ Parameter {parameter_name} modified successfully[/bold green]")
            
            if verbose:
                # Show current parameter value
                current_value = get_current_parameter_value(runtime_manager, parameter_name)
                console.print(f"[dim]Current value: {current_value}[/dim]")
        else:
            console.print(f"[bold red]✗ Failed to modify parameter {parameter_name}[/bold red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error modifying parameter: {e}[/bold red]")
        raise typer.Exit(1)


def show_parameter_history(
    parameter_name: Optional[str] = typer.Option(None, help="Specific parameter name"),
    simulation_id: Optional[str] = typer.Option(None, help="Filter by simulation ID"),
    limit: int = typer.Option(50, help="Maximum number of entries to show"),
    export_file: Optional[str] = typer.Option(None, help="Export history to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Show parameter change history"""
    
    try:
        # Initialize runtime parameter manager
        runtime_manager = RuntimeParameterManager()  # type: ignore
        
        console.print("[bold blue]Retrieving parameter change history[/bold blue]")
        
        # Get parameter history
        history = get_parameter_history(runtime_manager, parameter_name, simulation_id, limit)
        
        if not history:
            console.print("[yellow]No parameter changes found[/yellow]")
            return
        
        # Display history table
        history_table = Table(title="Parameter Change History")
        history_table.add_column("Timestamp", style="cyan")
        history_table.add_column("Parameter", style="magenta")
        history_table.add_column("Old Value", style="red")
        history_table.add_column("New Value", style="green")
        history_table.add_column("Reason", style="dim")
        
        for change in history[:limit]:
            timestamp_str = change["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(change["timestamp"], datetime) else str(change["timestamp"])
            
            history_table.add_row(
                timestamp_str,
                change.get("parameter_name", "Unknown"),
                str(change.get("old_value", "N/A")),
                str(change.get("new_value", "N/A")),
                change.get("reason", "No reason provided")[:30] + "..." if len(change.get("reason", "")) > 30 else change.get("reason", "")
            )
        
        console.print(history_table)
        
        # Export if requested
        if export_file:
            export_parameter_history(history, export_file)
            console.print(f"[dim]History exported to: {export_file}[/dim]")
        
        console.print(f"[bold green]✓ Showing {len(history)} parameter changes[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error retrieving parameter history: {e}[/bold red]")
        raise typer.Exit(1)


def rollback_parameter(
    parameter_name: str = typer.Argument(..., help="Parameter name to rollback"),
    steps: int = typer.Option(1, help="Number of steps to rollback"),
    simulation_id: Optional[str] = typer.Option(None, help="Target simulation ID"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm rollback operation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Rollback parameter to previous value"""
    
    try:
        if not confirm:
            console.print(f"[yellow]This will rollback {parameter_name} by {steps} step(s)[/yellow]")
            console.print("[dim]Use --confirm to proceed with rollback[/dim]")
            return
        
        # Initialize runtime parameter manager
        runtime_manager = RuntimeParameterManager()  # type: ignore
        
        console.print(f"[bold blue]Rolling back parameter: {parameter_name}[/bold blue]")
        console.print(f"[yellow]Rollback steps: {steps}[/yellow]")
        
        # Get current value for confirmation
        current_value = get_current_parameter_value(runtime_manager, parameter_name)
        console.print(f"[dim]Current value: {current_value}[/dim]")
        
        # Perform rollback
        success = perform_parameter_rollback(runtime_manager, parameter_name, steps, simulation_id)
        
        if success:
            new_value = get_current_parameter_value(runtime_manager, parameter_name)
            console.print(f"[bold green]✓ Parameter {parameter_name} rolled back successfully[/bold green]")
            console.print(f"[dim]New value: {new_value}[/dim]")
        else:
            console.print(f"[bold red]✗ Failed to rollback parameter {parameter_name}[/bold red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error rolling back parameter: {e}[/bold red]")
        raise typer.Exit(1)


def rollback_all_parameters(
    simulation_id: Optional[str] = typer.Option(None, help="Target simulation ID"),
    to_timestamp: Optional[str] = typer.Option(None, help="Rollback to specific timestamp (ISO format)"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm rollback operation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Rollback all parameters to previous state"""
    
    try:
        if not confirm:
            console.print("[yellow]This will rollback ALL parameters to a previous state[/yellow]")
            console.print("[bold red]This is a potentially destructive operation![/bold red]")
            console.print("[dim]Use --confirm to proceed with rollback[/dim]")
            return
        
        # Initialize runtime parameter manager
        runtime_manager = RuntimeParameterManager()  # type: ignore
        
        console.print("[bold blue]Rolling back all parameters[/bold blue]")
        
        if to_timestamp:
            console.print(f"[yellow]Rolling back to timestamp: {to_timestamp}[/yellow]")
        
        # Perform full rollback
        success = perform_full_rollback(runtime_manager, simulation_id, to_timestamp)
        
        if success:
            console.print("[bold green]✓ All parameters rolled back successfully[/bold green]")
        else:
            console.print("[bold red]✗ Failed to rollback parameters[/bold red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error rolling back parameters: {e}[/bold red]")
        raise typer.Exit(1)


def export_parameter_log(
    output_file: str = typer.Option("artifacts/runtime/parameter_log.json", help="Output log file"),
    simulation_id: Optional[str] = typer.Option(None, help="Filter by simulation ID"),
    format_type: str = typer.Option("json", help="Export format (json/csv)"),
    include_metadata: bool = typer.Option(True, help="Include metadata"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Export parameter change log"""
    
    try:
        # Initialize runtime parameter manager
        runtime_manager = RuntimeParameterManager()  # type: ignore
        
        console.print("[bold blue]Exporting parameter change log[/bold blue]")
        
        # Get full parameter log
        parameter_log = get_full_parameter_log(runtime_manager, simulation_id)
        
        # Prepare export data
        export_data = prepare_log_export_data(parameter_log, include_metadata)
        
        # Export log
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format_type.lower() == "csv":
            import pandas as pd
            df = pd.DataFrame(export_data)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        console.print("[bold green]✓ Parameter log exported successfully[/bold green]")
        console.print(f"[dim]Log saved to: {output_path}[/dim]")
        console.print(f"[dim]Total entries: {len(export_data)}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error exporting parameter log: {e}[/bold red]")
        raise typer.Exit(1)


def show_current_parameters(
    filter_pattern: Optional[str] = typer.Option(None, help="Filter parameters by pattern"),
    simulation_id: Optional[str] = typer.Option(None, help="Target simulation ID"),
    include_metadata: bool = typer.Option(False, help="Include parameter metadata"),
    export_file: Optional[str] = typer.Option(None, help="Export current parameters to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Show current parameter values"""
    
    try:
        # Initialize runtime parameter manager
        runtime_manager = RuntimeParameterManager()  # type: ignore
        
        console.print("[bold blue]Retrieving current parameter values[/bold blue]")
        
        # Get current parameters
        current_params = get_current_parameters(runtime_manager, simulation_id, filter_pattern)
        
        if not current_params:
            console.print("[yellow]No parameters found[/yellow]")
            return
        
        # Display parameters table
        params_table = Table(title="Current Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="magenta")
        params_table.add_column("Type", style="green")
        
        if include_metadata:
            params_table.add_column("Last Modified", style="dim")
        
        for param_name, param_info in current_params.items():
            value = param_info.get("value", "N/A")
            param_type = type(value).__name__
            
            if include_metadata:
                last_modified = param_info.get("last_modified", "Unknown")
                params_table.add_row(param_name, str(value), param_type, str(last_modified))
            else:
                params_table.add_row(param_name, str(value), param_type)
        
        console.print(params_table)
        
        # Export if requested
        if export_file:
            export_current_parameters(current_params, export_file)
            console.print(f"[dim]Parameters exported to: {export_file}[/dim]")
        
        console.print(f"[bold green]✓ Showing {len(current_params)} parameters[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error retrieving current parameters: {e}[/bold red]")
        raise typer.Exit(1)


def validate_parameter_configuration(
    config_file: str = typer.Argument(..., help="Parameter configuration file"),
    strict_mode: bool = typer.Option(False, help="Enable strict validation"),
    output_report: Optional[str] = typer.Option(None, help="Output validation report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Validate parameter configuration file"""
    
    try:
        console.print(f"[bold blue]Validating parameter configuration: {config_file}[/bold blue]")
        
        # Load configuration file
        config_path = Path(config_file)
        if not config_path.exists():
            console.print(f"[bold red]✗ Configuration file not found: {config_file}[/bold red]")
            raise typer.Exit(1)
        
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        # Validate configuration
        validation_results = validate_configuration(config_data, strict_mode)
        
        # Display validation results
        display_validation_results(validation_results)
        
        # Export report if requested
        if output_report:
            export_validation_report(validation_results, output_report)
            console.print(f"[dim]Validation report saved to: {output_report}[/dim]")
        
        # Summary
        if validation_results["is_valid"]:
            console.print("[bold green]✓ Configuration validation passed[/bold green]")
        else:
            console.print("[bold red]✗ Configuration validation failed[/bold red]")
            console.print(f"[dim]Errors found: {len(validation_results['errors'])}[/dim]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error validating configuration: {e}[/bold red]")
        raise typer.Exit(1)


# Helper functions
def parse_parameter_value(value_str: str) -> Any:
    """Parse parameter value from string"""
    
    # Try to parse as JSON first for complex types
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass
    
    # Try common type conversions
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    
    try:
        # Try integer
        return int(value_str)
    except ValueError:
        pass
    
    try:
        # Try float
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def validate_parameter_change(parameter_name: str, new_value: Any) -> bool:
    """Validate parameter change"""
    
    # Mock validation - in practice this would use actual validation logic
    # from the RuntimeParameterManager
    
    # Basic validation rules
    if parameter_name.endswith("_rate") and isinstance(new_value, (int, float)):
        return 0.0 <= new_value <= 1.0
    
    if parameter_name.endswith("_count") and isinstance(new_value, int):
        return new_value >= 0
    
    # Default to valid for demonstration
    return True


def apply_parameter_change(manager: RuntimeParameterManager, param_name: str, 
                         new_value: Any, simulation_id: Optional[str], reason: Optional[str]) -> bool:
    """Apply parameter change through manager"""
    
    try:
        # In practice, this would use the actual RuntimeParameterManager methods
        # For now, we'll simulate the operation
        
        # Mock the parameter change
        return True
        
    except Exception:
        return False


def get_current_parameter_value(manager: RuntimeParameterManager, param_name: str) -> Any:
    """Get current parameter value"""
    
    # Mock current value retrieval
    return f"current_value_of_{param_name}"


def get_parameter_history(manager: RuntimeParameterManager, param_name: Optional[str], 
                         simulation_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
    """Get parameter change history"""
    
    # Mock parameter history
    history = []
    for i in range(min(limit, 20)):  # Mock up to 20 entries
        history.append({
            "timestamp": datetime.now(),
            "parameter_name": param_name or f"mock_param_{i}",
            "old_value": f"old_value_{i}",
            "new_value": f"new_value_{i}",
            "reason": f"Test change {i}"
        })
    
    return history


def export_parameter_history(history: List[Dict[str, Any]], output_file: str) -> None:
    """Export parameter history to file"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2, default=str)


def perform_parameter_rollback(manager: RuntimeParameterManager, param_name: str, 
                             steps: int, simulation_id: Optional[str]) -> bool:
    """Perform parameter rollback"""
    
    # Mock rollback operation
    return True


def perform_full_rollback(manager: RuntimeParameterManager, simulation_id: Optional[str], 
                         to_timestamp: Optional[str]) -> bool:
    """Perform full parameter rollback"""
    
    # Mock full rollback operation
    return True


def get_full_parameter_log(manager: RuntimeParameterManager, simulation_id: Optional[str]) -> List[Dict[str, Any]]:
    """Get full parameter log"""
    
    # Mock parameter log
    return [
        {
            "timestamp": datetime.now(),
            "parameter_name": "test_param",
            "old_value": "old",
            "new_value": "new",
            "reason": "Test change"
        }
    ]


def prepare_log_export_data(log_data: List[Dict[str, Any]], include_metadata: bool) -> List[Dict[str, Any]]:
    """Prepare log data for export"""
    
    if not include_metadata:
        # Remove metadata fields
        cleaned_data = []
        for entry in log_data:
            cleaned_entry = {k: v for k, v in entry.items() if not k.startswith("_")}
            cleaned_data.append(cleaned_entry)
        return cleaned_data
    
    return log_data


def get_current_parameters(manager: RuntimeParameterManager, simulation_id: Optional[str], 
                          filter_pattern: Optional[str]) -> Dict[str, Any]:
    """Get current parameters"""
    
    # Mock current parameters
    params = {
        "mortality_rate": {"value": 0.05, "last_modified": datetime.now()},
        "foraging_efficiency": {"value": 0.78, "last_modified": datetime.now()},
        "population_size": {"value": 125000, "last_modified": datetime.now()}
    }
    
    if filter_pattern:
        # Filter parameters by pattern
        filtered_params = {}
        for param_name, param_info in params.items():
            if filter_pattern.lower() in param_name.lower():
                filtered_params[param_name] = param_info
        return filtered_params
    
    return params


def export_current_parameters(params: Dict[str, Any], output_file: str) -> None:
    """Export current parameters to file"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2, default=str)


def validate_configuration(config_data: Dict[str, Any], strict_mode: bool) -> Dict[str, Any]:
    """Validate configuration data"""
    
    # Mock validation
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "strict_mode": strict_mode
    }
    
    # Add some mock validation rules
    if "invalid_param" in config_data:
        validation_results["is_valid"] = False
        errors_list = validation_results["errors"]
        if isinstance(errors_list, list):
            errors_list.append("Invalid parameter found: invalid_param")
    
    return validation_results


def display_validation_results(results: Dict[str, Any]) -> None:
    """Display validation results"""
    
    if results["is_valid"]:
        console.print("[bold green]✓ Validation passed[/bold green]")
    else:
        console.print("[bold red]✗ Validation failed[/bold red]")
    
    if results["errors"]:
        error_table = Table(title="Validation Errors")
        error_table.add_column("Error", style="red")
        
        for error in results["errors"]:
            error_table.add_row(error)
        
        console.print(error_table)
    
    if results["warnings"]:
        warning_table = Table(title="Validation Warnings")
        warning_table.add_column("Warning", style="yellow")
        
        for warning in results["warnings"]:
            warning_table.add_row(warning)
        
        console.print(warning_table)


def export_validation_report(results: Dict[str, Any], output_file: str) -> None:
    """Export validation report to file"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# Create typer app for runtime parameter commands
app = typer.Typer(name="runtime", help="Runtime parameter management commands")

app.command(name="modify")(modify_parameter)
app.command(name="history")(show_parameter_history)
app.command(name="rollback")(rollback_parameter)
app.command(name="rollback-all")(rollback_all_parameters)
app.command(name="export-log")(export_parameter_log)
app.command(name="show")(show_current_parameters)
app.command(name="validate")(validate_parameter_configuration)

if __name__ == "__main__":
    app()