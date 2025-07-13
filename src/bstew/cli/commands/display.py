"""
Display Configuration CLI Commands
=================================

Command-line interface for display configuration management in BSTEW.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import json

from ...visualization.display_system import (
    DisplayStateManager, 
    DisplayToggle, 
    ColorScheme, 
    DisplayControlInterface
)

console = Console()

def show_display_status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Show current display configuration status"""
    
    try:
        # Initialize display manager
        display_manager = DisplayStateManager()
        
        # Get current status
        status = {
            "current_color_scheme": display_manager.current_color_scheme.value,
            "available_configurations": display_manager.get_available_configurations(),
            "toggle_states": {
                toggle.value: display_manager.get_toggle_state(toggle)
                for toggle in DisplayToggle
            },
            "custom_colors": display_manager.custom_colors,
            "visibility_filters": display_manager.visibility_filters,
            "display_parameters": display_manager.display_parameters
        }
        
        # Display current configuration
        console.print(Panel(f"[bold green]Current Color Scheme:[/bold green] {status['current_color_scheme']}", title="Display Status"))
        
        # Display toggle states
        toggle_table = Table(title="Display Toggles")
        toggle_table.add_column("Toggle", style="cyan")
        toggle_table.add_column("Status", style="magenta")
        toggle_table.add_column("Description", style="dim")
        
        toggle_descriptions = {
            "show_cohorts": "Show bee cohorts",
            "show_dead_cols": "Show dead colonies",
            "show_paths": "Show movement paths",
            "show_resources": "Show resource patches",
            "show_territories": "Show territory boundaries",
            "show_activity": "Show activity colors",
            "show_species": "Show species colors",
            "show_health": "Show health indicators",
            "show_connectivity": "Show connectivity networks",
            "show_grid": "Show coordinate grid",
            "show_labels": "Show entity labels",
            "show_legend": "Show plot legend",
            "show_statistics": "Show statistics panel",
            "show_stewardship": "Show stewardship options",
            "show_management": "Show management areas"
        }
        
        for toggle_name, enabled in status["toggle_states"].items():
            status_icon = "✓" if enabled else "✗"
            status_color = "green" if enabled else "red"
            description = toggle_descriptions.get(toggle_name, "")
            
            toggle_table.add_row(
                toggle_name.replace("_", " ").title(),
                f"[{status_color}]{status_icon}[/{status_color}]",
                description
            )
        
        console.print(toggle_table)
        
        # Display available configurations
        if status["available_configurations"]:
            config_table = Table(title="Available Configurations")
            config_table.add_column("Configuration", style="cyan")
            config_table.add_column("Status", style="magenta")
            
            for config_name in status["available_configurations"]:
                config_table.add_row(config_name, "Available")
            
            console.print(config_table)
        
        # Display custom colors if any
        if status["custom_colors"] and verbose:
            colors_table = Table(title="Custom Colors")
            colors_table.add_column("Category", style="cyan")
            colors_table.add_column("Color", style="magenta")
            
            for category, color in status["custom_colors"].items():
                colors_table.add_row(category, color)
            
            console.print(colors_table)
        
        # Display visibility filters if any
        if status["visibility_filters"] and verbose:
            filters_table = Table(title="Visibility Filters")
            filters_table.add_column("Filter", style="cyan")
            filters_table.add_column("Value", style="magenta")
            
            for filter_name, filter_value in status["visibility_filters"].items():
                filters_table.add_row(filter_name, str(filter_value))
            
            console.print(filters_table)
        
        console.print("[bold green]✓ Display status retrieved successfully[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error retrieving display status: {e}[/bold red]")
        raise typer.Exit(1)


def toggle_display_option(
    option: str = typer.Argument(..., help="Display option to toggle"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Toggle a display option"""
    
    try:
        # Initialize display manager
        display_manager = DisplayStateManager()
        control_interface = DisplayControlInterface(display_manager)
        
        # Toggle the option
        success = control_interface.toggle_display_option(option)
        
        if success:
            new_state = display_manager.get_toggle_state(DisplayToggle(option))
            status_text = "enabled" if new_state else "disabled"
            status_color = "green" if new_state else "red"
            
            console.print(f"[bold {status_color}]✓ {option.replace('_', ' ').title()} {status_text}[/bold {status_color}]")
        else:
            console.print(f"[bold red]✗ Invalid display option: {option}[/bold red]")
            console.print("[dim]Available options: show_cohorts, show_dead_cols, show_paths, show_resources, show_territories, show_activity, show_species, show_health, show_connectivity, show_grid, show_labels, show_legend, show_statistics, show_stewardship, show_management[/dim]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error toggling display option: {e}[/bold red]")
        raise typer.Exit(1)


def set_display_configuration(
    config_name: str = typer.Argument(..., help="Configuration name to apply"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Set display configuration"""
    
    try:
        # Initialize display manager
        display_manager = DisplayStateManager()
        control_interface = DisplayControlInterface(display_manager)
        
        # Apply configuration
        success = control_interface.set_display_configuration(config_name)
        
        if success:
            console.print(f"[bold green]✓ Applied configuration: {config_name}[/bold green]")
            
            if verbose:
                # Show what was applied
                status = control_interface.get_display_status()
                console.print(f"[dim]Color scheme: {status['color_scheme']}[/dim]")
                
                enabled_toggles = [toggle for toggle, enabled in status["toggles"].items() if enabled]
                console.print(f"[dim]Enabled toggles: {', '.join(enabled_toggles)}[/dim]")
        else:
            console.print(f"[bold red]✗ Configuration not found: {config_name}[/bold red]")
            
            # Show available configurations
            available_configs = display_manager.get_available_configurations()
            console.print(f"[dim]Available configurations: {', '.join(available_configs)}[/dim]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error setting display configuration: {e}[/bold red]")
        raise typer.Exit(1)


def set_color_scheme(
    scheme: str = typer.Argument(..., help="Color scheme to apply"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Set color scheme"""
    
    try:
        # Initialize display manager
        display_manager = DisplayStateManager()
        
        # Set color scheme
        try:
            color_scheme = ColorScheme(scheme)
            display_manager.set_color_scheme(color_scheme)
            
            console.print(f"[bold green]✓ Color scheme set to: {scheme}[/bold green]")
            
        except ValueError:
            console.print(f"[bold red]✗ Invalid color scheme: {scheme}[/bold red]")
            console.print("[dim]Available schemes: species_colors, activity_colors, health_colors, resource_colors, connectivity_colors, management_colors, custom[/dim]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error setting color scheme: {e}[/bold red]")
        raise typer.Exit(1)


def set_visibility_filter(
    filter_name: str = typer.Argument(..., help="Filter name"),
    filter_value: str = typer.Argument(..., help="Filter value (JSON format for complex filters)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Set visibility filter"""
    
    try:
        # Initialize display manager
        display_manager = DisplayStateManager()
        control_interface = DisplayControlInterface(display_manager)
        
        # Parse filter value
        try:
            # Try to parse as JSON for complex filters
            parsed_value = json.loads(filter_value)
        except json.JSONDecodeError:
            # Use as string for simple filters
            parsed_value = filter_value
        
        # Set filter
        control_interface.set_visibility_filter(filter_name, parsed_value)
        
        console.print(f"[bold green]✓ Visibility filter set: {filter_name} = {parsed_value}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error setting visibility filter: {e}[/bold red]")
        raise typer.Exit(1)


def clear_visibility_filters(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Clear all visibility filters"""
    
    try:
        # Initialize display manager
        display_manager = DisplayStateManager()
        control_interface = DisplayControlInterface(display_manager)
        
        # Clear filters
        control_interface.clear_visibility_filters()
        
        console.print("[bold green]✓ All visibility filters cleared[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error clearing visibility filters: {e}[/bold red]")
        raise typer.Exit(1)


def create_custom_configuration(
    name: str = typer.Argument(..., help="Configuration name"),
    description: str = typer.Argument(..., help="Configuration description"),
    from_current: bool = typer.Option(True, help="Create from current settings"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Create custom display configuration"""
    
    try:
        # Initialize display manager
        display_manager = DisplayStateManager()
        
        # Create configuration
        if from_current:
            display_manager.create_custom_configuration(name, description)
        else:
            # Create with default settings
            display_manager.create_custom_configuration(name, description, {})
        
        console.print(f"[bold green]✓ Custom configuration created: {name}[/bold green]")
        console.print(f"[dim]Description: {description}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error creating custom configuration: {e}[/bold red]")
        raise typer.Exit(1)


def export_display_configuration(
    output_file: str = typer.Option("artifacts/display_config_export.json", help="Output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Export display configuration"""
    
    try:
        # Initialize display manager
        display_manager = DisplayStateManager()
        
        # Export configuration
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get complete configuration
        config_data = {
            "current_color_scheme": display_manager.current_color_scheme.value,
            "toggle_states": {
                toggle.value: display_manager.get_toggle_state(toggle)
                for toggle in DisplayToggle
            },
            "custom_colors": display_manager.custom_colors,
            "visibility_filters": display_manager.visibility_filters,
            "display_parameters": display_manager.display_parameters,
            "available_configurations": display_manager.get_available_configurations()
        }
        
        with open(output_path, "w") as f:
            json.dump(config_data, f, indent=2)
        
        console.print("[bold green]✓ Display configuration exported[/bold green]")
        console.print(f"[dim]Exported to: {output_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error exporting display configuration: {e}[/bold red]")
        raise typer.Exit(1)


def import_display_configuration(
    input_file: str = typer.Argument(..., help="Input configuration file"),
    apply_immediately: bool = typer.Option(False, help="Apply configuration immediately"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Import display configuration"""
    
    try:
        # Check if file exists
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[bold red]✗ Configuration file not found: {input_file}[/bold red]")
            raise typer.Exit(1)
        
        # Load configuration
        with open(input_path, "r") as f:
            config_data = json.load(f)
        
        # Initialize display manager
        display_manager = DisplayStateManager()
        
        # Apply configuration if requested
        if apply_immediately:
            # Set color scheme
            if "current_color_scheme" in config_data:
                display_manager.set_color_scheme(ColorScheme(config_data["current_color_scheme"]))
            
            # Set toggle states
            if "toggle_states" in config_data:
                for toggle_name, state in config_data["toggle_states"].items():
                    try:
                        toggle = DisplayToggle(toggle_name)
                        display_manager.set_toggle_state(toggle, state)
                    except ValueError:
                        console.print(f"[yellow]Warning: Unknown toggle '{toggle_name}' skipped[/yellow]")
            
            # Set custom colors
            if "custom_colors" in config_data:
                for category, color in config_data["custom_colors"].items():
                    display_manager.set_custom_color(category, color)
            
            # Set visibility filters
            if "visibility_filters" in config_data:
                for filter_name, filter_value in config_data["visibility_filters"].items():
                    display_manager.set_visibility_filter(filter_name, filter_value)
            
            # Set display parameters
            if "display_parameters" in config_data:
                for param_name, param_value in config_data["display_parameters"].items():
                    display_manager.set_display_parameter(param_name, param_value)
            
            console.print("[bold green]✓ Display configuration imported and applied[/bold green]")
        else:
            console.print("[bold green]✓ Display configuration imported[/bold green]")
            console.print("[dim]Use --apply-immediately to apply the configuration[/dim]")
        
        console.print(f"[dim]Imported from: {input_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error importing display configuration: {e}[/bold red]")
        raise typer.Exit(1)


def reset_display_configuration(
    confirm: bool = typer.Option(False, "--confirm", help="Confirm reset"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Reset display configuration to defaults"""
    
    try:
        if not confirm:
            console.print("[yellow]This will reset all display settings to defaults.[/yellow]")
            console.print("[dim]Use --confirm to proceed with reset[/dim]")
            return
        
        # Initialize display manager
        display_manager = DisplayStateManager()
        
        # Apply default configuration
        display_manager.apply_configuration("default")
        
        console.print("[bold green]✓ Display configuration reset to defaults[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error resetting display configuration: {e}[/bold red]")
        raise typer.Exit(1)


# Create typer app for display commands
app = typer.Typer(name="display", help="Display configuration commands")

app.command(name="status")(show_display_status)
app.command(name="toggle")(toggle_display_option)
app.command(name="config")(set_display_configuration)
app.command(name="scheme")(set_color_scheme)
app.command(name="filter")(set_visibility_filter)
app.command(name="clear-filters")(clear_visibility_filters)
app.command(name="create-config")(create_custom_configuration)
app.command(name="export")(export_display_configuration)
app.command(name="import")(import_display_configuration)
app.command(name="reset")(reset_display_configuration)

if __name__ == "__main__":
    app()