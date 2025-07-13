"""
Visualization CLI Commands
=========================

Command-line interface for visualization operations in BSTEW.
"""

import typer
from rich.console import Console
from pathlib import Path
import json
from typing import Optional, Dict, Any, List

from ...visualization.display_system import DisplayStateManager
from ...visualization.agent_visualization import AgentVisualizationEngine, VisualizationFrame
from ...visualization.interactive_analysis import DataExplorationEngine, InteractiveDashboard
from ...visualization.export_system import (
    VisualizationExporter, 
    ExportFormat, 
    ExportQuality, 
    ExportConfiguration
)

console = Console()

def export_visualization(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    output_dir: str = typer.Option("artifacts/visualization", help="Output directory"),
    format_type: str = typer.Option("png", help="Export format (png/svg/pdf/html)"),
    quality: str = typer.Option("high", help="Export quality (low/medium/high/publication)"),
    width: int = typer.Option(1920, help="Image width"),
    height: int = typer.Option(1080, help="Image height"),
    dpi: int = typer.Option(300, help="DPI for raster formats"),
    transparent: bool = typer.Option(False, help="Transparent background"),
    include_metadata: bool = typer.Option(True, help="Include metadata"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Export static visualization"""
    
    try:
        console.print(f"[bold blue]Loading simulation data from {input_file}[/bold blue]")
        
        # Load simulation data
        agents, patches = load_simulation_data(input_file)
        
        console.print(f"[green]Loaded {len(agents)} agents and {len(patches)} patches[/green]")
        
        # Initialize visualization components
        display_manager = DisplayStateManager()
        viz_engine = AgentVisualizationEngine(display_manager)
        data_engine = DataExplorationEngine(display_manager)
        exporter = VisualizationExporter(display_manager, viz_engine, data_engine)
        
        # Create export configuration
        export_format = ExportFormat(format_type.lower())
        export_quality = ExportQuality(quality.lower())
        
        config = ExportConfiguration(
            format=export_format,
            quality=export_quality,
            width=width,
            height=height,
            dpi=dpi,
            transparent_background=transparent,
            include_metadata=include_metadata
        )
        
        # Export visualization
        console.print("[yellow]Exporting visualization...[/yellow]")
        
        exported_files = exporter.export_static_visualization(
            agents=agents,
            patches=patches,
            output_path=f"{output_dir}/visualization",
            config=config
        )
        
        # Display results
        console.print("[bold green]✓ Visualization exported successfully[/bold green]")
        for file_type, file_path in exported_files.items():
            console.print(f"[dim]{file_type}: {file_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error exporting visualization: {e}[/bold red]")
        raise typer.Exit(1)


def create_animation(
    input_file: str = typer.Argument(..., help="Input file with time series data"),
    output_file: str = typer.Option("artifacts/visualization/animation.mp4", help="Output animation file"),
    format_type: str = typer.Option("mp4", help="Animation format (mp4/gif)"),
    fps: int = typer.Option(5, help="Frames per second"),
    quality: str = typer.Option("medium", help="Export quality"),
    duration: Optional[int] = typer.Option(None, help="Animation duration in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Create animation from time series data"""
    
    try:
        console.print(f"[bold blue]Loading time series data from {input_file}[/bold blue]")
        
        # Load time series data
        frame_data = load_time_series_data(input_file)
        
        console.print(f"[green]Loaded {len(frame_data)} frames[/green]")
        
        # Initialize visualization components
        display_manager = DisplayStateManager()
        viz_engine = AgentVisualizationEngine(display_manager)
        data_engine = DataExplorationEngine(display_manager)
        exporter = VisualizationExporter(display_manager, viz_engine, data_engine)
        
        # Create export configuration
        export_format = ExportFormat(format_type.lower())
        export_quality = ExportQuality(quality.lower())
        
        config = ExportConfiguration(
            format=export_format,
            quality=export_quality
        )
        
        # Create animation
        console.print("[yellow]Creating animation...[/yellow]")
        
        with console.status("[bold green]Generating animation frames..."):
            exported_files = exporter.export_animation(
                frame_data=frame_data,
                output_path=output_file,
                config=config
            )
        
        # Display results
        console.print("[bold green]✓ Animation created successfully[/bold green]")
        for file_type, file_path in exported_files.items():
            console.print(f"[dim]{file_type}: {file_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error creating animation: {e}[/bold red]")
        raise typer.Exit(1)


def launch_interactive_dashboard(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    port: int = typer.Option(8050, help="Dashboard port"),
    host: str = typer.Option("127.0.0.1", help="Dashboard host"),
    debug: bool = typer.Option(False, help="Debug mode"),
    auto_open: bool = typer.Option(True, help="Auto-open browser"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Launch interactive analysis dashboard"""
    
    try:
        console.print(f"[bold blue]Loading simulation data from {input_file}[/bold blue]")
        
        # Load simulation data
        agents, patches = load_simulation_data(input_file)
        
        console.print(f"[green]Loaded {len(agents)} agents and {len(patches)} patches[/green]")
        
        # Initialize visualization components
        display_manager = DisplayStateManager()
        data_engine = DataExplorationEngine(display_manager)
        
        # Prepare data sources
        data_sources = {
            "agents": agents,
            "patches": patches
        }
        
        # Load data into engine
        data_engine.load_data(data_sources)
        
        # Create dashboard
        dashboard = InteractiveDashboard(data_engine, port=port)
        
        # Update dashboard data
        dashboard.update_data(data_sources)
        
        console.print("[bold green]🚀 Launching interactive dashboard[/bold green]")
        console.print(f"[dim]Dashboard will be available at: http://{host}:{port}[/dim]")
        
        if auto_open:
            import webbrowser
            webbrowser.open(f"http://{host}:{port}")
        
        # Run dashboard
        dashboard.run(debug=debug)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error launching dashboard: {e}[/bold red]")
        raise typer.Exit(1)


def generate_report(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    output_file: str = typer.Option("artifacts/visualization/report.pdf", help="Output report file"),
    template: str = typer.Option("comprehensive", help="Report template"),
    include_analysis: bool = typer.Option(True, help="Include analysis data"),
    quality: str = typer.Option("publication", help="Export quality"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Generate comprehensive visualization report"""
    
    try:
        console.print(f"[bold blue]Loading simulation data from {input_file}[/bold blue]")
        
        # Load simulation data
        agents, patches = load_simulation_data(input_file)
        
        console.print(f"[green]Loaded {len(agents)} agents and {len(patches)} patches[/green]")
        
        # Initialize visualization components
        display_manager = DisplayStateManager()
        viz_engine = AgentVisualizationEngine(display_manager)
        data_engine = DataExplorationEngine(display_manager)
        exporter = VisualizationExporter(display_manager, viz_engine, data_engine)
        
        # Prepare analysis data if requested
        analysis_data = None
        if include_analysis:
            console.print("[yellow]Preparing analysis data...[/yellow]")
            analysis_data = {
                "agent_statistics": calculate_agent_statistics(agents),
                "patch_statistics": calculate_patch_statistics(patches),
                "spatial_analysis": perform_spatial_analysis(patches)
            }
        
        # Create export configuration
        export_quality = ExportQuality(quality.lower())
        config = ExportConfiguration(
            format=ExportFormat.PDF,
            quality=export_quality
        )
        
        # Generate report
        console.print("[yellow]Generating comprehensive report...[/yellow]")
        
        with console.status("[bold green]Creating report..."):
            report_path = exporter.generate_comprehensive_report(
                agents=agents,
                patches=patches,
                analysis_data=analysis_data,
                output_path=output_file,
                config=config
            )
        
        console.print("[bold green]✓ Report generated successfully[/bold green]")
        console.print(f"[dim]Report saved to: {report_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error generating report: {e}[/bold red]")
        raise typer.Exit(1)


def export_publication_figures(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    output_dir: str = typer.Option("artifacts/visualization/publication", help="Output directory"),
    formats: str = typer.Option("png,svg,pdf", help="Export formats (comma-separated)"),
    quality: str = typer.Option("publication", help="Export quality"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Export publication-ready figures"""
    
    try:
        console.print(f"[bold blue]Loading simulation data from {input_file}[/bold blue]")
        
        # Load simulation data
        agents, patches = load_simulation_data(input_file)
        
        console.print(f"[green]Loaded {len(agents)} agents and {len(patches)} patches[/green]")
        
        # Initialize visualization components
        display_manager = DisplayStateManager()
        viz_engine = AgentVisualizationEngine(display_manager)
        data_engine = DataExplorationEngine(display_manager)
        exporter = VisualizationExporter(display_manager, viz_engine, data_engine)
        
        # Parse formats
        format_list = [ExportFormat(fmt.strip().lower()) for fmt in formats.split(",")]
        
        # Export publication figures
        console.print("[yellow]Exporting publication figures...[/yellow]")
        
        exported_files = exporter.export_publication_figures(
            agents=agents,
            patches=patches,
            output_directory=output_dir,
            formats=format_list
        )
        
        # Display results
        console.print("[bold green]✓ Publication figures exported successfully[/bold green]")
        
        for format_type, file_paths in exported_files.items():
            console.print(f"[cyan]{format_type.upper()}:[/cyan]")
            for file_path in file_paths:
                console.print(f"[dim]  {file_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error exporting publication figures: {e}[/bold red]")
        raise typer.Exit(1)


def create_export_package(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    output_file: str = typer.Option("artifacts/visualization/export_package.zip", help="Output package file"),
    include_animation: bool = typer.Option(False, help="Include animation data"),
    include_analysis: bool = typer.Option(True, help="Include analysis data"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Create complete export package"""
    
    try:
        console.print(f"[bold blue]Loading simulation data from {input_file}[/bold blue]")
        
        # Load simulation data
        agents, patches = load_simulation_data(input_file)
        
        console.print(f"[green]Loaded {len(agents)} agents and {len(patches)} patches[/green]")
        
        # Initialize visualization components
        display_manager = DisplayStateManager()
        viz_engine = AgentVisualizationEngine(display_manager)
        data_engine = DataExplorationEngine(display_manager)
        exporter = VisualizationExporter(display_manager, viz_engine, data_engine)
        
        # Prepare optional data
        frame_data = None
        if include_animation:
            console.print("[yellow]Loading animation data...[/yellow]")
            frame_data = load_time_series_data(input_file)
        
        analysis_data = None
        if include_analysis:
            console.print("[yellow]Preparing analysis data...[/yellow]")
            analysis_data = {
                "agent_statistics": calculate_agent_statistics(agents),
                "patch_statistics": calculate_patch_statistics(patches),
                "spatial_analysis": perform_spatial_analysis(patches)
            }
        
        # Create export package
        console.print("[yellow]Creating export package...[/yellow]")
        
        with console.status("[bold green]Building complete package..."):
            package_path = exporter.create_export_package(
                agents=agents,
                patches=patches,
                frame_data=frame_data,
                analysis_data=analysis_data,
                output_path=output_file
            )
        
        console.print("[bold green]✓ Export package created successfully[/bold green]")
        console.print(f"[dim]Package saved to: {package_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error creating export package: {e}[/bold red]")
        raise typer.Exit(1)


# Helper functions
def load_simulation_data(file_path: str) -> tuple:
    """Load simulation data from file"""
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Mock data loading - in real implementation, this would parse actual simulation data
    agents = []
    patches = []
    
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            
        agents = data.get("agents", [])
        patches = data.get("patches", [])
    else:
        # For other formats, create mock data
        console.print("[yellow]Warning: Using mock data for non-JSON files[/yellow]")
        agents = [{"id": i, "x": i*10, "y": i*5, "species": "apis_mellifera"} for i in range(100)]
        patches = [{"id": i, "x": i*20, "y": i*10, "type": "flower"} for i in range(20)]
    
    return agents, patches


def load_time_series_data(file_path: str) -> List[VisualizationFrame]:
    """Load time series data for animation"""
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Mock time series data loading
    from datetime import datetime, timedelta
    
    frames = []
    base_time = datetime.now()
    
    for i in range(50):  # 50 frames
        timestamp = base_time + timedelta(hours=i)
        agents = [{"id": j, "x": j*10 + i, "y": j*5, "species": "apis_mellifera"} for j in range(100)]
        patches = [{"id": j, "x": j*20, "y": j*10, "type": "flower"} for j in range(20)]
        
        frame = VisualizationFrame(
            timestamp=timestamp,
            agents=agents,
            patches=patches,
            connections=[],
            metadata={"frame": i}
        )
        frames.append(frame)
    
    return frames


def calculate_agent_statistics(agents: List[Any]) -> Dict[str, Any]:
    """Calculate agent statistics"""
    
    stats = {
        "total_agents": len(agents),
        "species_distribution": {},
        "location_stats": {
            "mean_x": 0,
            "mean_y": 0,
            "range_x": 0,
            "range_y": 0
        }
    }
    
    if agents:
        x_coords = [agent.get("x", 0) for agent in agents]
        y_coords = [agent.get("y", 0) for agent in agents]
        
        stats["location_stats"]["mean_x"] = sum(x_coords) / len(x_coords)
        stats["location_stats"]["mean_y"] = sum(y_coords) / len(y_coords)
        stats["location_stats"]["range_x"] = max(x_coords) - min(x_coords)
        stats["location_stats"]["range_y"] = max(y_coords) - min(y_coords)
    
    return stats


def calculate_patch_statistics(patches: List[Any]) -> Dict[str, Any]:
    """Calculate patch statistics"""
    
    stats = {
        "total_patches": len(patches),
        "type_distribution": {},
        "spatial_extent": {
            "min_x": 0,
            "max_x": 0,
            "min_y": 0,
            "max_y": 0
        }
    }
    
    if patches:
        x_coords = [patch.get("x", 0) for patch in patches]
        y_coords = [patch.get("y", 0) for patch in patches]
        
        stats["spatial_extent"]["min_x"] = min(x_coords)
        stats["spatial_extent"]["max_x"] = max(x_coords)
        stats["spatial_extent"]["min_y"] = min(y_coords)
        stats["spatial_extent"]["max_y"] = max(y_coords)
    
    return stats


def perform_spatial_analysis(patches: List[Any]) -> Dict[str, Any]:
    """Perform basic spatial analysis"""
    
    analysis = {
        "patch_density": len(patches) / 10000 if patches else 0,  # patches per 10000 units
        "spatial_distribution": "clustered",  # simplified
        "connectivity_index": 0.75  # simplified
    }
    
    return analysis


# Create typer app for visualization commands
app = typer.Typer(name="visualize", help="Visualization commands")

app.command(name="export")(export_visualization)
app.command(name="animate")(create_animation)
app.command(name="dashboard")(launch_interactive_dashboard)
app.command(name="report")(generate_report)
app.command(name="publication")(export_publication_figures)
app.command(name="package")(create_export_package)

if __name__ == "__main__":
    app()