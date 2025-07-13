"""
Spatial Analysis CLI Commands
============================

Command-line interface for spatial analysis operations in BSTEW.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path
import json
from typing import Optional, Dict, Any, List

from ...core.spatial_analysis import (
    PatchConnectivityAnalyzer, 
    SpatialPatch,
    PatchType
)

console = Console()

def analyze_spatial_data(
    input_file: str = typer.Argument(..., help="Input file with patch data (JSON/CSV)"),
    output_dir: str = typer.Option("artifacts/spatial_analysis", help="Output directory"),
    connectivity_threshold: float = typer.Option(0.1, help="Connectivity threshold"),
    cluster_method: str = typer.Option("dbscan", help="Clustering method (dbscan/kmeans)"),
    min_cluster_size: int = typer.Option(5, help="Minimum cluster size"),
    export_graphs: bool = typer.Option(True, help="Export network graphs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Run comprehensive spatial connectivity analysis"""
    
    try:
        console.print(f"[bold blue]Loading spatial data from {input_file}[/bold blue]")
        
        # Load patch data
        patches = load_patch_data(input_file)
        
        console.print(f"[green]Loaded {len(patches)} patches[/green]")
        
        # Initialize analyzer
        analyzer = PatchConnectivityAnalyzer()
        
        # Add patches with progress tracking
        for patch in track(patches, description="Adding patches..."):
            analyzer.add_patch(patch)
        
        # Build spatial index
        console.print("[yellow]Building spatial index...[/yellow]")
        analyzer.build_spatial_index()
        
        # Calculate connectivity
        console.print("[yellow]Calculating connectivity indices...[/yellow]")
        analyzer.calculate_connectivity_indices()
        
        # Identify clusters
        console.print("[yellow]Identifying patch clusters...[/yellow]")
        cluster_results = analyzer.identify_patch_clusters(
            method=cluster_method,
            min_cluster_size=min_cluster_size,
            connectivity_threshold=connectivity_threshold
        )
        
        # Calculate landscape metrics
        console.print("[yellow]Calculating landscape metrics...[/yellow]")
        landscape_metrics = analyzer.calculate_landscape_metrics()
        
        # Export results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[yellow]Exporting results to {output_path}...[/yellow]")
        analyzer.export_connectivity_data(str(output_path))
        
        # Export analysis summary
        summary = {
            "total_patches": len(patches),
            "clusters_found": len(cluster_results.get("clusters", [])),
            "landscape_metrics": landscape_metrics,
            "connectivity_threshold": connectivity_threshold,
            "cluster_method": cluster_method
        }
        
        with open(output_path / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Display results
        display_spatial_results(summary, cluster_results, landscape_metrics)
        
        console.print("[bold green]✓ Spatial analysis completed successfully[/bold green]")
        console.print(f"[dim]Results saved to: {output_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error during spatial analysis: {e}[/bold red]")
        raise typer.Exit(1)


def query_spatial_patches(
    input_file: str = typer.Argument(..., help="Input file with patch data"),
    center_x: float = typer.Option(..., help="Query center X coordinate"),
    center_y: float = typer.Option(..., help="Query center Y coordinate"),
    radius: float = typer.Option(100.0, help="Query radius"),
    patch_type: Optional[str] = typer.Option(None, help="Filter by patch type"),
    min_quality: Optional[float] = typer.Option(None, help="Minimum habitat quality"),
    max_results: int = typer.Option(50, help="Maximum results to return"),
    output_file: Optional[str] = typer.Option(None, help="Output file for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Query patches within spatial criteria"""
    
    try:
        console.print(f"[bold blue]Loading spatial data from {input_file}[/bold blue]")
        
        # Load patch data
        patches = load_patch_data(input_file)
        patch_dict = {patch.patch_id: patch for patch in patches}
        
        console.print(f"[green]Loaded {len(patches)} patches[/green]")
        
        # Initialize patch connectivity analyzer
        query_engine = PatchConnectivityAnalyzer(patch_dict)
        
        # Build query filters
        filters = {}
        if patch_type:
            filters["patch_type"] = patch_type
        if min_quality is not None:
            filters["habitat_quality"] = {"min": min_quality}
        
        # Execute query
        console.print(f"[yellow]Querying patches near ({center_x}, {center_y}) within radius {radius}...[/yellow]")
        
        results = query_engine.multi_criteria_search(
            center_point=(center_x, center_y),
            radius=radius,
            filters=filters,
            max_results=max_results
        )
        
        # Display results
        if results:
            display_query_results(results, patch_dict)
            
            # Export results if requested
            if output_file:
                export_query_results(results, patch_dict, output_file)
                console.print(f"[dim]Results exported to: {output_file}[/dim]")
        else:
            console.print("[yellow]No patches found matching the criteria[/yellow]")
        
        console.print(f"[bold green]✓ Query completed: {len(results)} patches found[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error during spatial query: {e}[/bold red]")
        raise typer.Exit(1)


def calculate_landscape_metrics(
    input_file: str = typer.Argument(..., help="Input file with patch data"),
    output_file: str = typer.Option("artifacts/spatial_analysis/landscape_metrics.json", help="Output file"),
    export_csv: bool = typer.Option(True, help="Also export as CSV"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Calculate landscape-level metrics"""
    
    try:
        console.print(f"[bold blue]Loading spatial data from {input_file}[/bold blue]")
        
        # Load patch data
        patches = load_patch_data(input_file)
        
        console.print(f"[green]Loaded {len(patches)} patches[/green]")
        
        # Initialize analyzer
        analyzer = PatchConnectivityAnalyzer()
        
        # Add patches
        for patch in track(patches, description="Processing patches..."):
            analyzer.add_patch(patch)
        
        # Build spatial index
        analyzer.build_spatial_index()
        
        # Calculate metrics
        console.print("[yellow]Calculating landscape metrics...[/yellow]")
        metrics = analyzer.calculate_landscape_metrics()
        
        # Export results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        if export_csv:
            csv_path = output_path.with_suffix(".csv")
            import pandas as pd
            pd.DataFrame([metrics]).to_csv(csv_path, index=False)
            console.print(f"[dim]CSV exported to: {csv_path}[/dim]")
        
        # Display metrics
        display_landscape_metrics(metrics)
        
        console.print("[bold green]✓ Landscape metrics calculated successfully[/bold green]")
        console.print(f"[dim]Results saved to: {output_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error calculating landscape metrics: {e}[/bold red]")
        raise typer.Exit(1)


def export_spatial_network(
    input_file: str = typer.Argument(..., help="Input file with patch data"),
    output_dir: str = typer.Option("artifacts/spatial_analysis/network", help="Output directory"),
    connectivity_threshold: float = typer.Option(0.1, help="Connectivity threshold"),
    format_type: str = typer.Option("graphml", help="Export format (graphml/gexf/json)"),
    include_attributes: bool = typer.Option(True, help="Include node attributes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Export spatial network graph"""
    
    try:
        console.print(f"[bold blue]Loading spatial data from {input_file}[/bold blue]")
        
        # Load patch data
        patches = load_patch_data(input_file)
        
        console.print(f"[green]Loaded {len(patches)} patches[/green]")
        
        # Initialize analyzer
        analyzer = PatchConnectivityAnalyzer()
        
        # Add patches
        for patch in track(patches, description="Building network..."):
            analyzer.add_patch(patch)
        
        # Build spatial index and connectivity
        analyzer.build_spatial_index()
        analyzer.calculate_connectivity_indices()
        
        # Build network graph
        console.print("[yellow]Building network graph...[/yellow]")
        graph = analyzer.build_network_graph(connectivity_threshold)
        
        # Export network
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format_type == "graphml":
            import networkx as nx
            nx.write_graphml(graph, output_path / "spatial_network.graphml")
        elif format_type == "gexf":
            import networkx as nx
            nx.write_gexf(graph, output_path / "spatial_network.gexf")
        elif format_type == "json":
            import networkx as nx
            from networkx.readwrite import json_graph
            graph_data = json_graph.node_link_data(graph)
            with open(output_path / "spatial_network.json", "w") as f:
                json.dump(graph_data, f, indent=2)
        
        # Export network statistics
        network_stats = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2),
            "connectivity_threshold": connectivity_threshold
        }
        
        with open(output_path / "network_stats.json", "w") as f:
            json.dump(network_stats, f, indent=2)
        
        console.print("[bold green]✓ Network exported successfully[/bold green]")
        console.print(f"[dim]Network saved to: {output_path}[/dim]")
        console.print(f"[dim]Nodes: {network_stats['nodes']}, Edges: {network_stats['edges']}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error exporting network: {e}[/bold red]")
        raise typer.Exit(1)


# Helper functions
def load_patch_data(file_path: str) -> List[SpatialPatch]:
    """Load patch data from file"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    patches = []
    
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            
        if isinstance(data, list):
            for i, patch_data in enumerate(data):
                patch = SpatialPatch(
                    patch_id=patch_data.get("patch_id", i),
                    x=patch_data.get("x", 0),
                    y=patch_data.get("y", 0),
                    patch_type=PatchType(patch_data.get("patch_type", "open")),
                    area=patch_data.get("area", 1.0),
                    habitat_quality=patch_data.get("habitat_quality", 0.5),
                    resource_density=patch_data.get("resource_density", 0.0),
                    carrying_capacity=patch_data.get("carrying_capacity", 100)
                )
                patches.append(patch)
    else:
        # Try CSV format
        import pandas as pd
        df = pd.read_csv(path)
        
        for i, row in df.iterrows():
            patch = SpatialPatch(
                patch_id=row.get("patch_id", i),
                x=row.get("x", 0),
                y=row.get("y", 0),
                patch_type=PatchType(row.get("patch_type", "open")),
                area=row.get("area", 1.0),
                habitat_quality=row.get("habitat_quality", 0.5),
                resource_density=row.get("resource_density", 0.0),
                carrying_capacity=row.get("carrying_capacity", 100)
            )
            patches.append(patch)
    
    return patches


def display_spatial_results(summary: Dict[str, Any], cluster_results: Dict[str, Any], landscape_metrics: Dict[str, Any]) -> None:
    """Display spatial analysis results"""
    
    # Summary table
    table = Table(title="Spatial Analysis Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Patches", str(summary["total_patches"]))
    table.add_row("Clusters Found", str(summary["clusters_found"]))
    table.add_row("Connectivity Threshold", f"{summary['connectivity_threshold']:.3f}")
    table.add_row("Cluster Method", summary["cluster_method"])
    
    console.print(table)
    
    # Landscape metrics table
    if landscape_metrics:
        metrics_table = Table(title="Landscape Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")
        
        for metric, value in landscape_metrics.items():
            if isinstance(value, float):
                metrics_table.add_row(metric.replace("_", " ").title(), f"{value:.4f}")
            else:
                metrics_table.add_row(metric.replace("_", " ").title(), str(value))
        
        console.print(metrics_table)


def display_query_results(results: List[int], patch_dict: Dict[int, SpatialPatch]) -> None:
    """Display spatial query results"""
    
    table = Table(title="Spatial Query Results")
    table.add_column("Patch ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("X", style="magenta")
    table.add_column("Y", style="magenta")
    table.add_column("Quality", style="yellow")
    table.add_column("Area", style="blue")
    
    for patch_id in results[:20]:  # Show first 20 results
        patch = patch_dict[patch_id]
        table.add_row(
            str(patch.patch_id),
            patch.patch_type.value,
            f"{patch.x:.2f}",
            f"{patch.y:.2f}",
            f"{patch.habitat_quality:.3f}",
            f"{patch.area:.2f}"
        )
    
    console.print(table)
    
    if len(results) > 20:
        console.print(f"[dim]... and {len(results) - 20} more patches[/dim]")


def display_landscape_metrics(metrics: Dict[str, Any]) -> None:
    """Display landscape metrics"""
    
    table = Table(title="Landscape Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="dim")
    
    metric_descriptions = {
        "total_area": "Total landscape area",
        "patch_density": "Number of patches per unit area",
        "mean_patch_size": "Average patch size",
        "landscape_diversity": "Shannon diversity index",
        "connectivity_index": "Overall connectivity measure",
        "fragmentation_index": "Landscape fragmentation level"
    }
    
    for metric, value in metrics.items():
        description = metric_descriptions.get(metric, "")
        if isinstance(value, float):
            table.add_row(metric.replace("_", " ").title(), f"{value:.4f}", description)
        else:
            table.add_row(metric.replace("_", " ").title(), str(value), description)
    
    console.print(table)


def export_query_results(results: List[int], patch_dict: Dict[int, SpatialPatch], output_file: str) -> None:
    """Export query results to file"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    export_data = []
    for patch_id in results:
        patch = patch_dict[patch_id]
        export_data.append({
            "patch_id": patch.patch_id,
            "x": patch.x,
            "y": patch.y,
            "patch_type": patch.patch_type.value,
            "area": patch.area,
            "habitat_quality": patch.habitat_quality,
            "resource_density": patch.resource_density,
            "carrying_capacity": patch.carrying_capacity
        })
    
    # Export based on file extension
    if output_path.suffix.lower() == ".json":
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
    else:
        # Export as CSV
        import pandas as pd
        pd.DataFrame(export_data).to_csv(output_path, index=False)


# Create typer app for spatial commands
app = typer.Typer(name="spatial", help="Spatial analysis commands")

app.command(name="analyze")(analyze_spatial_data)
app.command(name="query")(query_spatial_patches)
app.command(name="metrics")(calculate_landscape_metrics)
app.command(name="export-network")(export_spatial_network)

if __name__ == "__main__":
    app()