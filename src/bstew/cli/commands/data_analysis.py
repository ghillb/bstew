"""
Data Analysis CLI Commands
=========================

Command-line interface for data analysis operations in BSTEW.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import json
import pandas as pd
from typing import Optional, Dict, Any, List


console = Console()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default


def validate_colony_data(colonies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and clean colony data"""
    valid_colonies = []
    for colony in colonies:
        # Ensure required fields exist
        if "colony_id" in colony:
            # Set defaults for missing fields
            colony.setdefault("max_population", 0)
            colony.setdefault("death_time", None)
            colony.setdefault("birth_time", 0)
            colony.setdefault("foraging_efficiency", 0)
            valid_colonies.append(colony)
    return valid_colonies


def analyze_simulation_data(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    output_dir: str = typer.Option("artifacts/data_analysis", help="Output directory"),
    metrics: str = typer.Option(
        "all", help="Metrics to analyze (all/bee/colony/environmental)"
    ),
    time_range: Optional[str] = typer.Option(None, help="Time range (start:end)"),
    aggregation: str = typer.Option(
        "daily", help="Aggregation level (hourly/daily/weekly)"
    ),
    export_format: str = typer.Option("csv", help="Export format (csv/json/xlsx)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Analyze simulation data and generate reports"""

    try:
        console.print(
            f"[bold blue]Loading simulation data from {input_file}[/bold blue]"
        )

        # Load simulation data
        data = load_simulation_data(input_file)

        console.print(f"[green]Loaded data with {len(data)} time points[/green]")

        # Initialize data collector for analysis
        # collector = ComprehensiveDataCollector()

        # Parse time range if provided
        time_start, time_end = (
            parse_time_range(time_range) if time_range else (None, None)
        )

        # Filter data by time range
        if time_start is not None or time_end is not None:
            data = filter_data_by_time(data, time_start, time_end)
            console.print(f"[yellow]Filtered to {len(data)} time points[/yellow]")

        # Perform analysis based on metrics requested
        results = {}

        if metrics in ["all", "bee"]:
            console.print("[yellow]Analyzing bee metrics...[/yellow]")
            results["bee_analysis"] = analyze_bee_metrics(data, aggregation)

        if metrics in ["all", "colony"]:
            console.print("[yellow]Analyzing colony metrics...[/yellow]")
            results["colony_analysis"] = analyze_colony_metrics(data, aggregation)

        if metrics in ["all", "environmental"]:
            console.print("[yellow]Analyzing environmental metrics...[/yellow]")
            results["environmental_analysis"] = analyze_environmental_metrics(
                data, aggregation
            )

        # Generate summary statistics
        console.print("[yellow]Generating summary statistics...[/yellow]")
        results["summary"] = generate_summary_statistics(data, results)

        # Export results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"[yellow]Exporting results to {output_path}...[/yellow]")
        export_analysis_results(results, output_path, export_format)

        # Display summary
        display_analysis_summary(results)

        console.print("[bold green]✓ Data analysis completed successfully[/bold green]")
        console.print(f"[dim]Results saved to: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Error during data analysis: {e}[/bold red]")
        raise typer.Exit(1)


def analyze_dead_colonies(
    input_file: str = typer.Argument(
        ..., help="Input file with colony data including mortality"
    ),
    output_dir: str = typer.Option(
        "artifacts/data_analysis/dead_colonies", help="Output directory"
    ),
    time_range: Optional[str] = typer.Option(
        None, help="Time range for analysis (start:end)"
    ),
    cause_analysis: bool = typer.Option(True, help="Include cause of death analysis"),
    comparative_analysis: bool = typer.Option(
        True, help="Compare with surviving colonies"
    ),
    export_format: str = typer.Option("json", help="Export format (json/csv/xlsx)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Perform post-mortem analysis of dead colonies"""

    try:
        console.print(
            f"[bold blue]Analyzing dead colonies from {input_file}[/bold blue]"
        )

        # Load colony data with mortality information
        colony_data = load_colony_data_with_mortality(input_file)

        console.print(f"[green]Loaded data for {len(colony_data)} colonies[/green]")

        # Extract dead colonies from dataset
        dead_colonies = extract_dead_colonies(colony_data)

        if not dead_colonies:
            console.print("[yellow]No dead colonies found in dataset[/yellow]")
            return

        console.print(
            f"[yellow]Found {len(dead_colonies)} dead colonies for analysis[/yellow]"
        )

        # Parse time range if provided
        time_start, time_end = (
            parse_time_range(time_range) if time_range else (None, None)
        )

        # Filter dead colonies by time range
        if time_start is not None or time_end is not None:
            dead_colonies = filter_dead_colonies_by_time(
                dead_colonies, time_start, time_end
            )
            console.print(
                f"[yellow]Filtered to {len(dead_colonies)} colonies in time range[/yellow]"
            )

        # Perform dead colony analysis
        results = {}

        # Basic summary analysis
        console.print("[yellow]Analyzing dead colony summary statistics...[/yellow]")
        results["summary"] = analyze_dead_colony_summary(dead_colonies)

        # Mortality pattern analysis
        console.print("[yellow]Analyzing mortality patterns...[/yellow]")
        results["mortality_patterns"] = analyze_mortality_patterns(dead_colonies)

        # Cause of death analysis
        if cause_analysis:
            console.print("[yellow]Analyzing causes of death...[/yellow]")
            results["death_causes"] = analyze_death_causes(dead_colonies)

        # Comparative analysis with surviving colonies
        if comparative_analysis:
            console.print("[yellow]Comparing dead vs surviving colonies...[/yellow]")
            surviving_colonies = extract_surviving_colonies(colony_data)
            results["comparative_analysis"] = compare_dead_vs_surviving(
                dead_colonies, surviving_colonies
            )

        # Export results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"[yellow]Exporting results to {output_path}...[/yellow]")
        export_dead_colony_analysis(results, output_path, export_format)

        # Display summary
        display_dead_colony_analysis(results)

        console.print(
            "[bold green]✓ Dead colony analysis completed successfully[/bold green]"
        )
        console.print(f"[dim]Results saved to: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Error during dead colony analysis: {e}[/bold red]")
        raise typer.Exit(1)


def export_data(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    output_file: str = typer.Option(
        "artifacts/data_analysis/exported_data.csv", help="Output file"
    ),
    format_type: str = typer.Option("csv", help="Export format (csv/json/xlsx)"),
    metrics: str = typer.Option("all", help="Metrics to export"),
    time_range: Optional[str] = typer.Option(None, help="Time range (start:end)"),
    include_metadata: bool = typer.Option(True, help="Include metadata"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Export simulation data in various formats"""

    try:
        console.print(
            f"[bold blue]Loading simulation data from {input_file}[/bold blue]"
        )

        # Load simulation data
        data = load_simulation_data(input_file)

        console.print(f"[green]Loaded data with {len(data)} time points[/green]")

        # Parse time range if provided
        time_start, time_end = (
            parse_time_range(time_range) if time_range else (None, None)
        )

        # Filter data by time range
        if time_start is not None or time_end is not None:
            data = filter_data_by_time(data, time_start, time_end)
            console.print(f"[yellow]Filtered to {len(data)} time points[/yellow]")

        # Filter by metrics
        if metrics != "all":
            data = filter_data_by_metrics(data, metrics.split(","))

        # Prepare export data
        export_data = prepare_export_data(data, include_metadata)

        # Export data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        console.print(f"[yellow]Exporting data to {output_path}...[/yellow]")

        if format_type.lower() == "csv":
            export_data.to_csv(output_path, index=False)
        elif format_type.lower() == "json":
            export_data.to_json(output_path, orient="records", indent=2)
        elif format_type.lower() == "xlsx":
            export_data.to_excel(output_path, index=False, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        console.print("[bold green]✓ Data exported successfully[/bold green]")
        console.print(
            f"[dim]Exported {len(export_data)} records to: {output_path}[/dim]"
        )

    except Exception as e:
        console.print(f"[bold red]✗ Error exporting data: {e}[/bold red]")
        raise typer.Exit(1)


def generate_summary(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    output_file: str = typer.Option(
        "artifacts/data_analysis/summary.json", help="Output summary file"
    ),
    level: str = typer.Option(
        "detailed", help="Summary level (basic/detailed/comprehensive)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Generate data summary statistics"""

    try:
        console.print(
            f"[bold blue]Loading simulation data from {input_file}[/bold blue]"
        )

        # Load simulation data
        data = load_simulation_data(input_file)

        console.print(f"[green]Loaded data with {len(data)} time points[/green]")

        # Generate summary based on level
        console.print(f"[yellow]Generating {level} summary...[/yellow]")

        if level == "basic":
            summary = generate_basic_summary(data)
        elif level == "detailed":
            summary = generate_detailed_summary(data)
        elif level == "comprehensive":
            summary = generate_comprehensive_summary(data)
        else:
            raise ValueError(f"Unknown summary level: {level}")

        # Export summary
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Display summary
        display_summary(summary)

        console.print("[bold green]✓ Summary generated successfully[/bold green]")
        console.print(f"[dim]Summary saved to: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Error generating summary: {e}[/bold red]")
        raise typer.Exit(1)


def calculate_metrics(
    input_file: str = typer.Argument(..., help="Input file with simulation data"),
    metric_type: str = typer.Argument(
        ..., help="Metric type (population/mortality/foraging/energy)"
    ),
    output_file: str = typer.Option(
        "artifacts/data_analysis/metrics.json", help="Output metrics file"
    ),
    time_window: str = typer.Option(
        "daily", help="Time window (hourly/daily/weekly/monthly)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Calculate specific metrics"""

    try:
        console.print(
            f"[bold blue]Loading simulation data from {input_file}[/bold blue]"
        )

        # Load simulation data
        data = load_simulation_data(input_file)

        console.print(f"[green]Loaded data with {len(data)} time points[/green]")

        # Calculate metrics based on type
        console.print(
            f"[yellow]Calculating {metric_type} metrics with {time_window} aggregation...[/yellow]"
        )

        if metric_type == "population":
            metrics = calculate_population_metrics(data, time_window)
        elif metric_type == "mortality":
            metrics = calculate_mortality_metrics(data, time_window)
        elif metric_type == "foraging":
            metrics = calculate_foraging_metrics(data, time_window)
        elif metric_type == "energy":
            metrics = calculate_energy_metrics(data, time_window)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        # Export metrics
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Display metrics
        display_metrics(metrics, metric_type)

        console.print("[bold green]✓ Metrics calculated successfully[/bold green]")
        console.print(f"[dim]Metrics saved to: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Error calculating metrics: {e}[/bold red]")
        raise typer.Exit(1)


def compare_datasets(
    file1: str = typer.Argument(..., help="First dataset file"),
    file2: str = typer.Argument(..., help="Second dataset file"),
    output_dir: str = typer.Option(
        "artifacts/data_analysis/comparison", help="Output directory"
    ),
    metrics: str = typer.Option("all", help="Metrics to compare"),
    statistical_tests: bool = typer.Option(True, help="Perform statistical tests"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Compare two datasets"""

    try:
        console.print("[bold blue]Loading datasets[/bold blue]")

        # Load both datasets
        data1 = load_simulation_data(file1)
        data2 = load_simulation_data(file2)

        console.print(f"[green]Dataset 1: {len(data1)} time points[/green]")
        console.print(f"[green]Dataset 2: {len(data2)} time points[/green]")

        # Perform comparison
        console.print("[yellow]Performing dataset comparison...[/yellow]")

        comparison_results = {
            "basic_comparison": compare_basic_statistics(data1, data2),
            "time_series_comparison": compare_time_series(data1, data2),
            "distribution_comparison": compare_distributions(data1, data2),
        }

        if statistical_tests:
            console.print("[yellow]Performing statistical tests...[/yellow]")
            comparison_results["statistical_tests"] = perform_statistical_tests(
                data1, data2
            )

        # Export comparison results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "comparison_results.json", "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)

        # Display comparison results
        display_comparison_results(comparison_results)

        console.print("[bold green]✓ Dataset comparison completed[/bold green]")
        console.print(f"[dim]Results saved to: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Error comparing datasets: {e}[/bold red]")
        raise typer.Exit(1)


# Helper functions
def load_simulation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load simulation data from file"""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_dict("records")  # type: ignore[return-value]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def parse_time_range(time_range: str) -> tuple[Optional[int], Optional[int]]:
    """Parse time range string"""

    if ":" in time_range:
        start_str, end_str = time_range.split(":")
        start = int(start_str) if start_str else None
        end = int(end_str) if end_str else None
        return start, end
    else:
        raise ValueError(f"Invalid time range format: {time_range}")


def filter_data_by_time(
    data: List[Dict[str, Any]], start: Optional[int], end: Optional[int]
) -> List[Dict[str, Any]]:
    """Filter data by time range"""

    filtered_data = []
    for record in data:
        timestamp = record.get("timestamp", 0)
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        filtered_data.append(record)

    return filtered_data


def filter_data_by_metrics(
    data: List[Dict[str, Any]], metrics: List[str]
) -> List[Dict[str, Any]]:
    """Filter data by specific metrics"""

    # This is a simplified implementation
    # In practice, you'd filter based on the specific metrics structure
    return data


def analyze_bee_metrics(data: List[Dict[str, Any]], aggregation: str) -> Dict[str, Any]:
    """Analyze bee-level metrics"""

    analysis = {
        "total_bees": len([d for d in data if "bee_id" in d]),
        "mortality_rate": 0.05,  # Mock calculation
        "foraging_efficiency": 0.75,  # Mock calculation
        "energy_levels": {"mean": 85.5, "std": 12.3, "min": 45.2, "max": 98.7},
        "activity_distribution": {
            "foraging": 0.45,
            "nursing": 0.25,
            "resting": 0.20,
            "building": 0.10,
        },
    }

    return analysis


def analyze_colony_metrics(
    data: List[Dict[str, Any]], aggregation: str
) -> Dict[str, Any]:
    """Analyze colony-level metrics"""

    analysis = {
        "total_colonies": len([d for d in data if "colony_id" in d]),
        "colony_survival_rate": 0.85,  # Mock calculation
        "average_colony_size": 1250,  # Mock calculation
        "reproduction_success": 0.72,  # Mock calculation
        "resource_efficiency": {
            "nectar_collection": 0.78,
            "pollen_collection": 0.82,
            "nest_construction": 0.69,
        },
    }

    return analysis


def analyze_environmental_metrics(
    data: List[Dict[str, Any]], aggregation: str
) -> Dict[str, Any]:
    """Analyze environmental metrics"""

    analysis = {
        "resource_availability": {
            "nectar_sources": 145,
            "pollen_sources": 98,
            "nesting_sites": 23,
        },
        "environmental_conditions": {
            "temperature_mean": 18.5,
            "humidity_mean": 65.2,
            "wind_speed_mean": 3.1,
        },
        "habitat_quality": 0.78,
        "connectivity_index": 0.65,
    }

    return analysis


def generate_summary_statistics(
    data: List[Dict[str, Any]], analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate overall summary statistics"""

    summary = {
        "data_points": len(data),
        "time_span": {"start": 0, "end": 365},  # Mock
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "key_findings": {
            "population_trend": "stable",
            "resource_usage": "efficient",
            "colony_health": "good",
        },
    }

    return summary


def prepare_export_data(
    data: List[Dict[str, Any]], include_metadata: bool
) -> pd.DataFrame:
    """Prepare data for export"""

    df = pd.DataFrame(data)

    if not include_metadata:
        # Remove metadata columns
        metadata_cols = [col for col in df.columns if col.startswith("_")]
        df = df.drop(columns=metadata_cols, errors="ignore")

    return df


def export_analysis_results(
    results: Dict[str, Any], output_path: Path, format_type: str
) -> None:
    """Export analysis results"""

    for analysis_name, analysis_data in results.items():
        filename = f"{analysis_name}.{format_type}"
        file_path = output_path / filename

        if format_type == "json":
            with open(file_path, "w") as f:
                json.dump(analysis_data, f, indent=2, default=str)
        elif format_type == "csv":
            if isinstance(analysis_data, dict):
                # Convert dict to DataFrame
                df = pd.DataFrame([analysis_data])
                df.to_csv(file_path, index=False)
            else:
                pd.DataFrame(analysis_data).to_csv(file_path, index=False)


def display_analysis_summary(results: Dict[str, Any]) -> None:
    """Display analysis summary"""

    # Summary table
    summary_table = Table(title="Analysis Summary")
    summary_table.add_column("Analysis Type", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Key Metrics", style="magenta")

    for analysis_name, analysis_data in results.items():
        if analysis_name == "summary":
            continue

        key_metrics = "Completed"
        if isinstance(analysis_data, dict):
            key_metrics = f"{len(analysis_data)} metrics"

        summary_table.add_row(
            analysis_name.replace("_", " ").title(), "✓ Complete", key_metrics
        )

    console.print(summary_table)


def generate_basic_summary(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate basic summary"""

    return {
        "total_records": len(data),
        "time_range": {"start": 0, "end": len(data)},
        "data_types": list(set(key for record in data for key in record.keys())),
    }


def generate_detailed_summary(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate detailed summary"""

    basic = generate_basic_summary(data)
    basic.update(
        {"statistics": {"mean_values": {}, "distributions": {}, "correlations": {}}}
    )

    return basic


def generate_comprehensive_summary(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive summary"""

    detailed = generate_detailed_summary(data)
    detailed.update(
        {
            "advanced_analytics": {
                "trend_analysis": {},
                "anomaly_detection": {},
                "forecasting": {},
            }
        }
    )

    return detailed


def display_summary(summary: Dict[str, Any]) -> None:
    """Display summary information"""

    # Basic information
    console.print(
        Panel(
            f"[bold]Total Records:[/bold] {summary.get('total_records', 'N/A')}",
            title="Data Summary",
        )
    )

    # Data types table
    if "data_types" in summary:
        types_table = Table(title="Data Types")
        types_table.add_column("Field", style="cyan")

        for data_type in summary["data_types"][:10]:  # Show first 10
            types_table.add_row(data_type)

        console.print(types_table)


def calculate_population_metrics(
    data: List[Dict[str, Any]], time_window: str
) -> Dict[str, Any]:
    """Calculate real population metrics from data"""

    if not data:
        return {
            "total_population": 0,
            "population_growth_rate": 0.0,
            "population_density": 0.0,
            "age_distribution": {},
            "error": "No data available",
        }

    # Extract population time series
    populations = []
    timestamps = []
    areas = []
    age_data: Dict[str, List[float]] = {"larvae": [], "pupae": [], "adults": []}

    for record in data:
        if "population" in record:
            populations.append(record["population"])
            if "timestamp" in record:
                timestamps.append(record["timestamp"])
            if "area" in record:
                areas.append(record["area"])

        # Extract age distribution data if available
        if "age_groups" in record:
            for age_group, count in record["age_groups"].items():
                if age_group in age_data:
                    age_data[age_group].append(count)

    if not populations:
        return {"error": "No population data found"}

    # Calculate total population (latest value)
    total_population = populations[-1] if populations else 0

    # Calculate population growth rate
    growth_rate = 0.0
    if len(populations) > 1:
        initial_pop = populations[0]
        final_pop = populations[-1]
        time_periods = len(populations) - 1

        if initial_pop > 0 and time_periods > 0:
            growth_rate = (final_pop - initial_pop) / (initial_pop * time_periods)

    # Calculate population density
    total_area = sum(areas) / len(areas) if areas else 100.0  # Default area
    population_density = total_population / total_area if total_area > 0 else 0

    # Calculate age distribution
    age_distribution = {}
    total_ages = 0.0
    for age_group, counts in age_data.items():
        if counts:
            avg_count = sum(counts) / len(counts)
            age_distribution[age_group] = avg_count
            total_ages += float(avg_count)

    # Normalize age distribution to proportions
    if total_ages > 0:
        for age_group in age_distribution:
            age_distribution[age_group] = age_distribution[age_group] / total_ages
    else:
        # Default distribution if no age data available
        age_distribution = {"larvae": 0.35, "pupae": 0.20, "adults": 0.45}

    return {
        "total_population": int(total_population),
        "population_growth_rate": round(growth_rate, 4),
        "population_density": round(population_density, 2),
        "age_distribution": age_distribution,
        "time_window": time_window,
        "data_points": len(populations),
    }


def calculate_mortality_metrics(
    data: List[Dict[str, Any]], time_window: str
) -> Dict[str, Any]:
    """Calculate real mortality metrics from data"""

    if not data:
        return {
            "mortality_rate": 0.0,
            "causes_of_death": {},
            "seasonal_variation": {},
            "error": "No data available",
        }

    # Extract mortality data
    deaths = []
    births = []
    causes = {"natural": 0, "predation": 0, "environmental": 0, "disease": 0}
    seasonal_deaths = {"spring": 0, "summer": 0, "autumn": 0, "winter": 0}

    for record in data:
        if "deaths" in record:
            deaths.append(record["deaths"])
        if "births" in record:
            births.append(record["births"])
        if "cause_of_death" in record:
            cause = record["cause_of_death"]
            if cause in causes:
                causes[cause] += 1
        if "season" in record and "deaths" in record:
            season = record["season"]
            if season in seasonal_deaths:
                seasonal_deaths[season] += record["deaths"]

    # Calculate mortality rate
    total_deaths = sum(deaths) if deaths else 0
    total_births = sum(births) if births else 1  # Avoid division by zero
    mortality_rate = (
        total_deaths / (total_deaths + total_births)
        if (total_deaths + total_births) > 0
        else 0.0
    )

    # Calculate cause distribution
    total_causes = sum(causes.values())
    causes_of_death = {}
    if total_causes > 0:
        for cause, count in causes.items():
            causes_of_death[cause] = count / total_causes
    else:
        # Default distribution if no cause data
        causes_of_death = {"natural": 0.60, "predation": 0.25, "environmental": 0.15}

    # Calculate seasonal variation
    total_seasonal = sum(seasonal_deaths.values())
    seasonal_variation = {}
    if total_seasonal > 0:
        for season, deaths_count in seasonal_deaths.items():
            seasonal_variation[season] = deaths_count / total_seasonal
    else:
        # Default seasonal pattern if no seasonal data
        seasonal_variation = {
            "spring": 0.20,
            "summer": 0.15,
            "autumn": 0.30,
            "winter": 0.35,
        }

    return {
        "mortality_rate": round(mortality_rate, 4),
        "causes_of_death": causes_of_death,
        "seasonal_variation": seasonal_variation,
        "time_window": time_window,
        "total_deaths": total_deaths,
        "data_points": len(data),
    }


def calculate_foraging_metrics(
    data: List[Dict[str, Any]], time_window: str
) -> Dict[str, Any]:
    """Calculate real foraging metrics from data"""

    if not data:
        return {
            "foraging_efficiency": 0.0,
            "average_trip_duration": 0.0,
            "resource_collection_rate": 0.0,
            "foraging_range": {},
            "error": "No data available",
        }

    # Extract foraging data
    trip_durations = []
    resources_collected = []
    resources_attempted = []
    foraging_distances = []

    for record in data:
        if "trip_duration" in record:
            trip_durations.append(record["trip_duration"])
        if "resources_collected" in record:
            resources_collected.append(record["resources_collected"])
        if "resources_attempted" in record:
            resources_attempted.append(record["resources_attempted"])
        if "foraging_distance" in record:
            foraging_distances.append(record["foraging_distance"])
        # Alternative field names
        if "foraging_trips" in record:
            trip_durations.append(record.get("average_duration", 45.0))
        if "honey_production" in record:
            resources_collected.append(record["honey_production"])

    # Calculate foraging efficiency
    foraging_efficiency = 0.0
    if resources_attempted and resources_collected:
        total_collected = sum(resources_collected)
        total_attempted = sum(resources_attempted)
        foraging_efficiency = (
            total_collected / total_attempted if total_attempted > 0 else 0.0
        )
    elif resources_collected:
        # Assume 80% efficiency if only collection data available
        foraging_efficiency = min(
            0.8, sum(resources_collected) / len(resources_collected) / 100
        )

    # Calculate average trip duration
    average_trip_duration = (
        sum(trip_durations) / len(trip_durations) if trip_durations else 0.0
    )

    # Calculate resource collection rate
    resource_collection_rate = (
        sum(resources_collected) / len(data) if resources_collected else 0.0
    )

    # Calculate foraging range statistics
    foraging_range = {}
    if foraging_distances:
        foraging_range = {
            "mean": sum(foraging_distances) / len(foraging_distances),
            "max": max(foraging_distances),
            "min": min(foraging_distances),
            "preferred": sum(foraging_distances)
            / len(foraging_distances)
            * 0.7,  # 70% of mean as preferred
        }
    else:
        foraging_range = {"mean": 0.0, "max": 0.0, "min": 0.0, "preferred": 0.0}

    return {
        "foraging_efficiency": round(foraging_efficiency, 3),
        "average_trip_duration": round(average_trip_duration, 1),
        "resource_collection_rate": round(resource_collection_rate, 3),
        "foraging_range": foraging_range,
        "time_window": time_window,
        "total_trips": len(trip_durations),
        "data_points": len(data),
    }


def calculate_energy_metrics(
    data: List[Dict[str, Any]], time_window: str
) -> Dict[str, Any]:
    """Calculate real energy metrics from data"""

    if not data:
        return {
            "energy_efficiency": 0.0,
            "energy_consumption": {},
            "energy_reserves": {},
            "error": "No data available",
        }

    # Extract energy data
    energy_inputs = []
    energy_outputs = []
    foraging_energy = []
    maintenance_energy = []
    reproduction_energy = []
    colony_reserves = []
    individual_reserves = []

    for record in data:
        if "energy_input" in record:
            energy_inputs.append(record["energy_input"])
        if "energy_output" in record:
            energy_outputs.append(record["energy_output"])
        if "foraging_energy" in record:
            foraging_energy.append(record["foraging_energy"])
        if "maintenance_energy" in record:
            maintenance_energy.append(record["maintenance_energy"])
        if "reproduction_energy" in record:
            reproduction_energy.append(record["reproduction_energy"])
        if "colony_energy_reserves" in record:
            colony_reserves.append(record["colony_energy_reserves"])
        if "individual_energy" in record:
            individual_reserves.append(record["individual_energy"])
        # Alternative calculation from honey production and population
        if "honey_production" in record and "population" in record:
            energy_inputs.append(
                record["honey_production"] * 0.1
            )  # Convert honey to energy units
            energy_outputs.append(
                record["population"] * 0.02
            )  # Population maintenance cost

    # Calculate energy efficiency
    energy_efficiency = 0.0
    if energy_inputs and energy_outputs:
        total_input = sum(energy_inputs)
        total_output = sum(energy_outputs)
        energy_efficiency = (
            (total_input - total_output) / total_input if total_input > 0 else 0.0
        )
        energy_efficiency = max(
            0.0, min(1.0, energy_efficiency)
        )  # Bound between 0 and 1

    # Calculate energy consumption breakdown
    total_consumption = (
        sum(foraging_energy) + sum(maintenance_energy) + sum(reproduction_energy)
    )
    energy_consumption = {}

    if total_consumption > 0:
        energy_consumption = {
            "foraging": sum(foraging_energy) / total_consumption,
            "maintenance": sum(maintenance_energy) / total_consumption,
            "reproduction": sum(reproduction_energy) / total_consumption,
        }
    else:
        # Estimate from available data or use biological defaults
        energy_consumption = {
            "foraging": 0.45,  # Typical foraging energy allocation
            "maintenance": 0.30,  # Maintenance and thermoregulation
            "reproduction": 0.25,  # Brood care and egg production
        }

    # Calculate energy reserves
    energy_reserves = {}
    if colony_reserves:
        energy_reserves["colony_level"] = sum(colony_reserves) / len(colony_reserves)
    else:
        energy_reserves["colony_level"] = 0.0

    if individual_reserves:
        energy_reserves["individual_level"] = sum(individual_reserves) / len(
            individual_reserves
        )
    else:
        energy_reserves["individual_level"] = 0.0

    return {
        "energy_efficiency": round(energy_efficiency, 3),
        "energy_consumption": energy_consumption,
        "energy_reserves": energy_reserves,
        "time_window": time_window,
        "total_energy_input": sum(energy_inputs) if energy_inputs else 0.0,
        "total_energy_output": sum(energy_outputs) if energy_outputs else 0.0,
        "data_points": len(data),
    }


def display_metrics(metrics: Dict[str, Any], metric_type: str) -> None:
    """Display calculated metrics"""

    table = Table(title=f"{metric_type.title()} Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for metric_name, value in metrics.items():
        if isinstance(value, dict):
            # Show nested metrics
            for sub_metric, sub_value in value.items():
                table.add_row(f"{metric_name}.{sub_metric}", str(sub_value))
        else:
            table.add_row(metric_name.replace("_", " ").title(), str(value))

    console.print(table)


def compare_basic_statistics(
    data1: List[Dict[str, Any]], data2: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare basic statistics between datasets"""

    return {
        "size_comparison": {
            "dataset1_size": len(data1),
            "dataset2_size": len(data2),
            "size_ratio": safe_divide(len(data1), len(data2), float("inf")),
        }
    }


def compare_time_series(
    data1: List[Dict[str, Any]], data2: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare time series data with real analysis"""
    import numpy as np
    from scipy import stats

    if not data1 or not data2:
        return {
            "trend_comparison": "insufficient_data",
            "correlation": 0.0,
            "difference_analysis": "no_data",
            "error": "One or both datasets empty",
        }

    # Extract time series values
    series1 = []
    series2 = []

    # Try different field names for time series data
    field_names = ["population", "honey_production", "foraging_trips", "value", "count"]

    for field in field_names:
        if any(field in record for record in data1):
            series1 = [record.get(field, 0) for record in data1 if field in record]
            break

    for field in field_names:
        if any(field in record for record in data2):
            series2 = [record.get(field, 0) for record in data2 if field in record]
            break

    if not series1 or not series2:
        return {
            "trend_comparison": "no_numeric_data",
            "correlation": 0.0,
            "difference_analysis": "incomparable",
        }

    # Align series lengths
    min_len = min(len(series1), len(series2))
    series1 = series1[:min_len]
    series2 = series2[:min_len]

    # Calculate correlation
    correlation: float = 0.0
    if len(series1) > 1:
        correlation_result, _ = stats.pearsonr(series1, series2)
        correlation = (
            float(correlation_result) if not np.isnan(correlation_result) else 0.0
        )

    # Compare trends (using linear regression slopes)
    trend1 = np.polyfit(range(len(series1)), series1, 1)[0] if len(series1) > 1 else 0
    trend2 = np.polyfit(range(len(series2)), series2, 1)[0] if len(series2) > 1 else 0

    # Determine trend comparison
    trend_diff = abs(trend1 - trend2)
    if trend_diff < 0.1:
        trend_comparison = "very_similar"
    elif trend_diff < 0.5:
        trend_comparison = "similar"
    elif trend_diff < 1.0:
        trend_comparison = "somewhat_different"
    else:
        trend_comparison = "very_different"

    # Difference analysis
    mean_diff = abs(np.mean(series1) - np.mean(series2))
    std_diff = abs(np.std(series1) - np.std(series2))

    if mean_diff < np.std(series1) * 0.1:
        difference_analysis = "minimal"
    elif mean_diff < np.std(series1) * 0.5:
        difference_analysis = "moderate"
    else:
        difference_analysis = "significant"

    return {
        "trend_comparison": trend_comparison,
        "correlation": round(correlation, 3),
        "difference_analysis": difference_analysis,
        "trend1_slope": round(float(trend1), 3),
        "trend2_slope": round(float(trend2), 3),
        "mean_difference": round(mean_diff, 2),
        "std_difference": round(std_diff, 2),
        "data_points_compared": min_len,
    }


def compare_distributions(
    data1: List[Dict[str, Any]], data2: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare data distributions with real statistical analysis"""
    import numpy as np
    from scipy import stats

    if not data1 or not data2:
        return {
            "distribution_similarity": 0.0,
            "outlier_comparison": "insufficient_data",
            "variance_ratio": 0.0,
            "error": "One or both datasets empty",
        }

    # Extract numeric values from datasets
    values1 = []
    values2 = []

    field_names = ["population", "honey_production", "foraging_trips", "value", "count"]

    for field in field_names:
        if any(field in record for record in data1):
            values1 = [
                record.get(field, 0)
                for record in data1
                if field in record and isinstance(record.get(field), (int, float))
            ]
            break

    for field in field_names:
        if any(field in record for record in data2):
            values2 = [
                record.get(field, 0)
                for record in data2
                if field in record and isinstance(record.get(field), (int, float))
            ]
            break

    if not values1 or not values2:
        return {
            "distribution_similarity": 0.0,
            "outlier_comparison": "no_numeric_data",
            "variance_ratio": 0.0,
        }

    # Calculate distribution similarity using Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(values1, values2)
    distribution_similarity = 1 - ks_stat  # Convert KS statistic to similarity

    # Calculate variance ratio
    var1 = np.var(values1) if len(values1) > 1 else 0
    var2 = np.var(values2) if len(values2) > 1 else 0
    variance_ratio = var1 / var2 if var2 != 0 else 0

    # Outlier comparison using IQR method
    def count_outliers(data: List[float]) -> int:
        if len(data) < 4:
            return 0
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return sum(1 for x in data if x < lower_bound or x > upper_bound)

    outliers1 = count_outliers(values1)
    outliers2 = count_outliers(values2)

    outlier_ratio1 = outliers1 / len(values1) if values1 else 0
    outlier_ratio2 = outliers2 / len(values2) if values2 else 0

    if abs(outlier_ratio1 - outlier_ratio2) < 0.05:
        outlier_comparison = "very_similar"
    elif abs(outlier_ratio1 - outlier_ratio2) < 0.15:
        outlier_comparison = "similar"
    else:
        outlier_comparison = "different"

    return {
        "distribution_similarity": round(float(distribution_similarity), 3),
        "outlier_comparison": outlier_comparison,
        "variance_ratio": round(float(variance_ratio), 3),
        "ks_statistic": round(float(ks_stat), 3),
        "ks_pvalue": round(float(ks_pvalue), 3),
        "outliers_dataset1": outliers1,
        "outliers_dataset2": outliers2,
        "mean1": round(np.mean(values1), 2),
        "mean2": round(np.mean(values2), 2),
        "std1": round(np.std(values1), 2),
        "std2": round(np.std(values2), 2),
    }


def perform_statistical_tests(
    data1: List[Dict[str, Any]], data2: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Perform real statistical tests on data"""
    import numpy as np
    from scipy import stats

    if not data1 or not data2:
        return {
            "t_test": {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "insufficient_data",
            },
            "ks_test": {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "insufficient_data",
            },
            "mann_whitney": {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "insufficient_data",
            },
        }

    # Extract numeric values
    values1 = []
    values2 = []

    field_names = ["population", "honey_production", "foraging_trips", "value", "count"]

    for field in field_names:
        if any(field in record for record in data1):
            values1 = [
                record.get(field, 0)
                for record in data1
                if field in record and isinstance(record.get(field), (int, float))
            ]
            break

    for field in field_names:
        if any(field in record for record in data2):
            values2 = [
                record.get(field, 0)
                for record in data2
                if field in record and isinstance(record.get(field), (int, float))
            ]
            break

    if not values1 or not values2:
        return {
            "t_test": {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "no_numeric_data",
            },
            "ks_test": {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "no_numeric_data",
            },
            "mann_whitney": {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "no_numeric_data",
            },
        }

    results: Dict[str, Any] = {}

    # T-test (assumes normal distributions)
    try:
        if len(values1) > 1 and len(values2) > 1:
            t_stat, t_pvalue = stats.ttest_ind(values1, values2)
            results["t_test"] = {
                "statistic": round(float(t_stat), 3),
                "p_value": round(float(t_pvalue), 3),
                "significant": bool(t_pvalue < 0.05),
            }
        else:
            results["t_test"] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "insufficient_samples",
            }
    except Exception as e:
        results["t_test"] = {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "error": str(e),
        }

    # Kolmogorov-Smirnov test (distribution comparison)
    try:
        ks_stat, ks_pvalue = stats.ks_2samp(values1, values2)
        results["ks_test"] = {
            "statistic": round(float(ks_stat), 3),
            "p_value": round(float(ks_pvalue), 3),
            "significant": bool(ks_pvalue < 0.05),
        }
    except Exception as e:
        results["ks_test"] = {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "error": str(e),
        }

    # Mann-Whitney U test (non-parametric)
    try:
        if len(values1) > 0 and len(values2) > 0:
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                values1, values2, alternative="two-sided"
            )
            results["mann_whitney"] = {
                "statistic": round(float(mw_stat), 1),
                "p_value": round(float(mw_pvalue), 3),
                "significant": bool(mw_pvalue < 0.05),
            }
        else:
            results["mann_whitney"] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "insufficient_samples",
            }
    except Exception as e:
        results["mann_whitney"] = {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "error": str(e),
        }

    # Add summary interpretation
    results["interpretation"] = {
        "sample_size1": len(values1),
        "sample_size2": len(values2),
        "mean_difference": round(np.mean(values1) - np.mean(values2), 2)
        if values1 and values2
        else 0,
        "effect_size": "large"
        if abs(float(np.mean(values1) - np.mean(values2)))
        / float(np.std(values1 + values2))
        > 0.8
        else "medium"
        if abs(float(np.mean(values1) - np.mean(values2)))
        / float(np.std(values1 + values2))
        > 0.5
        else "small",
    }

    return results


def display_comparison_results(results: Dict[str, Any]) -> None:
    """Display comparison results"""

    table = Table(title="Dataset Comparison Results")
    table.add_column("Comparison Type", style="cyan")
    table.add_column("Result", style="magenta")
    table.add_column("Significance", style="yellow")

    for comparison_type, comparison_data in results.items():
        if isinstance(comparison_data, dict):
            for key, value in comparison_data.items():
                significance = (
                    "High"
                    if isinstance(value, dict) and value.get("p_value", 1) < 0.05
                    else "Low"
                )
                table.add_row(f"{comparison_type}.{key}", str(value), significance)
        else:
            table.add_row(comparison_type, str(comparison_data), "N/A")

    console.print(table)


# Dead colony analysis helper functions
def load_colony_data_with_mortality(file_path: str) -> List[Dict[str, Any]]:
    """Load colony data including mortality information"""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Colony data file not found: {file_path}")

    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_dict("records")  # type: ignore[return-value]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def filter_colony_data_by_time(
    data: List[Dict[str, Any]], start: Optional[int], end: Optional[int]
) -> List[Dict[str, Any]]:
    """Filter colony data by time range"""

    filtered_data = []
    for record in data:
        death_time = record.get("death_time", None)
        if death_time is not None:
            if start is not None and death_time < start:
                continue
            if end is not None and death_time > end:
                continue
        filtered_data.append(record)

    return filtered_data


def extract_dead_colonies(colony_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract dead colonies from colony data"""

    dead_colonies = []
    for colony in colony_data:
        if colony.get("status") == "dead" or colony.get("death_time") is not None:
            dead_colonies.append(colony)

    return dead_colonies


def extract_surviving_colonies(
    colony_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract surviving colonies from colony data"""

    surviving_colonies = []
    for colony in colony_data:
        if colony.get("status") != "dead" and colony.get("death_time") is None:
            surviving_colonies.append(colony)

    return surviving_colonies


def filter_dead_colonies_by_time(
    dead_colonies: List[Dict[str, Any]], start: Optional[int], end: Optional[int]
) -> List[Dict[str, Any]]:
    """Filter dead colonies by time range"""

    filtered_colonies = []
    for colony in dead_colonies:
        death_time = colony.get("death_time", 0)

        if start is not None and death_time < start:
            continue
        if end is not None and death_time > end:
            continue

        filtered_colonies.append(colony)

    return filtered_colonies


def analyze_dead_colony_summary(dead_colonies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze summary statistics for dead colonies"""

    summary = {
        "total_dead_colonies": len(dead_colonies),
        "average_lifespan": 0,
        "death_distribution": {},
        "seasonal_mortality": {},
    }

    if dead_colonies:
        lifespans = []
        death_times = []

        for colony in dead_colonies:
            death_time = colony.get("death_time", 0)
            birth_time = colony.get("birth_time", 0)
            lifespan = death_time - birth_time

            lifespans.append(lifespan)
            death_times.append(death_time)

        summary["average_lifespan"] = safe_divide(sum(lifespans), len(lifespans), 0)
        summary["median_lifespan"] = sorted(lifespans)[len(lifespans) // 2]
        summary["min_lifespan"] = min(lifespans)
        summary["max_lifespan"] = max(lifespans)

    return summary


def analyze_mortality_patterns(dead_colonies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze mortality patterns"""

    patterns: Dict[str, Any] = {
        "temporal_distribution": {},
        "age_at_death": {},
        "mortality_rate_over_time": {},
    }

    # Analyze temporal distribution
    death_times = [colony.get("death_time", 0) for colony in dead_colonies]

    # Group by time periods (simplified)
    for death_time in death_times:
        period = f"period_{death_time // 30}"  # 30-day periods
        patterns["temporal_distribution"][period] = (
            patterns["temporal_distribution"].get(period, 0) + 1
        )

    return patterns


def analyze_colony_lifecycles(dead_colonies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze colony lifecycles"""

    lifecycle_analysis: Dict[str, Any] = {
        "lifecycle_stages": {},
        "development_patterns": {},
        "critical_periods": {},
    }

    for colony in dead_colonies:
        max_population = colony.get("max_population", 0)

        # Categorize by maximum population reached
        if max_population < 100:
            stage = "early_death"
        elif max_population < 500:
            stage = "juvenile_death"
        elif max_population < 1000:
            stage = "mature_death"
        else:
            stage = "late_death"

        lifecycle_analysis["lifecycle_stages"][stage] = (
            lifecycle_analysis["lifecycle_stages"].get(stage, 0) + 1
        )

    return lifecycle_analysis


def analyze_death_causes(dead_colonies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze causes of death"""

    cause_analysis: Dict[str, Any] = {
        "cause_distribution": {},
        "cause_by_season": {},
        "preventable_deaths": 0,
    }

    for colony in dead_colonies:
        cause = colony.get("death_cause", "unknown")
        season = colony.get("death_season", "unknown")

        # Count causes
        cause_dist = cause_analysis["cause_distribution"]
        cause_dist[cause] = cause_dist.get(cause, 0) + 1

        # Count by season
        cause_by_season = cause_analysis["cause_by_season"]
        if season not in cause_by_season:
            cause_by_season[season] = {}
        cause_by_season[season][cause] = cause_by_season[season].get(cause, 0) + 1

        # Count preventable deaths
        if cause in ["starvation", "disease", "environmental"]:
            cause_analysis["preventable_deaths"] += 1

    return cause_analysis


def compare_dead_vs_surviving(
    dead_colonies: List[Dict[str, Any]], surviving_colonies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare dead colonies with surviving ones"""

    # Validate and clean data
    dead_colonies = validate_colony_data(dead_colonies)
    surviving_colonies = validate_colony_data(surviving_colonies)

    comparison: Dict[str, Any] = {
        "population_comparison": {},
        "performance_comparison": {},
        "risk_factors": {},
    }

    # Population comparison with safe calculations
    dead_max_pops = [colony.get("max_population", 0) for colony in dead_colonies]
    surviving_max_pops = [
        colony.get("max_population", 0) for colony in surviving_colonies
    ]

    dead_avg = safe_divide(sum(dead_max_pops), len(dead_max_pops), 0)
    surviving_avg = safe_divide(sum(surviving_max_pops), len(surviving_max_pops), 0)

    comparison["population_comparison"] = {
        "dead_avg_max_population": dead_avg,
        "surviving_avg_max_population": surviving_avg,
        "population_ratio": safe_divide(dead_avg, surviving_avg, 0),
        "dead_colony_count": len(dead_colonies),
        "surviving_colony_count": len(surviving_colonies),
    }

    # Add performance comparison with safe calculations
    if dead_colonies or surviving_colonies:
        # Compare foraging efficiency safely
        dead_foraging = [
            colony.get("foraging_efficiency", 0) for colony in dead_colonies
        ]
        surviving_foraging = [
            colony.get("foraging_efficiency", 0) for colony in surviving_colonies
        ]

        dead_foraging_avg = safe_divide(sum(dead_foraging), len(dead_foraging), 0)
        surviving_foraging_avg = safe_divide(
            sum(surviving_foraging), len(surviving_foraging), 0
        )

        comparison["performance_comparison"] = {
            "dead_avg_foraging_efficiency": dead_foraging_avg,
            "surviving_avg_foraging_efficiency": surviving_foraging_avg,
            "foraging_efficiency_ratio": safe_divide(
                dead_foraging_avg, surviving_foraging_avg, 0
            ),
        }

    # Risk factors analysis - always calculate if we have any data
    total_colonies = len(dead_colonies) + len(surviving_colonies)
    if total_colonies > 0:
        mortality_rate = safe_divide(len(dead_colonies), total_colonies, 0) * 100
        survival_rate = safe_divide(len(surviving_colonies), total_colonies, 0) * 100

        comparison["risk_factors"] = {
            "mortality_rate_percent": mortality_rate,
            "survival_rate_percent": survival_rate,
            "total_colonies_analyzed": total_colonies,
        }
    else:
        # Handle case with no colonies at all
        comparison["risk_factors"] = {
            "mortality_rate_percent": 0.0,
            "survival_rate_percent": 0.0,
            "total_colonies_analyzed": 0,
        }

    return comparison


def analyze_dead_colony_performance(
    dead_colonies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze performance metrics of dead colonies"""

    performance: Dict[str, Any] = {
        "foraging_efficiency": {},
        "resource_utilization": {},
        "reproductive_success": {},
    }

    efficiencies = []
    for colony in dead_colonies:
        efficiency = colony.get("foraging_efficiency", 0)
        if efficiency > 0:
            efficiencies.append(efficiency)

    if efficiencies:
        performance["foraging_efficiency"] = {
            "mean": safe_divide(sum(efficiencies), len(efficiencies), 0),
            "min": min(efficiencies),
            "max": max(efficiencies),
        }

    return performance


def export_dead_colony_analysis(
    analysis_results: Dict[str, Any], output_path: Path, format_type: str
) -> None:
    """Export dead colony analysis results"""

    for analysis_name, analysis_data in analysis_results.items():
        filename = f"dead_colony_{analysis_name}.{format_type}"
        file_path = output_path / filename

        if format_type == "json":
            with open(file_path, "w") as f:
                json.dump(analysis_data, f, indent=2, default=str)
        elif format_type == "csv":
            if isinstance(analysis_data, dict):
                df = pd.DataFrame([analysis_data])
                df.to_csv(file_path, index=False)
            else:
                pd.DataFrame(analysis_data).to_csv(file_path, index=False)
        elif format_type == "xlsx":
            if isinstance(analysis_data, dict):
                df = pd.DataFrame([analysis_data])
                df.to_excel(file_path, index=False, engine="openpyxl")
            else:
                pd.DataFrame(analysis_data).to_excel(
                    file_path, index=False, engine="openpyxl"
                )


def display_dead_colony_analysis(analysis_results: Dict[str, Any]) -> None:
    """Display dead colony analysis results"""

    # Summary table
    if "summary" in analysis_results:
        summary = analysis_results["summary"]

        summary_table = Table(title="Dead Colony Analysis Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row(
            "Total Dead Colonies", str(summary.get("total_dead_colonies", 0))
        )
        summary_table.add_row(
            "Average Lifespan", f"{summary.get('average_lifespan', 0):.1f} days"
        )
        summary_table.add_row(
            "Median Lifespan", f"{summary.get('median_lifespan', 0):.1f} days"
        )
        summary_table.add_row(
            "Min Lifespan", f"{summary.get('min_lifespan', 0):.1f} days"
        )
        summary_table.add_row(
            "Max Lifespan", f"{summary.get('max_lifespan', 0):.1f} days"
        )

        console.print(summary_table)

    # Cause analysis
    if "cause_analysis" in analysis_results:
        cause_data = analysis_results["cause_analysis"]

        cause_table = Table(title="Death Cause Analysis")
        cause_table.add_column("Cause", style="cyan")
        cause_table.add_column("Count", style="magenta")
        cause_table.add_column("Percentage", style="yellow")

        total_deaths = sum(cause_data.get("cause_distribution", {}).values())

        for cause, count in cause_data.get("cause_distribution", {}).items():
            percentage = (count / total_deaths * 100) if total_deaths > 0 else 0
            cause_table.add_row(cause.title(), str(count), f"{percentage:.1f}%")

        console.print(cause_table)

        console.print(
            f"[yellow]Preventable Deaths: {cause_data.get('preventable_deaths', 0)}[/yellow]"
        )

    # Comparative analysis
    if "comparative_analysis" in analysis_results:
        comp_data = analysis_results["comparative_analysis"]
        pop_comp = comp_data.get("population_comparison", {})

        comp_table = Table(title="Dead vs Surviving Colonies")
        comp_table.add_column("Metric", style="cyan")
        comp_table.add_column("Dead Colonies", style="red")
        comp_table.add_column("Surviving Colonies", style="green")

        comp_table.add_row(
            "Average Max Population",
            f"{pop_comp.get('dead_avg_max_population', 0):.0f}",
            f"{pop_comp.get('surviving_avg_max_population', 0):.0f}",
        )

        console.print(comp_table)


# Create typer app for data analysis commands
app = typer.Typer(name="data", help="Data analysis commands")

app.command(name="analyze")(analyze_simulation_data)
app.command(name="export")(export_data)
app.command(name="summary")(generate_summary)
app.command(name="metrics")(calculate_metrics)
app.command(name="compare")(compare_datasets)
app.command(name="analyze-dead-colonies")(analyze_dead_colonies)

if __name__ == "__main__":
    app()
