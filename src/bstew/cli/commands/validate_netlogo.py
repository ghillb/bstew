"""
NetLogo Validation CLI Commands
==============================

CLI commands for validating BSTEW behavior against NetLogo BEE-STEWARD v2.
"""

import typer
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any, List
from rich.console import Console

from ...validation.netlogo_validation import NetLogoValidationSuite, NetLogoDataLoader
from ...core.data_collection import ComprehensiveDataCollector
from ...core.model import BeeModel
from ...data.netlogo_parser import NetLogoDataParser
import yaml
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from datetime import datetime

# Initialize console for rich output
console = Console()

# Create Typer app for NetLogo validation commands
app = typer.Typer(
    name="netlogo",
    help="NetLogo BEE-STEWARD v2 validation commands",
    add_completion=False,
)


@app.command()
def validate(
    netlogo_data: Path = typer.Option(
        ...,
        "--netlogo-data",
        help="Path to NetLogo output data directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    bstew_data: Optional[Path] = typer.Option(
        None,
        "--bstew-data",
        help="Path to BSTEW data collector file (JSON)",
        exists=True,
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to BSTEW configuration file", exists=True
    ),
    output: Path = typer.Option(
        "validation_results", "--output", help="Output directory for validation results"
    ),
    run_simulation: bool = typer.Option(
        False, "--run-simulation", help="Run a new BSTEW simulation for comparison"
    ),
    simulation_days: int = typer.Option(
        365,
        "--simulation-days",
        help="Number of days to simulate (if running new simulation)",
    ),
    tolerance_config: Optional[Path] = typer.Option(
        None,
        "--tolerance-config",
        help="Path to validation tolerance configuration (JSON)",
        exists=True,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Validate BSTEW behavior against NetLogo BEE-STEWARD v2"""

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger(__name__)

    console.print(
        "🧪 [bold blue]Starting NetLogo BEE-STEWARD v2 Validation[/bold blue]"
    )

    try:
        # Load tolerance configuration if provided
        tolerance_config_dict = None
        if tolerance_config:
            with open(tolerance_config, "r") as f:
                tolerance_config_dict = json.load(f)
            console.print(f"✓ Loaded tolerance configuration from {tolerance_config}")

        # Get BSTEW data collector
        data_collector = None

        if run_simulation:
            console.print(f"🚀 Running new BSTEW simulation for {simulation_days} days")
            data_collector = _run_bstew_simulation(config, simulation_days)

        elif bstew_data:
            console.print(f"📁 Loading BSTEW data from {bstew_data}")
            data_collector = _load_bstew_data(bstew_data)

        else:
            console.print(
                "❌ [red]Must provide either --bstew-data or --run-simulation[/red]"
            )
            raise typer.Exit(1)

        if not data_collector:
            console.print("❌ [red]Failed to obtain BSTEW data[/red]")
            raise typer.Exit(1)

        # Initialize validation suite
        console.print("🔧 Initializing validation suite...")
        validation_suite = NetLogoValidationSuite(
            netlogo_data_path=str(netlogo_data), output_path=str(output)
        )

        if tolerance_config_dict:
            validation_suite.validator.tolerance_config = tolerance_config_dict

        # Run validation
        console.print("⚡ Running behavioral validation...")
        validation_results = validation_suite.run_complete_validation(data_collector)

        # Print summary results
        _print_validation_summary(validation_results)

        console.print(
            f"✅ [green]Validation completed. Results saved to {output}[/green]"
        )

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"❌ [red]Validation failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def inspect(
    netlogo_data: Path = typer.Option(
        ...,
        "--netlogo-data",
        help="Path to NetLogo output data directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: Path = typer.Option(
        "artifacts/netlogo/netlogo_data_summary.json",
        "--output",
        help="Output file for data summary",
    ),
) -> None:
    """Inspect available NetLogo data for validation"""

    console.print(
        f"🔍 [bold blue]Inspecting NetLogo data in {netlogo_data}[/bold blue]"
    )

    try:
        # Load NetLogo data
        loader = NetLogoDataLoader(str(netlogo_data))
        data = loader.load_netlogo_outputs()

        # Create summary
        summary = {}
        for category, category_data in data.items():
            if category_data:
                summary[category] = {
                    "available": True,
                    "metrics": list(category_data.keys()),
                    "metric_count": len(category_data),
                }

                # Add data shape information
                for metric, values in category_data.items():
                    if isinstance(values, list) and values:
                        summary[category][f"{metric}_length"] = len(values)
                        summary[category][f"{metric}_sample"] = (
                            values[:5] if len(values) >= 5 else values
                        )
            else:
                summary[category] = {"available": False}

        # Save summary
        with open(output, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Print summary using Rich
        console.print("\n📊 [bold]NetLogo Data Summary:[/bold]")
        console.print("=" * 50)
        for category, info in summary.items():
            status = "✅" if info.get("available", False) else "❌"
            metric_count = info.get("metric_count", 0)
            console.print(f"{status} [cyan]{category}[/cyan]: {metric_count} metrics")

            if info.get("available") and info.get("metrics"):
                metrics_data = info.get("metrics", [])
                metrics = (
                    list(metrics_data)
                    if isinstance(metrics_data, (list, tuple))
                    else []
                )
                for metric in metrics[:3]:  # Show first 3 metrics
                    console.print(f"    • {metric}")
                if len(metrics) > 3:
                    console.print(f"    ... and {len(metrics) - 3} more")

        console.print(f"\n✅ [green]Detailed summary saved to {output}[/green]")

    except Exception as e:
        console.print(f"❌ [red]Failed to inspect NetLogo data: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def generate_bstew_data(
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to BSTEW configuration file", exists=True
    ),
    days: int = typer.Option(365, "--days", help="Number of days to simulate"),
    output: Path = typer.Option(
        "bstew_validation_data", "--output", help="Output directory for BSTEW data"
    ),
) -> None:
    """Generate BSTEW data for NetLogo validation"""

    console.print(f"🚀 [bold blue]Generating BSTEW data for {days} days[/bold blue]")

    try:
        # Run simulation
        data_collector = _run_bstew_simulation(config, days)

        if data_collector:
            # Save data collector state
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)

            # Export data in various formats for analysis
            console.print("💾 Exporting simulation data...")
            _export_bstew_data(data_collector, output_path)

            console.print(
                f"✅ [green]BSTEW data generated and saved to {output}[/green]"
            )
        else:
            console.print("❌ [red]Failed to generate BSTEW data[/red]")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"❌ [red]Failed to generate BSTEW data: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def convert(
    netlogo_directory: Path = typer.Argument(
        ...,
        help="Directory containing NetLogo BEE-STEWARD data files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_directory: Path = typer.Option(
        "artifacts/netlogo_converted",
        "--output",
        "-o",
        help="Output directory for converted BSTEW data",
    ),
    validate_first: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate NetLogo data compatibility before conversion",
    ),
    include_parameters: bool = typer.Option(
        True,
        "--include-parameters/--no-parameters",
        help="Include parameter file conversion",
    ),
    include_species: bool = typer.Option(
        True, "--include-species/--no-species", help="Include species data conversion"
    ),
    include_spatial: bool = typer.Option(
        True,
        "--include-spatial/--no-spatial",
        help="Include spatial/landscape data conversion",
    ),
    format_type: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format for converted data (json, csv, yaml)",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing output directory"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Convert NetLogo BEE-STEWARD data to BSTEW format"""

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    console.print("🔄 [bold blue]Starting NetLogo to BSTEW Data Conversion[/bold blue]")
    console.print(f"📂 Source: {netlogo_directory}")
    console.print(f"📁 Target: {output_directory}")

    try:
        # Check output directory
        if output_directory.exists() and not overwrite:
            console.print(
                f"❌ [red]Output directory already exists: {output_directory}[/red]"
            )
            console.print("Use --overwrite to replace existing directory")
            raise typer.Exit(1)

        # Create output directory
        output_directory.mkdir(parents=True, exist_ok=True)

        # Validate NetLogo data first if requested
        if validate_first:
            console.print("🔍 Validating NetLogo data compatibility...")
            validation_success = _validate_netlogo_data(netlogo_directory)
            if not validation_success:
                console.print(
                    "⚠️ [yellow]Validation warnings found, but continuing conversion...[/yellow]"
                )

        # Import the NetLogo conversion functionality
        from ..commands.parameters import ParameterManagerCLI
        from ..core.base import CLIContext

        # Create parameter manager for conversion
        cli_context = CLIContext(console=console)
        param_manager = ParameterManagerCLI(cli_context)

        # Step 1: Convert parameters if requested
        if include_parameters:
            console.print("⚙️ Converting NetLogo parameters...")
            param_manager.convert_netlogo_data(
                str(netlogo_directory),
                str(output_directory / "parameters"),
                validate_first=False,  # Already validated above
            )

        # Step 2: Convert species data if requested
        if include_species:
            console.print("🐝 Converting species data...")
            _convert_species_data(netlogo_directory, output_directory, format_type)

        # Step 3: Convert spatial/landscape data if requested
        if include_spatial:
            console.print("🗺️ Converting spatial data...")
            _convert_spatial_data(netlogo_directory, output_directory, format_type)

        # Step 4: Create conversion summary report
        console.print("📋 Generating conversion report...")
        _create_conversion_report(
            netlogo_directory,
            output_directory,
            {
                "include_parameters": include_parameters,
                "include_species": include_species,
                "include_spatial": include_spatial,
                "format_type": format_type,
                "validated": validate_first,
            },
        )

        console.print("✅ [green]Conversion completed successfully![/green]")
        console.print(f"📁 Converted data available in: {output_directory}")
        console.print("📋 See conversion_report.json for details")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"❌ [red]Conversion failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _run_bstew_simulation(
    config_path: Optional[Path], days: int
) -> Optional[ComprehensiveDataCollector]:
    """Run BSTEW simulation and return data collector"""

    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = None
        if config_path:
            import json

            with open(config_path, "r") as f:
                config = json.load(f)

        # Create and run model
        model = BeeModel(config=config)

        # Enable comprehensive data collection
        if not hasattr(model, "comprehensive_data_collector"):
            model.comprehensive_data_collector = ComprehensiveDataCollector()

        # Run simulation
        logger.info(f"Running simulation for {days} days...")
        for day in range(days):
            model.step()

            if day % 50 == 0:  # Progress update every 50 days
                logger.info(f"Simulation progress: {day}/{days} days")

        model.cleanup()

        # Return the data collector from system integrator if available
        if hasattr(model, "system_integrator") and model.system_integrator:
            return model.system_integrator.data_collector
        else:
            return getattr(model, "comprehensive_data_collector", None)

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return None


def _load_bstew_data(data_path: Path) -> Optional[ComprehensiveDataCollector]:
    """Load BSTEW data collector from file"""

    logger = logging.getLogger(__name__)

    try:
        # This would need to be implemented based on how data is serialized
        logger.warning("Loading BSTEW data from file not yet implemented")
        return None

    except Exception as e:
        logger.error(f"Failed to load BSTEW data: {e}")
        return None


def _export_bstew_data(
    data_collector: ComprehensiveDataCollector, output_path: Path
) -> None:
    """Export BSTEW data collector data to files"""

    logger = logging.getLogger(__name__)

    try:
        # Export population data
        pop_data = []
        for colony_id, metrics in data_collector.colony_metrics.items():
            if hasattr(metrics, "population_size_day_list"):
                for day, population in enumerate(metrics.population_size_day_list):
                    pop_data.append(
                        {
                            "day": day,
                            "colony_id": colony_id,
                            "total_population": population,
                            "egg_count": metrics.egg_count_day_list[day]
                            if day < len(metrics.egg_count_day_list)
                            else 0,
                            "larva_count": metrics.larva_count_day_list[day]
                            if day < len(metrics.larva_count_day_list)
                            else 0,
                            "pupa_count": metrics.pupa_count_day_list[day]
                            if day < len(metrics.pupa_count_day_list)
                            else 0,
                            "adult_count": metrics.adult_count_day_list[day]
                            if day < len(metrics.adult_count_day_list)
                            else 0,
                        }
                    )

        if pop_data:
            import pandas as pd

            pop_df = pd.DataFrame(pop_data)
            pop_df.to_csv(output_path / "population_data.csv", index=False)

        # Export foraging data
        foraging_data = []
        for bee_id, metrics in data_collector.bee_metrics.items():
            if hasattr(metrics, "foraging_trips") and metrics.foraging_trips > 0:
                foraging_data.append(
                    {
                        "bee_id": bee_id,
                        "total_trips": metrics.foraging_trips,
                        "successful_trips": getattr(metrics, "successful_trips", 0),
                        "foraging_efficiency": getattr(
                            metrics, "foraging_success_rate", 0
                        ),
                        "total_energy_collected": getattr(
                            metrics, "nectar_collected_total", 0
                        )
                        + getattr(metrics, "pollen_collected_total", 0),
                    }
                )

        if foraging_data:
            import pandas as pd

            foraging_df = pd.DataFrame(foraging_data)
            foraging_df.to_csv(output_path / "foraging_data.csv", index=False)

        logger.info("BSTEW data exported successfully")

    except Exception as e:
        logger.error(f"Failed to export BSTEW data: {e}")


def _validate_netlogo_data(netlogo_directory: Path) -> bool:
    """Validate NetLogo data compatibility"""
    try:
        # Basic validation - check for required files and structure
        required_patterns = [
            "*.csv",  # Parameter or data files
            "*.txt",  # NetLogo output files
        ]

        found_files = []
        for pattern in required_patterns:
            found_files.extend(list(netlogo_directory.glob(pattern)))

        if not found_files:
            console.print(
                "⚠️ [yellow]Warning: No recognized NetLogo data files found[/yellow]"
            )
            return False

        console.print(f"✓ Found {len(found_files)} data files")
        return True

    except Exception as e:
        console.print(f"❌ [red]Validation error: {e}[/red]")
        return False


def _convert_species_data(
    netlogo_directory: Path, output_directory: Path, format_type: str
) -> None:
    """Convert NetLogo species data to BSTEW format"""
    try:
        from ...data.netlogo_parser import NetLogoStringParser

        # Look for species-related files
        species_files = (
            list(netlogo_directory.glob("*species*.csv"))
            + list(netlogo_directory.glob("*flower*.csv"))
            + list(netlogo_directory.glob("*bee*.csv"))
        )

        if not species_files:
            console.print("ℹ️ No species data files found")
            return

        species_output_dir = output_directory / "species"
        species_output_dir.mkdir(exist_ok=True)

        for species_file in species_files:
            console.print(f"  📝 Processing {species_file.name}")

            # Read and convert species file
            import pandas as pd

            df = pd.read_csv(species_file)

            # Convert NetLogo format to BSTEW format
            converted_data = []
            for _, row in df.iterrows():
                converted_row = {}
                for col, value in row.items():
                    # Parse NetLogo-specific formats
                    converted_row[col] = NetLogoStringParser.parse_netlogo_string(
                        str(value)
                    )
                converted_data.append(converted_row)

            # Save in requested format
            output_file = species_output_dir / f"{species_file.stem}_converted"
            if format_type == "json":
                import json

                with open(f"{output_file}.json", "w") as f:
                    json.dump(converted_data, f, indent=2, default=str)
            elif format_type == "csv":
                converted_df = pd.DataFrame(converted_data)
                converted_df.to_csv(f"{output_file}.csv", index=False)
            elif format_type == "yaml":
                import yaml

                with open(f"{output_file}.yaml", "w") as f:
                    yaml.safe_dump(converted_data, f, default_flow_style=False)

        console.print(f"✓ Converted {len(species_files)} species files")

    except Exception as e:
        console.print(f"⚠️ [yellow]Species conversion warning: {e}[/yellow]")


def _convert_spatial_data(
    netlogo_directory: Path, output_directory: Path, format_type: str
) -> None:
    """Convert NetLogo spatial/landscape data to BSTEW format"""
    try:
        # Look for spatial-related files
        spatial_files = (
            list(netlogo_directory.glob("*landscape*.csv"))
            + list(netlogo_directory.glob("*patch*.csv"))
            + list(netlogo_directory.glob("*coordinates*.csv"))
            + list(netlogo_directory.glob("*location*.csv"))
        )

        if not spatial_files:
            console.print("ℹ️ No spatial data files found")
            return

        spatial_output_dir = output_directory / "spatial"
        spatial_output_dir.mkdir(exist_ok=True)

        for spatial_file in spatial_files:
            console.print(f"  🗺️ Processing {spatial_file.name}")

            # Read and convert spatial file
            import pandas as pd

            df = pd.read_csv(spatial_file)

            # Convert coordinate systems and formats as needed
            converted_df = df.copy()

            # Basic coordinate validation and conversion
            if "x" in converted_df.columns and "y" in converted_df.columns:
                # Ensure numeric coordinates
                converted_df["x"] = pd.to_numeric(converted_df["x"], errors="coerce")
                converted_df["y"] = pd.to_numeric(converted_df["y"], errors="coerce")

            # Save in requested format
            output_file = spatial_output_dir / f"{spatial_file.stem}_converted"
            if format_type == "json":
                converted_df.to_json(f"{output_file}.json", orient="records", indent=2)
            elif format_type == "csv":
                converted_df.to_csv(f"{output_file}.csv", index=False)
            elif format_type == "yaml":
                import yaml

                with open(f"{output_file}.yaml", "w") as f:
                    yaml.safe_dump(
                        converted_df.to_dict("records"), f, default_flow_style=False
                    )

        console.print(f"✓ Converted {len(spatial_files)} spatial files")

    except Exception as e:
        console.print(f"⚠️ [yellow]Spatial conversion warning: {e}[/yellow]")


def _create_conversion_report(
    netlogo_directory: Path, output_directory: Path, config: dict
) -> None:
    """Create detailed conversion report"""
    try:
        import json
        from datetime import datetime

        # Gather conversion statistics
        source_files = list(netlogo_directory.glob("*"))
        source_data_files = [
            f for f in source_files if f.suffix in [".csv", ".txt", ".json"]
        ]

        converted_files = []
        for subdir in ["parameters", "species", "spatial"]:
            subdir_path = output_directory / subdir
            if subdir_path.exists():
                converted_files.extend(list(subdir_path.glob("*")))

        report: Dict[str, Any] = {
            "conversion_timestamp": datetime.now().isoformat(),
            "source_directory": str(netlogo_directory),
            "output_directory": str(output_directory),
            "conversion_config": config,
            "source_files": {
                "total_files": len(source_files),
                "data_files": len(source_data_files),
                "file_list": [f.name for f in source_data_files],
            },
            "converted_files": {
                "total_converted": len(converted_files),
                "by_category": {},
                "file_list": [f.name for f in converted_files],
            },
            "conversion_summary": {
                "parameters_converted": config.get("include_parameters", False),
                "species_converted": config.get("include_species", False),
                "spatial_converted": config.get("include_spatial", False),
                "format_used": config.get("format_type", "json"),
                "validation_performed": config.get("validated", False),
            },
        }

        # Count files by category
        for subdir in ["parameters", "species", "spatial"]:
            subdir_path = output_directory / subdir
            if subdir_path.exists():
                count = len(list(subdir_path.glob("*")))
                report["converted_files"]["by_category"][subdir] = count

        # Save report
        report_file = output_directory / "conversion_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        console.print(f"✓ Conversion report saved to {report_file}")

    except Exception as e:
        console.print(f"⚠️ [yellow]Report generation warning: {e}[/yellow]")


def _print_validation_summary(results: dict) -> None:
    """Print validation results summary using Rich console"""

    console.print("\n" + "=" * 60)
    console.print("[bold blue]NetLogo BEE-STEWARD v2 Validation Results[/bold blue]")
    console.print("=" * 60)

    total_metrics = 0
    total_passed = 0

    for category, result in results.items():
        if result.pass_rate >= 0.8:
            status = "✅ [green]PASS[/green]"
        elif result.pass_rate >= 0.6:
            status = "⚠️ [yellow]WARN[/yellow]"
        else:
            status = "❌ [red]FAIL[/red]"

        console.print(f"\n{status} [cyan]{category.replace('_', ' ').title()}[/cyan]")
        console.print(
            f"    Pass Rate: {result.pass_rate:.1%} ({result.passed_metrics}/{result.total_metrics})"
        )
        console.print(
            f"    Mean Difference: {result.summary_statistics.get('mean_relative_difference', 0):.3f}"
        )

        total_metrics += result.total_metrics
        total_passed += result.passed_metrics

        # Show failed metrics
        failed_metrics = [
            m for m in result.individual_results if not m.passes_validation
        ]
        if failed_metrics:
            console.print("    [red]Failed metrics:[/red]")
            for metric in failed_metrics[:3]:  # Show first 3 failed
                console.print(
                    f"      • {metric.metric_name}: {metric.relative_difference:.3f} (>{metric.tolerance:.3f})"
                )
            if len(failed_metrics) > 3:
                console.print(f"      ... and {len(failed_metrics) - 3} more")

    overall_rate = total_passed / total_metrics if total_metrics > 0 else 0
    if overall_rate >= 0.8:
        overall_status = "✅ [green]PASS[/green]"
    elif overall_rate >= 0.6:
        overall_status = "⚠️ [yellow]WARN[/yellow]"
    else:
        overall_status = "❌ [red]FAIL[/red]"

    console.print(f"\n{'-' * 60}")
    console.print(
        f"{overall_status} [bold]Overall Validation: {overall_rate:.1%} ({total_passed}/{total_metrics})[/bold]"
    )
    console.print("=" * 60)


@app.command()
def compare(
    bstew_results: Path = typer.Option(
        ...,
        "--bstew",
        help="Path to BSTEW simulation results directory or CSV file",
        exists=True,
    ),
    netlogo_results: Path = typer.Option(
        ...,
        "--netlogo",
        help="Path to NetLogo simulation results directory or CSV file",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for comparison report"
    ),
    metrics: str = typer.Option(
        "all", "--metrics", help="Metrics to compare: population,foraging,mortality,all"
    ),
    tolerance: float = typer.Option(
        0.05,
        "--tolerance",
        help="Relative tolerance for accuracy validation (0.05 = 5%)",
    ),
    detailed: bool = typer.Option(
        False, "--detailed", help="Generate detailed statistical analysis"
    ),
    plot: bool = typer.Option(False, "--plot", help="Generate comparison plots"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Compare BSTEW vs NetLogo simulation results for validation"""

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    console.print("🔍 [bold blue]NetLogo vs BSTEW Comparison[/bold blue]")
    console.print(f"📊 BSTEW Results: {bstew_results}")
    console.print(f"📈 NetLogo Results: {netlogo_results}")
    console.print(f"📏 Tolerance: ±{tolerance * 100:.1f}%")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load and process data
            task1 = progress.add_task("Loading simulation results...", total=None)

            bstew_data = _load_bstew_results(bstew_results)
            netlogo_data = _load_netlogo_results(netlogo_results)

            progress.update(task1, description="Preprocessing data...")

            # Align datasets for comparison
            aligned_data = _align_datasets(bstew_data, netlogo_data)

            progress.update(task1, description="Running behavioral comparisons...")

            # Determine which metrics to compare
            comparison_metrics = _parse_metrics_option(metrics)

            # Run comparisons
            comparison_results = {}
            for metric in comparison_metrics:
                if metric == "population":
                    comparison_results["population"] = _compare_population_dynamics(
                        aligned_data, tolerance
                    )
                elif metric == "foraging":
                    comparison_results["foraging"] = _compare_foraging_behavior(
                        aligned_data, tolerance
                    )
                elif metric == "mortality":
                    comparison_results["mortality"] = _compare_mortality_rates(
                        aligned_data, tolerance
                    )

            progress.update(task1, description="Computing statistical metrics...")

            # Calculate overall statistical accuracy
            overall_accuracy = _calculate_overall_accuracy(comparison_results)

            progress.advance(task1)

        # Display results
        _display_comparison_results(comparison_results, overall_accuracy, detailed)

        # Generate plots if requested
        if plot:
            console.print("📊 Generating comparison plots...")
            _generate_comparison_plots(aligned_data, comparison_results, output)

        # Save detailed report if requested
        if output:
            _save_comparison_report(
                comparison_results, overall_accuracy, output, detailed
            )
            console.print(f"💾 Comparison report saved to: [green]{output}[/green]")

        # Determine overall validation status
        if overall_accuracy >= 0.95:
            console.print(
                "✅ [green]EXCELLENT: Mathematical accuracy validation PASSED[/green]"
            )
            console.print("   BSTEW shows >95% behavioral equivalence with NetLogo")
        elif overall_accuracy >= 0.90:
            console.print(
                "✅ [yellow]GOOD: Mathematical accuracy validation PASSED[/yellow]"
            )
            console.print("   BSTEW shows >90% behavioral equivalence with NetLogo")
        elif overall_accuracy >= 0.80:
            console.print(
                "⚠️  [yellow]ACCEPTABLE: Some behavioral differences detected[/yellow]"
            )
            console.print("   BSTEW shows >80% behavioral equivalence with NetLogo")
        else:
            console.print(
                "❌ [red]FAILED: Significant behavioral differences detected[/red]"
            )
            console.print("   BSTEW shows <80% behavioral equivalence with NetLogo")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Error during comparison: [red]{e}[/red]")
        if verbose:
            import traceback

            console.print(f"[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)

    console.print("✅ [green]NetLogo comparison completed successfully![/green]")


class ParseType(str, Enum):
    """NetLogo parse types"""

    rotation = "rotation"
    species = "species"
    config = "config"
    all = "all"


@app.command()
def parse(
    input_file: Path = typer.Argument(
        ..., help="Input file or directory containing NetLogo data"
    ),
    parse_type: Optional[ParseType] = typer.Option(
        None,
        "--type",
        "-t",
        help="Type of data to parse (rotation, species, config, all)",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    show_samples: bool = typer.Option(
        False, "--show-samples", help="Show sample data from parsed files"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Parse NetLogo data files and convert to structured format"""

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    console.print("🔍 [bold blue]NetLogo Data Parser[/bold blue]")

    # Determine if input is file or directory
    is_directory = input_file.is_dir()

    # Determine parse type based on file extension or input argument
    if parse_type is None:
        if input_file.name.endswith(("crop_rotation.txt", "rotation.txt")):
            parse_type = ParseType.rotation
        elif input_file.name.endswith(("BumbleSpecies", "species.csv")):
            parse_type = ParseType.species
        elif input_file.name.endswith(("Parameters.csv", "config.yaml", "config.yml")):
            parse_type = ParseType.config
        elif is_directory:
            parse_type = ParseType.all
        else:
            parse_type = ParseType.all

    console.print(f"📂 Input: {input_file}")
    console.print(f"🔧 Parse Type: {parse_type.value}")

    parser = NetLogoDataParser()
    netlogo_data = {}

    try:
        if parse_type == ParseType.all and is_directory:
            # Parse all data files in directory
            console.print("🔄 Parsing all NetLogo data files...")
            netlogo_data = parser.parse_all_data_files(str(input_file))

            # Create summary table
            from rich.table import Table

            table = Table(title="📊 Parsing Results")
            table.add_column("Data Type", style="cyan")
            table.add_column("Count", style="magenta")

            table.add_row("Parameters", str(len(netlogo_data.get("parameters", {}))))
            table.add_row("Species", str(len(netlogo_data.get("species", {}))))
            table.add_row("Flowers", str(len(netlogo_data.get("flowers", {}))))
            table.add_row(
                "Food Sources", str(len(netlogo_data.get("food_sources", {})))
            )
            table.add_row("Habitats", str(len(netlogo_data.get("habitats", {}))))

            console.print(table)

        elif parse_type == ParseType.rotation:
            # Parse crop rotation data
            console.print("🌾 Parsing crop rotation data...")
            netlogo_data = _parse_crop_rotation(parser, str(input_file))
            console.print(f"✅ Parsed crop rotation data from {input_file}")

        elif parse_type == ParseType.species:
            # Parse species data
            console.print("🐝 Parsing species data...")
            if input_file.suffix.lower() == ".csv":
                species_data = parser.species_parser.parse_species_file(str(input_file))
                netlogo_data = {"species": species_data}
                console.print(
                    f"✅ Parsed {len(species_data)} species from {input_file}"
                )
            else:
                raise typer.BadParameter("Species files must be CSV format")

        elif parse_type == ParseType.config:
            # Parse config/parameters data
            console.print("⚙️  Parsing config/parameters data...")
            if input_file.suffix.lower() == ".csv":
                params_data = parser.parameter_parser.parse_parameters_file(
                    str(input_file)
                )
                netlogo_data = {"parameters": params_data}
                console.print(
                    f"✅ Parsed {len(params_data)} parameters from {input_file}"
                )
            elif input_file.suffix.lower() in [".yaml", ".yml"]:
                # Handle YAML config files
                with open(input_file, "r") as f:
                    config_data = yaml.safe_load(f)
                netlogo_data = {"config": config_data}
                console.print(f"✅ Parsed YAML config from {input_file}")
            else:
                raise typer.BadParameter("Config files must be CSV or YAML format")

        # Save output if requested
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)

            if output.suffix.lower() == ".json":
                with open(output, "w") as f:
                    json.dump(netlogo_data, f, indent=2, default=str)
            elif output.suffix.lower() in [".yaml", ".yml"]:
                with open(output, "w") as f:
                    yaml.dump(netlogo_data, f, default_flow_style=False)
            else:
                raise typer.BadParameter(f"Unsupported output format: {output.suffix}")

            console.print(f"💾 Data saved to: [green]{output}[/green]")

        # Show sample data if requested
        if show_samples:
            _show_sample_data(netlogo_data, console, parse_type)

    except Exception as e:
        console.print(f"❌ Error parsing NetLogo data: [red]{e}[/red]")
        raise typer.Exit(1)

    console.print("✅ [green]NetLogo data parsing completed successfully![/green]")


def _parse_crop_rotation(parser: NetLogoDataParser, file_path: str) -> Dict[str, Any]:
    """Parse crop rotation data from NetLogo file"""
    crop_data: Dict[str, Any] = {"crop_rotation": []}

    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                # Parse crop rotation format: crop_type,start_day,end_day,yield_multiplier
                parts = line.split(",")

                if len(parts) >= 3:
                    try:
                        crop_entry = {
                            "crop_type": parts[0].strip(),
                            "start_day": int(parts[1].strip()) if len(parts) > 1 else 1,
                            "end_day": int(parts[2].strip()) if len(parts) > 2 else 365,
                            "yield_multiplier": float(parts[3].strip())
                            if len(parts) > 3
                            else 1.0,
                            "line_number": line_num,
                        }
                        crop_data["crop_rotation"].append(crop_entry)

                    except ValueError as e:
                        console.print(
                            f"⚠️  Warning: Error parsing line {line_num}: {e}",
                            style="yellow",
                        )
                        continue

        crop_data["metadata"] = {
            "total_crops": len(crop_data["crop_rotation"]),
            "source_file": file_path,
        }

    except FileNotFoundError:
        raise typer.BadParameter(f"Crop rotation file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing crop rotation file {file_path}: {e}")

    return crop_data


def _show_sample_data(
    netlogo_data: Dict[str, Any], console: Console, parse_type: ParseType
) -> None:
    """Show sample data based on parse type"""
    console.print("📋 [bold blue]Sample Data:[/bold blue]")

    if parse_type == ParseType.rotation:
        # Show crop rotation samples
        crops = netlogo_data.get("crop_rotation", [])
        if crops:
            console.print("  🌾 Crop Rotation (first 3):")
            for crop in crops[:3]:
                console.print(
                    f"    {crop.get('crop_type')}: days {crop.get('start_day')}-{crop.get('end_day')}"
                )

    elif parse_type == ParseType.species:
        # Show species samples
        species = netlogo_data.get("species", {})
        if species:
            sample_species = list(species.items())[:2]
            console.print("  🐝 Species (first 2):")
            for name, spec in sample_species:
                console.print(
                    f"    {name}: {spec.species_id if hasattr(spec, 'species_id') else name}"
                )

    elif parse_type == ParseType.config:
        # Show config/parameter samples
        params = netlogo_data.get("parameters", {})
        config = netlogo_data.get("config", {})

        if params:
            sample_params = list(params.items())[:3]
            console.print("  ⚙️  Parameters (first 3):")
            for name, param in sample_params:
                value = param.value if hasattr(param, "value") else param
                console.print(f"    {name}: {value}")

        if config:
            sample_config = list(config.items())[:3]
            console.print("  🔧 Config (first 3):")
            for key, value in sample_config:
                console.print(f"    {key}: {value}")

    else:  # ParseType.all
        # Sample parameters
        params = netlogo_data.get("parameters", {})
        if params:
            sample_params = list(params.items())[:3]
            console.print("  ⚙️  Parameters (first 3):")
            for name, param in sample_params:
                value = param.value if hasattr(param, "value") else param
                console.print(f"    {name}: {value}")

        # Sample species
        species = netlogo_data.get("species", {})
        if species:
            sample_species = list(species.items())[:2]
            console.print("  🐝 Species (first 2):")
            for name, spec in sample_species:
                console.print(
                    f"    {name}: {spec.species_id if hasattr(spec, 'species_id') else name}"
                )

        # Sample flowers
        flowers = netlogo_data.get("flowers", {})
        if flowers:
            sample_flowers = list(flowers.items())[:3]
            console.print("  🌸 Flowers (first 3):")
            for name, flower in sample_flowers:
                depth = (
                    flower.corolla_depth_mm
                    if hasattr(flower, "corolla_depth_mm")
                    else "N/A"
                )
                console.print(f"    {name}: depth={depth}mm")


# Helper functions for NetLogo comparison
def _load_bstew_results(results_path: Path) -> pd.DataFrame:
    """Load BSTEW simulation results from file or directory"""
    if results_path.is_file():
        if results_path.suffix.lower() == ".csv":
            return pd.read_csv(results_path)
        elif results_path.suffix.lower() == ".json":
            data = json.loads(results_path.read_text())
            # Convert JSON to DataFrame format expected by comparison functions
            if isinstance(data, list):
                return pd.DataFrame(data)
            else:
                # Handle nested JSON structure
                return pd.json_normalize(data)
        else:
            raise typer.BadParameter(
                f"Unsupported BSTEW results format: {results_path.suffix}"
            )

    elif results_path.is_dir():
        # Look for common BSTEW output files
        csv_files = list(results_path.glob("*.csv"))
        if csv_files:
            # Use the first CSV file found, or combine multiple
            if len(csv_files) == 1:
                return pd.read_csv(csv_files[0])
            else:
                # Combine multiple CSV files
                combined_data = []
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    df["source_file"] = csv_file.name
                    combined_data.append(df)
                return pd.concat(combined_data, ignore_index=True)
        else:
            raise typer.BadParameter(
                f"No CSV files found in BSTEW results directory: {results_path}"
            )

    else:
        raise typer.BadParameter(f"BSTEW results path does not exist: {results_path}")


def _load_netlogo_results(results_path: Path) -> pd.DataFrame:
    """Load NetLogo simulation results from file or directory"""
    if results_path.is_file():
        if results_path.suffix.lower() == ".csv":
            return pd.read_csv(results_path)
        else:
            raise typer.BadParameter(
                f"Unsupported NetLogo results format: {results_path.suffix}"
            )

    elif results_path.is_dir():
        # Look for NetLogo output files
        csv_files = list(results_path.glob("*.csv"))
        if csv_files:
            if len(csv_files) == 1:
                return pd.read_csv(csv_files[0])
            else:
                # Look for specific NetLogo output files
                behaviorspace_files = [
                    f for f in csv_files if "behaviorspace" in f.name.lower()
                ]
                if behaviorspace_files:
                    return pd.read_csv(behaviorspace_files[0])
                else:
                    return pd.read_csv(csv_files[0])
        else:
            raise typer.BadParameter(
                f"No CSV files found in NetLogo results directory: {results_path}"
            )

    else:
        raise typer.BadParameter(f"NetLogo results path does not exist: {results_path}")


def _align_datasets(
    bstew_data: pd.DataFrame, netlogo_data: pd.DataFrame
) -> Dict[str, Any]:
    """Align BSTEW and NetLogo datasets for comparison"""
    aligned: Dict[str, Any] = {}

    # Standardize column names for comparison
    bstew_cols = _standardize_column_names(bstew_data.columns.tolist(), "bstew")
    netlogo_cols = _standardize_column_names(netlogo_data.columns.tolist(), "netlogo")

    # Find common metrics between datasets
    common_metrics = set(bstew_cols.keys()) & set(netlogo_cols.keys())

    if not common_metrics:
        console.print("⚠️  Warning: No common metrics found between datasets")
        console.print(f"BSTEW columns: {list(bstew_cols.keys())[:5]}...")
        console.print(f"NetLogo columns: {list(netlogo_cols.keys())[:5]}...")

    # Align data on common time dimension (usually 'day' or 'step')
    time_col_bstew: Optional[str] = None
    time_col_netlogo: Optional[str] = None

    for col in bstew_data.columns:
        if col.lower() in ["day", "step", "tick", "time"]:
            time_col_bstew = col
            break

    for col in netlogo_data.columns:
        if col.lower() in ["day", "step", "tick", "time"]:
            time_col_netlogo = col
            break

    aligned["bstew_data"] = bstew_data
    aligned["netlogo_data"] = netlogo_data
    aligned["common_metrics"] = list(common_metrics)
    aligned["time_col_bstew"] = time_col_bstew
    aligned["time_col_netlogo"] = time_col_netlogo
    aligned["bstew_cols"] = bstew_cols
    aligned["netlogo_cols"] = netlogo_cols

    return aligned


def _standardize_column_names(columns: List[str], source: str) -> Dict[str, str]:
    """Standardize column names for comparison between BSTEW and NetLogo"""
    standardized = {}

    for col in columns:
        col_lower = col.lower().replace("_", "").replace("-", "").replace(" ", "")

        # Population metrics
        if (
            "population" in col_lower
            or "totalbees" in col_lower
            or "total_bees" in col_lower
        ):
            standardized["total_population"] = col
        elif "queens" in col_lower:
            standardized["queens"] = col
        elif "workers" in col_lower:
            standardized["workers"] = col
        elif "foragers" in col_lower:
            standardized["foragers"] = col
        elif "eggs" in col_lower:
            standardized["eggs"] = col
        elif "larvae" in col_lower:
            standardized["larvae"] = col
        elif "pupae" in col_lower:
            standardized["pupae"] = col

        # Foraging metrics
        elif "nectar" in col_lower and (
            "collected" in col_lower or "total" in col_lower
        ):
            standardized["nectar_collected"] = col
        elif "pollen" in col_lower and (
            "collected" in col_lower or "total" in col_lower
        ):
            standardized["pollen_collected"] = col
        elif "foraging" in col_lower and "trips" in col_lower:
            standardized["foraging_trips"] = col

        # Mortality metrics
        elif "deaths" in col_lower or "mortality" in col_lower:
            standardized["deaths"] = col
        elif "died" in col_lower:
            standardized["deaths"] = col

    return standardized


def _parse_metrics_option(metrics_str: str) -> List[str]:
    """Parse the metrics option string into list of metrics"""
    if metrics_str.lower() == "all":
        return ["population", "foraging", "mortality"]
    else:
        return [m.strip().lower() for m in metrics_str.split(",")]


def _compare_population_dynamics(
    aligned_data: Dict[str, Any], tolerance: float
) -> Dict[str, Any]:
    """Compare population dynamics between BSTEW and NetLogo"""
    bstew_data = aligned_data["bstew_data"]
    netlogo_data = aligned_data["netlogo_data"]
    common_metrics = aligned_data["common_metrics"]
    bstew_cols = aligned_data["bstew_cols"]
    netlogo_cols = aligned_data["netlogo_cols"]

    results: Dict[str, Any] = {"metric_type": "population", "comparisons": []}

    # Compare total population
    if "total_population" in common_metrics:
        bstew_col = bstew_cols["total_population"]
        netlogo_col = netlogo_cols["total_population"]

        comparison = _compute_statistical_comparison(
            bstew_data[bstew_col].dropna(),
            netlogo_data[netlogo_col].dropna(),
            "Total Population",
            tolerance,
        )
        results["comparisons"].append(comparison)

    # Compare life stages
    for stage in ["queens", "workers", "foragers", "eggs", "larvae", "pupae"]:
        if stage in common_metrics:
            bstew_col = bstew_cols[stage]
            netlogo_col = netlogo_cols[stage]

            comparison = _compute_statistical_comparison(
                bstew_data[bstew_col].dropna(),
                netlogo_data[netlogo_col].dropna(),
                stage.title(),
                tolerance,
            )
            results["comparisons"].append(comparison)

    return results


def _compare_foraging_behavior(
    aligned_data: Dict[str, Any], tolerance: float
) -> Dict[str, Any]:
    """Compare foraging behavior between BSTEW and NetLogo"""
    bstew_data = aligned_data["bstew_data"]
    netlogo_data = aligned_data["netlogo_data"]
    common_metrics = aligned_data["common_metrics"]
    bstew_cols = aligned_data["bstew_cols"]
    netlogo_cols = aligned_data["netlogo_cols"]

    results: Dict[str, Any] = {"metric_type": "foraging", "comparisons": []}

    # Compare resource collection
    for resource in ["nectar_collected", "pollen_collected", "foraging_trips"]:
        if resource in common_metrics:
            bstew_col = bstew_cols[resource]
            netlogo_col = netlogo_cols[resource]

            comparison = _compute_statistical_comparison(
                bstew_data[bstew_col].dropna(),
                netlogo_data[netlogo_col].dropna(),
                resource.replace("_", " ").title(),
                tolerance,
            )
            results["comparisons"].append(comparison)

    return results


def _compare_mortality_rates(
    aligned_data: Dict[str, Any], tolerance: float
) -> Dict[str, Any]:
    """Compare mortality rates between BSTEW and NetLogo"""
    bstew_data = aligned_data["bstew_data"]
    netlogo_data = aligned_data["netlogo_data"]
    common_metrics = aligned_data["common_metrics"]
    bstew_cols = aligned_data["bstew_cols"]
    netlogo_cols = aligned_data["netlogo_cols"]

    results: Dict[str, Any] = {"metric_type": "mortality", "comparisons": []}

    if "deaths" in common_metrics:
        bstew_col = bstew_cols["deaths"]
        netlogo_col = netlogo_cols["deaths"]

        comparison = _compute_statistical_comparison(
            bstew_data[bstew_col].dropna(),
            netlogo_data[netlogo_col].dropna(),
            "Deaths",
            tolerance,
        )
        results["comparisons"].append(comparison)

    return results


def _compute_statistical_comparison(
    bstew_series: pd.Series,
    netlogo_series: pd.Series,
    metric_name: str,
    tolerance: float,
) -> Dict[str, Any]:
    """Compute statistical comparison between two data series"""

    # Ensure series have same length for comparison
    min_length = min(len(bstew_series), len(netlogo_series))
    bstew_values = bstew_series.iloc[:min_length].values
    netlogo_values = netlogo_series.iloc[:min_length].values

    # Calculate statistical metrics
    rmse = np.sqrt(mean_squared_error(netlogo_values, bstew_values))
    mae = mean_absolute_error(netlogo_values, bstew_values)
    r2 = r2_score(netlogo_values, bstew_values) if len(set(netlogo_values)) > 1 else 0.0
    correlation, p_value = (
        stats.pearsonr(bstew_values.astype(float), netlogo_values.astype(float))
        if len(bstew_values) > 1
        else (0.0, 1.0)
    )

    # Calculate relative error
    netlogo_mean = (
        float(np.mean(netlogo_values.astype(float))) if len(netlogo_values) > 0 else 0.0
    )
    relative_rmse = rmse / netlogo_mean if netlogo_mean != 0 else float("inf")

    # Determine if comparison passes tolerance
    passes_tolerance = relative_rmse <= tolerance

    # Calculate accuracy score (1 - relative_rmse, capped at 0)
    accuracy_score = max(0.0, 1.0 - relative_rmse)

    return {
        "metric_name": metric_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": correlation,
        "p_value": p_value,
        "relative_rmse": relative_rmse,
        "passes_tolerance": passes_tolerance,
        "accuracy_score": accuracy_score,
        "bstew_mean": float(np.mean(bstew_values.astype(float))),
        "netlogo_mean": netlogo_mean,
        "bstew_std": float(np.std(bstew_values.astype(float))),
        "netlogo_std": float(np.std(netlogo_values.astype(float))),
        "sample_size": min_length,
    }


def _calculate_overall_accuracy(comparison_results: Dict[str, Any]) -> float:
    """Calculate overall accuracy score across all comparisons"""
    all_scores = []

    for metric_type, results in comparison_results.items():
        for comparison in results["comparisons"]:
            all_scores.append(comparison["accuracy_score"])

    if not all_scores:
        return 0.0

    return float(np.mean(all_scores))


def _display_comparison_results(
    comparison_results: Dict[str, Any], overall_accuracy: float, detailed: bool
) -> None:
    """Display comparison results in formatted tables"""

    console.print("\n📊 [bold blue]Comparison Results Summary[/bold blue]")
    console.print(f"🎯 Overall Accuracy: {overall_accuracy:.1%}")

    # Create summary table
    summary_table = Table(title="📋 Behavioral Comparison Summary")
    summary_table.add_column("Metric Type", style="cyan")
    summary_table.add_column("Comparisons", style="white")
    summary_table.add_column("Avg Accuracy", style="green")
    summary_table.add_column("Status", style="magenta")

    for metric_type, results in comparison_results.items():
        comparisons = results["comparisons"]
        if comparisons:
            avg_accuracy = np.mean([c["accuracy_score"] for c in comparisons])
            status = (
                "✅ PASS"
                if avg_accuracy >= 0.8
                else "⚠️ WARN"
                if avg_accuracy >= 0.6
                else "❌ FAIL"
            )

            summary_table.add_row(
                metric_type.title(),
                str(len(comparisons)),
                f"{avg_accuracy:.1%}",
                status,
            )

    console.print(summary_table)

    if detailed:
        # Show detailed results for each metric type
        for metric_type, results in comparison_results.items():
            console.print(
                f"\n📈 [bold blue]{metric_type.title()} Detailed Results[/bold blue]"
            )

            detailed_table = Table(title=f"{metric_type.title()} Comparison Details")
            detailed_table.add_column("Metric", style="cyan")
            detailed_table.add_column("BSTEW Mean", style="green")
            detailed_table.add_column("NetLogo Mean", style="blue")
            detailed_table.add_column("RMSE", style="yellow")
            detailed_table.add_column("R²", style="magenta")
            detailed_table.add_column("Accuracy", style="white")
            detailed_table.add_column("Status", style="red")

            for comparison in results["comparisons"]:
                status = "✅" if comparison["passes_tolerance"] else "❌"

                detailed_table.add_row(
                    comparison["metric_name"],
                    f"{comparison['bstew_mean']:.2f}",
                    f"{comparison['netlogo_mean']:.2f}",
                    f"{comparison['rmse']:.3f}",
                    f"{comparison['r2']:.3f}",
                    f"{comparison['accuracy_score']:.1%}",
                    status,
                )

            console.print(detailed_table)


def _generate_comparison_plots(
    aligned_data: Dict[str, Any],
    comparison_results: Dict[str, Any],
    output: Optional[Path],
) -> None:
    """Generate comparison plots (placeholder - would require matplotlib)"""
    console.print("📊 Plot generation not implemented in this version")
    console.print(
        "   Consider using external plotting tools with the saved comparison data"
    )


def _save_comparison_report(
    comparison_results: Dict[str, Any],
    overall_accuracy: float,
    output: Path,
    detailed: bool,
) -> None:
    """Save detailed comparison report to file"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_accuracy": overall_accuracy,
        "summary": {
            "total_comparisons": sum(
                len(results["comparisons"]) for results in comparison_results.values()
            ),
            "passed_comparisons": sum(
                sum(1 for c in results["comparisons"] if c["passes_tolerance"])
                for results in comparison_results.values()
            ),
        },
        "detailed_results": comparison_results if detailed else {},
        "metadata": {
            "version": "1.0.0",
            "tolerance_used": 0.05,
            "report_type": "detailed" if detailed else "summary",
        },
    }

    if output.suffix.lower() == ".json":
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
    elif output.suffix.lower() in [".yaml", ".yml"]:
        with open(output, "w") as f:
            yaml.dump(report, f, default_flow_style=False)
    else:
        # Default to JSON
        output_json = output.with_suffix(".json")
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)


@app.command()
def map(
    netlogo_params: Path = typer.Argument(
        ...,
        help="Path to NetLogo parameters file (CSV or YAML)",
        exists=True,
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, csv, json, yaml",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (if not specified, prints to console)",
    ),
    show_unmapped: bool = typer.Option(
        True,
        "--show-unmapped/--hide-unmapped",
        help="Show parameters that couldn't be mapped",
    ),
    species: Optional[str] = typer.Option(
        None,
        "--species",
        "-s",
        help="Filter mapping for specific species",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed mapping information including units and validation",
    ),
) -> None:
    """Map NetLogo parameters to BSTEW configuration format.

    This command helps translate NetLogo BEE-STEWARD parameters to their
    BSTEW equivalents, showing parameter correspondences and identifying
    any unmapped parameters that need manual configuration.

    Examples:
        # Show parameter mapping as a table
        bstew netlogo map netlogo_params.csv

        # Export mapping to CSV
        bstew netlogo map netlogo_params.yaml --format csv --output mapping.csv

        # Show detailed mapping with units and validation
        bstew netlogo map params.csv --detailed

        # Filter for specific species
        bstew netlogo map params.csv --species "B_terrestris"
    """
    from ...config.netlogo_mapping import NetLogoParameterMapper

    console.print("[bold blue]NetLogo to BSTEW Parameter Mapping[/bold blue]")
    console.print(f"Input: {netlogo_params}")

    try:
        # Initialize mapper
        mapper = NetLogoParameterMapper()

        # Load NetLogo parameters
        netlogo_data = {}
        if netlogo_params.suffix.lower() == ".csv":
            # Load CSV parameters
            df = pd.read_csv(netlogo_params)
            # Convert to dict format expected by mapper
            for _, row in df.iterrows():
                if "parameter" in df.columns and "value" in df.columns:
                    netlogo_data[row["parameter"]] = row["value"]
                else:
                    # Assume first column is parameter, second is value
                    netlogo_data[row.iloc[0]] = row.iloc[1]
        elif netlogo_params.suffix.lower() in [".yaml", ".yml"]:
            with open(netlogo_params, "r") as f:
                netlogo_data = yaml.safe_load(f)
        else:
            with open(netlogo_params, "r") as f:
                netlogo_data = json.load(f)

        # Perform mapping
        mapped_params = {}
        unmapped_params = []
        mapping_details = []

        # Map system parameters
        for param_name, param_value in netlogo_data.items():
            if param_name in mapper.parameter_mappings:
                mapping = mapper.parameter_mappings[param_name]

                # Apply species filter if specified
                if (
                    species
                    and hasattr(mapping, "species")
                    and mapping.species != species
                ):
                    continue

                try:
                    converted_value = mapping.convert_value(param_value)
                    mapped_params[mapping.bstew_path] = converted_value

                    mapping_details.append(
                        {
                            "netlogo_param": param_name,
                            "netlogo_value": param_value,
                            "bstew_path": mapping.bstew_path,
                            "bstew_value": converted_value,
                            "type": mapping.parameter_type.value,
                            "unit": mapping.unit_type.value,
                            "conversion_factor": mapping.conversion_factor,
                            "description": mapping.description,
                        }
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to map {param_name}: {e}[/yellow]"
                    )
                    unmapped_params.append(
                        {
                            "parameter": param_name,
                            "value": param_value,
                            "reason": str(e),
                        }
                    )
            else:
                unmapped_params.append(
                    {
                        "parameter": param_name,
                        "value": param_value,
                        "reason": "No mapping defined",
                    }
                )

        # Display results based on format
        if output_format == "table":
            _display_mapping_table(
                mapping_details, unmapped_params, detailed, show_unmapped
            )

        # Generate output
        output_data = {
            "mapped_parameters": mapping_details,
            "unmapped_parameters": unmapped_params if show_unmapped else [],
            "summary": {
                "total_parameters": len(netlogo_data),
                "mapped_count": len(mapping_details),
                "unmapped_count": len(unmapped_params),
                "mapping_rate": f"{len(mapping_details) / len(netlogo_data) * 100:.1f}%",
            },
            "metadata": {
                "source_file": str(netlogo_params),
                "timestamp": datetime.now().isoformat(),
                "mapper_version": "1.0.0",
            },
        }

        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_format == "csv" or output_file.suffix.lower() == ".csv":
                # Convert to CSV format
                df = pd.DataFrame(mapping_details)
                df.to_csv(output_file, index=False)
                console.print(f"✅ Mapping saved to: [green]{output_file}[/green]")
            elif output_format == "json" or output_file.suffix.lower() == ".json":
                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                console.print(f"✅ Mapping saved to: [green]{output_file}[/green]")
            elif output_format == "yaml" or output_file.suffix.lower() in [
                ".yaml",
                ".yml",
            ]:
                with open(output_file, "w") as f:
                    yaml.dump(output_data, f, default_flow_style=False)
                console.print(f"✅ Mapping saved to: [green]{output_file}[/green]")

        # Show summary
        console.print("\n[bold]Mapping Summary:[/bold]")
        console.print(f"  Total parameters: {len(netlogo_data)}")
        console.print(f"  [green]Mapped: {len(mapping_details)}[/green]")
        console.print(f"  [yellow]Unmapped: {len(unmapped_params)}[/yellow]")
        console.print(
            f"  Mapping rate: {len(mapping_details) / len(netlogo_data) * 100:.1f}%"
        )

    except Exception as e:
        console.print(f"❌ Error mapping parameters: [red]{e}[/red]")
        raise typer.Exit(1)


def _display_mapping_table(
    mapping_details: List[Dict[str, Any]],
    unmapped_params: List[Dict[str, Any]],
    detailed: bool,
    show_unmapped: bool,
) -> None:
    """Display parameter mapping as a formatted table"""

    # Create mapping table
    table = Table(title="Parameter Mapping", show_header=True, header_style="bold")
    table.add_column("NetLogo Parameter", style="cyan")
    table.add_column("NetLogo Value", style="yellow")
    table.add_column("BSTEW Path", style="green")
    table.add_column("BSTEW Value", style="green")

    if detailed:
        table.add_column("Type", style="blue")
        table.add_column("Unit", style="blue")
        table.add_column("Factor", style="blue")
        table.add_column("Description", style="dim")

    for detail in mapping_details:
        row = [
            detail["netlogo_param"],
            str(detail["netlogo_value"]),
            detail["bstew_path"],
            str(detail["bstew_value"]),
        ]

        if detailed:
            row.extend(
                [
                    detail["type"],
                    detail["unit"],
                    str(detail["conversion_factor"]),
                    detail["description"][:50] + "..."
                    if len(detail["description"]) > 50
                    else detail["description"],
                ]
            )

        table.add_row(*row)

    console.print(table)

    # Show unmapped parameters if requested
    if show_unmapped and unmapped_params:
        console.print("\n[bold yellow]Unmapped Parameters:[/bold yellow]")
        unmapped_table = Table(show_header=True, header_style="bold")
        unmapped_table.add_column("Parameter", style="yellow")
        unmapped_table.add_column("Value", style="dim")
        unmapped_table.add_column("Reason", style="red")

        for param in unmapped_params:
            unmapped_table.add_row(
                param["parameter"],
                str(param["value"]),
                param["reason"],
            )

        console.print(unmapped_table)


@app.command()
def test(
    test_suite: str = typer.Argument(
        "all",
        help="Test suite to run: all, population, foraging, mortality, spatial, reproduction",
    ),
    scenario_file: Optional[Path] = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Path to test scenario definition file",
    ),
    netlogo_data: Optional[Path] = typer.Option(
        None,
        "--netlogo-data",
        help="Path to NetLogo test data directory",
    ),
    bstew_config: Optional[Path] = typer.Option(
        None,
        "--bstew-config",
        help="Path to BSTEW configuration for test runs",
    ),
    output_dir: Path = typer.Option(
        Path("test_results"),
        "--output",
        "-o",
        help="Output directory for test results",
    ),
    tolerance: float = typer.Option(
        0.05,
        "--tolerance",
        "-t",
        help="Relative tolerance for numerical comparisons (0.05 = 5%)",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        "-p",
        help="Run tests in parallel",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop on first test failure",
    ),
    generate_plots: bool = typer.Option(
        True,
        "--plots/--no-plots",
        help="Generate comparison plots",
    ),
) -> None:
    """Run behavioral validation tests comparing NetLogo and BSTEW.

    This command executes predefined test scenarios to validate that BSTEW
    correctly replicates NetLogo BEE-STEWARD v2 behavior across different
    aspects of the simulation.

    Test Suites:
    - all: Run all available tests
    - population: Test population dynamics and demographics
    - foraging: Test foraging behavior and resource collection
    - mortality: Test mortality patterns and survival rates
    - spatial: Test spatial distribution and movement
    - reproduction: Test reproduction and colony growth

    Examples:
        # Run all tests with default settings
        bstew netlogo test

        # Run only population tests with custom tolerance
        bstew netlogo test population --tolerance 0.01

        # Run tests from scenario file
        bstew netlogo test --scenario scenarios/basic_tests.yaml

        # Run tests in parallel with custom output
        bstew netlogo test all --parallel --output my_test_results/
    """
    from concurrent.futures import ProcessPoolExecutor
    from ...validation.test_scenarios import ValidationScenarioLoader, ValidationRunner

    console.print("[bold blue]🧪 NetLogo Behavioral Validation Tests[/bold blue]")
    console.print(f"Test Suite: [cyan]{test_suite}[/cyan]")
    console.print(f"Tolerance: [yellow]±{tolerance * 100:.1f}%[/yellow]")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load test scenarios
        if scenario_file:
            console.print(f"📄 Loading test scenarios from: {scenario_file}")
            scenario_loader = ValidationScenarioLoader(scenario_file)
            scenarios = scenario_loader.load_scenarios()
        else:
            console.print("📄 Loading default test scenarios")
            scenarios = _get_default_test_scenarios(test_suite)

        if not scenarios:
            console.print("[red]❌ No test scenarios found![/red]")
            raise typer.Exit(1)

        console.print(f"✅ Loaded {len(scenarios)} test scenarios")

        # Initialize test runner
        runner = ValidationRunner(
            tolerance=tolerance,
            output_dir=output_dir,
            verbose=verbose,
            generate_plots=generate_plots,
        )

        # Configure data sources
        if netlogo_data:
            runner.set_netlogo_data_source(netlogo_data)
        if bstew_config:
            runner.set_bstew_config(bstew_config)

        # Run tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            if parallel and len(scenarios) > 1:
                # Run tests in parallel
                task = progress.add_task("Running tests...", total=len(scenarios))
                results = []

                with ProcessPoolExecutor() as executor:
                    futures = []
                    for scenario in scenarios:
                        future = executor.submit(runner.run_scenario, scenario)
                        futures.append((scenario, future))

                    for scenario, future in futures:
                        try:
                            result = future.result()
                            results.append(result)
                            progress.update(task, advance=1)

                            if fail_fast and not result.passed:
                                console.print(
                                    f"[red]❌ Test failed: {scenario.name}[/red]"
                                )
                                executor.shutdown(wait=False)
                                break
                        except Exception as e:
                            console.print(
                                f"[red]❌ Error in test {scenario.name}: {e}[/red]"
                            )
                            if fail_fast:
                                executor.shutdown(wait=False)
                                break
            else:
                # Run tests sequentially
                task = progress.add_task("Running tests...", total=len(scenarios))
                results = []

                for scenario in scenarios:
                    progress.update(task, description=f"Running: {scenario.name}")

                    try:
                        result = runner.run_scenario(scenario)
                        results.append(result)
                        progress.update(task, advance=1)

                        if fail_fast and not result.passed:
                            console.print(f"[red]❌ Test failed: {scenario.name}[/red]")
                            break
                    except Exception as e:
                        console.print(
                            f"[red]❌ Error in test {scenario.name}: {e}[/red]"
                        )
                        if fail_fast:
                            break

        # Generate test report
        console.print("\n[bold]📊 Generating Test Report[/bold]")
        report_path = runner.generate_report(results)

        # Display summary
        _display_test_summary(results, console)

        # Save detailed results
        results_file = output_dir / "test_results.json"
        _save_test_results(results, results_file)

        console.print(f"\n✅ Test results saved to: [green]{output_dir}[/green]")
        console.print(f"📄 Detailed report: [green]{report_path}[/green]")

        # Exit with appropriate code
        all_passed = all(r.passed for r in results)
        if not all_passed:
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Test execution failed: {e}[/red]")
        raise typer.Exit(1)


def _get_default_test_scenarios(test_suite: str) -> List[Any]:
    """Load default test scenarios for the specified suite"""
    from ...validation.test_scenarios import ValidationScenario

    scenarios = []

    # Population dynamics tests
    if test_suite in ["all", "population"]:
        scenarios.extend(
            [
                ValidationScenario(
                    name="population_growth_basic",
                    description="Test basic population growth dynamics",
                    category="population",
                    netlogo_params={
                        "BeeSpeciesInitialQueensListAsString": ["B_terrestris", 1],
                        "MaxForagingRange_m": 5000,
                        "Weather": "Constant 8 hrs",
                    },
                    bstew_config_overrides={
                        "simulation.duration_days": 180,
                        "colony.initial_queens": ["B_terrestris", 1],
                    },
                    metrics_to_compare=[
                        "total_population",
                        "egg_count",
                        "larva_count",
                        "pupa_count",
                        "adult_count",
                    ],
                    expected_patterns={
                        "population_increases": True,
                        "follows_growth_curve": True,
                    },
                ),
                ValidationScenario(
                    name="population_seasonal_variation",
                    description="Test seasonal population variations",
                    category="population",
                    netlogo_params={
                        "BeeSpeciesInitialQueensListAsString": ["B_terrestris", 1],
                        "Weather": "Variable",
                    },
                    bstew_config_overrides={
                        "simulation.duration_days": 365,
                        "environment.weather_type": "variable",
                    },
                    metrics_to_compare=[
                        "total_population",
                        "worker_count",
                        "drone_count",
                    ],
                    expected_patterns={
                        "seasonal_peak": "summer",
                        "winter_decline": True,
                    },
                ),
            ]
        )

    # Foraging behavior tests
    if test_suite in ["all", "foraging"]:
        scenarios.extend(
            [
                ValidationScenario(
                    name="foraging_efficiency_basic",
                    description="Test basic foraging efficiency",
                    category="foraging",
                    netlogo_params={
                        "MaxForagingRange_m": 3000,
                        "FoodSourceLimit": 10,
                    },
                    bstew_config_overrides={
                        "foraging.max_range_m": 3000,
                        "resources.food_source_limit": 10,
                    },
                    metrics_to_compare=[
                        "foraging_trips",
                        "successful_trips",
                        "energy_collected",
                        "foraging_efficiency",
                    ],
                    expected_patterns={
                        "efficiency_above_threshold": 0.7,
                    },
                ),
                ValidationScenario(
                    name="foraging_distance_impact",
                    description="Test impact of foraging distance on efficiency",
                    category="foraging",
                    netlogo_params={
                        "MaxForagingRange_m": 10000,
                        "Lambda_detectProb": -0.005,
                    },
                    bstew_config_overrides={
                        "foraging.max_range_m": 10000,
                        "detection.lambda_detect_prob": -0.005,
                    },
                    metrics_to_compare=[
                        "trip_duration",
                        "energy_expenditure",
                        "net_energy_gain",
                    ],
                    expected_patterns={
                        "distance_efficiency_tradeoff": True,
                    },
                ),
            ]
        )

    # Mortality tests
    if test_suite in ["all", "mortality"]:
        scenarios.extend(
            [
                ValidationScenario(
                    name="mortality_baseline",
                    description="Test baseline mortality rates",
                    category="mortality",
                    netlogo_params={
                        "ForagingMortalityFactor": 1.0,
                        "WinterMortality?": True,
                    },
                    bstew_config_overrides={
                        "mortality.foraging_factor": 1.0,
                        "mortality.winter_mortality_enabled": True,
                    },
                    metrics_to_compare=[
                        "daily_mortality_rate",
                        "forager_mortality",
                        "winter_survival_rate",
                    ],
                    expected_patterns={
                        "mortality_within_bounds": [0.001, 0.01],
                    },
                ),
            ]
        )

    # Spatial distribution tests
    if test_suite in ["all", "spatial"]:
        scenarios.extend(
            [
                ValidationScenario(
                    name="spatial_foraging_distribution",
                    description="Test spatial distribution of foraging activity",
                    category="spatial",
                    netlogo_params={
                        "Gridsize": 500,
                        "MaxPatchRadius_m": 500,
                    },
                    bstew_config_overrides={
                        "landscape.grid_size": 500,
                        "landscape.max_patch_radius_m": 500,
                    },
                    metrics_to_compare=[
                        "foraging_spatial_distribution",
                        "patch_visitation_frequency",
                        "resource_depletion_pattern",
                    ],
                    expected_patterns={
                        "follows_optimal_foraging": True,
                    },
                ),
            ]
        )

    # Reproduction tests
    if test_suite in ["all", "reproduction"]:
        scenarios.extend(
            [
                ValidationScenario(
                    name="reproduction_timing",
                    description="Test reproduction timing and patterns",
                    category="reproduction",
                    netlogo_params={
                        "SexLocus?": False,
                        "UnlimitedMales?": True,
                    },
                    bstew_config_overrides={
                        "genetics.csd_enabled": False,
                        "genetics.unlimited_males": True,
                    },
                    metrics_to_compare=[
                        "queen_production_timing",
                        "drone_production_timing",
                        "reproductive_success_rate",
                    ],
                    expected_patterns={
                        "reproduction_seasonal": True,
                    },
                ),
            ]
        )

    return scenarios


def _display_test_summary(results: List[Any], console: Console) -> None:
    """Display test results summary"""

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests

    console.print("\n[bold]Test Summary[/bold]")
    console.print("=" * 60)

    # Overall status
    if failed_tests == 0:
        console.print(
            f"✅ [green]All tests passed! ({passed_tests}/{total_tests})[/green]"
        )
    else:
        console.print(
            f"❌ [red]{failed_tests} tests failed ({passed_tests}/{total_tests} passed)[/red]"
        )

    # Category breakdown
    categories = {}
    for result in results:
        category = result.scenario.category
        if category not in categories:
            categories[category] = {"passed": 0, "failed": 0}

        if result.passed:
            categories[category]["passed"] += 1
        else:
            categories[category]["failed"] += 1

    console.print("\n[bold]Results by Category:[/bold]")
    for category, counts in sorted(categories.items()):
        total = counts["passed"] + counts["failed"]
        if counts["failed"] == 0:
            console.print(f"  {category}: [green]✓ {counts['passed']}/{total}[/green]")
        else:
            console.print(f"  {category}: [red]✗ {counts['passed']}/{total}[/red]")

    # Failed tests details
    if failed_tests > 0:
        console.print("\n[bold red]Failed Tests:[/bold red]")
        for result in results:
            if not result.passed:
                console.print(f"  • {result.scenario.name}: {result.failure_reason}")
                for metric in result.failed_metrics:
                    console.print(
                        f"    - {metric.name}: {metric.difference:.2%} > {metric.tolerance:.2%}"
                    )


def _save_test_results(results: List[Any], output_file: Path) -> None:
    """Save detailed test results to JSON file"""

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        },
        "results": [
            {
                "scenario": r.scenario.name,
                "category": r.scenario.category,
                "passed": r.passed,
                "execution_time": r.execution_time,
                "metrics": [
                    {
                        "name": m.name,
                        "passed": m.passed,
                        "netlogo_value": m.netlogo_value,
                        "bstew_value": m.bstew_value,
                        "difference": m.difference,
                        "tolerance": m.tolerance,
                    }
                    for m in r.metrics
                ],
                "failure_reason": r.failure_reason if not r.passed else None,
            }
            for r in results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)


if __name__ == "__main__":
    app()
