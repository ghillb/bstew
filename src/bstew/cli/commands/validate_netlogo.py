"""
NetLogo Validation CLI Commands
==============================

CLI commands for validating BSTEW behavior against NetLogo BEE-STEWARD v2.
"""

import typer
import logging
from pathlib import Path
import json
from typing import Optional
from rich.console import Console

from ...validation.netlogo_validation import NetLogoValidationSuite, NetLogoDataLoader
from ...core.data_collection import ComprehensiveDataCollector
from ...core.model import BeeModel

# Initialize console for rich output
console = Console()

# Create Typer app for NetLogo validation commands
app = typer.Typer(
    name="netlogo",
    help="NetLogo BEE-STEWARD v2 validation commands",
    add_completion=False
)


@app.command()
def validate(
    netlogo_data: Path = typer.Option(
        ..., 
        "--netlogo-data",
        help="Path to NetLogo output data directory",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    bstew_data: Optional[Path] = typer.Option(
        None,
        "--bstew-data", 
        help="Path to BSTEW data collector file (JSON)",
        exists=True
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to BSTEW configuration file",
        exists=True
    ),
    output: Path = typer.Option(
        "validation_results",
        "--output",
        help="Output directory for validation results"
    ),
    run_simulation: bool = typer.Option(
        False,
        "--run-simulation",
        help="Run a new BSTEW simulation for comparison"
    ),
    simulation_days: int = typer.Option(
        365,
        "--simulation-days",
        help="Number of days to simulate (if running new simulation)"
    ),
    tolerance_config: Optional[Path] = typer.Option(
        None,
        "--tolerance-config",
        help="Path to validation tolerance configuration (JSON)",
        exists=True
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """Validate BSTEW behavior against NetLogo BEE-STEWARD v2"""
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger(__name__)
    
    console.print("🧪 [bold blue]Starting NetLogo BEE-STEWARD v2 Validation[/bold blue]")
    
    try:
        # Load tolerance configuration if provided
        tolerance_config_dict = None
        if tolerance_config:
            with open(tolerance_config, 'r') as f:
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
            console.print("❌ [red]Must provide either --bstew-data or --run-simulation[/red]")
            raise typer.Exit(1)
        
        if not data_collector:
            console.print("❌ [red]Failed to obtain BSTEW data[/red]")
            raise typer.Exit(1)
        
        # Initialize validation suite
        console.print("🔧 Initializing validation suite...")
        validation_suite = NetLogoValidationSuite(
            netlogo_data_path=str(netlogo_data),
            output_path=str(output)
        )
        
        if tolerance_config_dict:
            validation_suite.validator.tolerance_config = tolerance_config_dict
        
        # Run validation
        console.print("⚡ Running behavioral validation...")
        validation_results = validation_suite.run_complete_validation(data_collector)
        
        # Print summary results
        _print_validation_summary(validation_results)
        
        console.print(f"✅ [green]Validation completed. Results saved to {output}[/green]")
        
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
        dir_okay=True
    ),
    output: Path = typer.Option(
        "netlogo_data_summary.json",
        "--output",
        help="Output file for data summary"
    )
):
    """Inspect available NetLogo data for validation"""
    
    console.print(f"🔍 [bold blue]Inspecting NetLogo data in {netlogo_data}[/bold blue]")
    
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
                    "metric_count": len(category_data)
                }
                
                # Add data shape information
                for metric, values in category_data.items():
                    if isinstance(values, list) and values:
                        summary[category][f"{metric}_length"] = len(values)
                        summary[category][f"{metric}_sample"] = values[:5] if len(values) >= 5 else values
            else:
                summary[category] = {"available": False}
        
        # Save summary
        with open(output, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary using Rich
        console.print("\n📊 [bold]NetLogo Data Summary:[/bold]")
        console.print("=" * 50)
        for category, info in summary.items():
            status = "✅" if info.get("available", False) else "❌"
            metric_count = info.get("metric_count", 0)
            console.print(f"{status} [cyan]{category}[/cyan]: {metric_count} metrics")
            
            if info.get("available") and info.get("metrics"):
                for metric in info["metrics"][:3]:  # Show first 3 metrics
                    console.print(f"    • {metric}")
                if len(info["metrics"]) > 3:
                    console.print(f"    ... and {len(info['metrics']) - 3} more")
        
        console.print(f"\n✅ [green]Detailed summary saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to inspect NetLogo data: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def generate_bstew_data(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to BSTEW configuration file",
        exists=True
    ),
    days: int = typer.Option(
        365,
        "--days",
        help="Number of days to simulate"
    ),
    output: Path = typer.Option(
        "bstew_validation_data",
        "--output",
        help="Output directory for BSTEW data"
    )
):
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
            
            console.print(f"✅ [green]BSTEW data generated and saved to {output}[/green]")
        else:
            console.print("❌ [red]Failed to generate BSTEW data[/red]")
            raise typer.Exit(1)
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"❌ [red]Failed to generate BSTEW data: {e}[/red]")
        raise typer.Exit(1)


def _run_bstew_simulation(config_path: Optional[Path], days: int) -> Optional[ComprehensiveDataCollector]:
    """Run BSTEW simulation and return data collector"""
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = None
        if config_path:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Create and run model
        model = BeeModel(config=config)
        
        # Enable comprehensive data collection
        if not hasattr(model, 'comprehensive_data_collector'):
            model.comprehensive_data_collector = ComprehensiveDataCollector()
        
        # Run simulation
        logger.info(f"Running simulation for {days} days...")
        for day in range(days):
            model.step()
            
            if day % 50 == 0:  # Progress update every 50 days
                logger.info(f"Simulation progress: {day}/{days} days")
        
        model.cleanup()
        
        # Return the data collector from system integrator if available
        if hasattr(model, 'system_integrator') and model.system_integrator:
            return model.system_integrator.data_collector
        else:
            return getattr(model, 'comprehensive_data_collector', None)
        
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


def _export_bstew_data(data_collector: ComprehensiveDataCollector, output_path: Path) -> None:
    """Export BSTEW data collector data to files"""
    
    logger = logging.getLogger(__name__)
    
    try:
        # Export population data
        pop_data = []
        for colony_id, metrics in data_collector.colony_metrics.items():
            if hasattr(metrics, 'population_size_day_list'):
                for day, population in enumerate(metrics.population_size_day_list):
                    pop_data.append({
                        'day': day,
                        'colony_id': colony_id,
                        'total_population': population,
                        'egg_count': metrics.egg_count_day_list[day] if day < len(metrics.egg_count_day_list) else 0,
                        'larva_count': metrics.larva_count_day_list[day] if day < len(metrics.larva_count_day_list) else 0,
                        'pupa_count': metrics.pupa_count_day_list[day] if day < len(metrics.pupa_count_day_list) else 0,
                        'adult_count': metrics.adult_count_day_list[day] if day < len(metrics.adult_count_day_list) else 0
                    })
        
        if pop_data:
            import pandas as pd
            pop_df = pd.DataFrame(pop_data)
            pop_df.to_csv(output_path / "population_data.csv", index=False)
        
        # Export foraging data
        foraging_data = []
        for bee_id, metrics in data_collector.bee_metrics.items():
            if hasattr(metrics, 'foraging_trips') and metrics.foraging_trips > 0:
                foraging_data.append({
                    'bee_id': bee_id,
                    'total_trips': metrics.foraging_trips,
                    'successful_trips': getattr(metrics, 'successful_trips', 0),
                    'foraging_efficiency': getattr(metrics, 'foraging_success_rate', 0),
                    'total_energy_collected': getattr(metrics, 'nectar_collected_total', 0) + getattr(metrics, 'pollen_collected_total', 0)
                })
        
        if foraging_data:
            import pandas as pd
            foraging_df = pd.DataFrame(foraging_data)
            foraging_df.to_csv(output_path / "foraging_data.csv", index=False)
        
        logger.info("BSTEW data exported successfully")
        
    except Exception as e:
        logger.error(f"Failed to export BSTEW data: {e}")


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
        console.print(f"    Pass Rate: {result.pass_rate:.1%} ({result.passed_metrics}/{result.total_metrics})")
        console.print(f"    Mean Difference: {result.summary_statistics.get('mean_relative_difference', 0):.3f}")
        
        total_metrics += result.total_metrics
        total_passed += result.passed_metrics
        
        # Show failed metrics
        failed_metrics = [m for m in result.individual_results if not m.passes_validation]
        if failed_metrics:
            console.print("    [red]Failed metrics:[/red]")
            for metric in failed_metrics[:3]:  # Show first 3 failed
                console.print(f"      • {metric.metric_name}: {metric.relative_difference:.3f} (>{metric.tolerance:.3f})")
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
    console.print(f"{overall_status} [bold]Overall Validation: {overall_rate:.1%} ({total_passed}/{total_metrics})[/bold]")
    console.print("=" * 60)


if __name__ == "__main__":
    app()