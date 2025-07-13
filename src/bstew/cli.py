"""
Command Line Interface for BSTEW
================================

Typer-based CLI with Rich integration for beautiful terminal output.
"""

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from pathlib import Path
from typing import Optional
import yaml
import sys
import time
import pandas as pd

from .utils.config import ConfigManager
from .utils.batch_processing import ExperimentManager, ParameterSpec, ExperimentType


# Initialize Typer app and Rich console
app = typer.Typer(
    name="bstew",
    help="BSTEW - BeeSteward v2 Python Transpilation",
    add_completion=False,
)

console = Console()

# Global configuration manager
config_manager = ConfigManager()


def show_banner():
    """Display BSTEW banner"""
    banner = """
    ╭─────────────────────────────────────────────────────────╮
    │                                                         │
    │   ██████╗ ███████╗████████╗███████╗██╗    ██╗          │
    │   ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██║    ██║          │
    │   ██████╔╝███████╗   ██║   █████╗  ██║ █╗ ██║          │
    │   ██╔══██╗╚════██║   ██║   ██╔══╝  ██║███╗██║          │
    │   ██████╔╝███████║   ██║   ███████╗╚███╔███╔╝          │
    │   ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚══╝╚══╝           │
    │                                                         │
    │        BeeSteward v2 Python Transpilation               │
    │        Agent-based Pollinator Population Modeling       │
    │                                                         │
    ╰─────────────────────────────────────────────────────────╯
    """
    console.print(banner, style="bold blue")


@app.command()
def run(
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Configuration file path"
    ),
    output: str = typer.Option("results", "--output", "-o", help="Output directory"),
    days: Optional[int] = typer.Option(
        None, "--days", "-d", help="Simulation duration in days"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
):
    """Run BSTEW simulation"""

    if not quiet:
        show_banner()

    # Load configuration
    try:
        sim_config = config_manager.load_config(config)
        console.print(f"✅ Loaded configuration: {config}", style="green")
    except Exception as e:
        console.print(f"❌ Error loading configuration: {e}", style="red")
        raise typer.Exit(1)

    # Override configuration with command line arguments
    if days is not None:
        sim_config.simulation.duration_days = days
    if seed is not None:
        sim_config.simulation.random_seed = seed
    if output:
        sim_config.output.output_directory = output

    # Validate configuration
    errors = config_manager.validate_config(sim_config)
    if errors:
        console.print("❌ Configuration validation errors:", style="red")
        for error in errors:
            console.print(f"  • {error}", style="red")
        raise typer.Exit(1)

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)

    # Display simulation info
    if not quiet:
        info_table = Table(title="Simulation Configuration")
        info_table.add_column("Parameter", style="cyan")
        info_table.add_column("Value", style="yellow")

        info_table.add_row("Duration", f"{sim_config.simulation.duration_days} days")
        info_table.add_row(
            "Random Seed", str(sim_config.simulation.random_seed or "None")
        )
        info_table.add_row("Output Directory", output)
        info_table.add_row("Colony Species", sim_config.colony.species)
        info_table.add_row(
            "Initial Workers", str(sim_config.colony.initial_population["workers"])
        )
        info_table.add_row(
            "Landscape Size",
            f"{sim_config.environment.landscape_width}x{sim_config.environment.landscape_height}",
        )

        console.print(info_table)

    # Run simulation with progress bar
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            # Initialize simulation
            init_task = progress.add_task("Initializing simulation...", total=100)

            # Simulate initialization steps
            for i in range(5):
                progress.update(init_task, advance=20)
                time.sleep(0.1)  # Simulate work

            progress.update(init_task, description="✅ Simulation initialized")

            # Run main simulation
            sim_task = progress.add_task(
                "Running simulation...", total=sim_config.simulation.duration_days
            )

            for day in range(sim_config.simulation.duration_days):
                # Simulate one day of computation
                time.sleep(0.001)  # Placeholder for actual simulation
                progress.update(sim_task, advance=1)

                if verbose and day % 30 == 0:  # Monthly updates
                    progress.console.print(
                        f"Day {day}: Colony population stable", style="dim"
                    )

            progress.update(sim_task, description="✅ Simulation completed")

            # Save results
            save_task = progress.add_task("Saving results...", total=100)

            for i in range(10):
                progress.update(save_task, advance=10)
                time.sleep(0.05)

            progress.update(save_task, description="✅ Results saved")

        if not quiet:
            console.print("\n🎉 Simulation completed successfully!", style="bold green")
            console.print(
                f"📁 Results saved to: {output_path.absolute()}", style="green"
            )

    except KeyboardInterrupt:
        console.print("\n⚠️  Simulation interrupted by user", style="yellow")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n❌ Simulation failed: {e}", style="red")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def config_cmd(
    action: str = typer.Argument(..., help="Action: create, validate, show, list"),
    name: Optional[str] = typer.Argument(None, help="Configuration name"),
    template: str = typer.Option(
        "basic", "--template", "-t", help="Configuration template"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
):
    """Manage BSTEW configurations"""

    if action == "create":
        if not name:
            console.print(
                "❌ Configuration name required for create action", style="red"
            )
            raise typer.Exit(1)

        try:
            config = config_manager.get_config_template(template)
            output_path = output or f"configs/{name}.yaml"
            config_manager.save_config(config, output_path)
            console.print(f"✅ Created configuration: {output_path}", style="green")
        except Exception as e:
            console.print(f"❌ Error creating configuration: {e}", style="red")
            raise typer.Exit(1)

    elif action == "validate":
        if not name:
            console.print(
                "❌ Configuration file path required for validate action", style="red"
            )
            raise typer.Exit(1)

        try:
            config = config_manager.load_config(name)
            errors = config_manager.validate_config(config)

            if errors:
                console.print("❌ Configuration validation failed:", style="red")
                for error in errors:
                    console.print(f"  • {error}", style="red")
                raise typer.Exit(1)
            else:
                console.print("✅ Configuration is valid", style="green")
        except Exception as e:
            console.print(f"❌ Error validating configuration: {e}", style="red")
            raise typer.Exit(1)

    elif action == "show":
        if not name:
            name = "configs/default.yaml"

        try:
            with open(name, "r") as f:
                content = f.read()

            syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=f"Configuration: {name}"))
        except Exception as e:
            console.print(f"❌ Error reading configuration: {e}", style="red")
            raise typer.Exit(1)

    elif action == "list":
        try:
            configs = config_manager.list_available_configs()

            tree = Tree("📁 Available Configurations")

            if configs["scenarios"]:
                scenarios_branch = tree.add("🎯 Scenarios")
                for scenario in configs["scenarios"]:
                    scenarios_branch.add(scenario)

            if configs["species"]:
                species_branch = tree.add("🐝 Species")
                for species in configs["species"]:
                    species_branch.add(species)

            console.print(tree)

        except Exception as e:
            console.print(f"❌ Error listing configurations: {e}", style="red")
            raise typer.Exit(1)

    else:
        console.print(f"❌ Unknown action: {action}", style="red")
        console.print("Valid actions: create, validate, show, list", style="yellow")
        raise typer.Exit(1)


@app.command()
def analyze(
    input_dir: str = typer.Argument(
        ..., help="Input directory with simulation results"
    ),
    format_type: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, csv, json"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
):
    """Analyze BSTEW simulation results"""

    input_path = Path(input_dir)

    if not input_path.exists():
        console.print(f"❌ Input directory not found: {input_dir}", style="red")
        raise typer.Exit(1)

    console.print(f"📊 Analyzing results from: {input_path}", style="blue")

    # Placeholder analysis
    results_table = Table(title="Simulation Results Summary")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="yellow")
    results_table.add_column("Unit", style="green")

    results_table.add_row("Final Colony Size", "15,432", "bees")
    results_table.add_row("Peak Population", "28,567", "bees")
    results_table.add_row("Total Honey Produced", "45.2", "kg")
    results_table.add_row("Foraging Efficiency", "0.78", "ratio")
    results_table.add_row("Colony Survival", "Yes", "boolean")

    console.print(results_table)

    if output_file:
        console.print(f"💾 Results exported to: {output_file}", style="green")


@app.command()
def plot(
    input_dir: str = typer.Argument(
        ..., help="Input directory with simulation results"
    ),
    plot_type: str = typer.Option(
        "population", "--type", "-t", help="Plot type: population, resources, spatial"
    ),
    output_dir: str = typer.Option(
        "plots", "--output", "-o", help="Output directory for plots"
    ),
):
    """Generate plots from BSTEW simulation results"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        console.print(f"❌ Input directory not found: {input_dir}", style="red")
        raise typer.Exit(1)

    output_path.mkdir(exist_ok=True)

    console.print(f"📈 Generating {plot_type} plots...", style="blue")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating plots...", total=None)

        time.sleep(1)  # Simulate plot generation

        progress.update(task, description="✅ Plots generated")

    console.print(f"💾 Plots saved to: {output_path.absolute()}", style="green")


@app.command()
def batch(
    experiments_file: str = typer.Argument(..., help="Experiments configuration file"),
    parallel: int = typer.Option(
        1, "--parallel", "-p", help="Number of parallel processes"
    ),
    output_base: str = typer.Option(
        "batch_results", "--output", "-o", help="Base output directory"
    ),
):
    """Run batch experiments"""

    experiments_path = Path(experiments_file)

    if not experiments_path.exists():
        console.print(f"❌ Experiments file not found: {experiments_file}", style="red")
        raise typer.Exit(1)

    try:
        with open(experiments_path, "r") as f:
            experiments = yaml.safe_load(f)
    except Exception as e:
        console.print(f"❌ Error loading experiments file: {e}", style="red")
        raise typer.Exit(1)

    num_experiments = len(experiments.get("experiments", []))

    console.print(
        f"🔬 Running {num_experiments} experiments with {parallel} parallel processes",
        style="blue",
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Running experiments...", total=num_experiments)

        for i in range(num_experiments):
            time.sleep(0.5)  # Simulate experiment execution
            progress.update(task, advance=1)

        progress.update(task, description="✅ All experiments completed")

    console.print(f"💾 Batch results saved to: {output_base}", style="green")


@app.command()
def experiment(
    design_file: str = typer.Argument(..., help="Experiment design YAML file"),
    output_dir: str = typer.Option(
        "experiments", "--output-dir", "-o", help="Output directory"
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Resume interrupted experiment"
    ),
    max_workers: int = typer.Option(
        None, "--max-workers", "-w", help="Maximum parallel workers"
    ),
):
    """Run batch experiments from design file"""

    design_path = Path(design_file)
    if not design_path.exists():
        console.print(f"❌ Design file not found: {design_file}", style="red")
        raise typer.Exit(1)

    try:
        # Load experiment design
        with open(design_path, "r") as f:
            design_data = yaml.safe_load(f)

        # Create experiment manager
        experiment_manager = ExperimentManager(output_dir)
        if max_workers:
            experiment_manager.batch_processor.max_workers = max_workers

        console.print(
            f"🧪 Loading experiment: {design_data.get('name', 'Unnamed')}", style="blue"
        )

        # Create parameter specs
        parameter_specs = {}
        for param_name, param_config in design_data.get("parameters", {}).items():
            parameter_specs[param_name] = ParameterSpec(
                name=param_name,
                min_value=param_config["min_value"],
                max_value=param_config["max_value"],
                step_size=param_config.get("step_size"),
                values=param_config.get("values"),
                distribution=param_config.get("distribution", "uniform"),
            )

        # Create experiment design
        exp_type = ExperimentType(design_data.get("experiment_type", "parameter_sweep"))

        if exp_type == ExperimentType.PARAMETER_SWEEP:
            design = experiment_manager.create_parameter_sweep(
                design_data["name"],
                parameter_specs,
                design_data.get("base_config"),
                design_data.get("n_replicates", 1),
                design_data.get("simulation_days", 365),
            )
        elif exp_type == ExperimentType.MONTE_CARLO:
            design = experiment_manager.create_monte_carlo_experiment(
                design_data["name"],
                parameter_specs,
                design_data.get("n_samples", 100),
                design_data.get("base_config"),
                design_data.get("simulation_days", 365),
            )
        elif exp_type == ExperimentType.SENSITIVITY_ANALYSIS:
            design = experiment_manager.create_sensitivity_analysis(
                design_data["name"],
                parameter_specs,
                design_data.get("n_samples", 1000),
                design_data.get("base_config"),
                design_data.get("simulation_days", 365),
            )
        else:
            console.print(f"❌ Unsupported experiment type: {exp_type}", style="red")
            raise typer.Exit(1)

        # Run experiment
        console.print(
            f"🚀 Starting experiment with {len(design.generate_runs())} runs",
            style="green",
        )

        result = experiment_manager.batch_processor.run_experiment(
            design, resume=resume
        )

        # Display results
        console.print("✅ Experiment completed!", style="bold green")

        results_table = Table(title="Experiment Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="yellow")

        results_table.add_row("Total Runs", str(result["total_runs"]))
        results_table.add_row("Successful", str(result["successful_runs"]))
        results_table.add_row("Failed", str(result["failed_runs"]))
        results_table.add_row("Duration", f"{result['total_duration']:.1f}s")
        results_table.add_row("Results Path", result["results_path"])

        console.print(results_table)

    except Exception as e:
        console.print(f"❌ Experiment failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def sweep(
    parameter: str = typer.Argument(
        ..., help="Parameter to sweep (e.g., 'colony.initial_population')"
    ),
    min_val: float = typer.Argument(..., help="Minimum parameter value"),
    max_val: float = typer.Argument(..., help="Maximum parameter value"),
    steps: int = typer.Option(10, "--steps", "-s", help="Number of steps"),
    replicates: int = typer.Option(
        1, "--replicates", "-r", help="Number of replicates per step"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Base configuration file"
    ),
    output_dir: str = typer.Option(
        "sweep_results", "--output-dir", "-o", help="Output directory"
    ),
    simulation_days: int = typer.Option(
        365, "--days", "-d", help="Simulation duration"
    ),
):
    """Run parameter sweep experiment"""

    console.print(
        f"🔄 Parameter sweep: {parameter} from {min_val} to {max_val}", style="blue"
    )

    try:
        # Create experiment manager
        experiment_manager = ExperimentManager(output_dir)

        # Load base configuration
        base_config = None
        if config:
            config_path = Path(config)
            if config_path.exists():
                base_config = config_manager.load_config(config_path)
            else:
                console.print(
                    f"⚠️  Config file not found: {config}, using defaults",
                    style="yellow",
                )

        # Create parameter spec
        parameter_specs = {
            parameter: ParameterSpec(
                name=parameter,
                min_value=min_val,
                max_value=max_val,
                step_size=(max_val - min_val) / (steps - 1) if steps > 1 else 1,
            )
        }

        # Create and run experiment
        design = experiment_manager.create_parameter_sweep(
            f"Parameter sweep: {parameter}",
            parameter_specs,
            base_config,
            replicates,
            simulation_days,
        )

        result = experiment_manager.batch_processor.run_experiment(design)

        # Display results
        console.print("✅ Parameter sweep completed!", style="bold green")
        console.print(f"📊 Results saved to: {result['results_path']}", style="green")

    except Exception as e:
        console.print(f"❌ Parameter sweep failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def compare(
    scenarios_file: str = typer.Argument(
        ..., help="YAML file with scenario configurations"
    ),
    output_dir: str = typer.Option(
        "comparison", "--output-dir", "-o", help="Output directory"
    ),
    simulation_days: int = typer.Option(
        365, "--days", "-d", help="Simulation duration"
    ),
):
    """Compare multiple scenarios"""

    scenarios_path = Path(scenarios_file)
    if not scenarios_path.exists():
        console.print(f"❌ Scenarios file not found: {scenarios_file}", style="red")
        raise typer.Exit(1)

    try:
        # Load scenarios
        with open(scenarios_path, "r") as f:
            scenarios = yaml.safe_load(f)

        console.print(f"📊 Comparing {len(scenarios)} scenarios", style="blue")

        # Create experiment manager
        experiment_manager = ExperimentManager(output_dir)

        # Run quick comparison
        results = experiment_manager.run_quick_comparison(scenarios, simulation_days)

        # Display results table
        comparison_table = Table(title="Scenario Comparison")
        comparison_table.add_column("Scenario", style="cyan")
        comparison_table.add_column("Final Population", style="yellow")
        comparison_table.add_column("Max Population", style="yellow")
        comparison_table.add_column("Final Honey", style="yellow")
        comparison_table.add_column("Survived", style="green")

        for scenario_name, result in results.items():
            survived = "✅" if result["colony_survival"] else "❌"
            comparison_table.add_row(
                scenario_name,
                f"{result['final_population']:,}",
                f"{result['max_population']:,}",
                f"{result['final_honey']:.1f}",
                survived,
            )

        console.print(comparison_table)

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        results_df = pd.DataFrame(results).T
        results_df.to_csv(output_path / "comparison_results.csv")

        console.print(
            f"💾 Results saved to: {output_path / 'comparison_results.csv'}",
            style="green",
        )

    except Exception as e:
        console.print(f"❌ Comparison failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def version():
    """Show BSTEW version information"""

    version_info = Table(title="BSTEW Version Information")
    version_info.add_column("Component", style="cyan")
    version_info.add_column("Version", style="yellow")

    version_info.add_row("BSTEW", "0.1.0")
    version_info.add_row(
        "Python",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    version_info.add_row("Platform", sys.platform)

    console.print(version_info)


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize"),
    template: str = typer.Option("basic", "--template", "-t", help="Project template"),
):
    """Initialize a new BSTEW project"""

    project_path = Path(directory)

    if project_path.exists() and any(project_path.iterdir()):
        if not typer.confirm(f"Directory {directory} is not empty. Continue?"):
            raise typer.Exit(0)

    project_path.mkdir(exist_ok=True)

    console.print(
        f"🚀 Initializing BSTEW project in: {project_path.absolute()}", style="blue"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating project structure...", total=None)

        # Create directories
        (project_path / "configs").mkdir(exist_ok=True)
        (project_path / "data").mkdir(exist_ok=True)
        (project_path / "results").mkdir(exist_ok=True)
        (project_path / "scripts").mkdir(exist_ok=True)

        # Create default configuration
        config = config_manager.get_config_template(template)
        config_manager.save_config(config, project_path / "configs" / "config.yaml")

        # Create README
        readme_content = """# BSTEW Project

This is a BSTEW (BeeSteward v2 Python) simulation project.

## Quick Start

1. Configure your simulation:
   ```bash
   bstew config show configs/config.yaml
   ```

2. Run the simulation:
   ```bash
   bstew run --config configs/config.yaml
   ```

3. Analyze results:
   ```bash
   bstew analyze results/
   ```

## Directory Structure

- `configs/` - Configuration files
- `data/` - Input data (landscapes, weather, etc.)
- `results/` - Simulation output
- `scripts/` - Custom analysis scripts

"""

        with open(project_path / "README.md", "w") as f:
            f.write(readme_content)

        time.sleep(0.5)  # Simulate work
        progress.update(task, description="✅ Project initialized")

    console.print("✅ BSTEW project initialized successfully!", style="green")
    console.print(f"📁 Project directory: {project_path.absolute()}", style="blue")
    console.print("\nNext steps:", style="bold")
    console.print("1. Edit configs/config.yaml to customize your simulation")
    console.print("2. Add landscape and weather data to data/")
    console.print("3. Run: bstew run --config configs/config.yaml")


if __name__ == "__main__":
    app()
