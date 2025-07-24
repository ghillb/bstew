"""
BSTEW CLI Interface
==================

Modular command-line interface for BSTEW with Rich integration and advanced features.
"""

import typer
from rich.console import Console
from typing import Optional, List

from .core.base import CLIContext, command_registry
from .core.progress import StatusDisplay  # noqa: F401
from .types import VerbosityLevel
from .commands import (
    RunCommand,
    ConfigCommand,
    AnalyzeCommand,
    PlotCommand,
    ExperimentCommand,
    BatchCommand,
    SweepCommand,
    CompareCommand,
    OptimizePerformanceCommand,
    OptimizeParametersCommand,
    CalibrateCommand,
    SensitivityCommand,
    UncertaintyCommand,
    ValidateCommand,
    VersionCommand,
    InitCommand,
)
from .commands.economics import EconomicAnalysisCommand
from .commands.parameters import (
    load_parameters_cmd,
    inspect_parameters_cmd,
    validate_parameters_cmd,
    convert_parameters_cmd,
    discover_netlogo_cmd,
    validate_netlogo_cmd,
    convert_netlogo_cmd,
    create_template_cmd,
)
from .commands import (
    spatial,
    visualization,
    display,
    data_analysis,
    excel_reports,
    runtime,
)
from .commands.validate_netlogo import app as netlogo_app
from .commands.benchmark import app as benchmark_app

# Initialize Typer app
app = typer.Typer(
    name="bstew",
    help="BSTEW - BeeSteward v2 Python Transpilation",
    add_completion=False,
    rich_markup_mode="rich",
)

# Global console and context
console = Console()


def version_callback(value: bool) -> None:
    """Handle --version flag"""
    if value:
        context = CLIContext(console=console, verbosity=VerbosityLevel.NORMAL)
        command = VersionCommand(context)
        result = command.execute()
        raise typer.Exit(code=result.exit_code)


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version information and exit",
    ),
) -> None:
    """BSTEW - BeeSteward v2 Python Transpilation"""
    pass


def get_cli_context(
    verbose: bool = False,
    quiet: bool = False,
    debug: bool = False,
) -> CLIContext:
    """Create CLI context based on verbosity flags"""

    if debug:
        verbosity = VerbosityLevel.DEBUG
    elif verbose:
        verbosity = VerbosityLevel.VERBOSE
    elif quiet:
        verbosity = VerbosityLevel.QUIET
    else:
        verbosity = VerbosityLevel.NORMAL

    return CLIContext(console=console, verbosity=verbosity)


# Register all commands
def register_commands() -> None:
    """Register all command classes"""

    command_registry.register("run", RunCommand)
    command_registry.register("config", ConfigCommand)
    command_registry.register("analyze", AnalyzeCommand)
    command_registry.register("experiment", ExperimentCommand)
    command_registry.register("batch", BatchCommand)
    command_registry.register("sweep", SweepCommand)
    command_registry.register("compare", CompareCommand)
    command_registry.register("optimize_performance", OptimizePerformanceCommand)
    command_registry.register("optimize_parameters", OptimizeParametersCommand)
    command_registry.register("calibrate", CalibrateCommand)
    command_registry.register("sensitivity", SensitivityCommand)
    command_registry.register("uncertainty", UncertaintyCommand)
    command_registry.register("validate", ValidateCommand)
    command_registry.register("version", VersionCommand)
    command_registry.register("init", InitCommand)
    command_registry.register("economics", EconomicAnalysisCommand)


# Run command
@app.command()
def run(
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Configuration file path"
    ),
    output: str = typer.Option(
        "artifacts/outputs", "--output", "-o", help="Output directory"
    ),
    days: Optional[int] = typer.Option(
        None, "--days", "-d", help="Simulation duration in days"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    parallel: bool = typer.Option(
        False, "--parallel", help="Enable parallel execution for multiple runs"
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        help="Maximum number of parallel workers (default: CPU count)",
    ),
    replicates: int = typer.Option(
        1, "--replicates", "-r", help="Number of simulation replicates to run"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Run BSTEW simulation with optional parallel execution

    When --parallel is enabled, multiple simulation replicates can be run
    concurrently using ProcessPoolExecutor. Use --replicates to specify
    the number of replicate runs and --max-workers to control parallelism.

    Examples:
    - Single simulation: bstew run --config my_config.yaml
    - Multiple replicates: bstew run --replicates 5
    - Parallel execution: bstew run --parallel --replicates 10 --max-workers 4
    """

    context = get_cli_context(verbose, quiet, debug)
    command = RunCommand(context)

    # Validate inputs
    errors = command.validate_inputs(
        config=config,
        output=output,
        days=days,
        seed=seed,
        parallel=parallel,
        max_workers=max_workers,
        replicates=replicates,
        verbose=verbose,
        quiet=quiet,
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(
        config=config,
        output=output,
        days=days,
        seed=seed,
        parallel=parallel,
        max_workers=max_workers,
        replicates=replicates,
        verbose=verbose,
        quiet=quiet,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


# Config command
@app.command()
def config(
    action: str = typer.Argument(
        ..., help="Action: create, validate, show, list, diff"
    ),
    name: Optional[str] = typer.Argument(
        None, help="Configuration name or first file for diff"
    ),
    template: str = typer.Option(
        "basic",
        "--template",
        "-t",
        help="Configuration template or second file for diff",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Manage BSTEW configurations"""

    context = get_cli_context(verbose, quiet, debug)
    command = ConfigCommand(context)

    # Validate inputs
    errors = command.validate_inputs(
        action=action, name=name, template=template, output=output
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(action=action, name=name, template=template, output=output)

    if not result.success:
        raise typer.Exit(result.exit_code)


# Analyze command
@app.command()
def analyze(
    input_dir: str = typer.Argument(
        ..., help="Input directory with simulation results"
    ),
    format_type: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, csv, json, yaml, html, summary",
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    analysis_type: str = typer.Option(
        "comprehensive",
        "--type",
        "-t",
        help="Analysis type: comprehensive, population, foraging, survival, efficiency, summary",
    ),
    species: Optional[str] = typer.Option(
        None, "--species", "-s", help="Filter by species (e.g., bombus_terrestris)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Analyze BSTEW simulation results with comprehensive statistics

    Analysis types:
    - comprehensive: Complete analysis including all metrics
    - population: Population trends, growth rates, and demographics
    - foraging: Foraging efficiency and resource optimization
    - survival: Survival curves and life expectancy analysis
    - efficiency: Combined efficiency metrics
    - summary: Quick summary statistics only

    Examples:
    - Basic analysis: bstew analyze results/
    - Export to CSV: bstew analyze results/ --format csv --output analysis.csv
    - Species-specific: bstew analyze results/ --species bombus_terrestris
    - HTML report: bstew analyze results/ --format html --output report.html
    """

    context = get_cli_context(verbose, quiet, debug)
    command = AnalyzeCommand(context)

    # Validate inputs
    errors = command.validate_inputs(
        input_dir=input_dir,
        format_type=format_type,
        output_file=output_file,
        analysis_type=analysis_type,
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(
        input_dir=input_dir,
        format_type=format_type,
        output_file=output_file,
        analysis_type=analysis_type,
        species=species,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


# Advanced optimization commands
@app.command(name="optimize-performance")
def optimize_performance(
    config: str = typer.Option("configs/default.yaml", help="Configuration file"),
    enable_caching: bool = typer.Option(True, help="Enable result caching"),
    parallel_workers: Optional[int] = typer.Option(
        None, help="Number of parallel workers"
    ),
    memory_limit: Optional[int] = typer.Option(None, help="Memory limit in MB"),
    profile: bool = typer.Option(False, help="Enable performance profiling"),
    output: str = typer.Option("artifacts/performance", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Run simulation with performance optimizations"""

    context = get_cli_context(verbose, quiet, debug)
    command = OptimizePerformanceCommand(context)

    # Validate inputs
    errors = command.validate_inputs(
        config=config, parallel_workers=parallel_workers, memory_limit=memory_limit
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(
        config=config,
        enable_caching=enable_caching,
        parallel_workers=parallel_workers,
        memory_limit=memory_limit,
        profile=profile,
        output=output,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command(name="optimize-parameters")
def optimize_parameters(
    target_data: str = typer.Argument(
        ..., help="Target/field data file for calibration"
    ),
    config: str = typer.Option("configs/default.yaml", help="Base configuration"),
    method: str = typer.Option("genetic_algorithm", help="Optimization method"),
    max_iterations: int = typer.Option(1000, help="Maximum iterations"),
    population_size: int = typer.Option(
        50, help="Population size for genetic algorithms"
    ),
    parallel: bool = typer.Option(True, help="Enable parallel evaluation"),
    output: str = typer.Option("artifacts/experiments", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Optimize model parameters against target data"""

    context = get_cli_context(verbose, quiet, debug)
    command = OptimizeParametersCommand(context)

    # Validate inputs
    errors = command.validate_inputs(
        target_data=target_data,
        method=method,
        max_iterations=max_iterations,
        population_size=population_size,
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(
        target_data=target_data,
        config=config,
        method=method,
        max_iterations=max_iterations,
        population_size=population_size,
        parallel=parallel,
        output=output,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def calibrate(
    field_data: str = typer.Argument(..., help="Field observation data"),
    config: str = typer.Option("configs/default.yaml", help="Base configuration"),
    parameters: Optional[str] = typer.Option(
        None, help="Parameters file specifying ranges"
    ),
    objective: str = typer.Option("rmse", help="Objective function"),
    output: str = typer.Option("artifacts/experiments", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Calibrate model against field observations"""

    context = get_cli_context(verbose, quiet, debug)
    command = CalibrateCommand(context)

    # Execute command
    result = command.execute(
        field_data=field_data,
        config=config,
        parameters=parameters,
        objective=objective,
        output=output,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def sensitivity(
    config: str = typer.Option("configs/default.yaml", help="Configuration file"),
    method: str = typer.Option("sobol", help="Sensitivity method: sobol, morris"),
    parameters: Optional[str] = typer.Option(None, help="Parameters file"),
    samples: int = typer.Option(1000, help="Number of samples"),
    output: str = typer.Option("artifacts/experiments", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Run sensitivity analysis on model parameters"""

    context = get_cli_context(verbose, quiet, debug)
    command = SensitivityCommand(context)

    # Execute command
    result = command.execute(
        config=config,
        method=method,
        parameters=parameters,
        samples=samples,
        output=output,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def uncertainty(
    config: str = typer.Option("configs/default.yaml", help="Configuration file"),
    method: str = typer.Option("monte_carlo", help="Uncertainty method"),
    samples: int = typer.Option(1000, help="Number of Monte Carlo samples"),
    output: str = typer.Option("artifacts/experiments", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Run uncertainty quantification analysis"""

    context = get_cli_context(verbose, quiet, debug)
    command = UncertaintyCommand(context)

    # Execute command
    result = command.execute(
        config=config, method=method, samples=samples, output=output
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def validate(
    model_results: str = typer.Argument(..., help="Model results directory"),
    field_data: str = typer.Argument(..., help="Field data for comparison"),
    metrics: str = typer.Option("all", help="Validation metrics to compute"),
    output: str = typer.Option("artifacts/validation", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Validate model outputs against field data"""

    context = get_cli_context(verbose, quiet, debug)
    command = ValidateCommand(context)

    # Validate inputs
    errors = command.validate_inputs(
        model_results=model_results, field_data=field_data, metrics=metrics
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(
        model_results=model_results,
        field_data=field_data,
        metrics=metrics,
        output=output,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


# Utility commands
@app.command()
def version() -> None:
    """Show BSTEW version information"""

    context = get_cli_context()
    command = VersionCommand(context)

    result = command.execute()

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize"),
    template: str = typer.Option("basic", "--template", "-t", help="Project template"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Initialize a new BSTEW project"""

    context = get_cli_context(verbose, quiet, debug)
    command = InitCommand(context)

    # Validate inputs
    errors = command.validate_inputs(directory=directory, template=template)
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(directory=directory, template=template)

    if not result.success:
        raise typer.Exit(result.exit_code)


# Command modules imported at top of file

# Add new command groups
app.add_typer(spatial.app, name="spatial")
app.add_typer(visualization.app, name="visualize")
app.add_typer(display.app, name="display")
app.add_typer(data_analysis.app, name="data")
app.add_typer(excel_reports.app, name="excel")
app.add_typer(runtime.app, name="runtime")

app.add_typer(netlogo_app, name="netlogo")
app.add_typer(benchmark_app, name="benchmark")


# Plot command with full functionality
@app.command()
def plot(
    results_dir: str = typer.Argument(
        ..., help="Input directory with simulation results"
    ),
    plot_type: str = typer.Option(
        "population",
        "--type",
        "-t",
        help="Plot type: population, spatial, foraging, resources, temporal, comparison",
    ),
    species: Optional[str] = typer.Option(
        None, "--species", "-s", help="Filter by species (e.g., bombus_terrestris)"
    ),
    output_dir: str = typer.Option(
        "artifacts/plots", "--output", "-o", help="Output directory for plots"
    ),
    format: str = typer.Option(
        "png", "--format", "-f", help="Output format: png, svg, pdf, html"
    ),
    time_range: Optional[str] = typer.Option(
        None,
        "--time-range",
        help="Time range filter (format: start:end, e.g., 2024-01-01:2024-12-31)",
    ),
    save_data: bool = typer.Option(
        False, "--save-data", help="Save underlying data alongside plots"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Open interactive plot viewer"
    ),
    dpi: int = typer.Option(300, "--dpi", help="DPI for raster formats"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Generate plots from BSTEW simulation results

    Plot types:
    - population: Population dynamics over time
    - spatial: Spatial distribution of colonies and territories
    - foraging: Foraging patterns and efficiency
    - resources: Resource distribution and quality
    - temporal: Multi-metric temporal analysis
    - comparison: Compare multiple scenarios
    """

    context = get_cli_context(verbose, quiet, debug)
    command = PlotCommand(context)

    # Validate inputs
    errors = command.validate_inputs(
        plot_type=plot_type,
        format=format,
        time_range=time_range,
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(
        input_dir=results_dir,
        plot_type=plot_type,
        output_dir=output_dir,
        species=species,
        format=format,
        time_range=time_range,
        save_data=save_data,
        interactive=interactive,
        dpi=dpi,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def batch(
    experiments_file: str = typer.Argument(..., help="Experiments configuration file"),
    parallel: int = typer.Option(
        1, "--parallel", "-p", help="Number of parallel processes"
    ),
    output_base: str = typer.Option(
        "artifacts/experiments", "--output", "-o", help="Base output directory"
    ),
) -> None:
    """Run batch experiments"""

    context = get_cli_context()
    command = BatchCommand(context)

    result = command.execute(
        experiments_file=experiments_file, parallel=parallel, output_base=output_base
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def experiment(
    design_file: Optional[str] = typer.Argument(
        None, help="Experiment design YAML file (optional if using --parameter)"
    ),
    parameter: Optional[List[str]] = typer.Option(
        None,
        "--parameter",
        "-p",
        help="Parameter variations (e.g., 'learning_rate=0.1,0.2,0.3')",
    ),
    output_dir: str = typer.Option(
        "artifacts/experiments", "--output-dir", "-o", help="Output directory"
    ),
    base_config: Optional[str] = typer.Option(
        None, "--base-config", "-c", help="Base configuration file"
    ),
    replicates: int = typer.Option(
        1, "--replicates", "-r", help="Number of replicates per combination"
    ),
    simulation_days: int = typer.Option(
        365, "--days", "-d", help="Simulation duration in days"
    ),
    experiment_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Experiment name"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume interrupted experiment"
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", "-w", help="Maximum parallel workers"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Run experiments with parameter variations

    Examples:
    - Using parameter syntax:
      bstew experiment --parameter learning_rate=0.1,0.2,0.3 --parameter foraging_radius=500,1000,1500

    - From design file:
      bstew experiment experiment_design.yaml

    - Combining both:
      bstew experiment design.yaml --parameter extra_param=1,2,3

    - With replicates:
      bstew experiment --parameter colony_size=1000,5000,10000 --replicates 5
    """

    context = get_cli_context(verbose, quiet, debug)
    command = ExperimentCommand(context)

    # Validate inputs
    errors = command.validate_inputs(design_file=design_file, parameters=parameter)
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    result = command.execute(
        design_file=design_file,
        output_dir=output_dir,
        resume=resume,
        max_workers=max_workers,
        parameters=parameter,
        base_config=base_config,
        replicates=replicates,
        simulation_days=simulation_days,
        experiment_name=experiment_name,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def sweep(
    param_range: List[str] = typer.Option(
        ...,
        "--param-range",
        "-p",
        help="Parameter range (e.g., 'learning_rate=0.1:0.5:0.1')",
    ),
    base_config: Optional[str] = typer.Option(
        None, "--base-config", "-c", help="Base configuration file"
    ),
    replicates: int = typer.Option(
        1, "--replicates", "-r", help="Number of replicates per combination"
    ),
    output_dir: str = typer.Option(
        "artifacts/sweeps", "--output-dir", "-o", help="Output directory"
    ),
    simulation_days: int = typer.Option(
        365, "--days", "-d", help="Simulation duration in days"
    ),
    sweep_name: Optional[str] = typer.Option(None, "--name", "-n", help="Sweep name"),
    save_interval: int = typer.Option(
        10, "--save-interval", help="Save results every N runs"
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Generate visualizations"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Run systematic parameter sweep with range notation

    Examples:
    - Single parameter sweep:
      bstew sweep --param-range learning_rate=0.1:0.5:0.1

    - Multiple parameter sweep (full factorial):
      bstew sweep --param-range learning_rate=0.1:0.5:0.1 --param-range colony_size=1000:10000:1000

    - With replicates and base config:
      bstew sweep --param-range foraging_radius=100:500:50 --base-config config.yaml --replicates 3

    - With visualization:
      bstew sweep --param-range nectar_threshold=0.1:1.0:0.1 --visualize

    Range format: parameter_name=start:end:step
    - start: Starting value
    - end: Ending value (inclusive)
    - step: Step size between values
    """

    context = get_cli_context(verbose, quiet, debug)
    command = SweepCommand(context)

    # Validate inputs
    errors = command.validate_inputs(param_ranges=param_range)
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    result = command.execute(
        param_ranges=param_range,
        base_config=base_config,
        replicates=replicates,
        output_dir=output_dir,
        simulation_days=simulation_days,
        sweep_name=sweep_name,
        save_interval=save_interval,
        visualize=visualize,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def compare(
    scenarios_file: str = typer.Argument(
        ..., help="YAML file with scenario configurations"
    ),
    output_dir: str = typer.Option(
        "artifacts/experiments", "--output-dir", "-o", help="Output directory"
    ),
    simulation_days: int = typer.Option(
        365, "--days", "-d", help="Simulation duration"
    ),
) -> None:
    """Compare multiple scenarios"""

    context = get_cli_context()
    command = CompareCommand(context)

    result = command.execute(
        scenarios_file=scenarios_file,
        output_dir=output_dir,
        simulation_days=simulation_days,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


# Parameter management commands
@app.command(name="load-params")
def load_parameters(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    param_type: str = typer.Option(
        "behavioral", help="Parameter type (behavioral/environmental/colony)"
    ),
    format_type: str = typer.Option("csv", help="File format (csv/json/xlsx)"),
    validate: bool = typer.Option(True, help="Validate parameters after loading"),
    output_dir: str = typer.Option(
        "artifacts/parameters", help="Output directory for validated parameters"
    ),
) -> None:
    """Load parameters from file"""
    load_parameters_cmd(file_path, param_type, format_type, validate, output_dir)


@app.command(name="inspect-params")
def inspect_parameters(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    details: bool = typer.Option(False, "--details", help="Show detailed data"),
    schema: bool = typer.Option(False, "--schema", help="Export schema"),
) -> None:
    """Inspect parameter file contents"""
    inspect_parameters_cmd(file_path, details, schema)


@app.command(name="validate-params")
def validate_parameters(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    param_type: str = typer.Option("behavioral", help="Parameter type"),
    strict: bool = typer.Option(False, "--strict", help="Strict validation mode"),
) -> None:
    """Validate parameter file"""
    validate_parameters_cmd(file_path, param_type, strict)


@app.command(name="convert-params")
def convert_parameters(
    input_file: str = typer.Argument(..., help="Input parameter file"),
    output_file: str = typer.Argument(..., help="Output parameter file"),
    input_format: str = typer.Option("csv", help="Input format"),
    output_format: str = typer.Option("json", help="Output format"),
    param_type: str = typer.Option("behavioral", help="Parameter type"),
) -> None:
    """Convert parameters between formats"""
    convert_parameters_cmd(
        input_file, output_file, input_format, output_format, param_type
    )


@app.command(name="discover-netlogo")
def discover_netlogo(
    directory: str = typer.Argument(..., help="Directory to search for NetLogo files"),
    export: bool = typer.Option(False, "--export", help="Export discovery report"),
    output_dir: str = typer.Option(
        "artifacts/netlogo_discovery", help="Output directory for reports"
    ),
) -> None:
    """Discover NetLogo files in directory"""
    discover_netlogo_cmd(directory, export, output_dir)


@app.command(name="validate-netlogo")
def validate_netlogo(
    directory: str = typer.Argument(..., help="Directory containing NetLogo files"),
    fix: bool = typer.Option(
        False, "--fix", help="Attempt to fix compatibility issues"
    ),
) -> None:
    """Validate NetLogo data compatibility"""
    validate_netlogo_cmd(directory, fix)


@app.command(name="convert-netlogo")
def convert_netlogo(
    netlogo_directory: str = typer.Argument(..., help="NetLogo data directory"),
    output_directory: str = typer.Option(
        "artifacts/netlogo_converted", help="Output directory"
    ),
    validate_first: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate before conversion"
    ),
) -> None:
    """Convert NetLogo data to BSTEW format"""
    convert_netlogo_cmd(netlogo_directory, output_directory, validate_first)


@app.command(name="create-template")
def create_template(
    param_type: str = typer.Option("behavioral", help="Parameter type"),
    output_file: str = typer.Option(
        "artifacts/parameters/template.csv", help="Output file"
    ),
    format_type: str = typer.Option("csv", help="File format"),
) -> None:
    """Create parameter template file"""
    create_template_cmd(param_type, output_file, format_type)


@app.command()
def economics(
    input_dir: str = typer.Argument(..., help="Directory with simulation results"),
    analysis_type: str = typer.Option(
        "comprehensive",
        "--type",
        "-t",
        help="Analysis type: comprehensive, crop_valuation, cost_benefit, scenario_comparison, landscape",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, csv, json, html, excel, pdf",
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    crop_config: Optional[str] = typer.Option(
        None, "--crop-config", help="YAML file with crop configuration"
    ),
    scenario_config: Optional[str] = typer.Option(
        None, "--scenario-config", help="YAML file with scenario definitions"
    ),
    baseline_data: Optional[str] = typer.Option(
        None, "--baseline", help="Baseline economic scenario data"
    ),
    time_horizon: int = typer.Option(
        10, "--time-horizon", help="Analysis time horizon in years"
    ),
    discount_rate: float = typer.Option(
        0.03, "--discount-rate", help="Discount rate for NPV calculations"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Perform economic analysis of pollinator services

    Analysis types:
    - comprehensive: Complete economic assessment including all metrics
    - crop_valuation: Estimate pollination service value for crops
    - cost_benefit: Cost-benefit analysis of conservation strategies
    - scenario_comparison: Compare multiple economic scenarios
    - landscape: Landscape-level economic valuation

    Examples:
    - Basic analysis: bstew economics results/
    - Crop valuation: bstew economics results/ --type crop_valuation --crop-config crops.yaml
    - Export to Excel: bstew economics results/ --format excel --output economic_report.xlsx
    - Scenario comparison: bstew economics results/ --type scenario_comparison --scenario-config scenarios.yaml
    """

    context = get_cli_context(verbose, quiet, debug)
    command = EconomicAnalysisCommand(context)

    # Validate inputs
    errors = command.validate_inputs(
        input_dir=input_dir,
        analysis_type=analysis_type,
        output_format=output_format,
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)

    # Execute command
    result = command.execute(
        input_dir=input_dir,
        analysis_type=analysis_type,
        output_format=output_format,
        output_file=output_file,
        crop_config=crop_config,
        scenario_config=scenario_config,
        baseline_data=baseline_data,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
    )

    if not result.success:
        raise typer.Exit(result.exit_code)


# Initialize commands on import
register_commands()


def main() -> None:
    """Main CLI entry point"""
    app()


__all__ = ["main", "netlogo_main", "app"]

# Keep NetLogo compatibility
from .netlogo_cli import main as netlogo_main  # noqa: E402


if __name__ == "__main__":
    main()
