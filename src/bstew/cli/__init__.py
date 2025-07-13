"""
BSTEW CLI Interface
==================

Modular command-line interface for BSTEW with Rich integration and advanced features.
"""

import typer
from rich.console import Console
from typing import Optional

from .core.base import CLIContext, command_registry
from .core.progress import StatusDisplay  # noqa: F401
from .types import VerbosityLevel
from .commands import (
    RunCommand,
    ConfigCommand,
    AnalyzeCommand,
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
from .commands import spatial, visualization, display, data_analysis, excel_reports, runtime

# Initialize Typer app
app = typer.Typer(
    name="bstew",
    help="BSTEW - BeeSteward v2 Python Transpilation",
    add_completion=False,
    rich_markup_mode="rich",
)

# Global console and context
console = Console()


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


# Run command
@app.command()
def run(
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Configuration file path"
    ),
    output: str = typer.Option("artifacts/outputs", "--output", "-o", help="Output directory"),
    days: Optional[int] = typer.Option(
        None, "--days", "-d", help="Simulation duration in days"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Run BSTEW simulation"""
    
    context = get_cli_context(verbose, quiet, debug)
    command = RunCommand(context)
    
    # Validate inputs
    errors = command.validate_inputs(
        config=config, output=output, days=days, seed=seed, 
        verbose=verbose, quiet=quiet
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)
    
    # Execute command
    result = command.execute(
        config=config, output=output, days=days, seed=seed,
        verbose=verbose, quiet=quiet
    )
    
    if not result.success:
        raise typer.Exit(result.exit_code)


# Config command
@app.command()
def config(
    action: str = typer.Argument(..., help="Action: create, validate, show, list"),
    name: Optional[str] = typer.Argument(None, help="Configuration name"),
    template: str = typer.Option(
        "basic", "--template", "-t", help="Configuration template"
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
    result = command.execute(
        action=action, name=name, template=template, output=output
    )
    
    if not result.success:
        raise typer.Exit(result.exit_code)


# Analyze command
@app.command()
def analyze(
    input_dir: str = typer.Argument(
        ..., help="Input directory with simulation results"
    ),
    format_type: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, csv, json, yaml"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """Analyze BSTEW simulation results"""
    
    context = get_cli_context(verbose, quiet, debug)
    command = AnalyzeCommand(context)
    
    # Validate inputs
    errors = command.validate_inputs(
        input_dir=input_dir, format_type=format_type, output_file=output_file
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)
    
    # Execute command
    result = command.execute(
        input_dir=input_dir, format_type=format_type, output_file=output_file
    )
    
    if not result.success:
        raise typer.Exit(result.exit_code)


# Advanced optimization commands
@app.command(name="optimize-performance")
def optimize_performance(
    config: str = typer.Option("configs/default.yaml", help="Configuration file"),
    enable_caching: bool = typer.Option(True, help="Enable result caching"),
    parallel_workers: Optional[int] = typer.Option(None, help="Number of parallel workers"),
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
        config=config, enable_caching=enable_caching, 
        parallel_workers=parallel_workers, memory_limit=memory_limit,
        profile=profile, output=output
    )
    
    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command(name="optimize-parameters")
def optimize_parameters(
    target_data: str = typer.Argument(..., help="Target/field data file for calibration"),
    config: str = typer.Option("configs/default.yaml", help="Base configuration"),
    method: str = typer.Option("genetic_algorithm", help="Optimization method"),
    max_iterations: int = typer.Option(1000, help="Maximum iterations"),
    population_size: int = typer.Option(50, help="Population size for genetic algorithms"),
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
        target_data=target_data, method=method, 
        max_iterations=max_iterations, population_size=population_size
    )
    if errors:
        for error in errors:
            context.print_error(error)
        raise typer.Exit(1)
    
    # Execute command
    result = command.execute(
        target_data=target_data, config=config, method=method,
        max_iterations=max_iterations, population_size=population_size,
        parallel=parallel, output=output
    )
    
    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def calibrate(
    field_data: str = typer.Argument(..., help="Field observation data"),
    config: str = typer.Option("configs/default.yaml", help="Base configuration"),
    parameters: Optional[str] = typer.Option(None, help="Parameters file specifying ranges"),
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
        field_data=field_data, config=config, parameters=parameters,
        objective=objective, output=output
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
        config=config, method=method, parameters=parameters,
        samples=samples, output=output
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
        model_results=model_results, field_data=field_data,
        metrics=metrics, output=output
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

# Add NetLogo validation commands
from .commands.validate_netlogo import app as netlogo_app
app.add_typer(netlogo_app, name="netlogo")

# Add benchmark commands
from .commands.benchmark import app as benchmark_app
app.add_typer(benchmark_app, name="benchmark")

# Legacy command stubs (to be implemented with remaining functionality)
@app.command()
def plot(
    input_dir: str = typer.Argument(..., help="Input directory with simulation results"),
    plot_type: str = typer.Option("population", "--type", "-t", help="Plot type"),
    output_dir: str = typer.Option("artifacts/outputs", "--output", "-o", help="Output directory"),
) -> None:
    """Generate plots from BSTEW simulation results"""
    
    context = get_cli_context()
    # Placeholder - would use PlotCommand
    context.print_info(f"Generating {plot_type} plots from {input_dir}")
    context.print_success(f"Plots would be saved to: {output_dir}")


@app.command()
def batch(
    experiments_file: str = typer.Argument(..., help="Experiments configuration file"),
    parallel: int = typer.Option(1, "--parallel", "-p", help="Number of parallel processes"),
    output_base: str = typer.Option("artifacts/experiments", "--output", "-o", help="Base output directory"),
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
    design_file: str = typer.Argument(..., help="Experiment design YAML file"),
    output_dir: str = typer.Option("artifacts/experiments", "--output-dir", "-o", help="Output directory"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume interrupted experiment"),
    max_workers: Optional[int] = typer.Option(None, "--max-workers", "-w", help="Maximum parallel workers"),
) -> None:
    """Run batch experiments from design file"""
    
    context = get_cli_context()
    command = ExperimentCommand(context)
    
    result = command.execute(
        design_file=design_file, output_dir=output_dir,
        resume=resume, max_workers=max_workers
    )
    
    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def sweep(
    parameter: str = typer.Argument(..., help="Parameter to sweep"),
    min_val: float = typer.Argument(..., help="Minimum parameter value"),
    max_val: float = typer.Argument(..., help="Maximum parameter value"),
    steps: int = typer.Option(10, "--steps", "-s", help="Number of steps"),
    replicates: int = typer.Option(1, "--replicates", "-r", help="Number of replicates per step"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Base configuration file"),
    output_dir: str = typer.Option("artifacts/experiments", "--output-dir", "-o", help="Output directory"),
    simulation_days: int = typer.Option(365, "--days", "-d", help="Simulation duration"),
) -> None:
    """Run parameter sweep experiment"""
    
    context = get_cli_context()
    command = SweepCommand(context)
    
    result = command.execute(
        parameter=parameter, min_val=min_val, max_val=max_val,
        steps=steps, replicates=replicates, config=config,
        output_dir=output_dir, simulation_days=simulation_days
    )
    
    if not result.success:
        raise typer.Exit(result.exit_code)


@app.command()
def compare(
    scenarios_file: str = typer.Argument(..., help="YAML file with scenario configurations"),
    output_dir: str = typer.Option("artifacts/experiments", "--output-dir", "-o", help="Output directory"),
    simulation_days: int = typer.Option(365, "--days", "-d", help="Simulation duration"),
) -> None:
    """Compare multiple scenarios"""
    
    context = get_cli_context()
    command = CompareCommand(context)
    
    result = command.execute(
        scenarios_file=scenarios_file, output_dir=output_dir,
        simulation_days=simulation_days
    )
    
    if not result.success:
        raise typer.Exit(result.exit_code)


# Parameter management commands
@app.command(name="load-params")
def load_parameters(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    param_type: str = typer.Option("behavioral", help="Parameter type (behavioral/environmental/colony)"),
    format_type: str = typer.Option("csv", help="File format (csv/json/xlsx)"),
    validate: bool = typer.Option(True, help="Validate parameters after loading"),
    output_dir: str = typer.Option("artifacts/parameters", help="Output directory for validated parameters")
) -> None:
    """Load parameters from file"""
    load_parameters_cmd(file_path, param_type, format_type, validate, output_dir)


@app.command(name="inspect-params")
def inspect_parameters(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    details: bool = typer.Option(False, "--details", help="Show detailed data"),
    schema: bool = typer.Option(False, "--schema", help="Export schema")
) -> None:
    """Inspect parameter file contents"""
    inspect_parameters_cmd(file_path, details, schema)


@app.command(name="validate-params")
def validate_parameters(
    file_path: str = typer.Argument(..., help="Path to parameter file"),
    param_type: str = typer.Option("behavioral", help="Parameter type"),
    strict: bool = typer.Option(False, "--strict", help="Strict validation mode")
) -> None:
    """Validate parameter file"""
    validate_parameters_cmd(file_path, param_type, strict)


@app.command(name="convert-params")
def convert_parameters(
    input_file: str = typer.Argument(..., help="Input parameter file"),
    output_file: str = typer.Argument(..., help="Output parameter file"),
    input_format: str = typer.Option("csv", help="Input format"),
    output_format: str = typer.Option("json", help="Output format"),
    param_type: str = typer.Option("behavioral", help="Parameter type")
) -> None:
    """Convert parameters between formats"""
    convert_parameters_cmd(input_file, output_file, input_format, output_format, param_type)


@app.command(name="discover-netlogo")
def discover_netlogo(
    directory: str = typer.Argument(..., help="Directory to search for NetLogo files"),
    export: bool = typer.Option(False, "--export", help="Export discovery report"),
    output_dir: str = typer.Option("artifacts/netlogo_discovery", help="Output directory for reports")
) -> None:
    """Discover NetLogo files in directory"""
    discover_netlogo_cmd(directory, export, output_dir)


@app.command(name="validate-netlogo")
def validate_netlogo(
    directory: str = typer.Argument(..., help="Directory containing NetLogo files"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix compatibility issues")
) -> None:
    """Validate NetLogo data compatibility"""
    validate_netlogo_cmd(directory, fix)


@app.command(name="convert-netlogo")
def convert_netlogo(
    netlogo_directory: str = typer.Argument(..., help="NetLogo data directory"),
    output_directory: str = typer.Option("artifacts/netlogo_converted", help="Output directory"),
    validate_first: bool = typer.Option(True, "--validate/--no-validate", help="Validate before conversion")
) -> None:
    """Convert NetLogo data to BSTEW format"""
    convert_netlogo_cmd(netlogo_directory, output_directory, validate_first)


@app.command(name="create-template")
def create_template(
    param_type: str = typer.Option("behavioral", help="Parameter type"),
    output_file: str = typer.Option("artifacts/parameters/template.csv", help="Output file"),
    format_type: str = typer.Option("csv", help="File format")
) -> None:
    """Create parameter template file"""
    create_template_cmd(param_type, output_file, format_type)


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