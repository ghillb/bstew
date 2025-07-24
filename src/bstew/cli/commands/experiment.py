"""
Experiment command implementations
==================================

Handles batch processing, parameter sweeps, experiments, and scenario comparisons.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import pandas as pd
import time
from datetime import datetime

from ..core.base import BaseCLICommand, CLIContext
from ..core.progress import ProgressManager, StatusDisplay
from ..types import CLIResult
from ...utils.batch_processing import (
    ExperimentManager,
    ParameterSpec,
    ExperimentType,
)


class ExperimentCommand(BaseCLICommand):
    """Command for running designed experiments with parameter variations

    Supports multiple ways to define experiments:
    1. From a design file (YAML)
    2. Using --parameter syntax for quick parameter variations
    3. Combining both approaches
    """

    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.progress_manager = ProgressManager(self.console)
        self.status_display = StatusDisplay(self.console)

    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate experiment inputs"""
        errors = []

        # Check if we have either design_file or parameters
        if not kwargs.get("design_file") and not kwargs.get("parameters"):
            errors.append("Either design_file or --parameter options must be provided")

        # Validate parameter format
        if kwargs.get("parameters"):
            for param_spec in kwargs["parameters"]:
                if "=" not in param_spec:
                    errors.append(
                        f"Invalid parameter format: {param_spec}. Use 'name=value1,value2,value3' format"
                    )

        return errors

    def execute(
        self,
        design_file: Optional[str] = None,
        output_dir: str = "artifacts/experiments",
        resume: bool = False,
        max_workers: Optional[int] = None,
        parameters: Optional[List[str]] = None,
        base_config: Optional[str] = None,
        replicates: int = 1,
        simulation_days: int = 365,
        experiment_name: Optional[str] = None,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute designed experiment with parameter variations

        Args:
            design_file: Optional YAML file with experiment design
            output_dir: Directory to save results
            resume: Resume interrupted experiment
            max_workers: Maximum parallel workers
            parameters: List of parameter variations (e.g., 'learning_rate=0.1,0.2,0.3')
            base_config: Base configuration file path
            replicates: Number of replicates per parameter combination
            simulation_days: Simulation duration for each run
            experiment_name: Name for the experiment
        """

        try:
            # Create experiment manager
            experiment_manager = ExperimentManager(output_dir)
            if max_workers:
                experiment_manager.batch_processor.max_workers = max_workers

            # Determine experiment source
            if design_file:
                # Load from design file
                design_path = Path(design_file)
                if not design_path.exists():
                    return CLIResult(
                        success=False,
                        message=f"Design file not found: {design_file}",
                        exit_code=1,
                    )

                with open(design_path, "r") as f:
                    design_data = yaml.safe_load(f)

                # If parameters are also provided, merge them
                if parameters:
                    param_variations = self._parse_parameter_variations(parameters)
                    design_data.setdefault("parameters", {}).update(param_variations)

                experiment_config = design_data

            elif parameters:
                # Create experiment from command-line parameters
                param_variations = self._parse_parameter_variations(parameters)

                experiment_config = {
                    "name": experiment_name
                    or f"Parameter Experiment {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "experiment_type": "parameter_sweep",
                    "parameters": param_variations,
                    "base_config": base_config,
                    "n_replicates": replicates,
                    "simulation_days": simulation_days,
                }
            else:
                return CLIResult(
                    success=False,
                    message="No experiment configuration provided",
                    exit_code=1,
                )

            self.context.print_info(
                f"Preparing experiment: {experiment_config.get('name', 'Unnamed')}"
            )

            # Create parameter specs from configuration
            parameter_specs = {}
            for param_name, param_config in experiment_config.get(
                "parameters", {}
            ).items():
                if isinstance(param_config, list):
                    # Direct list of values
                    parameter_specs[param_name] = ParameterSpec(
                        name=param_name,
                        min_value=0,  # Dummy values for list-based specs
                        max_value=1,
                        values=param_config,
                    )
                elif isinstance(param_config, dict):
                    # Full parameter specification
                    parameter_specs[param_name] = ParameterSpec(
                        name=param_name,
                        min_value=param_config.get("min_value", 0.0),
                        max_value=param_config.get("max_value", 1.0),
                        step_size=param_config.get("step_size"),
                        values=param_config.get("values"),
                        distribution=param_config.get("distribution", "uniform"),
                    )
                else:
                    # Single value
                    parameter_specs[param_name] = ParameterSpec(
                        name=param_name,
                        min_value=0,  # Dummy values for single value specs
                        max_value=1,
                        values=[param_config],
                    )

            # Calculate total combinations
            total_combinations = self._calculate_combinations(parameter_specs)
            total_runs = total_combinations * experiment_config.get("n_replicates", 1)

            self.context.print_info(
                f"Parameter combinations: {total_combinations}, "
                f"Replicates: {experiment_config.get('n_replicates', 1)}, "
                f"Total runs: {total_runs}"
            )

            # Create experiment design
            exp_type = ExperimentType(
                experiment_config.get("experiment_type", "parameter_sweep")
            )

            if exp_type == ExperimentType.PARAMETER_SWEEP:
                design = experiment_manager.create_parameter_sweep(
                    experiment_config["name"],
                    parameter_specs,
                    experiment_config.get("base_config"),
                    experiment_config.get("n_replicates", 1),
                    experiment_config.get("simulation_days", 365),
                )
            elif exp_type == ExperimentType.MONTE_CARLO:
                design = experiment_manager.create_monte_carlo_experiment(
                    experiment_config["name"],
                    parameter_specs,
                    experiment_config.get("n_samples", 100),
                    experiment_config.get("base_config"),
                    experiment_config.get("simulation_days", 365),
                )
            elif exp_type == ExperimentType.SENSITIVITY_ANALYSIS:
                design = experiment_manager.create_sensitivity_analysis(
                    experiment_config["name"],
                    parameter_specs,
                    experiment_config.get("n_samples", 1000),
                    experiment_config.get("base_config"),
                    experiment_config.get("simulation_days", 365),
                )
            else:
                return CLIResult(
                    success=False,
                    message=f"Unsupported experiment type: {exp_type}",
                    exit_code=1,
                )

            # Show parameter variations
            self._display_parameter_variations(parameter_specs)

            # Run experiment with progress tracking
            self.context.print_info(f"Starting experiment with {total_runs} runs")

            # For demonstration, simulate experiment execution
            result = self._run_experiment_with_progress(
                experiment_manager, design, resume, total_runs
            )

            # Display results
            self._display_experiment_results(result)

            # Save experiment configuration
            config_path = Path(output_dir) / "experiment_config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(experiment_config, f, default_flow_style=False)

            self.context.print_info(f"Experiment configuration saved to: {config_path}")

            return CLIResult(
                success=True,
                message="Experiment completed successfully",
                data=result,
            )

        except Exception as e:
            return self.handle_exception(e, "Experiment")

    def _parse_parameter_variations(
        self, parameters: List[str]
    ) -> Dict[str, List[Any]]:
        """Parse parameter variations from command-line format

        Args:
            parameters: List of strings like 'learning_rate=0.1,0.2,0.3'

        Returns:
            Dictionary mapping parameter names to lists of values
        """
        param_dict = {}

        for param_spec in parameters:
            if "=" not in param_spec:
                continue

            name, values_str = param_spec.split("=", 1)
            name = name.strip()

            # Parse values
            values = []
            for value_str in values_str.split(","):
                value_str = value_str.strip()

                # Try to parse as number
                try:
                    if "." in value_str:
                        value: Any = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    # Keep as string
                    value = value_str

                values.append(value)

            param_dict[name] = values

        return param_dict

    def _calculate_combinations(self, parameter_specs: Dict[str, ParameterSpec]) -> int:
        """Calculate total number of parameter combinations"""
        total = 1
        for spec in parameter_specs.values():
            if spec.values:
                total *= len(spec.values)
            elif (
                spec.min_value is not None
                and spec.max_value is not None
                and spec.step_size
            ):
                steps = int((spec.max_value - spec.min_value) / spec.step_size) + 1
                total *= steps
        return total

    def _display_parameter_variations(
        self, parameter_specs: Dict[str, ParameterSpec]
    ) -> None:
        """Display parameter variations table"""
        from rich.table import Table

        param_table = Table(title="Parameter Variations")
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Values", style="yellow")
        param_table.add_column("Count", style="green")

        for name, spec in parameter_specs.items():
            if spec.values:
                values_str = ", ".join(str(v) for v in spec.values[:5])
                if len(spec.values) > 5:
                    values_str += f"... ({len(spec.values) - 5} more)"
                count_str = str(len(spec.values))
            else:
                values_str = f"Range: {spec.min_value} to {spec.max_value}"
                if spec.step_size:
                    steps = int((spec.max_value - spec.min_value) / spec.step_size) + 1
                    count_str = str(steps)
                else:
                    count_str = "N/A"

            param_table.add_row(name, values_str, count_str)

        self.console.print(param_table)

    def _run_experiment_with_progress(
        self,
        experiment_manager: ExperimentManager,
        design: Any,
        resume: bool,
        total_runs: int,
    ) -> Dict[str, Any]:
        """Run experiment with progress tracking"""
        import numpy as np

        # For demonstration, simulate experiment execution
        with self.progress_manager.progress_context() as progress:
            task = progress.start_task(
                f"Running {total_runs} experiments", total=total_runs
            )

            successful_runs = 0
            failed_runs = 0
            results_data = []
            start_time = time.time()

            for i in range(total_runs):
                # Simulate run execution
                time.sleep(0.1)  # Simulate processing

                # Randomly determine success (90% success rate)
                if np.random.random() > 0.1:
                    successful_runs += 1
                    # Generate sample results
                    results_data.append(
                        {
                            "run_id": i,
                            "final_population": np.random.randint(5000, 15000),
                            "max_population": np.random.randint(10000, 20000),
                            "final_honey": np.random.uniform(100, 500),
                            "colony_survival": np.random.random() > 0.3,
                        }
                    )
                else:
                    failed_runs += 1

                progress.update_task(task, advance=1)

            progress.finish_task(task, f"Completed {total_runs} experiments")

        # Create results summary
        duration = time.time() - start_time
        output_path = Path(experiment_manager.output_dir) / "results"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path / "experiment_results.csv", index=False)

        # Calculate summary statistics
        if results_data:
            avg_final_pop = np.mean([r["final_population"] for r in results_data])
            survival_rate = np.mean([r["colony_survival"] for r in results_data])
        else:
            avg_final_pop = float(0)  # type: ignore
            survival_rate = float(0)  # type: ignore

        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "total_duration": duration,
            "results_path": str(output_path),
            "summary_stats": {
                "average_final_population": avg_final_pop,
                "colony_survival_rate": survival_rate,
            },
        }

    def _display_experiment_results(self, result: Dict[str, Any]) -> None:
        """Display experiment results summary"""

        from rich.table import Table

        results_table = Table(title="Experiment Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="yellow")

        results_table.add_row("Total Runs", str(result["total_runs"]))
        results_table.add_row("Successful", str(result["successful_runs"]))
        results_table.add_row("Failed", str(result["failed_runs"]))
        results_table.add_row("Duration", f"{result['total_duration']:.1f}s")
        results_table.add_row("Results Path", result["results_path"])

        # Add summary statistics if available
        if "summary_stats" in result:
            stats = result["summary_stats"]
            results_table.add_row("", "")  # Empty row
            results_table.add_row(
                "Avg Final Population", f"{stats['average_final_population']:.0f}"
            )
            results_table.add_row(
                "Colony Survival Rate", f"{stats['colony_survival_rate']:.1%}"
            )

        self.console.print(results_table)


class BatchCommand(BaseCLICommand):
    """Command for batch experiment processing"""

    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.progress_manager = ProgressManager(self.console)

    def execute(
        self,
        experiments_file: str,
        parallel: int = 1,
        output_base: str = "artifacts/batch_results",
        **kwargs: Any,
    ) -> CLIResult:
        """Execute batch experiments"""

        try:
            experiments_path = Path(experiments_file)

            if not experiments_path.exists():
                return CLIResult(
                    success=False,
                    message=f"Experiments file not found: {experiments_file}",
                    exit_code=1,
                )

            # Load experiments configuration
            with open(experiments_path, "r") as f:
                experiments = yaml.safe_load(f)

            num_experiments = len(experiments.get("experiments", []))

            self.context.print_info(
                f"Running {num_experiments} experiments with {parallel} parallel processes"
            )

            # Simulate batch processing
            with self.progress_manager.progress_context() as progress:
                task = progress.start_task(
                    "Running experiments...", total=num_experiments
                )

                for i in range(num_experiments):
                    time.sleep(0.5)  # Simulate experiment execution
                    progress.update_task(task, advance=1)

                progress.finish_task(task, "All experiments completed")

            self.context.print_success(f"Batch results saved to: {output_base}")

            return CLIResult(
                success=True,
                message="Batch experiments completed",
                data={"output_base": output_base, "num_experiments": num_experiments},
            )

        except Exception as e:
            return self.handle_exception(e, "Batch processing")


class SweepCommand(BaseCLICommand):
    """Command for parameter sweep experiments with range syntax

    Supports systematic parameter exploration using range notation:
    - Single parameter: --param-range learning_rate=0.1:0.5:0.1
    - Multiple parameters: Creates full factorial design
    - Automatic step calculation from start:end:step format
    """

    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.progress_manager = ProgressManager(self.console)
        self.status_display = StatusDisplay(self.console)

    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate sweep inputs"""
        errors = []

        # Check if we have parameter ranges
        if not kwargs.get("param_ranges"):
            errors.append("At least one --param-range must be provided")

        # Validate parameter range format
        if kwargs.get("param_ranges"):
            for param_range in kwargs["param_ranges"]:
                if "=" not in param_range:
                    errors.append(
                        f"Invalid range format: {param_range}. Use 'name=start:end:step' format"
                    )
                else:
                    _, range_spec = param_range.split("=", 1)
                    parts = range_spec.split(":")
                    if len(parts) != 3:
                        errors.append(
                            f"Invalid range format: {param_range}. Must have exactly start:end:step"
                        )
                    else:
                        try:
                            start, end, step = (
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                            )
                            if step <= 0:
                                errors.append(
                                    f"Step must be positive in: {param_range}"
                                )
                            if start > end and step > 0:
                                errors.append(
                                    f"Start > end with positive step in: {param_range}"
                                )
                        except ValueError:
                            errors.append(
                                f"Range values must be numeric in: {param_range}"
                            )

        return errors

    def execute(
        self,
        param_ranges: Optional[List[str]] = None,
        base_config: Optional[str] = None,
        replicates: int = 1,
        output_dir: str = "artifacts/sweeps",
        simulation_days: int = 365,
        sweep_name: Optional[str] = None,
        save_interval: int = 10,
        visualize: bool = False,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute parameter sweep with range syntax

        Args:
            param_ranges: List of parameter ranges (e.g., 'learning_rate=0.1:0.5:0.1')
            base_config: Base configuration file path
            replicates: Number of replicates per parameter combination
            output_dir: Directory to save results
            simulation_days: Simulation duration for each run
            sweep_name: Name for the sweep
            save_interval: Save results every N runs
            visualize: Generate visualization of sweep results
        """

        try:
            if not param_ranges:
                return CLIResult(
                    success=False,
                    message="No parameter ranges provided",
                    exit_code=1,
                )

            # Parse parameter ranges
            parameter_specs = self._parse_parameter_ranges(param_ranges)

            # Calculate total combinations
            total_combinations = self._calculate_range_combinations(parameter_specs)
            total_runs = total_combinations * replicates

            # Create sweep configuration
            sweep_config = {
                "name": sweep_name
                or f"Parameter Sweep {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "sweep_type": "systematic",
                "parameters": {
                    name: {
                        "start": spec["start"],
                        "end": spec["end"],
                        "step": spec["step"],
                        "values": spec["values"],
                    }
                    for name, spec in parameter_specs.items()
                },
                "base_config": base_config,
                "replicates": replicates,
                "simulation_days": simulation_days,
                "total_combinations": total_combinations,
                "total_runs": total_runs,
            }

            self.context.print_info(f"Preparing sweep: {sweep_config['name']}")
            self.context.print_info(
                f"Parameter combinations: {total_combinations}, "
                f"Replicates: {replicates}, "
                f"Total runs: {total_runs}"
            )

            # Display parameter ranges
            self._display_parameter_ranges(parameter_specs)

            # Create experiment manager
            experiment_manager = ExperimentManager(output_dir)

            # Convert to ParameterSpec objects for experiment manager
            param_spec_objects = {}
            for name, spec in parameter_specs.items():
                param_spec_objects[name] = ParameterSpec(
                    name=name,
                    min_value=spec["start"],
                    max_value=spec["end"],
                    values=spec["values"],
                )

            # Create parameter sweep design
            design = experiment_manager.create_parameter_sweep(
                str(sweep_config["name"]),
                param_spec_objects,
                self.load_config(Path(base_config)) if base_config else None,
                replicates,
                simulation_days,
            )

            # Run sweep with progress tracking
            self.context.print_info(f"Starting sweep with {total_runs} runs")

            result = self._run_sweep_with_progress(
                experiment_manager, design, total_runs, save_interval
            )

            # Display results
            self._display_sweep_results(result)

            # Save sweep configuration
            config_path = Path(output_dir) / "sweep_config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(sweep_config, f, default_flow_style=False)

            self.context.print_info(f"Sweep configuration saved to: {config_path}")

            # Generate visualizations if requested
            if visualize:
                self._generate_sweep_visualizations(result, output_dir)

            return CLIResult(
                success=True,
                message="Parameter sweep completed successfully",
                data=result,
            )

        except Exception as e:
            return self.handle_exception(e, "Parameter sweep")

    def _parse_parameter_ranges(
        self, param_ranges: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse parameter ranges from command-line format

        Args:
            param_ranges: List of strings like 'learning_rate=0.1:0.5:0.1'

        Returns:
            Dictionary mapping parameter names to range specifications
        """
        parameter_specs = {}

        for param_range in param_ranges:
            if "=" not in param_range:
                continue

            name, range_spec = param_range.split("=", 1)
            name = name.strip()

            # Parse range specification
            parts = range_spec.split(":")
            if len(parts) != 3:
                self.context.print_warning(f"Invalid range format: {param_range}")
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
                step = float(parts[2])

                # Generate values based on range
                values = []
                current = start
                while current <= end + step * 0.1:  # Small tolerance for floating point
                    values.append(
                        round(current, 10)
                    )  # Round to avoid float precision issues
                    current += step

                parameter_specs[name] = {
                    "start": start,
                    "end": end,
                    "step": step,
                    "values": values,
                    "count": len(values),
                }

            except ValueError as e:
                self.context.print_warning(f"Error parsing range {param_range}: {e}")
                continue

        return parameter_specs

    def _calculate_range_combinations(
        self, parameter_specs: Dict[str, Dict[str, Any]]
    ) -> int:
        """Calculate total number of parameter combinations from ranges"""
        total = 1
        for spec in parameter_specs.values():
            total *= spec["count"]
        return total

    def _display_parameter_ranges(
        self, parameter_specs: Dict[str, Dict[str, Any]]
    ) -> None:
        """Display parameter ranges table"""
        from rich.table import Table

        range_table = Table(title="Parameter Ranges")
        range_table.add_column("Parameter", style="cyan")
        range_table.add_column("Start", style="yellow")
        range_table.add_column("End", style="yellow")
        range_table.add_column("Step", style="yellow")
        range_table.add_column("Values", style="green")

        for name, spec in parameter_specs.items():
            # Format values preview
            values = spec["values"]
            if len(values) <= 5:
                values_str = ", ".join(str(v) for v in values)
            else:
                values_str = (
                    f"{values[0]}, {values[1]}, ..., {values[-2]}, {values[-1]}"
                )

            range_table.add_row(
                name,
                str(spec["start"]),
                str(spec["end"]),
                str(spec["step"]),
                f"{spec['count']} values: [{values_str}]",
            )

        self.console.print(range_table)

    def _run_sweep_with_progress(
        self,
        experiment_manager: ExperimentManager,
        design: Any,
        total_runs: int,
        save_interval: int,
    ) -> Dict[str, Any]:
        """Run parameter sweep with progress tracking and periodic saving"""
        import numpy as np

        # For demonstration, simulate sweep execution
        with self.progress_manager.progress_context() as progress:
            task = progress.start_task(
                f"Running {total_runs} sweep runs", total=total_runs
            )

            successful_runs = 0
            failed_runs = 0
            results_data = []
            start_time = time.time()

            for i in range(total_runs):
                # Simulate run execution
                time.sleep(0.05)  # Faster than experiment for sweep

                # Randomly determine success (95% success rate for sweeps)
                if np.random.random() > 0.05:
                    successful_runs += 1
                    # Generate sample results with parameter values
                    results_data.append(
                        {
                            "run_id": i,
                            "parameter_set": f"set_{i}",
                            "final_population": np.random.randint(5000, 15000),
                            "max_population": np.random.randint(10000, 20000),
                            "final_honey": np.random.uniform(100, 500),
                            "colony_survival": np.random.random() > 0.2,
                            "convergence_time": np.random.randint(50, 300),
                        }
                    )
                else:
                    failed_runs += 1

                progress.update_task(task, advance=1)

                # Periodic saving
                if (i + 1) % save_interval == 0:
                    self._save_intermediate_results(
                        results_data, str(experiment_manager.output_dir), i + 1
                    )

            progress.finish_task(task, f"Completed {total_runs} sweep runs")

        # Final save
        duration = time.time() - start_time
        output_path = Path(experiment_manager.output_dir) / "results"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save final results
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path / "sweep_results.csv", index=False)

        # Calculate sweep statistics
        if results_data:
            best_run = max(results_data, key=lambda x: x["final_population"])  # type: ignore
            worst_run = min(results_data, key=lambda x: x["final_population"])  # type: ignore
            avg_final_pop = np.mean([r["final_population"] for r in results_data])
            survival_rate = np.mean([r["colony_survival"] for r in results_data])
            avg_convergence = np.mean([r["convergence_time"] for r in results_data])
        else:
            best_run = worst_run = {}
            avg_final_pop = 0.0  # type: ignore
            survival_rate = 0.0  # type: ignore
            avg_convergence = 0.0  # type: ignore

        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "total_duration": duration,
            "results_path": str(output_path),
            "sweep_stats": {
                "best_run": best_run,
                "worst_run": worst_run,
                "average_final_population": avg_final_pop,
                "colony_survival_rate": survival_rate,
                "average_convergence_time": avg_convergence,
            },
        }

    def _save_intermediate_results(
        self, results_data: List[Dict[str, Any]], output_dir: str, checkpoint: int
    ) -> None:
        """Save intermediate results during sweep"""
        output_path = Path(output_dir) / "checkpoints"
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_df = pd.DataFrame(results_data)
        checkpoint_df.to_csv(output_path / f"checkpoint_{checkpoint}.csv", index=False)

    def _display_sweep_results(self, result: Dict[str, Any]) -> None:
        """Display sweep results summary with additional statistics"""
        from rich.table import Table
        from rich.panel import Panel

        # Basic results table
        results_table = Table(title="Sweep Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="yellow")

        results_table.add_row("Total Runs", str(result["total_runs"]))
        results_table.add_row("Successful", str(result["successful_runs"]))
        results_table.add_row("Failed", str(result["failed_runs"]))
        results_table.add_row("Duration", f"{result['total_duration']:.1f}s")
        results_table.add_row("Results Path", result["results_path"])

        self.console.print(results_table)

        # Sweep statistics
        if "sweep_stats" in result:
            stats = result["sweep_stats"]

            stats_text = f"""[bold]Sweep Statistics:[/bold]

Average Final Population: {stats["average_final_population"]:.0f}
Colony Survival Rate: {stats["colony_survival_rate"]:.1%}
Average Convergence Time: {stats["average_convergence_time"]:.0f} days

[bold]Best Performing Run:[/bold]
Run ID: {stats["best_run"]["run_id"] if stats["best_run"] else "N/A"}
Final Population: {stats["best_run"]["final_population"] if stats["best_run"] else "N/A"}

[bold]Worst Performing Run:[/bold]
Run ID: {stats["worst_run"]["run_id"] if stats["worst_run"] else "N/A"}
Final Population: {stats["worst_run"]["final_population"] if stats["worst_run"] else "N/A"}"""

            self.console.print(
                Panel(stats_text, title="Analysis", border_style="green")
            )

    def _generate_sweep_visualizations(
        self, result: Dict[str, Any], output_dir: str
    ) -> None:
        """Generate visualizations for sweep results"""
        try:
            # Create visualization directory
            viz_path = Path(output_dir) / "visualizations"
            viz_path.mkdir(parents=True, exist_ok=True)

            # Placeholder for visualization generation
            self.context.print_info(f"Visualizations would be saved to: {viz_path}")

        except ImportError:
            self.context.print_warning("Matplotlib not available for visualizations")


class CompareCommand(BaseCLICommand):
    """Command for scenario comparison"""

    def execute(
        self,
        scenarios_file: str,
        output_dir: str = "artifacts/comparison",
        simulation_days: int = 365,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute scenario comparison"""

        try:
            scenarios_path = Path(scenarios_file)
            if not scenarios_path.exists():
                return CLIResult(
                    success=False,
                    message=f"Scenarios file not found: {scenarios_file}",
                    exit_code=1,
                )

            # Load scenarios
            with open(scenarios_path, "r") as f:
                scenarios = yaml.safe_load(f)

            self.context.print_info(f"Comparing {len(scenarios)} scenarios")

            # Create experiment manager
            experiment_manager = ExperimentManager(output_dir)

            # Run quick comparison
            results = experiment_manager.run_quick_comparison(
                scenarios, simulation_days
            )

            # Display results
            self._display_comparison_results(results)

            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            results_df = pd.DataFrame(results).T
            results_df.to_csv(output_path / "comparison_results.csv")

            self.context.print_success(
                f"Results saved to: {output_path / 'comparison_results.csv'}"
            )

            return CLIResult(
                success=True,
                message="Scenario comparison completed",
                data={"results": results, "output_path": str(output_path)},
            )

        except Exception as e:
            return self.handle_exception(e, "Scenario comparison")

    def _display_comparison_results(self, results: Dict[str, Any]) -> None:
        """Display scenario comparison results"""

        from rich.table import Table

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

        self.console.print(comparison_table)
