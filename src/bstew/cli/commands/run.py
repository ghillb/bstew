"""
Run command implementation
=========================

Handles simulation execution with configuration overrides and progress tracking.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core.base import BaseCLICommand, CLIContext
from ..core.progress import ProgressManager, StatusDisplay
from ..types import CLIResult


class RunCommand(BaseCLICommand):
    """Command for running BSTEW simulations"""

    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.progress_manager = ProgressManager(self.console)
        self.status_display = StatusDisplay(self.console)

    def execute(
        self,
        config: str = "configs/default.yaml",
        output: str = "artifacts/results",
        days: Optional[int] = None,
        seed: Optional[int] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        replicates: int = 1,
        verbose: bool = False,
        quiet: bool = False,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute simulation run"""

        try:
            # Show banner unless in quiet mode
            if not quiet:
                self.status_display.show_banner()

            # Load and validate configuration
            sim_config = self.load_config(config)

            # Apply command line overrides
            overrides = {
                "simulation.duration_days": days,
                "simulation.random_seed": seed,
                "output.output_directory": output,
            }
            sim_config = self.apply_config_overrides(sim_config, overrides)

            # Validate final configuration
            self.validate_config(sim_config)

            # Create output directory
            output_path = self.ensure_output_dir(output)

            # Display simulation info
            if not quiet:
                self.status_display.show_simulation_info(sim_config)

            # Determine execution mode
            if parallel and replicates > 1:
                # Run parallel simulations
                results = self._run_parallel_simulations(
                    sim_config, replicates, max_workers, output_path, verbose, quiet
                )

                if not quiet:
                    self.context.print_success(
                        f"Parallel simulation completed! {replicates} replicates finished."
                    )
                    self.context.print_info(
                        f"Results saved to: {output_path.absolute()}"
                    )

                return CLIResult(
                    success=True,
                    message=f"Parallel simulation completed ({replicates} replicates)",
                    data={
                        "results": results,
                        "output_path": str(output_path),
                        "replicates": replicates,
                    },
                )

            elif replicates > 1:
                # Run sequential replicates
                results = self._run_sequential_replicates(
                    sim_config, replicates, output_path, verbose, quiet
                )

                if not quiet:
                    self.context.print_success(
                        f"Sequential simulation completed! {replicates} replicates finished."
                    )
                    self.context.print_info(
                        f"Results saved to: {output_path.absolute()}"
                    )

                return CLIResult(
                    success=True,
                    message=f"Sequential simulation completed ({replicates} replicates)",
                    data={
                        "results": results,
                        "output_path": str(output_path),
                        "replicates": replicates,
                    },
                )

            else:
                # Run single simulation
                results = self._run_simulation_with_progress(sim_config, verbose, quiet)

                if not quiet:
                    self.context.print_success("Simulation completed successfully!")
                    self.context.print_info(
                        f"Results saved to: {output_path.absolute()}"
                    )

                return CLIResult(
                    success=True,
                    message="Simulation completed successfully",
                    data={"results": results, "output_path": str(output_path)},
                )

        except KeyboardInterrupt:
            self.context.print_warning("Simulation interrupted by user")
            return CLIResult(
                success=False,
                message="Simulation interrupted",
                exit_code=130,
            )
        except Exception as e:
            return self.handle_exception(e, "Simulation")

    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate command inputs"""
        errors = []

        # Validate config file exists
        config_path = kwargs.get("config", "configs/default.yaml")
        if not Path(config_path).exists():
            errors.append(f"Configuration file not found: {config_path}")

        # Validate optional parameters
        days = kwargs.get("days")
        if days is not None and (not isinstance(days, int) or days <= 0):
            errors.append("Days must be a positive integer")

        seed = kwargs.get("seed")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            errors.append("Seed must be a non-negative integer")

        # Validate parallel execution parameters
        parallel = kwargs.get("parallel", False)
        max_workers = kwargs.get("max_workers")
        replicates = kwargs.get("replicates", 1)

        if parallel and max_workers is not None:
            if not isinstance(max_workers, int) or max_workers <= 0:
                errors.append("max_workers must be a positive integer")

        if not isinstance(replicates, int) or replicates <= 0:
            errors.append("replicates must be a positive integer")

        if parallel and replicates == 1:
            errors.append("parallel execution requires replicates > 1")

        return errors

    def _run_simulation_with_progress(
        self,
        config: Dict[str, Any],
        verbose: bool,
        quiet: bool,
    ) -> Dict[str, Any]:
        """Run simulation with progress tracking"""

        # Extract simulation parameters
        simulation_config = config.get("simulation", {})
        duration_days = simulation_config.get("duration_days", 365)

        if quiet:
            # Run without progress bars
            return self._run_simulation(config, duration_days, verbose)

        # Run with progress tracking
        from ...core.model import BeeModel

        with self.progress_manager.progress_context() as progress:
            # Initialization
            init_task = progress.start_task("Initializing simulation...", total=5)

            # Create BeeModel instance from config
            seed = config.get("simulation", {}).get("random_seed")
            model = BeeModel(config=config, random_seed=seed)
            progress.update_task(init_task, advance=5)

            progress.finish_task(init_task, "Simulation initialized")

            # Main simulation
            sim_task = progress.start_task("Running simulation...", total=duration_days)

            # Run simulation steps
            for day in range(duration_days):
                model.step()
                progress.update_task(sim_task, advance=1)

                # Periodic verbose updates
                if verbose and day % 30 == 0:
                    pop = model.get_total_population()
                    self.context.print_verbose(f"Day {day}: Colony population {pop}")

            progress.finish_task(sim_task, "Simulation completed")

            # Results saving
            save_task = progress.start_task("Saving results...", total=3)

            # Export results
            output_dir = config.get("output", {}).get("output_directory", "results")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            progress.update_task(save_task, advance=1)

            try:
                model.export_results(output_dir)
                progress.update_task(save_task, advance=1)
                if verbose:
                    self.context.print_verbose(f"Results exported to {output_dir}")
            except Exception as e:
                # Always show export errors
                self.context.print_warning(f"Export failed: {e}")
                if verbose:
                    import traceback

                    self.context.print_debug(traceback.format_exc())

            progress.update_task(save_task, advance=1)
            progress.finish_task(save_task, "Results saved")

            # Get summary statistics
            if hasattr(model, "datacollector") and model.datacollector:
                model_data = model.datacollector.get_model_vars_dataframe()
                if not model_data.empty:
                    final_pop = (
                        model_data["TotalPopulation"].iloc[-1]
                        if "TotalPopulation" in model_data.columns
                        else 0
                    )
                    max_pop = (
                        model_data["TotalPopulation"].max()
                        if "TotalPopulation" in model_data.columns
                        else 0
                    )
                else:
                    final_pop = max_pop = 0
            else:
                final_pop = max_pop = 0

            return {
                "final_population": final_pop,
                "max_population": max_pop,
                "total_honey_produced": 0.0,  # Would need to calculate from model data
                "foraging_efficiency": 0.0,  # Would need to calculate from model data
                "colony_survival": True,
                "simulation_days": duration_days,
            }

    def _run_simulation(
        self,
        config: Dict[str, Any],
        duration_days: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        """Run actual simulation with BeeModel integration"""

        from ...core.model import BeeModel
        from pathlib import Path

        if verbose:
            self.context.print_verbose("Initializing bee model...")

        # Create BeeModel instance from config
        seed = config.get("simulation", {}).get("random_seed")
        model = BeeModel(config=config, random_seed=seed)

        if verbose:
            self.context.print_verbose(
                f"Running simulation for {duration_days} days..."
            )

        # Run the simulation
        model.run_simulation(days=duration_days)

        if verbose:
            self.context.print_verbose("Collecting results...")

        # Export results to output directory
        output_dir = config.get("output", {}).get("output_directory", "results")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            model.export_results(output_dir)
            if verbose:
                self.context.print_verbose(f"Results exported to {output_dir}")
        except Exception as e:
            # Always show export errors, not just in verbose mode
            self.context.print_warning(f"Export failed: {e}")
            if verbose:
                import traceback

                self.context.print_debug(traceback.format_exc())

        # Get summary statistics from data collector
        if hasattr(model, "datacollector") and model.datacollector:
            model_data = model.datacollector.get_model_vars_dataframe()
            if not model_data.empty:
                final_pop = (
                    model_data["TotalPopulation"].iloc[-1]
                    if "TotalPopulation" in model_data.columns
                    else 0
                )
                max_pop = (
                    model_data["TotalPopulation"].max()
                    if "TotalPopulation" in model_data.columns
                    else 0
                )
            else:
                final_pop = max_pop = 0
        else:
            final_pop = max_pop = 0

        return {
            "final_population": final_pop,
            "max_population": max_pop,
            "total_honey_produced": 0.0,  # Would need to calculate from model data
            "foraging_efficiency": 0.0,  # Would need to calculate from model data
            "colony_survival": True,
            "simulation_days": duration_days,
        }

    def _run_parallel_simulations(
        self,
        config: Dict[str, Any],
        replicates: int,
        max_workers: Optional[int],
        output_path: Path,
        verbose: bool,
        quiet: bool,
    ) -> Dict[str, Any]:
        """Run multiple simulation replicates in parallel using ProcessPoolExecutor"""

        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from copy import deepcopy
        import time

        # Determine number of workers
        if max_workers is None:
            max_workers = min(replicates, os.cpu_count() or 1)
        else:
            max_workers = min(max_workers, replicates)

        if not quiet:
            self.context.print_info(
                f"Running {replicates} replicates with {max_workers} parallel workers"
            )

        # Prepare replicate configurations with unique output directories and seeds
        replicate_configs = []
        for i in range(replicates):
            replicate_config = deepcopy(config)

            # Set unique output directory for each replicate
            replicate_output = output_path / f"replicate_{i + 1:03d}"
            replicate_config["output"] = {"output_directory": str(replicate_output)}

            # Set unique seed if not specified
            base_seed = config.get("simulation", {}).get("random_seed")
            if base_seed is not None:
                replicate_config["simulation"]["random_seed"] = base_seed + i
            else:
                import random

                replicate_config["simulation"]["random_seed"] = (
                    random.randint(1, 10000) + i
                )

            replicate_configs.append(
                {"config": replicate_config, "replicate_id": i + 1, "verbose": verbose}
            )

        # Execute parallel simulations
        results = []
        failed_replicates = []
        start_time = time.time()

        if quiet:
            # Run without progress bars
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_replicate = {
                    executor.submit(_run_single_replicate, rep_config): rep_config[
                        "replicate_id"
                    ]
                    for rep_config in replicate_configs
                }

                for future in as_completed(future_to_replicate):
                    replicate_id = future_to_replicate[future]
                    try:
                        result = future.result()
                        result["replicate_id"] = replicate_id
                        results.append(result)
                    except Exception as e:
                        failed_replicates.append(
                            {"replicate_id": replicate_id, "error": str(e)}
                        )
        else:
            # Run with progress tracking
            from rich.progress import (
                Progress,
                SpinnerColumn,
                TextColumn,
                BarColumn,
                TaskProgressColumn,
                TimeElapsedColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "Running parallel simulations...", total=replicates
                )

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_replicate = {
                        executor.submit(_run_single_replicate, rep_config): rep_config[
                            "replicate_id"
                        ]
                        for rep_config in replicate_configs
                    }

                    for future in as_completed(future_to_replicate):
                        replicate_id = future_to_replicate[future]
                        try:
                            result = future.result()
                            result["replicate_id"] = replicate_id
                            results.append(result)

                            if verbose:
                                self.context.print_verbose(
                                    f"Replicate {replicate_id} completed"
                                )

                        except Exception as e:
                            failed_replicates.append(
                                {"replicate_id": replicate_id, "error": str(e)}
                            )
                            if verbose:
                                self.context.print_verbose(
                                    f"Replicate {replicate_id} failed: {e}"
                                )

                        progress.advance(task, 1)

        execution_time = time.time() - start_time

        # Generate summary results
        if results:
            aggregated_results = self._aggregate_replicate_results(results)
        else:
            aggregated_results = {"error": "All replicates failed"}

        aggregated_results.update(
            {
                "execution_mode": "parallel",
                "total_replicates": replicates,
                "successful_replicates": len(results),
                "failed_replicates": len(failed_replicates),
                "max_workers": max_workers,
                "total_execution_time": execution_time,
                "failed_replicate_details": failed_replicates
                if failed_replicates
                else None,
            }
        )

        return aggregated_results

    def _run_sequential_replicates(
        self,
        config: Dict[str, Any],
        replicates: int,
        output_path: Path,
        verbose: bool,
        quiet: bool,
    ) -> Dict[str, Any]:
        """Run multiple simulation replicates sequentially"""

        from copy import deepcopy
        import time

        if not quiet:
            self.context.print_info(f"Running {replicates} replicates sequentially")

        results = []
        failed_replicates = []
        start_time = time.time()

        if quiet:
            # Run without progress bars
            for i in range(replicates):
                try:
                    # Prepare replicate configuration
                    replicate_config = deepcopy(config)

                    # Set unique output directory
                    replicate_output = output_path / f"replicate_{i + 1:03d}"
                    replicate_config["output"] = {
                        "output_directory": str(replicate_output)
                    }

                    # Set unique seed
                    base_seed = config.get("simulation", {}).get("random_seed")
                    if base_seed is not None:
                        replicate_config["simulation"]["random_seed"] = base_seed + i
                    else:
                        import random

                        replicate_config["simulation"]["random_seed"] = (
                            random.randint(1, 10000) + i
                        )

                    # Run simulation
                    result = self._run_simulation(
                        replicate_config,
                        replicate_config.get("simulation", {}).get(
                            "duration_days", 365
                        ),
                        verbose,
                    )
                    result["replicate_id"] = i + 1
                    results.append(result)

                except Exception as e:
                    failed_replicates.append({"replicate_id": i + 1, "error": str(e)})
        else:
            # Run with progress tracking
            from rich.progress import (
                Progress,
                SpinnerColumn,
                TextColumn,
                BarColumn,
                TaskProgressColumn,
                TimeElapsedColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "Running sequential simulations...", total=replicates
                )

                for i in range(replicates):
                    try:
                        # Prepare replicate configuration
                        replicate_config = deepcopy(config)

                        # Set unique output directory
                        replicate_output = output_path / f"replicate_{i + 1:03d}"
                        replicate_config["output"] = {
                            "output_directory": str(replicate_output)
                        }

                        # Set unique seed
                        base_seed = config.get("simulation", {}).get("random_seed")
                        if base_seed is not None:
                            replicate_config["simulation"]["random_seed"] = (
                                base_seed + i
                            )
                        else:
                            import random

                            replicate_config["simulation"]["random_seed"] = (
                                random.randint(1, 10000) + i
                            )

                        # Run simulation
                        result = self._run_simulation(
                            replicate_config,
                            replicate_config.get("simulation", {}).get(
                                "duration_days", 365
                            ),
                            verbose,
                        )
                        result["replicate_id"] = i + 1
                        results.append(result)

                        if verbose:
                            self.context.print_verbose(f"Replicate {i + 1} completed")

                    except Exception as e:
                        failed_replicates.append(
                            {"replicate_id": i + 1, "error": str(e)}
                        )
                        if verbose:
                            self.context.print_verbose(f"Replicate {i + 1} failed: {e}")

                    progress.advance(task, 1)

        execution_time = time.time() - start_time

        # Generate summary results
        if results:
            aggregated_results = self._aggregate_replicate_results(results)
        else:
            aggregated_results = {"error": "All replicates failed"}

        aggregated_results.update(
            {
                "execution_mode": "sequential",
                "total_replicates": replicates,
                "successful_replicates": len(results),
                "failed_replicates": len(failed_replicates),
                "total_execution_time": execution_time,
                "failed_replicate_details": failed_replicates
                if failed_replicates
                else None,
            }
        )

        return aggregated_results

    def _aggregate_replicate_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple replicates"""

        import numpy as np

        if not results:
            return {"error": "No results to aggregate"}

        # Extract metrics from all replicates
        metrics = [
            "final_population",
            "max_population",
            "total_honey_produced",
            "foraging_efficiency",
        ]
        aggregated = {}

        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))
                aggregated[f"{metric}_min"] = float(np.min(values))
                aggregated[f"{metric}_max"] = float(np.max(values))
                aggregated[f"{metric}_median"] = float(np.median(values))

        # Add survival rate
        survival_count = sum(1 for r in results if r.get("colony_survival", False))
        aggregated["colony_survival_rate"] = survival_count / len(results)

        # Add individual replicate results
        aggregated["individual_results"] = results  # type: ignore[assignment]

        return aggregated


def _run_single_replicate(replicate_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function to run a single replicate (for ProcessPoolExecutor)

    This function must be defined at module level to be pickle-able for multiprocessing
    """

    from ...core.model import BeeModel
    from pathlib import Path

    config = replicate_config["config"]
    verbose = replicate_config["verbose"]

    try:
        # Extract simulation parameters
        simulation_config = config.get("simulation", {})
        duration_days = simulation_config.get("duration_days", 365)

        # Create BeeModel instance from config
        seed = config.get("simulation", {}).get("random_seed")
        model = BeeModel(config=config, random_seed=seed)

        # Run the simulation
        model.run_simulation(days=duration_days)

        # Export results to output directory
        output_dir = config.get("output", {}).get("output_directory", "results")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            model.export_results(output_dir)
        except Exception as e:
            if verbose:
                print(f"Export failed for replicate: {e}")

        # Get summary statistics from data collector
        if hasattr(model, "datacollector") and model.datacollector:
            model_data = model.datacollector.get_model_vars_dataframe()
            if not model_data.empty:
                final_pop = (
                    model_data["TotalPopulation"].iloc[-1]
                    if "TotalPopulation" in model_data.columns
                    else 0
                )
                max_pop = (
                    model_data["TotalPopulation"].max()
                    if "TotalPopulation" in model_data.columns
                    else 0
                )
            else:
                final_pop = max_pop = 0
        else:
            final_pop = max_pop = 0

        return {
            "final_population": final_pop,
            "max_population": max_pop,
            "total_honey_produced": 0.0,  # Would need to calculate from model data
            "foraging_efficiency": 0.0,  # Would need to calculate from model data
            "colony_survival": True,
            "simulation_days": duration_days,
            "output_directory": output_dir,
        }

    except Exception as e:
        raise RuntimeError(f"Replicate simulation failed: {str(e)}")
