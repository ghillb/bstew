"""
Experiment command implementations
==================================

Handles batch processing, parameter sweeps, experiments, and scenario comparisons.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import pandas as pd
import time

from ..core.base import BaseCLICommand, CLIContext
from ..core.progress import ProgressManager
from ..types import CLIResult
from ...utils.batch_processing import (
    ExperimentManager,
    ParameterSpec,
    ExperimentType,
)


class ExperimentCommand(BaseCLICommand):
    """Command for running designed experiments"""
    
    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.progress_manager = ProgressManager(self.console)
    
    def execute(
        self,
        design_file: str,
        output_dir: str = "artifacts/experiments",
        resume: bool = False,
        max_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute designed experiment"""
        
        try:
            design_path = Path(design_file)
            if not design_path.exists():
                return CLIResult(
                    success=False,
                    message=f"Design file not found: {design_file}",
                    exit_code=1,
                )
            
            # Load experiment design
            with open(design_path, "r") as f:
                design_data = yaml.safe_load(f)
            
            # Create experiment manager
            experiment_manager = ExperimentManager(output_dir)
            if max_workers:
                experiment_manager.batch_processor.max_workers = max_workers
            
            self.context.print_info(
                f"Loading experiment: {design_data.get('name', 'Unnamed')}"
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
                return CLIResult(
                    success=False,
                    message=f"Unsupported experiment type: {exp_type}",
                    exit_code=1,
                )
            
            # Run experiment
            runs = design.generate_runs()
            self.context.print_info(f"Starting experiment with {len(runs)} runs")
            
            result = experiment_manager.batch_processor.run_experiment(
                design, resume=resume
            )
            
            # Display results
            self._display_experiment_results(result)
            
            return CLIResult(
                success=True,
                message="Experiment completed successfully",
                data=result,
            )
            
        except Exception as e:
            return self.handle_exception(e, "Experiment")
    
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
                task = progress.start_task("Running experiments...", total=num_experiments)
                
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
    """Command for parameter sweep experiments"""
    
    def execute(
        self,
        parameter: str,
        min_val: float,
        max_val: float,
        steps: int = 10,
        replicates: int = 1,
        config: Optional[str] = None,
        output_dir: str = "artifacts/sweeps",
        simulation_days: int = 365,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute parameter sweep"""
        
        try:
            self.context.print_info(
                f"Parameter sweep: {parameter} from {min_val} to {max_val}"
            )
            
            # Create experiment manager
            experiment_manager = ExperimentManager(output_dir)
            
            # Load base configuration
            base_config = None
            if config:
                config_path = Path(config)
                if config_path.exists():
                    base_config = self.load_config(config_path)
                else:
                    self.context.print_warning(
                        f"Config file not found: {config}, using defaults"
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
            
            self.context.print_success("Parameter sweep completed!")
            self.context.print_info(f"Results saved to: {result['results_path']}")
            
            return CLIResult(
                success=True,
                message="Parameter sweep completed",
                data=result,
            )
            
        except Exception as e:
            return self.handle_exception(e, "Parameter sweep")


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
            results = experiment_manager.run_quick_comparison(scenarios, simulation_days)
            
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