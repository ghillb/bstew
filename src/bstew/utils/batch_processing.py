"""
Batch processing and experiment management for BSTEW
===================================================

Handles parameter sweeps, sensitivity analysis, Monte Carlo simulations,
and experimental design for systematic bee colony modeling research.
"""

import numpy as np
import pandas as pd
import yaml
import json
import itertools
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, computed_field, field_validator
from enum import Enum
import logging
from datetime import datetime
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import random

from ..core.model import BeeModel
from ..utils.config import ConfigManager


class ExperimentType(Enum):
    """Types of experimental designs"""

    PARAMETER_SWEEP = "parameter_sweep"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    MONTE_CARLO = "monte_carlo"
    FACTORIAL_DESIGN = "factorial_design"
    LATIN_HYPERCUBE = "latin_hypercube"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"


class ExperimentStatus(Enum):
    """Experiment execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ParameterSpec(BaseModel):
    """Parameter specification for experiments"""

    model_config = {"validate_assignment": True}

    name: str = Field(description="Parameter name")
    min_value: float = Field(description="Minimum parameter value")
    max_value: float = Field(description="Maximum parameter value")
    step_size: Optional[float] = Field(
        default=None, gt=0.0, description="Step size for parameter sweep"
    )
    values: Optional[List[Any]] = Field(
        default=None, description="Explicit list of parameter values"
    )
    distribution: str = Field(
        default="uniform", description="Distribution type: uniform, normal, log_uniform"
    )

    @field_validator("distribution")
    @classmethod
    def validate_distribution(cls, v: str) -> str:
        valid_distributions = ["uniform", "normal", "log_uniform"]
        if v not in valid_distributions:
            raise ValueError(f"Distribution must be one of {valid_distributions}")
        return v

    @field_validator("max_value")
    @classmethod
    def validate_max_greater_than_min(cls, v: float, info: Any) -> float:
        if "min_value" in info.data and v <= info.data["min_value"]:
            raise ValueError("max_value must be greater than min_value")
        return v

    def generate_values(self, n_samples: Optional[int] = None) -> List[Any]:
        """Generate parameter values based on specification"""
        if self.values:
            return self.values

        if n_samples is None:
            if self.step_size:
                n_samples = int((self.max_value - self.min_value) / self.step_size) + 1
            else:
                n_samples = 10

        if self.distribution == "uniform":
            values = np.linspace(self.min_value, self.max_value, n_samples).tolist()
            return [float(v) for v in values]
        elif self.distribution == "normal":
            mean = (self.min_value + self.max_value) / 2
            std = (self.max_value - self.min_value) / 6  # 3-sigma range
            values = np.random.normal(mean, std, n_samples)
            clipped_values = np.clip(values, self.min_value, self.max_value).tolist()
            return [float(v) for v in clipped_values]
        elif self.distribution == "log_uniform":
            log_min = np.log10(max(self.min_value, 1e-10))
            log_max = np.log10(self.max_value)
            log_values = np.linspace(log_min, log_max, n_samples)
            power_values = (10**log_values).tolist()
            return [float(v) for v in power_values]
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class ExperimentRun(BaseModel):
    """Individual experiment run specification"""

    model_config = {"validate_assignment": True}

    run_id: str = Field(description="Unique identifier for this run")
    parameters: Dict[str, Any] = Field(description="Parameter values for this run")
    random_seed: int = Field(ge=0, description="Random seed for reproducibility")
    status: ExperimentStatus = Field(
        default=ExperimentStatus.PENDING, description="Current run status"
    )
    start_time: Optional[datetime] = Field(
        default=None, description="Run start timestamp"
    )
    end_time: Optional[datetime] = Field(default=None, description="Run end timestamp")
    results: Optional[Dict[str, Any]] = Field(
        default=None, description="Run results data"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if run failed"
    )

    @computed_field
    def duration(self) -> Optional[float]:
        """Calculate run duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ExperimentDesign(BaseModel):
    """Complete experiment design specification"""

    model_config = {"validate_assignment": True}

    experiment_id: str = Field(description="Unique experiment identifier")
    experiment_type: ExperimentType = Field(description="Type of experiment")
    name: str = Field(description="Human-readable experiment name")
    description: str = Field(description="Detailed experiment description")
    parameters: Dict[str, ParameterSpec] = Field(description="Parameter specifications")
    base_config: Dict[str, Any] = Field(description="Base configuration for all runs")
    n_replicates: int = Field(
        default=1, ge=1, description="Number of replicates per parameter combination"
    )
    n_processes: int = Field(
        default=1, ge=1, description="Number of parallel processes"
    )
    simulation_days: int = Field(
        default=365, ge=1, le=3650, description="Simulation duration in days"
    )
    output_metrics: List[str] = Field(
        default_factory=list, description="Metrics to collect"
    )
    save_agent_data: bool = Field(
        default=False, description="Whether to save agent-level data"
    )
    save_spatial_data: bool = Field(
        default=False, description="Whether to save spatial data"
    )

    def generate_runs(self) -> List[ExperimentRun]:
        """Generate all experiment runs based on design"""
        runs = []

        if self.experiment_type == ExperimentType.PARAMETER_SWEEP:
            runs = self._generate_parameter_sweep()
        elif self.experiment_type == ExperimentType.FACTORIAL_DESIGN:
            runs = self._generate_factorial_design()
        elif self.experiment_type == ExperimentType.MONTE_CARLO:
            runs = self._generate_monte_carlo()
        elif self.experiment_type == ExperimentType.LATIN_HYPERCUBE:
            runs = self._generate_latin_hypercube()
        elif self.experiment_type == ExperimentType.SENSITIVITY_ANALYSIS:
            runs = self._generate_sensitivity_analysis()
        else:
            raise ValueError(f"Unsupported experiment type: {self.experiment_type}")

        return runs

    def _generate_parameter_sweep(self) -> List[ExperimentRun]:
        """Generate parameter sweep runs"""
        runs = []

        # Single parameter sweeps
        for param_name, param_spec in self.parameters.items():
            values = param_spec.generate_values()

            for value in values:
                for replicate in range(self.n_replicates):
                    params = self.base_config.copy()
                    params[param_name] = value

                    run = ExperimentRun(
                        run_id=f"{self.experiment_id}_{param_name}_{value}_rep{replicate}",
                        parameters=params,
                        random_seed=random.randint(1, 1000000),
                    )
                    runs.append(run)

        return runs

    def _generate_factorial_design(self) -> List[ExperimentRun]:
        """Generate full factorial design"""
        runs = []

        # Generate all parameter combinations
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[name].generate_values() for name in param_names]

        for combination in itertools.product(*param_values):
            for replicate in range(self.n_replicates):
                params = self.base_config.copy()

                for i, param_name in enumerate(param_names):
                    params[param_name] = combination[i]

                combination_str = "_".join(
                    f"{name}_{val}" for name, val in zip(param_names, combination)
                )
                run = ExperimentRun(
                    run_id=f"{self.experiment_id}_{combination_str}_rep{replicate}",
                    parameters=params,
                    random_seed=random.randint(1, 1000000),
                )
                runs.append(run)

        return runs

    def _generate_monte_carlo(self) -> List[ExperimentRun]:
        """Generate Monte Carlo sampling runs"""
        runs = []

        for replicate in range(self.n_replicates):
            params = self.base_config.copy()

            for param_name, param_spec in self.parameters.items():
                # Sample from parameter distribution
                if param_spec.distribution == "uniform":
                    value = random.uniform(param_spec.min_value, param_spec.max_value)
                elif param_spec.distribution == "normal":
                    mean = (param_spec.min_value + param_spec.max_value) / 2
                    std = (param_spec.max_value - param_spec.min_value) / 6
                    value = np.clip(
                        random.gauss(mean, std),
                        param_spec.min_value,
                        param_spec.max_value,
                    )
                elif param_spec.distribution == "log_uniform":
                    log_min = np.log10(max(param_spec.min_value, 1e-10))
                    log_max = np.log10(param_spec.max_value)
                    log_value = random.uniform(log_min, log_max)
                    value = 10**log_value
                else:
                    value = random.uniform(param_spec.min_value, param_spec.max_value)

                params[param_name] = value

            run = ExperimentRun(
                run_id=f"{self.experiment_id}_mc_rep{replicate}",
                parameters=params,
                random_seed=random.randint(1, 1000000),
            )
            runs.append(run)

        return runs

    def _generate_latin_hypercube(self) -> List[ExperimentRun]:
        """Generate Latin Hypercube sampling"""
        runs = []
        n_params = len(self.parameters)

        # Generate Latin Hypercube samples
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=n_params, scramble=False)
        # Set seed for reproducibility
        np.random.seed(42)
        samples = sampler.random(n=self.n_replicates)

        param_names = list(self.parameters.keys())

        for i, sample in enumerate(samples):
            params = self.base_config.copy()

            for j, param_name in enumerate(param_names):
                param_spec = self.parameters[param_name]
                # Transform uniform sample to parameter range
                value = param_spec.min_value + sample[j] * (
                    param_spec.max_value - param_spec.min_value
                )
                params[param_name] = value

            run = ExperimentRun(
                run_id=f"{self.experiment_id}_lhs_{i}",
                parameters=params,
                random_seed=random.randint(1, 1000000),
            )
            runs.append(run)

        return runs

    def _generate_sensitivity_analysis(self) -> List[ExperimentRun]:
        """Generate Sobol sensitivity analysis runs"""
        runs = []
        n_params = len(self.parameters)

        # Generate Sobol samples
        try:
            from SALib.sample import sobol

            problem = {
                "num_vars": n_params,
                "names": list(self.parameters.keys()),
                "bounds": [
                    [spec.min_value, spec.max_value]
                    for spec in self.parameters.values()
                ],
            }

            samples = sobol.sample(problem, self.n_replicates, calc_second_order=True)

            for i, sample in enumerate(samples):
                params = self.base_config.copy()

                names_data = problem.get("names", [])
                param_names = (
                    list(names_data) if isinstance(names_data, (list, tuple)) else []
                )
                for j, param_name in enumerate(param_names):
                    params[param_name] = sample[j]

                run = ExperimentRun(
                    run_id=f"{self.experiment_id}_sobol_{i}",
                    parameters=params,
                    random_seed=random.randint(1, 1000000),
                )
                runs.append(run)

        except ImportError:
            # Fallback to simple parameter perturbation
            logging.warning("SALib not available, using simple sensitivity analysis")
            runs = self._generate_simple_sensitivity()

        return runs

    def _generate_simple_sensitivity(self) -> List[ExperimentRun]:
        """Simple sensitivity analysis without SALib"""
        runs = []

        # Base case
        base_run = ExperimentRun(
            run_id=f"{self.experiment_id}_base",
            parameters=self.base_config.copy(),
            random_seed=random.randint(1, 1000000),
        )
        runs.append(base_run)

        # Perturbation around base case
        for param_name, param_spec in self.parameters.items():
            for factor in [0.8, 1.2]:  # ±20% perturbation
                params = self.base_config.copy()
                base_value = params.get(
                    param_name, (param_spec.min_value + param_spec.max_value) / 2
                )
                perturbed_value = base_value * factor
                perturbed_value = np.clip(
                    perturbed_value, param_spec.min_value, param_spec.max_value
                )
                params[param_name] = perturbed_value

                run = ExperimentRun(
                    run_id=f"{self.experiment_id}_{param_name}_pert_{factor}",
                    parameters=params,
                    random_seed=random.randint(1, 1000000),
                )
                runs.append(run)

        return runs


class BatchProcessor:
    """
    Batch processing engine for running multiple simulations.

    Handles:
    - Parallel execution
    - Progress monitoring
    - Result collection
    - Error handling
    - Checkpoint/resume functionality
    """

    def __init__(
        self,
        output_dir: str = "artifacts/experiments",
        max_workers: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers or mp.cpu_count() // 2
        self.logger = logging.getLogger(__name__)

    def run_experiment(
        self, design: ExperimentDesign, resume: bool = False
    ) -> Dict[str, Any]:
        """Run complete experiment with progress tracking"""

        experiment_dir = self.output_dir / design.experiment_id
        experiment_dir.mkdir(exist_ok=True)

        # Save experiment design
        design_path = experiment_dir / "design.yaml"
        with open(design_path, "w") as f:
            yaml.dump(design.model_dump(), f, default_flow_style=False)

        # Generate runs
        runs = design.generate_runs()

        # Resume from checkpoint if requested
        if resume:
            runs = self._load_checkpoint(experiment_dir, runs)

        self.logger.info(
            f"Starting experiment {design.experiment_id} with {len(runs)} runs"
        )

        # Execute runs
        results = self._execute_runs(runs, design, experiment_dir)

        # Analyze results
        analysis = self._analyze_experiment_results(results, design)

        # Save final results
        self._save_experiment_results(results, analysis, experiment_dir)

        return {
            "experiment_id": design.experiment_id,
            "total_runs": len(runs),
            "successful_runs": len(
                [r for r in results if r.status == ExperimentStatus.COMPLETED]
            ),
            "failed_runs": len(
                [r for r in results if r.status == ExperimentStatus.FAILED]
            ),
            "total_duration": sum(
                r.duration for r in results if r.duration is not None
            ),
            "results_path": str(experiment_dir),
            "analysis": analysis,
        }

    def _execute_runs(
        self, runs: List[ExperimentRun], design: ExperimentDesign, experiment_dir: Path
    ) -> List[ExperimentRun]:
        """Execute simulation runs in parallel"""

        completed_runs: List[ExperimentRun] = []

        # Create checkpoint function
        def save_checkpoint() -> None:
            checkpoint_path = experiment_dir / "checkpoint.pkl"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(completed_runs, f)

        try:
            if design.n_processes == 1:
                # Serial execution
                for i, run in enumerate(runs):
                    self.logger.info(f"Running {run.run_id} ({i + 1}/{len(runs)})")
                    completed_run = self._execute_single_run(run, design)
                    completed_runs.append(completed_run)

                    # Save checkpoint every 10 runs
                    if (i + 1) % 10 == 0:
                        save_checkpoint()

            else:
                # Parallel execution
                with ProcessPoolExecutor(max_workers=design.n_processes) as executor:
                    # Submit all runs
                    future_to_run = {
                        executor.submit(
                            self._execute_single_run_wrapper, run, design
                        ): run
                        for run in runs
                    }

                    # Collect results as they complete
                    for i, future in enumerate(as_completed(future_to_run)):
                        run = future_to_run[future]

                        try:
                            completed_run = future.result(
                                timeout=3600
                            )  # 1 hour timeout
                            completed_runs.append(completed_run)
                            self.logger.info(
                                f"Completed {run.run_id} ({i + 1}/{len(runs)})"
                            )

                        except Exception as e:
                            self.logger.error(f"Run {run.run_id} failed: {e}")
                            run.status = ExperimentStatus.FAILED
                            run.error_message = str(e)
                            run.end_time = datetime.now()
                            completed_runs.append(run)

                        # Save checkpoint every 10 runs
                        if (i + 1) % 10 == 0:
                            save_checkpoint()

        except KeyboardInterrupt:
            self.logger.info("Experiment interrupted by user")

        finally:
            save_checkpoint()

        return completed_runs

    def _execute_single_run_wrapper(
        self, run: ExperimentRun, design: ExperimentDesign
    ) -> ExperimentRun:
        """Wrapper for parallel execution"""
        return self._execute_single_run(run, design)

    def _execute_single_run(
        self, run: ExperimentRun, design: ExperimentDesign
    ) -> ExperimentRun:
        """Execute a single simulation run"""
        run.start_time = datetime.now()
        run.status = ExperimentStatus.RUNNING

        try:
            # Create model with run parameters
            config = ConfigManager()
            config_dict = config.load_default_config().model_dump()
            config_dict.update(run.parameters)

            model = BeeModel(config=config_dict, random_seed=run.random_seed)

            # Run simulation
            for day in range(design.simulation_days):
                model.step()

                # Check for early termination conditions
                if model.get_colony_count() == 0:
                    break

            # Collect results
            model_data = model.datacollector.get_model_vars_dataframe()

            results = {
                "model_data": model_data.to_dict(),
                "final_metrics": self._extract_final_metrics(
                    model_data, design.output_metrics
                ),
                "simulation_days": len(model_data),
                "final_colony_count": model.get_colony_count(),
                "total_bees": (
                    model_data["Total_Bees"].iloc[-1] if len(model_data) > 0 else 0
                ),
            }

            # Save agent data if requested
            if design.save_agent_data:
                agent_data = model.datacollector.get_agent_vars_dataframe()
                results["agent_data"] = agent_data.to_dict()

            # Save spatial data if requested
            if design.save_spatial_data:
                spatial_data = model.landscape.export_to_dict()
                results["spatial_data"] = spatial_data

            run.results = results
            run.status = ExperimentStatus.COMPLETED

        except Exception as e:
            self.logger.error(f"Error in run {run.run_id}: {e}")
            run.status = ExperimentStatus.FAILED
            run.error_message = str(e)

        finally:
            run.end_time = datetime.now()

        return run

    def _extract_final_metrics(
        self, model_data: pd.DataFrame, output_metrics: List[str]
    ) -> Dict[str, Any]:
        """Extract final metrics from model data"""
        if len(model_data) == 0:
            return {}

        metrics = {}

        # Standard metrics
        for col in ["Total_Bees", "Total_Brood", "Total_Honey", "Active_Colonies"]:
            if col in model_data.columns:
                metrics[f"final_{col.lower()}"] = model_data[col].iloc[-1]
                metrics[f"max_{col.lower()}"] = model_data[col].max()
                metrics[f"mean_{col.lower()}"] = model_data[col].mean()

        # Custom metrics
        for metric in output_metrics:
            if metric in model_data.columns:
                metrics[f"final_{metric}"] = model_data[metric].iloc[-1]

        return metrics

    def _analyze_experiment_results(
        self, runs: List[ExperimentRun], design: ExperimentDesign
    ) -> Dict[str, Any]:
        """Analyze experiment results"""

        successful_runs = [r for r in runs if r.status == ExperimentStatus.COMPLETED]

        if not successful_runs:
            return {"error": "No successful runs to analyze"}

        # Create results DataFrame
        results_data = []
        for run in successful_runs:
            row = {"run_id": run.run_id}
            row.update(run.parameters)
            if run.results and "final_metrics" in run.results:
                row.update(run.results["final_metrics"])
            results_data.append(row)

        results_df = pd.DataFrame(results_data)

        # Basic statistics
        analysis: Dict[str, Any] = {
            "summary_statistics": results_df.describe().to_dict(),
            "parameter_correlations": {},
            "sensitivity_indices": {},
        }

        # Parameter correlations
        param_names = list(design.parameters.keys())
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns

        if len(param_names) > 1 and len(numeric_cols) > 0:
            correlation_matrix = results_df[numeric_cols].corr()
            analysis["parameter_correlations"] = correlation_matrix.to_dict()

        # Sensitivity analysis for specific experiment types
        if design.experiment_type == ExperimentType.SENSITIVITY_ANALYSIS:
            analysis["sensitivity_indices"] = self._calculate_sensitivity_indices(
                results_df, param_names, design
            )

        return analysis

    def _calculate_sensitivity_indices(
        self, results_df: pd.DataFrame, param_names: List[str], design: ExperimentDesign
    ) -> Dict[str, Any]:
        """Calculate sensitivity indices"""

        sensitivity_indices = {}

        # Simple sensitivity calculation (correlation-based)
        output_cols = [
            col
            for col in results_df.columns
            if col.startswith("final_") and col not in param_names
        ]

        for output_col in output_cols:
            if output_col in results_df.columns:
                indices = {}

                for param in param_names:
                    if param in results_df.columns:
                        correlation = results_df[param].corr(results_df[output_col])
                        indices[param] = (
                            abs(correlation) if not np.isnan(correlation) else 0.0
                        )

                sensitivity_indices[output_col] = indices

        return sensitivity_indices

    def _save_experiment_results(
        self, runs: List[ExperimentRun], analysis: Dict[str, Any], experiment_dir: Path
    ) -> None:
        """Save experiment results to files"""

        # Save run results
        results_data = []
        for run in runs:
            row = {
                "run_id": run.run_id,
                "status": run.status.value,
                "duration": run.duration,
                "error_message": run.error_message,
            }
            row.update(run.parameters)

            if run.results and "final_metrics" in run.results:
                row.update(run.results["final_metrics"])

            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(experiment_dir / "results.csv", index=False)

        # Save analysis
        with open(experiment_dir / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Save individual run data
        data_dir = experiment_dir / "run_data"
        data_dir.mkdir(exist_ok=True)

        for run in runs:
            if run.results and run.status == ExperimentStatus.COMPLETED:
                run_file = data_dir / f"{run.run_id}.json"
                with open(run_file, "w") as f:
                    json.dump(run.results, f, indent=2, default=str)

    def _load_checkpoint(
        self, experiment_dir: Path, runs: List[ExperimentRun]
    ) -> List[ExperimentRun]:
        """Load checkpoint and return remaining runs"""

        checkpoint_path = experiment_dir / "checkpoint.pkl"

        if not checkpoint_path.exists():
            return runs

        try:
            with open(checkpoint_path, "rb") as f:
                completed_runs = pickle.load(f)

            completed_run_ids = {run.run_id for run in completed_runs}
            remaining_runs = [
                run for run in runs if run.run_id not in completed_run_ids
            ]

            self.logger.info(
                f"Resuming experiment: {len(completed_runs)} completed, "
                f"{len(remaining_runs)} remaining"
            )

            return remaining_runs

        except Exception as e:
            self.logger.warning(f"Could not load checkpoint: {e}")
            return runs


class ExperimentManager:
    """
    High-level experiment management interface.

    Provides convenience methods for common experimental designs.
    """

    def __init__(self, output_dir: str = "artifacts/experiments"):
        self.output_dir = Path(output_dir)
        self.batch_processor = BatchProcessor(output_dir)
        self.config_manager = ConfigManager()
        self.logger = logging.getLogger(__name__)

    def create_parameter_sweep(
        self,
        experiment_name: str,
        parameter_specs: Dict[str, ParameterSpec],
        base_config: Optional[Dict[str, Any]] = None,
        n_replicates: int = 1,
        simulation_days: int = 365,
    ) -> ExperimentDesign:
        """Create parameter sweep experiment"""

        if base_config is None:
            base_config = self.config_manager.load_default_config().to_dict()

        design = ExperimentDesign(
            experiment_id=f"param_sweep_{int(time.time())}",
            experiment_type=ExperimentType.PARAMETER_SWEEP,
            name=experiment_name,
            description=f"Parameter sweep: {', '.join(parameter_specs.keys())}",
            parameters=parameter_specs,
            base_config=base_config,
            n_replicates=n_replicates,
            simulation_days=simulation_days,
        )

        return design

    def create_monte_carlo_experiment(
        self,
        experiment_name: str,
        parameter_specs: Dict[str, ParameterSpec],
        n_samples: int = 100,
        base_config: Optional[Dict[str, Any]] = None,
        simulation_days: int = 365,
    ) -> ExperimentDesign:
        """Create Monte Carlo experiment"""

        if base_config is None:
            base_config = self.config_manager.load_default_config().to_dict()

        design = ExperimentDesign(
            experiment_id=f"monte_carlo_{int(time.time())}",
            experiment_type=ExperimentType.MONTE_CARLO,
            name=experiment_name,
            description=f"Monte Carlo sampling with {n_samples} samples",
            parameters=parameter_specs,
            base_config=base_config,
            n_replicates=n_samples,
            simulation_days=simulation_days,
        )

        return design

    def create_sensitivity_analysis(
        self,
        experiment_name: str,
        parameter_specs: Dict[str, ParameterSpec],
        n_samples: int = 1000,
        base_config: Optional[Dict[str, Any]] = None,
        simulation_days: int = 365,
    ) -> ExperimentDesign:
        """Create sensitivity analysis experiment"""

        if base_config is None:
            base_config = self.config_manager.load_default_config().to_dict()

        design = ExperimentDesign(
            experiment_id=f"sensitivity_{int(time.time())}",
            experiment_type=ExperimentType.SENSITIVITY_ANALYSIS,
            name=experiment_name,
            description=f"Sensitivity analysis for {', '.join(parameter_specs.keys())}",
            parameters=parameter_specs,
            base_config=base_config,
            n_replicates=n_samples,
            simulation_days=simulation_days,
        )

        return design

    def run_quick_comparison(
        self, scenarios: Dict[str, Dict[str, Any]], simulation_days: int = 365
    ) -> Dict[str, Any]:
        """Run quick scenario comparison"""

        results = {}

        for scenario_name, config in scenarios.items():
            self.logger.info(f"Running scenario: {scenario_name}")

            model = BeeModel(config=config)

            for day in range(simulation_days):
                model.step()

            model_data = model.datacollector.get_model_vars_dataframe()

            results[scenario_name] = {
                "final_population": (
                    model_data["Total_Bees"].iloc[-1] if len(model_data) > 0 else 0
                ),
                "max_population": (
                    model_data["Total_Bees"].max() if len(model_data) > 0 else 0
                ),
                "final_honey": (
                    model_data["Total_Honey"].iloc[-1] if len(model_data) > 0 else 0
                ),
                "colony_survival": model.get_colony_count() > 0,
                "simulation_days": len(model_data),
            }

        return results
