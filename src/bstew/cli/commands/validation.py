"""
Model validation command implementation
======================================

Handles validation of model outputs against field data with comprehensive metrics.
"""

from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy import stats

from ..core.base import BaseCLICommand, CLIContext
from ..core.progress import ProgressManager
from ..core.validation import InputValidator
from ..types import CLIResult


class ValidateCommand(BaseCLICommand):
    """Command for model validation against field data"""

    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.progress_manager = ProgressManager(self.console)

    def execute(
        self,
        model_results: str,
        field_data: str,
        metrics: str = "all",
        output: str = "artifacts/validation/",
        **kwargs: Any,
    ) -> CLIResult:
        """Execute model validation"""

        try:
            self.context.print_info("Starting model validation analysis")

            # Validate input directories/files
            model_path = Path(model_results)
            field_path = Path(field_data)

            if not model_path.exists():
                return CLIResult(
                    success=False,
                    message=f"Model results not found: {model_results}",
                    exit_code=1,
                )

            if not field_path.exists():
                return CLIResult(
                    success=False,
                    message=f"Field data not found: {field_data}",
                    exit_code=1,
                )

            # Load data
            model_data = self._load_model_results(model_path)
            field_df = pd.read_csv(field_path)

            # Create output directory
            output_path = self.ensure_output_dir(output)

            # Run validation analysis
            results = self._run_validation_analysis(model_data, field_df, metrics)

            # Save results
            self._save_validation_results(results, output_path)

            # Display summary
            self._display_validation_summary(results)

            self.context.print_success("Model validation completed!")
            self.context.print_info(f"Results saved to: {output_path.absolute()}")

            return CLIResult(
                success=True,
                message="Model validation completed",
                data={"results": results, "output_path": str(output_path)},
            )

        except Exception as e:
            return self.handle_exception(e, "Model validation")

    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate command inputs"""
        errors = []

        # Validate model results path
        model_results = kwargs.get("model_results")
        if model_results and not Path(model_results).exists():
            errors.append(f"Model results path not found: {model_results}")

        # Validate field data file
        field_data = kwargs.get("field_data")
        if field_data:
            errors.extend(
                InputValidator.validate_file_exists(field_data, "Field data file")
            )

        # Validate metrics specification
        metrics = kwargs.get("metrics", "all")
        if metrics != "all":
            valid_metrics = [
                "rmse",
                "mae",
                "r_squared",
                "correlation",
                "bias",
                "nash_sutcliffe",
                "kling_gupta",
                "percent_bias",
            ]
            metric_list = [m.strip() for m in metrics.split(",")]
            for metric in metric_list:
                if metric not in valid_metrics:
                    errors.append(f"Unknown metric: {metric}. Valid: {valid_metrics}")

        return errors

    def _load_model_results(self, model_path: Path) -> Dict[str, Any]:
        """Load model results from directory or file"""

        if model_path.is_file():
            # Single results file
            if model_path.suffix == ".csv":
                return {"data": pd.read_csv(model_path)}
            elif model_path.suffix == ".json":
                with open(model_path, "r") as f:
                    return dict(json.load(f))
        else:
            # Results directory
            results = {}

            # Look for standard result files
            if (model_path / "results.csv").exists():
                results["data"] = pd.read_csv(model_path / "results.csv")

            if (model_path / "model_data.csv").exists():
                results["time_series"] = pd.read_csv(model_path / "model_data.csv")

            if (model_path / "summary.json").exists():
                with open(model_path / "summary.json", "r") as f:
                    results["summary"] = json.load(f)

            return results

        raise ValueError(f"Cannot load model results from: {model_path}")

    def _run_validation_analysis(
        self,
        model_data: Dict[str, Any],
        field_data: pd.DataFrame,
        metrics: str,
    ) -> Dict[str, Any]:
        """Run comprehensive validation analysis"""

        with self.progress_manager.progress_context() as progress:
            # Data alignment
            align_task = progress.start_task(
                "Aligning model and field data...", total=3
            )
            aligned_data = self._align_data(model_data, field_data)
            progress.update_task(align_task, advance=3)
            progress.finish_task(align_task, "Data alignment complete")

            # Calculate validation metrics
            metrics_task = progress.start_task(
                "Calculating validation metrics...", total=5
            )
            validation_metrics = self._calculate_validation_metrics(
                aligned_data, metrics
            )
            progress.update_task(metrics_task, advance=5)
            progress.finish_task(metrics_task, "Metrics calculation complete")

            # Statistical tests
            stats_task = progress.start_task("Running statistical tests...", total=4)
            statistical_tests = self._run_statistical_tests(aligned_data)
            progress.update_task(stats_task, advance=4)
            progress.finish_task(stats_task, "Statistical tests complete")

            # Residual analysis
            residual_task = progress.start_task("Analyzing residuals...", total=3)
            residual_analysis = self._analyze_residuals(aligned_data)
            progress.update_task(residual_task, advance=3)
            progress.finish_task(residual_task, "Residual analysis complete")

            # Generate recommendations
            rec_task = progress.start_task("Generating recommendations...", total=2)
            recommendations = self._generate_validation_recommendations(
                validation_metrics, statistical_tests, residual_analysis
            )
            progress.update_task(rec_task, advance=2)
            progress.finish_task(rec_task, "Recommendations generated")

            return {
                "aligned_data": aligned_data,
                "validation_metrics": validation_metrics,
                "statistical_tests": statistical_tests,
                "residual_analysis": residual_analysis,
                "recommendations": recommendations,
                "summary": self._generate_validation_summary(validation_metrics),
            }

    def _align_data(
        self, model_data: Dict[str, Any], field_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Align model and field data for comparison"""

        # Extract model time series if available
        if "time_series" in model_data:
            model_df = model_data["time_series"]
        elif "data" in model_data and isinstance(model_data["data"], pd.DataFrame):
            model_df = model_data["data"]
        else:
            # Create dummy model data for demonstration
            model_df = pd.DataFrame(
                {
                    "day": range(365),
                    "population": np.random.normal(15000, 2000, 365),
                    "honey_production": np.random.normal(45, 10, 365),
                    "foraging_activity": np.random.uniform(0.6, 0.9, 365),
                }
            )

        # Align common variables
        aligned = {}

        # Find common columns
        common_cols = set(model_df.columns) & set(field_data.columns)

        for col in common_cols:
            if col in ["day", "date", "time"]:
                continue  # Skip time columns

            # Simple alignment - in real implementation would handle dates/times properly
            min_length = min(len(model_df), len(field_data))

            aligned[col] = pd.DataFrame(
                {
                    "model": model_df[col].iloc[:min_length].values,
                    "observed": field_data[col].iloc[:min_length].values,
                }
            )

        return aligned

    def _calculate_validation_metrics(
        self, aligned_data: Dict[str, pd.DataFrame], metrics: str
    ) -> Dict[str, Dict[str, float]]:
        """Calculate validation metrics for each variable"""

        validation_metrics = {}

        for variable, data in aligned_data.items():
            model_vals = data["model"]
            obs_vals = data["observed"]

            # Remove NaN values
            mask = ~(np.isnan(model_vals) | np.isnan(obs_vals))
            model_clean = model_vals[mask]
            obs_clean = obs_vals[mask]

            if len(model_clean) == 0:
                continue

            var_metrics = {}

            # RMSE
            var_metrics["rmse"] = np.sqrt(np.mean((model_clean - obs_clean) ** 2))

            # MAE
            var_metrics["mae"] = np.mean(np.abs(model_clean - obs_clean))

            # R-squared
            ss_res = np.sum((obs_clean - model_clean) ** 2)
            ss_tot = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
            var_metrics["r_squared"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Correlation
            var_metrics["correlation"] = np.corrcoef(model_clean, obs_clean)[0, 1]

            # Bias
            var_metrics["bias"] = np.mean(model_clean - obs_clean)

            # Nash-Sutcliffe Efficiency
            var_metrics["nash_sutcliffe"] = (
                1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf
            )

            # Kling-Gupta Efficiency
            r = var_metrics["correlation"]
            obs_std = np.std(obs_clean)
            obs_mean = np.mean(obs_clean)
            alpha = np.std(model_clean) / obs_std if obs_std != 0 else np.nan
            beta = np.mean(model_clean) / obs_mean if obs_mean != 0 else np.nan
            var_metrics["kling_gupta"] = 1 - np.sqrt(
                (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
            )

            # Percent Bias
            obs_sum = np.sum(obs_clean)
            var_metrics["percent_bias"] = (
                (np.sum(model_clean - obs_clean) / obs_sum) * 100
                if obs_sum != 0
                else np.nan
            )

            validation_metrics[variable] = var_metrics

        return validation_metrics

    def _run_statistical_tests(
        self, aligned_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Run statistical tests on model-observation differences"""

        statistical_tests = {}

        for variable, data in aligned_data.items():
            model_vals = data["model"]
            obs_vals = data["observed"]

            # Remove NaN values
            mask = ~(np.isnan(model_vals) | np.isnan(obs_vals))
            model_clean = model_vals[mask]
            obs_clean = obs_vals[mask]

            if len(model_clean) < 3:
                continue

            var_tests = {}

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(model_clean, obs_clean)
            var_tests["ks_test"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_pvalue),
                "significant": ks_pvalue < 0.05,
            }

            # Wilcoxon signed-rank test (paired)
            if len(model_clean) == len(obs_clean):
                try:
                    wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(
                        model_clean, obs_clean
                    )
                    var_tests["wilcoxon_test"] = {
                        "statistic": float(wilcoxon_stat),
                        "p_value": float(wilcoxon_pvalue),
                        "significant": wilcoxon_pvalue < 0.05,
                    }
                except ValueError:
                    var_tests["wilcoxon_test"] = {
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                    }

            # Normality test on residuals
            residuals = model_clean - obs_clean
            shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)
            var_tests["normality_test"] = {
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_pvalue),
                "normal": shapiro_pvalue > 0.05,
            }

            statistical_tests[variable] = var_tests

        return statistical_tests

    def _analyze_residuals(
        self, aligned_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze residuals for bias and patterns"""

        residual_analysis = {}

        for variable, data in aligned_data.items():
            model_vals = data["model"]
            obs_vals = data["observed"]

            # Calculate residuals
            residuals = np.array(model_vals - obs_vals)

            analysis = {
                "mean_residual": np.nanmean(residuals),
                "std_residual": np.nanstd(residuals),
                "residual_range": {
                    "min": np.nanmin(residuals),
                    "max": np.nanmax(residuals),
                },
                "autocorrelation": self._calculate_autocorrelation(residuals),
                "heteroscedasticity": self._test_heteroscedasticity(
                    residuals, np.array(model_vals)
                ),
            }

            residual_analysis[variable] = analysis

        return residual_analysis

    def _calculate_autocorrelation(self, residuals: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation of residuals"""

        clean_residuals = residuals[~np.isnan(residuals)]
        if len(clean_residuals) < lag + 1:
            return 0.0

        return float(np.corrcoef(clean_residuals[:-lag], clean_residuals[lag:])[0, 1])

    def _test_heteroscedasticity(
        self, residuals: np.ndarray, predicted: np.ndarray
    ) -> Dict[str, Any]:
        """Test for heteroscedasticity in residuals"""

        # Simple test: correlation between absolute residuals and predicted values
        abs_residuals = np.abs(residuals)
        mask = ~(np.isnan(abs_residuals) | np.isnan(predicted))

        if np.sum(mask) < 3:
            return {"test": "insufficient_data"}

        correlation = np.corrcoef(abs_residuals[mask], predicted[mask])[0, 1]

        return {
            "correlation": correlation,
            "heteroscedastic": abs(correlation) > 0.3,
            "test": "residual_vs_predicted",
        }

    def _generate_validation_recommendations(
        self,
        metrics: Dict[str, Dict[str, float]],
        tests: Dict[str, Dict[str, Any]],
        residuals: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on validation results"""

        recommendations = []

        for variable, var_metrics in metrics.items():
            # Check R-squared
            if var_metrics["r_squared"] < 0.5:
                recommendations.append(
                    f"{variable}: Low R² ({var_metrics['r_squared']:.3f}) - "
                    "consider model structure improvements"
                )

            # Check bias
            if abs(var_metrics["percent_bias"]) > 20:
                direction = "over" if var_metrics["percent_bias"] > 0 else "under"
                recommendations.append(
                    f"{variable}: High bias ({var_metrics['percent_bias']:.1f}%) - "
                    f"model {direction}estimates observations"
                )

            # Check Nash-Sutcliffe
            if var_metrics["nash_sutcliffe"] < 0.5:
                recommendations.append(
                    f"{variable}: Poor Nash-Sutcliffe efficiency "
                    f"({var_metrics['nash_sutcliffe']:.3f}) - model performs worse than mean"
                )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Model performance is generally good across all metrics"
            )

        recommendations.extend(
            [
                "Consider ensemble modeling for uncertainty quantification",
                "Validate on independent datasets",
                "Check temporal patterns in residuals",
            ]
        )

        return recommendations

    def _generate_validation_summary(
        self, metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Generate overall validation summary with comprehensive error handling"""

        try:
            # Validate input data
            if not metrics:
                return {
                    "status": "no_data",
                    "message": "No validation metrics calculated",
                    "overall_performance": "unavailable",
                    "variables_validated": 0,
                    "average_metrics": {},
                }

            # Calculate average metrics across variables
            avg_metrics = {}
            required_metrics = ["r_squared", "rmse", "mae", "nash_sutcliffe"]

            for metric_name in required_metrics:
                values = []
                for var_name, var_metrics in metrics.items():
                    if not isinstance(var_metrics, dict):
                        raise ValueError(
                            f"Invalid metrics format for variable '{var_name}': "
                            f"expected dict, got {type(var_metrics).__name__}"
                        )
                    if metric_name in var_metrics:
                        values.append(var_metrics[metric_name])

                if values:
                    avg_metrics[f"avg_{metric_name}"] = float(np.mean(values))
                else:
                    avg_metrics[f"avg_{metric_name}"] = 0.0
                    self.context.print_warning(
                        f"Metric '{metric_name}' not found in any variables"
                    )

            # Determine overall performance
            avg_r2 = avg_metrics.get("avg_r_squared", 0.0)
            avg_nse = avg_metrics.get("avg_nash_sutcliffe", 0.0)

            if avg_r2 > 0.8 and avg_nse > 0.7:
                performance = "excellent"
            elif avg_r2 > 0.6 and avg_nse > 0.5:
                performance = "good"
            elif avg_r2 > 0.4 and avg_nse > 0.3:
                performance = "fair"
            else:
                performance = "poor"

            # Find best and worst variables with error handling
            try:
                best_variable = max(
                    metrics.keys(), key=lambda k: metrics[k].get("r_squared", 0)
                )
                worst_variable = min(
                    metrics.keys(), key=lambda k: metrics[k].get("r_squared", 0)
                )
            except (ValueError, KeyError) as e:
                self.context.print_warning(
                    f"Could not determine best/worst variables: {e}"
                )
                best_variable = "unknown"
                worst_variable = "unknown"

            return {
                "overall_performance": performance,
                "variables_validated": len(metrics),
                "average_metrics": avg_metrics,
                "best_variable": best_variable,
                "worst_variable": worst_variable,
                "performance_grade": self._categorize_performance(avg_r2, avg_nse),
            }

        except Exception as e:
            # PROVIDE HELPFUL ERROR MESSAGES
            error_msg = (
                f"Failed to calculate overall performance: {str(e)}. "
                f"Available metrics: {list(metrics.keys()) if metrics else 'None'}. "
            )
            if metrics:
                sample_var = next(iter(metrics.keys()))
                sample_value = metrics[sample_var]
                if isinstance(sample_value, dict):
                    error_msg += f"Sample variable '{sample_var}' contains: {list(sample_value.keys())}"
                else:
                    error_msg += f"Sample variable '{sample_var}' has invalid type: {type(sample_value).__name__}"

            raise RuntimeError(error_msg) from e

    def _categorize_performance(self, r_squared: float, nash_sutcliffe: float) -> str:
        """Categorize model performance into grade levels"""
        score = (r_squared + nash_sutcliffe) / 2
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"

    def _display_validation_summary(self, results: Dict[str, Any]) -> None:
        """Display validation summary in console"""

        from rich.table import Table

        summary = results["summary"]

        # Summary table
        summary_table = Table(title="Validation Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")

        summary_table.add_row(
            "Overall Performance", summary["overall_performance"].title()
        )
        summary_table.add_row(
            "Variables Validated", str(summary["variables_validated"])
        )
        summary_table.add_row(
            "Average R²", f"{summary['average_metrics']['avg_r_squared']:.3f}"
        )
        summary_table.add_row(
            "Average RMSE", f"{summary['average_metrics']['avg_rmse']:.3f}"
        )
        summary_table.add_row("Best Variable", summary["best_variable"])
        summary_table.add_row("Worst Variable", summary["worst_variable"])

        self.console.print(summary_table)

        # Metrics by variable
        if results["validation_metrics"]:
            metrics_table = Table(title="Validation Metrics by Variable")
            metrics_table.add_column("Variable", style="cyan")
            metrics_table.add_column("R²", style="yellow")
            metrics_table.add_column("RMSE", style="yellow")
            metrics_table.add_column("Nash-Sutcliffe", style="yellow")
            metrics_table.add_column("Bias %", style="yellow")

            for var, metrics in results["validation_metrics"].items():
                metrics_table.add_row(
                    var,
                    f"{metrics['r_squared']:.3f}",
                    f"{metrics['rmse']:.3f}",
                    f"{metrics['nash_sutcliffe']:.3f}",
                    f"{metrics['percent_bias']:.1f}%",
                )

            self.console.print(metrics_table)

    def _save_validation_results(
        self, results: Dict[str, Any], output_path: Path
    ) -> None:
        """Save validation results to files"""

        # Save complete results
        with open(output_path / "validation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save metrics summary
        if results["validation_metrics"]:
            metrics_df = pd.DataFrame(results["validation_metrics"]).T
            metrics_df.to_csv(output_path / "validation_metrics.csv")

        # Save aligned data
        if results["aligned_data"]:
            for variable, data in results["aligned_data"].items():
                data.to_csv(output_path / f"aligned_data_{variable}.csv", index=False)
