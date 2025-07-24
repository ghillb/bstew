"""
Validation framework for NetLogo comparison
==========================================

Implements systematic validation of BSTEW against original NetLogo model
including parameter compatibility, output comparison, and statistical tests.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, computed_field
from pathlib import Path
import logging
import json
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ValidationResult(BaseModel):
    """Result of a validation test"""

    model_config = {"validate_assignment": True}

    test_name: str = Field(description="Name of the validation test")
    passed: bool = Field(description="Whether the test passed")
    score: float = Field(
        ge=0.0, le=1.0, description="Test score (0-1 where 1 is perfect match)"
    )
    p_value: Optional[float] = Field(
        default=None, description="Statistical p-value if applicable"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional test details"
    )
    recommendation: str = Field(
        default="", description="Recommendation for improvement"
    )


class ComparisonMetrics(BaseModel):
    """Statistical comparison metrics between two datasets"""

    model_config = {"validate_assignment": True}

    rmse: float = Field(ge=0.0, description="Root mean squared error")
    mae: float = Field(ge=0.0, description="Mean absolute error")
    r2: float = Field(description="R-squared score")
    correlation: float = Field(
        ge=-1.0, le=1.0, description="Pearson correlation coefficient"
    )
    p_value: float = Field(ge=0.0, le=1.0, description="Correlation p-value")
    ks_statistic: float = Field(ge=0.0, description="Kolmogorov-Smirnov statistic")
    ks_p_value: float = Field(ge=0.0, le=1.0, description="Kolmogorov-Smirnov p-value")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def agreement_score(self) -> float:
        """Overall agreement score (0-1)"""
        return (
            self.r2
            + self.correlation
            + (1 - min(1, self.rmse))
            + (1 - min(1, self.mae))
        ) / 4


class ParameterValidator:
    """
    Validates parameter compatibility between NetLogo and BSTEW.

    Ensures parameters are correctly mapped and within valid ranges.
    """

    def __init__(self) -> None:
        self.parameter_mappings = self._load_parameter_mappings()
        self.validation_rules = self._load_validation_rules()

    def _load_parameter_mappings(self) -> Dict[str, str]:
        """Load NetLogo to BSTEW parameter mappings"""
        # This would typically load from a configuration file
        return {
            # NetLogo parameter -> BSTEW parameter path
            "initial-bees": "colony.initial_population",
            "initial-brood": "colony.initial_brood",
            "max-age": "biology.max_lifespan",
            "egg-laying-rate": "biology.egg_laying_rate",
            "foraging-range": "foraging.max_range",
            "dance-threshold": "communication.dance_threshold",
            "mortality-rate": "biology.base_mortality",
            "resource-quality": "environment.resource_quality_base",
        }

    def _load_validation_rules(self) -> Dict[str, Dict]:
        """Load parameter validation rules"""
        return {
            "colony.initial_population": {
                "min": 1000,
                "max": 100000,
                "type": "integer",
                "default": 20000,
            },
            "colony.initial_brood": {
                "min": 0,
                "max": 50000,
                "type": "integer",
                "default": 5000,
            },
            "biology.max_lifespan": {
                "min": 10,
                "max": 200,
                "type": "integer",
                "default": 45,
            },
            "biology.egg_laying_rate": {
                "min": 100,
                "max": 3000,
                "type": "integer",
                "default": 1500,
            },
            "foraging.max_range": {
                "min": 100,
                "max": 10000,
                "type": "float",
                "default": 2000.0,
            },
        }

    def validate_parameter_mapping(
        self, netlogo_params: Dict[str, Any], bstew_config: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate parameter mapping from NetLogo to BSTEW"""

        results = []

        for netlogo_param, bstew_path in self.parameter_mappings.items():
            if netlogo_param not in netlogo_params:
                results.append(
                    ValidationResult(
                        test_name=f"parameter_exists_{netlogo_param}",
                        passed=False,
                        score=0.0,
                        details={
                            "message": f"NetLogo parameter '{netlogo_param}' not found"
                        },
                        recommendation=f"Add '{netlogo_param}' to NetLogo parameter set",
                    )
                )
                continue

            # Navigate BSTEW config path
            bstew_value = self._get_nested_value(bstew_config, bstew_path)
            netlogo_value = netlogo_params[netlogo_param]

            if bstew_value is None:
                results.append(
                    ValidationResult(
                        test_name=f"parameter_mapping_{netlogo_param}",
                        passed=False,
                        score=0.0,
                        details={
                            "message": f"BSTEW parameter path '{bstew_path}' not found"
                        },
                        recommendation=f"Check BSTEW config structure for '{bstew_path}'",
                    )
                )
                continue

            # Validate parameter values
            validation_result = self._validate_parameter_value(
                netlogo_param, netlogo_value, bstew_value, bstew_path
            )
            results.append(validation_result)

        return results

    def _get_nested_value(self, config: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _validate_parameter_value(
        self, netlogo_param: str, netlogo_value: Any, bstew_value: Any, bstew_path: str
    ) -> ValidationResult:
        """Validate individual parameter value"""

        # Check if values are equal (within tolerance for floats)
        if isinstance(netlogo_value, (int, float)) and isinstance(
            bstew_value, (int, float)
        ):
            tolerance = 1e-6 if isinstance(netlogo_value, float) else 0
            values_match = abs(netlogo_value - bstew_value) <= tolerance
            score = (
                1.0
                if values_match
                else max(
                    0.0,
                    1.0 - abs(netlogo_value - bstew_value) / max(abs(netlogo_value), 1),
                )
            )
        else:
            values_match = netlogo_value == bstew_value
            score = 1.0 if values_match else 0.0

        # Check parameter rules
        rules = self.validation_rules.get(bstew_path, {})
        rule_violations = []

        if "min" in rules and bstew_value < rules["min"]:
            rule_violations.append(f"Value {bstew_value} below minimum {rules['min']}")

        if "max" in rules and bstew_value > rules["max"]:
            rule_violations.append(f"Value {bstew_value} above maximum {rules['max']}")

        if "type" in rules:
            expected_type = rules["type"]
            if expected_type == "integer" and not isinstance(bstew_value, int):
                rule_violations.append(
                    f"Expected integer, got {type(bstew_value).__name__}"
                )

        passed = values_match and not rule_violations

        recommendation = ""
        if not values_match:
            recommendation = f"Adjust BSTEW parameter '{bstew_path}' to match NetLogo value {netlogo_value}"
        if rule_violations:
            recommendation += f" Violations: {'; '.join(rule_violations)}"

        return ValidationResult(
            test_name=f"parameter_value_{netlogo_param}",
            passed=passed,
            score=score * (0.5 if rule_violations else 1.0),
            details={
                "netlogo_value": netlogo_value,
                "bstew_value": bstew_value,
                "rule_violations": rule_violations,
            },
            recommendation=recommendation.strip(),
        )


class OutputComparator:
    """
    Compares simulation outputs between NetLogo and BSTEW.

    Performs statistical tests and visual comparisons.
    """

    def __init__(self) -> None:
        self.comparison_metrics: Dict[str, Any] = {}
        self.significance_level = 0.05

    def compare_time_series(
        self,
        netlogo_data: pd.DataFrame,
        bstew_data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, ValidationResult]:
        """Compare time series outputs between models"""

        if metrics is None:
            metrics = ["Total_Bees", "Total_Brood", "Total_Honey"]

        results = {}

        for metric in metrics:
            if metric not in netlogo_data.columns or metric not in bstew_data.columns:
                results[metric] = ValidationResult(
                    test_name=f"timeseries_{metric}",
                    passed=False,
                    score=0.0,
                    details={
                        "error": f"Metric '{metric}' not found in one or both datasets"
                    },
                )
                continue

            # Align time series (interpolate to common time points)
            aligned_netlogo, aligned_bstew = self._align_time_series(
                netlogo_data, bstew_data, metric
            )

            if len(aligned_netlogo) == 0 or len(aligned_bstew) == 0:
                results[metric] = ValidationResult(
                    test_name=f"timeseries_{metric}",
                    passed=False,
                    score=0.0,
                    details={"error": "No overlapping time points found"},
                )
                continue

            # Calculate comparison metrics
            comp_metrics = self._calculate_comparison_metrics(
                aligned_netlogo, aligned_bstew
            )

            # Determine if comparison passes
            passed = (
                comp_metrics.r2 > 0.7
                and comp_metrics.correlation > 0.8
                and comp_metrics.ks_p_value > self.significance_level
            )

            recommendation = self._generate_comparison_recommendation(
                comp_metrics, metric
            )

            results[metric] = ValidationResult(
                test_name=f"timeseries_{metric}",
                passed=passed,
                score=comp_metrics.agreement_score,
                p_value=comp_metrics.p_value,
                details=comp_metrics.__dict__,
                recommendation=recommendation,
            )

        return results

    def _align_time_series(
        self, netlogo_data: pd.DataFrame, bstew_data: pd.DataFrame, metric: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align time series to common time points"""

        # Assume both have 'Day' or 'Step' column
        time_col = "Day" if "Day" in netlogo_data.columns else "Step"

        if time_col not in netlogo_data.columns or time_col not in bstew_data.columns:
            # Fall back to index alignment
            min_len = min(len(netlogo_data), len(bstew_data))
            return (
                np.array(netlogo_data[metric].iloc[:min_len].values),
                np.array(bstew_data[metric].iloc[:min_len].values),
            )

        # Find common time range
        netlogo_times = netlogo_data[time_col].values
        bstew_times = bstew_data[time_col].values

        common_start = max(
            float(np.array(netlogo_times).min()), float(np.array(bstew_times).min())
        )
        common_end = min(
            float(np.array(netlogo_times).max()), float(np.array(bstew_times).max())
        )

        # Filter to common time range
        netlogo_mask = (np.array(netlogo_times) >= common_start) & (
            np.array(netlogo_times) <= common_end
        )
        bstew_mask = (np.array(bstew_times) >= common_start) & (
            np.array(bstew_times) <= common_end
        )

        netlogo_filtered = netlogo_data[netlogo_mask]
        bstew_filtered = bstew_data[bstew_mask]

        # Interpolate to common time points if needed
        if len(netlogo_filtered) != len(bstew_filtered):
            common_times = np.linspace(
                common_start,
                common_end,
                min(len(netlogo_filtered), len(bstew_filtered)),
            )

            netlogo_interp = np.interp(
                common_times, netlogo_filtered[time_col], netlogo_filtered[metric]
            )
            bstew_interp = np.interp(
                common_times, bstew_filtered[time_col], bstew_filtered[metric]
            )

            return netlogo_interp, bstew_interp
        else:
            return np.asarray(netlogo_filtered[metric].values), np.asarray(
                bstew_filtered[metric].values
            )

    def _calculate_comparison_metrics(
        self, netlogo_values: np.ndarray, bstew_values: np.ndarray
    ) -> ComparisonMetrics:
        """Calculate statistical comparison metrics"""

        # Remove any NaN values
        valid_mask = ~(np.isnan(netlogo_values) | np.isnan(bstew_values))
        netlogo_clean = netlogo_values[valid_mask]
        bstew_clean = bstew_values[valid_mask]

        if len(netlogo_clean) == 0:
            return ComparisonMetrics(
                rmse=0.0,
                mae=0.0,
                r2=0.0,
                correlation=0.0,
                p_value=1.0,
                ks_statistic=1.0,
                ks_p_value=1.0,
            )

        # Basic metrics
        rmse = np.sqrt(mean_squared_error(netlogo_clean, bstew_clean))
        mae = mean_absolute_error(netlogo_clean, bstew_clean)
        r2 = r2_score(netlogo_clean, bstew_clean)

        # Correlation
        correlation, p_value = stats.pearsonr(netlogo_clean, bstew_clean)

        # Kolmogorov-Smirnov test for distribution similarity
        ks_statistic, ks_p_value = stats.ks_2samp(netlogo_clean, bstew_clean)

        return ComparisonMetrics(
            rmse=float(rmse),
            mae=float(mae),
            r2=float(r2),
            correlation=float(correlation),
            p_value=float(p_value),
            ks_statistic=float(ks_statistic),
            ks_p_value=float(ks_p_value),
        )

    def _generate_comparison_recommendation(
        self, metrics: ComparisonMetrics, metric_name: str
    ) -> str:
        """Generate recommendation based on comparison metrics"""

        recommendations = []

        if metrics.r2 < 0.5:
            recommendations.append(
                f"Low R² ({metrics.r2:.3f}) indicates poor model fit for {metric_name}"
            )

        if metrics.correlation < 0.7:
            recommendations.append(
                f"Low correlation ({metrics.correlation:.3f}) suggests different dynamics"
            )

        if metrics.ks_p_value < 0.05:
            recommendations.append(
                f"Significant distribution difference (KS p={metrics.ks_p_value:.3f})"
            )

        if not recommendations:
            return f"Good agreement for {metric_name} (R²={metrics.r2:.3f}, r={metrics.correlation:.3f})"
        else:
            return f"Issues with {metric_name}: " + "; ".join(recommendations)

    def create_comparison_plots(
        self,
        netlogo_data: pd.DataFrame,
        bstew_data: pd.DataFrame,
        output_dir: str = "artifacts/validation_plots",
    ) -> List[str]:
        """Create visual comparison plots"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        plot_files = []
        metrics = ["Total_Bees", "Total_Brood", "Total_Honey"]

        for metric in metrics:
            if metric not in netlogo_data.columns or metric not in bstew_data.columns:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Model Comparison: {metric}", fontsize=16)

            # Time series comparison
            time_col = (
                "Day" if "Day" in netlogo_data.columns else range(len(netlogo_data))
            )
            axes[0, 0].plot(time_col, netlogo_data[metric], label="NetLogo", alpha=0.8)
            axes[0, 0].plot(time_col, bstew_data[metric], label="BSTEW", alpha=0.8)
            axes[0, 0].set_title("Time Series Comparison")
            axes[0, 0].set_xlabel("Time")
            axes[0, 0].set_ylabel(metric)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Scatter plot
            aligned_netlogo, aligned_bstew = self._align_time_series(
                netlogo_data, bstew_data, metric
            )
            axes[0, 1].scatter(aligned_netlogo, aligned_bstew, alpha=0.6)
            axes[0, 1].plot(
                [min(aligned_netlogo), max(aligned_netlogo)],
                [min(aligned_netlogo), max(aligned_netlogo)],
                "r--",
            )
            axes[0, 1].set_title("Scatter Plot (Perfect Agreement = Red Line)")
            axes[0, 1].set_xlabel(f"NetLogo {metric}")
            axes[0, 1].set_ylabel(f"BSTEW {metric}")
            axes[0, 1].grid(True, alpha=0.3)

            # Residuals
            residuals = aligned_bstew - aligned_netlogo
            axes[1, 0].plot(residuals)
            axes[1, 0].axhline(y=0, color="r", linestyle="--")
            axes[1, 0].set_title("Residuals (BSTEW - NetLogo)")
            axes[1, 0].set_xlabel("Time")
            axes[1, 0].set_ylabel("Residual")
            axes[1, 0].grid(True, alpha=0.3)

            # Distribution comparison
            axes[1, 1].hist(
                aligned_netlogo, bins=30, alpha=0.5, label="NetLogo", density=True
            )
            axes[1, 1].hist(
                aligned_bstew, bins=30, alpha=0.5, label="BSTEW", density=True
            )
            axes[1, 1].set_title("Distribution Comparison")
            axes[1, 1].set_xlabel(metric)
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            plot_file = output_path / f"{metric}_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()

            plot_files.append(str(plot_file))

        return plot_files


class ModelValidator:
    """
    Comprehensive model validation framework.

    Integrates parameter validation, output comparison, and statistical tests.
    """

    def __init__(self) -> None:
        self.parameter_validator = ParameterValidator()
        self.output_comparator = OutputComparator()
        self.logger = logging.getLogger(__name__)

    def run_full_validation(
        self,
        netlogo_data: Dict[str, Any],
        bstew_data: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run complete validation suite"""

        if validation_config is None:
            validation_config = self._get_default_validation_config()

        validation_results: Dict[str, Any] = {
            "parameter_validation": [],
            "output_comparison": {},
            "summary": {},
            "recommendations": [],
            "plots": [],
        }

        try:
            # Parameter validation
            if "parameters" in netlogo_data and "config" in bstew_data:
                self.logger.info("Running parameter validation...")
                param_results = self.parameter_validator.validate_parameter_mapping(
                    netlogo_data["parameters"], bstew_data["config"]
                )
                validation_results["parameter_validation"] = [
                    r.__dict__ for r in param_results
                ]

            # Output comparison
            if "output_data" in netlogo_data and "output_data" in bstew_data:
                self.logger.info("Running output comparison...")
                comparison_metrics = validation_config.get("comparison_metrics")
                if comparison_metrics is not None:
                    output_results = self.output_comparator.compare_time_series(
                        netlogo_data["output_data"],
                        bstew_data["output_data"],
                        comparison_metrics,
                    )
                    validation_results["output_comparison"] = {
                        k: v.__dict__ for k, v in output_results.items()
                    }

                # Create comparison plots
                if validation_config.get("create_plots", True):
                    self.logger.info("Creating comparison plots...")
                    plot_files = self.output_comparator.create_comparison_plots(
                        netlogo_data["output_data"],
                        bstew_data["output_data"],
                        validation_config.get("plot_dir", "validation_plots"),
                    )
                    validation_results["plots"] = plot_files

            # Generate summary and recommendations
            validation_results["summary"] = self._generate_validation_summary(
                validation_results
            )
            validation_results["recommendations"] = self._generate_recommendations(
                validation_results
            )

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            validation_results["error"] = str(e)

        return validation_results

    def _get_default_validation_config(self) -> Dict[str, Any]:
        """Get default validation configuration"""
        return {
            "comparison_metrics": ["Total_Bees", "Total_Brood", "Total_Honey"],
            "create_plots": True,
            "plot_dir": "validation_plots",
            "significance_level": 0.05,
            "min_r2_threshold": 0.7,
            "min_correlation_threshold": 0.8,
        }

    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary statistics"""

        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "overall_score": 0.0,
            "parameter_issues": 0,
            "output_issues": 0,
        }

        # Parameter validation summary
        param_results = results.get("parameter_validation", [])
        if param_results:
            param_passed = sum(1 for r in param_results if r["passed"])
            summary["total_tests"] += len(param_results)
            summary["passed_tests"] += param_passed
            summary["failed_tests"] += len(param_results) - param_passed
            summary["parameter_issues"] = len(param_results) - param_passed

        # Output comparison summary
        output_results = results.get("output_comparison", {})
        if output_results:
            output_passed = sum(1 for r in output_results.values() if r["passed"])
            summary["total_tests"] += len(output_results)
            summary["passed_tests"] += output_passed
            summary["failed_tests"] += len(output_results) - output_passed
            summary["output_issues"] = len(output_results) - output_passed

        # Overall score calculation
        if summary["total_tests"] > 0:
            all_scores = []

            # Collect parameter scores
            all_scores.extend([r["score"] for r in param_results])

            # Collect output scores
            all_scores.extend([r["score"] for r in output_results.values()])

            summary["overall_score"] = np.mean(all_scores) if all_scores else 0.0
            summary["pass_rate"] = summary["passed_tests"] / summary["total_tests"]

        return summary

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate validation recommendations"""

        recommendations = []
        summary = results.get("summary", {})

        # Overall recommendations
        if summary.get("pass_rate", 0) < 0.8:
            recommendations.append(
                "Overall validation pass rate is low. Review failed tests for critical issues."
            )

        if summary.get("overall_score", 0) < 0.7:
            recommendations.append(
                "Overall validation score is low. Focus on improving model agreement."
            )

        # Parameter-specific recommendations
        param_results = results.get("parameter_validation", [])
        failed_params = [r for r in param_results if not r["passed"]]

        if failed_params:
            recommendations.append(
                f"Parameter validation failed for {len(failed_params)} parameters. "
                "Review parameter mappings and ranges."
            )

        # Output-specific recommendations
        output_results = results.get("output_comparison", {})
        failed_outputs = [k for k, v in output_results.items() if not v["passed"]]

        if failed_outputs:
            recommendations.append(
                f"Output comparison failed for: {', '.join(failed_outputs)}. "
                "Review model dynamics and equations."
            )

        # Specific metric recommendations
        for metric, result in output_results.items():
            if result.get("recommendation"):
                recommendations.append(f"{metric}: {result['recommendation']}")

        return recommendations

    def save_validation_report(self, results: Dict[str, Any], output_file: str) -> None:
        """Save validation report to file"""

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "validation_results": results,
            "metadata": {
                "validator_version": "1.0.0",
                "validation_framework": "BSTEW-NetLogo Comparison",
            },
        }

        output_path = Path(output_file)

        if output_path.suffix.lower() == ".json":
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif output_path.suffix.lower() in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.dump(report, f, default_flow_style=False)
        else:
            # Default to JSON
            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Validation report saved to {output_path}")


class NetLogoSpecificValidator:
    """
    NetLogo-specific validation tests for BEE-STEWARD compatibility.

    Tests biological parameters, genetic system, and species-specific behaviors.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def validate_genetic_parameters(
        self, netlogo_data: Dict[str, Any], bstew_data: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate genetic system parameters"""
        results = []

        # Test CSD (Complementary Sex Determination) parameters
        csd_tests = [
            ("sex_locus_enabled", "genetics.csd_enabled", "boolean"),
            ("diploid_male_mortality", "genetics.diploid_male_mortality", "float"),
            ("allele_count", "genetics.allele_pool_size", "integer"),
            ("inbreeding_depression", "genetics.inbreeding_coefficient", "float"),
        ]

        for netlogo_param, bstew_param, expected_type in csd_tests:
            result = self._validate_parameter_match(
                netlogo_data, bstew_data, netlogo_param, bstew_param, expected_type
            )
            results.append(result)

        return results

    def validate_species_parameters(
        self, netlogo_species: Dict[str, Any], bstew_species: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate species-specific parameters"""
        results = []

        # Test proboscis length compatibility
        for species_id in netlogo_species.keys():
            if species_id in bstew_species:
                # NetLogoSpeciesData has parameters stored in .parameters field
                netlogo_data = netlogo_species[species_id]
                if hasattr(netlogo_data, "parameters"):
                    netlogo_params = netlogo_data.parameters
                    netlogo_min = netlogo_params.get("proboscis_min_mm", 0)
                    netlogo_max = netlogo_params.get("proboscis_max_mm", 0)
                else:
                    # Fallback if netlogo_data is already a dict
                    netlogo_min = netlogo_data.get("proboscis_min_mm", 0)
                    netlogo_max = netlogo_data.get("proboscis_max_mm", 0)

                # Handle bstew_species which should be a dict
                bstew_data = bstew_species[species_id]
                if isinstance(bstew_data, dict):
                    bstew_min = bstew_data.get("proboscis_length_min", 0)
                    bstew_max = bstew_data.get("proboscis_length_max", 0)
                else:
                    # Fallback if bstew_data is an object with attributes
                    bstew_min = getattr(bstew_data, "proboscis_length_min", 0)
                    bstew_max = getattr(bstew_data, "proboscis_length_max", 0)

                proboscis_match = (
                    abs(netlogo_min - bstew_min) < 0.1
                    and abs(netlogo_max - bstew_max) < 0.1
                )

                results.append(
                    ValidationResult(
                        test_name=f"proboscis_length_{species_id}",
                        passed=proboscis_match,
                        score=1.0 if proboscis_match else 0.0,
                        details={
                            "netlogo_range": f"{netlogo_min}-{netlogo_max}",
                            "bstew_range": f"{bstew_min}-{bstew_max}",
                        },
                        recommendation=(
                            "Proboscis lengths must match for flower accessibility"
                            if not proboscis_match
                            else ""
                        ),
                    )
                )

        return results

    def validate_flower_accessibility(
        self, netlogo_flowers: Dict[str, Any], bstew_flowers: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate flower corolla depth parameters"""
        results = []

        for flower_name in netlogo_flowers.keys():
            if flower_name in bstew_flowers:
                # NetLogoFlowerData has corolla_depth_mm as a direct attribute
                netlogo_data = netlogo_flowers[flower_name]
                if hasattr(netlogo_data, "corolla_depth_mm"):
                    netlogo_depth = netlogo_data.corolla_depth_mm
                else:
                    # Fallback if netlogo_data is already a dict
                    netlogo_depth = netlogo_data.get("corolla_depth_mm", 0)

                # Handle bstew_flowers which should be a dict
                bstew_data = bstew_flowers[flower_name]
                if isinstance(bstew_data, dict):
                    bstew_depth = bstew_data.get("corolla_depth", 0)
                else:
                    # Fallback if bstew_data is an object with attributes
                    bstew_depth = getattr(bstew_data, "corolla_depth", 0)

                depth_match = abs(netlogo_depth - bstew_depth) < 0.1

                results.append(
                    ValidationResult(
                        test_name=f"corolla_depth_{flower_name}",
                        passed=depth_match,
                        score=1.0 if depth_match else 0.0,
                        details={
                            "netlogo_depth": netlogo_depth,
                            "bstew_depth": bstew_depth,
                        },
                        recommendation=(
                            "Corolla depths must match for accessibility calculations"
                            if not depth_match
                            else ""
                        ),
                    )
                )

        return results

    def validate_badger_parameters(
        self, netlogo_data: Dict[str, Any], bstew_data: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate badger predation parameters"""
        results = []

        badger_tests = [
            ("n_badgers", "predation.badger_count", "integer"),
            ("badger_foraging_range", "predation.badger_range_m", "float"),
            ("badger_encounter_prob", "predation.encounter_probability", "float"),
            ("badger_attack_prob", "predation.attack_probability", "float"),
        ]

        for netlogo_param, bstew_param, expected_type in badger_tests:
            result = self._validate_parameter_match(
                netlogo_data, bstew_data, netlogo_param, bstew_param, expected_type
            )
            results.append(result)

        return results

    def validate_development_parameters(
        self, netlogo_species: Dict[str, Any], bstew_species: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate development phase parameters"""
        results = []

        dev_params = [
            ("devAgeHatchingMin_d", "dev_age_hatching_min_d"),
            ("devAgePupationMin_d", "dev_age_pupation_min_d"),
            ("devAgeEmergingMin_d", "dev_age_emerging_min_d"),
            ("devWeightEgg_mg", "dev_weight_egg_mg"),
            ("devWeightPupationMin_mg", "dev_weight_pupation_min_mg"),
            ("devWeightPupationMax_mg", "dev_weight_pupation_max_mg"),
        ]

        for species_id in netlogo_species.keys():
            if species_id in bstew_species:
                for netlogo_param, bstew_param in dev_params:
                    # NetLogoSpeciesData has parameters stored in .parameters field
                    netlogo_data = netlogo_species[species_id]
                    if hasattr(netlogo_data, "parameters"):
                        netlogo_value = netlogo_data.parameters.get(netlogo_param, 0)
                    else:
                        # Fallback if netlogo_data is already a dict
                        netlogo_value = netlogo_data.get(netlogo_param, 0)

                    # Handle bstew_species which should be a dict
                    bstew_data = bstew_species[species_id]
                    if hasattr(bstew_data, "get"):
                        bstew_value = bstew_data.get(bstew_param, 0)
                    else:
                        # Fallback if bstew_data is an object with attributes
                        bstew_value = getattr(bstew_data, bstew_param, 0)

                    param_match = abs(netlogo_value - bstew_value) < 0.1

                    results.append(
                        ValidationResult(
                            test_name=f"development_{species_id}_{netlogo_param}",
                            passed=param_match,
                            score=1.0 if param_match else 0.0,
                            details={
                                "netlogo_value": netlogo_value,
                                "bstew_value": bstew_value,
                            },
                            recommendation=(
                                f"Development parameter {netlogo_param} must match"
                                if not param_match
                                else ""
                            ),
                        )
                    )

        return results

    def _validate_parameter_match(
        self,
        netlogo_data: Dict[str, Any],
        bstew_data: Dict[str, Any],
        netlogo_param: str,
        bstew_param: str,
        expected_type: str,
    ) -> ValidationResult:
        """Helper method to validate parameter matches"""

        netlogo_value = self._get_nested_value(netlogo_data, netlogo_param)
        bstew_value = self._get_nested_value(bstew_data, bstew_param)

        if netlogo_value is None or bstew_value is None:
            return ValidationResult(
                test_name=f"parameter_{netlogo_param}",
                passed=False,
                score=0.0,
                details={
                    "netlogo_value": netlogo_value,
                    "bstew_value": bstew_value,
                    "error": "Parameter not found",
                },
                recommendation=f"Ensure {netlogo_param} maps to {bstew_param}",
            )

        # Type validation
        if expected_type == "boolean":
            values_match = bool(netlogo_value) == bool(bstew_value)
        elif expected_type == "integer":
            values_match = int(netlogo_value) == int(bstew_value)
        elif expected_type == "float":
            values_match = abs(float(netlogo_value) - float(bstew_value)) < 1e-6
        else:
            values_match = netlogo_value == bstew_value

        return ValidationResult(
            test_name=f"parameter_{netlogo_param}",
            passed=values_match,
            score=1.0 if values_match else 0.0,
            details={
                "netlogo_value": netlogo_value,
                "bstew_value": bstew_value,
                "expected_type": expected_type,
            },
            recommendation=(
                f"Parameter {netlogo_param} must match {bstew_param}"
                if not values_match
                else ""
            ),
        )

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value


def validate_model_compatibility(
    netlogo_results_path: str,
    bstew_results_path: str,
    output_dir: str = "artifacts/validation_results",
) -> Dict[str, Any]:
    """
    Convenience function for model validation.

    Args:
        netlogo_results_path: Path to NetLogo simulation results
        bstew_results_path: Path to BSTEW simulation results
        output_dir: Output directory for validation results

    Returns:
        Validation results dictionary
    """

    # Load data
    netlogo_data = {}
    bstew_data = {}

    # This would load actual data - simplified for example
    if Path(netlogo_results_path).exists():
        netlogo_data["output_data"] = pd.read_csv(netlogo_results_path)

    if Path(bstew_results_path).exists():
        bstew_data["output_data"] = pd.read_csv(bstew_results_path)

    # Run validation
    validator = ModelValidator()
    results = validator.run_full_validation(netlogo_data, bstew_data)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    validator.save_validation_report(
        results, str(output_path / "validation_report.json")
    )

    return results


def validate_netlogo_compatibility(
    netlogo_data_dir: str,
    bstew_config: Dict[str, Any],
    output_dir: str = "artifacts/netlogo_validation",
) -> Dict[str, Any]:
    """
    Comprehensive NetLogo compatibility validation.

    Args:
        netlogo_data_dir: Directory containing NetLogo data files
        bstew_config: BSTEW configuration dictionary
        output_dir: Output directory for validation results

    Returns:
        Comprehensive validation results
    """
    from ..data.netlogo_parser import NetLogoDataParser

    # Parse NetLogo data
    parser = NetLogoDataParser()
    netlogo_data = parser.parse_all_data_files(netlogo_data_dir)

    # Initialize validators
    param_validator = ParameterValidator()
    netlogo_validator = NetLogoSpecificValidator()

    # Run all validation tests
    results = {
        "parameter_validation": [],
        "genetic_validation": [],
        "species_validation": [],
        "flower_validation": [],
        "badger_validation": [],
        "development_validation": [],
        "compatibility_score": 0.0,
        "recommendations": [],
    }

    try:
        # Parameter validation
        if "parameters" in netlogo_data:
            param_results = param_validator.validate_parameter_mapping(
                netlogo_data["parameters"], bstew_config
            )
            results["parameter_validation"] = [r.__dict__ for r in param_results]

        # Genetic system validation
        genetic_results = netlogo_validator.validate_genetic_parameters(
            netlogo_data.get("parameters", {}), bstew_config
        )
        results["genetic_validation"] = [r.__dict__ for r in genetic_results]

        # Species validation
        if "species" in netlogo_data:
            species_results = netlogo_validator.validate_species_parameters(
                netlogo_data["species"], bstew_config.get("species", {})
            )
            results["species_validation"] = [r.__dict__ for r in species_results]

        # Flower validation
        if "flowers" in netlogo_data:
            flower_results = netlogo_validator.validate_flower_accessibility(
                netlogo_data["flowers"], bstew_config.get("flowers", {})
            )
            results["flower_validation"] = [r.__dict__ for r in flower_results]

        # Badger validation
        badger_results = netlogo_validator.validate_badger_parameters(
            netlogo_data.get("parameters", {}), bstew_config
        )
        results["badger_validation"] = [r.__dict__ for r in badger_results]

        # Development validation
        if "species" in netlogo_data:
            dev_results = netlogo_validator.validate_development_parameters(
                netlogo_data["species"], bstew_config.get("species", {})
            )
            results["development_validation"] = [r.__dict__ for r in dev_results]

        # Calculate overall compatibility score
        all_results: List[Any] = []
        for validation_type in [
            "parameter_validation",
            "genetic_validation",
            "species_validation",
            "flower_validation",
            "badger_validation",
            "development_validation",
        ]:
            validation_result = results[validation_type]
            if isinstance(validation_result, list):
                all_results.extend(validation_result)
            else:
                all_results.append(validation_result)

        if all_results:
            results["compatibility_score"] = sum(r["score"] for r in all_results) / len(
                all_results
            )

        # Generate recommendations
        failed_tests = [r for r in all_results if not r["passed"]]
        results["recommendations"] = [
            r["recommendation"] for r in failed_tests if r["recommendation"]
        ]

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "netlogo_compatibility_report.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    except Exception as e:
        results["error"] = str(e)

    return results
