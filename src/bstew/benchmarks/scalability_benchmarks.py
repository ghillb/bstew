"""
Scalability Benchmarks for BSTEW
===============================

Tests system behavior under varying loads and scales.
"""

import time
import psutil
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass

from ..core.model import BeeModel


@dataclass
class ScalabilityResult:
    """Result of a scalability test"""

    test_name: str
    parameter_value: Any
    execution_time: float
    memory_peak: float
    steps_per_second: float
    success: bool
    error_message: Optional[str] = None


class ScalabilityBenchmarks:
    """Scalability testing suite"""

    def __init__(self, output_directory: str = "artifacts/benchmarks/scalability"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def run_population_scaling_test(self) -> List[ScalabilityResult]:
        """Test performance scaling with population size"""

        population_sizes = [50, 100, 200, 500, 1000, 1500, 2000]
        results = []

        for pop_size in population_sizes:
            self.logger.info(f"Testing population scaling: {pop_size} bees")

            config = {
                "simulation": {"steps": 100, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": int(pop_size * 0.7),
                        "foragers": int(pop_size * 0.2),
                        "drones": int(pop_size * 0.05),
                        "brood": int(pop_size * 0.05),
                    }
                },
                "environment": {"patches": 50, "flower_density": 0.3},
            }

            result = self._run_scaling_test(
                test_name="population_scaling",
                parameter_value=pop_size,
                config=config,
                steps=100,
            )
            results.append(result)

        return results

    def run_duration_scaling_test(self) -> List[ScalabilityResult]:
        """Test performance scaling with simulation duration"""

        durations = [50, 100, 250, 500, 1000, 2000]
        results = []

        for duration in durations:
            self.logger.info(f"Testing duration scaling: {duration} steps")

            config = {
                "simulation": {"steps": duration, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 350,
                        "foragers": 100,
                        "drones": 25,
                        "brood": 25,
                    }
                },
                "environment": {"patches": 50, "flower_density": 0.3},
            }

            result = self._run_scaling_test(
                test_name="duration_scaling",
                parameter_value=duration,
                config=config,
                steps=duration,
            )
            results.append(result)

        return results

    def run_environment_scaling_test(self) -> List[ScalabilityResult]:
        """Test performance scaling with environment complexity"""

        patch_counts = [10, 25, 50, 100, 200, 400]
        results = []

        for patch_count in patch_counts:
            self.logger.info(f"Testing environment scaling: {patch_count} patches")

            config = {
                "simulation": {"steps": 100, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 350,
                        "foragers": 100,
                        "drones": 25,
                        "brood": 25,
                    }
                },
                "environment": {"patches": patch_count, "flower_density": 0.3},
            }

            result = self._run_scaling_test(
                test_name="environment_scaling",
                parameter_value=patch_count,
                config=config,
                steps=100,
            )
            results.append(result)

        return results

    def _run_scaling_test(
        self, test_name: str, parameter_value: Any, config: Dict[str, Any], steps: int
    ) -> ScalabilityResult:
        """Run individual scaling test"""

        try:
            process = psutil.Process()
            process.memory_info().rss / 1024 / 1024  # MB

            start_time = time.time()

            # Create and run model
            model = BeeModel(config=config)

            for step in range(steps):
                model.step()

                # Check memory every 50 steps to catch potential issues
                if step % 50 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    if current_memory > 4000:  # 4GB limit
                        raise MemoryError(
                            f"Memory usage exceeded 4GB: {current_memory:.1f}MB"
                        )

            end_time = time.time()

            # Calculate metrics
            execution_time = end_time - start_time
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_peak = end_memory
            steps_per_second = steps / execution_time if execution_time > 0 else 0

            model.cleanup()

            return ScalabilityResult(
                test_name=test_name,
                parameter_value=parameter_value,
                execution_time=execution_time,
                memory_peak=memory_peak,
                steps_per_second=steps_per_second,
                success=True,
            )

        except Exception as e:
            self.logger.error(
                f"Scaling test failed for {test_name} = {parameter_value}: {e}"
            )
            return ScalabilityResult(
                test_name=test_name,
                parameter_value=parameter_value,
                execution_time=0.0,
                memory_peak=0.0,
                steps_per_second=0.0,
                success=False,
                error_message=str(e),
            )

    def analyze_scaling_behavior(
        self, results: List[ScalabilityResult]
    ) -> Dict[str, Any]:
        """Analyze scaling behavior from test results"""

        successful_results = [r for r in results if r.success]

        if len(successful_results) < 2:
            return {"analysis": "insufficient_data"}

        # Extract data for analysis
        parameter_values = np.array([r.parameter_value for r in successful_results])
        execution_times = np.array([r.execution_time for r in successful_results])
        memory_peaks = np.array([r.memory_peak for r in successful_results])
        steps_per_second = np.array([r.steps_per_second for r in successful_results])

        # Fit scaling curves
        try:
            # Log-log regression for scaling analysis
            log_params = np.log(parameter_values)
            log_times = np.log(execution_times)
            log_memory = np.log(memory_peaks)

            # Time scaling coefficient
            time_coeffs = np.polyfit(log_params, log_times, 1)
            time_scaling_exponent = time_coeffs[0]

            # Memory scaling coefficient
            memory_coeffs = np.polyfit(log_params, log_memory, 1)
            memory_scaling_exponent = memory_coeffs[0]

            # Efficiency analysis
            efficiency_trend = self._analyze_efficiency_trend(
                parameter_values, steps_per_second
            )

            analysis = {
                "time_scaling": {
                    "exponent": time_scaling_exponent,
                    "interpretation": self._interpret_scaling_exponent(
                        time_scaling_exponent
                    ),
                    "is_acceptable": time_scaling_exponent < 1.5,
                },
                "memory_scaling": {
                    "exponent": memory_scaling_exponent,
                    "interpretation": self._interpret_scaling_exponent(
                        memory_scaling_exponent
                    ),
                    "is_acceptable": memory_scaling_exponent < 1.2,
                },
                "efficiency": efficiency_trend,
                "performance_breakdown": self._identify_performance_breakdown(
                    successful_results
                ),
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Scaling analysis failed: {e}")
            return {"analysis": "failed", "error": str(e)}

    def _interpret_scaling_exponent(self, exponent: float) -> str:
        """Interpret scaling exponent"""

        if exponent < 0.9:
            return "sub-linear (excellent)"
        elif exponent < 1.1:
            return "linear (good)"
        elif exponent < 1.5:
            return "super-linear (acceptable)"
        elif exponent < 2.0:
            return "quadratic (concerning)"
        else:
            return "exponential (problematic)"

    def _analyze_efficiency_trend(
        self, parameter_values: np.ndarray, steps_per_second: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze efficiency trends"""

        # Normalize efficiency by parameter size
        normalized_efficiency = steps_per_second / parameter_values

        # Calculate trend
        coeffs = np.polyfit(parameter_values, normalized_efficiency, 1)
        trend_slope = coeffs[0]

        return {
            "trend_slope": trend_slope,
            "trend_direction": "improving"
            if trend_slope > 0
            else "degrading"
            if trend_slope < -0.001
            else "stable",
            "efficiency_loss": (normalized_efficiency[0] - normalized_efficiency[-1])
            / normalized_efficiency[0]
            if len(normalized_efficiency) > 1
            else 0,
        }

    def _identify_performance_breakdown(
        self, results: List[ScalabilityResult]
    ) -> Optional[Dict[str, Any]]:
        """Identify where performance starts to break down"""

        if len(results) < 3:
            return None

        # Look for sudden changes in performance trends
        steps_per_second = [r.steps_per_second for r in results]
        parameter_values = [r.parameter_value for r in results]

        # Calculate performance ratios between consecutive points
        performance_ratios = []
        for i in range(1, len(steps_per_second)):
            if steps_per_second[i - 1] > 0:
                ratio = steps_per_second[i] / steps_per_second[i - 1]
                performance_ratios.append(
                    {
                        "parameter_value": parameter_values[i],
                        "ratio": ratio,
                        "performance_drop": ratio < 0.8,  # 20% performance drop
                    }
                )

        # Find breakdown point
        breakdown_points = [r for r in performance_ratios if r["performance_drop"]]

        if breakdown_points:
            first_breakdown = min(breakdown_points, key=lambda x: x["parameter_value"])
            return {
                "breakdown_detected": True,
                "breakdown_parameter": first_breakdown["parameter_value"],
                "performance_ratio": first_breakdown["ratio"],
            }

        return {"breakdown_detected": False}

    def save_results(
        self, test_type: str, results: List[ScalabilityResult], analysis: Dict[str, Any]
    ) -> str:
        """Save scalability test results"""

        output_data = {
            "test_type": test_type,
            "timestamp": time.time(),
            "results": [
                {
                    "test_name": r.test_name,
                    "parameter_value": r.parameter_value,
                    "execution_time": r.execution_time,
                    "memory_peak": r.memory_peak,
                    "steps_per_second": r.steps_per_second,
                    "success": r.success,
                    "error_message": r.error_message,
                }
                for r in results
            ],
            "analysis": analysis,
        }

        output_file = self.output_directory / f"{test_type}_results.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        self.logger.info(f"Scalability results saved to {output_file}")
        return str(output_file)
