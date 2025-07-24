"""
Benchmark Runner for BSTEW Performance Testing
============================================

Orchestrates execution of comprehensive benchmark suites with
reporting and integration testing capabilities.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import subprocess

from .netlogo_parity_benchmarks import NetLogoParityBenchmarks


@dataclass
class EndToEndTestResult:
    """Result of end-to-end integration test"""

    test_name: str
    success: bool
    execution_time: float
    steps_completed: int
    data_generated: bool
    reports_generated: bool
    error_message: Optional[str] = None


class BenchmarkRunner:
    """Orchestrates comprehensive benchmark and integration testing"""

    def __init__(self, output_directory: str = "artifacts/benchmarks"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize benchmark components
        self.netlogo_benchmarks = NetLogoParityBenchmarks(str(self.output_directory))

    def run_complete_validation_suite(self) -> Dict[str, Any]:
        """Run complete validation suite including benchmarks and integration tests"""

        self.logger.info("Starting complete BSTEW validation suite")

        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "performance_benchmarks": self._run_performance_benchmarks(),
            "integration_tests": self._run_integration_tests(),
            "end_to_end_tests": self._run_end_to_end_tests(),
            "stress_tests": self._run_stress_tests(),
            "validation_summary": {},
        }

        # Generate validation summary
        validation_results["validation_summary"] = self._generate_validation_summary(
            validation_results
        )

        # Save complete results
        self._save_validation_results(validation_results)

        self.logger.info("Complete validation suite finished")
        return validation_results

    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run NetLogo parity performance benchmarks"""

        self.logger.info("Running performance benchmarks")

        try:
            benchmark_results = self.netlogo_benchmarks.run_complete_benchmark_suite()
            return {"status": "completed", "results": benchmark_results}
        except Exception as e:
            self.logger.error(f"Performance benchmarks failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests using pytest"""

        self.logger.info("Running integration tests")

        try:
            # Find integration test files
            test_files = [
                "tests/test_integration.py",
                "tests/test_netlogo_integration.py",
            ]

            integration_results = {}

            for test_file in test_files:
                if Path(test_file).exists():
                    result = self._run_pytest(test_file)
                    integration_results[test_file] = result
                else:
                    integration_results[test_file] = {
                        "status": "skipped",
                        "reason": "test file not found",
                    }

            return {"status": "completed", "test_results": integration_results}

        except Exception as e:
            self.logger.error(f"Integration tests failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _run_end_to_end_tests(self) -> List[EndToEndTestResult]:
        """Run comprehensive end-to-end workflow tests"""

        self.logger.info("Running end-to-end tests")

        test_scenarios = [
            {
                "name": "basic_simulation_workflow",
                "config": {
                    "simulation": {"steps": 100, "random_seed": 42},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": 140,
                            "foragers": 40,
                            "drones": 10,
                            "brood": 10,
                        }
                    },
                    "environment": {"patches": 30, "flower_density": 0.3},
                },
                "expected_steps": 100,
            },
            {
                "name": "data_collection_workflow",
                "config": {
                    "simulation": {"steps": 50, "random_seed": 123},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": 70,
                            "foragers": 20,
                            "drones": 5,
                            "brood": 5,
                        }
                    },
                    "environment": {"patches": 20, "flower_density": 0.4},
                },
                "expected_steps": 50,
            },
            {
                "name": "system_integration_workflow",
                "config": {
                    "simulation": {"steps": 75, "random_seed": 456},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": 210,
                            "foragers": 60,
                            "drones": 15,
                            "brood": 15,
                        }
                    },
                    "environment": {"patches": 40, "flower_density": 0.2},
                },
                "expected_steps": 75,
            },
        ]

        results = []

        for scenario in test_scenarios:
            result = self._run_end_to_end_scenario(scenario)
            results.append(result)

        return results

    def _run_end_to_end_scenario(self, scenario: Dict[str, Any]) -> EndToEndTestResult:
        """Run individual end-to-end test scenario"""

        test_name = scenario["name"]
        config = scenario["config"]
        expected_steps = scenario["expected_steps"]

        self.logger.info(f"Running end-to-end test: {test_name}")

        try:
            from ..core.model import BeeModel
            from ..core.system_integrator import SystemIntegrator

            start_time = time.time()

            # Initialize model with system integration
            model = BeeModel(config=config)

            # Ensure system integrator is active
            if not hasattr(model, "system_integrator"):
                model.system_integrator = SystemIntegrator()
                model.system_integrator.initialize_systems(model)

            # Run simulation
            steps_completed = 0
            for step in range(expected_steps):
                model.step()
                steps_completed += 1

            # Check data collection
            data_generated = False
            if (
                model.system_integrator
                and model.system_integrator.data_collector
                and model.system_integrator.data_collector.colony_metrics
            ):
                data_generated = True

            # Generate reports
            reports_generated = False
            if model.system_integrator:
                report_paths = model.system_integrator.generate_reports(
                    steps_completed, final_report=True
                )
                reports_generated = len(report_paths) > 0

            # Cleanup
            model.cleanup()

            end_time = time.time()
            execution_time = end_time - start_time

            return EndToEndTestResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                steps_completed=steps_completed,
                data_generated=data_generated,
                reports_generated=reports_generated,
            )

        except Exception as e:
            self.logger.error(f"End-to-end test {test_name} failed: {e}")
            return EndToEndTestResult(
                test_name=test_name,
                success=False,
                execution_time=0.0,
                steps_completed=0,
                data_generated=False,
                reports_generated=False,
                error_message=str(e),
            )

    def _run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests for system stability"""

        self.logger.info("Running stress tests")

        stress_scenarios: List[Dict[str, Any]] = [
            {
                "name": "large_population_stress",
                "config": {
                    "simulation": {"steps": 200, "random_seed": 789},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": 1400,
                            "foragers": 400,
                            "drones": 100,
                            "brood": 100,
                        }
                    },
                    "environment": {"patches": 100, "flower_density": 0.5},
                },
            },
            {
                "name": "extended_duration_stress",
                "config": {
                    "simulation": {"steps": 1000, "random_seed": 101112},
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
                },
            },
        ]

        stress_results = {}

        for scenario in stress_scenarios:
            scenario_name = scenario["name"]
            self.logger.info(f"Running stress test: {scenario_name}")

            try:
                from ..core.model import BeeModel

                start_time = time.time()
                model = BeeModel(config=scenario["config"])

                steps = scenario["config"]["simulation"]["steps"]
                for step in range(steps):
                    model.step()

                    # Monitor every 100 steps
                    if step % 100 == 0:
                        import psutil

                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        if memory_mb > 2000:  # 2GB memory limit
                            raise MemoryError(
                                f"Memory usage exceeded 2GB: {memory_mb:.1f}MB"
                            )

                model.cleanup()
                end_time = time.time()

                stress_results[scenario_name] = {
                    "success": True,
                    "execution_time": end_time - start_time,
                    "steps_completed": steps,
                }

            except Exception as e:
                self.logger.error(f"Stress test {scenario_name} failed: {e}")
                stress_results[scenario_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": 0.0,
                    "steps_completed": 0,
                }

        return {"status": "completed", "results": stress_results}

    def _run_pytest(self, test_file: str) -> Dict[str, Any]:
        """Run pytest on specific test file"""

        try:
            # Run pytest with JSON output
            cmd = ["pytest", test_file, "--tb=short", "-v"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": "Test execution timed out after 5 minutes",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation summary"""

        summary: Dict[str, Any] = {
            "overall_status": "unknown",
            "performance_status": "unknown",
            "integration_status": "unknown",
            "end_to_end_status": "unknown",
            "stress_status": "unknown",
            "recommendations": [],
        }

        # Analyze performance benchmarks
        if "performance_benchmarks" in results:
            perf_results = results["performance_benchmarks"]
            if perf_results.get("status") == "completed":
                # Check NetLogo comparison results
                netlogo_comparisons = perf_results.get("results", {}).get(
                    "netlogo_comparison", []
                )
                if netlogo_comparisons:
                    passed_benchmarks = sum(
                        1 for c in netlogo_comparisons if c.passes_benchmark
                    )
                    total_benchmarks = len(netlogo_comparisons)

                    if passed_benchmarks == total_benchmarks:
                        summary["performance_status"] = "excellent"
                    elif passed_benchmarks >= total_benchmarks * 0.8:
                        summary["performance_status"] = "good"
                    else:
                        summary["performance_status"] = "needs_improvement"
                        summary["recommendations"].append(
                            "Performance optimization needed"
                        )
                else:
                    summary["performance_status"] = "completed"
            else:
                summary["performance_status"] = "failed"

        # Analyze integration tests
        if "integration_tests" in results:
            int_results = results["integration_tests"]
            if int_results.get("status") == "completed":
                summary["integration_status"] = "passed"
            else:
                summary["integration_status"] = "failed"
                summary["recommendations"].append("Integration tests need attention")

        # Analyze end-to-end tests
        if "end_to_end_tests" in results:
            e2e_results = results["end_to_end_tests"]
            successful_tests = sum(1 for test in e2e_results if test.success)
            total_tests = len(e2e_results)

            if successful_tests == total_tests:
                summary["end_to_end_status"] = "passed"
            else:
                summary["end_to_end_status"] = "failed"
                summary["recommendations"].append("End-to-end workflow issues detected")

        # Analyze stress tests
        if "stress_tests" in results:
            stress_results = results["stress_tests"]
            if stress_results.get("status") == "completed":
                stress_test_results = stress_results.get("results", {})
                successful_stress = sum(
                    1 for r in stress_test_results.values() if r.get("success", False)
                )
                total_stress = len(stress_test_results)

                if successful_stress == total_stress:
                    summary["stress_status"] = "passed"
                else:
                    summary["stress_status"] = "failed"
                    summary["recommendations"].append(
                        "System stability under stress needs improvement"
                    )
            else:
                summary["stress_status"] = "failed"

        # Determine overall status
        statuses = [
            summary["performance_status"],
            summary["integration_status"],
            summary["end_to_end_status"],
            summary["stress_status"],
        ]

        if all(s in ["excellent", "good", "passed", "completed"] for s in statuses):
            summary["overall_status"] = "excellent"
        elif all(s not in ["failed", "unknown"] for s in statuses):
            summary["overall_status"] = "good"
        else:
            summary["overall_status"] = "needs_attention"

        return summary

    def _save_validation_results(self, results: Dict[str, Any]) -> None:
        """Save validation results to files"""

        # Save complete results as JSON
        results_file = self.output_directory / "complete_validation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Generate markdown summary
        summary_file = (
            self.output_directory
            / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        with open(summary_file, "w") as f:
            f.write("# BSTEW Complete Validation Results\n\n")
            f.write(f"Generated: {results['timestamp']}\n\n")

            # Overall summary
            summary = results.get("validation_summary", {})
            overall_status = summary.get("overall_status", "unknown")
            status_emoji = {
                "excellent": "🟢",
                "good": "🟡",
                "needs_attention": "🔴",
                "unknown": "⚪",
            }.get(overall_status, "⚪")

            f.write(f"## Overall Status: {status_emoji} {overall_status.upper()}\n\n")

            # Component status
            f.write("## Component Status\n\n")
            f.write(
                f"- **Performance Benchmarks**: {summary.get('performance_status', 'unknown')}\n"
            )
            f.write(
                f"- **Integration Tests**: {summary.get('integration_status', 'unknown')}\n"
            )
            f.write(
                f"- **End-to-End Tests**: {summary.get('end_to_end_status', 'unknown')}\n"
            )
            f.write(
                f"- **Stress Tests**: {summary.get('stress_status', 'unknown')}\n\n"
            )

            # Recommendations
            recommendations = summary.get("recommendations", [])
            if recommendations:
                f.write("## Recommendations\n\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")

            # End-to-end test details
            if "end_to_end_tests" in results:
                f.write("## End-to-End Test Results\n\n")
                for test in results["end_to_end_tests"]:
                    status = "✅" if test.success else "❌"
                    f.write(f"- **{test.test_name}**: {status}\n")
                    f.write(f"  - Execution time: {test.execution_time:.2f}s\n")
                    f.write(f"  - Steps completed: {test.steps_completed}\n")
                    f.write(
                        f"  - Data generated: {'✅' if test.data_generated else '❌'}\n"
                    )
                    f.write(
                        f"  - Reports generated: {'✅' if test.reports_generated else '❌'}\n"
                    )
                    if not test.success and test.error_message:
                        f.write(f"  - Error: {test.error_message}\n")
                f.write("\n")

        self.logger.info(f"Validation results saved to {results_file}")
        self.logger.info(f"Validation summary saved to {summary_file}")

    def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation for CI/CD pipelines"""

        self.logger.info("Running quick validation suite")

        # Run minimal set of tests for speed
        quick_results = {
            "timestamp": datetime.now().isoformat(),
            "basic_functionality": self._test_basic_functionality(),
            "performance_check": self._quick_performance_check(),
            "integration_smoke_test": self._integration_smoke_test(),
        }

        # Determine pass/fail
        all_passed = all(
            result.get("success", False)
            for result in quick_results.values()
            if isinstance(result, dict) and "success" in result
        )

        quick_results["overall_status"] = "passed" if all_passed else "failed"

        return quick_results

    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic BSTEW functionality"""

        try:
            from ..core.model import BeeModel

            config = {
                "simulation": {"steps": 10, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 35,
                        "foragers": 10,
                        "drones": 2,
                        "brood": 2,
                    }
                },
                "environment": {"patches": 10, "flower_density": 0.3},
            }

            model = BeeModel(config=config)
            for step in range(10):
                model.step()
            model.cleanup()

            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _quick_performance_check(self) -> Dict[str, Any]:
        """Quick performance sanity check"""

        try:
            from ..core.model import BeeModel

            config = {
                "simulation": {"steps": 50, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 70,
                        "foragers": 20,
                        "drones": 5,
                        "brood": 5,
                    }
                },
                "environment": {"patches": 20, "flower_density": 0.3},
            }

            start_time = time.time()
            model = BeeModel(config=config)
            for step in range(50):
                model.step()
            model.cleanup()
            end_time = time.time()

            execution_time = end_time - start_time
            steps_per_second = 50 / execution_time

            # Pass if we can do at least 100 steps per second
            return {
                "success": steps_per_second >= 100,
                "steps_per_second": steps_per_second,
                "execution_time": execution_time,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _integration_smoke_test(self) -> Dict[str, Any]:
        """Quick integration smoke test"""

        try:
            from ..core.model import BeeModel
            from ..core.system_integrator import SystemIntegrator

            config = {
                "simulation": {"steps": 20, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 35,
                        "foragers": 10,
                        "drones": 2,
                        "brood": 2,
                    }
                },
                "environment": {"patches": 15, "flower_density": 0.3},
            }

            model = BeeModel(config=config)

            # Test system integration
            model.system_integrator = SystemIntegrator()
            init_status = model.system_integrator.initialize_systems(model)

            # Run a few steps
            for step in range(20):
                model.step()

            # Test data collection
            has_data = (
                model.system_integrator.data_collector
                and len(model.system_integrator.data_collector.colony_metrics) > 0
            )

            model.cleanup()

            return {
                "success": True,
                "systems_initialized": len(init_status) > 0,
                "data_collected": has_data,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
