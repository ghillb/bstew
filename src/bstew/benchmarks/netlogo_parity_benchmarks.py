"""
NetLogo Parity Performance Benchmarks
====================================

Comprehensive benchmarking suite comparing BSTEW performance against 
NetLogo BEE-STEWARD v2 for parity validation and optimization.
"""

import time
import psutil
import logging
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from ..core.model import BeeModel
from ..utils.performance import PerformanceProfiler


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    benchmark_name: str
    execution_time: float
    memory_peak: float
    memory_average: float
    cpu_usage: float
    steps_per_second: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class NetLogoComparisonResult:
    """NetLogo vs BSTEW comparison result"""
    metric_name: str
    bstew_value: float
    netlogo_reference: float
    performance_ratio: float
    difference: float
    relative_difference: float
    passes_benchmark: bool


class NetLogoParityBenchmarks:
    """Performance benchmarks for NetLogo parity validation"""
    
    def __init__(self, output_directory: str = "artifacts/benchmarks"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.profiler = PerformanceProfiler()
        
        # NetLogo reference performance metrics (baseline targets)
        self.netlogo_references = {
            "simulation_speed": {
                "small_colony": 1000.0,  # steps/second for 100 bees
                "medium_colony": 500.0,  # steps/second for 500 bees
                "large_colony": 200.0,   # steps/second for 1000 bees
            },
            "memory_usage": {
                "small_colony": 50.0,    # MB for 100 bees
                "medium_colony": 150.0,  # MB for 500 bees
                "large_colony": 300.0,   # MB for 1000 bees
            },
            "initialization_time": {
                "small_colony": 0.5,     # seconds
                "medium_colony": 1.5,    # seconds
                "large_colony": 3.0,     # seconds
            }
        }
    
    def run_complete_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete NetLogo parity benchmark suite"""
        
        self.logger.info("Starting NetLogo parity benchmark suite")
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "simulation_speed": self._benchmark_simulation_speed(),
            "memory_efficiency": self._benchmark_memory_efficiency(),
            "initialization_performance": self._benchmark_initialization(),
            "scalability": self._benchmark_scalability(),
            "behavioral_accuracy": self._benchmark_behavioral_accuracy(),
            "netlogo_comparison": self._compare_with_netlogo_references(),
            "regression_analysis": self._check_performance_regression()
        }
        
        # Generate benchmark report
        self._generate_benchmark_report(benchmark_results)
        
        self.logger.info("NetLogo parity benchmark suite completed")
        return benchmark_results
    
    def _benchmark_simulation_speed(self) -> Dict[str, BenchmarkResult]:
        """Benchmark simulation execution speed for different colony sizes"""
        
        results = {}
        colony_sizes = [
            ("small_colony", 100),
            ("medium_colony", 500), 
            ("large_colony", 1000)
        ]
        
        for size_name, bee_count in colony_sizes:
            self.logger.info(f"Benchmarking simulation speed for {size_name} ({bee_count} bees)")
            
            try:
                # Create model with specific colony size
                config = {
                    "simulation": {"steps": 100, "random_seed": 42},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": int(bee_count * 0.7),
                            "foragers": int(bee_count * 0.2),
                            "drones": int(bee_count * 0.05),
                            "brood": int(bee_count * 0.05)
                        }
                    },
                    "environment": {"patches": 50, "flower_density": 0.3}
                }
                
                # Start performance monitoring
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                process.cpu_percent()
                
                start_time = time.time()
                
                # Run simulation
                model = BeeModel(config=config)
                for step in range(100):
                    model.step()
                
                end_time = time.time()
                
                # Calculate metrics
                execution_time = end_time - start_time
                steps_per_second = 100 / execution_time
                
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = end_memory - start_memory
                cpu_usage = process.cpu_percent()
                
                model.cleanup()
                
                results[size_name] = BenchmarkResult(
                    benchmark_name=f"simulation_speed_{size_name}",
                    execution_time=execution_time,
                    memory_peak=end_memory,
                    memory_average=memory_used,
                    cpu_usage=cpu_usage,
                    steps_per_second=steps_per_second,
                    success=True,
                    metadata={"bee_count": bee_count, "steps": 100}
                )
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {size_name}: {e}")
                results[size_name] = BenchmarkResult(
                    benchmark_name=f"simulation_speed_{size_name}",
                    execution_time=0.0,
                    memory_peak=0.0,
                    memory_average=0.0,
                    cpu_usage=0.0,
                    steps_per_second=0.0,
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def _benchmark_memory_efficiency(self) -> Dict[str, BenchmarkResult]:
        """Benchmark memory usage and efficiency"""
        
        results = {}
        scenarios = [
            ("memory_baseline", 100, 100),
            ("memory_stress", 1000, 500),
            ("memory_extended", 500, 1000)
        ]
        
        for scenario_name, bee_count, steps in scenarios:
            self.logger.info(f"Benchmarking memory efficiency for {scenario_name}")
            
            try:
                config = {
                    "simulation": {"steps": steps, "random_seed": 42},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": int(bee_count * 0.7),
                            "foragers": int(bee_count * 0.2),
                            "drones": int(bee_count * 0.05),
                            "brood": int(bee_count * 0.05)
                        }
                    },
                    "environment": {"patches": 50, "flower_density": 0.3}
                }
                
                # Monitor memory throughout simulation
                process = psutil.Process()
                memory_samples = []
                
                start_time = time.time()
                model = BeeModel(config=config)
                
                for step in range(steps):
                    model.step()
                    if step % 10 == 0:  # Sample every 10 steps
                        memory_samples.append(process.memory_info().rss / 1024 / 1024)
                
                end_time = time.time()
                
                memory_peak = max(memory_samples)
                memory_average = statistics.mean(memory_samples)
                memory_growth = memory_samples[-1] - memory_samples[0] if len(memory_samples) > 1 else 0
                
                model.cleanup()
                
                results[scenario_name] = BenchmarkResult(
                    benchmark_name=f"memory_efficiency_{scenario_name}",
                    execution_time=end_time - start_time,
                    memory_peak=memory_peak,
                    memory_average=memory_average,
                    cpu_usage=process.cpu_percent(),
                    steps_per_second=steps / (end_time - start_time),
                    success=True,
                    metadata={
                        "memory_growth": memory_growth,
                        "memory_samples": len(memory_samples),
                        "bee_count": bee_count,
                        "steps": steps
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Memory benchmark failed for {scenario_name}: {e}")
                results[scenario_name] = BenchmarkResult(
                    benchmark_name=f"memory_efficiency_{scenario_name}",
                    execution_time=0.0,
                    memory_peak=0.0,
                    memory_average=0.0,
                    cpu_usage=0.0,
                    steps_per_second=0.0,
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def _benchmark_initialization(self) -> Dict[str, BenchmarkResult]:
        """Benchmark model initialization performance"""
        
        results = {}
        colony_sizes = [100, 500, 1000, 2000]
        
        for bee_count in colony_sizes:
            size_name = f"init_{bee_count}_bees"
            self.logger.info(f"Benchmarking initialization for {bee_count} bees")
            
            try:
                config = {
                    "simulation": {"steps": 1, "random_seed": 42},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": int(bee_count * 0.7),
                            "foragers": int(bee_count * 0.2),
                            "drones": int(bee_count * 0.05),
                            "brood": int(bee_count * 0.05)
                        }
                    },
                    "environment": {"patches": 50, "flower_density": 0.3}
                }
                
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                model = BeeModel(config=config)
                end_time = time.time()
                
                end_memory = process.memory_info().rss / 1024 / 1024
                initialization_time = end_time - start_time
                
                model.cleanup()
                
                results[size_name] = BenchmarkResult(
                    benchmark_name=f"initialization_{size_name}",
                    execution_time=initialization_time,
                    memory_peak=end_memory,
                    memory_average=end_memory - start_memory,
                    cpu_usage=process.cpu_percent(),
                    steps_per_second=0.0,  # Not applicable for initialization
                    success=True,
                    metadata={"bee_count": bee_count}  # type: ignore
                )
                
            except Exception as e:
                self.logger.error(f"Initialization benchmark failed for {bee_count} bees: {e}")
                results[size_name] = BenchmarkResult(
                    benchmark_name=f"initialization_{size_name}",
                    execution_time=0.0,
                    memory_peak=0.0,
                    memory_average=0.0,
                    cpu_usage=0.0,
                    steps_per_second=0.0,
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def _benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability characteristics"""
        
        bee_counts = [50, 100, 200, 500, 1000, 2000]
        scalability_results = []
        
        for bee_count in bee_counts:
            self.logger.info(f"Scalability test with {bee_count} bees")
            
            try:
                config = {
                    "simulation": {"steps": 50, "random_seed": 42},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": int(bee_count * 0.7),
                            "foragers": int(bee_count * 0.2),
                            "drones": int(bee_count * 0.05),
                            "brood": int(bee_count * 0.05)
                        }
                    },
                    "environment": {"patches": 50, "flower_density": 0.3}
                }
                
                start_time = time.time()
                model = BeeModel(config=config)
                
                for step in range(50):
                    model.step()
                
                end_time = time.time()
                execution_time = end_time - start_time
                steps_per_second = 50 / execution_time
                
                model.cleanup()
                
                scalability_results.append({
                    "bee_count": bee_count,
                    "execution_time": execution_time,
                    "steps_per_second": steps_per_second,
                    "efficiency": steps_per_second / bee_count  # Steps per second per bee
                })
                
            except Exception as e:
                self.logger.error(f"Scalability test failed for {bee_count} bees: {e}")
                scalability_results.append({
                    "bee_count": bee_count,
                    "execution_time": 0.0,
                    "steps_per_second": 0.0,
                    "efficiency": 0.0,
                    "error": str(e)
                })
        
        # Analyze scalability trends
        successful_results = [r for r in scalability_results if "error" not in r]
        if len(successful_results) > 1:
            # Calculate scalability coefficients
            bee_counts_array = np.array([r["bee_count"] for r in successful_results])
            exec_times_array = np.array([r["execution_time"] for r in successful_results])
            
            # Linear regression to find scaling relationship
            coeffs = np.polyfit(np.log(bee_counts_array), np.log(exec_times_array), 1)
            scaling_exponent = coeffs[0]  # Should be close to 1.0 for linear scaling
            
            return {
                "results": scalability_results,
                "scaling_analysis": {
                    "scaling_exponent": scaling_exponent,
                    "is_linear": abs(scaling_exponent - 1.0) < 0.2,
                    "efficiency_trend": "decreasing" if scaling_exponent > 1.1 else "stable"
                }
            }
        
        return {"results": scalability_results, "scaling_analysis": None}
    
    def _benchmark_behavioral_accuracy(self) -> Dict[str, Any]:
        """Benchmark behavioral accuracy and consistency"""
        
        # Run multiple simulations with same parameters to test consistency
        consistency_results = []
        
        for run in range(5):  # 5 replicate runs
            self.logger.info(f"Behavioral consistency test run {run + 1}")
            
            try:
                config = {
                    "simulation": {"steps": 200, "random_seed": run},
                    "colony": {
                        "initial_population": {
                            "queens": 1,
                            "workers": 350,
                            "foragers": 100,
                            "drones": 25,
                            "brood": 25
                        }
                    },
                    "environment": {"patches": 50, "flower_density": 0.3}
                }
                
                model = BeeModel(config=config)
                
                # Enable comprehensive data collection
                if not hasattr(model, 'system_integrator'):
                    from ..core.system_integrator import SystemIntegrator
                    model.system_integrator = SystemIntegrator()
                    model.system_integrator.initialize_systems(model)
                
                # Run simulation
                for step in range(200):
                    model.step()
                
                # Extract key behavioral metrics
                data_collector = model.system_integrator.data_collector
                
                # Calculate consistency metrics
                final_population = 0
                total_foraging_trips = 0
                
                for colony_metrics in data_collector.colony_metrics.values():
                    if hasattr(colony_metrics, 'population_size_day_list') and colony_metrics.population_size_day_list:
                        final_population += colony_metrics.population_size_day_list[-1]
                
                for bee_metrics in data_collector.bee_metrics.values():
                    if hasattr(bee_metrics, 'foraging_trips'):
                        total_foraging_trips += bee_metrics.foraging_trips
                
                consistency_results.append({
                    "run": run,
                    "final_population": final_population,
                    "total_foraging_trips": total_foraging_trips,
                    "success": True
                })
                
                model.cleanup()
                
            except Exception as e:
                self.logger.error(f"Behavioral consistency test run {run} failed: {e}")
                consistency_results.append({
                    "run": run,
                    "final_population": 0,
                    "total_foraging_trips": 0,
                    "success": False,
                    "error": str(e)
                })
        
        # Analyze consistency
        successful_runs = [r for r in consistency_results if r["success"]]
        if len(successful_runs) > 1:
            populations = [r["final_population"] for r in successful_runs]
            foraging_trips = [r["total_foraging_trips"] for r in successful_runs]
            
            return {
                "results": consistency_results,
                "consistency_analysis": {
                    "population_cv": statistics.stdev(populations) / statistics.mean(populations) if populations else 0,
                    "foraging_cv": statistics.stdev(foraging_trips) / statistics.mean(foraging_trips) if foraging_trips else 0,
                    "successful_runs": len(successful_runs),
                    "total_runs": len(consistency_results)
                }
            }
        
        return {"results": consistency_results, "consistency_analysis": None}
    
    def _compare_with_netlogo_references(self) -> List[NetLogoComparisonResult]:
        """Compare BSTEW performance with NetLogo reference values"""
        
        comparisons = []
        
        # Get simulation speed results
        speed_results = self._benchmark_simulation_speed()
        
        for size_name, result in speed_results.items():
            if result.success and size_name in self.netlogo_references["simulation_speed"]:
                netlogo_ref = self.netlogo_references["simulation_speed"][size_name]
                
                comparison = NetLogoComparisonResult(
                    metric_name=f"simulation_speed_{size_name}",
                    bstew_value=result.steps_per_second,
                    netlogo_reference=netlogo_ref,
                    performance_ratio=result.steps_per_second / netlogo_ref,
                    difference=result.steps_per_second - netlogo_ref,
                    relative_difference=abs(result.steps_per_second - netlogo_ref) / netlogo_ref,
                    passes_benchmark=result.steps_per_second >= netlogo_ref * 0.8  # 80% of NetLogo performance
                )
                comparisons.append(comparison)
        
        return comparisons
    
    def _check_performance_regression(self) -> Dict[str, Any]:
        """Check for performance regression against historical data"""
        
        # Look for previous benchmark results
        previous_results_file = self.output_directory / "latest_benchmark_results.json"
        
        if not previous_results_file.exists():
            return {"regression_check": "no_historical_data"}
        
        try:
            with open(previous_results_file, 'r') as f:
                previous_results = json.load(f)
            
            # Run current benchmarks for comparison
            current_speed_results = self._benchmark_simulation_speed()
            
            regression_analysis = []
            
            for size_name, current_result in current_speed_results.items():
                if (current_result.success and 
                    "simulation_speed" in previous_results and 
                    size_name in previous_results["simulation_speed"]):
                    
                    previous_result = previous_results["simulation_speed"][size_name]
                    
                    if "steps_per_second" in previous_result:
                        previous_sps = previous_result["steps_per_second"]
                        current_sps = current_result.steps_per_second
                        
                        performance_change = (current_sps - previous_sps) / previous_sps
                        
                        regression_analysis.append({
                            "metric": f"simulation_speed_{size_name}",
                            "previous_value": previous_sps,
                            "current_value": current_sps,
                            "performance_change": performance_change,
                            "is_regression": performance_change < -0.1  # 10% performance drop
                        })
            
            return {
                "regression_check": "completed",
                "analysis": regression_analysis,
                "has_regressions": any(r["is_regression"] for r in regression_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Regression check failed: {e}")
            return {"regression_check": "failed", "error": str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context"""
        import sys
        import platform
        
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.system()
        }
    
    def _generate_benchmark_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive benchmark report"""
        
        # Save raw results
        results_file = self.output_directory / "latest_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = self.output_directory / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(summary_file, 'w') as f:
            f.write("# NetLogo Parity Benchmark Results\n\n")
            f.write(f"Generated: {results['timestamp']}\n\n")
            
            # System info
            f.write("## System Information\n\n")
            sys_info = results['system_info']
            f.write(f"- CPU Cores: {sys_info['cpu_count']}\n")
            f.write(f"- Memory: {sys_info['memory_total']:.1f} GB\n")
            f.write(f"- Platform: {sys_info.get('platform', 'unknown')}\n\n")
            
            # Simulation speed results
            if 'simulation_speed' in results:
                f.write("## Simulation Speed Benchmarks\n\n")
                for size_name, result in results['simulation_speed'].items():
                    if hasattr(result, 'success') and result.success:
                        f.write(f"- **{size_name}**: {result.steps_per_second:.1f} steps/second\n")
                        f.write(f"  - Execution time: {result.execution_time:.3f}s\n")
                        f.write(f"  - Memory peak: {result.memory_peak:.1f} MB\n")
                f.write("\n")
            
            # NetLogo comparison
            if 'netlogo_comparison' in results:
                f.write("## NetLogo Performance Comparison\n\n")
                for comparison in results['netlogo_comparison']:
                    status = "✅ PASS" if comparison.passes_benchmark else "❌ FAIL"
                    f.write(f"- **{comparison.metric_name}**: {status}\n")
                    f.write(f"  - BSTEW: {comparison.bstew_value:.1f}\n")
                    f.write(f"  - NetLogo Reference: {comparison.netlogo_reference:.1f}\n")
                    f.write(f"  - Performance Ratio: {comparison.performance_ratio:.2f}x\n")
                f.write("\n")
        
        self.logger.info(f"Benchmark report generated: {summary_file}")