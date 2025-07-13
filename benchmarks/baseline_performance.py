"""
Baseline Performance Benchmarking
=================================

Comprehensive baseline performance measurement for BSTEW simulation
to establish performance metrics before optimization integration.
"""

import time
import psutil
import gc
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path
from datetime import datetime
import cProfile
import pstats
import io
import tracemalloc


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    simulation_name: str
    agent_count: int
    step_count: int
    total_time: float
    steps_per_second: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    gc_collections: Dict[str, int]
    method_timings: Dict[str, float] = field(default_factory=dict)
    memory_allocations: Dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BaselineProfiler:
    """Comprehensive performance profiler for baseline measurements"""
    
    def __init__(self, enable_memory_profiling: bool = True):
        self.enable_memory_profiling = enable_memory_profiling
        self.results: List[PerformanceMetrics] = []
        
        # Profiling tools
        self.profiler: Optional[cProfile.Profile] = None
        self.process = psutil.Process()
        
        # Memory tracking
        self.memory_snapshots: List[Tuple[float, float]] = []
        
    def start_profiling(self, test_name: str) -> None:
        """Start comprehensive profiling"""
        print(f"Starting baseline profiling for: {test_name}")
        
        # Start CPU profiling
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        # Start memory tracking
        if self.enable_memory_profiling:
            tracemalloc.start()
        
        # Reset garbage collection stats
        gc.collect()
        self.gc_stats_start = {
            gen: gc.get_stats()[gen]['collections'] 
            for gen in range(len(gc.get_stats()))
        }
        
        # Start memory monitoring
        self.memory_snapshots = [(time.time(), self.process.memory_info().rss / 1024 / 1024)]
        self.start_time = time.time()
        
    def stop_profiling(self, test_name: str, agent_count: int, step_count: int) -> PerformanceMetrics:
        """Stop profiling and collect metrics"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Stop CPU profiling
        if self.profiler:
            self.profiler.disable()
        
        # Collect final memory usage
        final_memory = self.process.memory_info().rss / 1024 / 1024
        peak_memory = max(snapshot[1] for snapshot in self.memory_snapshots)
        
        # Collect garbage collection stats
        gc.collect()
        gc_stats_end = {
            gen: gc.get_stats()[gen]['collections'] 
            for gen in range(len(gc.get_stats()))
        }
        
        gc_collections = {
            f"gen_{gen}": gc_stats_end[gen] - self.gc_stats_start[gen]
            for gen in range(len(gc.get_stats()))
        }
        
        # Extract method timings from profiler
        method_timings = {}
        if self.profiler:
            method_timings = self._extract_method_timings()
        
        # Extract memory allocations
        memory_allocations = {}
        if self.enable_memory_profiling:
            memory_allocations = self._extract_memory_allocations()
            tracemalloc.stop()
        
        # Calculate derived metrics
        steps_per_second = step_count / max(total_time, 0.001)
        cpu_percent = self.process.cpu_percent()
        
        metrics = PerformanceMetrics(
            simulation_name=test_name,
            agent_count=agent_count,
            step_count=step_count,
            total_time=total_time,
            steps_per_second=steps_per_second,
            memory_usage_mb=final_memory,
            peak_memory_mb=peak_memory,
            cpu_percent=cpu_percent,
            gc_collections=gc_collections,
            method_timings=method_timings,
            memory_allocations=memory_allocations
        )
        
        self.results.append(metrics)
        print(f"Completed profiling for: {test_name}")
        print(f"  Steps/sec: {steps_per_second:.2f}")
        print(f"  Memory: {final_memory:.1f} MB (peak: {peak_memory:.1f} MB)")
        print(f"  GC collections: {sum(gc_collections.values())}")
        
        return metrics
    
    def _extract_method_timings(self) -> Dict[str, float]:
        """Extract method timing information from profiler"""
        if not self.profiler:
            return {}
        
        # Capture profiler output
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 methods
        
        # Parse method timings (simplified extraction)
        method_timings = {}
        lines = s.getvalue().split('\n')
        
        for line in lines:
            if 'bstew' in line and '(' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        cumtime = float(parts[3])
                        method_name = parts[-1]
                        method_timings[method_name] = cumtime
                    except (ValueError, IndexError):
                        continue
        
        return method_timings
    
    def _extract_memory_allocations(self) -> Dict[str, int]:
        """Extract memory allocation information"""
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('filename')
            
            allocations = {}
            for stat in top_stats[:10]:  # Top 10 allocators
                filename = stat.traceback.format()[-1] if stat.traceback.format() else "unknown"
                allocations[filename] = stat.size
            
            return allocations
        except Exception:
            return {}
    
    def monitor_memory(self) -> None:
        """Monitor memory usage during execution"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.memory_snapshots.append((time.time(), current_memory))
    
    def save_results(self, output_path: str) -> None:
        """Save profiling results to file"""
        results_data = []
        for metrics in self.results:
            result_dict = {
                "simulation_name": metrics.simulation_name,
                "agent_count": metrics.agent_count,
                "step_count": metrics.step_count,
                "total_time": metrics.total_time,
                "steps_per_second": metrics.steps_per_second,
                "memory_usage_mb": metrics.memory_usage_mb,
                "peak_memory_mb": metrics.peak_memory_mb,
                "cpu_percent": metrics.cpu_percent,
                "gc_collections": metrics.gc_collections,
                "method_timings": metrics.method_timings,
                "memory_allocations": metrics.memory_allocations,
                "timestamp": metrics.timestamp
            }
            results_data.append(result_dict)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Baseline performance results saved to: {output_path}")


class SimulationBenchmarkSuite:
    """Comprehensive simulation benchmark test suite"""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiler = BaselineProfiler()
        
        # Test scenarios
        self.test_scenarios = [
            {"name": "small_sim", "agents": 100, "steps": 100},
            {"name": "medium_sim", "agents": 500, "steps": 200},
            {"name": "large_sim", "agents": 1000, "steps": 100},
            {"name": "stress_test", "agents": 2000, "steps": 50},
        ]
    
    def run_baseline_benchmarks(self, iterations: int = 3) -> Dict[str, List[PerformanceMetrics]]:
        """Run comprehensive baseline benchmarks"""
        print("=" * 60)
        print("BSTEW BASELINE PERFORMANCE BENCHMARKING")
        print("=" * 60)
        
        all_results = {}
        
        for scenario in self.test_scenarios:
            scenario_name = scenario["name"]
            agent_count = scenario["agents"]
            step_count = scenario["steps"]
            
            print(f"\nRunning scenario: {scenario_name}")
            print(f"  Agents: {agent_count}, Steps: {step_count}")
            print(f"  Iterations: {iterations}")
            
            scenario_results = []
            
            for iteration in range(iterations):
                print(f"\n  Iteration {iteration + 1}/{iterations}")
                
                try:
                    metrics = self._run_single_benchmark(
                        f"{scenario_name}_iter_{iteration + 1}",
                        agent_count,
                        step_count
                    )
                    scenario_results.append(metrics)
                    
                except Exception as e:
                    print(f"    ERROR in iteration {iteration + 1}: {e}")
                    continue
            
            all_results[scenario_name] = scenario_results
            
            # Calculate statistics for this scenario
            if scenario_results:
                self._print_scenario_stats(scenario_name, scenario_results)
        
        # Save all results
        output_file = self.output_dir / "baseline_performance.json"
        self.profiler.save_results(str(output_file))
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _run_single_benchmark(self, test_name: str, agent_count: int, step_count: int) -> PerformanceMetrics:
        """Run a single benchmark test"""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.bstew.core.model import BeeModel
        
        # Start profiling
        self.profiler.start_profiling(test_name)
        
        try:
            # Create minimal configuration for testing
            config = {
                "simulation": {
                    "duration_days": step_count,
                    "random_seed": 42
                },
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": max(1, agent_count - 10),
                        "foragers": min(10, agent_count // 10),
                        "drones": 0,
                        "brood": 0
                    },
                    "species": "apis_mellifera",
                    "location": [0, 0],
                    "colony_strength": 0.8,
                    "genetic_diversity": 0.7
                },
                "environment": {
                    "landscape_width": 50,
                    "landscape_height": 50,
                    "cell_size": 10.0,
                    "weather_variation": 0.1,
                    "seasonal_effects": False
                },
                "foraging": {
                    "max_foraging_range": 500.0,
                    "dance_threshold": 0.7,
                    "recruitment_efficiency": 0.8,
                    "energy_cost_per_meter": 0.01
                },
                "disease": {
                    "enable_varroa": False,
                    "enable_viruses": False,
                    "enable_nosema": False,
                    "natural_resistance": 0.8
                },
                "output": {
                    "output_directory": "benchmark_output",
                    "log_level": "ERROR",  # Reduce logging during benchmarks
                    "save_plots": False,
                    "save_csv": False,
                    "save_spatial_data": False,
                    "compress_output": False
                }
            }
            
            # Create and run simulation
            model = BeeModel(config=config)
            
            # Run simulation steps with periodic memory monitoring
            for step in range(step_count):
                model.step()
                
                # Monitor memory every 10 steps
                if step % 10 == 0:
                    self.profiler.monitor_memory()
            
            # Final memory monitoring
            self.profiler.monitor_memory()
            
        except Exception as e:
            print(f"    Simulation error: {e}")
            raise
        
        finally:
            # Stop profiling and collect metrics
            metrics = self.profiler.stop_profiling(test_name, agent_count, step_count)
            
            # Cleanup
            try:
                del model
                gc.collect()
            except Exception:
                pass
        
        return metrics
    
    def _print_scenario_stats(self, scenario_name: str, results: List[PerformanceMetrics]) -> None:
        """Print statistical summary for a scenario"""
        if not results:
            return
        
        steps_per_sec = [r.steps_per_second for r in results]
        memory_usage = [r.memory_usage_mb for r in results]
        total_times = [r.total_time for r in results]
        
        print(f"\n  Results for {scenario_name}:")
        print(f"    Steps/sec: {statistics.mean(steps_per_sec):.2f} ± {statistics.stdev(steps_per_sec) if len(steps_per_sec) > 1 else 0:.2f}")
        print(f"    Memory: {statistics.mean(memory_usage):.1f} ± {statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0:.1f} MB")
        print(f"    Time: {statistics.mean(total_times):.2f} ± {statistics.stdev(total_times) if len(total_times) > 1 else 0:.2f} sec")
    
    def _generate_summary_report(self, all_results: Dict[str, List[PerformanceMetrics]]) -> None:
        """Generate comprehensive summary report"""
        summary_file = self.output_dir / "baseline_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("BSTEW BASELINE PERFORMANCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for scenario_name, results in all_results.items():
                if not results:
                    continue
                
                f.write(f"Scenario: {scenario_name}\n")
                f.write("-" * 30 + "\n")
                
                # Calculate statistics
                steps_per_sec = [r.steps_per_second for r in results]
                memory_usage = [r.memory_usage_mb for r in results]
                
                if steps_per_sec:
                    f.write(f"Steps/sec: {statistics.mean(steps_per_sec):.2f} ± {statistics.stdev(steps_per_sec) if len(steps_per_sec) > 1 else 0:.2f}\n")
                    f.write(f"Memory: {statistics.mean(memory_usage):.1f} ± {statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0:.1f} MB\n")
                    f.write(f"Agent count: {results[0].agent_count}\n")
                    f.write(f"Step count: {results[0].step_count}\n")
                
                f.write("\n")
        
        print(f"\nBaseline summary report saved to: {summary_file}")


def run_baseline_performance_measurement():
    """Main function to run baseline performance measurement"""
    benchmark_suite = SimulationBenchmarkSuite()
    results = benchmark_suite.run_baseline_benchmarks(iterations=3)
    
    print("\n" + "=" * 60)
    print("BASELINE PERFORMANCE MEASUREMENT COMPLETE")
    print("=" * 60)
    print("\nBaseline metrics established for:")
    for scenario_name in results.keys():
        print(f"  ✓ {scenario_name}")
    
    print(f"\nResults saved to: {benchmark_suite.output_dir}")
    print("Ready for optimization integration and comparison!")
    
    return results


if __name__ == "__main__":
    run_baseline_performance_measurement()