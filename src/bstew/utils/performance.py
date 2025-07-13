"""
Performance optimization utilities for BSTEW
============================================

Implements memory management, parallel processing, caching,
and performance monitoring for large-scale simulations.
"""

import numpy as np
import pandas as pd
import time
import psutil
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Callable, Optional
from pydantic import BaseModel, Field, computed_field
from collections import defaultdict, deque
import logging
import pickle
import sqlite3
from pathlib import Path
import functools
import weakref


class PerformanceMetrics(BaseModel):
    """Performance monitoring metrics"""

    model_config = {"validate_assignment": True}

    cpu_usage: float = Field(ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(ge=0.0, description="Memory usage percentage")
    memory_peak: float = Field(ge=0.0, description="Peak memory usage in MB")
    simulation_time: float = Field(
        ge=0.0, description="Total simulation time in seconds"
    )
    steps_per_second: float = Field(ge=0.0, description="Simulation steps per second")
    gc_collections: int = Field(ge=0, description="Number of garbage collections")
    cache_hits: int = Field(ge=0, description="Number of cache hits")
    cache_misses: int = Field(ge=0, description="Number of cache misses")
    efficiency_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall efficiency score"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate efficiency score after model creation"""
        self.efficiency_score = self._calculate_efficiency()

    def _calculate_efficiency(self) -> float:
        """Calculate overall efficiency score"""
        # Combine multiple metrics into efficiency score (0-100)
        cpu_score = max(0, 100 - self.cpu_usage)
        memory_score = max(0, 100 - self.memory_usage)
        speed_score = min(100, self.steps_per_second * 10)  # Normalize
        cache_score = (
            self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        ) * 100

        return (cpu_score + memory_score + speed_score + cache_score) / 4


class CacheStatistics(BaseModel):
    """Cache performance statistics with validation"""

    model_config = {"validate_assignment": True}

    hits: int = Field(default=0, ge=0, description="Number of cache hits")
    misses: int = Field(default=0, ge=0, description="Number of cache misses")
    memory_cache_size_mb: float = Field(
        default=0.0, ge=0.0, description="Memory cache size in MB"
    )
    memory_cache_items: int = Field(
        default=0, ge=0, description="Number of items in memory cache"
    )
    memory_utilization: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Memory utilization ratio"
    )

    @computed_field
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MemoryReport(BaseModel):
    """Memory usage report with validation"""

    model_config = {"validate_assignment": True}

    current_usage_mb: float = Field(ge=0.0, description="Current memory usage in MB")
    peak_usage_mb: float = Field(ge=0.0, description="Peak memory usage in MB")
    virtual_memory_mb: float = Field(ge=0.0, description="Virtual memory usage in MB")
    memory_percent: float = Field(ge=0.0, description="Memory usage percentage")
    gc_counts: tuple = Field(description="Garbage collection counts")
    pool_sizes: Dict[str, int] = Field(
        default_factory=dict, description="Object pool sizes"
    )
    weak_refs_count: int = Field(ge=0, description="Number of weak references")


class PerformanceConfig(BaseModel):
    """Performance optimization configuration with validation"""

    model_config = {"validate_assignment": True}

    memory_limit_gb: float = Field(
        default=8.0, gt=0.0, le=256.0, description="Memory limit in GB"
    )
    cache_memory_mb: float = Field(
        default=512.0, gt=0.0, description="Cache memory limit in MB"
    )
    max_workers: int = Field(
        default=8, ge=1, le=32, description="Maximum worker threads"
    )
    gc_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="GC trigger threshold"
    )
    optimization_enabled: bool = Field(
        default=True, description="Enable optimization features"
    )
    cache_dir: str = Field(default="artifacts/cache", description="Cache directory path")


class MemoryManager:
    """
    Memory management and optimization system.

    Handles:
    - Memory usage monitoring
    - Automatic garbage collection
    - Object pooling
    - Memory-efficient data structures
    """

    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit = memory_limit_gb * 1024**3  # Convert to bytes
        self.memory_history: deque = deque(maxlen=100)
        self.gc_threshold = 0.8  # Trigger GC at 80% memory usage
        self.object_pools: Dict[str, List[Any]] = {}
        self.weak_refs: weakref.WeakSet = weakref.WeakSet()

        self.logger = logging.getLogger(__name__)

    def monitor_memory(self) -> float:
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss
        memory_percent = (memory_usage / self.memory_limit) * 100

        self.memory_history.append(
            {
                "timestamp": time.time(),
                "usage_bytes": memory_usage,
                "usage_percent": memory_percent,
            }
        )

        # Trigger garbage collection if needed
        if memory_percent > self.gc_threshold * 100:
            self.force_garbage_collection()

        return float(memory_percent)

    def force_garbage_collection(self) -> None:
        """Force garbage collection and cleanup"""
        collected = gc.collect()
        self.logger.info(f"Garbage collection freed {collected} objects")

        # Clear weakly referenced objects
        for obj in list(self.weak_refs):
            if hasattr(obj, "cleanup"):
                obj.cleanup()

    def get_object_from_pool(self, obj_type: str, constructor: Callable) -> Any:
        """Get object from pool or create new one"""
        if obj_type not in self.object_pools:
            self.object_pools[obj_type] = []

        pool = self.object_pools[obj_type]

        if pool:
            obj = pool.pop()
            if hasattr(obj, "reset"):
                obj.reset()
            return obj
        else:
            return constructor()

    def return_to_pool(self, obj: Any, obj_type: str) -> None:
        """Return object to pool for reuse"""
        if obj_type not in self.object_pools:
            self.object_pools[obj_type] = []

        # Limit pool size to prevent memory bloat
        if len(self.object_pools[obj_type]) < 100:
            self.object_pools[obj_type].append(obj)

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        for col in df.columns:
            col_type = df[col].dtype

            if col_type != "object":
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)

                elif str(col_type)[:5] == "float":
                    if (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)

        return df

    def get_memory_report(self) -> Dict[str, Any]:
        """Get detailed memory usage report"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "current_usage_mb": memory_info.rss / (1024**2),
            "peak_usage_mb": (
                memory_info.peak_wss / (1024**2)
                if hasattr(memory_info, "peak_wss")
                else 0
            ),
            "virtual_memory_mb": memory_info.vms / (1024**2),
            "memory_percent": (memory_info.rss / self.memory_limit) * 100,
            "gc_counts": gc.get_count(),
            "pool_sizes": {k: len(v) for k, v in self.object_pools.items()},
            "weak_refs_count": len(self.weak_refs),
        }


class CacheManager:
    """
    Intelligent caching system for simulation results.

    Implements:
    - LRU cache for frequently accessed data
    - Persistent cache storage
    - Cache invalidation strategies
    - Memory-aware cache sizing
    """

    def __init__(self, max_memory_mb: float = 512, cache_dir: str = "cache"):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.memory_cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_sizes: Dict[str, int] = {}

        # Persistent cache database
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize persistent cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER
                )
            """
            )

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get item from cache"""
        # Check memory cache first
        if key in self.memory_cache:
            self.cache_stats["hits"] += 1
            self.cache_access_times[key] = time.time()
            return self.memory_cache[key]

        # Check persistent cache
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data FROM cache_entries WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

            if row:
                self.cache_stats["hits"] += 1
                data = pickle.loads(row[0])

                # Promote to memory cache if space available
                data_size = len(row[0])
                if self._get_memory_cache_size() + data_size < self.max_memory:
                    self.memory_cache[key] = data
                    self.cache_access_times[key] = time.time()
                    self.cache_sizes[key] = data_size

                # Update access count
                conn.execute(
                    "UPDATE cache_entries SET access_count = access_count + 1 WHERE key = ?",
                    (key,),
                )

                return data

        self.cache_stats["misses"] += 1
        return default

    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Set item in cache"""
        # Serialize data
        data_bytes = pickle.dumps(value)
        data_size = len(data_bytes)

        # Add to memory cache if space available
        if self._get_memory_cache_size() + data_size < self.max_memory:
            self.memory_cache[key] = value
            self.cache_access_times[key] = time.time()
            self.cache_sizes[key] = data_size
        else:
            # Evict least recently used items
            self._evict_lru_items(data_size)
            self.memory_cache[key] = value
            self.cache_access_times[key] = time.time()
            self.cache_sizes[key] = data_size

        # Store in persistent cache if requested
        if persist:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries 
                    (key, data, timestamp, size_bytes) 
                    VALUES (?, ?, ?, ?)
                """,
                    (key, data_bytes, time.time(), data_size),
                )

    def _get_memory_cache_size(self) -> int:
        """Get current memory cache size in bytes"""
        return sum(self.cache_sizes.values())

    def _evict_lru_items(self, space_needed: int) -> None:
        """Evict least recently used items to make space"""
        # Sort by access time
        sorted_items = sorted(self.cache_access_times.items(), key=lambda x: x[1])

        space_freed = 0
        for key, _ in sorted_items:
            if space_freed >= space_needed:
                break

            if key in self.memory_cache:
                space_freed += self.cache_sizes.get(key, 0)
                del self.memory_cache[key]
                del self.cache_access_times[key]
                del self.cache_sizes[key]

    def clear(self, memory_only: bool = False) -> None:
        """Clear cache"""
        self.memory_cache.clear()
        self.cache_access_times.clear()
        self.cache_sizes.clear()

        if not memory_only:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.cache_stats["hits"] / max(1, sum(self.cache_stats.values()))

        return {
            "hit_rate": hit_rate,
            "total_hits": self.cache_stats["hits"],
            "total_misses": self.cache_stats["misses"],
            "memory_cache_size_mb": self._get_memory_cache_size() / (1024 * 1024),
            "memory_cache_items": len(self.memory_cache),
            "memory_utilization": self._get_memory_cache_size() / self.max_memory,
        }


def memoize_with_ttl(ttl_seconds: int = 300) -> Callable:
    """Decorator for memoizing function results with TTL"""

    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Any] = {}
        cache_times: Dict[str, float] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()

            # Check if cached result exists and is still valid
            if key in cache and (current_time - cache_times[key]) < ttl_seconds:
                return cache[key]

            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = current_time

            # Clean old entries periodically
            if len(cache) > 1000:
                old_keys = [
                    k
                    for k, t in cache_times.items()
                    if (current_time - t) >= ttl_seconds
                ]
                for k in old_keys:
                    cache.pop(k, None)
                    cache_times.pop(k, None)

            return result

        return wrapper

    return decorator


class ParallelProcessingManager:
    """
    Parallel processing utilities for BSTEW simulations.

    Handles:
    - Multi-threaded agent updates
    - Parallel patch processing
    - Load balancing
    - Thread safety
    """

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool: Optional[ProcessPoolExecutor] = None  # Created on demand

    def parallel_agent_update(
        self, agents: List[Any], update_func: Callable
    ) -> List[Any]:
        """Update agents in parallel using threads"""

        def update_chunk(agent_chunk: List[Any]) -> List[Any]:
            return [update_func(agent) for agent in agent_chunk]

        # Split agents into chunks
        chunk_size = max(1, len(agents) // self.max_workers)
        chunks = [agents[i : i + chunk_size] for i in range(0, len(agents), chunk_size)]

        # Process chunks in parallel
        futures = [self.thread_pool.submit(update_chunk, chunk) for chunk in chunks]

        # Collect results
        updated_agents = []
        for future in futures:
            updated_agents.extend(future.result())

        return updated_agents

    def parallel_patch_processing(
        self, patches: List[Any], process_func: Callable
    ) -> List[Any]:
        """Process patches in parallel"""

        # Use thread pool for I/O bound patch operations
        futures = [self.thread_pool.submit(process_func, patch) for patch in patches]
        return [future.result() for future in futures]

    def parallel_simulation_runs(
        self, run_configs: List[Dict], simulation_func: Callable
    ) -> List[Any]:
        """Run multiple simulations in parallel using processes"""

        if not self.process_pool:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

        futures = [
            self.process_pool.submit(simulation_func, config) for config in run_configs
        ]
        return [future.result() for future in futures]

    def cleanup(self) -> None:
        """Clean up thread and process pools"""
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class PerformanceProfiler:
    """
    Performance profiling and optimization recommendations.

    Monitors:
    - Function execution times
    - Memory allocations
    - CPU usage patterns
    - Bottleneck identification
    """

    def __init__(self) -> None:
        self.profile_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.current_profiles: Dict[str, Dict[str, Any]] = {}
        self.memory_manager = MemoryManager()
        self.start_time = time.time()

    def start_profile(self, name: str) -> None:
        """Start profiling a section"""
        self.current_profiles[name] = {
            "start_time": time.time(),
            "start_memory": self.memory_manager.monitor_memory(),
        }

    def end_profile(self, name: str) -> None:
        """End profiling a section"""
        if name not in self.current_profiles:
            return

        profile_data = self.current_profiles[name]
        end_time = time.time()
        end_memory = self.memory_manager.monitor_memory()

        duration = end_time - profile_data["start_time"]
        memory_delta = end_memory - profile_data["start_memory"]

        self.profile_data[name].append(
            {"duration": duration, "memory_delta": memory_delta, "timestamp": end_time}
        )

        del self.current_profiles[name]

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            self.start_profile(func_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.end_profile(func_name)

        return wrapper

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance analysis report"""

        report: Dict[str, Any] = {
            "total_runtime": time.time() - self.start_time,
            "memory_report": self.memory_manager.get_memory_report(),
            "function_profiles": {},
            "bottlenecks": [],
            "recommendations": [],
        }

        # Analyze function profiles
        for func_name, profiles in self.profile_data.items():
            if not profiles:
                continue

            durations = [p["duration"] for p in profiles]
            memory_deltas = [p["memory_delta"] for p in profiles]

            func_stats = {
                "call_count": len(profiles),
                "total_time": sum(durations),
                "avg_time": np.mean(durations),
                "max_time": max(durations),
                "min_time": min(durations),
                "avg_memory_delta": np.mean(memory_deltas),
                "total_memory_delta": sum(memory_deltas),
            }

            report["function_profiles"][func_name] = func_stats

            # Identify bottlenecks
            total_runtime = float(report["total_runtime"])
            if func_stats["total_time"] > total_runtime * 0.1:  # >10% of total time
                if isinstance(report["bottlenecks"], list):
                    report["bottlenecks"].append(
                        {
                            "function": func_name,
                            "impact": func_stats["total_time"] / total_runtime,
                            "avg_time": func_stats["avg_time"],
                        }
                    )

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Memory usage recommendations
        memory_usage = report["memory_report"]["memory_percent"]
        if memory_usage > 80:
            recommendations.append(
                "High memory usage detected. Consider reducing data retention or using data streaming."
            )

        # Function-specific recommendations
        for bottleneck in report["bottlenecks"]:
            func_name = bottleneck["function"]
            if bottleneck["impact"] > 0.3:
                recommendations.append(
                    f"Function '{func_name}' is a major bottleneck ({bottleneck['impact']:.1%} of runtime). Consider optimization or parallelization."
                )

        # General recommendations
        if len(report["bottlenecks"]) > 3:
            recommendations.append(
                "Multiple performance bottlenecks detected. Consider code profiling and optimization."
            )

        return recommendations
    
    def profile_simulation(self, config: Dict[str, Any], steps: int, 
                          output_directory: str) -> Dict[str, Any]:
        """Profile a complete simulation run"""
        import psutil
        from pathlib import Path
        
        # Ensure output directory exists
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Start overall profiling
        self.start_profile("simulation_complete")
        
        # Monitor initial state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from ..core.model import BeeModel
            
            # Profile model initialization
            self.start_profile("model_initialization")
            model = BeeModel(config=config)
            self.end_profile("model_initialization")
            
            # Profile simulation steps
            self.start_profile("simulation_steps")
            for step in range(steps):
                step_start = time.time()
                model.step()
                step_time = time.time() - step_start
                
                # Log step performance data
                self.profile_data["step_performance"].append({
                    "step": step,
                    "duration": step_time,
                    "memory": process.memory_info().rss / 1024 / 1024,
                    "timestamp": time.time()
                })
            
            self.end_profile("simulation_steps")
            
            # Profile cleanup
            self.start_profile("model_cleanup")
            model.cleanup()
            self.end_profile("model_cleanup")
            
        except Exception as e:
            # Still end profiling even if simulation fails
            self.end_profile("simulation_complete")
            raise e
        
        # End overall profiling
        self.end_profile("simulation_complete")
        
        # Calculate final metrics
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024
        total_time = end_time - start_time
        
        # Generate comprehensive results
        performance_report = self.get_performance_report()
        
        results = {
            "performance_metrics": {
                "simulation_time": total_time,
                "steps_per_second": steps / total_time if total_time > 0 else 0,
                "memory_peak": final_memory,
                "memory_usage": final_memory - initial_memory,
                "cpu_usage": process.cpu_percent(),
                "efficiency_score": self._calculate_efficiency_score(steps, total_time, final_memory)
            },
            "function_timings": performance_report.get("function_profiles", {}),
            "bottlenecks": performance_report.get("bottlenecks", []),
            "recommendations": performance_report.get("recommendations", []),
            "step_performance": self.profile_data.get("step_performance", [])
        }
        
        # Save results to file
        import json
        output_file = Path(output_directory) / "profiling_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _calculate_efficiency_score(self, steps: int, total_time: float, memory_mb: float) -> float:
        """Calculate an efficiency score based on performance metrics"""
        # Normalize metrics (these are rough estimates for scoring)
        time_score = min(1.0, 1000 / (total_time * 1000 / steps))  # Prefer faster step execution
        memory_score = min(1.0, 500 / memory_mb)  # Prefer lower memory usage
        
        # Weighted average (time is more important than memory)
        efficiency = (time_score * 0.7) + (memory_score * 0.3)
        return efficiency


class SimulationOptimizer:
    """
    High-level simulation optimization coordinator.

    Integrates all performance optimization components.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_manager = MemoryManager(
            memory_limit_gb=config.get("memory_limit_gb", 8.0)
        )
        self.cache_manager = CacheManager(
            max_memory_mb=config.get("cache_memory_mb", 512),
            cache_dir=config.get("cache_dir", "cache"),
        )
        self.parallel_manager = ParallelProcessingManager(
            max_workers=config.get("max_workers", None)
        )
        self.profiler = PerformanceProfiler()

        # Performance monitoring
        self.performance_history: deque = deque(maxlen=1000)
        self.optimization_enabled = config.get("optimization_enabled", True)

    def optimize_simulation_step(
        self, simulation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize a single simulation step"""

        if not self.optimization_enabled:
            return simulation_state

        self.profiler.start_profile("simulation_step")

        try:
            # Memory management
            self.memory_manager.monitor_memory()

            # Cache frequently accessed data
            step_key = f"step_{simulation_state.get('current_step', 0)}"
            cached_result = self.cache_manager.get(step_key)

            if cached_result is not None:
                return dict(cached_result)

            # Optimize data structures
            if "dataframes" in simulation_state:
                for key, df in simulation_state["dataframes"].items():
                    simulation_state["dataframes"][key] = (
                        self.memory_manager.optimize_dataframe(df)
                    )

            # Store result in cache
            self.cache_manager.set(step_key, simulation_state)

            return simulation_state

        finally:
            self.profiler.end_profile("simulation_step")

    def get_optimization_metrics(self) -> PerformanceMetrics:
        """Get current optimization metrics"""

        memory_report = self.memory_manager.get_memory_report()
        cache_stats = self.cache_manager.get_cache_stats()

        # Calculate metrics
        process = psutil.Process()
        cpu_usage = process.cpu_percent()

        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_report["memory_percent"],
            memory_peak=memory_report["peak_usage_mb"],
            simulation_time=time.time() - self.profiler.start_time,
            steps_per_second=len(self.performance_history)
            / max(1, time.time() - self.profiler.start_time),
            gc_collections=sum(memory_report["gc_counts"]),
            cache_hits=cache_stats["total_hits"],
            cache_misses=cache_stats["total_misses"],
        )

    def cleanup(self) -> None:
        """Clean up optimization resources"""
        self.parallel_manager.cleanup()
        self.cache_manager.clear(memory_only=True)
        self.memory_manager.force_garbage_collection()

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""

        performance_report = self.profiler.get_performance_report()
        cache_stats = self.cache_manager.get_cache_stats()
        memory_report = self.memory_manager.get_memory_report()
        current_metrics = self.get_optimization_metrics()

        return {
            "current_metrics": current_metrics.__dict__,
            "performance_profile": performance_report,
            "cache_statistics": cache_stats,
            "memory_statistics": memory_report,
            "optimization_recommendations": self._get_optimization_recommendations(
                current_metrics
            ),
        }

    def _get_optimization_recommendations(
        self, metrics: PerformanceMetrics
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if metrics.memory_usage > 80:
            recommendations.append(
                "Consider reducing memory usage or increasing memory limit"
            )

        if metrics.cpu_usage > 90:
            recommendations.append(
                "High CPU usage detected. Consider reducing simulation complexity or using more workers"
            )

        if metrics.steps_per_second < 1:
            recommendations.append(
                "Low simulation speed. Enable parallelization or optimize bottlenecks"
            )

        cache_hit_rate = metrics.cache_hits / max(
            1, metrics.cache_hits + metrics.cache_misses
        )
        if cache_hit_rate < 0.3:
            recommendations.append(
                "Low cache hit rate. Consider increasing cache size or improving cache strategies"
            )

        if metrics.efficiency_score < 50:
            recommendations.append(
                "Overall efficiency is low. Review performance profile for optimization opportunities"
            )

        return recommendations
