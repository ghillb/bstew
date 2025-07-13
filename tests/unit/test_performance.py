"""
Unit tests for performance optimization functionality
===================================================

Comprehensive tests for performance monitoring, memory management,
caching, and simulation optimization.
"""

from unittest.mock import Mock, patch
import tempfile
import os

from src.bstew.utils.performance import (
    MemoryManager,
    CacheManager,
    ParallelProcessingManager,
    PerformanceProfiler,
    SimulationOptimizer,
    PerformanceMetrics,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality"""

    def test_performance_metrics_creation(self):
        """Test basic performance metrics creation"""
        metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_peak=80.0,
            simulation_time=120.0,
            steps_per_second=10.0,
            gc_collections=5,
            cache_hits=100,
            cache_misses=20,
        )

        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.memory_peak == 80.0
        assert metrics.simulation_time == 120.0
        assert metrics.steps_per_second == 10.0
        assert metrics.gc_collections == 5
        assert metrics.cache_hits == 100
        assert metrics.cache_misses == 20

    def test_performance_metrics_efficiency_score(self):
        """Test efficiency score calculation"""
        metrics = PerformanceMetrics(
            cpu_usage=20.0,
            memory_usage=30.0,
            memory_peak=40.0,
            simulation_time=60.0,
            steps_per_second=5.0,
            gc_collections=2,
            cache_hits=80,
            cache_misses=20,
        )

        assert hasattr(metrics, "efficiency_score")
        assert isinstance(metrics.efficiency_score, float)
        assert 0 <= metrics.efficiency_score <= 100


class TestMemoryManager:
    """Test MemoryManager functionality"""

    def test_memory_manager_creation(self):
        """Test basic memory manager creation"""
        manager = MemoryManager(memory_limit_gb=4.0)

        assert manager.memory_limit == 4.0 * 1024**3
        assert manager.gc_threshold == 0.8
        assert len(manager.memory_history) == 0
        assert isinstance(manager.object_pools, dict)

    @patch("src.bstew.utils.performance.psutil.Process")
    def test_memory_monitoring(self, mock_process):
        """Test memory monitoring"""
        # Mock process memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB
        mock_process.return_value.memory_info.return_value = mock_memory_info

        manager = MemoryManager(memory_limit_gb=4.0)
        memory_percent = manager.monitor_memory()

        assert isinstance(memory_percent, float)
        assert memory_percent >= 0
        assert len(manager.memory_history) == 1

    @patch("src.bstew.utils.performance.gc.collect")
    def test_garbage_collection(self, mock_gc_collect):
        """Test garbage collection"""
        mock_gc_collect.return_value = 10

        manager = MemoryManager()
        manager.force_garbage_collection()

        mock_gc_collect.assert_called_once()

    def test_object_pool_management(self):
        """Test object pool management"""
        manager = MemoryManager()

        # Create a simple constructor
        def create_object():
            return {"value": 42}

        # Get object from empty pool
        obj1 = manager.get_object_from_pool("test_type", create_object)
        assert obj1 == {"value": 42}

        # Return object to pool
        manager.return_to_pool(obj1, "test_type")
        assert len(manager.object_pools["test_type"]) == 1

        # Get object from pool
        obj2 = manager.get_object_from_pool("test_type", create_object)
        assert obj2 == obj1  # Should be same object
        assert len(manager.object_pools["test_type"]) == 0

    def test_dataframe_optimization(self):
        """Test DataFrame optimization"""
        import pandas as pd

        manager = MemoryManager()

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "string_col": ["a", "b", "c", "d", "e"],
            }
        )

        # Optimize DataFrame
        optimized_df = manager.optimize_dataframe(df)

        assert isinstance(optimized_df, pd.DataFrame)
        assert len(optimized_df) == 5
        assert list(optimized_df.columns) == ["int_col", "float_col", "string_col"]

    @patch("src.bstew.utils.performance.psutil.Process")
    def test_memory_report(self, mock_process):
        """Test memory report generation"""
        # Mock process memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB
        mock_memory_info.vms = 2 * 1024 * 1024 * 1024  # 2GB
        mock_memory_info.peak_wss = 1.5 * 1024 * 1024 * 1024  # 1.5GB
        mock_process.return_value.memory_info.return_value = mock_memory_info

        manager = MemoryManager(memory_limit_gb=4.0)
        report = manager.get_memory_report()

        assert isinstance(report, dict)
        assert "current_usage_mb" in report
        assert "virtual_memory_mb" in report
        assert "memory_percent" in report
        assert "gc_counts" in report
        assert "pool_sizes" in report
        assert "weak_refs_count" in report


class TestCacheManager:
    """Test CacheManager functionality"""

    def test_cache_manager_creation(self):
        """Test basic cache manager creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "cache")
            manager = CacheManager(max_memory_mb=100, cache_dir=cache_dir)

            assert manager.max_memory == 100 * 1024 * 1024
            assert manager.cache_dir.exists()
            assert len(manager.memory_cache) == 0
            assert manager.cache_stats == {"hits": 0, "misses": 0}

    def test_cache_set_get(self):
        """Test cache set and get operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "cache")
            manager = CacheManager(max_memory_mb=100, cache_dir=cache_dir)

            # Test set and get
            manager.set("key1", "value1")
            value = manager.get("key1")

            assert value == "value1"
            assert manager.cache_stats["hits"] == 1
            assert manager.cache_stats["misses"] == 0

    def test_cache_miss(self):
        """Test cache miss"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "cache")
            manager = CacheManager(max_memory_mb=100, cache_dir=cache_dir)

            # Test cache miss
            value = manager.get("nonexistent_key", "default")

            assert value == "default"
            assert manager.cache_stats["hits"] == 0
            assert manager.cache_stats["misses"] == 1

    def test_cache_clear(self):
        """Test cache clearing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "cache")
            manager = CacheManager(max_memory_mb=100, cache_dir=cache_dir)

            # Add some data
            manager.set("key1", "value1")
            manager.set("key2", "value2")

            # Clear cache
            manager.clear(memory_only=True)

            assert len(manager.memory_cache) == 0

    def test_cache_statistics(self):
        """Test cache statistics"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "cache")
            manager = CacheManager(max_memory_mb=100, cache_dir=cache_dir)

            # Perform some operations
            manager.set("key1", "value1")
            manager.get("key1")  # Hit
            manager.get("key2")  # Miss

            stats = manager.get_cache_stats()

            assert isinstance(stats, dict)
            assert "hit_rate" in stats
            assert "total_hits" in stats
            assert "total_misses" in stats
            assert "memory_cache_size_mb" in stats
            assert "memory_cache_items" in stats
            assert "memory_utilization" in stats

            assert stats["total_hits"] == 1
            assert stats["total_misses"] == 1
            assert stats["hit_rate"] == 0.5


class TestParallelProcessingManager:
    """Test ParallelProcessingManager functionality"""

    def test_parallel_manager_creation(self):
        """Test basic parallel manager creation"""
        manager = ParallelProcessingManager(max_workers=4)

        assert manager.max_workers == 4
        assert manager.thread_pool is not None
        assert manager.process_pool is None

    def test_parallel_agent_update(self):
        """Test parallel agent update"""
        manager = ParallelProcessingManager(max_workers=2)

        # Create mock agents
        agents = [Mock(id=i) for i in range(5)]

        # Define update function
        def update_agent(agent):
            agent.updated = True
            return agent

        # Update agents in parallel
        updated_agents = manager.parallel_agent_update(agents, update_agent)

        assert len(updated_agents) == 5
        assert all(hasattr(agent, "updated") for agent in updated_agents)

    def test_parallel_patch_processing(self):
        """Test parallel patch processing"""
        manager = ParallelProcessingManager(max_workers=2)

        # Create mock patches
        patches = [Mock(id=i) for i in range(3)]

        # Define process function
        def process_patch(patch):
            patch.processed = True
            return patch

        # Process patches in parallel
        processed_patches = manager.parallel_patch_processing(patches, process_patch)

        assert len(processed_patches) == 3
        assert all(hasattr(patch, "processed") for patch in processed_patches)

    def test_cleanup(self):
        """Test cleanup"""
        manager = ParallelProcessingManager(max_workers=2)

        # Should not raise any exceptions
        manager.cleanup()


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality"""

    def test_profiler_creation(self):
        """Test basic profiler creation"""
        profiler = PerformanceProfiler()

        assert profiler.profile_data is not None
        assert profiler.current_profiles is not None
        assert profiler.memory_manager is not None
        assert profiler.start_time is not None

    def test_profiler_start_end(self):
        """Test profiler start and end"""
        profiler = PerformanceProfiler()

        # Start profiling
        profiler.start_profile("test_section")
        assert "test_section" in profiler.current_profiles

        # End profiling
        profiler.end_profile("test_section")
        assert "test_section" not in profiler.current_profiles
        assert "test_section" in profiler.profile_data

    def test_profiler_function_decorator(self):
        """Test function profiling decorator"""
        profiler = PerformanceProfiler()

        @profiler.profile_function
        def test_function():
            return 42

        result = test_function()

        assert result == 42
        assert "test_function" in profiler.profile_data

    def test_performance_report(self):
        """Test performance report generation"""
        profiler = PerformanceProfiler()

        # Add some profile data
        profiler.start_profile("test_section")
        profiler.end_profile("test_section")

        report = profiler.get_performance_report()

        assert isinstance(report, dict)
        assert "total_runtime" in report
        assert "memory_report" in report
        assert "function_profiles" in report
        assert "bottlenecks" in report
        assert "recommendations" in report


class TestSimulationOptimizer:
    """Test SimulationOptimizer functionality"""

    def test_optimizer_creation(self):
        """Test basic optimizer creation"""
        config = {
            "memory_limit_gb": 4.0,
            "cache_memory_mb": 256,
            "max_workers": 2,
            "optimization_enabled": True,
        }

        optimizer = SimulationOptimizer(config)

        assert optimizer.config == config
        assert optimizer.memory_manager is not None
        assert optimizer.cache_manager is not None
        assert optimizer.parallel_manager is not None
        assert optimizer.profiler is not None
        assert optimizer.optimization_enabled

    def test_optimization_metrics(self):
        """Test optimization metrics"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Get metrics
        metrics = optimizer.get_optimization_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, "cpu_usage")
        assert hasattr(metrics, "memory_usage")
        assert hasattr(metrics, "cache_hits")
        assert hasattr(metrics, "cache_misses")

    def test_optimization_report(self):
        """Test optimization report generation"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Get report
        report = optimizer.get_optimization_report()

        assert isinstance(report, dict)
        assert "current_metrics" in report
        assert "performance_profile" in report
        assert "cache_statistics" in report
        assert "memory_statistics" in report
        assert "optimization_recommendations" in report

    def test_optimization_cleanup(self):
        """Test optimization cleanup"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Should not raise any exceptions
        optimizer.cleanup()

    def test_simulation_step_optimization(self):
        """Test simulation step optimization"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Create mock simulation state
        simulation_state = {"current_step": 10, "dataframes": {}}

        # Optimize step
        optimized_state = optimizer.optimize_simulation_step(simulation_state)

        assert isinstance(optimized_state, dict)
        assert "current_step" in optimized_state

    def test_optimization_disabled(self):
        """Test behavior when optimization is disabled"""
        config = {"optimization_enabled": False}
        optimizer = SimulationOptimizer(config)

        # Create mock simulation state
        simulation_state = {"current_step": 10, "dataframes": {}}

        # Should return unchanged state
        optimized_state = optimizer.optimize_simulation_step(simulation_state)

        assert optimized_state == simulation_state

    def test_adaptive_optimization(self):
        """Test adaptive optimization features"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Should have adaptive optimization capabilities
        assert hasattr(optimizer, "memory_manager")
        assert hasattr(optimizer, "cache_manager")
        assert hasattr(optimizer, "parallel_manager")

    def test_performance_benchmarking(self):
        """Test performance benchmarking"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Should be able to get performance metrics
        metrics = optimizer.get_optimization_metrics()
        assert isinstance(metrics, PerformanceMetrics)

    def test_resource_monitoring(self):
        """Test resource monitoring"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Should have resource monitoring capabilities
        assert optimizer.memory_manager is not None
        assert optimizer.cache_manager is not None

    def test_optimization_strategies(self):
        """Test different optimization strategies"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Should have different optimization components
        assert optimizer.memory_manager is not None
        assert optimizer.cache_manager is not None
        assert optimizer.parallel_manager is not None
        assert optimizer.profiler is not None

    def test_memory_optimization_integration(self):
        """Test integration with memory optimization"""
        config = {"optimization_enabled": True, "memory_limit_gb": 2.0}
        optimizer = SimulationOptimizer(config)

        # Should have memory optimization enabled
        assert optimizer.memory_manager is not None

    def test_cache_optimization_integration(self):
        """Test integration with cache optimization"""
        config = {"optimization_enabled": True, "cache_memory_mb": 128}
        optimizer = SimulationOptimizer(config)

        # Should have cache optimization enabled
        assert optimizer.cache_manager is not None

    def test_parallel_optimization_integration(self):
        """Test integration with parallel optimization"""
        config = {"optimization_enabled": True, "max_workers": 4}
        optimizer = SimulationOptimizer(config)

        # Should have parallel optimization enabled
        assert optimizer.parallel_manager is not None

    def test_profiling_integration(self):
        """Test integration with profiling"""
        config = {"optimization_enabled": True}
        optimizer = SimulationOptimizer(config)

        # Should have profiling enabled
        assert optimizer.profiler is not None
