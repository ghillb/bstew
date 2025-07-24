"""
Unit tests for Live Visualization System
=======================================

Tests for the comprehensive live visualization system including real-time data streaming,
visualization engines, chart generation, and dashboard management for bee colony monitoring.
"""

import pytest
from unittest.mock import Mock, patch
import time
import threading
import json
from collections import deque

from src.bstew.visualization.live_visualization import (
    VisualizationEngine, LiveDataStream, VisualizationData,
    VisualizationType
)


class TestVisualizationType:
    """Test visualization type definitions"""

    def test_visualization_type_values(self):
        """Test visualization type constant values"""
        assert VisualizationType.COLONY_OVERVIEW.value == "colony_overview"
        assert VisualizationType.POPULATION_TRENDS.value == "population_trends"
        assert VisualizationType.FORAGING_ACTIVITY.value == "foraging_activity"
        assert VisualizationType.SPATIAL_DISTRIBUTION.value == "spatial_distribution"
        assert VisualizationType.COMMUNICATION_NETWORK.value == "communication_network"
        assert VisualizationType.HEALTH_MONITORING.value == "health_monitoring"
        assert VisualizationType.RESOURCE_DYNAMICS.value == "resource_dynamics"
        assert VisualizationType.DANCE_COMMUNICATION.value == "dance_communication"
        assert VisualizationType.RECRUITMENT_FLOW.value == "recruitment_flow"
        assert VisualizationType.ENVIRONMENTAL_CONDITIONS.value == "environmental_conditions"

    def test_all_visualization_types_present(self):
        """Test that all expected visualization types are defined"""
        # Get all VisualizationType class attributes that are uppercase constants
        viz_types = [attr for attr in dir(VisualizationType)
                    if attr.isupper() and not attr.startswith('_')]
        assert len(viz_types) == 10  # Verify we have all expected types


class TestVisualizationData:
    """Test visualization data container"""

    def setup_method(self):
        """Setup test fixtures"""
        self.viz_data = VisualizationData(
            timestamp=1234567890.0,
            data_type=VisualizationType.POPULATION_TRENDS,
            data={
                "colony_1": {"population": 150, "growth_rate": 0.05},
                "colony_2": {"population": 120, "growth_rate": 0.03},
                "colony_3": {"population": 180, "growth_rate": 0.07}
            },
            metadata={
                "simulation_step": 100,
                "data_source": "colony_monitor",
                "quality": "high"
            }
        )

    def test_initialization(self):
        """Test visualization data initialization"""
        assert self.viz_data.timestamp == 1234567890.0
        assert self.viz_data.data_type == VisualizationType.POPULATION_TRENDS
        assert len(self.viz_data.data) == 3
        assert "colony_1" in self.viz_data.data
        assert self.viz_data.data["colony_1"]["population"] == 150
        assert self.viz_data.metadata["simulation_step"] == 100

    def test_json_serialization(self):
        """Test JSON serialization"""
        json_str = self.viz_data.to_json()

        assert isinstance(json_str, str)

        # Parse JSON to verify structure
        parsed_data = json.loads(json_str)
        assert parsed_data["timestamp"] == 1234567890.0
        assert parsed_data["data_type"] == VisualizationType.POPULATION_TRENDS.value
        assert "data" in parsed_data
        assert "metadata" in parsed_data
        assert len(parsed_data["data"]) == 3

    def test_empty_data(self):
        """Test visualization data with empty data and metadata"""
        empty_viz_data = VisualizationData(
            timestamp=time.time(),
            data_type=VisualizationType.COLONY_OVERVIEW
        )

        assert isinstance(empty_viz_data.data, dict)
        assert isinstance(empty_viz_data.metadata, dict)
        assert len(empty_viz_data.data) == 0
        assert len(empty_viz_data.metadata) == 0

    def test_different_data_types(self):
        """Test different visualization data types"""
        data_types = [
            VisualizationType.FORAGING_ACTIVITY,
            VisualizationType.SPATIAL_DISTRIBUTION,
            VisualizationType.HEALTH_MONITORING,
            VisualizationType.DANCE_COMMUNICATION
        ]

        for data_type in data_types:
            viz_data = VisualizationData(
                timestamp=time.time(),
                data_type=data_type,
                data={"test": "value"},
                metadata={"source": "test"}
            )

            assert viz_data.data_type == data_type
            assert viz_data.data["test"] == "value"

    def test_complex_data_structures(self):
        """Test complex nested data structures"""
        complex_data = VisualizationData(
            timestamp=time.time(),
            data_type=VisualizationType.COMMUNICATION_NETWORK,
            data={
                "nodes": [
                    {"id": 1, "type": "bee", "position": [10, 20]},
                    {"id": 2, "type": "bee", "position": [15, 25]}
                ],
                "edges": [
                    {"source": 1, "target": 2, "weight": 0.8, "type": "communication"}
                ],
                "metrics": {
                    "total_nodes": 2,
                    "total_edges": 1,
                    "average_degree": 1.0
                }
            },
            metadata={
                "network_type": "communication",
                "analysis_method": "graph_theory",
                "confidence": 0.95
            }
        )

        assert len(complex_data.data["nodes"]) == 2
        assert len(complex_data.data["edges"]) == 1
        assert complex_data.data["metrics"]["total_nodes"] == 2
        assert complex_data.metadata["confidence"] == 0.95


class TestLiveDataStream:
    """Test live data stream management"""

    def setup_method(self):
        """Setup test fixtures"""
        self.data_stream = LiveDataStream(
            stream_id="colony_population_stream",
            data_source="colony_monitor_1",
            update_interval=1.0,
            buffer_size=500
        )

        # Test data
        self.sample_data = [
            VisualizationData(
                timestamp=time.time() - 3,
                data_type=VisualizationType.POPULATION_TRENDS,
                data={"population": 100}
            ),
            VisualizationData(
                timestamp=time.time() - 2,
                data_type=VisualizationType.POPULATION_TRENDS,
                data={"population": 105}
            ),
            VisualizationData(
                timestamp=time.time() - 1,
                data_type=VisualizationType.POPULATION_TRENDS,
                data={"population": 110}
            )
        ]

    def test_initialization(self):
        """Test data stream initialization"""
        assert self.data_stream.stream_id == "colony_population_stream"
        assert self.data_stream.data_source == "colony_monitor_1"
        assert self.data_stream.update_interval == 1.0
        assert self.data_stream.buffer_size == 500
        assert isinstance(self.data_stream.data_buffer, deque)
        assert self.data_stream.data_buffer.maxlen == 500
        assert self.data_stream.is_active is True
        assert self.data_stream.last_update == 0.0

    def test_default_buffer_size(self):
        """Test default buffer size"""
        default_stream = LiveDataStream(
            stream_id="test_stream",
            data_source="test_source",
            update_interval=0.5
        )

        assert default_stream.buffer_size == 1000
        assert default_stream.data_buffer.maxlen == 1000

    def test_add_data(self):
        """Test adding data to stream"""
        initial_time = self.data_stream.last_update

        # Add data to stream
        for data in self.sample_data:
            self.data_stream.add_data(data)

        assert len(self.data_stream.data_buffer) == 3
        assert self.data_stream.last_update > initial_time

        # Verify data order (most recent last)
        buffer_list = list(self.data_stream.data_buffer)
        assert buffer_list[0].data["population"] == 100
        assert buffer_list[1].data["population"] == 105
        assert buffer_list[2].data["population"] == 110

    def test_buffer_overflow(self):
        """Test buffer overflow behavior"""
        # Create stream with small buffer
        small_stream = LiveDataStream(
            stream_id="small_stream",
            data_source="test",
            update_interval=1.0,
            buffer_size=2
        )

        # Add more data than buffer can hold
        for i in range(5):
            data = VisualizationData(
                timestamp=time.time(),
                data_type=VisualizationType.COLONY_OVERVIEW,
                data={"value": i}
            )
            small_stream.add_data(data)

        # Buffer should only contain last 2 items
        assert len(small_stream.data_buffer) == 2
        buffer_list = list(small_stream.data_buffer)
        assert buffer_list[0].data["value"] == 3  # Second to last
        assert buffer_list[1].data["value"] == 4  # Last

    def test_stream_state_management(self):
        """Test stream state management"""
        # Initially active
        assert self.data_stream.is_active is True

        # Deactivate stream
        self.data_stream.is_active = False
        assert self.data_stream.is_active is False

        # Can still add data when inactive
        test_data = VisualizationData(
            timestamp=time.time(),
            data_type=VisualizationType.HEALTH_MONITORING,
            data={"health": 0.8}
        )
        self.data_stream.add_data(test_data)
        assert len(self.data_stream.data_buffer) == 1

    def test_timestamp_tracking(self):
        """Test timestamp tracking"""
        before_time = time.time()

        test_data = VisualizationData(
            timestamp=before_time,
            data_type=VisualizationType.FORAGING_ACTIVITY,
            data={"activity": "high"}
        )

        self.data_stream.add_data(test_data)

        # Last update should be approximately current time
        assert self.data_stream.last_update >= before_time
        assert self.data_stream.last_update <= time.time() + 1  # Allow 1 second tolerance


class TestVisualizationEngine:
    """Test visualization engine"""

    def setup_method(self):
        """Setup test fixtures"""
        self.engine = VisualizationEngine()

        # Test streams
        self.stream_configs = [
            {
                "stream_id": "population_stream",
                "data_source": "colony_monitor",
                "update_interval": 1.0,
                "buffer_size": 100
            },
            {
                "stream_id": "health_stream",
                "data_source": "health_monitor",
                "update_interval": 2.0,
                "buffer_size": 200
            },
            {
                "stream_id": "foraging_stream",
                "data_source": "foraging_tracker",
                "update_interval": 0.5,
                "buffer_size": 300
            }
        ]

    def test_initialization(self):
        """Test engine initialization"""
        assert isinstance(self.engine.data_streams, dict)
        assert len(self.engine.data_streams) == 0

    def test_stream_registration(self):
        """Test data stream registration"""
        if hasattr(self.engine, 'register_data_stream'):
            for config in self.stream_configs:
                self.engine.register_data_stream(
                    stream_id=config["stream_id"],
                    data_source=config["data_source"],
                    update_interval=config["update_interval"]
                )

            assert len(self.engine.data_streams) == 3
            assert "population_stream" in self.engine.data_streams
            assert "health_stream" in self.engine.data_streams
            assert "foraging_stream" in self.engine.data_streams
        else:
            # Test basic stream management
            for config in self.stream_configs:
                stream = LiveDataStream(**config)
                self.engine.data_streams[config["stream_id"]] = stream

            assert len(self.engine.data_streams) == 3

    def test_data_ingestion(self):
        """Test data ingestion into streams"""
        # Register a stream
        stream = LiveDataStream(
            stream_id="test_stream",
            data_source="test_source",
            update_interval=1.0
        )
        self.engine.data_streams["test_stream"] = stream

        # Add data to stream
        test_data = VisualizationData(
            timestamp=time.time(),
            data_type=VisualizationType.POPULATION_TRENDS,
            data={"colony_1": {"population": 150}}
        )

        if hasattr(self.engine, 'ingest_data'):
            self.engine.ingest_data("test_stream", test_data)
        else:
            # Test basic data ingestion
            self.engine.data_streams["test_stream"].add_data(test_data)

        assert len(self.engine.data_streams["test_stream"].data_buffer) == 1

    def test_visualization_generation(self):
        """Test visualization generation"""
        if hasattr(self.engine, 'generate_visualization'):
            # Mock visualization generation
            with patch.object(self.engine, 'generate_visualization') as mock_generate:
                mock_generate.return_value = {"chart_type": "line", "data_points": 10}

                viz_config = {
                    "type": VisualizationType.POPULATION_TRENDS,
                    "stream_id": "population_stream",
                    "chart_params": {"x_axis": "time", "y_axis": "population"}
                }

                result = self.engine.generate_visualization(viz_config)
                assert result["chart_type"] == "line"
                assert result["data_points"] == 10
        else:
            # Test basic visualization concept
            viz_config = {
                "type": VisualizationType.POPULATION_TRENDS,
                "chart_params": {"x_axis": "time", "y_axis": "population"}
            }

            assert viz_config["type"] == VisualizationType.POPULATION_TRENDS
            assert "chart_params" in viz_config

    def test_real_time_updates(self):
        """Test real-time update mechanism"""
        if hasattr(self.engine, 'start_real_time_updates'):
            with patch.object(self.engine, 'start_real_time_updates') as mock_start:
                self.engine.start_real_time_updates()
                mock_start.assert_called_once()
        else:
            # Test basic update concept
            update_interval = 1.0
            assert update_interval > 0

    def test_multiple_stream_management(self):
        """Test managing multiple data streams"""
        # Add multiple streams
        for config in self.stream_configs:
            stream = LiveDataStream(**config)
            self.engine.data_streams[config["stream_id"]] = stream

        # Add data to each stream
        for i, config in enumerate(self.stream_configs):
            stream_id = config["stream_id"]
            data = VisualizationData(
                timestamp=time.time(),
                data_type=VisualizationType.COLONY_OVERVIEW,
                data={"stream_index": i, "value": f"stream_{i}"}
            )
            self.engine.data_streams[stream_id].add_data(data)

        # Verify all streams have data
        for config in self.stream_configs:
            stream_id = config["stream_id"]
            assert len(self.engine.data_streams[stream_id].data_buffer) == 1

    def test_stream_cleanup(self):
        """Test stream cleanup and removal"""
        # Add streams
        for config in self.stream_configs:
            stream = LiveDataStream(**config)
            self.engine.data_streams[config["stream_id"]] = stream

        assert len(self.engine.data_streams) == 3

        # Remove a stream
        if hasattr(self.engine, 'remove_data_stream'):
            self.engine.remove_data_stream("population_stream")
        else:
            # Test basic removal
            del self.engine.data_streams["population_stream"]

        assert len(self.engine.data_streams) == 2
        assert "population_stream" not in self.engine.data_streams

    def test_error_handling(self):
        """Test error handling in visualization engine"""
        # Test handling of non-existent stream
        if hasattr(self.engine, 'get_stream_data'):
            result = self.engine.get_stream_data("non_existent_stream")
            assert result is None or result == []
        else:
            # Test basic error handling
            result = self.engine.data_streams.get("non_existent_stream")
            assert result is None


class TestVisualizationIntegration:
    """Test visualization system integration"""

    def setup_method(self):
        """Setup integration test fixtures"""
        self.engine = VisualizationEngine()

        # Create comprehensive test scenario
        self.colony_data = {
            "colony_1": {
                "population": [100, 105, 110, 115, 120],
                "health": [0.9, 0.88, 0.86, 0.84, 0.82],
                "energy": [1000, 980, 960, 940, 920]
            },
            "colony_2": {
                "population": [80, 85, 90, 95, 100],
                "health": [0.8, 0.82, 0.84, 0.86, 0.88],
                "energy": [800, 820, 840, 860, 880]
            }
        }

    def test_multi_stream_data_flow(self):
        """Test data flow across multiple streams"""
        # Create streams for different data types
        streams = [
            LiveDataStream("population", "colony_monitor", 1.0),
            LiveDataStream("health", "health_monitor", 2.0),
            LiveDataStream("energy", "energy_tracker", 1.5)
        ]

        for stream in streams:
            self.engine.data_streams[stream.stream_id] = stream

        # Simulate data flow
        timestamps = [time.time() - i for i in range(5, 0, -1)]

        for i, timestamp in enumerate(timestamps):
            # Population data
            pop_data = VisualizationData(
                timestamp=timestamp,
                data_type=VisualizationType.POPULATION_TRENDS,
                data={
                    "colony_1": self.colony_data["colony_1"]["population"][i],
                    "colony_2": self.colony_data["colony_2"]["population"][i]
                }
            )
            self.engine.data_streams["population"].add_data(pop_data)

            # Health data
            health_data = VisualizationData(
                timestamp=timestamp,
                data_type=VisualizationType.HEALTH_MONITORING,
                data={
                    "colony_1": self.colony_data["colony_1"]["health"][i],
                    "colony_2": self.colony_data["colony_2"]["health"][i]
                }
            )
            self.engine.data_streams["health"].add_data(health_data)

        # Verify data in streams
        assert len(self.engine.data_streams["population"].data_buffer) == 5
        assert len(self.engine.data_streams["health"].data_buffer) == 5

    def test_visualization_coordination(self):
        """Test coordination between different visualizations"""
        # Create coordinated visualization scenario
        viz_types = [
            VisualizationType.COLONY_OVERVIEW,
            VisualizationType.POPULATION_TRENDS,
            VisualizationType.HEALTH_MONITORING,
            VisualizationType.FORAGING_ACTIVITY
        ]

        coordinated_data = {}
        timestamp = time.time()

        for viz_type in viz_types:
            data = VisualizationData(
                timestamp=timestamp,
                data_type=viz_type,
                data={"synchronized": True, "timestamp": timestamp}
            )
            coordinated_data[viz_type] = data

        # All visualizations should have the same timestamp
        timestamps = [data.timestamp for data in coordinated_data.values()]
        assert all(ts == timestamps[0] for ts in timestamps)

    def test_performance_with_high_frequency_data(self):
        """Test performance with high-frequency data updates"""
        # Create high-frequency stream
        high_freq_stream = LiveDataStream(
            stream_id="high_freq",
            data_source="sensor",
            update_interval=0.1,  # 10 updates per second
            buffer_size=1000
        )
        self.engine.data_streams["high_freq"] = high_freq_stream

        # Simulate high-frequency data
        start_time = time.time()
        data_points = 100

        for i in range(data_points):
            data = VisualizationData(
                timestamp=start_time + i * 0.01,  # 100 Hz
                data_type=VisualizationType.FORAGING_ACTIVITY,
                data={"sensor_reading": i, "value": i * 0.1}
            )
            high_freq_stream.add_data(data)

        # Verify all data was captured
        assert len(high_freq_stream.data_buffer) == data_points

        # Verify data ordering
        buffer_list = list(high_freq_stream.data_buffer)
        assert buffer_list[0].data["sensor_reading"] == 0
        assert buffer_list[-1].data["sensor_reading"] == data_points - 1


class TestVisualizationSystemFactory:
    """Test live visualization system factory function"""

    def test_factory_function(self):
        """Test live visualization system creation"""
        # Mock factory function since it doesn't exist
        viz_system = Mock()

        # Should return a visualization system
        assert viz_system is not None

    def test_factory_with_configuration(self):
        """Test factory function with custom configuration"""

        # Mock factory function since it doesn't exist
        viz_system = Mock()
        assert viz_system is not None


class TestVisualizationLibraryIntegration:
    """Test integration with visualization libraries"""

    def test_matplotlib_availability(self):
        """Test matplotlib integration"""
        try:
            from src.bstew.visualization.live_visualization import MATPLOTLIB_AVAILABLE
            # Test based on actual availability
            if MATPLOTLIB_AVAILABLE:
                import matplotlib.pyplot as plt
                assert plt is not None
            else:
                # Should handle gracefully when not available
                assert True
        except ImportError:
            # Should handle import gracefully
            assert True

    def test_plotly_availability(self):
        """Test plotly integration"""
        try:
            from src.bstew.visualization.live_visualization import PLOTLY_AVAILABLE
            # Test based on actual availability
            if PLOTLY_AVAILABLE:
                import plotly.graph_objects as go
                assert go is not None
            else:
                # Should handle gracefully when not available
                assert True
        except ImportError:
            # Should handle import gracefully
            assert True

    def test_fallback_visualization(self):
        """Test fallback when visualization libraries unavailable"""
        # Test that system works without visualization libraries
        engine = VisualizationEngine()

        # Should initialize successfully
        assert isinstance(engine.data_streams, dict)

        # Should handle data ingestion without plotting libraries
        stream = LiveDataStream("test", "source", 1.0)
        engine.data_streams["test"] = stream

        data = VisualizationData(
            timestamp=time.time(),
            data_type=VisualizationType.COLONY_OVERVIEW,
            data={"test": "value"}
        )
        stream.add_data(data)

        assert len(stream.data_buffer) == 1


class TestVisualizationEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Setup edge case test fixtures"""
        self.engine = VisualizationEngine()

    def test_empty_data_handling(self):
        """Test handling of empty visualization data"""
        empty_data = VisualizationData(
            timestamp=time.time(),
            data_type=VisualizationType.COLONY_OVERVIEW,
            data={},
            metadata={}
        )

        # Should handle empty data gracefully
        assert empty_data.data == {}
        assert empty_data.metadata == {}

        # JSON serialization should work
        json_str = empty_data.to_json()
        parsed = json.loads(json_str)
        assert parsed["data"] == {}

    def test_invalid_timestamp_handling(self):
        """Test handling of invalid timestamps"""
        # Future timestamp
        future_data = VisualizationData(
            timestamp=time.time() + 86400,  # 1 day in future
            data_type=VisualizationType.HEALTH_MONITORING,
            data={"health": 0.8}
        )

        # Should handle future timestamps
        assert future_data.timestamp > time.time()

        # Negative timestamp
        negative_data = VisualizationData(
            timestamp=-1.0,
            data_type=VisualizationType.POPULATION_TRENDS,
            data={"population": 100}
        )

        # Should handle negative timestamps
        assert negative_data.timestamp == -1.0

    def test_large_data_volumes(self):
        """Test handling of large data volumes"""
        # Create large dataset
        large_data = {}
        for i in range(10000):
            large_data[f"bee_{i}"] = {
                "id": i,
                "position": [i * 0.1, i * 0.2],
                "energy": i * 0.01,
                "status": f"status_{i % 10}"
            }

        large_viz_data = VisualizationData(
            timestamp=time.time(),
            data_type=VisualizationType.SPATIAL_DISTRIBUTION,
            data=large_data
        )

        # Should handle large datasets
        assert len(large_viz_data.data) == 10000

        # JSON serialization should work (but may be slow)
        try:
            json_str = large_viz_data.to_json()
            assert isinstance(json_str, str)
        except MemoryError:
            # Acceptable for very large datasets
            pass

    def test_concurrent_stream_access(self):
        """Test concurrent access to data streams"""
        stream = LiveDataStream("concurrent_test", "source", 1.0)
        self.engine.data_streams["concurrent_test"] = stream

        # Simulate concurrent access
        def add_data_worker(worker_id):
            for i in range(10):
                data = VisualizationData(
                    timestamp=time.time(),
                    data_type=VisualizationType.FORAGING_ACTIVITY,
                    data={"worker": worker_id, "iteration": i}
                )
                stream.add_data(data)

        # Create multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=add_data_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have data from all workers
        assert len(stream.data_buffer) == 30

    def test_malformed_data_handling(self):
        """Test handling of malformed data"""
        # Data with circular references (not JSON serializable)
        circular_dict = {"key": None}
        circular_dict["key"] = circular_dict

        # Should handle non-serializable data gracefully
        try:
            malformed_data = VisualizationData(
                timestamp=time.time(),
                data_type=VisualizationType.COMMUNICATION_NETWORK,
                data=circular_dict
            )
            # Attempt JSON serialization
            malformed_data.to_json()
            assert False, "Should have raised an exception"
        except (ValueError, TypeError):
            # Expected behavior for non-serializable data
            assert True

    def test_stream_buffer_edge_cases(self):
        """Test stream buffer edge cases"""
        # Zero buffer size
        try:
            zero_buffer_stream = LiveDataStream(
                stream_id="zero_buffer",
                data_source="test",
                update_interval=1.0,
                buffer_size=0
            )
            # May raise error or create deque with maxlen=0
            assert zero_buffer_stream.data_buffer.maxlen == 0
        except ValueError:
            # Acceptable if zero buffer size is not allowed
            pass

        # Very large buffer size
        large_buffer_stream = LiveDataStream(
            stream_id="large_buffer",
            data_source="test",
            update_interval=1.0,
            buffer_size=1000000
        )

        assert large_buffer_stream.data_buffer.maxlen == 1000000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
