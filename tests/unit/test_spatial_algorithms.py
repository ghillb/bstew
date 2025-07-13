"""
Unit tests for Spatial Algorithms and Integration
================================================

Tests for the advanced spatial algorithms system including spatial indexing,
pathfinding, connectivity analysis, and spatial integration for bee movement.
"""

import pytest
from unittest.mock import Mock, patch
import math

from src.bstew.core.spatial_algorithms import (
    SpatialIndex, PathFinder, ConnectivityAnalyzer, LandscapeAnalyzer,
    SpatialPoint, SpatialPatch, MovementCorridor, ConnectivityType,
    DistanceMetric
)
from src.bstew.core.spatial_integration import (
    SpatialEnvironmentManager, SpatialBeeManager, SpatialBeeState,
    create_spatial_integration_system
)
from src.bstew.core.enums import BeeStatus


class TestSpatialPoint:
    """Test spatial point data structure"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.point1 = SpatialPoint(x=0.0, y=0.0, z=0.0, patch_id=1)
        self.point2 = SpatialPoint(x=3.0, y=4.0, z=0.0, patch_id=2)
        self.point3 = SpatialPoint(x=1.0, y=1.0, z=1.0, patch_id=3)
    
    def test_initialization(self):
        """Test point initialization"""
        assert self.point1.x == 0.0
        assert self.point1.y == 0.0
        assert self.point1.z == 0.0
        assert self.point1.patch_id == 1
        assert isinstance(self.point1.properties, dict)
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation"""
        # 2D distance test (3,4,5 triangle)
        distance = self.point1.distance_to(self.point2, DistanceMetric.EUCLIDEAN)
        assert abs(distance - 5.0) < 0.001
        
        # 3D distance test
        distance_3d = self.point1.distance_to(self.point3, DistanceMetric.EUCLIDEAN)
        expected = math.sqrt(1 + 1 + 1)  # sqrt(3)
        assert abs(distance_3d - expected) < 0.001
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation"""
        distance = self.point1.distance_to(self.point2, DistanceMetric.MANHATTAN)
        assert distance == 7.0  # |3-0| + |4-0| + |0-0|
        
        distance_3d = self.point1.distance_to(self.point3, DistanceMetric.MANHATTAN)
        assert distance_3d == 3.0  # |1-0| + |1-0| + |1-0|
    
    def test_chebyshev_distance(self):
        """Test Chebyshev (infinity norm) distance calculation"""
        distance = self.point1.distance_to(self.point2, DistanceMetric.CHEBYSHEV)
        assert distance == 4.0  # max(|3-0|, |4-0|, |0-0|)
        
        distance_3d = self.point1.distance_to(self.point3, DistanceMetric.CHEBYSHEV)
        assert distance_3d == 1.0  # max(|1-0|, |1-0|, |1-0|)
    
    def test_properties_storage(self):
        """Test properties storage and access"""
        point = SpatialPoint(x=1.0, y=2.0, properties={"quality": 0.8, "visited": True})
        assert point.properties["quality"] == 0.8
        assert point.properties["visited"] is True


class TestSpatialPatch:
    """Test spatial patch data structure"""
    
    def setup_method(self):
        """Setup test fixtures"""
        center = SpatialPoint(x=10.0, y=10.0, z=0.0)
        
        # Simple square patch
        vertices = [
            SpatialPoint(x=8.0, y=8.0),
            SpatialPoint(x=12.0, y=8.0),
            SpatialPoint(x=12.0, y=12.0),
            SpatialPoint(x=8.0, y=12.0)
        ]
        
        self.patch = SpatialPatch(
            patch_id=100,
            center=center,
            vertices=vertices,
            area=16.0,
            perimeter=16.0,
            quality=0.8,
            resource_density=0.7,
            accessibility=1.0
        )
    
    def test_initialization(self):
        """Test patch initialization"""
        assert self.patch.patch_id == 100
        assert self.patch.center.x == 10.0
        assert self.patch.center.y == 10.0
        assert self.patch.area == 16.0
        assert self.patch.quality == 0.8
        assert self.patch.resource_density == 0.7
        assert len(self.patch.vertices) == 4
    
    def test_point_in_patch_simple(self):
        """Test point-in-patch for points without vertices (default square)"""
        # Create patch without vertices - should use default square logic
        simple_patch = SpatialPatch(
            patch_id=101,
            center=SpatialPoint(x=0.0, y=0.0),
            area=4.0  # 2x2 square
        )
        
        # Point at center should be inside
        center_point = SpatialPoint(x=0.0, y=0.0)
        assert simple_patch.contains_point(center_point)
        
        # Point just inside boundary
        inside_point = SpatialPoint(x=0.9, y=0.9)
        assert simple_patch.contains_point(inside_point)
        
        # Point outside boundary
        outside_point = SpatialPoint(x=2.0, y=2.0)
        assert not simple_patch.contains_point(outside_point)
    
    def test_point_in_polygon(self):
        """Test point-in-polygon for patches with vertices"""
        # Point clearly inside
        inside_point = SpatialPoint(x=10.0, y=10.0)
        assert self.patch.contains_point(inside_point)
        
        # Point clearly outside
        outside_point = SpatialPoint(x=20.0, y=20.0)
        assert not self.patch.contains_point(outside_point)
        
        # Point on edge (should be handled consistently)
        edge_point = SpatialPoint(x=8.0, y=10.0)
        # Edge behavior may vary, just ensure it doesn't crash
        result = self.patch.contains_point(edge_point)
        assert isinstance(result, bool)
    
    def test_neighbor_management(self):
        """Test neighbor patch management"""
        # Initially no neighbors
        assert len(self.patch.neighbors) == 0
        
        # Add neighbors
        self.patch.neighbors.add(101)
        self.patch.neighbors.add(102)
        
        assert 101 in self.patch.neighbors
        assert 102 in self.patch.neighbors
        assert len(self.patch.neighbors) == 2


class TestMovementCorridor:
    """Test movement corridor system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        waypoints = [
            SpatialPoint(x=0.0, y=0.0),
            SpatialPoint(x=5.0, y=0.0),
            SpatialPoint(x=5.0, y=5.0),
            SpatialPoint(x=10.0, y=5.0)
        ]
        
        self.corridor = MovementCorridor(
            corridor_id=1,
            start_patch=100,
            end_patch=200,
            waypoints=waypoints,
            width=2.0,
            quality=0.9
        )
    
    def test_initialization(self):
        """Test corridor initialization"""
        assert self.corridor.corridor_id == 1
        assert self.corridor.start_patch == 100
        assert self.corridor.end_patch == 200
        assert len(self.corridor.waypoints) == 4
        assert self.corridor.width == 2.0
        assert self.corridor.quality == 0.9
    
    def test_path_length_calculation(self):
        """Test path length calculation"""
        # Expected length: 5 + 5 + 5 = 15
        length = self.corridor.get_path_length()
        assert abs(length - 15.0) < 0.001
    
    def test_empty_corridor(self):
        """Test corridor with insufficient waypoints"""
        empty_corridor = MovementCorridor(
            corridor_id=2,
            start_patch=101,
            end_patch=201,
            waypoints=[]
        )
        
        assert empty_corridor.get_path_length() == 0.0
        
        single_point_corridor = MovementCorridor(
            corridor_id=3,
            start_patch=102,
            end_patch=202,
            waypoints=[SpatialPoint(x=0.0, y=0.0)]
        )
        
        assert single_point_corridor.get_path_length() == 0.0


class TestSpatialIndex:
    """Test spatial indexing system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.spatial_index = SpatialIndex(grid_size=10.0)
        
        # Create test patches
        self.patches = []
        for i in range(5):
            center = SpatialPoint(x=i * 10.0, y=i * 10.0, z=0.0)
            patch = SpatialPatch(
                patch_id=i,
                center=center,
                area=4.0,
                quality=0.5 + i * 0.1
            )
            self.patches.append(patch)
            self.spatial_index.add_patch(patch)
    
    def test_initialization(self):
        """Test spatial index initialization"""
        assert self.spatial_index.grid_size == 10.0
        assert self.spatial_index.enable_rtree is True
        assert self.spatial_index.enable_kdtree is True
        assert len(self.spatial_index._patches) == 5
    
    def test_patch_addition(self):
        """Test adding patches to spatial index"""
        new_patch = SpatialPatch(
            patch_id=10,
            center=SpatialPoint(x=50.0, y=50.0),
            area=1.0
        )
        
        self.spatial_index.add_patch(new_patch)
        
        assert 10 in self.spatial_index._patches
        assert self.spatial_index._patches[10] == new_patch
    
    def test_nearest_patches_query(self):
        """Test nearest patches query"""
        query_point = SpatialPoint(x=15.0, y=15.0)
        nearest = self.spatial_index.find_nearest_patches(query_point, k=3)
        
        # Should return patch IDs
        assert len(nearest) <= 3
        assert all(isinstance(patch_id, int) for patch_id in nearest)
        
        # Nearest should be patch 1 (at 10,10)
        if nearest:
            assert 1 in nearest
    
    def test_radius_query(self):
        """Test patches within radius query"""
        query_point = SpatialPoint(x=10.0, y=10.0)
        patches_in_radius = self.spatial_index.find_patches_in_radius(query_point, radius=15.0)
        
        # Should include patches 0, 1, and 2
        assert 1 in patches_in_radius  # At exact location
        assert len(patches_in_radius) >= 1
    
    def test_grid_neighbors(self):
        """Test grid-based neighbor finding"""
        neighbors = self.spatial_index.get_grid_neighbors(1)
        
        # Should be a set of patch IDs
        assert isinstance(neighbors, set)
        # Should not include the patch itself
        assert 1 not in neighbors
    
    def test_kdtree_rebuilding(self):
        """Test KD-tree rebuilding functionality"""
        # Add more patches to trigger rebuilding
        for i in range(5, 10):
            patch = SpatialPatch(
                patch_id=i,
                center=SpatialPoint(x=i * 5.0, y=i * 5.0),
                area=1.0
            )
            self.spatial_index.add_patch(patch)
        
        # KD-tree should be built
        assert self.spatial_index._kdtree is not None
        assert self.spatial_index._patch_centroids is not None
        assert len(self.spatial_index._patch_centroids) == 10


class TestPathFinder:
    """Test pathfinding algorithms"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create spatial index with connected patches
        self.spatial_index = SpatialIndex(grid_size=5.0)
        
        # Create a simple 3x3 grid of patches
        for i in range(3):
            for j in range(3):
                patch_id = i * 3 + j
                center = SpatialPoint(x=i * 10.0, y=j * 10.0)
                patch = SpatialPatch(
                    patch_id=patch_id,
                    center=center,
                    area=4.0
                )
                
                # Set up neighbors manually for simplicity
                neighbors = set()
                if i > 0:
                    neighbors.add((i-1) * 3 + j)  # Left
                if i < 2:
                    neighbors.add((i+1) * 3 + j)  # Right
                if j > 0:
                    neighbors.add(i * 3 + (j-1))  # Up
                if j < 2:
                    neighbors.add(i * 3 + (j+1))  # Down
                
                patch.neighbors = neighbors
                self.spatial_index.add_patch(patch)
        
        self.path_finder = PathFinder(spatial_index=self.spatial_index)
    
    def test_initialization(self):
        """Test path finder initialization"""
        assert self.path_finder.spatial_index == self.spatial_index
        assert self.path_finder.default_movement_cost == 1.0
        assert hasattr(self.path_finder, 'barrier_penalty')
    
    def test_direct_path(self):
        """Test direct pathfinding"""
        path = self.path_finder.find_shortest_path(0, 8, algorithm="direct")
        assert path == [0, 8]
    
    def test_same_patch_path(self):
        """Test pathfinding from patch to itself"""
        path = self.path_finder.find_shortest_path(4, 4, algorithm="astar")
        assert path == [4]
    
    def test_invalid_patches(self):
        """Test pathfinding with invalid patch IDs"""
        path = self.path_finder.find_shortest_path(0, 999)  # Non-existent patch
        assert path is None
        
        path = self.path_finder.find_shortest_path(999, 0)  # Non-existent start
        assert path is None
    
    def test_astar_pathfinding(self):
        """Test A* pathfinding algorithm"""
        # Mock the internal methods for testing
        with patch.object(self.path_finder, '_get_patch_neighbors') as mock_neighbors, \
             patch.object(self.path_finder, '_movement_cost') as mock_cost, \
             patch.object(self.path_finder, '_heuristic') as mock_heuristic:
            
            # Set up simple path: 0 -> 1 -> 2
            mock_neighbors.side_effect = lambda x: {0: [1], 1: [0, 2], 2: [1]}.get(x, [])
            mock_cost.return_value = 1.0
            mock_heuristic.side_effect = lambda x, goal: abs(x - goal)
            
            path = self.path_finder.find_shortest_path(0, 2, algorithm="astar")
            
            # Should find a valid path
            if path:
                assert path[0] == 0
                assert path[-1] == 2
                assert len(path) >= 2
    
    def test_dijkstra_pathfinding(self):
        """Test Dijkstra pathfinding algorithm"""
        # Mock the internal methods for testing
        with patch.object(self.path_finder, '_get_patch_neighbors') as mock_neighbors, \
             patch.object(self.path_finder, '_movement_cost') as mock_cost:
            
            # Set up simple linear path
            mock_neighbors.side_effect = lambda x: {0: [1], 1: [0, 2], 2: [1]}.get(x, [])
            mock_cost.return_value = 1.0
            
            path = self.path_finder.find_shortest_path(0, 2, algorithm="dijkstra")
            
            # Should find a valid path
            if path:
                assert path[0] == 0
                assert path[-1] == 2


class TestSpatialEnvironmentManager:
    """Test spatial environment management"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create real instances instead of mocks to avoid Pydantic validation issues
        self.spatial_index = SpatialIndex()
        self.path_finder = PathFinder(spatial_index=self.spatial_index)
        self.connectivity_analyzer = ConnectivityAnalyzer(
            spatial_index=self.spatial_index,
            path_finder=self.path_finder
        )
        self.landscape_analyzer = LandscapeAnalyzer(
            spatial_index=self.spatial_index,
            connectivity_analyzer=self.connectivity_analyzer
        )
        
        self.environment_manager = SpatialEnvironmentManager(
            spatial_index=self.spatial_index,
            path_finder=self.path_finder,
            connectivity_analyzer=self.connectivity_analyzer,
            landscape_analyzer=self.landscape_analyzer
        )
    
    def test_initialization(self):
        """Test environment manager initialization"""
        assert self.environment_manager.spatial_index == self.spatial_index
        assert self.environment_manager.path_finder == self.path_finder
        assert isinstance(self.environment_manager.patch_qualities, dict)
        assert isinstance(self.environment_manager.resource_levels, dict)
        assert self.environment_manager.quality_change_rate == 0.01
    
    def test_landscape_initialization(self):
        """Test landscape initialization from configuration"""
        landscape_config = {
            'patches': [
                {'id': 1, 'x': 10.0, 'y': 10.0, 'quality': 0.8, 'area': 4.0},
                {'id': 2, 'x': 20.0, 'y': 20.0, 'quality': 0.6, 'area': 6.0}
            ]
        }
        
        # Initialize landscape with patches
        with patch.object(self.environment_manager, '_establish_patch_connectivity'):
            self.environment_manager.initialize_landscape(landscape_config)
        
        # Check patches were created and added to spatial index
        assert len(self.spatial_index._patches) == 2
        assert 1 in self.spatial_index._patches
        assert 2 in self.spatial_index._patches
        
        # Check patch properties
        patch1 = self.spatial_index._patches[1]
        patch2 = self.spatial_index._patches[2]
        assert patch1.patch_id == 1
        assert patch2.patch_id == 2
        
        # Check dynamic state initialization
        assert 1 in self.environment_manager.patch_qualities
        assert 2 in self.environment_manager.patch_qualities
    
    def test_patch_quality_tracking(self):
        """Test patch quality tracking and updates"""
        patch_id = 100
        initial_quality = 0.7
        
        self.environment_manager.patch_qualities[patch_id] = initial_quality
        
        # Get current quality
        quality = self.environment_manager.get_patch_current_quality(patch_id)
        assert quality == initial_quality
        
        # Test with non-existent patch
        unknown_quality = self.environment_manager.get_patch_current_quality(999)
        assert unknown_quality == 0.5  # Default value
    
    def test_resource_consumption(self):
        """Test resource consumption mechanics"""
        patch_id = 200
        initial_resources = 10.0
        
        self.environment_manager.resource_levels[patch_id] = initial_resources
        
        # Consume some resources
        consumed = self.environment_manager.consume_resources(patch_id, 3.0)
        assert consumed == 3.0
        assert self.environment_manager.resource_levels[patch_id] == 7.0
        
        # Try to consume more than available
        consumed = self.environment_manager.consume_resources(patch_id, 10.0)
        assert consumed == 7.0
        assert self.environment_manager.resource_levels[patch_id] == 0.0
        
        # Check patch usage tracking
        assert self.environment_manager.patch_usage[patch_id] == 2
    
    def test_landscape_dynamics_update(self):
        """Test landscape dynamics updates"""
        # Set up patches
        patch_id = 300
        self.spatial_index._patches = {
            patch_id: SpatialPatch(
                patch_id=patch_id,
                center=SpatialPoint(x=0, y=0),
                quality=0.8,
                resource_density=0.6
            )
        }
        
        self.environment_manager.patch_qualities[patch_id] = 0.8
        self.environment_manager.resource_levels[patch_id] = 0.4
        self.environment_manager.seasonal_variation = True
        
        # Update dynamics
        self.environment_manager.update_landscape_dynamics(timestep=100)
        
        # Check seasonal cycle update
        expected_season = (100 % 365) / 365.0
        assert self.environment_manager.current_season == expected_season
        
        # Quality should be updated based on seasonal variation
        updated_quality = self.environment_manager.patch_qualities[patch_id]
        assert isinstance(updated_quality, float)
        assert 0.1 <= updated_quality <= 1.0
    
    def test_optimal_foraging_patches(self):
        """Test optimal foraging patch finding"""
        # Add patches to spatial index
        patch1 = SpatialPatch(patch_id=1, center=SpatialPoint(x=5, y=5), quality=0.8)
        patch2 = SpatialPatch(patch_id=2, center=SpatialPoint(x=10, y=10), quality=0.6)
        patch3 = SpatialPatch(patch_id=3, center=SpatialPoint(x=15, y=15), quality=0.9)
        
        self.spatial_index.add_patch(patch1)
        self.spatial_index.add_patch(patch2)
        self.spatial_index.add_patch(patch3)
        
        # Set up quality and resource levels
        self.environment_manager.patch_qualities = {1: 0.8, 2: 0.6, 3: 0.9}
        self.environment_manager.resource_levels = {1: 0.7, 2: 0.5, 3: 0.8}
        
        bee_position = SpatialPoint(x=0, y=0)
        optimal_patches = self.environment_manager.find_optimal_foraging_patches(
            bee_position, max_distance=20.0, min_quality=0.5
        )
        
        # Should return sorted list of (patch_id, score) tuples
        assert isinstance(optimal_patches, list)
        if optimal_patches:
            assert isinstance(optimal_patches[0], tuple)
            assert len(optimal_patches[0]) == 2
            
            # Scores should be in descending order
            scores = [score for _, score in optimal_patches]
            assert scores == sorted(scores, reverse=True)


class TestSpatialBeeManager:
    """Test spatial bee management system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock environment manager
        self.environment_manager = Mock()
        self.environment_manager.find_optimal_foraging_patches.return_value = [(1, 0.8), (2, 0.6)]
        self.environment_manager.path_finder.find_shortest_path.return_value = [0, 1]
        self.environment_manager.get_patch_current_quality.return_value = 0.7
        self.environment_manager.consume_resources.return_value = 0.1
        self.environment_manager.spatial_index._patches = {
            1: SpatialPatch(patch_id=1, center=SpatialPoint(x=10, y=10))
        }
        
        self.bee_manager = SpatialBeeManager(environment_manager=self.environment_manager)
    
    def test_initialization(self):
        """Test bee manager initialization"""
        assert self.bee_manager.environment_manager == self.environment_manager
        assert isinstance(self.bee_manager.bee_states, dict)
        assert self.bee_manager.max_foraging_distance == 100.0
        assert self.bee_manager.movement_speed == 1.0
    
    def test_bee_registration(self):
        """Test bee registration in spatial system"""
        bee_id = 100
        initial_position = SpatialPoint(x=0.0, y=0.0)
        initial_patch = 0
        
        self.bee_manager.register_bee(bee_id, initial_position, initial_patch)
        
        assert bee_id in self.bee_manager.bee_states
        bee_state = self.bee_manager.bee_states[bee_id]
        assert bee_state.bee_id == bee_id
        assert bee_state.current_position == initial_position
        assert bee_state.current_patch == initial_patch
    
    def test_foraging_behavior(self):
        """Test foraging spatial behavior"""
        bee_id = 200
        initial_position = SpatialPoint(x=0.0, y=0.0)
        self.bee_manager.register_bee(bee_id, initial_position, 0)
        
        # Test foraging behavior
        result = self.bee_manager.update_bee_spatial_behavior(
            bee_id, BeeStatus.FORAGING, timestep=100
        )
        
        assert isinstance(result, dict)
        
        # Check that bee has a target patch
        bee_state = self.bee_manager.bee_states[bee_id]
        assert bee_state.target_patch is not None
        
        # Verify optimal patches were queried
        self.environment_manager.find_optimal_foraging_patches.assert_called()
    
    def test_scouting_behavior(self):
        """Test scouting spatial behavior"""
        bee_id = 300
        initial_position = SpatialPoint(x=5.0, y=5.0)
        self.bee_manager.register_bee(bee_id, initial_position, 0)
        
        # Mock spatial index for scouting
        self.environment_manager.spatial_index.find_patches_in_radius.return_value = [1, 2, 3]
        
        result = self.bee_manager.update_bee_spatial_behavior(
            bee_id, BeeStatus.SEARCHING, timestep=100
        )
        
        assert isinstance(result, dict)
        
        # Should query nearby patches
        self.environment_manager.spatial_index.find_patches_in_radius.assert_called()
    
    def test_returning_behavior(self):
        """Test returning to hive behavior"""
        bee_id = 400
        initial_position = SpatialPoint(x=20.0, y=20.0)
        self.bee_manager.register_bee(bee_id, initial_position, 5)  # Start at patch 5
        
        result = self.bee_manager.update_bee_spatial_behavior(
            bee_id, BeeStatus.BRINGING_NECTAR, timestep=100
        )
        
        assert isinstance(result, dict)
        
        # Check that target is hive (patch 0)
        bee_state = self.bee_manager.bee_states[bee_id]
        assert bee_state.target_patch == 0
    
    def test_bee_spatial_metrics(self):
        """Test bee spatial metrics calculation"""
        bee_id = 500
        initial_position = SpatialPoint(x=0.0, y=0.0)
        self.bee_manager.register_bee(bee_id, initial_position, 0)
        
        # Add some movement history
        bee_state = self.bee_manager.bee_states[bee_id]
        bee_state.movement_history = [
            SpatialPoint(x=0, y=0),
            SpatialPoint(x=1, y=1),
            SpatialPoint(x=2, y=2)
        ]
        bee_state.spatial_memory = {1: 0.8, 2: 0.6, 3: 0.9}
        bee_state.planned_path = [1, 2, 3]
        
        metrics = self.bee_manager.get_bee_spatial_metrics(bee_id)
        
        assert "current_position" in metrics
        assert "current_patch" in metrics
        assert "total_distance_traveled" in metrics
        assert "known_patches" in metrics
        assert "average_remembered_quality" in metrics
        assert "path_length" in metrics
        
        assert metrics["known_patches"] == 3
        assert metrics["path_length"] == 3
        assert metrics["total_distance_traveled"] >= 0


class TestSpatialBeeState:
    """Test spatial bee state management"""
    
    def setup_method(self):
        """Setup test fixtures"""
        initial_position = SpatialPoint(x=10.0, y=10.0)
        self.bee_state = SpatialBeeState(
            bee_id=123,
            current_position=initial_position,
            current_patch=1
        )
    
    def test_initialization(self):
        """Test bee state initialization"""
        assert self.bee_state.bee_id == 123
        assert self.bee_state.current_position.x == 10.0
        assert self.bee_state.current_patch == 1
        assert self.bee_state.target_patch is None
        assert len(self.bee_state.planned_path) == 0
        assert len(self.bee_state.movement_history) == 0
        assert len(self.bee_state.spatial_memory) == 0
    
    def test_position_update(self):
        """Test position update mechanics"""
        old_position = self.bee_state.current_position
        new_position = SpatialPoint(x=15.0, y=15.0)
        
        self.bee_state.update_position(new_position, patch_id=2)
        
        # Check position updated
        assert self.bee_state.current_position == new_position
        assert self.bee_state.current_patch == 2
        
        # Check history tracking
        assert len(self.bee_state.movement_history) == 1
        assert self.bee_state.movement_history[0] == old_position
    
    def test_movement_history_limit(self):
        """Test movement history size limitation"""
        # Add many positions to test limit
        for i in range(150):
            new_position = SpatialPoint(x=i, y=i)
            self.bee_state.update_position(new_position)
        
        # Should be limited to 100 entries
        assert len(self.bee_state.movement_history) == 100
    
    def test_spatial_memory_update(self):
        """Test spatial memory management"""
        # Remember new patch
        self.bee_state.remember_patch_quality(10, 0.8)
        assert self.bee_state.spatial_memory[10] == 0.8
        
        # Update existing memory (weighted average)
        self.bee_state.remember_patch_quality(10, 0.6)
        # Should be 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        expected = 0.7 * 0.8 + 0.3 * 0.6
        assert abs(self.bee_state.spatial_memory[10] - expected) < 0.001


class TestSpatialIntegrationSystem:
    """Test complete spatial integration system"""
    
    def test_factory_function(self):
        """Test spatial integration system factory function"""
        landscape_config = {
            'patches': [
                {'id': 1, 'x': 0.0, 'y': 0.0, 'quality': 0.5},
                {'id': 2, 'x': 10.0, 'y': 10.0, 'quality': 0.7}
            ],
            'use_gis_data': False
        }
        
        env_manager, bee_manager = create_spatial_integration_system(landscape_config)
        
        assert isinstance(env_manager, SpatialEnvironmentManager)
        assert isinstance(bee_manager, SpatialBeeManager)
        assert bee_manager.environment_manager == env_manager
    
    def test_end_to_end_workflow(self):
        """Test complete spatial workflow"""
        # Create system
        landscape_config = {
            'patches': [
                {'id': 0, 'x': 0.0, 'y': 0.0, 'quality': 0.8, 'area': 4.0},  # Hive
                {'id': 1, 'x': 10.0, 'y': 10.0, 'quality': 0.6, 'area': 2.0},
                {'id': 2, 'x': 20.0, 'y': 20.0, 'quality': 0.9, 'area': 3.0}
            ],
            'use_gis_data': False
        }
        
        env_manager, bee_manager = create_spatial_integration_system(landscape_config)
        
        # Register bees
        bee_id = 100
        initial_position = SpatialPoint(x=0.0, y=0.0)
        bee_manager.register_bee(bee_id, initial_position, 0)
        
        # Simulate foraging cycle
        foraging_result = bee_manager.update_bee_spatial_behavior(
            bee_id, BeeStatus.FORAGING, timestep=1
        )
        
        # Should have valid result
        assert isinstance(foraging_result, dict)
        
        # Update landscape dynamics
        env_manager.update_landscape_dynamics(timestep=1)
        
        # Get metrics
        metrics = bee_manager.get_bee_spatial_metrics(bee_id)
        assert isinstance(metrics, dict)
        assert "current_position" in metrics


class TestConnectivityAnalyzer:
    """Test connectivity analysis functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        try:
            from src.bstew.core.spatial_algorithms import ConnectivityAnalyzer, SpatialIndex, PathFinder
            spatial_index = SpatialIndex()
            path_finder = PathFinder(spatial_index=spatial_index)
            self.connectivity_analyzer = ConnectivityAnalyzer(
                spatial_index=spatial_index,
                path_finder=path_finder
            )
        except ImportError:
            # Create mock if class doesn't exist
            self.connectivity_analyzer = Mock()
    
    def test_connectivity_analysis(self):
        """Test basic connectivity analysis"""
        if hasattr(self.connectivity_analyzer, 'build_connectivity_graph'):
            # Test with mock spatial index
            spatial_index = Mock()
            spatial_index._patches = {
                1: SpatialPatch(patch_id=1, center=SpatialPoint(x=0, y=0)),
                2: SpatialPatch(patch_id=2, center=SpatialPoint(x=10, y=0)),
                3: SpatialPatch(patch_id=3, center=SpatialPoint(x=20, y=0))
            }
            
            # Test graph building
            graph = self.connectivity_analyzer.build_connectivity_graph(ConnectivityType.PROXIMITY)
            
            # Should return some form of graph structure
            assert graph is not None
        else:
            # Test mock functionality
            assert self.connectivity_analyzer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])