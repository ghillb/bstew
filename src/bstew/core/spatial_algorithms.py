"""
Spatial Algorithms and Connectivity Analysis for NetLogo BEE-STEWARD v2 Parity
==============================================================================

Advanced spatial analysis algorithms for bee movement, patch connectivity,
landscape analysis, and spatial optimization matching NetLogo's spatial capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import heapq
import math
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import networkx as nx

class DistanceMetric(Enum):
    """Distance calculation methods"""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    GEODESIC = "geodesic"
    WEIGHTED = "weighted"

class ConnectivityType(Enum):
    """Types of spatial connectivity"""
    ADJACENCY = "adjacency"  # Direct neighbors
    PROXIMITY = "proximity"  # Within threshold distance
    VISIBILITY = "visibility"  # Line of sight
    ACCESSIBILITY = "accessibility"  # Accounting for barriers
    FUNCTIONAL = "functional"  # Based on bee movement patterns

@dataclass
class SpatialPoint:
    """Spatial point with coordinates and properties"""
    x: float
    y: float
    z: float = 0.0
    patch_id: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def distance_to(self, other: 'SpatialPoint', metric: DistanceMetric = DistanceMetric.EUCLIDEAN) -> float:
        """Calculate distance to another point"""
        if metric == DistanceMetric.EUCLIDEAN:
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
        elif metric == DistanceMetric.MANHATTAN:
            return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)
        elif metric == DistanceMetric.CHEBYSHEV:
            return max(abs(self.x - other.x), abs(self.y - other.y), abs(self.z - other.z))
        else:
            return self.distance_to(other, DistanceMetric.EUCLIDEAN)

@dataclass
class SpatialPatch:
    """Spatial patch with geometric and ecological properties"""
    patch_id: int
    center: SpatialPoint
    vertices: List[SpatialPoint] = field(default_factory=list)
    area: float = 1.0
    perimeter: float = 4.0
    quality: float = 0.5
    resource_density: float = 0.5
    accessibility: float = 1.0
    barriers: List[SpatialPoint] = field(default_factory=list)
    neighbors: Set[int] = field(default_factory=set)
    
    def contains_point(self, point: SpatialPoint) -> bool:
        """Check if point is within patch boundaries"""
        if not self.vertices:
            # Default square patch
            half_size = math.sqrt(self.area) / 2
            return (abs(point.x - self.center.x) <= half_size and 
                   abs(point.y - self.center.y) <= half_size)
        
        # Point-in-polygon test for arbitrary shapes
        return self._point_in_polygon(point, self.vertices)
    
    def _point_in_polygon(self, point: SpatialPoint, vertices: List[SpatialPoint]) -> bool:
        """Ray casting algorithm for point-in-polygon test"""
        x, y = point.x, point.y
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0].x, vertices[0].y
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n].x, vertices[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

@dataclass
class MovementCorridor:
    """Spatial corridor for bee movement"""
    corridor_id: int
    start_patch: int
    end_patch: int
    waypoints: List[SpatialPoint]
    width: float = 1.0
    quality: float = 1.0
    barriers: List[SpatialPoint] = field(default_factory=list)
    usage_frequency: float = 0.0
    
    def get_path_length(self) -> float:
        """Calculate total path length"""
        if len(self.waypoints) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self.waypoints) - 1):
            total_length += self.waypoints[i].distance_to(self.waypoints[i + 1])
        
        return total_length

class SpatialIndex(BaseModel):
    """Spatial indexing for efficient spatial queries"""
    
    model_config = {"validate_assignment": True}
    
    # Configuration
    grid_size: float = 10.0
    enable_rtree: bool = True
    enable_kdtree: bool = True
    
    # Internal structures
    _patches: Dict[int, SpatialPatch] = {}
    _grid_index: Dict[Tuple[int, int], Set[int]] = {}
    _kdtree: Optional[Any] = None
    _patch_centroids: Optional[np.ndarray] = None
    
    def add_patch(self, patch: SpatialPatch) -> None:
        """Add patch to spatial index"""
        self._patches[patch.patch_id] = patch
        
        # Add to grid index
        grid_x = int(patch.center.x // self.grid_size)
        grid_y = int(patch.center.y // self.grid_size)
        grid_key = (grid_x, grid_y)
        
        if grid_key not in self._grid_index:
            self._grid_index[grid_key] = set()
        self._grid_index[grid_key].add(patch.patch_id)
        
        # Rebuild KD-tree if enabled
        if self.enable_kdtree:
            self._rebuild_kdtree()
    
    def _rebuild_kdtree(self) -> None:
        """Rebuild KD-tree for efficient nearest neighbor queries"""
        if not self._patches:
            return
        
        centroids = []
        for patch in self._patches.values():
            centroids.append([patch.center.x, patch.center.y, patch.center.z])
        
        self._patch_centroids = np.array(centroids)
        self._kdtree = cKDTree(self._patch_centroids)
    
    def find_nearest_patches(self, point: SpatialPoint, k: int = 5) -> List[int]:
        """Find k nearest patches to point"""
        if not self._kdtree:
            return []
        
        query_point = np.array([[point.x, point.y, point.z]])
        distances, indices = self._kdtree.query(query_point, k=min(k, len(self._patches)))
        
        return [list(self._patches.keys())[i] for i in indices[0]]
    
    def find_patches_in_radius(self, point: SpatialPoint, radius: float) -> List[int]:
        """Find all patches within radius of point"""
        if not self._kdtree:
            return []
        
        query_point = np.array([[point.x, point.y, point.z]])
        indices = self._kdtree.query_ball_point(query_point, radius)
        
        if len(indices) > 0:
            return [list(self._patches.keys())[i] for i in indices[0]]
        return []
    
    def get_grid_neighbors(self, patch_id: int) -> Set[int]:
        """Get patches in neighboring grid cells"""
        if patch_id not in self._patches:
            return set()
        
        patch = self._patches[patch_id]
        grid_x = int(patch.center.x // self.grid_size)
        grid_y = int(patch.center.y // self.grid_size)
        
        neighbors = set()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_key = (grid_x + dx, grid_y + dy)
                if neighbor_key in self._grid_index:
                    neighbors.update(self._grid_index[neighbor_key])
        
        neighbors.discard(patch_id)  # Remove self
        return neighbors

class PathFinder(BaseModel):
    """Advanced pathfinding algorithms for bee movement"""
    
    model_config = {"validate_assignment": True}
    
    # Configuration
    spatial_index: SpatialIndex
    default_movement_cost: float = 1.0
    barrier_penalty: float = 100.0
    energy_cost_factor: float = 0.1
    
    def find_shortest_path(self, start_patch: int, end_patch: int, 
                          algorithm: str = "astar") -> Optional[List[int]]:
        """Find shortest path between patches"""
        
        if algorithm == "astar":
            return self._astar_pathfinding(start_patch, end_patch)
        elif algorithm == "dijkstra":
            return self._dijkstra_pathfinding(start_patch, end_patch)
        elif algorithm == "direct":
            return [start_patch, end_patch]
        else:
            return self._astar_pathfinding(start_patch, end_patch)
    
    def _astar_pathfinding(self, start: int, goal: int) -> Optional[List[int]]:
        """A* pathfinding algorithm"""
        if start == goal:
            return [start]
        
        if start not in self.spatial_index._patches or goal not in self.spatial_index._patches:
            return None
        
        # Priority queue: (f_score, patch_id)
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            # Get neighbors
            neighbors = self._get_patch_neighbors(current)
            
            for neighbor in neighbors:
                tentative_g = g_score[current] + self._movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    if (f_score[neighbor], neighbor) not in open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _dijkstra_pathfinding(self, start: int, goal: int) -> Optional[List[int]]:
        """Dijkstra's pathfinding algorithm"""
        if start == goal:
            return [start]
        
        distances = {start: 0}
        previous = {}
        unvisited = set(self.spatial_index._patches.keys())
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda x: distances.get(x, float('inf')))
            
            if distances.get(current, float('inf')) == float('inf'):
                break  # No more reachable nodes
            
            unvisited.remove(current)
            
            if current == goal:
                return self._reconstruct_path(previous, current)
            
            neighbors = self._get_patch_neighbors(current)
            for neighbor in neighbors:
                if neighbor in unvisited:
                    alt_distance = distances[current] + self._movement_cost(current, neighbor)
                    if alt_distance < distances.get(neighbor, float('inf')):
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
        
        return None
    
    def _heuristic(self, patch1: int, patch2: int) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        if patch1 not in self.spatial_index._patches or patch2 not in self.spatial_index._patches:
            return float('inf')
        
        p1 = self.spatial_index._patches[patch1].center
        p2 = self.spatial_index._patches[patch2].center
        
        return p1.distance_to(p2)
    
    def _movement_cost(self, from_patch: int, to_patch: int) -> float:
        """Calculate movement cost between patches"""
        if from_patch not in self.spatial_index._patches or to_patch not in self.spatial_index._patches:
            return float('inf')
        
        patch1 = self.spatial_index._patches[from_patch]
        patch2 = self.spatial_index._patches[to_patch]
        
        # Base distance cost
        distance = patch1.center.distance_to(patch2.center)
        
        # Quality factors
        quality_factor = 1.0 / max(0.1, (patch1.quality + patch2.quality) / 2)
        
        # Accessibility factors
        accessibility_factor = 1.0 / max(0.1, (patch1.accessibility + patch2.accessibility) / 2)
        
        # Barrier penalties
        barrier_penalty = 0
        if patch1.barriers or patch2.barriers:
            barrier_penalty = self.barrier_penalty
        
        return distance * quality_factor * accessibility_factor + barrier_penalty
    
    def _get_patch_neighbors(self, patch_id: int) -> Set[int]:
        """Get neighboring patches for pathfinding"""
        if patch_id not in self.spatial_index._patches:
            return set()
        
        patch = self.spatial_index._patches[patch_id]
        
        # Use explicit neighbors if available
        if patch.neighbors:
            return patch.neighbors
        
        # Otherwise use spatial proximity
        point = patch.center
        return set(self.spatial_index.find_nearest_patches(point, k=8))
    
    def _reconstruct_path(self, came_from: Dict[int, int], current: int) -> List[int]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

class ConnectivityAnalyzer(BaseModel):
    """Analyze spatial connectivity and network properties"""
    
    model_config = {"validate_assignment": True}
    
    # Configuration
    spatial_index: SpatialIndex
    path_finder: PathFinder
    connectivity_threshold: float = 10.0
    
    def build_connectivity_graph(self, connectivity_type: ConnectivityType = ConnectivityType.PROXIMITY) -> nx.Graph:
        """Build connectivity graph based on specified criteria"""
        
        graph = nx.Graph()
        
        # Add all patches as nodes
        for patch_id, patch in self.spatial_index._patches.items():
            graph.add_node(patch_id, 
                          x=patch.center.x, 
                          y=patch.center.y,
                          quality=patch.quality,
                          area=patch.area)
        
        # Add edges based on connectivity type
        if connectivity_type == ConnectivityType.PROXIMITY:
            self._add_proximity_edges(graph)
        elif connectivity_type == ConnectivityType.ADJACENCY:
            self._add_adjacency_edges(graph)
        elif connectivity_type == ConnectivityType.VISIBILITY:
            self._add_visibility_edges(graph)
        elif connectivity_type == ConnectivityType.ACCESSIBILITY:
            self._add_accessibility_edges(graph)
        elif connectivity_type == ConnectivityType.FUNCTIONAL:
            self._add_functional_edges(graph)
        
        return graph
    
    def _add_proximity_edges(self, graph: nx.Graph) -> None:
        """Add edges based on proximity threshold"""
        patches = list(self.spatial_index._patches.values())
        
        for i, patch1 in enumerate(patches):
            for patch2 in patches[i+1:]:
                distance = patch1.center.distance_to(patch2.center)
                if distance <= self.connectivity_threshold:
                    graph.add_edge(patch1.patch_id, patch2.patch_id, weight=distance)
    
    def _add_adjacency_edges(self, graph: nx.Graph) -> None:
        """Add edges for directly adjacent patches"""
        for patch_id, patch in self.spatial_index._patches.items():
            for neighbor_id in patch.neighbors:
                if neighbor_id in self.spatial_index._patches:
                    distance = patch.center.distance_to(
                        self.spatial_index._patches[neighbor_id].center
                    )
                    graph.add_edge(patch_id, neighbor_id, weight=distance)
    
    def _add_visibility_edges(self, graph: nx.Graph) -> None:
        """Add edges based on line-of-sight visibility"""
        patches = list(self.spatial_index._patches.values())
        
        for i, patch1 in enumerate(patches):
            for patch2 in patches[i+1:]:
                if self._has_line_of_sight(patch1, patch2):
                    distance = patch1.center.distance_to(patch2.center)
                    graph.add_edge(patch1.patch_id, patch2.patch_id, weight=distance)
    
    def _add_accessibility_edges(self, graph: nx.Graph) -> None:
        """Add edges based on accessibility (accounting for barriers)"""
        patches = list(self.spatial_index._patches.values())
        
        for i, patch1 in enumerate(patches):
            for patch2 in patches[i+1:]:
                accessibility = min(patch1.accessibility, patch2.accessibility)
                if accessibility > 0.5:  # Threshold for accessibility
                    distance = patch1.center.distance_to(patch2.center)
                    # Weight by inverse accessibility
                    weight = distance / accessibility
                    graph.add_edge(patch1.patch_id, patch2.patch_id, weight=weight)
    
    def _add_functional_edges(self, graph: nx.Graph) -> None:
        """Add edges based on functional bee movement patterns"""
        patches = list(self.spatial_index._patches.values())
        
        for i, patch1 in enumerate(patches):
            for patch2 in patches[i+1:]:
                # Functional connectivity based on quality and distance
                distance = patch1.center.distance_to(patch2.center)
                if distance <= self.connectivity_threshold:
                    quality_factor = (patch1.quality + patch2.quality) / 2
                    functional_weight = distance / max(0.1, quality_factor)
                    graph.add_edge(patch1.patch_id, patch2.patch_id, weight=functional_weight)
    
    def _has_line_of_sight(self, patch1: SpatialPatch, patch2: SpatialPatch) -> bool:
        """Check if two patches have line of sight (simplified)"""
        # Simple implementation - check if there are barriers between patches
        
        if patch1.barriers or patch2.barriers:
            return False
        
        # For now, assume line of sight if within threshold and no barriers
        distance = patch1.center.distance_to(patch2.center)
        return distance <= self.connectivity_threshold
    
    def analyze_connectivity_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze connectivity metrics of the spatial network"""
        
        metrics = {}
        
        # Basic graph metrics
        metrics['num_nodes'] = graph.number_of_nodes()
        metrics['num_edges'] = graph.number_of_edges()
        metrics['density'] = nx.density(graph)
        
        # Connectivity metrics
        if graph.number_of_nodes() > 0:
            metrics['is_connected'] = nx.is_connected(graph)
            metrics['num_components'] = nx.number_connected_components(graph)
            
            if nx.is_connected(graph):
                metrics['diameter'] = nx.diameter(graph)
                metrics['average_shortest_path'] = nx.average_shortest_path_length(graph)
                metrics['radius'] = nx.radius(graph)
            
            # Centrality measures
            metrics['degree_centrality'] = nx.degree_centrality(graph)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(graph)
            metrics['closeness_centrality'] = nx.closeness_centrality(graph)
            
            # Clustering
            metrics['average_clustering'] = nx.average_clustering(graph)
            metrics['transitivity'] = nx.transitivity(graph)
        
        return metrics
    
    def find_critical_patches(self, graph: nx.Graph) -> List[int]:
        """Identify critical patches for connectivity"""
        
        if not nx.is_connected(graph):
            return []
        
        critical_patches = []
        original_components = nx.number_connected_components(graph)
        
        for node in graph.nodes():
            # Create a copy without this node
            temp_graph = graph.copy()
            temp_graph.remove_node(node)
            
            # Check if removing this node increases number of components
            new_components = nx.number_connected_components(temp_graph)
            if new_components > original_components:
                critical_patches.append(node)
        
        return critical_patches
    
    def identify_corridors(self, graph: nx.Graph, min_quality: float = 0.5) -> List[MovementCorridor]:
        """Identify movement corridors between high-quality patches"""
        
        corridors = []
        corridor_id = 0
        
        # Find high-quality patches
        high_quality_patches = []
        for patch_id, patch in self.spatial_index._patches.items():
            if patch.quality >= min_quality:
                high_quality_patches.append(patch_id)
        
        # Find corridors between high-quality patches
        for i, start_patch in enumerate(high_quality_patches):
            for end_patch in high_quality_patches[i+1:]:
                # Find shortest path
                try:
                    path = nx.shortest_path(graph, start_patch, end_patch, weight='weight')
                    if len(path) > 2:  # Only consider non-direct connections
                        # Create waypoints
                        waypoints = []
                        for patch_id in path:
                            if patch_id in self.spatial_index._patches:
                                waypoints.append(self.spatial_index._patches[patch_id].center)
                        
                        corridor = MovementCorridor(
                            corridor_id=corridor_id,
                            start_patch=start_patch,
                            end_patch=end_patch,
                            waypoints=waypoints
                        )
                        corridors.append(corridor)
                        corridor_id += 1
                
                except nx.NetworkXNoPath:
                    continue  # No path exists
        
        return corridors

class LandscapeAnalyzer(BaseModel):
    """Comprehensive landscape analysis for bee habitat assessment"""
    
    model_config = {"validate_assignment": True}
    
    # Configuration
    spatial_index: SpatialIndex
    connectivity_analyzer: ConnectivityAnalyzer
    
    def analyze_patch_isolation(self) -> Dict[int, float]:
        """Analyze isolation of each patch"""
        
        isolation_metrics = {}
        
        for patch_id, patch in self.spatial_index._patches.items():
            # Find nearest neighbors
            neighbors = self.spatial_index.find_nearest_patches(patch.center, k=5)
            neighbors = [n for n in neighbors if n != patch_id]
            
            if not neighbors:
                isolation_metrics[patch_id] = float('inf')
                continue
            
            # Calculate average distance to nearest neighbors
            distances = []
            for neighbor_id in neighbors:
                if neighbor_id in self.spatial_index._patches:
                    neighbor = self.spatial_index._patches[neighbor_id]
                    distance = patch.center.distance_to(neighbor.center)
                    distances.append(distance)
            
            isolation_metrics[patch_id] = np.mean(distances) if distances else float('inf')
        
        return isolation_metrics
    
    def calculate_landscape_fragmentation(self) -> Dict[str, float]:
        """Calculate landscape fragmentation metrics"""
        
        patches = list(self.spatial_index._patches.values())
        
        if not patches:
            return {}
        
        # Total landscape area
        total_area = sum(patch.area for patch in patches)
        
        # Number of patches
        num_patches = len(patches)
        
        # Mean patch size
        mean_patch_size = total_area / num_patches
        
        # Patch size standard deviation
        patch_sizes = [patch.area for patch in patches]
        patch_size_std = np.std(patch_sizes)
        
        # Edge density calculation (simplified)
        total_perimeter = sum(patch.perimeter for patch in patches)
        edge_density = total_perimeter / total_area
        
        # Connectivity index
        graph = self.connectivity_analyzer.build_connectivity_graph()
        connectivity_index = nx.density(graph) if graph.number_of_nodes() > 0 else 0
        
        return {
            'num_patches': num_patches,
            'total_area': total_area,
            'mean_patch_size': mean_patch_size,
            'patch_size_std': patch_size_std,
            'edge_density': edge_density,
            'connectivity_index': connectivity_index
        }
    
    def identify_habitat_clusters(self, min_quality: float = 0.6) -> List[List[int]]:
        """Identify clusters of high-quality habitat patches"""
        
        # Get high-quality patches
        high_quality_patches = []
        coordinates = []
        
        for patch_id, patch in self.spatial_index._patches.items():
            if patch.quality >= min_quality:
                high_quality_patches.append(patch_id)
                coordinates.append([patch.center.x, patch.center.y])
        
        if len(coordinates) < 2:
            return [high_quality_patches] if high_quality_patches else []
        
        # Apply DBSCAN clustering
        coordinates = np.array(coordinates)
        clustering = DBSCAN(eps=self.connectivity_analyzer.connectivity_threshold, 
                           min_samples=2).fit(coordinates)
        
        # Group patches by cluster
        clusters = defaultdict(list)
        for i, cluster_id in enumerate(clustering.labels_):
            if cluster_id != -1:  # -1 indicates noise/outliers
                clusters[cluster_id].append(high_quality_patches[i])
        
        return list(clusters.values())
    
    def assess_landscape_quality(self) -> Dict[str, float]:
        """Comprehensive landscape quality assessment"""
        
        patches = list(self.spatial_index._patches.values())
        
        if not patches:
            return {}
        
        # Quality metrics
        qualities = [patch.quality for patch in patches]
        mean_quality = np.mean(qualities)
        quality_variance = np.var(qualities)
        
        # Resource density metrics
        densities = [patch.resource_density for patch in patches]
        mean_density = np.mean(densities)
        
        # Accessibility metrics
        accessibilities = [patch.accessibility for patch in patches]
        mean_accessibility = np.mean(accessibilities)
        
        # Spatial distribution of quality
        high_quality_patches = sum(1 for q in qualities if q > 0.7)
        low_quality_patches = sum(1 for q in qualities if q < 0.3)
        
        quality_balance = 1.0 - abs(high_quality_patches - low_quality_patches) / len(patches)
        
        # Connectivity quality
        graph = self.connectivity_analyzer.build_connectivity_graph(ConnectivityType.FUNCTIONAL)
        connectivity_quality = nx.density(graph) if graph.number_of_nodes() > 0 else 0
        
        return {
            'mean_quality': mean_quality,
            'quality_variance': quality_variance,
            'mean_resource_density': mean_density,
            'mean_accessibility': mean_accessibility,
            'quality_balance': quality_balance,
            'connectivity_quality': connectivity_quality,
            'high_quality_patch_ratio': high_quality_patches / len(patches),
            'low_quality_patch_ratio': low_quality_patches / len(patches)
        }


def create_spatial_analysis_system(grid_size: float = 10.0, 
                                 connectivity_threshold: float = 15.0) -> Tuple[SpatialIndex, PathFinder, ConnectivityAnalyzer, LandscapeAnalyzer]:
    """Factory function to create complete spatial analysis system"""
    
    # Create spatial index
    spatial_index = SpatialIndex(grid_size=grid_size)
    
    # Create pathfinder
    path_finder = PathFinder(spatial_index=spatial_index)
    
    # Create connectivity analyzer
    connectivity_analyzer = ConnectivityAnalyzer(
        spatial_index=spatial_index,
        path_finder=path_finder,
        connectivity_threshold=connectivity_threshold
    )
    
    # Create landscape analyzer
    landscape_analyzer = LandscapeAnalyzer(
        spatial_index=spatial_index,
        connectivity_analyzer=connectivity_analyzer
    )
    
    return spatial_index, path_finder, connectivity_analyzer, landscape_analyzer