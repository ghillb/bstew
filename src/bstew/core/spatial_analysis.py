"""
Advanced Spatial Analysis Systems for BSTEW
==========================================

Enhanced patch relationships, spatial queries, and connectivity analysis
matching NetLogo's complex spatial modeling capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
from scipy.spatial import cKDTree, distance_matrix
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import json


class PatchType(Enum):
    """Types of patches in the landscape"""

    NEST = "nest"
    FLOWER = "flower"
    NECTAR_SOURCE = "nectar_source"
    POLLEN_SOURCE = "pollen_source"
    SHELTER = "shelter"
    OBSTACLE = "obstacle"
    WATER = "water"
    OPEN = "open"
    MANAGED = "managed"
    NATURAL = "natural"


class ConnectivityType(Enum):
    """Types of connectivity between patches"""

    DIRECT = "direct"
    FUNCTIONAL = "functional"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    EUCLIDEAN = "euclidean"
    NETWORK = "network"


@dataclass
class PatchConnection:
    """Connection between two patches"""

    patch1_id: int
    patch2_id: int
    distance: float
    connectivity_type: ConnectivityType
    strength: float
    cost: float
    traversal_time: float
    attributes: Dict[str, Any]


@dataclass
class PatchNeighborhood:
    """Neighborhood definition for a patch"""

    patch_id: int
    neighbors: List[int]
    distances: List[float]
    weights: List[float]
    neighborhood_type: str
    radius: float


class SpatialPatch(BaseModel):
    """Enhanced spatial patch with connectivity information"""

    model_config = {"validate_assignment": True}

    # Basic patch properties
    patch_id: int = Field(..., description="Unique patch identifier")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    patch_type: PatchType = Field(default=PatchType.OPEN, description="Type of patch")

    # Spatial properties
    area: float = Field(default=1.0, ge=0.0, description="Patch area")
    perimeter: float = Field(default=4.0, ge=0.0, description="Patch perimeter")
    elevation: float = Field(default=0.0, description="Elevation above sea level")
    slope: float = Field(default=0.0, ge=0.0, description="Slope angle")
    aspect: float = Field(default=0.0, ge=0.0, le=360.0, description="Aspect direction")

    # Ecological properties
    habitat_quality: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Habitat quality"
    )
    resource_density: float = Field(default=0.0, ge=0.0, description="Resource density")
    carrying_capacity: int = Field(default=100, ge=0, description="Carrying capacity")

    # Connectivity properties
    connectivity_index: float = Field(
        default=0.0, ge=0.0, description="Overall connectivity index"
    )
    centrality_score: float = Field(
        default=0.0, ge=0.0, description="Network centrality score"
    )
    isolation_index: float = Field(default=0.0, ge=0.0, description="Isolation index")

    # Dynamic properties
    is_accessible: bool = Field(default=True, description="Whether patch is accessible")
    management_status: str = Field(default="unmanaged", description="Management status")
    disturbance_level: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Disturbance level"
    )

    # Temporal properties
    seasonal_availability: Dict[str, float] = Field(
        default_factory=dict, description="Seasonal availability"
    )
    phenology_stage: str = Field(
        default="active", description="Current phenology stage"
    )

    # Adjacency information
    adjacent_patches: List[int] = Field(
        default_factory=list, description="Adjacent patch IDs"
    )
    neighbor_distances: List[float] = Field(
        default_factory=list, description="Distances to neighbors"
    )
    connection_strengths: List[float] = Field(
        default_factory=list, description="Connection strengths"
    )


class PatchConnectivityAnalyzer:
    """Analyzes patch connectivity and relationships"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.patches: Dict[int, SpatialPatch] = {}
        self.connections: List[PatchConnection] = []
        self.connectivity_matrix: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
        self.spatial_tree: Optional[cKDTree] = None
        self.network_graph: Optional[nx.Graph] = None

    def add_patch(self, patch: SpatialPatch) -> None:
        """Add a patch to the analysis system"""
        self.patches[patch.patch_id] = patch

    def add_patches(self, patches: List[SpatialPatch]) -> None:
        """Add multiple patches to the analysis system"""
        for patch in patches:
            self.add_patch(patch)

    def build_spatial_index(self) -> None:
        """Build spatial index for efficient spatial queries"""
        if not self.patches:
            return

        # Extract coordinates
        coords = np.array([[p.x, p.y] for p in self.patches.values()])

        # Build KDTree
        self.spatial_tree = cKDTree(coords)

        self.logger.info(f"Built spatial index for {len(self.patches)} patches")

    def calculate_distance_matrix(self) -> None:
        """Calculate distance matrix between all patches"""
        if not self.patches:
            return

        # Extract coordinates
        coords = np.array([[p.x, p.y] for p in self.patches.values()])

        # Calculate Euclidean distances
        self.distance_matrix = distance_matrix(coords, coords)

        self.logger.info(f"Calculated distance matrix for {len(self.patches)} patches")

    def define_neighborhoods(
        self,
        radius: Optional[float] = None,
        k_neighbors: Optional[int] = None,
        connectivity_threshold: Optional[float] = None,
    ) -> Dict[int, PatchNeighborhood]:
        """Define neighborhoods for each patch"""
        neighborhoods = {}

        if not self.spatial_tree:
            self.build_spatial_index()

        patch_ids = list(self.patches.keys())
        coords = np.array([[p.x, p.y] for p in self.patches.values()])

        for i, patch_id in enumerate(patch_ids):
            if radius is not None and self.spatial_tree is not None:
                # Radius-based neighborhood
                neighbor_indices = self.spatial_tree.query_ball_point(coords[i], radius)
                neighbor_indices = [idx for idx in neighbor_indices if idx != i]

                if neighbor_indices and self.distance_matrix is not None:
                    distances = [
                        self.distance_matrix[i, idx] for idx in neighbor_indices
                    ]
                    weights = [
                        1.0 / (d + 1e-6) for d in distances
                    ]  # Inverse distance weights
                    neighbor_ids = [patch_ids[idx] for idx in neighbor_indices]

                    neighborhoods[patch_id] = PatchNeighborhood(
                        patch_id=patch_id,
                        neighbors=neighbor_ids,
                        distances=distances,
                        weights=weights,
                        neighborhood_type="radius",
                        radius=radius,
                    )

            elif k_neighbors is not None:
                # K-nearest neighbors
                if self.distance_matrix is None:
                    self.calculate_distance_matrix()

                if self.distance_matrix is not None:
                    distances = self.distance_matrix[i, :].tolist()
                    # Get k+1 nearest (excluding self)
                    nearest_indices = np.argsort(distances)[1 : k_neighbors + 1]

                    neighbor_distances = [
                        float(distances[idx]) for idx in nearest_indices
                    ]
                    weights = [1.0 / (d + 1e-6) for d in neighbor_distances]
                    neighbor_ids = [patch_ids[idx] for idx in nearest_indices]

                    neighborhoods[patch_id] = PatchNeighborhood(
                        patch_id=patch_id,
                        neighbors=neighbor_ids,
                        distances=neighbor_distances,
                        weights=weights,
                        neighborhood_type="k_nearest",
                        radius=max(neighbor_distances) if neighbor_distances else 0.0,
                    )

        return neighborhoods

    def calculate_connectivity_indices(self) -> None:
        """Calculate various connectivity indices for patches"""
        if self.distance_matrix is None:
            self.calculate_distance_matrix()

        patch_ids = list(self.patches.keys())
        n_patches = len(patch_ids)

        # Calculate connectivity matrix based on inverse distance
        connectivity_matrix = np.zeros((n_patches, n_patches))

        for i in range(n_patches):
            for j in range(n_patches):
                if i != j and self.distance_matrix is not None:
                    dist = self.distance_matrix[i, j]
                    # Exponential decay connectivity
                    connectivity_matrix[i, j] = np.exp(
                        -dist / 1000.0
                    )  # 1000m characteristic distance

        self.connectivity_matrix = connectivity_matrix

        # Calculate connectivity indices for each patch
        for i, patch_id in enumerate(patch_ids):
            patch = self.patches[patch_id]

            # Total connectivity (sum of connections to all other patches)
            total_connectivity = float(np.sum(connectivity_matrix[i, :]))

            # Isolation index (inverse of connectivity)
            isolation_index = 1.0 / (total_connectivity + 1e-6)

            # Update patch
            patch.connectivity_index = total_connectivity
            patch.isolation_index = isolation_index

        self.logger.info(f"Calculated connectivity indices for {n_patches} patches")

    def build_network_graph(self, connectivity_threshold: float = 0.1) -> nx.Graph:
        """Build network graph from patch connectivity"""
        if self.connectivity_matrix is None:
            self.calculate_connectivity_indices()

        # Create network graph
        graph: nx.Graph = nx.Graph()
        patch_ids = list(self.patches.keys())

        # Add nodes (patches)
        for patch_id in patch_ids:
            patch = self.patches[patch_id]
            graph.add_node(
                patch_id,
                x=patch.x,
                y=patch.y,
                patch_type=patch.patch_type.value,
                habitat_quality=patch.habitat_quality,
                area=patch.area,
            )

        # Add edges (connections)
        n_patches = len(patch_ids)
        for i in range(n_patches):
            for j in range(i + 1, n_patches):
                if self.connectivity_matrix is not None:
                    connectivity = self.connectivity_matrix[i, j]
                else:
                    continue

                if (
                    float(connectivity) > connectivity_threshold
                    and self.distance_matrix is not None
                ):
                    distance = float(self.distance_matrix[i, j])

                    graph.add_edge(
                        patch_ids[i],
                        patch_ids[j],
                        weight=float(connectivity),
                        distance=distance,
                        connectivity=float(connectivity),
                    )

        self.network_graph = graph

        # Calculate network centrality measures
        self._calculate_network_centrality()

        return graph

    def _calculate_network_centrality(self) -> None:
        """Calculate network centrality measures"""
        if not self.network_graph:
            return

        # Calculate various centrality measures
        betweenness = nx.betweenness_centrality(self.network_graph, weight="distance")
        closeness = nx.closeness_centrality(self.network_graph, distance="distance")
        degree = nx.degree_centrality(self.network_graph)
        eigenvector = nx.eigenvector_centrality(
            self.network_graph, weight="connectivity"
        )

        # Update patch centrality scores
        for patch_id in self.patches:
            if patch_id in betweenness:
                centrality_score = (
                    betweenness[patch_id] * 0.3
                    + closeness[patch_id] * 0.3
                    + degree[patch_id] * 0.2
                    + eigenvector[patch_id] * 0.2
                )
                self.patches[patch_id].centrality_score = centrality_score

    def identify_patch_clusters(
        self,
        method: str = "dbscan",
        eps: float = 500.0,
        min_samples: int = 3,
        n_clusters: int = 5,
    ) -> Dict[str, Any]:
        """Identify clusters of patches"""
        if not self.patches:
            return {
                "clusters": {},
                "n_clusters": 0,
                "method": method,
                "parameters": {"eps": eps, "min_samples": min_samples},
            }

        # Extract coordinates and attributes
        coords = np.array([[p.x, p.y] for p in self.patches.values()])
        attributes = np.array(
            [
                [p.habitat_quality, p.resource_density, p.area]
                for p in self.patches.values()
            ]
        )

        # Combine spatial and attribute information
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        attributes_scaled = scaler.fit_transform(attributes)

        # Combined feature matrix
        features = np.hstack([coords_scaled, attributes_scaled])

        # Apply clustering
        if method == "dbscan":
            clustering = DBSCAN(eps=eps / 1000.0, min_samples=min_samples)
        elif method == "kmeans":
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        cluster_labels = clustering.fit_predict(features)

        # Create patch ID to cluster mapping
        patch_ids = list(self.patches.keys())
        cluster_mapping = {
            patch_ids[i]: int(cluster_labels[i]) if cluster_labels[i] >= 0 else -1
            for i in range(len(patch_ids))
        }

        self.logger.info(
            f"Identified {len(set(cluster_labels))} clusters using {method}"
        )

        # Return structured result as specified in the spec
        return {
            "clusters": cluster_mapping,
            "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "method": method,
            "parameters": {"eps": eps, "min_samples": min_samples},
        }

    def analyze_connectivity_corridors(
        self,
        source_patches: List[int],
        target_patches: List[int],
        max_cost: float = 10000.0,
    ) -> Dict[str, Any]:
        """Analyze connectivity corridors between patch sets"""
        if not self.network_graph:
            self.build_network_graph()

        corridors: List[Dict[str, Any]] = []

        # Check if network graph exists
        if self.network_graph is None:
            return {
                "n_corridors": 0,
                "corridors": [],
                "average_length": 0,
                "average_steps": 0,
                "connectivity_index": 0,
            }

        # Find shortest paths between all source-target pairs
        for source_id in source_patches:
            for target_id in target_patches:
                if source_id != target_id:
                    try:
                        # Find shortest path using distance as weight
                        path = nx.shortest_path(
                            self.network_graph,
                            source=source_id,
                            target=target_id,
                            weight="distance",
                        )

                        # Calculate path metrics
                        path_length = nx.shortest_path_length(
                            self.network_graph,
                            source=source_id,
                            target=target_id,
                            weight="distance",
                        )

                        if path_length <= max_cost:
                            corridors.append(
                                {
                                    "source": source_id,
                                    "target": target_id,
                                    "path": path,
                                    "length": path_length,
                                    "n_steps": len(path) - 1,
                                }
                            )

                    except nx.NetworkXNoPath:
                        # No path exists between these patches
                        continue

        # Analyze corridor characteristics
        analysis = {
            "n_corridors": len(corridors),
            "corridors": corridors,
            "average_length": np.mean([c["length"] for c in corridors])
            if corridors
            else 0,
            "average_steps": np.mean([c["n_steps"] for c in corridors])
            if corridors
            else 0,
            "connectivity_index": len(corridors)
            / (len(source_patches) * len(target_patches)),
        }

        return analysis

    def calculate_landscape_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive landscape metrics"""
        # Calculate basic landscape metrics
        total_area = sum(p.area for p in self.patches.values())
        patch_areas = [p.area for p in self.patches.values()]

        metrics = {
            "total_area": total_area,
            "patch_density": len(self.patches) / total_area if total_area > 0 else 0,
            "mean_patch_size": np.mean(patch_areas) if patch_areas else 0,
            "landscape_diversity": self._calculate_shannon_diversity(),
            "connectivity_index": self._calculate_overall_connectivity(),
            "fragmentation_index": self._calculate_fragmentation_index(),
        }

        # Build network graph if not exists
        if self.network_graph is None:
            self.build_network_graph()

        if self.network_graph:
            # Basic network metrics
            n_patches = len(self.patches)
            n_connections = self.network_graph.number_of_edges()

            # Calculate component sizes
            components = list(nx.connected_components(self.network_graph))
            component_sizes = [len(comp) for comp in components]

            # Calculate edge density
            total_perimeter = sum(p.perimeter for p in self.patches.values())
            edge_density = total_perimeter / total_area if total_area > 0 else 0

            # Calculate betweenness centrality
            betweenness = nx.betweenness_centrality(self.network_graph)
            avg_betweenness = np.mean(list(betweenness.values())) if betweenness else 0

            metrics.update(
                {
                    "largest_component_size": max(component_sizes)
                    if component_sizes
                    else 0,
                    "n_components": len(components),
                    "mean_component_size": np.mean(component_sizes)
                    if component_sizes
                    else 0,
                    "network_density": nx.density(self.network_graph),
                    "average_clustering": nx.average_clustering(self.network_graph),
                    "edge_density": edge_density,
                    "betweenness_centrality": avg_betweenness,
                    "clustering_coefficient": nx.average_clustering(self.network_graph),
                    "n_patches": n_patches,
                    "n_connections": n_connections,
                    "connection_density": n_connections
                    / (n_patches * (n_patches - 1) / 2)
                    if n_patches > 1
                    else 0,
                    "average_degree": self._calculate_average_degree(),
                    "isolation_index": 1.0 - (max(component_sizes) / n_patches)
                    if n_patches > 0 and component_sizes
                    else 1.0,
                }
            )

            # Diameter and efficiency for largest component
            if len(components) > 0:
                largest_comp = max(components, key=len)
                if len(largest_comp) > 1:
                    largest_component = self.network_graph.subgraph(largest_comp)
                    try:
                        diameter = nx.diameter(largest_component)
                        efficiency = nx.global_efficiency(largest_component)
                        metrics.update({"diameter": diameter, "efficiency": efficiency})
                    except Exception:
                        metrics.update({"diameter": float("inf"), "efficiency": 0.0})
                else:
                    metrics.update({"diameter": float("inf"), "efficiency": 0.0})
        else:
            # Provide default values when no network exists
            metrics.update(
                {
                    "largest_component_size": 0,
                    "n_components": 0,
                    "mean_component_size": 0,
                    "network_density": 0,
                    "average_clustering": 0,
                    "edge_density": 0,
                    "betweenness_centrality": 0,
                    "clustering_coefficient": 0,
                    "n_patches": len(self.patches),
                    "n_connections": 0,
                    "connection_density": 0,
                    "average_degree": 0,
                    "isolation_index": 1.0,
                    "diameter": float("inf"),
                    "efficiency": 0.0,
                }
            )

        return metrics

    def _calculate_average_degree(self) -> float:
        """Calculate average degree handling type issues"""
        try:
            if self.network_graph is None or self.network_graph.number_of_nodes() == 0:
                return 0.0
            # Handle both cases: when degree() returns a view or an int
            degree_data = self.network_graph.degree()
            if hasattr(degree_data, "__iter__") and not isinstance(degree_data, int):
                degrees = [d for _, d in degree_data]
                return float(np.mean(degrees)) if degrees else 0.0
            else:
                return 0.0
        except (TypeError, AttributeError):
            return 0.0

    def _calculate_shannon_diversity(self) -> float:
        """Calculate Shannon diversity index for patch types"""
        if not self.patches:
            return 0.0

        # Count patches by type
        type_counts: Dict[str, int] = {}
        for patch in self.patches.values():
            patch_type = patch.patch_type.value
            type_counts[patch_type] = type_counts.get(patch_type, 0) + 1

        # Calculate proportions
        total_patches = len(self.patches)
        shannon = 0.0
        for count in type_counts.values():
            if count > 0:
                proportion = count / total_patches
                shannon -= proportion * np.log(proportion)

        return shannon

    def _calculate_overall_connectivity(self) -> float:
        """Calculate overall connectivity index"""
        if not self.patches:
            return 0.0

        # Use mean connectivity index of all patches
        connectivity_values = [p.connectivity_index for p in self.patches.values()]
        return float(np.mean(connectivity_values)) if connectivity_values else 0.0

    def _calculate_fragmentation_index(self) -> float:
        """Calculate landscape fragmentation index"""
        if not self.patches:
            return 0.0

        # Fragmentation based on patch size variance and isolation
        patch_areas = [p.area for p in self.patches.values()]
        if not patch_areas:
            return 0.0

        # Coefficient of variation in patch sizes
        cv = (
            np.std(patch_areas) / np.mean(patch_areas)
            if np.mean(patch_areas) > 0
            else 0
        )

        # Mean isolation index
        isolation_values = [p.isolation_index for p in self.patches.values()]
        mean_isolation = np.mean(isolation_values) if isolation_values else 0.0

        # Combined fragmentation index (0-1 scale)
        fragmentation = (cv + mean_isolation) / 2.0
        return float(min(float(fragmentation), 1.0))

    def export_connectivity_data(self, output_directory: str) -> None:
        """Export connectivity analysis results"""
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export patch data
        patch_data = []
        for patch in self.patches.values():
            patch_data.append(
                {
                    "patch_id": patch.patch_id,
                    "x": patch.x,
                    "y": patch.y,
                    "patch_type": patch.patch_type.value,
                    "area": patch.area,
                    "habitat_quality": patch.habitat_quality,
                    "connectivity_index": patch.connectivity_index,
                    "centrality_score": patch.centrality_score,
                    "isolation_index": patch.isolation_index,
                }
            )

        patch_df = pd.DataFrame(patch_data)
        patch_df.to_csv(output_path / "patch_connectivity.csv", index=False)

        # Export connection data
        if self.network_graph:
            connection_data = []
            for edge in self.network_graph.edges(data=True):
                connection_data.append(
                    {
                        "patch1": edge[0],
                        "patch2": edge[1],
                        "distance": edge[2]["distance"],
                        "connectivity": edge[2]["connectivity"],
                        "weight": edge[2]["weight"],
                    }
                )

            connection_df = pd.DataFrame(connection_data)
            connection_df.to_csv(output_path / "patch_connections.csv", index=False)

        # Export landscape metrics
        landscape_metrics = self.calculate_landscape_metrics()
        with open(output_path / "landscape_metrics.json", "w") as f:
            json.dump(landscape_metrics, f, indent=2)

        # Export network graph
        if self.network_graph:
            nx.write_gml(self.network_graph, output_path / "network_graph.gml")

        self.logger.info(f"Connectivity data exported to {output_path}")

    def multi_criteria_search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform multi-criteria search on patches"""
        results: List[Dict[str, Any]] = []

        for patch_id, patch in self.patches.items():
            match = True
            for criterion, value in criteria.items():
                # Simple criteria matching - can be expanded
                if criterion == "quality" and hasattr(patch, "quality"):
                    if patch.quality < value:
                        match = False
                        break
                elif criterion == "distance" and hasattr(patch, "distance"):
                    if patch.distance > value:
                        match = False
                        break

            if match:
                results.append(
                    {
                        "patch_id": patch_id,
                        "location": (patch.x, patch.y),
                        "patch": patch,
                    }
                )

        return results

    def update_agent_positions(self, agents: List[Any]) -> None:
        """Update agent positions in the spatial system"""
        for agent in agents:
            if hasattr(agent, "location") and hasattr(agent, "patch_id"):
                # Update agent's patch assignment based on location
                closest_patch = self._find_closest_patch(agent.location)
                if closest_patch:
                    agent.patch_id = closest_patch.patch_id

    def _find_closest_patch(
        self, location: Tuple[float, float]
    ) -> Optional[SpatialPatch]:
        """Find the closest patch to a given location"""
        if not self.patches:
            return None

        min_distance = float("inf")
        closest_patch = None

        for patch in self.patches.values():
            distance = (
                (patch.x - location[0]) ** 2 + (patch.y - location[1]) ** 2
            ) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_patch = patch

        return closest_patch


class AdvancedSpatialQueries:
    """Advanced spatial query capabilities"""

    def __init__(self, patches: Dict[int, SpatialPatch]):
        self.patches = patches
        self.spatial_tree: Optional[cKDTree] = None
        self.logger = logging.getLogger(__name__)
        self._build_spatial_index()

    def _build_spatial_index(self) -> None:
        """Build spatial index for efficient queries"""
        if not self.patches:
            return

        coords = np.array([[p.x, p.y] for p in self.patches.values()])
        self.spatial_tree = cKDTree(coords)

    def radius_query(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Find patches within radius of a point"""
        if not self.spatial_tree:
            return []

        # Query spatial index
        indices = self.spatial_tree.query_ball_point([center_x, center_y], radius)

        # Get patch IDs
        patch_ids = list(self.patches.keys())
        result_patches = [patch_ids[i] for i in indices]

        # Apply filters if provided
        if filters:
            result_patches = self._apply_filters(result_patches, filters)

        return result_patches

    def multi_criteria_search(
        self,
        spatial_criteria: Dict[str, Any],
        attribute_criteria: Dict[str, Any],
        ranking_criteria: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-criteria spatial search with ranking"""
        results: List[Dict[str, Any]] = []

        # Start with all patches
        candidate_patches = list(self.patches.keys())

        # Apply spatial criteria
        if "center" in spatial_criteria and "radius" in spatial_criteria:
            center = spatial_criteria["center"]
            radius = spatial_criteria["radius"]
            candidate_patches = self.radius_query(center[0], center[1], radius)

        # Apply attribute filters
        candidate_patches = self._apply_filters(candidate_patches, attribute_criteria)

        # Calculate scores and rank
        for patch_id in candidate_patches:
            patch = self.patches[patch_id]

            # Calculate spatial score
            spatial_score = self._calculate_spatial_score(patch, spatial_criteria)

            # Calculate attribute score
            attribute_score = self._calculate_attribute_score(patch, attribute_criteria)

            # Calculate ranking score
            ranking_score = (
                self._calculate_ranking_score(patch, ranking_criteria)
                if ranking_criteria
                else 0.5
            )

            # Combined score
            total_score = (
                spatial_score * 0.3 + attribute_score * 0.4 + ranking_score * 0.3
            )

            results.append(
                {
                    "patch_id": patch_id,
                    "patch": patch,
                    "spatial_score": spatial_score,
                    "attribute_score": attribute_score,
                    "ranking_score": ranking_score,
                    "total_score": total_score,
                }
            )

        # Sort by total score
        results.sort(key=lambda x: float(x.get("total_score", 0)), reverse=True)

        return results

    def _apply_filters(
        self, patch_ids: List[int], filters: Dict[str, Any]
    ) -> List[int]:
        """Apply attribute filters to patch list"""
        filtered_patches = []

        for patch_id in patch_ids:
            patch = self.patches[patch_id]
            include_patch = True

            for attr_name, criteria in filters.items():
                if hasattr(patch, attr_name):
                    value = getattr(patch, attr_name)

                    if isinstance(criteria, dict):
                        # Range or comparison criteria
                        if "min" in criteria and value < criteria["min"]:
                            include_patch = False
                            break
                        if "max" in criteria and value > criteria["max"]:
                            include_patch = False
                            break
                        if "equals" in criteria and value != criteria["equals"]:
                            include_patch = False
                            break
                    else:
                        # Direct comparison
                        if value != criteria:
                            include_patch = False
                            break

            if include_patch:
                filtered_patches.append(patch_id)

        return filtered_patches

    def _calculate_spatial_score(
        self, patch: SpatialPatch, spatial_criteria: Dict[str, Any]
    ) -> float:
        """Calculate spatial score for a patch"""
        score = 0.5  # Default score

        if "center" in spatial_criteria and "radius" in spatial_criteria:
            center = spatial_criteria["center"]
            radius = spatial_criteria["radius"]

            # Distance from center
            distance = np.sqrt((patch.x - center[0]) ** 2 + (patch.y - center[1]) ** 2)

            # Normalized distance score (closer is better)
            if distance <= radius:
                score = 1.0 - (distance / radius)
            else:
                score = 0.0

        return score

    def _calculate_attribute_score(
        self, patch: SpatialPatch, attribute_criteria: Dict[str, Any]
    ) -> float:
        """Calculate attribute score for a patch"""
        scores = []

        for attr_name, criteria in attribute_criteria.items():
            if hasattr(patch, attr_name):
                value = getattr(patch, attr_name)

                if attr_name == "habitat_quality":
                    scores.append(value)
                elif attr_name == "resource_density":
                    scores.append(min(value / 10.0, 1.0))  # Normalize to 0-1
                elif attr_name == "area":
                    scores.append(min(value / 100.0, 1.0))  # Normalize to 0-1
                else:
                    scores.append(0.5)  # Default score

        return np.mean(scores) if scores else 0.5

    def _calculate_ranking_score(
        self, patch: SpatialPatch, ranking_criteria: Dict[str, float]
    ) -> float:
        """Calculate ranking score based on weighted criteria"""
        score = 0.0
        total_weight = 0.0

        for attr_name, weight in ranking_criteria.items():
            if hasattr(patch, attr_name):
                value = getattr(patch, attr_name)

                # Normalize different attributes
                if attr_name == "connectivity_index":
                    normalized_value = min(value / 10.0, 1.0)
                elif attr_name == "centrality_score":
                    normalized_value = value
                elif attr_name == "habitat_quality":
                    normalized_value = value
                else:
                    normalized_value = 0.5

                score += normalized_value * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.5


class GISDataIntegrator:
    """GIS data import and integration capabilities"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def import_shapefile(
        self,
        shapefile_path: str,
        patch_id_field: str = "id",
        coordinate_system: str = "EPSG:4326",
    ) -> List[SpatialPatch]:
        """Import patches from shapefile"""
        try:
            import geopandas as gpd

            # Read shapefile
            gdf = gpd.read_file(shapefile_path)

            # Ensure coordinate system
            if gdf.crs is None:
                gdf = gdf.set_crs(coordinate_system)
            elif gdf.crs.to_string() != coordinate_system:
                gdf = gdf.to_crs(coordinate_system)

            patches = []

            for idx, row in gdf.iterrows():
                geom = row.geometry

                # Get centroid coordinates
                centroid = geom.centroid
                x, y = centroid.x, centroid.y

                # Calculate area and perimeter
                area = geom.area
                perimeter = geom.length

                # Get patch ID
                patch_id = row.get(patch_id_field, idx)

                # Extract other attributes
                patch_type = PatchType.OPEN
                if "patch_type" in row:
                    try:
                        patch_type = PatchType(row["patch_type"])
                    except ValueError:
                        patch_type = PatchType.OPEN

                habitat_quality = row.get("habitat_quality", 0.5)
                resource_density = row.get("resource_density", 0.0)
                elevation = row.get("elevation", 0.0)

                # Create patch
                patch = SpatialPatch(
                    patch_id=int(patch_id),
                    x=x,
                    y=y,
                    patch_type=patch_type,
                    area=area,
                    perimeter=perimeter,
                    elevation=elevation,
                    habitat_quality=float(habitat_quality),
                    resource_density=float(resource_density),
                )

                patches.append(patch)

            self.logger.info(f"Imported {len(patches)} patches from shapefile")
            return patches

        except ImportError:
            self.logger.error("GeoPandas not available for shapefile import")
            return []
        except Exception as e:
            self.logger.error(f"Error importing shapefile: {e}")
            return []

    def import_raster_data(
        self,
        raster_path: str,
        patches: List[SpatialPatch],
        attribute_name: str = "raster_value",
    ) -> List[SpatialPatch]:
        """Import raster data and assign values to patches"""
        try:
            import rasterio

            # Open raster
            with rasterio.open(raster_path) as src:
                # Sample raster at patch locations
                coords = [(patch.x, patch.y) for patch in patches]

                # Sample raster values
                raster_values = list(src.sample(coords))

                # Update patches with raster values
                for patch, value in zip(patches, raster_values):
                    if hasattr(patch, attribute_name):
                        setattr(patch, attribute_name, float(value[0]))
                    else:
                        # For elevation, habitat_quality, etc.
                        if attribute_name == "elevation":
                            patch.elevation = float(value[0])
                        elif attribute_name == "habitat_quality":
                            patch.habitat_quality = min(max(float(value[0]), 0.0), 1.0)
                        elif attribute_name == "resource_density":
                            patch.resource_density = max(float(value[0]), 0.0)

            self.logger.info(f"Imported raster data for {len(patches)} patches")
            return patches

        except ImportError:
            self.logger.error("Rasterio not available for raster import")
            return patches
        except Exception as e:
            self.logger.error(f"Error importing raster data: {e}")
            return patches

    def import_csv_coordinates(
        self,
        csv_path: str,
        x_field: str = "x",
        y_field: str = "y",
        id_field: str = "id",
    ) -> List[SpatialPatch]:
        """Import patch coordinates from CSV file"""
        try:
            df = pd.read_csv(csv_path)

            patches = []

            for idx, row in df.iterrows():
                x = row[x_field]
                y = row[y_field]
                patch_id = row.get(id_field, idx)

                # Extract other attributes if present
                patch_type = PatchType.OPEN
                if "patch_type" in row:
                    try:
                        patch_type = PatchType(row["patch_type"])
                    except ValueError:
                        patch_type = PatchType.OPEN

                area = row.get("area", 1.0)
                habitat_quality = row.get("habitat_quality", 0.5)
                resource_density = row.get("resource_density", 0.0)
                elevation = row.get("elevation", 0.0)

                patch = SpatialPatch(
                    patch_id=int(patch_id),
                    x=float(x),
                    y=float(y),
                    patch_type=patch_type,
                    area=float(area),
                    habitat_quality=float(habitat_quality),
                    resource_density=float(resource_density),
                    elevation=float(elevation),
                )

                patches.append(patch)

            self.logger.info(f"Imported {len(patches)} patches from CSV")
            return patches

        except Exception as e:
            self.logger.error(f"Error importing CSV coordinates: {e}")
            return []

    def export_to_shapefile(
        self,
        patches: List[SpatialPatch],
        output_path: str,
        coordinate_system: str = "EPSG:4326",
    ) -> None:
        """Export patches to shapefile"""
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            # Create GeoDataFrame
            data = []
            geometries = []

            for patch in patches:
                # Create point geometry
                point = Point(patch.x, patch.y)
                geometries.append(point)

                # Create attribute data
                data.append(
                    {
                        "patch_id": patch.patch_id,
                        "patch_type": patch.patch_type.value,
                        "area": patch.area,
                        "perimeter": patch.perimeter,
                        "elevation": patch.elevation,
                        "habitat_q": patch.habitat_quality,
                        "resource_d": patch.resource_density,
                        "connect_i": patch.connectivity_index,
                        "central_s": patch.centrality_score,
                        "isolation": patch.isolation_index,
                        "accessible": patch.is_accessible,
                        "management": patch.management_status,
                        "disturb_l": patch.disturbance_level,
                    }
                )

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(data, geometry=geometries, crs=coordinate_system)

            # Save to shapefile
            gdf.to_file(output_path)

            self.logger.info(
                f"Exported {len(patches)} patches to shapefile: {output_path}"
            )

        except ImportError:
            self.logger.error("GeoPandas not available for shapefile export")
        except Exception as e:
            self.logger.error(f"Error exporting to shapefile: {e}")

    def coordinate_transformation(
        self, patches: List[SpatialPatch], source_crs: str, target_crs: str
    ) -> List[SpatialPatch]:
        """Transform patch coordinates between coordinate systems"""
        try:
            import pyproj

            # Create transformer
            transformer = pyproj.Transformer.from_crs(
                source_crs, target_crs, always_xy=True
            )

            # Transform coordinates
            for patch in patches:
                new_x, new_y = transformer.transform(patch.x, patch.y)
                patch.x = new_x
                patch.y = new_y

            self.logger.info(
                f"Transformed {len(patches)} patches from {source_crs} to {target_crs}"
            )
            return patches

        except ImportError:
            self.logger.error("PyProj not available for coordinate transformation")
            return patches
        except Exception as e:
            self.logger.error(f"Error in coordinate transformation: {e}")
            return patches


class SpatialStatisticsCalculator:
    """Calculate spatial statistics and analysis"""

    def __init__(self, patches: List[SpatialPatch]):
        self.patches = {p.patch_id: p for p in patches}
        self.logger = logging.getLogger(__name__)

    def calculate_nearest_neighbor_statistics(self) -> Dict[str, float]:
        """Calculate nearest neighbor statistics"""
        if len(self.patches) < 2:
            return {}

        # Calculate coordinates
        coords = np.array([[p.x, p.y] for p in self.patches.values()])

        # Build spatial tree
        tree = cKDTree(coords)

        # Find nearest neighbor distances
        distances, indices = tree.query(coords, k=2)  # k=2 to exclude self
        # Ensure distances is 2D array and handle indexing safely
        if isinstance(distances, (int, float)):
            nn_distances = np.array([distances])
        else:
            distances = np.atleast_2d(distances)
            if distances.shape[1] > 1:
                nn_distances = distances[:, 1]
            else:
                nn_distances = distances[:, 0]

        # Calculate statistics
        stats = {
            "mean_nn_distance": float(np.mean(nn_distances)),
            "median_nn_distance": float(np.median(nn_distances)),
            "min_nn_distance": float(np.min(nn_distances)),
            "max_nn_distance": float(np.max(nn_distances)),
            "std_nn_distance": float(np.std(nn_distances)),
            "cv_nn_distance": float(np.std(nn_distances) / np.mean(nn_distances))
            if np.mean(nn_distances) > 0
            else 0.0,
        }

        # Calculate nearest neighbor index (observed vs expected)
        area = self._calculate_study_area()
        density = len(self.patches) / area if area > 0 else 0
        expected_nn_distance = 1 / (2 * np.sqrt(density)) if density > 0 else 0

        stats["expected_nn_distance"] = float(expected_nn_distance)
        stats["nn_index"] = float(
            stats["mean_nn_distance"] / expected_nn_distance
            if expected_nn_distance > 0
            else 0.0
        )

        return stats

    def calculate_spatial_autocorrelation(
        self, attribute: str = "habitat_quality"
    ) -> Dict[str, float]:
        """Calculate spatial autocorrelation (Moran's I)"""
        if len(self.patches) < 3:
            return {}

        # Get attribute values
        values = []
        coords = []

        for patch in self.patches.values():
            if hasattr(patch, attribute):
                values.append(getattr(patch, attribute))
                coords.append([patch.x, patch.y])

        if not values:
            return {}

        coords_array = np.array(coords)

        # Calculate distance matrix
        distances = distance_matrix(coords_array, coords_array)

        # Create spatial weights matrix (inverse distance)
        weights = np.zeros_like(distances)
        for i in range(len(distances)):
            for j in range(len(distances)):
                if i != j and distances[i, j] > 0:
                    weights[i, j] = 1 / distances[i, j]

        # Row standardize weights
        row_sums = np.sum(weights, axis=1)
        for i in range(len(weights)):
            if row_sums[i] > 0:
                weights[i, :] /= row_sums[i]

        # Calculate Moran's I
        n = len(values)
        mean_value = np.mean(values)

        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                numerator += (
                    weights[i, j] * (values[i] - mean_value) * (values[j] - mean_value)
                )

            denominator += (values[i] - mean_value) ** 2

        morans_i = (
            (n / np.sum(weights)) * (numerator / denominator) if denominator > 0 else 0
        )

        return {"morans_i": float(morans_i), "n_patches": float(n)}

    def calculate_landscape_composition(self) -> Dict[str, Any]:
        """Calculate landscape composition metrics"""
        # Count patches by type
        patch_counts = {}
        total_area = 0.0

        for patch in self.patches.values():
            patch_type = patch.patch_type.value
            if patch_type not in patch_counts:
                patch_counts[patch_type] = {"count": 0, "area": 0.0}
            patch_counts[patch_type]["count"] += 1
            patch_counts[patch_type]["area"] += float(patch.area)
            total_area += float(patch.area)

        # Calculate proportions
        composition = {}
        for patch_type, data in patch_counts.items():
            composition[patch_type] = {
                "count": data["count"],
                "area": data["area"],
                "proportion_count": data["count"] / len(self.patches),
                "proportion_area": data["area"] / total_area if total_area > 0 else 0,
            }

        # Calculate diversity indices
        proportions = [data["proportion_area"] for data in composition.values()]

        # Shannon diversity
        shannon = -sum(p * np.log(p) for p in proportions if p > 0)

        # Simpson diversity
        simpson = 1 - sum(p**2 for p in proportions)

        # Evenness
        max_shannon = np.log(len(composition))
        evenness = shannon / max_shannon if max_shannon > 0 else 0

        return {
            "composition": composition,
            "total_patches": len(self.patches),
            "total_area": total_area,
            "n_patch_types": len(composition),
            "shannon_diversity": shannon,
            "simpson_diversity": simpson,
            "evenness": evenness,
        }

    def calculate_fragmentation_metrics(self) -> Dict[str, float]:
        """Calculate landscape fragmentation metrics"""
        # Basic fragmentation metrics
        n_patches = len(self.patches)
        total_area = sum(p.area for p in self.patches.values())
        mean_patch_size = total_area / n_patches if n_patches > 0 else 0

        # Patch size distribution
        patch_sizes = [p.area for p in self.patches.values()]

        # Edge density (simplified - assumes square patches)
        total_perimeter = sum(p.perimeter for p in self.patches.values())
        edge_density = total_perimeter / total_area if total_area > 0 else 0

        # Largest patch index
        largest_patch = max(patch_sizes) if patch_sizes else 0
        largest_patch_index = largest_patch / total_area if total_area > 0 else 0

        return {
            "n_patches": n_patches,
            "total_area": total_area,
            "mean_patch_size": mean_patch_size,
            "patch_size_cv": float(np.std(patch_sizes) / mean_patch_size)
            if mean_patch_size > 0
            else 0.0,
            "edge_density": edge_density,
            "largest_patch_index": largest_patch_index,
            "patch_density": n_patches / total_area if total_area > 0 else 0,
        }

    def _calculate_study_area(self) -> float:
        """Calculate the area of the study region"""
        if not self.patches:
            return 0

        # Get bounding box
        x_coords = [p.x for p in self.patches.values()]
        y_coords = [p.y for p in self.patches.values()]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Calculate area (simple rectangular approximation)
        area = (x_max - x_min) * (y_max - y_min)

        return area

    def export_spatial_statistics(self, output_directory: str) -> None:
        """Export spatial statistics to files"""
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate all statistics
        nn_stats = self.calculate_nearest_neighbor_statistics()
        autocorr_stats = self.calculate_spatial_autocorrelation()
        composition_stats = self.calculate_landscape_composition()
        fragmentation_stats = self.calculate_fragmentation_metrics()

        # Combine all statistics
        all_stats = {
            "nearest_neighbor": nn_stats,
            "spatial_autocorrelation": autocorr_stats,
            "landscape_composition": composition_stats,
            "fragmentation": fragmentation_stats,
        }

        # Export to JSON
        with open(output_path / "spatial_statistics.json", "w") as f:
            json.dump(all_stats, f, indent=2, default=str)

        # Export summary table
        summary_data = []

        # Add key metrics
        if nn_stats:
            summary_data.append(
                {
                    "metric": "Mean Nearest Neighbor Distance",
                    "value": nn_stats["mean_nn_distance"],
                    "category": "Nearest Neighbor",
                }
            )

        if autocorr_stats:
            summary_data.append(
                {
                    "metric": "Moran's I",
                    "value": autocorr_stats["morans_i"],
                    "category": "Spatial Autocorrelation",
                }
            )

        if composition_stats:
            summary_data.append(
                {
                    "metric": "Shannon Diversity",
                    "value": composition_stats["shannon_diversity"],
                    "category": "Landscape Composition",
                }
            )

        if fragmentation_stats:
            summary_data.append(
                {
                    "metric": "Mean Patch Size",
                    "value": fragmentation_stats["mean_patch_size"],
                    "category": "Fragmentation",
                }
            )

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / "spatial_statistics_summary.csv", index=False)

        self.logger.info(f"Spatial statistics exported to {output_path}")
