"""
Visual Landscape Modification Engine
====================================

Real-time visualization system for landscape modifications, including
before/after rendering, habitat quality mapping, and interactive updates
for agricultural stewardship planning.
"""

from typing import Dict, List, Optional, Tuple, Any, Set, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, date
import logging
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

from .stewardship import HabitatType
from ..spatial.patches import ResourcePatch


class ModificationType(Enum):
    """Types of landscape modifications"""

    MARGIN_CREATION = "margin_creation"
    WILDFLOWER_STRIP = "wildflower_strip"
    HEDGE_PLANTING = "hedge_planting"
    POND_CREATION = "pond_creation"
    TREE_PLANTING = "tree_planting"
    GRASSLAND_RESTORATION = "grassland_restoration"
    WETLAND_CREATION = "wetland_creation"
    HABITAT_CONNECTIVITY = "habitat_connectivity"


class VisualizationMode(Enum):
    """Visualization rendering modes"""

    CURRENT_STATE = "current_state"
    PROPOSED_CHANGES = "proposed_changes"
    BEFORE_AFTER_SPLIT = "before_after_split"
    QUALITY_HEATMAP = "quality_heatmap"
    RESOURCE_DENSITY = "resource_density"
    CONNECTIVITY_MAP = "connectivity_map"
    TEMPORAL_SEQUENCE = "temporal_sequence"


class QualityMetric(Enum):
    """Habitat quality assessment metrics"""

    NECTAR_AVAILABILITY = "nectar_availability"
    POLLEN_AVAILABILITY = "pollen_availability"
    NESTING_SUITABILITY = "nesting_suitability"
    BIODIVERSITY_INDEX = "biodiversity_index"
    CONNECTIVITY_SCORE = "connectivity_score"
    SEASONAL_CONSISTENCY = "seasonal_consistency"


@dataclass
class ColorScheme:
    """Color scheme for visualization"""

    habitat_colors: Dict[HabitatType, str] = field(
        default_factory=lambda: {
            HabitatType.CROPLAND: "#F4E04D",  # Yellow
            HabitatType.GRASSLAND: "#90BE6D",  # Light green
            HabitatType.HEDGEROW: "#43732A",  # Dark green
            HabitatType.WOODLAND: "#2A4D3A",  # Forest green
            HabitatType.URBAN: "#8D8D8D",  # Gray
            HabitatType.WATER: "#277DA1",  # Blue
            HabitatType.WILDFLOWER: "#EA9AB2",  # Pink
            HabitatType.BARE_SOIL: "#B07C3A",  # Brown
        }
    )

    quality_gradient: List[str] = field(
        default_factory=lambda: [
            "#440154",  # Very low (purple)
            "#414487",  # Low
            "#2A788E",  # Medium-low
            "#22A884",  # Medium
            "#7AD151",  # Medium-high
            "#FDE725",  # High (yellow)
        ]
    )

    modification_colors: Dict[ModificationType, str] = field(
        default_factory=lambda: {
            ModificationType.MARGIN_CREATION: "#E63946",  # Red
            ModificationType.WILDFLOWER_STRIP: "#F72585",  # Magenta
            ModificationType.HEDGE_PLANTING: "#3A5A40",  # Dark green
            ModificationType.POND_CREATION: "#1D3557",  # Navy
            ModificationType.TREE_PLANTING: "#344E41",  # Forest
            ModificationType.GRASSLAND_RESTORATION: "#95D5B2",  # Mint
            ModificationType.WETLAND_CREATION: "#52B788",  # Sea green
            ModificationType.HABITAT_CONNECTIVITY: "#FF9F1C",  # Orange
        }
    )


@dataclass
class RenderSettings:
    """Settings for visualization rendering"""

    width: int = 1200
    height: int = 800
    scale: float = 1.0  # Meters per pixel
    opacity: float = 1.0
    show_grid: bool = True
    grid_size: int = 100  # Grid spacing in meters
    show_labels: bool = True
    show_legend: bool = True
    animation_speed: float = 1.0  # Seconds per frame
    quality_resolution: int = 10  # Resolution for quality maps (meters)


@dataclass
class ModificationRecord:
    """Record of a landscape modification"""

    modification_id: str
    modification_type: ModificationType
    timestamp: datetime
    geometry: Dict[str, Any]  # GeoJSON-like geometry
    affected_patches: List[int]
    parameters: Dict[str, Any]
    predicted_impact: Dict[str, float]
    cost_estimate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "modification_id": self.modification_id,
            "modification_type": self.modification_type.value,
            "timestamp": self.timestamp.isoformat(),
            "geometry": self.geometry,
            "affected_patches": self.affected_patches,
            "parameters": self.parameters,
            "predicted_impact": self.predicted_impact,
            "cost_estimate": self.cost_estimate,
        }


class LandscapeLayer(BaseModel):
    """Visual layer for landscape rendering"""

    model_config = {"validate_assignment": True}

    layer_id: str = Field(description="Unique layer identifier")
    layer_type: str = Field(
        description="Type of layer (habitat, modification, quality)"
    )
    visible: bool = Field(default=True, description="Layer visibility")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Layer opacity")
    z_order: int = Field(default=0, description="Drawing order (higher = on top)")

    # Data
    features: List[Dict[str, Any]] = Field(
        default_factory=list, description="GeoJSON features"
    )
    style: Dict[str, Any] = Field(default_factory=dict, description="Layer styling")

    # Interaction
    interactive: bool = Field(default=True, description="Allow user interaction")
    selectable: bool = Field(default=True, description="Allow feature selection")
    editable: bool = Field(default=False, description="Allow editing")


class LandscapeVisualization(BaseModel):
    """Complete landscape visualization state"""

    model_config = {"validate_assignment": True}

    visualization_id: str = Field(description="Unique visualization ID")
    creation_time: datetime = Field(default_factory=datetime.now)

    # Spatial bounds
    bounds: Dict[str, float] = Field(
        description="Spatial bounds {min_x, min_y, max_x, max_y}"
    )
    center: Tuple[float, float] = Field(description="Center coordinates")
    zoom: float = Field(default=1.0, description="Zoom level")

    # Layers
    layers: List[LandscapeLayer] = Field(default_factory=list)
    active_layer_id: Optional[str] = None

    # Rendering
    render_settings: RenderSettings = Field(default_factory=RenderSettings)
    color_scheme: ColorScheme = Field(default_factory=ColorScheme)

    # State
    mode: VisualizationMode = Field(default=VisualizationMode.CURRENT_STATE)
    selected_features: List[str] = Field(default_factory=list)
    highlights: Dict[str, str] = Field(default_factory=dict)  # feature_id -> color


class LandscapeModificationEngine:
    """
    Visual Landscape Modification Engine

    Provides real-time visualization and analysis of landscape modifications
    for agricultural stewardship planning, including before/after comparisons,
    habitat quality mapping, and impact assessment.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Active visualizations
        self.visualizations: Dict[str, LandscapeVisualization] = {}

        # Modification history
        self.modifications: Dict[str, ModificationRecord] = {}
        self.modification_history: List[str] = []  # Ordered list of modification IDs

        # Cached calculations
        self.quality_cache: Dict[
            Tuple[int, int, str], float
        ] = {}  # (x, y, metric) -> value
        self.connectivity_cache: Dict[
            Tuple[int, int], float
        ] = {}  # (patch1, patch2) -> score

        # Rendering engine (simplified - would integrate with actual graphics library)
        self.render_cache: Dict[str, Any] = {}

        # Change tracking
        self.pending_modifications: List[ModificationRecord] = []
        self.applied_modifications: Set[str] = set()

    def create_visualization(
        self,
        patches: List[ResourcePatch],
        bounds: Optional[Dict[str, float]] = None,
        mode: VisualizationMode = VisualizationMode.CURRENT_STATE,
    ) -> LandscapeVisualization:
        """Create a new landscape visualization"""

        # Calculate bounds if not provided
        if bounds is None:
            bounds = self._calculate_bounds(patches)

        center = (
            (bounds["min_x"] + bounds["max_x"]) / 2,
            (bounds["min_y"] + bounds["max_y"]) / 2,
        )

        # Create visualization
        viz_id = f"viz_{datetime.now().timestamp()}"
        visualization = LandscapeVisualization(
            visualization_id=viz_id, bounds=bounds, center=center, mode=mode
        )

        # Add base habitat layer
        habitat_layer = self._create_habitat_layer(patches)
        visualization.layers.append(habitat_layer)

        # Add quality layer if in quality mode
        if mode == VisualizationMode.QUALITY_HEATMAP:
            quality_layer = self._create_quality_layer(
                patches, QualityMetric.BIODIVERSITY_INDEX
            )
            visualization.layers.append(quality_layer)

        self.visualizations[viz_id] = visualization

        self.logger.info(f"Created visualization {viz_id} with {len(patches)} patches")

        return visualization

    def _calculate_bounds(self, patches: List[ResourcePatch]) -> Dict[str, float]:
        """Calculate spatial bounds from patches"""
        if not patches:
            return {"min_x": 0, "min_y": 0, "max_x": 100, "max_y": 100}

        xs = [p.x for p in patches]
        ys = [p.y for p in patches]

        # Add buffer
        buffer = 50

        return {
            "min_x": min(xs) - buffer,
            "min_y": min(ys) - buffer,
            "max_x": max(xs) + buffer,
            "max_y": max(ys) + buffer,
        }

    def _create_habitat_layer(self, patches: List[ResourcePatch]) -> LandscapeLayer:
        """Create habitat visualization layer"""
        features = []

        for patch in patches:
            # Create polygon for patch (simplified as square)
            size = np.sqrt(getattr(patch, "area_ha", 1.0) * 10000)  # Convert to meters
            half_size = size / 2

            feature = {
                "type": "Feature",
                "id": f"patch_{patch.id}",
                "properties": {
                    "patch_id": patch.id,
                    "habitat_type": patch.habitat_type.value,
                    "area_ha": getattr(patch, "area_ha", 1.0),
                    "quality_score": getattr(patch, "quality_score", 0.5),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [patch.x - half_size, patch.y - half_size],
                            [patch.x + half_size, patch.y - half_size],
                            [patch.x + half_size, patch.y + half_size],
                            [patch.x - half_size, patch.y + half_size],
                            [patch.x - half_size, patch.y - half_size],
                        ]
                    ],
                },
            }
            features.append(feature)

        color_scheme = ColorScheme()

        return LandscapeLayer(
            layer_id="habitat_base",
            layer_type="habitat",
            features=features,
            style={
                "fill_color_property": "habitat_type",
                "fill_colors": color_scheme.habitat_colors,
                "stroke_color": "#333333",
                "stroke_width": 1,
            },
            z_order=0,
        )

    def _create_quality_layer(
        self, patches: List[ResourcePatch], metric: QualityMetric, resolution: int = 10
    ) -> LandscapeLayer:
        """Create habitat quality heatmap layer"""
        # Calculate quality grid
        bounds = self._calculate_bounds(patches)

        width = int((bounds["max_x"] - bounds["min_x"]) / resolution)
        height = int((bounds["max_y"] - bounds["min_y"]) / resolution)

        quality_grid = np.zeros((height, width))

        # Calculate quality at each grid point
        for i in range(height):
            for j in range(width):
                x = bounds["min_x"] + j * resolution + resolution / 2
                y = bounds["min_y"] + i * resolution + resolution / 2

                quality = self._calculate_quality_at_point(x, y, patches, metric)
                quality_grid[i, j] = quality

        # Convert to features (grid cells)
        features = []
        color_scheme = ColorScheme()

        for i in range(height):
            for j in range(width):
                if quality_grid[i, j] > 0:
                    x = bounds["min_x"] + j * resolution
                    y = bounds["min_y"] + i * resolution

                    # Map quality to color
                    quality_normalized = quality_grid[i, j]
                    color_index = int(
                        quality_normalized * (len(color_scheme.quality_gradient) - 1)
                    )
                    color = color_scheme.quality_gradient[color_index]

                    feature = {
                        "type": "Feature",
                        "id": f"quality_{i}_{j}",
                        "properties": {
                            "quality": quality_grid[i, j],
                            "metric": metric.value,
                            "color": color,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [x, y],
                                    [x + resolution, y],
                                    [x + resolution, y + resolution],
                                    [x, y + resolution],
                                    [x, y],
                                ]
                            ],
                        },
                    }
                    features.append(feature)

        return LandscapeLayer(
            layer_id=f"quality_{metric.value}",
            layer_type="quality",
            features=features,
            style={"fill_color_property": "color", "opacity": 0.7, "stroke_width": 0},
            z_order=10,
            interactive=False,
        )

    def _calculate_quality_at_point(
        self, x: float, y: float, patches: List[ResourcePatch], metric: QualityMetric
    ) -> float:
        """Calculate habitat quality at a specific point"""
        # Check cache
        cache_key = (int(x), int(y), metric.value)
        if cache_key in self.quality_cache:
            return self.quality_cache[cache_key]

        quality = 0.0
        total_weight = 0.0

        for patch in patches:
            # Distance-based influence
            distance = np.sqrt((patch.x - x) ** 2 + (patch.y - y) ** 2)

            # Skip if too far
            if distance > 500:  # 500m influence radius
                continue

            # Calculate weight (inverse distance)
            weight = 1.0 / (1.0 + distance / 100)

            # Get patch quality for metric
            patch_quality = self._get_patch_quality(patch, metric)

            quality += patch_quality * weight
            total_weight += weight

        final_quality = quality / total_weight if total_weight > 0 else 0

        # Cache result
        self.quality_cache[cache_key] = final_quality

        return final_quality

    def _get_patch_quality(self, patch: ResourcePatch, metric: QualityMetric) -> float:
        """Get quality score for a patch based on metric"""
        if metric == QualityMetric.NECTAR_AVAILABILITY:
            return min(patch.base_nectar_production / 10.0, 1.0)  # Normalize
        elif metric == QualityMetric.POLLEN_AVAILABILITY:
            return min(patch.base_pollen_production / 10.0, 1.0)
        elif metric == QualityMetric.NESTING_SUITABILITY:
            # Based on habitat type
            nesting_scores = {
                HabitatType.HEDGEROW: 0.9,
                HabitatType.WOODLAND: 0.8,
                HabitatType.GRASSLAND: 0.7,
                HabitatType.WILDFLOWER: 0.6,
                HabitatType.CROPLAND: 0.2,
                HabitatType.URBAN: 0.1,
                HabitatType.WATER: 0.0,
                HabitatType.BARE_SOIL: 0.3,
            }
            return nesting_scores.get(patch.habitat_type, 0.5)
        elif metric == QualityMetric.BIODIVERSITY_INDEX:
            # Composite metric
            nectar = self._get_patch_quality(patch, QualityMetric.NECTAR_AVAILABILITY)
            pollen = self._get_patch_quality(patch, QualityMetric.POLLEN_AVAILABILITY)
            nesting = self._get_patch_quality(patch, QualityMetric.NESTING_SUITABILITY)
            return (nectar + pollen + nesting) / 3
        else:
            return 0.5  # Default

    def add_modification(
        self,
        visualization_id: str,
        modification_type: ModificationType,
        geometry: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> ModificationRecord:
        """Add a proposed modification to the landscape"""

        if visualization_id not in self.visualizations:
            raise ValueError(f"Visualization {visualization_id} not found")

        # Create modification record
        mod_id = f"mod_{datetime.now().timestamp()}"

        # Determine affected patches
        affected_patches = self._find_affected_patches(geometry)

        # Predict impact
        impact = self._predict_modification_impact(
            modification_type, geometry, parameters, affected_patches
        )

        # Estimate cost
        cost = self._estimate_modification_cost(modification_type, geometry, parameters)

        modification = ModificationRecord(
            modification_id=mod_id,
            modification_type=modification_type,
            timestamp=datetime.now(),
            geometry=geometry,
            affected_patches=affected_patches,
            parameters=parameters,
            predicted_impact=impact,
            cost_estimate=cost,
        )

        self.modifications[mod_id] = modification
        self.pending_modifications.append(modification)

        # Add modification layer to visualization
        self._add_modification_layer(visualization_id, modification)

        self.logger.info(
            f"Added {modification_type.value} modification affecting "
            f"{len(affected_patches)} patches, estimated cost £{cost:.2f}"
        )

        return modification

    def _find_affected_patches(self, geometry: Dict[str, Any]) -> List[int]:
        """Find patches affected by a modification geometry"""
        affected = []

        # Simplified - would use proper spatial intersection
        if geometry["type"] == "LineString":
            # For strips and corridors
            coords = geometry["coordinates"]
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                # Find patches along line
                # Simplified - just return mock data
                affected.extend([1001, 1002, 1003])
        elif geometry["type"] == "Polygon":
            # For areas
            affected.extend([2001, 2002])

        return list(set(affected))

    def _predict_modification_impact(
        self,
        mod_type: ModificationType,
        geometry: Dict[str, Any],
        parameters: Dict[str, Any],
        affected_patches: List[int],
    ) -> Dict[str, float]:
        """Predict the impact of a modification"""

        impact = {
            "nectar_increase": 0.0,
            "pollen_increase": 0.0,
            "habitat_area_change": 0.0,
            "connectivity_improvement": 0.0,
            "biodiversity_increase": 0.0,
        }

        # Calculate area
        area_ha = self._calculate_geometry_area(geometry)

        if mod_type == ModificationType.WILDFLOWER_STRIP:
            # Based on seed mix
            impact["nectar_increase"] = area_ha * 5.0  # kg/year estimate
            impact["pollen_increase"] = area_ha * 3.0
            impact["habitat_area_change"] = area_ha
            impact["biodiversity_increase"] = 0.15  # 15% increase

        elif mod_type == ModificationType.MARGIN_CREATION:
            width = parameters.get("width", 6)
            impact["nectar_increase"] = area_ha * 3.0
            impact["pollen_increase"] = area_ha * 2.0
            impact["habitat_area_change"] = area_ha
            impact["connectivity_improvement"] = 0.1 * width / 6  # Width bonus

        elif mod_type == ModificationType.HEDGE_PLANTING:
            length_km = parameters.get("length_m", 0) / 1000
            impact["habitat_area_change"] = length_km * 0.3  # 3m width assumed
            impact["connectivity_improvement"] = 0.3  # High connectivity value
            impact["biodiversity_increase"] = 0.2

        elif mod_type == ModificationType.POND_CREATION:
            impact["habitat_area_change"] = area_ha
            impact["biodiversity_increase"] = 0.25  # High biodiversity value

        return impact

    def _calculate_geometry_area(self, geometry: Dict[str, Any]) -> float:
        """Calculate area of geometry in hectares"""
        if geometry["type"] == "Polygon":
            # Simplified - would use proper area calculation
            coords = geometry["coordinates"][0]
            # Shoelace formula
            area = 0.0
            for i in range(len(coords) - 1):
                area += coords[i][0] * coords[i + 1][1]
                area -= coords[i + 1][0] * coords[i][1]
            area = abs(area) / 2.0 / 10000  # Convert to hectares
            return area
        elif geometry["type"] == "LineString":
            # For strips - width from parameters
            coords = geometry["coordinates"]
            length = 0.0
            for i in range(len(coords) - 1):
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]
                length += np.sqrt(dx**2 + dy**2)
            width = 6  # Default 6m
            return (length * width) / 10000
        return 0.0

    def _estimate_modification_cost(
        self,
        mod_type: ModificationType,
        geometry: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> float:
        """Estimate cost of modification"""
        area_ha = self._calculate_geometry_area(geometry)

        # Base costs per hectare
        cost_per_ha = {
            ModificationType.WILDFLOWER_STRIP: 750.0,
            ModificationType.MARGIN_CREATION: 500.0,
            ModificationType.HEDGE_PLANTING: 8000.0,  # Per km, not ha
            ModificationType.POND_CREATION: 5000.0,
            ModificationType.TREE_PLANTING: 3000.0,
            ModificationType.GRASSLAND_RESTORATION: 1200.0,
            ModificationType.WETLAND_CREATION: 4000.0,
            ModificationType.HABITAT_CONNECTIVITY: 1000.0,
        }

        base_cost = cost_per_ha.get(mod_type, 1000.0)

        if mod_type == ModificationType.HEDGE_PLANTING:
            # Calculate by length instead
            coords = geometry["coordinates"]
            length_m = sum(
                np.sqrt(
                    (coords[i + 1][0] - coords[i][0]) ** 2
                    + (coords[i + 1][1] - coords[i][1]) ** 2
                )
                for i in range(len(coords) - 1)
            )
            return float(base_cost * (length_m / 1000))

        return base_cost * area_ha

    def _add_modification_layer(
        self, visualization_id: str, modification: ModificationRecord
    ) -> None:
        """Add modification as a layer to visualization"""
        viz = self.visualizations[visualization_id]

        color_scheme = ColorScheme()

        feature = {
            "type": "Feature",
            "id": modification.modification_id,
            "properties": {
                "modification_type": modification.modification_type.value,
                "cost": modification.cost_estimate,
                "impact": modification.predicted_impact,
            },
            "geometry": modification.geometry,
        }

        layer = LandscapeLayer(
            layer_id=f"mod_{modification.modification_id}",
            layer_type="modification",
            features=[feature],
            style={
                "fill_color": color_scheme.modification_colors.get(
                    modification.modification_type, "#999999"
                ),
                "stroke_color": "#FFFFFF",
                "stroke_width": 2,
                "stroke_dasharray": "5,5",  # Dashed line for proposed
            },
            z_order=20,
            editable=True,
        )

        viz.layers.append(layer)

    def render_before_after(
        self, visualization_id: str, modifications: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Render before/after comparison view"""

        if visualization_id not in self.visualizations:
            raise ValueError(f"Visualization {visualization_id} not found")

        viz = self.visualizations[visualization_id]

        # Select modifications to show
        if modifications is None:
            modifications = [m.modification_id for m in self.pending_modifications]

        # Create split view
        render_data = {
            "type": "split_view",
            "left": {
                "title": "Current State",
                "layers": [
                    layer for layer in viz.layers if layer.layer_type != "modification"
                ],
            },
            "right": {
                "title": "After Modifications",
                "layers": viz.layers,  # All layers including modifications
            },
            "modifications": [
                self.modifications[mod_id].to_dict()
                for mod_id in modifications
                if mod_id in self.modifications
            ],
            "summary": self._calculate_modification_summary(modifications),
        }

        return render_data

    def _calculate_modification_summary(
        self, modification_ids: List[str]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for modifications"""
        total_cost = 0.0
        total_area = 0.0
        impact_totals: Dict[str, float] = defaultdict(float)

        for mod_id in modification_ids:
            if mod_id in self.modifications:
                mod = self.modifications[mod_id]
                total_cost += mod.cost_estimate
                total_area += self._calculate_geometry_area(mod.geometry)

                for key, value in mod.predicted_impact.items():
                    impact_totals[key] += value

        return {
            "total_cost": total_cost,
            "total_area_ha": total_area,
            "modification_count": len(modification_ids),
            "predicted_impacts": dict(impact_totals),
            "cost_per_hectare": total_cost / total_area if total_area > 0 else 0,
        }

    def update_quality_map(
        self,
        visualization_id: str,
        metric: QualityMetric,
        include_modifications: bool = True,
    ) -> None:
        """Update habitat quality heatmap with current or proposed state"""

        if visualization_id not in self.visualizations:
            raise ValueError(f"Visualization {visualization_id} not found")

        viz = self.visualizations[visualization_id]

        # Remove existing quality layers
        viz.layers = [layer for layer in viz.layers if layer.layer_type != "quality"]

        # Get patches (would be from actual data source)
        patches: List[Any] = []  # Would load from current state

        # Apply modifications if requested
        if include_modifications:
            patches = self._apply_modifications_to_patches(patches)

        # Create new quality layer
        quality_layer = self._create_quality_layer(patches, metric)
        viz.layers.append(quality_layer)

        # Clear quality cache as it may have changed
        self.quality_cache.clear()

        self.logger.info(f"Updated quality map for {metric.value}")

    def _apply_modifications_to_patches(
        self, patches: List[ResourcePatch]
    ) -> List[ResourcePatch]:
        """Apply pending modifications to patches (simulation)"""
        # Copy patches
        modified_patches = [p for p in patches]  # Would deep copy

        for mod in self.pending_modifications:
            # Apply modification effects
            if mod.modification_type == ModificationType.WILDFLOWER_STRIP:
                # Increase resources in affected patches
                for patch_id in mod.affected_patches:
                    # Find patch and modify
                    for patch in modified_patches:
                        if patch.id == patch_id:
                            patch.base_nectar_production *= 1.5
                            patch.base_pollen_production *= 1.5

        return modified_patches

    def analyze_connectivity(
        self, visualization_id: str, threshold_distance: float = 200.0
    ) -> Dict[str, Any]:
        """Analyze habitat connectivity in the landscape"""

        if visualization_id not in self.visualizations:
            raise ValueError(f"Visualization {visualization_id} not found")

        # Get habitat patches
        habitat_patches: List[Any] = []  # Would extract from visualization

        # Build connectivity graph
        connections = []
        isolated_patches = []

        for i, patch1 in enumerate(habitat_patches):
            connected = False
            for j, patch2 in enumerate(habitat_patches):
                if i >= j:
                    continue

                distance = np.sqrt(
                    (patch1.x - patch2.x) ** 2 + (patch1.y - patch2.y) ** 2
                )

                if distance <= threshold_distance:
                    connections.append(
                        {
                            "from": patch1.id,
                            "to": patch2.id,
                            "distance": distance,
                            "quality": 1.0 - (distance / threshold_distance),
                        }
                    )
                    connected = True

            if not connected:
                isolated_patches.append(patch1.id)

        # Calculate metrics
        connectivity_score = (
            len(connections) / (len(habitat_patches) * (len(habitat_patches) - 1) / 2)
            if len(habitat_patches) > 1
            else 0
        )

        return {
            "connectivity_score": connectivity_score,
            "connections": connections,
            "isolated_patches": isolated_patches,
            "average_connection_distance": np.mean([c["distance"] for c in connections])
            if connections
            else 0,
            "recommendations": self._generate_connectivity_recommendations(
                isolated_patches, connections, threshold_distance
            ),
        }

    def _generate_connectivity_recommendations(
        self,
        isolated_patches: List[int],
        connections: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for improving connectivity"""
        recommendations = []

        if isolated_patches:
            recommendations.append(
                {
                    "type": "connect_isolated",
                    "priority": "high",
                    "description": f"Create habitat corridors to connect {len(isolated_patches)} isolated patches",
                    "estimated_length_m": len(isolated_patches) * threshold * 0.7,
                }
            )

        # Find weak connections
        weak_connections = [c for c in connections if c["quality"] < 0.5]
        if weak_connections:
            recommendations.append(
                {
                    "type": "strengthen_connections",
                    "priority": "medium",
                    "description": f"Enhance {len(weak_connections)} weak habitat connections",
                    "targets": weak_connections[:5],  # Top 5
                }
            )

        return recommendations

    def animate_temporal_changes(
        self,
        visualization_id: str,
        start_date: date,
        end_date: date,
        interval_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Generate animation frames showing temporal changes"""

        frames = []
        current_date = start_date

        while current_date <= end_date:
            # Create frame for current date
            frame: Dict[str, Any] = {
                "date": current_date.isoformat(),
                "day_of_year": current_date.timetuple().tm_yday,
                "changes": [],
            }

            # Check modifications becoming active
            for mod in self.modifications.values():
                if mod.timestamp.date() == current_date:
                    frame["changes"].append(
                        {"type": "modification_added", "modification": mod.to_dict()}
                    )

            # Seasonal changes (simplified)
            season = self._get_season(current_date)
            frame["season"] = season
            frame["bloom_intensity"] = self._get_bloom_intensity(current_date)

            frames.append(frame)

            # Advance date using timedelta to handle month boundaries correctly
            from datetime import timedelta

            current_date = current_date + timedelta(days=interval_days)

        return frames

    def _get_season(self, date: date) -> str:
        """Get season for date"""
        month = date.month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    def _get_bloom_intensity(self, date: date) -> float:
        """Get bloom intensity for date (0-1)"""
        day_of_year = date.timetuple().tm_yday

        # Peak in mid-summer (day 180-210)
        if 180 <= day_of_year <= 210:
            return 1.0
        elif 150 <= day_of_year <= 240:
            return 0.8
        elif 120 <= day_of_year <= 270:
            return 0.6
        elif 90 <= day_of_year <= 300:
            return 0.4
        else:
            return 0.1

    def export_visualization(
        self, visualization_id: str, format: str = "geojson"
    ) -> Union[Dict[str, Any], str]:
        """Export visualization in specified format"""

        if visualization_id not in self.visualizations:
            raise ValueError(f"Visualization {visualization_id} not found")

        viz = self.visualizations[visualization_id]

        if format == "geojson":
            # Create FeatureCollection
            all_features = []
            for layer in viz.layers:
                all_features.extend(layer.features)

            return {
                "type": "FeatureCollection",
                "features": all_features,
                "properties": {
                    "visualization_id": viz.visualization_id,
                    "creation_time": viz.creation_time.isoformat(),
                    "bounds": viz.bounds,
                    "mode": viz.mode.value,
                },
            }
        elif format == "summary":
            # Create summary report
            return {
                "visualization_id": viz.visualization_id,
                "layer_count": len(viz.layers),
                "modification_count": len(self.pending_modifications),
                "total_estimated_cost": sum(
                    m.cost_estimate for m in self.pending_modifications
                ),
                "affected_area_ha": sum(
                    self._calculate_geometry_area(m.geometry)
                    for m in self.pending_modifications
                ),
            }
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_visualization_state(self, visualization_id: str) -> Dict[str, Any]:
        """Get current state of visualization"""
        if visualization_id not in self.visualizations:
            raise ValueError(f"Visualization {visualization_id} not found")

        viz = self.visualizations[visualization_id]

        return {
            "visualization_id": viz.visualization_id,
            "mode": viz.mode.value,
            "bounds": viz.bounds,
            "center": viz.center,
            "zoom": viz.zoom,
            "layer_count": len(viz.layers),
            "visible_layers": [layer.layer_id for layer in viz.layers if layer.visible],
            "selected_features": viz.selected_features,
            "pending_modifications": len(self.pending_modifications),
            "render_settings": {
                "width": viz.render_settings.width,
                "height": viz.render_settings.height,
                "scale": viz.render_settings.scale,
            },
        }
