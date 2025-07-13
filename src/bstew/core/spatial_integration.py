"""
Spatial Integration System for NetLogo BEE-STEWARD v2 Parity
===========================================================

Integration of spatial algorithms with the main BSTEW simulation system,
providing spatial-aware bee behavior and landscape dynamics.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import logging
import numpy as np
import math

from .spatial_algorithms import (
    SpatialIndex, PathFinder, ConnectivityAnalyzer, LandscapeAnalyzer,
    SpatialPoint, SpatialPatch, ConnectivityType,
    create_spatial_analysis_system
)
from .gis_integration import GISIntegrationManager, create_gis_integration_system
from .enums import BeeStatus

@dataclass
class SpatialBeeState:
    """Spatial state information for individual bees"""
    bee_id: int
    current_position: SpatialPoint
    current_patch: Optional[int] = None
    target_patch: Optional[int] = None
    planned_path: List[int] = field(default_factory=list)
    movement_history: List[SpatialPoint] = field(default_factory=list)
    spatial_memory: Dict[int, float] = field(default_factory=dict)  # patch_id -> quality memory
    
    def update_position(self, new_position: SpatialPoint, patch_id: Optional[int] = None) -> None:
        """Update bee's spatial position"""
        self.movement_history.append(self.current_position)
        self.current_position = new_position
        if patch_id is not None:
            self.current_patch = patch_id
        
        # Limit movement history size
        if len(self.movement_history) > 100:
            self.movement_history.pop(0)
    
    def remember_patch_quality(self, patch_id: int, quality: float) -> None:
        """Update spatial memory of patch quality"""
        # Weighted average with existing memory
        if patch_id in self.spatial_memory:
            self.spatial_memory[patch_id] = 0.7 * self.spatial_memory[patch_id] + 0.3 * quality
        else:
            self.spatial_memory[patch_id] = quality

class SpatialEnvironmentManager(BaseModel):
    """Manages spatial environment and landscape dynamics"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Core spatial systems
    spatial_index: SpatialIndex
    path_finder: PathFinder
    connectivity_analyzer: ConnectivityAnalyzer
    landscape_analyzer: LandscapeAnalyzer
    
    # GIS integration
    gis_manager: Optional[GISIntegrationManager] = None
    
    # Dynamic landscape state
    patch_qualities: Dict[int, float] = Field(default_factory=dict)
    resource_levels: Dict[int, float] = Field(default_factory=dict)
    patch_usage: Dict[int, int] = Field(default_factory=dict)  # visits per patch
    
    # Landscape dynamics
    quality_change_rate: float = 0.01
    resource_regeneration_rate: float = 0.05
    seasonal_variation: bool = True
    current_season: float = 0.0  # 0-1 representing seasonal cycle
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def initialize_landscape(self, landscape_config: Dict[str, Any]) -> None:
        """Initialize spatial landscape from configuration"""
        
        patches = []
        
        # Load patches from GIS data if available
        if landscape_config.get('use_gis_data', False):
            patches.extend(self._load_patches_from_gis(landscape_config))
        
        # Create patches from direct configuration
        patches_config = landscape_config.get('patches', [])
        for patch_config in patches_config:
            patch = self._create_patch_from_config(patch_config)
            patches.append(patch)
        
        # Add all patches to spatial index
        for patch in patches:
            self.spatial_index.add_patch(patch)
            
            # Initialize dynamic state
            self.patch_qualities[patch.patch_id] = patch.quality
            self.resource_levels[patch.patch_id] = patch.resource_density
            self.patch_usage[patch.patch_id] = 0
        
        # Establish connectivity
        self._establish_patch_connectivity()
        
        self.logger.info(f"Initialized landscape with {len(self.spatial_index._patches)} patches")
    
    def _create_patch_from_config(self, config: Dict[str, Any]) -> SpatialPatch:
        """Create spatial patch from configuration"""
        
        center = SpatialPoint(
            x=config.get('x', 0.0),
            y=config.get('y', 0.0),
            z=config.get('z', 0.0),
            patch_id=config.get('id')
        )
        
        # Create vertices if specified
        vertices = []
        if 'vertices' in config:
            for vertex_config in config['vertices']:
                vertex = SpatialPoint(
                    x=vertex_config['x'],
                    y=vertex_config['y'],
                    z=vertex_config.get('z', 0.0)
                )
                vertices.append(vertex)
        
        patch = SpatialPatch(
            patch_id=config['id'],
            center=center,
            vertices=vertices,
            area=config.get('area', 1.0),
            perimeter=config.get('perimeter', 4.0),
            quality=config.get('quality', 0.5),
            resource_density=config.get('resource_density', 0.5),
            accessibility=config.get('accessibility', 1.0)
        )
        
        return patch
    
    def _load_patches_from_gis(self, landscape_config: Dict[str, Any]) -> List[SpatialPatch]:
        """Load spatial patches from GIS data sources"""
        
        if not self.gis_manager:
            # Initialize GIS manager if not provided
            gis_config = landscape_config.get('gis_config', {})
            source_crs = gis_config.get('source_crs', 'EPSG:4326')
            target_crs = gis_config.get('target_crs', 'LOCAL')
            
            self.gis_manager = create_gis_integration_system(source_crs, target_crs)
        
        try:
            # Load spatial data from GIS sources
            gis_patches = self.gis_manager.load_spatial_data_from_config(landscape_config)
            
            # Renumber patch IDs to avoid conflicts
            next_id = max([p.patch_id for p in self.spatial_index._patches.values()], default=-1) + 1
            
            for patch in gis_patches:
                patch.patch_id = next_id
                next_id += 1
            
            self.logger.info(f"Loaded {len(gis_patches)} patches from GIS data sources")
            return gis_patches
            
        except Exception as e:
            self.logger.error(f"Failed to load GIS data: {e}")
            return []
    
    def _establish_patch_connectivity(self) -> None:
        """Establish connectivity between patches"""
        
        # Build connectivity graph
        graph = self.connectivity_analyzer.build_connectivity_graph(ConnectivityType.PROXIMITY)
        
        # Update patch neighbor information
        for patch_id in self.spatial_index._patches:
            if patch_id in graph:
                neighbors = set(graph.neighbors(patch_id))
                self.spatial_index._patches[patch_id].neighbors = neighbors
    
    def update_landscape_dynamics(self, timestep: int) -> None:
        """Update dynamic landscape properties"""
        
        # Update seasonal cycle
        if self.seasonal_variation:
            self.current_season = (timestep % 365) / 365.0
        
        # Update patch qualities and resources
        for patch_id, patch in self.spatial_index._patches.items():
            self._update_patch_dynamics(patch_id, patch)
        
        # Resource regeneration
        self._regenerate_resources()
    
    def _update_patch_dynamics(self, patch_id: int, patch: SpatialPatch) -> None:
        """Update dynamics for individual patch"""
        
        # Seasonal quality variation
        if self.seasonal_variation:
            seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * self.current_season)
            base_quality = patch.quality
            current_quality = base_quality * (0.7 + 0.3 * seasonal_factor)
            self.patch_qualities[patch_id] = max(0.1, min(1.0, current_quality))
        
        # Usage-based degradation
        usage = self.patch_usage.get(patch_id, 0)
        if usage > 10:  # High usage threshold
            degradation = min(0.1, usage * 0.001)
            self.patch_qualities[patch_id] = max(0.1, self.patch_qualities[patch_id] - degradation)
    
    def _regenerate_resources(self) -> None:
        """Regenerate resources in patches"""
        
        for patch_id in self.resource_levels:
            current_level = self.resource_levels[patch_id]
            patch = self.spatial_index._patches.get(patch_id)
            
            if patch:
                max_level = patch.resource_density
                if current_level < max_level:
                    regeneration = self.resource_regeneration_rate * (max_level - current_level)
                    self.resource_levels[patch_id] = min(max_level, current_level + regeneration)
    
    def get_patch_current_quality(self, patch_id: int) -> float:
        """Get current quality of patch (may differ from base quality)"""
        return self.patch_qualities.get(patch_id, 0.5)
    
    def get_patch_resource_level(self, patch_id: int) -> float:
        """Get current resource level of patch"""
        return self.resource_levels.get(patch_id, 0.5)
    
    def consume_resources(self, patch_id: int, amount: float) -> float:
        """Consume resources from patch, return actual amount consumed"""
        
        if patch_id not in self.resource_levels:
            return 0.0
        
        current_level = self.resource_levels[patch_id]
        actual_consumption = min(amount, current_level)
        self.resource_levels[patch_id] = current_level - actual_consumption
        
        # Record patch usage
        self.patch_usage[patch_id] = self.patch_usage.get(patch_id, 0) + 1
        
        return actual_consumption
    
    def find_optimal_foraging_patches(self, bee_position: SpatialPoint, 
                                    max_distance: float = 50.0,
                                    min_quality: float = 0.3) -> List[Tuple[int, float]]:
        """Find optimal foraging patches within range"""
        
        # Find patches within range
        nearby_patches = self.spatial_index.find_patches_in_radius(bee_position, max_distance)
        
        # Evaluate patches
        patch_scores = []
        for patch_id in nearby_patches:
            if patch_id in self.spatial_index._patches:
                patch = self.spatial_index._patches[patch_id]
                current_quality = self.get_patch_current_quality(patch_id)
                resource_level = self.get_patch_resource_level(patch_id)
                
                if current_quality >= min_quality and resource_level > 0.1:
                    distance = bee_position.distance_to(patch.center)
                    
                    # Score based on quality, resources, and distance
                    score = (current_quality * resource_level) / max(1.0, distance / 10.0)
                    patch_scores.append((patch_id, score))
        
        # Sort by score (descending)
        patch_scores.sort(key=lambda x: x[1], reverse=True)
        
        return patch_scores[:10]  # Return top 10 patches
    
    def export_landscape_to_gis(self, output_path: str, output_format: str = "geojson") -> str:
        """Export current landscape state to GIS format"""
        
        if not self.gis_manager:
            raise RuntimeError("GIS manager not initialized - cannot export landscape data")
        
        try:
            from .gis_integration import DataFormat
            
            # Map string format to enum
            format_mapping = {
                "geojson": DataFormat.GEOJSON,
                "json": DataFormat.GEOJSON
            }
            
            data_format = format_mapping.get(output_format.lower(), DataFormat.GEOJSON)
            
            # Export spatial data
            result_path = self.gis_manager.export_spatial_data(
                self.spatial_index, output_path, data_format
            )
            
            self.logger.info(f"Exported landscape to: {result_path}")
            return result_path
            
        except Exception as e:
            self.logger.error(f"Failed to export landscape: {e}")
            raise
    
    def get_gis_analysis_results(self) -> Dict[str, Any]:
        """Get GIS-based analysis results"""
        
        results = {}
        
        # Basic spatial statistics
        patches = list(self.spatial_index._patches.values())
        if patches:
            areas = [patch.area for patch in patches]
            qualities = [self.get_patch_current_quality(patch.patch_id) for patch in patches]
            
            results['spatial_statistics'] = {
                'total_patches': len(patches),
                'total_area': sum(areas),
                'mean_patch_area': sum(areas) / len(areas),
                'mean_quality': sum(qualities) / len(qualities),
                'quality_variance': sum((q - sum(qualities)/len(qualities))**2 for q in qualities) / len(qualities)
            }
        
        # Coordinate system information
        if self.gis_manager:
            results['coordinate_system'] = {
                'source_crs': self.gis_manager.coordinate_transformer.source_crs,
                'target_crs': self.gis_manager.coordinate_transformer.target_crs
            }
        
        # Landscape quality assessment
        landscape_quality = self.landscape_analyzer.assess_landscape_quality()
        results['landscape_quality'] = landscape_quality
        
        return results

class SpatialBeeManager(BaseModel):
    """Manages spatial behavior and movement of individual bees"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Dependencies
    environment_manager: Union[SpatialEnvironmentManager, Any]
    
    # Bee spatial states
    bee_states: Dict[int, SpatialBeeState] = Field(default_factory=dict)
    
    # Movement parameters
    max_foraging_distance: float = 100.0
    movement_speed: float = 1.0  # units per timestep
    path_recalculation_interval: int = 10
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def register_bee(self, bee_id: int, initial_position: SpatialPoint, initial_patch: Optional[int] = None) -> None:
        """Register a bee in the spatial system"""
        
        bee_state = SpatialBeeState(
            bee_id=bee_id,
            current_position=initial_position,
            current_patch=initial_patch
        )
        
        self.bee_states[bee_id] = bee_state
    
    def update_bee_spatial_behavior(self, bee_id: int, bee_status: BeeStatus, timestep: int) -> Dict[str, Any]:
        """Update spatial behavior based on bee status"""
        
        if bee_id not in self.bee_states:
            return {}
        
        bee_state = self.bee_states[bee_id]
        behavior_result = {}
        
        if bee_status == BeeStatus.FORAGING:
            behavior_result = self._handle_foraging_behavior(bee_state, timestep)
        elif bee_status == BeeStatus.SEARCHING:
            behavior_result = self._handle_scouting_behavior(bee_state, timestep)
        elif bee_status == BeeStatus.DANCING:
            behavior_result = self._handle_following_behavior(bee_state, timestep)
        elif bee_status in [BeeStatus.BRINGING_POLLEN, BeeStatus.BRINGING_NECTAR]:
            behavior_result = self._handle_returning_behavior(bee_state, timestep)
        else:
            behavior_result = self._handle_default_movement(bee_state, timestep)
        
        return behavior_result
    
    def _handle_foraging_behavior(self, bee_state: SpatialBeeState, timestep: int) -> Dict[str, Any]:
        """Handle spatial behavior for foraging bees"""
        
        # Check if bee needs a new target
        if not bee_state.target_patch or not bee_state.planned_path:
            optimal_patches = self.environment_manager.find_optimal_foraging_patches(
                bee_state.current_position,
                self.max_foraging_distance
            )
            
            if optimal_patches:
                target_patch, score = optimal_patches[0]
                bee_state.target_patch = target_patch
                
                # Plan path to target
                if bee_state.current_patch:
                    path = self.environment_manager.path_finder.find_shortest_path(
                        bee_state.current_patch,
                        target_patch
                    )
                    if path:
                        bee_state.planned_path = path[1:]  # Exclude current patch
        
        # Move along planned path
        movement_result = self._execute_movement(bee_state)
        
        # Check if reached target patch
        if bee_state.current_patch == bee_state.target_patch:
            # Forage in the patch
            if bee_state.target_patch:
                quality = self.environment_manager.get_patch_current_quality(bee_state.target_patch)
                resources_consumed = self.environment_manager.consume_resources(bee_state.target_patch, 0.1)
                
                # Update bee's spatial memory
                bee_state.remember_patch_quality(bee_state.target_patch, quality)
                
                movement_result.update({
                    'foraging_success': resources_consumed > 0,
                    'resources_collected': resources_consumed,
                    'patch_quality': quality
                })
        
        return movement_result
    
    def _handle_scouting_behavior(self, bee_state: SpatialBeeState, timestep: int) -> Dict[str, Any]:
        """Handle spatial behavior for scouting bees"""
        
        # Scout for new patches within range
        nearby_patches = self.environment_manager.spatial_index.find_patches_in_radius(
            bee_state.current_position, 
            self.max_foraging_distance
        )
        
        # Look for unexplored or high-quality patches
        best_patch = None
        best_score = 0
        
        for patch_id in nearby_patches:
            if patch_id not in bee_state.spatial_memory:
                # Unexplored patch - high priority
                best_patch = patch_id
                best_score = 1.0
                break
            else:
                # Check if it's a high-quality patch worth revisiting
                quality = bee_state.spatial_memory[patch_id]
                if quality > best_score:
                    best_patch = patch_id
                    best_score = quality
        
        # Move toward best patch
        if best_patch and best_patch != bee_state.current_patch:
            bee_state.target_patch = best_patch
            if bee_state.current_patch:
                path = self.environment_manager.path_finder.find_shortest_path(
                    bee_state.current_patch,
                    best_patch
                )
                if path:
                    bee_state.planned_path = path[1:]
        
        movement_result = self._execute_movement(bee_state)
        
        # If reached target, evaluate patch
        if bee_state.current_patch == bee_state.target_patch and bee_state.target_patch:
            quality = self.environment_manager.get_patch_current_quality(bee_state.target_patch)
            bee_state.remember_patch_quality(bee_state.target_patch, quality)
            
            movement_result.update({
                'scouting_result': {
                    'patch_id': bee_state.target_patch,
                    'quality': quality,
                    'position': bee_state.current_position
                }
            })
        
        return movement_result
    
    def _handle_following_behavior(self, bee_state: SpatialBeeState, timestep: int) -> Dict[str, Any]:
        """Handle spatial behavior for bees following dance directions"""
        
        # Following behavior - move toward communicated patch location
        # This would integrate with the dance communication system
        
        return self._execute_movement(bee_state)
    
    def _handle_returning_behavior(self, bee_state: SpatialBeeState, timestep: int) -> Dict[str, Any]:
        """Handle spatial behavior for bees returning to hive"""
        
        # Return to hive (patch 0 by convention)
        if bee_state.current_patch != 0:
            bee_state.target_patch = 0
            if bee_state.current_patch:
                path = self.environment_manager.path_finder.find_shortest_path(
                    bee_state.current_patch,
                    0
                )
                if path:
                    bee_state.planned_path = path[1:]
        
        return self._execute_movement(bee_state)
    
    def _handle_default_movement(self, bee_state: SpatialBeeState, timestep: int) -> Dict[str, Any]:
        """Handle default spatial movement"""
        
        # Random local movement or staying in place
        if np.random.random() < 0.1:  # 10% chance to move
            # Small random movement
            dx = np.random.normal(0, 2.0)
            dy = np.random.normal(0, 2.0)
            
            new_position = SpatialPoint(
                x=bee_state.current_position.x + dx,
                y=bee_state.current_position.y + dy,
                z=bee_state.current_position.z
            )
            
            bee_state.update_position(new_position)
        
        return {'movement_type': 'local_random'}
    
    def _execute_movement(self, bee_state: SpatialBeeState) -> Dict[str, Any]:
        """Execute movement along planned path"""
        
        if not bee_state.planned_path:
            return {'movement_type': 'stationary'}
        
        # Get next patch in path
        next_patch_id = bee_state.planned_path[0]
        
        if next_patch_id in self.environment_manager.spatial_index._patches:
            next_patch = self.environment_manager.spatial_index._patches[next_patch_id]
            target_position = next_patch.center
            
            # Calculate movement vector
            current_pos = bee_state.current_position
            dx = target_position.x - current_pos.x
            dy = target_position.y - current_pos.y
            dz = target_position.z - current_pos.z
            
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if distance <= self.movement_speed:
                # Reached the patch
                bee_state.update_position(target_position, next_patch_id)
                bee_state.planned_path.pop(0)  # Remove reached patch from path
                
                return {
                    'movement_type': 'path_following',
                    'reached_patch': next_patch_id,
                    'distance_traveled': distance
                }
            else:
                # Move toward the patch
                movement_factor = self.movement_speed / distance
                new_x = current_pos.x + dx * movement_factor
                new_y = current_pos.y + dy * movement_factor
                new_z = current_pos.z + dz * movement_factor
                
                new_position = SpatialPoint(x=new_x, y=new_y, z=new_z)
                bee_state.update_position(new_position)
                
                return {
                    'movement_type': 'path_following',
                    'distance_traveled': self.movement_speed,
                    'progress_to_target': movement_factor
                }
        
        return {'movement_type': 'path_error'}
    
    def get_bee_spatial_metrics(self, bee_id: int) -> Dict[str, Any]:
        """Get spatial metrics for a bee"""
        
        if bee_id not in self.bee_states:
            return {}
        
        bee_state = self.bee_states[bee_id]
        
        # Calculate movement statistics
        total_distance = 0.0
        if len(bee_state.movement_history) > 1:
            for i in range(1, len(bee_state.movement_history)):
                total_distance += bee_state.movement_history[i-1].distance_to(bee_state.movement_history[i])
        
        # Spatial memory statistics
        known_patches = len(bee_state.spatial_memory)
        avg_remembered_quality = np.mean(list(bee_state.spatial_memory.values())) if bee_state.spatial_memory else 0
        
        return {
            'current_position': bee_state.current_position,
            'current_patch': bee_state.current_patch,
            'target_patch': bee_state.target_patch,
            'total_distance_traveled': total_distance,
            'known_patches': known_patches,
            'average_remembered_quality': avg_remembered_quality,
            'path_length': len(bee_state.planned_path)
        }

def create_spatial_integration_system(landscape_config: Dict[str, Any]) -> Tuple[SpatialEnvironmentManager, SpatialBeeManager]:
    """Factory function to create integrated spatial system"""
    
    # Create core spatial analysis components
    spatial_index, path_finder, connectivity_analyzer, landscape_analyzer = create_spatial_analysis_system()
    
    # Create GIS integration if requested
    gis_manager = None
    if landscape_config.get('use_gis_data', False):
        gis_config = landscape_config.get('gis_config', {})
        source_crs = gis_config.get('source_crs', 'EPSG:4326')
        target_crs = gis_config.get('target_crs', 'LOCAL')
        gis_manager = create_gis_integration_system(source_crs, target_crs)
    
    # Create environment manager
    environment_manager = SpatialEnvironmentManager(
        spatial_index=spatial_index,
        path_finder=path_finder,
        connectivity_analyzer=connectivity_analyzer,
        landscape_analyzer=landscape_analyzer,
        gis_manager=gis_manager
    )
    
    # Initialize landscape
    environment_manager.initialize_landscape(landscape_config)
    
    # Create bee manager
    bee_manager = SpatialBeeManager(environment_manager=environment_manager)
    
    return environment_manager, bee_manager