"""
Landscape grid system for BSTEW
===============================

Spatial grid system replacing NetLogo patches with NumPy arrays.
Handles landscape representation, resource distribution, and spatial queries.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import math
from scipy.spatial import cKDTree

from .patches import ResourcePatch, HabitatType
from ..components.masterpatch_system import MasterPatchSystem


class LandscapeGrid:
    """
    Spatial grid system replacing NetLogo patches.

    Manages:
    - Landscape representation from RGB images
    - Resource patch grid
    - Spatial queries and neighbor finding
    - Habitat type mapping
    """

    def __init__(self, width: int, height: int, cell_size: float = 20.0):
        """
        Initialize landscape grid.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            cell_size: Size of each cell in meters
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.total_cells = width * height

        # Spatial dimensions
        self.world_width = width * cell_size
        self.world_height = height * cell_size

        # Grid arrays
        self.habitat_grid = np.zeros((height, width), dtype=int)
        self.patch_grid = np.empty((height, width), dtype=object)

        # Masterpatch system integration
        landscape_bounds = (0, 0, self.world_width, self.world_height)
        self.masterpatch_system = MasterPatchSystem(landscape_bounds)
        self.use_masterpatch_system = True

        # Patch storage
        self.patches: Dict[int, ResourcePatch] = {}
        self.patch_counter = 0

        # Spatial index for efficient queries
        self.spatial_index: Optional[cKDTree] = None
        self.patch_positions: List[Tuple[float, float]] = []

        # Color mapping for habitat types
        self.color_to_habitat = self.get_default_color_mapping()

        # Initialize empty grid
        self.initialize_empty_grid()

    def get_default_color_mapping(self) -> Dict[Tuple[int, int, int], HabitatType]:
        """Default color mapping for habitat types"""
        return {
            (0, 255, 0): HabitatType.GRASSLAND,  # Green
            (255, 255, 0): HabitatType.CROPLAND,  # Yellow
            (255, 0, 255): HabitatType.WILDFLOWER,  # Magenta
            (128, 64, 0): HabitatType.WOODLAND,  # Brown
            (0, 128, 0): HabitatType.HEDGEROW,  # Dark green
            (128, 128, 128): HabitatType.URBAN,  # Gray
            (0, 0, 255): HabitatType.WATER,  # Blue
            (255, 255, 255): HabitatType.BARE_SOIL,  # White
            (64, 64, 64): HabitatType.ROAD,  # Dark gray
            (255, 0, 0): HabitatType.BUILDING,  # Red
        }

    def initialize_empty_grid(self) -> None:
        """Initialize grid with empty patches"""
        for y in range(self.height):
            for x in range(self.width):
                # Convert grid coordinates to world coordinates
                world_x = (x + 0.5) * self.cell_size
                world_y = (y + 0.5) * self.cell_size

                # Create patch
                patch = ResourcePatch(
                    self.patch_counter,
                    world_x,
                    world_y,
                    HabitatType.GRASSLAND,  # Default habitat
                )

                self.patches[self.patch_counter] = patch
                self.patch_grid[y, x] = patch
                self.patch_counter += 1

        self.build_spatial_index()

    def load_from_image(
        self,
        image_path: str,
        color_mapping: Optional[Dict[Tuple[int, int, int], HabitatType]] = None,
    ) -> None:
        """
        Load landscape from RGB image.

        Args:
            image_path: Path to landscape image
            color_mapping: Custom color to habitat mapping
        """
        if color_mapping is None:
            color_mapping = self.color_to_habitat

        # Load and resize image
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((self.width, self.height), Image.Resampling.NEAREST)
            rgb_array = np.array(img)

        # Map colors to habitat types
        for y in range(self.height):
            for x in range(self.width):
                pixel_color = tuple(rgb_array[y, x])

                # Find closest color match
                habitat_type = self.find_closest_habitat(pixel_color, color_mapping)

                # Update patch
                patch = self.patch_grid[y, x]
                patch.habitat_type = habitat_type
                patch.initialize_habitat_properties()

                # Update habitat grid
                self.habitat_grid[y, x] = list(HabitatType).index(habitat_type)

        # Initialize masterpatch system if enabled
        if self.use_masterpatch_system:
            self.initialize_masterpatch_system()

    def find_closest_habitat(
        self,
        pixel_color: Tuple[int, int, int],
        color_mapping: Dict[Tuple[int, int, int], HabitatType],
    ) -> HabitatType:
        """Find closest habitat type for a pixel color"""
        min_distance = float("inf")
        closest_habitat = HabitatType.GRASSLAND

        for color, habitat in color_mapping.items():
            # Calculate Euclidean distance in RGB space
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pixel_color, color)))

            if distance < min_distance:
                min_distance = distance
                closest_habitat = habitat

        return closest_habitat

    def build_spatial_index(self) -> None:
        """Build spatial index for efficient neighbor queries"""
        self.patch_positions = []

        for patch in self.patches.values():
            self.patch_positions.append((patch.x, patch.y))

        if self.patch_positions:
            self.spatial_index = cKDTree(self.patch_positions)
        else:
            self.spatial_index = None

    def get_patch(self, patch_id: int) -> Optional[ResourcePatch]:
        """Get patch by ID"""
        return self.patches.get(patch_id)

    def get_patch_at_position(self, x: float, y: float) -> Optional[ResourcePatch]:
        """Get patch at world coordinates"""
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)

        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            patch = self.patch_grid[grid_y, grid_x]
            return patch if isinstance(patch, ResourcePatch) else None
        else:
            return None

    def get_patches_in_radius(
        self, center: Tuple[float, float], radius: float
    ) -> List[ResourcePatch]:
        """Get all patches within radius of center point"""
        if self.spatial_index is None:
            return []

        # Query spatial index
        indices = self.spatial_index.query_ball_point(center, radius)

        # Return corresponding patches
        patches = []
        for idx in indices:
            patch_id = list(self.patches.keys())[idx]
            patches.append(self.patches[patch_id])

        return patches

    def get_patches_in_rectangle(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> List[ResourcePatch]:
        """Get all patches within rectangular bounds"""
        patches = []

        for patch in self.patches.values():
            if min_x <= patch.x <= max_x and min_y <= patch.y <= max_y:
                patches.append(patch)

        return patches

    def get_patches_by_habitat(self, habitat_type: HabitatType) -> List[ResourcePatch]:
        """Get all patches of specific habitat type"""
        return [
            patch
            for patch in self.patches.values()
            if patch.habitat_type == habitat_type
        ]

    def get_neighbor_patches(
        self, center_patch: ResourcePatch, neighborhood_type: str = "moore"
    ) -> List[ResourcePatch]:
        """
        Get neighboring patches.

        Args:
            center_patch: Center patch
            neighborhood_type: "moore" (8-connected) or "von_neumann" (4-connected)
        """
        # Convert to grid coordinates
        grid_x = int(center_patch.x / self.cell_size)
        grid_y = int(center_patch.y / self.cell_size)

        neighbors = []

        if neighborhood_type == "moore":
            # 8-connected neighborhood
            offsets = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        else:
            # 4-connected neighborhood
            offsets = [(0, -1), (-1, 0), (1, 0), (0, 1)]

        for dx, dy in offsets:
            nx, ny = grid_x + dx, grid_y + dy

            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append(self.patch_grid[ny, nx])

        return neighbors

    def calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx**2 + dy**2)

    def calculate_manhattan_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def find_path(
        self, start: Tuple[float, float], end: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """
        Find path between two points using A* algorithm.
        Simplified implementation for basic pathfinding.
        """
        # Convert to grid coordinates
        start_grid = (int(start[0] / self.cell_size), int(start[1] / self.cell_size))
        end_grid = (int(end[0] / self.cell_size), int(end[1] / self.cell_size))

        # Simple direct line for now (could implement full A*)
        path = []

        steps = max(abs(end_grid[0] - start_grid[0]), abs(end_grid[1] - start_grid[1]))

        if steps == 0:
            return [start, end]

        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            path.append((x, y))

        return path

    def update_all_patches(self, day_of_year: int, weather: Dict[str, float]) -> None:
        """Update all patches for current time step"""
        for patch in self.patches.values():
            patch.update_resources(day_of_year, weather)

    def get_habitat_distribution(self) -> Dict[HabitatType, float]:
        """Get distribution of habitat types as percentages"""
        distribution = {}

        for habitat_type in HabitatType:
            count = sum(
                1
                for patch in self.patches.values()
                if patch.habitat_type == habitat_type
            )
            percentage = (count / len(self.patches)) * 100.0
            distribution[habitat_type] = percentage

        return distribution

    def get_resource_hotspots(self, min_quality: float = 0.5) -> List[ResourcePatch]:
        """Get patches with high resource quality"""
        return [
            patch
            for patch in self.patches.values()
            if patch.get_resource_quality() >= min_quality
        ]

    def get_total_resources(self) -> Dict[str, float]:
        """Get total available resources in landscape"""
        total_nectar = sum(patch.current_nectar for patch in self.patches.values())
        total_pollen = sum(patch.current_pollen for patch in self.patches.values())

        return {
            "nectar": total_nectar,
            "pollen": total_pollen,
            "total": total_nectar + total_pollen,
        }

    def calculate_landscape_connectivity(self, max_distance: float = 500.0) -> float:
        """
        Calculate landscape connectivity for pollinators.
        Based on habitat patch connectivity within flight range.
        """
        resource_patches = [
            patch for patch in self.patches.values() if patch.has_resources()
        ]

        if len(resource_patches) < 2:
            return 0.0

        connected_pairs = 0
        total_pairs = 0

        for i, patch1 in enumerate(resource_patches):
            for patch2 in resource_patches[i + 1 :]:
                total_pairs += 1
                distance = self.calculate_distance(patch1.location, patch2.location)

                if distance <= max_distance:
                    connected_pairs += 1

        return connected_pairs / total_pairs if total_pairs > 0 else 0.0

    def export_to_dict(self) -> Dict[str, Any]:
        """Export landscape data to dictionary"""
        return {
            "width": self.width,
            "height": self.height,
            "cell_size": self.cell_size,
            "patch_count": len(self.patches),
            "habitat_distribution": {
                ht.value: pct for ht, pct in self.get_habitat_distribution().items()
            },
            "total_resources": self.get_total_resources(),
            "patches": [patch.to_dict() for patch in self.patches.values()],
        }

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import landscape data from dictionary"""
        # Clear existing data
        self.patches.clear()

        # Import patches
        for patch_data in data["patches"]:
            patch = ResourcePatch.from_dict(patch_data)
            self.patches[patch.id] = patch

            # Update grid
            grid_x = int(patch.x / self.cell_size)
            grid_y = int(patch.y / self.cell_size)

            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                self.patch_grid[grid_y, grid_x] = patch

        # Rebuild spatial index
        self.build_spatial_index()

    def save_to_file(self, filepath: str) -> None:
        """Save landscape to file"""
        import json

        data = self.export_to_dict()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str) -> None:
        """Load landscape from file"""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        self.import_from_dict(data)

        # Add logger if not exists
        if not hasattr(self, "logger"):
            import logging

            self.logger = logging.getLogger(__name__)

    def initialize_masterpatch_system(self) -> None:
        """Initialize masterpatch system from current landscape"""
        if not self.use_masterpatch_system:
            return

        # Initialize species database
        from .resources import ResourceDistribution

        resource_dist = ResourceDistribution(self)
        species_list = list(resource_dist.species_database.values())
        self.masterpatch_system.initialize_species_database(species_list)

        # Create masterpatches from current patches
        patch_count = self.masterpatch_system.create_habitat_based_landscape(
            self.habitat_grid, self.cell_size
        )

        self.logger.info(f"Initialized masterpatch system with {patch_count} patches")

    def get_masterpatch_in_radius(
        self, center: Tuple[float, float], radius: float
    ) -> List[Any]:
        """Get masterpatches within radius using masterpatch system"""
        if self.use_masterpatch_system:
            return self.masterpatch_system.get_patches_in_radius(center, radius)
        else:
            # Fallback to regular patches
            return self.get_patches_in_radius(center, radius)

    def get_best_foraging_patches(
        self,
        bee_species: str,
        center: Tuple[float, float],
        radius: float,
        proboscis_system: Any,
    ) -> List[Tuple[Any, float]]:
        """Get best foraging patches for bee species"""
        if self.use_masterpatch_system:
            return self.masterpatch_system.get_best_patches_for_species(
                bee_species, center, radius, proboscis_system
            )
        else:
            # Fallback to regular patch scoring
            nearby_patches = self.get_patches_in_radius(center, radius)
            patch_scores = []

            for patch in nearby_patches:
                quality = patch.get_resource_quality()
                distance = self.calculate_distance(center, patch.location)
                score = quality / (distance + 1)
                patch_scores.append((patch, score))

            patch_scores.sort(key=lambda x: x[1], reverse=True)
            return patch_scores

    def update_masterpatch_system(
        self, day_of_year: int, weather_conditions: Dict[str, float]
    ) -> None:
        """Update masterpatch system for current day"""
        if self.use_masterpatch_system:
            self.masterpatch_system.update_all_patches(day_of_year, weather_conditions)

    def simulate_foraging_on_masterpatch(
        self,
        patch_id: str,
        bee_species: str,
        duration_hours: float,
        proboscis_system: Any,
    ) -> Dict[str, Any]:
        """Simulate foraging impact on masterpatch"""
        if self.use_masterpatch_system:
            return self.masterpatch_system.simulate_foraging_impact(
                patch_id, bee_species, duration_hours, proboscis_system
            )
        else:
            return {}

    def get_landscape_carrying_capacity_masterpatch(self) -> Dict[str, float]:
        """Get landscape carrying capacity using masterpatch system"""
        if self.use_masterpatch_system:
            return self.masterpatch_system.get_landscape_carrying_capacity()
        else:
            # Fallback to regular calculation
            return self.calculate_carrying_capacity()

    def schedule_management_on_patches(
        self,
        day_of_year: int,
        patch_ids: List[str],
        management_type: str,
        intensity: float,
    ) -> None:
        """Schedule management on specific patches"""
        if self.use_masterpatch_system:
            self.masterpatch_system.schedule_management(
                day_of_year, patch_ids, management_type, intensity
            )

    def export_masterpatch_data(self, day_of_year: int) -> Dict[str, Any]:
        """Export masterpatch data for analysis"""
        if self.use_masterpatch_system:
            return self.masterpatch_system.export_patch_data(day_of_year)
        else:
            return self.export_to_dict()

    def get_species_layers_at_location(
        self, location: Tuple[float, float]
    ) -> List[str]:
        """Get species layers available at location"""
        if not self.use_masterpatch_system:
            return []

        # Find nearest masterpatch
        nearest_patches = self.masterpatch_system.get_patches_in_radius(
            location, self.cell_size
        )

        if nearest_patches:
            patch = nearest_patches[0]
            return [layer.flower_species.name for layer in patch.layers.values()]

        return []

    def get_patch_species_accessibility(
        self, patch_id: str, bee_species: str, proboscis_system: Any
    ) -> Dict[str, float]:
        """Get species accessibility scores for patch"""
        if not self.use_masterpatch_system:
            return {}

        if patch_id in self.masterpatch_system.masterpatches:
            patch = self.masterpatch_system.masterpatches[patch_id]
            return patch.get_species_accessibility_for_bee(
                bee_species, proboscis_system
            )

        return {}

    def calculate_carrying_capacity(self) -> Dict[str, float]:
        """Calculate basic carrying capacity (fallback method)"""
        # Simple calculation for backward compatibility
        total_nectar = 0.0
        total_pollen = 0.0

        for patch in self.patches.values():
            total_nectar += patch.current_nectar
            total_pollen += patch.current_pollen

        # Estimate annual production (simplified)
        annual_nectar = total_nectar * 100  # Rough multiplier
        annual_pollen = total_pollen * 100

        nectar_colonies = annual_nectar / 50000  # 50g per colony
        pollen_colonies = annual_pollen / 20000  # 20g per colony

        return {
            "nectar_limited": nectar_colonies,
            "pollen_limited": pollen_colonies,
            "overall": min(nectar_colonies, pollen_colonies),
        }
