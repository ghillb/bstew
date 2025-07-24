"""
Habitat Creation Algorithms
===========================

Advanced algorithms for optimizing habitat creation, including nesting site
selection, foraging corridor design, and resource density calculations for
bee population support.
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import heapq

from .stewardship import HabitatType
from ..spatial.patches import ResourcePatch


class NestingSiteType(Enum):
    """Types of bee nesting sites"""

    GROUND_CAVITY = "ground_cavity"  # Bumblebees
    HOLLOW_STEMS = "hollow_stems"  # Small solitary bees
    WOOD_CAVITIES = "wood_cavities"  # Carpenter bees
    BARE_GROUND = "bare_ground"  # Mining bees
    SANDY_BANKS = "sandy_banks"  # Aggregation sites
    HEDGE_BANKS = "hedge_banks"  # Multiple species
    ARTIFICIAL_BOXES = "artificial_boxes"  # Managed sites


class BeeSpeciesGroup(Enum):
    """Bee species groups with different requirements"""

    BUMBLEBEES = "bumblebees"
    HONEYBEES = "honeybees"
    SOLITARY_GROUND = "solitary_ground"
    SOLITARY_CAVITY = "solitary_cavity"
    MINING_BEES = "mining_bees"
    MASON_BEES = "mason_bees"
    LEAFCUTTER_BEES = "leafcutter_bees"


class CorridorType(Enum):
    """Types of foraging corridors"""

    LINEAR = "linear"  # Direct path
    STEPPING_STONE = "stepping_stone"  # Discrete patches
    RIPARIAN = "riparian"  # Following water
    HEDGEROW = "hedgerow"  # Along field boundaries
    WILDFLOWER_STRIP = "wildflower_strip"  # Dedicated strips
    MIXED = "mixed"  # Combination


@dataclass
class NestingSiteRequirements:
    """Requirements for bee nesting sites"""

    species_group: BeeSpeciesGroup
    site_type: NestingSiteType
    min_area_m2: float
    max_distance_to_forage_m: float
    aspect_preference: Optional[str] = None  # north, south, east, west
    slope_preference: Optional[str] = None  # flat, gentle, steep
    soil_type_preference: Optional[List[str]] = None
    vegetation_cover_percent: Tuple[float, float] = (0, 100)  # min, max
    proximity_requirements: Dict[str, float] = field(
        default_factory=dict
    )  # feature -> max_distance


@dataclass
class ForagingRequirements:
    """Foraging requirements for bee populations"""

    species_group: BeeSpeciesGroup
    foraging_range_m: float
    min_patch_size_m2: float
    preferred_flower_types: List[str]
    nectar_demand_mg_per_day: float
    pollen_demand_mg_per_day: float
    active_months: List[int]  # 1-12
    flight_speed_m_per_s: float = 4.0  # Average flight speed


@dataclass
class NestingSite:
    """Optimized nesting site location"""

    site_id: str
    location: Tuple[float, float]  # x, y coordinates
    site_type: NestingSiteType
    species_groups: List[BeeSpeciesGroup]
    capacity: int  # Number of nests
    quality_score: float  # 0-1
    area_m2: float
    nearby_resources: Dict[str, float]  # resource_type -> distance
    establishment_cost: float
    maintenance_requirements: List[str]


@dataclass
class ForagingCorridor:
    """Foraging corridor connecting habitats"""

    corridor_id: str
    corridor_type: CorridorType
    path: List[Tuple[float, float]]  # Waypoints
    width_m: float
    length_m: float
    connected_patches: List[int]  # Patch IDs
    resource_types: List[str]
    quality_score: float  # 0-1
    establishment_cost: float
    maintenance_cost_annual: float


class ResourceDensityGrid:
    """Grid-based resource density calculations"""

    def __init__(self, bounds: Dict[str, float], resolution_m: float = 50):
        self.bounds = bounds
        self.resolution = resolution_m

        # Calculate grid dimensions
        self.width = int((bounds["max_x"] - bounds["min_x"]) / resolution_m)
        self.height = int((bounds["max_y"] - bounds["min_y"]) / resolution_m)

        # Resource grids
        self.nectar_grid = np.zeros((self.height, self.width))
        self.pollen_grid = np.zeros((self.height, self.width))
        self.nesting_grid = np.zeros((self.height, self.width))

        # Accessibility grid (considers barriers)
        self.accessibility_grid = np.ones((self.height, self.width))

    def add_resource_patch(
        self, patch: ResourcePatch, influence_radius_m: float = 200
    ) -> None:
        """Add a resource patch to the density grid"""
        # Convert patch location to grid coordinates
        grid_x = int((patch.x - self.bounds["min_x"]) / self.resolution)
        grid_y = int((patch.y - self.bounds["min_y"]) / self.resolution)

        # Calculate influence area
        influence_cells = int(influence_radius_m / self.resolution)

        for dy in range(-influence_cells, influence_cells + 1):
            for dx in range(-influence_cells, influence_cells + 1):
                y = grid_y + dy
                x = grid_x + dx

                if 0 <= y < self.height and 0 <= x < self.width:
                    # Distance-based influence
                    distance = np.sqrt(dx**2 + dy**2) * self.resolution
                    if distance <= influence_radius_m:
                        influence = 1.0 - (distance / influence_radius_m)

                        self.nectar_grid[y, x] += (
                            patch.base_nectar_production * influence
                        )
                        self.pollen_grid[y, x] += (
                            patch.base_pollen_production * influence
                        )

                        # Nesting suitability based on habitat type
                        nesting_values = {
                            HabitatType.HEDGEROW: 0.9,
                            HabitatType.WOODLAND: 0.8,
                            HabitatType.GRASSLAND: 0.6,
                            HabitatType.WILDFLOWER: 0.5,
                            HabitatType.CROPLAND: 0.2,
                            HabitatType.URBAN: 0.1,
                            HabitatType.WATER: 0.0,
                            HabitatType.BARE_SOIL: 0.3,
                        }
                        self.nesting_grid[y, x] += (
                            nesting_values.get(patch.habitat_type, 0.3) * influence
                        )

    def get_resource_at_point(self, x: float, y: float) -> Dict[str, float]:
        """Get resource values at a specific point"""
        grid_x = int((x - self.bounds["min_x"]) / self.resolution)
        grid_y = int((y - self.bounds["min_y"]) / self.resolution)

        if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
            return {
                "nectar": self.nectar_grid[grid_y, grid_x],
                "pollen": self.pollen_grid[grid_y, grid_x],
                "nesting": self.nesting_grid[grid_y, grid_x],
                "accessibility": self.accessibility_grid[grid_y, grid_x],
            }
        return {"nectar": 0, "pollen": 0, "nesting": 0, "accessibility": 0}


class HabitatCreationAlgorithms:
    """
    Advanced algorithms for habitat creation optimization.

    Includes:
    - Nesting site optimization using spatial analysis
    - Foraging corridor design using graph algorithms
    - Resource density calculations and hotspot identification
    - Multi-species habitat requirements balancing
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Species requirements database
        self.nesting_requirements = self._initialize_nesting_requirements()
        self.foraging_requirements = self._initialize_foraging_requirements()

        # Optimization parameters
        self.optimization_weights = {
            "resource_proximity": 0.3,
            "site_quality": 0.3,
            "connectivity": 0.2,
            "cost_efficiency": 0.2,
        }

        # Cached calculations
        self.distance_matrix_cache: Dict[str, np.ndarray] = {}
        self.corridor_cache: Dict[Tuple[int, int], ForagingCorridor] = {}

    def _initialize_nesting_requirements(
        self,
    ) -> Dict[BeeSpeciesGroup, NestingSiteRequirements]:
        """Initialize nesting requirements for different bee groups"""
        return {
            BeeSpeciesGroup.BUMBLEBEES: NestingSiteRequirements(
                species_group=BeeSpeciesGroup.BUMBLEBEES,
                site_type=NestingSiteType.GROUND_CAVITY,
                min_area_m2=1.0,
                max_distance_to_forage_m=1500,
                aspect_preference="south",
                slope_preference="gentle",
                vegetation_cover_percent=(30, 70),
                proximity_requirements={"water": 500, "flowers": 200},
            ),
            BeeSpeciesGroup.SOLITARY_GROUND: NestingSiteRequirements(
                species_group=BeeSpeciesGroup.SOLITARY_GROUND,
                site_type=NestingSiteType.BARE_GROUND,
                min_area_m2=0.5,
                max_distance_to_forage_m=300,
                aspect_preference="south",
                slope_preference="flat",
                soil_type_preference=["sandy", "loamy"],
                vegetation_cover_percent=(0, 30),
            ),
            BeeSpeciesGroup.MINING_BEES: NestingSiteRequirements(
                species_group=BeeSpeciesGroup.MINING_BEES,
                site_type=NestingSiteType.SANDY_BANKS,
                min_area_m2=2.0,
                max_distance_to_forage_m=500,
                aspect_preference="south",
                slope_preference="steep",
                soil_type_preference=["sandy"],
                vegetation_cover_percent=(0, 20),
            ),
            BeeSpeciesGroup.MASON_BEES: NestingSiteRequirements(
                species_group=BeeSpeciesGroup.MASON_BEES,
                site_type=NestingSiteType.HOLLOW_STEMS,
                min_area_m2=0.1,
                max_distance_to_forage_m=300,
                proximity_requirements={"flowers": 100, "mud": 50},
            ),
            BeeSpeciesGroup.LEAFCUTTER_BEES: NestingSiteRequirements(
                species_group=BeeSpeciesGroup.LEAFCUTTER_BEES,
                site_type=NestingSiteType.HOLLOW_STEMS,
                min_area_m2=0.1,
                max_distance_to_forage_m=400,
                proximity_requirements={"flowers": 150, "leaves": 100},
            ),
        }

    def _initialize_foraging_requirements(
        self,
    ) -> Dict[BeeSpeciesGroup, ForagingRequirements]:
        """Initialize foraging requirements for different bee groups"""
        return {
            BeeSpeciesGroup.BUMBLEBEES: ForagingRequirements(
                species_group=BeeSpeciesGroup.BUMBLEBEES,
                foraging_range_m=1500,
                min_patch_size_m2=100,
                preferred_flower_types=["tubular", "complex", "open"],
                nectar_demand_mg_per_day=500,
                pollen_demand_mg_per_day=300,
                active_months=[3, 4, 5, 6, 7, 8, 9, 10],
                flight_speed_m_per_s=5.0,
            ),
            BeeSpeciesGroup.HONEYBEES: ForagingRequirements(
                species_group=BeeSpeciesGroup.HONEYBEES,
                foraging_range_m=3000,
                min_patch_size_m2=500,
                preferred_flower_types=["open", "shallow"],
                nectar_demand_mg_per_day=1000,  # Colony demand
                pollen_demand_mg_per_day=500,
                active_months=[3, 4, 5, 6, 7, 8, 9, 10],
                flight_speed_m_per_s=6.0,
            ),
            BeeSpeciesGroup.SOLITARY_GROUND: ForagingRequirements(
                species_group=BeeSpeciesGroup.SOLITARY_GROUND,
                foraging_range_m=300,
                min_patch_size_m2=10,
                preferred_flower_types=["open", "flat"],
                nectar_demand_mg_per_day=50,
                pollen_demand_mg_per_day=40,
                active_months=[4, 5, 6, 7, 8],
                flight_speed_m_per_s=3.0,
            ),
            BeeSpeciesGroup.MINING_BEES: ForagingRequirements(
                species_group=BeeSpeciesGroup.MINING_BEES,
                foraging_range_m=500,
                min_patch_size_m2=20,
                preferred_flower_types=["spring_flowers", "willow", "fruit_trees"],
                nectar_demand_mg_per_day=60,
                pollen_demand_mg_per_day=50,
                active_months=[3, 4, 5, 6],
                flight_speed_m_per_s=3.5,
            ),
        }

    def optimize_nesting_sites(
        self,
        available_area: List[
            Tuple[float, float, float, float]
        ],  # (x, y, width, height)
        resource_patches: List[ResourcePatch],
        target_species: List[BeeSpeciesGroup],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[NestingSite]:
        """
        Optimize nesting site locations using spatial analysis.

        Args:
            available_area: List of available areas (rectangles)
            resource_patches: Existing resource patches
            target_species: Target bee species groups
            constraints: Optional constraints (budget, area limits, etc.)

        Returns:
            List of optimized nesting site locations
        """

        nesting_sites = []

        # Create resource density grid
        bounds = self._calculate_area_bounds(available_area, resource_patches)
        density_grid = ResourceDensityGrid(bounds, resolution_m=25)

        # Add resource patches to grid
        for patch in resource_patches:
            density_grid.add_resource_patch(patch)

        # For each target species, find optimal sites
        for species_group in target_species:
            requirements = self.nesting_requirements.get(species_group)
            if not requirements:
                continue

            # Find candidate locations
            candidates = self._find_nesting_candidates(
                available_area, density_grid, requirements
            )

            # Score and rank candidates
            scored_candidates = []
            for candidate in candidates:
                score = self._score_nesting_site(
                    candidate, resource_patches, requirements, density_grid
                )
                scored_candidates.append((score, candidate))

            # Select top candidates
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Create nesting sites from top candidates
            num_sites = min(
                len(scored_candidates),
                constraints.get("max_sites_per_species", 5) if constraints else 5,
            )

            for i in range(num_sites):
                score, location = scored_candidates[i]

                site = NestingSite(
                    site_id=f"nest_{species_group.value}_{i}",
                    location=location,
                    site_type=requirements.site_type,
                    species_groups=[species_group],
                    capacity=self._estimate_site_capacity(requirements, score),
                    quality_score=score,
                    area_m2=requirements.min_area_m2,
                    nearby_resources=self._find_nearby_resources(
                        location,
                        resource_patches,
                        requirements.max_distance_to_forage_m,
                    ),
                    establishment_cost=self._estimate_nesting_site_cost(
                        requirements.site_type, requirements.min_area_m2
                    ),
                    maintenance_requirements=self._get_maintenance_requirements(
                        requirements.site_type
                    ),
                )

                nesting_sites.append(site)

        # Merge nearby sites if they can support multiple species
        merged_sites = self._merge_compatible_sites(nesting_sites)

        self.logger.info(
            f"Optimized {len(merged_sites)} nesting sites for "
            f"{len(target_species)} species groups"
        )

        return merged_sites

    def _calculate_area_bounds(
        self,
        areas: List[Tuple[float, float, float, float]],
        patches: List[ResourcePatch],
    ) -> Dict[str, float]:
        """Calculate overall bounds for analysis area"""
        all_x = []
        all_y = []

        for x, y, w, h in areas:
            all_x.extend([x, x + w])
            all_y.extend([y, y + h])

        for patch in patches:
            all_x.append(patch.x)
            all_y.append(patch.y)

        return {
            "min_x": min(all_x) - 100,
            "max_x": max(all_x) + 100,
            "min_y": min(all_y) - 100,
            "max_y": max(all_y) + 100,
        }

    def _find_nesting_candidates(
        self,
        available_areas: List[Tuple[float, float, float, float]],
        density_grid: ResourceDensityGrid,
        requirements: NestingSiteRequirements,
    ) -> List[Tuple[float, float]]:
        """Find candidate nesting locations within available areas"""
        candidates = []

        # Sample points within available areas
        for x, y, width, height in available_areas:
            # Grid sampling
            sample_spacing = max(10, int(np.sqrt(requirements.min_area_m2)))

            for dx in range(0, int(width), sample_spacing):
                for dy in range(0, int(height), sample_spacing):
                    point_x = x + dx
                    point_y = y + dy

                    # Check basic suitability
                    resources = density_grid.get_resource_at_point(point_x, point_y)

                    if resources["accessibility"] > 0.5:  # Basic accessibility check
                        candidates.append((point_x, point_y))

        return candidates

    def _score_nesting_site(
        self,
        location: Tuple[float, float],
        resource_patches: List[ResourcePatch],
        requirements: NestingSiteRequirements,
        density_grid: ResourceDensityGrid,
    ) -> float:
        """Score a potential nesting site location"""
        x, y = location

        # Get resource density at location
        resources = density_grid.get_resource_at_point(x, y)

        # Base nesting suitability
        nesting_score = resources["nesting"]

        # Resource proximity score
        nectar_score = min(resources["nectar"] / 10.0, 1.0)  # Normalize
        pollen_score = min(resources["pollen"] / 10.0, 1.0)
        resource_score = (nectar_score + pollen_score) / 2

        # Distance to nearest foraging patches
        min_distance = float("inf")
        suitable_patches = 0

        for patch in resource_patches:
            distance = np.sqrt((patch.x - x) ** 2 + (patch.y - y) ** 2)

            if distance <= requirements.max_distance_to_forage_m:
                suitable_patches += 1
                min_distance = min(min_distance, distance)

        distance_score = (
            1.0 - (min_distance / requirements.max_distance_to_forage_m)
            if min_distance < float("inf")
            else 0
        )
        patch_diversity_score = min(
            suitable_patches / 10.0, 1.0
        )  # Normalize to 10 patches

        # Combine scores
        final_score = (
            self.optimization_weights["site_quality"] * nesting_score
            + self.optimization_weights["resource_proximity"] * resource_score
            + self.optimization_weights["connectivity"] * distance_score
            + 0.1 * patch_diversity_score  # Bonus for diversity
        )

        return min(final_score, 1.0)

    def _estimate_site_capacity(
        self, requirements: NestingSiteRequirements, quality_score: float
    ) -> int:
        """Estimate nesting capacity based on requirements and quality"""
        base_capacity = {
            NestingSiteType.GROUND_CAVITY: 10,
            NestingSiteType.HOLLOW_STEMS: 50,
            NestingSiteType.WOOD_CAVITIES: 20,
            NestingSiteType.BARE_GROUND: 100,
            NestingSiteType.SANDY_BANKS: 200,
            NestingSiteType.HEDGE_BANKS: 30,
            NestingSiteType.ARTIFICIAL_BOXES: 40,
        }

        capacity = base_capacity.get(requirements.site_type, 20)

        # Adjust by area
        area_factor = requirements.min_area_m2 / 1.0  # Normalized to 1m²

        # Adjust by quality
        quality_factor = 0.5 + quality_score  # 0.5 to 1.5 multiplier

        return int(capacity * area_factor * quality_factor)

    def _find_nearby_resources(
        self,
        location: Tuple[float, float],
        patches: List[ResourcePatch],
        max_distance: float,
    ) -> Dict[str, float]:
        """Find resources near a location"""
        x, y = location
        resources = defaultdict(list)

        for patch in patches:
            distance = np.sqrt((patch.x - x) ** 2 + (patch.y - y) ** 2)

            if distance <= max_distance:
                resources[patch.habitat_type.value].append(distance)

        # Return minimum distance to each resource type
        return {
            resource_type: min(distances)
            for resource_type, distances in resources.items()
        }

    def _estimate_nesting_site_cost(
        self, site_type: NestingSiteType, area_m2: float
    ) -> float:
        """Estimate cost to establish nesting site"""
        cost_per_m2 = {
            NestingSiteType.GROUND_CAVITY: 10.0,
            NestingSiteType.HOLLOW_STEMS: 50.0,  # Bee hotels
            NestingSiteType.WOOD_CAVITIES: 30.0,
            NestingSiteType.BARE_GROUND: 5.0,  # Clearing/preparation
            NestingSiteType.SANDY_BANKS: 20.0,  # Earthworks
            NestingSiteType.HEDGE_BANKS: 40.0,
            NestingSiteType.ARTIFICIAL_BOXES: 100.0,  # Purchase and install
        }

        base_cost = cost_per_m2.get(site_type, 20.0)
        return base_cost * area_m2

    def _get_maintenance_requirements(self, site_type: NestingSiteType) -> List[str]:
        """Get maintenance requirements for nesting site type"""
        requirements = {
            NestingSiteType.GROUND_CAVITY: [
                "vegetation_management",
                "predator_control",
            ],
            NestingSiteType.HOLLOW_STEMS: ["annual_replacement", "cleaning"],
            NestingSiteType.WOOD_CAVITIES: ["monitoring", "repair"],
            NestingSiteType.BARE_GROUND: ["vegetation_removal", "soil_management"],
            NestingSiteType.SANDY_BANKS: ["erosion_control", "vegetation_removal"],
            NestingSiteType.HEDGE_BANKS: ["hedge_maintenance", "access_management"],
            NestingSiteType.ARTIFICIAL_BOXES: ["cleaning", "replacement", "monitoring"],
        }

        return requirements.get(site_type, ["general_maintenance"])

    def _merge_compatible_sites(self, sites: List[NestingSite]) -> List[NestingSite]:
        """Merge nearby compatible nesting sites"""
        if len(sites) <= 1:
            return sites

        # Group sites by proximity
        merge_distance = 50  # meters
        merged = []
        used = set()

        for i, site1 in enumerate(sites):
            if i in used:
                continue

            # Find nearby sites
            nearby = [site1]
            used.add(i)

            for j, site2 in enumerate(sites):
                if j in used:
                    continue

                distance = np.sqrt(
                    (site1.location[0] - site2.location[0]) ** 2
                    + (site1.location[1] - site2.location[1]) ** 2
                )

                if distance <= merge_distance:
                    # Check compatibility
                    if self._are_sites_compatible(site1, site2):
                        nearby.append(site2)
                        used.add(j)

            # Create merged site if multiple found
            if len(nearby) > 1:
                merged_site = self._create_merged_site(nearby)
                merged.append(merged_site)
            else:
                merged.append(site1)

        return merged

    def _are_sites_compatible(self, site1: NestingSite, site2: NestingSite) -> bool:
        """Check if two nesting sites can be merged"""
        # Same site type or compatible types
        compatible_types = {
            (NestingSiteType.HOLLOW_STEMS, NestingSiteType.WOOD_CAVITIES),
            (NestingSiteType.GROUND_CAVITY, NestingSiteType.HEDGE_BANKS),
            (NestingSiteType.BARE_GROUND, NestingSiteType.SANDY_BANKS),
        }

        if site1.site_type == site2.site_type:
            return True

        type_pair = (site1.site_type, site2.site_type)
        return type_pair in compatible_types or type_pair[::-1] in compatible_types

    def _create_merged_site(self, sites: List[NestingSite]) -> NestingSite:
        """Create a merged nesting site from multiple sites"""
        # Average location
        avg_x = np.mean([s.location[0] for s in sites])
        avg_y = np.mean([s.location[1] for s in sites])

        # Combine species groups
        all_species = []
        for site in sites:
            all_species.extend(site.species_groups)
        unique_species = list(set(all_species))

        # Sum capacities and areas
        total_capacity = sum(s.capacity for s in sites)
        total_area = sum(s.area_m2 for s in sites)

        # Average quality
        avg_quality = np.mean([s.quality_score for s in sites])

        # Combine maintenance requirements
        all_maintenance = []
        for site in sites:
            all_maintenance.extend(site.maintenance_requirements)
        unique_maintenance = list(set(all_maintenance))

        return NestingSite(
            site_id=f"merged_{sites[0].site_id}",
            location=(float(avg_x), float(avg_y)),
            site_type=sites[0].site_type,  # Use primary type
            species_groups=unique_species,
            capacity=total_capacity,
            quality_score=float(avg_quality),
            area_m2=total_area,
            nearby_resources=sites[0].nearby_resources,  # Use first site's resources
            establishment_cost=sum(s.establishment_cost for s in sites),
            maintenance_requirements=unique_maintenance,
        )

    def design_foraging_corridors(
        self,
        habitat_patches: List[ResourcePatch],
        target_connectivity: float = 0.8,
        corridor_types: Optional[List[CorridorType]] = None,
        budget_constraint: Optional[float] = None,
    ) -> List[ForagingCorridor]:
        """
        Design optimal foraging corridors between habitat patches.

        Uses graph algorithms to create efficient corridor networks that
        maximize connectivity while minimizing cost.

        Args:
            habitat_patches: Existing habitat patches to connect
            target_connectivity: Target connectivity score (0-1)
            corridor_types: Allowed corridor types
            budget_constraint: Maximum budget for corridors

        Returns:
            List of designed foraging corridors
        """

        if not corridor_types:
            corridor_types = list(CorridorType)

        # Build distance matrix
        n_patches = len(habitat_patches)
        distance_matrix = np.zeros((n_patches, n_patches))

        for i in range(n_patches):
            for j in range(i + 1, n_patches):
                distance = np.sqrt(
                    (habitat_patches[i].x - habitat_patches[j].x) ** 2
                    + (habitat_patches[i].y - habitat_patches[j].y) ** 2
                )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Find critical connections (minimum spanning tree)
        critical_connections = self._find_critical_connections(
            habitat_patches, distance_matrix
        )

        # Design corridors for critical connections
        corridors = []
        total_cost = 0.0

        for i, j, distance in critical_connections:
            if budget_constraint and total_cost >= budget_constraint:
                break

            # Select corridor type based on distance and terrain
            corridor_type = self._select_corridor_type(
                habitat_patches[i], habitat_patches[j], distance, corridor_types
            )

            # Design corridor path
            path = self._design_corridor_path(
                habitat_patches[i], habitat_patches[j], corridor_type
            )

            # Calculate corridor properties
            width = self._calculate_corridor_width(distance, corridor_type)
            quality = self._calculate_corridor_quality(
                habitat_patches[i], habitat_patches[j], distance, corridor_type
            )

            # Estimate costs
            establishment_cost = self._estimate_corridor_cost(
                distance, width, corridor_type
            )

            if (
                budget_constraint
                and total_cost + establishment_cost > budget_constraint
            ):
                continue

            corridor = ForagingCorridor(
                corridor_id=f"corridor_{i}_{j}",
                corridor_type=corridor_type,
                path=path,
                width_m=width,
                length_m=distance,
                connected_patches=[habitat_patches[i].id, habitat_patches[j].id],
                resource_types=self._get_corridor_resources(corridor_type),
                quality_score=quality,
                establishment_cost=establishment_cost,
                maintenance_cost_annual=establishment_cost * 0.1,  # 10% annual
            )

            corridors.append(corridor)
            total_cost += establishment_cost

            # Cache corridor
            self.corridor_cache[(i, j)] = corridor

        # Add additional corridors if needed for target connectivity
        current_connectivity = self._calculate_network_connectivity(
            habitat_patches, corridors
        )

        if current_connectivity < target_connectivity:
            additional_corridors = self._add_redundant_corridors(
                habitat_patches,
                corridors,
                distance_matrix,
                target_connectivity,
                budget_constraint,
                total_cost,
            )
            corridors.extend(additional_corridors)

        self.logger.info(
            f"Designed {len(corridors)} corridors connecting "
            f"{len(habitat_patches)} patches"
        )

        return corridors

    def _find_critical_connections(
        self, patches: List[ResourcePatch], distance_matrix: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """Find critical connections using minimum spanning tree"""
        n = len(patches)
        if n < 2:
            return []

        # Prim's algorithm for MST
        visited = [False] * n
        min_heap = [(0, 0, -1)]  # (distance, current, parent)
        connections = []

        while min_heap:
            dist, current, parent = heapq.heappop(min_heap)

            if visited[current]:
                continue

            visited[current] = True

            if parent != -1:
                connections.append((parent, current, float(dist)))

            # Add neighbors
            for next_node in range(n):
                if not visited[next_node]:
                    heapq.heappush(
                        min_heap,
                        (distance_matrix[current, next_node], next_node, current),
                    )

        return connections

    def _select_corridor_type(
        self,
        patch1: ResourcePatch,
        patch2: ResourcePatch,
        distance: float,
        allowed_types: List[CorridorType],
    ) -> CorridorType:
        """Select appropriate corridor type based on patches and distance"""
        # Short distances - direct corridors
        if distance < 200 and CorridorType.LINEAR in allowed_types:
            return CorridorType.LINEAR

        # Medium distances - stepping stones
        elif distance < 1000 and CorridorType.STEPPING_STONE in allowed_types:
            return CorridorType.STEPPING_STONE

        # Along existing features
        elif CorridorType.HEDGEROW in allowed_types:
            # Check if patches are along field edge (simplified)
            if (
                patch1.habitat_type == HabitatType.CROPLAND
                or patch2.habitat_type == HabitatType.CROPLAND
            ):
                return CorridorType.HEDGEROW

        # Default to wildflower strip
        elif CorridorType.WILDFLOWER_STRIP in allowed_types:
            return CorridorType.WILDFLOWER_STRIP

        # Fallback
        return allowed_types[0] if allowed_types else CorridorType.LINEAR

    def _design_corridor_path(
        self, patch1: ResourcePatch, patch2: ResourcePatch, corridor_type: CorridorType
    ) -> List[Tuple[float, float]]:
        """Design the path for a corridor"""
        start = (patch1.x, patch1.y)
        end = (patch2.x, patch2.y)

        if corridor_type == CorridorType.LINEAR:
            # Direct path
            return [start, end]

        elif corridor_type == CorridorType.STEPPING_STONE:
            # Create intermediate points
            num_steps = max(
                2,
                int(np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) / 200),
            )  # One step per 200m

            path = [start]
            for i in range(1, num_steps):
                t = i / num_steps
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])

                # Add some variation
                x += np.random.normal(0, 20)
                y += np.random.normal(0, 20)

                path.append((x, y))
            path.append(end)

            return path

        else:
            # Default curved path
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2

            # Add curve
            perpendicular_x = -(end[1] - start[1]) / 10
            perpendicular_y = (end[0] - start[0]) / 10

            return [start, (mid_x + perpendicular_x, mid_y + perpendicular_y), end]

    def _calculate_corridor_width(
        self, length: float, corridor_type: CorridorType
    ) -> float:
        """Calculate appropriate corridor width"""
        base_widths = {
            CorridorType.LINEAR: 6.0,
            CorridorType.STEPPING_STONE: 10.0,  # Per stepping stone
            CorridorType.RIPARIAN: 10.0,
            CorridorType.HEDGEROW: 3.0,
            CorridorType.WILDFLOWER_STRIP: 6.0,
            CorridorType.MIXED: 8.0,
        }

        width = base_widths.get(corridor_type, 6.0)

        # Adjust for length - wider for longer corridors
        if length > 500:
            width *= 1.5
        elif length > 1000:
            width *= 2.0

        return width

    def _calculate_corridor_quality(
        self,
        patch1: ResourcePatch,
        patch2: ResourcePatch,
        distance: float,
        corridor_type: CorridorType,
    ) -> float:
        """Calculate corridor quality score"""
        # Base quality by type
        type_quality = {
            CorridorType.LINEAR: 0.7,
            CorridorType.STEPPING_STONE: 0.8,
            CorridorType.RIPARIAN: 0.9,
            CorridorType.HEDGEROW: 0.85,
            CorridorType.WILDFLOWER_STRIP: 0.9,
            CorridorType.MIXED: 0.8,
        }

        base_quality = type_quality.get(corridor_type, 0.7)

        # Adjust for distance (shorter is better)
        distance_factor = 1.0 - min(distance / 2000, 0.5)  # Max 50% reduction

        # Adjust for patch quality
        patch_quality = (
            getattr(patch1, "quality_score", 0.5)
            + getattr(patch2, "quality_score", 0.5)
        ) / 2

        return base_quality * distance_factor * (0.5 + 0.5 * patch_quality)

    def _estimate_corridor_cost(
        self, length: float, width: float, corridor_type: CorridorType
    ) -> float:
        """Estimate corridor establishment cost"""
        # Cost per meter
        cost_per_m = {
            CorridorType.LINEAR: 10.0,
            CorridorType.STEPPING_STONE: 15.0,
            CorridorType.RIPARIAN: 20.0,
            CorridorType.HEDGEROW: 25.0,
            CorridorType.WILDFLOWER_STRIP: 12.0,
            CorridorType.MIXED: 18.0,
        }

        base_cost = cost_per_m.get(corridor_type, 15.0)

        # Adjust for width
        width_factor = width / 6.0  # Normalized to 6m

        return length * base_cost * width_factor

    def _get_corridor_resources(self, corridor_type: CorridorType) -> List[str]:
        """Get resource types provided by corridor"""
        resources = {
            CorridorType.LINEAR: ["nectar", "pollen"],
            CorridorType.STEPPING_STONE: ["nectar", "pollen", "nesting"],
            CorridorType.RIPARIAN: ["water", "nectar", "pollen", "nesting"],
            CorridorType.HEDGEROW: ["nectar", "pollen", "nesting", "shelter"],
            CorridorType.WILDFLOWER_STRIP: ["nectar", "pollen"],
            CorridorType.MIXED: ["nectar", "pollen", "nesting", "shelter"],
        }

        return resources.get(corridor_type, ["nectar", "pollen"])

    def _calculate_network_connectivity(
        self, patches: List[ResourcePatch], corridors: List[ForagingCorridor]
    ) -> float:
        """Calculate overall network connectivity score"""
        if len(patches) < 2:
            return 1.0

        # Build adjacency matrix
        n = len(patches)
        connected = np.zeros((n, n), dtype=bool)

        # Map patch IDs to indices
        patch_id_to_index = {p.id: i for i, p in enumerate(patches)}

        # Mark connected patches
        for corridor in corridors:
            if len(corridor.connected_patches) >= 2:
                i = patch_id_to_index.get(corridor.connected_patches[0])
                j = patch_id_to_index.get(corridor.connected_patches[1])

                if i is not None and j is not None:
                    connected[i, j] = True
                    connected[j, i] = True

        # Calculate connectivity as ratio of connected pairs
        total_pairs = n * (n - 1) / 2
        connected_pairs = np.sum(connected) / 2

        return connected_pairs / total_pairs if total_pairs > 0 else 0

    def _add_redundant_corridors(
        self,
        patches: List[ResourcePatch],
        existing_corridors: List[ForagingCorridor],
        distance_matrix: np.ndarray,
        target_connectivity: float,
        budget_constraint: Optional[float],
        current_cost: float,
    ) -> List[ForagingCorridor]:
        """Add additional corridors to improve connectivity"""
        additional_corridors: List[ForagingCorridor] = []

        # Find unconnected patch pairs
        n = len(patches)
        connected_pairs = set()

        for corridor in existing_corridors:
            if len(corridor.connected_patches) >= 2:
                pair = tuple(sorted(corridor.connected_patches[:2]))
                connected_pairs.add(pair)

        # Sort potential connections by distance
        potential_connections = []
        for i in range(n):
            for j in range(i + 1, n):
                pair = tuple(sorted([patches[i].id, patches[j].id]))
                if pair not in connected_pairs:
                    potential_connections.append((distance_matrix[i, j], i, j))

        potential_connections.sort()

        # Add connections until target reached or budget exhausted
        for distance, i, j in potential_connections:
            if budget_constraint and current_cost >= budget_constraint:
                break

            # Check if this improves connectivity significantly
            test_corridor = ForagingCorridor(
                corridor_id=f"redundant_{i}_{j}",
                corridor_type=CorridorType.LINEAR,
                path=[(patches[i].x, patches[i].y), (patches[j].x, patches[j].y)],
                width_m=6.0,
                length_m=distance,
                connected_patches=[patches[i].id, patches[j].id],
                resource_types=["nectar", "pollen"],
                quality_score=0.7,
                establishment_cost=distance * 10,
                maintenance_cost_annual=distance * 1,
            )

            test_corridors = existing_corridors + additional_corridors + [test_corridor]
            new_connectivity = self._calculate_network_connectivity(
                patches, test_corridors
            )

            if new_connectivity >= target_connectivity:
                additional_corridors.append(test_corridor)
                current_cost += test_corridor.establishment_cost
                break
            elif new_connectivity > self._calculate_network_connectivity(
                patches, existing_corridors + additional_corridors
            ):
                additional_corridors.append(test_corridor)
                current_cost += test_corridor.establishment_cost

        return additional_corridors

    def calculate_resource_density(
        self,
        patches: List[ResourcePatch],
        modifications: Optional[List[Dict[str, Any]]] = None,
        resolution_m: float = 50,
    ) -> ResourceDensityGrid:
        """
        Calculate resource density across the landscape.

        Creates a grid-based representation of resource availability,
        accounting for both existing patches and proposed modifications.

        Args:
            patches: Existing resource patches
            modifications: Proposed landscape modifications
            resolution_m: Grid resolution in meters

        Returns:
            ResourceDensityGrid with calculated densities
        """

        # Calculate bounds
        bounds = self._calculate_area_bounds(
            [(p.x - 50, p.y - 50, 100, 100) for p in patches], patches
        )

        # Create density grid
        density_grid = ResourceDensityGrid(bounds, resolution_m)

        # Add existing patches
        for patch in patches:
            density_grid.add_resource_patch(patch)

        # Apply modifications if provided
        if modifications:
            for mod in modifications:
                self._apply_modification_to_grid(density_grid, mod)

        self.logger.info(
            f"Calculated resource density grid: "
            f"{density_grid.width}x{density_grid.height} cells at {resolution_m}m resolution"
        )

        return density_grid

    def _apply_modification_to_grid(
        self, grid: ResourceDensityGrid, modification: Dict[str, Any]
    ) -> None:
        """Apply a landscape modification to the resource grid"""
        mod_type = modification.get("type")
        geometry = modification.get("geometry", {})

        if mod_type == "wildflower_strip":
            # Add nectar and pollen along strip
            if geometry.get("type") == "LineString":
                coords = geometry.get("coordinates", [])
                for i in range(len(coords) - 1):
                    # Interpolate points along line
                    start = coords[i]
                    end = coords[i + 1]

                    distance = np.sqrt(
                        (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                    )

                    num_points = int(distance / grid.resolution)
                    for j in range(num_points):
                        t = j / num_points
                        x = start[0] + t * (end[0] - start[0])
                        y = start[1] + t * (end[1] - start[1])

                        # Add resources at point
                        grid_x = int((x - grid.bounds["min_x"]) / grid.resolution)
                        grid_y = int((y - grid.bounds["min_y"]) / grid.resolution)

                        if 0 <= grid_y < grid.height and 0 <= grid_x < grid.width:
                            grid.nectar_grid[grid_y, grid_x] += 5.0
                            grid.pollen_grid[grid_y, grid_x] += 3.0

        elif mod_type == "margin":
            """Implement field margin habitat modification"""
            # Field margins typically have different resource density and types
            margin_width = modification.get("width", 2.0)  # meters
            resource_multiplier = modification.get("resource_multiplier", 0.8)

            for coord in modification["coordinates"]:
                x, y = coord

                # Create margin zone around coordinates
                for dx in range(-int(margin_width), int(margin_width) + 1):
                    for dy in range(-int(margin_width), int(margin_width) + 1):
                        grid_x = int((x - grid.bounds["min_x"]) / grid.resolution) + dx
                        grid_y = int((y - grid.bounds["min_y"]) / grid.resolution) + dy

                        if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
                            # Margins have diverse but lower resource density
                            grid.nectar_grid[grid_y, grid_x] += (
                                3.0 * resource_multiplier
                            )
                            grid.pollen_grid[grid_y, grid_x] += (
                                2.0 * resource_multiplier
                            )
                            # Add habitat diversity index if available
                            if hasattr(grid, "habitat_diversity"):
                                grid.habitat_diversity[grid_y, grid_x] = 0.7

        elif mod_type == "pond":
            """Implement pond habitat modification for water resources"""
            pond_radius = modification.get("radius", 5.0)  # meters

            for coord in modification["coordinates"]:
                center_x, center_y = coord

                # Convert to grid coordinates
                center_grid_x = int((center_x - grid.bounds["min_x"]) / grid.resolution)
                center_grid_y = int((center_y - grid.bounds["min_y"]) / grid.resolution)

                # Create circular pond area
                for dx in range(-int(pond_radius), int(pond_radius) + 1):
                    for dy in range(-int(pond_radius), int(pond_radius) + 1):
                        distance = (dx**2 + dy**2) ** 0.5

                        if distance <= pond_radius:
                            grid_x, grid_y = center_grid_x + dx, center_grid_y + dy

                            if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
                                # Ponds provide water resources, reduce other resources
                                if hasattr(grid, "water_grid"):
                                    grid.water_grid[grid_y, grid_x] = 1.0

                                # Reduce nectar/pollen in water areas
                                grid.nectar_grid[grid_y, grid_x] *= 0.1
                                grid.pollen_grid[grid_y, grid_x] *= 0.1

    def identify_resource_hotspots(
        self, density_grid: ResourceDensityGrid, threshold_percentile: float = 80
    ) -> List[Dict[str, Any]]:
        """Identify resource hotspots in the landscape"""
        hotspots = []

        # Calculate thresholds
        nectar_threshold = np.percentile(
            density_grid.nectar_grid[density_grid.nectar_grid > 0], threshold_percentile
        )
        pollen_threshold = np.percentile(
            density_grid.pollen_grid[density_grid.pollen_grid > 0], threshold_percentile
        )

        # Find hotspot cells
        nectar_hotspots = np.argwhere(density_grid.nectar_grid > nectar_threshold)
        pollen_hotspots = np.argwhere(density_grid.pollen_grid > pollen_threshold)

        # Combine and cluster hotspots
        all_hotspot_cells = np.vstack([nectar_hotspots, pollen_hotspots])

        # Simple clustering (grid-based)
        hotspot_clusters = self._cluster_hotspot_cells(all_hotspot_cells, density_grid)

        for cluster in hotspot_clusters:
            center_y, center_x = np.mean(cluster, axis=0)

            # Convert to world coordinates
            world_x = density_grid.bounds["min_x"] + center_x * density_grid.resolution
            world_y = density_grid.bounds["min_y"] + center_y * density_grid.resolution

            # Calculate cluster properties
            nectar_values = [
                density_grid.nectar_grid[int(y), int(x)]
                for y, x in cluster
                if 0 <= int(y) < density_grid.height
                and 0 <= int(x) < density_grid.width
            ]
            pollen_values = [
                density_grid.pollen_grid[int(y), int(x)]
                for y, x in cluster
                if 0 <= int(y) < density_grid.height
                and 0 <= int(x) < density_grid.width
            ]

            hotspot = {
                "location": (world_x, world_y),
                "area_m2": len(cluster) * density_grid.resolution**2,
                "avg_nectar": np.mean(nectar_values) if nectar_values else 0,
                "avg_pollen": np.mean(pollen_values) if pollen_values else 0,
                "max_nectar": np.max(nectar_values) if nectar_values else 0,
                "max_pollen": np.max(pollen_values) if pollen_values else 0,
                "importance_score": len(cluster)
                * (
                    np.mean(nectar_values + pollen_values)
                    if nectar_values or pollen_values
                    else 0
                ),
            }

            hotspots.append(hotspot)

        # Sort by importance
        hotspots.sort(key=lambda h: h["importance_score"], reverse=True)

        return hotspots

    def _cluster_hotspot_cells(
        self, cells: np.ndarray, grid: ResourceDensityGrid
    ) -> List[List[Tuple[int, int]]]:
        """Cluster nearby hotspot cells"""
        if len(cells) == 0:
            return []

        # Remove duplicates
        unique_cells = np.unique(cells, axis=0)

        # Simple distance-based clustering
        clusters = []
        used = set()

        for i, cell in enumerate(unique_cells):
            if i in used:
                continue

            cluster = [tuple(cell)]
            used.add(i)

            # Find nearby cells
            for j, other_cell in enumerate(unique_cells):
                if j in used:
                    continue

                # Manhattan distance in grid cells
                distance = abs(cell[0] - other_cell[0]) + abs(cell[1] - other_cell[1])

                if distance <= 3:  # Within 3 cells
                    cluster.append(tuple(other_cell))
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def evaluate_habitat_network(
        self,
        nesting_sites: List[NestingSite],
        foraging_corridors: List[ForagingCorridor],
        resource_patches: List[ResourcePatch],
        target_species: List[BeeSpeciesGroup],
    ) -> Dict[str, Any]:
        """
        Evaluate the overall habitat network quality.

        Provides metrics on network connectivity, resource availability,
        and suitability for target species.
        """

        evaluation: Dict[str, Any] = {
            "network_metrics": {},
            "species_suitability": {},
            "resource_availability": {},
            "recommendations": [],
        }

        # Initialize network metrics dict
        network_metrics: Dict[str, Any] = {}
        species_suitability: Dict[str, Any] = {}
        resource_availability: Dict[str, Any] = {}
        recommendations: List[Dict[str, str]] = []

        # Network connectivity metrics
        connectivity_score = self._calculate_network_connectivity(
            resource_patches, foraging_corridors
        )

        network_metrics["connectivity"] = connectivity_score
        network_metrics["nesting_sites"] = len(nesting_sites)
        network_metrics["corridors"] = len(foraging_corridors)
        network_metrics["total_corridor_length_km"] = (
            sum(c.length_m for c in foraging_corridors) / 1000
        )

        # Species-specific evaluation
        for species in target_species:
            species_eval = self._evaluate_species_requirements(
                species, nesting_sites, resource_patches, foraging_corridors
            )
            species_suitability[species.value] = species_eval

        # Resource availability
        density_grid = self.calculate_resource_density(resource_patches)

        total_nectar = np.sum(density_grid.nectar_grid)
        total_pollen = np.sum(density_grid.pollen_grid)

        resource_availability["total_nectar_kg"] = total_nectar / 1000
        resource_availability["total_pollen_kg"] = total_pollen / 1000
        resource_availability["nectar_density_avg"] = (
            np.mean(density_grid.nectar_grid[density_grid.nectar_grid > 0])
            if np.any(density_grid.nectar_grid > 0)
            else 0
        )

        # Generate recommendations
        if connectivity_score < 0.6:
            recommendations.append(
                {
                    "type": "connectivity",
                    "priority": "high",
                    "description": "Add more corridors to improve habitat connectivity",
                }
            )

        # Check species requirements
        for species_key, suitability in species_suitability.items():
            if suitability["overall_score"] < 0.6:
                recommendations.append(
                    {
                        "type": "species_habitat",
                        "priority": "medium",
                        "description": f"Improve habitat for {species_key}: {suitability['limiting_factors']}",
                    }
                )

        # Assemble final evaluation
        evaluation["network_metrics"] = network_metrics
        evaluation["species_suitability"] = species_suitability
        evaluation["resource_availability"] = resource_availability
        evaluation["recommendations"] = recommendations

        return evaluation

    def _evaluate_species_requirements(
        self,
        species: BeeSpeciesGroup,
        nesting_sites: List[NestingSite],
        patches: List[ResourcePatch],
        corridors: List[ForagingCorridor],
    ) -> Dict[str, Any]:
        """Evaluate habitat suitability for a specific species"""

        requirements = self.foraging_requirements.get(species)
        nesting_reqs = self.nesting_requirements.get(species)

        if not requirements or not nesting_reqs:
            return {
                "overall_score": 0,
                "nesting_score": 0,
                "nectar_score": 0,
                "pollen_score": 0,
                "suitable_nests": 0,
                "limiting_factors": ["Unknown species"],
            }

        # Check nesting sites
        suitable_nests = [
            site for site in nesting_sites if species in site.species_groups
        ]

        nesting_score = min(len(suitable_nests) / 3, 1.0)  # Target 3+ sites

        # Check foraging resources within range
        total_nectar_available = 0.0
        total_pollen_available = 0.0

        for nest in suitable_nests:
            for patch in patches:
                distance = np.sqrt(
                    (patch.x - nest.location[0]) ** 2
                    + (patch.y - nest.location[1]) ** 2
                )

                if distance <= requirements.foraging_range_m:
                    # Simple availability estimate
                    total_nectar_available += patch.base_nectar_production
                    total_pollen_available += patch.base_pollen_production

        # Calculate resource adequacy
        nectar_score = min(
            total_nectar_available / (requirements.nectar_demand_mg_per_day * 100), 1.0
        )  # For 100 individuals
        pollen_score = min(
            total_pollen_available / (requirements.pollen_demand_mg_per_day * 100), 1.0
        )

        # Overall score
        overall_score = (nesting_score + nectar_score + pollen_score) / 3

        # Identify limiting factors
        limiting_factors = []
        if nesting_score < 0.6:
            limiting_factors.append("Insufficient nesting sites")
        if nectar_score < 0.6:
            limiting_factors.append("Low nectar availability")
        if pollen_score < 0.6:
            limiting_factors.append("Low pollen availability")

        return {
            "overall_score": overall_score,
            "nesting_score": nesting_score,
            "nectar_score": nectar_score,
            "pollen_score": pollen_score,
            "suitable_nests": len(suitable_nests),
            "limiting_factors": limiting_factors,
        }
