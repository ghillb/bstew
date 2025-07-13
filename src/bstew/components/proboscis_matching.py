"""
Proboscis-Corolla Matching System for BSTEW
==========================================

Implements realistic flower accessibility based on bee proboscis length
and flower corolla depth/width constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import logging
import math
from dataclasses import dataclass

from ..spatial.patches import FlowerSpecies


@dataclass
class FlowerGeometry3D:
    """3D geometric properties of flowers for realistic accessibility modeling"""
    corolla_depth_mm: float  # Depth from opening to nectary
    corolla_width_mm: float  # Width of corolla opening
    corolla_angle_degrees: float = 0.0  # Angle from vertical (0 = upright, 90 = horizontal)
    nectary_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D position of nectary
    corolla_curvature: float = 0.0  # Curvature of corolla tube (0 = straight, 1 = highly curved)
    constriction_points: List[Tuple[float, float]] = None  # (position, width) tuples for constrictions
    
    def __post_init__(self):
        if self.constriction_points is None:
            self.constriction_points = []
    
    def calculate_effective_depth(self, approach_angle_degrees: float) -> float:
        """Calculate effective depth considering approach angle"""
        angle_factor = math.cos(math.radians(abs(approach_angle_degrees - self.corolla_angle_degrees)))
        return self.corolla_depth_mm / max(0.1, angle_factor)
    
    def has_path_clearance(self, proboscis_width: float, insertion_depth: float) -> bool:
        """Check if proboscis has clearance through constriction points"""
        for position, width in self.constriction_points:
            if position <= insertion_depth and proboscis_width * 1.1 > width:  # 10% safety margin
                return False
        return True


@dataclass
class Geometric3DAccessibility:
    """3D geometric accessibility analysis result"""
    optimal_approach_angle: float  # Best approach angle in degrees
    insertion_depth_possible: float  # Maximum safe insertion depth
    path_clearance_score: float  # 0-1 score for path clearance
    geometric_efficiency: float  # Overall geometric efficiency 0-1
    bending_requirement: float  # Required bending angle
    collision_risk: float  # Risk of collision with flower parts 0-1


class AccessibilityLevel(Enum):
    """Flower accessibility levels"""

    INACCESSIBLE = "inaccessible"  # Cannot access at all
    POOR = "poor"  # Can access with difficulty
    MODERATE = "moderate"  # Can access reasonably well
    GOOD = "good"  # Can access easily
    EXCELLENT = "excellent"  # Optimal match


class ProboscisCharacteristics(BaseModel):
    """Bee proboscis characteristics"""

    model_config = {"validate_assignment": True}

    length_mm: float = Field(ge=0.0, description="Proboscis length in millimeters")
    width_mm: float = Field(
        default=0.2, ge=0.0, description="Proboscis width in millimeters"
    )
    flexibility: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Flexibility factor (0-1 scale)"
    )
    extension_efficiency: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Efficiency when fully extended"
    )
    # 3D geometric properties
    cross_sectional_area_mm2: float = Field(
        default=0.031, ge=0.0, description="Cross-sectional area in mm²"
    )
    insertion_angle_degrees: float = Field(
        default=0.0, ge=-45.0, le=45.0, description="Optimal insertion angle in degrees"
    )
    bending_radius_mm: float = Field(
        default=2.0, ge=0.0, description="Minimum bending radius in mm"
    )

    def get_effective_length(self) -> float:
        """Get effective proboscis length considering flexibility"""
        return self.length_mm * self.flexibility
    
    def get_3d_reach_volume(self) -> float:
        """Calculate 3D reach volume considering bending and flexibility"""
        # Simplified ellipsoid volume calculation
        effective_length = self.get_effective_length()
        lateral_reach = min(effective_length * 0.3, self.bending_radius_mm)
        return (4/3) * math.pi * effective_length * lateral_reach * lateral_reach
    
    def can_bend_to_angle(self, required_angle_degrees: float) -> float:
        """Calculate bending capability for required angle (0-1 scale)"""
        max_bend = 45.0 * self.flexibility  # Max bend angle based on flexibility
        if abs(required_angle_degrees) <= max_bend:
            return 1.0 - (abs(required_angle_degrees) / max_bend) * 0.3
        else:
            return 0.0


class AccessibilityResult(BaseModel):
    """Result of proboscis-corolla matching"""

    model_config = {"validate_assignment": True}

    accessibility_level: AccessibilityLevel = Field(
        description="Accessibility level category"
    )
    accessibility_score: float = Field(
        ge=0.0, le=1.0, description="Accessibility score (0-1 scale)"
    )
    nectar_extraction_efficiency: float = Field(
        ge=0.0, le=1.0, description="Nectar extraction efficiency (0-1 scale)"
    )
    pollen_extraction_efficiency: float = Field(
        ge=0.0, le=1.0, description="Pollen extraction efficiency (0-1 scale)"
    )
    energy_cost_multiplier: float = Field(
        ge=0.0, description="Energy cost multiplier for accessing this flower"
    )
    handling_time_multiplier: float = Field(
        ge=0.0, description="Time multiplier for handling"
    )
    # 3D geometric analysis
    geometric_3d: Optional[Geometric3DAccessibility] = Field(
        default=None, description="3D geometric accessibility analysis"
    )
    spatial_efficiency: float = Field(
        default=1.0, ge=0.0, le=1.0, description="3D spatial efficiency factor"
    )

    def is_accessible(self) -> bool:
        """Check if flower is accessible at all"""
        return self.accessibility_level != AccessibilityLevel.INACCESSIBLE


class ProboscisCorollaSystem:
    """
    Manages proboscis-corolla matching for realistic flower accessibility.

    Implements:
    - Proboscis length vs corolla depth matching
    - Corolla width constraints
    - Species-specific proboscis characteristics
    - Accessibility scoring and efficiency calculations
    - Energy cost adjustments based on accessibility
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Species-specific proboscis characteristics
        self.species_proboscis = self.initialize_species_proboscis()
        
        # 3D flower geometry database
        self.flower_geometry_3d = self.initialize_3d_flower_geometry()

        # Accessibility thresholds
        self.accessibility_thresholds = {
            AccessibilityLevel.INACCESSIBLE: 0.0,
            AccessibilityLevel.POOR: 0.2,
            AccessibilityLevel.MODERATE: 0.4,
            AccessibilityLevel.GOOD: 0.7,
            AccessibilityLevel.EXCELLENT: 0.9,
        }

    def initialize_species_proboscis(self) -> Dict[str, ProboscisCharacteristics]:
        """Initialize species-specific proboscis characteristics"""

        # Based on UK bumblebee species data
        species_data = {
            # Short-tongued species
            "Bombus_terrestris": ProboscisCharacteristics(
                length_mm=7.2, width_mm=0.25, flexibility=0.85, extension_efficiency=0.9
            ),
            "Bombus_lucorum": ProboscisCharacteristics(
                length_mm=7.8, width_mm=0.23, flexibility=0.82, extension_efficiency=0.88,
                cross_sectional_area_mm2=0.042, insertion_angle_degrees=0.0, bending_radius_mm=2.3
            ),
            "Bombus_lapidarius": ProboscisCharacteristics(
                length_mm=8.1, width_mm=0.24, flexibility=0.83, extension_efficiency=0.87,
                cross_sectional_area_mm2=0.045, insertion_angle_degrees=0.0, bending_radius_mm=2.4
            ),
            # Medium-tongued species
            "Bombus_pratorum": ProboscisCharacteristics(
                length_mm=9.4, width_mm=0.22, flexibility=0.88, extension_efficiency=0.91,
                cross_sectional_area_mm2=0.038, insertion_angle_degrees=0.0, bending_radius_mm=2.8
            ),
            "Bombus_pascuorum": ProboscisCharacteristics(
                length_mm=11.2, width_mm=0.21, flexibility=0.92, extension_efficiency=0.93,
                cross_sectional_area_mm2=0.035, insertion_angle_degrees=0.0, bending_radius_mm=3.2
            ),
            # Long-tongued species
            "Bombus_hortorum": ProboscisCharacteristics(
                length_mm=15.3, width_mm=0.19, flexibility=0.95, extension_efficiency=0.96,
                cross_sectional_area_mm2=0.028, insertion_angle_degrees=0.0, bending_radius_mm=4.5
            ),
            "Bombus_ruderatus": ProboscisCharacteristics(
                length_mm=17.1, width_mm=0.18, flexibility=0.97, extension_efficiency=0.98,
                cross_sectional_area_mm2=0.025, insertion_angle_degrees=0.0, bending_radius_mm=5.0
            ),
            # Honeybee for comparison
            "Apis_mellifera": ProboscisCharacteristics(
                length_mm=6.4, width_mm=0.15, flexibility=0.8, extension_efficiency=0.85,
                cross_sectional_area_mm2=0.018, insertion_angle_degrees=0.0, bending_radius_mm=1.8
            ),
            # Lowercase variants for compatibility
            "apis_mellifera": ProboscisCharacteristics(
                length_mm=6.4, width_mm=0.15, flexibility=0.8, extension_efficiency=0.85,
                cross_sectional_area_mm2=0.018, insertion_angle_degrees=0.0, bending_radius_mm=1.8
            ),
            "bombus_terrestris": ProboscisCharacteristics(
                length_mm=7.2, width_mm=0.25, flexibility=0.85, extension_efficiency=0.9,
                cross_sectional_area_mm2=0.049, insertion_angle_degrees=0.0, bending_radius_mm=2.5
            ),
        }

        return species_data
    
    def initialize_3d_flower_geometry(self) -> Dict[str, FlowerGeometry3D]:
        """Initialize 3D geometric data for flower species"""
        
        geometry_data = {
            "White Clover": FlowerGeometry3D(
                corolla_depth_mm=4.5, corolla_width_mm=1.8, corolla_angle_degrees=15.0,
                nectary_position=(0.0, 0.0, -4.2), corolla_curvature=0.2,
                constriction_points=[(2.0, 1.2)]
            ),
            "Red Clover": FlowerGeometry3D(
                corolla_depth_mm=9.2, corolla_width_mm=2.1, corolla_angle_degrees=10.0,
                nectary_position=(0.0, 0.0, -8.8), corolla_curvature=0.1,
                constriction_points=[(3.0, 1.5), (6.0, 1.2)]
            ),
            "Dandelion": FlowerGeometry3D(
                corolla_depth_mm=3.8, corolla_width_mm=2.5, corolla_angle_degrees=0.0,
                nectary_position=(0.0, 0.0, -3.5), corolla_curvature=0.0,
                constriction_points=[]
            ),
            "Bramble": FlowerGeometry3D(
                corolla_depth_mm=5.1, corolla_width_mm=3.2, corolla_angle_degrees=0.0,
                nectary_position=(0.0, 0.0, -4.8), corolla_curvature=0.0,
                constriction_points=[]
            ),
            "Foxglove": FlowerGeometry3D(
                corolla_depth_mm=25.0, corolla_width_mm=12.0, corolla_angle_degrees=45.0,
                nectary_position=(0.0, 8.0, -22.0), corolla_curvature=0.6,
                constriction_points=[(8.0, 6.0), (15.0, 4.0), (20.0, 3.0)]
            ),
            "Honeysuckle": FlowerGeometry3D(
                corolla_depth_mm=18.7, corolla_width_mm=3.5, corolla_angle_degrees=30.0,
                nectary_position=(0.0, 5.0, -17.5), corolla_curvature=0.8,
                constriction_points=[(5.0, 2.5), (12.0, 2.0)]
            ),
            "Salvia": FlowerGeometry3D(
                corolla_depth_mm=15.2, corolla_width_mm=8.0, corolla_angle_degrees=60.0,
                nectary_position=(0.0, 12.0, -8.0), corolla_curvature=0.4,
                constriction_points=[(4.0, 5.0), (10.0, 3.5)]
            ),
        }
        
        return geometry_data

    def analyze_3d_geometry(
        self, proboscis: ProboscisCharacteristics, flower_name: str, flower: FlowerSpecies
    ) -> Geometric3DAccessibility:
        """Perform detailed 3D geometric analysis of proboscis-flower interaction"""
        
        # Get 3D geometry data for this flower
        if flower_name not in self.flower_geometry_3d:
            # Use basic geometry from FlowerSpecies
            geometry = FlowerGeometry3D(
                corolla_depth_mm=flower.corolla_depth_mm,
                corolla_width_mm=flower.corolla_width_mm,
                corolla_angle_degrees=0.0,
                nectary_position=(0.0, 0.0, -flower.corolla_depth_mm * 0.9),
                corolla_curvature=0.0
            )
        else:
            geometry = self.flower_geometry_3d[flower_name]
        
        # Calculate optimal approach angle
        optimal_angle = self.calculate_optimal_approach_angle(proboscis, geometry)
        
        # Calculate maximum safe insertion depth
        insertion_depth = self.calculate_safe_insertion_depth(proboscis, geometry, optimal_angle)
        
        # Analyze path clearance through constrictions
        path_clearance = self.calculate_path_clearance(proboscis, geometry, insertion_depth)
        
        # Calculate overall geometric efficiency
        geometric_efficiency = self.calculate_geometric_efficiency(
            proboscis, geometry, optimal_angle, insertion_depth, path_clearance
        )
        
        # Calculate required bending
        bending_requirement = abs(optimal_angle - proboscis.insertion_angle_degrees)
        
        # Assess collision risk
        collision_risk = self.calculate_collision_risk(proboscis, geometry, optimal_angle)
        
        return Geometric3DAccessibility(
            optimal_approach_angle=optimal_angle,
            insertion_depth_possible=insertion_depth,
            path_clearance_score=path_clearance,
            geometric_efficiency=geometric_efficiency,
            bending_requirement=bending_requirement,
            collision_risk=collision_risk
        )
    
    def calculate_optimal_approach_angle(self, proboscis: ProboscisCharacteristics, geometry: FlowerGeometry3D) -> float:
        """Calculate optimal approach angle for proboscis insertion"""
        
        # Start with flower's natural angle
        base_angle = geometry.corolla_angle_degrees
        
        # Adjust for proboscis characteristics
        if proboscis.flexibility > 0.9:
            # Highly flexible can adapt to flower angle
            return base_angle
        elif proboscis.flexibility > 0.7:
            # Moderately flexible - partial adaptation
            return base_angle * 0.7
        else:
            # Rigid proboscis prefers straight approach
            return base_angle * 0.3
    
    def calculate_safe_insertion_depth(self, proboscis: ProboscisCharacteristics, 
                                     geometry: FlowerGeometry3D, approach_angle: float) -> float:
        """Calculate maximum safe insertion depth"""
        
        # Calculate effective depth considering approach angle
        effective_depth = geometry.calculate_effective_depth(approach_angle)
        
        # Limit by proboscis reach
        max_reach = proboscis.get_effective_length()
        
        # Account for curvature - curved corollas reduce effective reach
        curvature_factor = 1.0 - (geometry.corolla_curvature * 0.3)
        effective_reach = max_reach * curvature_factor
        
        # Account for bending limitations
        required_bend = abs(approach_angle - proboscis.insertion_angle_degrees)
        bend_capability = proboscis.can_bend_to_angle(required_bend)
        if bend_capability < 0.5:
            effective_reach *= 0.7  # Reduced reach due to difficult bending
        
        return min(effective_depth, effective_reach)
    
    def calculate_path_clearance(self, proboscis: ProboscisCharacteristics, 
                               geometry: FlowerGeometry3D, insertion_depth: float) -> float:
        """Calculate path clearance score through flower constrictions"""
        
        if not geometry.constriction_points:
            return 1.0  # No constrictions
        
        clearance_scores = []
        proboscis_width = proboscis.width_mm
        
        for position, width in geometry.constriction_points:
            if position <= insertion_depth:
                if proboscis_width <= width * 0.8:
                    clearance_scores.append(1.0)  # Easy clearance
                elif proboscis_width <= width:
                    clearance_scores.append(0.7)  # Tight clearance
                elif proboscis_width <= width * 1.2:
                    clearance_scores.append(0.3)  # Very tight
                else:
                    clearance_scores.append(0.0)  # Blocked
        
        return min(clearance_scores) if clearance_scores else 1.0
    
    def calculate_geometric_efficiency(self, proboscis: ProboscisCharacteristics, 
                                     geometry: FlowerGeometry3D, approach_angle: float,
                                     insertion_depth: float, path_clearance: float) -> float:
        """Calculate overall 3D geometric efficiency"""
        
        # Base efficiency from depth ratio
        depth_ratio = insertion_depth / geometry.corolla_depth_mm
        depth_efficiency = min(1.0, depth_ratio)
        
        # Angle efficiency
        angle_diff = abs(approach_angle - geometry.corolla_angle_degrees)
        angle_efficiency = max(0.0, 1.0 - (angle_diff / 90.0))
        
        # Flexibility efficiency
        required_bend = abs(approach_angle - proboscis.insertion_angle_degrees)
        flex_efficiency = proboscis.can_bend_to_angle(required_bend)
        
        # Curvature handling
        curvature_efficiency = 1.0 - (geometry.corolla_curvature * (1.0 - proboscis.flexibility))
        
        # Combine all factors
        overall_efficiency = (
            depth_efficiency * 0.3 +
            angle_efficiency * 0.2 +
            flex_efficiency * 0.2 +
            path_clearance * 0.2 +
            curvature_efficiency * 0.1
        )
        
        return max(0.0, min(1.0, overall_efficiency))
    
    def calculate_collision_risk(self, proboscis: ProboscisCharacteristics, 
                               geometry: FlowerGeometry3D, approach_angle: float) -> float:
        """Calculate risk of collision with flower parts"""
        
        # Base risk from approach angle mismatch
        angle_mismatch = abs(approach_angle - geometry.corolla_angle_degrees)
        base_risk = angle_mismatch / 90.0
        
        # Width-based collision risk
        width_ratio = proboscis.width_mm / geometry.corolla_width_mm
        width_risk = max(0.0, (width_ratio - 0.5) * 2.0)
        
        # Curvature increases collision risk for rigid proboscis
        curvature_risk = geometry.corolla_curvature * (1.0 - proboscis.flexibility)
        
        total_risk = min(1.0, base_risk + width_risk + curvature_risk)
        return total_risk

    def calculate_accessibility(
        self, proboscis: ProboscisCharacteristics, flower: FlowerSpecies
    ) -> AccessibilityResult:
        """Calculate flower accessibility for given proboscis-corolla combination"""

        # Basic length accessibility
        effective_length = proboscis.get_effective_length()
        length_ratio = effective_length / flower.corolla_depth_mm

        # Width constraint check
        width_constraint = self.check_width_constraint(proboscis, flower)

        # Calculate base accessibility score
        base_accessibility = self.calculate_base_accessibility(
            length_ratio, width_constraint
        )

        # Apply flower-specific modifiers
        modified_accessibility = base_accessibility * flower.nectar_accessibility

        # Perform 3D geometric analysis
        geometric_3d = self.analyze_3d_geometry(proboscis, flower.name, flower)
        
        # Apply 3D geometric factors to accessibility
        spatial_efficiency = geometric_3d.geometric_efficiency
        modified_accessibility *= spatial_efficiency
        
        # Recalculate accessibility level with 3D factors
        accessibility_level = self.determine_accessibility_level(modified_accessibility)

        # Calculate extraction efficiencies with 3D factors
        nectar_efficiency = self.calculate_nectar_efficiency(
            length_ratio, width_constraint
        ) * spatial_efficiency
        pollen_efficiency = self.calculate_pollen_efficiency(proboscis, flower) * spatial_efficiency

        # Calculate energy cost and handling time with 3D factors
        base_energy_cost = self.calculate_energy_cost(
            length_ratio, accessibility_level
        )
        collision_penalty = 1.0 + (geometric_3d.collision_risk * 0.5)
        energy_cost_multiplier = base_energy_cost * collision_penalty
        
        base_handling_time = self.calculate_handling_time(
            length_ratio, accessibility_level
        )
        geometric_time_penalty = 1.0 + ((1.0 - spatial_efficiency) * 0.8)
        handling_time_multiplier = base_handling_time * geometric_time_penalty

        return AccessibilityResult(
            accessibility_level=accessibility_level,
            accessibility_score=modified_accessibility,
            nectar_extraction_efficiency=nectar_efficiency,
            pollen_extraction_efficiency=pollen_efficiency,
            energy_cost_multiplier=energy_cost_multiplier,
            handling_time_multiplier=handling_time_multiplier,
            geometric_3d=geometric_3d,
            spatial_efficiency=spatial_efficiency,
        )

    def check_width_constraint(
        self, proboscis: ProboscisCharacteristics, flower: FlowerSpecies
    ) -> float:
        """Check if proboscis can fit through corolla opening"""

        # Proboscis can compress slightly
        effective_width = proboscis.width_mm * 0.8

        if effective_width <= flower.corolla_width_mm:
            return 1.0  # No width constraint
        elif effective_width <= flower.corolla_width_mm * 1.2:
            return 0.7  # Slight constraint
        elif effective_width <= flower.corolla_width_mm * 1.5:
            return 0.4  # Significant constraint
        else:
            return 0.0  # Cannot fit

    def calculate_base_accessibility(
        self, length_ratio: float, width_constraint: float
    ) -> float:
        """Calculate base accessibility score"""

        # Optimal length ratio is around 1.2 (slightly longer than corolla)
        if length_ratio >= 1.2:
            length_score = 1.0
        elif length_ratio >= 1.0:
            length_score = 0.5 + 0.5 * (length_ratio - 1.0) / 0.2
        elif length_ratio >= 0.8:
            length_score = 0.2 + 0.3 * (length_ratio - 0.8) / 0.2
        elif length_ratio >= 0.6:
            length_score = 0.1 + 0.1 * (length_ratio - 0.6) / 0.2
        else:
            length_score = 0.0

        # Combine with width constraint
        return length_score * width_constraint

    def determine_accessibility_level(
        self, accessibility_score: float
    ) -> AccessibilityLevel:
        """Determine accessibility level from score"""

        if (
            accessibility_score
            >= self.accessibility_thresholds[AccessibilityLevel.EXCELLENT]
        ):
            return AccessibilityLevel.EXCELLENT
        elif (
            accessibility_score
            >= self.accessibility_thresholds[AccessibilityLevel.GOOD]
        ):
            return AccessibilityLevel.GOOD
        elif (
            accessibility_score
            >= self.accessibility_thresholds[AccessibilityLevel.MODERATE]
        ):
            return AccessibilityLevel.MODERATE
        elif (
            accessibility_score
            >= self.accessibility_thresholds[AccessibilityLevel.POOR]
        ):
            return AccessibilityLevel.POOR
        else:
            return AccessibilityLevel.INACCESSIBLE

    def calculate_nectar_efficiency(
        self, length_ratio: float, width_constraint: float
    ) -> float:
        """Calculate nectar extraction efficiency"""

        # Optimal nectar extraction at length ratio 1.0-1.3
        if 1.0 <= length_ratio <= 1.3:
            length_efficiency = 1.0
        elif length_ratio > 1.3:
            # Longer proboscis is less efficient due to excess length
            length_efficiency = max(0.7, 1.0 - (length_ratio - 1.3) * 0.3)
        else:
            # Shorter proboscis can't reach all nectar
            length_efficiency = length_ratio * 0.8

        return length_efficiency * width_constraint

    def calculate_pollen_efficiency(
        self, proboscis: ProboscisCharacteristics, flower: FlowerSpecies
    ) -> float:
        """Calculate pollen extraction efficiency"""

        # Pollen is generally easier to access than nectar
        base_efficiency = 0.8

        # Proboscis flexibility affects pollen collection
        flexibility_bonus = proboscis.flexibility * 0.2

        # Flower structure affects pollen accessibility
        structure_factor = min(1.0, flower.corolla_width_mm / 2.0)

        return min(1.0, base_efficiency + flexibility_bonus) * structure_factor

    def calculate_energy_cost(
        self, length_ratio: float, accessibility_level: AccessibilityLevel
    ) -> float:
        """Calculate energy cost multiplier for accessing flower"""

        base_cost = 1.0

        # Length ratio effects
        if length_ratio < 0.8:
            # Difficult to reach - high energy cost
            base_cost += (0.8 - length_ratio) * 2.0
        elif length_ratio > 1.5:
            # Inefficient use of long proboscis
            base_cost += (length_ratio - 1.5) * 0.5

        # Accessibility level effects
        accessibility_multipliers = {
            AccessibilityLevel.INACCESSIBLE: 10.0,  # Extremely high cost
            AccessibilityLevel.POOR: 2.0,
            AccessibilityLevel.MODERATE: 1.3,
            AccessibilityLevel.GOOD: 1.0,
            AccessibilityLevel.EXCELLENT: 0.8,
        }

        return base_cost * accessibility_multipliers[accessibility_level]

    def calculate_handling_time(
        self, length_ratio: float, accessibility_level: AccessibilityLevel
    ) -> float:
        """Calculate handling time multiplier"""

        base_time = 1.0

        # Poor accessibility increases handling time
        if length_ratio < 1.0:
            base_time += (1.0 - length_ratio) * 1.5

        # Accessibility level effects
        time_multipliers = {
            AccessibilityLevel.INACCESSIBLE: 5.0,
            AccessibilityLevel.POOR: 2.5,
            AccessibilityLevel.MODERATE: 1.5,
            AccessibilityLevel.GOOD: 1.0,
            AccessibilityLevel.EXCELLENT: 0.8,
        }

        return base_time * time_multipliers[accessibility_level]

    def get_species_proboscis(self, species_name: str) -> ProboscisCharacteristics:
        """Get proboscis characteristics for species"""

        if species_name in self.species_proboscis:
            return self.species_proboscis[species_name]
        else:
            # Default proboscis for unknown species
            self.logger.warning(
                f"Unknown species {species_name}, using default proboscis"
            )
            return ProboscisCharacteristics(
                length_mm=8.0, width_mm=0.2, flexibility=0.8, extension_efficiency=0.85
            )

    def evaluate_flower_patch(
        self, species_name: str, flower_species_list: List[FlowerSpecies]
    ) -> Dict[str, AccessibilityResult]:
        """Evaluate accessibility of all flowers in a patch for given species"""

        proboscis = self.get_species_proboscis(species_name)
        accessibility_results = {}

        for flower in flower_species_list:
            result = self.calculate_accessibility(proboscis, flower)
            accessibility_results[flower.name] = result

        return accessibility_results

    def filter_accessible_flowers(
        self, species_name: str, flower_species_list: List[FlowerSpecies]
    ) -> List[FlowerSpecies]:
        """Filter flowers to only accessible ones for given species"""

        accessibility_results = self.evaluate_flower_patch(
            species_name, flower_species_list
        )
        accessible_flowers = []

        for flower in flower_species_list:
            result = accessibility_results[flower.name]
            if result.is_accessible():
                accessible_flowers.append(flower)

        return accessible_flowers

    def calculate_patch_accessibility_score(
        self, species_name: str, flower_species_list: List[FlowerSpecies]
    ) -> float:
        """Calculate overall accessibility score for a patch"""

        if not flower_species_list:
            return 0.0

        accessibility_results = self.evaluate_flower_patch(
            species_name, flower_species_list
        )

        # Weight by flower density and production
        total_weighted_score = 0.0
        total_weight = 0.0

        for flower in flower_species_list:
            result = accessibility_results[flower.name]

            # Weight by flower density and resource production
            weight = flower.flower_density * (
                flower.nectar_production + flower.pollen_production
            )

            total_weighted_score += result.accessibility_score * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def get_foraging_efficiency_modifier(
        self, species_name: str, flower: FlowerSpecies
    ) -> Dict[str, float]:
        """Get foraging efficiency modifiers for species-flower combination"""

        proboscis = self.get_species_proboscis(species_name)
        result = self.calculate_accessibility(proboscis, flower)

        modifiers = {
            "nectar_efficiency": result.nectar_extraction_efficiency,
            "pollen_efficiency": result.pollen_extraction_efficiency,
            "energy_cost": result.energy_cost_multiplier,
            "handling_time": result.handling_time_multiplier,
            "accessibility_score": result.accessibility_score,
            "spatial_efficiency": result.spatial_efficiency,
        }
        
        # Add 3D geometric details if available
        if result.geometric_3d:
            modifiers.update({
                "optimal_angle": result.geometric_3d.optimal_approach_angle,
                "insertion_depth": result.geometric_3d.insertion_depth_possible,
                "collision_risk": result.geometric_3d.collision_risk,
                "path_clearance": result.geometric_3d.path_clearance_score,
            })
        
        return modifiers

    def get_community_analysis(
        self, species_list: List[str], flower_species_list: List[FlowerSpecies]
    ) -> Dict[str, Any]:
        """Analyze flower accessibility for multiple bee species"""

        community_data: Dict[str, Any] = {
            "species_accessibility": {},
            "flower_specialization": {},
            "resource_partitioning": {},
            "competition_analysis": {},
        }

        # Calculate accessibility for each species
        for species in species_list:
            accessibility_results = self.evaluate_flower_patch(
                species, flower_species_list
            )

            accessible_flowers = [
                flower.name
                for flower, result in zip(
                    flower_species_list, accessibility_results.values()
                )
                if result.is_accessible()
            ]

            community_data["species_accessibility"][species] = {
                "accessible_flowers": accessible_flowers,
                "accessibility_count": len(accessible_flowers),
                "average_accessibility": np.mean(
                    [
                        result.accessibility_score
                        for result in accessibility_results.values()
                    ]
                ),
                "patch_score": self.calculate_patch_accessibility_score(
                    species, flower_species_list
                ),
            }

        # Analyze flower specialization
        for flower in flower_species_list:
            accessing_species = []
            accessibility_scores = []

            for species in species_list:
                proboscis = self.get_species_proboscis(species)
                result = self.calculate_accessibility(proboscis, flower)

                if result.is_accessible():
                    accessing_species.append(species)
                    accessibility_scores.append(result.accessibility_score)

            community_data["flower_specialization"][flower.name] = {
                "accessing_species": accessing_species,
                "specialist_score": 1.0 - (len(accessing_species) / len(species_list)),
                "average_accessibility": (
                    np.mean(accessibility_scores) if accessibility_scores else 0.0
                ),
                "corolla_depth": flower.corolla_depth_mm,
            }

        return community_data

    def get_optimal_flower_matches(
        self, species_name: str, flower_species_list: List[FlowerSpecies]
    ) -> List[Tuple[FlowerSpecies, float]]:
        """Get optimal flower matches for species, sorted by accessibility"""

        accessibility_results = self.evaluate_flower_patch(
            species_name, flower_species_list
        )

        # Create list of (flower, accessibility_score) tuples
        matches = [
            (flower, accessibility_results[flower.name].accessibility_score)
            for flower in flower_species_list
        ]

        # Sort by accessibility score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)

        # Filter to only accessible flowers
        accessible_matches = [
            (flower, score) for flower, score in matches if score > 0.0
        ]

        return accessible_matches

    def update_flower_species_corolla_data(
        self, flower_species_list: List[FlowerSpecies]
    ) -> None:
        """Update flower species with realistic corolla depth data"""

        # Realistic corolla depth data for common UK flowers
        corolla_depths = {
            "White Clover": 4.5,
            "Red Clover": 9.2,
            "Dandelion": 3.8,
            "Bramble": 5.1,
            "Hawthorn": 2.9,
            "Blackthorn": 3.2,
            "Oilseed Rape": 2.1,
            "Field Bean": 8.7,
            "Heather": 4.6,
            "Gorse": 7.3,
            "Lime Tree": 4.8,
            "Phacelia": 3.9,
            "Borage": 5.4,
            "Lavender": 6.8,
            "Rosemary": 8.1,
            "Foxglove": 25.0,  # Very deep
            "Comfrey": 12.5,
            "Viper's Bugloss": 7.9,
            "Salvia": 15.2,
            "Honeysuckle": 18.7,
        }

        # Update flower species with corolla depth data
        for flower in flower_species_list:
            if flower.name in corolla_depths:
                flower.corolla_depth_mm = corolla_depths[flower.name]
                flower.corolla_width_mm = max(
                    1.0, flower.corolla_depth_mm * 0.3
                )  # Approximate width

                # Adjust nectar accessibility based on depth
                if flower.corolla_depth_mm > 15.0:
                    flower.nectar_accessibility = 0.6  # Deep flowers less accessible
                elif flower.corolla_depth_mm > 10.0:
                    flower.nectar_accessibility = 0.8
                else:
                    flower.nectar_accessibility = (
                        1.0  # Shallow flowers fully accessible
                    )
