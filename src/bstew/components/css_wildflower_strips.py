"""
Wildflower Strips System for CSS Implementation
===============================================

Implements wildflower strips as part of the Countryside Stewardship Scheme (CSS),
including strip placement algorithms, seed mix compositions, flowering succession
planning, and maintenance scheduling.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import date
import logging
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from .stewardship import HabitatType
from ..spatial.patches import ResourcePatch


class StripWidth(Enum):
    """Standard wildflower strip widths"""

    NARROW = 3  # 3 meters - minimum effective width
    STANDARD = 6  # 6 meters - standard width
    WIDE = 12  # 12 meters - enhanced biodiversity
    EXTRA_WIDE = 20  # 20 meters - maximum biodiversity


class PlacementStrategy(Enum):
    """Strip placement strategies"""

    FIELD_EDGE = "field_edge"  # Along field boundaries
    IN_FIELD = "in_field"  # Within fields (beetle banks)
    WATERCOURSE = "watercourse"  # Along water features
    HEDGEROW = "hedgerow"  # Adjacent to hedgerows
    CONNECTIVITY = "connectivity"  # Linking habitat patches
    SLOPE_CONTOUR = "slope_contour"  # Following land contours


class SeedMixType(Enum):
    """Specialized seed mix types for different objectives"""

    GENERAL_WILDFLOWER = "general_wildflower"
    POLLINATOR_NECTAR = "pollinator_nectar"
    BIRD_SEED = "bird_seed"
    BUTTERFLY_LARVAE = "butterfly_larvae"
    BENEFICIAL_INSECTS = "beneficial_insects"
    GRASS_WILDFLOWER = "grass_wildflower"
    ANNUAL_FLOWERS = "annual_flowers"
    PERENNIAL_NATIVES = "perennial_natives"


class FloweringPeriod(Enum):
    """Flowering period categories"""

    EARLY_SPRING = "early_spring"  # March-April
    LATE_SPRING = "late_spring"  # May-June
    EARLY_SUMMER = "early_summer"  # June-July
    MID_SUMMER = "mid_summer"  # July-August
    LATE_SUMMER = "late_summer"  # August-September
    AUTUMN = "autumn"  # September-October


@dataclass
class SpeciesComposition:
    """Individual species in seed mix"""

    scientific_name: str
    common_name: str
    percentage: float  # Percentage of mix by weight
    flowering_start_day: int  # Day of year
    flowering_end_day: int  # Day of year
    flowering_period: FloweringPeriod
    nectar_rating: float  # 0-10 scale
    pollen_rating: float  # 0-10 scale
    height_cm: int  # Average height
    establishment_rate: float  # 0-1 success rate
    persistence_years: int  # Expected lifespan


@dataclass
class MaintenanceActivity:
    """Scheduled maintenance activity"""

    activity_type: str  # cutting, topping, scarifying, etc.
    timing_start_day: int
    timing_end_day: int
    frequency_years: float  # Can be 0.5 for twice yearly
    equipment: str
    height_cm: Optional[int] = None  # Cutting height if applicable
    notes: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


class StripConfiguration(BaseModel):
    """Configuration for a wildflower strip"""

    model_config = {"validate_assignment": True}

    strip_id: str = Field(description="Unique identifier")
    width: StripWidth = Field(description="Strip width category")
    length_meters: float = Field(gt=0, description="Strip length")
    placement: PlacementStrategy = Field(description="Placement strategy")
    seed_mix_type: SeedMixType = Field(description="Primary seed mix")
    establishment_year: int = Field(description="Year established")

    # Location and geometry
    start_x: float = Field(description="Start X coordinate")
    start_y: float = Field(description="Start Y coordinate")
    end_x: float = Field(description="End X coordinate")
    end_y: float = Field(description="End Y coordinate")
    adjacent_patches: List[int] = Field(
        default_factory=list, description="Adjacent patch IDs"
    )

    # Seed mix composition
    species_composition: List[SpeciesComposition] = Field(default_factory=list)
    total_seed_rate_kg_ha: float = Field(default=2.0, ge=0.5, le=5.0)

    # Management
    maintenance_schedule: List[MaintenanceActivity] = Field(default_factory=list)
    last_maintenance: Optional[date] = None

    # Economic
    establishment_cost_per_ha: float = Field(default=750.0)
    annual_payment_per_ha: float = Field(default=450.0)
    management_payment_per_ha: float = Field(default=100.0)

    @field_validator("species_composition")
    def validate_composition_percentages(
        cls, v: Optional[List[Any]]
    ) -> Optional[List[Any]]:
        """Ensure species percentages sum to 100%"""
        if v:
            total = sum(species.percentage for species in v)
            if not (99.5 <= total <= 100.5):  # Allow small rounding errors
                raise ValueError(f"Species percentages must sum to 100%, got {total}%")
        return v

    @property
    def area_hectares(self) -> float:
        """Calculate strip area in hectares"""
        return (self.length_meters * self.width.value) / 10000

    @property
    def annual_payment(self) -> float:
        """Calculate total annual payment"""
        area = self.area_hectares
        return (self.annual_payment_per_ha + self.management_payment_per_ha) * area


class WildflowerStripSystem:
    """
    Wildflower Strip System implementing CSS requirements.

    Manages:
    - Strategic strip placement algorithms
    - Seed mix design and succession planning
    - Maintenance scheduling and compliance
    - Integration with landscape connectivity
    - Economic and ecological monitoring
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Active strips
        self.active_strips: Dict[str, StripConfiguration] = {}
        self.patch_strips: Dict[int, List[str]] = {}  # patch_id -> strip_ids

        # Seed mix templates
        self.seed_mix_templates = self._initialize_seed_mixes()

        # Placement constraints
        self.placement_rules = self._initialize_placement_rules()

        # Maintenance templates
        self.maintenance_templates = self._initialize_maintenance_templates()

        # Flowering calendar for succession planning
        self.flowering_calendar: Dict[int, List[str]] = defaultdict(list)

    def _initialize_seed_mixes(self) -> Dict[SeedMixType, List[SpeciesComposition]]:
        """Initialize standard seed mix compositions"""
        return {
            SeedMixType.GENERAL_WILDFLOWER: [
                SpeciesComposition(
                    "Leucanthemum vulgare",
                    "Oxeye Daisy",
                    15.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    7.0,
                    6.0,
                    60,
                    0.85,
                    5,
                ),
                SpeciesComposition(
                    "Centaurea nigra",
                    "Common Knapweed",
                    12.0,
                    182,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    9.0,
                    8.0,
                    80,
                    0.80,
                    10,
                ),
                SpeciesComposition(
                    "Lotus corniculatus",
                    "Bird's-foot Trefoil",
                    10.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    8.0,
                    9.0,
                    30,
                    0.75,
                    8,
                ),
                SpeciesComposition(
                    "Achillea millefolium",
                    "Yarrow",
                    8.0,
                    182,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    6.0,
                    5.0,
                    60,
                    0.90,
                    10,
                ),
                SpeciesComposition(
                    "Plantago lanceolata",
                    "Ribwort Plantain",
                    10.0,
                    121,
                    213,
                    FloweringPeriod.EARLY_SUMMER,
                    3.0,
                    7.0,
                    40,
                    0.95,
                    10,
                ),
                SpeciesComposition(
                    "Prunella vulgaris",
                    "Selfheal",
                    8.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    7.0,
                    6.0,
                    20,
                    0.80,
                    8,
                ),
                SpeciesComposition(
                    "Ranunculus acris",
                    "Meadow Buttercup",
                    7.0,
                    121,
                    213,
                    FloweringPeriod.EARLY_SUMMER,
                    5.0,
                    6.0,
                    60,
                    0.85,
                    7,
                ),
                SpeciesComposition(
                    "Trifolium pratense",
                    "Red Clover",
                    12.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    9.0,
                    8.0,
                    40,
                    0.90,
                    3,
                ),
                SpeciesComposition(
                    "Festuca rubra",
                    "Red Fescue",
                    18.0,
                    152,
                    182,
                    FloweringPeriod.MID_SUMMER,
                    1.0,
                    3.0,
                    60,
                    0.95,
                    15,
                ),
            ],
            SeedMixType.POLLINATOR_NECTAR: [
                SpeciesComposition(
                    "Echium vulgare",
                    "Viper's Bugloss",
                    8.0,
                    152,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    10.0,
                    8.0,
                    90,
                    0.70,
                    2,
                ),
                SpeciesComposition(
                    "Centaurea cyanus",
                    "Cornflower",
                    10.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    9.0,
                    7.0,
                    80,
                    0.85,
                    1,
                ),
                SpeciesComposition(
                    "Borago officinalis",
                    "Borage",
                    8.0,
                    152,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    10.0,
                    9.0,
                    60,
                    0.90,
                    1,
                ),
                SpeciesComposition(
                    "Phacelia tanacetifolia",
                    "Phacelia",
                    12.0,
                    121,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    10.0,
                    8.0,
                    80,
                    0.95,
                    1,
                ),
                SpeciesComposition(
                    "Origanum vulgare",
                    "Wild Marjoram",
                    6.0,
                    182,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    9.0,
                    7.0,
                    50,
                    0.75,
                    10,
                ),
                SpeciesComposition(
                    "Malva moschata",
                    "Musk Mallow",
                    8.0,
                    182,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    8.0,
                    6.0,
                    80,
                    0.80,
                    5,
                ),
                SpeciesComposition(
                    "Knautia arvensis",
                    "Field Scabious",
                    10.0,
                    182,
                    304,
                    FloweringPeriod.AUTUMN,
                    9.0,
                    7.0,
                    80,
                    0.75,
                    8,
                ),
                SpeciesComposition(
                    "Trifolium hybridum",
                    "Alsike Clover",
                    10.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    9.0,
                    8.0,
                    40,
                    0.90,
                    3,
                ),
                SpeciesComposition(
                    "Calendula officinalis",
                    "Pot Marigold",
                    8.0,
                    152,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    7.0,
                    5.0,
                    40,
                    0.85,
                    1,
                ),
                SpeciesComposition(
                    "Agastache foeniculum",
                    "Anise Hyssop",
                    10.0,
                    182,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    10.0,
                    6.0,
                    90,
                    0.80,
                    3,
                ),
                SpeciesComposition(
                    "Melilotus officinalis",
                    "Yellow Sweet Clover",
                    10.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    9.0,
                    7.0,
                    120,
                    0.85,
                    2,
                ),
            ],
            SeedMixType.ANNUAL_FLOWERS: [
                SpeciesComposition(
                    "Papaver rhoeas",
                    "Common Poppy",
                    5.0,
                    152,
                    213,
                    FloweringPeriod.EARLY_SUMMER,
                    5.0,
                    9.0,
                    60,
                    0.90,
                    1,
                ),
                SpeciesComposition(
                    "Centaurea cyanus",
                    "Cornflower",
                    15.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    9.0,
                    7.0,
                    80,
                    0.85,
                    1,
                ),
                SpeciesComposition(
                    "Chrysanthemum segetum",
                    "Corn Marigold",
                    10.0,
                    152,
                    244,
                    FloweringPeriod.MID_SUMMER,
                    7.0,
                    5.0,
                    50,
                    0.85,
                    1,
                ),
                SpeciesComposition(
                    "Calendula officinalis",
                    "Pot Marigold",
                    15.0,
                    152,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    7.0,
                    5.0,
                    40,
                    0.85,
                    1,
                ),
                SpeciesComposition(
                    "Iberis umbellata",
                    "Garden Candytuft",
                    10.0,
                    121,
                    213,
                    FloweringPeriod.EARLY_SUMMER,
                    6.0,
                    4.0,
                    30,
                    0.80,
                    1,
                ),
                SpeciesComposition(
                    "Cosmos bipinnatus",
                    "Cosmos",
                    15.0,
                    182,
                    304,
                    FloweringPeriod.AUTUMN,
                    8.0,
                    5.0,
                    100,
                    0.90,
                    1,
                ),
                SpeciesComposition(
                    "Eschscholzia californica",
                    "California Poppy",
                    10.0,
                    152,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    5.0,
                    3.0,
                    30,
                    0.85,
                    1,
                ),
                SpeciesComposition(
                    "Fagopyrum esculentum",
                    "Buckwheat",
                    20.0,
                    182,
                    274,
                    FloweringPeriod.LATE_SUMMER,
                    9.0,
                    8.0,
                    100,
                    0.95,
                    1,
                ),
            ],
        }

    def _initialize_placement_rules(self) -> Dict[PlacementStrategy, Dict[str, Any]]:
        """Initialize placement strategy rules and constraints"""
        return {
            PlacementStrategy.FIELD_EDGE: {
                "min_field_size_ha": 2.0,
                "preferred_aspects": ["south", "west"],  # Sun exposure
                "buffer_from_hedge_m": 1.0,
                "max_shade_percentage": 30,
                "connectivity_bonus": 0.2,  # Bonus for connecting habitats
            },
            PlacementStrategy.IN_FIELD: {
                "min_field_size_ha": 5.0,
                "min_distance_from_edge_m": 20.0,
                "preferred_slope": "gentle",  # <5% slope
                "orientation": "contour_following",
                "beetle_bank_width_m": 2.0,  # Additional refuge
            },
            PlacementStrategy.WATERCOURSE: {
                "buffer_from_water_m": 3.0,
                "max_flood_risk": "low",
                "preferred_species": ["wetland_tolerant"],
                "additional_benefits": ["water_quality", "erosion_control"],
            },
            PlacementStrategy.CONNECTIVITY: {
                "max_gap_between_habitats_m": 100.0,
                "min_corridor_width_m": 6.0,
                "preferred_alignment": "straight_or_gentle_curve",
                "stepping_stone_spacing_m": 50.0,
            },
        }

    def _initialize_maintenance_templates(
        self,
    ) -> Dict[SeedMixType, List[MaintenanceActivity]]:
        """Initialize maintenance schedules for different mix types"""
        return {
            SeedMixType.GENERAL_WILDFLOWER: [
                MaintenanceActivity(
                    "autumn_cut",
                    274,
                    334,
                    1.0,
                    "flail_mower",
                    height_cm=10,
                    notes=["Remove cuttings", "After seed set"],
                ),
                MaintenanceActivity(
                    "spring_scarify",
                    60,
                    120,
                    3.0,
                    "chain_harrow",
                    notes=["Light scarification", "Encourage germination"],
                ),
            ],
            SeedMixType.POLLINATOR_NECTAR: [
                MaintenanceActivity(
                    "late_cut",
                    305,
                    365,
                    1.0,
                    "reciprocating_mower",
                    height_cm=15,
                    notes=["Retain winter structure", "Phased cutting"],
                ),
                MaintenanceActivity(
                    "selective_topping",
                    182,
                    213,
                    2.0,
                    "brush_cutter",
                    notes=["Control dominant species", "Maintain diversity"],
                ),
            ],
            SeedMixType.ANNUAL_FLOWERS: [
                MaintenanceActivity(
                    "cultivation",
                    60,
                    120,
                    1.0,
                    "rotovator",
                    notes=["Annual cultivation", "Prepare seedbed"],
                ),
                MaintenanceActivity(
                    "rolling",
                    121,
                    151,
                    1.0,
                    "cambridge_roller",
                    notes=["After sowing", "Improve germination"],
                ),
            ],
        }

    def create_wildflower_strip(
        self,
        strip_id: str,
        width: StripWidth,
        length_meters: float,
        placement: PlacementStrategy,
        seed_mix_type: SeedMixType,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        establishment_year: int,
        custom_species: Optional[List[SpeciesComposition]] = None,
    ) -> StripConfiguration:
        """Create a new wildflower strip"""

        # Use template or custom species composition
        if custom_species:
            species_composition = custom_species
        else:
            species_composition = self.seed_mix_templates.get(
                seed_mix_type, self.seed_mix_templates[SeedMixType.GENERAL_WILDFLOWER]
            ).copy()

        # Get maintenance template
        maintenance_schedule = self.maintenance_templates.get(
            seed_mix_type, self.maintenance_templates[SeedMixType.GENERAL_WILDFLOWER]
        ).copy()

        # Create configuration
        strip_config = StripConfiguration(
            strip_id=strip_id,
            width=width,
            length_meters=length_meters,
            placement=placement,
            seed_mix_type=seed_mix_type,
            establishment_year=establishment_year,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            species_composition=species_composition,
            maintenance_schedule=maintenance_schedule,
        )

        self.active_strips[strip_id] = strip_config

        # Update flowering calendar
        self._update_flowering_calendar(strip_config)

        self.logger.info(
            f"Created wildflower strip {strip_id}: {width.value}m x {length_meters}m "
            f"{seed_mix_type.value} ({strip_config.area_hectares:.2f}ha)"
        )

        return strip_config

    def _update_flowering_calendar(self, strip: StripConfiguration) -> None:
        """Update the flowering calendar with strip species"""
        for species in strip.species_composition:
            for day in range(
                species.flowering_start_day, species.flowering_end_day + 1
            ):
                self.flowering_calendar[day].append(strip.strip_id)

    def optimize_strip_placement(
        self,
        available_patches: List[ResourcePatch],
        target_area_ha: float,
        placement_strategy: PlacementStrategy,
        connectivity_targets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Optimize wildflower strip placement across landscape.

        Args:
            available_patches: Patches available for strip placement
            target_area_ha: Total target area for strips
            placement_strategy: Primary placement strategy
            connectivity_targets: Optional patch pairs to connect

        Returns:
            List of recommended strip placements
        """
        recommendations = []
        rules = self.placement_rules[placement_strategy]

        # Score patches based on placement criteria
        patch_scores = []
        for patch in available_patches:
            score = self._score_patch_for_strip(patch, placement_strategy, rules)

            # Bonus for connectivity
            if connectivity_targets:
                for source_id, target_id in connectivity_targets:
                    if patch.id in [source_id, target_id]:
                        score *= 1.5

            patch_scores.append((patch, score))

        # Sort by score
        patch_scores.sort(key=lambda x: x[1], reverse=True)

        # Generate strip recommendations
        current_area = 0.0
        for patch, score in patch_scores:
            if current_area >= target_area_ha:
                break

            # Determine optimal strip configuration for patch
            strip_config = self._configure_strip_for_patch(
                patch, placement_strategy, target_area_ha - current_area
            )

            if strip_config:
                recommendations.append(
                    {
                        "patch_id": patch.id,
                        "score": score,
                        "strip_config": strip_config,
                        "rationale": self._get_placement_rationale(
                            patch, placement_strategy
                        ),
                    }
                )

                current_area += strip_config["area_ha"]

        return recommendations

    def _score_patch_for_strip(
        self, patch: ResourcePatch, strategy: PlacementStrategy, rules: Dict[str, Any]
    ) -> float:
        """Score a patch for strip placement suitability"""
        score = 1.0

        # Base habitat suitability
        if patch.habitat_type == HabitatType.CROPLAND:
            score *= 1.2  # Preferred for conversion
        elif patch.habitat_type == HabitatType.GRASSLAND:
            score *= 0.8  # Already providing some benefit

        # Size constraints
        if "min_field_size_ha" in rules:
            if getattr(patch, "area_ha", 1.0) < rules["min_field_size_ha"]:
                score *= 0.5

        # Existing biodiversity (prefer lower for improvement)
        if hasattr(patch, "biodiversity_score"):
            score *= 2.0 - patch.biodiversity_score  # Inverse relationship

        # Landscape position factors
        if strategy == PlacementStrategy.CONNECTIVITY:
            # Favor patches between other habitats
            score *= 1.5  # Simplified - would check actual connectivity

        return score

    def _configure_strip_for_patch(
        self, patch: ResourcePatch, strategy: PlacementStrategy, max_area_ha: float
    ) -> Optional[Dict[str, Any]]:
        """Configure optimal strip dimensions for a patch"""

        # Estimate patch perimeter (simplified as square)
        patch_side = np.sqrt(
            getattr(patch, "area_ha", 1.0) * 10000
        )  # Convert to meters
        perimeter = 4 * patch_side

        # Determine width based on strategy
        if strategy == PlacementStrategy.FIELD_EDGE:
            preferred_width = StripWidth.STANDARD
        elif strategy == PlacementStrategy.CONNECTIVITY:
            preferred_width = StripWidth.WIDE
        else:
            preferred_width = StripWidth.STANDARD

        # Calculate strip area
        strip_area_ha = (perimeter * preferred_width.value) / 10000

        if strip_area_ha > max_area_ha:
            # Adjust width or length
            strip_area_ha = max_area_ha
            actual_length = (max_area_ha * 10000) / preferred_width.value
        else:
            actual_length = perimeter

        return {
            "width": preferred_width,
            "length_m": actual_length,
            "area_ha": strip_area_ha,
            "placement": strategy,
        }

    def _get_placement_rationale(
        self, patch: ResourcePatch, strategy: PlacementStrategy
    ) -> str:
        """Generate human-readable rationale for placement"""
        rationales = {
            PlacementStrategy.FIELD_EDGE: "Field edge placement maximizes accessibility and minimizes crop area loss",
            PlacementStrategy.CONNECTIVITY: "Strategic placement to enhance landscape connectivity",
            PlacementStrategy.IN_FIELD: "In-field strips provide refuge and break up large monocultures",
            PlacementStrategy.WATERCOURSE: "Riparian buffer provides water quality benefits",
        }
        return rationales.get(strategy, "Strategic placement for biodiversity")

    def plan_flowering_succession(
        self,
        target_months: List[int],
        available_area_ha: float,
        priority_species: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Plan species mix for continuous flowering across target months.

        Args:
            target_months: Months requiring flowering (1-12)
            available_area_ha: Total area available
            priority_species: Optional priority species to include

        Returns:
            Succession plan with species mix and timing
        """
        # Convert months to day ranges
        month_to_days = {
            1: (1, 31),
            2: (32, 59),
            3: (60, 90),
            4: (91, 120),
            5: (121, 151),
            6: (152, 181),
            7: (182, 212),
            8: (213, 243),
            9: (244, 273),
            10: (274, 304),
            11: (305, 334),
            12: (335, 365),
        }

        target_days: Set[int] = set()
        for month in target_months:
            start, end = month_to_days[month]
            target_days.update(range(start, end + 1))

        # Select species to cover target period
        selected_species = []
        covered_days: Set[int] = set()

        # First, include priority species
        all_species = []
        for mix_type, species_list in self.seed_mix_templates.items():
            all_species.extend(species_list)

        if priority_species:
            for species in all_species:
                if species.scientific_name in priority_species:
                    selected_species.append(species)
                    covered_days.update(
                        range(
                            species.flowering_start_day, species.flowering_end_day + 1
                        )
                    )

        # Fill gaps with additional species
        remaining_days = target_days - covered_days

        # Sort species by coverage of remaining days
        species_scores = []
        for species in all_species:
            if species not in selected_species:
                species_days = set(
                    range(species.flowering_start_day, species.flowering_end_day + 1)
                )
                coverage = len(species_days & remaining_days)
                nectar_score = species.nectar_rating + species.pollen_rating

                score = coverage * nectar_score
                species_scores.append((species, score, coverage))

        species_scores.sort(key=lambda x: x[1], reverse=True)

        # Add species until target period covered
        for species, score, coverage in species_scores:
            if not remaining_days:
                break

            selected_species.append(species)
            species_days = set(
                range(species.flowering_start_day, species.flowering_end_day + 1)
            )
            covered_days.update(species_days)
            remaining_days -= species_days

        # Calculate percentages to sum to 100%
        total_weight = len(selected_species)
        base_percentage = 100.0 / total_weight if total_weight > 0 else 0

        for i, species in enumerate(selected_species):
            if i == len(selected_species) - 1:
                # Last species gets remainder to ensure 100%
                species.percentage = 100.0 - sum(
                    s.percentage for s in selected_species[:-1]
                )
            else:
                species.percentage = round(base_percentage, 1)

        # Create succession timeline
        timeline = defaultdict(list)
        for species in selected_species:
            for day in range(
                species.flowering_start_day, species.flowering_end_day + 1
            ):
                timeline[day].append(species.common_name)

        # Calculate coverage statistics
        coverage_percent = (
            (len(covered_days & target_days) / len(target_days) * 100)
            if target_days
            else 0
        )

        return {
            "selected_species": selected_species,
            "total_species": len(selected_species),
            "coverage_percent": coverage_percent,
            "covered_days": len(covered_days & target_days),
            "target_days": len(target_days),
            "gap_days": sorted(list(target_days - covered_days)),
            "timeline_summary": {
                month: len(
                    [
                        d
                        for d in range(
                            month_to_days[month][0], month_to_days[month][1] + 1
                        )
                        if d in covered_days
                    ]
                )
                for month in target_months
            },
            "establishment_cost_per_ha": self._calculate_mix_cost(selected_species),
            "average_nectar_rating": np.mean(
                [s.nectar_rating for s in selected_species]
            )
            if selected_species
            else 0,
            "average_pollen_rating": np.mean(
                [s.pollen_rating for s in selected_species]
            )
            if selected_species
            else 0,
        }

    def _calculate_mix_cost(self, species_list: List[SpeciesComposition]) -> float:
        """Calculate establishment cost for species mix"""
        # Base cost plus complexity factor
        base_cost = 500.0
        species_cost = len(species_list) * 25.0  # More species = higher cost

        # Adjust for establishment difficulty
        avg_establishment = (
            np.mean([s.establishment_rate for s in species_list])
            if species_list
            else 0.5
        )
        difficulty_factor = (
            2.0 - avg_establishment
        )  # Harder to establish = more expensive

        return float(base_cost + species_cost * difficulty_factor)

    def schedule_maintenance(
        self, strip_id: str, current_date: date, years_ahead: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate maintenance schedule for specified time period.

        Args:
            strip_id: Strip to schedule maintenance for
            current_date: Current date
            years_ahead: Years to plan ahead

        Returns:
            List of scheduled maintenance activities
        """
        if strip_id not in self.active_strips:
            return []

        strip = self.active_strips[strip_id]
        schedule = []

        current_year = current_date.year

        for year_offset in range(years_ahead):
            year = current_year + year_offset

            for activity in strip.maintenance_schedule:
                # Check if activity due this year
                years_since_establishment = year - strip.establishment_year

                if years_since_establishment < 1:
                    continue  # No maintenance in establishment year

                if activity.frequency_years >= 1:
                    # Annual or less frequent
                    if years_since_establishment % int(activity.frequency_years) == 0:
                        schedule.append(
                            {
                                "year": year,
                                "activity": activity.activity_type,
                                "timing_window": f"Day {activity.timing_start_day}-{activity.timing_end_day}",
                                "equipment": activity.equipment,
                                "notes": activity.notes,
                                "area_ha": strip.area_hectares,
                                "status": "scheduled",
                            }
                        )
                else:
                    # Multiple times per year
                    times_per_year = int(1 / activity.frequency_years)
                    for occurrence in range(times_per_year):
                        adjusted_start = activity.timing_start_day + (
                            occurrence * 365 // times_per_year
                        )
                        adjusted_end = activity.timing_end_day + (
                            occurrence * 365 // times_per_year
                        )

                        schedule.append(
                            {
                                "year": year,
                                "activity": f"{activity.activity_type}_{occurrence + 1}",
                                "timing_window": f"Day {adjusted_start}-{adjusted_end}",
                                "equipment": activity.equipment,
                                "notes": activity.notes,
                                "area_ha": strip.area_hectares,
                                "status": "scheduled",
                            }
                        )

        return sorted(schedule, key=lambda x: (x["year"], x["timing_window"]))

    def calculate_strip_resources(
        self, strip_id: str, day_of_year: int
    ) -> Dict[str, float]:
        """Calculate resource provision from a strip on given day"""
        if strip_id not in self.active_strips:
            return {"nectar": 0.0, "pollen": 0.0}

        strip = self.active_strips[strip_id]
        total_nectar = 0.0
        total_pollen = 0.0

        for species in strip.species_composition:
            # Check if species is flowering
            if species.flowering_start_day <= day_of_year <= species.flowering_end_day:
                # Calculate production based on percentage of mix and area
                species_area = strip.area_hectares * (species.percentage / 100.0)

                # Adjust for establishment success
                establishment_factor = species.establishment_rate

                # Simple production model (mg/m²/day to kg/ha/day)
                nectar_production = (
                    species.nectar_rating * 0.5 * establishment_factor
                )  # kg/ha/day
                pollen_production = (
                    species.pollen_rating * 0.3 * establishment_factor
                )  # kg/ha/day

                total_nectar += nectar_production * species_area
                total_pollen += pollen_production * species_area

        return {
            "nectar": total_nectar,
            "pollen": total_pollen,
            "flowering_species": sum(
                1
                for s in strip.species_composition
                if s.flowering_start_day <= day_of_year <= s.flowering_end_day
            ),
        }

    def get_strip_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all wildflower strips"""
        if not self.active_strips:
            return {"status": "no_strips"}

        total_area = sum(s.area_hectares for s in self.active_strips.values())
        total_length = sum(s.length_meters for s in self.active_strips.values())

        width_breakdown: Dict[str, float] = defaultdict(float)
        placement_breakdown: Dict[str, float] = defaultdict(float)
        mix_breakdown: Dict[str, float] = defaultdict(float)

        for strip in self.active_strips.values():
            width_breakdown[str(strip.width.value)] += strip.area_hectares
            placement_breakdown[strip.placement.value] += strip.area_hectares
            mix_breakdown[strip.seed_mix_type.value] += strip.area_hectares

        # Calculate annual payments
        total_payments = sum(s.annual_payment for s in self.active_strips.values())

        return {
            "total_strips": len(self.active_strips),
            "total_area_hectares": total_area,
            "total_length_km": total_length / 1000,
            "average_width_m": sum(s.width.value for s in self.active_strips.values())
            / len(self.active_strips),
            "width_breakdown_ha": dict(width_breakdown),
            "placement_breakdown_ha": dict(placement_breakdown),
            "mix_type_breakdown_ha": dict(mix_breakdown),
            "total_annual_payments": total_payments,
            "average_payment_per_ha": total_payments / total_area
            if total_area > 0
            else 0,
            "total_species": len(
                set(
                    species.scientific_name
                    for strip in self.active_strips.values()
                    for species in strip.species_composition
                )
            ),
        }

    def validate_strip_compliance(
        self, strip_id: str, current_date: date
    ) -> Dict[str, Any]:
        """Validate strip compliance with CSS requirements"""
        if strip_id not in self.active_strips:
            return {"valid": False, "error": "Strip not found"}

        strip = self.active_strips[strip_id]
        issues = []
        warnings = []

        # Check establishment
        years_established = current_date.year - strip.establishment_year
        if years_established < 0:
            issues.append("Strip establishment year is in the future")

        # Check minimum width
        if strip.width.value < 3:
            issues.append("Strip width below minimum 3m requirement")

        # Check species composition
        if not strip.species_composition:
            issues.append("No species composition defined")
        else:
            total_percentage = sum(s.percentage for s in strip.species_composition)
            if not (99.5 <= total_percentage <= 100.5):
                issues.append(
                    f"Species percentages sum to {total_percentage}%, not 100%"
                )

        # Check maintenance
        if not strip.maintenance_schedule:
            warnings.append("No maintenance schedule defined")

        # Check last maintenance
        if strip.last_maintenance:
            days_since_maintenance = (current_date - strip.last_maintenance).days
            if days_since_maintenance > 365:
                warnings.append(f"No maintenance for {days_since_maintenance} days")

        return {
            "valid": len(issues) == 0,
            "compliance_issues": issues,
            "warnings": warnings,
            "years_established": years_established,
            "next_maintenance": self._get_next_maintenance(strip, current_date),
        }

    def _get_next_maintenance(
        self, strip: StripConfiguration, current_date: date
    ) -> Optional[str]:
        """Get next scheduled maintenance activity"""
        current_day = current_date.timetuple().tm_yday
        current_year = current_date.year

        next_activities = []

        for activity in strip.maintenance_schedule:
            years_since = current_year - strip.establishment_year

            if activity.frequency_years >= 1:
                next_year = current_year
                if years_since % int(activity.frequency_years) != 0:
                    years_until = int(activity.frequency_years) - (
                        years_since % int(activity.frequency_years)
                    )
                    next_year = current_year + years_until

                if next_year == current_year and current_day > activity.timing_end_day:
                    next_year += int(activity.frequency_years)

                next_activities.append(
                    {
                        "activity": activity.activity_type,
                        "year": next_year,
                        "start_day": activity.timing_start_day,
                    }
                )

        if next_activities:
            next_activities.sort(key=lambda x: (x["year"], x["start_day"]))
            next_act = next_activities[0]
            return f"{next_act['activity']} in {next_act['year']} (day {next_act['start_day']})"

        return None
