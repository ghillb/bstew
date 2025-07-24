"""
Enhanced Margins System for CSS Implementation
=============================================

Implements enhanced field margins with configurable widths, species mixes,
seasonal management, and economic payment calculations as part of the
Countryside Stewardship Scheme (CSS) features.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging
from dataclasses import dataclass

from ..spatial.patches import ResourcePatch


class MarginWidth(Enum):
    """Enhanced margin width options"""

    NARROW = 6  # 6 meters
    MEDIUM = 12  # 12 meters
    WIDE = 24  # 24 meters


class SpeciesMix(Enum):
    """Species mix selection for enhanced margins"""

    WILDFLOWER_GRASSLAND = "wildflower_grassland"
    POLLINATOR_MIX = "pollinator_mix"
    NECTAR_RICH = "nectar_rich"
    LEGUME_RICH = "legume_rich"
    BIRD_FOOD = "bird_food"
    TUSSOCKY_GRASS = "tussocky_grass"


class ManagementOperation(Enum):
    """Seasonal management operations"""

    CUTTING = "cutting"
    TOPPING = "topping"
    GRAZING = "grazing"
    OVERSEEDING = "overseeding"
    CULTIVATION = "cultivation"
    WEED_CONTROL = "weed_control"


@dataclass
class ManagementSchedule:
    """Scheduled management operation"""

    operation: ManagementOperation
    timing_window_start: int  # Day of year
    timing_window_end: int  # Day of year
    frequency_years: int = 1  # How often (every N years)
    area_percentage: float = 100.0  # Percentage of margin area
    restrictions: Optional[List[str]] = None  # e.g., ["no_cutting_during_nesting"]

    def __post_init__(self) -> None:
        if self.restrictions is None:
            self.restrictions = []


class EnhancedMarginConfiguration(BaseModel):
    """Configuration for enhanced field margins"""

    model_config = {"validate_assignment": True}

    margin_id: str = Field(description="Unique identifier for this margin")
    width: MarginWidth = Field(description="Margin width category")
    species_mix: SpeciesMix = Field(description="Selected species mix")
    establishment_year: int = Field(description="Year margin was established")
    target_area_hectares: float = Field(gt=0.0, description="Target area in hectares")

    # Payment rates per hectare (£/ha/year)
    base_payment_rate: float = Field(default=640.0, description="Base CSS payment rate")
    width_multiplier: float = Field(default=1.0, description="Width-based multiplier")
    species_bonus: float = Field(default=0.0, description="Species mix bonus payment")
    management_bonus: float = Field(default=0.0, description="Management quality bonus")

    # Ecological parameters
    nectar_production_rate: float = Field(default=0.0, description="Nectar mg/m²/day")
    pollen_production_rate: float = Field(default=0.0, description="Pollen mg/m²/day")
    bloom_start_day: int = Field(ge=1, le=365, description="Bloom start day of year")
    bloom_end_day: int = Field(ge=1, le=365, description="Bloom end day of year")
    bee_attractiveness_score: float = Field(
        ge=0.0, le=1.0, description="Bee attractiveness"
    )

    # Management
    management_schedule: List[ManagementSchedule] = Field(default_factory=list)
    last_management_operations: Dict[str, int] = Field(
        default_factory=dict
    )  # operation -> year

    def model_post_init(self, __context: Any) -> None:
        """Set width multiplier after model initialization"""
        multipliers = {
            MarginWidth.NARROW: 1.0,
            MarginWidth.MEDIUM: 1.2,
            MarginWidth.WIDE: 1.5,
        }
        self.width_multiplier = multipliers.get(self.width, 1.0)

    def calculate_annual_payment(self) -> float:
        """Calculate annual CSS payment for this margin"""
        base_payment = self.base_payment_rate * self.target_area_hectares
        width_adjustment = base_payment * (self.width_multiplier - 1.0)
        species_adjustment = self.species_bonus * self.target_area_hectares
        management_adjustment = self.management_bonus * self.target_area_hectares

        return (
            base_payment + width_adjustment + species_adjustment + management_adjustment
        )


class EnhancedMarginSystem:
    """
    Enhanced Margins System implementing CSS requirements.

    Manages:
    - Width configuration (6m, 12m, 24m options)
    - Species mix selection and establishment
    - Seasonal management schedules
    - Economic payment calculations per hectare
    - Integration with existing stewardship system
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Active margin configurations
        self.active_margins: Dict[str, EnhancedMarginConfiguration] = {}
        self.patch_margins: Dict[int, List[str]] = {}  # patch_id -> margin_ids

        # Species mix characteristics
        self.species_mix_data = self._initialize_species_mix_data()

        # Management templates
        self.management_templates = self._initialize_management_templates()

        # Payment calculation parameters
        self.css_payment_rates = self._initialize_payment_rates()

    def _initialize_species_mix_data(self) -> Dict[SpeciesMix, Dict[str, Any]]:
        """Initialize species mix ecological characteristics"""
        return {
            SpeciesMix.WILDFLOWER_GRASSLAND: {
                "nectar_production": 4.2,  # mg/m²/day during bloom
                "pollen_production": 2.8,
                "bloom_start": 105,  # Mid-April
                "bloom_end": 273,  # End September
                "bee_attractiveness": 0.85,
                "establishment_cost": 450.0,  # £/ha
                "species_bonus": 50.0,  # £/ha/year bonus
                "target_species": [
                    "Centaurea nigra",
                    "Leucanthemum vulgare",
                    "Trifolium pratense",
                    "Lotus corniculatus",
                    "Plantago lanceolata",
                    "Achillea millefolium",
                ],
            },
            SpeciesMix.POLLINATOR_MIX: {
                "nectar_production": 6.8,
                "pollen_production": 4.2,
                "bloom_start": 91,  # Early April
                "bloom_end": 304,  # Late October
                "bee_attractiveness": 0.92,
                "establishment_cost": 520.0,
                "species_bonus": 75.0,
                "target_species": [
                    "Echium vulgare",
                    "Centaurea cyanus",
                    "Papaver rhoeas",
                    "Calendula officinalis",
                    "Phacelia tanacetifolia",
                    "Borago officinalis",
                ],
            },
            SpeciesMix.NECTAR_RICH: {
                "nectar_production": 8.5,
                "pollen_production": 3.6,
                "bloom_start": 114,  # Late April
                "bloom_end": 258,  # Mid September
                "bee_attractiveness": 0.95,
                "establishment_cost": 580.0,
                "species_bonus": 85.0,
                "target_species": [
                    "Echium vulgare",
                    "Viper's bugloss",
                    "Borage",
                    "Phacelia",
                    "Cornflower",
                    "Field scabious",
                ],
            },
            SpeciesMix.LEGUME_RICH: {
                "nectar_production": 3.8,
                "pollen_production": 5.2,
                "bloom_start": 135,  # Mid May
                "bloom_end": 244,  # Early September
                "bee_attractiveness": 0.78,
                "establishment_cost": 380.0,
                "species_bonus": 40.0,
                "target_species": [
                    "Trifolium pratense",
                    "Trifolium repens",
                    "Lotus corniculatus",
                    "Vicia cracca",
                    "Medicago sativa",
                    "Onobrychis viciifolia",
                ],
            },
            SpeciesMix.BIRD_FOOD: {
                "nectar_production": 2.1,
                "pollen_production": 1.8,
                "bloom_start": 120,  # Early May
                "bloom_end": 200,  # Mid July
                "bee_attractiveness": 0.45,
                "establishment_cost": 320.0,
                "species_bonus": 25.0,
                "target_species": [
                    "Panicum miliaceum",
                    "Echinochloa crus-galli",
                    "Setaria italica",
                    "Brassica napus",
                    "Helianthus annuus",
                    "Linum usitatissimum",
                ],
            },
            SpeciesMix.TUSSOCKY_GRASS: {
                "nectar_production": 0.8,
                "pollen_production": 1.2,
                "bloom_start": 152,  # Early June
                "bloom_end": 180,  # Late June
                "bee_attractiveness": 0.25,
                "establishment_cost": 280.0,
                "species_bonus": 15.0,
                "target_species": [
                    "Dactylis glomerata",
                    "Festuca rubra",
                    "Poa pratensis",
                    "Agrostis capillaris",
                    "Holcus lanatus",
                ],
            },
        }

    def _initialize_management_templates(
        self,
    ) -> Dict[SpeciesMix, List[ManagementSchedule]]:
        """Initialize management schedule templates for each species mix"""
        return {
            SpeciesMix.WILDFLOWER_GRASSLAND: [
                ManagementSchedule(
                    operation=ManagementOperation.CUTTING,
                    timing_window_start=274,  # Early October
                    timing_window_end=334,  # Late November
                    frequency_years=1,
                    area_percentage=100.0,
                    restrictions=["no_cutting_march_july", "leave_15cm_stubble"],
                ),
                ManagementSchedule(
                    operation=ManagementOperation.OVERSEEDING,
                    timing_window_start=244,  # Early September
                    timing_window_end=273,  # End September
                    frequency_years=3,
                    area_percentage=50.0,
                    restrictions=["good_seedbed_conditions"],
                ),
            ],
            SpeciesMix.POLLINATOR_MIX: [
                ManagementSchedule(
                    operation=ManagementOperation.CUTTING,
                    timing_window_start=305,  # Early November
                    timing_window_end=365,  # End December
                    frequency_years=1,
                    area_percentage=50.0,  # Rotational cutting
                    restrictions=["leave_seed_heads", "cut_different_areas_annually"],
                ),
                ManagementSchedule(
                    operation=ManagementOperation.WEED_CONTROL,
                    timing_window_start=91,  # Early April
                    timing_window_end=120,  # Early May
                    frequency_years=2,
                    area_percentage=100.0,
                    restrictions=["spot_treatment_only", "no_herbicide_during_bloom"],
                ),
            ],
            SpeciesMix.NECTAR_RICH: [
                ManagementSchedule(
                    operation=ManagementOperation.CUTTING,
                    timing_window_start=289,  # Mid October
                    timing_window_end=334,  # Late November
                    frequency_years=1,
                    area_percentage=30.0,  # Light management
                    restrictions=["minimal_intervention", "preserve_structure"],
                )
            ],
            SpeciesMix.LEGUME_RICH: [
                ManagementSchedule(
                    operation=ManagementOperation.TOPPING,
                    timing_window_start=213,  # Early August
                    timing_window_end=243,  # End August
                    frequency_years=1,
                    area_percentage=100.0,
                    restrictions=["avoid_seed_setting_time", "maintain_15cm_height"],
                ),
                ManagementSchedule(
                    operation=ManagementOperation.OVERSEEDING,
                    timing_window_start=244,  # Early September
                    timing_window_end=273,  # End September
                    frequency_years=5,
                    area_percentage=100.0,
                    restrictions=["legume_seed_only"],
                ),
            ],
            SpeciesMix.BIRD_FOOD: [
                ManagementSchedule(
                    operation=ManagementOperation.CULTIVATION,
                    timing_window_start=60,  # Early March
                    timing_window_end=120,  # Early May
                    frequency_years=1,
                    area_percentage=100.0,
                    restrictions=["prepare_seedbed", "annual_resowing"],
                )
            ],
            SpeciesMix.TUSSOCKY_GRASS: [
                ManagementSchedule(
                    operation=ManagementOperation.CUTTING,
                    timing_window_start=274,  # Early October
                    timing_window_end=334,  # Late November
                    frequency_years=3,  # Every 3 years
                    area_percentage=50.0,
                    restrictions=["rotational_cutting", "maintain_tussock_structure"],
                )
            ],
        }

    def _initialize_payment_rates(self) -> Dict[str, float]:
        """Initialize CSS payment rates (2024 rates in £/ha/year)"""
        return {
            "base_enhanced_margins": 640.0,  # AB8 Enhanced field margins
            "width_6m_base": 640.0,  # 6m margins
            "width_12m_multiplier": 1.2,  # 12m margins (+20%)
            "width_24m_multiplier": 1.5,  # 24m margins (+50%)
            "nectar_rich_bonus": 85.0,  # Additional for nectar-rich mixes
            "pollinator_bonus": 75.0,  # Additional for pollinator mixes
            "management_quality_bonus": 50.0,  # Quality management bonus
            "establishment_grant": 500.0,  # One-time establishment grant
        }

    def create_enhanced_margin(
        self,
        margin_id: str,
        width: MarginWidth,
        species_mix: SpeciesMix,
        target_area_hectares: float,
        establishment_year: int,
    ) -> EnhancedMarginConfiguration:
        """Create a new enhanced margin configuration"""

        # Get species mix characteristics
        mix_data = self.species_mix_data[species_mix]

        # Calculate payment rates
        base_rate = self.css_payment_rates["base_enhanced_margins"]
        width_multiplier = {
            MarginWidth.NARROW: 1.0,
            MarginWidth.MEDIUM: self.css_payment_rates["width_12m_multiplier"],
            MarginWidth.WIDE: self.css_payment_rates["width_24m_multiplier"],
        }[width]

        species_bonus = mix_data["species_bonus"]

        # Create configuration
        margin_config = EnhancedMarginConfiguration(
            margin_id=margin_id,
            width=width,
            species_mix=species_mix,
            establishment_year=establishment_year,
            target_area_hectares=target_area_hectares,
            base_payment_rate=base_rate,
            width_multiplier=width_multiplier,
            species_bonus=species_bonus,
            nectar_production_rate=mix_data["nectar_production"],
            pollen_production_rate=mix_data["pollen_production"],
            bloom_start_day=mix_data["bloom_start"],
            bloom_end_day=mix_data["bloom_end"],
            bee_attractiveness_score=mix_data["bee_attractiveness"],
            management_schedule=self.management_templates[species_mix].copy(),
        )

        self.active_margins[margin_id] = margin_config

        self.logger.info(
            f"Created enhanced margin {margin_id}: {width.value}m {species_mix.value} "
            f"({target_area_hectares:.2f}ha, £{margin_config.calculate_annual_payment():.2f}/year)"
        )

        return margin_config

    def assign_margin_to_patch(self, patch_id: int, margin_id: str) -> bool:
        """Assign an enhanced margin to a resource patch"""
        if margin_id not in self.active_margins:
            self.logger.warning(f"Unknown margin ID: {margin_id}")
            return False

        if patch_id not in self.patch_margins:
            self.patch_margins[patch_id] = []

        if margin_id not in self.patch_margins[patch_id]:
            self.patch_margins[patch_id].append(margin_id)
            self.logger.info(f"Assigned margin {margin_id} to patch {patch_id}")
            return True

        return False

    def update_patch_resources(
        self, patch: ResourcePatch, day_of_year: int, current_year: int
    ) -> None:
        """Update patch resources based on enhanced margins"""
        if patch.id not in self.patch_margins:
            return

        total_nectar_bonus = 0.0
        total_pollen_bonus = 0.0
        total_attractiveness_bonus = 0.0

        for margin_id in self.patch_margins[patch.id]:
            margin = self.active_margins[margin_id]

            # Check if margin is blooming
            if self._is_margin_blooming(margin, day_of_year):
                # Apply margin area fraction to patch (assume margin covers part of patch)
                area_fraction = min(
                    margin.target_area_hectares / getattr(patch, "area_ha", 1.0), 1.0
                )

                total_nectar_bonus += margin.nectar_production_rate * area_fraction
                total_pollen_bonus += margin.pollen_production_rate * area_fraction
                total_attractiveness_bonus += (
                    margin.bee_attractiveness_score * area_fraction
                )

            # Apply management effects
            self._apply_management_effects(margin, patch, day_of_year, current_year)

        # Update patch resources
        patch.base_nectar_production += total_nectar_bonus
        patch.base_pollen_production += total_pollen_bonus

        # Apply attractiveness as a multiplier (capped at 2.0x)
        attractiveness_multiplier = min(1.0 + total_attractiveness_bonus, 2.0)
        patch.base_nectar_production *= attractiveness_multiplier
        patch.base_pollen_production *= attractiveness_multiplier

    def _is_margin_blooming(
        self, margin: EnhancedMarginConfiguration, day_of_year: int
    ) -> bool:
        """Check if margin is currently blooming"""
        start_day = margin.bloom_start_day
        end_day = margin.bloom_end_day

        if start_day <= end_day:
            return start_day <= day_of_year <= end_day
        else:
            # Handle winter blooming across year boundary
            return day_of_year >= start_day or day_of_year <= end_day

    def _apply_management_effects(
        self,
        margin: EnhancedMarginConfiguration,
        patch: ResourcePatch,
        day_of_year: int,
        current_year: int,
    ) -> None:
        """Apply management operation effects to patch resources"""
        for schedule in margin.management_schedule:
            # Check if management operation should occur this year
            years_since_establishment = current_year - margin.establishment_year
            if years_since_establishment % schedule.frequency_years != 0:
                continue

            # Check if we're in the timing window
            start_day = schedule.timing_window_start
            end_day = schedule.timing_window_end

            in_window = False
            if start_day <= end_day:
                in_window = start_day <= day_of_year <= end_day
            else:
                in_window = day_of_year >= start_day or day_of_year <= end_day

            if not in_window:
                continue

            # Apply operation effects
            area_fraction = schedule.area_percentage / 100.0

            if schedule.operation == ManagementOperation.CUTTING:
                # Cutting reduces immediate resources but may improve long-term quality
                patch.base_nectar_production *= (
                    1.0 - 0.3 * area_fraction
                )  # 30% reduction
                patch.base_pollen_production *= (
                    1.0 - 0.4 * area_fraction
                )  # 40% reduction

            elif schedule.operation == ManagementOperation.TOPPING:
                # Topping has lighter impact than cutting
                patch.base_nectar_production *= (
                    1.0 - 0.15 * area_fraction
                )  # 15% reduction
                patch.base_pollen_production *= (
                    1.0 - 0.20 * area_fraction
                )  # 20% reduction

            elif schedule.operation == ManagementOperation.OVERSEEDING:
                # Overseeding improves resources (delayed effect - simplified here)
                patch.base_nectar_production *= (
                    1.0 + 0.1 * area_fraction
                )  # 10% improvement
                patch.base_pollen_production *= (
                    1.0 + 0.1 * area_fraction
                )  # 10% improvement

            elif schedule.operation == ManagementOperation.WEED_CONTROL:
                # Weed control improves quality by reducing competition
                patch.base_nectar_production *= (
                    1.0 + 0.05 * area_fraction
                )  # 5% improvement
                patch.base_pollen_production *= (
                    1.0 + 0.05 * area_fraction
                )  # 5% improvement

    def calculate_total_css_payments(self, year: int) -> Dict[str, float]:
        """Calculate total CSS payments for all active margins"""
        payments = {}
        total_payment = 0.0

        for margin_id, margin in self.active_margins.items():
            # Only pay for margins that have been established
            if year >= margin.establishment_year:
                annual_payment = margin.calculate_annual_payment()
                payments[margin_id] = annual_payment
                total_payment += annual_payment

        payments["total"] = total_payment
        return payments

    def get_margin_summary(self) -> Dict[str, Any]:
        """Get summary statistics for enhanced margins"""
        if not self.active_margins:
            return {"status": "no_margins"}

        total_area = sum(m.target_area_hectares for m in self.active_margins.values())
        total_payments = sum(
            m.calculate_annual_payment() for m in self.active_margins.values()
        )

        width_breakdown = {width.value: 0.0 for width in MarginWidth}
        species_breakdown = {mix.value: 0.0 for mix in SpeciesMix}

        for margin in self.active_margins.values():
            width_breakdown[margin.width.value] += margin.target_area_hectares
            species_breakdown[margin.species_mix.value] += margin.target_area_hectares

        return {
            "total_margins": len(self.active_margins),
            "total_area_hectares": total_area,
            "total_annual_payments": total_payments,
            "average_payment_per_hectare": total_payments / total_area
            if total_area > 0
            else 0,
            "width_breakdown_hectares": width_breakdown,
            "species_breakdown_hectares": species_breakdown,
            "patches_with_margins": len(self.patch_margins),
        }

    def get_margin_details(self, margin_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific margin"""
        if margin_id not in self.active_margins:
            return None

        margin = self.active_margins[margin_id]

        return {
            "margin_id": margin.margin_id,
            "width_meters": margin.width.value,
            "species_mix": margin.species_mix.value,
            "area_hectares": margin.target_area_hectares,
            "establishment_year": margin.establishment_year,
            "annual_payment": margin.calculate_annual_payment(),
            "payment_breakdown": {
                "base_payment": margin.base_payment_rate * margin.target_area_hectares,
                "width_bonus": (margin.width_multiplier - 1.0)
                * margin.base_payment_rate
                * margin.target_area_hectares,
                "species_bonus": margin.species_bonus * margin.target_area_hectares,
                "management_bonus": margin.management_bonus
                * margin.target_area_hectares,
            },
            "ecological_characteristics": {
                "nectar_production": margin.nectar_production_rate,
                "pollen_production": margin.pollen_production_rate,
                "bloom_period": f"Day {margin.bloom_start_day} - Day {margin.bloom_end_day}",
                "bee_attractiveness": margin.bee_attractiveness_score,
            },
            "management_schedule": [
                {
                    "operation": schedule.operation.value,
                    "timing": f"Day {schedule.timing_window_start} - Day {schedule.timing_window_end}",
                    "frequency": f"Every {schedule.frequency_years} year(s)",
                    "area_percentage": schedule.area_percentage,
                    "restrictions": schedule.restrictions,
                }
                for schedule in margin.management_schedule
            ],
        }

    def validate_margin_compliance(
        self, margin_id: str, current_year: int
    ) -> Dict[str, Any]:
        """Validate margin compliance with CSS requirements"""
        if margin_id not in self.active_margins:
            return {"valid": False, "error": "Margin not found"}

        margin = self.active_margins[margin_id]
        compliance_issues = []
        warnings = []

        # Check establishment period
        years_established = current_year - margin.establishment_year
        if years_established < 0:
            compliance_issues.append("Margin establishment year is in the future")
        elif years_established < 2:
            warnings.append(
                "Margin recently established - monitor establishment success"
            )

        # Check area requirements
        if margin.target_area_hectares < 0.1:
            warnings.append(
                "Very small margin area - may not provide significant benefits"
            )
        elif margin.target_area_hectares > 10.0:
            warnings.append("Very large margin area - verify area calculation")

        # Check management compliance
        for schedule in margin.management_schedule:
            if schedule.frequency_years > 5:
                warnings.append(
                    f"Infrequent {schedule.operation.value} may reduce effectiveness"
                )

        return {
            "valid": len(compliance_issues) == 0,
            "compliance_issues": compliance_issues,
            "warnings": warnings,
            "years_established": years_established,
            "payment_eligible": years_established >= 0,
        }
