"""
Agricultural Stewardship System for BSTEW
==========================================

Implements agricultural management practices including crop rotation, habitat
management, and conservation strategies to support bee populations.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict
import logging

from ..spatial.patches import ResourcePatch, HabitatType


class CropType(Enum):
    """Agricultural crop types"""

    CEREALS = "cereals"
    OILSEED_RAPE = "oilseed_rape"
    FIELD_BEANS = "field_beans"
    MAIZE = "maize"
    POTATOES = "potatoes"
    SUGAR_BEET = "sugar_beet"
    WHEAT = "wheat"
    BARLEY = "barley"
    FALLOW = "fallow"
    WILDFLOWER_STRIPS = "wildflower_strips"
    COVER_CROPS = "cover_crops"


class ManagementPractice(Enum):
    """Agricultural management practices"""

    CONVENTIONAL = "conventional"
    ORGANIC = "organic"
    INTEGRATED_PEST_MANAGEMENT = "integrated_pest_management"
    REDUCED_TILLAGE = "reduced_tillage"
    NO_TILL = "no_till"
    PRECISION_AGRICULTURE = "precision_agriculture"
    AGROECOLOGY = "agroecology"


class CropRotationStage(BaseModel):
    """Single stage in a crop rotation cycle"""

    model_config = {"validate_assignment": True}

    crop_type: CropType = Field(description="Type of crop in this stage")
    duration_years: int = Field(ge=1, description="Duration of this stage in years")
    bloom_start: int = Field(ge=1, le=365, description="Bloom start day of year")
    bloom_end: int = Field(ge=1, le=365, description="Bloom end day of year")
    nectar_production: float = Field(
        ge=0.0, description="Nectar production (mg per m² per day)"
    )
    pollen_production: float = Field(
        ge=0.0, description="Pollen production (mg per m² per day)"
    )
    bee_attractiveness: float = Field(
        ge=0.0, le=1.0, description="Bee attractiveness (0-1 scale)"
    )
    pesticide_use: bool = Field(
        default=False, description="Whether pesticides are used"
    )
    management_practice: ManagementPractice = Field(
        default=ManagementPractice.CONVENTIONAL, description="Management practice type"
    )

    def is_blooming(self, day_of_year: int) -> bool:
        """Check if crop is blooming on given day"""
        if self.bloom_start <= self.bloom_end:
            return self.bloom_start <= day_of_year <= self.bloom_end
        else:
            # Handle winter blooming (crosses year boundary)
            return day_of_year >= self.bloom_start or day_of_year <= self.bloom_end


class CropRotationPlan(BaseModel):
    """Multi-year crop rotation plan"""

    model_config = {"validate_assignment": True}

    name: str = Field(description="Name of the rotation plan")
    stages: List[CropRotationStage] = Field(description="List of crop rotation stages")
    cycle_length_years: int = Field(ge=1, description="Total cycle length in years")
    current_stage: int = Field(default=0, ge=0, description="Current stage index")
    years_in_current_stage: int = Field(
        default=0, ge=0, description="Years completed in current stage"
    )

    def get_current_crop(self) -> CropRotationStage:
        """Get current crop stage"""
        return self.stages[self.current_stage]

    def advance_year(self) -> None:
        """Advance rotation to next year"""
        self.years_in_current_stage += 1
        current_stage = self.stages[self.current_stage]

        if self.years_in_current_stage >= current_stage.duration_years:
            self.current_stage = (self.current_stage + 1) % len(self.stages)
            self.years_in_current_stage = 0


class StewardshipAction(BaseModel):
    """Conservation/stewardship action"""

    model_config = {"validate_assignment": True}

    name: str = Field(description="Name of the stewardship action")
    action_type: str = Field(description="Type of stewardship action")
    habitat_targets: List[HabitatType] = Field(description="Target habitat types")
    nectar_benefit: float = Field(
        default=0.0, ge=0.0, description="Nectar benefit per m² per day"
    )
    pollen_benefit: float = Field(
        default=0.0, ge=0.0, description="Pollen benefit per m² per day"
    )
    bloom_extension: int = Field(
        default=0, ge=0, description="Bloom period extension in days"
    )
    implementation_cost: float = Field(
        default=0.0, ge=0.0, description="Implementation cost"
    )
    duration_years: int = Field(default=5, ge=0, description="Action duration in years")
    bee_species_benefits: Dict[str, float] = Field(
        default_factory=dict, description="Species-specific benefits"
    )


class AgriculturalStewardshipSystem:
    """
    Manages agricultural stewardship practices and their impacts on bee resources.

    Implements:
    - Crop rotation schedules
    - Habitat management interventions
    - Conservation practices
    - Pesticide management
    - Seasonal resource planning
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Crop rotation plans
        self.rotation_plans: Dict[str, CropRotationPlan] = {}
        self.patch_rotations: Dict[int, str] = {}  # patch_id -> rotation_plan_name

        # Stewardship actions
        self.active_actions: Dict[int, List[StewardshipAction]] = defaultdict(list)
        self.action_history: List[
            Tuple[int, int, StewardshipAction]
        ] = []  # (year, patch_id, action)

        # Environmental impact tracking
        self.pesticide_applications: Dict[int, List[Tuple[int, str]]] = defaultdict(
            list
        )  # patch_id -> [(day, chemical)]
        self.habitat_quality_modifiers: Dict[int, float] = defaultdict(lambda: 1.0)

        # Initialize standard rotation plans
        self._initialize_rotation_plans()
        self._initialize_stewardship_actions()

    def _initialize_rotation_plans(self) -> None:
        """Initialize standard crop rotation plans"""

        # 4-year Norfolk rotation
        norfolk_stages = [
            CropRotationStage(
                crop_type=CropType.WHEAT,
                duration_years=1,
                bloom_start=1,
                bloom_end=1,  # No bloom
                nectar_production=0.0,
                pollen_production=0.0,
                bee_attractiveness=0.0,
                pesticide_use=True,
            ),
            CropRotationStage(
                crop_type=CropType.FIELD_BEANS,
                duration_years=1,
                bloom_start=153,
                bloom_end=182,  # June-early July
                nectar_production=0.86,
                pollen_production=0.65,
                bee_attractiveness=0.7,
                pesticide_use=False,
            ),
            CropRotationStage(
                crop_type=CropType.OILSEED_RAPE,
                duration_years=1,
                bloom_start=114,
                bloom_end=136,  # April-May
                nectar_production=10.0,
                pollen_production=1.51,
                bee_attractiveness=0.9,
                pesticide_use=True,
            ),
            CropRotationStage(
                crop_type=CropType.FALLOW,
                duration_years=1,
                bloom_start=120,
                bloom_end=273,  # May-September
                nectar_production=2.0,
                pollen_production=1.0,
                bee_attractiveness=0.6,
                pesticide_use=False,
            ),
        ]

        self.rotation_plans["norfolk_rotation"] = CropRotationPlan(
            name="Norfolk 4-Year Rotation", stages=norfolk_stages, cycle_length_years=4
        )

        # Bee-friendly rotation
        bee_friendly_stages = [
            CropRotationStage(
                crop_type=CropType.OILSEED_RAPE,
                duration_years=1,
                bloom_start=114,
                bloom_end=136,
                nectar_production=10.0,
                pollen_production=1.51,
                bee_attractiveness=0.9,
                pesticide_use=False,
                management_practice=ManagementPractice.ORGANIC,
            ),
            CropRotationStage(
                crop_type=CropType.FIELD_BEANS,
                duration_years=1,
                bloom_start=153,
                bloom_end=182,
                nectar_production=0.86,
                pollen_production=0.65,
                bee_attractiveness=0.7,
                pesticide_use=False,
                management_practice=ManagementPractice.ORGANIC,
            ),
            CropRotationStage(
                crop_type=CropType.WILDFLOWER_STRIPS,
                duration_years=2,
                bloom_start=91,
                bloom_end=304,  # April-October
                nectar_production=5.0,
                pollen_production=3.0,
                bee_attractiveness=0.95,
                pesticide_use=False,
                management_practice=ManagementPractice.ORGANIC,
            ),
        ]

        self.rotation_plans["bee_friendly_rotation"] = CropRotationPlan(
            name="Bee-Friendly 4-Year Rotation",
            stages=bee_friendly_stages,
            cycle_length_years=4,
        )

        # Intensive agriculture rotation
        intensive_stages = [
            CropRotationStage(
                crop_type=CropType.WHEAT,
                duration_years=1,
                bloom_start=1,
                bloom_end=1,
                nectar_production=0.0,
                pollen_production=0.0,
                bee_attractiveness=0.0,
                pesticide_use=True,
            ),
            CropRotationStage(
                crop_type=CropType.MAIZE,
                duration_years=1,
                bloom_start=197,
                bloom_end=210,
                nectar_production=0.0,
                pollen_production=30.0,
                bee_attractiveness=0.3,
                pesticide_use=True,
            ),
            CropRotationStage(
                crop_type=CropType.SUGAR_BEET,
                duration_years=1,
                bloom_start=1,
                bloom_end=1,
                nectar_production=0.0,
                pollen_production=0.0,
                bee_attractiveness=0.0,
                pesticide_use=True,
            ),
        ]

        self.rotation_plans["intensive_rotation"] = CropRotationPlan(
            name="Intensive 3-Year Rotation",
            stages=intensive_stages,
            cycle_length_years=3,
        )

    def _initialize_stewardship_actions(self) -> None:
        """Initialize available stewardship actions"""
        self.available_actions = [
            StewardshipAction(
                name="Wildflower Strip Creation",
                action_type="habitat_creation",
                habitat_targets=[HabitatType.CROPLAND],
                nectar_benefit=3.0,
                pollen_benefit=2.0,
                bloom_extension=120,
                implementation_cost=500.0,
                duration_years=5,
                bee_species_benefits={
                    "B_terrestris": 0.3,
                    "B_pascuorum": 0.4,
                    "B_lapidarius": 0.2,
                },
            ),
            StewardshipAction(
                name="Hedgerow Enhancement",
                action_type="habitat_improvement",
                habitat_targets=[HabitatType.HEDGEROW],
                nectar_benefit=2.0,
                pollen_benefit=1.5,
                bloom_extension=90,
                implementation_cost=300.0,
                duration_years=10,
                bee_species_benefits={
                    "B_hortorum": 0.4,
                    "B_pratorum": 0.3,
                    "B_hypnorum": 0.2,
                },
            ),
            StewardshipAction(
                name="Cover Crop Establishment",
                action_type="resource_enhancement",
                habitat_targets=[HabitatType.CROPLAND],
                nectar_benefit=1.0,
                pollen_benefit=0.8,
                bloom_extension=60,
                implementation_cost=150.0,
                duration_years=1,
                bee_species_benefits={"B_terrestris": 0.2, "B_pascuorum": 0.3},
            ),
            StewardshipAction(
                name="Pesticide-Free Zones",
                action_type="chemical_reduction",
                habitat_targets=[HabitatType.CROPLAND, HabitatType.GRASSLAND],
                nectar_benefit=0.5,
                pollen_benefit=0.3,
                bloom_extension=0,
                implementation_cost=200.0,
                duration_years=3,
                bee_species_benefits={
                    "B_terrestris": 0.15,
                    "B_pascuorum": 0.15,
                    "B_lapidarius": 0.1,
                },
            ),
            StewardshipAction(
                name="Late Summer Flowering",
                action_type="resource_timing",
                habitat_targets=[HabitatType.CROPLAND, HabitatType.GRASSLAND],
                nectar_benefit=2.5,
                pollen_benefit=1.5,
                bloom_extension=45,
                implementation_cost=100.0,
                duration_years=1,
                bee_species_benefits={
                    "B_terrestris": 0.25,
                    "B_pascuorum": 0.2,
                    "B_lapidarius": 0.15,
                },
            ),
        ]

    def assign_rotation_plan(self, patch_id: int, rotation_name: str) -> None:
        """Assign a crop rotation plan to a patch"""
        if rotation_name not in self.rotation_plans:
            self.logger.warning(f"Unknown rotation plan: {rotation_name}")
            raise KeyError(f"Unknown rotation plan: {rotation_name}")

        self.patch_rotations[patch_id] = rotation_name
        self.logger.info(f"Assigned {rotation_name} to patch {patch_id}")

    def implement_stewardship_action(
        self, patch_id: int, action_name: str, year: int
    ) -> None:
        """Implement a stewardship action on a patch"""
        action = next(
            (a for a in self.available_actions if a.name == action_name), None
        )
        if not action:
            self.logger.warning(f"Unknown stewardship action: {action_name}")
            return

        self.active_actions[patch_id].append(action)
        self.action_history.append((year, patch_id, action))
        self.logger.info(
            f"Implemented {action_name} on patch {patch_id} in year {year}"
        )

    def update_patch_resources(
        self, patch: ResourcePatch, day_of_year: int, year: int
    ) -> None:
        """Update patch resources based on current crop rotation and stewardship"""
        base_nectar = 0.0
        base_pollen = 0.0

        # Get current crop rotation stage
        if patch.id in self.patch_rotations:
            rotation_name = self.patch_rotations[patch.id]
            rotation = self.rotation_plans[rotation_name]
            current_crop = rotation.get_current_crop()

            # Apply crop-specific resources
            if current_crop.is_blooming(day_of_year):
                base_nectar = current_crop.nectar_production
                base_pollen = current_crop.pollen_production

                # Apply pesticide penalty
                if current_crop.pesticide_use:
                    self._apply_pesticide_effects(patch, day_of_year)
                    base_nectar *= 0.7
                    base_pollen *= 0.8

            else:
                # Apply pesticide effects even during non-bloom if pesticide use
                if current_crop.pesticide_use:
                    self._apply_pesticide_effects(patch, day_of_year)

        # Apply stewardship benefits (regardless of crop rotation)
        for action in self.active_actions[patch.id]:
            if patch.habitat_type in action.habitat_targets:
                base_nectar += action.nectar_benefit
                base_pollen += action.pollen_benefit

        # Update patch resources
        patch.base_nectar_production = base_nectar
        patch.base_pollen_production = base_pollen

        # Apply habitat quality modifiers
        quality_modifier = self.habitat_quality_modifiers.get(patch.id, 1.0)
        patch.base_nectar_production *= quality_modifier
        patch.base_pollen_production *= quality_modifier

    def advance_year(self) -> None:
        """Advance all crop rotations by one year"""
        for rotation in self.rotation_plans.values():
            rotation.advance_year()

        # Decrement duration for all active actions and remove expired ones
        for patch_id in list(self.active_actions.keys()):
            # Decrement duration for all actions first
            for action in self.active_actions[patch_id]:
                action.duration_years -= 1

            # Remove expired actions (duration <= 0)
            self.active_actions[patch_id] = [
                action
                for action in self.active_actions[patch_id]
                if action.duration_years > 0
            ]

        # Reset habitat quality modifiers (pesticide effects fade over time)
        self.habitat_quality_modifiers.clear()

        self.logger.info("Advanced crop rotations and stewardship actions by one year")

    def _apply_pesticide_effects(self, patch: ResourcePatch, day_of_year: int) -> None:
        """Apply pesticide effects to patch"""
        # Record pesticide application
        self.pesticide_applications[patch.id].append((day_of_year, "generic_pesticide"))

        # Set habitat quality to reduced level (don't multiply existing)
        self.habitat_quality_modifiers[patch.id] = 0.5

    def get_rotation_status(self, patch_id: int) -> Dict[str, Any]:
        """Get current rotation status for a patch"""
        if patch_id not in self.patch_rotations:
            return {"status": "no_rotation"}

        rotation_name = self.patch_rotations[patch_id]
        rotation = self.rotation_plans[rotation_name]
        current_crop = rotation.get_current_crop()

        return {
            "status": "active",
            "rotation_name": rotation_name,
            "current_crop": current_crop.crop_type.value,
            "years_in_stage": rotation.years_in_current_stage,
            "stage_duration": current_crop.duration_years,
            "pesticide_use": current_crop.pesticide_use,
            "management_practice": current_crop.management_practice.value,
        }

    def get_stewardship_summary(self) -> Dict[str, Any]:
        """Get summary of stewardship implementation"""
        total_actions = sum(len(actions) for actions in self.active_actions.values())
        patches_managed = len(
            [p for p in self.active_actions.keys() if self.active_actions[p]]
        )

        action_types: Dict[str, int] = defaultdict(int)
        for actions in self.active_actions.values():
            for action in actions:
                action_types[action.action_type] += 1

        return {
            "total_active_actions": total_actions,
            "patches_under_management": patches_managed,
            "action_types": dict(action_types),
            "rotation_plans_active": len(self.patch_rotations),
            "total_pesticide_applications": sum(
                len(apps) for apps in self.pesticide_applications.values()
            ),
        }

    def get_bee_species_benefits(self, patch_id: int) -> Dict[str, float]:
        """Get species-specific benefits from stewardship actions"""
        benefits: Dict[str, float] = defaultdict(float)

        for action in self.active_actions[patch_id]:
            for species, benefit in action.bee_species_benefits.items():
                benefits[species] += benefit

        return dict(benefits)

    def parse_netlogo_crop_rotation(
        self, crop_rotation_string: str
    ) -> Optional[CropRotationPlan]:
        """Parse NetLogo crop rotation string format"""
        if not crop_rotation_string or crop_rotation_string.strip() == "":
            return None

        # Format: "crop1:years1,crop2:years2,..."
        try:
            stages = []
            entries = crop_rotation_string.split(",")

            for entry in entries:
                if ":" in entry:
                    crop_name, years_str = entry.split(":")
                    crop_name = crop_name.strip()
                    years = int(years_str.strip())

                    # Map NetLogo crop names to our crop types
                    crop_type = self._map_netlogo_crop_name(crop_name)
                    if crop_type:
                        stage = self._create_crop_stage(crop_type, years)
                        stages.append(stage)

            if stages:
                return CropRotationPlan(
                    name=f"custom_rotation_{len(self.rotation_plans)}",
                    stages=stages,
                    cycle_length_years=sum(s.duration_years for s in stages),
                )

        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing crop rotation string: {e}")

        return None

    def _map_netlogo_crop_name(self, netlogo_name: str) -> Optional[CropType]:
        """Map NetLogo crop names to CropType enum"""
        name_mappings = {
            "cereals": CropType.CEREALS,
            "oilseed_rape": CropType.OILSEED_RAPE,
            "field_beans": CropType.FIELD_BEANS,
            "maize": CropType.MAIZE,
            "wheat": CropType.WHEAT,
            "barley": CropType.BARLEY,
            "fallow": CropType.FALLOW,
            "wildflower": CropType.WILDFLOWER_STRIPS,
            "cover_crop": CropType.COVER_CROPS,
        }

        return name_mappings.get(netlogo_name.lower())

    def _create_crop_stage(
        self, crop_type: CropType, duration_years: int
    ) -> CropRotationStage:
        """Create crop rotation stage with default values"""
        # Default crop characteristics
        crop_defaults = {
            CropType.CEREALS: (1, 1, 0.0, 0.0, 0.0, True),
            CropType.OILSEED_RAPE: (114, 136, 10.0, 1.51, 0.9, True),
            CropType.FIELD_BEANS: (153, 182, 0.86, 0.65, 0.7, False),
            CropType.MAIZE: (197, 210, 0.0, 30.0, 0.3, True),
            CropType.WHEAT: (1, 1, 0.0, 0.0, 0.0, True),
            CropType.BARLEY: (1, 1, 0.0, 0.0, 0.0, True),
            CropType.FALLOW: (120, 273, 2.0, 1.0, 0.6, False),
            CropType.WILDFLOWER_STRIPS: (91, 304, 5.0, 3.0, 0.95, False),
            CropType.COVER_CROPS: (120, 200, 1.0, 0.8, 0.4, False),
        }

        bloom_start, bloom_end, nectar, pollen, attractiveness, pesticide = (
            crop_defaults.get(crop_type, (1, 1, 0.0, 0.0, 0.0, False))
        )

        return CropRotationStage(
            crop_type=crop_type,
            duration_years=duration_years,
            bloom_start=bloom_start,
            bloom_end=bloom_end,
            nectar_production=nectar,
            pollen_production=pollen,
            bee_attractiveness=attractiveness,
            pesticide_use=pesticide,
        )
