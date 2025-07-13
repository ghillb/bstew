"""
Tests for Agricultural Stewardship System
=========================================

Tests crop rotation, stewardship actions, and agricultural management.
"""

import pytest

from src.bstew.components.stewardship import (
    AgriculturalStewardshipSystem,
    CropType,
    CropRotationPlan,
    CropRotationStage,
    StewardshipAction,
)
from src.bstew.spatial.patches import ResourcePatch, HabitatType


class TestAgriculturalStewardshipSystem:
    """Test agricultural stewardship system"""

    def setup_method(self):
        """Setup test environment"""
        self.stewardship = AgriculturalStewardshipSystem()
        self.test_patch = ResourcePatch(
            patch_id=1, x=100.0, y=100.0, habitat_type=HabitatType.CROPLAND
        )

    def test_initialization(self):
        """Test stewardship system initialization"""
        assert len(self.stewardship.rotation_plans) == 3
        assert "norfolk_rotation" in self.stewardship.rotation_plans
        assert "bee_friendly_rotation" in self.stewardship.rotation_plans
        assert "intensive_rotation" in self.stewardship.rotation_plans
        assert len(self.stewardship.available_actions) == 5

    def test_crop_rotation_stage(self):
        """Test crop rotation stage functionality"""
        stage = CropRotationStage(
            crop_type=CropType.OILSEED_RAPE,
            duration_years=1,
            bloom_start=114,
            bloom_end=136,
            nectar_production=10.0,
            pollen_production=1.5,
            bee_attractiveness=0.9,
        )

        assert stage.is_blooming(125)  # Mid-bloom
        assert not stage.is_blooming(100)  # Before bloom
        assert not stage.is_blooming(150)  # After bloom

        # Test winter blooming (crosses year boundary)
        winter_stage = CropRotationStage(
            crop_type=CropType.COVER_CROPS,
            duration_years=1,
            bloom_start=330,
            bloom_end=60,
            nectar_production=1.0,
            pollen_production=0.5,
            bee_attractiveness=0.4,
        )

        assert winter_stage.is_blooming(350)  # Winter bloom
        assert winter_stage.is_blooming(30)  # Early spring bloom
        assert not winter_stage.is_blooming(200)  # Summer

    def test_crop_rotation_plan(self):
        """Test crop rotation plan advancement"""
        stages = [
            CropRotationStage(
                crop_type=CropType.WHEAT,
                duration_years=1,
                bloom_start=1,
                bloom_end=1,
                nectar_production=0.0,
                pollen_production=0.0,
                bee_attractiveness=0.0,
            ),
            CropRotationStage(
                crop_type=CropType.FIELD_BEANS,
                duration_years=1,
                bloom_start=153,
                bloom_end=182,
                nectar_production=0.86,
                pollen_production=0.65,
                bee_attractiveness=0.7,
            ),
            CropRotationStage(
                crop_type=CropType.OILSEED_RAPE,
                duration_years=1,
                bloom_start=114,
                bloom_end=136,
                nectar_production=10.0,
                pollen_production=1.5,
                bee_attractiveness=0.9,
            ),
            CropRotationStage(
                crop_type=CropType.FALLOW,
                duration_years=1,
                bloom_start=120,
                bloom_end=273,
                nectar_production=2.0,
                pollen_production=1.0,
                bee_attractiveness=0.6,
            ),
        ]

        plan = CropRotationPlan(
            name="test_rotation", stages=stages, cycle_length_years=4
        )

        # Initial state
        assert plan.current_stage == 0
        assert plan.years_in_current_stage == 0
        assert plan.get_current_crop().crop_type == CropType.WHEAT

        # Advance one year
        plan.advance_year()
        assert plan.current_stage == 1
        assert plan.years_in_current_stage == 0
        assert plan.get_current_crop().crop_type == CropType.FIELD_BEANS

        # Advance through full cycle
        plan.advance_year()  # Oilseed rape
        plan.advance_year()  # Fallow
        plan.advance_year()  # Back to wheat

        assert plan.current_stage == 0
        assert plan.get_current_crop().crop_type == CropType.WHEAT

    def test_assign_rotation_plan(self):
        """Test assigning rotation plans to patches"""
        # Valid assignment
        self.stewardship.assign_rotation_plan(1, "norfolk_rotation")
        assert self.stewardship.patch_rotations[1] == "norfolk_rotation"

        # Invalid assignment
        with pytest.raises(KeyError):
            self.stewardship.assign_rotation_plan(2, "invalid_rotation")

    def test_implement_stewardship_action(self):
        """Test implementing stewardship actions"""
        # Valid action
        self.stewardship.implement_stewardship_action(
            1, "Wildflower Strip Creation", 2023
        )

        actions = self.stewardship.active_actions[1]
        assert len(actions) == 1
        assert actions[0].name == "Wildflower Strip Creation"
        assert actions[0].duration_years == 5

        # Check action history
        assert len(self.stewardship.action_history) == 1
        assert self.stewardship.action_history[0][0] == 2023
        assert self.stewardship.action_history[0][1] == 1

    def test_update_patch_resources(self):
        """Test updating patch resources based on rotation and stewardship"""
        # Assign rotation plan
        self.stewardship.assign_rotation_plan(1, "bee_friendly_rotation")

        # Update during oilseed rape bloom period
        self.stewardship.update_patch_resources(self.test_patch, 125, 2023)

        # Should have oilseed rape production
        assert self.test_patch.base_nectar_production == 10.0
        assert self.test_patch.base_pollen_production == 1.51

        # Update during non-bloom period
        self.stewardship.update_patch_resources(self.test_patch, 200, 2023)

        # Should have no production
        assert self.test_patch.base_nectar_production == 0.0
        assert self.test_patch.base_pollen_production == 0.0

    def test_stewardship_benefits(self):
        """Test stewardship action benefits"""
        # Implement wildflower strip
        self.stewardship.implement_stewardship_action(
            1, "Wildflower Strip Creation", 2023
        )
        self.stewardship.assign_rotation_plan(1, "intensive_rotation")

        # Update during non-crop bloom period
        self.stewardship.update_patch_resources(self.test_patch, 200, 2023)

        # Should have stewardship benefits (3.0 nectar * 0.5 pesticide penalty)
        assert self.test_patch.base_nectar_production == 1.5
        assert self.test_patch.base_pollen_production == 1.0

    def test_pesticide_effects(self):
        """Test pesticide application effects"""
        # Assign intensive rotation with pesticide use
        self.stewardship.assign_rotation_plan(1, "intensive_rotation")

        # Update during wheat (non-bloom but pesticide use)
        self.stewardship.update_patch_resources(self.test_patch, 100, 2023)

        # Check pesticide application was recorded
        assert len(self.stewardship.pesticide_applications[1]) > 0

        # Check habitat quality reduction
        assert self.stewardship.habitat_quality_modifiers[1] == 0.5

    def test_advance_year(self):
        """Test advancing years"""
        # Implement action with 2-year duration
        action = StewardshipAction(
            name="Test Action",
            action_type="test",
            habitat_targets=[HabitatType.CROPLAND],
            duration_years=2,
        )
        self.stewardship.active_actions[1].append(action)

        # Advance one year
        self.stewardship.advance_year()
        assert self.stewardship.active_actions[1][0].duration_years == 1

        # Advance second year (should expire after decrementing to 0)
        self.stewardship.advance_year()
        assert len(self.stewardship.active_actions[1]) == 0

    def test_rotation_status(self):
        """Test getting rotation status"""
        # No rotation
        status = self.stewardship.get_rotation_status(1)
        assert status["status"] == "no_rotation"

        # With rotation
        self.stewardship.assign_rotation_plan(1, "norfolk_rotation")
        status = self.stewardship.get_rotation_status(1)

        assert status["status"] == "active"
        assert status["rotation_name"] == "norfolk_rotation"
        assert status["current_crop"] == "wheat"
        assert status["years_in_stage"] == 0
        assert status["pesticide_use"]

    def test_stewardship_summary(self):
        """Test stewardship summary"""
        # Implement some actions
        self.stewardship.implement_stewardship_action(
            1, "Wildflower Strip Creation", 2023
        )
        self.stewardship.implement_stewardship_action(2, "Hedgerow Enhancement", 2023)
        self.stewardship.assign_rotation_plan(1, "norfolk_rotation")

        summary = self.stewardship.get_stewardship_summary()

        assert summary["total_active_actions"] == 2
        assert summary["patches_under_management"] == 2
        assert summary["rotation_plans_active"] == 1
        assert "habitat_creation" in summary["action_types"]
        assert "habitat_improvement" in summary["action_types"]

    def test_bee_species_benefits(self):
        """Test species-specific benefits"""
        # Implement action with species benefits
        self.stewardship.implement_stewardship_action(
            1, "Wildflower Strip Creation", 2023
        )

        benefits = self.stewardship.get_bee_species_benefits(1)

        assert benefits["B_terrestris"] == 0.3
        assert benefits["B_pascuorum"] == 0.4
        assert benefits["B_lapidarius"] == 0.2

    def test_netlogo_crop_rotation_parsing(self):
        """Test parsing NetLogo crop rotation strings"""
        # Valid rotation string
        rotation_str = "oilseed_rape:1,field_beans:1,wheat:2,fallow:1"
        plan = self.stewardship.parse_netlogo_crop_rotation(rotation_str)

        assert plan is not None
        assert len(plan.stages) == 4
        assert plan.stages[0].crop_type == CropType.OILSEED_RAPE
        assert plan.stages[0].duration_years == 1
        assert plan.stages[1].crop_type == CropType.FIELD_BEANS
        assert plan.stages[2].duration_years == 2
        assert plan.cycle_length_years == 5

        # Empty string
        plan = self.stewardship.parse_netlogo_crop_rotation("")
        assert plan is None

        # Invalid format
        plan = self.stewardship.parse_netlogo_crop_rotation("invalid_format")
        assert plan is None

    def test_crop_stage_creation(self):
        """Test creating crop stages with defaults"""
        stage = self.stewardship._create_crop_stage(CropType.OILSEED_RAPE, 1)

        assert stage.crop_type == CropType.OILSEED_RAPE
        assert stage.duration_years == 1
        assert stage.bloom_start == 114
        assert stage.bloom_end == 136
        assert stage.nectar_production == 10.0
        assert stage.pollen_production == 1.51
        assert stage.bee_attractiveness == 0.9
        assert stage.pesticide_use

    def test_netlogo_crop_name_mapping(self):
        """Test mapping NetLogo crop names"""
        assert (
            self.stewardship._map_netlogo_crop_name("oilseed_rape")
            == CropType.OILSEED_RAPE
        )
        assert (
            self.stewardship._map_netlogo_crop_name("field_beans")
            == CropType.FIELD_BEANS
        )
        assert (
            self.stewardship._map_netlogo_crop_name("wildflower")
            == CropType.WILDFLOWER_STRIPS
        )
        assert self.stewardship._map_netlogo_crop_name("unknown_crop") is None

    def test_multiple_stewardship_actions(self):
        """Test multiple stewardship actions on same patch"""
        # Implement multiple actions
        self.stewardship.implement_stewardship_action(
            1, "Wildflower Strip Creation", 2023
        )
        self.stewardship.implement_stewardship_action(1, "Pesticide-Free Zones", 2023)

        # Should have cumulative benefits
        benefits = self.stewardship.get_bee_species_benefits(1)
        assert benefits["B_terrestris"] == 0.3 + 0.15  # Both actions

        # Resource benefits should be additive
        self.stewardship.assign_rotation_plan(1, "intensive_rotation")
        self.stewardship.update_patch_resources(self.test_patch, 200, 2023)

        # Wildflower (3.0 nectar) + Pesticide-free (0.5 nectar) = 3.5, then * 0.5 pesticide penalty
        assert self.test_patch.base_nectar_production == 1.75
        assert self.test_patch.base_pollen_production == 1.15  # (2.0 + 0.3) * 0.5

    def test_habitat_quality_modifiers(self):
        """Test habitat quality modifiers"""
        # Set quality modifier
        self.stewardship.habitat_quality_modifiers[1] = 0.8

        # Assign rotation and stewardship
        self.stewardship.assign_rotation_plan(1, "bee_friendly_rotation")
        self.stewardship.implement_stewardship_action(
            1, "Wildflower Strip Creation", 2023
        )

        # Update during bloom period
        self.stewardship.update_patch_resources(self.test_patch, 125, 2023)

        # Resources should be modified by quality factor
        expected_nectar = (10.0 + 3.0) * 0.8  # (crop + stewardship) * modifier
        expected_pollen = (1.51 + 2.0) * 0.8

        assert abs(self.test_patch.base_nectar_production - expected_nectar) < 0.01
        assert abs(self.test_patch.base_pollen_production - expected_pollen) < 0.01


class TestCropRotationIntegration:
    """Test integration with existing systems"""

    def setup_method(self):
        """Setup test environment"""
        self.stewardship = AgriculturalStewardshipSystem()

    def test_seasonal_resource_pattern(self):
        """Test seasonal resource patterns from rotations"""
        # Create bee-friendly rotation
        self.stewardship.assign_rotation_plan(1, "bee_friendly_rotation")

        patch = ResourcePatch(1, 100.0, 100.0, HabitatType.CROPLAND)

        # Test resource availability throughout the year
        resources_by_day = {}
        for day in range(1, 366, 30):  # Sample every 30 days
            self.stewardship.update_patch_resources(patch, day, 2023)
            resources_by_day[day] = {
                "nectar": patch.base_nectar_production,
                "pollen": patch.base_pollen_production,
            }

        # Should have high resources during oilseed rape bloom (April-May)
        assert resources_by_day[121]["nectar"] > 5.0  # May

        # Should have lower resources in winter
        assert resources_by_day[31]["nectar"] == 0.0  # January

    def test_rotation_cycle_progression(self):
        """Test full rotation cycle progression"""
        # Create custom rotation
        custom_rotation = self.stewardship.parse_netlogo_crop_rotation(
            "oilseed_rape:1,fallow:1"
        )
        self.stewardship.rotation_plans["custom"] = custom_rotation
        self.stewardship.assign_rotation_plan(1, "custom")

        patch = ResourcePatch(1, 100.0, 100.0, HabitatType.CROPLAND)

        # Year 1: Oilseed rape
        self.stewardship.update_patch_resources(patch, 125, 2023)
        year1_nectar = patch.base_nectar_production

        # Advance to Year 2: Fallow
        self.stewardship.advance_year()
        self.stewardship.update_patch_resources(patch, 125, 2024)
        year2_nectar = patch.base_nectar_production

        # Advance to Year 3: Back to oilseed rape
        self.stewardship.advance_year()
        self.stewardship.update_patch_resources(patch, 125, 2025)
        year3_nectar = patch.base_nectar_production

        # Should cycle back to original crop
        assert year1_nectar == year3_nectar
        assert year1_nectar != year2_nectar

    def test_stewardship_action_duration(self):
        """Test stewardship action duration and expiration"""
        # Implement short-duration action
        action = StewardshipAction(
            name="Test Short Action",
            action_type="test",
            habitat_targets=[HabitatType.CROPLAND],
            nectar_benefit=5.0,
            duration_years=1,
        )
        self.stewardship.active_actions[1].append(action)

        patch = ResourcePatch(1, 100.0, 100.0, HabitatType.CROPLAND)

        # Year 1: Action active
        self.stewardship.update_patch_resources(patch, 200, 2023)
        year1_nectar = patch.base_nectar_production

        # Check that action is still active
        assert len(self.stewardship.active_actions[1]) == 1
        assert self.stewardship.active_actions[1][0].duration_years == 1

        # Advance year: Action expires
        self.stewardship.advance_year()

        # Check that action has expired
        assert len(self.stewardship.active_actions[1]) == 0

        self.stewardship.update_patch_resources(patch, 200, 2024)
        year2_nectar = patch.base_nectar_production

        # Should have benefit in year 1, not in year 2
        assert year1_nectar > year2_nectar
        assert year2_nectar == 0.0  # No base production without action
