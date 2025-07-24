"""
Tests for Enhanced Margins System
=================================

Comprehensive test suite for CSS Enhanced Margins functionality.
"""

import pytest

from src.bstew.components.css_enhanced_margins import (
    EnhancedMarginSystem,
    EnhancedMarginConfiguration,
    MarginWidth,
    SpeciesMix,
    ManagementOperation,
    ManagementSchedule
)
from src.bstew.spatial.patches import ResourcePatch, HabitatType


class TestMarginWidth:
    """Test MarginWidth enum"""

    def test_margin_width_values(self):
        """Test margin width enum values"""
        assert MarginWidth.NARROW.value == 6
        assert MarginWidth.MEDIUM.value == 12
        assert MarginWidth.WIDE.value == 24


class TestSpeciesMix:
    """Test SpeciesMix enum"""

    def test_species_mix_values(self):
        """Test species mix enum values"""
        assert SpeciesMix.WILDFLOWER_GRASSLAND.value == "wildflower_grassland"
        assert SpeciesMix.POLLINATOR_MIX.value == "pollinator_mix"
        assert SpeciesMix.NECTAR_RICH.value == "nectar_rich"
        assert SpeciesMix.LEGUME_RICH.value == "legume_rich"
        assert SpeciesMix.BIRD_FOOD.value == "bird_food"
        assert SpeciesMix.TUSSOCKY_GRASS.value == "tussocky_grass"


class TestManagementSchedule:
    """Test ManagementSchedule dataclass"""

    def test_management_schedule_creation(self):
        """Test basic management schedule creation"""
        schedule = ManagementSchedule(
            operation=ManagementOperation.CUTTING,
            timing_window_start=274,
            timing_window_end=334,
            frequency_years=1,
            area_percentage=100.0
        )

        assert schedule.operation == ManagementOperation.CUTTING
        assert schedule.timing_window_start == 274
        assert schedule.timing_window_end == 334
        assert schedule.frequency_years == 1
        assert schedule.area_percentage == 100.0
        assert schedule.restrictions == []

    def test_management_schedule_with_restrictions(self):
        """Test management schedule with restrictions"""
        restrictions = ["no_cutting_march_july", "leave_15cm_stubble"]
        schedule = ManagementSchedule(
            operation=ManagementOperation.CUTTING,
            timing_window_start=274,
            timing_window_end=334,
            restrictions=restrictions
        )

        assert schedule.restrictions == restrictions


class TestEnhancedMarginConfiguration:
    """Test EnhancedMarginConfiguration model"""

    def test_margin_configuration_creation(self):
        """Test basic margin configuration creation"""
        config = EnhancedMarginConfiguration(
            margin_id="test_margin_1",
            width=MarginWidth.MEDIUM,
            species_mix=SpeciesMix.POLLINATOR_MIX,
            establishment_year=2024,
            target_area_hectares=2.5,
            bloom_start_day=100,
            bloom_end_day=200,
            bee_attractiveness_score=0.8
        )

        assert config.margin_id == "test_margin_1"
        assert config.width == MarginWidth.MEDIUM
        assert config.species_mix == SpeciesMix.POLLINATOR_MIX
        assert config.establishment_year == 2024
        assert config.target_area_hectares == 2.5
        assert config.base_payment_rate == 640.0  # Default

    def test_width_multiplier_validation(self):
        """Test width multiplier is set based on width"""
        narrow_config = EnhancedMarginConfiguration(
            margin_id="narrow",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            establishment_year=2024,
            target_area_hectares=1.0,
            bloom_start_day=105,
            bloom_end_day=273,
            bee_attractiveness_score=0.85
        )

        medium_config = EnhancedMarginConfiguration(
            margin_id="medium",
            width=MarginWidth.MEDIUM,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            establishment_year=2024,
            target_area_hectares=1.0,
            bloom_start_day=105,
            bloom_end_day=273,
            bee_attractiveness_score=0.85
        )

        wide_config = EnhancedMarginConfiguration(
            margin_id="wide",
            width=MarginWidth.WIDE,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            establishment_year=2024,
            target_area_hectares=1.0,
            bloom_start_day=105,
            bloom_end_day=273,
            bee_attractiveness_score=0.85
        )

        assert narrow_config.width_multiplier == 1.0
        assert medium_config.width_multiplier == 1.2
        assert wide_config.width_multiplier == 1.5

    def test_calculate_annual_payment_basic(self):
        """Test basic payment calculation"""
        config = EnhancedMarginConfiguration(
            margin_id="payment_test",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            establishment_year=2024,
            target_area_hectares=2.0,
            base_payment_rate=640.0,
            species_bonus=50.0,
            management_bonus=0.0,
            bloom_start_day=105,
            bloom_end_day=273,
            bee_attractiveness_score=0.85
        )

        # 2.0 ha * £640/ha + 0 width bonus + 2.0 ha * £50/ha species bonus + 0 management bonus
        expected = 2.0 * 640.0 + 2.0 * 50.0
        assert config.calculate_annual_payment() == expected

    def test_calculate_annual_payment_with_multipliers(self):
        """Test payment calculation with width and bonuses"""
        config = EnhancedMarginConfiguration(
            margin_id="complex_payment",
            width=MarginWidth.WIDE,
            species_mix=SpeciesMix.NECTAR_RICH,
            establishment_year=2024,
            target_area_hectares=1.5,
            base_payment_rate=640.0,
            width_multiplier=1.5,
            species_bonus=85.0,
            management_bonus=50.0,
            bloom_start_day=114,
            bloom_end_day=258,
            bee_attractiveness_score=0.95
        )

        # Base: 1.5 * 640 = 960
        # Width adjustment: 960 * (1.5 - 1.0) = 480
        # Species bonus: 1.5 * 85 = 127.5
        # Management bonus: 1.5 * 50 = 75
        # Total: 960 + 480 + 127.5 + 75 = 1642.5
        expected = 1642.5
        assert config.calculate_annual_payment() == expected


class TestEnhancedMarginSystem:
    """Test EnhancedMarginSystem class"""

    @pytest.fixture
    def margin_system(self):
        """Enhanced margin system instance for testing"""
        return EnhancedMarginSystem()

    def test_initialization(self, margin_system):
        """Test system initialization"""
        assert margin_system.active_margins == {}
        assert margin_system.patch_margins == {}
        assert len(margin_system.species_mix_data) == 6  # All species mixes
        assert len(margin_system.management_templates) == 6
        assert "base_enhanced_margins" in margin_system.css_payment_rates

    def test_species_mix_data_complete(self, margin_system):
        """Test all species mixes have required data"""
        required_keys = [
            "nectar_production", "pollen_production", "bloom_start", "bloom_end",
            "bee_attractiveness", "establishment_cost", "species_bonus", "target_species"
        ]

        for species_mix in SpeciesMix:
            mix_data = margin_system.species_mix_data[species_mix]
            for key in required_keys:
                assert key in mix_data, f"Missing {key} for {species_mix.value}"

            # Check data types and ranges
            assert isinstance(mix_data["nectar_production"], (int, float))
            assert mix_data["nectar_production"] >= 0
            assert isinstance(mix_data["pollen_production"], (int, float))
            assert mix_data["pollen_production"] >= 0
            assert 1 <= mix_data["bloom_start"] <= 365
            assert 1 <= mix_data["bloom_end"] <= 365
            assert 0 <= mix_data["bee_attractiveness"] <= 1
            assert isinstance(mix_data["target_species"], list)
            assert len(mix_data["target_species"]) > 0

    def test_management_templates_complete(self, margin_system):
        """Test all species mixes have management templates"""
        for species_mix in SpeciesMix:
            templates = margin_system.management_templates[species_mix]
            assert isinstance(templates, list)

            for schedule in templates:
                assert isinstance(schedule, ManagementSchedule)
                assert 1 <= schedule.timing_window_start <= 365
                assert 1 <= schedule.timing_window_end <= 365
                assert schedule.frequency_years >= 1
                assert 0 <= schedule.area_percentage <= 100

    def test_create_enhanced_margin(self, margin_system):
        """Test creating enhanced margin"""
        margin_config = margin_system.create_enhanced_margin(
            margin_id="test_margin",
            width=MarginWidth.MEDIUM,
            species_mix=SpeciesMix.POLLINATOR_MIX,
            target_area_hectares=3.0,
            establishment_year=2024
        )

        assert margin_config.margin_id == "test_margin"
        assert margin_config.width == MarginWidth.MEDIUM
        assert margin_config.species_mix == SpeciesMix.POLLINATOR_MIX
        assert margin_config.target_area_hectares == 3.0
        assert margin_config.establishment_year == 2024

        # Check margin is stored in system
        assert "test_margin" in margin_system.active_margins
        assert margin_system.active_margins["test_margin"] == margin_config

        # Check ecological parameters are set from species mix
        pollinator_data = margin_system.species_mix_data[SpeciesMix.POLLINATOR_MIX]
        assert margin_config.nectar_production_rate == pollinator_data["nectar_production"]
        assert margin_config.pollen_production_rate == pollinator_data["pollen_production"]
        assert margin_config.bloom_start_day == pollinator_data["bloom_start"]
        assert margin_config.bloom_end_day == pollinator_data["bloom_end"]
        assert margin_config.bee_attractiveness_score == pollinator_data["bee_attractiveness"]

    def test_assign_margin_to_patch(self, margin_system):
        """Test assigning margin to patch"""
        # Create margin first
        margin_system.create_enhanced_margin(
            margin_id="patch_test_margin",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            target_area_hectares=1.0,
            establishment_year=2024
        )

        # Assign to patch
        result = margin_system.assign_margin_to_patch(
            patch_id=100,
            margin_id="patch_test_margin"
        )

        assert result is True
        assert 100 in margin_system.patch_margins
        assert "patch_test_margin" in margin_system.patch_margins[100]

        # Test assigning same margin again (should not duplicate)
        result2 = margin_system.assign_margin_to_patch(100, "patch_test_margin")
        assert result2 is False  # Already assigned
        assert len(margin_system.patch_margins[100]) == 1

        # Test assigning non-existent margin
        result3 = margin_system.assign_margin_to_patch(100, "non_existent")
        assert result3 is False

    def test_is_margin_blooming(self, margin_system):
        """Test margin blooming check"""
        config = EnhancedMarginConfiguration(
            margin_id="bloom_test",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.POLLINATOR_MIX,
            establishment_year=2024,
            target_area_hectares=1.0,
            bloom_start_day=100,  # Mid April
            bloom_end_day=200,    # Mid July
            bee_attractiveness_score=0.92
        )

        # Test blooming period
        assert margin_system._is_margin_blooming(config, 150) is True   # Mid bloom
        assert margin_system._is_margin_blooming(config, 100) is True   # Start
        assert margin_system._is_margin_blooming(config, 200) is True   # End
        assert margin_system._is_margin_blooming(config, 50) is False   # Before
        assert margin_system._is_margin_blooming(config, 250) is False  # After

        # Test winter blooming (across year boundary)
        winter_config = EnhancedMarginConfiguration(
            margin_id="winter_bloom",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.TUSSOCKY_GRASS,
            establishment_year=2024,
            target_area_hectares=1.0,
            bloom_start_day=300,  # Late October
            bloom_end_day=60,     # Early March
            bee_attractiveness_score=0.25
        )

        assert margin_system._is_margin_blooming(winter_config, 330) is True  # Winter
        assert margin_system._is_margin_blooming(winter_config, 30) is True   # Early year
        assert margin_system._is_margin_blooming(winter_config, 150) is False # Summer

    def test_update_patch_resources(self, margin_system):
        """Test updating patch resources with margin effects"""
        # Create test patch
        patch = ResourcePatch(
            patch_id=200,
            x=10.0,
            y=10.0,
            habitat_type=HabitatType.CROPLAND
        )
        patch.area_ha = 5.0
        patch.base_nectar_production = 1.0
        patch.base_pollen_production = 1.0

        # Create and assign margin
        margin_system.create_enhanced_margin(
            margin_id="resource_test_margin",
            width=MarginWidth.MEDIUM,
            species_mix=SpeciesMix.NECTAR_RICH,
            target_area_hectares=2.0,  # 40% of patch area
            establishment_year=2024
        )

        margin_system.assign_margin_to_patch(200, "resource_test_margin")

        # Get nectar-rich characteristics
        nectar_rich_data = margin_system.species_mix_data[SpeciesMix.NECTAR_RICH]
        bloom_start = nectar_rich_data["bloom_start"]

        # Update resources during bloom period
        margin_system.update_patch_resources(patch, bloom_start + 10, 2024)

        # Check resources were increased
        area_fraction = 2.0 / 5.0  # margin_area / patch_area = 0.4
        expected_nectar_bonus = nectar_rich_data["nectar_production"] * area_fraction
        expected_pollen_bonus = nectar_rich_data["pollen_production"] * area_fraction
        expected_attractiveness = nectar_rich_data["bee_attractiveness"] * area_fraction
        attractiveness_multiplier = min(1.0 + expected_attractiveness, 2.0)

        # Base (1.0) + bonus, then multiplied by attractiveness
        expected_nectar = (1.0 + expected_nectar_bonus) * attractiveness_multiplier
        expected_pollen = (1.0 + expected_pollen_bonus) * attractiveness_multiplier

        assert patch.base_nectar_production == pytest.approx(expected_nectar, rel=1e-3)
        assert patch.base_pollen_production == pytest.approx(expected_pollen, rel=1e-3)

    def test_update_patch_resources_no_bloom(self, margin_system):
        """Test patch resource update when margin not blooming"""
        # Create test patch
        patch = ResourcePatch(
            patch_id=201,
            x=10.0,
            y=10.0,
            habitat_type=HabitatType.CROPLAND
        )
        patch.area_ha = 3.0
        patch.base_nectar_production = 2.0
        patch.base_pollen_production = 2.0

        # Create margin with specific bloom period
        margin_system.create_enhanced_margin(
            margin_id="no_bloom_margin",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            target_area_hectares=1.0,
            establishment_year=2024
        )

        margin_system.assign_margin_to_patch(201, "no_bloom_margin")

        # Update resources outside bloom period
        wildflower_data = margin_system.species_mix_data[SpeciesMix.WILDFLOWER_GRASSLAND]
        non_bloom_day = wildflower_data["bloom_start"] - 50  # Well before bloom

        margin_system.update_patch_resources(patch, non_bloom_day, 2024)

        # Resources should remain unchanged (no blooming bonus)
        assert patch.base_nectar_production == 2.0
        assert patch.base_pollen_production == 2.0

    def test_calculate_total_css_payments(self, margin_system):
        """Test total CSS payment calculation"""
        # Create multiple margins
        margin_system.create_enhanced_margin(
            margin_id="payment_margin_1",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            target_area_hectares=2.0,
            establishment_year=2023
        )

        margin_system.create_enhanced_margin(
            margin_id="payment_margin_2",
            width=MarginWidth.WIDE,
            species_mix=SpeciesMix.NECTAR_RICH,
            target_area_hectares=1.5,
            establishment_year=2024
        )

        # Calculate payments for 2024
        payments = margin_system.calculate_total_css_payments(2024)

        assert "payment_margin_1" in payments
        assert "payment_margin_2" in payments
        assert "total" in payments

        # Check individual payments match margin calculations
        margin1 = margin_system.active_margins["payment_margin_1"]
        margin2 = margin_system.active_margins["payment_margin_2"]

        assert payments["payment_margin_1"] == margin1.calculate_annual_payment()
        assert payments["payment_margin_2"] == margin2.calculate_annual_payment()
        assert payments["total"] == payments["payment_margin_1"] + payments["payment_margin_2"]

    def test_calculate_total_css_payments_not_established(self, margin_system):
        """Test payments for margins not yet established"""
        # Create margin established in future
        margin_system.create_enhanced_margin(
            margin_id="future_margin",
            width=MarginWidth.MEDIUM,
            species_mix=SpeciesMix.POLLINATOR_MIX,
            target_area_hectares=1.0,
            establishment_year=2025
        )

        # Calculate payments for 2024 (before establishment)
        payments = margin_system.calculate_total_css_payments(2024)

        # Should not include future margin
        assert "future_margin" not in payments
        assert payments["total"] == 0.0

    def test_get_margin_summary(self, margin_system):
        """Test margin summary statistics"""
        # Test empty system
        summary = margin_system.get_margin_summary()
        assert summary["status"] == "no_margins"

        # Add margins
        margin_system.create_enhanced_margin(
            margin_id="summary_margin_1",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            target_area_hectares=1.5,
            establishment_year=2024
        )

        margin_system.create_enhanced_margin(
            margin_id="summary_margin_2",
            width=MarginWidth.WIDE,
            species_mix=SpeciesMix.POLLINATOR_MIX,
            target_area_hectares=2.5,
            establishment_year=2024
        )

        margin_system.assign_margin_to_patch(300, "summary_margin_1")

        summary = margin_system.get_margin_summary()

        assert summary["total_margins"] == 2
        assert summary["total_area_hectares"] == 4.0  # 1.5 + 2.5
        assert summary["total_annual_payments"] > 0
        assert summary["width_breakdown_hectares"][6] == 1.5   # Narrow margin
        assert summary["width_breakdown_hectares"][24] == 2.5  # Wide margin
        assert summary["species_breakdown_hectares"]["wildflower_grassland"] == 1.5
        assert summary["species_breakdown_hectares"]["pollinator_mix"] == 2.5
        assert summary["patches_with_margins"] == 1

    def test_get_margin_details(self, margin_system):
        """Test getting detailed margin information"""
        margin_system.create_enhanced_margin(
            margin_id="detail_test_margin",
            width=MarginWidth.MEDIUM,
            species_mix=SpeciesMix.NECTAR_RICH,
            target_area_hectares=3.0,
            establishment_year=2024
        )

        details = margin_system.get_margin_details("detail_test_margin")

        assert details is not None
        assert details["margin_id"] == "detail_test_margin"
        assert details["width_meters"] == 12
        assert details["species_mix"] == "nectar_rich"
        assert details["area_hectares"] == 3.0
        assert details["establishment_year"] == 2024
        assert details["annual_payment"] > 0

        # Check payment breakdown structure
        assert "payment_breakdown" in details
        breakdown = details["payment_breakdown"]
        assert "base_payment" in breakdown
        assert "width_bonus" in breakdown
        assert "species_bonus" in breakdown
        assert "management_bonus" in breakdown

        # Check ecological characteristics
        assert "ecological_characteristics" in details
        eco = details["ecological_characteristics"]
        assert "nectar_production" in eco
        assert "pollen_production" in eco
        assert "bloom_period" in eco
        assert "bee_attractiveness" in eco

        # Check management schedule
        assert "management_schedule" in details
        assert isinstance(details["management_schedule"], list)

        # Test non-existent margin
        assert margin_system.get_margin_details("non_existent") is None

    def test_validate_margin_compliance(self, margin_system):
        """Test margin compliance validation"""
        margin_system.create_enhanced_margin(
            margin_id="compliance_test",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            target_area_hectares=2.0,
            establishment_year=2022
        )

        # Test validation for current year (2024)
        validation = margin_system.validate_margin_compliance("compliance_test", 2024)

        assert validation["valid"] is True
        assert validation["years_established"] == 2  # 2024 - 2022
        assert validation["payment_eligible"] is True
        assert isinstance(validation["compliance_issues"], list)
        assert isinstance(validation["warnings"], list)

        # Test future establishment year
        margin_system.create_enhanced_margin(
            margin_id="future_compliance_test",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            target_area_hectares=2.0,
            establishment_year=2026  # Future year
        )

        validation_future = margin_system.validate_margin_compliance("future_compliance_test", 2024)
        assert validation_future["valid"] is False
        assert "establishment year is in the future" in validation_future["compliance_issues"][0]

        # Test non-existent margin
        validation_missing = margin_system.validate_margin_compliance("non_existent", 2024)
        assert validation_missing["valid"] is False
        assert validation_missing["error"] == "Margin not found"


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""

    @pytest.fixture
    def configured_system(self):
        """System with multiple configured margins"""
        system = EnhancedMarginSystem()

        # Add various margins representing farm scenario
        system.create_enhanced_margin(
            margin_id="field1_north_margin",
            width=MarginWidth.NARROW,
            species_mix=SpeciesMix.WILDFLOWER_GRASSLAND,
            target_area_hectares=0.8,
            establishment_year=2023
        )

        system.create_enhanced_margin(
            margin_id="field2_pollinator_strip",
            width=MarginWidth.MEDIUM,
            species_mix=SpeciesMix.POLLINATOR_MIX,
            target_area_hectares=1.5,
            establishment_year=2024
        )

        system.create_enhanced_margin(
            margin_id="field3_nectar_margin",
            width=MarginWidth.WIDE,
            species_mix=SpeciesMix.NECTAR_RICH,
            target_area_hectares=2.2,
            establishment_year=2024
        )

        return system

    def test_farm_scale_scenario(self, configured_system):
        """Test realistic farm-scale margin implementation"""
        # Assign margins to patches (fields)
        configured_system.assign_margin_to_patch(1001, "field1_north_margin")
        configured_system.assign_margin_to_patch(1002, "field2_pollinator_strip")
        configured_system.assign_margin_to_patch(1003, "field3_nectar_margin")

        # Check total area and payments
        summary = configured_system.get_margin_summary()
        assert summary["total_area_hectares"] == 4.5  # 0.8 + 1.5 + 2.2
        assert summary["total_margins"] == 3
        assert summary["patches_with_margins"] == 3

        # Check payment calculations
        payments = configured_system.calculate_total_css_payments(2024)
        assert payments["total"] > 2000  # Should be reasonable payment level

        # All margins should be eligible for payment in 2024
        assert len(payments) == 4  # 3 margins + total

    def test_seasonal_resource_dynamics(self, configured_system):
        """Test how margins affect resources throughout the year"""
        # Create test patches
        patches = []
        for i, (patch_id, x, area) in enumerate([(2001, 0, 10.0), (2002, 10, 8.0), (2003, 20, 12.0)]):
            patch = ResourcePatch(patch_id=patch_id, x=x, y=0, habitat_type=HabitatType.CROPLAND)
            patch.area_ha = area
            patch.base_nectar_production = 0.5
            patch.base_pollen_production = 0.5
            patches.append(patch)

        # Assign margins
        configured_system.assign_margin_to_patch(2001, "field1_north_margin")
        configured_system.assign_margin_to_patch(2002, "field2_pollinator_strip")
        configured_system.assign_margin_to_patch(2003, "field3_nectar_margin")

        # Test resource levels at different times of year
        test_days = [60, 120, 180, 240, 300]  # Spring through winter

        for day in test_days:
            for patch in patches:
                # Reset patch resources
                patch.base_nectar_production = 0.5
                patch.base_pollen_production = 0.5

                # Apply margin effects
                configured_system.update_patch_resources(patch, day, 2024)

                # Resources may be affected by management operations
                # Some reductions are expected during management periods
                assert patch.base_nectar_production >= 0.0
                assert patch.base_pollen_production >= 0.0

    def test_compliance_monitoring(self, configured_system):
        """Test compliance monitoring across multiple margins"""
        compliance_results = {}

        for margin_id in configured_system.active_margins.keys():
            compliance = configured_system.validate_margin_compliance(margin_id, 2024)
            compliance_results[margin_id] = compliance

        # All should be compliant
        for margin_id, result in compliance_results.items():
            assert result["valid"] is True, f"Margin {margin_id} should be compliant"
            assert result["payment_eligible"] is True

        # Check establishment periods are reasonable
        for result in compliance_results.values():
            assert result["years_established"] >= 0
            assert result["years_established"] <= 10  # Reasonable time frame
