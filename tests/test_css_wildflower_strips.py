"""
Tests for Wildflower Strips System
==================================

Comprehensive test suite for CSS Wildflower Strips functionality.
"""

import pytest
from datetime import date

from src.bstew.components.css_wildflower_strips import (
    WildflowerStripSystem,
    StripConfiguration,
    StripWidth,
    PlacementStrategy,
    SeedMixType,
    FloweringPeriod,
    SpeciesComposition,
    MaintenanceActivity
)
from src.bstew.spatial.patches import ResourcePatch, HabitatType


class TestStripWidth:
    """Test StripWidth enum"""

    def test_strip_width_values(self):
        """Test strip width enum values"""
        assert StripWidth.NARROW.value == 3
        assert StripWidth.STANDARD.value == 6
        assert StripWidth.WIDE.value == 12
        assert StripWidth.EXTRA_WIDE.value == 20


class TestPlacementStrategy:
    """Test PlacementStrategy enum"""

    def test_placement_strategy_values(self):
        """Test placement strategy enum values"""
        assert PlacementStrategy.FIELD_EDGE.value == "field_edge"
        assert PlacementStrategy.IN_FIELD.value == "in_field"
        assert PlacementStrategy.WATERCOURSE.value == "watercourse"
        assert PlacementStrategy.HEDGEROW.value == "hedgerow"
        assert PlacementStrategy.CONNECTIVITY.value == "connectivity"
        assert PlacementStrategy.SLOPE_CONTOUR.value == "slope_contour"


class TestSpeciesComposition:
    """Test SpeciesComposition dataclass"""

    def test_species_composition_creation(self):
        """Test creating species composition"""
        species = SpeciesComposition(
            scientific_name="Centaurea nigra",
            common_name="Common Knapweed",
            percentage=15.0,
            flowering_start_day=182,
            flowering_end_day=274,
            flowering_period=FloweringPeriod.LATE_SUMMER,
            nectar_rating=9.0,
            pollen_rating=8.0,
            height_cm=80,
            establishment_rate=0.80,
            persistence_years=10
        )

        assert species.scientific_name == "Centaurea nigra"
        assert species.percentage == 15.0
        assert species.nectar_rating == 9.0
        assert species.establishment_rate == 0.80


class TestMaintenanceActivity:
    """Test MaintenanceActivity dataclass"""

    def test_maintenance_activity_creation(self):
        """Test creating maintenance activity"""
        activity = MaintenanceActivity(
            activity_type="autumn_cut",
            timing_start_day=274,
            timing_end_day=334,
            frequency_years=1.0,
            equipment="flail_mower",
            height_cm=10,
            notes=["Remove cuttings", "After seed set"]
        )

        assert activity.activity_type == "autumn_cut"
        assert activity.frequency_years == 1.0
        assert activity.height_cm == 10
        assert len(activity.notes) == 2


class TestStripConfiguration:
    """Test StripConfiguration model"""

    def test_strip_configuration_creation(self):
        """Test basic strip configuration"""
        species = [
            SpeciesComposition(
                "Leucanthemum vulgare", "Oxeye Daisy", 50.0,
                152, 244, FloweringPeriod.MID_SUMMER,
                7.0, 6.0, 60, 0.85, 5
            ),
            SpeciesComposition(
                "Centaurea nigra", "Common Knapweed", 50.0,
                182, 274, FloweringPeriod.LATE_SUMMER,
                9.0, 8.0, 80, 0.80, 10
            )
        ]

        config = StripConfiguration(
            strip_id="test_strip_1",
            width=StripWidth.STANDARD,
            length_meters=200.0,
            placement=PlacementStrategy.FIELD_EDGE,
            seed_mix_type=SeedMixType.GENERAL_WILDFLOWER,
            establishment_year=2024,
            start_x=0.0,
            start_y=0.0,
            end_x=200.0,
            end_y=0.0,
            species_composition=species
        )

        assert config.strip_id == "test_strip_1"
        assert config.width == StripWidth.STANDARD
        assert config.length_meters == 200.0
        assert len(config.species_composition) == 2

    def test_area_calculation(self):
        """Test strip area calculation"""
        config = StripConfiguration(
            strip_id="area_test",
            width=StripWidth.STANDARD,  # 6m
            length_meters=1000.0,
            placement=PlacementStrategy.FIELD_EDGE,
            seed_mix_type=SeedMixType.GENERAL_WILDFLOWER,
            establishment_year=2024,
            start_x=0.0,
            start_y=0.0,
            end_x=1000.0,
            end_y=0.0
        )

        # 6m * 1000m = 6000m² = 0.6ha
        assert config.area_hectares == 0.6

    def test_annual_payment_calculation(self):
        """Test annual payment calculation"""
        config = StripConfiguration(
            strip_id="payment_test",
            width=StripWidth.WIDE,  # 12m
            length_meters=500.0,
            placement=PlacementStrategy.FIELD_EDGE,
            seed_mix_type=SeedMixType.POLLINATOR_NECTAR,
            establishment_year=2024,
            start_x=0.0,
            start_y=0.0,
            end_x=500.0,
            end_y=0.0,
            annual_payment_per_ha=450.0,
            management_payment_per_ha=100.0
        )

        # Area: 12m * 500m = 6000m² = 0.6ha
        # Payment: (450 + 100) * 0.6 = 330
        assert config.annual_payment == 330.0

    def test_species_percentage_validation(self):
        """Test species percentage validation"""
        species = [
            SpeciesComposition(
                "Species1", "Common 1", 60.0,
                100, 200, FloweringPeriod.MID_SUMMER,
                5.0, 5.0, 50, 0.8, 5
            ),
            SpeciesComposition(
                "Species2", "Common 2", 30.0,
                150, 250, FloweringPeriod.LATE_SUMMER,
                6.0, 6.0, 60, 0.8, 5
            )
        ]

        # Total: 90%, should fail validation
        with pytest.raises(ValueError, match="Species percentages must sum to 100%"):
            StripConfiguration(
                strip_id="invalid_percentage",
                width=StripWidth.STANDARD,
                length_meters=100.0,
                placement=PlacementStrategy.FIELD_EDGE,
                seed_mix_type=SeedMixType.GENERAL_WILDFLOWER,
                establishment_year=2024,
                start_x=0.0,
                start_y=0.0,
                end_x=100.0,
                end_y=0.0,
                species_composition=species
            )


class TestWildflowerStripSystem:
    """Test WildflowerStripSystem class"""

    @pytest.fixture
    def strip_system(self):
        """Wildflower strip system instance for testing"""
        return WildflowerStripSystem()

    def test_initialization(self, strip_system):
        """Test system initialization"""
        assert strip_system.active_strips == {}
        assert strip_system.patch_strips == {}
        assert len(strip_system.seed_mix_templates) > 0
        assert len(strip_system.placement_rules) > 0
        assert len(strip_system.maintenance_templates) > 0

    def test_seed_mix_templates_complete(self, strip_system):
        """Test seed mix templates have valid data"""
        for mix_type, species_list in strip_system.seed_mix_templates.items():
            assert len(species_list) > 0

            # Check total percentage approximately 100%
            total_percentage = sum(s.percentage for s in species_list)
            assert 99.0 <= total_percentage <= 101.0, f"{mix_type} total: {total_percentage}%"

            # Check all species have required attributes
            for species in species_list:
                assert species.scientific_name
                assert species.common_name
                assert 0 < species.percentage <= 100
                assert 1 <= species.flowering_start_day <= 365
                assert 1 <= species.flowering_end_day <= 365
                assert 0 <= species.nectar_rating <= 10
                assert 0 <= species.pollen_rating <= 10
                assert species.height_cm > 0
                assert 0 <= species.establishment_rate <= 1
                assert species.persistence_years > 0

    def test_create_wildflower_strip(self, strip_system):
        """Test creating a wildflower strip"""
        strip = strip_system.create_wildflower_strip(
            strip_id="test_strip",
            width=StripWidth.STANDARD,
            length_meters=300.0,
            placement=PlacementStrategy.FIELD_EDGE,
            seed_mix_type=SeedMixType.POLLINATOR_NECTAR,
            start_x=100.0,
            start_y=100.0,
            end_x=400.0,
            end_y=100.0,
            establishment_year=2024
        )

        assert strip.strip_id == "test_strip"
        assert strip.width == StripWidth.STANDARD
        assert strip.length_meters == 300.0
        assert strip.seed_mix_type == SeedMixType.POLLINATOR_NECTAR

        # Check strip is stored
        assert "test_strip" in strip_system.active_strips

        # Check species composition assigned
        assert len(strip.species_composition) > 0

        # Check maintenance schedule assigned
        assert len(strip.maintenance_schedule) > 0

    def test_create_strip_with_custom_species(self, strip_system):
        """Test creating strip with custom species mix"""
        custom_species = [
            SpeciesComposition(
                "Custom species 1", "Custom 1", 40.0,
                100, 200, FloweringPeriod.EARLY_SUMMER,
                8.0, 7.0, 70, 0.85, 5
            ),
            SpeciesComposition(
                "Custom species 2", "Custom 2", 60.0,
                150, 250, FloweringPeriod.MID_SUMMER,
                9.0, 8.0, 80, 0.90, 6
            )
        ]

        strip = strip_system.create_wildflower_strip(
            strip_id="custom_strip",
            width=StripWidth.WIDE,
            length_meters=200.0,
            placement=PlacementStrategy.CONNECTIVITY,
            seed_mix_type=SeedMixType.GENERAL_WILDFLOWER,
            start_x=0.0,
            start_y=0.0,
            end_x=200.0,
            end_y=0.0,
            establishment_year=2024,
            custom_species=custom_species
        )

        assert len(strip.species_composition) == 2
        assert strip.species_composition[0].scientific_name == "Custom species 1"
        assert strip.species_composition[1].scientific_name == "Custom species 2"

    def test_optimize_strip_placement(self, strip_system):
        """Test strip placement optimization"""
        # Create test patches
        patches = []
        for i in range(5):
            patch = ResourcePatch(
                patch_id=1000 + i,
                x=i * 100,
                y=0,
                habitat_type=HabitatType.CROPLAND
            )
            patch.area_ha = 10.0 + i * 2.0  # Varying sizes
            patches.append(patch)

        recommendations = strip_system.optimize_strip_placement(
            available_patches=patches,
            target_area_ha=2.0,
            placement_strategy=PlacementStrategy.FIELD_EDGE
        )

        assert len(recommendations) > 0

        for rec in recommendations:
            assert "patch_id" in rec
            assert "score" in rec
            assert "strip_config" in rec
            assert "rationale" in rec

            # Check strip config
            config = rec["strip_config"]
            assert "width" in config
            assert "length_m" in config
            assert "area_ha" in config
            assert config["area_ha"] > 0

    def test_plan_flowering_succession(self, strip_system):
        """Test flowering succession planning"""
        # Plan for summer months (June-August)
        succession_plan = strip_system.plan_flowering_succession(
            target_months=[6, 7, 8],
            available_area_ha=1.0,
            priority_species=["Centaurea nigra"]
        )

        assert "selected_species" in succession_plan
        assert "coverage_percent" in succession_plan
        assert "timeline_summary" in succession_plan

        # Check priority species included
        species_names = [s.scientific_name for s in succession_plan["selected_species"]]
        assert "Centaurea nigra" in species_names

        # Check coverage
        assert succession_plan["coverage_percent"] > 80  # Should have good summer coverage

        # Check timeline
        assert 6 in succession_plan["timeline_summary"]
        assert 7 in succession_plan["timeline_summary"]
        assert 8 in succession_plan["timeline_summary"]

    def test_schedule_maintenance(self, strip_system):
        """Test maintenance scheduling"""
        # Create strip
        strip_system.create_wildflower_strip(
            strip_id="maintenance_test",
            width=StripWidth.STANDARD,
            length_meters=100.0,
            placement=PlacementStrategy.FIELD_EDGE,
            seed_mix_type=SeedMixType.GENERAL_WILDFLOWER,
            start_x=0.0,
            start_y=0.0,
            end_x=100.0,
            end_y=0.0,
            establishment_year=2023
        )

        # Schedule maintenance
        current_date = date(2024, 6, 15)
        schedule = strip_system.schedule_maintenance(
            strip_id="maintenance_test",
            current_date=current_date,
            years_ahead=3
        )

        assert len(schedule) > 0

        for activity in schedule:
            assert "year" in activity
            assert "activity" in activity
            assert "timing_window" in activity
            assert "equipment" in activity
            assert activity["year"] >= 2024
            assert activity["year"] <= 2026

    def test_calculate_strip_resources(self, strip_system):
        """Test resource calculation for strips"""
        # Create strip with known species
        strip_system.create_wildflower_strip(
            strip_id="resource_test",
            width=StripWidth.STANDARD,
            length_meters=100.0,
            placement=PlacementStrategy.FIELD_EDGE,
            seed_mix_type=SeedMixType.POLLINATOR_NECTAR,
            start_x=0.0,
            start_y=0.0,
            end_x=100.0,
            end_y=0.0,
            establishment_year=2024
        )

        # Calculate resources during flowering period
        # Most species flower in summer (day 180-250)
        resources = strip_system.calculate_strip_resources("resource_test", 200)

        assert "nectar" in resources
        assert "pollen" in resources
        assert "flowering_species" in resources

        assert resources["nectar"] > 0
        assert resources["pollen"] > 0
        assert resources["flowering_species"] > 0

        # Test outside flowering period
        winter_resources = strip_system.calculate_strip_resources("resource_test", 30)
        assert winter_resources["flowering_species"] == 0

    def test_get_strip_summary(self, strip_system):
        """Test strip summary statistics"""
        # Test empty system
        summary = strip_system.get_strip_summary()
        assert summary["status"] == "no_strips"

        # Create multiple strips
        strip_system.create_wildflower_strip(
            "strip1", StripWidth.NARROW, 200.0,
            PlacementStrategy.FIELD_EDGE, SeedMixType.GENERAL_WILDFLOWER,
            0, 0, 200, 0, 2024
        )

        strip_system.create_wildflower_strip(
            "strip2", StripWidth.WIDE, 300.0,
            PlacementStrategy.CONNECTIVITY, SeedMixType.POLLINATOR_NECTAR,
            0, 100, 300, 100, 2024
        )

        summary = strip_system.get_strip_summary()

        assert summary["total_strips"] == 2
        assert summary["total_area_hectares"] > 0
        assert summary["total_length_km"] == 0.5  # 200m + 300m = 500m = 0.5km
        assert summary["width_breakdown_ha"]["3"] > 0  # Narrow strip
        assert summary["width_breakdown_ha"]["12"] > 0  # Wide strip
        assert summary["total_annual_payments"] > 0
        assert summary["total_species"] > 0

    def test_validate_strip_compliance(self, strip_system):
        """Test strip compliance validation"""
        strip_system.create_wildflower_strip(
            strip_id="compliance_test",
            width=StripWidth.STANDARD,
            length_meters=150.0,
            placement=PlacementStrategy.FIELD_EDGE,
            seed_mix_type=SeedMixType.GENERAL_WILDFLOWER,
            start_x=0.0,
            start_y=0.0,
            end_x=150.0,
            end_y=0.0,
            establishment_year=2023
        )

        current_date = date(2024, 6, 15)
        validation = strip_system.validate_strip_compliance("compliance_test", current_date)

        assert validation["valid"] is True
        assert validation["years_established"] == 1
        assert len(validation["compliance_issues"]) == 0

        # Test future establishment
        strip_system.create_wildflower_strip(
            strip_id="future_test",
            width=StripWidth.STANDARD,
            length_meters=100.0,
            placement=PlacementStrategy.FIELD_EDGE,
            seed_mix_type=SeedMixType.GENERAL_WILDFLOWER,
            start_x=0.0,
            start_y=0.0,
            end_x=100.0,
            end_y=0.0,
            establishment_year=2026
        )

        future_validation = strip_system.validate_strip_compliance("future_test", current_date)
        assert future_validation["valid"] is False
        assert "future" in future_validation["compliance_issues"][0]


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""

    @pytest.fixture
    def configured_system(self):
        """System with multiple configured strips"""
        system = WildflowerStripSystem()

        # Create various strips
        system.create_wildflower_strip(
            "edge_strip_1", StripWidth.STANDARD, 500.0,
            PlacementStrategy.FIELD_EDGE, SeedMixType.GENERAL_WILDFLOWER,
            0, 0, 500, 0, 2023
        )

        system.create_wildflower_strip(
            "pollinator_corridor", StripWidth.WIDE, 800.0,
            PlacementStrategy.CONNECTIVITY, SeedMixType.POLLINATOR_NECTAR,
            100, 100, 900, 100, 2024
        )

        system.create_wildflower_strip(
            "beetle_bank", StripWidth.NARROW, 300.0,
            PlacementStrategy.IN_FIELD, SeedMixType.PERENNIAL_NATIVES,
            200, 200, 500, 200, 2024
        )

        return system

    def test_landscape_scale_implementation(self, configured_system):
        """Test landscape-scale strip implementation"""
        summary = configured_system.get_strip_summary()

        assert summary["total_strips"] == 3
        assert summary["total_length_km"] == 1.6  # 500 + 800 + 300 = 1600m

        # Check diversity of placements
        placement_breakdown = summary["placement_breakdown_ha"]
        assert len(placement_breakdown) == 3  # Three different strategies

        # Check mix diversity
        mix_breakdown = summary["mix_type_breakdown_ha"]
        assert len(mix_breakdown) >= 2  # At least two different mixes

    def test_seasonal_flowering_coverage(self, configured_system):
        """Test flowering coverage across seasons"""
        # Check resource provision across year
        test_days = [90, 150, 210, 270]  # Spring, early summer, late summer, autumn

        total_resources = {day: {"nectar": 0, "pollen": 0} for day in test_days}

        for strip_id in configured_system.active_strips:
            for day in test_days:
                resources = configured_system.calculate_strip_resources(strip_id, day)
                total_resources[day]["nectar"] += resources["nectar"]
                total_resources[day]["pollen"] += resources["pollen"]

        # Should have resources in summer months at least
        assert total_resources[150]["nectar"] > 0
        assert total_resources[210]["nectar"] > 0

        # Verify seasonal variation exists
        nectar_values = [total_resources[day]["nectar"] for day in test_days]
        assert max(nectar_values) > min(nectar_values) * 2  # Significant variation

    def test_multi_year_maintenance_planning(self, configured_system):
        """Test multi-year maintenance scheduling"""
        current_date = date(2024, 6, 15)

        all_maintenance = []
        for strip_id in configured_system.active_strips:
            schedule = configured_system.schedule_maintenance(
                strip_id, current_date, years_ahead=5
            )
            all_maintenance.extend(schedule)

        # Should have maintenance activities scheduled
        assert len(all_maintenance) > 0

        # Check distribution across years
        years = set(activity["year"] for activity in all_maintenance)
        assert len(years) >= 3  # Activities spread across multiple years

        # Check different activity types
        activity_types = set(activity["activity"] for activity in all_maintenance)
        assert len(activity_types) >= 2  # Multiple maintenance types

    def test_succession_planning_integration(self, configured_system):
        """Test integration of succession planning with existing strips"""
        # Plan for extending flowering period
        succession_plan = configured_system.plan_flowering_succession(
            target_months=[4, 5, 9, 10],  # Spring and autumn gaps
            available_area_ha=0.5,
            priority_species=["Primula veris", "Aster amellus"]  # Early/late species
        )

        assert succession_plan["selected_species"]

        # Check coverage of target months
        timeline = succession_plan["timeline_summary"]
        assert 4 in timeline or 5 in timeline  # Spring coverage
        assert 9 in timeline or 10 in timeline  # Autumn coverage

        # Verify cost calculation
        assert succession_plan["establishment_cost_per_ha"] > 0
        assert succession_plan["average_nectar_rating"] > 0
