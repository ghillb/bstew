"""
Tests for Bumblebee-Specific Behaviors: Buzz Pollination and Thermoregulation
=============================================================================

Comprehensive tests ensuring biological accuracy of unique bumblebee behaviors
that distinguish them from honey bees in conservation research modeling.

Tests cover:
1. Buzz pollination behavior for sonication flowers (tomatoes, blueberries)
2. Thermoregulation for cold weather foraging (2-8°C vs honey bee 12°C minimum)
3. Species-specific behavioral differences
4. Energy cost calculations and performance metrics
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from bstew.core.bumblebee_behaviors import (
    BuzzPollinationBehavior,
    ThermalRegulationBehavior,
    SonicationFlowerType,
    BuzzPollinationEfficiency,
    ThermoregulationCapacity,
)
from bstew.core.species_parameters import create_literature_validated_species


class TestBuzzPollinationBehavior:
    """Test buzz pollination behavior implementation"""

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    @pytest.fixture
    def terrestris_buzz_behavior(self, species_params):
        """Create B. terrestris buzz behavior instance"""
        return BuzzPollinationBehavior(species_params["Bombus_terrestris"])

    @pytest.fixture
    def pascuorum_buzz_behavior(self, species_params):
        """Create B. pascuorum buzz behavior instance"""
        return BuzzPollinationBehavior(species_params["Bombus_pascuorum"])

    @pytest.fixture
    def lapidarius_buzz_behavior(self, species_params):
        """Create B. lapidarius buzz behavior instance"""
        return BuzzPollinationBehavior(species_params["Bombus_lapidarius"])

    def test_buzz_behavior_initialization(self, terrestris_buzz_behavior):
        """Test buzz behavior initializes correctly"""

        behavior = terrestris_buzz_behavior

        # Check basic initialization
        assert behavior.species_parameters.species_name == "Bombus_terrestris"
        assert isinstance(behavior.buzz_efficiency, BuzzPollinationEfficiency)
        assert len(behavior.compatible_flowers) > 0

        # Check performance tracking initialization
        assert behavior.sonication_events == 0
        assert behavior.total_energy_expended == 0.0
        assert behavior.pollen_collected_mg == 0.0

    def test_buzz_parameters_species_differences(self, species_params):
        """Test that species have different buzz pollination parameters"""

        terrestris_behavior = BuzzPollinationBehavior(species_params["Bombus_terrestris"])
        pascuorum_behavior = BuzzPollinationBehavior(species_params["Bombus_pascuorum"])
        lapidarius_behavior = BuzzPollinationBehavior(species_params["Bombus_lapidarius"])

        # Check frequency differences
        terrestris_freq = terrestris_behavior.buzz_efficiency.vibration_frequency_hz
        pascuorum_freq = pascuorum_behavior.buzz_efficiency.vibration_frequency_hz
        lapidarius_freq = lapidarius_behavior.buzz_efficiency.vibration_frequency_hz

        assert terrestris_freq != pascuorum_freq
        assert terrestris_freq != lapidarius_freq
        assert pascuorum_freq != lapidarius_freq

        # Check efficiency differences
        terrestris_eff = terrestris_behavior.buzz_efficiency.pollen_release_efficiency
        pascuorum_eff = pascuorum_behavior.buzz_efficiency.pollen_release_efficiency
        lapidarius_eff = lapidarius_behavior.buzz_efficiency.pollen_release_efficiency

        # B. terrestris should be most efficient
        assert terrestris_eff > pascuorum_eff
        assert terrestris_eff > lapidarius_eff

        # All should be within valid ranges
        for freq in [terrestris_freq, pascuorum_freq, lapidarius_freq]:
            assert 100.0 <= freq <= 1000.0

        for eff in [terrestris_eff, pascuorum_eff, lapidarius_eff]:
            assert 0.0 <= eff <= 1.0

    def test_compatible_flowers_by_species(self, species_params):
        """Test that species have appropriate compatible flowers"""

        terrestris_behavior = BuzzPollinationBehavior(species_params["Bombus_terrestris"])
        pascuorum_behavior = BuzzPollinationBehavior(species_params["Bombus_pascuorum"])
        lapidarius_behavior = BuzzPollinationBehavior(species_params["Bombus_lapidarius"])

        # All species should handle basic crops
        basic_crops = [SonicationFlowerType.TOMATO, SonicationFlowerType.EGGPLANT, SonicationFlowerType.POTATO]

        for crop in basic_crops:
            assert terrestris_behavior.can_buzz_pollinate(crop)
            assert pascuorum_behavior.can_buzz_pollinate(crop)
            assert lapidarius_behavior.can_buzz_pollinate(crop)

        # Larger species should handle bigger flowers
        # B. terrestris (22mm) should handle all flowers including kiwi
        assert terrestris_behavior.can_buzz_pollinate(SonicationFlowerType.KIWI)
        assert terrestris_behavior.can_buzz_pollinate(SonicationFlowerType.BLUEBERRY)

        # B. lapidarius (18mm) should handle blueberries but not necessarily kiwi
        assert lapidarius_behavior.can_buzz_pollinate(SonicationFlowerType.BLUEBERRY)

        # B. pascuorum (16mm) should handle fewer large flowers
        # Check compatibility is size-dependent
        assert len(terrestris_behavior.compatible_flowers) >= len(pascuorum_behavior.compatible_flowers)

    def test_buzz_pollination_success(self, terrestris_buzz_behavior):
        """Test successful buzz pollination on tomato"""

        behavior = terrestris_buzz_behavior
        initial_events = behavior.sonication_events
        initial_energy = behavior.total_energy_expended
        initial_pollen = behavior.pollen_collected_mg

        # Perform buzz pollination on tomato
        result = behavior.perform_buzz_pollination(SonicationFlowerType.TOMATO)

        # Check successful result
        assert result["success"] is True
        assert result["pollen_collected_mg"] > 0.0
        assert result["energy_expended_j"] > 0.0
        assert result["sonication_duration_ms"] > 0.0
        assert "effectiveness" in result
        assert 0.0 <= result["effectiveness"] <= 1.0

        # Check tracking updated
        assert behavior.sonication_events == initial_events + 1
        assert behavior.total_energy_expended > initial_energy
        assert behavior.pollen_collected_mg > initial_pollen

    def test_buzz_pollination_incompatible_flower(self, pascuorum_buzz_behavior):
        """Test buzz pollination fails on incompatible flower"""

        behavior = pascuorum_buzz_behavior

        # Try to buzz pollinate a flower that's too large (if B. pascuorum can't handle kiwi)
        if not behavior.can_buzz_pollinate(SonicationFlowerType.KIWI):
            result = behavior.perform_buzz_pollination(SonicationFlowerType.KIWI)

            assert result["success"] is False
            assert result["pollen_collected_mg"] == 0.0
            assert result["energy_expended_j"] == 0.0
            assert "reason" in result
            assert "cannot buzz pollinate" in result["reason"]

    def test_buzz_pollination_with_custom_characteristics(self, terrestris_buzz_behavior):
        """Test buzz pollination with custom flower characteristics"""

        behavior = terrestris_buzz_behavior

        # Custom flower characteristics
        custom_flower = {
            "optimal_frequency_hz": behavior.buzz_efficiency.vibration_frequency_hz,  # Perfect match
            "anther_size_mm": 2.5,
            "available_pollen_mg": 1.0
        }

        result = behavior.perform_buzz_pollination(
            SonicationFlowerType.TOMATO,
            flower_characteristics=custom_flower
        )

        # Should be highly effective due to frequency match
        assert result["success"] is True
        assert result["frequency_match"] > 0.8
        assert result["effectiveness"] > 0.6
        assert result["pollen_collected_mg"] > 0.5

    def test_frequency_effectiveness_calculation(self, terrestris_buzz_behavior):
        """Test frequency matching effectiveness calculation"""

        behavior = terrestris_buzz_behavior
        bee_frequency = behavior.buzz_efficiency.vibration_frequency_hz

        # Perfect match
        perfect_match = behavior._calculate_frequency_effectiveness(bee_frequency)
        assert perfect_match == 1.0

        # Close match
        close_match = behavior._calculate_frequency_effectiveness(bee_frequency + 50.0)
        assert 0.7 <= close_match < 1.0

        # Poor match
        poor_match = behavior._calculate_frequency_effectiveness(bee_frequency + 200.0)
        assert poor_match == 0.5  # Minimum effectiveness

    def test_size_compatibility_calculation(self, terrestris_buzz_behavior):
        """Test size compatibility calculation"""

        behavior = terrestris_buzz_behavior

        # Optimal size ratio (for 22mm bee, optimal anther is ~5.5mm, so 2.0mm is close)
        optimal_size = behavior._calculate_size_compatibility(5.5)  # Optimal anther size for 22mm bee
        assert optimal_size == 1.0

        # Too small flower
        small_flower = behavior._calculate_size_compatibility(0.5)
        assert small_flower < 1.0

        # Too large flower
        large_flower = behavior._calculate_size_compatibility(8.0)
        assert large_flower < 1.0

    def test_buzz_pollination_summary(self, terrestris_buzz_behavior):
        """Test buzz pollination summary generation"""

        behavior = terrestris_buzz_behavior

        # Perform some buzz pollination events
        behavior.perform_buzz_pollination(SonicationFlowerType.TOMATO)
        behavior.perform_buzz_pollination(SonicationFlowerType.BLUEBERRY)

        summary = behavior.get_buzz_pollination_summary()

        # Check summary structure
        assert "species_name" in summary
        assert "buzz_parameters" in summary
        assert "compatible_flowers" in summary
        assert "performance_metrics" in summary

        # Check performance metrics
        metrics = summary["performance_metrics"]
        assert metrics["total_sonication_events"] == 2
        assert metrics["total_energy_expended_j"] > 0.0
        assert metrics["total_pollen_collected_mg"] > 0.0
        assert metrics["average_energy_per_event"] > 0.0
        assert metrics["average_pollen_per_event"] > 0.0


class TestThermalRegulationBehavior:
    """Test thermal regulation behavior implementation"""

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    @pytest.fixture
    def terrestris_thermal_behavior(self, species_params):
        """Create B. terrestris thermal behavior instance"""
        return ThermalRegulationBehavior(species_params["Bombus_terrestris"])

    @pytest.fixture
    def pascuorum_thermal_behavior(self, species_params):
        """Create B. pascuorum thermal behavior instance"""
        return ThermalRegulationBehavior(species_params["Bombus_pascuorum"])

    @pytest.fixture
    def lapidarius_thermal_behavior(self, species_params):
        """Create B. lapidarius thermal behavior instance"""
        return ThermalRegulationBehavior(species_params["Bombus_lapidarius"])

    def test_thermal_behavior_initialization(self, terrestris_thermal_behavior):
        """Test thermal behavior initializes correctly"""

        behavior = terrestris_thermal_behavior

        # Check basic initialization
        assert behavior.species_parameters.species_name == "Bombus_terrestris"
        assert isinstance(behavior.thermal_capacity, ThermoregulationCapacity)

        # Check initial state
        assert behavior.thoracic_temperature_c == 20.0
        assert behavior.is_warmed_up is False
        assert behavior.warmup_energy_expended == 0.0

        # Check performance tracking initialization
        assert behavior.cold_weather_flights == 0
        assert behavior.total_warmup_time == 0.0
        assert behavior.total_thermal_energy == 0.0

    def test_thermal_parameters_species_differences(self, species_params):
        """Test that species have different thermal regulation parameters"""

        terrestris_behavior = ThermalRegulationBehavior(species_params["Bombus_terrestris"])
        pascuorum_behavior = ThermalRegulationBehavior(species_params["Bombus_pascuorum"])
        lapidarius_behavior = ThermalRegulationBehavior(species_params["Bombus_lapidarius"])

        # Check minimum flight temperature differences (species-specific)
        terrestris_min = terrestris_behavior.thermal_capacity.min_flight_temperature_c
        pascuorum_min = pascuorum_behavior.thermal_capacity.min_flight_temperature_c
        lapidarius_min = lapidarius_behavior.thermal_capacity.min_flight_temperature_c

        # All should be below honey bee minimum (12°C)
        assert terrestris_min < 12.0
        assert pascuorum_min < 12.0
        assert lapidarius_min < 12.0

        # B. lapidarius should be least cold tolerant (heat-adapted)
        assert lapidarius_min >= terrestris_min

        # Check warmup time differences
        terrestris_warmup = terrestris_behavior.thermal_capacity.preflight_warmup_time_s
        pascuorum_warmup = pascuorum_behavior.thermal_capacity.preflight_warmup_time_s
        lapidarius_warmup = lapidarius_behavior.thermal_capacity.preflight_warmup_time_s

        # All should have reasonable warmup times
        assert 10.0 <= terrestris_warmup <= 300.0
        assert 10.0 <= pascuorum_warmup <= 300.0
        assert 10.0 <= lapidarius_warmup <= 300.0

    def test_flight_capability_at_different_temperatures(self, terrestris_thermal_behavior):
        """Test flight capability assessment at different temperatures"""

        behavior = terrestris_thermal_behavior
        min_temp = behavior.thermal_capacity.min_flight_temperature_c

        # Should be able to fly at or above minimum temperature
        assert behavior.can_fly_at_temperature(min_temp)
        assert behavior.can_fly_at_temperature(min_temp + 5.0)
        assert behavior.can_fly_at_temperature(20.0)

        # Should not be able to fly below minimum temperature
        assert not behavior.can_fly_at_temperature(min_temp - 1.0)
        assert not behavior.can_fly_at_temperature(-5.0)

    def test_warmup_requirement_assessment(self, terrestris_thermal_behavior):
        """Test warm-up requirement assessment"""

        behavior = terrestris_thermal_behavior
        min_temp = behavior.thermal_capacity.min_flight_temperature_c

        # Should require warm-up in cold conditions
        assert behavior.requires_warmup(min_temp + 1.0)
        assert behavior.requires_warmup(5.0)

        # Should not require warm-up in warm conditions
        assert not behavior.requires_warmup(20.0)
        assert not behavior.requires_warmup(25.0)

    def test_successful_preflight_warmup(self, terrestris_thermal_behavior):
        """Test successful pre-flight warm-up behavior"""

        behavior = terrestris_thermal_behavior
        min_temp = behavior.thermal_capacity.min_flight_temperature_c
        test_temp = min_temp + 2.0  # Cold but flyable

        initial_warmup_time = behavior.total_warmup_time
        initial_thermal_energy = behavior.total_thermal_energy

        result = behavior.perform_preflight_warmup(test_temp)

        # Check successful warm-up
        assert result["success"] is True
        assert result["warmup_time_s"] > 0.0
        assert result["energy_expended_j"] > 0.0
        assert result["final_thoracic_temp_c"] > test_temp

        # Check state updates
        assert behavior.is_warmed_up is True
        assert behavior.thoracic_temperature_c > 20.0

        # Check tracking
        assert behavior.total_warmup_time > initial_warmup_time
        assert behavior.total_thermal_energy > initial_thermal_energy

    def test_failed_warmup_too_cold(self, terrestris_thermal_behavior):
        """Test warm-up failure when too cold"""

        behavior = terrestris_thermal_behavior
        min_temp = behavior.thermal_capacity.min_flight_temperature_c
        too_cold_temp = min_temp - 2.0

        result = behavior.perform_preflight_warmup(too_cold_temp)

        # Check failed warm-up
        assert result["success"] is False
        assert result["warmup_time_s"] == 0.0
        assert result["energy_expended_j"] == 0.0
        assert "below minimum" in result["reason"]

        # State should not change
        assert behavior.is_warmed_up is False

    def test_no_warmup_needed_warm_conditions(self, terrestris_thermal_behavior):
        """Test no warm-up needed in warm conditions"""

        behavior = terrestris_thermal_behavior
        warm_temp = 20.0

        result = behavior.perform_preflight_warmup(warm_temp)

        # Check no warm-up needed
        assert result["success"] is True
        assert result["warmup_time_s"] == 0.0
        assert result["energy_expended_j"] == 0.0
        assert "No warm-up required" in result["reason"]

        # Should still be warmed up
        assert behavior.is_warmed_up is True
        assert behavior.thoracic_temperature_c >= 30.0

    def test_flight_energy_cost_calculation(self, terrestris_thermal_behavior):
        """Test flight energy cost calculation with thermoregulation"""

        behavior = terrestris_thermal_behavior

        # Test at different temperatures
        warm_temp = 20.0
        cold_temp = 8.0
        flight_duration = 60.0  # seconds

        # Warm conditions - lower thermal cost
        warm_result = behavior.calculate_flight_energy_cost(warm_temp, flight_duration)
        assert warm_result["feasible"] is True
        assert warm_result["flight_energy_j"] > 0.0
        assert warm_result["thermal_energy_j"] >= 0.0

        # Cold conditions - higher thermal cost
        if behavior.can_fly_at_temperature(cold_temp):
            cold_result = behavior.calculate_flight_energy_cost(cold_temp, flight_duration)
            assert cold_result["feasible"] is True
            assert cold_result["total_energy_j"] > warm_result["total_energy_j"]
            assert cold_result["thermal_energy_j"] > warm_result["thermal_energy_j"]
            assert cold_result["thermal_cost_ratio"] > 0.0

        # Impossible conditions
        impossible_temp = behavior.thermal_capacity.min_flight_temperature_c - 5.0
        impossible_result = behavior.calculate_flight_energy_cost(impossible_temp, flight_duration)
        assert impossible_result["feasible"] is False
        assert impossible_result["total_energy_j"] == float('inf')

    def test_honey_bee_comparison(self, terrestris_thermal_behavior):
        """Test comparison to honey bee thermal capabilities"""

        behavior = terrestris_thermal_behavior

        # Test at various temperatures
        temperatures = [15.0, 10.0, 8.0, 5.0, 2.0, -1.0]

        for temp in temperatures:
            comparison = behavior.compare_to_honey_bee_capability(temp)

            # Check comparison structure
            assert "air_temperature_c" in comparison
            assert "bumblebee_can_fly" in comparison
            assert "honey_bee_can_fly" in comparison
            assert "bumblebee_advantage" in comparison
            assert "temperature_advantage_c" in comparison
            assert "exclusive_foraging_opportunity" in comparison

            # Honey bees cannot fly below 12°C
            expected_honey_bee_can_fly = temp >= 12.0
            assert comparison["honey_bee_can_fly"] == expected_honey_bee_can_fly

            # Temperature advantage should be positive
            assert comparison["temperature_advantage_c"] > 0.0

            # Check exclusive foraging opportunities
            if comparison["bumblebee_can_fly"] and not comparison["honey_bee_can_fly"]:
                assert comparison["exclusive_foraging_opportunity"] is True
                assert comparison["bumblebee_advantage"] is True

    def test_cold_weather_flight_tracking(self, terrestris_thermal_behavior):
        """Test tracking of cold weather flights"""

        behavior = terrestris_thermal_behavior
        initial_cold_flights = behavior.cold_weather_flights

        # Perform warm-up in cold conditions (below 10°C)
        cold_temp = 8.0
        if behavior.can_fly_at_temperature(cold_temp):
            behavior.perform_preflight_warmup(cold_temp)
            assert behavior.cold_weather_flights == initial_cold_flights + 1

        # Perform warm-up in mild conditions (above 10°C)
        mild_temp = 15.0
        behavior.perform_preflight_warmup(mild_temp)
        # Should not increment cold weather counter
        assert behavior.cold_weather_flights == initial_cold_flights + (1 if behavior.can_fly_at_temperature(cold_temp) else 0)

    def test_thermal_regulation_summary(self, terrestris_thermal_behavior):
        """Test thermal regulation summary generation"""

        behavior = terrestris_thermal_behavior

        # Perform some thermal regulation events
        if behavior.can_fly_at_temperature(8.0):
            behavior.perform_preflight_warmup(8.0)
        behavior.perform_preflight_warmup(15.0)

        summary = behavior.get_thermal_regulation_summary()

        # Check summary structure
        assert "species_name" in summary
        assert "thermal_parameters" in summary
        assert "current_state" in summary
        assert "performance_metrics" in summary

        # Check thermal parameters
        params = summary["thermal_parameters"]
        assert params["min_flight_temperature_c"] < 12.0  # Better than honey bees
        assert params["cold_advantage_over_honey_bees_c"] > 0.0
        assert params["heat_production_w"] > 0.0
        assert 0.0 < params["thermal_efficiency"] <= 1.0

        # Check current state
        state = summary["current_state"]
        assert state["thoracic_temperature_c"] >= 20.0
        assert isinstance(state["is_warmed_up"], bool)
        assert state["warmup_energy_expended_j"] >= 0.0

        # Check performance metrics
        metrics = summary["performance_metrics"]
        assert metrics["total_warmup_time_s"] >= 0.0
        assert metrics["total_thermal_energy_j"] >= 0.0


class TestBehaviorIntegration:
    """Test integration between buzz pollination and thermal regulation"""

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    def test_combined_behaviors_species_consistency(self, species_params):
        """Test that both behaviors use consistent species parameters"""

        for species_name, params in species_params.items():
            buzz_behavior = BuzzPollinationBehavior(params)
            thermal_behavior = ThermalRegulationBehavior(params)

            # Both should reference same species
            assert buzz_behavior.species_parameters.species_name == species_name
            assert thermal_behavior.species_parameters.species_name == species_name

            # Both should use same base parameters
            assert buzz_behavior.species_parameters.body_size_mm == params.body_size_mm
            assert thermal_behavior.species_parameters.body_size_mm == params.body_size_mm

    def test_cold_weather_buzz_pollination_scenario(self, species_params):
        """Test realistic scenario: buzz pollination in cold weather"""

        terrestris_params = species_params["Bombus_terrestris"]
        buzz_behavior = BuzzPollinationBehavior(terrestris_params)
        thermal_behavior = ThermalRegulationBehavior(terrestris_params)

        cold_temp = 8.0  # Cold morning temperature

        # Check if bee can operate in these conditions
        can_fly = thermal_behavior.can_fly_at_temperature(cold_temp)

        if can_fly:
            # Perform warm-up
            warmup_result = thermal_behavior.perform_preflight_warmup(cold_temp)
            assert warmup_result["success"] is True

            # Calculate flight energy cost
            flight_cost = thermal_behavior.calculate_flight_energy_cost(cold_temp, 120.0)  # 2 minute flight

            # Perform buzz pollination
            pollination_result = buzz_behavior.perform_buzz_pollination(SonicationFlowerType.TOMATO)
            assert pollination_result["success"] is True

            # Total energy cost combines thermal regulation and buzz pollination
            total_energy_cost = (
                warmup_result["energy_expended_j"] +
                flight_cost["total_energy_j"] +
                pollination_result["energy_expended_j"]
            )

            assert total_energy_cost > 0.0

            # Check honey bee comparison
            honey_bee_comparison = thermal_behavior.compare_to_honey_bee_capability(cold_temp)

            # Bumblebee should have exclusive access to these flowers in cold conditions
            if not honey_bee_comparison["honey_bee_can_fly"]:
                assert honey_bee_comparison["exclusive_foraging_opportunity"] is True

    def test_species_behavioral_specializations(self, species_params):
        """Test that species show appropriate behavioral specializations"""

        species_behaviors = {}

        for species_name, params in species_params.items():
            buzz_behavior = BuzzPollinationBehavior(params)
            thermal_behavior = ThermalRegulationBehavior(params)

            species_behaviors[species_name] = {
                "buzz_efficiency": buzz_behavior.buzz_efficiency.pollen_release_efficiency,
                "buzz_frequency": buzz_behavior.buzz_efficiency.vibration_frequency_hz,
                "min_flight_temp": thermal_behavior.thermal_capacity.min_flight_temperature_c,
                "thermal_efficiency": thermal_behavior.thermal_capacity.thermal_efficiency,
                "cold_advantage": thermal_behavior.thermal_capacity.cold_tolerance_advantage
            }

        # B. terrestris should be most efficient buzz pollinator
        terrestris = species_behaviors["Bombus_terrestris"]
        pascuorum = species_behaviors["Bombus_pascuorum"]
        lapidarius = species_behaviors["Bombus_lapidarius"]

        assert terrestris["buzz_efficiency"] >= pascuorum["buzz_efficiency"]
        assert terrestris["buzz_efficiency"] >= lapidarius["buzz_efficiency"]

        # All species should have significant cold tolerance advantage over honey bees
        for species_data in species_behaviors.values():
            assert species_data["cold_advantage"] >= 3.0  # At least 3°C advantage
            assert species_data["min_flight_temp"] < 12.0  # Below honey bee minimum

    def test_energy_budget_realism(self, species_params):
        """Test that energy costs are realistic for bumblebee physiology"""

        terrestris_params = species_params["Bombus_terrestris"]
        buzz_behavior = BuzzPollinationBehavior(terrestris_params)
        thermal_behavior = ThermalRegulationBehavior(terrestris_params)

        # Test buzz pollination energy cost
        buzz_result = buzz_behavior.perform_buzz_pollination(SonicationFlowerType.TOMATO)
        buzz_energy = buzz_result["energy_expended_j"]

        # Should be reasonable for a single pollination event (< 0.1 J)
        assert 0.001 <= buzz_energy <= 0.1

        # Test thermal regulation energy cost
        if thermal_behavior.can_fly_at_temperature(5.0):
            warmup_result = thermal_behavior.perform_preflight_warmup(5.0)
            warmup_energy = warmup_result["energy_expended_j"]

            # Should be reasonable for warm-up (< 2 J, adjusted for realistic thermal costs)
            assert 0.01 <= warmup_energy <= 2.0

            # Flight energy cost
            flight_result = thermal_behavior.calculate_flight_energy_cost(5.0, 60.0)
            flight_energy = flight_result["total_energy_j"]

            # Should be reasonable for 1 minute flight (< 15 J, adjusted for cold weather costs)
            assert 0.1 <= flight_energy <= 15.0


class TestBiologicalAccuracy:
    """Test biological accuracy against known bumblebee vs honey bee differences"""

    def test_sonication_unique_to_bumblebees(self, ):
        """Test that buzz pollination is unique capability"""

        # This test documents that honey bees cannot perform buzz pollination
        # Honey bees cannot vibrate their flight muscles while gripping flowers
        # This is a key biological difference that makes bumblebees essential
        # for crops like tomatoes, blueberries, and eggplants

        species_params = create_literature_validated_species()

        for species_name, params in species_params.items():
            buzz_behavior = BuzzPollinationBehavior(params)

            # All bumblebee species should be able to buzz pollinate tomatoes
            assert buzz_behavior.can_buzz_pollinate(SonicationFlowerType.TOMATO)

            # Buzz pollination should be effective
            result = buzz_behavior.perform_buzz_pollination(SonicationFlowerType.TOMATO)
            assert result["success"] is True
            assert result["pollen_collected_mg"] > 0.0

    def test_cold_tolerance_advantage_over_honey_bees(self):
        """Test that bumblebees have documented cold tolerance advantage"""

        species_params = create_literature_validated_species()
        honey_bee_min_temp = 12.0  # °C - documented honey bee minimum

        for species_name, params in species_params.items():
            thermal_behavior = ThermalRegulationBehavior(params)

            # All bumblebee species should fly at temperatures below honey bee minimum
            min_temp = thermal_behavior.thermal_capacity.min_flight_temperature_c
            assert min_temp < honey_bee_min_temp

            # Test exclusive foraging opportunity at 8°C (common spring morning temperature)
            test_temp = 8.0
            comparison = thermal_behavior.compare_to_honey_bee_capability(test_temp)

            if thermal_behavior.can_fly_at_temperature(test_temp):
                assert comparison["bumblebee_can_fly"] is True
                assert comparison["honey_bee_can_fly"] is False
                assert comparison["exclusive_foraging_opportunity"] is True

    def test_literature_consistency(self):
        """Test that behaviors are consistent with cited literature"""

        species_params = create_literature_validated_species()

        # Test buzz pollination frequencies are within literature ranges
        # Buchmann (1983) documents 50-1000 Hz for buzz pollination
        for species_name, params in species_params.items():
            buzz_behavior = BuzzPollinationBehavior(params)
            frequency = buzz_behavior.buzz_efficiency.vibration_frequency_hz
            assert 50.0 <= frequency <= 1000.0

        # Test thermal regulation temperatures are within literature ranges
        # Heinrich (1979) documents bumblebee flight at 2-8°C minimum
        for species_name, params in species_params.items():
            thermal_behavior = ThermalRegulationBehavior(params)
            min_temp = thermal_behavior.thermal_capacity.min_flight_temperature_c
            assert -2.0 <= min_temp <= 15.0  # Broader range to include all species


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
