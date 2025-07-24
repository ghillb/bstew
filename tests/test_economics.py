"""
Comprehensive Tests for Economic Assessment System
=================================================

Tests for the economic assessment module including crop valuation,
cost-benefit analysis, and policy scenario evaluation.
"""

import pytest
from unittest.mock import patch

from src.bstew.components.economics import (
    EconomicAssessment,
    EconomicScenario,
    CropType,
    CropEconomics,
    PollinationServiceValue,
    CostBenefitAnalysis,
)
from src.bstew.components.species_system import SpeciesSystem


class TestCropEconomics:
    """Test crop economics data structures"""

    def test_crop_economics_creation(self):
        """Test creating CropEconomics instance"""
        crop = CropEconomics(
            crop_type=CropType.APPLES,
            price_per_kg=1.20,
            yield_baseline_kg_ha=8000,
            yield_with_pollinators_kg_ha=25000,
            pollinator_dependency=0.65,
            hectares_planted=1000,
            production_cost_per_ha=8500,
            harvest_cost_per_kg=0.15,
        )

        assert crop.crop_type == CropType.APPLES
        assert crop.price_per_kg == 1.20
        assert crop.pollinator_dependency == 0.65

    def test_crop_economics_validation(self):
        """Test that crop economics can be created with valid data"""
        # Test that valid economics work
        crop = CropEconomics(
            crop_type=CropType.APPLES,
            price_per_kg=1.20,  # Valid positive price
            yield_baseline_kg_ha=8000,
            yield_with_pollinators_kg_ha=25000,
            pollinator_dependency=0.65,
            hectares_planted=1000,
            production_cost_per_ha=8500,
            harvest_cost_per_kg=0.15,
        )
        assert crop.price_per_kg > 0


class TestEconomicScenario:
    """Test economic scenario configurations"""

    def test_economic_scenario_creation(self):
        """Test creating EconomicScenario"""
        scenario = EconomicScenario(
            scenario_name="Test Scenario",
            crop_data={},
            stewardship_costs={"wildflower_strips": 500},
            discount_rate=0.03,
            time_horizon_years=10,
        )

        assert scenario.scenario_name == "Test Scenario"
        assert scenario.discount_rate == 0.03
        assert scenario.time_horizon_years == 10

    def test_economic_scenario_validation(self):
        """Test economic scenario validation"""
        # Test invalid discount rate
        with pytest.raises(ValueError):
            EconomicScenario(
                scenario_name="Invalid Scenario",
                crop_data={},
                stewardship_costs={},
                discount_rate=1.5,  # > 1.0 should be invalid
                time_horizon_years=10,
            )

        # Test invalid time horizon
        with pytest.raises(ValueError):
            EconomicScenario(
                scenario_name="Invalid Scenario",
                crop_data={},
                stewardship_costs={},
                discount_rate=0.03,
                time_horizon_years=0,  # < 1 should be invalid
            )


class TestPollinationServiceValue:
    """Test pollination service value calculations"""

    def test_pollination_service_value_creation(self):
        """Test creating PollinationServiceValue"""
        value = PollinationServiceValue(
            crop_type=CropType.APPLES,
            annual_value_gbp=10000.0,
            yield_increase_kg=5000.0,
            value_per_hectare=100.0,
            primary_species=["Bombus_terrestris"],
            service_reliability=0.85,
        )

        assert value.crop_type == CropType.APPLES
        assert value.annual_value_gbp == 10000.0
        assert value.service_reliability == 0.85

    def test_service_reliability_validation(self):
        """Test service reliability bounds"""
        with pytest.raises(ValueError):
            PollinationServiceValue(
                crop_type=CropType.APPLES,
                annual_value_gbp=10000.0,
                yield_increase_kg=5000.0,
                value_per_hectare=100.0,
                primary_species=["Bombus_terrestris"],
                service_reliability=1.5,  # > 1.0 should be invalid
            )


class TestEconomicAssessment:
    """Test the main EconomicAssessment class"""

    @pytest.fixture
    def species_system(self):
        """Create a species system for testing"""
        return SpeciesSystem()

    @pytest.fixture
    def economic_assessment(self, species_system):
        """Create an EconomicAssessment instance"""
        return EconomicAssessment(species_system)

    @pytest.fixture
    def sample_species_abundance(self):
        """Sample species abundance data"""
        return {
            "Bombus_terrestris": 150,
            "Bombus_lucorum": 120,
            "Bombus_hortorum": 80,
            "Bombus_ruderatus": 40,
        }

    @pytest.fixture
    def sample_foraging_efficiency(self):
        """Sample foraging efficiency data"""
        return {
            "Bombus_terrestris": 0.85,
            "Bombus_lucorum": 0.78,
            "Bombus_hortorum": 0.92,
            "Bombus_ruderatus": 0.88,
        }

    def test_economic_assessment_initialization(self, species_system):
        """Test EconomicAssessment initialization"""
        assessment = EconomicAssessment(species_system)

        assert assessment.species_system is not None
        assert len(assessment.crop_economics) > 0
        assert len(assessment.species_efficiency) > 0

    def test_crop_economics_initialization(self, economic_assessment):
        """Test that crop economics are properly initialized"""
        crop_econ = economic_assessment.crop_economics

        # Check that key crops are present
        assert CropType.APPLES in crop_econ
        assert CropType.BLUEBERRIES in crop_econ
        assert CropType.OILSEED_RAPE in crop_econ

        # Check apple economics
        apple_econ = crop_econ[CropType.APPLES]
        assert apple_econ.price_per_kg == 1.20
        assert apple_econ.pollinator_dependency == 0.65

    def test_species_efficiency_initialization(self, economic_assessment):
        """Test that species efficiency data is properly initialized"""
        efficiency = economic_assessment.species_efficiency

        # Check that key species are present
        assert "Bombus_terrestris" in efficiency
        assert "Bombus_hortorum" in efficiency

        # Check that each species has crop-specific efficiencies
        terrestris_eff = efficiency["Bombus_terrestris"]
        assert CropType.APPLES in terrestris_eff
        assert 0.0 <= terrestris_eff[CropType.APPLES] <= 1.0

    def test_calculate_pollination_service_value(
        self,
        economic_assessment,
        sample_species_abundance,
        sample_foraging_efficiency
    ):
        """Test pollination service value calculation"""

        value = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            sample_species_abundance,
            sample_foraging_efficiency,
            area_hectares=100.0,
        )

        assert isinstance(value, PollinationServiceValue)
        assert value.crop_type == CropType.APPLES
        assert value.annual_value_gbp > 0
        assert value.yield_increase_kg > 0
        assert 0.0 <= value.service_reliability <= 1.0
        assert len(value.primary_species) > 0

    def test_pollination_value_with_no_species(self, economic_assessment):
        """Test pollination value calculation with no species"""

        value = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            {},  # No species
            {},  # No foraging efficiency
            area_hectares=100.0,
        )

        assert value.annual_value_gbp == 0.0
        assert value.yield_increase_kg == 0.0
        assert value.service_reliability == 0.0
        assert len(value.primary_species) == 0

    def test_pollination_value_scaling_with_area(
        self,
        economic_assessment,
        sample_species_abundance,
        sample_foraging_efficiency
    ):
        """Test that pollination value scales with area"""

        value_100ha = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            sample_species_abundance,
            sample_foraging_efficiency,
            area_hectares=100.0,
        )

        value_200ha = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            sample_species_abundance,
            sample_foraging_efficiency,
            area_hectares=200.0,
        )

        # Value should scale roughly with area
        assert value_200ha.annual_value_gbp > value_100ha.annual_value_gbp
        assert abs(value_200ha.annual_value_gbp - 2 * value_100ha.annual_value_gbp) < value_100ha.annual_value_gbp * 0.1

    def test_invalid_crop_type(
        self,
        economic_assessment,
        sample_species_abundance,
        sample_foraging_efficiency
    ):
        """Test handling of invalid crop type"""

        # Mock a crop type that doesn't exist in economics data
        with patch.object(economic_assessment, 'crop_economics', {}):
            with pytest.raises(ValueError, match="Crop economics not defined"):
                economic_assessment.calculate_pollination_service_value(
                    CropType.APPLES,
                    sample_species_abundance,
                    sample_foraging_efficiency,
                )

    def test_present_value_calculation(self, economic_assessment):
        """Test present value calculation"""

        # Test simple case
        pv = economic_assessment._present_value_series(1000, 5, 0.05)

        # Should be sum of: 1000 + 1000/1.05 + 1000/1.05^2 + ... + 1000/1.05^4
        expected = sum(1000 / (1.05 ** year) for year in range(5))
        assert abs(pv - expected) < 0.01

    def test_present_value_zero_discount(self, economic_assessment):
        """Test present value with zero discount rate"""

        pv = economic_assessment._present_value_series(1000, 5, 0.0)
        assert pv == 5000  # Simple sum with no discounting

    def test_assess_stewardship_scenario(
        self,
        economic_assessment,
        sample_species_abundance,
        sample_foraging_efficiency
    ):
        """Test stewardship scenario assessment"""

        scenario = EconomicScenario(
            scenario_name="Test Stewardship",
            crop_data={
                "apples": CropEconomics(
                    crop_type=CropType.APPLES,
                    price_per_kg=1.20,
                    yield_baseline_kg_ha=8000,
                    yield_with_pollinators_kg_ha=25000,
                    pollinator_dependency=0.65,
                    hectares_planted=100,
                    production_cost_per_ha=8500,
                    harvest_cost_per_kg=0.15,
                ),
            },
            stewardship_costs={"wildflower_strips": 500},
            discount_rate=0.03,
            time_horizon_years=10,
        )

        # Create baseline and enhanced scenarios
        baseline_abundance = {k: int(v * 0.7) for k, v in sample_species_abundance.items()}
        baseline_foraging = {k: v * 0.8 for k, v in sample_foraging_efficiency.items()}

        result = economic_assessment.assess_stewardship_scenario(
            scenario,
            baseline_abundance,
            sample_species_abundance,
            baseline_foraging,
            sample_foraging_efficiency,
        )

        assert isinstance(result, CostBenefitAnalysis)
        assert result.scenario_name == "Test Stewardship"
        assert result.total_costs_gbp > 0
        assert result.total_benefits_gbp >= 0  # Could be zero if no improvement
        assert result.benefit_cost_ratio >= 0

    def test_landscape_assessment(
        self,
        economic_assessment,
        sample_species_abundance,
        sample_foraging_efficiency
    ):
        """Test landscape-scale economic assessment"""

        landscape_data = {
            "crops": {
                "apples": {"area_hectares": 100},
                "blueberries": {"area_hectares": 50},
            }
        }

        species_abundance_data = {
            species: {"patch_1": abundance, "patch_2": abundance}
            for species, abundance in sample_species_abundance.items()
        }

        foraging_data = {
            species: {"patch_1": efficiency, "patch_2": efficiency}
            for species, efficiency in sample_foraging_efficiency.items()
        }

        result = economic_assessment.generate_landscape_assessment(
            landscape_data, species_abundance_data, foraging_data
        )

        assert "total_pollination_value" in result
        assert "crop_breakdown" in result
        assert "species_contributions" in result
        assert "risk_assessment" in result
        assert "recommendations" in result

        assert result["total_pollination_value"] >= 0
        assert len(result["crop_breakdown"]) > 0
        assert isinstance(result["recommendations"], list)

    def test_policy_scenario_comparison(
        self,
        economic_assessment,
        sample_species_abundance,
        sample_foraging_efficiency
    ):
        """Test policy scenario comparison"""

        scenarios = [
            ("Baseline",
             {k: int(v * 0.7) for k, v in sample_species_abundance.items()},
             {k: v * 0.8 for k, v in sample_foraging_efficiency.items()}),
            ("Enhanced",
             sample_species_abundance,
             sample_foraging_efficiency),
        ]

        base_crop_areas = {
            CropType.APPLES: 100,
            CropType.BLUEBERRIES: 50,
        }

        result = economic_assessment.compare_policy_scenarios(scenarios, base_crop_areas)

        assert len(result) == 2
        assert "Baseline" in result
        assert "Enhanced" in result

        for scenario_name, scenario_result in result.items():
            assert "total_annual_value" in scenario_result
            assert "crop_breakdown" in scenario_result
            assert "value_per_hectare" in scenario_result
            assert "relative_performance" in scenario_result
            assert scenario_result["total_annual_value"] >= 0


class TestCropTypeEnum:
    """Test CropType enumeration"""

    def test_crop_type_values(self):
        """Test that crop types have expected values"""
        assert CropType.APPLES.value == "apples"
        assert CropType.BLUEBERRIES.value == "blueberries"
        assert CropType.OILSEED_RAPE.value == "oilseed_rape"

    def test_crop_type_iteration(self):
        """Test iterating over crop types"""
        crop_types = list(CropType)
        assert len(crop_types) >= 4  # At least 4 crop types defined
        assert CropType.APPLES in crop_types


class TestEconomicEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def economic_assessment(self):
        """Create assessment for edge case testing"""
        species_system = SpeciesSystem()
        return EconomicAssessment(species_system)

    def test_zero_abundance_species(self, economic_assessment):
        """Test with zero abundance for all species"""

        value = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            {"Bombus_terrestris": 0},
            {"Bombus_terrestris": 0.85},
            area_hectares=100.0,
        )

        assert value.annual_value_gbp == 0.0
        assert value.service_reliability == 0.0

    def test_very_high_abundance(self, economic_assessment):
        """Test with very high species abundance"""

        high_abundance = {"Bombus_terrestris": 10000}
        high_efficiency = {"Bombus_terrestris": 0.95}

        value = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            high_abundance,
            high_efficiency,
            area_hectares=100.0,
        )

        # Service reliability should be capped at 1.0
        assert value.service_reliability <= 1.0
        assert value.annual_value_gbp > 0

    def test_zero_area(self, economic_assessment):
        """Test with zero hectares"""

        species_abundance = {"Bombus_terrestris": 100}
        foraging_efficiency = {"Bombus_terrestris": 0.85}

        value = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            species_abundance,
            foraging_efficiency,
            area_hectares=0.0,
        )

        assert value.annual_value_gbp == 0.0
        assert value.yield_increase_kg == 0.0

    def test_unknown_species(self, economic_assessment):
        """Test with species not in efficiency matrix"""

        unknown_species = {"Unknown_species": 100}
        foraging_efficiency = {"Unknown_species": 0.85}

        value = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            unknown_species,
            foraging_efficiency,
            area_hectares=100.0,
        )

        # Should handle gracefully with zero contribution
        assert value.annual_value_gbp == 0.0

    def test_negative_efficiency(self, economic_assessment):
        """Test with negative foraging efficiency"""

        species_abundance = {"Bombus_terrestris": 100}
        negative_efficiency = {"Bombus_terrestris": -0.5}

        # Should handle gracefully without crashing
        value = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            species_abundance,
            negative_efficiency,
            area_hectares=100.0,
        )

        # Economic value should not be negative due to negative efficiency
        assert value.annual_value_gbp >= 0


class TestEconomicCalculationAccuracy:
    """Test accuracy of economic calculations"""

    @pytest.fixture
    def economic_assessment(self):
        """Create assessment for calculation testing"""
        species_system = SpeciesSystem()
        return EconomicAssessment(species_system)

    def test_manual_calculation_verification(self, economic_assessment):
        """Verify economic calculation against manual computation"""

        # Use known values for manual verification
        species_abundance = {"Bombus_terrestris": 100}
        foraging_efficiency = {"Bombus_terrestris": 0.8}
        area_hectares = 100.0

        # Get crop economics
        crop_econ = economic_assessment.crop_economics[CropType.APPLES]

        # Manual calculation
        species_efficiency = economic_assessment.species_efficiency["Bombus_terrestris"][CropType.APPLES]
        pollination_capacity = 100 * species_efficiency * 0.8  # abundance × efficiency × foraging
        service_reliability = min(1.0, pollination_capacity / 100.0)

        max_yield_increase = (crop_econ.yield_with_pollinators_kg_ha - crop_econ.yield_baseline_kg_ha) * crop_econ.pollinator_dependency
        actual_yield_increase = max_yield_increase * service_reliability * area_hectares

        gross_value = actual_yield_increase * crop_econ.price_per_kg
        harvest_costs = actual_yield_increase * crop_econ.harvest_cost_per_kg
        expected_net_value = gross_value - harvest_costs

        # Calculate using function
        value = economic_assessment.calculate_pollination_service_value(
            CropType.APPLES,
            species_abundance,
            foraging_efficiency,
            area_hectares,
        )

        # Compare (allow small floating point differences)
        assert abs(value.annual_value_gbp - expected_net_value) < 0.01
        assert abs(value.service_reliability - service_reliability) < 0.01

    def test_cost_benefit_calculation_accuracy(self, economic_assessment):
        """Test accuracy of cost-benefit analysis calculations"""

        scenario = EconomicScenario(
            scenario_name="Test Accuracy",
            crop_data={
                "apples": CropEconomics(
                    crop_type=CropType.APPLES,
                    price_per_kg=1.0,  # Simplified for calculation
                    yield_baseline_kg_ha=1000,
                    yield_with_pollinators_kg_ha=2000,
                    pollinator_dependency=1.0,  # 100% dependency
                    hectares_planted=100,
                    production_cost_per_ha=0,
                    harvest_cost_per_kg=0,  # No costs for simplicity
                ),
            },
            stewardship_costs={"strategy": 100},  # £100/ha
            discount_rate=0.0,  # No discounting for simplicity
            time_horizon_years=1,
        )

        baseline_abundance = {"Bombus_terrestris": 0}  # No pollinators
        enhanced_abundance = {"Bombus_terrestris": 100}  # With pollinators
        baseline_foraging = {"Bombus_terrestris": 0.0}
        enhanced_foraging = {"Bombus_terrestris": 1.0}  # Perfect foraging

        result = economic_assessment.assess_stewardship_scenario(
            scenario,
            baseline_abundance,
            enhanced_abundance,
            baseline_foraging,
            enhanced_foraging,
        )

        # Manual calculation
        # Cost: 100 hectares × £100/ha × 0.1 (applied to 10% of area) = £1000
        expected_cost = 100 * 100 * 0.1

        # Benefit calculation is more complex due to species efficiency
        # But we can verify that calculations are consistent
        assert abs(result.total_costs_gbp - expected_cost) < 1.0  # Within £1
        assert result.total_benefits_gbp >= 0
