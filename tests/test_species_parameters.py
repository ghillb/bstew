"""
Tests for Literature-Validated Bumblebee Species Parameters
==========================================================

Comprehensive tests ensuring biological accuracy and literature validation
for the three target bumblebee species in conservation research.
"""

import pytest
from typing import Dict, Any

from bstew.core.species_parameters import (
    BumblebeeParameterValidator,
    LiteratureValidatedSpeciesParameters,
    create_literature_validated_species,
    ValidationResult,
    LiteratureRange,
)
from bstew.components.species_system import SpeciesType
from bstew.components.proboscis_matching import ProboscisCharacteristics
from bstew.components.development import DevelopmentParameters


class TestBumblebeeParameterValidator:
    """Test parameter validation against literature"""

    def test_validator_initialization(self):
        """Test validator initializes with literature ranges"""
        validator = BumblebeeParameterValidator()

        assert len(validator.literature_ranges) == 3
        assert "Bombus_terrestris" in validator.literature_ranges
        assert "Bombus_pascuorum" in validator.literature_ranges
        assert "Bombus_lapidarius" in validator.literature_ranges

    def test_literature_ranges_structure(self):
        """Test literature ranges have required parameters"""
        validator = BumblebeeParameterValidator()

        required_params = [
            "foraging_range_m",
            "max_colony_size",
            "tongue_length_mm",
            "temperature_tolerance_min_c",
            "memory_capacity_patches",
            "flight_velocity_ms",
        ]

        for species in validator.literature_ranges:
            species_ranges = validator.literature_ranges[species]
            for param in required_params:
                assert param in species_ranges
                lit_range = species_ranges[param]
                assert isinstance(lit_range, LiteratureRange)
                assert lit_range.min_value <= lit_range.max_value
                assert lit_range.min_value <= lit_range.mean_value <= lit_range.max_value
                assert lit_range.source_reference != ""

    def test_parameter_validation_valid_case(self):
        """Test validation with valid parameter values"""
        validator = BumblebeeParameterValidator()

        # Test valid foraging range for B. terrestris
        result = validator.validate_parameter(
            "Bombus_terrestris",
            "foraging_range_m",
            800.0  # Within 200-1500 range
        )

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.parameter_name == "foraging_range_m"
        assert result.current_value == 800.0
        assert result.literature_min == 200.0
        assert result.literature_max == 1500.0
        assert "Knight et al." in result.source_reference

    def test_parameter_validation_invalid_case(self):
        """Test validation with invalid parameter values"""
        validator = BumblebeeParameterValidator()

        # Test invalid foraging range (too high, like honey bees)
        result = validator.validate_parameter(
            "Bombus_terrestris",
            "foraging_range_m",
            5000.0  # Honey bee range, invalid for bumblebees
        )

        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert result.current_value == 5000.0
        assert "Value 5000.0 outside range" in result.notes

    def test_species_not_found(self):
        """Test validation with unknown species"""
        validator = BumblebeeParameterValidator()

        result = validator.validate_parameter(
            "Apis_mellifera",  # Honey bee, not in database
            "foraging_range_m",
            1000.0
        )

        assert not result.is_valid
        assert "Species Apis_mellifera not in literature database" in result.notes

    def test_parameter_not_found(self):
        """Test validation with unknown parameter"""
        validator = BumblebeeParameterValidator()

        result = validator.validate_parameter(
            "Bombus_terrestris",
            "unknown_parameter",
            100.0
        )

        assert not result.is_valid
        assert "Parameter unknown_parameter not in literature database" in result.notes

    def test_literature_recommendations(self):
        """Test getting literature-recommended values"""
        validator = BumblebeeParameterValidator()

        recs = validator.get_literature_recommendations("Bombus_terrestris")

        assert isinstance(recs, dict)
        assert len(recs) > 0
        assert "foraging_range_m" in recs
        assert "max_colony_size" in recs
        assert "tongue_length_mm" in recs

        # Check values are within expected ranges
        assert 200 <= recs["foraging_range_m"] <= 1500
        assert 150 <= recs["max_colony_size"] <= 400
        assert 6.9 <= recs["tongue_length_mm"] <= 7.5

    def test_honey_bee_comparison(self):
        """Test comparison to honey bee parameters"""
        validator = BumblebeeParameterValidator()

        comparison = validator.compare_to_honey_bees()

        assert isinstance(comparison, dict)
        assert "foraging_range" in comparison
        assert "colony_size" in comparison
        assert "temperature_tolerance" in comparison
        assert "communication" in comparison
        assert "lifecycle" in comparison

        # Check key differences are highlighted
        assert "3-5x closer" in comparison["foraging_range"]["ratio"]
        assert "50-200x smaller" in comparison["colony_size"]["ratio"]
        assert "colder" in comparison["temperature_tolerance"]["advantage"]


class TestLiteratureValidatedSpeciesParameters:
    """Test enhanced species parameters with validation"""

    def test_memory_capacity_validation(self):
        """Test memory capacity validation"""

        # Valid memory capacity
        params = self._create_test_species_params(memory_capacity_patches=12)
        assert params.memory_capacity_patches == 12

        # Invalid memory capacity (too high) - Pydantic validation
        with pytest.raises(Exception):  # Pydantic will raise ValidationError
            self._create_test_species_params(memory_capacity_patches=25)

    def test_social_recruitment_validation(self):
        """Test social recruitment rate validation"""

        # Valid social recruitment rate
        params = self._create_test_species_params(social_recruitment_rate=0.05)
        assert params.social_recruitment_rate == 0.05

        # Invalid social recruitment rate (too high, like honey bees) - Pydantic validation
        with pytest.raises(Exception):  # Pydantic will raise ValidationError
            self._create_test_species_params(social_recruitment_rate=0.3)  # Honey bee level

    def test_temperature_tolerance_range(self):
        """Test temperature tolerance validation"""

        # Valid temperature tolerance
        params = self._create_test_species_params(temperature_tolerance_min_c=8.0)
        assert params.temperature_tolerance_min_c == 8.0

        # Edge cases should work
        params_cold = self._create_test_species_params(temperature_tolerance_min_c=2.0)
        assert params_cold.temperature_tolerance_min_c == 2.0

        params_warm = self._create_test_species_params(temperature_tolerance_min_c=12.0)
        assert params_warm.temperature_tolerance_min_c == 12.0

    def test_patch_fidelity_range(self):
        """Test patch fidelity strength validation"""

        params = self._create_test_species_params(patch_fidelity_strength=0.7)
        assert params.patch_fidelity_strength == 0.7

        # Test range boundaries
        params_low = self._create_test_species_params(patch_fidelity_strength=0.3)
        assert params_low.patch_fidelity_strength == 0.3

        params_high = self._create_test_species_params(patch_fidelity_strength=0.95)
        assert params_high.patch_fidelity_strength == 0.95

    def _create_test_species_params(self, **overrides):
        """Helper to create test species parameters"""

        default_params = {
            "species_name": "Bombus_terrestris",
            "species_type": SpeciesType.BOMBUS_TERRESTRIS,
            "proboscis_characteristics": ProboscisCharacteristics(
                length_mm=7.2, width_mm=0.25, flexibility=0.85, extension_efficiency=0.9
            ),
            "body_size_mm": 22.0,
            "wing_length_mm": 18.0,
            "weight_mg": 850.0,
            "development_parameters": DevelopmentParameters(
                dev_age_hatching_min=3.0,
                dev_age_pupation_min=12.0,
                dev_age_emerging_min=18.0,
                dev_weight_egg=0.15,
                dev_weight_pupation_min=110.0,
                temperature_optimal=32.0,
            ),
            "max_lifespan_workers": 35,
            "max_lifespan_queens": 365,
            "max_lifespan_drones": 28,
            "emerging_day_mean": 60,
            "emerging_day_sd": 10.0,
            "active_season_start": 50,
            "active_season_end": 280,
            "flight_velocity_ms": 3.2,
            "foraging_range_m": 800.0,
            "search_length_m": 400.0,
            "nectar_load_capacity_mg": 45.0,
            "pollen_load_capacity_mg": 12.0,
            "max_colony_size": 250,
            "typical_colony_size": 175,
            "brood_development_time": 21.0,
            "nest_habitat_preferences": {"woodland": 0.3, "hedgerow": 0.7},
            "foraging_habitat_preferences": {"cropland": 0.9, "wildflower": 0.8},
            "cold_tolerance": 0.8,
            "drought_tolerance": 0.7,
            "competition_strength": 0.9,
            "foraging_aggressiveness": 0.8,
            "site_fidelity": 0.6,
            "social_dominance": 0.9,
            "memory_capacity_patches": 12,
            "temperature_tolerance_min_c": 8.0,
            "social_recruitment_rate": 0.05,
            "patch_fidelity_strength": 0.7,
        }

        default_params.update(overrides)
        return LiteratureValidatedSpeciesParameters(**default_params)


class TestLiteratureValidatedSpeciesCreation:
    """Test creation of literature-validated species"""

    def test_create_all_target_species(self):
        """Test creation of all three target species"""

        species_params = create_literature_validated_species()

        assert len(species_params) == 3
        assert "Bombus_terrestris" in species_params
        assert "Bombus_pascuorum" in species_params
        assert "Bombus_lapidarius" in species_params

        # Check all are LiteratureValidatedSpeciesParameters
        for species_name, params in species_params.items():
            assert isinstance(params, LiteratureValidatedSpeciesParameters)

    def test_terrestris_parameters(self):
        """Test B. terrestris parameters against literature"""

        species_params = create_literature_validated_species()
        terrestris = species_params["Bombus_terrestris"]

        # Check critical parameters are within literature ranges
        assert 200 <= terrestris.foraging_range_m <= 1500  # Literature range
        assert 150 <= terrestris.max_colony_size <= 400    # Literature range
        assert 6.9 <= terrestris.proboscis_characteristics.length_mm <= 7.5  # Short-tongued
        assert 2.0 <= terrestris.temperature_tolerance_min_c <= 8.0  # Cold tolerant
        assert 8 <= terrestris.memory_capacity_patches <= 15  # Individual memory
        assert terrestris.social_recruitment_rate < 0.1  # <10% for bumblebees

        # Check species-specific traits
        assert terrestris.species_type == SpeciesType.BOMBUS_TERRESTRIS
        assert terrestris.emerging_day_mean == 60  # Early March emergence
        assert terrestris.competition_strength == 0.9  # Dominant species

    def test_pascuorum_parameters(self):
        """Test B. pascuorum parameters against literature"""

        species_params = create_literature_validated_species()
        pascuorum = species_params["Bombus_pascuorum"]

        # Check critical parameters are within literature ranges
        assert 300 <= pascuorum.foraging_range_m <= 1400  # Medium-tongued range
        assert 80 <= pascuorum.max_colony_size <= 200     # Smaller colonies
        assert 10.5 <= pascuorum.proboscis_characteristics.length_mm <= 11.8  # Medium-tongued
        assert 4.0 <= pascuorum.temperature_tolerance_min_c <= 10.0
        assert 10 <= pascuorum.memory_capacity_patches <= 14
        assert pascuorum.social_recruitment_rate < 0.1

        # Check species-specific traits
        assert pascuorum.species_type == SpeciesType.BOMBUS_PASCUORUM
        assert pascuorum.emerging_day_mean == 100  # Mid April emergence (later)
        assert pascuorum.active_season_end == 310  # Long season
        assert pascuorum.patch_fidelity_strength == 0.8  # Higher fidelity

    def test_lapidarius_parameters(self):
        """Test B. lapidarius parameters against literature"""

        species_params = create_literature_validated_species()
        lapidarius = species_params["Bombus_lapidarius"]

        # Check critical parameters are within literature ranges
        assert 400 <= lapidarius.foraging_range_m <= 1800  # Longest range
        assert 120 <= lapidarius.max_colony_size <= 300    # Medium colonies
        assert 7.8 <= lapidarius.proboscis_characteristics.length_mm <= 8.4  # Short-tongued
        assert 6.0 <= lapidarius.temperature_tolerance_min_c <= 12.0  # Heat tolerant
        assert 9 <= lapidarius.memory_capacity_patches <= 13
        assert lapidarius.social_recruitment_rate < 0.1

        # Check species-specific traits
        assert lapidarius.species_type == SpeciesType.BOMBUS_LAPIDARIUS
        assert lapidarius.emerging_day_mean == 80  # Late March emergence
        assert lapidarius.drought_tolerance == 0.8  # Heat tolerant
        assert lapidarius.social_recruitment_rate == 0.03  # Lowest social recruitment

    def test_species_differences(self):
        """Test that species have appropriate biological differences"""

        species_params = create_literature_validated_species()
        terrestris = species_params["Bombus_terrestris"]
        pascuorum = species_params["Bombus_pascuorum"]
        lapidarius = species_params["Bombus_lapidarius"]

        # Proboscis length differences (biological constraint)
        assert terrestris.proboscis_characteristics.length_mm < pascuorum.proboscis_characteristics.length_mm
        assert terrestris.proboscis_characteristics.length_mm < lapidarius.proboscis_characteristics.length_mm
        assert lapidarius.proboscis_characteristics.length_mm < pascuorum.proboscis_characteristics.length_mm

        # Colony size differences
        assert pascuorum.max_colony_size < terrestris.max_colony_size
        assert lapidarius.max_colony_size < terrestris.max_colony_size

        # Foraging range differences
        assert terrestris.foraging_range_m < lapidarius.foraging_range_m  # Heat tolerance extends range

        # Emergence timing differences
        assert terrestris.emerging_day_mean < lapidarius.emerging_day_mean < pascuorum.emerging_day_mean

        # Competition strength differences
        assert terrestris.competition_strength > lapidarius.competition_strength
        assert terrestris.competition_strength > pascuorum.competition_strength

    def test_bumblebee_vs_honey_bee_distinctions(self):
        """Test that all species maintain bumblebee vs honey bee distinctions"""

        species_params = create_literature_validated_species()

        for species_name, params in species_params.items():
            # Foraging range: bumblebees much shorter than honey bees (5-6km)
            assert params.foraging_range_m < 2000, f"{species_name} foraging range too high (honey bee-like)"

            # Colony size: bumblebees much smaller than honey bees (20,000-80,000)
            assert params.max_colony_size < 500, f"{species_name} colony size too high (honey bee-like)"

            # Social recruitment: bumblebees <5% vs honey bees 30-70%
            assert params.social_recruitment_rate < 0.1, f"{species_name} social recruitment too high"

            # Individual memory: bumblebees limited vs honey bees unlimited sharing
            assert params.memory_capacity_patches <= 20, f"{species_name} memory capacity too high"

            # Temperature tolerance: bumblebees can work at lower temps than honey bees
            assert params.temperature_tolerance_min_c < 15, f"{species_name} not cold tolerant enough"


class TestValidationIntegration:
    """Test integration of validation with species parameters"""

    def test_validate_created_species(self):
        """Test that created species pass literature validation"""

        validator = BumblebeeParameterValidator()
        species_params = create_literature_validated_species()

        for species_name, params in species_params.items():
            # Convert LiteratureValidatedSpeciesParameters to SpeciesParameters for validation
            # (This tests the inheritance chain works correctly)
            validation_results = validator.validate_species_parameters(species_name, params)

            # All parameters should be valid
            invalid_results = [r for r in validation_results if not r.is_valid]
            assert len(invalid_results) == 0, f"Invalid parameters found for {species_name}: {[r.parameter_name for r in invalid_results]}"

    def test_validation_report_generation(self):
        """Test generation of validation reports"""

        validator = BumblebeeParameterValidator()
        species_params = create_literature_validated_species()

        for species_name, params in species_params.items():
            report = validator.generate_validation_report(species_name, params)

            assert isinstance(report, dict)
            assert report["species_name"] == species_name
            assert report["total_parameters_checked"] > 0
            assert report["validation_success_rate"] == 1.0  # All should be valid
            assert len(report["critical_issues"]) == 0  # No critical issues
            assert len(report["literature_recommendations"]) > 0
            assert report["species_specific_notes"] != ""

    def test_literature_source_traceability(self):
        """Test that all parameters have traceable literature sources"""

        validator = BumblebeeParameterValidator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            species_ranges = validator.literature_ranges[species_name]

            for param_name, lit_range in species_ranges.items():
                # Check source reference exists and is meaningful
                assert lit_range.source_reference != ""
                assert len(lit_range.source_reference) > 10  # More than just a name

                # Check confidence level is specified
                assert lit_range.confidence_level in ["high", "medium", "low"]

                # Check notes provide context
                assert lit_range.notes != ""
                assert len(lit_range.notes) > 20  # Meaningful notes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
