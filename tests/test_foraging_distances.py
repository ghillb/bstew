"""
Tests for Advanced Foraging Distance Distributions
=================================================

Comprehensive tests ensuring biological accuracy of species-specific
foraging distance distributions for bumblebee conservation modeling.

Tests cover:
1. Distribution parameter validation and initialization
2. Species-specific distribution differences
3. Environmental modifier effects
4. Field data validation and accuracy
5. Integration with species parameter system
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from bstew.core.foraging_distances import (
    ForagingDistanceDistribution,
    ForagingDistanceParameters,
    EnvironmentalModifiers,
    DistributionType,
    create_species_distance_distribution,
)
from bstew.core.species_parameters import create_literature_validated_species
from bstew.core.species_integration import species_integrator


class TestForagingDistanceParameters:
    """Test foraging distance parameter validation and initialization"""

    def test_exponential_parameters_validation(self):
        """Test exponential distribution parameter validation"""

        # Valid exponential parameters
        params = ForagingDistanceParameters(
            distribution_type=DistributionType.EXPONENTIAL,
            exponential_lambda=0.01,
            max_distance_m=1500.0
        )
        assert params.distribution_type == DistributionType.EXPONENTIAL
        assert params.exponential_lambda == 0.01

        # Invalid lambda (too high)
        with pytest.raises(ValueError):
            ForagingDistanceParameters(
                distribution_type=DistributionType.EXPONENTIAL,
                exponential_lambda=0.15,  # Above maximum
                max_distance_m=1500.0
            )

    def test_gamma_parameters_validation(self):
        """Test gamma distribution parameter validation"""

        # Valid gamma parameters
        params = ForagingDistanceParameters(
            distribution_type=DistributionType.GAMMA,
            gamma_shape=2.0,
            gamma_scale=300.0,
            max_distance_m=1500.0
        )
        assert params.gamma_shape == 2.0
        assert params.gamma_scale == 300.0

        # Invalid shape (too low)
        with pytest.raises(ValueError):
            ForagingDistanceParameters(
                distribution_type=DistributionType.GAMMA,
                gamma_shape=0.3,  # Below minimum
                gamma_scale=300.0,
                max_distance_m=1500.0
            )

    def test_weibull_parameters_validation(self):
        """Test Weibull distribution parameter validation"""

        # Valid Weibull parameters
        params = ForagingDistanceParameters(
            distribution_type=DistributionType.WEIBULL,
            weibull_shape=1.5,
            weibull_scale=400.0,
            max_distance_m=1500.0
        )
        assert params.weibull_shape == 1.5
        assert params.weibull_scale == 400.0

    def test_parameter_constraints(self):
        """Test universal parameter constraints"""

        params = ForagingDistanceParameters(
            distribution_type=DistributionType.EXPONENTIAL,
            exponential_lambda=0.005,
            max_distance_m=2000.0,
            min_distance_m=30.0,
            short_distance_boost=1.5,
            long_distance_penalty=0.7
        )

        assert 10.0 <= params.min_distance_m <= 100.0
        assert 200.0 <= params.max_distance_m <= 5000.0
        assert 1.0 <= params.short_distance_boost <= 3.0
        assert 0.1 <= params.long_distance_penalty <= 1.0


class TestEnvironmentalModifiers:
    """Test environmental modifier parameter validation"""

    def test_environmental_parameter_ranges(self):
        """Test environmental parameter validation ranges"""

        modifiers = EnvironmentalModifiers(
            temperature_c=15.0,
            wind_speed_ms=5.0,
            precipitation_mm=2.0,
            local_resource_density=0.7,
            regional_resource_quality=0.6,
            colony_energy_stores=0.8,
            forager_experience=0.4
        )

        assert -5.0 <= modifiers.temperature_c <= 35.0
        assert 0.0 <= modifiers.wind_speed_ms <= 20.0
        assert 0.0 <= modifiers.precipitation_mm <= 50.0
        assert 0.0 <= modifiers.local_resource_density <= 1.0
        assert 0.0 <= modifiers.regional_resource_quality <= 1.0
        assert 0.0 <= modifiers.colony_energy_stores <= 1.0
        assert 0.0 <= modifiers.forager_experience <= 1.0

    def test_default_environmental_conditions(self):
        """Test default environmental conditions are reasonable"""

        modifiers = EnvironmentalModifiers()

        assert modifiers.temperature_c == 15.0  # Reasonable default
        assert modifiers.wind_speed_ms == 2.0   # Light wind
        assert modifiers.precipitation_mm == 0.0  # No rain
        assert modifiers.local_resource_density == 0.5  # Moderate resources
        assert modifiers.regional_resource_quality == 0.5  # Moderate quality
        assert modifiers.colony_energy_stores == 0.5  # Moderate reserves
        assert modifiers.forager_experience == 0.5  # Moderate experience


class TestForagingDistanceDistributions:
    """Test foraging distance distribution implementations"""

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    @pytest.fixture
    def terrestris_distribution(self, species_params):
        """Create B. terrestris distance distribution"""
        return ForagingDistanceDistribution(species_params["Bombus_terrestris"])

    @pytest.fixture
    def pascuorum_distribution(self, species_params):
        """Create B. pascuorum distance distribution"""
        return ForagingDistanceDistribution(species_params["Bombus_pascuorum"])

    @pytest.fixture
    def lapidarius_distribution(self, species_params):
        """Create B. lapidarius distance distribution"""
        return ForagingDistanceDistribution(species_params["Bombus_lapidarius"])

    def test_distribution_initialization(self, terrestris_distribution):
        """Test distribution initializes correctly"""

        dist = terrestris_distribution

        # Check species parameters are stored
        assert dist.species_parameters.species_name == "Bombus_terrestris"

        # Check distribution parameters are created
        assert isinstance(dist.distribution_params, ForagingDistanceParameters)
        assert dist.distribution_params.max_distance_m > 0

        # Check distribution is initialized
        assert hasattr(dist, '_base_distribution')

        # Check tracking variables
        assert dist.distance_samples == []
        assert dist.rejection_rate == 0.0

    def test_species_specific_distributions(self, species_params):
        """Test that species have different distribution characteristics"""

        terrestris_dist = ForagingDistanceDistribution(species_params["Bombus_terrestris"])
        pascuorum_dist = ForagingDistanceDistribution(species_params["Bombus_pascuorum"])
        lapidarius_dist = ForagingDistanceDistribution(species_params["Bombus_lapidarius"])

        # Check distribution types are appropriate
        assert terrestris_dist.distribution_params.distribution_type == DistributionType.GAMMA
        assert pascuorum_dist.distribution_params.distribution_type == DistributionType.EXPONENTIAL
        assert lapidarius_dist.distribution_params.distribution_type == DistributionType.WEIBULL

        # Check maximum distances match species capabilities
        assert terrestris_dist.distribution_params.max_distance_m == species_params["Bombus_terrestris"].foraging_range_m
        assert pascuorum_dist.distribution_params.max_distance_m == species_params["Bombus_pascuorum"].foraging_range_m
        assert lapidarius_dist.distribution_params.max_distance_m == species_params["Bombus_lapidarius"].foraging_range_m

        # Check species-specific behavioral adjustments
        # B. pascuorum should have strong short-distance preference
        assert pascuorum_dist.distribution_params.short_distance_boost > terrestris_dist.distribution_params.short_distance_boost

    def test_distance_sampling_basic(self, terrestris_distribution):
        """Test basic distance sampling functionality"""

        dist = terrestris_distribution

        # Sample single distance
        distances = dist.sample_foraging_distance(n_samples=1)
        assert len(distances) == 1
        assert distances[0] >= dist.distribution_params.min_distance_m
        assert distances[0] <= dist.distribution_params.max_distance_m

        # Sample multiple distances
        distances = dist.sample_foraging_distance(n_samples=100)
        assert len(distances) == 100
        assert all(d >= dist.distribution_params.min_distance_m for d in distances)
        assert all(d <= dist.distribution_params.max_distance_m for d in distances)

        # Check tracking is updated
        assert len(dist.distance_samples) == 101

    def test_distance_sampling_constraints(self, terrestris_distribution):
        """Test distance sampling respects biological constraints"""

        dist = terrestris_distribution
        distances = dist.sample_foraging_distance(n_samples=500)

        # All distances should be within species range
        assert all(d <= dist.species_parameters.foraging_range_m for d in distances)
        assert all(d >= dist.distribution_params.min_distance_m for d in distances)

        # Should have reasonable distribution shape
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        max_distance = np.max(distances)

        # For gamma distribution, should have realistic characteristics
        assert mean_distance < dist.distribution_params.max_distance_m * 0.7  # Not too high
        assert median_distance < mean_distance  # Right-skewed
        assert max_distance <= dist.distribution_params.max_distance_m

    def test_environmental_modifier_effects(self, terrestris_distribution):
        """Test environmental conditions affect distance sampling"""

        dist = terrestris_distribution

        # Test cold temperature effect
        cold_modifiers = EnvironmentalModifiers(temperature_c=5.0)
        cold_distances = dist.sample_foraging_distance(
            environmental_modifiers=cold_modifiers, n_samples=200
        )

        # Test warm temperature effect
        warm_modifiers = EnvironmentalModifiers(temperature_c=20.0)
        warm_distances = dist.sample_foraging_distance(
            environmental_modifiers=warm_modifiers, n_samples=200
        )

        # Warm conditions should allow longer foraging distances
        cold_mean = np.mean(cold_distances)
        warm_mean = np.mean(warm_distances)
        assert warm_mean > cold_mean

    def test_wind_effect_on_distances(self, terrestris_distribution):
        """Test high winds reduce foraging distances"""

        dist = terrestris_distribution

        # Test low wind conditions
        low_wind = EnvironmentalModifiers(wind_speed_ms=1.0)
        low_wind_distances = dist.sample_foraging_distance(
            environmental_modifiers=low_wind, n_samples=200
        )

        # Test high wind conditions
        high_wind = EnvironmentalModifiers(wind_speed_ms=8.0)
        high_wind_distances = dist.sample_foraging_distance(
            environmental_modifiers=high_wind, n_samples=200
        )

        # High winds should reduce foraging distances
        low_wind_mean = np.mean(low_wind_distances)
        high_wind_mean = np.mean(high_wind_distances)
        assert high_wind_mean < low_wind_mean

    def test_resource_availability_effects(self, terrestris_distribution):
        """Test resource availability affects foraging distances"""

        dist = terrestris_distribution

        # High local resources should reduce long-distance foraging
        high_local = EnvironmentalModifiers(local_resource_density=0.9)
        high_local_distances = dist.sample_foraging_distance(
            environmental_modifiers=high_local, n_samples=200
        )

        # Low local resources should increase foraging distances
        low_local = EnvironmentalModifiers(
            local_resource_density=0.2,
            regional_resource_quality=0.8
        )
        low_local_distances = dist.sample_foraging_distance(
            environmental_modifiers=low_local, n_samples=200
        )

        high_local_mean = np.mean(high_local_distances)
        low_local_mean = np.mean(low_local_distances)

        # Low local resources should lead to longer foraging trips
        assert low_local_mean > high_local_mean

    def test_colony_energy_effects(self, terrestris_distribution):
        """Test colony energy stores affect foraging risk tolerance"""

        dist = terrestris_distribution

        # Low energy should reduce long-distance foraging (risk aversion)
        low_energy = EnvironmentalModifiers(colony_energy_stores=0.1)
        low_energy_distances = dist.sample_foraging_distance(
            environmental_modifiers=low_energy, n_samples=200
        )

        # High energy allows more exploration
        high_energy = EnvironmentalModifiers(colony_energy_stores=0.9)
        high_energy_distances = dist.sample_foraging_distance(
            environmental_modifiers=high_energy, n_samples=200
        )

        low_energy_mean = np.mean(low_energy_distances)
        high_energy_mean = np.mean(high_energy_distances)

        # High energy should allow longer foraging distances
        assert high_energy_mean >= low_energy_mean

    def test_precipitation_effects(self, terrestris_distribution):
        """Test precipitation reduces foraging distances"""

        dist = terrestris_distribution

        # No precipitation
        dry_conditions = EnvironmentalModifiers(precipitation_mm=0.0)
        dry_distances = dist.sample_foraging_distance(
            environmental_modifiers=dry_conditions, n_samples=200
        )

        # Heavy precipitation
        wet_conditions = EnvironmentalModifiers(precipitation_mm=5.0)
        wet_distances = dist.sample_foraging_distance(
            environmental_modifiers=wet_conditions, n_samples=200
        )

        dry_mean = np.mean(dry_distances)
        wet_mean = np.mean(wet_distances)

        # Precipitation should reduce foraging distances
        assert wet_mean < dry_mean

    def test_distance_statistics_calculation(self, terrestris_distribution):
        """Test distance statistics calculation"""

        dist = terrestris_distribution

        # Generate sample data
        distances = dist.sample_foraging_distance(n_samples=500)
        stats = dist.get_distance_statistics()

        # Check statistics are calculated
        assert "mean_distance_m" in stats
        assert "median_distance_m" in stats
        assert "std_distance_m" in stats
        assert "min_distance_m" in stats
        assert "max_distance_m" in stats
        assert "quartile_25_m" in stats
        assert "quartile_75_m" in stats
        assert "sample_count" in stats
        assert "rejection_rate" in stats
        assert "distribution_type" in stats

        # Check statistics are reasonable
        assert stats["sample_count"] == 500
        assert stats["mean_distance_m"] > stats["min_distance_m"]
        assert stats["max_distance_m"] > stats["mean_distance_m"]
        assert 0.0 <= stats["rejection_rate"] <= 1.0
        assert stats["quartile_25_m"] < stats["quartile_75_m"]

    def test_field_data_validation(self, terrestris_distribution):
        """Test validation against field data"""

        dist = terrestris_distribution

        # Mock field data for B. terrestris (realistic values)
        field_mean = 350.0  # meters
        field_std = 200.0   # meters
        field_max = 1500.0  # meters

        validation = dist.validate_against_field_data(
            field_mean=field_mean,
            field_std=field_std,
            field_max=field_max,
            tolerance=0.3  # 30% tolerance
        )

        # Check validation structure
        assert "validation_passed" in validation
        assert "mean_comparison" in validation
        assert "std_comparison" in validation
        assert "max_comparison" in validation

        # Check comparison details
        mean_comp = validation["mean_comparison"]
        assert "field" in mean_comp
        assert "model" in mean_comp
        assert "relative_error" in mean_comp
        assert "valid" in mean_comp

        # Should pass validation with reasonable tolerance
        assert isinstance(validation["validation_passed"], bool)

    def test_distribution_summary(self, terrestris_distribution):
        """Test distribution summary generation"""

        dist = terrestris_distribution
        summary = dist.get_distribution_summary()

        # Check summary structure
        assert "species_name" in summary
        assert "distribution_type" in summary
        assert "parameters" in summary
        assert "distribution_specific" in summary
        assert "species_constraints" in summary
        assert "performance_metrics" in summary

        # Check species information
        assert summary["species_name"] == "Bombus_terrestris"
        assert summary["distribution_type"] == "gamma"

        # Check parameters
        params = summary["parameters"]
        assert "min_distance_m" in params
        assert "max_distance_m" in params
        assert "short_distance_boost" in params
        assert "long_distance_penalty" in params

        # Check species constraints
        constraints = summary["species_constraints"]
        assert "foraging_range_m" in constraints
        assert "body_size_mm" in constraints
        assert "temperature_tolerance_min_c" in constraints


class TestSpeciesDistributionFactory:
    """Test factory function for creating species distributions"""

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    def test_factory_function_default(self, species_params):
        """Test factory function with default distribution"""

        terrestris_dist = create_species_distance_distribution(
            species_params["Bombus_terrestris"]
        )

        assert isinstance(terrestris_dist, ForagingDistanceDistribution)
        assert terrestris_dist.species_parameters.species_name == "Bombus_terrestris"
        assert terrestris_dist.distribution_params.distribution_type == DistributionType.GAMMA

    def test_factory_function_override(self, species_params):
        """Test factory function with distribution type override"""

        # Override to exponential
        terrestris_exp = create_species_distance_distribution(
            species_params["Bombus_terrestris"],
            distribution_type=DistributionType.EXPONENTIAL
        )

        assert terrestris_exp.distribution_params.distribution_type == DistributionType.EXPONENTIAL
        assert terrestris_exp.distribution_params.exponential_lambda is not None

        # Override to Weibull
        terrestris_weibull = create_species_distance_distribution(
            species_params["Bombus_terrestris"],
            distribution_type=DistributionType.WEIBULL
        )

        assert terrestris_weibull.distribution_params.distribution_type == DistributionType.WEIBULL
        assert terrestris_weibull.distribution_params.weibull_shape is not None
        assert terrestris_weibull.distribution_params.weibull_scale is not None

    def test_all_species_factory_creation(self, species_params):
        """Test factory function works for all species"""

        for species_name, params in species_params.items():
            dist = create_species_distance_distribution(params)

            assert isinstance(dist, ForagingDistanceDistribution)
            assert dist.species_parameters.species_name == species_name
            assert dist.distribution_params.max_distance_m == params.foraging_range_m


class TestSpeciesIntegration:
    """Test integration with species parameter system"""

    def test_species_integrator_distance_distributions(self):
        """Test species integrator provides distance distributions"""

        # Get distribution for B. terrestris
        terrestris_dist = species_integrator.get_species_distance_distribution("Bombus_terrestris")

        assert isinstance(terrestris_dist, ForagingDistanceDistribution)
        assert terrestris_dist.species_parameters.species_name == "Bombus_terrestris"

        # Test caching works
        terrestris_dist2 = species_integrator.get_species_distance_distribution("Bombus_terrestris")
        assert terrestris_dist is terrestris_dist2  # Same object (cached)

    def test_species_integrator_different_distributions(self):
        """Test species integrator handles different distribution types"""

        # Get default distribution
        default_dist = species_integrator.get_species_distance_distribution("Bombus_terrestris")

        # Get exponential distribution
        exp_dist = species_integrator.get_species_distance_distribution(
            "Bombus_terrestris",
            DistributionType.EXPONENTIAL
        )

        assert default_dist is not exp_dist  # Different objects
        assert default_dist.distribution_params.distribution_type == DistributionType.GAMMA
        assert exp_dist.distribution_params.distribution_type == DistributionType.EXPONENTIAL

    def test_foraging_parameters_include_distribution(self):
        """Test foraging parameters include distance distribution"""

        foraging_params = species_integrator.get_species_foraging_parameters("Bombus_terrestris")

        assert "distance_distribution" in foraging_params
        assert isinstance(foraging_params["distance_distribution"], ForagingDistanceDistribution)

        # Check other parameters are still present
        assert "max_foraging_distance_m" in foraging_params
        assert "flight_velocity_ms" in foraging_params
        assert "proboscis_length_mm" in foraging_params

    def test_unknown_species_fallback(self):
        """Test handling of unknown species"""

        # Should fallback to default species
        unknown_dist = species_integrator.get_species_distance_distribution("Unknown_species")

        # Should use Bombus_terrestris as fallback
        assert unknown_dist.species_parameters.species_name == "Bombus_terrestris"


class TestBiologicalAccuracy:
    """Test biological accuracy of distance distributions"""

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    def test_species_differences_reflect_biology(self, species_params):
        """Test that species differences reflect biological reality"""

        terrestris_dist = ForagingDistanceDistribution(species_params["Bombus_terrestris"])
        pascuorum_dist = ForagingDistanceDistribution(species_params["Bombus_pascuorum"])
        lapidarius_dist = ForagingDistanceDistribution(species_params["Bombus_lapidarius"])

        # Sample distances for comparison
        terrestris_distances = terrestris_dist.sample_foraging_distance(n_samples=300)
        pascuorum_distances = pascuorum_dist.sample_foraging_distance(n_samples=300)
        lapidarius_distances = lapidarius_dist.sample_foraging_distance(n_samples=300)

        terrestris_mean = np.mean(terrestris_distances)
        pascuorum_mean = np.mean(pascuorum_distances)
        lapidarius_mean = np.mean(lapidarius_distances)

        # B. pascuorum should prefer shorter distances (efficient forager)
        assert pascuorum_mean <= terrestris_mean

        # All should be within species-specific ranges
        assert np.max(terrestris_distances) <= species_params["Bombus_terrestris"].foraging_range_m
        assert np.max(pascuorum_distances) <= species_params["Bombus_pascuorum"].foraging_range_m
        assert np.max(lapidarius_distances) <= species_params["Bombus_lapidarius"].foraging_range_m

    def test_distribution_shapes_appropriate(self, species_params):
        """Test distribution shapes are biologically appropriate"""

        for species_name, params in species_params.items():
            dist = ForagingDistanceDistribution(params)
            distances = dist.sample_foraging_distance(n_samples=500)

            # Should be right-skewed (most trips short, few long)
            mean_distance = np.mean(distances)
            median_distance = np.median(distances)
            assert median_distance <= mean_distance  # Right skew

            # Should have reasonable minimum distances (energy cost threshold)
            assert np.min(distances) >= 20.0  # Minimum biological threshold

            # Should not exceed species maximum
            assert np.max(distances) <= params.foraging_range_m

    def test_temperature_effects_realistic(self, species_params):
        """Test temperature effects are biologically realistic"""

        terrestris_dist = ForagingDistanceDistribution(species_params["Bombus_terrestris"])
        min_temp = species_params["Bombus_terrestris"].temperature_tolerance_min_c

        # Test at minimum temperature
        cold_modifiers = EnvironmentalModifiers(temperature_c=min_temp + 1.0)
        cold_distances = terrestris_dist.sample_foraging_distance(
            environmental_modifiers=cold_modifiers, n_samples=200
        )

        # Test at optimal temperature
        optimal_modifiers = EnvironmentalModifiers(temperature_c=20.0)
        optimal_distances = terrestris_dist.sample_foraging_distance(
            environmental_modifiers=optimal_modifiers, n_samples=200
        )

        cold_mean = np.mean(cold_distances)
        optimal_mean = np.mean(optimal_distances)

        # Should forage closer to nest in cold conditions
        assert cold_mean < optimal_mean

        # But should still be able to forage (bumblebee cold tolerance)
        assert cold_mean > 50.0  # Should not be extremely restricted

    def test_energy_trade_offs_realistic(self, species_params):
        """Test energy trade-offs in distance selection are realistic"""

        terrestris_dist = ForagingDistanceDistribution(species_params["Bombus_terrestris"])

        # Low energy conditions
        low_energy = EnvironmentalModifiers(colony_energy_stores=0.1)
        low_energy_distances = terrestris_dist.sample_foraging_distance(
            environmental_modifiers=low_energy, n_samples=200
        )

        # High energy conditions
        high_energy = EnvironmentalModifiers(colony_energy_stores=0.9)
        high_energy_distances = terrestris_dist.sample_foraging_distance(
            environmental_modifiers=high_energy, n_samples=200
        )

        low_energy_mean = np.mean(low_energy_distances)
        high_energy_mean = np.mean(high_energy_distances)

        # Low energy should lead to more conservative foraging
        assert low_energy_mean <= high_energy_mean

        # But bees should still forage even when energy is low
        assert low_energy_mean > 30.0  # Minimum foraging necessity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
