"""
Advanced Foraging Distance Distributions for Bumblebees
=======================================================

CRITICAL: Implements species-specific foraging distance distributions
based on field data and biological constraints for accurate bumblebee
conservation modeling.

Key distributions:
1. Exponential: Most trips are short-distance, exponential decay
2. Gamma: More realistic with peak at intermediate distances
3. Weibull: Flexible shape for different species/conditions
4. Truncated: Enforces maximum foraging range constraints

Based on:
- Osborne et al. (2008): Bumblebee flight distances in relation to landscape
- Knight et al. (2005): Landscape-scale influences on pollinator movements
- Greenleaf et al. (2007): Bee foraging ranges and their relationship to body size
- Carvell et al. (2017): Bumblebee foraging distances and habitat connectivity
"""

import numpy as np
from typing import Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging
from scipy import stats

from .species_parameters import LiteratureValidatedSpeciesParameters


class DistributionType(Enum):
    """Foraging distance distribution types"""

    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    WEIBULL = "weibull"
    TRUNCATED_NORMAL = "truncated_normal"
    CUSTOM = "custom"


class ForagingDistanceParameters(BaseModel):
    """Species-specific foraging distance distribution parameters"""

    model_config = {"validate_assignment": True}

    # Distribution type and core parameters
    distribution_type: DistributionType = Field(
        description="Type of distance distribution"
    )

    # Exponential parameters
    exponential_lambda: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=0.1,
        description="Exponential decay rate (1/mean_distance)",
    )

    # Gamma parameters
    gamma_shape: Optional[float] = Field(
        default=None, ge=0.5, le=5.0, description="Gamma distribution shape parameter"
    )
    gamma_scale: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=1000.0,
        description="Gamma distribution scale parameter (meters)",
    )

    # Weibull parameters
    weibull_shape: Optional[float] = Field(
        default=None, ge=0.5, le=3.0, description="Weibull distribution shape parameter"
    )
    weibull_scale: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=2000.0,
        description="Weibull distribution scale parameter (meters)",
    )

    # Truncated normal parameters
    normal_mean: Optional[float] = Field(
        default=None,
        ge=100.0,
        le=1500.0,
        description="Normal distribution mean (meters)",
    )
    normal_std: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=500.0,
        description="Normal distribution standard deviation",
    )

    # Universal constraints
    min_distance_m: float = Field(
        ge=10.0,
        le=100.0,
        default=25.0,
        description="Minimum foraging distance (meters)",
    )
    max_distance_m: float = Field(
        ge=200.0, le=5000.0, description="Maximum foraging distance (meters)"
    )

    # Probability adjustments
    short_distance_boost: float = Field(
        ge=1.0,
        le=3.0,
        default=1.0,
        description="Multiplier for distances < 200m (energy efficiency)",
    )
    long_distance_penalty: float = Field(
        ge=0.1,
        le=1.0,
        default=1.0,
        description="Multiplier for distances > 80% of max range",
    )


class EnvironmentalModifiers(BaseModel):
    """Environmental factors that modify foraging distance distributions"""

    model_config = {"validate_assignment": True}

    # Weather conditions
    temperature_c: float = Field(
        ge=-5.0, le=35.0, default=15.0, description="Air temperature (°C)"
    )
    wind_speed_ms: float = Field(
        ge=0.0, le=20.0, default=2.0, description="Wind speed (m/s)"
    )
    precipitation_mm: float = Field(
        ge=0.0, le=50.0, default=0.0, description="Precipitation in last hour (mm)"
    )

    # Resource availability
    local_resource_density: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Resource density within 200m of colony (0-1)",
    )
    regional_resource_quality: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Average resource quality at distance (0-1)",
    )

    # Colony state
    colony_energy_stores: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Colony energy reserves (0-1, affects risk tolerance)",
    )
    forager_experience: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Average forager experience level (0-1)",
    )


class ForagingDistanceDistribution:
    """
    Implements advanced foraging distance distributions for bumblebees.

    Provides species-specific distance sampling with environmental modifiers
    and biological constraints for realistic foraging behavior modeling.
    """

    def __init__(
        self,
        species_parameters: LiteratureValidatedSpeciesParameters,
        distribution_params: Optional[ForagingDistanceParameters] = None,
    ) -> None:
        self.species_parameters = species_parameters
        self.logger = logging.getLogger(__name__)

        # Initialize distribution parameters
        if distribution_params is None:
            self.distribution_params = self._create_species_default_distribution()
        else:
            self.distribution_params = distribution_params

        # Validate distribution parameters
        self._validate_distribution_parameters()

        # Initialize scipy distributions for performance
        self._base_distribution: Any = None  # Will be set in _initialize_distributions
        self._initialize_distributions()

        # Performance tracking
        self.distance_samples: list[float] = []
        self.rejection_rate = 0.0

    def _create_species_default_distribution(self) -> ForagingDistanceParameters:
        """Create species-specific default distribution parameters"""

        species_name = self.species_parameters.species_name
        max_range = self.species_parameters.foraging_range_m

        # Species-specific distribution assignments based on literature
        if species_name == "Bombus_terrestris":
            # Generalist, broad distance range with gamma distribution
            return ForagingDistanceParameters(
                distribution_type=DistributionType.GAMMA,
                gamma_shape=1.5,
                gamma_scale=max_range / 4.0,  # Peak at ~25% of max range
                max_distance_m=max_range,
                short_distance_boost=1.3,  # Efficient close-range foraging
                long_distance_penalty=0.8,
            )

        elif species_name == "Bombus_pascuorum":
            # Later-emerging, efficient forager with exponential preference for near
            return ForagingDistanceParameters(
                distribution_type=DistributionType.EXPONENTIAL,
                exponential_lambda=1.0 / (max_range * 0.3),  # Mean at 30% of max
                max_distance_m=max_range,
                short_distance_boost=1.5,  # Strong preference for nearby resources
                long_distance_penalty=0.6,
            )

        elif species_name == "Bombus_lapidarius":
            # Heat-adapted, more consistent distances with Weibull
            return ForagingDistanceParameters(
                distribution_type=DistributionType.WEIBULL,
                weibull_shape=2.0,  # More peaked distribution
                weibull_scale=max_range * 0.4,
                max_distance_m=max_range,
                short_distance_boost=1.1,
                long_distance_penalty=0.9,
            )

        else:
            # Default gamma distribution for other species
            return ForagingDistanceParameters(
                distribution_type=DistributionType.GAMMA,
                gamma_shape=1.8,
                gamma_scale=max_range / 5.0,
                max_distance_m=max_range,
                short_distance_boost=1.2,
                long_distance_penalty=0.8,
            )

    def _validate_distribution_parameters(self) -> None:
        """Validate distribution parameters are complete and consistent"""

        dist_type = self.distribution_params.distribution_type

        if dist_type == DistributionType.EXPONENTIAL:
            if self.distribution_params.exponential_lambda is None:
                raise ValueError("Exponential distribution requires lambda parameter")

        elif dist_type == DistributionType.GAMMA:
            if (
                self.distribution_params.gamma_shape is None
                or self.distribution_params.gamma_scale is None
            ):
                raise ValueError(
                    "Gamma distribution requires shape and scale parameters"
                )

        elif dist_type == DistributionType.WEIBULL:
            if (
                self.distribution_params.weibull_shape is None
                or self.distribution_params.weibull_scale is None
            ):
                raise ValueError(
                    "Weibull distribution requires shape and scale parameters"
                )

        elif dist_type == DistributionType.TRUNCATED_NORMAL:
            if (
                self.distribution_params.normal_mean is None
                or self.distribution_params.normal_std is None
            ):
                raise ValueError("Truncated normal requires mean and std parameters")

        # Validate max distance doesn't exceed species capability
        if (
            self.distribution_params.max_distance_m
            > self.species_parameters.foraging_range_m
        ):
            self.logger.warning(
                f"Distribution max distance ({self.distribution_params.max_distance_m}m) "
                f"exceeds species range ({self.species_parameters.foraging_range_m}m)"
            )

    def _initialize_distributions(self) -> None:
        """Initialize scipy distributions for efficient sampling"""

        dist_type = self.distribution_params.distribution_type

        try:
            if dist_type == DistributionType.EXPONENTIAL:
                lambda_param = self.distribution_params.exponential_lambda
                if lambda_param is not None:
                    scale = 1.0 / lambda_param
                    self._base_distribution = stats.expon(scale=scale)
                else:
                    raise ValueError("Exponential lambda parameter is required")

            elif dist_type == DistributionType.GAMMA:
                gamma_shape = self.distribution_params.gamma_shape
                gamma_scale = self.distribution_params.gamma_scale
                if gamma_shape is not None and gamma_scale is not None:
                    self._base_distribution = stats.gamma(
                        a=gamma_shape, scale=gamma_scale
                    )
                else:
                    raise ValueError("Gamma shape and scale parameters are required")

            elif dist_type == DistributionType.WEIBULL:
                weibull_shape = self.distribution_params.weibull_shape
                weibull_scale = self.distribution_params.weibull_scale
                if weibull_shape is not None and weibull_scale is not None:
                    self._base_distribution = stats.weibull_min(
                        c=weibull_shape, scale=weibull_scale
                    )
                else:
                    raise ValueError("Weibull shape and scale parameters are required")

            elif dist_type == DistributionType.TRUNCATED_NORMAL:
                mean = self.distribution_params.normal_mean
                std = self.distribution_params.normal_std
                if mean is not None and std is not None:
                    min_val = self.distribution_params.min_distance_m
                    max_val = self.distribution_params.max_distance_m

                    # Convert to standard form for truncnorm
                    a = (min_val - mean) / std
                    b = (max_val - mean) / std
                    self._base_distribution = stats.truncnorm(
                        a=a, b=b, loc=mean, scale=std
                    )
                else:
                    raise ValueError("Normal mean and std parameters are required")

        except Exception as e:
            self.logger.error(f"Failed to initialize distribution: {e}")
            # Fallback to exponential
            self._base_distribution = stats.expon(scale=200.0)

    def sample_foraging_distance(
        self,
        environmental_modifiers: Optional[EnvironmentalModifiers] = None,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Sample foraging distances with environmental and biological constraints.

        Args:
            environmental_modifiers: Environmental conditions affecting distances
            n_samples: Number of distance samples to generate

        Returns:
            Array of foraging distances in meters
        """

        if environmental_modifiers is None:
            environmental_modifiers = EnvironmentalModifiers()

        # Generate base samples from distribution
        base_samples = self._generate_base_samples(n_samples)

        # Apply environmental modifiers
        modified_samples = self._apply_environmental_modifiers(
            base_samples, environmental_modifiers
        )

        # Apply biological constraints and adjustments
        final_samples = self._apply_biological_constraints(modified_samples)

        # Track samples for analysis
        self.distance_samples.extend(final_samples.tolist())

        return final_samples

    def _generate_base_samples(self, n_samples: int) -> np.ndarray:
        """Generate base samples from the configured distribution"""

        max_attempts = n_samples * 5  # Prevent infinite loops
        samples: list[float] = []
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            # Generate candidate samples
            candidates = self._base_distribution.rvs(size=min(n_samples * 2, 1000))

            # Apply basic constraints
            valid_candidates = candidates[
                (candidates >= self.distribution_params.min_distance_m)
                & (candidates <= self.distribution_params.max_distance_m)
            ]

            samples.extend(valid_candidates[: n_samples - len(samples)])
            attempts += len(candidates)

        # Calculate rejection rate for monitoring
        if attempts > 0:
            self.rejection_rate = 1.0 - (len(samples) / attempts)

        # Fill any remaining with fallback values if needed
        while len(samples) < n_samples:
            fallback_distance = np.random.uniform(
                self.distribution_params.min_distance_m,
                min(500.0, self.distribution_params.max_distance_m),
            )
            samples.append(fallback_distance)

        return np.array(samples[:n_samples])

    def _apply_environmental_modifiers(
        self, distances: np.ndarray, modifiers: EnvironmentalModifiers
    ) -> np.ndarray:
        """Apply environmental conditions to modify foraging distances"""

        modified_distances = distances.copy()

        # Temperature effects
        temp_effect = self._calculate_temperature_effect(modifiers.temperature_c)
        modified_distances *= temp_effect

        # Wind effects (reduce long-distance foraging in high winds)
        wind_effect = self._calculate_wind_effect(modifiers.wind_speed_ms, distances)
        modified_distances *= wind_effect

        # Precipitation effects (reduce all foraging distances)
        if modifiers.precipitation_mm > 0.5:
            precip_effect = max(0.3, 1.0 - modifiers.precipitation_mm * 0.1)
            modified_distances *= precip_effect

        # Resource availability effects
        resource_effect = self._calculate_resource_effect(
            modifiers.local_resource_density,
            modifiers.regional_resource_quality,
            distances,
        )
        modified_distances *= resource_effect

        # Colony state effects
        energy_effect = self._calculate_energy_effect(
            modifiers.colony_energy_stores, modifiers.forager_experience, distances
        )
        modified_distances *= energy_effect

        return modified_distances

    def _calculate_temperature_effect(self, temperature_c: float) -> float:
        """Calculate temperature effect on foraging distance"""

        optimal_temp = 20.0  # Optimal foraging temperature
        min_temp = self.species_parameters.temperature_tolerance_min_c

        if temperature_c < min_temp:
            return 0.1  # Very limited foraging below minimum
        elif temperature_c < optimal_temp:
            # Linear increase from min to optimal
            return 0.4 + 0.6 * ((temperature_c - min_temp) / (optimal_temp - min_temp))
        else:
            # Slight decrease in very hot conditions
            return max(0.7, 1.0 - (temperature_c - optimal_temp) * 0.02)

    def _calculate_wind_effect(
        self, wind_speed_ms: float, distances: np.ndarray
    ) -> np.ndarray:
        """Calculate wind effect on foraging distances (stronger effect on longer distances)"""

        if wind_speed_ms < 3.0:
            return np.ones_like(distances)  # No effect in light winds

        # Wind effect increases with distance (flight becomes more difficult)
        wind_factor = 1.0 - (wind_speed_ms - 3.0) * 0.05
        distance_factor = (
            1.0 - (distances / self.distribution_params.max_distance_m) * 0.3
        )

        return np.maximum(0.2, wind_factor * distance_factor)

    def _calculate_resource_effect(
        self, local_density: float, regional_quality: float, distances: np.ndarray
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Calculate resource availability effect on foraging distances"""

        # High local density reduces long-distance foraging
        if local_density > 0.7:
            # Strong local resources - prefer short distances
            local_effect = 2.0 - distances / 200.0  # Boost short, reduce long
            return np.array(
                np.maximum(0.3, np.minimum(2.0, local_effect)), dtype=np.float64
            )

        # Low local density with good regional quality increases long-distance foraging
        elif local_density < 0.3 and regional_quality > 0.6:
            # Incentivize longer foraging
            result = 1.0 + (distances / self.distribution_params.max_distance_m) * 0.5
            return np.array(result, dtype=np.float64)

        else:
            # Moderate conditions - minimal effect
            return np.array(np.ones_like(distances), dtype=np.float64)

    def _calculate_energy_effect(
        self, energy_stores: float, experience: float, distances: np.ndarray
    ) -> np.ndarray:
        """Calculate colony energy and forager experience effects"""

        # Low energy stores reduce long-distance foraging (risk aversion)
        if energy_stores < 0.3:
            energy_penalty = (
                1.0 - (distances / self.distribution_params.max_distance_m) * 0.4
            )
            energy_penalty = np.maximum(0.5, energy_penalty)
        else:
            energy_penalty = np.ones_like(distances)

        # High experience enables more efficient long-distance foraging
        if experience > 0.7:
            experience_bonus = (
                1.0 + (distances / self.distribution_params.max_distance_m) * 0.2
            )
        else:
            experience_bonus = np.ones_like(distances)

        return energy_penalty * experience_bonus

    def _apply_biological_constraints(self, distances: np.ndarray) -> np.ndarray:
        """Apply final biological constraints and preference adjustments"""

        # Ensure within species maximum range
        distances = np.minimum(distances, self.distribution_params.max_distance_m)
        distances = np.maximum(distances, self.distribution_params.min_distance_m)

        # Apply short-distance boost (energy efficiency)
        short_distance_mask = distances < 200.0
        distances[short_distance_mask] *= self.distribution_params.short_distance_boost

        # Apply long-distance penalty (increased risk/energy cost)
        long_distance_threshold = self.distribution_params.max_distance_m * 0.8
        long_distance_mask = distances > long_distance_threshold
        distances[long_distance_mask] *= self.distribution_params.long_distance_penalty

        # Final constraint check
        distances = np.clip(
            distances,
            self.distribution_params.min_distance_m,
            self.distribution_params.max_distance_m,
        )

        return distances

    def get_distance_statistics(self) -> Dict[str, Union[float, int, str]]:
        """Get statistics of sampled distances"""

        if not self.distance_samples:
            return {"error": "No distance samples available", "sample_count": 0}

        samples = np.array(self.distance_samples)

        return {
            "mean_distance_m": float(np.mean(samples)),
            "median_distance_m": float(np.median(samples)),
            "std_distance_m": float(np.std(samples)),
            "min_distance_m": float(np.min(samples)),
            "max_distance_m": float(np.max(samples)),
            "quartile_25_m": float(np.percentile(samples, 25)),
            "quartile_75_m": float(np.percentile(samples, 75)),
            "sample_count": len(samples),
            "rejection_rate": self.rejection_rate,
            "distribution_type": self.distribution_params.distribution_type.value,
        }

    def validate_against_field_data(
        self,
        field_mean: float,
        field_std: float,
        field_max: float,
        tolerance: float = 0.2,
    ) -> Dict[str, Any]:
        """Validate distribution against field data"""

        if not self.distance_samples:
            # Generate validation sample
            validation_samples = self.sample_foraging_distance(n_samples=1000)
        else:
            validation_samples = np.array(
                self.distance_samples[-1000:]
            )  # Use recent samples

        model_mean = np.mean(validation_samples)
        model_std = np.std(validation_samples)
        model_max = np.max(validation_samples)

        # Check if model statistics are within tolerance of field data
        mean_valid = abs(model_mean - field_mean) / field_mean <= tolerance
        std_valid = abs(model_std - field_std) / field_std <= tolerance
        max_valid = model_max <= field_max * (1 + tolerance)

        return {
            "validation_passed": bool(mean_valid and std_valid and max_valid),
            "mean_comparison": {
                "field": field_mean,
                "model": model_mean,
                "relative_error": abs(model_mean - field_mean) / field_mean,
                "valid": mean_valid,
            },
            "std_comparison": {
                "field": field_std,
                "model": model_std,
                "relative_error": abs(model_std - field_std) / field_std,
                "valid": std_valid,
            },
            "max_comparison": {
                "field": field_max,
                "model": model_max,
                "within_bounds": model_max <= field_max * (1 + tolerance),
                "valid": max_valid,
            },
        }

    def get_distribution_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of distribution configuration"""

        return {
            "species_name": self.species_parameters.species_name,
            "distribution_type": self.distribution_params.distribution_type.value,
            "parameters": {
                "min_distance_m": self.distribution_params.min_distance_m,
                "max_distance_m": self.distribution_params.max_distance_m,
                "short_distance_boost": self.distribution_params.short_distance_boost,
                "long_distance_penalty": self.distribution_params.long_distance_penalty,
            },
            "distribution_specific": self._get_distribution_specific_params(),
            "species_constraints": {
                "foraging_range_m": self.species_parameters.foraging_range_m,
                "body_size_mm": self.species_parameters.body_size_mm,
                "temperature_tolerance_min_c": self.species_parameters.temperature_tolerance_min_c,
            },
            "performance_metrics": {
                "samples_generated": len(self.distance_samples),
                "rejection_rate": self.rejection_rate,
            },
        }

    def _get_distribution_specific_params(self) -> Dict[str, Any]:
        """Get parameters specific to the current distribution type"""

        dist_type = self.distribution_params.distribution_type

        if dist_type == DistributionType.EXPONENTIAL:
            return {
                "lambda": self.distribution_params.exponential_lambda,
                "mean_distance": 1.0 / self.distribution_params.exponential_lambda
                if self.distribution_params.exponential_lambda
                else None,
            }
        elif dist_type == DistributionType.GAMMA:
            return {
                "shape": self.distribution_params.gamma_shape,
                "scale": self.distribution_params.gamma_scale,
                "mean_distance": self.distribution_params.gamma_shape
                * self.distribution_params.gamma_scale
                if self.distribution_params.gamma_shape
                and self.distribution_params.gamma_scale
                else None,
            }
        elif dist_type == DistributionType.WEIBULL:
            return {
                "shape": self.distribution_params.weibull_shape,
                "scale": self.distribution_params.weibull_scale,
            }
        elif dist_type == DistributionType.TRUNCATED_NORMAL:
            return {
                "mean": self.distribution_params.normal_mean,
                "std": self.distribution_params.normal_std,
            }
        else:
            return {}


# Factory function for easy species-specific distribution creation
def create_species_distance_distribution(
    species_parameters: LiteratureValidatedSpeciesParameters,
    distribution_type: Optional[DistributionType] = None,
) -> ForagingDistanceDistribution:
    """Create species-appropriate foraging distance distribution"""

    if distribution_type is not None:
        # Override default with specified distribution type
        dist_params = ForagingDistanceParameters(
            distribution_type=distribution_type,
            max_distance_m=species_parameters.foraging_range_m,
        )
        # Set default parameters based on type
        if distribution_type == DistributionType.EXPONENTIAL:
            dist_params.exponential_lambda = 1.0 / (
                species_parameters.foraging_range_m * 0.3
            )
        elif distribution_type == DistributionType.GAMMA:
            dist_params.gamma_shape = 1.5
            dist_params.gamma_scale = species_parameters.foraging_range_m / 4.0
        elif distribution_type == DistributionType.WEIBULL:
            dist_params.weibull_shape = 2.0
            dist_params.weibull_scale = species_parameters.foraging_range_m * 0.4

        return ForagingDistanceDistribution(species_parameters, dist_params)
    else:
        # Use species-specific defaults
        return ForagingDistanceDistribution(species_parameters)


# Export for use in other modules
__all__ = [
    "DistributionType",
    "ForagingDistanceParameters",
    "EnvironmentalModifiers",
    "ForagingDistanceDistribution",
    "create_species_distance_distribution",
]
