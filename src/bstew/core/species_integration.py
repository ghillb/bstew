"""
Species Parameter Integration with Bumblebee Systems
====================================================

CRITICAL: This integrates literature-validated species parameters with
the biologically accurate bumblebee communication, colony lifecycle,
and foraging systems.

Ensures all bumblebee systems use scientifically validated parameters
for accurate conservation research modeling.
"""

from typing import Dict, Any, Optional
import logging

from .species_parameters import (
    create_literature_validated_species,
    BumblebeeParameterValidator,
    LiteratureValidatedSpeciesParameters,
)
from .bumblebee_communication import (
    BumblebeeCommunicationModel,
    BumblebeeCommunicationSystem,
)
from .bumblebee_colony_lifecycle import BumblebeeColonyLifecycleModel, BumblebeeColony
from .bumblebee_recruitment_mechanisms import BumblebeeRecruitmentModel
from .foraging_distances import (
    ForagingDistanceDistribution,
    create_species_distance_distribution,
    DistributionType,
)


class SpeciesParameterIntegrator:
    """Integrates species-specific parameters with bumblebee systems"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Load literature-validated species parameters
        self.species_parameters = create_literature_validated_species()
        self.parameter_validator = BumblebeeParameterValidator()

        # Cache for system configurations
        self._communication_configs: Dict[str, BumblebeeCommunicationModel] = {}
        self._lifecycle_configs: Dict[str, BumblebeeColonyLifecycleModel] = {}
        self._recruitment_configs: Dict[str, BumblebeeRecruitmentModel] = {}
        self._distance_distributions: Dict[str, ForagingDistanceDistribution] = {}

    def get_species_communication_config(
        self, species_name: str
    ) -> BumblebeeCommunicationModel:
        """Get species-specific communication system configuration"""

        if species_name in self._communication_configs:
            return self._communication_configs[species_name]

        if species_name not in self.species_parameters:
            self.logger.warning(
                f"Species {species_name} not found, using default B. terrestris parameters"
            )
            species_name = "Bombus_terrestris"

        params = self.species_parameters[species_name]

        # Create species-specific communication configuration
        config = BumblebeeCommunicationModel(
            # Individual memory parameters based on species
            memory_decay_rate=self._calculate_memory_decay_rate(params),
            memory_capacity=params.memory_capacity_patches,
            # Scent marking parameters (species-specific)
            scent_mark_probability=self._calculate_scent_mark_probability(params),
            scent_detection_range=self._calculate_scent_detection_range(params),
            scent_decay_rate=0.2,  # Standard for all bumblebees
            # Social arousal parameters (minimal for all bumblebees)
            nestmate_arousal_probability=params.social_recruitment_rate,
            arousal_radius=1.0,  # Standard for all species
            # Exploration parameters (species-specific)
            individual_exploration_rate=min(0.95, 1.0 - params.social_recruitment_rate),
            patch_fidelity_strength=params.patch_fidelity_strength,
        )

        self._communication_configs[species_name] = config
        return config

    def get_species_lifecycle_config(
        self, species_name: str
    ) -> BumblebeeColonyLifecycleModel:
        """Get species-specific colony lifecycle configuration"""

        if species_name in self._lifecycle_configs:
            return self._lifecycle_configs[species_name]

        if species_name not in self.species_parameters:
            self.logger.warning(
                f"Species {species_name} not found, using default B. terrestris parameters"
            )
            species_name = "Bombus_terrestris"

        params = self.species_parameters[species_name]

        # Create species-specific lifecycle configuration
        config = BumblebeeColonyLifecycleModel(
            # Timing parameters based on species phenology
            hibernation_end_day=max(60, min(120, params.active_season_start - 10)),
            colony_founding_duration=30,  # Standard for all species
            worker_production_peak_day=max(
                120, min(180, params.emerging_day_mean + 60)
            ),
            reproduction_start_day=max(150, min(210, params.emerging_day_mean + 120)),
            colony_decline_start_day=max(210, min(270, params.active_season_end - 40)),
            hibernation_start_day=max(270, min(330, params.active_season_end)),
            # Population parameters (species-specific)
            max_colony_size=params.max_colony_size,
            foundress_mortality_rate=self._calculate_foundress_mortality(params),
            worker_development_time=int(params.brood_development_time),
            male_development_time=int(params.brood_development_time) + 3,
            gyne_development_time=int(params.brood_development_time) + 4,
            # Reproductive parameters
            foundress_egg_laying_rate=self._calculate_egg_laying_rate(params),
            peak_egg_laying_rate=self._calculate_peak_egg_laying_rate(params),
            switch_point_trigger=0.7,  # Standard for all species
            # Survival parameters (species-specific)
            worker_daily_mortality=self._calculate_worker_mortality(params),
            winter_queen_survival=self._calculate_winter_survival(params),
        )

        self._lifecycle_configs[species_name] = config
        return config

    def get_species_recruitment_config(
        self, species_name: str
    ) -> BumblebeeRecruitmentModel:
        """Get species-specific recruitment mechanism configuration"""

        if species_name in self._recruitment_configs:
            return self._recruitment_configs[species_name]

        if species_name not in self.species_parameters:
            self.logger.warning(
                f"Species {species_name} not found, using default B. terrestris parameters"
            )
            species_name = "Bombus_terrestris"

        params = self.species_parameters[species_name]

        # Create species-specific recruitment configuration
        config = BumblebeeRecruitmentModel(
            # Social influence parameters (very low for all bumblebees)
            nest_arousal_probability=params.social_recruitment_rate,
            arousal_radius=1.0,
            max_aroused_bees=self._calculate_max_aroused_bees(params),
            # Chemical communication parameters (limited)
            scent_following_probability=params.social_recruitment_rate * 0.6,
            scent_information_accuracy=0.2,  # Low for all bumblebees
            # Individual decision dominance (species-specific)
            individual_decision_weight=min(
                1.0, max(0.8, 1.0 - params.social_recruitment_rate)
            ),
            # Recruitment success rates (species-specific)
            base_recruitment_success=params.social_recruitment_rate
            * 3.0,  # Scale up base success
        )

        self._recruitment_configs[species_name] = config
        return config

    def create_species_communication_system(
        self, species_name: str
    ) -> BumblebeeCommunicationSystem:
        """Create configured communication system for species"""

        config = self.get_species_communication_config(species_name)
        return BumblebeeCommunicationSystem(model=config)

    def create_species_colony(
        self, species_name: str, colony_id: str, founded_day: int = 0
    ) -> BumblebeeColony:
        """Create configured colony for species"""

        lifecycle_config = self.get_species_lifecycle_config(species_name)

        return BumblebeeColony(
            colony_id=colony_id,
            species=species_name,
            founded_day=founded_day,
            lifecycle_model=lifecycle_config,
        )

    def get_species_distance_distribution(
        self, species_name: str, distribution_type: Optional[DistributionType] = None
    ) -> ForagingDistanceDistribution:
        """Get species-specific foraging distance distribution"""

        cache_key = f"{species_name}_{distribution_type.value if distribution_type else 'default'}"

        if cache_key in self._distance_distributions:
            return self._distance_distributions[cache_key]

        if species_name not in self.species_parameters:
            self.logger.warning(
                f"Species {species_name} not found, using default B. terrestris parameters"
            )
            species_name = "Bombus_terrestris"

        params = self.species_parameters[species_name]

        # Create species-specific distance distribution
        distribution = create_species_distance_distribution(params, distribution_type)

        self._distance_distributions[cache_key] = distribution
        return distribution

    def get_species_foraging_parameters(self, species_name: str) -> Dict[str, Any]:
        """Get species-specific foraging parameters for integration"""

        if species_name not in self.species_parameters:
            self.logger.warning(
                f"Species {species_name} not found, using default B. terrestris parameters"
            )
            species_name = "Bombus_terrestris"

        params = self.species_parameters[species_name]

        return {
            "max_foraging_distance_m": params.foraging_range_m,
            "flight_velocity_ms": params.flight_velocity_ms,
            "nectar_capacity_mg": params.nectar_load_capacity_mg,
            "pollen_capacity_mg": params.pollen_load_capacity_mg,
            "min_temperature_c": params.temperature_tolerance_min_c,
            "memory_capacity": params.memory_capacity_patches,
            "patch_fidelity": params.patch_fidelity_strength,
            "social_recruitment_rate": params.social_recruitment_rate,
            "proboscis_length_mm": params.proboscis_characteristics.length_mm,
            "body_size_mm": params.body_size_mm,
            "competition_strength": params.competition_strength,
            "distance_distribution": self.get_species_distance_distribution(
                species_name
            ),
        }

    def validate_species_configuration(self, species_name: str) -> Dict[str, Any]:
        """Validate species configuration against literature"""

        if species_name not in self.species_parameters:
            return {
                "valid": False,
                "error": f"Species {species_name} not found in literature database",
            }

        params = self.species_parameters[species_name]

        # Generate validation report
        validation_report = self.parameter_validator.generate_validation_report(
            species_name, params
        )

        # Add system integration checks
        config_checks = self._validate_system_integration(species_name)

        return {
            "valid": validation_report["validation_success_rate"] == 1.0,
            "species_name": species_name,
            "literature_validation": validation_report,
            "system_integration": config_checks,
            "honey_bee_comparison": self.parameter_validator.compare_to_honey_bees(),
        }

    def _validate_system_integration(self, species_name: str) -> Dict[str, Any]:
        """Validate integration with bumblebee systems"""

        checks: Dict[str, Any] = {}

        try:
            # Test communication system creation
            comm_system = self.create_species_communication_system(species_name)
            checks["communication_system"] = {
                "valid": True,
                "memory_capacity": comm_system.model.memory_capacity,
                "social_recruitment": comm_system.model.nestmate_arousal_probability,
            }
        except Exception as e:
            checks["communication_system"] = {"valid": False, "error": str(e)}

        try:
            # Test colony creation
            colony = self.create_species_colony(species_name, "test_colony")
            checks["colony_lifecycle"] = {
                "valid": True,
                "max_colony_size": colony.lifecycle_model.max_colony_size,
                "hibernation_end": colony.lifecycle_model.hibernation_end_day,
            }
        except Exception as e:
            checks["colony_lifecycle"] = {"valid": False, "error": str(e)}

        try:
            # Test foraging parameters
            foraging_params = self.get_species_foraging_parameters(species_name)
            checks["foraging_parameters"] = {
                "valid": True,
                "foraging_range_m": foraging_params["max_foraging_distance_m"],
                "temperature_tolerance": foraging_params["min_temperature_c"],
            }
        except Exception as e:
            checks["foraging_parameters"] = {"valid": False, "error": str(e)}

        return checks

    def get_species_summary(self, species_name: str) -> Dict[str, Any]:
        """Get comprehensive species summary"""

        if species_name not in self.species_parameters:
            return {"error": f"Species {species_name} not found"}

        params = self.species_parameters[species_name]
        validation = self.validate_species_configuration(species_name)

        return {
            "species_name": species_name,
            "common_name": self._get_common_name(species_name),
            "biological_parameters": {
                "proboscis_length_mm": params.proboscis_characteristics.length_mm,
                "foraging_range_m": params.foraging_range_m,
                "max_colony_size": params.max_colony_size,
                "temperature_tolerance_min_c": params.temperature_tolerance_min_c,
                "memory_capacity_patches": params.memory_capacity_patches,
                "social_recruitment_rate": params.social_recruitment_rate,
                "patch_fidelity_strength": params.patch_fidelity_strength,
            },
            "phenology": {
                "emergence_day": params.emerging_day_mean,
                "active_season_start": params.active_season_start,
                "active_season_end": params.active_season_end,
                "season_length_days": params.active_season_end
                - params.active_season_start,
            },
            "ecological_traits": {
                "cold_tolerance": params.cold_tolerance,
                "drought_tolerance": params.drought_tolerance,
                "competition_strength": params.competition_strength,
                "social_dominance": params.social_dominance,
            },
            "validation": validation,
            "system_notes": self.parameter_validator._get_species_notes(species_name),
        }

    def _get_common_name(self, species_name: str) -> str:
        """Get common name for species"""

        common_names = {
            "Bombus_terrestris": "Buff-tailed bumblebee",
            "Bombus_pascuorum": "Common carder bee",
            "Bombus_lapidarius": "Red-tailed bumblebee",
        }

        return common_names.get(species_name, species_name)

    def _calculate_memory_decay_rate(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> float:
        """Calculate memory decay rate based on species characteristics"""

        # Higher site fidelity = slower memory decay
        base_decay = 0.05
        fidelity_factor = 1.0 - params.site_fidelity
        return base_decay * (0.5 + fidelity_factor)

    def _calculate_scent_mark_probability(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> float:
        """Calculate scent marking probability based on species"""

        # Social species mark more frequently
        return params.social_recruitment_rate * 3.0

    def _calculate_scent_detection_range(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> float:
        """Calculate scent detection range based on body size"""

        # Larger bees can detect scents from further away
        base_range = 2.0
        size_factor = params.body_size_mm / 20.0  # Normalize to 20mm
        return base_range * size_factor

    def _calculate_foundress_mortality(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> float:
        """Calculate foundress mortality rate"""

        # Higher cold tolerance = lower mortality
        base_mortality = 0.8
        cold_factor = params.cold_tolerance
        return base_mortality * (1.0 - cold_factor * 0.3)

    def _calculate_egg_laying_rate(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> float:
        """Calculate foundress egg laying rate"""

        # Larger species lay more eggs
        base_rate = 8.0
        size_factor = params.body_size_mm / 20.0
        return base_rate * size_factor

    def _calculate_peak_egg_laying_rate(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> float:
        """Calculate peak egg laying rate"""

        return self._calculate_egg_laying_rate(params) * 3.0

    def _calculate_worker_mortality(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> float:
        """Calculate worker daily mortality rate"""

        # Base on worker lifespan
        base_mortality = 0.02
        lifespan_factor = 35.0 / params.max_lifespan_workers  # Normalize to 35 days
        return base_mortality * lifespan_factor

    def _calculate_winter_survival(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> float:
        """Calculate winter queen survival rate"""

        # Cold tolerance affects winter survival
        base_survival = 0.1
        cold_factor = params.cold_tolerance
        return base_survival * (0.5 + cold_factor)

    def _calculate_max_aroused_bees(
        self, params: LiteratureValidatedSpeciesParameters
    ) -> int:
        """Calculate maximum number of bees aroused in recruitment event"""

        # Social dominance affects recruitment strength
        base_aroused = 2
        dominance_factor = params.social_dominance
        return max(1, int(base_aroused * dominance_factor))


# Global instance for easy access
species_integrator = SpeciesParameterIntegrator()


# Export for use in other modules
__all__ = [
    "SpeciesParameterIntegrator",
    "species_integrator",
]
