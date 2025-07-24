"""
Tests for Species Parameter Integration with Bumblebee Systems
============================================================

Tests ensuring literature-validated species parameters are correctly
integrated with bumblebee communication, colony lifecycle, and
recruitment systems.
"""

import pytest
from typing import Dict, Any

from bstew.core.species_integration import (
    SpeciesParameterIntegrator,
    species_integrator,
)
from bstew.core.bumblebee_communication import BumblebeeCommunicationModel, BumblebeeCommunicationSystem
from bstew.core.bumblebee_colony_lifecycle import BumblebeeColonyLifecycleModel, BumblebeeColony
from bstew.core.bumblebee_recruitment_mechanisms import BumblebeeRecruitmentModel


class TestSpeciesParameterIntegrator:
    """Test species parameter integration"""

    def test_integrator_initialization(self):
        """Test integrator initializes with species parameters"""

        integrator = SpeciesParameterIntegrator()

        assert len(integrator.species_parameters) == 3
        assert "Bombus_terrestris" in integrator.species_parameters
        assert "Bombus_pascuorum" in integrator.species_parameters
        assert "Bombus_lapidarius" in integrator.species_parameters
        assert integrator.parameter_validator is not None

    def test_communication_config_creation(self):
        """Test species-specific communication configuration"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            config = integrator.get_species_communication_config(species_name)

            assert isinstance(config, BumblebeeCommunicationModel)

            # Check species-specific parameters
            params = integrator.species_parameters[species_name]
            assert config.memory_capacity == params.memory_capacity_patches
            assert config.nestmate_arousal_probability == params.social_recruitment_rate
            assert config.patch_fidelity_strength == params.patch_fidelity_strength

            # Check bumblebee-specific constraints
            assert config.memory_capacity <= 20  # Biological limit
            assert config.nestmate_arousal_probability < 0.1  # <10% for bumblebees
            assert config.individual_exploration_rate > 0.9  # >90% individual decisions

    def test_lifecycle_config_creation(self):
        """Test species-specific lifecycle configuration"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            config = integrator.get_species_lifecycle_config(species_name)

            assert isinstance(config, BumblebeeColonyLifecycleModel)

            # Check species-specific parameters
            params = integrator.species_parameters[species_name]
            assert config.max_colony_size == params.max_colony_size
            # Hibernation days are bounded within validation ranges
            assert 60 <= config.hibernation_end_day <= 120
            assert 270 <= config.hibernation_start_day <= 330

            # Check bumblebee-specific constraints
            assert config.max_colony_size < 500  # Much smaller than honey bees
            assert config.hibernation_end_day < config.hibernation_start_day  # Annual cycle

    def test_recruitment_config_creation(self):
        """Test species-specific recruitment configuration"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            config = integrator.get_species_recruitment_config(species_name)

            assert isinstance(config, BumblebeeRecruitmentModel)

            # Check species-specific parameters
            params = integrator.species_parameters[species_name]
            assert config.nest_arousal_probability == params.social_recruitment_rate
            assert config.individual_decision_weight == 1.0 - params.social_recruitment_rate

            # Check bumblebee-specific constraints
            assert config.nest_arousal_probability < 0.1  # <10% for bumblebees
            assert config.individual_decision_weight > 0.9  # >90% individual decisions
            assert config.max_aroused_bees <= 5  # Small recruitment events

    def test_communication_system_creation(self):
        """Test creation of configured communication systems"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            comm_system = integrator.create_species_communication_system(species_name)

            assert isinstance(comm_system, BumblebeeCommunicationSystem)

            # Check configuration is applied
            params = integrator.species_parameters[species_name]
            assert comm_system.model.memory_capacity == params.memory_capacity_patches
            assert comm_system.model.nestmate_arousal_probability == params.social_recruitment_rate

    def test_colony_creation(self):
        """Test creation of configured colonies"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            colony = integrator.create_species_colony(species_name, f"test_colony_{species_name}")

            assert isinstance(colony, BumblebeeColony)
            assert colony.species == species_name
            assert colony.colony_id == f"test_colony_{species_name}"

            # Check configuration is applied
            params = integrator.species_parameters[species_name]
            assert colony.lifecycle_model.max_colony_size == params.max_colony_size

    def test_foraging_parameters_extraction(self):
        """Test extraction of foraging parameters"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            foraging_params = integrator.get_species_foraging_parameters(species_name)

            assert isinstance(foraging_params, dict)

            # Check required parameters
            required_params = [
                "max_foraging_distance_m",
                "flight_velocity_ms",
                "min_temperature_c",
                "memory_capacity",
                "patch_fidelity",
                "social_recruitment_rate",
                "proboscis_length_mm",
            ]

            for param in required_params:
                assert param in foraging_params

            # Check parameter ranges
            assert 200 <= foraging_params["max_foraging_distance_m"] <= 2000
            assert 2.5 <= foraging_params["flight_velocity_ms"] <= 4.0
            assert -2 <= foraging_params["min_temperature_c"] <= 15
            assert 5 <= foraging_params["memory_capacity"] <= 20
            assert 0 <= foraging_params["social_recruitment_rate"] < 0.1
            assert 6 <= foraging_params["proboscis_length_mm"] <= 12

    def test_unknown_species_handling(self):
        """Test handling of unknown species"""

        integrator = SpeciesParameterIntegrator()

        # Should default to B. terrestris for unknown species
        config = integrator.get_species_communication_config("Unknown_species")
        terrestris_config = integrator.get_species_communication_config("Bombus_terrestris")

        assert config.memory_capacity == terrestris_config.memory_capacity
        assert config.nestmate_arousal_probability == terrestris_config.nestmate_arousal_probability

    def test_species_validation(self):
        """Test species configuration validation"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            validation = integrator.validate_species_configuration(species_name)

            assert isinstance(validation, dict)
            assert validation["valid"] is True
            assert validation["species_name"] == species_name
            assert "literature_validation" in validation
            assert "system_integration" in validation
            assert "honey_bee_comparison" in validation

            # Check system integration tests passed
            integration = validation["system_integration"]
            assert integration["communication_system"]["valid"] is True
            assert integration["colony_lifecycle"]["valid"] is True
            assert integration["foraging_parameters"]["valid"] is True

    def test_species_summary(self):
        """Test comprehensive species summary"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            summary = integrator.get_species_summary(species_name)

            assert isinstance(summary, dict)
            assert summary["species_name"] == species_name
            assert "common_name" in summary
            assert "biological_parameters" in summary
            assert "phenology" in summary
            assert "ecological_traits" in summary
            assert "validation" in summary
            assert "system_notes" in summary

            # Check biological parameters
            bio_params = summary["biological_parameters"]
            assert 6 <= bio_params["proboscis_length_mm"] <= 12
            assert 200 <= bio_params["foraging_range_m"] <= 2000
            assert 80 <= bio_params["max_colony_size"] <= 400
            assert bio_params["social_recruitment_rate"] < 0.1

            # Check phenology makes sense
            phenology = summary["phenology"]
            assert phenology["active_season_start"] < phenology["active_season_end"]
            assert phenology["season_length_days"] > 0


class TestSpeciesComparisons:
    """Test comparisons between species"""

    def test_species_biological_differences(self):
        """Test that species show correct biological differences"""

        integrator = SpeciesParameterIntegrator()

        terrestris_summary = integrator.get_species_summary("Bombus_terrestris")
        pascuorum_summary = integrator.get_species_summary("Bombus_pascuorum")
        lapidarius_summary = integrator.get_species_summary("Bombus_lapidarius")

        # Proboscis length differences
        terrestris_proboscis = terrestris_summary["biological_parameters"]["proboscis_length_mm"]
        pascuorum_proboscis = pascuorum_summary["biological_parameters"]["proboscis_length_mm"]
        lapidarius_proboscis = lapidarius_summary["biological_parameters"]["proboscis_length_mm"]

        assert terrestris_proboscis < pascuorum_proboscis  # B. pascuorum is medium-tongued
        assert terrestris_proboscis < lapidarius_proboscis  # B. lapidarius slightly longer
        assert lapidarius_proboscis < pascuorum_proboscis  # B. pascuorum longest

        # Colony size differences
        terrestris_colony = terrestris_summary["biological_parameters"]["max_colony_size"]
        pascuorum_colony = pascuorum_summary["biological_parameters"]["max_colony_size"]
        lapidarius_colony = lapidarius_summary["biological_parameters"]["max_colony_size"]

        assert terrestris_colony > pascuorum_colony  # B. terrestris has largest colonies
        assert terrestris_colony > lapidarius_colony

        # Emergence timing differences
        terrestris_emergence = terrestris_summary["phenology"]["emergence_day"]
        pascuorum_emergence = pascuorum_summary["phenology"]["emergence_day"]
        lapidarius_emergence = lapidarius_summary["phenology"]["emergence_day"]

        assert terrestris_emergence < lapidarius_emergence < pascuorum_emergence  # Emergence order

    def test_species_vs_honey_bee_distinctions(self):
        """Test that all species maintain bumblebee characteristics vs honey bees"""

        integrator = SpeciesParameterIntegrator()
        honey_bee_comparison = integrator.parameter_validator.compare_to_honey_bees()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            summary = integrator.get_species_summary(species_name)
            bio_params = summary["biological_parameters"]

            # Foraging range: bumblebees much shorter than honey bees
            assert bio_params["foraging_range_m"] < 2000  # vs 5000-6000m for honey bees

            # Colony size: bumblebees much smaller than honey bees
            assert bio_params["max_colony_size"] < 500  # vs 20,000-80,000 for honey bees

            # Social recruitment: bumblebees <5% vs honey bees 30-70%
            assert bio_params["social_recruitment_rate"] < 0.1

            # Individual memory: bumblebees limited vs honey bees unlimited sharing
            assert bio_params["memory_capacity_patches"] <= 20

            # Temperature tolerance: bumblebees work at lower temps
            assert bio_params["temperature_tolerance_min_c"] < 15  # vs 12°C for honey bees


class TestGlobalIntegratorInstance:
    """Test global species integrator instance"""

    def test_global_instance_available(self):
        """Test global integrator instance is available"""

        assert species_integrator is not None
        assert isinstance(species_integrator, SpeciesParameterIntegrator)

        # Test it works
        config = species_integrator.get_species_communication_config("Bombus_terrestris")
        assert isinstance(config, BumblebeeCommunicationModel)

    def test_global_instance_consistency(self):
        """Test global instance provides consistent results"""

        # Get the same config multiple times
        config1 = species_integrator.get_species_communication_config("Bombus_terrestris")
        config2 = species_integrator.get_species_communication_config("Bombus_terrestris")

        # Should be the same cached instance
        assert config1 is config2
        assert config1.memory_capacity == config2.memory_capacity


class TestIntegrationRobustness:
    """Test integration robustness and edge cases"""

    def test_parameter_calculation_stability(self):
        """Test parameter calculations are stable and reasonable"""

        integrator = SpeciesParameterIntegrator()

        for species_name in ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]:
            params = integrator.species_parameters[species_name]

            # Test private calculation methods don't crash
            memory_decay = integrator._calculate_memory_decay_rate(params)
            scent_prob = integrator._calculate_scent_mark_probability(params)
            scent_range = integrator._calculate_scent_detection_range(params)

            # Check results are reasonable
            assert 0.0 <= memory_decay <= 1.0
            assert 0.0 <= scent_prob <= 1.0
            assert 0.0 <= scent_range <= 10.0

    def test_error_handling(self):
        """Test error handling for invalid inputs"""

        integrator = SpeciesParameterIntegrator()

        # Test unknown species handling
        validation = integrator.validate_species_configuration("Unknown_species")
        assert validation["valid"] is False
        assert "not found" in validation["error"]

        summary = integrator.get_species_summary("Unknown_species")
        assert "error" in summary

    def test_configuration_caching(self):
        """Test configuration caching works correctly"""

        integrator = SpeciesParameterIntegrator()

        # Get config multiple times
        config1 = integrator.get_species_communication_config("Bombus_terrestris")
        config2 = integrator.get_species_communication_config("Bombus_terrestris")

        # Should be cached (same instance)
        assert config1 is config2

        # Different species should be different
        config3 = integrator.get_species_communication_config("Bombus_pascuorum")
        assert config1 is not config3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
