"""
Test cases for the multi-species system
"""

from src.bstew.components.species_system import (
    SpeciesSystem,
    SpeciesParameters,
    SpeciesType,
)
from src.bstew.components.proboscis_matching import ProboscisCharacteristics
from src.bstew.components.development import DevelopmentParameters


class TestSpeciesSystem:
    """Test multi-species system functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.species_system = SpeciesSystem()

    def test_system_initialization(self):
        """Test species system initialization"""
        assert len(self.species_system.species_parameters) == 7
        assert "Bombus_terrestris" in self.species_system.species_parameters
        assert "Bombus_ruderatus" in self.species_system.species_parameters

    def test_species_parameters_structure(self):
        """Test species parameters contain required fields"""
        for species_name, params in self.species_system.species_parameters.items():
            assert isinstance(params, SpeciesParameters)
            assert hasattr(params, "proboscis_characteristics")
            assert hasattr(params, "development_parameters")
            assert hasattr(params, "max_colony_size")
            assert hasattr(params, "foraging_range_m")

    def test_proboscis_length_range(self):
        """Test proboscis lengths span expected range"""
        lengths = []
        for params in self.species_system.species_parameters.values():
            lengths.append(params.proboscis_characteristics.length_mm)

        assert min(lengths) < 8.0  # Short-tongued species
        assert max(lengths) > 15.0  # Long-tongued species
        assert len(set(lengths)) == 7  # All different lengths

    def test_species_interactions_matrix(self):
        """Test species interactions are calculated"""
        species_names = list(self.species_system.species_parameters.keys())

        # Check interaction matrix is populated
        for species1 in species_names:
            for species2 in species_names:
                if species1 != species2:
                    key = (species1, species2)
                    assert key in self.species_system.species_interactions

    def test_temporal_overlap_calculation(self):
        """Test temporal overlap calculation"""
        # Complete overlap
        overlap = self.species_system.calculate_temporal_overlap(
            "Bombus_terrestris", "Bombus_lucorum"
        )
        assert 0.0 <= overlap <= 1.0

        # Check with early vs late species
        early_species = "Bombus_terrestris"  # Early emerging
        late_species = "Bombus_pascuorum"  # Late emerging

        overlap = self.species_system.calculate_temporal_overlap(
            early_species, late_species
        )
        assert overlap > 0.0  # Should have some overlap

    def test_resource_overlap_calculation(self):
        """Test resource overlap calculation"""
        # Similar proboscis lengths
        overlap = self.species_system.calculate_resource_overlap(
            "Bombus_terrestris", "Bombus_lucorum"
        )
        assert overlap > 0.5  # Should be high overlap

        # Very different proboscis lengths
        overlap = self.species_system.calculate_resource_overlap(
            "Bombus_terrestris", "Bombus_ruderatus"
        )
        assert overlap < 0.6  # Should be lower overlap than similar species

    def test_available_species_by_day(self):
        """Test getting available species by day"""
        # Early season
        early_species = self.species_system.get_available_species(60)
        assert "Bombus_terrestris" in early_species
        assert "Bombus_lucorum" in early_species

        # Late season
        late_species = self.species_system.get_available_species(300)
        assert len(late_species) < len(early_species)  # Fewer active species

    def test_emerging_species_by_day(self):
        """Test getting emerging species by day"""
        # Peak emergence period
        emerging = self.species_system.get_emerging_species(80)
        assert len(emerging) > 0

        # Off-season
        emerging = self.species_system.get_emerging_species(300)
        assert len(emerging) == 0

    def test_competition_effect_calculation(self):
        """Test competition effect calculation"""
        # Self-competition
        effect = self.species_system.calculate_competition_effect(
            "Bombus_terrestris", "Bombus_terrestris", 0.5
        )
        assert effect == 0.0

        # Strong competitor vs weak competitor
        effect = self.species_system.calculate_competition_effect(
            "Bombus_ruderatus", "Bombus_terrestris", 0.8
        )
        assert effect > 0.0

    def test_habitat_preferences(self):
        """Test habitat preference retrieval"""
        nest_pref, foraging_pref = self.species_system.get_optimal_habitats(
            "Bombus_terrestris", "urban"
        )
        assert 0.0 <= nest_pref <= 1.0
        assert 0.0 <= foraging_pref <= 1.0

        # Test unknown species
        nest_pref, foraging_pref = self.species_system.get_optimal_habitats(
            "Unknown_species", "urban"
        )
        assert nest_pref == 0.5
        assert foraging_pref == 0.5

    def test_community_assembly_simulation(self):
        """Test community assembly simulation"""
        landscape_capacity = {"overall": 100}
        environmental_conditions = {"day_of_year": 150}

        community = self.species_system.simulate_community_assembly(
            landscape_capacity, environmental_conditions
        )

        assert isinstance(community, dict)
        assert sum(community.values()) <= 100  # Should not exceed capacity

        # Check all species have reasonable colony counts
        for species, count in community.items():
            assert count >= 0

    def test_species_phenology_info(self):
        """Test species phenology information"""
        phenology = self.species_system.get_species_phenology("Bombus_terrestris")

        assert "emerging_day_mean" in phenology
        assert "active_season_start" in phenology
        assert "active_season_end" in phenology
        assert "season_length" in phenology

        # Test unknown species
        phenology = self.species_system.get_species_phenology("Unknown_species")
        assert phenology == {}

    def test_diversity_metrics(self):
        """Test community diversity metrics"""
        # Balanced community
        community = {
            "Bombus_terrestris": 10,
            "Bombus_lucorum": 10,
            "Bombus_lapidarius": 10,
        }

        metrics = self.species_system.get_community_diversity_metrics(community)

        assert "richness" in metrics
        assert "shannon" in metrics
        assert "evenness" in metrics
        assert metrics["richness"] == 3
        assert metrics["shannon"] > 0

        # Empty community
        empty_metrics = self.species_system.get_community_diversity_metrics({})
        assert empty_metrics["richness"] == 0.0
        assert empty_metrics["shannon"] == 0.0

    def test_trait_summary(self):
        """Test species trait summary"""
        summary = self.species_system.get_species_trait_summary()

        assert summary["species_count"] == 7
        assert "proboscis_lengths" in summary
        assert "body_sizes" in summary
        assert "foraging_ranges" in summary
        assert "colony_sizes" in summary
        assert "phenology" in summary

        # Check all species are included
        for species_name in self.species_system.species_parameters.keys():
            assert species_name in summary["proboscis_lengths"]

    def test_management_response_prediction(self):
        """Test species response to management practices"""
        # Wildflower strips
        responses = self.species_system.predict_species_response_to_management(
            "wildflower_strips", 1.0
        )

        assert len(responses) == 7

        # Long-tongued species should benefit more
        long_tongued_response = responses["Bombus_ruderatus"]
        short_tongued_response = responses["Bombus_terrestris"]
        assert long_tongued_response > short_tongued_response

        # Mowing
        mowing_responses = self.species_system.predict_species_response_to_management(
            "mowing", 1.0
        )

        # All responses should be negative
        for response in mowing_responses.values():
            assert response <= 0.0

        # Pesticide
        pesticide_responses = (
            self.species_system.predict_species_response_to_management("pesticide", 1.0)
        )

        # All responses should be negative
        for response in pesticide_responses.values():
            assert response <= 0.0

    def test_species_parameter_ranges(self):
        """Test species parameters are within realistic ranges"""
        for species_name, params in self.species_system.species_parameters.items():
            # Proboscis length
            assert 5.0 <= params.proboscis_characteristics.length_mm <= 20.0

            # Body size
            assert 10.0 <= params.body_size_mm <= 30.0

            # Colony size
            assert 50 <= params.max_colony_size <= 500

            # Foraging range
            assert 500.0 <= params.foraging_range_m <= 3000.0

            # Tolerance values
            assert 0.0 <= params.cold_tolerance <= 1.0
            assert 0.0 <= params.drought_tolerance <= 1.0
            assert 0.0 <= params.competition_strength <= 1.0

    def test_species_ecological_traits(self):
        """Test species ecological trait differentiation"""
        terrestris = self.species_system.species_parameters["Bombus_terrestris"]
        ruderatus = self.species_system.species_parameters["Bombus_ruderatus"]

        # B. terrestris should be more competitive and dominant
        assert terrestris.competition_strength > ruderatus.competition_strength
        assert terrestris.social_dominance > ruderatus.social_dominance

        # B. ruderatus should have longer proboscis and range
        assert (
            ruderatus.proboscis_characteristics.length_mm
            > terrestris.proboscis_characteristics.length_mm
        )
        assert ruderatus.foraging_range_m > terrestris.foraging_range_m

    def test_phenological_separation(self):
        """Test species show phenological separation"""
        emergence_times = []
        for params in self.species_system.species_parameters.values():
            emergence_times.append(params.emerging_day_mean)

        # Should have range of emergence times
        time_range = max(emergence_times) - min(emergence_times)
        assert time_range > 30  # At least 30 days separation

    def test_species_competition_asymmetry(self):
        """Test competition is asymmetric between species"""
        # Competition should not be symmetric
        effect_ab = self.species_system.calculate_competition_effect(
            "Bombus_terrestris", "Bombus_ruderatus", 0.5
        )
        effect_ba = self.species_system.calculate_competition_effect(
            "Bombus_ruderatus", "Bombus_terrestris", 0.5
        )

        # Effects should be different (asymmetric competition)
        assert effect_ab != effect_ba

    def test_habitat_specialization(self):
        """Test species show habitat specialization"""
        # Urban preference should vary between species
        urban_preferences = []
        for species_name in self.species_system.species_parameters.keys():
            nest_pref, _ = self.species_system.get_optimal_habitats(
                species_name, "urban"
            )
            urban_preferences.append(nest_pref)

        # Should have range of urban preferences
        assert max(urban_preferences) > min(urban_preferences) + 0.2


class TestSpeciesParameters:
    """Test species parameters dataclass"""

    def test_species_parameters_creation(self):
        """Test creating species parameters"""
        proboscis = ProboscisCharacteristics(
            length_mm=10.0, width_mm=0.2, flexibility=0.9, extension_efficiency=0.95
        )
        development = DevelopmentParameters(
            dev_age_hatching_min=3.0,
            dev_age_pupation_min=12.0,
            dev_age_emerging_min=18.0,
            dev_weight_egg=0.15,
            dev_weight_pupation_min=100.0,
            dev_weight_adult_min=32.0,
        )

        params = SpeciesParameters(
            species_name="Test_species",
            species_type=SpeciesType.BOMBUS_TERRESTRIS,
            proboscis_characteristics=proboscis,
            body_size_mm=20.0,
            wing_length_mm=18.0,
            weight_mg=800.0,
            development_parameters=development,
            max_lifespan_workers=30,
            max_lifespan_queens=300,
            max_lifespan_drones=25,
            emerging_day_mean=60,
            emerging_day_sd=10.0,
            active_season_start=50,
            active_season_end=250,
            flight_velocity_ms=3.0,
            foraging_range_m=1500.0,
            search_length_m=800.0,
            nectar_load_capacity_mg=40.0,
            pollen_load_capacity_mg=10.0,
            max_colony_size=300,
            typical_colony_size=150,
            brood_development_time=20.0,
            nest_habitat_preferences={"woodland": 0.8},
            foraging_habitat_preferences={"wildflower": 0.9},
            cold_tolerance=0.8,
            drought_tolerance=0.6,
            competition_strength=0.7,
            foraging_aggressiveness=0.6,
            site_fidelity=0.8,
            social_dominance=0.7,
        )

        assert params.species_name == "Test_species"
        assert params.body_size_mm == 20.0
        assert params.max_colony_size == 300
        assert params.nest_habitat_preferences["woodland"] == 0.8
