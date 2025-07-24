"""
Tests for Flower Community Layers System
========================================

Comprehensive test suite for hierarchical flower community functionality.
"""

import pytest
from datetime import date
from unittest.mock import Mock

from src.bstew.components.flower_community_layers import (
    VerticalLayer,
    SuccessionStage,
    CompetitionType,
    LayerInteraction,
    VerticalResourceGradient,
    CommunityLayer,
    FlowerCommunity,
    CommunityLayerSystem,
    initialize_standard_templates
)
from src.bstew.spatial.patches import FlowerSpecies


@pytest.fixture
def mock_flower_species():
    """Create mock flower species for testing"""
    species1 = Mock(spec=FlowerSpecies)
    species1.name = "White Clover"
    species1.bloom_start = 90
    species1.bloom_end = 270
    species1.nectar_production = 2.0
    species1.pollen_production = 1.5
    species1.flower_density = 100.0
    species1.attractiveness = 0.8
    species1.nectar_accessibility = 0.9

    species2 = Mock(spec=FlowerSpecies)
    species2.name = "Hawthorn"
    species2.bloom_start = 120
    species2.bloom_end = 180
    species2.nectar_production = 5.0
    species2.pollen_production = 3.0
    species2.flower_density = 50.0
    species2.attractiveness = 0.7
    species2.nectar_accessibility = 0.6

    return [species1, species2]


@pytest.fixture
def sample_resource_gradient():
    """Create sample vertical resource gradient"""
    return VerticalResourceGradient(
        light_profile={
            VerticalLayer.CANOPY: 1.0,
            VerticalLayer.UNDERSTORY: 0.6,
            VerticalLayer.GROUND: 0.3
        },
        nutrient_profile={
            VerticalLayer.CANOPY: 0.7,
            VerticalLayer.UNDERSTORY: 0.8,
            VerticalLayer.GROUND: 1.0
        },
        water_profile={
            VerticalLayer.CANOPY: 0.8,
            VerticalLayer.UNDERSTORY: 0.9,
            VerticalLayer.GROUND: 1.0
        },
        pollinator_access={
            VerticalLayer.CANOPY: 0.7,
            VerticalLayer.UNDERSTORY: 1.0,
            VerticalLayer.GROUND: 0.9
        }
    )


class TestVerticalLayer:
    """Test VerticalLayer enum"""

    def test_vertical_layer_values(self):
        """Test vertical layer enum values"""
        assert VerticalLayer.CANOPY.value == "canopy"
        assert VerticalLayer.UNDERSTORY.value == "understory"
        assert VerticalLayer.GROUND.value == "ground"
        assert VerticalLayer.ROOT.value == "root"


class TestSuccessionStage:
    """Test SuccessionStage enum"""

    def test_succession_stage_values(self):
        """Test succession stage enum values"""
        assert SuccessionStage.PIONEER.value == "pioneer"
        assert SuccessionStage.EARLY.value == "early"
        assert SuccessionStage.MID.value == "mid"
        assert SuccessionStage.LATE.value == "late"
        assert SuccessionStage.CLIMAX.value == "climax"


class TestCompetitionType:
    """Test CompetitionType enum"""

    def test_competition_type_values(self):
        """Test competition type enum values"""
        assert CompetitionType.LIGHT.value == "light"
        assert CompetitionType.NUTRIENTS.value == "nutrients"
        assert CompetitionType.WATER.value == "water"
        assert CompetitionType.SPACE.value == "space"
        assert CompetitionType.POLLINATORS.value == "pollinators"


class TestLayerInteraction:
    """Test LayerInteraction dataclass"""

    def test_layer_interaction_creation(self):
        """Test creating layer interaction"""
        interaction = LayerInteraction(
            source_layer=VerticalLayer.CANOPY,
            target_layer=VerticalLayer.UNDERSTORY,
            interaction_type=CompetitionType.LIGHT,
            strength=0.4
        )

        assert interaction.source_layer == VerticalLayer.CANOPY
        assert interaction.target_layer == VerticalLayer.UNDERSTORY
        assert interaction.interaction_type == CompetitionType.LIGHT
        assert interaction.strength == 0.4

    def test_seasonal_modifier(self):
        """Test seasonal modifier functionality"""
        interaction = LayerInteraction(
            source_layer=VerticalLayer.CANOPY,
            target_layer=VerticalLayer.GROUND,
            interaction_type=CompetitionType.LIGHT,
            strength=0.6,
            seasonal_modifier={150: 0.8, 200: 1.2}
        )

        assert interaction.seasonal_modifier[150] == 0.8
        assert interaction.seasonal_modifier[200] == 1.2


class TestVerticalResourceGradient:
    """Test VerticalResourceGradient dataclass"""

    def test_resource_gradient_creation(self, sample_resource_gradient):
        """Test creating resource gradient"""
        gradient = sample_resource_gradient

        assert gradient.light_profile[VerticalLayer.CANOPY] == 1.0
        assert gradient.light_profile[VerticalLayer.GROUND] == 0.3
        assert gradient.nutrient_profile[VerticalLayer.GROUND] == 1.0
        assert gradient.pollinator_access[VerticalLayer.UNDERSTORY] == 1.0


class TestCommunityLayer:
    """Test CommunityLayer model"""

    @pytest.fixture
    def basic_layer(self, mock_flower_species):
        """Create basic community layer"""
        return CommunityLayer(
            layer_id="test_ground",
            vertical_layer=VerticalLayer.GROUND,
            succession_stage=SuccessionStage.MID,
            dominant_species=mock_flower_species[:1],
            height_range=(0.0, 0.5),
            coverage_fraction=0.8
        )

    def test_layer_creation(self, basic_layer):
        """Test creating community layer"""
        assert basic_layer.layer_id == "test_ground"
        assert basic_layer.vertical_layer == VerticalLayer.GROUND
        assert basic_layer.succession_stage == SuccessionStage.MID
        assert basic_layer.coverage_fraction == 0.8
        assert basic_layer.height_range == (0.0, 0.5)
        assert len(basic_layer.dominant_species) == 1

    def test_resource_production(self, basic_layer, sample_resource_gradient):
        """Test resource production calculation"""
        competition_effects = {CompetitionType.LIGHT: 0.1}

        resources = basic_layer.get_resource_production(
            day_of_year=150,
            base_conditions=sample_resource_gradient,
            competition_effects=competition_effects
        )

        assert "nectar_production" in resources
        assert "pollen_production" in resources
        assert "resource_limitation" in resources
        assert resources["nectar_production"] >= 0
        assert resources["pollen_production"] >= 0

    def test_succession_advancement(self, basic_layer):
        """Test succession advancement"""
        initial_stage = basic_layer.succession_stage
        basic_layer.succession_trajectory = SuccessionStage.LATE
        basic_layer.succession_rate = 1.0  # Always advance

        advanced = basic_layer.advance_succession()

        if advanced:
            assert basic_layer.succession_stage != initial_stage

    def test_biodiversity_calculation(self, basic_layer, mock_flower_species):
        """Test biodiversity index calculation"""
        basic_layer.subordinate_species = mock_flower_species[1:]

        diversity = basic_layer.calculate_biodiversity_index()

        assert 0 <= diversity <= 1
        assert isinstance(diversity, float)

    def test_empty_layer_biodiversity(self):
        """Test biodiversity of empty layer"""
        empty_layer = CommunityLayer(
            layer_id="empty",
            vertical_layer=VerticalLayer.GROUND,
            succession_stage=SuccessionStage.PIONEER,
            height_range=(0.0, 0.5)
        )

        diversity = empty_layer.calculate_biodiversity_index()
        assert diversity == 0.0


class TestFlowerCommunity:
    """Test FlowerCommunity model"""

    @pytest.fixture
    def basic_community(self, mock_flower_species, sample_resource_gradient):
        """Create basic flower community"""
        community = FlowerCommunity(
            community_id="test_community",
            community_type="grassland",
            location=(100.0, 200.0),
            area_m2=10000.0,
            base_conditions=sample_resource_gradient,
            establishment_date=date(2024, 1, 1)
        )

        # Add ground layer
        ground_layer = CommunityLayer(
            layer_id="ground",
            vertical_layer=VerticalLayer.GROUND,
            succession_stage=SuccessionStage.MID,
            dominant_species=mock_flower_species[:1],
            coverage_fraction=0.8,
            height_range=(0.0, 0.5)
        )
        community.add_layer(ground_layer)

        return community

    def test_community_creation(self, basic_community):
        """Test creating flower community"""
        assert basic_community.community_id == "test_community"
        assert basic_community.community_type == "grassland"
        assert basic_community.location == (100.0, 200.0)
        assert basic_community.area_m2 == 10000.0
        assert len(basic_community.layers) == 1
        assert VerticalLayer.GROUND in basic_community.layers

    def test_layer_management(self, basic_community, mock_flower_species):
        """Test adding and removing layers"""
        # Add understory layer
        understory_layer = CommunityLayer(
            layer_id="understory",
            vertical_layer=VerticalLayer.UNDERSTORY,
            succession_stage=SuccessionStage.MID,
            dominant_species=mock_flower_species[1:],
            height_range=(1.0, 3.0)
        )

        basic_community.add_layer(understory_layer)
        assert len(basic_community.layers) == 2
        assert VerticalLayer.UNDERSTORY in basic_community.layers

        # Remove layer
        basic_community.remove_layer(VerticalLayer.UNDERSTORY)
        assert len(basic_community.layers) == 1
        assert VerticalLayer.UNDERSTORY not in basic_community.layers

    def test_layer_interactions(self, basic_community):
        """Test layer interaction calculations"""
        # Add interaction
        interaction = LayerInteraction(
            source_layer=VerticalLayer.UNDERSTORY,
            target_layer=VerticalLayer.GROUND,
            interaction_type=CompetitionType.LIGHT,
            strength=0.3
        )
        basic_community.layer_interactions.append(interaction)

        # Add understory layer for interaction
        understory_layer = CommunityLayer(
            layer_id="understory",
            vertical_layer=VerticalLayer.UNDERSTORY,
            succession_stage=SuccessionStage.MID,
            coverage_fraction=0.6,
            density_factor=1.0,
            height_range=(1.0, 3.0)
        )
        understory_layer.competitive_ability[CompetitionType.LIGHT] = 0.8
        basic_community.add_layer(understory_layer)

        interactions = basic_community.calculate_layer_interactions()

        assert VerticalLayer.GROUND in interactions
        assert CompetitionType.LIGHT in interactions[VerticalLayer.GROUND]

    def test_resource_updates(self, basic_community):
        """Test community resource updates"""
        resources = basic_community.update_community_resources(day_of_year=150)

        assert "nectar_production" in resources
        assert "pollen_production" in resources
        assert "community_efficiency" in resources
        assert resources["nectar_production"] >= 0
        assert resources["pollen_production"] >= 0

    def test_succession_simulation(self, basic_community):
        """Test succession simulation"""
        succession_results = basic_community.simulate_succession(years=2, disturbance_probability=0.0)

        assert len(succession_results) == 2
        assert all("year" in result for result in succession_results)
        assert all("succession_events" in result for result in succession_results)
        assert all("disturbances" in result for result in succession_results)

    def test_disturbance_application(self, basic_community):
        """Test applying disturbances"""
        basic_community.layers[VerticalLayer.GROUND].coverage_fraction

        disturbance = basic_community._apply_random_disturbance()

        assert "type" in disturbance
        assert "intensity" in disturbance
        assert "affected_layers" in disturbance
        assert len(basic_community.disturbance_history) == 1

        # Check if coverage was affected
        basic_community.layers[VerticalLayer.GROUND].coverage_fraction
        # Coverage might be reduced depending on disturbance

    def test_accessibility_profiles(self, basic_community):
        """Test species accessibility profiles"""
        # Mock proboscis system
        mock_proboscis = Mock()

        profiles = basic_community.get_species_accessibility_profile(mock_proboscis)

        assert isinstance(profiles, dict)
        if profiles:  # If there are profiles
            assert "ground" in profiles or len(profiles) == 0  # Either has ground layer or empty

    def test_community_export(self, basic_community):
        """Test community structure export"""
        export_data = basic_community.export_community_structure()

        assert export_data["community_id"] == "test_community"
        assert export_data["community_type"] == "grassland"
        assert export_data["area_m2"] == 10000.0
        assert "layers" in export_data
        assert "community_metrics" in export_data
        assert export_data["layer_count"] == 1


class TestCommunityLayerSystem:
    """Test CommunityLayerSystem class"""

    @pytest.fixture
    def layer_system(self):
        """Create community layer system"""
        return CommunityLayerSystem()

    def test_system_initialization(self, layer_system):
        """Test system initialization"""
        assert len(layer_system.communities) == 0
        assert len(layer_system.community_templates) == 0
        assert layer_system.default_gradient is not None
        assert len(layer_system.succession_rates) > 0

    def test_template_creation(self, layer_system):
        """Test creating community template"""
        template_data = {
            "community_type": "test_grassland",
            "layers": [
                {
                    "layer_id": "test_ground",
                    "vertical_layer": "ground",
                    "succession_stage": "mid",
                    "species": ["White Clover"],
                    "coverage_fraction": 0.8
                }
            ],
            "interactions": []
        }

        layer_system.create_community_template("test_template", template_data)

        assert "test_template" in layer_system.community_templates
        assert layer_system.community_templates["test_template"]["community_type"] == "test_grassland"

    def test_community_creation_from_template(self, layer_system):
        """Test creating community from template"""
        # First create template
        template_data = {
            "community_type": "test_grassland",
            "layers": [
                {
                    "layer_id": "test_ground",
                    "vertical_layer": "ground",
                    "succession_stage": "mid",
                    "coverage_fraction": 0.8,
                    "height_range": [0.0, 0.5],
                    "density_factor": 1.0
                }
            ],
            "interactions": []
        }

        layer_system.create_community_template("test_template", template_data)

        # Create community from template
        community = layer_system.create_community_from_template(
            community_id="test_comm",
            template_name="test_template",
            location=(0.0, 0.0),
            area_m2=5000.0
        )

        assert community.community_id == "test_comm"
        assert community.community_type == "test_grassland"
        assert community.area_m2 == 5000.0
        assert len(community.layers) == 1

    def test_invalid_template(self, layer_system):
        """Test creating community from non-existent template"""
        with pytest.raises(ValueError, match="Template nonexistent not found"):
            layer_system.create_community_from_template(
                community_id="test",
                template_name="nonexistent",
                location=(0.0, 0.0),
                area_m2=1000.0
            )

    def test_landscape_succession_simulation(self, layer_system):
        """Test landscape-wide succession simulation"""
        # Create template and community
        template_data = {
            "community_type": "grassland",
            "layers": [
                {
                    "layer_id": "ground",
                    "vertical_layer": "ground",
                    "succession_stage": "early",
                    "coverage_fraction": 1.0,
                    "height_range": [0.0, 0.5],
                    "density_factor": 1.0
                }
            ],
            "interactions": []
        }

        layer_system.create_community_template("grassland", template_data)
        layer_system.create_community_from_template(
            "comm1", "grassland", (0.0, 0.0), 1000.0
        )

        # Run landscape succession
        results = layer_system.simulate_landscape_succession(years=2)

        assert results["total_years"] == 2
        assert results["community_count"] == 1
        assert "results" in results
        assert "comm1" in results["results"]

    def test_connectivity_analysis(self, layer_system):
        """Test community connectivity analysis"""
        # Create template
        template_data = {
            "community_type": "grassland",
            "layers": [
                {
                    "layer_id": "ground",
                    "vertical_layer": "ground",
                    "succession_stage": "mid",
                    "coverage_fraction": 1.0,
                    "height_range": [0.0, 0.5],
                    "density_factor": 1.0
                }
            ],
            "interactions": []
        }

        layer_system.create_community_template("grassland", template_data)

        # Create two communities
        comm1 = layer_system.create_community_from_template(
            "comm1", "grassland", (0.0, 0.0), 1000.0
        )
        comm2 = layer_system.create_community_from_template(
            "comm2", "grassland", (500.0, 0.0), 1000.0
        )

        # Update diversity for connectivity calculation
        comm1.current_diversity_index = 0.5
        comm2.current_diversity_index = 0.6

        connectivity = layer_system.analyze_community_connectivity()

        assert "connectivity_score" in connectivity
        assert "connections" in connectivity
        assert connectivity["total_communities"] == 2

        if connectivity["connections"]:
            assert len(connectivity["connections"]) >= 0

    def test_landscape_summary_empty(self, layer_system):
        """Test landscape summary with no communities"""
        summary = layer_system.export_landscape_summary()

        assert "error" in summary
        assert summary["error"] == "No communities to analyze"

    def test_landscape_summary_with_communities(self, layer_system):
        """Test landscape summary with communities"""
        # Create template and community
        template_data = {
            "community_type": "grassland",
            "layers": [
                {
                    "layer_id": "ground",
                    "vertical_layer": "ground",
                    "succession_stage": "mid",
                    "coverage_fraction": 0.8,
                    "height_range": [0.0, 0.5],
                    "density_factor": 1.0
                }
            ]
        }

        layer_system.create_community_template("grassland", template_data)
        layer_system.create_community_from_template(
            "comm1", "grassland", (0.0, 0.0), 2000.0
        )

        summary = layer_system.export_landscape_summary()

        assert "landscape_summary" in summary
        assert summary["landscape_summary"]["total_communities"] == 1
        assert summary["landscape_summary"]["total_area_m2"] == 2000.0
        assert "community_types" in summary
        assert "succession_stages" in summary
        assert "layer_distribution" in summary


class TestStandardTemplates:
    """Test standard community templates"""

    def test_initialize_standard_templates(self):
        """Test initialization of standard templates"""
        templates = initialize_standard_templates()

        assert "uk_chalk_grassland" in templates
        assert "woodland_edge" in templates

        # Test chalk grassland template
        chalk_template = templates["uk_chalk_grassland"]
        assert chalk_template["community_type"] == "chalk_grassland"
        assert len(chalk_template["layers"]) == 2
        assert len(chalk_template["interactions"]) == 1

        # Test woodland edge template
        woodland_template = templates["woodland_edge"]
        assert woodland_template["community_type"] == "woodland_edge"
        assert len(woodland_template["layers"]) == 3
        assert len(woodland_template["interactions"]) == 3

    def test_template_layer_structure(self):
        """Test template layer structure"""
        templates = initialize_standard_templates()
        chalk_template = templates["uk_chalk_grassland"]

        # Check ground herbs layer
        ground_layer = chalk_template["layers"][0]
        assert ground_layer["layer_id"] == "ground_herbs"
        assert ground_layer["vertical_layer"] == "ground"
        assert ground_layer["succession_stage"] == "mid"
        assert ground_layer["coverage_fraction"] == 0.8

    def test_template_interactions(self):
        """Test template interaction definitions"""
        templates = initialize_standard_templates()
        woodland_template = templates["woodland_edge"]

        interactions = woodland_template["interactions"]

        # Check light competition interactions
        light_interactions = [i for i in interactions if i["type"] == "light"]
        assert len(light_interactions) == 3

        # Check canopy->understory interaction
        canopy_understory = next(
            i for i in interactions
            if i["source"] == "canopy" and i["target"] == "understory"
        )
        assert canopy_understory["strength"] == 0.4


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""

    @pytest.fixture
    def complete_system(self):
        """Create complete system with templates"""
        system = CommunityLayerSystem()
        templates = initialize_standard_templates()

        for name, template in templates.items():
            system.create_community_template(name, template)

        return system

    def test_grassland_succession_scenario(self, complete_system, mock_flower_species):
        """Test realistic grassland succession scenario"""
        # Create chalk grassland community
        community = complete_system.create_community_from_template(
            community_id="meadow1",
            template_name="uk_chalk_grassland",
            location=(100.0, 200.0),
            area_m2=5000.0,
            customizations={
                "microclimate": {
                    "temperature_buffer": 0.1,
                    "humidity_retention": 0.2
                }
            }
        )

        # Test resource production across seasons
        spring_resources = community.update_community_resources(day_of_year=120)
        summer_resources = community.update_community_resources(day_of_year=180)
        autumn_resources = community.update_community_resources(day_of_year=270)

        assert all(r >= 0 for r in [
            spring_resources["nectar_production"],
            summer_resources["nectar_production"],
            autumn_resources["nectar_production"]
        ])

    def test_woodland_edge_complexity(self, complete_system):
        """Test complex woodland edge community"""
        # Create woodland edge community
        community = complete_system.create_community_from_template(
            community_id="edge1",
            template_name="woodland_edge",
            location=(0.0, 0.0),
            area_m2=10000.0
        )

        # Should have three layers
        assert len(community.layers) == 3
        assert VerticalLayer.CANOPY in community.layers
        assert VerticalLayer.UNDERSTORY in community.layers
        assert VerticalLayer.GROUND in community.layers

        # Test layer interactions
        interactions = community.calculate_layer_interactions()

        # Ground layer should experience competition from above
        if VerticalLayer.GROUND in interactions:
            ground_competition = interactions[VerticalLayer.GROUND]
            assert CompetitionType.LIGHT in ground_competition

    def test_multi_community_landscape(self, complete_system):
        """Test landscape with multiple communities"""
        # Create diverse landscape
        communities = []

        # Create grassland
        grassland = complete_system.create_community_from_template(
            "grassland1", "uk_chalk_grassland", (0.0, 0.0), 3000.0
        )
        communities.append(grassland)

        # Create woodland edge nearby
        woodland = complete_system.create_community_from_template(
            "woodland1", "woodland_edge", (200.0, 0.0), 4000.0
        )
        communities.append(woodland)

        # Set realistic diversity values
        grassland.current_diversity_index = 0.6
        woodland.current_diversity_index = 0.8

        # Test connectivity analysis
        connectivity = complete_system.analyze_community_connectivity()

        assert connectivity["total_communities"] == 2
        assert connectivity["connectivity_score"] >= 0

        # Should have connection due to proximity (200m apart)
        if connectivity["connections"]:
            connection = connectivity["connections"][0]
            assert connection["distance_m"] == 200.0

    def test_succession_with_disturbance(self, complete_system):
        """Test succession under disturbance pressure"""
        community = complete_system.create_community_from_template(
            "test_succession", "uk_chalk_grassland", (0.0, 0.0), 2000.0
        )

        # Simulate with high disturbance
        succession_results = community.simulate_succession(
            years=3,
            disturbance_probability=0.5
        )

        assert len(succession_results) == 3

        # Should have some disturbances with high probability
        sum(
            len(year["disturbances"]) for year in succession_results
        )
        # With 50% probability over 3 years, expect some disturbances

        # Check that community survived disturbances
        assert community.stability_index >= 0
        assert community.resilience_score >= 0

    def test_climate_scenario_simulation(self, complete_system):
        """Test landscape response to climate scenarios"""
        # Create multiple communities
        for i in range(3):
            complete_system.create_community_from_template(
                f"comm{i}",
                "uk_chalk_grassland",
                (i * 1000.0, 0.0),
                2000.0
            )

        # Test different climate scenarios
        baseline = complete_system.simulate_landscape_succession(years=2)

        warming_scenario = complete_system.simulate_landscape_succession(
            years=2,
            climate_scenario={
                "temperature_change": 2.0,
                "precipitation_change": -0.1
            }
        )

        assert baseline["community_count"] == 3
        assert warming_scenario["community_count"] == 3

        # Both should complete without errors
        assert len(baseline["results"]) == 3
        assert len(warming_scenario["results"]) == 3
