"""
Test cases for the proboscis-corolla matching system
"""

from src.bstew.components.proboscis_matching import (
    ProboscisCorollaSystem,
    ProboscisCharacteristics,
    AccessibilityResult,
    AccessibilityLevel,
)
from src.bstew.spatial.patches import FlowerSpecies


class TestProboscisCorollaSystem:
    """Test proboscis-corolla matching system"""

    def setup_method(self):
        """Set up test fixtures"""
        self.matching_system = ProboscisCorollaSystem()

    def test_system_initialization(self):
        """Test system initialization"""
        assert len(self.matching_system.species_proboscis) > 0
        assert "Bombus_terrestris" in self.matching_system.species_proboscis
        assert "Bombus_hortorum" in self.matching_system.species_proboscis
        assert len(self.matching_system.accessibility_thresholds) == 5

    def test_species_proboscis_characteristics(self):
        """Test species-specific proboscis characteristics"""
        # Short-tongued species
        terrestris = self.matching_system.get_species_proboscis("Bombus_terrestris")
        assert terrestris.length_mm == 7.2
        assert terrestris.get_effective_length() < terrestris.length_mm

        # Long-tongued species
        hortorum = self.matching_system.get_species_proboscis("Bombus_hortorum")
        assert hortorum.length_mm == 15.3
        assert hortorum.length_mm > terrestris.length_mm

        # Unknown species should return default
        unknown = self.matching_system.get_species_proboscis("Unknown_species")
        assert unknown.length_mm == 8.0

    def test_width_constraint_checking(self):
        """Test proboscis width constraint checking"""
        narrow_proboscis = ProboscisCharacteristics(length_mm=10.0, width_mm=0.15)
        wide_proboscis = ProboscisCharacteristics(length_mm=10.0, width_mm=0.35)

        # Wide flower - no constraint
        wide_flower = FlowerSpecies(
            name="Test",
            bloom_start=100,
            bloom_end=200,
            nectar_production=5.0,
            pollen_production=2.0,
            flower_density=10.0,
            attractiveness=0.8,
            corolla_depth_mm=5.0,
            corolla_width_mm=2.0,
        )
        assert (
            self.matching_system.check_width_constraint(narrow_proboscis, wide_flower)
            == 1.0
        )
        assert (
            self.matching_system.check_width_constraint(wide_proboscis, wide_flower)
            == 1.0
        )

        # Narrow flower - constraint for wide proboscis
        narrow_flower = FlowerSpecies(
            name="Test",
            bloom_start=100,
            bloom_end=200,
            nectar_production=5.0,
            pollen_production=2.0,
            flower_density=10.0,
            attractiveness=0.8,
            corolla_depth_mm=5.0,
            corolla_width_mm=0.2,
        )
        assert (
            self.matching_system.check_width_constraint(narrow_proboscis, narrow_flower)
            == 1.0
        )
        assert (
            self.matching_system.check_width_constraint(wide_proboscis, narrow_flower)
            < 1.0
        )

    def test_base_accessibility_calculation(self):
        """Test base accessibility calculation"""
        # Optimal length ratio
        optimal_accessibility = self.matching_system.calculate_base_accessibility(
            1.2, 1.0
        )
        assert optimal_accessibility == 1.0

        # Short proboscis
        short_accessibility = self.matching_system.calculate_base_accessibility(
            0.5, 1.0
        )
        assert short_accessibility < optimal_accessibility

        # Width constraint
        constrained_accessibility = self.matching_system.calculate_base_accessibility(
            1.2, 0.5
        )
        assert constrained_accessibility < optimal_accessibility

    def test_accessibility_level_determination(self):
        """Test accessibility level determination"""
        assert (
            self.matching_system.determine_accessibility_level(0.95)
            == AccessibilityLevel.EXCELLENT
        )
        assert (
            self.matching_system.determine_accessibility_level(0.75)
            == AccessibilityLevel.GOOD
        )
        assert (
            self.matching_system.determine_accessibility_level(0.45)
            == AccessibilityLevel.MODERATE
        )
        assert (
            self.matching_system.determine_accessibility_level(0.25)
            == AccessibilityLevel.POOR
        )
        assert (
            self.matching_system.determine_accessibility_level(0.05)
            == AccessibilityLevel.INACCESSIBLE
        )

    def test_nectar_efficiency_calculation(self):
        """Test nectar extraction efficiency"""
        # Optimal length ratio
        optimal_efficiency = self.matching_system.calculate_nectar_efficiency(1.1, 1.0)
        assert optimal_efficiency == 1.0

        # Too short
        short_efficiency = self.matching_system.calculate_nectar_efficiency(0.8, 1.0)
        assert short_efficiency < optimal_efficiency

        # Too long
        long_efficiency = self.matching_system.calculate_nectar_efficiency(1.5, 1.0)
        assert long_efficiency < optimal_efficiency

    def test_energy_cost_calculation(self):
        """Test energy cost calculation"""
        # Good accessibility - low cost
        good_cost = self.matching_system.calculate_energy_cost(
            1.1, AccessibilityLevel.GOOD
        )
        assert good_cost <= 1.0

        # Poor accessibility - high cost
        poor_cost = self.matching_system.calculate_energy_cost(
            0.7, AccessibilityLevel.POOR
        )
        assert poor_cost > good_cost

        # Inaccessible - very high cost
        inaccessible_cost = self.matching_system.calculate_energy_cost(
            0.5, AccessibilityLevel.INACCESSIBLE
        )
        assert inaccessible_cost > poor_cost

    def test_handling_time_calculation(self):
        """Test handling time calculation"""
        # Good accessibility - standard time
        good_time = self.matching_system.calculate_handling_time(
            1.1, AccessibilityLevel.GOOD
        )
        assert good_time == 1.0

        # Poor accessibility - longer time
        poor_time = self.matching_system.calculate_handling_time(
            0.7, AccessibilityLevel.POOR
        )
        assert poor_time > good_time

    def test_complete_accessibility_calculation(self):
        """Test complete accessibility calculation"""
        proboscis = ProboscisCharacteristics(length_mm=8.0, width_mm=0.2)

        # Accessible flower
        accessible_flower = FlowerSpecies(
            name="Accessible",
            bloom_start=100,
            bloom_end=200,
            nectar_production=5.0,
            pollen_production=2.0,
            flower_density=10.0,
            attractiveness=0.8,
            corolla_depth_mm=6.0,
            corolla_width_mm=2.0,
        )
        result = self.matching_system.calculate_accessibility(
            proboscis, accessible_flower
        )

        assert isinstance(result, AccessibilityResult)
        assert result.is_accessible()
        assert result.accessibility_score > 0.0
        assert result.nectar_extraction_efficiency > 0.0
        assert result.energy_cost_multiplier >= 1.0

        # Inaccessible flower (very deep)
        inaccessible_flower = FlowerSpecies(
            name="Inaccessible",
            bloom_start=100,
            bloom_end=200,
            nectar_production=5.0,
            pollen_production=2.0,
            flower_density=10.0,
            attractiveness=0.8,
            corolla_depth_mm=25.0,
            corolla_width_mm=2.0,
        )
        result = self.matching_system.calculate_accessibility(
            proboscis, inaccessible_flower
        )

        assert not result.is_accessible()
        assert result.accessibility_score == 0.0

    def test_flower_patch_evaluation(self):
        """Test evaluation of flower patch"""
        flowers = [
            FlowerSpecies(
                name="Shallow",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=3.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Medium",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=8.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Deep",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=15.0,
                corolla_width_mm=2.0,
            ),
        ]

        # Short-tongued species
        results = self.matching_system.evaluate_flower_patch(
            "Bombus_terrestris", flowers
        )

        assert len(results) == 3
        assert (
            results["Shallow"].accessibility_score
            > results["Medium"].accessibility_score
        )
        assert (
            results["Medium"].accessibility_score > results["Deep"].accessibility_score
        )

    def test_accessible_flower_filtering(self):
        """Test filtering to accessible flowers only"""
        flowers = [
            FlowerSpecies(
                name="Accessible1",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=3.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Accessible2",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=7.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Inaccessible",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=25.0,
                corolla_width_mm=2.0,
            ),
        ]

        accessible = self.matching_system.filter_accessible_flowers(
            "Bombus_terrestris", flowers
        )

        assert len(accessible) == 2
        assert all(flower.corolla_depth_mm < 20.0 for flower in accessible)

    def test_patch_accessibility_scoring(self):
        """Test patch accessibility scoring"""
        # High accessibility flowers
        good_flowers = [
            FlowerSpecies(
                name="Good1",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=20.0,
                attractiveness=0.8,
                corolla_depth_mm=3.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Good2",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=15.0,
                attractiveness=0.8,
                corolla_depth_mm=5.0,
                corolla_width_mm=2.0,
            ),
        ]

        # Mixed accessibility flowers
        mixed_flowers = [
            FlowerSpecies(
                name="Good",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=3.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Poor",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=20.0,
                corolla_width_mm=2.0,
            ),
        ]

        good_score = self.matching_system.calculate_patch_accessibility_score(
            "Bombus_terrestris", good_flowers
        )
        mixed_score = self.matching_system.calculate_patch_accessibility_score(
            "Bombus_terrestris", mixed_flowers
        )

        assert good_score > mixed_score

    def test_foraging_efficiency_modifier(self):
        """Test foraging efficiency modifier calculation"""
        flower = FlowerSpecies(
            name="Test",
            bloom_start=100,
            bloom_end=200,
            nectar_production=5.0,
            pollen_production=2.0,
            flower_density=10.0,
            attractiveness=0.8,
            corolla_depth_mm=6.0,
            corolla_width_mm=2.0,
        )

        modifiers = self.matching_system.get_foraging_efficiency_modifier(
            "Bombus_terrestris", flower
        )

        assert "nectar_efficiency" in modifiers
        assert "pollen_efficiency" in modifiers
        assert "energy_cost" in modifiers
        assert "handling_time" in modifiers
        assert "accessibility_score" in modifiers

        assert 0.0 <= modifiers["nectar_efficiency"] <= 1.0
        assert 0.0 <= modifiers["pollen_efficiency"] <= 1.0
        assert modifiers["energy_cost"] >= 1.0
        assert modifiers["handling_time"] >= 0.8

    def test_community_analysis(self):
        """Test multi-species community analysis"""
        species = ["Bombus_terrestris", "Bombus_hortorum", "Bombus_pascuorum"]
        flowers = [
            FlowerSpecies(
                name="Shallow",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=3.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Medium",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=8.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Deep",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=15.0,
                corolla_width_mm=2.0,
            ),
        ]

        analysis = self.matching_system.get_community_analysis(species, flowers)

        assert "species_accessibility" in analysis
        assert "flower_specialization" in analysis
        assert len(analysis["species_accessibility"]) == 3
        assert len(analysis["flower_specialization"]) == 3

        # Long-tongued species should access more flowers
        hortorum_access = analysis["species_accessibility"]["Bombus_hortorum"][
            "accessibility_count"
        ]
        terrestris_access = analysis["species_accessibility"]["Bombus_terrestris"][
            "accessibility_count"
        ]
        assert hortorum_access >= terrestris_access

    def test_optimal_flower_matches(self):
        """Test optimal flower matching"""
        flowers = [
            FlowerSpecies(
                name="Best",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=7.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Good",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=5.0,
                corolla_width_mm=2.0,
            ),
            FlowerSpecies(
                name="Poor",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
                corolla_depth_mm=20.0,
                corolla_width_mm=2.0,
            ),
        ]

        matches = self.matching_system.get_optimal_flower_matches(
            "Bombus_terrestris", flowers
        )

        # Should be sorted by accessibility score
        assert len(matches) >= 2  # At least 2 should be accessible
        for i in range(len(matches) - 1):
            assert matches[i][1] >= matches[i + 1][1]  # Descending order

    def test_corolla_data_update(self):
        """Test flower species corolla data update"""
        flowers = [
            FlowerSpecies(
                name="White Clover",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
            ),
            FlowerSpecies(
                name="Red Clover",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
            ),
            FlowerSpecies(
                name="Foxglove",
                bloom_start=100,
                bloom_end=200,
                nectar_production=5.0,
                pollen_production=2.0,
                flower_density=10.0,
                attractiveness=0.8,
            ),
        ]

        # Update with realistic data
        self.matching_system.update_flower_species_corolla_data(flowers)

        # Check that known species got updated
        white_clover = flowers[0]
        red_clover = flowers[1]
        foxglove = flowers[2]

        assert white_clover.corolla_depth_mm == 4.5
        assert red_clover.corolla_depth_mm == 9.2
        assert foxglove.corolla_depth_mm == 25.0

        # Deep flowers should have reduced accessibility
        assert foxglove.nectar_accessibility < white_clover.nectar_accessibility


class TestProboscisCharacteristics:
    """Test proboscis characteristics"""

    def test_proboscis_creation(self):
        """Test proboscis characteristics creation"""
        proboscis = ProboscisCharacteristics(
            length_mm=10.0, width_mm=0.25, flexibility=0.9, extension_efficiency=0.95
        )

        assert proboscis.length_mm == 10.0
        assert proboscis.width_mm == 0.25
        assert proboscis.flexibility == 0.9
        assert proboscis.extension_efficiency == 0.95

    def test_effective_length_calculation(self):
        """Test effective length calculation"""
        flexible_proboscis = ProboscisCharacteristics(length_mm=10.0, flexibility=0.9)
        rigid_proboscis = ProboscisCharacteristics(length_mm=10.0, flexibility=0.7)

        flexible_length = flexible_proboscis.get_effective_length()
        rigid_length = rigid_proboscis.get_effective_length()

        assert flexible_length == 9.0
        assert rigid_length == 7.0
        assert flexible_length > rigid_length


class TestAccessibilityResult:
    """Test accessibility result"""

    def test_accessibility_result_creation(self):
        """Test accessibility result creation"""
        result = AccessibilityResult(
            accessibility_level=AccessibilityLevel.GOOD,
            accessibility_score=0.75,
            nectar_extraction_efficiency=0.8,
            pollen_extraction_efficiency=0.85,
            energy_cost_multiplier=1.2,
            handling_time_multiplier=1.1,
        )

        assert result.accessibility_level == AccessibilityLevel.GOOD
        assert result.accessibility_score == 0.75
        assert result.nectar_extraction_efficiency == 0.8
        assert result.pollen_extraction_efficiency == 0.85
        assert result.energy_cost_multiplier == 1.2
        assert result.handling_time_multiplier == 1.1

    def test_accessibility_checking(self):
        """Test accessibility checking"""
        accessible_result = AccessibilityResult(
            accessibility_level=AccessibilityLevel.GOOD,
            accessibility_score=0.75,
            nectar_extraction_efficiency=0.8,
            pollen_extraction_efficiency=0.85,
            energy_cost_multiplier=1.2,
            handling_time_multiplier=1.1,
        )

        inaccessible_result = AccessibilityResult(
            accessibility_level=AccessibilityLevel.INACCESSIBLE,
            accessibility_score=0.0,
            nectar_extraction_efficiency=0.0,
            pollen_extraction_efficiency=0.0,
            energy_cost_multiplier=10.0,
            handling_time_multiplier=5.0,
        )

        assert accessible_result.is_accessible()
        assert not inaccessible_result.is_accessible()
