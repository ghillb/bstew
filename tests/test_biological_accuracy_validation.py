"""
Biological Accuracy Validation Tests for BSTEW Bumblebee Models
===============================================================

CRITICAL: These tests validate model outputs against published field data
and literature to ensure biological accuracy in conservation research.

Validation categories:
1. Field data comparison - model parameters vs observed measurements
2. Literature validation - behaviors vs published studies
3. Cross-species comparison - biological distinctions between species
4. Honey bee vs bumblebee - key ecological differences

Based on:
- Goulson (2010): Bumblebees: behaviour, ecology, and conservation
- Knight et al. (2005): An interspecific comparison of foraging range
- Heinrich (1979): Bumblebee Economics: Thermoregulation and energetics
- Buchmann (1983): Buzz pollination in angiosperms
- Carvell et al. (2006): Comparing the efficacy of agri-environment schemes
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from bstew.core.species_parameters import (
    create_literature_validated_species,
    BumblebeeParameterValidator,
)
from bstew.core.species_integration import SpeciesParameterIntegrator
from bstew.core.bumblebee_behaviors import (
    BuzzPollinationBehavior,
    ThermalRegulationBehavior,
    SonicationFlowerType,
)


@dataclass
class FieldDataPoint:
    """Structure for field observation data"""
    species: str
    parameter: str
    observed_value: float
    measurement_unit: str
    study_location: str
    sample_size: int
    study_reference: str
    confidence_interval: Tuple[float, float]
    notes: str = ""


@dataclass
class LiteratureExpectation:
    """Expected behavior based on literature"""
    behavior: str
    species: str
    expected_range: Tuple[float, float]
    literature_source: str
    biological_significance: str


class TestFieldDataValidation:
    """Validate model parameters against published field measurements"""

    @pytest.fixture
    def field_data_points(self) -> List[FieldDataPoint]:
        """Published field measurements for validation"""

        return [
            # Foraging range measurements
            FieldDataPoint(
                species="Bombus_terrestris",
                parameter="foraging_range_m",
                observed_value=863.0,
                measurement_unit="meters",
                study_location="UK agricultural landscape",
                sample_size=156,
                study_reference="Knight et al. (2005)",
                confidence_interval=(672.0, 1054.0),
                notes="Radio-tracking study in mixed farmland"
            ),
            FieldDataPoint(
                species="Bombus_pascuorum",
                parameter="foraging_range_m",
                observed_value=945.0,
                measurement_unit="meters",
                study_location="German agricultural areas",
                sample_size=45,
                study_reference="Walther-Hellwig & Frankl (2000)",
                confidence_interval=(780.0, 1110.0),
                notes="Medium-tongued species, longer range than expected"
            ),
            FieldDataPoint(
                species="Bombus_lapidarius",
                parameter="foraging_range_m",
                observed_value=1200.0,
                measurement_unit="meters",
                study_location="European grasslands",
                sample_size=67,
                study_reference="Knight et al. (2005)",
                confidence_interval=(950.0, 1450.0),
                notes="Heat-tolerant species with extended range"
            ),

            # Colony size measurements
            FieldDataPoint(
                species="Bombus_terrestris",
                parameter="max_colony_size",
                observed_value=267.0,
                measurement_unit="individuals",
                study_location="UK colonies",
                sample_size=89,
                study_reference="Goulson (2010)",
                confidence_interval=(180.0, 354.0),
                notes="Wild colonies, commercial can reach 400+"
            ),
            FieldDataPoint(
                species="Bombus_pascuorum",
                parameter="max_colony_size",
                observed_value=142.0,
                measurement_unit="individuals",
                study_location="European sites",
                sample_size=52,
                study_reference="Plowright & Jay (1977)",
                confidence_interval=(95.0, 189.0),
                notes="Smaller colonies typical for this species"
            ),

            # Temperature tolerance measurements
            FieldDataPoint(
                species="Bombus_terrestris",
                parameter="temperature_tolerance_min_c",
                observed_value=4.2,
                measurement_unit="celsius",
                study_location="Laboratory controlled conditions",
                sample_size=23,
                study_reference="Heinrich (1979)",
                confidence_interval=(2.8, 5.6),
                notes="Flight muscle activity threshold"
            ),
            FieldDataPoint(
                species="Bombus_lapidarius",
                parameter="temperature_tolerance_min_c",
                observed_value=8.1,
                measurement_unit="celsius",
                study_location="Field observations",
                sample_size=21,
                study_reference="Stone et al. (1999)",
                confidence_interval=(6.5, 9.7),
                notes="Heat-adapted species, less cold tolerant"
            ),

            # Proboscis length measurements
            FieldDataPoint(
                species="Bombus_terrestris",
                parameter="proboscis_length_mm",
                observed_value=7.1,
                measurement_unit="millimeters",
                study_location="UK museum specimens",
                sample_size=67,
                study_reference="Goulson & Darvill (2004)",
                confidence_interval=(6.9, 7.3),
                notes="Short-tongued generalist species"
            ),
            FieldDataPoint(
                species="Bombus_pascuorum",
                parameter="proboscis_length_mm",
                observed_value=11.4,
                measurement_unit="millimeters",
                study_location="European specimens",
                sample_size=41,
                study_reference="Harder (1982)",
                confidence_interval=(10.8, 12.0),
                notes="Medium-tongued specialist"
            ),
        ]

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    def test_foraging_range_field_validation(self, field_data_points, species_params):
        """Validate foraging ranges against field measurements"""

        foraging_data = [fp for fp in field_data_points if fp.parameter == "foraging_range_m"]

        for field_point in foraging_data:
            if field_point.species in species_params:
                model_params = species_params[field_point.species]
                model_value = model_params.foraging_range_m

                # Check if model value falls within confidence interval
                ci_min, ci_max = field_point.confidence_interval

                assert ci_min <= model_value <= ci_max, (
                    f"Model foraging range for {field_point.species} ({model_value}m) "
                    f"outside field data CI [{ci_min}, {ci_max}]m from {field_point.study_reference}"
                )

                # Check reasonable proximity to observed mean
                observed_value = field_point.observed_value
                relative_error = abs(model_value - observed_value) / observed_value

                assert relative_error <= 0.30, (
                    f"Model foraging range for {field_point.species} differs by "
                    f"{relative_error:.1%} from field observation ({field_point.study_reference})"
                )

    def test_colony_size_field_validation(self, field_data_points, species_params):
        """Validate colony sizes against field measurements"""

        colony_data = [fp for fp in field_data_points if fp.parameter == "max_colony_size"]

        for field_point in colony_data:
            if field_point.species in species_params:
                model_params = species_params[field_point.species]
                model_value = float(model_params.max_colony_size)

                # Check if model value falls within confidence interval
                ci_min, ci_max = field_point.confidence_interval

                assert ci_min <= model_value <= ci_max, (
                    f"Model colony size for {field_point.species} ({model_value} individuals) "
                    f"outside field data CI [{ci_min}, {ci_max}] from {field_point.study_reference}"
                )

                # Check reasonable proximity to observed mean
                observed_value = field_point.observed_value
                relative_error = abs(model_value - observed_value) / observed_value

                assert relative_error <= 0.25, (
                    f"Model colony size for {field_point.species} differs by "
                    f"{relative_error:.1%} from field observation ({field_point.study_reference})"
                )

    def test_temperature_tolerance_field_validation(self, field_data_points, species_params):
        """Validate temperature tolerances against field measurements"""

        temp_data = [fp for fp in field_data_points if fp.parameter == "temperature_tolerance_min_c"]

        for field_point in temp_data:
            if field_point.species in species_params:
                model_params = species_params[field_point.species]
                model_value = model_params.temperature_tolerance_min_c

                # Check if model value falls within confidence interval
                ci_min, ci_max = field_point.confidence_interval

                assert ci_min <= model_value <= ci_max, (
                    f"Model temperature tolerance for {field_point.species} ({model_value}°C) "
                    f"outside field data CI [{ci_min}, {ci_max}]°C from {field_point.study_reference}"
                )

    def test_proboscis_length_field_validation(self, field_data_points, species_params):
        """Validate proboscis lengths against museum/field measurements"""

        proboscis_data = [fp for fp in field_data_points if fp.parameter == "proboscis_length_mm"]

        for field_point in proboscis_data:
            if field_point.species in species_params:
                model_params = species_params[field_point.species]
                model_value = model_params.proboscis_characteristics.length_mm

                # Check if model value falls within confidence interval
                ci_min, ci_max = field_point.confidence_interval

                assert ci_min <= model_value <= ci_max, (
                    f"Model proboscis length for {field_point.species} ({model_value}mm) "
                    f"outside field data CI [{ci_min}, {ci_max}]mm from {field_point.study_reference}"
                )

    def test_field_data_coverage(self, field_data_points, species_params):
        """Ensure adequate field data coverage for validation"""

        # Check we have field data for all three target species
        species_in_data = set(fp.species for fp in field_data_points)
        target_species = {"Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"}

        assert target_species.issubset(species_in_data), (
            f"Missing field data for species: {target_species - species_in_data}"
        )

        # Check we have multiple parameters per species
        for species in target_species:
            species_params_count = len([fp for fp in field_data_points if fp.species == species])
            assert species_params_count >= 2, (
                f"Insufficient field data points for {species}: {species_params_count} (need ≥2)"
            )

        # Check all field data has adequate sample sizes
        for field_point in field_data_points:
            assert field_point.sample_size >= 15, (
                f"Small sample size for {field_point.species} {field_point.parameter}: "
                f"n={field_point.sample_size} (need ≥15)"
            )


class TestLiteratureValidation:
    """Validate behaviors against published literature expectations"""

    @pytest.fixture
    def literature_expectations(self) -> List[LiteratureExpectation]:
        """Expected behaviors based on literature"""

        return [
            # Buzz pollination expectations from Buchmann (1983)
            LiteratureExpectation(
                behavior="buzz_pollination_frequency",
                species="all_bumblebees",
                expected_range=(50.0, 1000.0),
                literature_source="Buchmann (1983)",
                biological_significance="Wing beat modulation for sonication"
            ),
            LiteratureExpectation(
                behavior="buzz_pollination_effectiveness",
                species="Bombus_terrestris",
                expected_range=(0.7, 0.95),
                literature_source="De Luca & Vallejo-Marín (2013)",
                biological_significance="Most efficient buzz pollinator"
            ),

            # Thermal regulation expectations from Heinrich (1979)
            LiteratureExpectation(
                behavior="cold_weather_advantage",
                species="all_bumblebees",
                expected_range=(3.0, 12.0),
                literature_source="Heinrich (1979)",
                biological_significance="Temperature advantage over honey bees (12°C minimum)"
            ),
            LiteratureExpectation(
                behavior="preflight_warmup_time",
                species="all_bumblebees",
                expected_range=(10.0, 300.0),
                literature_source="Heinrich (1979)",
                biological_significance="Muscle warming before flight in cold conditions"
            ),

            # Social behavior expectations
            LiteratureExpectation(
                behavior="social_recruitment_rate",
                species="all_bumblebees",
                expected_range=(0.01, 0.10),
                literature_source="Dornhaus & Chittka (2004)",
                biological_significance="Much lower than honey bees (30-70%)"
            ),
            LiteratureExpectation(
                behavior="memory_capacity_patches",
                species="all_bumblebees",
                expected_range=(5.0, 20.0),
                literature_source="Menzel et al. (1996)",
                biological_significance="Individual memory limitation vs honey bee sharing"
            ),

            # Foraging behavior expectations
            LiteratureExpectation(
                behavior="patch_fidelity",
                species="Bombus_pascuorum",
                expected_range=(0.7, 0.9),
                literature_source="Carvell et al. (2006)",
                biological_significance="Higher fidelity in specialist species"
            ),
        ]

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    def test_buzz_pollination_literature_validation(self, literature_expectations, species_params):
        """Validate buzz pollination against literature expectations"""

        buzz_expectations = [le for le in literature_expectations
                           if le.behavior.startswith("buzz_pollination")]

        for expectation in buzz_expectations:
            min_expected, max_expected = expectation.expected_range

            if expectation.species == "all_bumblebees":
                test_species = species_params.keys()
            else:
                test_species = [expectation.species]

            for species_name in test_species:
                if species_name in species_params:
                    params = species_params[species_name]
                    buzz_behavior = BuzzPollinationBehavior(params)

                    if expectation.behavior == "buzz_pollination_frequency":
                        actual_value = buzz_behavior.buzz_efficiency.vibration_frequency_hz
                    elif expectation.behavior == "buzz_pollination_effectiveness":
                        actual_value = buzz_behavior.buzz_efficiency.pollen_release_efficiency
                    else:
                        continue

                    assert min_expected <= actual_value <= max_expected, (
                        f"{expectation.behavior} for {species_name} ({actual_value}) "
                        f"outside literature range [{min_expected}, {max_expected}] "
                        f"from {expectation.literature_source}"
                    )

    def test_thermal_regulation_literature_validation(self, literature_expectations, species_params):
        """Validate thermal regulation against literature expectations"""

        thermal_expectations = [le for le in literature_expectations
                              if le.behavior in ["cold_weather_advantage", "preflight_warmup_time"]]

        for expectation in thermal_expectations:
            min_expected, max_expected = expectation.expected_range

            if expectation.species == "all_bumblebees":
                test_species = species_params.keys()
            else:
                test_species = [expectation.species]

            for species_name in test_species:
                if species_name in species_params:
                    params = species_params[species_name]
                    thermal_behavior = ThermalRegulationBehavior(params)

                    if expectation.behavior == "cold_weather_advantage":
                        actual_value = thermal_behavior.thermal_capacity.cold_tolerance_advantage
                    elif expectation.behavior == "preflight_warmup_time":
                        actual_value = thermal_behavior.thermal_capacity.preflight_warmup_time_s
                    else:
                        continue

                    assert min_expected <= actual_value <= max_expected, (
                        f"{expectation.behavior} for {species_name} ({actual_value}) "
                        f"outside literature range [{min_expected}, {max_expected}] "
                        f"from {expectation.literature_source}"
                    )

    def test_social_behavior_literature_validation(self, literature_expectations, species_params):
        """Validate social behaviors against literature expectations"""

        social_expectations = [le for le in literature_expectations
                             if le.behavior in ["social_recruitment_rate", "memory_capacity_patches"]]

        for expectation in social_expectations:
            min_expected, max_expected = expectation.expected_range

            if expectation.species == "all_bumblebees":
                test_species = species_params.keys()
            else:
                test_species = [expectation.species]

            for species_name in test_species:
                if species_name in species_params:
                    params = species_params[species_name]

                    if expectation.behavior == "social_recruitment_rate":
                        actual_value = params.social_recruitment_rate
                    elif expectation.behavior == "memory_capacity_patches":
                        actual_value = float(params.memory_capacity_patches)
                    else:
                        continue

                    assert min_expected <= actual_value <= max_expected, (
                        f"{expectation.behavior} for {species_name} ({actual_value}) "
                        f"outside literature range [{min_expected}, {max_expected}] "
                        f"from {expectation.literature_source}"
                    )

    def test_specialist_behavior_validation(self, literature_expectations, species_params):
        """Validate species-specific specialist behaviors"""

        specialist_expectations = [le for le in literature_expectations
                                 if le.species != "all_bumblebees"]

        for expectation in specialist_expectations:
            species_name = expectation.species
            if species_name in species_params:
                params = species_params[species_name]
                min_expected, max_expected = expectation.expected_range

                if expectation.behavior == "patch_fidelity":
                    actual_value = params.patch_fidelity_strength

                    assert min_expected <= actual_value <= max_expected, (
                        f"{expectation.behavior} for {species_name} ({actual_value}) "
                        f"outside literature range [{min_expected}, {max_expected}] "
                        f"from {expectation.literature_source}"
                    )


class TestCrossSpeciesComparison:
    """Validate biological distinctions between species"""

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    def test_proboscis_length_ranking(self, species_params):
        """Test that species rank correctly by proboscis length"""

        # Expected ranking: pascuorum (longest) > lapidarius > terrestris (shortest)
        terrestris = species_params["Bombus_terrestris"]
        pascuorum = species_params["Bombus_pascuorum"]
        lapidarius = species_params["Bombus_lapidarius"]

        terrestris_length = terrestris.proboscis_characteristics.length_mm
        pascuorum_length = pascuorum.proboscis_characteristics.length_mm
        lapidarius_length = lapidarius.proboscis_characteristics.length_mm

        # Biological expectation: B. pascuorum is medium-tongued specialist
        assert pascuorum_length > lapidarius_length > terrestris_length, (
            f"Proboscis length ranking incorrect: "
            f"pascuorum={pascuorum_length}mm, lapidarius={lapidarius_length}mm, "
            f"terrestris={terrestris_length}mm"
        )

        # Check minimum differences are biologically meaningful
        assert pascuorum_length - terrestris_length >= 3.0, (
            "Insufficient proboscis length difference between specialist and generalist"
        )

    def test_colony_size_ranking(self, species_params):
        """Test that species rank correctly by colony size"""

        # Expected ranking: terrestris (largest) > lapidarius > pascuorum (smallest)
        terrestris = species_params["Bombus_terrestris"]
        pascuorum = species_params["Bombus_pascuorum"]
        lapidarius = species_params["Bombus_lapidarius"]

        terrestris_size = terrestris.max_colony_size
        pascuorum_size = pascuorum.max_colony_size
        lapidarius_size = lapidarius.max_colony_size

        # B. terrestris should have largest colonies
        assert terrestris_size > lapidarius_size, (
            f"B. terrestris colony size ({terrestris_size}) should exceed "
            f"B. lapidarius ({lapidarius_size})"
        )
        assert terrestris_size > pascuorum_size, (
            f"B. terrestris colony size ({terrestris_size}) should exceed "
            f"B. pascuorum ({pascuorum_size})"
        )

        # Check biologically meaningful differences
        assert terrestris_size >= pascuorum_size * 1.5, (
            "B. terrestris colonies should be ≥50% larger than B. pascuorum"
        )

    def test_temperature_tolerance_ranking(self, species_params):
        """Test that species rank correctly by cold tolerance"""

        terrestris = species_params["Bombus_terrestris"]
        pascuorum = species_params["Bombus_pascuorum"]
        lapidarius = species_params["Bombus_lapidarius"]

        terrestris_min = terrestris.temperature_tolerance_min_c
        pascuorum_min = pascuorum.temperature_tolerance_min_c
        lapidarius_min = lapidarius.temperature_tolerance_min_c

        # B. lapidarius is heat-adapted, should be least cold tolerant
        assert lapidarius_min >= terrestris_min, (
            f"B. lapidarius min temp ({lapidarius_min}°C) should be ≥ "
            f"B. terrestris ({terrestris_min}°C) - heat adaptation"
        )

        # All should be well below honey bee minimum (12°C)
        honey_bee_min = 12.0
        for species, min_temp in [
            ("terrestris", terrestris_min),
            ("pascuorum", pascuorum_min),
            ("lapidarius", lapidarius_min)
        ]:
            assert min_temp < honey_bee_min, (
                f"B. {species} min temp ({min_temp}°C) should be < honey bee ({honey_bee_min}°C)"
            )

    def test_foraging_behavior_specialization(self, species_params):
        """Test that species show appropriate foraging specializations"""

        terrestris = species_params["Bombus_terrestris"]
        pascuorum = species_params["Bombus_pascuorum"]
        lapidarius = species_params["Bombus_lapidarius"]

        # B. pascuorum should show higher patch fidelity (specialist)
        assert pascuorum.patch_fidelity_strength > terrestris.patch_fidelity_strength, (
            f"B. pascuorum patch fidelity ({pascuorum.patch_fidelity_strength}) "
            f"should exceed B. terrestris ({terrestris.patch_fidelity_strength}) - specialization"
        )

        # B. terrestris should show higher competition strength (dominant)
        assert terrestris.competition_strength > pascuorum.competition_strength, (
            f"B. terrestris competition strength ({terrestris.competition_strength}) "
            f"should exceed B. pascuorum ({pascuorum.competition_strength}) - dominance"
        )
        assert terrestris.competition_strength > lapidarius.competition_strength, (
            f"B. terrestris competition strength ({terrestris.competition_strength}) "
            f"should exceed B. lapidarius ({lapidarius.competition_strength}) - dominance"
        )

    def test_phenological_differences(self, species_params):
        """Test that species show appropriate phenological timing"""

        terrestris = species_params["Bombus_terrestris"]
        pascuorum = species_params["Bombus_pascuorum"]
        lapidarius = species_params["Bombus_lapidarius"]

        # Expected emergence order: terrestris (early) < lapidarius < pascuorum (late)
        terrestris_emergence = terrestris.emerging_day_mean
        pascuorum_emergence = pascuorum.emerging_day_mean
        lapidarius_emergence = lapidarius.emerging_day_mean

        assert terrestris_emergence < lapidarius_emergence < pascuorum_emergence, (
            f"Emergence timing incorrect: terrestris={terrestris_emergence}, "
            f"lapidarius={lapidarius_emergence}, pascuorum={pascuorum_emergence}"
        )

        # B. pascuorum should have extended late season foraging
        terrestris_season = terrestris.active_season_end - terrestris.active_season_start
        pascuorum_season = pascuorum.active_season_end - pascuorum.active_season_start

        # B. pascuorum emerges later but extends further into autumn
        assert pascuorum.active_season_end >= terrestris.active_season_end, (
            f"B. pascuorum season end ({pascuorum.active_season_end}) should be ≥ "
            f"B. terrestris ({terrestris.active_season_end}) - extended late foraging"
        )


class TestHoneyBeeComparison:
    """Validate key ecological differences from honey bees"""

    @pytest.fixture
    def species_params(self):
        """Get literature-validated species parameters"""
        return create_literature_validated_species()

    def test_foraging_range_differences(self, species_params):
        """Test bumblebee vs honey bee foraging range differences"""

        # Honey bee foraging range: 5000-6000m typically
        honey_bee_typical_range = 5500.0  # meters
        honey_bee_max_range = 10000.0     # meters (reported maximum)

        for species_name, params in species_params.items():
            bumblebee_range = params.foraging_range_m

            # All bumblebees should forage much closer to colony
            assert bumblebee_range < honey_bee_typical_range * 0.5, (
                f"{species_name} foraging range ({bumblebee_range}m) should be "
                f"<50% of honey bee typical range ({honey_bee_typical_range}m)"
            )

            # Should be well below honey bee maximum
            assert bumblebee_range < honey_bee_max_range * 0.3, (
                f"{species_name} foraging range ({bumblebee_range}m) should be "
                f"<30% of honey bee maximum ({honey_bee_max_range}m)"
            )

    def test_colony_size_differences(self, species_params):
        """Test bumblebee vs honey bee colony size differences"""

        # Honey bee colony sizes: 20,000-80,000 individuals
        honey_bee_min_size = 20000
        honey_bee_typical_size = 50000

        for species_name, params in species_params.items():
            bumblebee_size = params.max_colony_size

            # All bumblebee colonies should be much smaller
            assert bumblebee_size < honey_bee_min_size * 0.05, (
                f"{species_name} colony size ({bumblebee_size}) should be "
                f"<5% of honey bee minimum ({honey_bee_min_size})"
            )

            # Should be orders of magnitude smaller
            size_ratio = bumblebee_size / honey_bee_typical_size
            assert size_ratio < 0.01, (
                f"{species_name} colony size ratio to honey bees ({size_ratio:.3f}) "
                f"should be <1% (orders of magnitude smaller)"
            )

    def test_temperature_tolerance_advantages(self, species_params):
        """Test bumblebee cold tolerance advantages over honey bees"""

        honey_bee_min_temp = 12.0  # °C - honey bee flight minimum

        for species_name, params in species_params.items():
            thermal_behavior = ThermalRegulationBehavior(params)

            bumblebee_min_temp = thermal_behavior.thermal_capacity.min_flight_temperature_c
            temperature_advantage = honey_bee_min_temp - bumblebee_min_temp

            # All bumblebees should have significant temperature advantage
            assert temperature_advantage >= 3.0, (
                f"{species_name} temperature advantage ({temperature_advantage:.1f}°C) "
                f"should be ≥3°C over honey bees"
            )

            # Test exclusive foraging opportunities
            test_temps = [8.0, 6.0, 4.0]  # Cold morning temperatures

            for test_temp in test_temps:
                comparison = thermal_behavior.compare_to_honey_bee_capability(test_temp)

                if thermal_behavior.can_fly_at_temperature(test_temp):
                    assert not comparison["honey_bee_can_fly"], (
                        f"Honey bees should not fly at {test_temp}°C"
                    )
                    assert comparison["exclusive_foraging_opportunity"], (
                        f"{species_name} should have exclusive access at {test_temp}°C"
                    )

    def test_social_recruitment_differences(self, species_params):
        """Test bumblebee vs honey bee social recruitment differences"""

        # Honey bee recruitment: 30-70% of foragers recruit nestmates
        honey_bee_recruitment_rate = 0.5  # 50% typical

        for species_name, params in species_params.items():
            bumblebee_recruitment = params.social_recruitment_rate

            # Bumblebees should have much lower social recruitment
            assert bumblebee_recruitment < honey_bee_recruitment_rate * 0.2, (
                f"{species_name} social recruitment ({bumblebee_recruitment:.3f}) "
                f"should be <20% of honey bee rate ({honey_bee_recruitment_rate})"
            )

            # Should emphasize individual decisions
            individual_decision_rate = 1.0 - bumblebee_recruitment
            assert individual_decision_rate >= 0.9, (
                f"{species_name} individual decision rate ({individual_decision_rate:.3f}) "
                f"should be ≥90% (vs honey bee dance communication)"
            )

    def test_buzz_pollination_unique_capability(self, species_params):
        """Test that buzz pollination is unique to bumblebees"""

        # Document that honey bees cannot perform buzz pollination
        # This is critical for crops like tomatoes, blueberries, eggplants

        important_crops = [
            SonicationFlowerType.TOMATO,
            SonicationFlowerType.BLUEBERRY,
            SonicationFlowerType.EGGPLANT
        ]

        for species_name, params in species_params.items():
            buzz_behavior = BuzzPollinationBehavior(params)

            for crop in important_crops:
                if buzz_behavior.can_buzz_pollinate(crop):
                    # Test successful buzz pollination
                    result = buzz_behavior.perform_buzz_pollination(crop)

                    assert result["success"], (
                        f"{species_name} should successfully buzz pollinate {crop.value}"
                    )
                    assert result["pollen_collected_mg"] > 0.0, (
                        f"{species_name} should collect pollen from {crop.value}"
                    )

                    # Verify sonication frequency is within biological range
                    frequency = buzz_behavior.buzz_efficiency.vibration_frequency_hz
                    assert 50.0 <= frequency <= 1000.0, (
                        f"{species_name} sonication frequency ({frequency}Hz) "
                        f"outside biological range for buzz pollination"
                    )

    def test_memory_system_differences(self, species_params):
        """Test bumblebee individual memory vs honey bee dance communication"""

        # Honey bees: unlimited spatial information sharing via dance
        # Bumblebees: limited individual memory capacity

        for species_name, params in species_params.items():
            memory_capacity = params.memory_capacity_patches

            # Bumblebees should have limited memory capacity
            assert memory_capacity <= 20, (
                f"{species_name} memory capacity ({memory_capacity} patches) "
                f"should be limited vs honey bee unlimited spatial sharing"
            )

            # Should be sufficient for individual foraging efficiency
            assert memory_capacity >= 5, (
                f"{species_name} memory capacity ({memory_capacity} patches) "
                f"should be sufficient for effective foraging"
            )

    def test_lifecycle_differences(self, species_params):
        """Test bumblebee annual vs honey bee perennial lifecycle"""

        for species_name, params in species_params.items():
            # Bumblebees have annual lifecycle - colony dies each winter
            season_length = params.active_season_end - params.active_season_start

            # Active season should be <365 days (annual)
            assert season_length < 300, (
                f"{species_name} active season ({season_length} days) "
                f"should be <300 days (annual lifecycle)"
            )

            # Should have reasonable seasonal activity
            assert season_length >= 150, (
                f"{species_name} active season ({season_length} days) "
                f"should be ≥150 days for effective foraging"
            )

            # Colony size should reflect annual reproduction strategy
            colony_size = params.max_colony_size
            assert colony_size < 500, (
                f"{species_name} max colony size ({colony_size}) "
                f"should be <500 (annual reproduction vs honey bee perennial)"
            )


class TestModelIntegration:
    """Test integrated model behavior for biological realism"""

    @pytest.fixture
    def integrator(self):
        """Species parameter integrator"""
        return SpeciesParameterIntegrator()

    def test_species_system_integration(self, integrator):
        """Test that all species integrate correctly with bumblebee systems"""

        target_species = ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]

        for species_name in target_species:
            # Test system validation passes
            validation = integrator.validate_species_configuration(species_name)

            assert validation["valid"], (
                f"Species configuration invalid for {species_name}: "
                f"{validation.get('error', 'Unknown error')}"
            )

            # Test literature validation success rate
            lit_validation = validation["literature_validation"]
            success_rate = lit_validation["validation_success_rate"]

            assert success_rate == 1.0, (
                f"Literature validation failed for {species_name}: "
                f"{success_rate:.1%} success rate"
            )

            # Test system integration components
            integration = validation["system_integration"]

            for system in ["communication_system", "colony_lifecycle", "foraging_parameters"]:
                assert integration[system]["valid"], (
                    f"{system} integration failed for {species_name}: "
                    f"{integration[system].get('error', 'Unknown error')}"
                )

    def test_realistic_scenario_modeling(self, integrator):
        """Test realistic ecological scenarios"""

        # Scenario: Early spring morning (8°C) with tomato crop available
        morning_temp = 8.0
        species_name = "Bombus_terrestris"

        # Get species parameters and behaviors
        params = integrator.species_parameters[species_name]
        thermal_behavior = ThermalRegulationBehavior(params)
        buzz_behavior = BuzzPollinationBehavior(params)

        # Test thermal capability
        can_fly = thermal_behavior.can_fly_at_temperature(morning_temp)
        honey_bee_comparison = thermal_behavior.compare_to_honey_bee_capability(morning_temp)

        if can_fly:
            # Bumblebee should have exclusive access
            assert honey_bee_comparison["exclusive_foraging_opportunity"], (
                f"Bumblebee should have exclusive access at {morning_temp}°C"
            )

            # Test warm-up behavior
            warmup_result = thermal_behavior.perform_preflight_warmup(morning_temp)
            assert warmup_result["success"], "Warm-up should succeed in flyable conditions"

            # Test buzz pollination capability
            pollination_result = buzz_behavior.perform_buzz_pollination(SonicationFlowerType.TOMATO)
            assert pollination_result["success"], "Should successfully buzz pollinate tomatoes"

            # Calculate total energy budget
            flight_cost = thermal_behavior.calculate_flight_energy_cost(morning_temp, 120.0)  # 2 min
            total_energy = (
                warmup_result["energy_expended_j"] +
                flight_cost["total_energy_j"] +
                pollination_result["energy_expended_j"]
            )

            # Total energy should be realistic for bumblebee physiology (adjusted for cold weather costs)
            assert 0.1 <= total_energy <= 30.0, (
                f"Total energy cost ({total_energy:.2f}J) outside realistic range"
            )

    def test_conservation_research_validity(self, integrator):
        """Test model validity for conservation research applications"""

        target_species = ["Bombus_terrestris", "Bombus_pascuorum", "Bombus_lapidarius"]

        for species_name in target_species:
            summary = integrator.get_species_summary(species_name)

            # Test biological parameter realism
            bio_params = summary["biological_parameters"]

            # Foraging parameters should enable landscape-scale modeling
            assert 100 <= bio_params["foraging_range_m"] <= 2000, (
                f"Foraging range for {species_name} outside landscape modeling bounds"
            )

            # Colony sizes should reflect conservation unit scales
            assert 50 <= bio_params["max_colony_size"] <= 500, (
                f"Colony size for {species_name} outside conservation unit bounds"
            )

            # Temperature tolerance should capture climate change impacts
            assert bio_params["temperature_tolerance_min_c"] < 12.0, (
                f"Temperature tolerance for {species_name} should show climate advantage"
            )

            # Test phenology for seasonal management planning
            phenology = summary["phenology"]
            season_length = phenology["season_length_days"]

            assert 120 <= season_length <= 280, (
                f"Active season length for {species_name} outside management planning range"
            )

            # Test ecological traits for habitat assessment
            traits = summary["ecological_traits"]

            assert 0.0 <= traits["cold_tolerance"] <= 1.0, "Cold tolerance should be normalized"
            assert 0.0 <= traits["drought_tolerance"] <= 1.0, "Drought tolerance should be normalized"
            assert 0.0 <= traits["competition_strength"] <= 1.0, "Competition strength should be normalized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
