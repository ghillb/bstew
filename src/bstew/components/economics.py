"""
Economic Assessment System for BSTEW
====================================

Provides comprehensive economic valuation of pollinator services, cost-benefit
analysis for conservation strategies, and agricultural impact assessment.

This module transforms biological simulation data into economic insights for
evidence-based policy making and agricultural decision support.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import logging
from dataclasses import dataclass

from .species_system import SpeciesSystem


class CropType(Enum):
    """Agricultural crop types with pollinator dependencies"""

    APPLES = "apples"
    BLUEBERRIES = "blueberries"
    CHERRIES = "cherries"
    STRAWBERRIES = "strawberries"
    OILSEED_RAPE = "oilseed_rape"
    SUNFLOWER = "sunflower"
    CLOVER = "clover"
    BUCKWHEAT = "buckwheat"
    ALMONDS = "almonds"
    SQUASH = "squash"


@dataclass
class CropEconomics:
    """Economic parameters for agricultural crops"""

    crop_type: CropType
    price_per_kg: float  # Market price per kilogram
    yield_baseline_kg_ha: float  # Baseline yield without pollinators
    yield_with_pollinators_kg_ha: float  # Yield with optimal pollination
    pollinator_dependency: float  # Fraction dependent on pollinators (0-1)
    hectares_planted: float  # Area under cultivation
    production_cost_per_ha: float  # Production costs per hectare
    harvest_cost_per_kg: float  # Harvest and processing costs


class EconomicScenario(BaseModel):
    """Configuration for economic assessment scenarios"""

    model_config = {"validate_assignment": True}

    scenario_name: str = Field(description="Name of the economic scenario")
    crop_data: Dict[str, CropEconomics] = Field(description="Crop economic parameters")
    stewardship_costs: Dict[str, float] = Field(
        description="Cost per hectare for stewardship strategies"
    )
    discount_rate: float = Field(
        ge=0.0, le=1.0, description="Annual discount rate for NPV calculations"
    )
    time_horizon_years: int = Field(
        ge=1, description="Assessment time horizon in years"
    )
    baseline_scenario: bool = Field(
        default=False, description="Whether this is the baseline scenario"
    )


class PollinationServiceValue(BaseModel):
    """Calculated pollination service values"""

    model_config = {"validate_assignment": True}

    crop_type: CropType = Field(description="Type of crop")
    annual_value_gbp: float = Field(
        description="Annual pollination service value in GBP"
    )
    yield_increase_kg: float = Field(description="Yield increase due to pollination")
    value_per_hectare: float = Field(description="Value per hectare")
    primary_species: List[str] = Field(description="Primary pollinating species")
    service_reliability: float = Field(
        ge=0.0, le=1.0, description="Reliability of service delivery"
    )


class CostBenefitAnalysis(BaseModel):
    """Cost-benefit analysis results"""

    model_config = {"validate_assignment": True}

    scenario_name: str = Field(description="Scenario name")
    total_costs_gbp: float = Field(description="Total intervention costs")
    total_benefits_gbp: float = Field(description="Total economic benefits")
    net_present_value: float = Field(description="Net present value")
    benefit_cost_ratio: float = Field(description="Benefit-to-cost ratio")
    payback_period_years: float = Field(description="Payback period in years")
    annual_roi_percent: float = Field(
        description="Annual return on investment percentage"
    )


class EconomicAssessment:
    """
    Main economic assessment system for BSTEW.

    Integrates biological simulation outputs with economic models to provide
    quantitative assessment of pollinator services and conservation investments.
    """

    def __init__(self, species_system: SpeciesSystem) -> None:
        self.species_system = species_system
        self.logger = logging.getLogger(__name__)

        # Initialize crop economic data with UK agricultural values
        self.crop_economics = self._initialize_crop_economics()

        # Pollination efficiency coefficients by species
        self.species_efficiency = self._initialize_species_efficiency()

    def _initialize_crop_economics(self) -> Dict[CropType, CropEconomics]:
        """Initialize crop economic parameters with UK agricultural data"""

        return {
            CropType.APPLES: CropEconomics(
                crop_type=CropType.APPLES,
                price_per_kg=1.20,
                yield_baseline_kg_ha=8000,
                yield_with_pollinators_kg_ha=25000,
                pollinator_dependency=0.65,
                hectares_planted=1000,
                production_cost_per_ha=8500,
                harvest_cost_per_kg=0.15,
            ),
            CropType.BLUEBERRIES: CropEconomics(
                crop_type=CropType.BLUEBERRIES,
                price_per_kg=8.50,
                yield_baseline_kg_ha=1200,
                yield_with_pollinators_kg_ha=4500,
                pollinator_dependency=0.90,
                hectares_planted=200,
                production_cost_per_ha=12000,
                harvest_cost_per_kg=1.20,
            ),
            CropType.OILSEED_RAPE: CropEconomics(
                crop_type=CropType.OILSEED_RAPE,
                price_per_kg=0.45,
                yield_baseline_kg_ha=2500,
                yield_with_pollinators_kg_ha=3800,
                pollinator_dependency=0.35,
                hectares_planted=5000,
                production_cost_per_ha=850,
                harvest_cost_per_kg=0.08,
            ),
            CropType.STRAWBERRIES: CropEconomics(
                crop_type=CropType.STRAWBERRIES,
                price_per_kg=4.20,
                yield_baseline_kg_ha=8000,
                yield_with_pollinators_kg_ha=18000,
                pollinator_dependency=0.40,
                hectares_planted=300,
                production_cost_per_ha=15000,
                harvest_cost_per_kg=0.80,
            ),
        }

    def _initialize_species_efficiency(self) -> Dict[str, Dict[CropType, float]]:
        """Initialize pollination efficiency by species and crop"""

        return {
            "Bombus_terrestris": {
                CropType.APPLES: 0.85,
                CropType.BLUEBERRIES: 0.70,
                CropType.OILSEED_RAPE: 0.90,
                CropType.STRAWBERRIES: 0.75,
            },
            "Bombus_lucorum": {
                CropType.APPLES: 0.75,
                CropType.BLUEBERRIES: 0.80,
                CropType.OILSEED_RAPE: 0.85,
                CropType.STRAWBERRIES: 0.70,
            },
            "Bombus_hortorum": {
                CropType.APPLES: 0.95,
                CropType.BLUEBERRIES: 0.85,
                CropType.OILSEED_RAPE: 0.60,
                CropType.STRAWBERRIES: 0.90,
            },
            "Bombus_ruderatus": {
                CropType.APPLES: 0.90,
                CropType.BLUEBERRIES: 0.75,
                CropType.OILSEED_RAPE: 0.50,
                CropType.STRAWBERRIES: 0.85,
            },
        }

    def calculate_pollination_service_value(
        self,
        crop_type: CropType,
        species_abundance: Dict[str, int],
        foraging_efficiency: Dict[str, float],
        area_hectares: float = 1.0,
    ) -> PollinationServiceValue:
        """Calculate economic value of pollination services for a crop"""

        crop_econ = self.crop_economics.get(crop_type)
        if not crop_econ:
            raise ValueError(f"Crop economics not defined for {crop_type}")

        # Calculate effective pollination capacity
        total_pollination_capacity = 0.0
        primary_species = []

        for species_name, abundance in species_abundance.items():
            if species_name in self.species_efficiency:
                efficiency = self.species_efficiency[species_name].get(crop_type, 0.0)
                foraging_eff = foraging_efficiency.get(species_name, 0.5)

                # Species contribution = abundance × efficiency × foraging effectiveness
                species_contribution = abundance * efficiency * foraging_eff
                total_pollination_capacity += species_contribution

                if species_contribution > 10:  # Significant contributor
                    primary_species.append(species_name)

        # Convert pollination capacity to service reliability (0-1 scale)
        # Assumes saturation curve - diminishing returns with high abundance
        service_reliability = min(1.0, max(0.0, total_pollination_capacity / 100.0))

        # Calculate yield increase due to pollination
        max_yield_increase = (
            crop_econ.yield_with_pollinators_kg_ha - crop_econ.yield_baseline_kg_ha
        ) * crop_econ.pollinator_dependency

        actual_yield_increase = max_yield_increase * service_reliability * area_hectares

        # Calculate economic value
        # Value = yield increase × price - additional harvest costs
        gross_value = actual_yield_increase * crop_econ.price_per_kg
        harvest_costs = actual_yield_increase * crop_econ.harvest_cost_per_kg
        net_value = gross_value - harvest_costs

        return PollinationServiceValue(
            crop_type=crop_type,
            annual_value_gbp=net_value,
            yield_increase_kg=actual_yield_increase,
            value_per_hectare=net_value / area_hectares if area_hectares > 0 else 0,
            primary_species=primary_species,
            service_reliability=service_reliability,
        )

    def assess_stewardship_scenario(
        self,
        scenario: EconomicScenario,
        baseline_species_abundance: Dict[str, int],
        stewardship_species_abundance: Dict[str, int],
        baseline_foraging: Dict[str, float],
        stewardship_foraging: Dict[str, float],
    ) -> CostBenefitAnalysis:
        """Assess cost-benefit of stewardship scenario vs baseline"""

        total_costs = 0.0
        total_benefits = 0.0

        # Calculate stewardship costs
        for strategy, cost_per_ha in scenario.stewardship_costs.items():
            # Assume strategy applied to 10% of agricultural area
            area_treated = (
                sum(crop.hectares_planted for crop in scenario.crop_data.values()) * 0.1
            )
            annual_cost = area_treated * cost_per_ha

            # Present value of costs over time horizon
            pv_costs = self._present_value_series(
                annual_cost, scenario.time_horizon_years, scenario.discount_rate
            )
            total_costs += pv_costs

        # Calculate benefits for each crop
        for crop_name, crop_econ in scenario.crop_data.items():
            crop_type = crop_econ.crop_type

            # Baseline pollination value
            baseline_value = self.calculate_pollination_service_value(
                crop_type,
                baseline_species_abundance,
                baseline_foraging,
                crop_econ.hectares_planted,
            )

            # Stewardship scenario pollination value
            stewardship_value = self.calculate_pollination_service_value(
                crop_type,
                stewardship_species_abundance,
                stewardship_foraging,
                crop_econ.hectares_planted,
            )

            # Annual benefit = difference in pollination values
            annual_benefit = (
                stewardship_value.annual_value_gbp - baseline_value.annual_value_gbp
            )

            # Present value of benefits
            pv_benefits = self._present_value_series(
                annual_benefit, scenario.time_horizon_years, scenario.discount_rate
            )
            total_benefits += pv_benefits

        # Calculate economic indicators
        net_present_value = total_benefits - total_costs
        benefit_cost_ratio = total_benefits / total_costs if total_costs > 0 else 0

        # Calculate payback period (simplified)
        annual_net_benefit = (
            total_benefits / scenario.time_horizon_years
            - total_costs / scenario.time_horizon_years
        )
        payback_period = (
            total_costs / annual_net_benefit if annual_net_benefit > 0 else float("inf")
        )

        # Calculate annual ROI
        annual_roi_percent = (
            (annual_net_benefit / total_costs * 100) if total_costs > 0 else 0
        )

        return CostBenefitAnalysis(
            scenario_name=scenario.scenario_name,
            total_costs_gbp=total_costs,
            total_benefits_gbp=total_benefits,
            net_present_value=net_present_value,
            benefit_cost_ratio=benefit_cost_ratio,
            payback_period_years=min(payback_period, scenario.time_horizon_years),
            annual_roi_percent=annual_roi_percent,
        )

    def _present_value_series(
        self, annual_amount: float, years: int, discount_rate: float
    ) -> float:
        """Calculate present value of annual amount over multiple years"""

        pv = 0.0
        for year in range(years):
            pv += annual_amount / ((1 + discount_rate) ** year)
        return pv

    def generate_landscape_assessment(
        self,
        landscape_data: Dict[str, Any],
        species_abundance_data: Dict[str, Dict[str, int]],
        foraging_data: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Generate comprehensive economic assessment for a landscape"""

        assessment_results: Dict[str, Any] = {
            "total_pollination_value": 0.0,
            "crop_breakdown": {},
            "species_contributions": {},
            "risk_assessment": {},
            "recommendations": [],
        }

        # Calculate total landscape pollination value
        for crop_type_str, crop_data in landscape_data.get("crops", {}).items():
            try:
                crop_type = CropType(crop_type_str)

                # Use average species abundance across landscape
                avg_abundance = {}
                avg_foraging = {}

                for species in species_abundance_data:
                    species_abundances = list(species_abundance_data[species].values())
                    species_foraging_values = list(foraging_data[species].values())
                    avg_abundance[species] = int(np.mean(species_abundances))
                    avg_foraging[species] = float(np.mean(species_foraging_values))

                # Calculate pollination value
                pollination_value = self.calculate_pollination_service_value(
                    crop_type,
                    avg_abundance,
                    avg_foraging,
                    crop_data.get("area_hectares", 100),
                )

                assessment_results["total_pollination_value"] += float(
                    pollination_value.annual_value_gbp
                )
                crop_breakdown = assessment_results["crop_breakdown"]
                crop_breakdown[crop_type_str] = {
                    "annual_value": pollination_value.annual_value_gbp,
                    "value_per_hectare": pollination_value.value_per_hectare,
                    "primary_species": pollination_value.primary_species,
                    "service_reliability": pollination_value.service_reliability,
                }

            except ValueError:
                self.logger.warning(f"Unknown crop type: {crop_type_str}")
                continue

        # Analyze species contributions
        for species_name in species_abundance_data:
            species_value = 0.0

            for crop_type in self.crop_economics:
                if species_name in self.species_efficiency:
                    efficiency = self.species_efficiency[species_name].get(
                        crop_type, 0.0
                    )
                    # Rough estimate of species contribution to total value
                    total_value = float(assessment_results["total_pollination_value"])
                    species_value += total_value * efficiency * 0.1

            species_contributions = assessment_results["species_contributions"]
            species_contributions[species_name] = species_value

        # Risk assessment
        primary_species_count = len(
            [species for species, abundance in avg_abundance.items() if abundance > 50]
        )

        species_contributions = assessment_results["species_contributions"]
        total_value = float(assessment_results["total_pollination_value"])
        crop_breakdown = assessment_results["crop_breakdown"]

        if species_contributions and total_value > 0:
            max_contribution = max(species_contributions.values())
            single_species_dep = max_contribution / total_value
        else:
            single_species_dep = 0.0

        reliability_values = [
            crop["service_reliability"] for crop in crop_breakdown.values()
        ]
        service_stability = (
            float(np.mean(reliability_values)) if reliability_values else 0.0
        )

        risk_assessment = assessment_results["risk_assessment"]
        risk_assessment["diversification_score"] = min(1.0, primary_species_count / 4.0)
        risk_assessment["single_species_dependency"] = single_species_dep
        risk_assessment["service_stability"] = service_stability

        # Generate recommendations
        recommendations = assessment_results["recommendations"]
        if risk_assessment["diversification_score"] < 0.5:
            recommendations.append(
                "Increase pollinator species diversity to reduce service risk"
            )

        if risk_assessment["single_species_dependency"] > 0.6:
            recommendations.append(
                "Reduce dependency on single species through habitat diversification"
            )

        return assessment_results

    def compare_policy_scenarios(
        self,
        scenarios: List[Tuple[str, Dict[str, int], Dict[str, float]]],
        base_crop_areas: Dict[CropType, float],
    ) -> Dict[str, Any]:
        """Compare multiple policy scenarios for economic impact"""

        scenario_results = {}

        for scenario_name, species_abundance, foraging_efficiency in scenarios:
            total_value = 0.0
            crop_values = {}

            for crop_type, area_ha in base_crop_areas.items():
                pollination_value = self.calculate_pollination_service_value(
                    crop_type, species_abundance, foraging_efficiency, area_ha
                )

                crop_values[crop_type.value] = pollination_value.annual_value_gbp
                total_value += pollination_value.annual_value_gbp

            scenario_results[scenario_name] = {
                "total_annual_value": total_value,
                "crop_breakdown": crop_values,
                "value_per_hectare": total_value / sum(base_crop_areas.values()),
            }

        # Calculate relative performance
        all_values: List[float] = []
        for result in scenario_results.values():
            value = result["total_annual_value"]
            if isinstance(value, (int, float)):
                all_values.append(float(value))

        baseline_value = max(all_values) if all_values else 1.0

        for scenario_name, results in scenario_results.items():
            value = results["total_annual_value"]
            total_value = float(value) if isinstance(value, (int, float)) else 0.0
            results["relative_performance"] = (
                total_value / baseline_value if baseline_value > 0 else 0.0
            )
            results["additional_value"] = total_value - baseline_value

        return scenario_results
