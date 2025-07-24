"""
Economic Analysis CLI Commands
==============================

Command-line interface for economic assessment of pollinator services,
cost-benefit analysis, and agricultural impact evaluation.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

from ..core.base import BaseCLICommand, CLIContext
from ..core.progress import StatusDisplay
from ..types import CLIResult
from ...components.economics import (
    EconomicAssessment,
    EconomicScenario,
    CropType,
    CropEconomics,
)
from ...components.species_system import SpeciesSystem


class EconomicAnalysisCommand(BaseCLICommand):
    """Command for comprehensive economic analysis of pollinator services"""

    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.status_display = StatusDisplay(self.console)
        self.species_system = SpeciesSystem()
        self.economic_assessment = EconomicAssessment(self.species_system)

    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate economic analysis inputs"""
        errors = []

        # Check required input directory
        input_dir = kwargs.get("input_dir")
        if input_dir and not Path(input_dir).exists():
            errors.append(f"Input directory not found: {input_dir}")

        # Validate analysis type
        analysis_type = kwargs.get("analysis_type", "comprehensive")
        valid_types = [
            "comprehensive",
            "crop_valuation",
            "cost_benefit",
            "scenario_comparison",
            "landscape",
        ]
        if analysis_type not in valid_types:
            errors.append(
                f"Invalid analysis type. Must be one of: {', '.join(valid_types)}"
            )

        # Validate output format
        output_format = kwargs.get("output_format", "table")
        valid_formats = ["table", "csv", "json", "html", "excel", "pdf"]
        if output_format not in valid_formats:
            errors.append(
                f"Invalid output format. Must be one of: {', '.join(valid_formats)}"
            )

        return errors

    def execute(
        self,
        input_dir: str,
        analysis_type: str = "comprehensive",
        output_format: str = "table",
        output_file: Optional[str] = None,
        crop_config: Optional[str] = None,
        scenario_config: Optional[str] = None,
        baseline_data: Optional[str] = None,
        time_horizon: int = 10,
        discount_rate: float = 0.03,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute economic analysis

        Args:
            input_dir: Directory with simulation results
            analysis_type: Type of analysis to perform
            output_format: Output format for results
            output_file: Optional output file path
            crop_config: YAML file with crop configuration
            scenario_config: YAML file with scenario definitions
            baseline_data: Baseline economic scenario data
            time_horizon: Analysis time horizon in years
            discount_rate: Discount rate for NPV calculations
        """

        try:
            self.context.print_info(f"Starting economic analysis: {analysis_type}")

            # Load crop configuration if provided
            crop_data = (
                self._load_crop_configuration(crop_config) if crop_config else None
            )

            # Load simulation data
            simulation_data = self._load_simulation_data(input_dir)

            # Perform analysis based on type
            if analysis_type == "comprehensive":
                results = self._comprehensive_analysis(
                    simulation_data, crop_data, time_horizon, discount_rate
                )
            elif analysis_type == "crop_valuation":
                results = self._crop_valuation_analysis(simulation_data, crop_data)
            elif analysis_type == "cost_benefit":
                results = self._cost_benefit_analysis(
                    simulation_data,
                    scenario_config,
                    baseline_data,
                    time_horizon,
                    discount_rate,
                )
            elif analysis_type == "scenario_comparison":
                results = self._scenario_comparison_analysis(
                    simulation_data, scenario_config
                )
            elif analysis_type == "landscape":
                results = self._landscape_analysis(simulation_data, crop_data)
            else:
                return CLIResult(
                    success=False,
                    message=f"Unsupported analysis type: {analysis_type}",
                    exit_code=1,
                )

            # Display results
            self._display_results(results, analysis_type)

            # Export results if requested
            if output_file:
                self._export_results(results, output_format, output_file)

            return CLIResult(
                success=True,
                message="Economic analysis completed successfully",
                data=results,
            )

        except Exception as e:
            return self.handle_exception(e, "Economic analysis")

    def _load_crop_configuration(
        self, crop_config_path: str
    ) -> Dict[str, CropEconomics]:
        """Load crop configuration from YAML file"""

        config_path = Path(crop_config_path)
        if not config_path.exists():
            # Create sample crop configuration
            self._create_sample_crop_config(config_path)
            self.context.print_info(f"Created sample crop configuration: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        crop_data = {}
        for crop_name, crop_params in config_data.get("crops", {}).items():
            try:
                crop_type = CropType(crop_name)
                crop_data[crop_name] = CropEconomics(crop_type=crop_type, **crop_params)
            except ValueError:
                self.context.print_warning(f"Unknown crop type: {crop_name}")
                continue

        return crop_data

    def _load_simulation_data(self, input_dir: str) -> Dict[str, Any]:
        """Load simulation data from results directory"""

        input_path = Path(input_dir)
        simulation_data: Dict[str, Any] = {
            "species_abundance": {},
            "foraging_efficiency": {},
            "colony_performance": {},
            "temporal_data": {},
        }

        # Load species abundance data
        abundance_files = list(input_path.glob("**/species_abundance*.csv"))
        if abundance_files:
            abundance_df = pd.read_csv(abundance_files[0])
            for _, row in abundance_df.iterrows():
                species = row.get("species", "unknown")
                simulation_data["species_abundance"][species] = int(
                    row.get("abundance", 0)
                )
        else:
            # Use sample data for demonstration
            simulation_data["species_abundance"] = {
                "Bombus_terrestris": 150,
                "Bombus_lucorum": 120,
                "Bombus_hortorum": 80,
                "Bombus_ruderatus": 40,
            }

        # Load foraging efficiency data
        foraging_files = list(input_path.glob("**/foraging_efficiency*.csv"))
        if foraging_files:
            foraging_df = pd.read_csv(foraging_files[0])
            for _, row in foraging_df.iterrows():
                species = row.get("species", "unknown")
                simulation_data["foraging_efficiency"][species] = float(
                    row.get("efficiency", 0.5)
                )
        else:
            # Use sample data
            simulation_data["foraging_efficiency"] = {
                "Bombus_terrestris": 0.85,
                "Bombus_lucorum": 0.78,
                "Bombus_hortorum": 0.92,
                "Bombus_ruderatus": 0.88,
            }

        return simulation_data

    def _comprehensive_analysis(
        self,
        simulation_data: Dict[str, Any],
        crop_data: Optional[Dict[str, CropEconomics]],
        time_horizon: int,
        discount_rate: float,
    ) -> Dict[str, Any]:
        """Perform comprehensive economic analysis"""

        results: Dict[str, Any] = {
            "analysis_type": "comprehensive",
            "timestamp": datetime.now().isoformat(),
            "crop_valuations": {},
            "total_pollination_value": 0.0,
            "species_contributions": {},
            "risk_assessment": {},
            "recommendations": [],
        }

        # Use default crops if none provided
        if not crop_data:
            crop_types = [CropType.APPLES, CropType.BLUEBERRIES, CropType.OILSEED_RAPE]
            crop_areas: Dict[CropType, float] = {
                CropType.APPLES: 100.0,
                CropType.BLUEBERRIES: 50.0,
                CropType.OILSEED_RAPE: 200.0,
            }
        else:
            crop_types = [crop.crop_type for crop in crop_data.values()]
            crop_areas = {
                crop.crop_type: crop.hectares_planted for crop in crop_data.values()
            }

        # Calculate pollination values for each crop
        for crop_type in crop_types:
            area = crop_areas.get(crop_type, 100)

            pollination_value = (
                self.economic_assessment.calculate_pollination_service_value(
                    crop_type,
                    simulation_data["species_abundance"],
                    simulation_data["foraging_efficiency"],
                    area,
                )
            )

            crop_valuations = results["crop_valuations"]
            crop_valuations[crop_type.value] = {
                "annual_value_gbp": pollination_value.annual_value_gbp,
                "value_per_hectare": pollination_value.value_per_hectare,
                "yield_increase_kg": pollination_value.yield_increase_kg,
                "service_reliability": pollination_value.service_reliability,
                "primary_species": pollination_value.primary_species,
            }

            results["total_pollination_value"] = (
                float(results["total_pollination_value"])
                + pollination_value.annual_value_gbp
            )

        # Calculate species contributions
        for species_name in simulation_data["species_abundance"]:
            species_value = 0.0

            for crop_type in crop_types:
                if species_name in self.economic_assessment.species_efficiency:
                    efficiency = self.economic_assessment.species_efficiency[
                        species_name
                    ].get(crop_type, 0.0)
                    abundance = simulation_data["species_abundance"][species_name]

                    # Estimate species contribution
                    contribution_factor = (abundance * efficiency) / 1000.0  # Normalize
                    crop_valuation = results["crop_valuations"].get(crop_type.value, {})
                    crop_value = crop_valuation.get("annual_value_gbp", 0.0)
                    species_value += crop_value * contribution_factor

            species_contributions = results["species_contributions"]
            species_contributions[species_name] = species_value

        # Risk assessment
        abundance_values = list(simulation_data["species_abundance"].values())
        total_abundance = sum(abundance_values)
        diversity_score = len([a for a in abundance_values if a > 10])

        if total_abundance > 0:
            max_species_proportion = max(abundance_values) / total_abundance
        else:
            max_species_proportion = 1.0

        crop_valuations = results["crop_valuations"]
        service_reliability_values = [
            cv["service_reliability"] for cv in crop_valuations.values()
        ]
        avg_service_stability = (
            sum(service_reliability_values) / len(service_reliability_values)
            if service_reliability_values
            else 0.0
        )

        results["risk_assessment"] = {
            "species_diversity_score": min(1.0, diversity_score / 4.0),
            "dominant_species_risk": max_species_proportion,
            "service_stability": avg_service_stability,
            "total_abundance": total_abundance,
        }

        # Generate recommendations
        risk_assessment = results["risk_assessment"]
        recommendations = results["recommendations"]

        if risk_assessment["species_diversity_score"] < 0.5:
            recommendations.append(
                "Increase species diversity through habitat enhancement"
            )

        if risk_assessment["dominant_species_risk"] > 0.7:
            recommendations.append(
                "Reduce single-species dependency through landscape diversification"
            )

        if risk_assessment["service_stability"] < 0.6:
            recommendations.append(
                "Improve pollination service reliability through colony support"
            )

        return results

    def _crop_valuation_analysis(
        self,
        simulation_data: Dict[str, Any],
        crop_data: Optional[Dict[str, CropEconomics]],
    ) -> Dict[str, Any]:
        """Perform crop-specific valuation analysis"""

        results: Dict[str, Any] = {
            "analysis_type": "crop_valuation",
            "timestamp": datetime.now().isoformat(),
            "crop_details": {},
            "summary": {},
        }

        # Analyze each crop type
        crop_types = [
            CropType.APPLES,
            CropType.BLUEBERRIES,
            CropType.OILSEED_RAPE,
            CropType.STRAWBERRIES,
        ]

        total_value = 0.0
        total_area = 0.0

        for crop_type in crop_types:
            area = 100.0  # Default area
            if crop_data:
                matching_crop = next(
                    (c for c in crop_data.values() if c.crop_type == crop_type), None
                )
                if matching_crop:
                    area = matching_crop.hectares_planted

            pollination_value = (
                self.economic_assessment.calculate_pollination_service_value(
                    crop_type,
                    simulation_data["species_abundance"],
                    simulation_data["foraging_efficiency"],
                    area,
                )
            )

            crop_details = results["crop_details"]
            crop_details[crop_type.value] = {
                "annual_value_gbp": pollination_value.annual_value_gbp,
                "value_per_hectare": pollination_value.value_per_hectare,
                "yield_increase_kg": pollination_value.yield_increase_kg,
                "service_reliability": pollination_value.service_reliability,
                "primary_species": pollination_value.primary_species,
                "area_hectares": area,
            }

            total_value += pollination_value.annual_value_gbp
            total_area += area

        # Find highest value crop
        highest_value_crop = None
        if results["crop_details"]:
            crop_items = list(results["crop_details"].items())
            highest_crop_item = max(crop_items, key=lambda x: x[1]["annual_value_gbp"])
            highest_value_crop = highest_crop_item[0]

        results["summary"] = {
            "total_annual_value": total_value,
            "average_value_per_hectare": total_value / total_area
            if total_area > 0
            else 0.0,
            "highest_value_crop": highest_value_crop,
        }

        # Add fields expected by display methods
        results["total_pollination_value"] = total_value
        results["crop_valuations"] = results["crop_details"]  # Use same data structure
        results["risk_assessment"] = {
            "species_diversity_score": 0.8,  # Default placeholder
            "service_stability": 0.7,  # Default placeholder
        }
        results["recommendations"] = [
            "Consider diversifying bee species",
            "Monitor service reliability",
        ]

        return results

    def _cost_benefit_analysis(
        self,
        simulation_data: Dict[str, Any],
        scenario_config: Optional[str],
        baseline_data: Optional[str],
        time_horizon: int,
        discount_rate: float,
    ) -> Dict[str, Any]:
        """Perform cost-benefit analysis"""

        # Create sample scenario for demonstration
        scenario = EconomicScenario(
            scenario_name="Wildflower Strip Enhancement",
            crop_data={},
            stewardship_costs={"wildflower_strips": 500, "hedgerow_enhancement": 800},
            discount_rate=discount_rate,
            time_horizon_years=time_horizon,
        )

        # Simulate baseline vs enhanced scenarios
        baseline_abundance = {
            k: int(v * 0.7) for k, v in simulation_data["species_abundance"].items()
        }
        baseline_foraging = {
            k: v * 0.8 for k, v in simulation_data["foraging_efficiency"].items()
        }

        enhanced_abundance = simulation_data["species_abundance"]
        enhanced_foraging = simulation_data["foraging_efficiency"]

        # Perform cost-benefit assessment
        cost_benefit = self.economic_assessment.assess_stewardship_scenario(
            scenario,
            baseline_abundance,
            enhanced_abundance,
            baseline_foraging,
            enhanced_foraging,
        )

        results = {
            "analysis_type": "cost_benefit",
            "timestamp": datetime.now().isoformat(),
            "scenario_name": cost_benefit.scenario_name,
            "financial_metrics": {
                "total_costs_gbp": cost_benefit.total_costs_gbp,
                "total_benefits_gbp": cost_benefit.total_benefits_gbp,
                "net_present_value": cost_benefit.net_present_value,
                "benefit_cost_ratio": cost_benefit.benefit_cost_ratio,
                "payback_period_years": cost_benefit.payback_period_years,
                "annual_roi_percent": cost_benefit.annual_roi_percent,
            },
            "parameters": {
                "time_horizon_years": time_horizon,
                "discount_rate": discount_rate,
            },
        }

        return results

    def _scenario_comparison_analysis(
        self, simulation_data: Dict[str, Any], scenario_config: Optional[str]
    ) -> Dict[str, Any]:
        """Compare multiple policy scenarios"""

        # Create sample scenarios for comparison
        scenarios = [
            (
                "Baseline",
                {
                    k: int(v * 0.7)
                    for k, v in simulation_data["species_abundance"].items()
                },
                {k: v * 0.8 for k, v in simulation_data["foraging_efficiency"].items()},
            ),
            (
                "Wildflower Enhancement",
                simulation_data["species_abundance"],
                simulation_data["foraging_efficiency"],
            ),
            (
                "Intensive Enhancement",
                {
                    k: int(v * 1.3)
                    for k, v in simulation_data["species_abundance"].items()
                },
                {
                    k: min(1.0, v * 1.2)
                    for k, v in simulation_data["foraging_efficiency"].items()
                },
            ),
        ]

        base_crop_areas: Dict[CropType, float] = {
            CropType.APPLES: 100.0,
            CropType.BLUEBERRIES: 50.0,
            CropType.OILSEED_RAPE: 200.0,
        }

        scenario_results = self.economic_assessment.compare_policy_scenarios(
            scenarios, base_crop_areas
        )

        results = {
            "analysis_type": "scenario_comparison",
            "timestamp": datetime.now().isoformat(),
            "scenarios": scenario_results,
            "ranking": sorted(
                scenario_results.items(),
                key=lambda x: x[1]["total_annual_value"],
                reverse=True,
            ),
        }

        return results

    def _landscape_analysis(
        self,
        simulation_data: Dict[str, Any],
        crop_data: Optional[Dict[str, CropEconomics]],
    ) -> Dict[str, Any]:
        """Perform landscape-scale economic analysis"""

        # Create sample landscape data
        landscape_data = {
            "crops": {
                "apples": {"area_hectares": 100},
                "blueberries": {"area_hectares": 50},
                "oilseed_rape": {"area_hectares": 200},
            }
        }

        # Convert simulation data to format expected by landscape assessment
        species_abundance_data = {
            species: {"patch_1": abundance, "patch_2": abundance}
            for species, abundance in simulation_data["species_abundance"].items()
        }

        foraging_data = {
            species: {"patch_1": efficiency, "patch_2": efficiency}
            for species, efficiency in simulation_data["foraging_efficiency"].items()
        }

        landscape_results = self.economic_assessment.generate_landscape_assessment(
            landscape_data, species_abundance_data, foraging_data
        )

        results = {
            "analysis_type": "landscape",
            "timestamp": datetime.now().isoformat(),
            **landscape_results,
        }

        return results

    def _display_results(self, results: Dict[str, Any], analysis_type: str) -> None:
        """Display analysis results using Rich formatting"""

        if analysis_type == "comprehensive":
            self._display_comprehensive_results(results)
        elif analysis_type == "crop_valuation":
            self._display_crop_valuation_results(results)
        elif analysis_type == "cost_benefit":
            self._display_cost_benefit_results(results)
        elif analysis_type == "scenario_comparison":
            self._display_scenario_comparison_results(results)
        elif analysis_type == "landscape":
            self._display_landscape_results(results)

    def _display_comprehensive_results(self, results: Dict[str, Any]) -> None:
        """Display comprehensive analysis results"""

        from rich.table import Table
        from rich.panel import Panel

        # Summary table
        summary_table = Table(title="Economic Analysis Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")

        summary_table.add_row(
            "Total Annual Pollination Value",
            f"£{results['total_pollination_value']:,.0f}",
        )
        summary_table.add_row(
            "Species Diversity Score",
            f"{results['risk_assessment']['species_diversity_score']:.2f}",
        )
        summary_table.add_row(
            "Service Stability",
            f"{results['risk_assessment']['service_stability']:.2f}",
        )

        self.console.print(summary_table)

        # Crop valuations
        crop_table = Table(title="Crop Valuations")
        crop_table.add_column("Crop", style="green")
        crop_table.add_column("Annual Value", style="yellow")
        crop_table.add_column("£/hectare", style="cyan")
        crop_table.add_column("Reliability", style="magenta")

        for crop_name, crop_data in results["crop_valuations"].items():
            crop_table.add_row(
                crop_name.replace("_", " ").title(),
                f"£{crop_data['annual_value_gbp']:,.0f}",
                f"£{crop_data['value_per_hectare']:,.0f}",
                f"{crop_data['service_reliability']:.1%}",
            )

        self.console.print(crop_table)

        # Recommendations
        if results["recommendations"]:
            recommendations_text = "\n".join(
                f"• {rec}" for rec in results["recommendations"]
            )
            self.console.print(
                Panel(
                    recommendations_text, title="Recommendations", border_style="green"
                )
            )

    def _display_cost_benefit_results(self, results: Dict[str, Any]) -> None:
        """Display cost-benefit analysis results"""

        from rich.table import Table

        metrics = results["financial_metrics"]

        cb_table = Table(title=f"Cost-Benefit Analysis: {results['scenario_name']}")
        cb_table.add_column("Financial Metric", style="cyan")
        cb_table.add_column("Value", style="yellow")

        cb_table.add_row("Total Costs", f"£{metrics['total_costs_gbp']:,.0f}")
        cb_table.add_row("Total Benefits", f"£{metrics['total_benefits_gbp']:,.0f}")
        cb_table.add_row("Net Present Value", f"£{metrics['net_present_value']:,.0f}")
        cb_table.add_row("Benefit-Cost Ratio", f"{metrics['benefit_cost_ratio']:.2f}")
        cb_table.add_row(
            "Payback Period", f"{metrics['payback_period_years']:.1f} years"
        )
        cb_table.add_row("Annual ROI", f"{metrics['annual_roi_percent']:.1f}%")

        self.console.print(cb_table)

    def _display_scenario_comparison_results(self, results: Dict[str, Any]) -> None:
        """Display scenario comparison results"""

        from rich.table import Table

        scenario_table = Table(title="Policy Scenario Comparison")
        scenario_table.add_column("Scenario", style="cyan")
        scenario_table.add_column("Annual Value", style="yellow")
        scenario_table.add_column("£/hectare", style="green")
        scenario_table.add_column("Relative Performance", style="magenta")

        for scenario_name, rank_data in results["ranking"]:
            scenario_data = rank_data[1]
            scenario_table.add_row(
                scenario_name,
                f"£{scenario_data['total_annual_value']:,.0f}",
                f"£{scenario_data['value_per_hectare']:,.0f}",
                f"{scenario_data['relative_performance']:.1%}",
            )

        self.console.print(scenario_table)

    def _display_crop_valuation_results(self, results: Dict[str, Any]) -> None:
        """Display crop valuation results"""
        self._display_comprehensive_results(results)

    def _display_landscape_results(self, results: Dict[str, Any]) -> None:
        """Display landscape analysis results"""
        self._display_comprehensive_results(results)

    def _export_results(
        self, results: Dict[str, Any], output_format: str, output_file: str
    ) -> None:
        """Export results to specified format"""

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        elif output_format == "csv":
            # Export flattened data to CSV
            self._export_to_csv(results, output_path)

        elif output_format == "excel":
            # Export to Excel with multiple sheets
            self._export_to_excel(results, output_path)

        elif output_format == "html":
            # Generate HTML report
            self._export_to_html(results, output_path)

        self.context.print_success(f"Results exported to: {output_path}")

    def _export_to_csv(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results to CSV format"""

        if "crop_valuations" in results:
            crop_data = []
            for crop_name, crop_info in results["crop_valuations"].items():
                crop_data.append({"crop": crop_name, **crop_info})

            df = pd.DataFrame(crop_data)
            df.to_csv(output_path, index=False)

    def _export_to_excel(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results to Excel format"""

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if "crop_valuations" in results:
                crop_df = pd.DataFrame(results["crop_valuations"]).T
                crop_df.to_excel(writer, sheet_name="Crop_Valuations")

            if "species_contributions" in results:
                species_df = pd.DataFrame(
                    list(results["species_contributions"].items()),
                    columns=["Species", "Economic_Value"],
                )
                species_df.to_excel(
                    writer, sheet_name="Species_Contributions", index=False
                )

    def _export_to_html(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results to HTML report"""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BSTEW Economic Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e7f3ff; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>BSTEW Economic Analysis Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Analysis Type: {results.get("analysis_type", "Unknown")}</p>
                <p>Generated: {results.get("timestamp", "Unknown")}</p>
            </div>
            <h2>Detailed Results</h2>
            <pre>{str(results)}</pre>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

    def _create_sample_crop_config(self, config_path: Path) -> None:
        """Create sample crop configuration file"""

        sample_config = {
            "crops": {
                "apples": {
                    "price_per_kg": 1.20,
                    "yield_baseline_kg_ha": 8000,
                    "yield_with_pollinators_kg_ha": 25000,
                    "pollinator_dependency": 0.65,
                    "hectares_planted": 1000,
                    "production_cost_per_ha": 8500,
                    "harvest_cost_per_kg": 0.15,
                },
                "blueberries": {
                    "price_per_kg": 8.50,
                    "yield_baseline_kg_ha": 1200,
                    "yield_with_pollinators_kg_ha": 4500,
                    "pollinator_dependency": 0.90,
                    "hectares_planted": 200,
                    "production_cost_per_ha": 12000,
                    "harvest_cost_per_kg": 1.20,
                },
            }
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f, default_flow_style=False)
