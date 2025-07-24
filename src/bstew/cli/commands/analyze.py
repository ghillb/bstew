"""
Analysis and plotting command implementations
============================================

Handles analysis of simulation results and plot generation.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import json
import yaml
import typer

from ..core.base import BaseCLICommand, CLIContext
from ..core.progress import ProgressManager, StatusDisplay
from ..core.validation import InputValidator
from ..types import CLIResult
from ...analysis.population_plotter import PopulationPlotter, PlotConfig
from ...analysis.spatial_plotter import SpatialPlotter, SpatialConfig
from ...analysis.population_analyzer import PopulationAnalyzer
from ...analysis.foraging_analyzer import ForagingAnalyzer


class AnalyzeCommand(BaseCLICommand):
    """Command for analyzing simulation results"""

    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.status_display = StatusDisplay(self.console)
        self.progress_manager = ProgressManager(self.console)
        self.population_analyzer = PopulationAnalyzer()
        self.foraging_analyzer = ForagingAnalyzer()

    def execute(
        self,
        input_dir: str,
        format_type: str = "table",
        output_file: Optional[str] = None,
        analysis_type: str = "comprehensive",
        species: Optional[str] = None,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute results analysis"""

        try:
            input_path = Path(input_dir)

            if not input_path.exists():
                return CLIResult(
                    success=False,
                    message=f"Input directory not found: {input_dir}",
                    exit_code=1,
                )

            self.context.print_info(f"Analyzing results from: {input_path}")

            # Load and analyze results with progress tracking
            with self.progress_manager.progress_context() as progress:
                task = progress.start_task(
                    f"Performing {analysis_type} analysis...", total=None
                )

                results = self._analyze_results(input_path, analysis_type, species)

                progress.finish_task(task, "Analysis complete")

            # Display results
            if results.get("status") == "warning":
                # Display warning about empty results
                self._display_empty_results_warning(results)
            elif format_type == "table":
                self._display_results_table(results)
            elif format_type == "summary":
                self._display_summary(results)

            # Save to file if requested
            if output_file:
                # Don't try to save if we have warning status (empty results)
                if results.get("status") == "warning":
                    self.context.print_warning(
                        "Cannot save empty results. Please check the suggestions above."
                    )
                else:
                    try:
                        # Debug: log what we're trying to save
                        self.context.print_debug(
                            f"Saving results with keys: {list(results.keys())}"
                        )
                        files_created = self._save_results(
                            results, output_file, format_type
                        )

                        if len(files_created) == 1:
                            self.context.print_success(
                                f"Results exported to: {files_created[0]}"
                            )
                        elif len(files_created) > 1:
                            self.context.print_success(
                                f"Results exported to {len(files_created)} files:"
                            )
                            for file_path in files_created:
                                self.context.print_info(f"  - {file_path}")
                        # If no files created, the _save_results method will raise an error
                    except RuntimeError as e:
                        # Re-raise to be caught by outer handler with proper error message
                        raise RuntimeError(str(e)) from e

            return CLIResult(
                success=True,
                message="Analysis completed successfully",
                data={"results": results, "format": format_type},
            )

        except Exception as e:
            return self.handle_exception(e, "Analysis")

    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate command inputs"""
        errors = []

        # Validate input directory
        input_dir = kwargs.get("input_dir")
        if input_dir and not Path(input_dir).exists():
            errors.append(f"Input directory not found: {input_dir}")

        # Validate format
        format_type = kwargs.get("format_type", "table")
        valid_formats = ["table", "csv", "json", "yaml", "html", "summary"]
        errors.extend(
            InputValidator.validate_choice(format_type, "format_type", valid_formats)
        )

        # Validate analysis type
        analysis_type = kwargs.get("analysis_type", "comprehensive")
        valid_types = [
            "comprehensive",
            "population",
            "foraging",
            "survival",
            "efficiency",
            "summary",
        ]
        errors.extend(
            InputValidator.validate_choice(analysis_type, "analysis_type", valid_types)
        )

        return errors

    def _analyze_results(
        self, input_path: Path, analysis_type: str, species: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze simulation results using specialized analyzers"""

        results = {}

        # Load data files
        data = self._load_data_files(input_path, species)

        if not data:
            # Create sample data for demonstration
            self.context.print_warning(
                "No data files found, using sample data for demonstration..."
            )
            data = self._create_sample_analysis_data()

        # Perform analysis based on type
        if analysis_type in ["comprehensive", "population", "survival"]:
            if "population_data" in data:
                # Population analysis
                pop_trends = self.population_analyzer.calculate_trends(
                    data["population_data"], group_by="species" if not species else None
                )
                results["population_trends"] = self._convert_trends_to_dict(pop_trends)

                # Growth rate analysis
                growth_rates = self.population_analyzer.calculate_growth_rates(
                    data["population_data"]
                )
                results["growth_rates"] = self._convert_growth_rates_to_dict(
                    growth_rates
                )

            if "survival_data" in data and analysis_type in [
                "comprehensive",
                "survival",
            ]:
                # Survival analysis
                survival_results = self.population_analyzer.survival_analysis(
                    data["survival_data"], group_by="species" if not species else None
                )
                results["survival_analysis"] = self._convert_survival_to_dict(
                    survival_results
                )

        if analysis_type in ["comprehensive", "foraging", "efficiency"]:
            if "foraging_data" in data:
                # Foraging efficiency analysis
                efficiency_results = self.foraging_analyzer.analyze_foraging_efficiency(
                    data["foraging_data"]
                )
                results["foraging_efficiency"] = self._convert_efficiency_to_dict(
                    efficiency_results
                )

                # Resource optimization analysis
                if "resource_data" in data:
                    optimization = self.foraging_analyzer.optimize_resource_allocation(
                        data["foraging_data"], data["resource_data"]
                    )
                    results["resource_optimization"] = (
                        self._convert_optimization_to_dict(optimization)
                    )

                # Behavioral pattern analysis
                patterns = self.foraging_analyzer.analyze_behavioral_patterns(
                    data["foraging_data"]
                )
                results["behavioral_patterns"] = self._convert_patterns_to_dict(
                    patterns
                )

        # Add summary statistics
        results["summary_statistics"] = self._calculate_comprehensive_summary(
            data, results
        )

        # Handle empty results with helpful feedback
        results = self.handle_empty_analysis_results(results)

        return results

    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics from results"""

        summary = {}

        # Numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            summary[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
            }

        return summary

    def _analyze_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series data"""

        analysis = {}

        # Look for common time series columns
        if "Total_Bees" in df.columns:
            population_data = df["Total_Bees"]
            analysis["population"] = {
                "trend": "increasing"
                if population_data.iloc[-1] > population_data.iloc[0]
                else "decreasing",
                "peak_day": population_data.idxmax(),
                "peak_value": population_data.max(),
                "stability": population_data.std() / population_data.mean(),
            }

        if "Total_Honey" in df.columns:
            honey_data = df["Total_Honey"]
            analysis["honey_production"] = {
                "total_produced": honey_data.iloc[-1],
                "daily_average": honey_data.diff().mean(),
                "peak_production_day": honey_data.diff().idxmax(),
            }

        return analysis

    def _display_results_table(self, results: Dict[str, Any]) -> None:
        """Display results in table format"""

        # Use status display for results
        self.status_display.show_results_summary(results.get("summary_statistics", {}))

    def _save_results(
        self, results: Dict[str, Any], output_file: str, format_type: str
    ) -> List[Path]:
        """Save results to file with verification

        Returns list of files actually created
        """

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        files_created = []

        try:
            if format_type == "json":
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                files_created.append(output_path)

            elif format_type == "yaml":
                with open(output_path, "w") as f:
                    yaml.dump(results, f, default_flow_style=False)
                files_created.append(output_path)

            elif format_type == "csv":
                # Create multiple CSV files for different analysis types
                base_name = output_path.stem

                # Summary statistics
                if "summary_statistics" in results and results["summary_statistics"]:
                    summary_path = output_path.parent / f"{base_name}_summary.csv"
                    summary_df = pd.DataFrame([results["summary_statistics"]])
                    summary_df.to_csv(summary_path, index=False)
                    files_created.append(summary_path)

                # Population trends
                if "population_trends" in results and results["population_trends"]:
                    trends_data = []
                    for group, trend in results["population_trends"].items():
                        if isinstance(trend, dict):
                            trend_dict = {"group": group}
                            trend_dict.update(trend)
                            trends_data.append(trend_dict)
                    if trends_data:
                        trends_path = output_path.parent / f"{base_name}_trends.csv"
                        pd.DataFrame(trends_data).to_csv(trends_path, index=False)
                        files_created.append(trends_path)

                # Growth rates
                if "growth_rates" in results and results["growth_rates"]:
                    growth_data = []
                    for group, rates in results["growth_rates"].items():
                        if isinstance(rates, dict):
                            rate_dict = {"group": group}
                            rate_dict.update(rates)
                            growth_data.append(rate_dict)
                    if growth_data:
                        growth_path = output_path.parent / f"{base_name}_growth.csv"
                        pd.DataFrame(growth_data).to_csv(growth_path, index=False)
                        files_created.append(growth_path)

                # Foraging efficiency
                if "foraging_efficiency" in results and results["foraging_efficiency"]:
                    foraging_path = output_path.parent / f"{base_name}_foraging.csv"
                    efficiency_df = pd.DataFrame([results["foraging_efficiency"]])
                    efficiency_df.to_csv(foraging_path, index=False)
                    files_created.append(foraging_path)

                # If no specific data found, save flattened results as single CSV
                if not files_created:
                    flattened = self._flatten_results(results)
                    if flattened:
                        df = pd.DataFrame([flattened])
                        df.to_csv(output_path, index=False)
                        files_created.append(output_path)

            elif format_type == "html":
                # Generate HTML report
                html_content = self._generate_html_report(results)
                with open(output_path, "w") as f:
                    f.write(html_content)
                files_created.append(output_path)

            # CRITICAL: VERIFY FILES WERE CREATED
            if not files_created:
                raise RuntimeError(
                    f"No analysis files were created. "
                    f"Results may be empty or format '{format_type}' not properly handled."
                )

            # Verify each file exists and has content
            for file_path in files_created:
                if not file_path.exists():
                    raise RuntimeError(f"Analysis file was not created: {file_path}")

                if file_path.stat().st_size == 0:
                    raise RuntimeError(f"Analysis file is empty: {file_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to save analysis results: {e}")

        return files_created

    def _flatten_results(
        self, results: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""

        flattened = {}

        for key, value in results.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flattened.update(self._flatten_results(value, new_key))
            else:
                flattened[new_key] = value

        return flattened

    def _load_data_files(
        self, input_path: Path, species: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load simulation data files"""
        data = {}

        # Load population data
        pop_file = input_path / "population_data.csv"
        if pop_file.exists():
            df = pd.read_csv(pop_file)
            if species and "species" in df.columns:
                df = df[df["species"] == species]
            data["population_data"] = df

        # Load survival/mortality data
        survival_file = input_path / "survival_data.csv"
        if survival_file.exists():
            df = pd.read_csv(survival_file)
            if species and "species" in df.columns:
                df = df[df["species"] == species]
            data["survival_data"] = df

        # Load foraging data
        foraging_file = input_path / "foraging_data.csv"
        if foraging_file.exists():
            df = pd.read_csv(foraging_file)
            if species and "species" in df.columns:
                df = df[df["species"] == species]
            data["foraging_data"] = df

        # Load resource data
        resource_file = input_path / "resource_data.csv"
        if resource_file.exists():
            data["resource_data"] = pd.read_csv(resource_file)

        # Try to load from standard output files
        if not data:
            # Look for model_data.csv or results.csv
            model_file = input_path / "model_data.csv"
            results_file = input_path / "results.csv"

            if model_file.exists():
                data["model_data"] = pd.read_csv(model_file)
            if results_file.exists():
                data["results_data"] = pd.read_csv(results_file)

        return data

    def _create_sample_analysis_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample data for demonstration"""
        import numpy as np
        from datetime import datetime, timedelta

        # Create sample population data
        days = 365
        dates = [(datetime.now() - timedelta(days=365 - i)) for i in range(days)]
        species_list = ["bombus_terrestris", "bombus_lapidarius", "apis_mellifera"]

        pop_records = []
        for species in species_list:
            for i, date in enumerate(dates):
                base_pop = 10000 if species == "apis_mellifera" else 1000
                seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * i / 365)
                population = int(
                    base_pop * seasonal_factor * (1 + 0.1 * np.random.randn())
                )
                pop_records.append(
                    {
                        "time": date,
                        "species": species,
                        "population": max(0, population),
                        "queens": max(0, int(population * 0.01)),
                        "workers": max(0, int(population * 0.85)),
                        "drones": max(0, int(population * 0.14)),
                    }
                )

        # Create sample survival data
        survival_records = []
        for species in species_list:
            for i in range(100):  # 100 individuals per species
                birth_day = np.random.randint(0, 200)
                lifespan = np.random.normal(
                    60 if species == "apis_mellifera" else 45, 15
                )
                death_day = min(365, birth_day + max(1, int(lifespan)))
                survival_records.append(
                    {
                        "individual_id": f"{species}_{i}",
                        "species": species,
                        "birth_time": dates[birth_day],
                        "death_time": dates[death_day] if death_day < 365 else None,
                        "status": "dead" if death_day < 365 else "alive",
                        "lifespan_days": death_day - birth_day,
                        "cause_of_death": np.random.choice(
                            ["natural", "predation", "disease", "weather"]
                        )
                        if death_day < 365
                        else None,
                    }
                )

        # Create sample foraging data
        foraging_records = []
        for i in range(1000):  # 1000 foraging trips
            species = np.random.choice(species_list)
            success = np.random.random() > 0.2
            nectar = np.random.exponential(5) if np.random.random() > 0.3 else 0
            pollen = np.random.exponential(3) if np.random.random() > 0.4 else 0
            foraging_records.append(
                {
                    "trip_id": i,
                    "species": species,
                    "forager_id": f"{species}_f{np.random.randint(0, 100)}",
                    "success": success,
                    "nectar_collected": nectar,
                    "pollen_collected": pollen,
                    "energy_gained": (nectar * 2 + pollen * 1.5) if success else 0,
                    "distance_traveled": np.random.gamma(2, 200),
                    "time_spent": np.random.gamma(2, 30),
                    "energy_spent": np.random.gamma(2, 10),
                    "patch_id": np.random.randint(0, 50),
                    "patch_quality": np.random.uniform(0.1, 1.0),
                }
            )

        # Create sample resource data
        resource_records = []
        for i in range(50):  # 50 resource patches
            resource_records.append(
                {
                    "patch_id": i,
                    "x": np.random.uniform(0, 1000),
                    "y": np.random.uniform(0, 1000),
                    "resource_type": np.random.choice(["flower", "tree", "crop"]),
                    "quality": np.random.uniform(0.1, 1.0),
                    "quantity": np.random.gamma(2, 100),
                    "renewal_rate": np.random.uniform(0.01, 0.1),
                }
            )

        return {
            "population_data": pd.DataFrame(pop_records),
            "survival_data": pd.DataFrame(survival_records),
            "foraging_data": pd.DataFrame(foraging_records),
            "resource_data": pd.DataFrame(resource_records),
        }

    def _convert_trends_to_dict(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Convert trend results to dictionary format"""
        converted = {}
        for group, trend in trends.items():
            if hasattr(trend, "__dict__"):
                converted[group] = {
                    "trend_type": trend.trend_type.value
                    if hasattr(trend.trend_type, "value")
                    else str(trend.trend_type),
                    "slope": float(trend.slope),
                    "r_squared": float(trend.r_squared),
                    "p_value": float(trend.p_value),
                    "confidence_interval": list(trend.confidence_interval),
                    "significance": trend.significance,
                }
            else:
                converted[group] = trend
        return converted

    def _convert_growth_rates_to_dict(
        self, growth_rates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert growth rate results to dictionary format"""
        converted = {}
        for group, rates in growth_rates.items():
            if hasattr(rates, "__dict__"):
                converted[group] = {
                    "intrinsic_growth_rate": float(rates.intrinsic_growth_rate),
                    "doubling_time": float(rates.doubling_time)
                    if rates.doubling_time
                    else None,
                    "exponential_rate": float(rates.exponential_rate),
                    "logistic_rate": float(rates.logistic_rate),
                    "carrying_capacity": float(rates.carrying_capacity)
                    if rates.carrying_capacity
                    else None,
                    "growth_phase": rates.growth_phase,
                }
            else:
                converted[group] = rates
        return converted

    def _convert_survival_to_dict(
        self, survival_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert survival results to dictionary format"""
        converted = {}
        for group, survival in survival_results.items():
            if hasattr(survival, "__dict__"):
                converted[group] = {
                    "median_survival": float(survival.median_survival),
                    "life_expectancy": float(survival.life_expectancy),
                    "hazard_rates": survival.hazard_rates,
                    "mortality_factors": survival.mortality_factors,
                }
            else:
                converted[group] = survival
        return converted

    def _convert_efficiency_to_dict(self, efficiency_results: Any) -> Dict[str, Any]:
        """Convert foraging efficiency results to dictionary format"""
        if hasattr(efficiency_results, "__dict__"):
            return {
                "overall_efficiency": float(efficiency_results.overall_efficiency),
                "success_rate": float(efficiency_results.success_rate),
                "average_energy_gain": float(efficiency_results.average_energy_gain),
                "average_distance": float(efficiency_results.average_distance),
                "average_time_per_trip": float(
                    efficiency_results.average_time_per_trip
                ),
                "resource_collection_rate": float(
                    efficiency_results.resource_collection_rate
                ),
                "efficiency_category": efficiency_results.efficiency_category.value
                if hasattr(efficiency_results.efficiency_category, "value")
                else str(efficiency_results.efficiency_category),
                "bottlenecks": efficiency_results.bottlenecks,
            }
        return efficiency_results  # type: ignore

    def _convert_optimization_to_dict(self, optimization: Any) -> Dict[str, Any]:
        """Convert resource optimization results to dictionary format"""
        if hasattr(optimization, "__dict__"):
            return {
                "resource_allocation_efficiency": float(
                    optimization.resource_allocation_efficiency
                ),
                "spatial_efficiency": float(optimization.spatial_efficiency),
                "temporal_efficiency": float(optimization.temporal_efficiency),
                "optimization_recommendations": optimization.optimization_recommendations,
            }
        return optimization  # type: ignore

    def _convert_patterns_to_dict(self, patterns: Any) -> Dict[str, Any]:
        """Convert behavioral pattern results to dictionary format"""
        if hasattr(patterns, "__dict__"):
            return {
                "foraging_patterns": patterns.foraging_patterns,
                "temporal_patterns": patterns.temporal_patterns,
                "spatial_patterns": patterns.spatial_patterns,
                "decision_making_metrics": patterns.decision_making_metrics,
                "learning_indicators": patterns.learning_indicators,
            }
        return patterns  # type: ignore

    def _calculate_comprehensive_summary(
        self, data: Dict[str, pd.DataFrame], analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        summary = {}

        # Population summary
        if "population_data" in data:
            pop_df = data["population_data"]
            if len(pop_df) > 0:
                summary["final_population"] = (
                    int(pop_df["population"].iloc[-1])
                    if "population" in pop_df.columns
                    else 0
                )
                summary["peak_population"] = (
                    int(pop_df["population"].max())
                    if "population" in pop_df.columns
                    else 0
                )
                summary["average_population"] = (
                    int(pop_df["population"].mean())
                    if "population" in pop_df.columns
                    else 0
                )
                summary["population_volatility"] = (
                    int(pop_df["population"].std() / pop_df["population"].mean())
                    if "population" in pop_df.columns
                    and pop_df["population"].mean() > 0
                    else 0
                )

        # Foraging summary
        if "foraging_data" in data:
            forage_df = data["foraging_data"]
            if len(forage_df) > 0:
                summary["total_foraging_trips"] = len(forage_df)
                summary["foraging_success_rate"] = (
                    int(forage_df["success"].mean())
                    if "success" in forage_df.columns
                    else 0
                )
                summary["average_nectar_collected"] = (
                    int(forage_df["nectar_collected"].mean())
                    if "nectar_collected" in forage_df.columns
                    else 0
                )
                summary["average_pollen_collected"] = (
                    int(forage_df["pollen_collected"].mean())
                    if "pollen_collected" in forage_df.columns
                    else 0
                )

        # Survival summary
        if "survival_data" in data:
            survival_df = data["survival_data"]
            if len(survival_df) > 0 and "lifespan_days" in survival_df.columns:
                summary["average_lifespan"] = int(survival_df["lifespan_days"].mean())
                summary["max_lifespan"] = int(survival_df["lifespan_days"].max())
                summary["min_lifespan"] = int(survival_df["lifespan_days"].min())

        # Add analysis-derived summaries
        if "population_trends" in analysis_results:
            trends = analysis_results["population_trends"]
            if "overall" in trends:
                summary["population_trend"] = trends["overall"].get(
                    "trend_type", "unknown"
                )
                summary["population_growth_rate"] = trends["overall"].get("slope", 0)

        if "foraging_efficiency" in analysis_results:
            efficiency = analysis_results["foraging_efficiency"]
            summary["foraging_efficiency_score"] = efficiency.get(
                "overall_efficiency", 0
            )
            summary["foraging_efficiency_category"] = efficiency.get(
                "efficiency_category", "unknown"
            )

        return summary

    def _display_empty_results_warning(self, results: Dict[str, Any]) -> None:
        """Display warning about empty results with helpful suggestions"""

        self.context.print_warning(
            results.get("message", "No analysis results generated")
        )

        # Display diagnosis
        if "diagnosis" in results:
            self.context.print_info("\n" + results["diagnosis"])

        # Display suggestions
        if "suggestions" in results:
            self.context.print_info("\nSuggestions:")
            for suggestion in results["suggestions"]:
                self.context.print_info(f"  • {suggestion}")

        # Show any summary statistics that were generated
        if "summary_statistics" in results and results["summary_statistics"]:
            self.context.print_info("\nPartial summary statistics were generated:")
            self._display_summary({"summary_statistics": results["summary_statistics"]})

    def _display_summary(self, results: Dict[str, Any]) -> None:
        """Display analysis summary"""
        from rich.table import Table

        # Create summary table
        table = Table(title="Analysis Summary", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        if "summary_statistics" in results:
            for key, value in results["summary_statistics"].items():
                if isinstance(value, float):
                    table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
                else:
                    table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(table)

        # Show key insights
        if "population_trends" in results:
            self.console.print("\n[bold]Population Trends:[/bold]")
            for group, trend in results["population_trends"].items():
                self.console.print(
                    f"  {group}: {trend.get('trend_type', 'unknown')} (p={trend.get('p_value', 0):.3f})"
                )

        if "foraging_efficiency" in results:
            efficiency = results["foraging_efficiency"]
            self.console.print(
                f"\n[bold]Foraging Efficiency:[/bold] {efficiency.get('efficiency_category', 'unknown')}"
            )
            self.console.print(
                f"  Success Rate: {efficiency.get('success_rate', 0):.1%}"
            )
            self.console.print(
                f"  Resource Collection Rate: {efficiency.get('resource_collection_rate', 0):.2f} units/hour"
            )

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BSTEW Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .value {{ color: #0066cc; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>BSTEW Simulation Analysis Report</h1>
            <p>Generated: {timestamp}</p>

            {content}
        </body>
        </html>
        """

        content_parts = []

        # Summary section
        if "summary_statistics" in results:
            content_parts.append("<div class='section'>")
            content_parts.append("<h2>Summary Statistics</h2>")
            content_parts.append("<table>")
            for key, value in results["summary_statistics"].items():
                content_parts.append(
                    f"<tr><td class='metric'>{key.replace('_', ' ').title()}</td>"
                )
                if isinstance(value, float):
                    content_parts.append(f"<td class='value'>{value:.2f}</td></tr>")
                else:
                    content_parts.append(f"<td class='value'>{value}</td></tr>")
            content_parts.append("</table>")
            content_parts.append("</div>")

        # Population trends section
        if "population_trends" in results:
            content_parts.append("<div class='section'>")
            content_parts.append("<h2>Population Trends</h2>")
            content_parts.append("<table>")
            content_parts.append(
                "<tr><th>Group</th><th>Trend</th><th>Growth Rate</th><th>R²</th><th>P-value</th></tr>"
            )
            for group, trend in results["population_trends"].items():
                content_parts.append(f"<tr><td>{group}</td>")
                content_parts.append(f"<td>{trend.get('trend_type', 'unknown')}</td>")
                content_parts.append(f"<td>{trend.get('slope', 0):.3f}</td>")
                content_parts.append(f"<td>{trend.get('r_squared', 0):.3f}</td>")
                content_parts.append(f"<td>{trend.get('p_value', 0):.4f}</td></tr>")
            content_parts.append("</table>")
            content_parts.append("</div>")

        # Foraging efficiency section
        if "foraging_efficiency" in results:
            content_parts.append("<div class='section'>")
            content_parts.append("<h2>Foraging Efficiency Analysis</h2>")
            efficiency = results["foraging_efficiency"]
            content_parts.append("<table>")
            content_parts.append(
                f"<tr><td class='metric'>Overall Efficiency</td><td class='value'>{efficiency.get('overall_efficiency', 0):.1%}</td></tr>"
            )
            content_parts.append(
                f"<tr><td class='metric'>Success Rate</td><td class='value'>{efficiency.get('success_rate', 0):.1%}</td></tr>"
            )
            content_parts.append(
                f"<tr><td class='metric'>Average Energy Gain</td><td class='value'>{efficiency.get('average_energy_gain', 0):.2f}</td></tr>"
            )
            content_parts.append(
                f"<tr><td class='metric'>Average Distance</td><td class='value'>{efficiency.get('average_distance', 0):.1f} m</td></tr>"
            )
            content_parts.append(
                f"<tr><td class='metric'>Category</td><td class='value'>{efficiency.get('efficiency_category', 'unknown')}</td></tr>"
            )
            content_parts.append("</table>")
            content_parts.append("</div>")

        from datetime import datetime

        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content="\n".join(content_parts),
        )

    def handle_empty_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Handle empty analysis results with proper user feedback"""

        if not results or all(
            len(v) == 0 for v in results.values() if isinstance(v, (list, dict))
        ):
            # Provide helpful diagnosis
            diagnosis = self._diagnose_empty_results()

            return {
                "status": "warning",
                "message": "No analysis results generated",
                "diagnosis": diagnosis,
                "suggestions": [
                    "Check if input directory contains valid simulation data",
                    "Verify data format matches expected structure",
                    "Try running with --debug flag for detailed logging",
                    "Ensure simulation completed successfully before analysis",
                    "Check that data files have expected column names",
                ],
                # Keep any summary statistics that might have been generated
                "summary_statistics": results.get("summary_statistics", {}),
            }

        return results

    def _diagnose_empty_results(self) -> str:
        """Diagnose why analysis results are empty"""

        diagnosis_parts = []

        # Check common issues
        diagnosis_parts.append("Possible causes for empty results:")

        # Check if running from correct directory
        from pathlib import Path

        cwd = Path.cwd()
        if not (cwd / "artifacts").exists():
            diagnosis_parts.append("- Not running from project root directory")

        # Check for common data file patterns
        common_files = [
            "population_data.csv",
            "foraging_data.csv",
            "model_data.csv",
            "results.csv",
        ]
        missing_files = []
        for filename in common_files:
            if not any(Path(".").rglob(filename)):
                missing_files.append(filename)

        if missing_files:
            diagnosis_parts.append(
                f"- Missing expected data files: {', '.join(missing_files)}"
            )

        # Check for empty directories
        artifacts_dir = Path("artifacts")
        if artifacts_dir.exists():
            subdirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
            empty_dirs = [d.name for d in subdirs if not any(d.iterdir())]
            if empty_dirs:
                diagnosis_parts.append(
                    f"- Empty directories in artifacts: {', '.join(empty_dirs)}"
                )

        return "\n".join(diagnosis_parts)


@dataclass
class PlotOptions:
    """Options for plot generation"""

    species: Optional[str] = None
    time_range: Optional[Tuple[str, str]] = None
    save_data: bool = False
    interactive: bool = False
    format: str = "png"
    dpi: int = 300
    figure_size: Tuple[int, int] = (10, 6)


class PlotCommand(BaseCLICommand):
    """Command for generating plots from simulation results"""

    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.progress_manager = ProgressManager(self.console)
        self.population_plotter: Optional[PopulationPlotter] = None
        self.spatial_plotter: Optional[SpatialPlotter] = None

    def execute(
        self,
        input_dir: str,
        plot_type: str = "population",
        output_dir: str = "artifacts/plots",
        species: Optional[str] = None,
        format: str = "png",
        time_range: Optional[str] = None,
        save_data: bool = False,
        interactive: bool = False,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute plot generation"""

        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)

            if not input_path.exists():
                return CLIResult(
                    success=False,
                    message=f"Input directory not found: {input_dir}",
                    exit_code=1,
                )

            output_path.mkdir(parents=True, exist_ok=True)

            # Parse options
            options = PlotOptions(
                species=species,
                time_range=self._parse_time_range(time_range) if time_range else None,
                save_data=save_data,
                interactive=interactive,
                format=format.lower(),
                dpi=kwargs.get("dpi", 300),
                figure_size=kwargs.get("figure_size", (10, 6)),
            )

            self.context.print_info(f"Generating {plot_type} plots...")

            # Load data
            data = self._load_simulation_data(input_path, plot_type, options)

            if not data:
                return CLIResult(
                    success=False,
                    message=f"No data found for {plot_type} plots in {input_dir}",
                    exit_code=1,
                )

            # Generate plots with progress tracking
            with self.progress_manager.progress_context() as progress:
                task = progress.start_task(f"Creating {plot_type} plots...", total=None)

                # Create plots based on type
                plot_files = self._generate_plots(data, plot_type, output_path, options)

                # VALIDATE SUCCESS BEFORE CLAIMING IT
                if len(plot_files) == 0:
                    raise RuntimeError(
                        f"No plots generated for type '{plot_type}'. "
                        f"Possible causes: insufficient data, invalid data format, "
                        f"or missing required columns."
                    )

                progress.finish_task(task, f"Generated {len(plot_files)} plots")

            # Save data if requested
            if save_data:
                data_file = self._save_plot_data(data, output_path, plot_type)
                self.context.print_info(f"Data saved to: {data_file}")

            self.context.print_success(f"Plots saved to: {output_path.absolute()}")

            return CLIResult(
                success=True,
                message="Plot generation completed",
                data={
                    "plot_type": plot_type,
                    "output_path": str(output_path),
                    "plots_created": plot_files,
                    "species": species,
                },
            )

        except Exception as e:
            return self.handle_exception(e, "Plot generation")

    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate command inputs"""
        errors = []

        # Validate plot type
        plot_type = kwargs.get("plot_type", "population")
        valid_types = [
            "population",
            "spatial",
            "foraging",
            "resources",
            "temporal",
            "comparison",
        ]
        errors.extend(
            InputValidator.validate_choice(plot_type, "plot_type", valid_types)
        )

        # Validate format
        format_type = kwargs.get("format", "png")
        valid_formats = ["png", "svg", "pdf", "html"]
        errors.extend(
            InputValidator.validate_choice(format_type, "format", valid_formats)
        )

        # Validate time range if provided
        time_range = kwargs.get("time_range")
        if time_range:
            try:
                self._parse_time_range(time_range)
            except ValueError as e:
                errors.append(f"Invalid time range format: {e}")

        return errors

    def _load_simulation_data(
        self, input_path: Path, plot_type: str, options: PlotOptions
    ) -> Dict[str, Any]:
        """Load simulation data based on plot type"""
        data: Dict[str, Any] = {}

        # Look for standard result files
        if plot_type in ["population", "temporal"]:
            # Load population data
            pop_file = input_path / "population_data.csv"
            if pop_file.exists():
                df = pd.read_csv(pop_file)

                # Apply species filter if specified
                if options.species and "species" in df.columns:
                    df = df[df["species"] == options.species]

                # Apply time range filter if specified
                if options.time_range and "time" in df.columns:
                    df = self._filter_by_time_range(df, options.time_range)

                data["population_data"] = df

            # Load time series data
            ts_file = input_path / "time_series.csv"
            if ts_file.exists():
                df = pd.read_csv(ts_file)
                if options.time_range and "time" in df.columns:
                    df = self._filter_by_time_range(df, options.time_range)
                data["time_series"] = df

        elif plot_type in ["spatial", "foraging", "resources"]:
            # Load spatial data
            spatial_file = input_path / "spatial_data.csv"
            if spatial_file.exists():
                df = pd.read_csv(spatial_file)

                # Apply species filter if specified
                if options.species and "species" in df.columns:
                    df = df[df["species"] == options.species]

                data["spatial_data"] = df

            # Load colony data
            colony_file = input_path / "colony_data.csv"
            if colony_file.exists():
                data["colony_data"] = pd.read_csv(colony_file)

            # Load resource data
            resource_file = input_path / "resource_data.csv"
            if resource_file.exists():
                data["resource_data"] = pd.read_csv(resource_file)

            # Load foraging data
            if plot_type == "foraging":
                foraging_file = input_path / "foraging_data.csv"
                if foraging_file.exists():
                    df = pd.read_csv(foraging_file)
                    if options.species and "species" in df.columns:
                        df = df[df["species"] == options.species]
                    data["foraging_data"] = df

        elif plot_type == "comparison":
            # Load comparison data from multiple scenarios
            scenario_dirs = [d for d in input_path.iterdir() if d.is_dir()]
            comparison_data = {}

            for scenario_dir in scenario_dirs:
                scenario_name = scenario_dir.name
                pop_file = scenario_dir / "population_data.csv"
                if pop_file.exists():
                    df = pd.read_csv(pop_file)
                    if options.species and "species" in df.columns:
                        df = df[df["species"] == options.species]
                    comparison_data[scenario_name] = df

            data["comparison_data"] = comparison_data

        # If no real data found, show clear error with guidance
        if not data:
            self.context.print_error("No data files found for plotting.")
            self.context.print_info("To generate plots, first run a simulation:")
            self.context.print_info("  bstew run --config your_config.json")
            self.context.print_info("  bstew run --days 30  # Quick 30-day simulation")
            raise typer.Exit(1)

        return data

    def _generate_plots(
        self,
        data: Dict[str, Any],
        plot_type: str,
        output_path: Path,
        options: PlotOptions,
    ) -> List[str]:
        """Generate plots based on type"""
        plot_files = []

        # Initialize plotters if needed
        if plot_type in ["population", "temporal"]:
            if not self.population_plotter:
                config = PlotConfig(
                    figure_size=options.figure_size,
                    dpi=options.dpi,
                    save_format=options.format,
                )
                self.population_plotter = PopulationPlotter(config)

        elif plot_type in ["spatial", "foraging", "resources"]:
            if not self.spatial_plotter:
                spatial_config = SpatialConfig(
                    figure_size=options.figure_size, dpi=options.dpi
                )
                self.spatial_plotter = SpatialPlotter(spatial_config)

        # Generate plots based on type
        if plot_type == "population":
            plot_files.extend(
                self._generate_population_plots(data, output_path, options)
            )
        elif plot_type == "spatial":
            plot_files.extend(self._generate_spatial_plots(data, output_path, options))
        elif plot_type == "foraging":
            plot_files.extend(self._generate_foraging_plots(data, output_path, options))
        elif plot_type == "resources":
            plot_files.extend(self._generate_resource_plots(data, output_path, options))
        elif plot_type == "temporal":
            plot_files.extend(self._generate_temporal_plots(data, output_path, options))
        elif plot_type == "comparison":
            plot_files.extend(
                self._generate_comparison_plots(data, output_path, options)
            )

        return plot_files

    def _generate_population_plots(
        self, data: Dict[str, Any], output_path: Path, options: PlotOptions
    ) -> List[str]:
        """Generate population plots"""
        plot_files = []

        if "population_data" in data and self.population_plotter:
            # Population trends
            filename = f"population_trends.{options.format}"
            save_path = str(output_path / filename)

            self.population_plotter.plot_population_trends(
                data["population_data"],
                group_by="species" if not options.species else None,
                save_path=save_path,
            )
            plot_files.append(filename)

            # Growth charts
            filename = f"population_growth.{options.format}"
            save_path = str(output_path / filename)

            self.population_plotter.plot_growth_charts(
                data["population_data"], save_path=save_path
            )
            plot_files.append(filename)

            # Demographic composition
            if "role" in data["population_data"].columns:
                filename = f"demographic_composition.{options.format}"
                save_path = str(output_path / filename)

                self.population_plotter.plot_demographic_composition(
                    data["population_data"], save_path=save_path
                )
                plot_files.append(filename)

        return plot_files

    def _generate_spatial_plots(
        self, data: Dict[str, Any], output_path: Path, options: PlotOptions
    ) -> List[str]:
        """Generate spatial plots"""
        plot_files = []

        if self.spatial_plotter:
            # Colony distribution
            if "colony_data" in data:
                filename = f"colony_distribution.{options.format}"
                save_path = str(output_path / filename)

                self.spatial_plotter.plot_spatial_distribution(
                    data["colony_data"], entity_type="colony", save_path=save_path
                )
                plot_files.append(filename)

                # Colony territories
                filename = f"colony_territories.{options.format}"
                save_path = str(output_path / filename)

                foraging_data = data.get("foraging_data")
                self.spatial_plotter.plot_colony_territories(
                    data["colony_data"],
                    foraging_data=foraging_data,
                    save_path=save_path,
                )
                plot_files.append(filename)

        return plot_files

    def _generate_foraging_plots(
        self, data: Dict[str, Any], output_path: Path, options: PlotOptions
    ) -> List[str]:
        """Generate foraging plots"""
        plot_files = []

        if "foraging_data" in data and self.spatial_plotter:
            # Foraging patterns
            filename = f"foraging_patterns.{options.format}"
            save_path = str(output_path / filename)

            self.spatial_plotter.plot_foraging_patterns(
                data["foraging_data"],
                colony_data=data.get("colony_data"),
                pattern_type="heatmap",
                save_path=save_path,
            )
            plot_files.append(filename)

            # Foraging efficiency
            filename = f"foraging_efficiency.{options.format}"
            save_path = str(output_path / filename)

            self.spatial_plotter.plot_foraging_patterns(
                data["foraging_data"],
                colony_data=data.get("colony_data"),
                pattern_type="efficiency",
                save_path=save_path,
            )
            plot_files.append(filename)

        return plot_files

    def _generate_resource_plots(
        self, data: Dict[str, Any], output_path: Path, options: PlotOptions
    ) -> List[str]:
        """Generate resource plots"""
        plot_files = []

        if "resource_data" in data and self.spatial_plotter:
            filename = f"resource_mapping.{options.format}"
            save_path = str(output_path / filename)

            self.spatial_plotter.plot_resource_mapping(
                data["resource_data"], save_path=save_path
            )
            plot_files.append(filename)

        return plot_files

    def _generate_temporal_plots(
        self, data: Dict[str, Any], output_path: Path, options: PlotOptions
    ) -> List[str]:
        """Generate temporal analysis plots"""
        plot_files = []

        if self.population_plotter:
            # Create dashboard if we have both population and survival data
            if "population_data" in data:
                filename = f"temporal_dashboard.{options.format}"
                save_path = str(output_path / filename)

                self.population_plotter.create_dashboard(
                    data["population_data"],
                    survival_data=data.get("survival_data"),
                    save_path=save_path,
                )
                plot_files.append(filename)

        return plot_files

    def _generate_comparison_plots(
        self, data: Dict[str, Any], output_path: Path, options: PlotOptions
    ) -> List[str]:
        """Generate comparison plots"""
        plot_files = []

        if "comparison_data" in data and data["comparison_data"]:
            # For now, create a simple comparison message
            filename = "scenario_comparison.txt"
            save_path = output_path / filename

            with open(save_path, "w") as f:
                f.write("Scenario Comparison\n")
                f.write("===================\n\n")
                f.write(f"Scenarios: {', '.join(data['comparison_data'].keys())}\n")
                if options.species:
                    f.write(f"Species filter: {options.species}\n")
                f.write("\nDetailed comparison plots would be generated here.\n")

            plot_files.append(filename)

        return plot_files

    def _parse_time_range(self, time_range: str) -> Tuple[str, str]:
        """Parse time range string (e.g., '2024-01-01:2024-12-31')"""
        parts = time_range.split(":")
        if len(parts) != 2:
            raise ValueError("Time range must be in format 'start:end'")
        return (parts[0].strip(), parts[1].strip())

    def _filter_by_time_range(
        self, df: pd.DataFrame, time_range: Tuple[str, str]
    ) -> pd.DataFrame:
        """Filter dataframe by time range"""
        start, end = time_range

        # Convert time column to datetime if needed
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            mask = (df["time"] >= start) & (df["time"] <= end)
            return df[mask]

        return df

    def _save_plot_data(
        self, data: Dict[str, Any], output_path: Path, plot_type: str
    ) -> str:
        """Save plot data to CSV"""
        data_file = output_path / f"{plot_type}_data.csv"

        # Combine relevant dataframes
        if plot_type == "population" and "population_data" in data:
            data["population_data"].to_csv(data_file, index=False)
        elif plot_type in ["spatial", "foraging"] and "spatial_data" in data:
            data["spatial_data"].to_csv(data_file, index=False)
        elif plot_type == "resources" and "resource_data" in data:
            data["resource_data"].to_csv(data_file, index=False)

        return str(data_file)

    def _create_sample_data(
        self, plot_type: str, options: PlotOptions
    ) -> Dict[str, Any]:
        """Create sample data for demonstration"""
        import numpy as np

        data = {}

        if plot_type in ["population", "temporal"]:
            # Create sample population data
            days = 365
            species_list = ["bombus_terrestris", "bombus_lapidarius", "apis_mellifera"]
            if options.species:
                species_list = [options.species]

            records = []
            # Use proper date generation to avoid invalid dates
            from datetime import datetime, timedelta

            start_date = datetime(2024, 1, 1)

            for species in species_list:
                for day in range(days):
                    current_date = start_date + timedelta(days=day)
                    base_pop = 1000 if species == "apis_mellifera" else 100
                    population = base_pop + int(base_pop * 0.5 * np.sin(day / 30))
                    records.append(
                        {
                            "time": current_date.strftime("%Y-%m-%d"),
                            "species": species,
                            "population": max(0, population),
                            "role": "worker" if np.random.random() > 0.1 else "queen",
                        }
                    )

            data["population_data"] = pd.DataFrame(records)

        elif plot_type in ["spatial", "foraging", "resources"]:
            # Create sample spatial data
            n_colonies = 5
            data["colony_data"] = pd.DataFrame(
                {
                    "colony_id": range(n_colonies),
                    "x": np.random.uniform(0, 1000, n_colonies),
                    "y": np.random.uniform(0, 1000, n_colonies),
                    "species": np.random.choice(
                        ["bombus_terrestris", "apis_mellifera"], n_colonies
                    ),
                    "foraging_range": np.random.uniform(300, 700, n_colonies),
                }
            )

            # Create sample resource data
            n_resources = 20
            data["resource_data"] = pd.DataFrame(
                {
                    "resource_id": range(n_resources),
                    "x": np.random.uniform(0, 1000, n_resources),
                    "y": np.random.uniform(0, 1000, n_resources),
                    "quality": np.random.uniform(0.1, 1.0, n_resources),
                    "abundance": np.random.uniform(10, 100, n_resources),
                    "type": np.random.choice(["flower", "tree", "crop"], n_resources),
                }
            )

            if plot_type == "foraging":
                # Create sample foraging data
                n_trips = 100
                data["foraging_data"] = pd.DataFrame(
                    {
                        "trip_id": range(n_trips),
                        "colony_id": np.random.choice(range(n_colonies), n_trips),
                        "x": np.random.uniform(0, 1000, n_trips),
                        "y": np.random.uniform(0, 1000, n_trips),
                        "species": np.random.choice(
                            ["bombus_terrestris", "apis_mellifera"], n_trips
                        ),
                        "efficiency": np.random.uniform(0.3, 0.9, n_trips),
                    }
                )

        return data
