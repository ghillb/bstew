"""
NetLogo BEE-STEWARD v2 Behavioral Validation System
=================================================

Comprehensive validation framework to compare BSTEW simulation outputs
against NetLogo BEE-STEWARD v2 for behavioral accuracy and parity.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

from ..core.data_collection import ComprehensiveDataCollector


@dataclass
class ValidationMetric:
    """Individual validation metric result"""
    metric_name: str
    bstew_value: float
    netlogo_value: float
    difference: float
    relative_difference: float
    tolerance: float
    passes_validation: bool
    statistical_test: Optional[str] = None
    p_value: Optional[float] = None


@dataclass
class ValidationResult:
    """Complete validation result for a category"""
    category: str
    total_metrics: int
    passed_metrics: int
    failed_metrics: int
    pass_rate: float
    individual_results: List[ValidationMetric]
    summary_statistics: Dict[str, float]


class NetLogoDataLoader:
    """Loads and processes NetLogo BEE-STEWARD v2 output data"""
    
    def __init__(self, netlogo_data_path: str):
        self.netlogo_data_path = Path(netlogo_data_path)
        self.logger = logging.getLogger(__name__)
        self.loaded_data: Dict[str, Any] = {}
        
    def load_netlogo_outputs(self) -> Dict[str, Any]:
        """Load all available NetLogo output files"""
        
        netlogo_data = {
            "population_data": self._load_population_data(),
            "activity_patterns": self._load_activity_patterns(),
            "foraging_data": self._load_foraging_data(),
            "mortality_data": self._load_mortality_data(),
            "energy_data": self._load_energy_data(),
            "spatial_data": self._load_spatial_data()
        }
        
        self.loaded_data = netlogo_data
        return netlogo_data
    
    def _load_population_data(self) -> Dict[str, Any]:
        """Load NetLogo population dynamics data"""
        
        try:
            # Look for NetLogo population output files
            pop_files = list(self.netlogo_data_path.glob("*population*.csv"))
            if not pop_files:
                self.logger.warning("No NetLogo population data files found")
                return {}
            
            pop_data = {}
            for file_path in pop_files:
                df = pd.read_csv(file_path)
                
                # Extract key population metrics
                pop_data.update({
                    "total_population_time_series": df.get("total_population", pd.Series([])).tolist() if "total_population" in df else [],
                    "egg_count_time_series": df.get("egg_count", pd.Series([])).tolist() if "egg_count" in df else [],
                    "larva_count_time_series": df.get("larva_count", pd.Series([])).tolist() if "larva_count" in df else [],
                    "pupa_count_time_series": df.get("pupa_count", pd.Series([])).tolist() if "pupa_count" in df else [],
                    "adult_count_time_series": df.get("adult_count", pd.Series([])).tolist() if "adult_count" in df else [],
                    "queen_count": df.get("queen_count", pd.Series([])).tolist() if "queen_count" in df else [],
                    "drone_count": df.get("drone_count", pd.Series([])).tolist() if "drone_count" in df else [],
                    "worker_count": df.get("worker_count", pd.Series([])).tolist() if "worker_count" in df else []
                })
            
            return pop_data
            
        except Exception as e:
            self.logger.error(f"Error loading NetLogo population data: {e}")
            return {}
    
    def _load_activity_patterns(self) -> Dict[str, Any]:
        """Load NetLogo activity pattern data"""
        
        try:
            activity_files = list(self.netlogo_data_path.glob("*activity*.csv"))
            if not activity_files:
                return {}
            
            activity_data = {}
            for file_path in activity_files:
                df = pd.read_csv(file_path)
                
                # Extract activity distribution data
                activity_data.update({
                    "foraging_activity": df.get("foraging_bees", pd.Series([])).tolist() if "foraging_bees" in df else [],
                    "nursing_activity": df.get("nursing_bees", pd.Series([])).tolist() if "nursing_bees" in df else [],
                    "resting_activity": df.get("resting_bees", pd.Series([])).tolist() if "resting_bees" in df else [],
                    "building_activity": df.get("building_bees", pd.Series([])).tolist() if "building_bees" in df else [],
                    "activity_transitions": df.get("activity_transitions", pd.Series([])).tolist() if "activity_transitions" in df else [],
                    "daily_activity_cycles": self._extract_daily_cycles(df)
                })
            
            return activity_data
            
        except Exception as e:
            self.logger.error(f"Error loading NetLogo activity data: {e}")
            return {}
    
    def _load_foraging_data(self) -> Dict[str, Any]:
        """Load NetLogo foraging behavior data"""
        
        try:
            foraging_files = list(self.netlogo_data_path.glob("*foraging*.csv"))
            if not foraging_files:
                return {}
            
            foraging_data = {}
            for file_path in foraging_files:
                df = pd.read_csv(file_path)
                
                foraging_data.update({
                    "total_foraging_trips": df.get("total_trips", []).tolist(),
                    "successful_trips": df.get("successful_trips", []).tolist(),
                    "foraging_efficiency": df.get("foraging_efficiency", []).tolist(),
                    "trip_durations": df.get("trip_duration", []).tolist(),
                    "energy_collected": df.get("energy_collected", []).tolist(),
                    "patch_visitation": df.get("patches_visited", []).tolist(),
                    "dance_frequency": df.get("dances_performed", []).tolist()
                })
            
            return foraging_data
            
        except Exception as e:
            self.logger.error(f"Error loading NetLogo foraging data: {e}")
            return {}
    
    def _load_mortality_data(self) -> Dict[str, Any]:
        """Load NetLogo mortality data"""
        
        try:
            mortality_files = list(self.netlogo_data_path.glob("*mortality*.csv"))
            if not mortality_files:
                return {}
            
            mortality_data = {}
            for file_path in mortality_files:
                df = pd.read_csv(file_path)
                
                mortality_data.update({
                    "daily_deaths": df.get("daily_deaths", []).tolist(),
                    "mortality_rate": df.get("mortality_rate", []).tolist(),
                    "death_causes": df.get("death_causes", {}).to_dict() if "death_causes" in df else {},
                    "lifespan_distribution": df.get("lifespan", []).tolist(),
                    "age_at_death": df.get("age_at_death", []).tolist()
                })
            
            return mortality_data
            
        except Exception as e:
            self.logger.error(f"Error loading NetLogo mortality data: {e}")
            return {}
    
    def _load_energy_data(self) -> Dict[str, Any]:
        """Load NetLogo energy dynamics data"""
        
        try:
            energy_files = list(self.netlogo_data_path.glob("*energy*.csv"))
            if not energy_files:
                return {}
            
            energy_data = {}
            for file_path in energy_files:
                df = pd.read_csv(file_path)
                
                energy_data.update({
                    "colony_energy": df.get("colony_energy", []).tolist(),
                    "daily_energy_intake": df.get("daily_intake", []).tolist(),
                    "daily_energy_consumption": df.get("daily_consumption", []).tolist(),
                    "energy_balance": df.get("energy_balance", []).tolist(),
                    "individual_energy_levels": df.get("individual_energy", []).tolist()
                })
            
            return energy_data
            
        except Exception as e:
            self.logger.error(f"Error loading NetLogo energy data: {e}")
            return {}
    
    def _load_spatial_data(self) -> Dict[str, Any]:
        """Load NetLogo spatial behavior data"""
        
        try:
            spatial_files = list(self.netlogo_data_path.glob("*spatial*.csv"))
            if not spatial_files:
                return {}
            
            spatial_data = {}
            for file_path in spatial_files:
                df = pd.read_csv(file_path)
                
                spatial_data.update({
                    "foraging_distances": df.get("foraging_distance", []).tolist(),
                    "patch_preferences": df.get("patch_preferences", {}).to_dict() if "patch_preferences" in df else {},
                    "territory_size": df.get("territory_size", []).tolist(),
                    "spatial_clustering": df.get("clustering_coefficient", []).tolist()
                })
            
            return spatial_data
            
        except Exception as e:
            self.logger.error(f"Error loading NetLogo spatial data: {e}")
            return {}
    
    def _extract_daily_cycles(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract daily activity cycles from NetLogo data"""
        
        cycles = []
        if "hour" in df.columns and "activity_level" in df.columns:
            for day in df.get("day", [0]).unique():
                day_data = df[df["day"] == day] if "day" in df else df
                cycle = {
                    "day": day,
                    "hourly_activity": day_data.groupby("hour")["activity_level"].mean().to_dict()
                }
                cycles.append(cycle)
        
        return cycles


class BSTEWDataExtractor:
    """Extracts comparable data from BSTEW simulation"""
    
    def __init__(self, data_collector: ComprehensiveDataCollector):
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
    
    def extract_population_data(self) -> Dict[str, Any]:
        """Extract population dynamics data from BSTEW"""
        
        population_data = {}
        
        try:
            # Extract from colony metrics
            for colony_id, metrics in self.data_collector.colony_metrics.items():
                if hasattr(metrics, 'population_size_day_list'):
                    population_data[f"colony_{colony_id}_population"] = metrics.population_size_day_list
                    population_data[f"colony_{colony_id}_eggs"] = getattr(metrics, 'egg_count_day_list', [])
                    population_data[f"colony_{colony_id}_larvae"] = getattr(metrics, 'larva_count_day_list', [])
                    population_data[f"colony_{colony_id}_pupae"] = getattr(metrics, 'pupa_count_day_list', [])
                    population_data[f"colony_{colony_id}_adults"] = getattr(metrics, 'adult_count_day_list', [])
            
            # Aggregate across colonies
            if population_data:
                all_colonies_pop = []
                for key, values in population_data.items():
                    if "population" in key and values:
                        all_colonies_pop.append(values)
                
                if all_colonies_pop:
                    # Sum across colonies for each time step
                    max_len = max(len(series) for series in all_colonies_pop)
                    total_population = []
                    for i in range(max_len):
                        day_total = sum(series[i] if i < len(series) else 0 for series in all_colonies_pop)
                        total_population.append(day_total)
                    
                    population_data["total_population_time_series"] = total_population
            
            return population_data
            
        except Exception as e:
            self.logger.error(f"Error extracting BSTEW population data: {e}")
            return {}
    
    def extract_activity_patterns(self) -> Dict[str, Any]:
        """Extract activity pattern data from BSTEW"""
        
        activity_data = {}
        
        try:
            # Extract from bee metrics
            activity_counts = defaultdict(list)
            
            for bee_id, metrics in self.data_collector.bee_metrics.items():
                if hasattr(metrics, 'activity_time'):
                    for activity, time_spent in metrics.activity_time.items():
                        activity_counts[activity].append(time_spent)
            
            # Calculate activity distributions
            for activity, times in activity_counts.items():
                if times:
                    activity_data[f"{activity}_activity"] = {
                        "mean": np.mean(times),
                        "std": np.std(times),
                        "distribution": times
                    }
            
            return activity_data
            
        except Exception as e:
            self.logger.error(f"Error extracting BSTEW activity data: {e}")
            return {}
    
    def extract_foraging_data(self) -> Dict[str, Any]:
        """Extract foraging behavior data from BSTEW"""
        
        foraging_data = {}
        
        try:
            total_trips = []
            successful_trips = []
            trip_durations = []
            energy_collected = []
            
            for bee_id, metrics in self.data_collector.bee_metrics.items():
                if hasattr(metrics, 'foraging_trips'):
                    total_trips.append(metrics.foraging_trips)
                    successful_trips.append(getattr(metrics, 'successful_trips', 0))
                    
                    if hasattr(metrics, 'foraging_trip_durations'):
                        trip_durations.extend(metrics.foraging_trip_durations)
                    
                    if hasattr(metrics, 'foraging_trip_energies'):
                        energy_collected.extend(metrics.foraging_trip_energies)
            
            foraging_data.update({
                "total_foraging_trips": total_trips,
                "successful_trips": successful_trips,
                "trip_durations": trip_durations,
                "energy_collected": energy_collected,
                "foraging_efficiency": [
                    s/t if t > 0 else 0 for s, t in zip(successful_trips, total_trips)
                ] if total_trips else []
            })
            
            return foraging_data
            
        except Exception as e:
            self.logger.error(f"Error extracting BSTEW foraging data: {e}")
            return {}


class BehavioralValidator:
    """Validates BSTEW behavior against NetLogo BEE-STEWARD v2"""
    
    def __init__(self, tolerance_config: Optional[Dict[str, float]] = None):
        self.tolerance_config = tolerance_config or self._get_default_tolerances()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_tolerances(self) -> Dict[str, float]:
        """Default tolerance levels for validation metrics"""
        return {
            "population_dynamics": 0.05,  # 5% tolerance
            "activity_patterns": 0.10,    # 10% tolerance
            "foraging_behavior": 0.08,    # 8% tolerance
            "mortality_patterns": 0.12,   # 12% tolerance
            "energy_dynamics": 0.07,      # 7% tolerance
            "spatial_behavior": 0.15      # 15% tolerance
        }
    
    def validate_population_dynamics(self, bstew_data: Dict[str, Any], 
                                   netlogo_data: Dict[str, Any]) -> ValidationResult:
        """Validate population dynamics between BSTEW and NetLogo"""
        
        metrics = []
        tolerance = self.tolerance_config["population_dynamics"]
        
        # Compare total population trends
        if ("total_population_time_series" in bstew_data and 
            "total_population_time_series" in netlogo_data):
            
            bstew_pop = bstew_data["total_population_time_series"]
            netlogo_pop = netlogo_data["total_population_time_series"]
            
            if bstew_pop and netlogo_pop:
                # Compare final population sizes
                bstew_final = bstew_pop[-1] if bstew_pop else 0
                netlogo_final = netlogo_pop[-1] if netlogo_pop else 0
                
                metric = self._create_validation_metric(
                    "final_population_size", bstew_final, netlogo_final, tolerance
                )
                metrics.append(metric)
                
                # Compare population growth rates
                if len(bstew_pop) > 1 and len(netlogo_pop) > 1:
                    bstew_growth = (bstew_final - bstew_pop[0]) / len(bstew_pop)
                    netlogo_growth = (netlogo_final - netlogo_pop[0]) / len(netlogo_pop)
                    
                    metric = self._create_validation_metric(
                        "population_growth_rate", bstew_growth, netlogo_growth, tolerance
                    )
                    metrics.append(metric)
        
        # Compare life stage distributions
        life_stages = ["eggs", "larvae", "pupae", "adults"]
        for stage in life_stages:
            bstew_key = f"{stage[:-1]}_count_time_series"  # Remove 's' from end
            netlogo_key = f"{stage[:-1]}_count_time_series"
            
            if bstew_key in bstew_data and netlogo_key in netlogo_data:
                bstew_vals = bstew_data[bstew_key]
                netlogo_vals = netlogo_data[netlogo_key]
                
                if bstew_vals and netlogo_vals:
                    bstew_mean = np.mean(bstew_vals)
                    netlogo_mean = np.mean(netlogo_vals)
                    
                    metric = self._create_validation_metric(
                        f"average_{stage}_count", bstew_mean, netlogo_mean, tolerance
                    )
                    metrics.append(metric)
        
        return self._create_validation_result("population_dynamics", metrics)
    
    def validate_activity_patterns(self, bstew_data: Dict[str, Any], 
                                 netlogo_data: Dict[str, Any]) -> ValidationResult:
        """Validate activity patterns between BSTEW and NetLogo"""
        
        metrics = []
        tolerance = self.tolerance_config["activity_patterns"]
        
        # Compare activity distributions
        activities = ["foraging", "nursing", "resting", "building"]
        
        for activity in activities:
            bstew_key = f"{activity}_activity"
            netlogo_key = f"{activity}_activity"
            
            if bstew_key in bstew_data and netlogo_key in netlogo_data:
                bstew_activity = bstew_data[bstew_key]
                netlogo_activity = netlogo_data[netlogo_key]
                
                # Extract mean values for comparison
                if isinstance(bstew_activity, dict) and "mean" in bstew_activity:
                    bstew_val = bstew_activity["mean"]
                elif isinstance(bstew_activity, list):
                    bstew_val = np.mean(bstew_activity) if bstew_activity else 0
                else:
                    bstew_val = bstew_activity
                
                if isinstance(netlogo_activity, list):
                    netlogo_val = np.mean(netlogo_activity) if netlogo_activity else 0
                else:
                    netlogo_val = netlogo_activity
                
                metric = self._create_validation_metric(
                    f"{activity}_level", bstew_val, netlogo_val, tolerance
                )
                metrics.append(metric)
        
        return self._create_validation_result("activity_patterns", metrics)
    
    def validate_foraging_behavior(self, bstew_data: Dict[str, Any], 
                                 netlogo_data: Dict[str, Any]) -> ValidationResult:
        """Validate foraging behavior between BSTEW and NetLogo"""
        
        metrics = []
        tolerance = self.tolerance_config["foraging_behavior"]
        
        # Compare foraging metrics
        foraging_metrics = [
            "total_foraging_trips", "successful_trips", "foraging_efficiency",
            "trip_durations", "energy_collected"
        ]
        
        for metric_name in foraging_metrics:
            if metric_name in bstew_data and metric_name in netlogo_data:
                bstew_vals = bstew_data[metric_name]
                netlogo_vals = netlogo_data[metric_name]
                
                if bstew_vals and netlogo_vals:
                    # Calculate appropriate summary statistics
                    if metric_name in ["total_foraging_trips", "successful_trips"]:
                        bstew_val = sum(bstew_vals)
                        netlogo_val = sum(netlogo_vals)
                    else:
                        bstew_val = np.mean(bstew_vals)
                        netlogo_val = np.mean(netlogo_vals)
                    
                    metric = self._create_validation_metric(
                        metric_name, bstew_val, netlogo_val, tolerance
                    )
                    metrics.append(metric)
        
        return self._create_validation_result("foraging_behavior", metrics)
    
    def _create_validation_metric(self, name: str, bstew_val: float, netlogo_val: float, 
                                tolerance: float) -> ValidationMetric:
        """Create a validation metric object"""
        
        difference = bstew_val - netlogo_val
        relative_diff = abs(difference) / max(abs(netlogo_val), 1e-6)  # Avoid division by zero
        passes = relative_diff <= tolerance
        
        return ValidationMetric(
            metric_name=name,
            bstew_value=bstew_val,
            netlogo_value=netlogo_val,
            difference=difference,
            relative_difference=relative_diff,
            tolerance=tolerance,
            passes_validation=passes
        )
    
    def _create_validation_result(self, category: str, metrics: List[ValidationMetric]) -> ValidationResult:
        """Create a validation result object"""
        
        total_metrics = len(metrics)
        passed_metrics = sum(1 for m in metrics if m.passes_validation)
        failed_metrics = total_metrics - passed_metrics
        pass_rate = passed_metrics / total_metrics if total_metrics > 0 else 0
        
        # Calculate summary statistics
        summary_stats = {}
        if metrics:
            differences = [m.relative_difference for m in metrics]
            summary_stats = {
                "mean_relative_difference": np.mean(differences),
                "max_relative_difference": np.max(differences),
                "std_relative_difference": np.std(differences)
            }
        
        return ValidationResult(
            category=category,
            total_metrics=total_metrics,
            passed_metrics=passed_metrics,
            failed_metrics=failed_metrics,
            pass_rate=pass_rate,
            individual_results=metrics,
            summary_statistics=summary_stats
        )


class NetLogoValidationSuite:
    """Complete NetLogo validation suite"""
    
    def __init__(self, netlogo_data_path: str, output_path: str = "validation_results"):
        self.netlogo_loader = NetLogoDataLoader(netlogo_data_path)
        self.validator = BehavioralValidator()
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def run_complete_validation(self, data_collector: ComprehensiveDataCollector) -> Dict[str, ValidationResult]:
        """Run complete behavioral validation suite"""
        
        self.logger.info("Starting NetLogo BEE-STEWARD v2 behavioral validation")
        
        # Load NetLogo reference data
        netlogo_data = self.netlogo_loader.load_netlogo_outputs()
        
        # Extract BSTEW data
        bstew_extractor = BSTEWDataExtractor(data_collector)
        bstew_data = {
            "population_data": bstew_extractor.extract_population_data(),
            "activity_patterns": bstew_extractor.extract_activity_patterns(),
            "foraging_data": bstew_extractor.extract_foraging_data()
        }
        
        # Run validations
        validation_results = {}
        
        # Population dynamics validation
        if netlogo_data.get("population_data") and bstew_data.get("population_data"):
            validation_results["population_dynamics"] = self.validator.validate_population_dynamics(
                bstew_data["population_data"], netlogo_data["population_data"]
            )
        
        # Activity patterns validation
        if netlogo_data.get("activity_patterns") and bstew_data.get("activity_patterns"):
            validation_results["activity_patterns"] = self.validator.validate_activity_patterns(
                bstew_data["activity_patterns"], netlogo_data["activity_patterns"]
            )
        
        # Foraging behavior validation
        if netlogo_data.get("foraging_data") and bstew_data.get("foraging_data"):
            validation_results["foraging_behavior"] = self.validator.validate_foraging_behavior(
                bstew_data["foraging_data"], netlogo_data["foraging_data"]
            )
        
        # Generate validation report
        self._generate_validation_report(validation_results)
        
        self.logger.info("NetLogo validation completed")
        return validation_results
    
    def _generate_validation_report(self, results: Dict[str, ValidationResult]) -> None:
        """Generate comprehensive validation report"""
        
        # Create summary report
        summary_data = []
        for category, result in results.items():
            summary_data.append({
                "Category": category,
                "Total Metrics": result.total_metrics,
                "Passed": result.passed_metrics,
                "Failed": result.failed_metrics,
                "Pass Rate": f"{result.pass_rate:.1%}",
                "Mean Relative Difference": f"{result.summary_statistics.get('mean_relative_difference', 0):.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_path / "validation_summary.csv", index=False)
        
        # Create detailed report
        detailed_data = []
        for category, result in results.items():
            for metric in result.individual_results:
                detailed_data.append({
                    "Category": category,
                    "Metric": metric.metric_name,
                    "BSTEW Value": metric.bstew_value,
                    "NetLogo Value": metric.netlogo_value,
                    "Difference": metric.difference,
                    "Relative Difference": metric.relative_difference,
                    "Tolerance": metric.tolerance,
                    "Passes": metric.passes_validation
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(self.output_path / "validation_detailed.csv", index=False)
        
        # Generate visualization
        self._create_validation_plots(results)
        
        self.logger.info(f"Validation report generated in {self.output_path}")
    
    def _create_validation_plots(self, results: Dict[str, ValidationResult]) -> None:
        """Create validation visualization plots"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("NetLogo BEE-STEWARD v2 Validation Results", fontsize=16)
            
            # Pass rate by category
            categories = list(results.keys())
            pass_rates = [results[cat].pass_rate for cat in categories]
            
            axes[0, 0].bar(categories, pass_rates, color=['green' if pr >= 0.8 else 'orange' if pr >= 0.6 else 'red' for pr in pass_rates])
            axes[0, 0].set_title("Pass Rate by Category")
            axes[0, 0].set_ylabel("Pass Rate")
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Relative differences distribution
            all_diffs = []
            for result in results.values():
                all_diffs.extend([m.relative_difference for m in result.individual_results])
            
            if all_diffs:
                axes[0, 1].hist(all_diffs, bins=20, alpha=0.7, color='skyblue')
                axes[0, 1].set_title("Distribution of Relative Differences")
                axes[0, 1].set_xlabel("Relative Difference")
                axes[0, 1].set_ylabel("Frequency")
            
            # Metrics count by category
            metric_counts = [results[cat].total_metrics for cat in categories]
            axes[1, 0].bar(categories, metric_counts, color='lightblue')
            axes[1, 0].set_title("Number of Metrics by Category")
            axes[1, 0].set_ylabel("Metric Count")
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Overall validation status
            total_metrics = sum(result.total_metrics for result in results.values())
            total_passed = sum(result.passed_metrics for result in results.values())
            overall_rate = total_passed / total_metrics if total_metrics > 0 else 0
            
            colors = ['green', 'red']
            sizes = [total_passed, total_metrics - total_passed]
            labels = [f'Passed ({total_passed})', f'Failed ({total_metrics - total_passed})']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[1, 1].set_title(f"Overall Validation Status\n{overall_rate:.1%} Pass Rate")
            
            plt.tight_layout()
            plt.savefig(self.output_path / "validation_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating validation plots: {e}")