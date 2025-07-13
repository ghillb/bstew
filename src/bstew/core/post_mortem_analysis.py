"""
Post-Mortem Analysis and Reporting System for NetLogo BEE-STEWARD v2 Parity
==========================================================================

Comprehensive system for analyzing simulation results, generating detailed reports,
and providing insights into colony performance, failure modes, and optimization opportunities.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import logging
import statistics
import numpy as np
from collections import Counter
from datetime import datetime, timedelta

class AnalysisType(Enum):
    """Types of post-mortem analysis"""
    FULL_SIMULATION = "full_simulation"
    COLONY_SPECIFIC = "colony_specific"
    COMPARATIVE = "comparative"
    FAILURE_ANALYSIS = "failure_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    TREND_ANALYSIS = "trend_analysis"

class OutcomeCategory(Enum):
    """Simulation outcome categories"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    COLLAPSE = "collapse"
    INCONCLUSIVE = "inconclusive"

class FailureMode(Enum):
    """Colony failure modes"""
    STARVATION = "starvation"
    DISEASE_OUTBREAK = "disease_outbreak"
    POPULATION_CRASH = "population_crash"
    ENVIRONMENTAL_STRESS = "environmental_stress"
    FORAGING_FAILURE = "foraging_failure"
    REPRODUCTION_FAILURE = "reproduction_failure"
    RESOURCE_DEPLETION = "resource_depletion"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    UNKNOWN = "unknown"

@dataclass
class SimulationMetrics:
    """Key simulation metrics for analysis"""
    simulation_id: str
    duration_days: int
    start_date: datetime
    end_date: datetime
    
    # Population metrics
    initial_population: int
    final_population: int
    max_population: int
    min_population: int
    avg_population: float
    population_growth_rate: float
    population_std_deviation: float
    
    # Colony outcomes
    colonies_started: int
    colonies_survived: int
    colonies_collapsed: int
    survival_rate: float
    
    # Resource metrics
    total_energy_collected: float
    avg_foraging_efficiency: float
    resource_utilization_rate: float
    
    # Health metrics
    avg_health_score: float
    min_health_score: float
    health_alerts_generated: int
    critical_alerts: int
    
    # Environmental metrics
    avg_temperature: float
    weather_stress_days: int
    seasonal_effects: Dict[str, float]
    
    # Performance metrics
    simulation_time_seconds: float
    steps_per_second: float
    memory_usage_mb: float

@dataclass
class ColonyOutcome:
    """Individual colony outcome analysis"""
    colony_id: int
    species: str
    outcome: OutcomeCategory
    failure_mode: Optional[FailureMode]
    
    # Timeline
    lifespan_days: int
    collapse_day: Optional[int]
    last_healthy_day: Optional[int]
    
    # Performance metrics
    peak_population: int
    final_population: int
    total_energy_collected: float
    avg_foraging_efficiency: float
    
    # Health progression
    initial_health_score: float
    final_health_score: float
    min_health_score: float
    health_decline_rate: float
    
    # Failure analysis
    primary_cause: Optional[str]
    contributing_factors: List[str]
    warning_signs: List[str]
    intervention_opportunities: List[str]

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable", "fluctuating"
    slope: float
    r_squared: float
    significance: float
    
    # Change points
    change_points: List[int]
    trend_segments: List[Dict[str, Any]]
    
    # Statistics
    mean_value: float
    std_deviation: float
    coefficient_variation: float
    
    # Predictions
    predicted_trend: List[float]
    confidence_interval: Tuple[float, float]

@dataclass
class PerformanceInsight:
    """Performance optimization insight"""
    insight_id: str
    category: str
    priority: str  # "high", "medium", "low"
    
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    expected_impact: str = "unknown"
    implementation_difficulty: str = "medium"
    
    # Metrics
    affected_metrics: List[str] = field(default_factory=list)
    potential_improvement: float = 0.0

class PostMortemAnalyzer(BaseModel):
    """Comprehensive post-mortem analysis system"""
    
    model_config = {"validate_assignment": True}
    
    # Analysis configuration
    analysis_type: AnalysisType = AnalysisType.FULL_SIMULATION
    include_predictions: bool = True
    generate_recommendations: bool = True
    detailed_failure_analysis: bool = True
    
    # Data sources
    simulation_data: Optional[Dict[str, Any]] = None
    health_monitoring_data: Optional[Dict[str, Any]] = None
    foraging_analytics: Optional[Dict[str, Any]] = None
    comprehensive_data: Optional[Dict[str, Any]] = None
    
    # Analysis results
    simulation_metrics: Optional[SimulationMetrics] = None
    colony_outcomes: List[ColonyOutcome] = Field(default_factory=list)
    trend_analyses: Dict[str, TrendAnalysis] = Field(default_factory=dict)
    performance_insights: List[PerformanceInsight] = Field(default_factory=list)
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    analysis_duration: Optional[float] = None
    
    # Logging
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
    
    def analyze_simulation(self, model_data: Dict[str, Any], 
                          additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive post-mortem analysis"""
        
        start_time = datetime.now()
        self.logger.info("Starting post-mortem analysis...")
        
        # Store data sources
        self.simulation_data = model_data
        if additional_data:
            self.health_monitoring_data = additional_data.get("health_monitoring")
            self.foraging_analytics = additional_data.get("foraging_analytics")
            self.comprehensive_data = additional_data.get("comprehensive_data")
        
        analysis_results = {
            "analysis_id": f"postmortem_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "analysis_type": self.analysis_type.value,
            "timestamp": start_time.isoformat()
        }
        
        try:
            # Extract simulation metrics
            self.simulation_metrics = self._extract_simulation_metrics(model_data)
            analysis_results["simulation_metrics"] = self.simulation_metrics
            
            # Analyze colony outcomes
            self.colony_outcomes = self._analyze_colony_outcomes(model_data)
            analysis_results["colony_outcomes"] = self.colony_outcomes
            
            # Perform trend analysis
            if self.analysis_type in [AnalysisType.FULL_SIMULATION, AnalysisType.TREND_ANALYSIS]:
                self.trend_analyses = self._perform_trend_analysis(model_data)
                analysis_results["trend_analyses"] = self.trend_analyses
            
            # Generate performance insights
            if self.generate_recommendations:
                self.performance_insights = self._generate_performance_insights()
                analysis_results["performance_insights"] = self.performance_insights
            
            # Failure analysis
            if self.detailed_failure_analysis:
                failure_analysis = self._perform_failure_analysis()
                analysis_results["failure_analysis"] = failure_analysis
            
            # Comparative analysis
            if self.analysis_type == AnalysisType.COMPARATIVE:
                comparative_results = self._perform_comparative_analysis()
                analysis_results["comparative_analysis"] = comparative_results
            
            # Executive summary
            executive_summary = self._generate_executive_summary()
            analysis_results["executive_summary"] = executive_summary
            
            # Calculate analysis duration
            end_time = datetime.now()
            self.analysis_duration = (end_time - start_time).total_seconds()
            analysis_results["analysis_duration"] = self.analysis_duration
            
            self.logger.info(f"Post-mortem analysis completed in {self.analysis_duration:.2f} seconds")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Post-mortem analysis failed: {e}")
            analysis_results["error"] = str(e)
            analysis_results["status"] = "failed"
            return analysis_results
    
    def _extract_simulation_metrics(self, model_data: Dict[str, Any]) -> SimulationMetrics:
        """Extract key simulation metrics"""
        
        # Basic simulation info
        simulation_id = model_data.get("simulation_id", "unknown")
        duration_days = model_data.get("current_day", model_data.get("duration_days", 0))
        
        # Get time series data
        time_series = model_data.get("time_series", {})
        population_series = time_series.get("population", [])
        health_series = time_series.get("health_scores", [])
        
        # Population metrics
        if population_series:
            initial_population = population_series[0]
            final_population = population_series[-1]
            max_population = max(population_series)
            min_population = min(population_series)
            avg_population = statistics.mean(population_series)
            population_std_deviation = statistics.stdev(population_series) if len(population_series) > 1 else 0.0
            
            # Calculate growth rate
            if initial_population > 0 and duration_days > 0:
                population_growth_rate = (final_population - initial_population) / (initial_population * duration_days)
            else:
                population_growth_rate = 0.0
        else:
            initial_population = final_population = max_population = min_population = 0
            avg_population = population_growth_rate = population_std_deviation = 0.0
        
        # Colony outcomes
        colonies_data = model_data.get("colonies", [])
        colonies_started = len(colonies_data)
        colonies_survived = len([c for c in colonies_data if c.get("status") != "collapsed"])
        colonies_collapsed = colonies_started - colonies_survived
        survival_rate = colonies_survived / max(1, colonies_started)
        
        # Resource metrics
        foraging_data = model_data.get("foraging_analytics", {})
        total_energy_collected = foraging_data.get("total_energy_collected", 0.0)
        avg_foraging_efficiency = foraging_data.get("avg_efficiency", 0.0)
        resource_utilization_rate = model_data.get("resource_utilization", 0.0)
        
        # Health metrics
        if health_series:
            avg_health_score = statistics.mean(health_series)
            min_health_score = min(health_series)
        else:
            avg_health_score = min_health_score = 0.0
        
        health_alerts_generated = model_data.get("total_health_alerts", 0)
        critical_alerts = model_data.get("critical_health_alerts", 0)
        
        # Environmental metrics
        weather_data = model_data.get("weather_history", [])
        if weather_data:
            temperatures = [w.get("temperature", 20.0) for w in weather_data]
            avg_temperature = statistics.mean(temperatures)
            weather_stress_days = len([w for w in weather_data 
                                     if w.get("temperature", 20.0) < 10 or w.get("temperature", 20.0) > 30])
        else:
            avg_temperature = 20.0
            weather_stress_days = 0
        
        seasonal_effects = model_data.get("seasonal_effects", {
            "spring": 1.0, "summer": 1.0, "autumn": 0.8, "winter": 0.5
        })
        
        # Performance metrics
        simulation_time_seconds = model_data.get("simulation_time", 0.0)
        steps_per_second = duration_days / max(1, simulation_time_seconds) if simulation_time_seconds > 0 else 0.0
        memory_usage_mb = model_data.get("memory_usage", 0.0)
        
        return SimulationMetrics(
            simulation_id=simulation_id,
            duration_days=duration_days,
            start_date=datetime.now() - timedelta(days=duration_days),
            end_date=datetime.now(),
            initial_population=initial_population,
            final_population=final_population,
            max_population=max_population,
            min_population=min_population,
            avg_population=avg_population,
            population_growth_rate=population_growth_rate,
            population_std_deviation=population_std_deviation,
            colonies_started=colonies_started,
            colonies_survived=colonies_survived,
            colonies_collapsed=colonies_collapsed,
            survival_rate=survival_rate,
            total_energy_collected=total_energy_collected,
            avg_foraging_efficiency=avg_foraging_efficiency,
            resource_utilization_rate=resource_utilization_rate,
            avg_health_score=avg_health_score,
            min_health_score=min_health_score,
            health_alerts_generated=health_alerts_generated,
            critical_alerts=critical_alerts,
            avg_temperature=avg_temperature,
            weather_stress_days=weather_stress_days,
            seasonal_effects=seasonal_effects,
            simulation_time_seconds=simulation_time_seconds,
            steps_per_second=steps_per_second,
            memory_usage_mb=memory_usage_mb
        )
    
    def _analyze_colony_outcomes(self, model_data: Dict[str, Any]) -> List[ColonyOutcome]:
        """Analyze individual colony outcomes"""
        
        outcomes = []
        colonies_data = model_data.get("colonies", [])
        
        for colony_data in colonies_data:
            colony_id = colony_data.get("id", 0)
            species = colony_data.get("species", "unknown")
            
            # Determine outcome
            final_population = colony_data.get("final_population", 0)
            initial_population = colony_data.get("initial_population", 50)
            status = colony_data.get("status", "active")
            
            if status == "collapsed" or final_population == 0:
                outcome = OutcomeCategory.COLLAPSE
            elif final_population < initial_population * 0.5:
                outcome = OutcomeCategory.FAILURE
            elif final_population < initial_population * 0.8:
                outcome = OutcomeCategory.PARTIAL_SUCCESS
            else:
                outcome = OutcomeCategory.SUCCESS
            
            # Analyze failure mode
            failure_mode = self._determine_failure_mode(colony_data)
            
            # Timeline analysis
            lifespan_days = colony_data.get("lifespan", model_data.get("current_day", 0))
            collapse_day = colony_data.get("collapse_day")
            last_healthy_day = colony_data.get("last_healthy_day")
            
            # Performance metrics
            peak_population = colony_data.get("max_population", final_population)
            total_energy_collected = colony_data.get("energy_collected", 0.0)
            avg_foraging_efficiency = colony_data.get("avg_foraging_efficiency", 0.0)
            
            # Health progression
            health_history = colony_data.get("health_history", [])
            if health_history:
                initial_health_score = health_history[0]
                final_health_score = health_history[-1]
                min_health_score = min(health_history)
                
                # Calculate health decline rate
                if len(health_history) > 1:
                    health_decline_rate = (final_health_score - initial_health_score) / len(health_history)
                else:
                    health_decline_rate = 0.0
            else:
                initial_health_score = final_health_score = min_health_score = 0.5
                health_decline_rate = 0.0
            
            # Failure analysis
            primary_cause = self._identify_primary_cause(colony_data, failure_mode)
            contributing_factors = self._identify_contributing_factors(colony_data)
            warning_signs = self._identify_warning_signs(colony_data)
            intervention_opportunities = self._identify_intervention_opportunities(colony_data)
            
            outcome_analysis = ColonyOutcome(
                colony_id=colony_id,
                species=species,
                outcome=outcome,
                failure_mode=failure_mode,
                lifespan_days=lifespan_days,
                collapse_day=collapse_day,
                last_healthy_day=last_healthy_day,
                peak_population=peak_population,
                final_population=final_population,
                total_energy_collected=total_energy_collected,
                avg_foraging_efficiency=avg_foraging_efficiency,
                initial_health_score=initial_health_score,
                final_health_score=final_health_score,
                min_health_score=min_health_score,
                health_decline_rate=health_decline_rate,
                primary_cause=primary_cause,
                contributing_factors=contributing_factors,
                warning_signs=warning_signs,
                intervention_opportunities=intervention_opportunities
            )
            
            outcomes.append(outcome_analysis)
        
        return outcomes
    
    def _determine_failure_mode(self, colony_data: Dict[str, Any]) -> Optional[FailureMode]:
        """Determine primary failure mode for a colony"""
        
        # Check explicit failure mode
        if "failure_mode" in colony_data:
            return FailureMode(colony_data["failure_mode"])
        
        # Infer from colony data
        final_population = colony_data.get("final_population", 0)
        energy_per_bee = colony_data.get("final_energy_per_bee", 0.0)
        disease_prevalence = colony_data.get("final_disease_prevalence", 0.0)
        foraging_efficiency = colony_data.get("avg_foraging_efficiency", 0.0)
        
        # Check for starvation
        if energy_per_bee < 20.0:
            return FailureMode.STARVATION
        
        # Check for disease outbreak
        if disease_prevalence > 0.3:
            return FailureMode.DISEASE_OUTBREAK
        
        # Check for population crash
        initial_population = colony_data.get("initial_population", 50)
        if final_population < initial_population * 0.2:
            return FailureMode.POPULATION_CRASH
        
        # Check for foraging failure
        if foraging_efficiency < 0.3:
            return FailureMode.FORAGING_FAILURE
        
        # Check for environmental stress
        stress_events = colony_data.get("stress_events", 0)
        if stress_events > 10:
            return FailureMode.ENVIRONMENTAL_STRESS
        
        # Default to unknown if no clear pattern
        if final_population == 0 or colony_data.get("status") == "collapsed":
            return FailureMode.UNKNOWN
        
        return None
    
    def _identify_primary_cause(self, colony_data: Dict[str, Any], failure_mode: Optional[FailureMode]) -> Optional[str]:
        """Identify primary cause of colony issues"""
        
        if failure_mode == FailureMode.STARVATION:
            return "Insufficient food resources or poor foraging efficiency"
        elif failure_mode == FailureMode.DISEASE_OUTBREAK:
            return "High disease prevalence leading to population decline"
        elif failure_mode == FailureMode.POPULATION_CRASH:
            return "Rapid population decline due to multiple stressors"
        elif failure_mode == FailureMode.FORAGING_FAILURE:
            return "Poor foraging performance reducing resource acquisition"
        elif failure_mode == FailureMode.ENVIRONMENTAL_STRESS:
            return "Adverse environmental conditions"
        elif failure_mode == FailureMode.REPRODUCTION_FAILURE:
            return "Low reproduction rates preventing population maintenance"
        else:
            return None
    
    def _identify_contributing_factors(self, colony_data: Dict[str, Any]) -> List[str]:
        """Identify contributing factors to colony problems"""
        
        factors = []
        
        # Environmental factors
        avg_temp = colony_data.get("avg_temperature", 20.0)
        if avg_temp < 10.0 or avg_temp > 30.0:
            factors.append("Suboptimal temperature conditions")
        
        # Resource factors
        resource_scarcity = colony_data.get("avg_resource_scarcity", 0.0)
        if resource_scarcity > 0.7:
            factors.append("High resource scarcity")
        
        # Health factors
        min_health = colony_data.get("min_health_score", 1.0)
        if min_health < 0.5:
            factors.append("Prolonged poor health status")
        
        # Behavioral factors
        activity_level = colony_data.get("avg_activity_level", 1.0)
        if activity_level < 0.6:
            factors.append("Reduced colony activity")
        
        # Population density
        max_population = colony_data.get("max_population", 0)
        if max_population > 200:
            factors.append("Overcrowding stress")
        elif max_population < 20:
            factors.append("Small population vulnerability")
        
        return factors
    
    def _identify_warning_signs(self, colony_data: Dict[str, Any]) -> List[str]:
        """Identify early warning signs that were present"""
        
        warning_signs = []
        
        # Population decline
        population_trend = colony_data.get("population_trend", "stable")
        if population_trend == "declining":
            warning_signs.append("Declining population trend detected")
        
        # Health decline
        health_trend = colony_data.get("health_trend", "stable")
        if health_trend == "declining":
            warning_signs.append("Declining health score trend")
        
        # Foraging issues
        foraging_efficiency = colony_data.get("avg_foraging_efficiency", 0.8)
        if foraging_efficiency < 0.5:
            warning_signs.append("Low foraging efficiency")
        
        # Alert history
        alert_count = colony_data.get("total_alerts", 0)
        if alert_count > 10:
            warning_signs.append("High number of health alerts generated")
        
        # Mortality spikes
        max_mortality = colony_data.get("max_mortality_rate", 0.0)
        if max_mortality > 0.05:
            warning_signs.append("High mortality rate episodes")
        
        return warning_signs
    
    def _identify_intervention_opportunities(self, colony_data: Dict[str, Any]) -> List[str]:
        """Identify missed intervention opportunities"""
        
        opportunities = []
        
        # Early intervention points
        collapse_day = colony_data.get("collapse_day")
        first_alert_day = colony_data.get("first_alert_day")
        
        if collapse_day and first_alert_day:
            intervention_window = collapse_day - first_alert_day
            if intervention_window > 5:
                opportunities.append(f"Early intervention possible {intervention_window} days before collapse")
        
        # Resource supplementation
        min_energy = colony_data.get("min_energy_per_bee", 100.0)
        if min_energy < 50.0:
            opportunities.append("Resource supplementation could have prevented starvation")
        
        # Disease management
        max_disease = colony_data.get("max_disease_prevalence", 0.0)
        if max_disease > 0.15:
            opportunities.append("Disease management intervention needed")
        
        # Environmental mitigation
        stress_events = colony_data.get("stress_events", 0)
        if stress_events > 5:
            opportunities.append("Environmental stress mitigation possible")
        
        return opportunities
    
    def _perform_trend_analysis(self, model_data: Dict[str, Any]) -> Dict[str, TrendAnalysis]:
        """Perform trend analysis on key metrics"""
        
        trend_analyses = {}
        time_series = model_data.get("time_series", {})
        
        for metric_name, values in time_series.items():
            if len(values) < 5:  # Need sufficient data points
                continue
            
            trend = self._analyze_metric_trend(metric_name, values)
            trend_analyses[metric_name] = trend
        
        return trend_analyses
    
    def _analyze_metric_trend(self, metric_name: str, values: List[float]) -> TrendAnalysis:
        """Analyze trend for a specific metric"""
        
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction="insufficient_data",
                slope=0.0,
                r_squared=0.0,
                significance=0.0,
                change_points=[],
                trend_segments=[],
                mean_value=0.0,
                std_deviation=0.0,
                coefficient_variation=0.0,
                predicted_trend=[],
                confidence_interval=(0.0, 0.0)
            )
        
        # Convert to numpy arrays
        x = np.array(range(len(values)))
        y = np.array(values)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        # Calculate R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Calculate significance (simplified)
        significance = r_squared  # Use R-squared as significance proxy
        
        # Detect change points (simplified)
        change_points = self._detect_change_points(values)
        
        # Create trend segments
        trend_segments = self._create_trend_segments(values, change_points)
        
        # Basic statistics
        mean_value = np.mean(y)
        std_deviation = np.std(y)
        coefficient_variation = std_deviation / mean_value if mean_value != 0 else 0.0
        
        # Simple prediction (extend trend)
        prediction_steps = 5
        future_x = np.array(range(len(values), len(values) + prediction_steps))
        predicted_trend = (slope * future_x + intercept).tolist()
        
        # Confidence interval (simplified)
        prediction_std = np.std(y - y_pred)
        confidence_interval = (
            predicted_trend[-1] - 1.96 * prediction_std,
            predicted_trend[-1] + 1.96 * prediction_std
        )
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            slope=slope,
            r_squared=r_squared,
            significance=significance,
            change_points=change_points,
            trend_segments=trend_segments,
            mean_value=mean_value,
            std_deviation=std_deviation,
            coefficient_variation=coefficient_variation,
            predicted_trend=predicted_trend,
            confidence_interval=confidence_interval
        )
    
    def _detect_change_points(self, values: List[float]) -> List[int]:
        """Detect significant change points in time series"""
        
        change_points = []
        window_size = max(3, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            # Calculate means before and after point
            before_mean = np.mean(values[i-window_size:i])
            after_mean = np.mean(values[i:i+window_size])
            
            # Check for significant change
            if abs(after_mean - before_mean) > np.std(values) * 0.5:
                change_points.append(i)
        
        return change_points
    
    def _create_trend_segments(self, values: List[float], change_points: List[int]) -> List[Dict[str, Any]]:
        """Create trend segments based on change points"""
        
        segments = []
        start_points = [0] + change_points
        end_points = change_points + [len(values)]
        
        for start, end in zip(start_points, end_points):
            segment_values = values[start:end]
            if len(segment_values) > 1:
                segment_slope = np.polyfit(range(len(segment_values)), segment_values, 1)[0]
                segments.append({
                    "start_index": start,
                    "end_index": end,
                    "duration": end - start,
                    "slope": segment_slope,
                    "mean_value": np.mean(segment_values),
                    "trend": "increasing" if segment_slope > 0.01 else "decreasing" if segment_slope < -0.01 else "stable"
                })
        
        return segments
    
    def _generate_performance_insights(self) -> List[PerformanceInsight]:
        """Generate performance optimization insights"""
        
        insights = []
        
        if not self.simulation_metrics or not self.colony_outcomes:
            return insights
        
        # Survival rate insight
        if self.simulation_metrics.survival_rate < 0.8:
            insights.append(PerformanceInsight(
                insight_id="low_survival_rate",
                category="Colony Management",
                priority="high",
                title="Low Colony Survival Rate",
                description=f"Only {self.simulation_metrics.survival_rate:.1%} of colonies survived the simulation",
                evidence=[
                    f"{self.simulation_metrics.colonies_collapsed} out of {self.simulation_metrics.colonies_started} colonies collapsed",
                    f"Average health score: {self.simulation_metrics.avg_health_score:.2f}"
                ],
                recommendations=[
                    "Implement early warning system for colony health decline",
                    "Increase resource availability during critical periods",
                    "Improve disease management protocols"
                ],
                expected_impact="15-25% improvement in survival rate",
                implementation_difficulty="medium",
                affected_metrics=["survival_rate", "avg_health_score"],
                potential_improvement=0.2
            ))
        
        # Foraging efficiency insight
        if self.simulation_metrics.avg_foraging_efficiency < 0.6:
            insights.append(PerformanceInsight(
                insight_id="low_foraging_efficiency",
                category="Foraging Optimization",
                priority="high",
                title="Poor Foraging Performance",
                description=f"Average foraging efficiency of {self.simulation_metrics.avg_foraging_efficiency:.2f} is below optimal",
                evidence=[
                    f"Total energy collected: {self.simulation_metrics.total_energy_collected:.0f}",
                    f"Resource utilization rate: {self.simulation_metrics.resource_utilization_rate:.2f}"
                ],
                recommendations=[
                    "Optimize patch selection algorithms",
                    "Improve bee communication and recruitment",
                    "Reduce foraging distances through better patch distribution"
                ],
                expected_impact="20-30% increase in resource collection",
                implementation_difficulty="medium",
                affected_metrics=["avg_foraging_efficiency", "total_energy_collected"],
                potential_improvement=0.25
            ))
        
        # Population stability insight
        population_cv = self.simulation_metrics.population_std_deviation / max(1.0, self.simulation_metrics.avg_population)
        if population_cv > 0.3:
            insights.append(PerformanceInsight(
                insight_id="population_instability",
                category="Population Dynamics",
                priority="medium",
                title="High Population Variability",
                description="Colony populations show high variability indicating instability",
                evidence=[
                    f"Population range: {self.simulation_metrics.min_population} - {self.simulation_metrics.max_population}",
                    f"Average population: {self.simulation_metrics.avg_population:.0f}"
                ],
                recommendations=[
                    "Stabilize resource availability",
                    "Implement population control mechanisms",
                    "Reduce environmental stress factors"
                ],
                expected_impact="10-15% reduction in population variability",
                implementation_difficulty="low",
                affected_metrics=["population_stability", "avg_population"],
                potential_improvement=0.15
            ))
        
        # Environmental stress insight
        if self.simulation_metrics.weather_stress_days > self.simulation_metrics.duration_days * 0.2:
            insights.append(PerformanceInsight(
                insight_id="high_environmental_stress",
                category="Environmental Management",
                priority="medium",
                title="Excessive Environmental Stress",
                description=f"{self.simulation_metrics.weather_stress_days} days of weather stress detected",
                evidence=[
                    f"Average temperature: {self.simulation_metrics.avg_temperature:.1f}°C",
                    f"Stress days: {self.simulation_metrics.weather_stress_days}/{self.simulation_metrics.duration_days}"
                ],
                recommendations=[
                    "Implement weather prediction and response systems",
                    "Provide shelter or climate control options",
                    "Adjust colony behavior for weather conditions"
                ],
                expected_impact="5-10% improvement in overall performance",
                implementation_difficulty="high",
                affected_metrics=["avg_health_score", "survival_rate"],
                potential_improvement=0.08
            ))
        
        return insights
    
    def _perform_failure_analysis(self) -> Dict[str, Any]:
        """Perform detailed failure analysis"""
        
        failure_analysis = {
            "total_failures": 0,
            "failure_modes": {},
            "failure_timeline": {},
            "common_patterns": [],
            "prevention_strategies": []
        }
        
        failed_colonies = [c for c in self.colony_outcomes 
                          if c.outcome in [OutcomeCategory.FAILURE, OutcomeCategory.COLLAPSE]]
        
        failure_analysis["total_failures"] = len(failed_colonies)
        
        # Analyze failure modes
        failure_mode_counts = Counter([c.failure_mode.value for c in failed_colonies if c.failure_mode])
        failure_analysis["failure_modes"] = dict(failure_mode_counts)
        
        # Analyze failure timeline
        failure_days = [c.collapse_day for c in failed_colonies if c.collapse_day]
        if failure_days:
            failure_analysis["failure_timeline"] = {
                "earliest_failure": min(failure_days),
                "latest_failure": max(failure_days),
                "average_failure_day": statistics.mean(failure_days),
                "failure_distribution": self._create_failure_distribution(failure_days)
            }
        
        # Identify common patterns
        failure_analysis["common_patterns"] = self._identify_failure_patterns(failed_colonies)
        
        # Generate prevention strategies
        failure_analysis["prevention_strategies"] = self._generate_prevention_strategies(failed_colonies)
        
        return failure_analysis
    
    def _create_failure_distribution(self, failure_days: List[int]) -> Dict[str, int]:
        """Create failure distribution by time period"""
        
        distribution = {"early": 0, "mid": 0, "late": 0}
        duration = self.simulation_metrics.duration_days if self.simulation_metrics else 365
        
        for day in failure_days:
            if day < duration * 0.33:
                distribution["early"] += 1
            elif day < duration * 0.67:
                distribution["mid"] += 1
            else:
                distribution["late"] += 1
        
        return distribution
    
    def _identify_failure_patterns(self, failed_colonies: List[ColonyOutcome]) -> List[str]:
        """Identify common failure patterns"""
        
        patterns = []
        
        # Pattern: Rapid collapse
        rapid_collapses = [c for c in failed_colonies if c.lifespan_days < 30]
        if len(rapid_collapses) > len(failed_colonies) * 0.3:
            patterns.append("High rate of rapid colony collapse (< 30 days)")
        
        # Pattern: Health decline
        health_declines = [c for c in failed_colonies if c.health_decline_rate < -0.01]
        if len(health_declines) > len(failed_colonies) * 0.5:
            patterns.append("Consistent health decline pattern in failed colonies")
        
        # Pattern: Small population vulnerability
        small_pop_failures = [c for c in failed_colonies if c.peak_population < 50]
        if len(small_pop_failures) > len(failed_colonies) * 0.4:
            patterns.append("Small colonies more vulnerable to collapse")
        
        # Pattern: Species-specific failures
        species_failures = Counter([c.species for c in failed_colonies])
        if species_failures:
            most_vulnerable = species_failures.most_common(1)[0]
            if most_vulnerable[1] > len(failed_colonies) * 0.5:
                patterns.append(f"Species {most_vulnerable[0]} shows higher failure rate")
        
        return patterns
    
    def _generate_prevention_strategies(self, failed_colonies: List[ColonyOutcome]) -> List[str]:
        """Generate failure prevention strategies"""
        
        strategies = []
        
        # Analyze primary causes
        primary_causes = Counter([c.primary_cause for c in failed_colonies if c.primary_cause])
        
        for cause, count in primary_causes.most_common(3):
            if "starvation" in cause.lower():
                strategies.append("Implement resource monitoring and supplemental feeding protocols")
            elif "disease" in cause.lower():
                strategies.append("Establish disease surveillance and treatment programs")
            elif "population" in cause.lower():
                strategies.append("Monitor population dynamics and implement breeding support")
            elif "foraging" in cause.lower():
                strategies.append("Optimize foraging patch distribution and quality")
        
        # General strategies based on patterns
        if any("rapid" in p for p in self._identify_failure_patterns(failed_colonies)):
            strategies.append("Implement early intervention protocols for rapid decline detection")
        
        if any("health decline" in p for p in self._identify_failure_patterns(failed_colonies)):
            strategies.append("Enhance health monitoring frequency during critical periods")
        
        return list(set(strategies))  # Remove duplicates
    
    def _perform_comparative_analysis(self) -> Dict[str, Any]:
        """Perform comparative analysis between successful and failed colonies"""
        
        successful_colonies = [c for c in self.colony_outcomes if c.outcome == OutcomeCategory.SUCCESS]
        failed_colonies = [c for c in self.colony_outcomes 
                          if c.outcome in [OutcomeCategory.FAILURE, OutcomeCategory.COLLAPSE]]
        
        if not successful_colonies or not failed_colonies:
            return {"message": "Insufficient data for comparative analysis"}
        
        return {
            "success_factors": self._identify_success_factors(successful_colonies, failed_colonies),
            "performance_differences": self._analyze_performance_differences(successful_colonies, failed_colonies),
            "key_differentiators": self._identify_key_differentiators(successful_colonies, failed_colonies)
        }
    
    def _identify_success_factors(self, successful: List[ColonyOutcome], failed: List[ColonyOutcome]) -> List[str]:
        """Identify factors that correlate with success"""
        
        factors = []
        
        # Compare average metrics
        success_avg_health = statistics.mean([c.initial_health_score for c in successful])
        failed_avg_health = statistics.mean([c.initial_health_score for c in failed])
        
        if success_avg_health > failed_avg_health * 1.2:
            factors.append("Higher initial health scores correlate with success")
        
        success_avg_foraging = statistics.mean([c.avg_foraging_efficiency for c in successful])
        failed_avg_foraging = statistics.mean([c.avg_foraging_efficiency for c in failed])
        
        if success_avg_foraging > failed_avg_foraging * 1.3:
            factors.append("Superior foraging efficiency is key to survival")
        
        success_avg_population = statistics.mean([c.peak_population for c in successful])
        failed_avg_population = statistics.mean([c.peak_population for c in failed])
        
        if success_avg_population > failed_avg_population * 1.5:
            factors.append("Larger peak populations improve survival odds")
        
        return factors
    
    def _analyze_performance_differences(self, successful: List[ColonyOutcome], failed: List[ColonyOutcome]) -> Dict[str, Any]:
        """Analyze performance differences between groups"""
        
        return {
            "health_scores": {
                "successful_avg": statistics.mean([c.initial_health_score for c in successful]),
                "failed_avg": statistics.mean([c.initial_health_score for c in failed]),
                "difference": statistics.mean([c.initial_health_score for c in successful]) - 
                            statistics.mean([c.initial_health_score for c in failed])
            },
            "foraging_efficiency": {
                "successful_avg": statistics.mean([c.avg_foraging_efficiency for c in successful]),
                "failed_avg": statistics.mean([c.avg_foraging_efficiency for c in failed]),
                "difference": statistics.mean([c.avg_foraging_efficiency for c in successful]) - 
                            statistics.mean([c.avg_foraging_efficiency for c in failed])
            },
            "population_metrics": {
                "successful_peak": statistics.mean([c.peak_population for c in successful]),
                "failed_peak": statistics.mean([c.peak_population for c in failed]),
                "successful_final": statistics.mean([c.final_population for c in successful]),
                "failed_final": statistics.mean([c.final_population for c in failed])
            }
        }
    
    def _identify_key_differentiators(self, successful: List[ColonyOutcome], failed: List[ColonyOutcome]) -> List[str]:
        """Identify key differentiating factors"""
        
        differentiators = []
        
        # Health decline rates
        success_decline = statistics.mean([c.health_decline_rate for c in successful])
        failed_decline = statistics.mean([c.health_decline_rate for c in failed])
        
        if failed_decline < success_decline * 0.5:  # More negative decline in failed
            differentiators.append("Failed colonies show accelerated health decline")
        
        # Energy collection
        success_energy = statistics.mean([c.total_energy_collected for c in successful])
        failed_energy = statistics.mean([c.total_energy_collected for c in failed])
        
        if success_energy > failed_energy * 2:
            differentiators.append("Successful colonies collect significantly more energy")
        
        return differentiators
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of analysis"""
        
        if not self.simulation_metrics:
            return {"error": "Insufficient data for executive summary"}
        
        # Overall performance assessment
        if self.simulation_metrics.survival_rate >= 0.9:
            performance_grade = "Excellent"
        elif self.simulation_metrics.survival_rate >= 0.7:
            performance_grade = "Good"
        elif self.simulation_metrics.survival_rate >= 0.5:
            performance_grade = "Fair"
        else:
            performance_grade = "Poor"
        
        # Key findings
        key_findings = []
        
        key_findings.append(f"Colony survival rate: {self.simulation_metrics.survival_rate:.1%}")
        key_findings.append(f"Average health score: {self.simulation_metrics.avg_health_score:.2f}")
        key_findings.append(f"Foraging efficiency: {self.simulation_metrics.avg_foraging_efficiency:.2f}")
        
        if self.simulation_metrics.colonies_collapsed > 0:
            key_findings.append(f"{self.simulation_metrics.colonies_collapsed} colonies collapsed")
        
        # Top recommendations
        top_recommendations = []
        for insight in self.performance_insights[:3]:  # Top 3 insights
            top_recommendations.extend(insight.recommendations[:1])  # First recommendation from each
        
        return {
            "performance_grade": performance_grade,
            "simulation_duration": f"{self.simulation_metrics.duration_days} days",
            "key_findings": key_findings,
            "top_recommendations": top_recommendations,
            "critical_issues": len([i for i in self.performance_insights if i.priority == "high"]),
            "analysis_confidence": "High" if len(self.colony_outcomes) >= 3 else "Medium"
        }

def generate_comprehensive_report(analysis_results: Dict[str, Any], 
                                output_format: str = "json") -> Union[str, Dict[str, Any]]:
    """Generate comprehensive post-mortem report"""
    
    if output_format == "json":
        return analysis_results
    
    elif output_format == "markdown":
        return _generate_markdown_report(analysis_results)
    
    elif output_format == "html":
        return _generate_html_report(analysis_results)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def _generate_markdown_report(analysis_results: Dict[str, Any]) -> str:
    """Generate markdown format report"""
    
    report = []
    report.append("# BSTEW Simulation Post-Mortem Analysis Report\n")
    
    # Executive Summary
    exec_summary = analysis_results.get("executive_summary", {})
    report.append("## Executive Summary\n")
    report.append(f"**Performance Grade:** {exec_summary.get('performance_grade', 'N/A')}\n")
    report.append(f"**Simulation Duration:** {exec_summary.get('simulation_duration', 'N/A')}\n")
    
    if "key_findings" in exec_summary:
        report.append("### Key Findings\n")
        for finding in exec_summary["key_findings"]:
            report.append(f"- {finding}\n")
    
    if "top_recommendations" in exec_summary:
        report.append("### Top Recommendations\n")
        for rec in exec_summary["top_recommendations"]:
            report.append(f"- {rec}\n")
    
    # Simulation Metrics
    sim_metrics = analysis_results.get("simulation_metrics")
    if sim_metrics:
        report.append("\n## Simulation Metrics\n")
        report.append(f"- **Duration:** {sim_metrics.duration_days} days\n")
        report.append(f"- **Population Growth Rate:** {sim_metrics.population_growth_rate:.3f}\n")
        report.append(f"- **Survival Rate:** {sim_metrics.survival_rate:.1%}\n")
        report.append(f"- **Average Health Score:** {sim_metrics.avg_health_score:.2f}\n")
        report.append(f"- **Foraging Efficiency:** {sim_metrics.avg_foraging_efficiency:.2f}\n")
    
    # Performance Insights
    insights = analysis_results.get("performance_insights", [])
    if insights:
        report.append("\n## Performance Insights\n")
        for insight in insights:
            report.append(f"### {insight.title} ({insight.priority.upper()} Priority)\n")
            report.append(f"{insight.description}\n")
            report.append("**Recommendations:**\n")
            for rec in insight.recommendations:
                report.append(f"- {rec}\n")
            report.append(f"**Expected Impact:** {insight.expected_impact}\n\n")
    
    return "".join(report)

def _generate_html_report(analysis_results: Dict[str, Any]) -> str:
    """Generate HTML format report"""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BSTEW Post-Mortem Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .metric { background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }
            .insight { background-color: #fff3cd; padding: 15px; margin: 15px 0; border-radius: 5px; }
            .high-priority { border-left: 5px solid #dc3545; }
            .medium-priority { border-left: 5px solid #ffc107; }
            .low-priority { border-left: 5px solid #28a745; }
        </style>
    </head>
    <body>
    """
    
    # Add executive summary
    exec_summary = analysis_results.get("executive_summary", {})
    html += f"""
    <div class="header">
        <h1>BSTEW Simulation Post-Mortem Analysis</h1>
        <h2>Performance Grade: {exec_summary.get('performance_grade', 'N/A')}</h2>
        <p>Duration: {exec_summary.get('simulation_duration', 'N/A')}</p>
    </div>
    """
    
    # Add key findings
    if "key_findings" in exec_summary:
        html += "<h3>Key Findings</h3><ul>"
        for finding in exec_summary["key_findings"]:
            html += f"<li>{finding}</li>"
        html += "</ul>"
    
    # Add simulation metrics
    sim_metrics = analysis_results.get("simulation_metrics")
    if sim_metrics:
        html += """
        <h3>Simulation Metrics</h3>
        <div class="metric">
        """
        html += f"<p><strong>Survival Rate:</strong> {sim_metrics.survival_rate:.1%}</p>"
        html += f"<p><strong>Average Health Score:</strong> {sim_metrics.avg_health_score:.2f}</p>"
        html += f"<p><strong>Foraging Efficiency:</strong> {sim_metrics.avg_foraging_efficiency:.2f}</p>"
        html += "</div>"
    
    # Add performance insights
    insights = analysis_results.get("performance_insights", [])
    if insights:
        html += "<h3>Performance Insights</h3>"
        for insight in insights:
            priority_class = f"{insight.priority}-priority"
            html += f"""
            <div class="insight {priority_class}">
                <h4>{insight.title} ({insight.priority.upper()} Priority)</h4>
                <p>{insight.description}</p>
                <p><strong>Expected Impact:</strong> {insight.expected_impact}</p>
            </div>
            """
    
    html += "</body></html>"
    return html