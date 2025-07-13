"""
Dead Colony Tracking System for NetLogo BEE-STEWARD v2 Parity
============================================================

Comprehensive colony mortality tracking system monitoring colony health,
collapse conditions, survival factors, and population dynamics.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass
import time
from collections import defaultdict



class ColonyStatus(Enum):
    """Colony status levels"""
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DECLINING = "declining"
    CRITICAL = "critical"
    COLLAPSED = "collapsed"
    DEAD = "dead"


class CollapseReason(Enum):
    """Reasons for colony collapse"""
    QUEEN_DEATH = "queen_death"
    POPULATION_COLLAPSE = "population_collapse"
    RESOURCE_DEPLETION = "resource_depletion"
    DISEASE_OUTBREAK = "disease_outbreak"
    ENVIRONMENTAL_STRESS = "environmental_stress"
    GENETIC_BOTTLENECK = "genetic_bottleneck"
    FORAGING_FAILURE = "foraging_failure"
    ENERGY_EXHAUSTION = "energy_exhaustion"
    UNKNOWN = "unknown"


@dataclass
class ColonyHealthMetrics:
    """Comprehensive colony health metrics"""
    colony_id: int
    timestamp: float
    
    # Population metrics
    total_population: int
    queen_count: int
    worker_count: int
    drone_count: int
    brood_count: int
    
    # Age distribution
    average_age: float
    age_distribution: Dict[str, int]  # age_groups -> count
    
    # Resource metrics
    total_energy: float
    energy_reserves: float
    daily_energy_intake: float
    daily_energy_consumption: float
    energy_balance: float
    
    # Health indicators
    disease_prevalence: float
    average_stress_level: float
    mortality_rate: float
    birth_rate: float
    
    # Foraging metrics
    foraging_efficiency: float
    successful_foraging_trips: int
    failed_foraging_trips: int
    forager_count: int
    
    # Genetic metrics
    genetic_diversity: float
    inbreeding_coefficient: float
    
    # Environmental factors
    temperature_stress: float
    resource_availability: float
    competition_pressure: float
    
    def calculate_health_score(self) -> float:
        """Calculate overall colony health score (0-1)"""
        
        # Population health (0.3 weight)
        pop_health = min(1.0, self.total_population / 1000.0) * 0.3
        
        # Energy health (0.25 weight)
        energy_ratio = self.total_energy / max(1.0, self.total_population * 50.0)
        energy_health = min(1.0, energy_ratio) * 0.25
        
        # Foraging health (0.2 weight)
        if self.foraging_efficiency > 0:
            foraging_health = self.foraging_efficiency * 0.2
        else:
            foraging_health = 0.0
        
        # Genetic health (0.15 weight)
        genetic_health = self.genetic_diversity * 0.15
        
        # Disease health (0.1 weight)
        disease_health = (1.0 - self.disease_prevalence) * 0.1
        
        return pop_health + energy_health + foraging_health + genetic_health + disease_health


@dataclass
class CollapseEvent:
    """Record of colony collapse event"""
    colony_id: int
    collapse_timestamp: float
    detection_timestamp: float
    
    # Collapse characteristics
    primary_reason: CollapseReason
    contributing_factors: List[CollapseReason]
    collapse_duration: float  # Time from decline to collapse
    
    # Final state
    final_population: int
    final_energy: float
    final_health_score: float
    
    # Historical context
    peak_population: int
    peak_population_date: float
    days_since_peak: float
    
    # Environmental context
    environmental_conditions: Dict[str, float]
    resource_conditions: Dict[str, float]
    
    # Predictive factors
    warning_signs: List[str]
    prediction_accuracy: float = 0.0


class ColonyHealthMonitor(BaseModel):
    """Colony health monitoring system"""
    
    model_config = {"validate_assignment": True}
    
    # Health thresholds
    critical_population_threshold: int = Field(default=50, ge=0, description="Critical population threshold")
    energy_depletion_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Energy depletion threshold")
    disease_outbreak_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Disease outbreak threshold")
    
    # Monitoring parameters
    health_check_interval: float = Field(default=24.0, ge=0.0, description="Health check interval (hours)")
    historical_window: int = Field(default=30, ge=1, description="Historical data window (days)")
    
    # Alert thresholds
    declining_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Declining health threshold")
    critical_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Critical health threshold")
    
    def assess_colony_health(self, colony_data: Dict[str, Any]) -> ColonyHealthMetrics:
        """Assess current colony health"""
        
        # Extract basic metrics
        total_population = colony_data.get('total_population', 0)
        queen_count = colony_data.get('queen_count', 0)
        worker_count = colony_data.get('worker_count', 0)
        drone_count = colony_data.get('drone_count', 0)
        brood_count = colony_data.get('brood_count', 0)
        
        # Calculate age distribution
        age_distribution = self._calculate_age_distribution(colony_data.get('bee_ages', []))
        average_age = sum(age_distribution.keys()) / len(age_distribution) if age_distribution else 0.0
        
        # Resource metrics
        total_energy = colony_data.get('total_energy', 0.0)
        energy_reserves = colony_data.get('energy_reserves', 0.0)
        daily_intake = colony_data.get('daily_energy_intake', 0.0)
        daily_consumption = colony_data.get('daily_energy_consumption', 0.0)
        energy_balance = daily_intake - daily_consumption
        
        # Health indicators
        disease_prevalence = colony_data.get('disease_prevalence', 0.0)
        average_stress = colony_data.get('average_stress_level', 0.0)
        mortality_rate = colony_data.get('mortality_rate', 0.0)
        birth_rate = colony_data.get('birth_rate', 0.0)
        
        # Foraging metrics
        foraging_efficiency = colony_data.get('foraging_efficiency', 0.0)
        successful_trips = colony_data.get('successful_foraging_trips', 0)
        failed_trips = colony_data.get('failed_foraging_trips', 0)
        forager_count = colony_data.get('forager_count', 0)
        
        # Genetic metrics
        genetic_diversity = colony_data.get('genetic_diversity', 0.5)
        inbreeding_coeff = colony_data.get('inbreeding_coefficient', 0.0)
        
        # Environmental factors
        temp_stress = colony_data.get('temperature_stress', 0.0)
        resource_avail = colony_data.get('resource_availability', 1.0)
        competition = colony_data.get('competition_pressure', 0.0)
        
        return ColonyHealthMetrics(
            colony_id=colony_data.get('colony_id', 0),
            timestamp=time.time(),
            total_population=total_population,
            queen_count=queen_count,
            worker_count=worker_count,
            drone_count=drone_count,
            brood_count=brood_count,
            average_age=average_age,
            age_distribution=age_distribution,
            total_energy=total_energy,
            energy_reserves=energy_reserves,
            daily_energy_intake=daily_intake,
            daily_energy_consumption=daily_consumption,
            energy_balance=energy_balance,
            disease_prevalence=disease_prevalence,
            average_stress_level=average_stress,
            mortality_rate=mortality_rate,
            birth_rate=birth_rate,
            foraging_efficiency=foraging_efficiency,
            successful_foraging_trips=successful_trips,
            failed_foraging_trips=failed_trips,
            forager_count=forager_count,
            genetic_diversity=genetic_diversity,
            inbreeding_coefficient=inbreeding_coeff,
            temperature_stress=temp_stress,
            resource_availability=resource_avail,
            competition_pressure=competition
        )
    
    def _calculate_age_distribution(self, bee_ages: List[int]) -> Dict[str, int]:
        """Calculate age distribution by groups"""
        
        age_groups = {
            "young": 0,      # 0-50 days
            "adult": 0,      # 51-200 days
            "mature": 0,     # 201-400 days
            "old": 0         # 400+ days
        }
        
        for age in bee_ages:
            if age <= 50:
                age_groups["young"] += 1
            elif age <= 200:
                age_groups["adult"] += 1
            elif age <= 400:
                age_groups["mature"] += 1
            else:
                age_groups["old"] += 1
        
        return age_groups
    
    def determine_colony_status(self, health_metrics: ColonyHealthMetrics) -> ColonyStatus:
        """Determine colony status based on health metrics"""
        
        health_score = health_metrics.calculate_health_score()
        
        # Critical conditions
        if health_metrics.total_population <= self.critical_population_threshold:
            return ColonyStatus.CRITICAL
        
        if health_metrics.queen_count == 0:
            return ColonyStatus.CRITICAL
        
        if health_metrics.total_energy / max(1.0, health_metrics.total_population) < 10.0:
            return ColonyStatus.CRITICAL
        
        # Status based on health score
        if health_score >= 0.8:
            return ColonyStatus.HEALTHY
        elif health_score >= self.declining_threshold:
            return ColonyStatus.STRESSED
        elif health_score >= self.critical_threshold:
            return ColonyStatus.DECLINING
        else:
            return ColonyStatus.CRITICAL
    
    def identify_warning_signs(self, health_metrics: ColonyHealthMetrics,
                             historical_data: List[ColonyHealthMetrics]) -> List[str]:
        """Identify warning signs of potential colony collapse"""
        
        warning_signs = []
        
        # Population decline
        if len(historical_data) >= 7:
            recent_populations = [m.total_population for m in historical_data[-7:]]
            if all(recent_populations[i] <= recent_populations[i-1] for i in range(1, len(recent_populations))):
                warning_signs.append("consistent_population_decline")
        
        # Queen issues
        if health_metrics.queen_count == 0:
            warning_signs.append("queenless_colony")
        elif health_metrics.queen_count > 1:
            warning_signs.append("multiple_queens")
        
        # Energy depletion
        if health_metrics.energy_balance < 0:
            warning_signs.append("negative_energy_balance")
        
        energy_per_bee = health_metrics.total_energy / max(1.0, health_metrics.total_population)
        if energy_per_bee < 20.0:
            warning_signs.append("low_energy_per_bee")
        
        # Foraging problems
        if health_metrics.foraging_efficiency < 0.3:
            warning_signs.append("poor_foraging_efficiency")
        
        if health_metrics.forager_count / max(1.0, health_metrics.total_population) < 0.2:
            warning_signs.append("insufficient_foragers")
        
        # Disease outbreak
        if health_metrics.disease_prevalence > self.disease_outbreak_threshold:
            warning_signs.append("disease_outbreak")
        
        # High stress
        if health_metrics.average_stress_level > 0.7:
            warning_signs.append("high_colony_stress")
        
        # High mortality
        if health_metrics.mortality_rate > 0.1:
            warning_signs.append("high_mortality_rate")
        
        # Genetic issues
        if health_metrics.genetic_diversity < 0.3:
            warning_signs.append("low_genetic_diversity")
        
        if health_metrics.inbreeding_coefficient > 0.3:
            warning_signs.append("high_inbreeding")
        
        # Age distribution problems
        if health_metrics.age_distribution.get("young", 0) / max(1.0, health_metrics.total_population) < 0.2:
            warning_signs.append("lack_of_young_bees")
        
        return warning_signs


class CollapsePredictor(BaseModel):
    """System for predicting colony collapse"""
    
    model_config = {"validate_assignment": True}
    
    # Prediction parameters
    prediction_window: int = Field(default=14, ge=1, description="Prediction window (days)")
    minimum_data_points: int = Field(default=7, ge=1, description="Minimum data points for prediction")
    
    # Model weights
    population_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Population trend weight")
    energy_weight: float = Field(default=0.25, ge=0.0, le=1.0, description="Energy trend weight")
    health_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Health trend weight")
    foraging_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Foraging trend weight")
    genetic_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Genetic trend weight")
    
    def predict_collapse_risk(self, historical_data: List[ColonyHealthMetrics]) -> Dict[str, Any]:
        """Predict collapse risk based on historical data"""
        
        if len(historical_data) < self.minimum_data_points:
            return {
                "collapse_risk": 0.0,
                "confidence": 0.0,
                "predicted_days_to_collapse": None,
                "risk_factors": [],
                "insufficient_data": True
            }
        
        # Calculate trends
        population_trend = self._calculate_trend([m.total_population for m in historical_data])
        energy_trend = self._calculate_trend([m.total_energy for m in historical_data])
        health_trend = self._calculate_trend([m.calculate_health_score() for m in historical_data])
        foraging_trend = self._calculate_trend([m.foraging_efficiency for m in historical_data])
        genetic_trend = self._calculate_trend([m.genetic_diversity for m in historical_data])
        
        # Calculate risk scores
        pop_risk = max(0.0, -population_trend) * self.population_weight
        energy_risk = max(0.0, -energy_trend) * self.energy_weight
        health_risk = max(0.0, -health_trend) * self.health_weight
        foraging_risk = max(0.0, -foraging_trend) * self.foraging_weight
        genetic_risk = max(0.0, -genetic_trend) * self.genetic_weight
        
        # Combined risk
        collapse_risk = pop_risk + energy_risk + health_risk + foraging_risk + genetic_risk
        
        # Identify risk factors
        risk_factors = []
        if pop_risk > 0.1:
            risk_factors.append("population_decline")
        if energy_risk > 0.1:
            risk_factors.append("energy_depletion")
        if health_risk > 0.1:
            risk_factors.append("health_deterioration")
        if foraging_risk > 0.1:
            risk_factors.append("foraging_decline")
        if genetic_risk > 0.1:
            risk_factors.append("genetic_decline")
        
        # Predict days to collapse
        predicted_days = self._predict_collapse_timeline(historical_data, collapse_risk)
        
        # Calculate confidence
        confidence = min(1.0, len(historical_data) / (self.minimum_data_points * 2))
        
        return {
            "collapse_risk": collapse_risk,
            "confidence": confidence,
            "predicted_days_to_collapse": predicted_days,
            "risk_factors": risk_factors,
            "insufficient_data": False,
            "trend_analysis": {
                "population_trend": population_trend,
                "energy_trend": energy_trend,
                "health_trend": health_trend,
                "foraging_trend": foraging_trend,
                "genetic_trend": genetic_trend
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and strength"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize slope
        value_range = max(values) - min(values)
        if value_range == 0:
            return 0.0
        
        normalized_slope = slope / value_range
        
        return normalized_slope
    
    def _predict_collapse_timeline(self, historical_data: List[ColonyHealthMetrics], 
                                 collapse_risk: float) -> Optional[int]:
        """Predict timeline to collapse"""
        
        if collapse_risk < 0.3:
            return None
        
        # Get current population and trend
        current_population = historical_data[-1].total_population
        population_trend = self._calculate_trend([m.total_population for m in historical_data])
        
        if population_trend >= 0:
            return None  # Population not declining
        
        # Estimate days to reach critical threshold
        critical_threshold = 50  # Critical population threshold
        
        if current_population <= critical_threshold:
            return 0  # Already critical
        
        # Calculate decline rate (bees per day)
        decline_rate = -population_trend * current_population
        
        if decline_rate <= 0:
            return None
        
        # Calculate days to critical threshold
        days_to_critical = (current_population - critical_threshold) / decline_rate
        
        return max(1, int(days_to_critical))


class DeadColonyTracker(BaseModel):
    """Comprehensive dead colony tracking system"""
    
    model_config = {"validate_assignment": True}
    
    # Component systems
    health_monitor: ColonyHealthMonitor = Field(default_factory=ColonyHealthMonitor)
    collapse_predictor: CollapsePredictor = Field(default_factory=CollapsePredictor)
    
    # Tracking data
    colony_health_history: Dict[int, List[ColonyHealthMetrics]] = Field(default_factory=dict)
    colony_status_history: Dict[int, List[Tuple[float, ColonyStatus]]] = Field(default_factory=dict)
    collapse_events: List[CollapseEvent] = Field(default_factory=list)
    
    # Statistics
    total_colonies_tracked: int = Field(default=0, description="Total colonies ever tracked")
    active_colonies: Set[int] = Field(default_factory=set, description="Currently active colonies")
    dead_colonies: Set[int] = Field(default_factory=set, description="Dead colonies")
    
    def register_colony(self, colony_id: int) -> None:
        """Register new colony for tracking"""
        
        if colony_id not in self.colony_health_history:
            self.colony_health_history[colony_id] = []
            self.colony_status_history[colony_id] = []
            self.active_colonies.add(colony_id)
            self.total_colonies_tracked += 1
    
    def update_colony_health(self, colony_id: int, colony_data: Dict[str, Any], simulation_time: float = 0.0) -> ColonyStatus:
        """Update colony health metrics and status"""
        
        # Register colony if not already tracked
        if colony_id not in self.colony_health_history:
            self.register_colony(colony_id)
        
        # Assess health
        health_metrics = self.health_monitor.assess_colony_health(colony_data)
        
        # Store health data
        self.colony_health_history[colony_id].append(health_metrics)
        
        # Limit history size
        if len(self.colony_health_history[colony_id]) > 365:  # Keep 1 year of data
            self.colony_health_history[colony_id].pop(0)
        
        # Determine status
        colony_status = self.health_monitor.determine_colony_status(health_metrics)
        
        # Store status history
        self.colony_status_history[colony_id].append((time.time(), colony_status))
        
        # Limit status history size
        if len(self.colony_status_history[colony_id]) > 365:
            self.colony_status_history[colony_id].pop(0)
        
        # Check for collapse
        if colony_status == ColonyStatus.COLLAPSED and colony_id in self.active_colonies:
            self._record_colony_collapse(colony_id, health_metrics)
        
        return colony_status
    
    def _record_colony_collapse(self, colony_id: int, final_health: ColonyHealthMetrics) -> None:
        """Record colony collapse event"""
        
        # Remove from active colonies
        if colony_id in self.active_colonies:
            self.active_colonies.remove(colony_id)
        self.dead_colonies.add(colony_id)
        
        # Analyze collapse
        historical_data = self.colony_health_history[colony_id]
        status_history = self.colony_status_history[colony_id]
        
        # Determine primary collapse reason
        primary_reason = self._determine_collapse_reason(final_health, historical_data)
        
        # Find contributing factors
        contributing_factors = self._identify_contributing_factors(historical_data)
        
        # Calculate collapse duration
        collapse_duration = self._calculate_collapse_duration(status_history)
        
        # Find peak population
        peak_population = max(m.total_population for m in historical_data)
        peak_date = next(m.timestamp for m in historical_data if m.total_population == peak_population)
        days_since_peak = (time.time() - peak_date) / (24 * 3600)
        
        # Get warning signs
        warning_signs = self.health_monitor.identify_warning_signs(final_health, historical_data)
        
        # Create collapse event
        collapse_event = CollapseEvent(
            colony_id=colony_id,
            collapse_timestamp=time.time(),
            detection_timestamp=time.time(),
            primary_reason=primary_reason,
            contributing_factors=contributing_factors,
            collapse_duration=collapse_duration,
            final_population=final_health.total_population,
            final_energy=final_health.total_energy,
            final_health_score=final_health.calculate_health_score(),
            peak_population=peak_population,
            peak_population_date=peak_date,
            days_since_peak=days_since_peak,
            environmental_conditions={
                "temperature_stress": final_health.temperature_stress,
                "resource_availability": final_health.resource_availability,
                "competition_pressure": final_health.competition_pressure
            },
            resource_conditions={
                "total_energy": final_health.total_energy,
                "energy_balance": final_health.energy_balance,
                "foraging_efficiency": final_health.foraging_efficiency
            },
            warning_signs=warning_signs
        )
        
        self.collapse_events.append(collapse_event)
    
    def _determine_collapse_reason(self, final_health: ColonyHealthMetrics,
                                 historical_data: List[ColonyHealthMetrics]) -> CollapseReason:
        """Determine primary reason for colony collapse"""
        
        # Queen death
        if final_health.queen_count == 0:
            return CollapseReason.QUEEN_DEATH
        
        # Population collapse
        if final_health.total_population < 50:
            return CollapseReason.POPULATION_COLLAPSE
        
        # Energy exhaustion
        if final_health.total_energy < 100:
            return CollapseReason.ENERGY_EXHAUSTION
        
        # Disease outbreak
        if final_health.disease_prevalence > 0.5:
            return CollapseReason.DISEASE_OUTBREAK
        
        # Foraging failure
        if final_health.foraging_efficiency < 0.2:
            return CollapseReason.FORAGING_FAILURE
        
        # Genetic bottleneck
        if final_health.genetic_diversity < 0.2:
            return CollapseReason.GENETIC_BOTTLENECK
        
        # Environmental stress
        if final_health.temperature_stress > 0.7 or final_health.resource_availability < 0.3:
            return CollapseReason.ENVIRONMENTAL_STRESS
        
        return CollapseReason.UNKNOWN
    
    def _identify_contributing_factors(self, historical_data: List[ColonyHealthMetrics]) -> List[CollapseReason]:
        """Identify contributing factors to collapse"""
        
        factors = []
        
        if len(historical_data) < 7:
            return factors
        
        recent_data = historical_data[-7:]
        
        # Check for declining trends
        population_trend = self.collapse_predictor._calculate_trend([m.total_population for m in recent_data])
        if population_trend < -0.1:
            factors.append(CollapseReason.POPULATION_COLLAPSE)
        
        energy_trend = self.collapse_predictor._calculate_trend([m.total_energy for m in recent_data])
        if energy_trend < -0.1:
            factors.append(CollapseReason.RESOURCE_DEPLETION)
        
        foraging_trend = self.collapse_predictor._calculate_trend([m.foraging_efficiency for m in recent_data])
        if foraging_trend < -0.1:
            factors.append(CollapseReason.FORAGING_FAILURE)
        
        # Check for persistent problems
        avg_disease = sum(m.disease_prevalence for m in recent_data) / len(recent_data)
        if avg_disease > 0.3:
            factors.append(CollapseReason.DISEASE_OUTBREAK)
        
        avg_temp_stress = sum(m.temperature_stress for m in recent_data) / len(recent_data)
        if avg_temp_stress > 0.5:
            factors.append(CollapseReason.ENVIRONMENTAL_STRESS)
        
        return factors
    
    def _calculate_collapse_duration(self, status_history: List[Tuple[float, ColonyStatus]]) -> float:
        """Calculate duration from decline to collapse"""
        
        # Find first declining status
        decline_start = None
        for timestamp, status in status_history:
            if status in [ColonyStatus.DECLINING, ColonyStatus.CRITICAL]:
                decline_start = timestamp
                break
        
        if decline_start is None:
            return 0.0
        
        # Calculate duration to collapse
        collapse_time = time.time()
        duration = (collapse_time - decline_start) / (24 * 3600)  # Convert to days
        
        return duration
    
    def get_colony_collapse_prediction(self, colony_id: int) -> Dict[str, Any]:
        """Get collapse prediction for a colony"""
        
        if colony_id not in self.colony_health_history:
            return {"error": "Colony not tracked"}
        
        historical_data = self.colony_health_history[colony_id]
        
        return self.collapse_predictor.predict_collapse_risk(historical_data)
    
    def get_mortality_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mortality statistics"""
        
        if not self.collapse_events:
            return {
                "total_colonies": self.total_colonies_tracked,
                "active_colonies": len(self.active_colonies),
                "dead_colonies": 0,
                "survival_rate": 1.0 if self.total_colonies_tracked > 0 else 0.0
            }
        
        # Basic statistics
        total_dead = len(self.collapse_events)
        survival_rate = (self.total_colonies_tracked - total_dead) / self.total_colonies_tracked
        
        # Collapse reasons
        reason_counts = defaultdict(int)
        for event in self.collapse_events:
            reason_counts[event.primary_reason.value] += 1
        
        # Average metrics
        avg_lifespan = sum(event.days_since_peak for event in self.collapse_events) / len(self.collapse_events)
        avg_collapse_duration = sum(event.collapse_duration for event in self.collapse_events) / len(self.collapse_events)
        avg_peak_population = sum(event.peak_population for event in self.collapse_events) / len(self.collapse_events)
        
        return {
            "total_colonies": self.total_colonies_tracked,
            "active_colonies": len(self.active_colonies),
            "dead_colonies": total_dead,
            "survival_rate": survival_rate,
            "collapse_reasons": dict(reason_counts),
            "average_lifespan_days": avg_lifespan,
            "average_collapse_duration_days": avg_collapse_duration,
            "average_peak_population": avg_peak_population,
            "most_common_collapse_reason": max(reason_counts.items(), key=lambda x: x[1])[0] if reason_counts else None
        }
    
    def generate_survival_report(self) -> str:
        """Generate survival report for system integration (fixes missing method error)"""
        
        try:
            import json
            from pathlib import Path
            
            # Get mortality statistics
            stats = self.get_mortality_statistics()
            
            # Create report data
            report_data = {
                "generated_at": time.time(),
                "summary": stats,
                "colony_details": {},
                "collapse_events": []
            }
            
            # Add colony-specific details
            for colony_id in self.active_colonies.union(self.dead_colonies):
                if colony_id in self.colony_health_history:
                    history = self.colony_health_history[colony_id]
                    
                    if history:
                        latest = history[-1]
                        report_data["colony_details"][colony_id] = {
                            "status": "dead" if colony_id in self.dead_colonies else "active",
                            "latest_population": latest.total_population,
                            "latest_health_score": latest.health_score,
                            "data_points": len(history),
                            "tracking_duration": (latest.timestamp - history[0].timestamp) / (24 * 3600) if len(history) > 1 else 0
                        }
            
            # Add collapse events
            for event in self.collapse_events:
                report_data["collapse_events"].append({
                    "colony_id": event.colony_id,
                    "collapse_time": event.collapse_time,
                    "primary_reason": event.primary_reason.value,
                    "contributing_factors": [f.value for f in event.contributing_factors],
                    "peak_population": event.peak_population,
                    "final_population": event.final_population,
                    "days_since_peak": event.days_since_peak,
                    "collapse_duration": event.collapse_duration
                })
            
            # Save report to file
            output_path = Path("artifacts/survival_report.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            return str(output_path)
            
        except Exception as e:
            # Return error path to avoid breaking system integration
            return f"error_generating_survival_report: {e}"