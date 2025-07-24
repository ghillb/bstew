"""
Colony Health Monitoring System for NetLogo BEE-STEWARD v2 Parity
================================================================

Real-time colony health monitoring system that tracks vital signs, analyzes
health trends, and provides early warning systems for colony decline.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import logging
import time
import statistics
import numpy as np
from collections import defaultdict, deque


class HealthStatus(Enum):
    """Colony health status levels"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    DECLINING = "declining"
    COLLAPSED = "collapsed"


class HealthIndicator(Enum):
    """Individual health indicators"""

    POPULATION = "population"
    MORTALITY = "mortality"
    REPRODUCTION = "reproduction"
    NUTRITION = "nutrition"
    DISEASE = "disease"
    FORAGING = "foraging"
    STRESS = "stress"
    ENERGY = "energy"
    DEVELOPMENT = "development"
    BEHAVIORAL = "behavioral"


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthMetric:
    """Individual health metric measurement"""

    indicator: HealthIndicator
    value: float
    timestamp: float
    status: HealthStatus
    trend: str  # "improving", "stable", "declining"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthAlert:
    """Health monitoring alert"""

    alert_id: str
    level: AlertLevel
    indicator: HealthIndicator
    message: str
    timestamp: float
    colony_id: int
    current_value: float
    threshold_value: float
    recommendation: str = ""
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class HealthTrend:
    """Health trend analysis"""

    indicator: HealthIndicator
    trend_direction: str  # "improving", "stable", "declining"
    trend_strength: float  # 0-1
    slope: float
    r_squared: float
    prediction_horizon: int = 10  # Steps ahead
    predicted_values: List[float] = field(default_factory=list)


class ColonyHealthProfile(BaseModel):
    """Comprehensive colony health profile"""

    model_config = {"validate_assignment": True}

    colony_id: int
    last_update: float = 0.0

    # Current health metrics
    current_metrics: Dict[HealthIndicator, HealthMetric] = Field(default_factory=dict)
    overall_health_score: float = 0.0
    health_status: HealthStatus = HealthStatus.GOOD

    # Historical data (sliding windows)
    metric_history: Dict[HealthIndicator, deque] = Field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=100))
    )
    health_score_history: deque = Field(default_factory=lambda: deque(maxlen=100))

    # Trend analysis
    health_trends: Dict[HealthIndicator, HealthTrend] = Field(default_factory=dict)

    # Alert tracking
    active_alerts: List[HealthAlert] = Field(default_factory=list)
    alert_history: List[HealthAlert] = Field(default_factory=list)

    # Baseline values for comparison
    baseline_metrics: Dict[HealthIndicator, float] = Field(default_factory=dict)

    @property
    def health_grade(self) -> str:
        """Letter grade for health status"""
        if self.overall_health_score >= 0.9:
            return "A"
        elif self.overall_health_score >= 0.8:
            return "B"
        elif self.overall_health_score >= 0.7:
            return "C"
        elif self.overall_health_score >= 0.6:
            return "D"
        else:
            return "F"

    @property
    def critical_indicators(self) -> List[HealthIndicator]:
        """List of indicators in critical status"""
        return [
            indicator
            for indicator, metric in self.current_metrics.items()
            if metric.status in [HealthStatus.CRITICAL, HealthStatus.POOR]
        ]

    @property
    def declining_indicators(self) -> List[HealthIndicator]:
        """List of indicators showing declining trends"""
        return [
            indicator
            for indicator, trend in self.health_trends.items()
            if trend.trend_direction == "declining" and trend.trend_strength > 0.5
        ]


class HealthThresholds(BaseModel):
    """Health monitoring thresholds and limits"""

    model_config = {"validate_assignment": True}

    # Population thresholds
    min_population: int = 10
    max_population: int = 10000
    population_decline_rate: float = 0.05  # 5% decline triggers warning

    # Mortality thresholds
    max_daily_mortality_rate: float = 0.02  # 2% daily mortality
    max_weekly_mortality_rate: float = 0.1  # 10% weekly mortality

    # Nutrition thresholds (reduced for realistic startup conditions)
    min_energy_per_bee: float = 25.0  # Reduced from 50.0
    min_food_stores: float = 100.0
    nutrition_stress_threshold: float = 0.3

    # Foraging thresholds
    min_foraging_efficiency: float = 0.3
    max_foraging_failure_rate: float = 0.5
    foraging_distance_warning: float = 2000.0  # meters

    # Disease thresholds
    max_disease_prevalence: float = 0.1  # 10% infection rate
    disease_spread_rate_warning: float = 0.05

    # Behavioral thresholds (reduced for realistic startup conditions)
    min_activity_level: float = 0.2  # Reduced from 0.4
    stress_indicator_threshold: float = 0.7
    behavioral_anomaly_threshold: float = 0.6

    # Environmental stress thresholds
    temperature_stress_min: float = 5.0  # °C
    temperature_stress_max: float = 35.0  # °C
    weather_impact_threshold: float = 0.3

    # Colony startup grace period
    startup_grace_period_days: float = 7.0  # Grace period before strict monitoring


class HealthMonitoringSystem(BaseModel):
    """Real-time colony health monitoring system"""

    model_config = {"validate_assignment": True}

    # Colony profiles
    colony_profiles: Dict[int, ColonyHealthProfile] = Field(default_factory=dict)

    # System configuration
    thresholds: HealthThresholds = Field(default_factory=HealthThresholds)
    monitoring_enabled: bool = True
    update_frequency: int = 1  # Update every N simulation steps

    # Alert management
    alert_queue: deque = Field(default_factory=lambda: deque(maxlen=1000))
    alert_callbacks: List[Any] = Field(default_factory=list)

    # Alert debouncing system
    alert_debounce_cache: Dict[str, float] = Field(default_factory=dict)
    alert_cooldown_periods: Dict[AlertLevel, float] = Field(
        default_factory=lambda: {
            AlertLevel.INFO: 10.0,  # 10 steps cooldown for info
            AlertLevel.WARNING: 5.0,  # 5 steps cooldown for warnings
            AlertLevel.CRITICAL: 3.0,  # 3 steps cooldown for critical
            AlertLevel.EMERGENCY: 1.0,  # 1 step cooldown for emergency
        }
    )
    alert_rate_limits: Dict[str, List[float]] = Field(default_factory=dict)
    max_alerts_per_minute: int = 10

    # System metrics
    system_health_score: float = 0.0
    monitoring_statistics: Dict[str, Any] = Field(default_factory=dict)

    # Logging
    logger: Any = Field(
        default_factory=lambda: logging.getLogger(__name__), exclude=True
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    def register_colony(
        self, colony_id: int, initial_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a colony for health monitoring"""

        if self.logger:
            self.logger.info(f"Registering colony {colony_id} for health monitoring")

        profile = ColonyHealthProfile(colony_id=colony_id)

        if initial_data:
            # Set baseline metrics from initial data
            self._establish_baseline_metrics(profile, initial_data)

        self.colony_profiles[colony_id] = profile

        if self.logger:
            self.logger.info(f"Colony {colony_id} registered with health monitoring")

    def update_colony_health(
        self, colony_id: int, colony_data: Dict[str, Any], simulation_time: float
    ) -> ColonyHealthProfile:
        """Update health metrics for a colony"""

        if colony_id not in self.colony_profiles:
            self.register_colony(colony_id, colony_data)

        profile = self.colony_profiles[colony_id]
        profile.last_update = simulation_time

        # Calculate current health metrics
        new_metrics = self._calculate_health_metrics(colony_data, simulation_time)

        # Update profile with new metrics
        for indicator, metric in new_metrics.items():
            profile.current_metrics[indicator] = metric
            profile.metric_history[indicator].append(metric)

        # Calculate overall health score
        profile.overall_health_score = self._calculate_overall_health_score(profile)
        profile.health_score_history.append(profile.overall_health_score)

        # Determine health status
        profile.health_status = self._determine_health_status(
            profile.overall_health_score
        )

        # Update trend analysis
        self._update_health_trends(profile)

        # Check for alerts
        new_alerts = self._check_health_alerts(profile, simulation_time)

        # Process alerts
        for alert in new_alerts:
            self._process_health_alert(alert)
            profile.active_alerts.append(alert)
            self.alert_queue.append(alert)

        # Clean up resolved alerts
        self._cleanup_resolved_alerts(profile)

        return profile

    def _establish_baseline_metrics(
        self, profile: ColonyHealthProfile, initial_data: Dict[str, Any]
    ) -> None:
        """Establish baseline metrics for comparison"""

        # Population baseline
        population = initial_data.get(
            "adult_population", initial_data.get("population", 50)
        )
        profile.baseline_metrics[HealthIndicator.POPULATION] = population

        # Energy baseline
        total_energy = initial_data.get("total_energy", population * 100)
        profile.baseline_metrics[HealthIndicator.ENERGY] = total_energy / population

        # Foraging baseline
        profile.baseline_metrics[HealthIndicator.FORAGING] = (
            0.8  # Assume 80% efficiency baseline
        )

        # Other baselines
        profile.baseline_metrics[HealthIndicator.MORTALITY] = (
            0.01  # 1% daily mortality baseline
        )
        profile.baseline_metrics[HealthIndicator.STRESS] = 0.2  # 20% stress baseline
        profile.baseline_metrics[HealthIndicator.DISEASE] = 0.05  # 5% disease baseline
        profile.baseline_metrics[HealthIndicator.NUTRITION] = 0.8  # 80% nutrition score

        if self.logger:
            self.logger.info(
                f"Established baseline metrics for colony {profile.colony_id}"
            )

    def _calculate_health_metrics(
        self, colony_data: Dict[str, Any], timestamp: float
    ) -> Dict[HealthIndicator, HealthMetric]:
        """Calculate current health metrics from colony data"""

        metrics = {}

        # Population metrics
        population = colony_data.get(
            "adult_population", colony_data.get("population", 0)
        )
        pop_metric = HealthMetric(
            indicator=HealthIndicator.POPULATION,
            value=population,
            timestamp=timestamp,
            status=self._assess_population_health(population),
            trend="stable",  # Will be calculated later
        )
        metrics[HealthIndicator.POPULATION] = pop_metric

        # Mortality metrics
        mortality_rate = colony_data.get(
            "mortality_rate", colony_data.get("daily_mortality", 0.0)
        )
        mortality_metric = HealthMetric(
            indicator=HealthIndicator.MORTALITY,
            value=mortality_rate,
            timestamp=timestamp,
            status=self._assess_mortality_health(mortality_rate),
            trend="stable",
        )
        metrics[HealthIndicator.MORTALITY] = mortality_metric

        # Energy/nutrition metrics
        total_energy = colony_data.get("total_energy", 0.0)
        energy_per_bee = total_energy / max(1, population)
        energy_metric = HealthMetric(
            indicator=HealthIndicator.ENERGY,
            value=energy_per_bee,
            timestamp=timestamp,
            status=self._assess_energy_health(energy_per_bee),
            trend="stable",
        )
        metrics[HealthIndicator.ENERGY] = energy_metric

        # Foraging metrics
        foraging_efficiency = colony_data.get(
            "foraging_efficiency", colony_data.get("avg_foraging_efficiency", 0.5)
        )
        foraging_metric = HealthMetric(
            indicator=HealthIndicator.FORAGING,
            value=foraging_efficiency,
            timestamp=timestamp,
            status=self._assess_foraging_health(foraging_efficiency),
            trend="stable",
        )
        metrics[HealthIndicator.FORAGING] = foraging_metric

        # Disease metrics
        disease_prevalence = colony_data.get(
            "disease_prevalence", colony_data.get("infection_rate", 0.0)
        )
        disease_metric = HealthMetric(
            indicator=HealthIndicator.DISEASE,
            value=disease_prevalence,
            timestamp=timestamp,
            status=self._assess_disease_health(disease_prevalence),
            trend="stable",
        )
        metrics[HealthIndicator.DISEASE] = disease_metric

        # Stress metrics
        stress_level = self._calculate_stress_level(colony_data)
        stress_metric = HealthMetric(
            indicator=HealthIndicator.STRESS,
            value=stress_level,
            timestamp=timestamp,
            status=self._assess_stress_health(stress_level),
            trend="stable",
        )
        metrics[HealthIndicator.STRESS] = stress_metric

        # Reproduction metrics
        reproduction_rate = colony_data.get(
            "reproduction_rate", colony_data.get("egg_laying_rate", 0.0)
        )
        reproduction_metric = HealthMetric(
            indicator=HealthIndicator.REPRODUCTION,
            value=reproduction_rate,
            timestamp=timestamp,
            status=self._assess_reproduction_health(reproduction_rate),
            trend="stable",
        )
        metrics[HealthIndicator.REPRODUCTION] = reproduction_metric

        # Behavioral metrics
        activity_level = colony_data.get("activity_level", 0.8)
        behavioral_metric = HealthMetric(
            indicator=HealthIndicator.BEHAVIORAL,
            value=activity_level,
            timestamp=timestamp,
            status=self._assess_behavioral_health(activity_level),
            trend="stable",
        )
        metrics[HealthIndicator.BEHAVIORAL] = behavioral_metric

        return metrics

    def _calculate_stress_level(self, colony_data: Dict[str, Any]) -> float:
        """Calculate overall colony stress level"""

        stress_factors = []

        # Resource stress
        resource_scarcity = colony_data.get("resource_scarcity", 0.0)
        stress_factors.append(resource_scarcity)

        # Environmental stress
        temperature = colony_data.get("temperature", 20.0)
        temp_stress = 0.0
        if temperature < self.thresholds.temperature_stress_min:
            temp_stress = (self.thresholds.temperature_stress_min - temperature) / 10.0
        elif temperature > self.thresholds.temperature_stress_max:
            temp_stress = (temperature - self.thresholds.temperature_stress_max) / 10.0
        stress_factors.append(min(1.0, temp_stress))

        # Population stress
        population = colony_data.get("adult_population", 50)
        if population < self.thresholds.min_population:
            pop_stress = 1.0 - (population / self.thresholds.min_population)
            stress_factors.append(pop_stress)

        # Foraging stress
        foraging_efficiency = colony_data.get("foraging_efficiency", 0.8)
        if foraging_efficiency < self.thresholds.min_foraging_efficiency:
            forage_stress = 1.0 - (
                foraging_efficiency / self.thresholds.min_foraging_efficiency
            )
            stress_factors.append(forage_stress)

        # Disease stress
        disease_prevalence = colony_data.get("disease_prevalence", 0.0)
        if disease_prevalence > self.thresholds.max_disease_prevalence:
            disease_stress = disease_prevalence / self.thresholds.max_disease_prevalence
            stress_factors.append(min(1.0, disease_stress))

        # Calculate weighted average stress
        if stress_factors:
            mean_stress: float = float(statistics.mean(stress_factors))
            return min(1.0, mean_stress)
        else:
            return 0.0

    def _assess_population_health(self, population: int) -> HealthStatus:
        """Assess health status based on population"""
        if population >= self.thresholds.min_population * 5:
            return HealthStatus.EXCELLENT
        elif population >= self.thresholds.min_population * 3:
            return HealthStatus.GOOD
        elif population >= self.thresholds.min_population * 2:
            return HealthStatus.FAIR
        elif population >= self.thresholds.min_population:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _assess_mortality_health(self, mortality_rate: float) -> HealthStatus:
        """Assess health status based on mortality rate"""
        if mortality_rate <= self.thresholds.max_daily_mortality_rate * 0.5:
            return HealthStatus.EXCELLENT
        elif mortality_rate <= self.thresholds.max_daily_mortality_rate:
            return HealthStatus.GOOD
        elif mortality_rate <= self.thresholds.max_daily_mortality_rate * 1.5:
            return HealthStatus.FAIR
        elif mortality_rate <= self.thresholds.max_daily_mortality_rate * 2:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _assess_energy_health(self, energy_per_bee: float) -> HealthStatus:
        """Assess health status based on energy per bee"""
        if energy_per_bee >= self.thresholds.min_energy_per_bee * 2:
            return HealthStatus.EXCELLENT
        elif energy_per_bee >= self.thresholds.min_energy_per_bee * 1.5:
            return HealthStatus.GOOD
        elif energy_per_bee >= self.thresholds.min_energy_per_bee:
            return HealthStatus.FAIR
        elif energy_per_bee >= self.thresholds.min_energy_per_bee * 0.7:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _assess_foraging_health(self, efficiency: float) -> HealthStatus:
        """Assess health status based on foraging efficiency"""
        if efficiency >= 0.9:
            return HealthStatus.EXCELLENT
        elif efficiency >= 0.7:
            return HealthStatus.GOOD
        elif efficiency >= 0.5:
            return HealthStatus.FAIR
        elif efficiency >= 0.3:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _assess_disease_health(self, disease_prevalence: float) -> HealthStatus:
        """Assess health status based on disease prevalence"""
        if disease_prevalence <= 0.02:
            return HealthStatus.EXCELLENT
        elif disease_prevalence <= 0.05:
            return HealthStatus.GOOD
        elif disease_prevalence <= 0.1:
            return HealthStatus.FAIR
        elif disease_prevalence <= 0.2:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _assess_stress_health(self, stress_level: float) -> HealthStatus:
        """Assess health status based on stress level"""
        if stress_level <= 0.2:
            return HealthStatus.EXCELLENT
        elif stress_level <= 0.4:
            return HealthStatus.GOOD
        elif stress_level <= 0.6:
            return HealthStatus.FAIR
        elif stress_level <= 0.8:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _assess_reproduction_health(self, reproduction_rate: float) -> HealthStatus:
        """Assess health status based on reproduction rate"""
        if reproduction_rate >= 0.8:
            return HealthStatus.EXCELLENT
        elif reproduction_rate >= 0.6:
            return HealthStatus.GOOD
        elif reproduction_rate >= 0.4:
            return HealthStatus.FAIR
        elif reproduction_rate >= 0.2:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _assess_behavioral_health(self, activity_level: float) -> HealthStatus:
        """Assess health status based on behavioral indicators"""
        if activity_level >= 0.9:
            return HealthStatus.EXCELLENT
        elif activity_level >= 0.7:
            return HealthStatus.GOOD
        elif activity_level >= 0.5:
            return HealthStatus.FAIR
        elif activity_level >= 0.3:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _calculate_overall_health_score(self, profile: ColonyHealthProfile) -> float:
        """Calculate weighted overall health score"""

        if not profile.current_metrics:
            return 0.0

        # Health status to score mapping
        status_scores = {
            HealthStatus.EXCELLENT: 1.0,
            HealthStatus.GOOD: 0.8,
            HealthStatus.FAIR: 0.6,
            HealthStatus.POOR: 0.4,
            HealthStatus.CRITICAL: 0.2,
            HealthStatus.DECLINING: 0.3,
            HealthStatus.COLLAPSED: 0.0,
        }

        # Indicator weights
        weights = {
            HealthIndicator.POPULATION: 0.2,
            HealthIndicator.MORTALITY: 0.15,
            HealthIndicator.ENERGY: 0.15,
            HealthIndicator.FORAGING: 0.15,
            HealthIndicator.DISEASE: 0.1,
            HealthIndicator.STRESS: 0.1,
            HealthIndicator.REPRODUCTION: 0.1,
            HealthIndicator.BEHAVIORAL: 0.05,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for indicator, metric in profile.current_metrics.items():
            weight = weights.get(indicator, 0.1)
            score = status_scores.get(metric.status, 0.5)
            weighted_score += score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.5

    def _determine_health_status(self, health_score: float) -> HealthStatus:
        """Determine overall health status from score"""
        if health_score >= 0.9:
            return HealthStatus.EXCELLENT
        elif health_score >= 0.8:
            return HealthStatus.GOOD
        elif health_score >= 0.6:
            return HealthStatus.FAIR
        elif health_score >= 0.4:
            return HealthStatus.POOR
        elif health_score >= 0.2:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.COLLAPSED

    def _update_health_trends(self, profile: ColonyHealthProfile) -> None:
        """Update trend analysis for health indicators"""

        for indicator, history in profile.metric_history.items():
            if len(history) < 5:  # Need at least 5 points for trend analysis
                continue

            # Extract values and timestamps
            values = [metric.value for metric in history]
            timestamps = [metric.timestamp for metric in history]

            # Calculate trend
            trend = self._calculate_trend(values, timestamps)
            profile.health_trends[indicator] = trend

            # Update trend in current metric
            if indicator in profile.current_metrics:
                profile.current_metrics[indicator].trend = trend.trend_direction

    def _calculate_trend(
        self, values: List[float], timestamps: List[float]
    ) -> HealthTrend:
        """Calculate trend analysis for a series of values"""

        if len(values) < 2:
            return HealthTrend(
                indicator=HealthIndicator.POPULATION,  # Will be overridden
                trend_direction="stable",
                trend_strength=0.0,
                slope=0.0,
                r_squared=0.0,
            )

        # Simple linear regression
        n = len(values)
        x = np.array(range(n))
        y = np.array(values)

        # Calculate slope and R-squared
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            slope = 0.0
            r_squared = 0.0
        else:
            slope = numerator / denominator

            # Calculate R-squared
            y_pred = slope * (x - x_mean) + y_mean
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)

            if ss_tot == 0:
                r_squared = 1.0 if ss_res == 0 else 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)

        # Determine trend direction and strength
        if abs(slope) < 0.01:  # Very small slope
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "improving"
            trend_strength = min(1.0, abs(slope) * 10)  # Scale slope to 0-1 range
        else:
            trend_direction = "declining"
            trend_strength = min(1.0, abs(slope) * 10)

        # Generate predictions
        prediction_horizon = 5
        predicted_values = []
        for i in range(1, prediction_horizon + 1):
            pred_x = n + i - 1
            pred_y = slope * (pred_x - x_mean) + y_mean
            predicted_values.append(max(0.0, pred_y))  # Ensure non-negative predictions

        return HealthTrend(
            indicator=HealthIndicator.POPULATION,  # Will be set by caller
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            prediction_horizon=prediction_horizon,
            predicted_values=predicted_values,
        )

    def _should_emit_alert(
        self, alert_key: str, level: AlertLevel, timestamp: float
    ) -> bool:
        """Check if alert should be emitted based on debouncing rules"""

        # Check cooldown period
        if alert_key in self.alert_debounce_cache:
            last_alert_time = self.alert_debounce_cache[alert_key]
            cooldown_period = self.alert_cooldown_periods.get(level, 5.0)

            if timestamp - last_alert_time < cooldown_period:
                return False

        # Check rate limiting (alerts per minute)
        rate_key = f"{alert_key.split('_')[0]}_{alert_key.split('_')[1]}"  # Extract type and indicator

        if rate_key not in self.alert_rate_limits:
            self.alert_rate_limits[rate_key] = []

        # Clean old timestamps (older than 1 minute)
        cutoff_time = timestamp - 60.0
        self.alert_rate_limits[rate_key] = [
            t for t in self.alert_rate_limits[rate_key] if t > cutoff_time
        ]

        # Check if we've exceeded rate limit
        if len(self.alert_rate_limits[rate_key]) >= self.max_alerts_per_minute:
            return False

        # Alert is allowed
        self.alert_debounce_cache[alert_key] = timestamp
        self.alert_rate_limits[rate_key].append(timestamp)

        return True

    def _create_debounced_alert(
        self,
        alert_type: str,
        indicator: HealthIndicator,
        level: AlertLevel,
        colony_id: int,
        message: str,
        timestamp: float,
        current_value: float,
        threshold_value: float,
        recommendation: str = "",
    ) -> Optional[HealthAlert]:
        """Create alert with debouncing applied"""

        alert_key = f"{alert_type}_{indicator.value}_{colony_id}"

        if not self._should_emit_alert(alert_key, level, timestamp):
            return None

        alert_id = f"{alert_key}_{timestamp}"

        return HealthAlert(
            alert_id=alert_id,
            level=level,
            indicator=indicator,
            message=message,
            timestamp=timestamp,
            colony_id=colony_id,
            current_value=current_value,
            threshold_value=threshold_value,
            recommendation=recommendation,
        )

    def _check_health_alerts(
        self, profile: ColonyHealthProfile, timestamp: float
    ) -> List[HealthAlert]:
        """Check for health alerts based on current metrics and trends"""

        alerts = []

        # Check if colony is still in startup grace period
        colony_age_days = (
            timestamp / 24.0
        )  # Convert hours to days (assuming timestamp is in hours)
        in_grace_period = colony_age_days < self.thresholds.startup_grace_period_days

        for indicator, metric in profile.current_metrics.items():
            # Critical status alerts - reduce severity during grace period
            if metric.status == HealthStatus.CRITICAL:
                # Use WARNING level instead of CRITICAL during grace period
                alert_level = (
                    AlertLevel.WARNING if in_grace_period else AlertLevel.CRITICAL
                )
                alert_message = f"Colony {profile.colony_id} has {'startup warning' if in_grace_period else 'critical'} {indicator.value} status"
                alert = self._create_debounced_alert(
                    alert_type="startup_warning" if in_grace_period else "critical",
                    indicator=indicator,
                    level=alert_level,
                    colony_id=profile.colony_id,
                    message=alert_message,
                    timestamp=timestamp,
                    current_value=metric.value,
                    threshold_value=self._get_threshold_for_indicator(indicator),
                    recommendation=self._get_recommendation_for_indicator(
                        indicator, metric.status
                    ),
                )
                if alert:
                    alerts.append(alert)

            # Declining trend alerts
            if indicator in profile.health_trends:
                trend = profile.health_trends[indicator]
                if (
                    trend.trend_direction == "declining"
                    and trend.trend_strength > 0.7
                    and trend.r_squared > 0.5
                ):
                    alert = self._create_debounced_alert(
                        alert_type="declining",
                        indicator=indicator,
                        level=AlertLevel.WARNING,
                        colony_id=profile.colony_id,
                        message=f"Colony {profile.colony_id} shows declining {indicator.value} trend",
                        timestamp=timestamp,
                        current_value=metric.value,
                        threshold_value=trend.slope,
                        recommendation=f"Monitor {indicator.value} closely and consider intervention",
                    )
                    if alert:
                        alerts.append(alert)

        # System-wide alerts
        if profile.overall_health_score < 0.3:
            alert = self._create_debounced_alert(
                alert_type="overall_health",
                indicator=HealthIndicator.POPULATION,  # Use as general indicator
                level=AlertLevel.EMERGENCY,
                colony_id=profile.colony_id,
                message=f"Colony {profile.colony_id} overall health is critically low",
                timestamp=timestamp,
                current_value=profile.overall_health_score,
                threshold_value=0.3,
                recommendation="Immediate intervention required to prevent colony collapse",
            )
            if alert:
                alerts.append(alert)

        return alerts

    def _get_threshold_for_indicator(self, indicator: HealthIndicator) -> float:
        """Get threshold value for an indicator"""

        threshold_map = {
            HealthIndicator.POPULATION: self.thresholds.min_population,
            HealthIndicator.MORTALITY: self.thresholds.max_daily_mortality_rate,
            HealthIndicator.ENERGY: self.thresholds.min_energy_per_bee,
            HealthIndicator.FORAGING: self.thresholds.min_foraging_efficiency,
            HealthIndicator.DISEASE: self.thresholds.max_disease_prevalence,
            HealthIndicator.STRESS: self.thresholds.stress_indicator_threshold,
            HealthIndicator.NUTRITION: self.thresholds.min_food_stores,
            HealthIndicator.BEHAVIORAL: self.thresholds.min_activity_level,
        }

        return threshold_map.get(indicator, 0.5)

    def _get_recommendation_for_indicator(
        self, indicator: HealthIndicator, status: HealthStatus
    ) -> str:
        """Get recommendation for improving an indicator"""

        recommendations = {
            HealthIndicator.POPULATION: "Consider supplemental feeding and improving foraging conditions",
            HealthIndicator.MORTALITY: "Check for disease, parasites, and environmental stressors",
            HealthIndicator.ENERGY: "Increase food resources and reduce energy expenditure",
            HealthIndicator.FORAGING: "Improve flower patch quality and reduce foraging distance",
            HealthIndicator.DISEASE: "Implement disease management and treatment protocols",
            HealthIndicator.STRESS: "Reduce environmental stressors and improve colony conditions",
            HealthIndicator.NUTRITION: "Provide supplemental nutrition and improve food quality",
            HealthIndicator.BEHAVIORAL: "Monitor for behavioral anomalies and environmental factors",
            HealthIndicator.REPRODUCTION: "Improve queen health and colony nutritional status",
        }

        base_recommendation = recommendations.get(
            indicator, "Monitor closely and consult management protocols"
        )

        if status == HealthStatus.CRITICAL:
            return f"URGENT: {base_recommendation}"
        else:
            return base_recommendation

    def _process_health_alert(self, alert: HealthAlert) -> None:
        """Process and handle a health alert with debouncing"""

        # Create debouncing key
        alert_key = f"{alert.colony_id}_{alert.indicator.value}_{alert.level.value}"
        current_time = time.time()

        # Initialize debouncing data if needed
        if not hasattr(self, "_alert_debounce_cache"):
            self._alert_debounce_cache: Dict[str, Any] = {}
            self._alert_debounce_config = {
                "min_interval": 30.0,  # 30 seconds minimum between same alerts
                "max_count_per_minute": 2,  # Max 2 alerts of same type per minute
                "emergency_bypass": True,  # Always show emergency alerts
            }

        # Check if we should debounce this alert
        if alert.level != AlertLevel.EMERGENCY or not self._alert_debounce_config.get(
            "emergency_bypass", True
        ):
            if alert_key in self._alert_debounce_cache:
                last_time, count = self._alert_debounce_cache[alert_key]

                # Check minimum interval
                if (
                    current_time - last_time
                    < self._alert_debounce_config["min_interval"]
                ):
                    return  # Skip this alert

                # Check rate limiting
                if count >= self._alert_debounce_config["max_count_per_minute"]:
                    if current_time - last_time < 60.0:  # Within last minute
                        return  # Skip this alert
                    else:
                        count = 0  # Reset count after minute

        # Update debouncing cache
        self._alert_debounce_cache[alert_key] = (
            current_time,
            self._alert_debounce_cache.get(alert_key, (0, 0))[1] + 1,
        )

        # Log the alert
        log_level = {
            AlertLevel.INFO: "info",
            AlertLevel.WARNING: "warning",
            AlertLevel.CRITICAL: "error",
            AlertLevel.EMERGENCY: "critical",
        }.get(alert.level, "info")

        log_message = f"Health Alert [{alert.level.value.upper()}]: {alert.message}"
        getattr(self.logger, log_level)(log_message)

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Alert callback failed: {e}")

    def _cleanup_resolved_alerts(self, profile: ColonyHealthProfile) -> None:
        """Clean up resolved alerts"""

        resolved_alerts = []

        for alert in profile.active_alerts:
            # Check if alert condition is resolved
            if alert.indicator in profile.current_metrics:
                current_metric = profile.current_metrics[alert.indicator]

                # Alert is resolved if status improved significantly
                if alert.level == AlertLevel.CRITICAL and current_metric.status not in [
                    HealthStatus.CRITICAL,
                    HealthStatus.POOR,
                ]:
                    alert.resolved = True
                    resolved_alerts.append(alert)
                elif alert.level == AlertLevel.WARNING and current_metric.status in [
                    HealthStatus.GOOD,
                    HealthStatus.EXCELLENT,
                ]:
                    alert.resolved = True
                    resolved_alerts.append(alert)

        # Move resolved alerts to history
        for alert in resolved_alerts:
            profile.active_alerts.remove(alert)
            profile.alert_history.append(alert)

        if resolved_alerts:
            if self.logger:
                self.logger.info(
                    f"Resolved {len(resolved_alerts)} alerts for colony {profile.colony_id}"
                )

    def get_colony_health_summary(self, colony_id: int) -> Dict[str, Any]:
        """Get comprehensive health summary for a colony"""

        if colony_id not in self.colony_profiles:
            return {"error": f"Colony {colony_id} not registered for health monitoring"}

        profile = self.colony_profiles[colony_id]

        return {
            "colony_id": colony_id,
            "overall_health_score": profile.overall_health_score,
            "health_status": profile.health_status.value,
            "health_grade": profile.health_grade,
            "last_update": profile.last_update,
            "current_metrics": {
                indicator.value: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "trend": metric.trend,
                }
                for indicator, metric in profile.current_metrics.items()
            },
            "critical_indicators": [ind.value for ind in profile.critical_indicators],
            "declining_indicators": [ind.value for ind in profile.declining_indicators],
            "active_alerts": len(profile.active_alerts),
            "alert_summary": {
                level.value: len([a for a in profile.active_alerts if a.level == level])
                for level in AlertLevel
            },
            "trend_analysis": {
                indicator.value: {
                    "direction": trend.trend_direction,
                    "strength": trend.trend_strength,
                    "r_squared": trend.r_squared,
                }
                for indicator, trend in profile.health_trends.items()
            },
        }

    def get_system_health_overview(self) -> Dict[str, Any]:
        """Get system-wide health overview"""

        if not self.colony_profiles:
            return {"message": "No colonies registered for health monitoring"}

        # Calculate system metrics
        total_colonies = len(self.colony_profiles)
        health_scores = [p.overall_health_score for p in self.colony_profiles.values()]

        system_health_score = statistics.mean(health_scores) if health_scores else 0.0

        # Status distribution
        status_counts: Dict[str, int] = defaultdict(int)
        for profile in self.colony_profiles.values():
            status_counts[profile.health_status.value] += 1

        # Alert summary
        total_alerts = sum(len(p.active_alerts) for p in self.colony_profiles.values())
        alert_levels: Dict[str, int] = defaultdict(int)
        for profile in self.colony_profiles.values():
            for alert in profile.active_alerts:
                alert_levels[alert.level.value] += 1

        # Critical colonies
        critical_colonies = [
            p.colony_id
            for p in self.colony_profiles.values()
            if p.health_status
            in [HealthStatus.CRITICAL, HealthStatus.POOR, HealthStatus.COLLAPSED]
        ]

        return {
            "system_health_score": system_health_score,
            "total_colonies": total_colonies,
            "healthy_colonies": status_counts.get("good", 0)
            + status_counts.get("excellent", 0),
            "critical_colonies": len(critical_colonies),
            "critical_colony_ids": critical_colonies,
            "status_distribution": dict(status_counts),
            "alert_summary": {
                "total_alerts": total_alerts,
                "by_level": dict(alert_levels),
            },
            "monitoring_statistics": {
                "update_frequency": self.update_frequency,
                "monitoring_enabled": self.monitoring_enabled,
                "last_system_update": max(
                    (p.last_update for p in self.colony_profiles.values()), default=0.0
                ),
            },
        }

    def add_alert_callback(self, callback: Any) -> None:
        """Add callback function for health alerts"""
        self.alert_callbacks.append(callback)
        if self.logger:
            self.logger.info("Added health alert callback")

    def get_health_predictions(
        self, colony_id: int, horizon: int = 10
    ) -> Dict[str, Any]:
        """Get health predictions for a colony"""

        if colony_id not in self.colony_profiles:
            return {"error": f"Colony {colony_id} not registered"}

        profile = self.colony_profiles[colony_id]
        predictions = {}

        for indicator, trend in profile.health_trends.items():
            if len(trend.predicted_values) > 0:
                predictions[indicator.value] = {
                    "current_value": profile.current_metrics[indicator].value,
                    "predicted_values": trend.predicted_values[:horizon],
                    "trend_direction": trend.trend_direction,
                    "confidence": trend.r_squared,
                }

        return {
            "colony_id": colony_id,
            "prediction_horizon": horizon,
            "predictions": predictions,
            "overall_prognosis": self._assess_overall_prognosis(profile),
        }

    def _assess_overall_prognosis(self, profile: ColonyHealthProfile) -> str:
        """Assess overall colony prognosis"""

        declining_count = len(profile.declining_indicators)
        critical_count = len(profile.critical_indicators)

        if critical_count >= 3:
            return "Poor - Multiple critical indicators"
        elif declining_count >= 4:
            return "Concerning - Multiple declining trends"
        elif profile.overall_health_score > 0.8:
            return "Excellent - Strong health indicators"
        elif profile.overall_health_score > 0.6:
            return "Good - Stable health status"
        else:
            return "Fair - Monitor closely"
