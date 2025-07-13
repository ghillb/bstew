"""
Enhanced Mortality Tracking System for BSTEW
==========================================

Implements comprehensive mortality tracking with detailed death causes,
age-structured mortality rates, environmental factors, and statistical analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import logging
from collections import defaultdict, deque
import json
from datetime import datetime

from .genetics import Ploidy


class MortalityCategory(Enum):
    """High-level mortality categories"""

    DEVELOPMENT = "development"
    ENVIRONMENTAL = "environmental"
    BIOLOGICAL = "biological"
    MANAGEMENT = "management"
    PREDATION = "predation"
    DISEASE = "disease"
    SENESCENCE = "senescence"
    UNKNOWN = "unknown"


class DetailedDeathCause(Enum):
    """Detailed death causes for adult bees"""

    # Environmental causes
    COLD_EXPOSURE = "cold_exposure"
    HEAT_STRESS = "heat_stress"
    STARVATION = "starvation"
    DEHYDRATION = "dehydration"
    WEATHER_MORTALITY = "weather_mortality"
    PESTICIDE_EXPOSURE = "pesticide_exposure"

    # Biological causes
    OLD_AGE = "old_age"
    EXHAUSTION = "exhaustion"
    DISEASE_INFECTION = "disease_infection"
    PARASITE_LOAD = "parasite_load"
    GENETIC_DEFECT = "genetic_defect"
    IMMUNE_FAILURE = "immune_failure"

    # Behavioral/Foraging causes
    FORAGING_ACCIDENT = "foraging_accident"
    LOST_WHILE_FORAGING = "lost_while_foraging"
    ENERGY_DEPLETION = "energy_depletion"
    OVERWORK = "overwork"

    # Predation
    BIRD_PREDATION = "bird_predation"
    SPIDER_PREDATION = "spider_predation"
    WASP_PREDATION = "wasp_predation"
    BADGER_ATTACK = "badger_attack"

    # Management
    MOWING_MORTALITY = "mowing_mortality"
    HABITAT_DESTRUCTION = "habitat_destruction"
    CHEMICAL_TREATMENT = "chemical_treatment"

    # Colony dynamics
    EVICTION = "eviction"
    RESOURCE_COMPETITION = "resource_competition"
    QUEEN_SUPERSEDURE = "queen_supersedure"

    # Development causes (from development.py)
    INSUFFICIENT_INCUBATION = "insufficient_incubation"
    INSUFFICIENT_WEIGHT = "insufficient_weight"
    ENERGY_SHORTAGE = "energy_shortage"
    NO_ADULT_CARE = "no_adult_care"
    DIPLOID_MALE = "diploid_male"
    ENVIRONMENTAL_STRESS = "environmental_stress"

    # Unknown cause
    UNKNOWN = "unknown"


class MortalityEvent(BaseModel):
    """Individual mortality event record"""

    model_config = {"validate_assignment": True}

    bee_id: int = Field(ge=0, description="Unique bee identifier")
    death_day: int = Field(ge=0, description="Day of death in simulation")
    death_step: int = Field(ge=0, description="Step within day of death")
    age_at_death: float = Field(ge=0.0, description="Age at death in days")
    bee_role: str = Field(description="Bee role at time of death")
    bee_sex: str = Field(description="Bee sex")
    colony_id: Optional[str] = Field(default=None, description="Colony identifier")
    species: Optional[str] = Field(default=None, description="Bee species")

    # Death details
    primary_cause: DetailedDeathCause = Field(
        default=DetailedDeathCause.UNKNOWN, description="Primary cause of death"
    )
    contributing_factors: List[DetailedDeathCause] = Field(
        default_factory=list, description="Contributing death factors"
    )
    mortality_category: MortalityCategory = Field(
        default=MortalityCategory.UNKNOWN, description="Mortality category"
    )

    # Contextual information
    location: Optional[Tuple[float, float]] = Field(
        default=None, description="Death location coordinates"
    )
    environmental_conditions: Dict[str, float] = Field(
        default_factory=dict, description="Environmental conditions at death"
    )
    bee_state: Dict[str, float] = Field(
        default_factory=dict, description="Bee state variables at death"
    )

    @field_validator("bee_role", "bee_sex")
    @classmethod
    def validate_bee_attributes(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Bee role and sex cannot be empty")
        return v.strip().lower()

    # Genetic information
    genotype_summary: Optional[Dict[str, Any]] = None
    inbreeding_coefficient: Optional[float] = None

    # Development history (for bees that survived to adulthood)
    development_time: Optional[float] = None
    development_quality: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert mortality event to dictionary"""
        return {
            "bee_id": self.bee_id,
            "death_day": self.death_day,
            "death_step": self.death_step,
            "age_at_death": self.age_at_death,
            "bee_role": self.bee_role,
            "bee_sex": self.bee_sex,
            "colony_id": self.colony_id,
            "species": self.species,
            "primary_cause": self.primary_cause.value,
            "contributing_factors": [f.value for f in self.contributing_factors],
            "mortality_category": self.mortality_category.value,
            "location": self.location,
            "environmental_conditions": self.environmental_conditions.copy(),
            "bee_state": self.bee_state.copy(),
            "genotype_summary": (
                self.genotype_summary.copy() if self.genotype_summary else None
            ),
            "inbreeding_coefficient": self.inbreeding_coefficient,
            "development_time": self.development_time,
            "development_quality": self.development_quality,
        }


class MortalityStatistics(BaseModel):
    """Mortality statistics for analysis"""

    model_config = {"validate_assignment": True}

    total_deaths: int = Field(
        default=0, ge=0, description="Total number of recorded deaths"
    )
    deaths_by_cause: Dict[str, int] = Field(
        default_factory=dict, description="Deaths grouped by cause"
    )
    deaths_by_category: Dict[str, int] = Field(
        default_factory=dict, description="Deaths grouped by category"
    )
    deaths_by_age_group: Dict[str, int] = Field(
        default_factory=dict, description="Deaths grouped by age group"
    )
    deaths_by_role: Dict[str, int] = Field(
        default_factory=dict, description="Deaths grouped by bee role"
    )
    deaths_by_species: Dict[str, int] = Field(
        default_factory=dict, description="Deaths grouped by species"
    )

    # Temporal patterns
    deaths_by_day: Dict[int, int] = Field(
        default_factory=dict, description="Deaths by simulation day"
    )
    deaths_by_season: Dict[str, int] = Field(
        default_factory=dict, description="Deaths by season"
    )

    # Survival analysis
    mean_lifespan_by_role: Dict[str, float] = Field(
        default_factory=dict, description="Mean lifespan by role"
    )
    median_lifespan_by_role: Dict[str, float] = Field(
        default_factory=dict, description="Median lifespan by role"
    )
    survival_curves: Dict[str, List[float]] = Field(
        default_factory=dict, description="Survival curves by role"
    )

    # Environmental correlations
    mortality_temperature_correlation: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Correlation with temperature"
    )
    mortality_weather_correlation: float = 0.0

    def get_mortality_rate(self, population_size: int) -> float:
        """Calculate mortality rate"""
        return self.total_deaths / max(1, population_size)


class MortalityTracker:
    """
    Comprehensive mortality tracking system.

    Tracks all bee deaths with detailed cause analysis, environmental context,
    and statistical patterns for colony health assessment and research.
    """

    def __init__(self, max_history_days: int = 365):
        self.max_history_days = max_history_days
        self.mortality_events: List[MortalityEvent] = []
        self.daily_mortality: Dict[int, List[MortalityEvent]] = defaultdict(list)

        # Rolling statistics
        self.recent_mortality: deque[MortalityEvent] = deque(maxlen=30)  # Last 30 days
        self.statistics = MortalityStatistics()

        # Environmental tracking for correlation analysis
        self.environmental_history: Dict[int, Dict[str, float]] = {}

        # Risk factors and thresholds
        self.risk_thresholds: Dict[str, Any] = {
            "daily_mortality_rate": 0.05,  # 5% daily mortality is concerning
            "age_specific_mortality": {
                "young": 0.02,  # 0-7 days
                "adult": 0.01,  # 8-28 days
                "old": 0.05,  # >28 days
            },
            "environmental_stress": {
                "temperature_low": 5.0,  # °C
                "temperature_high": 35.0,  # °C
                "wind_high": 20.0,  # m/s
                "rain_heavy": 10.0,  # mm/day
            },
        }

        self.logger = logging.getLogger(__name__)

    def record_death(
        self,
        bee_agent: Any,
        primary_cause: DetailedDeathCause,
        contributing_factors: Optional[List[DetailedDeathCause]] = None,
        environmental_conditions: Optional[Dict[str, float]] = None,
        simulation_step: int = 0,
    ) -> MortalityEvent:
        """Record a bee death with comprehensive details"""

        if contributing_factors is None:
            contributing_factors = []
        if environmental_conditions is None:
            environmental_conditions = {}

        # Determine mortality category
        category = self._categorize_death_cause(primary_cause)

        # Extract bee state information
        bee_state = {
            "energy": getattr(bee_agent, "energy", 0.0),
            "disease_load": getattr(bee_agent, "disease_load", 0.0),
            "foraging_efficiency": getattr(bee_agent, "foraging_efficiency", 1.0),
            "longevity_factor": getattr(bee_agent, "longevity_factor", 1.0),
            "current_load": getattr(bee_agent, "current_load", 0.0),
        }

        # Extract genetic information
        genotype_summary = None
        inbreeding_coeff = None
        if hasattr(bee_agent, "genotype") and bee_agent.genotype:
            genotype_summary = {
                "sex": bee_agent.genotype.sex.value,
                "ploidy": bee_agent.genotype.ploidy.value,
                "allele_count": len(bee_agent.genotype.alleles),
                "is_diploid_male": bee_agent.genotype.is_diploid_male(),
            }
            # Calculate basic inbreeding coefficient (simplified)
            if bee_agent.genotype.ploidy == Ploidy.DIPLOID:
                allele_ids = bee_agent.genotype.get_allele_ids()
                inbreeding_coeff = (
                    1.0 if len(set(allele_ids)) < len(allele_ids) else 0.0
                )
            else:
                inbreeding_coeff = 0.0

        # Extract development history
        dev_time = None
        dev_quality = None
        if hasattr(bee_agent, "development_history") and bee_agent.development_history:
            dev_time = bee_agent.development_history.get("total_development_time")
            if dev_time is not None and isinstance(dev_time, (int, float)):
                care_received = bee_agent.development_history.get("care_received", 0)
                if isinstance(care_received, (int, float)):
                    dev_quality = care_received / max(1, dev_time or 1)

        # Create mortality event
        event = MortalityEvent(
            bee_id=bee_agent.unique_id,
            death_day=(
                simulation_step // 24 if simulation_step else 0
            ),  # Assuming 24 steps per day
            death_step=simulation_step,
            age_at_death=bee_agent.age,
            bee_role=bee_agent.role.value if hasattr(bee_agent, "role") else "unknown",
            bee_sex=(
                bee_agent.genotype.sex.value
                if (hasattr(bee_agent, "genotype") and bee_agent.genotype)
                else "unknown"
            ),
            colony_id=(
                getattr(bee_agent.colony, "species", None)
                if hasattr(bee_agent, "colony")
                else None
            ),
            species=(
                getattr(bee_agent.colony, "species", None)
                if hasattr(bee_agent, "colony")
                else None
            ),
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            mortality_category=category,
            location=getattr(bee_agent, "location", None),
            environmental_conditions=environmental_conditions.copy(),
            bee_state=bee_state,
            genotype_summary=genotype_summary,
            inbreeding_coefficient=inbreeding_coeff,
            development_time=dev_time,
            development_quality=dev_quality,
        )

        # Store the event
        self.mortality_events.append(event)
        self.daily_mortality[event.death_day].append(event)
        self.recent_mortality.append(event)

        # Update statistics
        self._update_statistics(event)

        # Log significant deaths
        if category in [MortalityCategory.PREDATION, MortalityCategory.MANAGEMENT]:
            self.logger.warning(
                f"Significant mortality: {bee_agent.role.value} bee died from {primary_cause.value}"
            )

        return event

    def _categorize_death_cause(self, cause: DetailedDeathCause) -> MortalityCategory:
        """Categorize death cause into high-level category"""

        environmental_causes = {
            DetailedDeathCause.COLD_EXPOSURE,
            DetailedDeathCause.HEAT_STRESS,
            DetailedDeathCause.STARVATION,
            DetailedDeathCause.DEHYDRATION,
            DetailedDeathCause.WEATHER_MORTALITY,
            DetailedDeathCause.PESTICIDE_EXPOSURE,
        }

        biological_causes = {
            DetailedDeathCause.OLD_AGE,
            DetailedDeathCause.EXHAUSTION,
            DetailedDeathCause.DISEASE_INFECTION,
            DetailedDeathCause.PARASITE_LOAD,
            DetailedDeathCause.GENETIC_DEFECT,
            DetailedDeathCause.IMMUNE_FAILURE,
            DetailedDeathCause.FORAGING_ACCIDENT,
            DetailedDeathCause.LOST_WHILE_FORAGING,
            DetailedDeathCause.ENERGY_DEPLETION,
            DetailedDeathCause.OVERWORK,
        }

        predation_causes = {
            DetailedDeathCause.BIRD_PREDATION,
            DetailedDeathCause.SPIDER_PREDATION,
            DetailedDeathCause.WASP_PREDATION,
            DetailedDeathCause.BADGER_ATTACK,
        }

        management_causes = {
            DetailedDeathCause.MOWING_MORTALITY,
            DetailedDeathCause.HABITAT_DESTRUCTION,
            DetailedDeathCause.CHEMICAL_TREATMENT,
        }

        development_causes = {
            DetailedDeathCause.INSUFFICIENT_INCUBATION,
            DetailedDeathCause.INSUFFICIENT_WEIGHT,
            DetailedDeathCause.ENERGY_SHORTAGE,
            DetailedDeathCause.NO_ADULT_CARE,
            DetailedDeathCause.DIPLOID_MALE,
            DetailedDeathCause.ENVIRONMENTAL_STRESS,
        }

        if cause in environmental_causes:
            return MortalityCategory.ENVIRONMENTAL
        elif cause in biological_causes:
            return MortalityCategory.BIOLOGICAL
        elif cause in predation_causes:
            return MortalityCategory.PREDATION
        elif cause in management_causes:
            return MortalityCategory.MANAGEMENT
        elif cause in development_causes:
            return MortalityCategory.DEVELOPMENT
        elif cause == DetailedDeathCause.OLD_AGE:
            return MortalityCategory.SENESCENCE
        else:
            return MortalityCategory.UNKNOWN

    def _update_statistics(self, event: MortalityEvent) -> None:
        """Update running statistics with new mortality event"""

        self.statistics.total_deaths += 1

        # By cause
        cause_key = event.primary_cause.value
        self.statistics.deaths_by_cause[cause_key] = (
            self.statistics.deaths_by_cause.get(cause_key, 0) + 1
        )

        # By category
        category_key = event.mortality_category.value
        self.statistics.deaths_by_category[category_key] = (
            self.statistics.deaths_by_category.get(category_key, 0) + 1
        )

        # By role
        role_key = event.bee_role
        self.statistics.deaths_by_role[role_key] = (
            self.statistics.deaths_by_role.get(role_key, 0) + 1
        )

        # By species
        if event.species:
            self.statistics.deaths_by_species[event.species] = (
                self.statistics.deaths_by_species.get(event.species, 0) + 1
            )

        # By age group
        age_group = self._get_age_group(event.age_at_death)
        self.statistics.deaths_by_age_group[age_group] = (
            self.statistics.deaths_by_age_group.get(age_group, 0) + 1
        )

        # By day
        self.statistics.deaths_by_day[event.death_day] = (
            self.statistics.deaths_by_day.get(event.death_day, 0) + 1
        )

        # By season
        season = self._get_season(event.death_day)
        self.statistics.deaths_by_season[season] = (
            self.statistics.deaths_by_season.get(season, 0) + 1
        )

    def _get_age_group(self, age: float) -> str:
        """Categorize bee age into groups"""
        if age <= 7:
            return "young"
        elif age <= 28:
            return "adult"
        else:
            return "old"

    def _get_season(self, day_of_year: int) -> str:
        """Determine season from day of year"""
        if day_of_year < 80:
            return "winter"
        elif day_of_year < 172:
            return "spring"
        elif day_of_year < 266:
            return "summer"
        elif day_of_year < 355:
            return "autumn"
        else:
            return "winter"

    def update_environmental_conditions(
        self, day: int, conditions: Dict[str, float]
    ) -> None:
        """Update environmental conditions for correlation analysis"""
        self.environmental_history[day] = conditions.copy()

        # Calculate correlations if we have enough data
        if len(self.environmental_history) > 30:
            self._calculate_environmental_correlations()

    def _calculate_environmental_correlations(self) -> None:
        """Calculate correlations between mortality and environmental factors"""

        # Get daily mortality counts
        daily_deaths = []
        temperatures = []

        for day in range(
            max(1, max(self.statistics.deaths_by_day.keys()) - 30),
            max(self.statistics.deaths_by_day.keys()) + 1,
        ):
            daily_deaths.append(self.statistics.deaths_by_day.get(day, 0))
            env_data = self.environmental_history.get(day, {})
            temperatures.append(env_data.get("temperature", 20.0))

        # Calculate correlation
        if len(daily_deaths) > 10 and len(temperatures) > 10:
            # Check for constant arrays to avoid division by zero
            deaths_array = np.array(daily_deaths)
            temp_array = np.array(temperatures)

            # Only calculate correlation if both arrays have variation
            if np.std(deaths_array) > 0 and np.std(temp_array) > 0:
                correlation_matrix = np.corrcoef(deaths_array, temp_array)
                # Check for NaN values (which can occur with insufficient data)
                if not np.isnan(correlation_matrix[0, 1]):
                    self.statistics.mortality_temperature_correlation = (
                        correlation_matrix[0, 1]
                    )
                else:
                    self.statistics.mortality_temperature_correlation = 0.0
            else:
                # If either array is constant, correlation is undefined - set to 0
                self.statistics.mortality_temperature_correlation = 0.0

    def get_mortality_risk_assessment(
        self, current_day: int, population_size: int
    ) -> Dict[str, Any]:
        """Assess current mortality risk levels"""

        # Recent mortality rate (last 7 days)
        recent_deaths = sum(
            self.statistics.deaths_by_day.get(day, 0)
            for day in range(max(1, current_day - 6), current_day + 1)
        )
        recent_mortality_rate = recent_deaths / max(1, population_size * 7)

        # Age-specific mortality rates
        age_specific_risk = {}
        for age_group in ["young", "adult", "old"]:
            deaths = self.statistics.deaths_by_age_group.get(age_group, 0)
            # Estimate population in age group (simplified)
            group_population = (
                population_size * {"young": 0.2, "adult": 0.6, "old": 0.2}[age_group]
            )
            rate = deaths / max(1, group_population)
            risk_level = (
                "high"
                if rate > self.risk_thresholds["age_specific_mortality"][age_group]
                else "normal"
            )
            age_specific_risk[age_group] = {"rate": rate, "risk_level": risk_level}

        # Primary causes of concern
        concerning_causes = []
        total_recent = max(1, recent_deaths)
        for cause, count in self.statistics.deaths_by_cause.items():
            if count / total_recent > 0.2:  # More than 20% of recent deaths
                concerning_causes.append(cause)

        # Environmental stress assessment
        current_env = self.environmental_history.get(current_day, {})
        environmental_stress = self._assess_environmental_stress(current_env)

        return {
            "overall_risk_level": (
                "high"
                if recent_mortality_rate > self.risk_thresholds["daily_mortality_rate"]
                else "normal"
            ),
            "recent_mortality_rate": recent_mortality_rate,
            "recent_deaths_7days": recent_deaths,
            "age_specific_risks": age_specific_risk,
            "primary_concerning_causes": concerning_causes,
            "environmental_stress": environmental_stress,
            "correlation_with_temperature": self.statistics.mortality_temperature_correlation,
        }

    def _assess_environmental_stress(
        self, conditions: Dict[str, float]
    ) -> Dict[str, str]:
        """Assess environmental stress levels"""

        stress_levels = {}
        thresholds: Dict[str, float] = self.risk_thresholds["environmental_stress"]

        temp = conditions.get("temperature", 20.0)
        if temp < thresholds["temperature_low"]:
            stress_levels["temperature"] = "cold_stress"
        elif temp > thresholds["temperature_high"]:
            stress_levels["temperature"] = "heat_stress"
        else:
            stress_levels["temperature"] = "normal"

        wind = conditions.get("wind_speed", 0.0)
        stress_levels["wind"] = "high" if wind > thresholds["wind_high"] else "normal"

        rain = conditions.get("rainfall", 0.0)
        stress_levels["precipitation"] = (
            "heavy" if rain > thresholds["rain_heavy"] else "normal"
        )

        return stress_levels

    def calculate_survival_curves(
        self, species: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Calculate survival curves by role"""

        survival_curves = {}

        # Filter events by species if specified
        events = self.mortality_events
        if species:
            events = [e for e in events if e.species == species]

        # Group by role
        by_role = defaultdict(list)
        for event in events:
            by_role[event.bee_role].append(event.age_at_death)

        # Calculate survival curves
        for role, ages in by_role.items():
            if len(ages) > 5:  # Need sufficient data
                ages.sort()
                max_age = max(ages)
                survival_curve = []

                for age in range(0, int(max_age) + 1):
                    survivors = sum(1 for a in ages if a >= age)
                    survival_rate = survivors / len(ages)
                    survival_curve.append(survival_rate)

                survival_curves[role] = survival_curve

        return survival_curves

    def export_mortality_data(
        self, filepath: str, include_individual_events: bool = True
    ) -> None:
        """Export mortality data to JSON file"""

        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_events": len(self.mortality_events),
                "tracking_period_days": self.max_history_days,
            },
            "statistics": {
                "total_deaths": self.statistics.total_deaths,
                "deaths_by_cause": dict(self.statistics.deaths_by_cause),
                "deaths_by_category": dict(self.statistics.deaths_by_category),
                "deaths_by_role": dict(self.statistics.deaths_by_role),
                "deaths_by_species": dict(self.statistics.deaths_by_species),
                "deaths_by_age_group": dict(self.statistics.deaths_by_age_group),
                "deaths_by_season": dict(self.statistics.deaths_by_season),
                "environmental_correlations": {
                    "temperature": self.statistics.mortality_temperature_correlation,
                    "weather": self.statistics.mortality_weather_correlation,
                },
            },
            "survival_curves": self.calculate_survival_curves(),
        }

        if include_individual_events:
            export_data["individual_events"] = [
                event.to_dict() for event in self.mortality_events
            ]

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Exported mortality data to {filepath}")

    def get_mortality_summary(self) -> Dict[str, Any]:
        """Get comprehensive mortality summary"""

        if not self.mortality_events:
            return {"message": "No mortality events recorded"}

        # Calculate key metrics
        total_deaths = len(self.mortality_events)

        # Most common causes
        cause_counts: Dict[str, int] = defaultdict(int)
        for event in self.mortality_events:
            cause_counts[event.primary_cause.value] += 1

        top_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Average lifespans by role
        role_lifespans = defaultdict(list)
        for event in self.mortality_events:
            role_lifespans[event.bee_role].append(event.age_at_death)

        avg_lifespans = {}
        for role, ages in role_lifespans.items():
            avg_lifespans[role] = {
                "mean": np.mean(ages),
                "median": np.median(ages),
                "std": np.std(ages),
                "sample_size": len(ages),
            }

        return {
            "total_deaths": total_deaths,
            "tracking_period": f"{len(self.daily_mortality)} days",
            "top_causes": top_causes,
            "average_lifespans": avg_lifespans,
            "deaths_by_category": dict(self.statistics.deaths_by_category),
            "recent_mortality_trend": (
                list(self.recent_mortality)[-10:] if self.recent_mortality else []
            ),
            "environmental_correlation": self.statistics.mortality_temperature_correlation,
        }

    def cleanup_old_data(self, current_day: int) -> None:
        """Remove old mortality data beyond retention period"""

        cutoff_day = current_day - self.max_history_days

        # Remove old daily mortality records
        old_days = [day for day in self.daily_mortality.keys() if day < cutoff_day]
        for day in old_days:
            del self.daily_mortality[day]

        # Remove old environmental records
        old_env_days = [
            day for day in self.environmental_history.keys() if day < cutoff_day
        ]
        for day in old_env_days:
            del self.environmental_history[day]

        # Keep recent events for statistics but could truncate if memory is a concern
        self.mortality_events = [
            e for e in self.mortality_events if e.death_day >= cutoff_day
        ]
