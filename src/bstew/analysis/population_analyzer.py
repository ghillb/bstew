"""
Population Analysis Framework for BSTEW
=======================================

Comprehensive population trend analysis, growth rate calculations,
and survival analysis for bee colony simulation data.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
import warnings
from enum import Enum

logger = logging.getLogger(__name__)


class TrendType(Enum):
    """Types of population trends"""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    FLUCTUATING = "fluctuating"
    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic"


@dataclass
class TrendResult:
    """Result of trend analysis"""

    trend_type: TrendType
    slope: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    significance: str  # 'significant', 'marginal', 'not_significant'


@dataclass
class GrowthRateResult:
    """Result of growth rate analysis"""

    intrinsic_growth_rate: float
    doubling_time: Optional[float]
    exponential_rate: float
    logistic_rate: float
    carrying_capacity: Optional[float]
    growth_phase: str  # 'exponential', 'logistic', 'declining', 'stable'


@dataclass
class SurvivalResult:
    """Result of survival analysis"""

    median_survival: float
    survival_curve: List[Tuple[float, float]]  # (time, survival_probability)
    hazard_rates: Dict[str, float]  # by age group, role, etc.
    mortality_factors: Dict[str, float]
    life_expectancy: float
    age_specific_mortality: List[Tuple[float, float]]


class PopulationAnalyzer:
    """
    Population analysis system for BSTEW simulation data.

    Provides comprehensive analysis of population dynamics including:
    - Trend analysis with statistical significance
    - Growth rate calculations (exponential and logistic)
    - Survival analysis with life tables
    - Demographic transitions and age structure analysis
    """

    def __init__(self, time_resolution: str = "daily"):
        """
        Initialize the population analyzer.

        Args:
            time_resolution: Time resolution for analysis ('daily', 'weekly', 'monthly')
        """
        self.time_resolution = time_resolution
        self.logger = logging.getLogger(__name__)

        # Analysis cache
        self._trend_cache: Dict[str, TrendResult] = {}
        self._growth_cache: Dict[str, GrowthRateResult] = {}
        self._survival_cache: Dict[str, SurvivalResult] = {}

    def calculate_trends(
        self,
        population_data: Union[List[Dict[str, Any]], pd.DataFrame],
        group_by: Optional[str] = None,
        time_column: str = "time",
        value_column: str = "population",
    ) -> Dict[str, TrendResult]:
        """
        Calculate population trends with statistical analysis.

        Args:
            population_data: Population time series data
            group_by: Column to group by (e.g., 'role', 'colony_id')
            time_column: Name of time column
            value_column: Name of population value column

        Returns:
            Dictionary mapping group names to trend results
        """
        # Convert to DataFrame if needed
        if isinstance(population_data, list):
            df = pd.DataFrame(population_data)
        else:
            df = population_data.copy()

        # Validate required columns
        if time_column not in df.columns or value_column not in df.columns:
            raise ValueError(
                f"Required columns '{time_column}' and '{value_column}' not found"
            )

        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])

        results = {}

        if group_by and group_by in df.columns:
            # Group-wise analysis
            for group_name, group_data in df.groupby(group_by):
                cache_key = f"{group_name}_{time_column}_{value_column}"
                if cache_key in self._trend_cache:
                    results[str(group_name)] = self._trend_cache[cache_key]
                else:
                    trend_result = self._calculate_single_trend(
                        group_data[time_column], group_data[value_column]
                    )
                    self._trend_cache[cache_key] = trend_result
                    results[str(group_name)] = trend_result
        else:
            # Overall analysis
            cache_key = f"overall_{time_column}_{value_column}"
            if cache_key in self._trend_cache:
                results["overall"] = self._trend_cache[cache_key]
            else:
                trend_result = self._calculate_single_trend(
                    df[time_column], df[value_column]
                )
                self._trend_cache[cache_key] = trend_result
                results["overall"] = trend_result

        return results

    def calculate_growth_rates(
        self,
        population_data: Union[List[Dict[str, Any]], pd.DataFrame],
        group_by: Optional[str] = None,
        time_column: str = "time",
        value_column: str = "population",
    ) -> Dict[str, GrowthRateResult]:
        """
        Calculate population growth rates using multiple models.

        Args:
            population_data: Population time series data
            group_by: Column to group by
            time_column: Name of time column
            value_column: Name of population value column

        Returns:
            Dictionary mapping group names to growth rate results
        """
        # Convert to DataFrame if needed
        if isinstance(population_data, list):
            df = pd.DataFrame(population_data)
        else:
            df = population_data.copy()

        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])

        results = {}

        if group_by and group_by in df.columns:
            # Group-wise analysis
            for group_name, group_data in df.groupby(group_by):
                cache_key = f"{group_name}_growth_{time_column}_{value_column}"
                if cache_key in self._growth_cache:
                    results[str(group_name)] = self._growth_cache[cache_key]
                else:
                    growth_result = self._calculate_growth_rates_single(
                        group_data[time_column], group_data[value_column]
                    )
                    self._growth_cache[cache_key] = growth_result
                    results[str(group_name)] = growth_result
        else:
            # Overall analysis
            cache_key = f"overall_growth_{time_column}_{value_column}"
            if cache_key in self._growth_cache:
                results["overall"] = self._growth_cache[cache_key]
            else:
                growth_result = self._calculate_growth_rates_single(
                    df[time_column], df[value_column]
                )
                self._growth_cache[cache_key] = growth_result
                results["overall"] = growth_result

        return results

    def survival_analysis(
        self,
        bee_data: Union[List[Dict[str, Any]], pd.DataFrame],
        group_by: Optional[str] = None,
        birth_column: str = "birth_time",
        death_column: str = "death_time",
        status_column: str = "status",
    ) -> Dict[str, SurvivalResult]:
        """
        Perform survival analysis on individual bee data.

        Args:
            bee_data: Individual bee lifecycle data
            group_by: Column to group by (e.g., 'role', 'colony_id')
            birth_column: Name of birth time column
            death_column: Name of death time column (None for alive bees)
            status_column: Status column ('alive', 'dead')

        Returns:
            Dictionary mapping group names to survival results
        """
        # Convert to DataFrame if needed
        if isinstance(bee_data, list):
            df = pd.DataFrame(bee_data)
        else:
            df = bee_data.copy()

        # Validate required columns
        required_cols = [birth_column, status_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        results = {}

        if group_by and group_by in df.columns:
            # Group-wise analysis
            for group_name, group_data in df.groupby(group_by):
                cache_key = f"{group_name}_survival_{birth_column}_{death_column}"
                if cache_key in self._survival_cache:
                    results[str(group_name)] = self._survival_cache[cache_key]
                else:
                    survival_result = self._calculate_survival_single(
                        group_data, birth_column, death_column, status_column
                    )
                    self._survival_cache[cache_key] = survival_result
                    results[str(group_name)] = survival_result
        else:
            # Overall analysis
            cache_key = f"overall_survival_{birth_column}_{death_column}"
            if cache_key in self._survival_cache:
                results["overall"] = self._survival_cache[cache_key]
            else:
                survival_result = self._calculate_survival_single(
                    df, birth_column, death_column, status_column
                )
                self._survival_cache[cache_key] = survival_result
                results["overall"] = survival_result

        return results

    def _calculate_single_trend(
        self, times: pd.Series, values: pd.Series
    ) -> TrendResult:
        """Calculate trend for a single time series"""
        # Convert times to numeric (days from first observation)
        time_numeric = (
            times - times.min()
        ).dt.total_seconds() / 86400  # Convert to days

        # Remove NaN values
        valid_mask = ~(np.isnan(time_numeric) | np.isnan(values))
        if valid_mask.sum() < 3:
            return TrendResult(
                trend_type=TrendType.STABLE,
                slope=0.0,
                r_squared=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                significance="not_significant",
            )

        x = time_numeric[valid_mask].values
        y = values[valid_mask].values

        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Simple p-value calculation (basic significance test)
        n = len(x)
        if n > 2:
            t_stat = abs(slope) * np.sqrt((n - 2) / (1 - r_squared + 1e-10))
            # Simplified p-value (would use scipy.stats.t.sf in production)
            p_value = max(0.001, 2 * (1 - min(0.999, t_stat / 10)))
        else:
            p_value = 1.0

        # Determine trend type
        if abs(slope) < 0.01:
            trend_type = TrendType.STABLE
        elif slope > 0:
            trend_type = TrendType.INCREASING if slope < 1.0 else TrendType.EXPONENTIAL
        else:
            trend_type = TrendType.DECREASING

        # Significance
        if p_value < 0.01:
            significance = "significant"
        elif p_value < 0.05:
            significance = "marginal"
        else:
            significance = "not_significant"

        # Confidence interval (simplified)
        se = (
            np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((x - np.mean(x)) ** 2))
            if n > 2
            else 0
        )
        ci_half = 1.96 * se  # 95% CI

        return TrendResult(
            trend_type=trend_type,
            slope=slope,
            r_squared=r_squared,
            p_value=p_value,
            confidence_interval=(slope - ci_half, slope + ci_half),
            significance=significance,
        )

    def _calculate_growth_rates_single(
        self, times: pd.Series, values: pd.Series
    ) -> GrowthRateResult:
        """Calculate growth rates for a single time series"""
        # Convert times to numeric
        time_numeric = (times - times.min()).dt.total_seconds() / 86400

        # Remove invalid values
        valid_mask = ~(np.isnan(time_numeric) | np.isnan(values)) & (values > 0)
        if valid_mask.sum() < 3:
            return GrowthRateResult(
                intrinsic_growth_rate=0.0,
                doubling_time=None,
                exponential_rate=0.0,
                logistic_rate=0.0,
                carrying_capacity=None,
                growth_phase="stable",
            )

        t = time_numeric[valid_mask].values
        N = values[valid_mask].values

        # Intrinsic growth rate (r = ln(N_t+1/N_t))
        if len(N) > 1:
            growth_rates = np.diff(np.log(N + 1e-10)) / np.diff(t)
            intrinsic_growth_rate = np.mean(growth_rates)
        else:
            intrinsic_growth_rate = 0.0

        # Exponential model: N(t) = N0 * exp(r*t)
        if len(N) > 2 and np.all(N > 0):
            log_N = np.log(N)
            coeffs = np.polyfit(t, log_N, 1)
            exponential_rate = coeffs[0]
        else:
            exponential_rate = 0.0

        # Doubling time
        doubling_time = np.log(2) / exponential_rate if exponential_rate > 0 else None

        # Logistic growth parameters (simplified estimation)
        max_pop = np.max(N)
        carrying_capacity = max_pop * 1.1  # Estimate K as slightly above max observed

        # Logistic rate (simplified)
        logistic_rate = exponential_rate * 0.8  # Typically lower than exponential

        # Determine growth phase
        if exponential_rate > 0.1:
            growth_phase = "exponential"
        elif exponential_rate > 0.01:
            growth_phase = "logistic"
        elif exponential_rate < -0.01:
            growth_phase = "declining"
        else:
            growth_phase = "stable"

        return GrowthRateResult(
            intrinsic_growth_rate=intrinsic_growth_rate,
            doubling_time=doubling_time,
            exponential_rate=exponential_rate,
            logistic_rate=logistic_rate,
            carrying_capacity=carrying_capacity,
            growth_phase=growth_phase,
        )

    def _calculate_survival_single(
        self, df: pd.DataFrame, birth_col: str, death_col: str, status_col: str
    ) -> SurvivalResult:
        """Calculate survival analysis for a single group"""
        # Calculate survival times
        if death_col in df.columns:
            df = df.copy()
            df["survival_time"] = np.where(
                df[status_col] == "dead",
                (
                    pd.to_datetime(df[death_col]) - pd.to_datetime(df[birth_col])
                ).dt.total_seconds()
                / 86400,
                (pd.Timestamp.now() - pd.to_datetime(df[birth_col])).dt.total_seconds()
                / 86400,
            )
        else:
            # All bees are alive, calculate current age
            df = df.copy()
            df["survival_time"] = (
                pd.Timestamp.now() - pd.to_datetime(df[birth_col])
            ).dt.total_seconds() / 86400

        survival_times = np.asarray(df["survival_time"].values)
        is_dead = np.asarray(
            (df[status_col] == "dead").values
            if status_col in df.columns
            else np.zeros(len(df), dtype=bool)
        )

        # Calculate survival curve (Kaplan-Meier style, simplified)
        unique_times = np.sort(np.unique(np.array(survival_times)))
        survival_curve = []

        n_at_risk = len(survival_times)
        survival_prob = 1.0

        for t in unique_times:
            # Number who died at time t
            deaths_at_t = np.sum((survival_times == t) & is_dead)
            # Number censored at time t (alive at time t)
            censored_at_t = np.sum((survival_times == t) & ~np.asarray(is_dead))

            if n_at_risk > 0:
                survival_prob *= 1 - deaths_at_t / n_at_risk

            survival_curve.append((t, survival_prob))
            n_at_risk -= deaths_at_t + censored_at_t

            if n_at_risk <= 0:
                break

        # Median survival
        median_survival = 0.0
        for t, prob in survival_curve:
            if prob <= 0.5:
                median_survival = t
                break
        if median_survival == 0.0 and survival_curve:
            median_survival = survival_curve[-1][
                0
            ]  # Use last time if 50% never reached

        # Life expectancy (mean survival time)
        life_expectancy = np.mean(np.array(survival_times))

        # Hazard rates by age groups
        hazard_rates = {}
        for age_group in ["young", "middle", "old"]:
            if age_group == "young":
                mask = survival_times <= 10
            elif age_group == "middle":
                mask = (survival_times > 10) & (survival_times <= 20)
            else:
                mask = survival_times > 20

            if np.sum(mask) > 0:
                deaths_in_group = np.sum(is_dead & mask)
                hazard_rates[age_group] = float(deaths_in_group / np.sum(mask))
            else:
                hazard_rates[age_group] = float(0.0)

        # Mortality factors (simplified)
        survival_array = np.array(survival_times)
        is_dead_array = np.array(is_dead.astype(float))

        # Check for valid correlation calculation (need variance in both arrays)
        if (
            len(survival_times) > 1
            and np.std(survival_array) > 0
            and np.std(is_dead_array) > 0
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                corr_coef = np.corrcoef(survival_array, is_dead_array)[0, 1]
                age_correlation = corr_coef if not np.isnan(corr_coef) else 0.0
        else:
            age_correlation = 0.0

        mortality_factors = {
            "age": age_correlation,
            "baseline": np.mean(is_dead_array),
        }

        # Age-specific mortality
        age_bins = np.linspace(0, np.max(np.array(survival_times)), 10)
        age_specific_mortality = []
        for i in range(len(age_bins) - 1):
            mask = (survival_times >= age_bins[i]) & (survival_times < age_bins[i + 1])
            if np.sum(mask) > 0:
                mortality_rate = np.mean(is_dead[mask])
                age_specific_mortality.append((age_bins[i], mortality_rate))

        return SurvivalResult(
            median_survival=median_survival,
            survival_curve=survival_curve,
            hazard_rates=hazard_rates,
            mortality_factors={k: float(v) for k, v in mortality_factors.items()},
            life_expectancy=float(life_expectancy),
            age_specific_mortality=age_specific_mortality,
        )

    def clear_cache(self) -> None:
        """Clear analysis cache"""
        self._trend_cache.clear()
        self._growth_cache.clear()
        self._survival_cache.clear()

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of cached analyses"""
        return {
            "trend_analyses": len(self._trend_cache),
            "growth_analyses": len(self._growth_cache),
            "survival_analyses": len(self._survival_cache),
        }
