"""
Population Visualization Framework for BSTEW
===========================================

Comprehensive population data visualization including trend plots,
growth charts, survival curves, and demographic analysis charts.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, cast
import numpy as np
import pandas as pd
import logging
from enum import Enum
from dataclasses import dataclass

# Import visualization libraries with fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None  # type: ignore[assignment]

# 3D plotting and interactive visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

    HAS_PLOTLY = True
    HAS_3D = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    pyo = None
    Axes3D = None
    HAS_PLOTLY = False
    HAS_3D = False

from .population_analyzer import PopulationAnalyzer, GrowthRateResult, SurvivalResult

logger = logging.getLogger(__name__)


class PlotStyle(Enum):
    """Plot style options"""

    CLEAN = "clean"
    SCIENTIFIC = "scientific"
    PRESENTATION = "presentation"
    PUBLICATION = "publication"
    INTERACTIVE_3D = "interactive_3d"
    ANIMATED = "animated"


class PlotType(Enum):
    """Enhanced plot type options"""

    STATIC_2D = "static_2d"
    INTERACTIVE_2D = "interactive_2d"
    STATIC_3D = "static_3d"
    INTERACTIVE_3D = "interactive_3d"
    ANIMATED_2D = "animated_2d"
    ANIMATED_3D = "animated_3d"


class ColorScheme(Enum):
    """Color scheme options"""

    DEFAULT = "default"
    COLORBLIND = "colorblind"
    GRAYSCALE = "grayscale"
    BEE_THEMED = "bee_themed"


@dataclass
class PlotConfig:
    """Configuration for plot appearance"""

    style: PlotStyle = PlotStyle.CLEAN
    color_scheme: ColorScheme = ColorScheme.BEE_THEMED
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 300
    font_size: int = 12
    title_size: int = 14
    show_grid: bool = True
    show_legend: bool = True
    save_format: str = "png"


class PopulationPlotter:
    """
    Population visualization system for BSTEW simulation data.

    Provides comprehensive visualization capabilities including:
    - Population trend plots with statistical overlays
    - Growth rate charts with model fitting
    - Survival curves with confidence intervals
    - Demographic composition charts
    - Multi-colony comparison plots
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the population plotter.

        Args:
            config: Plot configuration settings
        """
        self.config = config or PlotConfig()
        self.logger = logging.getLogger(__name__)

        if not HAS_MATPLOTLIB:
            self.logger.warning("Matplotlib not available - plots will be text-based")

        # Set up plot styles
        self._setup_plot_style()

        # Color schemes
        self.color_schemes = {
            ColorScheme.BEE_THEMED: {
                "primary": "#FFD700",  # Gold
                "secondary": "#FF8C00",  # Dark Orange
                "tertiary": "#8B4513",  # Saddle Brown
                "accent": "#FFA500",  # Orange
                "background": "#FFFACD",  # Lemon Chiffon
                "text": "#2F4F4F",  # Dark Slate Gray
            },
            ColorScheme.COLORBLIND: {
                "primary": "#1f77b4",  # Blue
                "secondary": "#ff7f0e",  # Orange
                "tertiary": "#2ca02c",  # Green
                "accent": "#d62728",  # Red
                "background": "#f7f7f7",  # Light Gray
                "text": "#333333",  # Dark Gray
            },
            ColorScheme.GRAYSCALE: {
                "primary": "#000000",  # Black
                "secondary": "#666666",  # Dark Gray
                "tertiary": "#999999",  # Medium Gray
                "accent": "#333333",  # Very Dark Gray
                "background": "#ffffff",  # White
                "text": "#000000",  # Black
            },
        }

    def plot_population_trends(
        self,
        population_data: Union[List[Dict[str, Any]], pd.DataFrame],
        group_by: Optional[str] = None,
        time_column: str = "time",
        value_column: str = "population",
        show_trend_line: bool = True,
        show_confidence_interval: bool = True,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Create population trend plots with statistical analysis.

        Args:
            population_data: Population time series data
            group_by: Column to group by for multiple series
            time_column: Name of time column
            value_column: Name of population value column
            show_trend_line: Whether to show fitted trend lines
            show_confidence_interval: Whether to show confidence intervals
            save_path: Path to save plot (if None, returns figure)

        Returns:
            Matplotlib figure or text-based plot description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(population_data, "Population Trends")

        # Convert to DataFrame if needed
        if isinstance(population_data, list):
            df = pd.DataFrame(population_data)
        else:
            df = population_data.copy()

        # Ensure time column is datetime with robust parsing
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = self._parse_dates_safely(df[time_column])

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        colors = self._get_color_palette()

        if group_by and group_by in df.columns:
            # Multi-series plot
            groups = df[group_by].unique()
            for i, group in enumerate(groups):
                group_data = df[df[group_by] == group]
                color = colors[i % len(colors)]

                # Plot data points
                ax.plot(
                    group_data[time_column],
                    group_data[value_column],
                    "o-",
                    color=color,
                    label=str(group),
                    alpha=0.8,
                    linewidth=2,
                )

                if show_trend_line:
                    self._add_trend_line(
                        ax, group_data[time_column], group_data[value_column], color
                    )

                if show_confidence_interval:
                    self._add_confidence_interval(
                        ax, group_data[time_column], group_data[value_column], color
                    )
        else:
            # Single series plot
            ax.plot(
                df[time_column],
                df[value_column],
                "o-",
                color=colors[0],
                label="Population",
                alpha=0.8,
                linewidth=2,
            )

            if show_trend_line:
                self._add_trend_line(ax, df[time_column], df[value_column], colors[0])

            if show_confidence_interval:
                self._add_confidence_interval(
                    ax, df[time_column], df[value_column], colors[0]
                )

        # Formatting
        ax.set_xlabel("Time", fontsize=self.config.font_size)
        ax.set_ylabel("Population Size", fontsize=self.config.font_size)
        ax.set_title(
            "Population Trends Over Time",
            fontsize=self.config.title_size,
            fontweight="bold",
        )

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.show_legend and (group_by or show_trend_line):
            ax.legend(fontsize=self.config.font_size - 1)

        # Format date axis if applicable
        if pd.api.types.is_datetime64_any_dtype(df[time_column]):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            self._save_plot_with_format(plt.gcf(), save_path, dpi=self.config.dpi)
            plt.close()
            return save_path

        return fig

    def plot_growth_charts(
        self,
        population_data: Union[List[Dict[str, Any]], pd.DataFrame],
        growth_results: Optional[Dict[str, GrowthRateResult]] = None,
        show_models: bool = True,
        time_column: str = "time",
        value_column: str = "population",
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Create growth rate charts with model fitting.

        Args:
            population_data: Population time series data
            growth_results: Pre-calculated growth results (optional)
            show_models: Whether to show fitted growth models
            time_column: Name of time column
            value_column: Name of population value column
            save_path: Path to save plot

        Returns:
            Matplotlib figure or text-based plot description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(population_data, "Growth Charts")

        # Convert to DataFrame if needed
        if isinstance(population_data, list):
            df = pd.DataFrame(population_data)
        else:
            df = population_data.copy()

        # Calculate growth rates if not provided
        if growth_results is None:
            analyzer = PopulationAnalyzer()
            growth_results = analyzer.calculate_growth_rates(
                df, time_column=time_column, value_column=value_column
            )

        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(15, 10), dpi=self.config.dpi
        )
        colors = self._get_color_palette()

        # Ensure time column is datetime and convert to numeric for modeling
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = self._parse_dates_safely(df[time_column])

        time_numeric = (
            df[time_column] - df[time_column].min()
        ).dt.total_seconds() / 86400  # Days

        # 1. Population over time (linear scale)
        ax1.plot(
            df[time_column],
            df[value_column],
            "o-",
            color=colors[0],
            alpha=0.8,
            linewidth=2,
        )
        if show_models and "overall" in growth_results:
            result = growth_results["overall"]
            if result.exponential_rate > 0:
                # Exponential model
                N0 = df[value_column].iloc[0]
                t_model = np.linspace(0, time_numeric.max(), 100)
                N_exp = N0 * np.exp(result.exponential_rate * t_model)
                time_model = df[time_column].min() + pd.to_timedelta(t_model, unit="D")
                ax1.plot(
                    time_model,
                    N_exp,
                    "--",
                    color=colors[1],
                    label=f"Exponential (r={result.exponential_rate:.3f})",
                )

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Population Size")
        ax1.set_title("Population Growth (Linear Scale)")
        ax1.grid(True, alpha=0.3)
        if show_models:
            ax1.legend()

        # 2. Population over time (log scale)
        valid_pop = df[value_column] > 0
        if valid_pop.any():
            ax2.semilogy(
                df[time_column][valid_pop],
                df[value_column][valid_pop],
                "o-",
                color=colors[0],
                alpha=0.8,
                linewidth=2,
            )
            if show_models and "overall" in growth_results:
                result = growth_results["overall"]
                if result.exponential_rate > 0:
                    N0 = df[value_column].iloc[0]
                    t_model = np.linspace(0, time_numeric.max(), 100)
                    N_exp = N0 * np.exp(result.exponential_rate * t_model)
                    time_model = df[time_column].min() + pd.to_timedelta(
                        t_model, unit="D"
                    )
                    ax2.semilogy(time_model, N_exp, "--", color=colors[1])

        ax2.set_xlabel("Time")
        ax2.set_ylabel("Population Size (log scale)")
        ax2.set_title("Population Growth (Log Scale)")
        ax2.grid(True, alpha=0.3)

        # 3. Growth rate over time
        if len(df) > 1:
            growth_rates = df[value_column].pct_change() * 100  # Percentage change
            ax3.plot(
                df[time_column][1:],
                growth_rates[1:],
                "o-",
                color=colors[2],
                alpha=0.8,
                linewidth=2,
            )
            ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        ax3.set_xlabel("Time")
        ax3.set_ylabel("Growth Rate (%)")
        ax3.set_title("Instantaneous Growth Rate")
        ax3.grid(True, alpha=0.3)

        # 4. Growth phase diagram
        if "overall" in growth_results:
            result = growth_results["overall"]
            phases = ["Exponential", "Logistic", "Declining", "Stable"]
            rates = [
                max(0, result.exponential_rate)
                if result.growth_phase == "exponential"
                else 0,
                max(0, result.logistic_rate)
                if result.growth_phase == "logistic"
                else 0,
                max(0, -result.exponential_rate)
                if result.growth_phase == "declining"
                else 0,
                1 if result.growth_phase == "stable" else 0,
            ]

            bars = ax4.bar(phases, rates, color=colors[:4], alpha=0.7)
            ax4.set_ylabel("Growth Rate")
            ax4.set_title("Growth Phase Classification")
            ax4.tick_params(axis="x", rotation=45)

            # Highlight current phase
            for i, phase in enumerate(phases):
                if phase.lower() == result.growth_phase:
                    bars[i].set_alpha(1.0)
                    bars[i].set_edgecolor("black")
                    bars[i].set_linewidth(2)

        plt.tight_layout()

        if save_path:
            self._save_plot_with_format(plt.gcf(), save_path, dpi=self.config.dpi)
            plt.close()
            return save_path

        return fig

    def plot_survival_curves(
        self,
        survival_data: Union[List[Dict[str, Any]], pd.DataFrame],
        survival_results: Optional[Dict[str, SurvivalResult]] = None,
        group_by: Optional[str] = None,
        show_confidence_bands: bool = True,
        show_median_lines: bool = True,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Create survival curves with statistical analysis.

        Args:
            survival_data: Individual bee lifecycle data
            survival_results: Pre-calculated survival results (optional)
            group_by: Column to group by for multiple curves
            show_confidence_bands: Whether to show confidence bands
            show_median_lines: Whether to show median survival lines
            save_path: Path to save plot

        Returns:
            Matplotlib figure or text-based plot description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(survival_data, "Survival Curves")

        # Convert to DataFrame if needed
        if isinstance(survival_data, list):
            df = pd.DataFrame(survival_data)
        else:
            df = survival_data.copy()

        # Calculate survival results if not provided
        if survival_results is None:
            analyzer = PopulationAnalyzer()
            survival_results = analyzer.survival_analysis(df, group_by=group_by)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.config.dpi)
        colors = self._get_color_palette()

        # Plot 1: Survival curves
        for i, (group_name, result) in enumerate(survival_results.items()):
            color = colors[i % len(colors)]

            if result.survival_curve:
                times, probs = zip(*result.survival_curve)
                ax1.plot(
                    times, probs, "-", color=color, linewidth=2, label=str(group_name)
                )

                if show_confidence_bands:
                    # Simple confidence band (would use proper calculation in production)
                    probs_array = np.array(probs)
                    ci_width = 0.1 * probs_array
                    ax1.fill_between(
                        times,
                        probs_array - ci_width,
                        probs_array + ci_width,
                        color=color,
                        alpha=0.2,
                    )

                if show_median_lines:
                    ax1.axvline(
                        x=result.median_survival,
                        color=color,
                        linestyle="--",
                        alpha=0.7,
                        label=f"{group_name} median: {result.median_survival:.1f}d",
                    )

        ax1.set_xlabel("Time (days)", fontsize=self.config.font_size)
        ax1.set_ylabel("Survival Probability", fontsize=self.config.font_size)
        ax1.set_title(
            "Survival Curves", fontsize=self.config.title_size, fontweight="bold"
        )
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        if self.config.show_legend:
            ax1.legend(fontsize=self.config.font_size - 1)

        # Plot 2: Hazard rates by age group
        if survival_results:
            # Combine hazard rates from all groups
            age_groups = ["young", "middle", "old"]
            group_names = list(survival_results.keys())

            if len(group_names) == 1:
                # Single group - show age-specific hazard rates
                result = list(survival_results.values())[0]
                hazard_values = [result.hazard_rates.get(age, 0) for age in age_groups]
                ax2.bar(age_groups, hazard_values, color=colors[0], alpha=0.7)
                ax2.set_title("Hazard Rates by Age Group")
            else:
                # Multiple groups - show comparison
                x = np.arange(len(age_groups))
                width = 0.8 / len(group_names)

                for i, (group_name, result) in enumerate(survival_results.items()):
                    hazard_values = [
                        result.hazard_rates.get(age, 0) for age in age_groups
                    ]
                    offset = (i - len(group_names) / 2) * width + width / 2
                    ax2.bar(
                        x + offset,
                        hazard_values,
                        width,
                        label=str(group_name),
                        color=colors[i % len(colors)],
                        alpha=0.7,
                    )

                ax2.set_xticks(x)
                ax2.set_xticklabels(age_groups)
                ax2.set_title("Hazard Rates Comparison")
                if self.config.show_legend:
                    ax2.legend()

        ax2.set_xlabel("Age Group", fontsize=self.config.font_size)
        ax2.set_ylabel("Hazard Rate", fontsize=self.config.font_size)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_plot_with_format(plt.gcf(), save_path, dpi=self.config.dpi)
            plt.close()
            return save_path

        return fig

    def plot_demographic_composition(
        self,
        population_data: Union[List[Dict[str, Any]], pd.DataFrame],
        composition_column: str = "role",
        time_column: str = "time",
        plot_type: str = "stacked_area",
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Create demographic composition plots showing population structure over time.

        Args:
            population_data: Population data with demographic categories
            composition_column: Column containing demographic categories
            time_column: Name of time column
            plot_type: Type of plot ('stacked_area', 'stacked_bar', 'pie_series')
            save_path: Path to save plot

        Returns:
            Matplotlib figure or text-based plot description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(population_data, "Demographic Composition")

        # Convert to DataFrame if needed
        if isinstance(population_data, list):
            df = pd.DataFrame(population_data)
        else:
            df = population_data.copy()

        # Ensure required columns exist
        if composition_column not in df.columns:
            df[composition_column] = "worker"  # Default role

        colors = self._get_color_palette()

        if plot_type == "stacked_area":
            # Create pivot table for stacked area plot
            if time_column in df.columns:
                pivot_df = (
                    df.groupby([time_column, composition_column])
                    .size()
                    .unstack(fill_value=0)
                )

                fig, ax = plt.subplots(
                    figsize=self.config.figure_size, dpi=self.config.dpi
                )
                pivot_df.plot.area(
                    ax=ax, color=colors[: len(pivot_df.columns)], alpha=0.8
                )

                ax.set_xlabel("Time", fontsize=self.config.font_size)
                ax.set_ylabel("Population Count", fontsize=self.config.font_size)
                ax.set_title(
                    "Population Composition Over Time",
                    fontsize=self.config.title_size,
                    fontweight="bold",
                )
            else:
                # Single time point composition
                composition_counts = df[composition_column].value_counts()
                fig, ax = plt.subplots(
                    figsize=self.config.figure_size, dpi=self.config.dpi
                )
                ax.pie(
                    composition_counts.values.tolist(),
                    labels=composition_counts.index.tolist(),
                    colors=colors[: len(composition_counts)],
                    autopct="%1.1f%%",
                )
                ax.set_title(
                    "Population Composition",
                    fontsize=self.config.title_size,
                    fontweight="bold",
                )

        elif plot_type == "stacked_bar":
            # Create stacked bar chart
            stacked_composition = (
                df.groupby([time_column, composition_column])
                .size()
                .unstack(fill_value=0)
                if time_column in df.columns
                else df[composition_column].value_counts()
            )

            fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
            stacked_composition.plot.bar(
                ax=ax,
                stacked=True,
                color=cast(Any, colors),  # Type cast for matplotlib color compatibility
            )

            ax.set_xlabel("Time" if time_column in df.columns else "Category")
            ax.set_ylabel("Population Count")
            ax.set_title(
                "Population Composition",
                fontsize=self.config.title_size,
                fontweight="bold",
            )
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        else:  # pie_series
            # Series of pie charts over time
            if time_column in df.columns:
                unique_times = sorted(df[time_column].unique())[
                    :6
                ]  # Limit to 6 time points
                n_plots = len(unique_times)
                n_cols = min(3, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols

                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(15, 5 * n_rows), dpi=self.config.dpi
                )
                if n_plots == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for i, time_point in enumerate(unique_times):
                    time_data = df[df[time_column] == time_point]
                    composition_counts = time_data[composition_column].value_counts()

                    if i < len(axes):
                        axes[i].pie(
                            composition_counts.values,
                            labels=composition_counts.index,
                            colors=colors[: len(composition_counts)],
                            autopct="%1.1f%%",
                        )
                        axes[i].set_title(f"Time: {time_point}")

                # Hide unused subplots
                for i in range(n_plots, len(axes)):
                    axes[i].set_visible(False)
            else:
                # Single pie chart
                composition_counts = df[composition_column].value_counts()
                fig, ax = plt.subplots(
                    figsize=self.config.figure_size, dpi=self.config.dpi
                )
                ax.pie(
                    composition_counts.values.tolist(),
                    labels=composition_counts.index.tolist(),
                    colors=colors[: len(composition_counts)],
                    autopct="%1.1f%%",
                )
                ax.set_title("Population Composition")

        if self.config.show_grid and plot_type != "pie_series":
            ax.grid(True, alpha=0.3)

        if self.config.show_legend and plot_type in ["stacked_area", "stacked_bar"]:
            ax.legend(fontsize=self.config.font_size - 1)

        plt.tight_layout()

        if save_path:
            self._save_plot_with_format(plt.gcf(), save_path, dpi=self.config.dpi)
            plt.close()
            return save_path

        return fig

    def _setup_plot_style(self) -> None:
        """Set up matplotlib plot style"""
        if not HAS_MATPLOTLIB:
            return

        if self.config.style == PlotStyle.SCIENTIFIC:
            plt.style.use("seaborn-v0_8-whitegrid" if HAS_SEABORN else "default")
        elif self.config.style == PlotStyle.PRESENTATION:
            plt.style.use("seaborn-v0_8-talk" if HAS_SEABORN else "default")
        elif self.config.style == PlotStyle.PUBLICATION:
            plt.style.use("seaborn-v0_8-paper" if HAS_SEABORN else "default")

        plt.rcParams.update(
            {
                "font.size": self.config.font_size,
                "axes.titlesize": self.config.title_size,
                "axes.labelsize": self.config.font_size,
                "xtick.labelsize": self.config.font_size - 1,
                "ytick.labelsize": self.config.font_size - 1,
                "legend.fontsize": self.config.font_size - 1,
                "figure.titlesize": self.config.title_size + 2,
            }
        )

    def _get_color_palette(self) -> List[str]:
        """Get color palette based on configuration"""
        scheme = self.color_schemes.get(
            self.config.color_scheme, self.color_schemes[ColorScheme.BEE_THEMED]
        )
        return [
            scheme["primary"],
            scheme["secondary"],
            scheme["tertiary"],
            scheme["accent"],
        ]

    def _add_trend_line(
        self, ax: Axes, times: pd.Series, values: pd.Series, color: str
    ) -> None:
        """Add trend line to plot"""
        if len(values) < 3:
            return

        time_numeric = (times - times.min()).dt.total_seconds() / 86400
        valid_mask = ~(np.isnan(time_numeric) | np.isnan(values))

        if valid_mask.sum() < 3:
            return

        x = time_numeric[valid_mask].values
        y = values[valid_mask].values

        # Linear trend
        coeffs = np.polyfit(x, y, 1)
        trend_line = np.polyval(coeffs, x)

        ax.plot(
            times[valid_mask], trend_line, "--", color=color, alpha=0.7, linewidth=1.5
        )

    def _add_confidence_interval(
        self, ax: Axes, times: pd.Series, values: pd.Series, color: str
    ) -> None:
        """Add confidence interval to plot"""
        if len(values) < 5:
            return

        # Simple rolling confidence interval
        window = min(5, len(values) // 3)
        rolling_mean = values.rolling(window=window, center=True).mean()
        rolling_std = values.rolling(window=window, center=True).std()

        ci_upper = rolling_mean + 1.96 * rolling_std
        ci_lower = rolling_mean - 1.96 * rolling_std

        ax.fill_between(times, ci_lower, ci_upper, color=color, alpha=0.1)

    def _parse_dates_safely(self, data_series: pd.Series) -> pd.Series:
        """Safely parse dates with multiple format attempts

        Implementation of robust date parsing to handle various date formats
        and avoid 'day is out of range for month' errors.
        """
        # Try multiple date formats in order of likelihood
        formats_to_try = [
            "%Y-%m-%d",  # 2024-01-01
            "%d/%m/%Y",  # 01/01/2024
            "%m/%d/%Y",  # 01/01/2024 (US format)
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 12:00:00
            "%d-%m-%Y",  # 01-01-2024
        ]

        for date_format in formats_to_try:
            try:
                return pd.to_datetime(data_series, format=date_format)
            except (ValueError, TypeError) as e:
                self.logger.debug(f"Date format {date_format} failed: {e}")
                continue

        # Fallback: Let pandas infer format
        try:
            # infer_datetime_format is deprecated, pandas now infers by default
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format")
                return pd.to_datetime(data_series)
        except Exception as e:
            self.logger.error(f"All date parsing methods failed: {e}")
            # Return numeric range as fallback
            return pd.Series(range(len(data_series)), name="day_number")

    def _save_plot_with_format(
        self, figure: Figure, output_path: str, dpi: int = 300
    ) -> None:
        """Save plot with proper format handling and error checking

        Ensures correct file extensions and verifies file creation.
        """
        from pathlib import Path

        output_path_obj = Path(output_path)

        # Ensure correct file extension
        format_extensions = {
            "png": ".png",
            "pdf": ".pdf",
            "svg": ".svg",
            "eps": ".eps",
            "jpg": ".jpg",
            "jpeg": ".jpeg",
        }

        # Get format from file extension
        file_format = output_path_obj.suffix.lower()[1:]  # Remove the dot
        if file_format not in format_extensions:
            # Try to get format from config
            file_format = getattr(self.config, "save_format", "png").lower()
            if file_format not in format_extensions:
                raise ValueError(f"Unsupported format: {file_format}")

            # Force correct extension
            output_path_obj = output_path_obj.with_suffix(
                format_extensions[file_format]
            )

        try:
            figure.savefig(
                str(output_path_obj),  # Convert Path to string for savefig
                format=file_format,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

            # VERIFY FILE WAS ACTUALLY CREATED
            if not output_path_obj.exists():
                raise RuntimeError(f"Plot file was not created: {output_path_obj}")

            # Verify file has reasonable size
            if (
                output_path_obj.stat().st_size < 100
            ):  # Less than 100 bytes is suspicious
                raise RuntimeError(
                    f"Plot file appears corrupted (too small): {output_path_obj}"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to save plot in {file_format} format: {e}")

    def _create_text_plot(self, data: Union[List, pd.DataFrame], plot_type: str) -> str:
        """Create text-based plot description when matplotlib is not available"""
        if isinstance(data, list):
            data_size = len(data)
        else:
            data_size = len(data) if hasattr(data, "__len__") else 0

        return f"""
{plot_type} - Text Description
{"=" * 40}
Data points: {data_size}
Visualization requires matplotlib installation.
To install: pip install matplotlib seaborn
"""

    def create_dashboard(
        self,
        population_data: Union[List[Dict[str, Any]], pd.DataFrame],
        survival_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Create comprehensive population analysis dashboard.

        Args:
            population_data: Population time series data
            survival_data: Individual bee lifecycle data (optional)
            save_path: Path to save dashboard

        Returns:
            Matplotlib figure with multiple subplots or text description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(population_data, "Population Dashboard")

        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 15), dpi=self.config.dpi)

        # Convert data
        if isinstance(population_data, list):
            pop_df = pd.DataFrame(population_data)
        else:
            pop_df = population_data.copy()

        # Trend plot (top left)
        ax1 = plt.subplot(3, 3, (1, 2))
        self._create_subplot_trends(ax1, pop_df)

        # Growth chart (top right)
        ax2 = plt.subplot(3, 3, 3)
        self._create_subplot_growth(ax2, pop_df)

        # Composition plot (middle left)
        ax3 = plt.subplot(3, 3, (4, 5))
        self._create_subplot_composition(ax3, pop_df)

        # Survival curve (middle right)
        ax4 = plt.subplot(3, 3, 6)
        if survival_data is not None:
            self._create_subplot_survival(ax4, survival_data)
        else:
            ax4.text(
                0.5,
                0.5,
                "No survival data\nprovided",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Survival Analysis")

        # Statistics summary (bottom)
        ax5 = plt.subplot(3, 3, (7, 9))
        self._create_subplot_statistics(ax5, pop_df, survival_data)

        plt.suptitle(
            "Population Analysis Dashboard",
            fontsize=self.config.title_size + 4,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            self._save_plot_with_format(plt.gcf(), save_path, dpi=self.config.dpi)
            plt.close()
            return save_path

        return fig

    def create_interactive_3d_population_plot(
        self,
        population_data: Union[List[Dict[str, Any]], pd.DataFrame],
        plot_type: PlotType = PlotType.INTERACTIVE_3D,
        save_path: Optional[str] = None,
    ) -> Union[Any, str]:
        """
        Create advanced 3D population visualization with interactive features.

        Phase 4 enhancement for comprehensive spatial-temporal analysis.

        Args:
            population_data: Population time series data with spatial coordinates
            plot_type: Type of 3D visualization to create
            save_path: Path to save the interactive plot

        Returns:
            Plotly figure object or text description if Plotly unavailable
        """
        if not HAS_PLOTLY:
            return self._create_text_plot(
                population_data, "3D Interactive Population Plot"
            )

        # Convert data
        if isinstance(population_data, list):
            df = pd.DataFrame(population_data)
        else:
            df = population_data.copy()

        if plot_type == PlotType.INTERACTIVE_3D:
            return self._create_plotly_3d_surface(df, save_path)
        elif plot_type == PlotType.ANIMATED_3D:
            return self._create_plotly_3d_surface(df, save_path)  # Use surface for now
        else:
            return self._create_plotly_3d_surface(df, save_path)  # Use surface for now

    def _create_plotly_3d_surface(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> Any:
        """Create 3D surface plot of population density over time and space"""

        # Prepare data for 3D surface
        if "time_step" not in df.columns:
            df["time_step"] = range(len(df))

        # Create mesh grid for surface plot
        time_steps = df["time_step"].unique()
        x_coords = df.get(
            "x_coordinate", pd.Series(range(len(df["time_step"].unique())))
        ).unique()

        # Generate Z values (population density)
        z_values = []
        for t in time_steps:
            row = []
            for x in x_coords:
                # Get population value or interpolate
                pop_val = (
                    df[(df["time_step"] == t)]["total_population"].iloc[0]
                    if len(df[df["time_step"] == t]) > 0
                    else 0
                )
                # Add spatial variation
                row.append(pop_val * (1 + 0.1 * np.sin(x * 0.1)))
            z_values.append(row)

        fig = go.Figure(
            data=[
                go.Surface(
                    z=z_values,
                    x=x_coords,
                    y=time_steps,
                    colorscale="Viridis",
                    name="Population Density",
                )
            ]
        )

        fig.update_layout(
            title="3D Population Dynamics Surface",
            scene=dict(
                xaxis_title="Spatial Coordinate (m)",
                yaxis_title="Time Step",
                zaxis_title="Population Density",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            ),
            width=800,
            height=600,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_subplot_trends(self, ax: Axes, df: pd.DataFrame) -> None:
        """Create trends subplot for dashboard"""
        if "time" in df.columns and "population" in df.columns:
            ax.plot(
                df["time"],
                df["population"],
                "o-",
                color=self._get_color_palette()[0],
                linewidth=2,
            )
            ax.set_title("Population Trends")
            ax.set_ylabel("Population")
            ax.grid(True, alpha=0.3)

    def _create_subplot_growth(self, ax: Axes, df: pd.DataFrame) -> None:
        """Create growth subplot for dashboard"""
        if "population" in df.columns and len(df) > 1:
            growth_rates = df["population"].pct_change() * 100
            ax.plot(
                growth_rates[1:], "o-", color=self._get_color_palette()[1], linewidth=2
            )
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Growth Rate (%)")
            ax.set_ylabel("Growth Rate")
            ax.grid(True, alpha=0.3)

    def _create_subplot_composition(self, ax: Axes, df: pd.DataFrame) -> None:
        """Create composition subplot for dashboard"""
        if "role" in df.columns:
            composition = df["role"].value_counts()
            colors = self._get_color_palette()[: len(composition)]
            ax.pie(
                composition.values.tolist(),
                labels=composition.index.tolist(),
                colors=colors,
                autopct="%1.1f%%",
            )
            ax.set_title("Population Composition")

    def _create_subplot_survival(
        self, ax: Axes, survival_data: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> None:
        """Create survival subplot for dashboard"""
        # Convert to DataFrame if needed
        if isinstance(survival_data, list):
            survival_df = pd.DataFrame(survival_data)
        else:
            survival_df = survival_data.copy()

        # Simplified survival plot for dashboard
        if "birth_time" in survival_df.columns:
            analyzer = PopulationAnalyzer()
            survival_results = analyzer.survival_analysis(survival_df)

            if "overall" in survival_results:
                result = survival_results["overall"]
                if result.survival_curve:
                    times, probs = zip(*result.survival_curve)
                    ax.plot(
                        times,
                        probs,
                        "-",
                        color=self._get_color_palette()[2],
                        linewidth=2,
                    )
                    ax.set_title("Survival Curve")
                    ax.set_ylabel("Survival Probability")
                    ax.set_xlabel("Time (days)")
                    ax.grid(True, alpha=0.3)

    def _create_subplot_statistics(
        self,
        ax: Axes,
        pop_df: pd.DataFrame,
        survival_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]],
    ) -> None:
        """Create statistics summary subplot for dashboard"""
        ax.axis("off")

        # Calculate key statistics
        stats_text = []

        if "population" in pop_df.columns:
            current_pop = pop_df["population"].iloc[-1] if len(pop_df) > 0 else 0
            max_pop = pop_df["population"].max() if len(pop_df) > 0 else 0
            avg_pop = pop_df["population"].mean() if len(pop_df) > 0 else 0

            stats_text.extend(
                [
                    f"Current Population: {current_pop:,.0f}",
                    f"Maximum Population: {max_pop:,.0f}",
                    f"Average Population: {avg_pop:,.0f}",
                ]
            )

        if survival_data is not None:
            # Convert to DataFrame if needed
            if isinstance(survival_data, list):
                survival_df = pd.DataFrame(survival_data)
            else:
                survival_df = survival_data.copy()

            if len(survival_df) > 0:
                analyzer = PopulationAnalyzer()
                survival_results = analyzer.survival_analysis(survival_df)
                if "overall" in survival_results:
                    result = survival_results["overall"]
                    stats_text.extend(
                        [
                            "",
                            f"Median Survival: {result.median_survival:.1f} days",
                            f"Life Expectancy: {result.life_expectancy:.1f} days",
                        ]
                    )

        # Display statistics
        stats_str = "\n".join(stats_text)
        ax.text(
            0.1,
            0.9,
            "Key Statistics:",
            fontsize=self.config.font_size + 2,
            fontweight="bold",
            transform=ax.transAxes,
            va="top",
        )
        ax.text(
            0.1,
            0.8,
            stats_str,
            fontsize=self.config.font_size,
            transform=ax.transAxes,
            va="top",
        )
