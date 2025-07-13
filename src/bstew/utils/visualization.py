"""
Visualization utilities for BSTEW
=================================

Matplotlib and Plotly-based plotting for simulation results and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..spatial.patches import HabitatType


class BstewPlotter:
    """
    Main plotting class for BSTEW visualizations.

    Supports:
    - Population dynamics plots
    - Spatial landscape visualization
    - Resource distribution maps
    - Colony health indicators
    - Foraging activity heatmaps
    """

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """Initialize plotter with style settings"""
        plt.style.use(style)
        self.figsize = figsize

        # Color schemes
        self.colony_colors = {
            "total_bees": "#1f77b4",
            "workers": "#ff7f0e",
            "foragers": "#2ca02c",
            "brood": "#d62728",
            "queens": "#9467bd",
            "drones": "#8c564b",
        }

        self.habitat_colors = {
            HabitatType.GRASSLAND: "#90EE90",
            HabitatType.CROPLAND: "#FFD700",
            HabitatType.WILDFLOWER: "#FF69B4",
            HabitatType.WOODLAND: "#8B4513",
            HabitatType.HEDGEROW: "#228B22",
            HabitatType.URBAN: "#808080",
            HabitatType.WATER: "#0000FF",
            HabitatType.BARE_SOIL: "#F5F5DC",
            HabitatType.ROAD: "#404040",
            HabitatType.BUILDING: "#FF0000",
        }

        self.resource_colormap = "viridis"

    def plot_population_dynamics(
        self,
        model_data: pd.DataFrame,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """Plot population dynamics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Colony Population Dynamics", fontsize=16, fontweight="bold")

        # Total population
        if "Total_Bees" in model_data.columns:
            axes[0, 0].plot(
                model_data["Day"],
                model_data["Total_Bees"],
                color=self.colony_colors["total_bees"],
                linewidth=2,
            )
            axes[0, 0].set_title("Total Adult Bee Population")
            axes[0, 0].set_xlabel("Day")
            axes[0, 0].set_ylabel("Number of Bees")
            axes[0, 0].grid(True, alpha=0.3)

        # Brood population
        if "Total_Brood" in model_data.columns:
            axes[0, 1].plot(
                model_data["Day"],
                model_data["Total_Brood"],
                color=self.colony_colors["brood"],
                linewidth=2,
            )
            axes[0, 1].set_title("Brood Population")
            axes[0, 1].set_xlabel("Day")
            axes[0, 1].set_ylabel("Number of Brood")
            axes[0, 1].grid(True, alpha=0.3)

        # Resource levels
        if "Total_Honey" in model_data.columns:
            axes[1, 0].plot(
                model_data["Day"],
                model_data["Total_Honey"],
                color="gold",
                linewidth=2,
                label="Honey",
            )
            axes[1, 0].set_title("Colony Resources")
            axes[1, 0].set_xlabel("Day")
            axes[1, 0].set_ylabel("Resource Amount (mg)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Colony health indicator
        if "Active_Colonies" in model_data.columns:
            axes[1, 1].plot(
                model_data["Day"],
                model_data["Active_Colonies"],
                color="red",
                linewidth=2,
            )
            axes[1, 1].set_title("Active Colonies")
            axes[1, 1].set_xlabel("Day")
            axes[1, 1].set_ylabel("Number of Active Colonies")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_seasonal_patterns(
        self,
        model_data: pd.DataFrame,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """Plot seasonal patterns in population and resources"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle("Seasonal Patterns", fontsize=16, fontweight="bold")

        # Add season backgrounds
        for ax in axes:
            self._add_season_backgrounds(ax, len(model_data))

        # Population with seasonal overlay
        if "Total_Bees" in model_data.columns:
            axes[0].plot(
                model_data["Day"],
                model_data["Total_Bees"],
                color=self.colony_colors["total_bees"],
                linewidth=2,
            )

            if "Temperature" in model_data.columns:
                # Add temperature on secondary axis
                ax_temp = axes[0].twinx()
                ax_temp.plot(
                    model_data["Day"],
                    model_data["Temperature"],
                    color="red",
                    alpha=0.7,
                    linestyle="--",
                    label="Temperature",
                )
                ax_temp.set_ylabel("Temperature (°C)", color="red")
                ax_temp.tick_params(axis="y", labelcolor="red")

            axes[0].set_title("Population and Temperature")
            axes[0].set_ylabel("Number of Bees")
            axes[0].grid(True, alpha=0.3)

        # Resource availability
        if "Available_Nectar" in model_data.columns:
            axes[1].plot(
                model_data["Day"],
                model_data["Available_Nectar"],
                color="purple",
                linewidth=2,
                label="Available Nectar",
            )
            axes[1].set_title("Landscape Resource Availability")
            axes[1].set_xlabel("Day of Year")
            axes[1].set_ylabel("Available Resources (mg)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def _add_season_backgrounds(self, ax: Any, max_day: int) -> None:
        """Add seasonal background colors to plot"""
        season_ranges = [
            (1, 79, "#E6F3FF", "Winter"),  # Light blue
            (80, 171, "#E6FFE6", "Spring"),  # Light green
            (172, 265, "#FFF9E6", "Summer"),  # Light yellow
            (266, 354, "#FFE6CC", "Autumn"),  # Light orange
            (355, max_day, "#E6F3FF", "Winter"),  # Light blue
        ]

        for start, end, color, label in season_ranges:
            if start <= max_day:
                end = min(end, max_day)
                ax.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color=color,
                    label=label if start < 100 else "",
                )

    def plot_landscape_map(
        self,
        landscape_grid: Any,
        overlay_data: Optional[Dict[str, np.ndarray]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """Plot landscape map with optional overlays"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle("Landscape Analysis", fontsize=16, fontweight="bold")

        # Habitat map
        habitat_array = self._create_habitat_image(landscape_grid.habitat_grid)
        axes[0].imshow(habitat_array, origin="lower")
        axes[0].set_title("Habitat Types")
        axes[0].set_xlabel("X coordinate")
        axes[0].set_ylabel("Y coordinate")

        # Create habitat legend
        legend_elements = []
        for habitat_type, color in self.habitat_colors.items():
            legend_elements.append(
                patches.Patch(
                    color=tuple(np.array(color) / 255.0), label=habitat_type.value
                )
            )
        axes[0].legend(
            handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left"
        )

        # Resource overlay or connectivity
        if overlay_data and "resources" in overlay_data:
            im = axes[1].imshow(
                overlay_data["resources"],
                origin="lower",
                cmap=self.resource_colormap,
                alpha=0.8,
            )
            axes[1].set_title("Resource Distribution")
            plt.colorbar(im, ax=axes[1], label="Resource Density")
        else:
            # Show connectivity or default overlay
            axes[1].imshow(habitat_array, origin="lower", alpha=0.5)
            axes[1].set_title("Landscape Structure")

        axes[1].set_xlabel("X coordinate")
        axes[1].set_ylabel("Y coordinate")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def _create_habitat_image(self, habitat_grid: np.ndarray) -> np.ndarray:
        """Convert habitat grid to RGB image"""
        height, width = habitat_grid.shape
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Map habitat types to colors
        habitat_type_to_id = {ht: i for i, ht in enumerate(HabitatType)}

        for habitat_type, color in self.habitat_colors.items():
            habitat_id = habitat_type_to_id[habitat_type]
            mask = habitat_grid == habitat_id
            rgb_array[mask] = color

        return rgb_array

    def plot_foraging_activity(
        self,
        agent_data: pd.DataFrame,
        landscape_grid: Any,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """Plot foraging activity heatmap"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Foraging Activity Analysis", fontsize=16, fontweight="bold")

        # Filter forager data
        forager_data = agent_data[agent_data["Role"] == "forager"]

        if not forager_data.empty:
            # Activity heatmap
            # Note: This would need actual spatial coordinates from agent data
            # For now, create example visualization

            # Forager energy distribution
            axes[0].hist(forager_data["Energy"], bins=30, alpha=0.7, color="orange")
            axes[0].set_title("Forager Energy Distribution")
            axes[0].set_xlabel("Energy Level")
            axes[0].set_ylabel("Number of Foragers")
            axes[0].grid(True, alpha=0.3)

            # Age distribution by role
            if "Age" in forager_data.columns:
                axes[1].hist(forager_data["Age"], bins=20, alpha=0.7, color="green")
                axes[1].set_title("Forager Age Distribution")
                axes[1].set_xlabel("Age (days)")
                axes[1].set_ylabel("Number of Foragers")
                axes[1].grid(True, alpha=0.3)
        else:
            axes[0].text(
                0.5,
                0.5,
                "No forager data available",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[1].text(
                0.5,
                0.5,
                "No forager data available",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig


class InteractivePlotter:
    """Interactive Plotly-based visualizations"""

    def __init__(self) -> None:
        self.colors = {
            "total_bees": "#1f77b4",
            "workers": "#ff7f0e",
            "foragers": "#2ca02c",
            "brood": "#d62728",
            "resources": "#9467bd",
        }

    def create_population_dashboard(self, model_data: pd.DataFrame) -> go.Figure:
        """Create interactive population dashboard"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Total Population",
                "Brood Development",
                "Resource Levels",
                "Colony Health",
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Total population with temperature overlay
        if "Total_Bees" in model_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_data["Day"],
                    y=model_data["Total_Bees"],
                    name="Total Bees",
                    line=dict(color=self.colors["total_bees"]),
                ),
                row=1,
                col=1,
            )

        if "Temperature" in model_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_data["Day"],
                    y=model_data["Temperature"],
                    name="Temperature",
                    line=dict(color="red", dash="dash"),
                    yaxis="y2",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        # Brood development
        if "Total_Brood" in model_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_data["Day"],
                    y=model_data["Total_Brood"],
                    name="Brood",
                    line=dict(color=self.colors["brood"]),
                ),
                row=1,
                col=2,
            )

        # Resources
        if "Total_Honey" in model_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_data["Day"],
                    y=model_data["Total_Honey"],
                    name="Honey",
                    line=dict(color="gold"),
                ),
                row=2,
                col=1,
            )

        # Colony health
        if "Active_Colonies" in model_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_data["Day"],
                    y=model_data["Active_Colonies"],
                    name="Active Colonies",
                    line=dict(color="red"),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title_text="Colony Population Dashboard", showlegend=True, height=600
        )

        return fig

    def create_resource_heatmap(self, landscape_grid: Any) -> go.Figure:
        """Create interactive resource heatmap"""
        # Calculate resource distribution
        resource_map = np.zeros(landscape_grid.habitat_grid.shape)

        for patch in landscape_grid.patches.values():
            grid_x = int(patch.x / landscape_grid.cell_size)
            grid_y = int(patch.y / landscape_grid.cell_size)

            if (
                0 <= grid_x < landscape_grid.width
                and 0 <= grid_y < landscape_grid.height
            ):
                resource_map[grid_y, grid_x] = patch.get_resource_quality()

        fig = go.Figure(
            data=go.Heatmap(
                z=resource_map,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Resource Quality"),
            )
        )

        fig.update_layout(
            title="Landscape Resource Distribution",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
        )

        return fig

    def create_foraging_animation(self, agent_data: pd.DataFrame) -> go.Figure:
        """Create animated foraging activity plot"""
        # Filter forager data and group by time steps
        forager_data = agent_data[agent_data["Role"] == "forager"]

        if forager_data.empty:
            # Create empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No forager data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        # Create animation frames
        frames = []
        steps = (
            sorted(forager_data["Step"].unique())
            if "Step" in forager_data.columns
            else [0]
        )

        for step in steps:
            step_data = (
                forager_data[forager_data["Step"] == step]
                if "Step" in forager_data.columns
                else forager_data
            )

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=step_data.get("X", [0] * len(step_data)),
                        y=step_data.get("Y", [0] * len(step_data)),
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=step_data.get("Energy", [50] * len(step_data)),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Energy Level"),
                        ),
                        text=step_data.get("AgentID", [""] * len(step_data)),
                        name="Foragers",
                    )
                ],
                name=str(step),
            )
            frames.append(frame)

        # Create initial figure
        fig = go.Figure(data=[go.Scatter(x=[], y=[], mode="markers")], frames=frames)

        # Add play/pause buttons
        fig.update_layout(
            title="Foraging Activity Animation",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )

        return fig


class ComparisonPlotter:
    """Plotting utilities for comparing simulation results"""

    def __init__(self) -> None:
        self.comparison_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def compare_scenarios(
        self,
        scenario_data: Dict[str, pd.DataFrame],
        metric: str = "Total_Bees",
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """Compare multiple scenarios"""
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (scenario_name, data) in enumerate(scenario_data.items()):
            if metric in data.columns:
                color = self.comparison_colors[i % len(self.comparison_colors)]
                ax.plot(
                    data["Day"],
                    data[metric],
                    label=scenario_name,
                    color=color,
                    linewidth=2,
                )

        ax.set_title(f"Scenario Comparison: {metric}")
        ax.set_xlabel("Day")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_sensitivity_analysis(
        self,
        sensitivity_results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """Plot sensitivity analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Sensitivity Analysis Results", fontsize=16, fontweight="bold")

        # Parameter sensitivity plot
        if "parameter_effects" in sensitivity_results:
            params = list(sensitivity_results["parameter_effects"].keys())
            effects = list(sensitivity_results["parameter_effects"].values())

            axes[0, 0].barh(params, effects)
            axes[0, 0].set_title("Parameter Sensitivity")
            axes[0, 0].set_xlabel("Effect Size")

        # Uncertainty quantification
        if "uncertainty_bands" in sensitivity_results:
            data = sensitivity_results["uncertainty_bands"]
            days = data.get("days", [])
            mean = data.get("mean", [])
            lower = data.get("lower", [])
            upper = data.get("upper", [])

            axes[0, 1].plot(days, mean, color="blue", linewidth=2, label="Mean")
            axes[0, 1].fill_between(
                days, lower, upper, alpha=0.3, color="blue", label="95% CI"
            )
            axes[0, 1].set_title("Uncertainty Quantification")
            axes[0, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig


def save_plots_to_pdf(figures: List[plt.Figure], output_path: str) -> None:
    """Save multiple figures to a single PDF"""
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:  # type: ignore[no-untyped-call]
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")


def create_simulation_report(
    model_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    landscape_grid: Any,
    output_dir: str = "artifacts/plots",
) -> List[str]:
    """Create complete set of plots for simulation report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plotter = BstewPlotter()
    plot_files = []

    # Population dynamics
    fig1 = plotter.plot_population_dynamics(
        model_data,
        save_path=str(output_path / "population_dynamics.png"),
        show_plot=False,
    )
    plot_files.append(str(output_path / "population_dynamics.png"))

    # Seasonal patterns
    fig2 = plotter.plot_seasonal_patterns(
        model_data,
        save_path=str(output_path / "seasonal_patterns.png"),
        show_plot=False,
    )
    plot_files.append(str(output_path / "seasonal_patterns.png"))

    # Landscape map
    fig3 = plotter.plot_landscape_map(
        landscape_grid,
        save_path=str(output_path / "landscape_map.png"),
        show_plot=False,
    )
    plot_files.append(str(output_path / "landscape_map.png"))

    # Foraging activity
    fig4 = plotter.plot_foraging_activity(
        agent_data,
        landscape_grid,
        save_path=str(output_path / "foraging_activity.png"),
        show_plot=False,
    )
    plot_files.append(str(output_path / "foraging_activity.png"))

    # Save all to PDF
    save_plots_to_pdf(
        [fig1, fig2, fig3, fig4], str(output_path / "simulation_report.pdf")
    )
    plot_files.append(str(output_path / "simulation_report.pdf"))

    return plot_files
