"""
Spatial Analysis Visualization Framework for BSTEW
=================================================

Comprehensive spatial visualization including distribution plots,
resource mapping, colony territory analysis, and foraging patterns.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from enum import Enum
from dataclasses import dataclass

# Import visualization libraries with fallbacks
try:
    import matplotlib.pyplot as plt
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
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore
    import plotly.figure_factory as ff

    HAS_PLOTLY = True
    HAS_3D = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    pyo = None
    Axes3D = None
    ff = None
    HAS_PLOTLY = False
    HAS_3D = False

logger = logging.getLogger(__name__)


class SpatialPlotType(Enum):
    """Types of spatial plots"""

    SCATTER = "scatter"
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    TERRITORY = "territory"
    NETWORK = "network"
    DENSITY = "density"
    SCATTER_3D = "scatter_3d"
    SURFACE_3D = "surface_3d"
    MESH_3D = "mesh_3d"
    ANIMATED_3D = "animated_3d"
    INTERACTIVE_3D = "interactive_3d"


class MapProjection(Enum):
    """Map projection types"""

    CARTESIAN = "cartesian"
    MERCATOR = "mercator"
    POLAR = "polar"


@dataclass
class SpatialConfig:
    """Configuration for spatial plots"""

    figure_size: Tuple[int, int] = (12, 10)
    dpi: int = 300
    projection: MapProjection = MapProjection.CARTESIAN
    show_grid: bool = True
    show_legend: bool = True
    show_scale: bool = True
    territory_alpha: float = 0.3
    resource_scale_factor: float = 1.0
    distance_units: str = "meters"
    coordinate_system: str = "utm"


class SpatialPlotter:
    """
    Spatial visualization system for BSTEW simulation data.

    Provides comprehensive spatial analysis visualization including:
    - Colony and resource distribution plots
    - Territory and foraging range visualization
    - Resource quality and availability mapping
    - Spatial foraging patterns and efficiency
    - Multi-colony spatial interaction analysis
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        """
        Initialize the spatial plotter.

        Args:
            config: Spatial plot configuration settings
        """
        self.config = config or SpatialConfig()
        self.logger = logging.getLogger(__name__)

        if not HAS_MATPLOTLIB:
            self.logger.warning("Matplotlib not available - plots will be text-based")

        # Color maps for different spatial features
        self.color_maps = {
            "colonies": "tab10",
            "resources": "viridis",
            "quality": "RdYlGn",
            "density": "plasma",
            "temperature": "coolwarm",
            "elevation": "terrain",
        }

    def plot_spatial_distribution(
        self,
        entities: Union[List[Dict[str, Any]], pd.DataFrame],
        entity_type: str = "colony",
        x_col: str = "x",
        y_col: str = "y",
        size_col: Optional[str] = None,
        color_col: Optional[str] = None,
        plot_type: SpatialPlotType = SpatialPlotType.SCATTER,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Create spatial distribution plots for colonies, resources, or agents.

        Args:
            entities: Spatial entity data with coordinates
            entity_type: Type of entities ("colony", "resource", "agent")
            x_col: Column name for x coordinates
            y_col: Column name for y coordinates
            size_col: Column name for size scaling (optional)
            color_col: Column name for color coding (optional)
            plot_type: Type of spatial plot
            save_path: Path to save plot

        Returns:
            Matplotlib figure or text-based plot description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(
                entities, f"Spatial Distribution - {entity_type}"
            )

        # Convert to DataFrame if needed
        if isinstance(entities, list):
            df = pd.DataFrame(entities)
        else:
            df = entities.copy()

        # Validate required columns
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(
                f"Required coordinate columns '{x_col}' and '{y_col}' not found"
            )

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Extract coordinates
        x = df[x_col].values
        y = df[y_col].values

        # Handle size scaling
        sizes = (
            df[size_col].values
            if size_col and size_col in df.columns
            else np.ones(len(df)) * 50
        )
        sizes = np.asarray(sizes) * self.config.resource_scale_factor

        # Handle color coding
        if color_col and color_col in df.columns:
            colors = df[color_col].values
            colormap = self.color_maps.get(entity_type, "viridis")
        else:
            colors = np.array(self._get_entity_colors(entity_type, len(df)))
            colormap = None

        if plot_type == SpatialPlotType.SCATTER:
            scatter = ax.scatter(
                np.asarray(x),
                np.asarray(y),
                s=np.asarray(sizes),
                c=np.asarray(colors),
                alpha=0.7,
                cmap=colormap,
                edgecolors="black",
                linewidth=0.5,
            )

            if colormap and color_col:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_col.replace("_", " ").title())

        elif plot_type == SpatialPlotType.HEATMAP:
            # Create 2D histogram/heatmap
            heatmap, xedges, yedges = np.histogram2d(
                np.asarray(x), np.asarray(y), bins=20, weights=sizes
            )
            extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

            im = ax.imshow(
                heatmap.T,
                extent=extent,
                origin="lower",
                cmap=self.color_maps.get(entity_type, "viridis"),
                alpha=0.8,
            )
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f"{entity_type.title()} Density")

        elif plot_type == SpatialPlotType.DENSITY:
            # Kernel density estimation visualization
            if HAS_SEABORN:
                sns.kdeplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    ax=ax,
                    cmap=self.color_maps.get(entity_type, "viridis"),
                    fill=True,
                    alpha=0.6,
                )
            else:
                # Fallback to scatter plot
                ax.scatter(
                    np.asarray(x),
                    np.asarray(y),
                    s=sizes,
                    c=np.asarray(colors),
                    alpha=0.7,
                )

        # Formatting
        ax.set_xlabel(f"X Coordinate ({self.config.distance_units})")
        ax.set_ylabel(f"Y Coordinate ({self.config.distance_units})")
        ax.set_title(f"{entity_type.title()} Spatial Distribution")

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.show_scale:
            self._add_scale_bar(ax, np.asarray(x), np.asarray(y))

        # Add entity labels if dataset is small
        if len(df) <= 20 and "id" in df.columns:
            for i, (xi, yi) in enumerate(zip(x, y)):
                ax.annotate(
                    str(df["id"].iloc[i]),
                    (xi, yi),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close()
            return save_path

        return fig

    def plot_resource_mapping(
        self,
        resource_data: Union[List[Dict[str, Any]], pd.DataFrame],
        quality_col: str = "quality",
        abundance_col: str = "abundance",
        resource_type_col: Optional[str] = None,
        x_col: str = "x",
        y_col: str = "y",
        show_quality_gradient: bool = True,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Create resource quality and abundance mapping visualization.

        Args:
            resource_data: Resource patch data with spatial information
            quality_col: Column for resource quality values
            abundance_col: Column for resource abundance values
            resource_type_col: Column for resource type categorization
            x_col: Column name for x coordinates
            y_col: Column name for y coordinates
            show_quality_gradient: Whether to show quality as color gradient
            save_path: Path to save plot

        Returns:
            Matplotlib figure or text-based plot description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(resource_data, "Resource Mapping")

        # Convert to DataFrame if needed
        if isinstance(resource_data, list):
            df = pd.DataFrame(resource_data)
        else:
            df = resource_data.copy()

        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(16, 12), dpi=self.config.dpi
        )

        x = df[x_col].values
        y = df[y_col].values

        # 1. Resource Quality Map
        if quality_col in df.columns:
            quality = df[quality_col].values
            scatter1 = ax1.scatter(
                x,
                y,
                c=quality,
                s=100,
                cmap="RdYlGn",
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label("Resource Quality")
            ax1.set_title("Resource Quality Distribution")
        else:
            ax1.scatter(x, y, s=100, alpha=0.8)
            ax1.set_title("Resource Locations")

        # 2. Resource Abundance Map
        if abundance_col in df.columns:
            abundance = df[abundance_col].values
            max_abundance = (
                abundance.max()
                if hasattr(abundance, "max")
                else np.max(np.asarray(abundance))
            )
            sizes = (abundance / max_abundance) * 200 + 20  # Scale for visibility
            ax2.scatter(
                x, y, s=sizes, c="orange", alpha=0.7, edgecolors="black", linewidth=0.5
            )
            ax2.set_title("Resource Abundance (Size = Abundance)")
        else:
            ax2.scatter(x, y, s=100, alpha=0.8)
            ax2.set_title("Resource Locations")

        # 3. Resource Type Distribution
        if resource_type_col and resource_type_col in df.columns:
            types = df[resource_type_col].unique()
            colors = plt.cm.get_cmap("Set3")(np.linspace(0, 1, len(types)))

            for i, resource_type in enumerate(types):
                mask = df[resource_type_col] == resource_type
                ax3.scatter(
                    x[mask],
                    y[mask],
                    c=[colors[i]],
                    label=resource_type,
                    s=100,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.5,
                )

            ax3.legend()
            ax3.set_title("Resource Types")
        else:
            ax3.scatter(x, y, s=100, alpha=0.8)
            ax3.set_title("Resource Locations")

        # 4. Combined Quality-Abundance Map
        if quality_col in df.columns and abundance_col in df.columns:
            quality = df[quality_col].values
            abundance = df[abundance_col].values
            max_abundance = (
                abundance.max()
                if hasattr(abundance, "max")
                else np.max(np.asarray(abundance))
            )
            sizes = (abundance / max_abundance) * 200 + 20

            scatter4 = ax4.scatter(
                x,
                y,
                c=quality,
                s=sizes,
                cmap="RdYlGn",
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )
            cbar4 = plt.colorbar(scatter4, ax=ax4)
            cbar4.set_label("Quality (Color) & Abundance (Size)")
            ax4.set_title("Combined Resource Assessment")
        else:
            ax4.scatter(x, y, s=100, alpha=0.8)
            ax4.set_title("Resource Locations")

        # Format all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel(f"X Coordinate ({self.config.distance_units})")
            ax.set_ylabel(f"Y Coordinate ({self.config.distance_units})")
            if self.config.show_grid:
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close()
            return save_path

        return fig

    def plot_colony_territories(
        self,
        colony_data: Union[List[Dict[str, Any]], pd.DataFrame],
        foraging_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        territory_radius_col: str = "foraging_range",
        x_col: str = "x",
        y_col: str = "y",
        colony_id_col: str = "colony_id",
        show_overlap: bool = True,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Visualize colony territories and foraging ranges.

        Args:
            colony_data: Colony location and territory data
            foraging_data: Individual foraging trip data (optional)
            territory_radius_col: Column with territory/foraging range radius
            x_col: Column name for x coordinates
            y_col: Column name for y coordinates
            colony_id_col: Column identifying different colonies
            show_overlap: Whether to highlight territory overlaps
            save_path: Path to save plot

        Returns:
            Matplotlib figure or text-based plot description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(colony_data, "Colony Territories")

        # Convert to DataFrame if needed
        if isinstance(colony_data, list):
            df = pd.DataFrame(colony_data)
        else:
            df = colony_data.copy()

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Get unique colonies
        colonies = df[colony_id_col].unique() if colony_id_col in df.columns else [1]
        colors = plt.cm.get_cmap("Set1")(np.linspace(0, 1, len(colonies)))

        territory_patches = []

        for i, colony_id in enumerate(colonies):
            if colony_id_col in df.columns:
                colony_row = df[df[colony_id_col] == colony_id].iloc[0]
            else:
                colony_row = df.iloc[i] if i < len(df) else df.iloc[0]

            x_center = colony_row[x_col]
            y_center = colony_row[y_col]

            # Plot colony location
            ax.scatter(
                x_center,
                y_center,
                s=200,
                c=[colors[i]],
                marker="H",
                edgecolors="black",
                linewidth=2,
                label=f"Colony {colony_id}",
                zorder=5,
            )

            # Plot territory circle
            if territory_radius_col in colony_row:
                radius = colony_row[territory_radius_col]
            else:
                radius = 500  # Default 500m radius

            territory_circle = plt.Circle(
                (x_center, y_center),
                radius,
                color=colors[i],
                alpha=self.config.territory_alpha,
                fill=True,
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(territory_circle)
            territory_patches.append(territory_circle)

            # Add colony label
            ax.annotate(
                f"Colony {colony_id}",
                (x_center, y_center),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        # Plot foraging data if provided
        if foraging_data is not None:
            if isinstance(foraging_data, list):
                foraging_df = pd.DataFrame(foraging_data)
            else:
                foraging_df = foraging_data.copy()

            if "x" in foraging_df.columns and "y" in foraging_df.columns:
                # Plot foraging locations
                ax.scatter(
                    foraging_df["x"],
                    foraging_df["y"],
                    s=20,
                    c="red",
                    alpha=0.6,
                    marker="o",
                    label="Foraging Locations",
                )

                # Draw foraging paths if colony association exists
                if colony_id_col in foraging_df.columns:
                    for colony_id in colonies:
                        colony_foraging = foraging_df[
                            foraging_df[colony_id_col] == colony_id
                        ]
                        if len(colony_foraging) > 0:
                            colony_row = df[df[colony_id_col] == colony_id].iloc[0]
                            x_center = colony_row[x_col]
                            y_center = colony_row[y_col]

                            for _, trip in colony_foraging.iterrows():
                                ax.plot(
                                    [x_center, trip["x"]],
                                    [y_center, trip["y"]],
                                    "k-",
                                    alpha=0.1,
                                    linewidth=0.5,
                                )

        # Highlight territory overlaps if requested
        if show_overlap and len(colonies) > 1:
            self._highlight_territory_overlaps(
                ax, df, x_col, y_col, colony_id_col, territory_radius_col
            )

        # Formatting
        ax.set_xlabel(f"X Coordinate ({self.config.distance_units})")
        ax.set_ylabel(f"Y Coordinate ({self.config.distance_units})")
        ax.set_title("Colony Territories and Foraging Ranges")
        ax.set_aspect("equal", adjustable="box")

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.show_legend:
            ax.legend(loc="upper right")

        if self.config.show_scale:
            self._add_scale_bar(ax, np.asarray(df[x_col]), np.asarray(df[y_col]))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close()
            return save_path

        return fig

    def plot_foraging_patterns(
        self,
        foraging_data: Union[List[Dict[str, Any]], pd.DataFrame],
        colony_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        pattern_type: str = "heatmap",
        time_window: Optional[str] = None,
        efficiency_col: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Visualize spatial foraging patterns and efficiency.

        Args:
            foraging_data: Foraging trip location and outcome data
            colony_data: Colony location data for reference
            pattern_type: Type of pattern visualization ("heatmap", "flow", "efficiency")
            time_window: Time window for temporal analysis
            efficiency_col: Column containing foraging efficiency data
            save_path: Path to save plot

        Returns:
            Matplotlib figure or text-based plot description
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(foraging_data, "Foraging Patterns")

        # Convert to DataFrame if needed
        if isinstance(foraging_data, list):
            df = pd.DataFrame(foraging_data)
        else:
            df = foraging_data.copy()

        # Convert colony_data to DataFrame if needed
        colony_df = None
        if colony_data is not None:
            if isinstance(colony_data, list):
                colony_df = pd.DataFrame(colony_data)
            else:
                colony_df = colony_data

        if pattern_type == "heatmap":
            return self._plot_foraging_heatmap(df, colony_df, save_path)
        elif pattern_type == "flow":
            return self._plot_foraging_flow(df, colony_df, save_path)
        elif pattern_type == "efficiency":
            return self._plot_foraging_efficiency(df, efficiency_col, save_path)
        else:
            # Default to combined view
            return self._plot_foraging_combined(
                df, colony_df, efficiency_col, save_path
            )

    def create_spatial_dashboard(
        self,
        colony_data: Union[List[Dict[str, Any]], pd.DataFrame],
        resource_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        foraging_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """
        Create comprehensive spatial analysis dashboard.

        Args:
            colony_data: Colony location and territory data
            resource_data: Resource patch spatial data
            foraging_data: Foraging trip spatial data
            save_path: Path to save dashboard

        Returns:
            Matplotlib figure with multiple spatial analysis plots
        """
        if not HAS_MATPLOTLIB:
            return self._create_text_plot(colony_data, "Spatial Analysis Dashboard")

        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 16), dpi=self.config.dpi)

        # Convert data
        if isinstance(colony_data, list):
            colony_df = pd.DataFrame(colony_data)
        else:
            colony_df = colony_data.copy()

        # 1. Colony territories (top left)
        ax1 = plt.subplot(3, 3, (1, 2))
        self._create_dashboard_territories(ax1, colony_df, foraging_data)

        # 2. Resource distribution (top right)
        ax2 = plt.subplot(3, 3, 3)
        if resource_data is not None:
            self._create_dashboard_resources(ax2, resource_data)
        else:
            ax2.text(
                0.5,
                0.5,
                "No resource data\nprovided",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Resource Distribution")

        # 3. Foraging patterns (middle left)
        ax3 = plt.subplot(3, 3, (4, 5))
        if foraging_data is not None:
            self._create_dashboard_foraging(ax3, foraging_data, colony_df)
        else:
            ax3.text(
                0.5,
                0.5,
                "No foraging data\nprovided",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Foraging Patterns")

        # 4. Spatial efficiency (middle right)
        ax4 = plt.subplot(3, 3, 6)
        if foraging_data is not None:
            self._create_dashboard_efficiency(ax4, foraging_data)
        else:
            ax4.text(
                0.5,
                0.5,
                "No foraging data\nprovided",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Spatial Efficiency")

        # 5. Distance analysis (bottom left)
        ax5 = plt.subplot(3, 3, 7)
        self._create_dashboard_distances(ax5, colony_df, resource_data, foraging_data)

        # 6. Territory overlap analysis (bottom middle)
        ax6 = plt.subplot(3, 3, 8)
        self._create_dashboard_overlaps(ax6, colony_df)

        # 7. Spatial statistics (bottom right)
        ax7 = plt.subplot(3, 3, 9)
        self._create_dashboard_statistics(ax7, colony_df, resource_data, foraging_data)

        plt.suptitle("Spatial Analysis Dashboard", fontsize=18, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close()
            return save_path

        return fig

    def _get_entity_colors(self, entity_type: str, count: int) -> List[str]:
        """Get appropriate colors for entity type"""
        if entity_type == "colony":
            return list(plt.cm.get_cmap("Set1")(np.linspace(0, 1, count)))
        elif entity_type == "resource":
            return list(plt.cm.get_cmap("viridis")(np.linspace(0, 1, count)))
        else:
            return list(plt.cm.get_cmap("tab10")(np.linspace(0, 1, count)))

    def _add_scale_bar(self, ax: Axes, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Add scale bar to spatial plot"""
        # Calculate appropriate scale bar length
        x_range = np.max(x_data) - np.min(x_data)
        scale_length = 10 ** (np.floor(np.log10(x_range / 4)))

        # Position scale bar
        x_pos = np.min(x_data) + 0.05 * x_range
        y_pos = np.min(y_data) + 0.05 * (np.max(y_data) - np.min(y_data))

        # Draw scale bar
        ax.plot([x_pos, x_pos + scale_length], [y_pos, y_pos], "k-", linewidth=3)
        ax.text(
            x_pos + scale_length / 2,
            y_pos - 0.02 * (np.max(y_data) - np.min(y_data)),
            f"{scale_length:.0f} {self.config.distance_units}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
        )

    def _highlight_territory_overlaps(
        self,
        ax: Axes,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        colony_id_col: str,
        territory_radius_col: str,
    ) -> None:
        """Highlight overlapping territories"""
        colonies = df[colony_id_col].unique()

        for i, colony1 in enumerate(colonies):
            for j, colony2 in enumerate(colonies[i + 1 :], i + 1):
                row1 = df[df[colony_id_col] == colony1].iloc[0]
                row2 = df[df[colony_id_col] == colony2].iloc[0]

                # Calculate distance between colonies
                distance = np.sqrt(
                    (row1[x_col] - row2[x_col]) ** 2 + (row1[y_col] - row2[y_col]) ** 2
                )

                # Check for overlap
                radius1 = row1.get(territory_radius_col, 500)
                radius2 = row2.get(territory_radius_col, 500)

                if distance < (radius1 + radius2):
                    # Draw overlap indicator
                    mid_x = (row1[x_col] + row2[x_col]) / 2
                    mid_y = (row1[y_col] + row2[y_col]) / 2
                    ax.plot(
                        [row1[x_col], row2[x_col]],
                        [row1[y_col], row2[y_col]],
                        "r--",
                        linewidth=2,
                        alpha=0.8,
                    )
                    ax.scatter(
                        mid_x,
                        mid_y,
                        s=100,
                        c="red",
                        marker="X",
                        edgecolors="black",
                        linewidth=1,
                        zorder=6,
                    )

    def _plot_foraging_heatmap(
        self,
        df: pd.DataFrame,
        colony_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """Create foraging activity heatmap"""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        if "x" in df.columns and "y" in df.columns:
            x = np.asarray(df["x"].values)
            y = np.asarray(df["y"].values)

            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=25)
            extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

            im = ax.imshow(
                heatmap.T, extent=extent, origin="lower", cmap="YlOrRd", alpha=0.8
            )
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Foraging Activity Density")

            # Add colony locations if provided
            if colony_data is not None:
                if isinstance(colony_data, list):
                    colony_df = pd.DataFrame(colony_data)
                else:
                    colony_df = colony_data.copy()

                if "x" in colony_df.columns and "y" in colony_df.columns:
                    ax.scatter(
                        colony_df["x"],
                        colony_df["y"],
                        s=200,
                        c="blue",
                        marker="H",
                        edgecolors="black",
                        linewidth=2,
                        label="Colonies",
                        zorder=5,
                    )

        ax.set_title("Foraging Activity Heatmap")
        ax.set_xlabel(f"X Coordinate ({self.config.distance_units})")
        ax.set_ylabel(f"Y Coordinate ({self.config.distance_units})")

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_foraging_flow(
        self,
        df: pd.DataFrame,
        colony_data: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """Create foraging flow visualization"""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Implementation would create flow field visualization
        ax.set_title("Foraging Flow Patterns")
        ax.text(
            0.5,
            0.5,
            "Flow visualization\nrequires additional\nanalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        plt.tight_layout()
        return fig

    def _plot_foraging_efficiency(
        self,
        df: pd.DataFrame,
        efficiency_col: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """Create spatial efficiency visualization"""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        if "x" in df.columns and "y" in df.columns:
            x = np.asarray(df["x"].values)
            y = np.asarray(df["y"].values)

            if efficiency_col and efficiency_col in df.columns:
                efficiency = np.asarray(df[efficiency_col].values)
                scatter = ax.scatter(
                    x,
                    y,
                    c=efficiency,
                    s=60,
                    cmap="RdYlGn",
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.5,
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Foraging Efficiency")
            else:
                ax.scatter(x, y, s=60, alpha=0.8)

        ax.set_title("Spatial Foraging Efficiency")
        ax.set_xlabel(f"X Coordinate ({self.config.distance_units})")
        ax.set_ylabel(f"Y Coordinate ({self.config.distance_units})")

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_foraging_combined(
        self,
        df: pd.DataFrame,
        colony_data: Optional[pd.DataFrame] = None,
        efficiency_col: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Union[Figure, str]:
        """Create combined foraging analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(16, 12), dpi=self.config.dpi
        )

        # Individual subplot implementations would go here
        ax1.set_title("Foraging Locations")
        ax2.set_title("Activity Density")
        ax3.set_title("Efficiency Map")
        ax4.set_title("Summary Statistics")

        plt.tight_layout()
        return fig

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

    # Dashboard subplot methods
    def _create_dashboard_territories(
        self, ax: Axes, colony_df: pd.DataFrame, foraging_data: Any
    ) -> None:
        """Create territories subplot for dashboard"""
        if "x" in colony_df.columns and "y" in colony_df.columns:
            x = np.asarray(colony_df["x"].values)
            y = np.asarray(colony_df["y"].values)

            ax.scatter(
                x, y, s=200, c="blue", marker="H", edgecolors="black", linewidth=2
            )

            # Add territory circles
            for i, (xi, yi) in enumerate(zip(x, y)):
                radius = colony_df.get(
                    "foraging_range", pd.Series([500] * len(colony_df))
                ).iloc[i]
                circle = plt.Circle(
                    (xi, yi),
                    radius,
                    fill=False,
                    color="blue",
                    linestyle="--",
                    alpha=0.6,
                )
                ax.add_patch(circle)

            ax.set_title("Colony Territories")
            ax.set_aspect("equal", adjustable="box")

    def _create_dashboard_resources(self, ax: Axes, resource_data: Any) -> None:
        """Create resources subplot for dashboard"""
        if isinstance(resource_data, list):
            resource_df = pd.DataFrame(resource_data)
        else:
            resource_df = resource_data.copy()

        if "x" in resource_df.columns and "y" in resource_df.columns:
            quality = resource_df.get("quality", pd.Series([0.5] * len(resource_df)))
            ax.scatter(
                resource_df["x"],
                resource_df["y"],
                c=quality,
                s=100,
                cmap="RdYlGn",
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )
            ax.set_title("Resource Distribution")

    def _create_dashboard_foraging(
        self, ax: Axes, foraging_data: Any, colony_df: pd.DataFrame
    ) -> None:
        """Create foraging subplot for dashboard"""
        if isinstance(foraging_data, list):
            foraging_df = pd.DataFrame(foraging_data)
        else:
            foraging_df = foraging_data.copy()

        if "x" in foraging_df.columns and "y" in foraging_df.columns:
            ax.scatter(
                foraging_df["x"], foraging_df["y"], s=20, c="red", alpha=0.6, marker="o"
            )
            ax.set_title("Foraging Patterns")
        else:
            ax.text(
                0.5,
                0.5,
                "No coordinate\ndata available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Foraging Patterns")

    def _create_dashboard_efficiency(self, ax: Axes, foraging_data: Any) -> None:
        """Create efficiency subplot for dashboard"""
        ax.text(
            0.5,
            0.5,
            "Foraging\nEfficiency\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Spatial Efficiency")

    def _create_dashboard_distances(
        self, ax: Axes, colony_df: Any, resource_data: Any, foraging_data: Any
    ) -> None:
        """Create distance analysis subplot for dashboard"""
        # Simple distance histogram
        if len(colony_df) > 1:
            distances = []
            x_coords = colony_df["x"].values if "x" in colony_df.columns else []
            y_coords = colony_df["y"].values if "y" in colony_df.columns else []

            for i in range(len(x_coords)):
                for j in range(i + 1, len(x_coords)):
                    dist = np.sqrt(
                        (x_coords[i] - x_coords[j]) ** 2
                        + (y_coords[i] - y_coords[j]) ** 2
                    )
                    distances.append(dist)

            if distances:
                ax.hist(distances, bins=10, alpha=0.7, color="skyblue")
                ax.set_title("Inter-colony Distances")
                ax.set_xlabel("Distance")
                ax.set_ylabel("Frequency")

    def _create_dashboard_overlaps(self, ax: Axes, colony_df: pd.DataFrame) -> None:
        """Create overlap analysis subplot for dashboard"""
        ax.text(
            0.5,
            0.5,
            "Territory\nOverlap\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Territory Overlaps")

    def _create_dashboard_statistics(
        self, ax: Axes, colony_df: Any, resource_data: Any, foraging_data: Any
    ) -> None:
        """Create statistics subplot for dashboard"""
        ax.axis("off")

        stats_text = []
        stats_text.append(f"Colonies: {len(colony_df)}")

        if resource_data is not None:
            resource_count = (
                len(resource_data) if hasattr(resource_data, "__len__") else 0
            )
            stats_text.append(f"Resource Patches: {resource_count}")

        if foraging_data is not None:
            foraging_count = (
                len(foraging_data) if hasattr(foraging_data, "__len__") else 0
            )
            stats_text.append(f"Foraging Trips: {foraging_count}")

        # Calculate spatial extent
        if "x" in colony_df.columns and "y" in colony_df.columns:
            x_range = colony_df["x"].max() - colony_df["x"].min()
            y_range = colony_df["y"].max() - colony_df["y"].min()
            area = x_range * y_range
            stats_text.append(f"Study Area: {area:,.0f} {self.config.distance_units}²")

        stats_str = "\n".join(stats_text)
        ax.text(
            0.1,
            0.9,
            "Spatial Statistics:",
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
            va="top",
        )
        ax.text(0.1, 0.7, stats_str, fontsize=10, transform=ax.transAxes, va="top")

    def create_3d_spatial_landscape(
        self,
        colony_data: Union[List[Dict[str, Any]], pd.DataFrame],
        resource_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        elevation_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        plot_type: SpatialPlotType = SpatialPlotType.INTERACTIVE_3D,
        save_path: Optional[str] = None,
    ) -> Union[Any, str]:
        """
        Create comprehensive 3D spatial landscape visualization.

        Phase 4 enhancement for advanced spatial analysis with elevation,
        colony positions, and resource distribution in 3D space.

        Args:
            colony_data: Colony location and characteristics data
            resource_data: Resource patch locations and qualities (optional)
            elevation_data: Terrain elevation data (optional)
            plot_type: Type of 3D visualization to create
            save_path: Path to save the interactive plot

        Returns:
            Plotly 3D figure or text description if Plotly unavailable
        """
        if not HAS_PLOTLY:
            return self._create_text_plot(colony_data, "3D Spatial Landscape")

        # Convert data
        if isinstance(colony_data, list):
            colony_df = pd.DataFrame(colony_data)
        else:
            colony_df = colony_data.copy()

        # Convert resource and elevation data to DataFrame if needed
        resource_df = None
        if resource_data is not None:
            if isinstance(resource_data, list):
                resource_df = pd.DataFrame(resource_data)
            else:
                resource_df = resource_data

        elevation_df = None
        if elevation_data is not None:
            if isinstance(elevation_data, list):
                elevation_df = pd.DataFrame(elevation_data)
            else:
                elevation_df = elevation_data

        if plot_type == SpatialPlotType.SURFACE_3D:
            return self._create_3d_surface_landscape(
                colony_df, resource_df, elevation_df, save_path
            )
        elif plot_type == SpatialPlotType.SCATTER_3D:
            return self._create_3d_scatter_landscape(
                colony_df, resource_df, elevation_df, save_path
            )
        elif plot_type == SpatialPlotType.MESH_3D:
            return self._create_3d_mesh_landscape(
                colony_df, resource_df, elevation_df, save_path
            )
        else:
            return self._create_interactive_3d_landscape(
                colony_df, resource_df, elevation_df, save_path
            )

    def _create_3d_surface_landscape(
        self,
        colony_df: pd.DataFrame,
        resource_data: Optional[pd.DataFrame],
        elevation_data: Optional[pd.DataFrame],
        save_path: Optional[str] = None,
    ) -> Any:
        """Create 3D surface plot of spatial landscape with elevation"""

        # Generate elevation surface
        x_range = np.linspace(
            colony_df["x"].min() - 500, colony_df["x"].max() + 500, 50
        )
        y_range = np.linspace(
            colony_df["y"].min() - 500, colony_df["y"].max() + 500, 50
        )
        X, Y = np.meshgrid(x_range, y_range)

        # Generate realistic elevation pattern
        if elevation_data is not None and "elevation" in elevation_data.columns:
            # Use actual elevation data if available
            Z = np.zeros_like(X)
            for i, x in enumerate(x_range):
                for j, y in enumerate(y_range):
                    # Simple interpolation from elevation data
                    distances = np.sqrt(
                        (elevation_data["x"] - x) ** 2 + (elevation_data["y"] - y) ** 2
                    )
                    nearest_idx = distances.idxmin()
                    Z[j, i] = elevation_data.loc[nearest_idx, "elevation"]
        else:
            # Generate synthetic elevation pattern
            Z = (
                100
                + 50 * np.sin(X * 0.001) * np.cos(Y * 0.001)
                + 25 * np.random.rand(*X.shape)
            )

        # Create surface
        fig = go.Figure(
            data=[
                go.Surface(
                    z=Z,
                    x=x_range,
                    y=y_range,
                    colorscale="Terrain",
                    name="Elevation",
                    opacity=0.8,
                    showscale=True,
                )
            ]
        )

        # Add colony positions as scatter points
        fig.add_trace(
            go.Scatter3d(
                x=colony_df["x"],
                y=colony_df["y"],
                z=colony_df.get(
                    "elevation", [np.max(Z)] * len(colony_df)
                ),  # Place on surface
                mode="markers+text",
                marker=dict(
                    size=colony_df.get("colony_size", 10) / 10,
                    color=colony_df.get("population", range(len(colony_df))),
                    colorscale="Viridis",
                    symbol="diamond",
                    line=dict(width=2, color="black"),
                ),
                text=colony_df.get(
                    "colony_id", [f"Colony {i}" for i in range(len(colony_df))]
                ),
                textposition="top center",
                name="Colonies",
                hovertemplate="<b>%{text}</b><br>Population: %{marker.color}<br>Location: (%{x}, %{y})<extra></extra>",
            )
        )

        # Add resource patches if available
        if resource_data is not None:
            if isinstance(resource_data, list):
                resource_df = pd.DataFrame(resource_data)
            else:
                resource_df = resource_data.copy()

            fig.add_trace(
                go.Scatter3d(
                    x=resource_df.get("x", []),
                    y=resource_df.get("y", []),
                    z=resource_df.get("elevation", [np.max(Z) + 20] * len(resource_df)),
                    mode="markers",
                    marker=dict(
                        size=resource_df.get("resource_quality", 5),
                        color="yellow",
                        symbol="circle",
                        opacity=0.7,
                        line=dict(width=1, color="orange"),
                    ),
                    name="Resources",
                    hovertemplate="Resource Quality: %{marker.size}<br>Location: (%{x}, %{y})<extra></extra>",
                )
            )

        fig.update_layout(
            title="3D Spatial Landscape with Colonies and Resources",
            scene=dict(
                xaxis_title="X Coordinate (m)",
                yaxis_title="Y Coordinate (m)",
                zaxis_title="Elevation (m)",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                aspectmode="cube",
            ),
            width=1000,
            height=700,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_3d_scatter_landscape(
        self,
        colony_df: pd.DataFrame,
        resource_data: Optional[pd.DataFrame],
        elevation_data: Optional[pd.DataFrame],
        save_path: Optional[str] = None,
    ) -> Any:
        """Create 3D scatter plot with colony distribution and foraging networks"""

        fig = go.Figure()

        # Add colony scatter
        fig.add_trace(
            go.Scatter3d(
                x=colony_df["x"],
                y=colony_df["y"],
                z=colony_df.get("z", colony_df.get("elevation", [0] * len(colony_df))),
                mode="markers",
                marker=dict(
                    size=np.clip(
                        np.array(colony_df.get("population", [100] * len(colony_df)))
                        / 50,
                        8,
                        25,
                    ),
                    color=colony_df.get("species_diversity", range(len(colony_df))),
                    colorscale="Spectral",
                    opacity=0.8,
                    colorbar=dict(title="Species Diversity"),
                    line=dict(width=2, color="darkblue"),
                ),
                text=colony_df.get(
                    "colony_id", [f"Colony {i}" for i in range(len(colony_df))]
                ),
                name="Colonies",
                hovertemplate="<b>%{text}</b><br>Population: %{marker.size}<br>Position: (%{x}, %{y}, %{z})<extra></extra>",
            )
        )

        # Add foraging connections (simplified network)
        if len(colony_df) > 1:
            for i in range(min(len(colony_df), 5)):  # Limit connections for clarity
                for j in range(i + 1, min(len(colony_df), 8)):
                    if np.random.random() < 0.3:  # 30% chance of connection
                        fig.add_trace(
                            go.Scatter3d(
                                x=[colony_df.iloc[i]["x"], colony_df.iloc[j]["x"]],
                                y=[colony_df.iloc[i]["y"], colony_df.iloc[j]["y"]],
                                z=[
                                    colony_df.iloc[i].get("z", 0),
                                    colony_df.iloc[j].get("z", 0),
                                ],
                                mode="lines",
                                line=dict(width=3, color="rgba(100,100,100,0.4)"),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

        # Add resource patches with different shapes
        if resource_data is not None:
            if isinstance(resource_data, list):
                resource_df = pd.DataFrame(resource_data)
            else:
                resource_df = resource_data.copy()

            # Different resource types with different symbols
            resource_types = resource_df.get(
                "resource_type", ["nectar"] * len(resource_df)
            )
            symbols = {"nectar": "circle", "pollen": "square", "mixed": "diamond"}
            colors = {"nectar": "gold", "pollen": "orange", "mixed": "red"}

            for res_type in set(resource_types):
                type_data = resource_df[
                    resource_df.get("resource_type", "nectar") == res_type
                ]

                fig.add_trace(
                    go.Scatter3d(
                        x=type_data.get("x", []),
                        y=type_data.get("y", []),
                        z=type_data.get(
                            "z", type_data.get("elevation", [10] * len(type_data))
                        ),
                        mode="markers",
                        marker=dict(
                            size=type_data.get("quality", 8),
                            color=colors.get(res_type, "yellow"),
                            symbol=symbols.get(res_type, "circle"),
                            opacity=0.7,
                        ),
                        name=f"{res_type.title()} Resources",
                        hovertemplate=f"{res_type.title()} Quality: %{{marker.size}}<br>Location: (%{{x}}, %{{y}}, %{{z}})<extra></extra>",
                    )
                )

        fig.update_layout(
            title="3D Colony and Resource Distribution Network",
            scene=dict(
                xaxis_title="X Coordinate (m)",
                yaxis_title="Y Coordinate (m)",
                zaxis_title="Elevation/Height (m)",
                camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
            ),
            width=900,
            height=600,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_3d_mesh_landscape(
        self,
        colony_df: pd.DataFrame,
        resource_data: Optional[pd.DataFrame],
        elevation_data: Optional[pd.DataFrame],
        save_path: Optional[str] = None,
    ) -> Any:
        """Create 3D mesh visualization of foraging territories and overlaps"""

        fig = go.Figure()

        # Create territory meshes for each colony
        for i, (idx, row) in enumerate(colony_df.iterrows()):
            if i >= 5:  # Limit to first 5 colonies for performance
                break

            # Create circular territory mesh
            theta = np.linspace(0, 2 * np.pi, 20)
            radius = float(row.get("territory_radius", 500))

            # Territory boundary
            x_circle = float(row["x"]) + radius * np.cos(theta)
            y_circle = float(row["y"]) + radius * np.sin(theta)
            z_circle = [float(row.get("z", 0))] * len(theta)

            # Add territory boundary
            fig.add_trace(
                go.Scatter3d(
                    x=x_circle,
                    y=y_circle,
                    z=z_circle,
                    mode="lines",
                    line=dict(
                        width=4, color=f"rgba({50 * i + 100}, {100}, {150}, 0.8)"
                    ),
                    name=f"Territory {i + 1}",
                    showlegend=True,
                )
            )

            # Create mesh surface for territory
            r = np.linspace(0, radius, 10)
            theta_mesh = np.linspace(0, 2 * np.pi, 20)
            R, THETA = np.meshgrid(r, theta_mesh)

            X_mesh = float(row["x"]) + R * np.cos(THETA)
            Y_mesh = float(row["y"]) + R * np.sin(THETA)
            Z_mesh = np.ones_like(X_mesh) * (
                float(row.get("z", 0)) + 5
            )  # Slightly elevated

            # Add mesh surface
            fig.add_trace(
                go.Surface(
                    x=X_mesh,
                    y=Y_mesh,
                    z=Z_mesh,
                    colorscale=[
                        [0, f"rgba({50 * i + 100}, {100}, {150}, 0.2)"],
                        [1, f"rgba({50 * i + 100}, {100}, {150}, 0.4)"],
                    ],
                    showscale=False,
                    name=f"Territory {i + 1} Area",
                )
            )

        # Add colony positions as elevated markers
        fig.add_trace(
            go.Scatter3d(
                x=colony_df["x"],
                y=colony_df["y"],
                z=np.array(colony_df.get("z", [0] * len(colony_df)))
                + 20,  # Elevated above territories
                mode="markers+text",
                marker=dict(
                    size=15,
                    color="red",
                    symbol="diamond",
                    line=dict(width=3, color="black"),
                ),
                text=[f"C{i + 1}" for i in range(len(colony_df))],
                textposition="top center",
                name="Colony Centers",
            )
        )

        fig.update_layout(
            title="3D Territory Mesh Analysis with Overlaps",
            scene=dict(
                xaxis_title="X Coordinate (m)",
                yaxis_title="Y Coordinate (m)",
                zaxis_title="Height (m)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=2.0)),
                aspectmode="cube",
            ),
            width=1000,
            height=700,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_interactive_3d_landscape(
        self,
        colony_df: pd.DataFrame,
        resource_data: Optional[pd.DataFrame],
        elevation_data: Optional[pd.DataFrame],
        save_path: Optional[str] = None,
    ) -> Any:
        """Create comprehensive interactive 3D landscape dashboard"""

        # Create subplot figure with multiple 3D visualizations
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "surface"}, {"type": "scatter3d"}],
                [{"type": "scatter3d"}, {"type": "surface"}],
            ],
            subplot_titles=[
                "Elevation Surface",
                "Colony Distribution",
                "Resource Networks",
                "Territory Analysis",
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        # 1. Elevation surface (top left)
        x_range = np.linspace(0, 2000, 25)
        y_range = np.linspace(0, 2000, 25)
        X, Y = np.meshgrid(x_range, y_range)
        Z = 50 + 30 * np.sin(X * 0.002) * np.cos(Y * 0.002)

        fig.add_trace(
            go.Surface(
                z=Z, x=x_range, y=y_range, colorscale="Terrain", showscale=False
            ),
            row=1,
            col=1,
        )

        # 2. Colony distribution (top right)
        fig.add_trace(
            go.Scatter3d(
                x=colony_df["x"],
                y=colony_df["y"],
                z=colony_df.get("population", [100] * len(colony_df)),
                mode="markers",
                marker=dict(size=10, color="red", opacity=0.8),
                name="Colonies",
            ),
            row=1,
            col=2,
        )

        # 3. Resource networks (bottom left)
        if resource_data is not None and len(resource_data) > 0:
            if isinstance(resource_data, list):
                res_df = pd.DataFrame(resource_data)
            else:
                res_df = resource_data.copy()

            fig.add_trace(
                go.Scatter3d(
                    x=res_df.get("x", []),
                    y=res_df.get("y", []),
                    z=res_df.get("quality", []),
                    mode="markers",
                    marker=dict(size=8, color="yellow", opacity=0.7),
                    name="Resources",
                ),
                row=2,
                col=1,
            )

        # 4. Territory analysis surface (bottom right)
        territory_Z = np.zeros_like(X)
        for i, (idx, row) in enumerate(colony_df.iterrows()):
            if i < 3:  # Limit for performance
                # Add Gaussian territory influence
                influence = np.exp(
                    -((X - float(row["x"])) ** 2 + (Y - float(row["y"])) ** 2)
                    / (2 * 500**2)
                )
                territory_Z += influence * float(row.get("population", 100))

        fig.add_trace(
            go.Surface(
                z=territory_Z, x=x_range, y=y_range, colorscale="Hot", showscale=False
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Interactive 3D Spatial Analysis Dashboard",
            height=800,
            showlegend=False,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_animated_spatial_evolution(
        self, time_series_data: List[Dict[str, Any]], save_path: Optional[str] = None
    ) -> Union[Any, str]:
        """
        Create animated visualization of spatial patterns evolving over time.

        Advanced Phase 4 feature showing colony expansion, resource depletion,
        and territory changes through animation.

        Args:
            time_series_data: List of spatial data snapshots over time
            save_path: Path to save the animated plot

        Returns:
            Plotly animated figure or text description
        """
        if not HAS_PLOTLY:
            return (
                "Animated spatial evolution visualization requires Plotly installation."
            )

        if not time_series_data:
            return "No time series data provided for animation."

        # Extract time points
        time_points = sorted(set(d.get("time_step", 0) for d in time_series_data))

        # Create animation frames
        frames = []

        for t in time_points:
            # Filter data for this time point
            frame_data = [d for d in time_series_data if d.get("time_step", 0) == t]

            if not frame_data:
                continue

            df = pd.DataFrame(frame_data)

            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=df.get("x", []),
                        y=df.get("y", []),
                        z=df.get("population", [0] * len(df)),
                        mode="markers",
                        marker=dict(
                            size=np.clip(
                                np.array(df.get("population", [100] * len(df))) / 100,
                                5,
                                20,
                            ),
                            color=df.get("population", [100] * len(df)),
                            colorscale="Viridis",
                            opacity=0.8,
                            line=dict(width=1, color="black"),
                        ),
                        text=[f"Time: {t}, Colony: {i}" for i in range(len(df))],
                        hovertemplate="<b>%{text}</b><br>Population: %{z}<br>Location: (%{x}, %{y})<extra></extra>",
                    )
                ],
                name=f"frame_{t}",
            )
            frames.append(frame)

        # Initial frame
        if frames:
            initial_data = time_series_data[0] if time_series_data else {}
            initial_df = pd.DataFrame([initial_data])

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=initial_df.get("x", [0]),
                        y=initial_df.get("y", [0]),
                        z=initial_df.get("population", [0]),
                        mode="markers",
                        marker=dict(size=10, color="blue"),
                    )
                ],
                frames=frames,
            )

            # Add animation controls
            fig.update_layout(
                title="Animated Spatial Colony Evolution",
                scene=dict(
                    xaxis_title="X Coordinate (m)",
                    yaxis_title="Y Coordinate (m)",
                    zaxis_title="Population Size",
                ),
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 300, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 200},
                                    },
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            },
                        ],
                    }
                ],
                width=900,
                height=600,
            )

            if save_path:
                fig.write_html(save_path)

            return fig

        return "No valid animation frames could be generated."
