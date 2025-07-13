"""
Enhanced Agent Visualization for BSTEW
=====================================

Agent visualization with species-specific colors, dynamic visualization updates,
and advanced visualization capabilities matching NetLogo's visual system.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import Dict, List, Optional, Any
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json

from .display_system import DisplayStateManager, ConditionalVisualization
from ..core.enums import BeeStatus
from ..core.spatial_analysis import SpatialPatch


class VisualizationMode(Enum):
    """Visualization modes for different display purposes"""
    STATIC = "static"
    ANIMATED = "animated"
    INTERACTIVE = "interactive"
    COMPARATIVE = "comparative"
    TIME_SERIES = "time_series"


class AgentVisualizationLayer(Enum):
    """Different layers for agent visualization"""
    BACKGROUND = "background"
    PATCHES = "patches"
    CONNECTIONS = "connections"
    AGENTS = "agents"
    PATHS = "paths"
    OVERLAYS = "overlays"
    ANNOTATIONS = "annotations"


@dataclass
class VisualizationFrame:
    """Single frame of visualization data"""
    timestamp: datetime
    agents: List[Any]
    patches: List[SpatialPatch]
    connections: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SpeciesColorManager:
    """Manages species-specific colors and visual properties"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Species-specific color schemes
        self.species_colors = {
            "apis_mellifera": {
                "primary": "#FFD700",    # Gold
                "secondary": "#FFA500",  # Orange
                "workers": "#FFD700",
                "foragers": "#FF8C00",
                "nurses": "#FFFF00",
                "queens": "#FF1493",
                "drones": "#8B4513"
            },
            "bombus_terrestris": {
                "primary": "#FF6B6B",    # Red
                "secondary": "#FF4444",  # Darker red
                "workers": "#FF6B6B",
                "foragers": "#FF0000",
                "nurses": "#FF69B4",
                "queens": "#8B0000",
                "drones": "#CD5C5C"
            },
            "bombus_lapidarius": {
                "primary": "#4ECDC4",    # Teal
                "secondary": "#20B2AA",  # Light sea green
                "workers": "#4ECDC4",
                "foragers": "#00CED1",
                "nurses": "#48D1CC",
                "queens": "#008B8B",
                "drones": "#5F9EA0"
            },
            "bombus_pascuorum": {
                "primary": "#45B7D1",    # Blue
                "secondary": "#1E90FF",  # Dodger blue
                "workers": "#45B7D1",
                "foragers": "#0000FF",
                "nurses": "#87CEEB",
                "queens": "#000080",
                "drones": "#4169E1"
            },
            "bombus_hortorum": {
                "primary": "#96CEB4",    # Green
                "secondary": "#90EE90",  # Light green
                "workers": "#96CEB4",
                "foragers": "#00FF00",
                "nurses": "#98FB98",
                "queens": "#006400",
                "drones": "#32CD32"
            },
            "bombus_ruderatus": {
                "primary": "#FFEAA7",    # Yellow
                "secondary": "#FFE4B5",  # Moccasin
                "workers": "#FFEAA7",
                "foragers": "#FFFF00",
                "nurses": "#FFFFE0",
                "queens": "#DAA520",
                "drones": "#F0E68C"
            },
            "bombus_humilis": {
                "primary": "#DDA0DD",    # Plum
                "secondary": "#DA70D6",  # Orchid
                "workers": "#DDA0DD",
                "foragers": "#FF00FF",
                "nurses": "#EE82EE",
                "queens": "#8B008B",
                "drones": "#BA55D3"
            },
            "bombus_muscorum": {
                "primary": "#F0A500",    # Orange
                "secondary": "#FF8C00",  # Dark orange
                "workers": "#F0A500",
                "foragers": "#FF4500",
                "nurses": "#FFA500",
                "queens": "#B22222",
                "drones": "#CD853F"
            },
            "default": {
                "primary": "#808080",    # Gray
                "secondary": "#A0A0A0",  # Light gray
                "workers": "#808080",
                "foragers": "#696969",
                "nurses": "#A9A9A9",
                "queens": "#2F4F4F",
                "drones": "#778899"
            }
        }
        
        # Activity-based color modifications
        self.activity_modifiers = {
            BeeStatus.FORAGING: {"brightness": 1.2, "saturation": 1.1},
            BeeStatus.NURSING: {"brightness": 0.9, "saturation": 1.0},
            BeeStatus.DANCING: {"brightness": 1.3, "saturation": 1.2},
            BeeStatus.RESTING: {"brightness": 0.7, "saturation": 0.8},
            BeeStatus.DEAD: {"brightness": 0.3, "saturation": 0.5},
            BeeStatus.HIBERNATING: {"brightness": 0.5, "saturation": 0.7}
        }
        
        # Health-based alpha modifications
        self.health_alpha_map = {
            "excellent": 1.0,
            "good": 0.9,
            "fair": 0.7,
            "poor": 0.5,
            "critical": 0.3,
            "dead": 0.2
        }
        
    def get_agent_color(self, agent: Any, color_mode: str = "species") -> str:
        """Get color for an agent based on various criteria"""
        
        if color_mode == "species":
            return self._get_species_color(agent)
        elif color_mode == "role":
            return self._get_role_color(agent)
        elif color_mode == "activity":
            return self._get_activity_color(agent)
        elif color_mode == "health":
            return self._get_health_color(agent)
        else:
            return "#808080"
            
    def _get_species_color(self, agent: Any) -> str:
        """Get species-specific color"""
        species = getattr(agent, "species", "default")
        role = getattr(agent, "role", "workers")
        
        if species in self.species_colors:
            if role in self.species_colors[species]:
                return self.species_colors[species][role]
            else:
                return self.species_colors[species]["primary"]
        else:
            return self.species_colors["default"]["primary"]
            
    def _get_role_color(self, agent: Any) -> str:
        """Get role-specific color"""
        role = getattr(agent, "role", "workers")
        species = getattr(agent, "species", "default")
        
        if species in self.species_colors and role in self.species_colors[species]:
            return self.species_colors[species][role]
        else:
            return self.species_colors["default"]["primary"]
            
    def _get_activity_color(self, agent: Any) -> str:
        """Get activity-based color"""
        base_color = self._get_species_color(agent)
        activity = getattr(agent, "status", BeeStatus.ALIVE)
        
        # Apply activity modifiers
        if activity in self.activity_modifiers:
            # This is a simplified approach - in reality, you'd want to
            # modify the RGB values based on brightness and saturation
            return base_color
        else:
            return base_color
            
    def _get_health_color(self, agent: Any) -> str:
        """Get health-based color"""
        health = getattr(agent, "health", 1.0)
        
        if health > 0.8:
            return "#2ECC71"  # Green
        elif health > 0.6:
            return "#F39C12"  # Orange
        elif health > 0.4:
            return "#E74C3C"  # Red
        elif health > 0.2:
            return "#8E44AD"  # Purple
        else:
            return "#2C3E50"  # Dark
            
    def get_agent_alpha(self, agent: Any) -> float:
        """Get alpha transparency for an agent"""
        health = getattr(agent, "health", 1.0)
        status = getattr(agent, "status", BeeStatus.ALIVE)
        
        # Base alpha from health
        alpha = max(health, 0.3)
        
        # Modify based on status
        if status == BeeStatus.DEAD:
            alpha *= 0.5
        elif status == BeeStatus.HIBERNATING:
            alpha *= 0.7
        elif status == BeeStatus.DANCING:
            alpha = min(alpha * 1.2, 1.0)
            
        return alpha
        
    def get_agent_size(self, agent: Any, base_size: float = 8.0) -> float:
        """Get size for an agent marker"""
        
        # Base size modifications
        role = getattr(agent, "role", "workers")
        
        if role == "queens":
            return base_size * 2.0
        elif role == "drones":
            return base_size * 1.5
        elif role == "foragers":
            return base_size * 1.2
        else:
            return base_size
            
    def get_agent_marker(self, agent: Any) -> str:
        """Get marker style for an agent"""
        role = getattr(agent, "role", "workers")
        status = getattr(agent, "status", BeeStatus.ALIVE)
        
        if status == BeeStatus.DEAD:
            return "x"
        elif role == "queens":
            return "D"  # Diamond
        elif role == "drones":
            return "s"  # Square
        elif role == "foragers":
            return "o"  # Circle
        else:
            return "o"  # Circle


class AgentVisualizationEngine:
    """Main engine for agent visualization"""
    
    def __init__(self, display_manager: DisplayStateManager):
        self.display_manager = display_manager
        self.color_manager = SpeciesColorManager()
        self.conditional_viz = ConditionalVisualization(display_manager)
        self.logger = logging.getLogger(__name__)
        
        # Visualization state
        self.current_frame = None
        self.animation_frames = []
        self.figure = None
        self.axes = {}
        
        # Configuration
        self.config = {
            "figure_size": (12, 10),
            "dpi": 100,
            "background_color": "#F8F9FA",
            "grid_alpha": 0.3,
            "agent_base_size": 8.0,
            "patch_alpha": 0.6,
            "connection_alpha": 0.4,
            "path_alpha": 0.7,
            "animation_interval": 200,  # milliseconds
            "color_mode": "species",  # species, role, activity, health
            "show_trails": False,
            "trail_length": 10
        }
        
    def create_static_visualization(self, 
                                  agents: List[Any], 
                                  patches: Optional[List[SpatialPatch]] = None,
                                  connections: Optional[List[Dict[str, Any]]] = None,
                                  title: str = "Agent Visualization") -> plt.Figure:
        """Create a static visualization of agents"""
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.config["figure_size"], dpi=self.config["dpi"])
        fig.patch.set_facecolor(self.config["background_color"])
        
        # Draw layers in order
        if patches:
            self._draw_patches(ax, patches)
            
        if connections:
            self._draw_connections(ax, connections)
            
        self._draw_agents(ax, agents)
        
        # Customize plot
        self._customize_plot(ax, title)
        
        # Add legend and statistics
        if self.display_manager.get_toggle_state("show_legend"):
            self._add_legend(ax, agents)
            
        if self.display_manager.get_toggle_state("show_statistics"):
            self._add_statistics(ax, agents)
            
        plt.tight_layout()
        return fig
        
    def _draw_patches(self, ax: plt.Axes, patches: List[SpatialPatch]) -> None:
        """Draw patches on the visualization"""
        
        if not self.display_manager.get_toggle_state("show_resources"):
            return
            
        for patch in patches:
            if not self.conditional_viz.should_display_entity(patch, "patch"):
                continue
                
            # Get patch properties
            color = self.conditional_viz.get_color_for_entity(patch, "patch_type")
            alpha = self.config["patch_alpha"]
            
            # Draw patch as circle
            circle = Circle((patch.x, patch.y), 
                          radius=np.sqrt(patch.area) / 2,
                          facecolor=color, 
                          alpha=alpha,
                          edgecolor='black',
                          linewidth=0.5)
            ax.add_patch(circle)
            
            # Add label if enabled
            if self.display_manager.get_toggle_state("show_labels"):
                ax.annotate(str(patch.patch_id), 
                          (patch.x, patch.y),
                          ha='center', va='center',
                          fontsize=8, alpha=0.8)
                          
    def _draw_connections(self, ax: plt.Axes, connections: List[Dict[str, Any]]) -> None:
        """Draw connections between patches or agents"""
        
        if not self.display_manager.get_toggle_state("show_connectivity"):
            return
            
        for connection in connections:
            x1, y1 = connection["start"]
            x2, y2 = connection["end"]
            strength = connection.get("strength", 1.0)
            
            # Line properties based on strength
            linewidth = max(strength * 3, 0.5)
            alpha = self.config["connection_alpha"] * strength
            
            ax.plot([x1, x2], [y1, y2], 
                   color='gray', 
                   linewidth=linewidth, 
                   alpha=alpha,
                   zorder=1)
                   
    def _draw_agents(self, ax: plt.Axes, agents: List[Any]) -> None:
        """Draw agents on the visualization"""
        
        # Group agents by species and role for efficient plotting
        agent_groups = {}
        
        for agent in agents:
            if not self.conditional_viz.should_display_entity(agent, "agent"):
                continue
                
            # Get agent properties
            species = getattr(agent, "species", "default")
            role = getattr(agent, "role", "workers")
            
            key = (species, role)
            if key not in agent_groups:
                agent_groups[key] = {
                    "x": [], "y": [], "colors": [], "sizes": [], "alphas": [], "markers": []
                }
                
            # Get visualization properties
            color = self.color_manager.get_agent_color(agent, self.config["color_mode"])
            size = self.color_manager.get_agent_size(agent, self.config["agent_base_size"])
            alpha = self.color_manager.get_agent_alpha(agent)
            marker = self.color_manager.get_agent_marker(agent)
            
            # Add to group
            agent_groups[key]["x"].append(getattr(agent, "x", 0))
            agent_groups[key]["y"].append(getattr(agent, "y", 0))
            agent_groups[key]["colors"].append(color)
            agent_groups[key]["sizes"].append(size)
            agent_groups[key]["alphas"].append(alpha)
            agent_groups[key]["markers"].append(marker)
            
        # Plot each group
        for (species, role), group_data in agent_groups.items():
            if not group_data["x"]:
                continue
                
            # Use scatter plot for efficiency
            ax.scatter(group_data["x"], group_data["y"],
                      c=group_data["colors"],
                      s=group_data["sizes"],
                      alpha=group_data["alphas"],
                      edgecolors='black',
                      linewidth=0.5,
                      label=f"{species} {role}",
                      zorder=2)
                      
    def _customize_plot(self, ax: plt.Axes, title: str) -> None:
        """Customize plot appearance"""
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("X Coordinate", fontsize=12)
        ax.set_ylabel("Y Coordinate", fontsize=12)
        
        # Grid
        if self.display_manager.get_toggle_state("show_grid"):
            ax.grid(True, alpha=self.config["grid_alpha"])
            
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    def _add_legend(self, ax: plt.Axes, agents: List[Any]) -> None:
        """Add legend to the plot"""
        
        # Get unique species and roles
        species_roles = set()
        for agent in agents:
            species = getattr(agent, "species", "default")
            role = getattr(agent, "role", "workers")
            species_roles.add((species, role))
            
        # Create legend elements
        legend_elements = []
        for species, role in sorted(species_roles):
            color = self.color_manager.species_colors.get(species, {}).get(role, "#808080")
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color, markersize=8,
                                            label=f"{species} {role}"))
                                            
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
            
    def _add_statistics(self, ax: plt.Axes, agents: List[Any]) -> None:
        """Add statistics panel to the plot"""
        
        # Calculate statistics
        total_agents = len(agents)
        species_counts = {}
        role_counts = {}
        status_counts = {}
        
        for agent in agents:
            species = getattr(agent, "species", "default")
            role = getattr(agent, "role", "workers")
            status = getattr(agent, "status", BeeStatus.ALIVE)
            
            species_counts[species] = species_counts.get(species, 0) + 1
            role_counts[role] = role_counts.get(role, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            
        # Create statistics text
        stats_text = f"Total Agents: {total_agents}\n"
        stats_text += f"Species: {len(species_counts)}\n"
        stats_text += f"Active: {status_counts.get(BeeStatus.ALIVE, 0)}\n"
        stats_text += f"Foraging: {status_counts.get(BeeStatus.FORAGING, 0)}\n"
        
        # Add text box
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=10)
                
    def create_animated_visualization(self, 
                                    frame_data: List[VisualizationFrame],
                                    output_path: Optional[str] = None) -> animation.FuncAnimation:
        """Create an animated visualization"""
        
        # Set up figure
        fig, ax = plt.subplots(figsize=self.config["figure_size"], dpi=self.config["dpi"])
        fig.patch.set_facecolor(self.config["background_color"])
        
        # Animation function
        def animate(frame_idx):
            ax.clear()
            
            if frame_idx < len(frame_data):
                frame = frame_data[frame_idx]
                
                # Draw patches
                if frame.patches:
                    self._draw_patches(ax, frame.patches)
                    
                # Draw connections
                if frame.connections:
                    self._draw_connections(ax, frame.connections)
                    
                # Draw agents
                self._draw_agents(ax, frame.agents)
                
                # Customize plot
                self._customize_plot(ax, f"Agent Visualization - Frame {frame_idx + 1}")
                
                # Add timestamp
                ax.text(0.02, 0.02, f"Time: {frame.timestamp.strftime('%H:%M:%S')}", 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                       
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(frame_data),
                                     interval=self.config["animation_interval"],
                                     blit=False, repeat=True)
                                     
        # Save animation if output path provided
        if output_path:
            anim.save(output_path, writer='pillow', fps=5)
            
        return anim
        
    def create_comparative_visualization(self, 
                                       agent_groups: Dict[str, List[Any]],
                                       title: str = "Comparative Agent Visualization") -> plt.Figure:
        """Create a comparative visualization of different agent groups"""
        
        n_groups = len(agent_groups)
        cols = min(n_groups, 3)
        rows = (n_groups + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.config["figure_size"][0] * cols, 
                                                     self.config["figure_size"][1] * rows))
        
        if n_groups == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
            
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for idx, (group_name, agents) in enumerate(agent_groups.items()):
            row = idx // cols
            col = idx % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
                
            # Draw agents for this group
            self._draw_agents(ax, agents)
            
            # Customize subplot
            ax.set_title(group_name, fontsize=14)
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.set_aspect('equal')
            
            if self.display_manager.get_toggle_state("show_grid"):
                ax.grid(True, alpha=self.config["grid_alpha"])
                
        # Hide empty subplots
        for idx in range(n_groups, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            ax.set_visible(False)
            
        plt.tight_layout()
        return fig
        
    def export_visualization_config(self, output_path: str) -> None:
        """Export visualization configuration"""
        
        config_data = {
            "visualization_config": self.config,
            "species_colors": self.color_manager.species_colors,
            "display_toggles": {
                toggle: self.display_manager.get_toggle_state(toggle)
                for toggle in self.display_manager.current_toggles
            },
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        self.logger.info(f"Visualization configuration exported to {output_path}")
        
    def set_visualization_config(self, **kwargs) -> None:
        """Update visualization configuration"""
        
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                
    def get_visualization_summary(self, agents: List[Any]) -> Dict[str, Any]:
        """Get summary of visualization data"""
        
        summary = {
            "total_agents": len(agents),
            "species_distribution": {},
            "role_distribution": {},
            "status_distribution": {},
            "spatial_bounds": {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0}
        }
        
        if not agents:
            return summary
            
        # Calculate distributions
        for agent in agents:
            species = getattr(agent, "species", "default")
            role = getattr(agent, "role", "workers")
            status = getattr(agent, "status", BeeStatus.ALIVE)
            
            summary["species_distribution"][species] = summary["species_distribution"].get(species, 0) + 1
            summary["role_distribution"][role] = summary["role_distribution"].get(role, 0) + 1
            summary["status_distribution"][str(status)] = summary["status_distribution"].get(str(status), 0) + 1
            
        # Calculate spatial bounds
        x_coords = [getattr(agent, "x", 0) for agent in agents]
        y_coords = [getattr(agent, "y", 0) for agent in agents]
        
        summary["spatial_bounds"] = {
            "x_min": min(x_coords),
            "x_max": max(x_coords),
            "y_min": min(y_coords),
            "y_max": max(y_coords)
        }
        
        return summary