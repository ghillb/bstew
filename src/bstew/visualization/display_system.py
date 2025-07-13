"""
Conditional Display System for BSTEW
===================================

NetLogo-compatible display toggle system with conditional visualization logic,
display state management, and visualization configuration persistence.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import logging
from datetime import datetime
import matplotlib.pyplot as plt


class DisplayToggle(Enum):
    """Display toggle options matching NetLogo interface"""
    SHOW_COHORTS = "show_cohorts"
    SHOW_DEAD_COLS = "show_dead_cols"
    SHOW_PATHS = "show_paths"
    SHOW_RESOURCES = "show_resources"
    SHOW_TERRITORIES = "show_territories"
    SHOW_ACTIVITY = "show_activity"
    SHOW_SPECIES = "show_species"
    SHOW_HEALTH = "show_health"
    SHOW_CONNECTIVITY = "show_connectivity"
    SHOW_GRID = "show_grid"
    SHOW_LABELS = "show_labels"
    SHOW_LEGEND = "show_legend"
    SHOW_STATISTICS = "show_statistics"
    SHOW_STEWARDSHIP = "show_stewardship"
    SHOW_MANAGEMENT = "show_management"


class ColorScheme(Enum):
    """Color schemes for different display modes"""
    SPECIES_COLORS = "species_colors"
    ACTIVITY_COLORS = "activity_colors"
    HEALTH_COLORS = "health_colors"
    RESOURCE_COLORS = "resource_colors"
    CONNECTIVITY_COLORS = "connectivity_colors"
    MANAGEMENT_COLORS = "management_colors"
    CUSTOM = "custom"


@dataclass
class DisplayConfiguration:
    """Configuration for a specific display mode"""
    name: str
    description: str
    toggle_states: Dict[DisplayToggle, bool]
    color_scheme: ColorScheme
    custom_colors: Optional[Dict[str, str]] = None
    visibility_filters: Optional[Dict[str, Any]] = None
    display_parameters: Optional[Dict[str, Any]] = None


class DisplayStateManager:
    """Manages display states and configuration persistence"""
    
    def __init__(self, config_file: str = "artifacts/visualization/display_config.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Current display state
        self.current_toggles: Dict[DisplayToggle, bool] = {}
        self.current_color_scheme: ColorScheme = ColorScheme.SPECIES_COLORS
        self.custom_colors: Dict[str, str] = {}
        self.visibility_filters: Dict[str, Any] = {}
        self.display_parameters: Dict[str, Any] = {}
        
        # Predefined configurations
        self.predefined_configs: Dict[str, DisplayConfiguration] = {}
        self._initialize_predefined_configs()
        
        # Load saved configuration
        self.load_configuration()
        
    def _initialize_predefined_configs(self) -> None:
        """Initialize predefined display configurations"""
        
        # Default configuration
        self.predefined_configs["default"] = DisplayConfiguration(
            name="Default View",
            description="Standard display with basic information",
            toggle_states={
                DisplayToggle.SHOW_COHORTS: True,
                DisplayToggle.SHOW_DEAD_COLS: False,
                DisplayToggle.SHOW_PATHS: False,
                DisplayToggle.SHOW_RESOURCES: True,
                DisplayToggle.SHOW_TERRITORIES: False,
                DisplayToggle.SHOW_ACTIVITY: False,
                DisplayToggle.SHOW_SPECIES: True,
                DisplayToggle.SHOW_HEALTH: False,
                DisplayToggle.SHOW_CONNECTIVITY: False,
                DisplayToggle.SHOW_GRID: True,
                DisplayToggle.SHOW_LABELS: False,
                DisplayToggle.SHOW_LEGEND: True,
                DisplayToggle.SHOW_STATISTICS: False,
                DisplayToggle.SHOW_STEWARDSHIP: False,
                DisplayToggle.SHOW_MANAGEMENT: False
            },
            color_scheme=ColorScheme.SPECIES_COLORS
        )
        
        # Research configuration
        self.predefined_configs["research"] = DisplayConfiguration(
            name="Research View",
            description="Detailed display for research and analysis",
            toggle_states={
                DisplayToggle.SHOW_COHORTS: True,
                DisplayToggle.SHOW_DEAD_COLS: True,
                DisplayToggle.SHOW_PATHS: True,
                DisplayToggle.SHOW_RESOURCES: True,
                DisplayToggle.SHOW_TERRITORIES: True,
                DisplayToggle.SHOW_ACTIVITY: True,
                DisplayToggle.SHOW_SPECIES: True,
                DisplayToggle.SHOW_HEALTH: True,
                DisplayToggle.SHOW_CONNECTIVITY: True,
                DisplayToggle.SHOW_GRID: True,
                DisplayToggle.SHOW_LABELS: True,
                DisplayToggle.SHOW_LEGEND: True,
                DisplayToggle.SHOW_STATISTICS: True,
                DisplayToggle.SHOW_STEWARDSHIP: True,
                DisplayToggle.SHOW_MANAGEMENT: True
            },
            color_scheme=ColorScheme.ACTIVITY_COLORS
        )
        
        # Management configuration
        self.predefined_configs["management"] = DisplayConfiguration(
            name="Management View",
            description="Focus on management and stewardship options",
            toggle_states={
                DisplayToggle.SHOW_COHORTS: False,
                DisplayToggle.SHOW_DEAD_COLS: False,
                DisplayToggle.SHOW_PATHS: False,
                DisplayToggle.SHOW_RESOURCES: True,
                DisplayToggle.SHOW_TERRITORIES: True,
                DisplayToggle.SHOW_ACTIVITY: False,
                DisplayToggle.SHOW_SPECIES: False,
                DisplayToggle.SHOW_HEALTH: False,
                DisplayToggle.SHOW_CONNECTIVITY: True,
                DisplayToggle.SHOW_GRID: False,
                DisplayToggle.SHOW_LABELS: False,
                DisplayToggle.SHOW_LEGEND: True,
                DisplayToggle.SHOW_STATISTICS: False,
                DisplayToggle.SHOW_STEWARDSHIP: True,
                DisplayToggle.SHOW_MANAGEMENT: True
            },
            color_scheme=ColorScheme.MANAGEMENT_COLORS
        )
        
        # Health monitoring configuration
        self.predefined_configs["health"] = DisplayConfiguration(
            name="Health Monitoring",
            description="Focus on colony health and mortality",
            toggle_states={
                DisplayToggle.SHOW_COHORTS: True,
                DisplayToggle.SHOW_DEAD_COLS: True,
                DisplayToggle.SHOW_PATHS: False,
                DisplayToggle.SHOW_RESOURCES: False,
                DisplayToggle.SHOW_TERRITORIES: False,
                DisplayToggle.SHOW_ACTIVITY: True,
                DisplayToggle.SHOW_SPECIES: False,
                DisplayToggle.SHOW_HEALTH: True,
                DisplayToggle.SHOW_CONNECTIVITY: False,
                DisplayToggle.SHOW_GRID: False,
                DisplayToggle.SHOW_LABELS: False,
                DisplayToggle.SHOW_LEGEND: True,
                DisplayToggle.SHOW_STATISTICS: True,
                DisplayToggle.SHOW_STEWARDSHIP: False,
                DisplayToggle.SHOW_MANAGEMENT: False
            },
            color_scheme=ColorScheme.HEALTH_COLORS
        )
        
    def get_toggle_state(self, toggle: DisplayToggle) -> bool:
        """Get the current state of a display toggle"""
        return self.current_toggles.get(toggle, False)
        
    def set_toggle_state(self, toggle: DisplayToggle, state: bool) -> None:
        """Set the state of a display toggle"""
        self.current_toggles[toggle] = state
        self.save_configuration()
        
    def toggle_display(self, toggle: DisplayToggle) -> bool:
        """Toggle a display option and return new state"""
        current_state = self.get_toggle_state(toggle)
        new_state = not current_state
        self.set_toggle_state(toggle, new_state)
        return new_state
        
    def apply_configuration(self, config_name: str) -> bool:
        """Apply a predefined configuration"""
        if config_name not in self.predefined_configs:
            self.logger.error(f"Configuration '{config_name}' not found")
            return False
            
        config = self.predefined_configs[config_name]
        
        # Apply toggle states
        self.current_toggles = config.toggle_states.copy()
        
        # Apply color scheme
        self.current_color_scheme = config.color_scheme
        
        # Apply custom colors if provided
        if config.custom_colors:
            self.custom_colors = config.custom_colors.copy()
            
        # Apply visibility filters
        if config.visibility_filters:
            self.visibility_filters = config.visibility_filters.copy()
            
        # Apply display parameters
        if config.display_parameters:
            self.display_parameters = config.display_parameters.copy()
            
        self.save_configuration()
        
        self.logger.info(f"Applied configuration: {config.name}")
        return True
        
    def create_custom_configuration(self, 
                                  name: str, 
                                  description: str,
                                  toggle_states: Optional[Dict[DisplayToggle, bool]] = None) -> None:
        """Create a custom configuration from current state"""
        
        if toggle_states is None:
            toggle_states = self.current_toggles.copy()
            
        config = DisplayConfiguration(
            name=name,
            description=description,
            toggle_states=toggle_states,
            color_scheme=self.current_color_scheme,
            custom_colors=self.custom_colors.copy() if self.custom_colors else None,
            visibility_filters=self.visibility_filters.copy() if self.visibility_filters else None,
            display_parameters=self.display_parameters.copy() if self.display_parameters else None
        )
        
        self.predefined_configs[name.lower().replace(" ", "_")] = config
        self.save_configuration()
        
        self.logger.info(f"Created custom configuration: {name}")
        
    def get_available_configurations(self) -> List[str]:
        """Get list of available configurations"""
        return list(self.predefined_configs.keys())
        
    def set_color_scheme(self, scheme: ColorScheme) -> None:
        """Set the current color scheme"""
        self.current_color_scheme = scheme
        self.save_configuration()
        
    def set_custom_color(self, category: str, color: str) -> None:
        """Set a custom color for a category"""
        self.custom_colors[category] = color
        self.save_configuration()
        
    def set_visibility_filter(self, filter_name: str, filter_value: Any) -> None:
        """Set a visibility filter"""
        self.visibility_filters[filter_name] = filter_value
        self.save_configuration()
        
    def set_display_parameter(self, parameter: str, value: Any) -> None:
        """Set a display parameter"""
        self.display_parameters[parameter] = value
        self.save_configuration()
        
    def save_configuration(self) -> None:
        """Save current configuration to file"""
        try:
            config_data = {
                "current_toggles": {toggle.value: state for toggle, state in self.current_toggles.items()},
                "current_color_scheme": self.current_color_scheme.value,
                "custom_colors": self.custom_colors,
                "visibility_filters": self.visibility_filters,
                "display_parameters": self.display_parameters,
                "predefined_configs": {
                    name: {
                        "name": config.name,
                        "description": config.description,
                        "toggle_states": {toggle.value: state for toggle, state in config.toggle_states.items()},
                        "color_scheme": config.color_scheme.value,
                        "custom_colors": config.custom_colors,
                        "visibility_filters": config.visibility_filters,
                        "display_parameters": config.display_parameters
                    }
                    for name, config in self.predefined_configs.items()
                },
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            
    def load_configuration(self) -> None:
        """Load configuration from file"""
        try:
            if not self.config_file.exists():
                # Initialize with default configuration
                self.apply_configuration("default")
                return
                
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                
            # Load current toggles
            if "current_toggles" in config_data:
                self.current_toggles = {
                    DisplayToggle(toggle): state 
                    for toggle, state in config_data["current_toggles"].items()
                }
                
            # Load color scheme
            if "current_color_scheme" in config_data:
                self.current_color_scheme = ColorScheme(config_data["current_color_scheme"])
                
            # Load custom colors
            if "custom_colors" in config_data:
                self.custom_colors = config_data["custom_colors"]
                
            # Load visibility filters
            if "visibility_filters" in config_data:
                self.visibility_filters = config_data["visibility_filters"]
                
            # Load display parameters
            if "display_parameters" in config_data:
                self.display_parameters = config_data["display_parameters"]
                
            # Load custom predefined configs
            if "predefined_configs" in config_data:
                for name, config_dict in config_data["predefined_configs"].items():
                    if name not in self.predefined_configs:  # Don't override built-in configs
                        toggle_states = {
                            DisplayToggle(toggle): state 
                            for toggle, state in config_dict["toggle_states"].items()
                        }
                        
                        config = DisplayConfiguration(
                            name=config_dict["name"],
                            description=config_dict["description"],
                            toggle_states=toggle_states,
                            color_scheme=ColorScheme(config_dict["color_scheme"]),
                            custom_colors=config_dict.get("custom_colors"),
                            visibility_filters=config_dict.get("visibility_filters"),
                            display_parameters=config_dict.get("display_parameters")
                        )
                        
                        self.predefined_configs[name] = config
                        
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Fall back to default configuration
            self.apply_configuration("default")


class ConditionalVisualization:
    """Conditional visualization logic based on display toggles"""
    
    def __init__(self, display_manager: DisplayStateManager):
        self.display_manager = display_manager
        self.logger = logging.getLogger(__name__)
        
        # Color schemes
        self.color_schemes = self._initialize_color_schemes()
        
    def _initialize_color_schemes(self) -> Dict[ColorScheme, Dict[str, Any]]:
        """Initialize color schemes for different display modes"""
        schemes = {}
        
        # Species colors
        schemes[ColorScheme.SPECIES_COLORS] = {
            "apis_mellifera": "#FFD700",  # Gold
            "bombus_terrestris": "#FF6B6B",  # Red
            "bombus_lapidarius": "#4ECDC4",  # Teal
            "bombus_pascuorum": "#45B7D1",  # Blue
            "bombus_hortorum": "#96CEB4",  # Green
            "bombus_ruderatus": "#FFEAA7",  # Yellow
            "bombus_humilis": "#DDA0DD",  # Plum
            "bombus_muscorum": "#F0A500",  # Orange
            "default": "#808080"  # Gray
        }
        
        # Activity colors
        schemes[ColorScheme.ACTIVITY_COLORS] = {
            "foraging": "#2ECC71",  # Green
            "nursing": "#3498DB",  # Blue
            "building": "#E67E22",  # Orange
            "guarding": "#E74C3C",  # Red
            "dancing": "#9B59B6",  # Purple
            "resting": "#95A5A6",  # Gray
            "dead": "#34495E",  # Dark gray
            "hibernating": "#17A2B8",  # Cyan
            "default": "#BDC3C7"  # Light gray
        }
        
        # Health colors
        schemes[ColorScheme.HEALTH_COLORS] = {
            "excellent": "#2ECC71",  # Green
            "good": "#F39C12",  # Orange
            "poor": "#E74C3C",  # Red
            "critical": "#8E44AD",  # Purple
            "dead": "#2C3E50",  # Dark
            "default": "#95A5A6"  # Gray
        }
        
        # Resource colors
        schemes[ColorScheme.RESOURCE_COLORS] = {
            "high": "#2ECC71",  # Green
            "medium": "#F39C12",  # Orange
            "low": "#E74C3C",  # Red
            "none": "#95A5A6",  # Gray
            "nectar": "#FFD700",  # Gold
            "pollen": "#FF6B6B",  # Red
            "default": "#BDC3C7"  # Light gray
        }
        
        # Connectivity colors
        schemes[ColorScheme.CONNECTIVITY_COLORS] = {
            "high": "#2ECC71",  # Green
            "medium": "#F39C12",  # Orange
            "low": "#E74C3C",  # Red
            "isolated": "#8E44AD",  # Purple
            "default": "#95A5A6"  # Gray
        }
        
        # Management colors
        schemes[ColorScheme.MANAGEMENT_COLORS] = {
            "protected": "#2ECC71",  # Green
            "managed": "#3498DB",  # Blue
            "restoration": "#E67E22",  # Orange
            "development": "#E74C3C",  # Red
            "unmanaged": "#95A5A6",  # Gray
            "default": "#BDC3C7"  # Light gray
        }
        
        return schemes
        
    def get_color_for_entity(self, entity: Any, attribute: str) -> str:
        """Get color for an entity based on current color scheme"""
        current_scheme = self.display_manager.current_color_scheme
        
        if current_scheme in self.color_schemes:
            color_map = self.color_schemes[current_scheme]
            
            # Get attribute value
            if hasattr(entity, attribute):
                value = getattr(entity, attribute)
                
                # Handle different value types
                if isinstance(value, str):
                    return color_map.get(value, color_map.get("default", "#808080"))
                elif isinstance(value, (int, float)):
                    # Map numeric values to categories
                    if attribute in ["health", "habitat_quality", "connectivity_index"]:
                        if value > 0.8:
                            return color_map.get("high", "#2ECC71")
                        elif value > 0.5:
                            return color_map.get("medium", "#F39C12")
                        elif value > 0.2:
                            return color_map.get("low", "#E74C3C")
                        else:
                            return color_map.get("none", "#95A5A6")
                            
            return color_map.get("default", "#808080")
            
        # Use custom colors if available
        if self.display_manager.custom_colors:
            return self.display_manager.custom_colors.get(attribute, "#808080")
            
        return "#808080"
        
    def should_display_entity(self, entity: Any, entity_type: str) -> bool:
        """Determine if an entity should be displayed based on current toggles"""
        
        # Check basic display toggles
        if entity_type == "cohort" and not self.display_manager.get_toggle_state(DisplayToggle.SHOW_COHORTS):
            return False
            
        if entity_type == "dead_colony" and not self.display_manager.get_toggle_state(DisplayToggle.SHOW_DEAD_COLS):
            return False
            
        if entity_type == "path" and not self.display_manager.get_toggle_state(DisplayToggle.SHOW_PATHS):
            return False
            
        if entity_type == "resource" and not self.display_manager.get_toggle_state(DisplayToggle.SHOW_RESOURCES):
            return False
            
        if entity_type == "territory" and not self.display_manager.get_toggle_state(DisplayToggle.SHOW_TERRITORIES):
            return False
            
        # Check visibility filters
        if self.display_manager.visibility_filters:
            for filter_name, filter_value in self.display_manager.visibility_filters.items():
                if hasattr(entity, filter_name):
                    entity_value = getattr(entity, filter_name)
                    
                    # Handle different filter types
                    if isinstance(filter_value, dict):
                        # Range filter
                        if "min" in filter_value and entity_value < filter_value["min"]:
                            return False
                        if "max" in filter_value and entity_value > filter_value["max"]:
                            return False
                    elif isinstance(filter_value, (list, tuple)):
                        # List filter
                        if entity_value not in filter_value:
                            return False
                    else:
                        # Exact match filter
                        if entity_value != filter_value:
                            return False
                            
        return True
        
    def get_display_properties(self, entity: Any, entity_type: str) -> Dict[str, Any]:
        """Get display properties for an entity"""
        properties = {
            "visible": self.should_display_entity(entity, entity_type),
            "color": "#808080",
            "size": 5,
            "alpha": 1.0,
            "marker": "o",
            "label": "",
            "show_label": False
        }
        
        if not properties["visible"]:
            return properties
            
        # Determine color based on current scheme
        if self.display_manager.get_toggle_state(DisplayToggle.SHOW_SPECIES):
            properties["color"] = self.get_color_for_entity(entity, "species")
        elif self.display_manager.get_toggle_state(DisplayToggle.SHOW_ACTIVITY):
            properties["color"] = self.get_color_for_entity(entity, "activity")
        elif self.display_manager.get_toggle_state(DisplayToggle.SHOW_HEALTH):
            properties["color"] = self.get_color_for_entity(entity, "health")
        elif self.display_manager.get_toggle_state(DisplayToggle.SHOW_CONNECTIVITY):
            properties["color"] = self.get_color_for_entity(entity, "connectivity_index")
        
        # Determine size based on entity attributes
        if hasattr(entity, "population") and entity_type == "cohort":
            # Scale size based on population
            properties["size"] = min(max(entity.population / 10, 2), 20)
        elif hasattr(entity, "area") and entity_type == "resource":
            # Scale size based on area
            properties["size"] = min(max(entity.area / 100, 2), 15)
            
        # Determine alpha based on health or activity
        if hasattr(entity, "health"):
            properties["alpha"] = max(entity.health, 0.3)
        elif hasattr(entity, "is_active"):
            properties["alpha"] = 1.0 if entity.is_active else 0.5
            
        # Determine marker based on entity type
        if entity_type == "cohort":
            properties["marker"] = "o"
        elif entity_type == "resource":
            properties["marker"] = "s"
        elif entity_type == "dead_colony":
            properties["marker"] = "x"
        elif entity_type == "territory":
            properties["marker"] = "^"
            
        # Show labels if enabled
        if self.display_manager.get_toggle_state(DisplayToggle.SHOW_LABELS):
            properties["show_label"] = True
            if hasattr(entity, "name"):
                properties["label"] = entity.name
            elif hasattr(entity, "id"):
                properties["label"] = str(entity.id)
                
        return properties
        
    def apply_conditional_styling(self, ax: plt.Axes, entities: List[Any], entity_type: str) -> None:
        """Apply conditional styling to a plot"""
        
        # Clear previous plots
        ax.clear()
        
        # Get display properties for all entities
        display_data = []
        for entity in entities:
            props = self.get_display_properties(entity, entity_type)
            if props["visible"]:
                display_data.append({
                    "entity": entity,
                    "x": getattr(entity, "x", 0),
                    "y": getattr(entity, "y", 0),
                    "color": props["color"],
                    "size": props["size"],
                    "alpha": props["alpha"],
                    "marker": props["marker"],
                    "label": props["label"],
                    "show_label": props["show_label"]
                })
                
        if not display_data:
            return
            
        # Create scatter plot
        x_coords = [d["x"] for d in display_data]
        y_coords = [d["y"] for d in display_data]
        colors = [d["color"] for d in display_data]
        sizes = [d["size"] for d in display_data]
        alphas = [d["alpha"] for d in display_data]
        
        # Plot points
        _scatter = ax.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=alphas, 
                           edgecolors='black', linewidth=0.5)
        
        # Add labels if enabled
        if self.display_manager.get_toggle_state(DisplayToggle.SHOW_LABELS):
            for d in display_data:
                if d["show_label"] and d["label"]:
                    ax.annotate(d["label"], (d["x"], d["y"]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
                               
        # Add grid if enabled
        if self.display_manager.get_toggle_state(DisplayToggle.SHOW_GRID):
            ax.grid(True, alpha=0.3)
            
        # Add legend if enabled
        if self.display_manager.get_toggle_state(DisplayToggle.SHOW_LEGEND):
            self._add_legend(ax, display_data)
            
        # Set axis properties
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"{entity_type.title()} Display")
        
    def _add_legend(self, ax: plt.Axes, display_data: List[Dict[str, Any]]) -> None:
        """Add legend to the plot"""
        
        # Get unique colors and their meanings
        color_meanings = {}
        current_scheme = self.display_manager.current_color_scheme
        
        if current_scheme in self.color_schemes:
            color_map = self.color_schemes[current_scheme]
            
            # Reverse mapping from color to meaning
            for meaning, color in color_map.items():
                if color in [d["color"] for d in display_data]:
                    color_meanings[color] = meaning
                    
        # Create legend elements
        legend_elements = []
        for color, meaning in color_meanings.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, 
                                            label=meaning.title()))
                                            
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
            
    def export_display_configuration(self, output_path: str) -> None:
        """Export current display configuration for documentation"""
        
        config_info = {
            "display_toggles": {
                toggle.value: self.display_manager.get_toggle_state(toggle)
                for toggle in DisplayToggle
            },
            "color_scheme": self.display_manager.current_color_scheme.value,
            "custom_colors": self.display_manager.custom_colors,
            "visibility_filters": self.display_manager.visibility_filters,
            "display_parameters": self.display_manager.display_parameters,
            "available_configurations": self.display_manager.get_available_configurations(),
            "color_schemes": {
                scheme.value: colors
                for scheme, colors in self.color_schemes.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_info, f, indent=2)
            
        self.logger.info(f"Display configuration exported to {output_path}")


class DisplayControlInterface:
    """Interface for controlling display options"""
    
    def __init__(self, display_manager: DisplayStateManager):
        self.display_manager = display_manager
        self.conditional_viz = ConditionalVisualization(display_manager)
        
    def toggle_display_option(self, option: str) -> bool:
        """Toggle a display option by name"""
        try:
            toggle = DisplayToggle(option)
            return self.display_manager.toggle_display(toggle)
        except ValueError:
            return False
            
    def set_display_configuration(self, config_name: str) -> bool:
        """Set a display configuration by name"""
        return self.display_manager.apply_configuration(config_name)
        
    def get_display_status(self) -> Dict[str, Any]:
        """Get current display status"""
        return {
            "toggles": {
                toggle.value: self.display_manager.get_toggle_state(toggle)
                for toggle in DisplayToggle
            },
            "color_scheme": self.display_manager.current_color_scheme.value,
            "available_configs": self.display_manager.get_available_configurations()
        }
        
    def set_visibility_filter(self, filter_name: str, filter_value: Any) -> None:
        """Set a visibility filter"""
        self.display_manager.set_visibility_filter(filter_name, filter_value)
        
    def clear_visibility_filters(self) -> None:
        """Clear all visibility filters"""
        self.display_manager.visibility_filters.clear()
        self.display_manager.save_configuration()
        
    def create_visualization(self, 
                           entities: List[Any], 
                           entity_type: str,
                           output_path: Optional[str] = None) -> plt.Figure:
        """Create a visualization with current display settings"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Apply conditional styling
        self.conditional_viz.apply_conditional_styling(ax, entities, entity_type)
        
        # Save if output path provided
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig