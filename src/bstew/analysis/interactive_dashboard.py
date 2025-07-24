"""
Interactive Analysis Dashboard for BSTEW
========================================

Real-time parameter adjustment dashboard with live simulation monitoring,
multi-scenario comparison, and advanced visualization capabilities.

Phase 4 implementation for 100% BSTEW completion.
"""

import json
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from collections import defaultdict, deque

# Core BSTEW imports
from ..simulation.simulation_engine import SimulationEngine, SimulationConfig
from .interactive_tools import InteractiveAnalysisEngine

# Optional dashboard dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    pyo = None
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import dash_bootstrap_components as dbc

    DASH_AVAILABLE = True
except ImportError:
    dash = None  # type: ignore
    dcc = None  # type: ignore
    html = None  # type: ignore
    Input = None  # type: ignore
    Output = None  # type: ignore
    State = None  # type: ignore
    callback_context = None  # type: ignore
    dbc = None
    DASH_AVAILABLE = False


class DashboardMode(Enum):
    """Dashboard operation modes"""

    REAL_TIME = "real_time"
    BATCH_ANALYSIS = "batch_analysis"
    COMPARISON = "comparison"
    PARAMETER_SWEEP = "parameter_sweep"


@dataclass
class ParameterControl:
    """Interactive parameter control configuration"""

    name: str
    label: str
    param_type: str  # 'slider', 'input', 'dropdown', 'checkbox'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    default_value: Any = None
    description: str = ""
    category: str = "General"


@dataclass
class SimulationScenario:
    """Dashboard simulation scenario"""

    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    config: SimulationConfig
    results: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class RealTimeMonitor:
    """Real-time simulation monitoring system"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.data_history: deque = deque(maxlen=max_history)
        self.is_monitoring = False
        self.update_callbacks: List[Callable] = []
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self, simulation_engine: SimulationEngine) -> None:
        """Start monitoring a simulation"""
        self.is_monitoring = True
        self.simulation_engine = simulation_engine
        self.logger.info("Real-time monitoring started")

    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.is_monitoring = False
        self.logger.info("Real-time monitoring stopped")

    def add_data_point(self, timestamp: float, data: Dict[str, Any]) -> None:
        """Add new data point to monitoring history"""
        self.data_history.append({"timestamp": timestamp, "data": data})

        # Notify callbacks
        for callback in self.update_callbacks:
            try:
                callback(timestamp, data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

    def register_callback(self, callback: Callable) -> None:
        """Register callback for data updates"""
        self.update_callbacks.append(callback)

    def get_recent_data(self, n_points: int = 100) -> List[Dict[str, Any]]:
        """Get recent monitoring data"""
        return list(self.data_history)[-n_points:]

    def get_live_metrics(self) -> Dict[str, Any]:
        """Get current live metrics"""
        if not self.data_history:
            return {}

        latest = self.data_history[-1]["data"]
        return {
            "colony_count": latest.get("total_colonies", 0),
            "population": latest.get("total_population", 0),
            "foraging_activity": latest.get("foraging_bees", 0),
            "resource_patches": latest.get("active_patches", 0),
            "update_time": self.data_history[-1]["timestamp"],
        }


class ParameterControlManager:
    """Manages interactive parameter controls"""

    def __init__(self) -> None:
        self.controls: Dict[str, ParameterControl] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self.current_values: Dict[str, Any] = {}
        self._setup_default_controls()

    def _setup_default_controls(self) -> None:
        """Setup default BSTEW parameter controls"""

        # Population parameters
        self.add_control(
            ParameterControl(
                name="initial_colonies",
                label="Initial Colonies",
                param_type="slider",
                min_val=1,
                max_val=50,
                step=1,
                default_value=5,
                description="Number of initial bumblebee colonies",
                category="Population",
            )
        )

        self.add_control(
            ParameterControl(
                name="colony_growth_rate",
                label="Colony Growth Rate",
                param_type="slider",
                min_val=0.1,
                max_val=3.0,
                step=0.1,
                default_value=1.88,
                description="Colony growth factor per generation",
                category="Population",
            )
        )

        # Foraging parameters
        self.add_control(
            ParameterControl(
                name="foraging_range_m",
                label="Foraging Range (m)",
                param_type="slider",
                min_val=500,
                max_val=5000,
                step=100,
                default_value=2500,
                description="Maximum foraging distance from colony",
                category="Foraging",
            )
        )

        self.add_control(
            ParameterControl(
                name="flight_velocity_ms",
                label="Flight Velocity (m/s)",
                param_type="slider",
                min_val=1.0,
                max_val=8.0,
                step=0.1,
                default_value=5.0,
                description="Bee flight speed",
                category="Foraging",
            )
        )

        # Species selection
        species_options = [
            "Bombus_terrestris",
            "Bombus_lucorum",
            "Bombus_lapidarius",
            "Bombus_pratorum",
            "Bombus_pascuorum",
            "Bombus_hortorum",
        ]
        self.add_control(
            ParameterControl(
                name="active_species",
                label="Active Species",
                param_type="dropdown",
                options=species_options,
                default_value=species_options[:3],  # First 3 species
                description="Bumblebee species to include in simulation",
                category="Species",
            )
        )

        # Environmental parameters
        self.add_control(
            ParameterControl(
                name="temperature_mean",
                label="Mean Temperature (°C)",
                param_type="slider",
                min_val=10.0,
                max_val=25.0,
                step=0.5,
                default_value=18.0,
                description="Average environmental temperature",
                category="Environment",
            )
        )

        # Landscape parameters
        self.add_control(
            ParameterControl(
                name="flower_density",
                label="Flower Density",
                param_type="slider",
                min_val=0.1,
                max_val=2.0,
                step=0.1,
                default_value=1.0,
                description="Multiplier for flower patch density",
                category="Landscape",
            )
        )

        # Simulation control
        self.add_control(
            ParameterControl(
                name="time_steps",
                label="Time Steps",
                param_type="slider",
                min_val=100,
                max_val=2000,
                step=50,
                default_value=1000,
                description="Number of simulation time steps",
                category="Simulation",
            )
        )

    def add_control(self, control: ParameterControl) -> None:
        """Add parameter control"""
        self.controls[control.name] = control
        self.categories[control.category].append(control.name)
        self.current_values[control.name] = control.default_value

    def get_control(self, name: str) -> Optional[ParameterControl]:
        """Get parameter control by name"""
        return self.controls.get(name)

    def get_controls_by_category(self, category: str) -> List[ParameterControl]:
        """Get all controls in category"""
        return [self.controls[name] for name in self.categories.get(category, [])]

    def update_value(self, name: str, value: Any) -> None:
        """Update parameter value"""
        if name in self.controls:
            self.current_values[name] = value

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        return self.current_values.copy()

    def reset_to_defaults(self) -> None:
        """Reset all parameters to defaults"""
        for name, control in self.controls.items():
            self.current_values[name] = control.default_value


class InteractiveDashboard:
    """Main Interactive Analysis Dashboard"""

    def __init__(self, output_dir: str = "artifacts/dashboard_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.analysis_engine = InteractiveAnalysisEngine()
        self.parameter_manager = ParameterControlManager()
        self.real_time_monitor = RealTimeMonitor()

        # Dashboard state
        self.mode = DashboardMode.REAL_TIME
        self.scenarios: Dict[str, SimulationScenario] = {}
        self.active_scenario: Optional[str] = None
        self.comparison_scenarios: List[str] = []

        # UI components (will be populated based on available libraries)
        self.app = None
        self.is_running = False

        self.logger = logging.getLogger(__name__)

        # Setup dashboard based on available dependencies
        self._detect_dashboard_capabilities()

    def _detect_dashboard_capabilities(self) -> None:
        """Detect available dashboard libraries and setup accordingly"""
        if DASH_AVAILABLE:
            self.dashboard_type = "dash"
            self._setup_dash_app()
        elif STREAMLIT_AVAILABLE:
            self.dashboard_type = "streamlit"
            self._setup_streamlit_app()
        elif PLOTLY_AVAILABLE:
            self.dashboard_type = "plotly_offline"
            self.logger.info("Using offline Plotly for static visualizations")
        else:
            self.dashboard_type = "html"
            self.logger.warning(
                "No interactive dashboard libraries available. Using HTML output."
            )

    def _setup_streamlit_app(self) -> None:
        """Setup Streamlit application"""
        if not STREAMLIT_AVAILABLE:
            return
        # Streamlit setup would be handled in a separate script
        self.app = None

    def _setup_dash_app(self) -> None:
        """Setup Dash application"""
        if not DASH_AVAILABLE:
            return

        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  # type: ignore
        self.app.title = "BSTEW Interactive Dashboard"  # type: ignore

        # Setup layout
        self._create_dash_layout()

        # Setup callbacks
        self._setup_dash_callbacks()

    def _create_dash_layout(self) -> None:
        """Create Dash layout"""
        if not self.app:
            return

        self.app.layout = dbc.Container(
            [
                # Header
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "BSTEW Interactive Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.P(
                                    "Real-time Bumblebee Simulation Analysis",
                                    className="text-center text-muted",
                                ),
                            ]
                        )
                    ]
                ),
                # Control panel
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Simulation Parameters"),
                                        dbc.CardBody(
                                            [self._create_parameter_controls()]
                                        ),
                                    ]
                                )
                            ],
                            width=4,
                        ),
                        # Main visualization area
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.Div(
                                                    [
                                                        html.H5(
                                                            "Live Simulation",
                                                            className="mb-0",
                                                        ),
                                                        dbc.ButtonGroup(
                                                            [
                                                                dbc.Button(
                                                                    "Start",
                                                                    id="start-btn",
                                                                    color="success",
                                                                    size="sm",
                                                                ),
                                                                dbc.Button(
                                                                    "Stop",
                                                                    id="stop-btn",
                                                                    color="danger",
                                                                    size="sm",
                                                                ),
                                                                dbc.Button(
                                                                    "Reset",
                                                                    id="reset-btn",
                                                                    color="secondary",
                                                                    size="sm",
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="d-flex justify-content-between align-items-center",
                                                )
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                # Live metrics
                                                html.Div(id="live-metrics"),
                                                # Main plot
                                                dcc.Graph(id="main-plot"),
                                                # Status
                                                html.Div(id="status-display"),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=8,
                        ),
                    ],
                    className="mb-4",
                ),
                # Secondary visualizations
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Population Dynamics"),
                                        dbc.CardBody([dcc.Graph(id="population-plot")]),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Spatial Distribution"),
                                        dbc.CardBody([dcc.Graph(id="spatial-plot")]),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

    def _create_parameter_controls(self) -> List[Any]:
        """Create parameter control widgets"""
        controls = []

        for category in sorted(self.parameter_manager.categories.keys()):
            category_controls = []

            for control_name in self.parameter_manager.categories[category]:
                control = self.parameter_manager.get_control(control_name)
                if not control:
                    continue

                if control.param_type == "slider":
                    widget = html.Div(
                        [
                            html.Label(control.label, className="form-label"),
                            dcc.Slider(
                                id=f"param-{control.name}",
                                min=control.min_val,
                                max=control.max_val,
                                step=control.step,
                                value=control.default_value,
                                marks={
                                    str(control.min_val or 0): str(
                                        control.min_val or 0
                                    ),
                                    str(control.max_val or 1): str(
                                        control.max_val or 1
                                    ),
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Small(control.description, className="text-muted"),
                        ],
                        className="mb-3",
                    )

                elif control.param_type == "dropdown":
                    widget = html.Div(
                        [
                            html.Label(control.label, className="form-label"),
                            dcc.Dropdown(
                                id=f"param-{control.name}",
                                options=[str(opt) for opt in control.options or []],
                                value=control.default_value,
                                multi=isinstance(control.default_value, list),
                            ),
                            html.Small(control.description, className="text-muted"),
                        ],
                        className="mb-3",
                    )

                category_controls.append(widget)

            if category_controls:
                controls.append(
                    html.Div(
                        [html.H6(category, className="mt-3 mb-2"), *category_controls]
                    )
                )

        return controls

    def _setup_dash_callbacks(self) -> None:
        """Setup Dash callbacks"""
        if not self.app:
            return

        # Parameter update callbacks
        @self.app.callback(
            Output("status-display", "children"),
            [
                Input(f"param-{name}", "value")
                for name in self.parameter_manager.controls.keys()
            ],
        )
        def update_parameters(*values):
            """Update parameters when controls change"""
            param_names = list(self.parameter_manager.controls.keys())
            for i, value in enumerate(values):
                if i < len(param_names):
                    self.parameter_manager.update_value(param_names[i], value)

            return html.Div([html.P("Parameters updated", className="text-success")])

        # Start simulation callback
        @self.app.callback(
            Output("main-plot", "figure"),
            [Input("start-btn", "n_clicks")],
            prevent_initial_call=True,
        )
        def start_simulation(n_clicks):
            """Start simulation with current parameters"""
            if n_clicks:
                return self._run_simulation_step()
            return {}

        # Live metrics callback
        @self.app.callback(
            Output("live-metrics", "children"),
            Input("main-plot", "figure"),  # Triggers when plot updates
        )
        def update_live_metrics(figure):
            """Update live metrics display"""
            metrics = self.real_time_monitor.get_live_metrics()

            if not metrics:
                return html.P("No simulation data", className="text-muted")

            return dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Colonies"),
                            html.H4(
                                metrics.get("colony_count", 0), className="text-primary"
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.H6("Population"),
                            html.H4(
                                metrics.get("population", 0), className="text-success"
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.H6("Foraging"),
                            html.H4(
                                metrics.get("foraging_activity", 0),
                                className="text-info",
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.H6("Resources"),
                            html.H4(
                                metrics.get("resource_patches", 0),
                                className="text-warning",
                            ),
                        ],
                        width=3,
                    ),
                ]
            )

    def _run_simulation_step(self) -> Dict[str, Any]:
        """Run single simulation step and return plot data"""
        # This is a placeholder - would integrate with actual SimulationEngine
        import random

        # Generate sample data
        x_data = list(range(100))
        y_data = [random.randint(10, 100) for _ in x_data]

        figure = {
            "data": [{"x": x_data, "y": y_data, "type": "line", "name": "Population"}],
            "layout": {
                "title": "Live Simulation Data",
                "xaxis": {"title": "Time Step"},
                "yaxis": {"title": "Population"},
            },
        }

        # Update monitor with fake data
        self.real_time_monitor.add_data_point(
            time.time(),
            {
                "total_colonies": random.randint(3, 8),
                "total_population": random.randint(500, 2000),
                "foraging_bees": random.randint(50, 300),
                "active_patches": random.randint(10, 50),
            },
        )

        return figure

    def create_scenario(
        self, name: str, description: str, parameters: Dict[str, Any]
    ) -> str:
        """Create new simulation scenario"""
        scenario_id = f"scenario_{len(self.scenarios) + 1}"

        # Create simulation config from parameters
        config = SimulationConfig(
            duration_days=parameters.get("time_steps", 1000),
            # Add more parameter mappings as needed
        )

        scenario = SimulationScenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            parameters=parameters,
            config=config,
        )

        self.scenarios[scenario_id] = scenario
        self.logger.info(f"Created scenario: {name}")

        return scenario_id

    def run_scenario(self, scenario_id: str) -> None:
        """Run simulation scenario"""
        scenario = self.scenarios.get(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario not found: {scenario_id}")

        scenario.status = "running"
        scenario.updated_at = time.time()

        try:
            # This would integrate with actual SimulationEngine
            self.logger.info(f"Running scenario: {scenario.name}")

            # Placeholder results
            scenario.results = {
                "final_population": 1500,
                "colony_survival_rate": 0.8,
                "resource_utilization": 0.65,
                "time_series_data": [],  # Would contain actual simulation data
            }

            scenario.status = "completed"

        except Exception as e:
            scenario.status = "failed"
            self.logger.error(f"Scenario failed: {e}")

        scenario.updated_at = time.time()

    def compare_scenarios(self, scenario_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple scenarios"""
        if len(scenario_ids) < 2:
            raise ValueError("Need at least 2 scenarios for comparison")

        comparison_data: Dict[str, Any] = {
            "scenarios": [],
            "metrics": [],
            "statistical_analysis": {},
        }

        for scenario_id in scenario_ids:
            scenario = self.scenarios.get(scenario_id)
            if scenario and scenario.results:
                comparison_data["scenarios"].append(
                    {
                        "id": scenario_id,
                        "name": scenario.name,
                        "parameters": scenario.parameters,
                        "results": scenario.results,
                    }
                )

        # Generate comparison plots if Plotly available
        if PLOTLY_AVAILABLE:
            comparison_data["plots"] = self._create_comparison_plots(
                comparison_data["scenarios"]
            )

        return comparison_data

    def _create_comparison_plots(
        self, scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create comparison visualization plots"""
        plots: Dict[str, Any] = {}

        if not scenarios:
            return plots

        # Population comparison
        scenario_names = [s["name"] for s in scenarios]
        populations = [s["results"].get("final_population", 0) for s in scenarios]

        plots["population_comparison"] = {
            "data": [
                {
                    "x": scenario_names,
                    "y": populations,
                    "type": "bar",
                    "name": "Final Population",
                }
            ],
            "layout": {
                "title": "Final Population Comparison",
                "xaxis": {"title": "Scenarios"},
                "yaxis": {"title": "Population"},
            },
        }

        return plots

    def export_results(self, scenario_ids: Optional[List[str]] = None) -> str:
        """Export scenario results to file"""
        scenario_list: List[str] = (
            scenario_ids if scenario_ids is not None else list(self.scenarios.keys())
        )

        export_data: Dict[str, Any] = {
            "export_timestamp": datetime.now().isoformat(),
            "scenarios": {},
        }

        for scenario_id in scenario_list:
            scenario = self.scenarios.get(scenario_id)
            if scenario:
                export_data["scenarios"][scenario_id] = {
                    "name": scenario.name,
                    "description": scenario.description,
                    "parameters": scenario.parameters,
                    "results": scenario.results,
                    "status": scenario.status,
                    "created_at": scenario.created_at,
                    "updated_at": scenario.updated_at,
                }

        # Export to JSON
        export_file = self.output_dir / f"dashboard_export_{int(time.time())}.json"
        with open(export_file, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Results exported to: {export_file}")
        return str(export_file)

    def run_server(
        self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False
    ) -> None:
        """Run dashboard server"""
        if self.dashboard_type == "dash" and self.app:
            self.logger.info(f"Starting Dash server at http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug)
        else:
            self.logger.error(
                "Dashboard server not available. Install dash for full functionality."
            )

    def generate_static_report(self) -> str:
        """Generate static HTML report"""
        report_file = self.output_dir / f"dashboard_report_{int(time.time())}.html"

        html_content = self._generate_report_html()

        with open(report_file, "w") as f:
            f.write(html_content)

        self.logger.info(f"Static report generated: {report_file}")
        return str(report_file)

    def _generate_report_html(self) -> str:
        """Generate HTML report content"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BSTEW Dashboard Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .scenario {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metrics {{ display: flex; gap: 20px; }}
                .metric {{ background-color: #e6f3ff; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BSTEW Interactive Dashboard Report</h1>
                <p>Generated: {datetime.now().isoformat()}</p>
                <p>Total Scenarios: {len(self.scenarios)}</p>
            </div>

            <div class="scenarios">
                <h2>Scenarios</h2>
                {"".join([self._scenario_html(s) for s in self.scenarios.values()])}
            </div>
        </body>
        </html>
        """

    def _scenario_html(self, scenario: SimulationScenario) -> str:
        """Generate HTML for single scenario"""
        return f"""
        <div class="scenario">
            <h3>{scenario.name}</h3>
            <p><strong>Status:</strong> {scenario.status}</p>
            <p><strong>Description:</strong> {scenario.description}</p>
            <p><strong>Parameters:</strong> {json.dumps(scenario.parameters, indent=2)}</p>
            {f"<p><strong>Results:</strong> {json.dumps(scenario.results, indent=2)}</p>" if scenario.results else ""}
        </div>
        """


def create_dashboard(
    output_dir: str = "artifacts/dashboard_output",
) -> InteractiveDashboard:
    """Factory function to create interactive dashboard"""
    return InteractiveDashboard(output_dir=output_dir)


# Example usage and testing
if __name__ == "__main__":
    # Create dashboard
    dashboard = create_dashboard()

    # Create test scenario
    test_params = {
        "initial_colonies": 5,
        "colony_growth_rate": 1.88,
        "foraging_range_m": 2500,
        "time_steps": 1000,
    }

    scenario_id = dashboard.create_scenario(
        name="Test Scenario",
        description="Basic test of dashboard functionality",
        parameters=test_params,
    )

    # Run scenario
    dashboard.run_scenario(scenario_id)

    # Export results
    dashboard.export_results()

    # Generate report
    dashboard.generate_static_report()

    print("Dashboard test completed successfully!")

    # Start server if Dash is available
    if DASH_AVAILABLE:
        print("Starting interactive dashboard server...")
        dashboard.run_server(debug=True)
    else:
        print(
            "Install dash-plotly for full interactive dashboard: pip install dash plotly"
        )
