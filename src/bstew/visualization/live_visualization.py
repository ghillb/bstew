"""
Live Visualization System for NetLogo BEE-STEWARD v2 Parity
=========================================================

Real-time visualization system that connects to live simulation data,
providing dynamic charts, maps, and analytics dashboards for bee colonies.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from collections import deque, defaultdict
import json
from pathlib import Path

# Visualization dependencies (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle, Polygon
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    animation = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

    # Type stubs for missing classes
    class Circle:  # type: ignore
        pass

    class Polygon:  # type: ignore
        pass


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


class VisualizationType(Enum):
    """Types of visualizations available"""

    COLONY_OVERVIEW = "colony_overview"
    POPULATION_TRENDS = "population_trends"
    FORAGING_ACTIVITY = "foraging_activity"
    SPATIAL_DISTRIBUTION = "spatial_distribution"
    COMMUNICATION_NETWORK = "communication_network"
    HEALTH_MONITORING = "health_monitoring"
    RESOURCE_DYNAMICS = "resource_dynamics"
    DANCE_COMMUNICATION = "dance_communication"
    RECRUITMENT_FLOW = "recruitment_flow"
    ENVIRONMENTAL_CONDITIONS = "environmental_conditions"


@dataclass
class VisualizationData:
    """Container for visualization data"""

    timestamp: float
    data_type: VisualizationType
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "data_type": self.data_type.value,
                "data": self.data,
                "metadata": self.metadata,
            }
        )


@dataclass
class LiveDataStream:
    """Live data stream configuration"""

    stream_id: str
    data_source: str
    update_interval: float  # seconds
    buffer_size: int = 1000
    data_buffer: deque = field(default_factory=deque)
    is_active: bool = True
    last_update: float = 0.0

    def __post_init__(self) -> None:
        """Initialize buffer with correct size"""
        if not self.data_buffer:
            self.data_buffer = deque(maxlen=self.buffer_size)

    def add_data(self, data: VisualizationData) -> None:
        """Add data to stream buffer"""
        self.data_buffer.append(data)
        self.last_update = time.time()


class VisualizationEngine(BaseModel):
    """Core visualization engine for live data"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    # Data streams
    data_streams: Dict[str, LiveDataStream] = Field(default_factory=dict)

    # Visualization configurations
    chart_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Update settings
    refresh_rate: float = 1.0  # seconds
    auto_save_interval: float = 300.0  # 5 minutes

    # Output settings
    output_directory: str = "visualization_output"
    save_format: str = "html"  # html, png, svg

    # Performance settings
    max_data_points: int = 1000
    data_aggregation_enabled: bool = True

    # Logging
    logger: Any = Field(
        default_factory=lambda: logging.getLogger(__name__), exclude=True
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.logger = logging.getLogger(__name__)
        self._setup_output_directory()
        self._is_running = False
        self._update_thread: Optional[threading.Thread] = None

    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist"""
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

    def register_data_stream(
        self, stream_id: str, data_source: str, update_interval: float = 1.0
    ) -> None:
        """Register a new data stream"""

        stream = LiveDataStream(
            stream_id=stream_id,
            data_source=data_source,
            update_interval=update_interval,
        )

        self.data_streams[stream_id] = stream
        self.logger.info(f"Registered data stream: {stream_id}")

    def add_data(self, stream_id: str, data: VisualizationData) -> None:
        """Add data to a specific stream"""

        if stream_id in self.data_streams:
            self.data_streams[stream_id].add_data(data)
        else:
            self.logger.warning(f"Stream {stream_id} not found")

    def create_colony_overview_chart(
        self, colony_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Create colony overview visualization"""

        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for colony overview chart")
            return None

        # Create subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Population", "Resources", "Health", "Activity"),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
            ],
        )

        # Population chart
        if "population_data" in colony_data:
            pop_data = colony_data["population_data"]
            fig.add_trace(
                go.Scatter(
                    x=pop_data.get("timestamps", []),
                    y=pop_data.get("total_bees", []),
                    name="Total Bees",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=pop_data.get("timestamps", []),
                    y=pop_data.get("foragers", []),
                    name="Foragers",
                    line=dict(color="green"),
                ),
                row=1,
                col=1,
            )

        # Resource chart
        if "resource_data" in colony_data:
            res_data = colony_data["resource_data"]
            fig.add_trace(
                go.Scatter(
                    x=res_data.get("timestamps", []),
                    y=res_data.get("honey_stores", []),
                    name="Honey Stores",
                    line=dict(color="orange"),
                ),
                row=1,
                col=2,
            )

        # Health chart
        if "health_data" in colony_data:
            health_data = colony_data["health_data"]
            fig.add_trace(
                go.Scatter(
                    x=health_data.get("timestamps", []),
                    y=health_data.get("health_score", []),
                    name="Health Score",
                    line=dict(color="red"),
                ),
                row=2,
                col=1,
            )

        # Activity chart
        if "activity_data" in colony_data:
            act_data = colony_data["activity_data"]
            fig.add_trace(
                go.Bar(
                    x=list(act_data.get("activities", {}).keys()),
                    y=list(act_data.get("activities", {}).values()),
                    name="Activity Distribution",
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title_text="Colony Overview Dashboard", showlegend=True, height=800
        )

        return fig

    def create_spatial_visualization(
        self, spatial_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Create spatial distribution visualization"""

        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for spatial visualization")
            return None

        fig = go.Figure()

        # Add patches
        if "patches" in spatial_data:
            patches = spatial_data["patches"]

            for patch in patches:
                # Add patch as scatter point
                fig.add_trace(
                    go.Scatter(
                        x=[patch.get("x", 0)],
                        y=[patch.get("y", 0)],
                        mode="markers",
                        marker=dict(
                            size=max(10, patch.get("quality", 0.5) * 20),
                            color=patch.get("quality", 0.5),
                            colorscale="Viridis",
                            showscale=True,
                        ),
                        name=f"Patch {patch.get('id', 0)}",
                        text=f"Quality: {patch.get('quality', 0.5):.2f}",
                        hoverinfo="text",
                    )
                )

        # Add bee positions
        if "bees" in spatial_data:
            bees = spatial_data["bees"]

            bee_x = [bee.get("x", 0) for bee in bees]
            bee_y = [bee.get("y", 0) for bee in bees]
            bee_status = [bee.get("status", "IDLE") for bee in bees]

            fig.add_trace(
                go.Scatter(
                    x=bee_x,
                    y=bee_y,
                    mode="markers",
                    marker=dict(size=5, color="red", symbol="circle"),
                    name="Bees",
                    text=bee_status,
                    hoverinfo="text",
                )
            )

        # Add colony locations
        if "colonies" in spatial_data:
            colonies = spatial_data["colonies"]

            for colony in colonies:
                fig.add_trace(
                    go.Scatter(
                        x=[colony.get("x", 0)],
                        y=[colony.get("y", 0)],
                        mode="markers",
                        marker=dict(size=15, color="black", symbol="square"),
                        name=f"Colony {colony.get('id', 0)}",
                        text=f"Population: {colony.get('population', 0)}",
                        hoverinfo="text",
                    )
                )

        fig.update_layout(
            title="Spatial Distribution",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            showlegend=True,
        )

        return fig

    def create_communication_network(
        self, communication_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Create communication network visualization"""

        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for communication network")
            return None

        fig = go.Figure()

        # Add dance communications
        if "dances" in communication_data:
            dances = communication_data["dances"]

            for dance in dances:
                # Add dance performance as node
                fig.add_trace(
                    go.Scatter(
                        x=[dance.get("x", 0)],
                        y=[dance.get("y", 0)],
                        mode="markers",
                        marker=dict(
                            size=dance.get("intensity", 0.5) * 20,
                            color="gold",
                            symbol="star",
                        ),
                        name=f"Dance {dance.get('id', 0)}",
                        text=f"Intensity: {dance.get('intensity', 0.5):.2f}<br>Recruits: {dance.get('recruits', 0)}",
                        hoverinfo="text",
                    )
                )

                # Add recruitment flows
                if "followers" in dance:
                    for follower in dance["followers"]:
                        fig.add_trace(
                            go.Scatter(
                                x=[dance.get("x", 0), follower.get("x", 0)],
                                y=[dance.get("y", 0), follower.get("y", 0)],
                                mode="lines",
                                line=dict(color="rgba(255, 165, 0, 0.5)", width=2),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

        # Add information flows
        if "information_flows" in communication_data:
            flows = communication_data["information_flows"]

            for flow in flows:
                fig.add_trace(
                    go.Scatter(
                        x=[flow.get("source_x", 0), flow.get("target_x", 0)],
                        y=[flow.get("source_y", 0), flow.get("target_y", 0)],
                        mode="lines",
                        line=dict(
                            color=f"rgba(0, 0, 255, {flow.get('accuracy', 0.5)})",
                            width=1,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        fig.update_layout(
            title="Communication Network",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            showlegend=True,
        )

        return fig

    def create_time_series_chart(
        self, time_series_data: Dict[str, Any], chart_type: VisualizationType
    ) -> Optional[Any]:
        """Create time series visualization"""

        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for time series chart")
            return None

        fig = go.Figure()

        # Add time series traces
        for series_name, series_data in time_series_data.items():
            if (
                isinstance(series_data, dict)
                and "x" in series_data
                and "y" in series_data
            ):
                fig.add_trace(
                    go.Scatter(
                        x=series_data["x"],
                        y=series_data["y"],
                        mode="lines+markers",
                        name=series_name,
                        line=dict(width=2),
                    )
                )

        fig.update_layout(
            title=f"{chart_type.value.replace('_', ' ').title()} - Time Series",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True,
        )

        return fig

    def generate_dashboard(self, data_snapshot: Dict[str, Any]) -> Optional[str]:
        """Generate complete dashboard with multiple visualizations"""

        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for dashboard generation")
            return None

        # Create individual charts
        charts = []

        # Colony overview
        if "colony_data" in data_snapshot:
            colony_chart = self.create_colony_overview_chart(
                data_snapshot["colony_data"]
            )
            if colony_chart:
                charts.append(("Colony Overview", colony_chart))

        # Spatial visualization
        if "spatial_data" in data_snapshot:
            spatial_chart = self.create_spatial_visualization(
                data_snapshot["spatial_data"]
            )
            if spatial_chart:
                charts.append(("Spatial Distribution", spatial_chart))

        # Communication network
        if "communication_data" in data_snapshot:
            comm_chart = self.create_communication_network(
                data_snapshot["communication_data"]
            )
            if comm_chart:
                charts.append(("Communication Network", comm_chart))

        # Time series charts
        for chart_type in [
            VisualizationType.POPULATION_TRENDS,
            VisualizationType.FORAGING_ACTIVITY,
            VisualizationType.HEALTH_MONITORING,
        ]:
            if chart_type.value in data_snapshot:
                ts_chart = self.create_time_series_chart(
                    data_snapshot[chart_type.value], chart_type
                )
                if ts_chart:
                    charts.append(
                        (chart_type.value.replace("_", " ").title(), ts_chart)
                    )

        # Combine charts into dashboard
        if charts:
            dashboard_html = self._create_dashboard_html(charts)

            # Save dashboard
            timestamp = int(time.time())
            output_file = f"{self.output_directory}/dashboard_{timestamp}.html"

            with open(output_file, "w") as f:
                f.write(dashboard_html)

            self.logger.info(f"Dashboard saved to {output_file}")
            return output_file

        return None

    def _create_dashboard_html(self, charts: List[Tuple[str, Any]]) -> str:
        """Create HTML dashboard from charts"""

        html_parts = [
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>BEE-STEWARD Live Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .chart-container { margin-bottom: 30px; }
                    .dashboard-title { text-align: center; color: #333; }
                    .timestamp { text-align: center; color: #666; font-size: 14px; }
                </style>
            </head>
            <body>
                <h1 class="dashboard-title">BEE-STEWARD Live Dashboard</h1>
                <p class="timestamp">Generated: """
            + time.strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
            """
        ]

        # Add each chart
        for i, (title, chart) in enumerate(charts):
            chart_div = f"chart_{i}"
            chart_json = chart.to_json()

            html_parts.append(f"""
                <div class="chart-container">
                    <h2>{title}</h2>
                    <div id="{chart_div}"></div>
                    <script>
                        Plotly.newPlot('{chart_div}', {chart_json});
                    </script>
                </div>
            """)

        html_parts.append("""
            </body>
            </html>
        """)

        return "\n".join(html_parts)

    def start_live_updates(self) -> None:
        """Start live data updates"""

        if self._is_running:
            self.logger.warning("Live updates already running")
            return

        self._is_running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        if self._update_thread is not None:
            self._update_thread.start()

        self.logger.info("Started live visualization updates")

    def stop_live_updates(self) -> None:
        """Stop live data updates"""

        self._is_running = False

        if self._update_thread:
            self._update_thread.join(timeout=5.0)

        self.logger.info("Stopped live visualization updates")

    def _update_loop(self) -> None:
        """Main update loop for live visualizations"""

        while self._is_running:
            try:
                # Process data streams
                for stream_id, stream in self.data_streams.items():
                    if stream.is_active and stream.data_buffer:
                        # Process latest data
                        latest_data = list(stream.data_buffer)[-self.max_data_points :]

                        # Create visualization if enough data
                        if len(latest_data) > 10:
                            self._create_live_visualization(stream_id, latest_data)

                # Wait for next update
                time.sleep(self.refresh_rate)

            except Exception as e:
                self.logger.error(f"Error in visualization update loop: {e}")
                time.sleep(self.refresh_rate)

    def _create_live_visualization(
        self, stream_id: str, data: List[VisualizationData]
    ) -> None:
        """Create live visualization for a data stream"""

        # Extract time series data
        timestamps = [d.timestamp for d in data]

        # Group data by type
        grouped_data = defaultdict(list)
        for d in data:
            for key, value in d.data.items():
                if isinstance(value, (int, float)):
                    grouped_data[key].append(value)

        # Create time series chart
        time_series_data = {}
        for key, values in grouped_data.items():
            if len(values) == len(timestamps):
                time_series_data[key] = {"x": timestamps, "y": values}

        if time_series_data:
            chart = self.create_time_series_chart(
                time_series_data, VisualizationType(stream_id)
            )

            if chart:
                # Save chart
                output_file = (
                    f"{self.output_directory}/live_{stream_id}_{int(time.time())}.html"
                )
                chart.write_html(output_file)

    def export_data_snapshot(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Export current data snapshot"""

        snapshot: Dict[str, Any] = {"timestamp": time.time(), "streams": {}}

        for stream_id, stream in self.data_streams.items():
            stream_data: Dict[str, Any] = {
                "data_points": len(stream.data_buffer),
                "last_update": stream.last_update,
                "is_active": stream.is_active,
            }

            if include_metadata:
                stream_data["recent_data"] = [
                    {"timestamp": d.timestamp, "data_type": d.data_type, "data": d.data}
                    for d in list(stream.data_buffer)[-10:]  # Last 10 data points
                ]

            snapshot["streams"][stream_id] = stream_data

        return snapshot


class LiveVisualizationManager(BaseModel):
    """Manager for live visualization system"""

    model_config = {"validate_assignment": True}

    # Visualization engine
    engine: VisualizationEngine

    # Data collectors
    data_collectors: Dict[str, Callable] = Field(default_factory=dict)

    # Update settings
    collection_interval: float = 5.0

    def __init__(self, **data: Any) -> None:
        if "engine" not in data:
            data["engine"] = VisualizationEngine()

        super().__init__(**data)
        self.logger = logging.getLogger(__name__)

    def register_model_data_collector(
        self, model: Any, stream_prefix: str = "model"
    ) -> None:
        """Register data collector for simulation model"""

        # Colony data stream
        def collect_colony_data() -> None:
            for colony in getattr(model, "get_colonies", lambda: [])():
                colony_id = getattr(colony, "unique_id", 0)

                data = VisualizationData(
                    timestamp=time.time(),
                    data_type=VisualizationType.COLONY_OVERVIEW,
                    data={
                        "colony_id": colony_id,
                        "population": getattr(
                            colony, "get_adult_population", lambda: 0
                        )(),
                        "health_score": getattr(
                            colony, "get_health_score", lambda: 0.5
                        )(),
                        "honey_stores": getattr(
                            colony, "get_honey_stores", lambda: 0.0
                        )(),
                        "forager_count": getattr(
                            colony, "get_forager_count", lambda: 0
                        )(),
                    },
                )

                self.engine.add_data(f"{stream_prefix}_colony_{colony_id}", data)

        # Spatial data stream
        def collect_spatial_data() -> None:
            if hasattr(model, "spatial_environment") and model.spatial_environment:
                spatial_data = model.spatial_environment.get_gis_analysis_results()

                data = VisualizationData(
                    timestamp=time.time(),
                    data_type=VisualizationType.SPATIAL_DISTRIBUTION,
                    data=spatial_data,
                )

                self.engine.add_data(f"{stream_prefix}_spatial", data)

        # Communication data stream
        def collect_communication_data() -> None:
            if (
                hasattr(model, "dance_communication_integrator")
                and model.dance_communication_integrator
            ):
                for colony in getattr(model, "get_colonies", lambda: [])():
                    colony_id = getattr(colony, "unique_id", 0)
                    comm_metrics = model.dance_communication_integrator.get_colony_communication_metrics(
                        colony_id
                    )

                    data = VisualizationData(
                        timestamp=time.time(),
                        data_type=VisualizationType.COMMUNICATION_NETWORK,
                        data=comm_metrics,
                    )

                    self.engine.add_data(
                        f"{stream_prefix}_communication_{colony_id}", data
                    )

        # Register collectors
        self.data_collectors["colony"] = collect_colony_data
        self.data_collectors["spatial"] = collect_spatial_data
        self.data_collectors["communication"] = collect_communication_data

        # Register streams
        self.engine.register_data_stream(
            f"{stream_prefix}_colonies", "model", self.collection_interval
        )
        self.engine.register_data_stream(
            f"{stream_prefix}_spatial", "model", self.collection_interval
        )
        self.engine.register_data_stream(
            f"{stream_prefix}_communication", "model", self.collection_interval
        )

        self.logger.info(
            f"Registered model data collectors with prefix: {stream_prefix}"
        )

    def start_data_collection(self) -> None:
        """Start automated data collection"""

        def collection_loop() -> None:
            while self.engine._is_running:
                try:
                    # Run all data collectors
                    for collector_name, collector_func in self.data_collectors.items():
                        collector_func()

                    time.sleep(self.collection_interval)

                except Exception as e:
                    self.logger.error(f"Error in data collection: {e}")
                    time.sleep(self.collection_interval)

        # Start collection thread
        collection_thread = threading.Thread(target=collection_loop, daemon=True)
        collection_thread.start()

        # Start visualization updates
        self.engine.start_live_updates()

        self.logger.info("Started live data collection and visualization")

    def stop_data_collection(self) -> None:
        """Stop automated data collection"""

        self.engine.stop_live_updates()
        self.logger.info("Stopped live data collection and visualization")


def create_live_visualization_system(
    output_dir: str = "visualization_output",
) -> LiveVisualizationManager:
    """Factory function to create live visualization system"""

    engine = VisualizationEngine(output_directory=output_dir)
    manager = LiveVisualizationManager(engine=engine)

    return manager
