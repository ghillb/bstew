"""
Interactive Analysis Tools for BSTEW
===================================

Interactive data exploration capabilities with real-time analysis,
dynamic filtering, and interactive parameter adjustment.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import logging
from datetime import datetime
import json
import threading
import time
from dataclasses import dataclass
from enum import Enum

from .display_system import DisplayStateManager


class InteractiveMode(Enum):
    """Interactive analysis modes"""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"


@dataclass
class InteractiveSession:
    """Interactive analysis session data"""
    session_id: str
    start_time: datetime
    mode: InteractiveMode
    data_sources: List[str]
    active_filters: Dict[str, Any]
    current_view: str
    user_preferences: Dict[str, Any]


class DataExplorationEngine:
    """Engine for interactive data exploration"""
    
    def __init__(self, display_manager: DisplayStateManager):
        self.display_manager = display_manager
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.current_data = {}
        self.historical_data = []
        self.filtered_data = {}
        
        # Analysis state
        self.active_filters = {}
        self.selected_metrics = []
        self.current_view = "overview"
        
        # Interactive callbacks
        self.update_callbacks = []
        self.filter_callbacks = []
        
    def load_data(self, data_sources: Dict[str, Any]) -> None:
        """Load data from multiple sources"""
        
        self.current_data = {}
        
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, list):
                # Convert to DataFrame for easier manipulation
                if source_data and hasattr(source_data[0], '__dict__'):
                    # Handle objects with attributes
                    records = []
                    for item in source_data:
                        record = {}
                        for attr in dir(item):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(item, attr)
                                    if not callable(value):
                                        record[attr] = value
                                except Exception:
                                    pass
                        records.append(record)
                    self.current_data[source_name] = pd.DataFrame(records)
                else:
                    self.current_data[source_name] = pd.DataFrame(source_data)
            elif isinstance(source_data, pd.DataFrame):
                self.current_data[source_name] = source_data
            else:
                self.logger.warning(f"Unsupported data type for {source_name}: {type(source_data)}")
                
        self.logger.info(f"Loaded data from {len(self.current_data)} sources")
        
    def apply_filters(self, filters: Dict[str, Any]) -> None:
        """Apply filters to current data"""
        
        self.active_filters = filters
        self.filtered_data = {}
        
        for source_name, df in self.current_data.items():
            filtered_df = df.copy()
            
            for filter_name, filter_value in filters.items():
                if filter_name in filtered_df.columns:
                    if isinstance(filter_value, dict):
                        # Range filter
                        if 'min' in filter_value:
                            filtered_df = filtered_df[filtered_df[filter_name] >= filter_value['min']]
                        if 'max' in filter_value:
                            filtered_df = filtered_df[filtered_df[filter_name] <= filter_value['max']]
                    elif isinstance(filter_value, (list, tuple)):
                        # Multiple values filter
                        filtered_df = filtered_df[filtered_df[filter_name].isin(filter_value)]
                    else:
                        # Exact match filter
                        filtered_df = filtered_df[filtered_df[filter_name] == filter_value]
                        
            self.filtered_data[source_name] = filtered_df
            
        # Trigger filter callbacks
        for callback in self.filter_callbacks:
            callback(self.filtered_data)
            
    def get_summary_statistics(self, source_name: str) -> Dict[str, Any]:
        """Get summary statistics for a data source"""
        
        if source_name not in self.current_data:
            return {}
            
        df = self.filtered_data.get(source_name, self.current_data[source_name])
        
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict()
        }
        
        # Add statistics for numeric columns
        numeric_stats = {}
        for col in summary['numeric_columns']:
            numeric_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        summary['numeric_statistics'] = numeric_stats
        
        # Add value counts for categorical columns
        categorical_stats = {}
        for col in summary['categorical_columns']:
            if df[col].nunique() < 50:  # Only for columns with reasonable number of unique values
                categorical_stats[col] = df[col].value_counts().to_dict()
        summary['categorical_statistics'] = categorical_stats
        
        return summary
        
    def create_interactive_plot(self, 
                               source_name: str, 
                               plot_type: str,
                               x_column: str,
                               y_column: Optional[str] = None,
                               color_column: Optional[str] = None,
                               size_column: Optional[str] = None) -> go.Figure:
        """Create an interactive plot"""
        
        if source_name not in self.current_data:
            return go.Figure()
            
        df = self.filtered_data.get(source_name, self.current_data[source_name])
        
        if plot_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_column, 
                           color=color_column, size=size_column,
                           hover_data=df.columns.tolist())
        elif plot_type == "line":
            fig = px.line(df, x=x_column, y=y_column, 
                         color=color_column)
        elif plot_type == "bar":
            fig = px.bar(df, x=x_column, y=y_column, 
                        color=color_column)
        elif plot_type == "histogram":
            fig = px.histogram(df, x=x_column, 
                             color=color_column)
        elif plot_type == "box":
            fig = px.box(df, x=x_column, y=y_column, 
                        color=color_column)
        elif plot_type == "violin":
            fig = px.violin(df, x=x_column, y=y_column, 
                           color=color_column)
        elif plot_type == "heatmap":
            # For heatmap, we need to prepare the data differently
            if y_column and x_column != y_column:
                pivot_df = df.pivot_table(values=y_column, 
                                        index=x_column, 
                                        columns=color_column if color_column else None,
                                        aggfunc='mean')
                fig = px.imshow(pivot_df, aspect="auto")
            else:
                # Correlation heatmap
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  x=corr_matrix.columns, 
                                  y=corr_matrix.columns,
                                  color_continuous_scale='RdBu_r',
                                  aspect="auto")
                else:
                    fig = go.Figure()
        else:
            fig = go.Figure()
            
        # Customize layout
        fig.update_layout(
            title=f"{plot_type.title()} Plot - {source_name}",
            template="plotly_white",
            height=500
        )
        
        return fig
        
    def create_time_series_plot(self, 
                               source_name: str,
                               time_column: str,
                               value_columns: List[str],
                               group_column: Optional[str] = None) -> go.Figure:
        """Create interactive time series plot"""
        
        if source_name not in self.current_data:
            return go.Figure()
            
        df = self.filtered_data.get(source_name, self.current_data[source_name])
        
        fig = go.Figure()
        
        if group_column and group_column in df.columns:
            # Multiple lines for different groups
            for group_value in df[group_column].unique():
                group_df = df[df[group_column] == group_value]
                
                for value_col in value_columns:
                    if value_col in group_df.columns:
                        fig.add_trace(go.Scatter(
                            x=group_df[time_column],
                            y=group_df[value_col],
                            mode='lines+markers',
                            name=f"{group_value} - {value_col}",
                            line=dict(width=2),
                            marker=dict(size=4)
                        ))
        else:
            # Single line for each value column
            for value_col in value_columns:
                if value_col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[time_column],
                        y=df[value_col],
                        mode='lines+markers',
                        name=value_col,
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
                    
        fig.update_layout(
            title=f"Time Series - {source_name}",
            xaxis_title=time_column,
            yaxis_title="Value",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        return fig
        
    def create_comparative_analysis(self, 
                                   source_names: List[str],
                                   metric_column: str,
                                   group_column: str) -> go.Figure:
        """Create comparative analysis plot"""
        
        all_data = []
        
        for source_name in source_names:
            if source_name in self.current_data:
                df = self.filtered_data.get(source_name, self.current_data[source_name])
                if metric_column in df.columns and group_column in df.columns:
                    df_subset = df[[metric_column, group_column]].copy()
                    df_subset['source'] = source_name
                    all_data.append(df_subset)
                    
        if not all_data:
            return go.Figure()
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create box plot for comparison
        fig = px.box(combined_df, x='source', y=metric_column, 
                    color=group_column,
                    title=f"Comparative Analysis - {metric_column}")
        
        fig.update_layout(
            template="plotly_white",
            height=500
        )
        
        return fig
        
    def create_correlation_matrix(self, 
                                 source_name: str,
                                 columns: Optional[List[str]] = None) -> go.Figure:
        """Create interactive correlation matrix"""
        
        if source_name not in self.current_data:
            return go.Figure()
            
        df = self.filtered_data.get(source_name, self.current_data[source_name])
        
        # Select numeric columns
        if columns:
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
        if len(numeric_cols) < 2:
            return go.Figure()
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Correlation Matrix - {source_name}",
            template="plotly_white",
            height=max(400, len(numeric_cols) * 30)
        )
        
        return fig
        
    def export_analysis_results(self, output_directory: str) -> None:
        """Export analysis results"""
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export summary statistics
        summary_data = {}
        for source_name in self.current_data:
            summary_data[source_name] = self.get_summary_statistics(source_name)
            
        with open(output_path / "summary_statistics.json", 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
            
        # Export filtered data
        for source_name, df in self.filtered_data.items():
            df.to_csv(output_path / f"{source_name}_filtered.csv", index=False)
            
        # Export analysis configuration
        config_data = {
            'active_filters': self.active_filters,
            'selected_metrics': self.selected_metrics,
            'current_view': self.current_view,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / "analysis_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
            
        self.logger.info(f"Analysis results exported to {output_path}")


class InteractiveDashboard:
    """Interactive dashboard for BSTEW analysis"""
    
    def __init__(self, data_engine: DataExplorationEngine, port: int = 8050):
        self.data_engine = data_engine
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, 
                           external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "BSTEW Interactive Analysis"
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
    def _setup_layout(self) -> None:
        """Setup dashboard layout"""
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("BSTEW Interactive Analysis", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Data Source"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="data-source-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select data source..."
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Plot Type"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="plot-type-dropdown",
                                options=[
                                    {"label": "Scatter", "value": "scatter"},
                                    {"label": "Line", "value": "line"},
                                    {"label": "Bar", "value": "bar"},
                                    {"label": "Histogram", "value": "histogram"},
                                    {"label": "Box Plot", "value": "box"},
                                    {"label": "Violin Plot", "value": "violin"},
                                    {"label": "Heatmap", "value": "heatmap"}
                                ],
                                value="scatter"
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("X-Axis"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="x-axis-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select X column..."
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Y-Axis"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="y-axis-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select Y column..."
                            )
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Color"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="color-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select color column..."
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Size"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="size-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select size column..."
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filters"),
                        dbc.CardBody([
                            dbc.Button("Configure Filters", id="filter-button", 
                                     color="primary", className="mb-2"),
                            html.Div(id="filter-status")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Actions"),
                        dbc.CardBody([
                            dbc.Button("Export Data", id="export-button", 
                                     color="success", className="mb-2"),
                            dbc.Button("Reset View", id="reset-button", 
                                     color="secondary")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Main Plot"),
                        dbc.CardBody([
                            dcc.Graph(id="main-plot", 
                                    style={"height": "500px"})
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Summary Statistics"),
                        dbc.CardBody([
                            html.Div(id="summary-stats")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Time Series Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="time-series-plot", 
                                    style={"height": "400px"})
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Correlation Matrix"),
                        dbc.CardBody([
                            dcc.Graph(id="correlation-plot", 
                                    style={"height": "400px"})
                        ])
                    ])
                ], width=6)
            ]),
            
            # Modal for filters
            dbc.Modal([
                dbc.ModalHeader("Configure Filters"),
                dbc.ModalBody([
                    html.Div(id="filter-controls")
                ]),
                dbc.ModalFooter([
                    dbc.Button("Apply Filters", id="apply-filters-button", 
                             color="primary"),
                    dbc.Button("Clear Filters", id="clear-filters-button", 
                             color="secondary"),
                    dbc.Button("Close", id="close-modal-button", 
                             color="dark")
                ])
            ], id="filter-modal", size="lg"),
            
            # Store components for data
            dcc.Store(id="current-filters"),
            dcc.Store(id="data-summary")
            
        ], fluid=True)
        
    def _setup_callbacks(self) -> None:
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output("data-source-dropdown", "options"),
             Output("data-source-dropdown", "value")],
            [Input("data-source-dropdown", "id")]
        )
        def update_data_sources(_):
            options = [{"label": name, "value": name} 
                      for name in self.data_engine.current_data.keys()]
            value = list(self.data_engine.current_data.keys())[0] if options else None
            return options, value
            
        @self.app.callback(
            [Output("x-axis-dropdown", "options"),
             Output("y-axis-dropdown", "options"),
             Output("color-dropdown", "options"),
             Output("size-dropdown", "options")],
            [Input("data-source-dropdown", "value")]
        )
        def update_column_options(selected_source):
            if not selected_source or selected_source not in self.data_engine.current_data:
                return [], [], [], []
                
            df = self.data_engine.current_data[selected_source]
            options = [{"label": col, "value": col} for col in df.columns]
            
            return options, options, options, options
            
        @self.app.callback(
            Output("main-plot", "figure"),
            [Input("data-source-dropdown", "value"),
             Input("plot-type-dropdown", "value"),
             Input("x-axis-dropdown", "value"),
             Input("y-axis-dropdown", "value"),
             Input("color-dropdown", "value"),
             Input("size-dropdown", "value")]
        )
        def update_main_plot(source, plot_type, x_col, y_col, color_col, size_col):
            if not source or not plot_type or not x_col:
                return go.Figure()
                
            return self.data_engine.create_interactive_plot(
                source, plot_type, x_col, y_col, color_col, size_col
            )
            
        @self.app.callback(
            Output("summary-stats", "children"),
            [Input("data-source-dropdown", "value")]
        )
        def update_summary_stats(selected_source):
            if not selected_source:
                return "No data source selected"
                
            stats = self.data_engine.get_summary_statistics(selected_source)
            
            return [
                html.P(f"Total Records: {stats.get('total_records', 0)}"),
                html.P(f"Columns: {len(stats.get('columns', []))}"),
                html.P(f"Numeric Columns: {len(stats.get('numeric_columns', []))}"),
                html.P(f"Categorical Columns: {len(stats.get('categorical_columns', []))}")
            ]
            
        @self.app.callback(
            Output("correlation-plot", "figure"),
            [Input("data-source-dropdown", "value")]
        )
        def update_correlation_plot(selected_source):
            if not selected_source:
                return go.Figure()
                
            return self.data_engine.create_correlation_matrix(selected_source)
            
        # Additional callbacks for filters, export, etc.
        @self.app.callback(
            Output("filter-modal", "is_open"),
            [Input("filter-button", "n_clicks"),
             Input("close-modal-button", "n_clicks")],
            [State("filter-modal", "is_open")]
        )
        def toggle_filter_modal(filter_clicks, close_clicks, is_open):
            if filter_clicks or close_clicks:
                return not is_open
            return is_open
            
    def run(self, debug: bool = False) -> None:
        """Run the dashboard"""
        
        self.logger.info(f"Starting interactive dashboard on port {self.port}")
        self.app.run_server(debug=debug, port=self.port, host="0.0.0.0")
        
    def update_data(self, data_sources: Dict[str, Any]) -> None:
        """Update dashboard data"""
        
        self.data_engine.load_data(data_sources)


class RealTimeAnalyzer:
    """Real-time analysis capabilities"""
    
    def __init__(self, data_engine: DataExplorationEngine):
        self.data_engine = data_engine
        self.logger = logging.getLogger(__name__)
        
        # Real-time state
        self.is_running = False
        self.update_interval = 1.0  # seconds
        self.data_buffer = []
        self.max_buffer_size = 1000
        
        # Threading
        self.update_thread = None
        self.stop_event = threading.Event()
        
    def start_real_time_analysis(self, 
                                data_source_callback: Callable[[], Dict[str, Any]],
                                update_interval: float = 1.0) -> None:
        """Start real-time analysis"""
        
        if self.is_running:
            self.logger.warning("Real-time analysis already running")
            return
            
        self.update_interval = update_interval
        self.is_running = True
        self.stop_event.clear()
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._real_time_update_loop,
            args=(data_source_callback,)
        )
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("Real-time analysis started")
        
    def stop_real_time_analysis(self) -> None:
        """Stop real-time analysis"""
        
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
            
        self.logger.info("Real-time analysis stopped")
        
    def _real_time_update_loop(self, data_source_callback: Callable[[], Dict[str, Any]]) -> None:
        """Real-time update loop"""
        
        while not self.stop_event.is_set():
            try:
                # Get new data
                new_data = data_source_callback()
                
                # Add timestamp
                new_data['timestamp'] = datetime.now()
                
                # Add to buffer
                self.data_buffer.append(new_data)
                
                # Trim buffer if too large
                if len(self.data_buffer) > self.max_buffer_size:
                    self.data_buffer.pop(0)
                    
                # Update data engine
                self.data_engine.load_data({"real_time": self.data_buffer})
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in real-time update loop: {e}")
                time.sleep(self.update_interval)
                
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics"""
        
        if not self.data_buffer:
            return {}
            
        # Calculate metrics from buffer
        recent_data = self.data_buffer[-100:]  # Last 100 data points
        
        metrics = {
            'buffer_size': len(self.data_buffer),
            'recent_data_points': len(recent_data),
            'update_frequency': 1.0 / self.update_interval,
            'last_update': recent_data[-1]['timestamp'] if recent_data else None
        }
        
        return metrics