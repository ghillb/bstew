"""
Interactive Analysis Tools for NetLogo BEE-STEWARD v2 Parity
===========================================================

Comprehensive interactive analysis tools for exploring bee simulation data,
including statistical analysis, pattern detection, and comparative studies.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import logging
import time
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum

# Statistical analysis dependencies (optional)
try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    np = None
    pd = None
    PANDAS_AVAILABLE = False

# Interactive plotting dependencies (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    from plotly.colors import qualitative
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    pyo = None
    qualitative = None
    PLOTLY_AVAILABLE = False

class AnalysisType(Enum):
    """Types of analysis available"""
    POPULATION_DYNAMICS = "population_dynamics"
    FORAGING_EFFICIENCY = "foraging_efficiency"
    COMMUNICATION_PATTERNS = "communication_patterns"
    SPATIAL_DISTRIBUTION = "spatial_distribution"
    HEALTH_TRENDS = "health_trends"
    RESOURCE_UTILIZATION = "resource_utilization"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    TEMPORAL_PATTERNS = "temporal_patterns"

class StatisticalTest(Enum):
    """Statistical tests available"""
    T_TEST = "t_test"
    ANOVA = "anova"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    CHI_SQUARE = "chi_square"
    TREND_ANALYSIS = "trend_analysis"
    DISTRIBUTION_FIT = "distribution_fit"

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    analysis_type: AnalysisType
    timestamp: float
    data_summary: Dict[str, Any] = field(default_factory=dict)
    statistical_results: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[Any] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'analysis_type': self.analysis_type.value,
            'timestamp': self.timestamp,
            'data_summary': self.data_summary,
            'statistical_results': self.statistical_results,
            'insights': self.insights,
            'recommendations': self.recommendations
        }

@dataclass
class DataFilter:
    """Data filtering configuration"""
    time_range: Optional[Tuple[float, float]] = None
    colony_ids: Optional[List[int]] = None
    bee_statuses: Optional[List[str]] = None
    patch_ids: Optional[List[int]] = None
    health_threshold: Optional[float] = None
    population_range: Optional[Tuple[int, int]] = None
    custom_filters: Dict[str, Any] = field(default_factory=dict)

class DataProcessor(BaseModel):
    """Data processing and filtering utilities"""
    
    model_config = {"validate_assignment": True}
    
    def filter_data(self, data: List[Dict[str, Any]], filter_config: DataFilter) -> List[Dict[str, Any]]:
        """Apply filters to data"""
        
        filtered_data = data.copy()
        
        # Time range filter
        if filter_config.time_range:
            start_time, end_time = filter_config.time_range
            filtered_data = [
                d for d in filtered_data 
                if start_time <= d.get('timestamp', 0) <= end_time
            ]
        
        # Colony ID filter
        if filter_config.colony_ids:
            filtered_data = [
                d for d in filtered_data 
                if d.get('colony_id') in filter_config.colony_ids
            ]
        
        # Bee status filter
        if filter_config.bee_statuses:
            filtered_data = [
                d for d in filtered_data 
                if d.get('bee_status') in filter_config.bee_statuses
            ]
        
        # Health threshold filter
        if filter_config.health_threshold is not None:
            filtered_data = [
                d for d in filtered_data 
                if d.get('health_score', 0) >= filter_config.health_threshold
            ]
        
        # Population range filter
        if filter_config.population_range:
            min_pop, max_pop = filter_config.population_range
            filtered_data = [
                d for d in filtered_data 
                if min_pop <= d.get('population', 0) <= max_pop
            ]
        
        return filtered_data
    
    def aggregate_data(self, data: List[Dict[str, Any]], 
                      group_by: str, aggregation: str = "mean") -> Dict[str, Any]:
        """Aggregate data by specified field"""
        
        grouped = defaultdict(list)
        
        for item in data:
            if group_by in item:
                grouped[item[group_by]].append(item)
        
        aggregated = {}
        
        for group_key, group_data in grouped.items():
            if aggregation == "mean":
                aggregated[group_key] = self._calculate_group_mean(group_data)
            elif aggregation == "sum":
                aggregated[group_key] = self._calculate_group_sum(group_data)
            elif aggregation == "count":
                aggregated[group_key] = len(group_data)
            elif aggregation == "std":
                aggregated[group_key] = self._calculate_group_std(group_data)
        
        return aggregated
    
    def _calculate_group_mean(self, group_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate mean for numeric fields in group"""
        
        numeric_fields = {}
        
        for item in group_data:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(value)
        
        return {
            field: statistics.mean(values) 
            for field, values in numeric_fields.items()
        }
    
    def _calculate_group_sum(self, group_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sum for numeric fields in group"""
        
        numeric_fields = {}
        
        for item in group_data:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = 0
                    numeric_fields[key] += value
        
        return numeric_fields
    
    def _calculate_group_std(self, group_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate standard deviation for numeric fields in group"""
        
        numeric_fields = {}
        
        for item in group_data:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(value)
        
        return {
            field: statistics.stdev(values) if len(values) > 1 else 0.0
            for field, values in numeric_fields.items()
        }

class StatisticalAnalyzer(BaseModel):
    """Statistical analysis tools"""
    
    model_config = {"validate_assignment": True}
    
    def perform_correlation_analysis(self, data: List[Dict[str, Any]], 
                                   fields: List[str]) -> Dict[str, Any]:
        """Perform correlation analysis between specified fields"""
        
        if not PANDAS_AVAILABLE:
            return {"error": "Pandas not available for correlation analysis"}
        
        # Extract numeric data
        df_data = {}
        for field in fields:
            values = []
            for item in data:
                if field in item and isinstance(item[field], (int, float)):
                    values.append(item[field])
                else:
                    values.append(None)
            df_data[field] = values
        
        df = pd.DataFrame(df_data)
        correlation_matrix = df.corr()
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "significant_correlations": self._find_significant_correlations(correlation_matrix),
            "sample_size": len(data)
        }
    
    def _find_significant_correlations(self, corr_matrix, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find statistically significant correlations"""
        
        significant = []
        
        for i, field1 in enumerate(corr_matrix.columns):
            for j, field2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    correlation = corr_matrix.iloc[i, j]
                    if abs(correlation) >= threshold:
                        significant.append({
                            "field1": field1,
                            "field2": field2,
                            "correlation": correlation,
                            "strength": self._correlation_strength(abs(correlation))
                        })
        
        return sorted(significant, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def _correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        
        if correlation >= 0.8:
            return "very_strong"
        elif correlation >= 0.6:
            return "strong"
        elif correlation >= 0.4:
            return "moderate"
        elif correlation >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def perform_trend_analysis(self, data: List[Dict[str, Any]], 
                             time_field: str, value_field: str) -> Dict[str, Any]:
        """Perform trend analysis on time series data"""
        
        # Extract time series
        time_series = []
        for item in data:
            if time_field in item and value_field in item:
                time_series.append((item[time_field], item[value_field]))
        
        if len(time_series) < 3:
            return {"error": "Insufficient data for trend analysis"}
        
        # Sort by time
        time_series.sort(key=lambda x: x[0])
        
        times = [t[0] for t in time_series]
        values = [t[1] for t in time_series]
        
        # Calculate simple linear trend
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate trend strength (R²)
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        ss_res = sum((values[i] - (slope * i + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "trend_strength": self._trend_strength(r_squared),
            "sample_size": n,
            "time_range": (min(times), max(times)),
            "value_range": (min(values), max(values))
        }
    
    def _trend_strength(self, r_squared: float) -> str:
        """Classify trend strength based on R²"""
        
        if r_squared >= 0.9:
            return "very_strong"
        elif r_squared >= 0.7:
            return "strong"
        elif r_squared >= 0.5:
            return "moderate"
        elif r_squared >= 0.3:
            return "weak"
        else:
            return "very_weak"

class VisualizationGenerator(BaseModel):
    """Generate interactive visualizations"""
    
    model_config = {"validate_assignment": True}
    
    def create_population_dynamics_chart(self, data: List[Dict[str, Any]]) -> Optional[Any]:
        """Create population dynamics visualization"""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        # Group data by colony
        colony_data = defaultdict(lambda: {"timestamps": [], "populations": [], "foragers": []})
        
        for item in data:
            colony_id = item.get("colony_id", 0)
            colony_data[colony_id]["timestamps"].append(item.get("timestamp", 0))
            colony_data[colony_id]["populations"].append(item.get("population", 0))
            colony_data[colony_id]["foragers"].append(item.get("forager_count", 0))
        
        fig = go.Figure()
        
        colors = qualitative.Plotly
        
        for i, (colony_id, colony_info) in enumerate(colony_data.items()):
            color = colors[i % len(colors)]
            
            # Population line
            fig.add_trace(go.Scatter(
                x=colony_info["timestamps"],
                y=colony_info["populations"],
                mode='lines+markers',
                name=f'Colony {colony_id} Population',
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
            
            # Forager line
            fig.add_trace(go.Scatter(
                x=colony_info["timestamps"],
                y=colony_info["foragers"],
                mode='lines',
                name=f'Colony {colony_id} Foragers',
                line=dict(color=color, width=1, dash='dash'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Population Dynamics Over Time",
            xaxis_title="Time",
            yaxis_title="Count",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_foraging_efficiency_chart(self, data: List[Dict[str, Any]]) -> Optional[Any]:
        """Create foraging efficiency visualization"""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Success Rate', 'Energy Gained', 'Distance Traveled', 'Time Efficiency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract foraging metrics
        timestamps = [item.get("timestamp", 0) for item in data]
        success_rates = [item.get("foraging_success_rate", 0) for item in data]
        energy_gained = [item.get("average_energy_gained", 0) for item in data]
        distances = [item.get("average_distance_traveled", 0) for item in data]
        time_efficiency = [item.get("time_efficiency", 0) for item in data]
        
        # Success rate
        fig.add_trace(
            go.Scatter(x=timestamps, y=success_rates, name='Success Rate', 
                      line=dict(color='green')),
            row=1, col=1
        )
        
        # Energy gained
        fig.add_trace(
            go.Scatter(x=timestamps, y=energy_gained, name='Energy Gained',
                      line=dict(color='orange')),
            row=1, col=2
        )
        
        # Distance traveled
        fig.add_trace(
            go.Scatter(x=timestamps, y=distances, name='Distance Traveled',
                      line=dict(color='blue')),
            row=2, col=1
        )
        
        # Time efficiency
        fig.add_trace(
            go.Scatter(x=timestamps, y=time_efficiency, name='Time Efficiency',
                      line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Foraging Efficiency Analysis",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def create_communication_network_graph(self, data: List[Dict[str, Any]]) -> Optional[Any]:
        """Create communication network visualization"""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        # Extract communication data
        nodes = set()
        edges = []
        
        for item in data:
            if "sender_id" in item and "receiver_id" in item:
                sender = item["sender_id"]
                receiver = item["receiver_id"]
                nodes.add(sender)
                nodes.add(receiver)
                edges.append((sender, receiver, item.get("information_quality", 0.5)))
        
        if not nodes:
            return None
        
        # Create network layout (simple circular layout)
        node_positions = {}
        n_nodes = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            angle = 2 * math.pi * i / n_nodes
            node_positions[node] = (math.cos(angle), math.sin(angle))
        
        fig = go.Figure()
        
        # Add edges
        for sender, receiver, quality in edges:
            x_coords = [node_positions[sender][0], node_positions[receiver][0], None]
            y_coords = [node_positions[sender][1], node_positions[receiver][1], None]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                line=dict(width=quality * 5, color=f'rgba(0,0,255,{quality})'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add nodes
        x_nodes = [pos[0] for pos in node_positions.values()]
        y_nodes = [pos[1] for pos in node_positions.values()]
        
        fig.add_trace(go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            marker=dict(size=15, color='red'),
            text=list(nodes),
            textposition='middle center',
            name='Bees',
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="Communication Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_data: Dict[str, Any]) -> Optional[Any]:
        """Create correlation heatmap"""
        
        if not PLOTLY_AVAILABLE or "correlation_matrix" not in correlation_data:
            return None
        
        corr_matrix = correlation_data["correlation_matrix"]
        
        # Convert to lists for Plotly
        fields = list(corr_matrix.keys())
        z_values = []
        
        for field1 in fields:
            row = []
            for field2 in fields:
                row.append(corr_matrix[field1].get(field2, 0))
            z_values.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=fields,
            y=fields,
            colorscale='RdBu',
            zmid=0,
            text=[[f"{val:.2f}" for val in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Analysis Heatmap",
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig

class InteractiveAnalysisEngine(BaseModel):
    """Main interactive analysis engine"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Core components
    data_processor: DataProcessor
    statistical_analyzer: StatisticalAnalyzer
    visualization_generator: VisualizationGenerator
    
    # Analysis results storage
    analysis_results: List[AnalysisResult] = Field(default_factory=list)
    
    # Configuration
    output_directory: str = "analysis_output"
    
    # Logging
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, **data):
        if 'data_processor' not in data:
            data['data_processor'] = DataProcessor()
        if 'statistical_analyzer' not in data:
            data['statistical_analyzer'] = StatisticalAnalyzer()
        if 'visualization_generator' not in data:
            data['visualization_generator'] = VisualizationGenerator()
        
        super().__init__(**data)
        self.logger = logging.getLogger(__name__)
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist"""
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
    
    def analyze_population_dynamics(self, data: List[Dict[str, Any]], 
                                  filter_config: Optional[DataFilter] = None) -> AnalysisResult:
        """Analyze population dynamics"""
        
        # Filter data if requested
        if filter_config:
            data = self.data_processor.filter_data(data, filter_config)
        
        result = AnalysisResult(
            analysis_type=AnalysisType.POPULATION_DYNAMICS,
            timestamp=time.time()
        )
        
        # Data summary
        populations = [item.get("population", 0) for item in data if "population" in item]
        if populations:
            result.data_summary = {
                "sample_size": len(populations),
                "mean_population": statistics.mean(populations),
                "median_population": statistics.median(populations),
                "std_population": statistics.stdev(populations) if len(populations) > 1 else 0,
                "min_population": min(populations),
                "max_population": max(populations)
            }
        
        # Trend analysis
        if len(data) > 3:
            trend_results = self.statistical_analyzer.perform_trend_analysis(
                data, "timestamp", "population"
            )
            result.statistical_results["trend_analysis"] = trend_results
            
            # Generate insights
            if trend_results.get("trend_direction") == "increasing":
                result.insights.append("Population shows an increasing trend over time")
            elif trend_results.get("trend_direction") == "decreasing":
                result.insights.append("Population shows a decreasing trend over time")
            else:
                result.insights.append("Population remains relatively stable over time")
        
        # Create visualization
        viz = self.visualization_generator.create_population_dynamics_chart(data)
        if viz:
            result.visualizations.append(viz)
        
        # Add recommendations
        if result.data_summary.get("mean_population", 0) < 5000:
            result.recommendations.append("Consider investigating factors causing low population")
        
        self.analysis_results.append(result)
        return result
    
    def analyze_foraging_efficiency(self, data: List[Dict[str, Any]], 
                                  filter_config: Optional[DataFilter] = None) -> AnalysisResult:
        """Analyze foraging efficiency patterns"""
        
        # Filter data if requested
        if filter_config:
            data = self.data_processor.filter_data(data, filter_config)
        
        result = AnalysisResult(
            analysis_type=AnalysisType.FORAGING_EFFICIENCY,
            timestamp=time.time()
        )
        
        # Data summary
        efficiency_metrics = ["foraging_success_rate", "average_energy_gained", "time_efficiency"]
        for metric in efficiency_metrics:
            values = [item.get(metric, 0) for item in data if metric in item]
            if values:
                result.data_summary[metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values)
                }
        
        # Correlation analysis
        if len(data) > 10:
            correlation_results = self.statistical_analyzer.perform_correlation_analysis(
                data, efficiency_metrics
            )
            result.statistical_results["correlations"] = correlation_results
        
        # Create visualization
        viz = self.visualization_generator.create_foraging_efficiency_chart(data)
        if viz:
            result.visualizations.append(viz)
        
        # Generate insights
        success_rate = result.data_summary.get("foraging_success_rate", {}).get("mean", 0)
        if success_rate > 0.8:
            result.insights.append("High foraging success rate indicates efficient resource utilization")
        elif success_rate < 0.5:
            result.insights.append("Low foraging success rate suggests potential issues with resource access")
        
        self.analysis_results.append(result)
        return result
    
    def analyze_communication_patterns(self, data: List[Dict[str, Any]], 
                                     filter_config: Optional[DataFilter] = None) -> AnalysisResult:
        """Analyze communication and recruitment patterns"""
        
        # Filter data if requested
        if filter_config:
            data = self.data_processor.filter_data(data, filter_config)
        
        result = AnalysisResult(
            analysis_type=AnalysisType.COMMUNICATION_PATTERNS,
            timestamp=time.time()
        )
        
        # Communication metrics
        dance_count = len([item for item in data if item.get("event_type") == "dance"])
        recruitment_count = len([item for item in data if item.get("event_type") == "recruitment"])
        
        result.data_summary = {
            "total_dances": dance_count,
            "total_recruitments": recruitment_count,
            "recruitment_rate": recruitment_count / dance_count if dance_count > 0 else 0
        }
        
        # Create network visualization
        viz = self.visualization_generator.create_communication_network_graph(data)
        if viz:
            result.visualizations.append(viz)
        
        # Generate insights
        recruitment_rate = result.data_summary["recruitment_rate"]
        if recruitment_rate > 0.7:
            result.insights.append("High recruitment efficiency indicates effective communication")
        elif recruitment_rate < 0.3:
            result.insights.append("Low recruitment efficiency suggests communication issues")
        
        self.analysis_results.append(result)
        return result
    
    def perform_comparative_analysis(self, datasets: Dict[str, List[Dict[str, Any]]], 
                                   metric: str) -> AnalysisResult:
        """Perform comparative analysis between different datasets"""
        
        result = AnalysisResult(
            analysis_type=AnalysisType.COMPARATIVE_ANALYSIS,
            timestamp=time.time()
        )
        
        # Compare datasets
        comparison_data = {}
        for dataset_name, data in datasets.items():
            values = [item.get(metric, 0) for item in data if metric in item]
            if values:
                comparison_data[dataset_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "sample_size": len(values)
                }
        
        result.data_summary = comparison_data
        
        # Generate comparative insights
        if len(comparison_data) >= 2:
            means = {name: data["mean"] for name, data in comparison_data.items()}
            best_performer = max(means, key=means.get)
            worst_performer = min(means, key=means.get)
            
            result.insights.append(f"Best performing dataset: {best_performer}")
            result.insights.append(f"Worst performing dataset: {worst_performer}")
            
            improvement = ((means[best_performer] - means[worst_performer]) / means[worst_performer]) * 100
            result.insights.append(f"Performance difference: {improvement:.1f}%")
        
        self.analysis_results.append(result)
        return result
    
    def generate_comprehensive_report(self, title: str = "Comprehensive Analysis Report") -> str:
        """Generate comprehensive analysis report"""
        
        if not self.analysis_results:
            return "No analysis results available"
        
        report_lines = [
            f"# {title}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Analyses: {len(self.analysis_results)}",
            "",
            "## Summary of Analyses",
            ""
        ]
        
        # Analysis summary
        analysis_types = Counter(result.analysis_type.value for result in self.analysis_results)
        for analysis_type, count in analysis_types.items():
            report_lines.append(f"- {analysis_type.replace('_', ' ').title()}: {count}")
        
        report_lines.extend(["", "## Detailed Results", ""])
        
        # Detailed results
        for i, result in enumerate(self.analysis_results):
            report_lines.extend([
                f"### Analysis {i+1}: {result.analysis_type.value.replace('_', ' ').title()}",
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.timestamp))}",
                ""
            ])
            
            # Data summary
            if result.data_summary:
                report_lines.append("**Data Summary:**")
                for key, value in result.data_summary.items():
                    if isinstance(value, dict):
                        report_lines.append(f"- {key}:")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, float):
                                report_lines.append(f"  - {subkey}: {subvalue:.3f}")
                            else:
                                report_lines.append(f"  - {subkey}: {subvalue}")
                    else:
                        if isinstance(value, float):
                            report_lines.append(f"- {key}: {value:.3f}")
                        else:
                            report_lines.append(f"- {key}: {value}")
                report_lines.append("")
            
            # Insights
            if result.insights:
                report_lines.append("**Key Insights:**")
                for insight in result.insights:
                    report_lines.append(f"- {insight}")
                report_lines.append("")
            
            # Recommendations
            if result.recommendations:
                report_lines.append("**Recommendations:**")
                for recommendation in result.recommendations:
                    report_lines.append(f"- {recommendation}")
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        timestamp = int(time.time())
        report_file = Path(self.output_directory) / f"analysis_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Analysis report saved to {report_file}")
        
        return report_content
    
    def export_analysis_results(self, format: str = "json") -> str:
        """Export analysis results to file"""
        
        if format == "json":
            results_data = [result.to_dict() for result in self.analysis_results]
            timestamp = int(time.time())
            output_file = Path(self.output_directory) / f"analysis_results_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
        
        return str(output_file)

def create_interactive_analysis_engine(output_dir: str = "analysis_output") -> InteractiveAnalysisEngine:
    """Factory function to create interactive analysis engine"""
    
    return InteractiveAnalysisEngine(output_directory=output_dir)