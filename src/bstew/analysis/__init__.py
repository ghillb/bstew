"""
Analysis Package for BEE-STEWARD
===============================

Interactive analysis tools and statistical utilities for exploring
bee simulation data and generating insights.
"""

from .interactive_tools import (
    InteractiveAnalysisEngine,
    AnalysisType,
    AnalysisResult,
    DataFilter,
    DataProcessor,
    StatisticalAnalyzer,
    VisualizationGenerator,
    create_interactive_analysis_engine,
)

from .population_analyzer import (
    PopulationAnalyzer,
    TrendType,
    TrendResult,
    GrowthRateResult,
    SurvivalResult,
)

from .foraging_analyzer import (
    ForagingAnalyzer,
    ForagingMetric,
    EfficiencyCategory,
    ForagingEfficiencyResult,
    ResourceOptimizationResult,
    BehavioralPatternResult,
)

from .population_plotter import PopulationPlotter, PlotStyle, ColorScheme, PlotConfig

from .spatial_plotter import (
    SpatialPlotter,
    SpatialPlotType,
    MapProjection,
    SpatialConfig,
)

__all__ = [
    "InteractiveAnalysisEngine",
    "AnalysisType",
    "AnalysisResult",
    "DataFilter",
    "DataProcessor",
    "StatisticalAnalyzer",
    "VisualizationGenerator",
    "create_interactive_analysis_engine",
    "PopulationAnalyzer",
    "TrendType",
    "TrendResult",
    "GrowthRateResult",
    "SurvivalResult",
    "ForagingAnalyzer",
    "ForagingMetric",
    "EfficiencyCategory",
    "ForagingEfficiencyResult",
    "ResourceOptimizationResult",
    "BehavioralPatternResult",
    "PopulationPlotter",
    "PlotStyle",
    "ColorScheme",
    "PlotConfig",
    "SpatialPlotter",
    "SpatialPlotType",
    "MapProjection",
    "SpatialConfig",
]
