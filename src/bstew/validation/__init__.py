"""
BSTEW Validation Package
=======================

Performance validation and NetLogo comparison systems for BSTEW.

This package provides comprehensive validation tools for ensuring BSTEW
maintains parity with the original NetLogo BEE-STEWARD model while
delivering superior performance.

Phase 5 - Final Phase for 100% BSTEW Completion
"""

from .performance_benchmarks import (
    PerformanceBenchmark,
    BenchmarkResult,
    BenchmarkType,
    PerformanceMetric,
    create_performance_benchmark,
)

from .netlogo_comparison import (
    NetLogoComparison,
    ValidationReport,
    ValidationResult,
    ComparisonMetric,
    create_netlogo_comparison,
)

__all__ = [
    "PerformanceBenchmark",
    "BenchmarkResult",
    "BenchmarkType",
    "PerformanceMetric",
    "create_performance_benchmark",
    "NetLogoComparison",
    "ValidationReport",
    "ValidationResult",
    "ComparisonMetric",
    "create_netlogo_comparison",
]

__version__ = "1.0.0"
