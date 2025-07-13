"""
Performance Regression Detection System
======================================

Automated detection of performance regressions during optimization integration.
Compares current performance against baseline metrics with statistical analysis.
"""

import json
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np
from enum import Enum


class RegressionSeverity(Enum):
    """Severity levels for performance regressions"""
    NONE = "none"
    MINOR = "minor"          # 1-5% degradation
    MODERATE = "moderate"    # 5-15% degradation  
    MAJOR = "major"         # 15-30% degradation
    CRITICAL = "critical"   # >30% degradation


@dataclass
class RegressionResult:
    """Result of regression analysis"""
    metric_name: str
    baseline_value: float
    current_value: float
    percentage_change: float
    severity: RegressionSeverity
    is_regression: bool
    confidence_level: float
    statistical_significance: bool


@dataclass
class RegressionReport:
    """Comprehensive regression analysis report"""
    test_name: str
    timestamp: str
    overall_status: str  # "PASS", "WARN", "FAIL"
    regressions: List[RegressionResult]
    improvements: List[RegressionResult]
    summary: Dict[str, Any]


class PerformanceRegressionDetector:
    """Detects performance regressions by comparing against baseline"""
    
    def __init__(self, 
                 baseline_file: str,
                 regression_threshold: float = 0.05,  # 5% threshold
                 confidence_level: float = 0.95):
        
        self.baseline_file = Path(baseline_file)
        self.regression_threshold = regression_threshold
        self.confidence_level = confidence_level
        
        # Load baseline data
        self.baseline_data = self._load_baseline_data()
        
        # Regression thresholds by severity
        self.severity_thresholds = {
            RegressionSeverity.MINOR: 0.01,     # 1%
            RegressionSeverity.MODERATE: 0.05,  # 5%
            RegressionSeverity.MAJOR: 0.15,     # 15%
            RegressionSeverity.CRITICAL: 0.30   # 30%
        }
    
    def _load_baseline_data(self) -> Dict[str, Any]:
        """Load baseline performance data"""
        if not self.baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {self.baseline_file}")
        
        with open(self.baseline_file, 'r') as f:
            return json.load(f)
    
    def _calculate_baseline_stats(self, scenario_name: str) -> Dict[str, Tuple[float, float]]:
        """Calculate baseline statistics (mean, std) for a scenario"""
        scenario_data = [
            result for result in self.baseline_data 
            if result['simulation_name'].startswith(scenario_name)
        ]
        
        if not scenario_data:
            return {}
        
        stats = {}
        
        # Key performance metrics
        metrics = ['steps_per_second', 'memory_usage_mb', 'total_time', 'peak_memory_mb']
        
        for metric in metrics:
            values = [result[metric] for result in scenario_data if metric in result]
            if values:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                stats[metric] = (mean_val, std_val)
        
        return stats
    
    def _determine_severity(self, percentage_change: float) -> RegressionSeverity:
        """Determine regression severity based on percentage change"""
        abs_change = abs(percentage_change)
        
        if abs_change >= self.severity_thresholds[RegressionSeverity.CRITICAL]:
            return RegressionSeverity.CRITICAL
        elif abs_change >= self.severity_thresholds[RegressionSeverity.MAJOR]:
            return RegressionSeverity.MAJOR
        elif abs_change >= self.severity_thresholds[RegressionSeverity.MODERATE]:
            return RegressionSeverity.MODERATE
        elif abs_change >= self.severity_thresholds[RegressionSeverity.MINOR]:
            return RegressionSeverity.MINOR
        else:
            return RegressionSeverity.NONE
    
    def _is_statistically_significant(self, baseline_mean: float, baseline_std: float,
                                    current_value: float, sample_size: int = 3) -> Tuple[bool, float]:
        """Check if difference is statistically significant"""
        if baseline_std == 0:
            # If no variance in baseline, any change is significant
            return abs(current_value - baseline_mean) > 0, 1.0
        
        # Calculate z-score
        z_score = abs(current_value - baseline_mean) / (baseline_std / np.sqrt(sample_size))
        
        # Critical value for 95% confidence (two-tailed)
        critical_value = 1.96
        
        is_significant = z_score > critical_value
        confidence = min(0.99, 1.0 - (2 * (1 - 0.5 * (1 + np.tanh(z_score - critical_value)))))
        
        return is_significant, confidence
    
    def analyze_regression(self, current_results: List[Dict[str, Any]], 
                         scenario_name: str) -> RegressionReport:
        """Analyze performance regression for a scenario"""
        
        # Get baseline statistics
        baseline_stats = self._calculate_baseline_stats(scenario_name)
        
        if not baseline_stats:
            raise ValueError(f"No baseline data found for scenario: {scenario_name}")
        
        # Calculate current statistics
        current_stats = {}
        metrics = ['steps_per_second', 'memory_usage_mb', 'total_time', 'peak_memory_mb']
        
        for metric in metrics:
            values = [result[metric] for result in current_results if metric in result]
            if values:
                current_stats[metric] = statistics.mean(values)
        
        # Analyze each metric
        regressions = []
        improvements = []
        
        for metric_name in metrics:
            if metric_name not in baseline_stats or metric_name not in current_stats:
                continue
            
            baseline_mean, baseline_std = baseline_stats[metric_name]
            current_value = current_stats[metric_name]
            
            # Calculate percentage change
            percentage_change = (current_value - baseline_mean) / baseline_mean
            
            # For memory and time metrics, negative change is improvement
            # For steps_per_second, positive change is improvement
            is_improvement = False
            if metric_name in ['memory_usage_mb', 'total_time', 'peak_memory_mb']:
                is_improvement = percentage_change < 0
                is_regression = percentage_change > self.regression_threshold
            else:  # steps_per_second
                is_improvement = percentage_change > 0
                is_regression = percentage_change < -self.regression_threshold
            
            # Statistical significance
            is_significant, confidence = self._is_statistically_significant(
                baseline_mean, baseline_std, current_value
            )
            
            # Determine severity
            severity = self._determine_severity(percentage_change)
            
            result = RegressionResult(
                metric_name=metric_name,
                baseline_value=baseline_mean,
                current_value=current_value,
                percentage_change=percentage_change,
                severity=severity,
                is_regression=is_regression and is_significant,
                confidence_level=confidence,
                statistical_significance=is_significant
            )
            
            if is_improvement and is_significant:
                improvements.append(result)
            elif is_regression and is_significant:
                regressions.append(result)
        
        # Determine overall status
        overall_status = self._determine_overall_status(regressions)
        
        # Generate summary
        summary = self._generate_summary(regressions, improvements, baseline_stats, current_stats)
        
        return RegressionReport(
            test_name=scenario_name,
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            regressions=regressions,
            improvements=improvements,
            summary=summary
        )
    
    def _determine_overall_status(self, regressions: List[RegressionResult]) -> str:
        """Determine overall regression status"""
        if not regressions:
            return "PASS"
        
        # Check for critical or major regressions
        for regression in regressions:
            if regression.severity in [RegressionSeverity.CRITICAL, RegressionSeverity.MAJOR]:
                return "FAIL"
        
        # Check for moderate regressions
        moderate_regressions = [r for r in regressions if r.severity == RegressionSeverity.MODERATE]
        if len(moderate_regressions) > 1:  # Multiple moderate regressions
            return "FAIL"
        elif moderate_regressions:
            return "WARN"
        
        # Only minor regressions
        return "WARN" if regressions else "PASS"
    
    def _generate_summary(self, regressions: List[RegressionResult], 
                         improvements: List[RegressionResult],
                         baseline_stats: Dict[str, Tuple[float, float]],
                         current_stats: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        summary = {
            "total_regressions": len(regressions),
            "total_improvements": len(improvements),
            "regression_severity_counts": {
                severity.value: len([r for r in regressions if r.severity == severity])
                for severity in RegressionSeverity
            },
            "baseline_performance": {
                metric: stats[0] for metric, stats in baseline_stats.items()
            },
            "current_performance": current_stats,
            "performance_changes": {}
        }
        
        # Calculate overall performance change
        for metric in baseline_stats:
            if metric in current_stats:
                baseline_val = baseline_stats[metric][0]
                current_val = current_stats[metric]
                change = (current_val - baseline_val) / baseline_val * 100
                summary["performance_changes"][metric] = f"{change:+.2f}%"
        
        return summary
    
    def print_regression_report(self, report: RegressionReport) -> None:
        """Print formatted regression report"""
        
        print("\n" + "=" * 80)
        print(f"PERFORMANCE REGRESSION REPORT: {report.test_name}")
        print("=" * 80)
        print(f"Status: {report.overall_status}")
        print(f"Timestamp: {report.timestamp}")
        
        # Print summary
        print("\nSUMMARY:")
        print(f"  Regressions: {report.summary['total_regressions']}")
        print(f"  Improvements: {report.summary['total_improvements']}")
        
        # Print severity breakdown
        severity_counts = report.summary['regression_severity_counts']
        for severity, count in severity_counts.items():
            if count > 0:
                print(f"    {severity.capitalize()}: {count}")
        
        # Print regressions
        if report.regressions:
            print("\nREGRESSIONS DETECTED:")
            for regression in report.regressions:
                print(f"  ⚠️  {regression.metric_name}:")
                print(f"      Baseline: {regression.baseline_value:.3f}")
                print(f"      Current:  {regression.current_value:.3f}")
                print(f"      Change:   {regression.percentage_change:+.2%}")
                print(f"      Severity: {regression.severity.value}")
                print(f"      Confidence: {regression.confidence_level:.2%}")
        
        # Print improvements
        if report.improvements:
            print("\nIMPROVEMENTS DETECTED:")
            for improvement in report.improvements:
                print(f"  ✅ {improvement.metric_name}:")
                print(f"      Baseline: {improvement.baseline_value:.3f}")
                print(f"      Current:  {improvement.current_value:.3f}")
                print(f"      Change:   {improvement.percentage_change:+.2%}")
        
        # Print performance changes
        print("\nPERFORMANCE CHANGES:")
        for metric, change in report.summary['performance_changes'].items():
            print(f"  {metric}: {change}")
        
        print("=" * 80)
    
    def save_regression_report(self, report: RegressionReport, output_path: str) -> None:
        """Save regression report to file"""
        
        # Convert to serializable format
        report_data = {
            "test_name": report.test_name,
            "timestamp": report.timestamp,
            "overall_status": report.overall_status,
            "regressions": [
                {
                    "metric_name": r.metric_name,
                    "baseline_value": r.baseline_value,
                    "current_value": r.current_value,
                    "percentage_change": r.percentage_change,
                    "severity": r.severity.value,
                    "is_regression": bool(r.is_regression),
                    "confidence_level": float(r.confidence_level),
                    "statistical_significance": bool(r.statistical_significance)
                }
                for r in report.regressions
            ],
            "improvements": [
                {
                    "metric_name": i.metric_name,
                    "baseline_value": i.baseline_value,
                    "current_value": i.current_value,
                    "percentage_change": i.percentage_change,
                    "confidence_level": float(i.confidence_level),
                    "statistical_significance": bool(i.statistical_significance)
                }
                for i in report.improvements
            ],
            "summary": report.summary
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Regression report saved to: {output_path}")


def validate_optimization_performance(baseline_file: str, current_results: List[Dict[str, Any]], 
                                    scenario_name: str) -> bool:
    """Validate that optimization doesn't introduce regressions"""
    
    detector = PerformanceRegressionDetector(baseline_file)
    report = detector.analyze_regression(current_results, scenario_name)
    
    # Print report
    detector.print_regression_report(report)
    
    # Save report
    output_path = f"benchmarks/results/regression_report_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    detector.save_regression_report(report, output_path)
    
    # Return pass/fail status
    return report.overall_status == "PASS"


if __name__ == "__main__":
    # Example usage
    print("Performance Regression Detection System Ready")
    print("Use validate_optimization_performance() to check for regressions")