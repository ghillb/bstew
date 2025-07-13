"""
Performance Regression Testing Suite
===================================

Automated performance regression detection and tracking.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from .netlogo_parity_benchmarks import NetLogoParityBenchmarks


@dataclass
class RegressionResult:
    """Performance regression test result"""
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    is_regression: bool
    severity: str  # "minor", "major", "critical"


class PerformanceRegressionSuite:
    """Automated performance regression testing"""
    
    def __init__(self, baseline_file: Optional[str] = None, 
                 output_directory: str = "artifacts/regression"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Regression thresholds
        self.thresholds = {
            "minor": 0.05,    # 5% performance drop
            "major": 0.15,    # 15% performance drop  
            "critical": 0.30   # 30% performance drop
        }
        
        # Load baseline if provided
        self.baseline_data = None
        if baseline_file and Path(baseline_file).exists():
            with open(baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
    
    def run_regression_check(self) -> Dict[str, Any]:
        """Run complete regression check"""
        
        self.logger.info("Starting performance regression check")
        
        # Run current benchmarks
        benchmarks = NetLogoParityBenchmarks(str(self.output_directory))
        current_results = benchmarks.run_complete_benchmark_suite()
        
        if not self.baseline_data:
            # No baseline - save current results as new baseline
            self._save_baseline(current_results)
            return {
                "status": "baseline_created",
                "message": "No baseline found. Current results saved as new baseline.",
                "baseline_file": str(self.output_directory / "performance_baseline.json")
            }
        
        # Compare with baseline
        regression_results = self._compare_with_baseline(current_results)
        
        # Generate regression report
        report = self._generate_regression_report(regression_results)
        
        # Save results
        self._save_regression_results(regression_results, current_results)
        
        return report
    
    def _compare_with_baseline(self, current_results: Dict[str, Any]) -> List[RegressionResult]:
        """Compare current results with baseline"""
        
        regressions = []
        
        # Compare simulation speed benchmarks
        baseline_data = self.baseline_data or {}
        if ("simulation_speed" in baseline_data and 
            "simulation_speed" in current_results):
            
            baseline_speed = baseline_data["simulation_speed"]
            current_speed = current_results["simulation_speed"]
            
            for size_name in baseline_speed:
                if size_name in current_speed:
                    baseline_result = baseline_speed[size_name]
                    current_result = current_speed[size_name]
                    
                    # Extract steps per second
                    if (isinstance(baseline_result, dict) and "steps_per_second" in baseline_result and
                        hasattr(current_result, 'steps_per_second')):
                        
                        baseline_sps = baseline_result["steps_per_second"]
                        current_sps = current_result.steps_per_second
                        
                        change_percent = (current_sps - baseline_sps) / baseline_sps
                        
                        regression = RegressionResult(
                            metric_name=f"simulation_speed_{size_name}",
                            current_value=current_sps,
                            baseline_value=baseline_sps,
                            change_percent=change_percent,
                            is_regression=change_percent < -self.thresholds["minor"],
                            severity=self._determine_severity(change_percent)
                        )
                        regressions.append(regression)
        
        # Compare memory usage
        if ("memory_efficiency" in baseline_data and 
            "memory_efficiency" in current_results):
            
            baseline_memory = baseline_data["memory_efficiency"]
            current_memory = current_results["memory_efficiency"]
            
            for scenario_name in baseline_memory:
                if scenario_name in current_memory:
                    baseline_result = baseline_memory[scenario_name]
                    current_result = current_memory[scenario_name]
                    
                    if (isinstance(baseline_result, dict) and "memory_peak" in baseline_result and
                        hasattr(current_result, 'memory_peak')):
                        
                        baseline_mem = baseline_result["memory_peak"]
                        current_mem = current_result.memory_peak
                        
                        # For memory, higher is worse
                        change_percent = (current_mem - baseline_mem) / baseline_mem
                        
                        regression = RegressionResult(
                            metric_name=f"memory_efficiency_{scenario_name}",
                            current_value=current_mem,
                            baseline_value=baseline_mem,
                            change_percent=change_percent,
                            is_regression=change_percent > self.thresholds["minor"],
                            severity=self._determine_severity(change_percent, higher_is_worse=True)
                        )
                        regressions.append(regression)
        
        return regressions
    
    def _determine_severity(self, change_percent: float, higher_is_worse: bool = False) -> str:
        """Determine regression severity"""
        
        if higher_is_worse:
            # For metrics where higher values are worse (like memory usage)
            if change_percent > self.thresholds["critical"]:
                return "critical"
            elif change_percent > self.thresholds["major"]:
                return "major"
            elif change_percent > self.thresholds["minor"]:
                return "minor"
            else:
                return "none"
        else:
            # For metrics where lower values are worse (like performance)
            if change_percent < -self.thresholds["critical"]:
                return "critical"
            elif change_percent < -self.thresholds["major"]:
                return "major"
            elif change_percent < -self.thresholds["minor"]:
                return "minor"
            else:
                return "none"
    
    def _generate_regression_report(self, regressions: List[RegressionResult]) -> Dict[str, Any]:
        """Generate regression analysis report"""
        
        # Categorize regressions by severity
        critical_regressions = [r for r in regressions if r.severity == "critical"]
        major_regressions = [r for r in regressions if r.severity == "major"]
        minor_regressions = [r for r in regressions if r.severity == "minor"]
        improvements = [r for r in regressions if not r.is_regression and r.change_percent > 0.05]
        
        # Determine overall status
        if critical_regressions:
            overall_status = "critical_regressions"
        elif major_regressions:
            overall_status = "major_regressions"
        elif minor_regressions:
            overall_status = "minor_regressions"
        else:
            overall_status = "no_regressions"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_metrics": len(regressions),
                "regressions_detected": len([r for r in regressions if r.is_regression]),
                "critical_count": len(critical_regressions),
                "major_count": len(major_regressions),
                "minor_count": len(minor_regressions),
                "improvements_count": len(improvements)
            },
            "regressions": {
                "critical": [self._regression_to_dict(r) for r in critical_regressions],
                "major": [self._regression_to_dict(r) for r in major_regressions],
                "minor": [self._regression_to_dict(r) for r in minor_regressions]
            },
            "improvements": [self._regression_to_dict(r) for r in improvements],
            "recommendations": self._generate_recommendations(regressions)
        }
        
        return report
    
    def _regression_to_dict(self, regression: RegressionResult) -> Dict[str, Any]:
        """Convert regression result to dictionary"""
        
        return {
            "metric_name": regression.metric_name,
            "current_value": regression.current_value,
            "baseline_value": regression.baseline_value,
            "change_percent": regression.change_percent * 100,  # Convert to percentage
            "is_regression": regression.is_regression,
            "severity": regression.severity
        }
    
    def _generate_recommendations(self, regressions: List[RegressionResult]) -> List[str]:
        """Generate recommendations based on regression analysis"""
        
        recommendations = []
        
        critical_regressions = [r for r in regressions if r.severity == "critical"]
        major_regressions = [r for r in regressions if r.severity == "major"]
        
        if critical_regressions:
            recommendations.append("URGENT: Critical performance regressions detected. Immediate investigation required.")
            
            # Identify problem areas
            speed_issues = [r for r in critical_regressions if "simulation_speed" in r.metric_name]
            memory_issues = [r for r in critical_regressions if "memory" in r.metric_name]
            
            if speed_issues:
                recommendations.append("Critical simulation speed regressions found. Check recent algorithm changes.")
            if memory_issues:
                recommendations.append("Critical memory usage regressions found. Check for memory leaks.")
        
        elif major_regressions:
            recommendations.append("Major performance regressions detected. Investigation recommended.")
            
        # Performance-specific recommendations
        speed_regressions = [r for r in regressions if r.is_regression and "simulation_speed" in r.metric_name]
        if speed_regressions:
            avg_speed_loss = sum(abs(r.change_percent) for r in speed_regressions) / len(speed_regressions)
            if avg_speed_loss > 0.2:
                recommendations.append(f"Average simulation speed loss: {avg_speed_loss:.1%}. Consider performance optimization.")
        
        memory_regressions = [r for r in regressions if r.is_regression and "memory" in r.metric_name]
        if memory_regressions:
            recommendations.append("Memory usage has increased. Review recent changes for potential memory leaks.")
        
        # If no regressions
        if not any(r.is_regression for r in regressions):
            improvements = [r for r in regressions if r.change_percent > 0.05]
            if improvements:
                recommendations.append("Performance improvements detected! Good work on optimization.")
            else:
                recommendations.append("Performance is stable. No significant changes detected.")
        
        return recommendations
    
    def _save_baseline(self, results: Dict[str, Any]) -> None:
        """Save current results as new baseline"""
        
        baseline_file = self.output_directory / "performance_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Baseline saved to {baseline_file}")
    
    def _save_regression_results(self, regressions: List[RegressionResult], 
                                current_results: Dict[str, Any]) -> None:
        """Save regression analysis results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed regression analysis
        regression_file = self.output_directory / f"regression_analysis_{timestamp}.json"
        regression_data = {
            "timestamp": datetime.now().isoformat(),
            "regressions": [self._regression_to_dict(r) for r in regressions],
            "current_results": current_results
        }
        
        with open(regression_file, 'w') as f:
            json.dump(regression_data, f, indent=2, default=str)
        
        # Update baseline with current results
        self._save_baseline(current_results)
        
        self.logger.info(f"Regression analysis saved to {regression_file}")
    
    def set_custom_thresholds(self, minor: float, major: float, critical: float) -> None:
        """Set custom regression thresholds"""
        
        self.thresholds = {
            "minor": minor,
            "major": major,
            "critical": critical
        }
        
        self.logger.info(f"Custom thresholds set: minor={minor:.1%}, major={major:.1%}, critical={critical:.1%}")
    
    def load_baseline_from_file(self, baseline_file: str) -> bool:
        """Load baseline from specific file"""
        
        try:
            with open(baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
            
            self.logger.info(f"Baseline loaded from {baseline_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load baseline from {baseline_file}: {e}")
            return False