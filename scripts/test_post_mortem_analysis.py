#!/usr/bin/env python3
"""
Test Script for Post-Mortem Analysis System
==========================================

Demonstrates comprehensive simulation analysis, failure detection,
trend analysis, and report generation capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import tempfile
from datetime import datetime, timedelta

from src.bstew.core.post_mortem_analysis import (
    PostMortemAnalyzer, AnalysisType, generate_comprehensive_report
)

def create_mock_simulation_data():
    """Create mock simulation data for testing"""
    
    # Base simulation info
    duration_days = 180
    start_date = datetime.now() - timedelta(days=duration_days)
    
    # Generate time series data
    population_series = []
    health_series = []
    
    # Simulate population and health trends
    for day in range(duration_days):
        # Base population with seasonal variation
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day / 365)
        base_population = 100 * seasonal_factor
        
        # Add some noise and trend
        noise = np.random.normal(0, 5)
        trend = -0.1 * day if day > 120 else 0  # Decline after day 120
        population = max(0, base_population + noise + trend)
        population_series.append(population)
        
        # Health correlates with population but has its own dynamics
        health_base = 0.8 - (day - 90) * 0.002 if day > 90 else 0.8
        health_noise = np.random.normal(0, 0.05)
        health = max(0.1, min(1.0, health_base + health_noise))
        health_series.append(health)
    
    # Create colony data
    colonies_data = []
    
    # Colony 1: Successful
    colonies_data.append({
        "id": 1,
        "species": "bombus_terrestris",
        "status": "active",
        "initial_population": 80,
        "final_population": 95,
        "max_population": 120,
        "lifespan": duration_days,
        "energy_collected": 8500.0,
        "avg_foraging_efficiency": 0.82,
        "health_history": [0.85, 0.83, 0.86, 0.84, 0.82, 0.80, 0.78, 0.79, 0.81],
        "avg_temperature": 22.5,
        "avg_resource_scarcity": 0.25,
        "min_health_score": 0.78,
        "total_alerts": 3,
        "stress_events": 2
    })
    
    # Colony 2: Failed (disease outbreak)
    colonies_data.append({
        "id": 2,
        "species": "apis_mellifera",
        "status": "collapsed",
        "initial_population": 60,
        "final_population": 0,
        "max_population": 75,
        "lifespan": 95,
        "collapse_day": 95,
        "last_healthy_day": 80,
        "energy_collected": 2800.0,
        "avg_foraging_efficiency": 0.45,
        "health_history": [0.75, 0.73, 0.68, 0.60, 0.52, 0.38, 0.25, 0.15, 0.05],
        "failure_mode": "disease_outbreak",
        "final_disease_prevalence": 0.35,
        "avg_temperature": 19.8,
        "avg_resource_scarcity": 0.45,
        "min_health_score": 0.05,
        "total_alerts": 15,
        "first_alert_day": 65,
        "max_mortality_rate": 0.08,
        "stress_events": 8
    })
    
    # Colony 3: Partial success
    colonies_data.append({
        "id": 3,
        "species": "bombus_lucorum",
        "status": "active",
        "initial_population": 50,
        "final_population": 35,
        "max_population": 65,
        "lifespan": duration_days,
        "energy_collected": 4200.0,
        "avg_foraging_efficiency": 0.58,
        "health_history": [0.70, 0.68, 0.65, 0.62, 0.58, 0.55, 0.52, 0.50, 0.48],
        "avg_temperature": 21.2,
        "avg_resource_scarcity": 0.55,
        "min_health_score": 0.48,
        "total_alerts": 8,
        "stress_events": 5,
        "min_energy_per_bee": 35.0
    })
    
    # Weather history
    weather_history = []
    for day in range(duration_days):
        temp = 20.0 + 5 * np.sin(2 * np.pi * day / 365) + np.random.normal(0, 2)
        weather_history.append({
            "day": day,
            "temperature": temp,
            "precipitation": max(0, np.random.normal(0, 2)),
            "wind_speed": max(0, np.random.normal(5, 2))
        })
    
    return {
        "simulation_id": "test_simulation_001",
        "current_day": duration_days,
        "duration_days": duration_days,
        "start_date": start_date.isoformat(),
        "time_series": {
            "population": population_series,
            "health_scores": health_series,
            "foraging_efficiency": [0.7 + 0.1 * np.sin(i/10) + np.random.normal(0, 0.05) for i in range(duration_days)],
            "energy_collected": [50 + 20 * np.random.random() for _ in range(duration_days)]
        },
        "colonies": colonies_data,
        "weather_history": weather_history,
        "total_health_alerts": 26,
        "critical_health_alerts": 8,
        "total_energy_collected": 15500.0,
        "resource_utilization": 0.68,
        "simulation_time": 45.2,
        "memory_usage": 128.5,
        "foraging_analytics": {
            "total_energy_collected": 15500.0,
            "avg_efficiency": 0.62,
            "total_sessions": 450
        }
    }

def test_post_mortem_analysis():
    """Test the post-mortem analysis system"""
    
    print("=== Post-Mortem Analysis System Test ===\n")
    
    # Test 1: Create mock simulation data
    print("1. Creating mock simulation data...")
    
    mock_data = create_mock_simulation_data()
    
    print(f"   - Simulation duration: {mock_data['duration_days']} days")
    print(f"   - Colonies analyzed: {len(mock_data['colonies'])}")
    print(f"   - Time series metrics: {len(mock_data['time_series'])}")
    print(f"   - Weather data points: {len(mock_data['weather_history'])}")
    print()
    
    # Test 2: Initialize post-mortem analyzer
    print("2. Initializing post-mortem analyzer...")
    
    analyzer = PostMortemAnalyzer(
        analysis_type=AnalysisType.FULL_SIMULATION,
        include_predictions=True,
        generate_recommendations=True,
        detailed_failure_analysis=True
    )
    
    print(f"   - Analysis type: {analyzer.analysis_type.value}")
    print(f"   - Include predictions: {analyzer.include_predictions}")
    print(f"   - Generate recommendations: {analyzer.generate_recommendations}")
    print(f"   - Detailed failure analysis: {analyzer.detailed_failure_analysis}")
    print()
    
    # Test 3: Perform comprehensive analysis
    print("3. Performing comprehensive analysis...")
    
    additional_data = {
        "health_monitoring": {"total_alerts": 26, "critical_alerts": 8},
        "foraging_analytics": mock_data["foraging_analytics"]
    }
    
    analysis_results = analyzer.analyze_simulation(mock_data, additional_data)
    
    print(f"   - Analysis ID: {analysis_results['analysis_id']}")
    print(f"   - Analysis duration: {analysis_results.get('analysis_duration', 0):.2f} seconds")
    print(f"   - Status: {'✓ SUCCESS' if 'error' not in analysis_results else '✗ FAILED'}")
    print()
    
    # Test 4: Examine simulation metrics
    print("4. Examining simulation metrics...")
    
    sim_metrics = analysis_results.get("simulation_metrics")
    if sim_metrics:
        print(f"   - Survival rate: {sim_metrics.survival_rate:.1%}")
        print(f"   - Population growth rate: {sim_metrics.population_growth_rate:.3f}")
        print(f"   - Average health score: {sim_metrics.avg_health_score:.2f}")
        print(f"   - Foraging efficiency: {sim_metrics.avg_foraging_efficiency:.2f}")
        print(f"   - Weather stress days: {sim_metrics.weather_stress_days}")
        print(f"   - Performance: {sim_metrics.steps_per_second:.2f} steps/sec")
    print()
    
    # Test 5: Analyze colony outcomes
    print("5. Analyzing colony outcomes...")
    
    colony_outcomes = analysis_results.get("colony_outcomes", [])
    
    for outcome in colony_outcomes:
        print(f"   Colony {outcome.colony_id} ({outcome.species}):")
        print(f"     - Outcome: {outcome.outcome.value}")
        print(f"     - Lifespan: {outcome.lifespan_days} days")
        print(f"     - Final population: {outcome.final_population}")
        print(f"     - Health score: {outcome.initial_health_score:.2f} → {outcome.final_health_score:.2f}")
        
        if outcome.failure_mode:
            print(f"     - Failure mode: {outcome.failure_mode.value}")
        
        if outcome.primary_cause:
            print(f"     - Primary cause: {outcome.primary_cause}")
        
        if outcome.warning_signs:
            print(f"     - Warning signs: {len(outcome.warning_signs)} detected")
        
        print()
    
    # Test 6: Review trend analysis
    print("6. Reviewing trend analysis...")
    
    trend_analyses = analysis_results.get("trend_analyses", {})
    
    for metric_name, trend in trend_analyses.items():
        print(f"   {metric_name}:")
        print(f"     - Trend: {trend.trend_direction}")
        print(f"     - Slope: {trend.slope:.4f}")
        print(f"     - R²: {trend.r_squared:.3f}")
        print(f"     - Significance: {trend.significance:.3f}")
        
        if trend.change_points:
            print(f"     - Change points: {len(trend.change_points)} detected")
        
        if trend.predicted_trend:
            print(f"     - Prediction: {trend.predicted_trend[-1]:.2f} (next 5 steps)")
        
        print()
    
    # Test 7: Examine performance insights
    print("7. Examining performance insights...")
    
    insights = analysis_results.get("performance_insights", [])
    
    print(f"   Total insights generated: {len(insights)}")
    
    for insight in insights:
        print(f"\n   {insight.title} ({insight.priority.upper()} Priority):")
        print(f"     Category: {insight.category}")
        print(f"     Description: {insight.description}")
        print(f"     Expected impact: {insight.expected_impact}")
        print(f"     Implementation difficulty: {insight.implementation_difficulty}")
        
        if insight.recommendations:
            print(f"     Top recommendation: {insight.recommendations[0]}")
    
    print()
    
    # Test 8: Review failure analysis
    print("8. Reviewing failure analysis...")
    
    failure_analysis = analysis_results.get("failure_analysis", {})
    
    if failure_analysis:
        print(f"   Total failures: {failure_analysis.get('total_failures', 0)}")
        
        failure_modes = failure_analysis.get("failure_modes", {})
        if failure_modes:
            print("   Failure modes:")
            for mode, count in failure_modes.items():
                print(f"     - {mode}: {count}")
        
        common_patterns = failure_analysis.get("common_patterns", [])
        if common_patterns:
            print("   Common patterns:")
            for pattern in common_patterns:
                print(f"     - {pattern}")
        
        prevention_strategies = failure_analysis.get("prevention_strategies", [])
        if prevention_strategies:
            print("   Prevention strategies:")
            for strategy in prevention_strategies[:3]:  # Show first 3
                print(f"     - {strategy}")
    
    print()
    
    # Test 9: Executive summary
    print("9. Executive summary...")
    
    exec_summary = analysis_results.get("executive_summary", {})
    
    if exec_summary:
        print(f"   Performance grade: {exec_summary.get('performance_grade', 'N/A')}")
        print(f"   Simulation duration: {exec_summary.get('simulation_duration', 'N/A')}")
        print(f"   Critical issues: {exec_summary.get('critical_issues', 0)}")
        print(f"   Analysis confidence: {exec_summary.get('analysis_confidence', 'Unknown')}")
        
        key_findings = exec_summary.get("key_findings", [])
        if key_findings:
            print("   Key findings:")
            for finding in key_findings[:3]:  # Show first 3
                print(f"     - {finding}")
        
        top_recommendations = exec_summary.get("top_recommendations", [])
        if top_recommendations:
            print("   Top recommendations:")
            for rec in top_recommendations[:3]:  # Show first 3
                print(f"     - {rec}")
    
    print()
    
    # Test 10: Generate reports in different formats
    print("10. Generating reports in different formats...")
    
    # JSON report
    json_report = generate_comprehensive_report(analysis_results, "json")
    print(f"   - JSON report: {len(str(json_report))} characters")
    
    # Markdown report
    markdown_report = generate_comprehensive_report(analysis_results, "markdown")
    print(f"   - Markdown report: {len(markdown_report)} characters")
    
    # HTML report
    html_report = generate_comprehensive_report(analysis_results, "html")
    print(f"   - HTML report: {len(html_report)} characters")
    
    # Save reports to temporary files
    temp_dir = tempfile.mkdtemp()
    
    # Save JSON
    json_file = os.path.join(temp_dir, "analysis_report.json")
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    
    # Save Markdown
    md_file = os.path.join(temp_dir, "analysis_report.md")
    with open(md_file, 'w') as f:
        f.write(markdown_report)
    
    # Save HTML
    html_file = os.path.join(temp_dir, "analysis_report.html")
    with open(html_file, 'w') as f:
        f.write(html_report)
    
    print(f"   - Reports saved to: {temp_dir}")
    print()
    
    # Test 11: Test different analysis types
    print("11. Testing different analysis types...")
    
    # Failure analysis
    failure_analyzer = PostMortemAnalyzer(analysis_type=AnalysisType.FAILURE_ANALYSIS)
    failure_results = failure_analyzer.analyze_simulation(mock_data)
    print(f"   - Failure analysis completed: {'✓' if 'error' not in failure_results else '✗'}")
    
    # Trend analysis
    trend_analyzer = PostMortemAnalyzer(analysis_type=AnalysisType.TREND_ANALYSIS)
    trend_results = trend_analyzer.analyze_simulation(mock_data)
    print(f"   - Trend analysis completed: {'✓' if 'error' not in trend_results else '✗'}")
    
    print()
    
    # Summary
    print("=== Test Summary ===")
    
    test_results = {
        "Mock data creation": True,
        "Analyzer initialization": True,
        "Comprehensive analysis": "error" not in analysis_results,
        "Simulation metrics extraction": sim_metrics is not None,
        "Colony outcome analysis": len(colony_outcomes) > 0,
        "Trend analysis": len(trend_analyses) > 0,
        "Performance insights": len(insights) > 0,
        "Failure analysis": bool(failure_analysis),
        "Executive summary": bool(exec_summary),
        "Report generation": all([json_report, markdown_report, html_report]),
        "Multiple analysis types": all([
            "error" not in failure_results,
            "error" not in trend_results
        ])
    }
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All post-mortem analysis tests passed!")
        print("✅ Post-mortem analysis system is fully functional")
    else:
        print("⚠️  Some tests failed - check implementation")
    
    # Final statistics
    print("\nFinal Statistics:")
    print(f"   Analysis duration: {analysis_results.get('analysis_duration', 0):.2f} seconds")
    print(f"   Colonies analyzed: {len(colony_outcomes)}")
    print(f"   Insights generated: {len(insights)}")
    print(f"   Trends analyzed: {len(trend_analyses)}")
    print("   Report formats: 3 (JSON, Markdown, HTML)")
    
    return temp_dir

if __name__ == "__main__":
    temp_dir = test_post_mortem_analysis()
    print(f"\nGenerated reports available at: {temp_dir}")
    print("You can open the HTML report in a web browser to view the formatted analysis.")