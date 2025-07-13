#!/usr/bin/env python3
"""
Interactive Analysis Tools Example for BSTEW
===========================================

Demonstrates how to use the interactive analysis tools to explore
bee simulation data, generate insights, and create statistical reports.
"""

import time
import random
from pathlib import Path
from typing import Dict, Any, List

def generate_sample_simulation_data(days: int = 30, colonies: int = 2) -> List[Dict[str, Any]]:
    """Generate sample simulation data for demonstration"""
    
    data = []
    
    for day in range(days):
        timestamp = time.time() - (days - day) * 86400  # Days ago
        
        for colony_id in range(1, colonies + 1):
            # Base population with some variation
            base_population = 8000 + random.randint(-1000, 2000)
            
            # Simulate seasonal effects
            seasonal_factor = 0.8 + 0.4 * (day / days)
            population = int(base_population * seasonal_factor)
            
            # Foraging metrics
            foraging_success_rate = random.uniform(0.4, 0.9)
            average_energy_gained = random.uniform(5.0, 25.0)
            time_efficiency = random.uniform(0.3, 0.8)
            
            # Health metrics
            health_score = random.uniform(0.6, 1.0)
            
            # Communication metrics
            dance_count = random.randint(0, 15)
            recruitment_count = int(dance_count * random.uniform(0.2, 0.8))
            
            # Colony data
            colony_data = {
                "timestamp": timestamp,
                "day": day,
                "colony_id": colony_id,
                "population": population,
                "forager_count": int(population * random.uniform(0.2, 0.4)),
                "health_score": health_score,
                "foraging_success_rate": foraging_success_rate,
                "average_energy_gained": average_energy_gained,
                "time_efficiency": time_efficiency,
                "average_distance_traveled": random.uniform(50, 300),
                "honey_stores": random.uniform(100, 2000),
                "dance_count": dance_count,
                "recruitment_count": recruitment_count
            }
            
            data.append(colony_data)
            
            # Add some communication events
            for dance_id in range(dance_count):
                comm_data = {
                    "timestamp": timestamp + random.uniform(0, 86400),
                    "event_type": "dance",
                    "colony_id": colony_id,
                    "sender_id": random.randint(1, 100),
                    "receiver_id": random.randint(1, 100),
                    "information_quality": random.uniform(0.3, 1.0),
                    "patch_id": random.randint(1, 10),
                    "dance_intensity": random.uniform(0.2, 1.0)
                }
                data.append(comm_data)
            
            # Add recruitment events
            for recruit_id in range(recruitment_count):
                recruit_data = {
                    "timestamp": timestamp + random.uniform(0, 86400),
                    "event_type": "recruitment",
                    "colony_id": colony_id,
                    "sender_id": random.randint(1, 100),
                    "receiver_id": random.randint(1, 100),
                    "recruitment_success": random.choice([True, False]),
                    "information_quality": random.uniform(0.4, 0.9)
                }
                data.append(recruit_data)
    
    return data

def run_interactive_analysis_demo():
    """Run interactive analysis demonstration"""
    
    print("🔬 BSTEW Interactive Analysis Demo")
    print("=" * 40)
    
    try:
        from bstew.analysis.interactive_tools import (
            create_interactive_analysis_engine,
            DataFilter
        )
        
        # Create analysis engine
        print("🔧 Creating interactive analysis engine...")
        analysis_engine = create_interactive_analysis_engine("analysis_demo")
        
        # Generate sample data
        print("📊 Generating sample simulation data...")
        simulation_data = generate_sample_simulation_data(days=30, colonies=3)
        
        print(f"   Generated {len(simulation_data)} data points")
        print(f"   Time range: {30} days")
        print("   Colonies: 3")
        
        # Separate data by type
        colony_data = [item for item in simulation_data if "population" in item]
        communication_data = [item for item in simulation_data if "event_type" in item]
        
        print(f"   Colony records: {len(colony_data)}")
        print(f"   Communication records: {len(communication_data)}")
        
        print("\n📈 Running Population Dynamics Analysis...")
        
        # Population dynamics analysis
        pop_result = analysis_engine.analyze_population_dynamics(colony_data)
        
        print("   Results:")
        print(f"   • Sample size: {pop_result.data_summary.get('sample_size', 0)}")
        print(f"   • Mean population: {pop_result.data_summary.get('mean_population', 0):.0f}")
        print(f"   • Population std: {pop_result.data_summary.get('std_population', 0):.0f}")
        
        if pop_result.statistical_results.get('trend_analysis'):
            trend = pop_result.statistical_results['trend_analysis']
            print(f"   • Trend direction: {trend.get('trend_direction', 'unknown')}")
            print(f"   • Trend strength: {trend.get('trend_strength', 'unknown')}")
        
        for insight in pop_result.insights:
            print(f"   💡 {insight}")
        
        print("\n⚡ Running Foraging Efficiency Analysis...")
        
        # Foraging efficiency analysis
        foraging_result = analysis_engine.analyze_foraging_efficiency(colony_data)
        
        print("   Results:")
        success_rate_data = foraging_result.data_summary.get('foraging_success_rate', {})
        energy_data = foraging_result.data_summary.get('average_energy_gained', {})
        
        if success_rate_data:
            print(f"   • Mean success rate: {success_rate_data.get('mean', 0):.3f}")
            print(f"   • Success rate range: {success_rate_data.get('min', 0):.3f} - {success_rate_data.get('max', 0):.3f}")
        
        if energy_data:
            print(f"   • Mean energy gained: {energy_data.get('mean', 0):.2f}")
        
        for insight in foraging_result.insights:
            print(f"   💡 {insight}")
        
        print("\n🗣️  Running Communication Patterns Analysis...")
        
        # Communication analysis
        comm_result = analysis_engine.analyze_communication_patterns(communication_data)
        
        print("   Results:")
        print(f"   • Total dances: {comm_result.data_summary.get('total_dances', 0)}")
        print(f"   • Total recruitments: {comm_result.data_summary.get('total_recruitments', 0)}")
        print(f"   • Recruitment rate: {comm_result.data_summary.get('recruitment_rate', 0):.3f}")
        
        for insight in comm_result.insights:
            print(f"   💡 {insight}")
        
        print("\n🔍 Running Filtered Analysis...")
        
        # Filtered analysis example
        filter_config = DataFilter(
            colony_ids=[1, 2],  # Only colonies 1 and 2
            health_threshold=0.7  # Only high health periods
        )
        
        filtered_result = analysis_engine.analyze_population_dynamics(colony_data, filter_config)
        
        print("   Filtered Results (High Health Periods Only):")
        print(f"   • Filtered sample size: {filtered_result.data_summary.get('sample_size', 0)}")
        print(f"   • Filtered mean population: {filtered_result.data_summary.get('mean_population', 0):.0f}")
        
        print("\n📊 Running Comparative Analysis...")
        
        # Comparative analysis between colonies
        colony_datasets = {}
        for colony_id in [1, 2, 3]:
            colony_datasets[f"Colony_{colony_id}"] = [
                item for item in colony_data if item.get("colony_id") == colony_id
            ]
        
        comp_result = analysis_engine.perform_comparative_analysis(
            colony_datasets, "foraging_success_rate"
        )
        
        print("   Comparative Results (Foraging Success Rate):")
        for dataset_name, data in comp_result.data_summary.items():
            print(f"   • {dataset_name}: {data.get('mean', 0):.3f} ± {data.get('std', 0):.3f}")
        
        for insight in comp_result.insights:
            print(f"   💡 {insight}")
        
        print("\n📝 Generating Comprehensive Report...")
        
        # Generate report
        report_content = analysis_engine.generate_comprehensive_report(
            "BSTEW Interactive Analysis Demo Report"
        )
        
        print("   ✅ Report generated")
        print("   📄 Report preview (first 500 characters):")
        print("   " + "─" * 50)
        print("   " + report_content[:500].replace("\n", "\n   ") + "...")
        print("   " + "─" * 50)
        
        print("\n💾 Exporting Analysis Results...")
        
        # Export results
        export_file = analysis_engine.export_analysis_results("json")
        print(f"   ✅ Results exported to: {export_file}")
        
        # Show output directory contents
        output_dir = Path("analysis_demo")
        if output_dir.exists():
            output_files = list(output_dir.glob("*"))
            print(f"\n📁 Generated files in {output_dir}:")
            for file in output_files:
                file_size = file.stat().st_size / 1024  # KB
                print(f"   • {file.name} ({file_size:.1f} KB)")
        
        print("\n📊 Visualization Capabilities:")
        
        if pop_result.visualizations:
            print("   ✅ Population dynamics chart generated")
        else:
            print("   ⚠️  Population chart not generated (plotly not available)")
        
        if foraging_result.visualizations:
            print("   ✅ Foraging efficiency chart generated")
        else:
            print("   ⚠️  Foraging chart not generated (plotly not available)")
        
        if comm_result.visualizations:
            print("   ✅ Communication network graph generated")
        else:
            print("   ⚠️  Communication graph not generated (plotly not available)")
        
        print("\n✅ Interactive analysis demo completed successfully!")
        print("\n📋 Summary:")
        print(f"   • Analyzed {len(simulation_data)} data points")
        print(f"   • Performed {len(analysis_engine.analysis_results)} analyses")
        print("   • Generated comprehensive report")
        print("   • Exported results to JSON")
        print("   • Output directory: analysis_demo/")
        
        # Show statistical capabilities
        print("\n🔬 Statistical Analysis Capabilities:")
        capabilities = [
            "Population trend analysis",
            "Correlation analysis between variables", 
            "Comparative analysis between groups",
            "Time series pattern detection",
            "Distribution analysis",
            "Efficiency metrics calculation",
            "Communication network analysis",
            "Filter-based subset analysis"
        ]
        
        for capability in capabilities:
            print(f"   ✓ {capability}")
        
        # Optional: Clean up demo files
        cleanup = input("\n🧹 Clean up demo files? (y/N): ").lower().strip()
        if cleanup == 'y':
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"   ✅ Cleaned up {output_dir}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Interactive analysis requires the core BSTEW package")
        print("   Make sure you're running from the correct directory")
        
    except Exception as e:
        print(f"❌ Error during interactive analysis demo: {e}")
        import traceback
        traceback.print_exc()

def show_analysis_features():
    """Show available analysis features"""
    
    print("🔬 BSTEW Interactive Analysis Features")
    print("=" * 40)
    
    features = [
        ("Population Dynamics Analysis", "Growth trends, statistics, population health"),
        ("Foraging Efficiency Analysis", "Success rates, energy gain, time efficiency"),
        ("Communication Pattern Analysis", "Dance frequency, recruitment success, network structure"),
        ("Spatial Distribution Analysis", "Movement patterns, patch utilization, territory analysis"),
        ("Health Trend Analysis", "Colony health metrics, disease progression, mortality"),
        ("Resource Utilization Analysis", "Honey stores, foraging patterns, resource depletion"),
        ("Behavioral Pattern Detection", "Activity cycles, state transitions, anomaly detection"),
        ("Comparative Analysis", "Multi-colony comparisons, scenario analysis, A/B testing"),
        ("Correlation Analysis", "Variable relationships, cause-effect identification"),
        ("Statistical Testing", "Trend analysis, distribution fitting, significance testing"),
        ("Interactive Visualizations", "Dynamic charts, network graphs, heatmaps"),
        ("Automated Reporting", "Comprehensive reports, insights generation, recommendations")
    ]
    
    for feature, description in features:
        print(f"✓ {feature}")
        print(f"  {description}")
        print()
    
    print("🛠️  Analysis Tools:")
    print("   • DataFilter: Flexible data filtering and subset selection")
    print("   • StatisticalAnalyzer: Correlation, trend, and distribution analysis")
    print("   • VisualizationGenerator: Interactive charts and graphs")
    print("   • DataProcessor: Data aggregation and transformation")
    print("   • InteractiveAnalysisEngine: Comprehensive analysis workflows")
    print()
    
    print("📊 Output Formats:")
    print("   • Interactive HTML visualizations (Plotly)")
    print("   • Markdown analysis reports")
    print("   • JSON data exports")
    print("   • Statistical summaries")
    print()
    
    print("📦 Optional Dependencies:")
    print("   pip install plotly pandas numpy")
    print("   (for full statistical and visualization functionality)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--features":
        show_analysis_features()
    else:
        run_interactive_analysis_demo()