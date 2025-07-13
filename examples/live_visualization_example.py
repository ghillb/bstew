#!/usr/bin/env python3
"""
Live Visualization Example for BSTEW
====================================

Demonstrates how to use the live visualization system to create
real-time dashboards and charts for bee simulation data.
"""

import time
from pathlib import Path
from typing import Dict, Any

def create_sample_config() -> Dict[str, Any]:
    """Create sample configuration with visualization enabled"""
    
    return {
        "simulation": {
            "duration_days": 30,
            "random_seed": 42
        },
        "colony": {
            "initial_population": {
                "queens": 1,
                "workers": 8000,
                "foragers": 1500,
                "drones": 500,
                "brood": 2000
            },
            "species": "apis_mellifera",
            "location": [0, 0],
            "colony_strength": 0.8,
            "genetic_diversity": 0.7
        },
        "environment": {
            "landscape_width": 100,
            "landscape_height": 100,
            "cell_size": 20.0,
            "weather_variation": 0.1,
            "seasonal_effects": True
        },
        "foraging": {
            "max_foraging_range": 800.0,
            "dance_threshold": 0.7,
            "recruitment_efficiency": 0.8,
            "energy_cost_per_meter": 0.01
        },
        "disease": {
            "enable_varroa": False,
            "enable_viruses": False,
            "enable_nosema": False,
            "natural_resistance": 0.8
        },
        "output": {
            "output_directory": "live_viz_demo",
            "log_level": "INFO",
            "save_plots": True,
            "save_csv": True,
            "save_spatial_data": True,
            "compress_output": False
        }
    }

def run_live_visualization_demo():
    """Run live visualization demonstration"""
    
    print("🎯 BSTEW Live Visualization Demo")
    print("=" * 40)
    
    try:
        from bstew.core.model import BeeModel
        
        # Create configuration
        print("📋 Creating simulation configuration...")
        config = create_sample_config()
        
        # Initialize model
        print("🏗️  Initializing simulation model...")
        model = BeeModel(config=config)
        
        # Start live visualization
        print("📊 Starting live visualization system...")
        model.start_live_visualization()
        
        print("\n🚀 Running simulation with live visualization...")
        print("   This will generate real-time charts and dashboards")
        print("   Check the 'live_viz_demo' directory for output files")
        
        # Run simulation for a few days
        simulation_days = 5
        for day in range(simulation_days):
            print(f"\n📅 Day {day + 1}/{simulation_days}")
            
            # Run one simulation day
            model.step()
            
            # Get current metrics
            summary = model.get_simulation_summary()
            
            print(f"   Total Bees: {summary['colony_summary']['total_bees']}")
            print(f"   Total Honey: {summary['colony_summary']['total_honey']:.2f}")
            print(f"   Average Health: {summary['health_summary']['average_health']:.3f}")
            
            # Display visualization metrics
            if model.live_visualization_manager:
                viz_snapshot = model.live_visualization_manager.engine.export_data_snapshot()
                active_streams = sum(1 for s in viz_snapshot['streams'].values() if s['is_active'])
                total_data_points = sum(s['data_points'] for s in viz_snapshot['streams'].values())
                
                print(f"   Active Streams: {active_streams}")
                print(f"   Total Data Points: {total_data_points}")
            
            # Brief pause to observe live updates
            time.sleep(3)
        
        print("\n📈 Generating final dashboard...")
        
        # Generate a comprehensive dashboard
        if model.live_visualization_manager:
            # Collect final data snapshot
            final_snapshot = {}
            
            # Colony data
            colony_data = {
                'population_data': {
                    'timestamps': list(range(simulation_days + 1)),
                    'total_bees': [summary['colony_summary']['total_bees']] * (simulation_days + 1),
                    'foragers': [summary['colony_summary']['total_bees'] // 3] * (simulation_days + 1)
                },
                'resource_data': {
                    'timestamps': list(range(simulation_days + 1)),
                    'honey_stores': [summary['colony_summary']['total_honey']] * (simulation_days + 1)
                },
                'health_data': {
                    'timestamps': list(range(simulation_days + 1)),
                    'health_score': [summary['health_summary']['average_health']] * (simulation_days + 1)
                },
                'activity_data': {
                    'activities': {
                        'Foraging': 30,
                        'Nursing': 25,
                        'Building': 20,
                        'Guarding': 15,
                        'Resting': 10
                    }
                }
            }
            
            final_snapshot['colony_data'] = colony_data
            
            # Spatial data
            if 'spatial' in summary:
                summary['spatial'].get('spatial_statistics', {})
                final_snapshot['spatial_data'] = {
                    'patches': [
                        {'id': 1, 'x': 50, 'y': 50, 'quality': 0.8},
                        {'id': 2, 'x': 80, 'y': 30, 'quality': 0.6},
                        {'id': 3, 'x': 30, 'y': 80, 'quality': 0.9}
                    ],
                    'colonies': [
                        {'id': 1, 'x': 0, 'y': 0, 'population': summary['colony_summary']['total_bees']}
                    ],
                    'bees': []  # Would be populated with individual bee positions
                }
            
            # Communication data
            if 'dance_communication' in summary:
                final_snapshot['communication_data'] = {
                    'dances': [],  # Would be populated with active dances
                    'information_flows': []  # Would be populated with information flows
                }
            
            # Generate dashboard
            dashboard_file = model.live_visualization_manager.engine.generate_dashboard(final_snapshot)
            
            if dashboard_file:
                print(f"   ✅ Dashboard saved to: {dashboard_file}")
            else:
                print("   ⚠️  Dashboard generation failed (visualization dependencies may be missing)")
        
        # Stop visualization
        print("\n🛑 Stopping live visualization...")
        model.stop_live_visualization()
        
        # Show output directory contents
        output_dir = Path(config['output']['output_directory'])
        if output_dir.exists():
            output_files = list(output_dir.glob("*"))
            print(f"\n📁 Generated files in {output_dir}:")
            for file in output_files:
                print(f"   • {file.name}")
        
        # Cleanup model
        model.cleanup()
        
        print("\n✅ Live visualization demo completed successfully!")
        print("\n📋 Summary:")
        print(f"   • Simulated {simulation_days} days")
        print("   • Generated real-time visualizations")
        print("   • Created dashboard and charts")
        print(f"   • Output directory: {config['output']['output_directory']}")
        
        # Optional: Clean up demo files
        cleanup = input("\n🧹 Clean up demo files? (y/N): ").lower().strip()
        if cleanup == 'y':
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"   ✅ Cleaned up {output_dir}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Live visualization requires the core BSTEW package")
        print("   Make sure you're running from the correct directory")
        
    except Exception as e:
        print(f"❌ Error during live visualization demo: {e}")
        import traceback
        traceback.print_exc()

def show_visualization_features():
    """Show available visualization features"""
    
    print("📊 BSTEW Live Visualization Features")
    print("=" * 40)
    
    features = [
        ("Real-time Colony Overview", "Population, health, resources, activity charts"),
        ("Spatial Distribution Maps", "Bee positions, patch locations, movement patterns"),
        ("Communication Networks", "Dance communications, recruitment flows"),
        ("Time Series Analysis", "Population trends, foraging efficiency over time"),
        ("Health Monitoring", "Colony health metrics, disease tracking"),
        ("Resource Dynamics", "Honey stores, foraging success rates"),
        ("Interactive Dashboards", "HTML dashboards with multiple charts"),
        ("Data Export", "JSON snapshots, CSV exports for further analysis"),
        ("Configurable Updates", "Adjustable refresh rates and data collection"),
        ("Multiple Formats", "HTML, PNG, SVG output formats")
    ]
    
    for feature, description in features:
        print(f"✓ {feature}")
        print(f"  {description}")
        print()
    
    print("🔧 Configuration Options:")
    print("   visualization.enabled: Enable/disable visualization")
    print("   visualization.output_directory: Output folder for files")
    print("   visualization.refresh_rate: Update frequency (seconds)")
    print("   visualization.auto_save_interval: Auto-save frequency")
    print()
    
    print("📦 Optional Dependencies:")
    print("   pip install plotly matplotlib numpy")
    print("   (for full visualization functionality)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--features":
        show_visualization_features()
    else:
        run_live_visualization_demo()