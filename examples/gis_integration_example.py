#!/usr/bin/env python3
"""
GIS Integration Example for BSTEW
=================================

Demonstrates how to use GIS integration capabilities to load real-world
spatial data and integrate it with bee simulation models.
"""

import tempfile
import json
from pathlib import Path

def create_example_geojson() -> str:
    """Create example GeoJSON file for demonstration"""
    
    # Create sample landscape data
    geojson_data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        },
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.5, 37.7, 0],
                        [-122.4, 37.7, 0],
                        [-122.4, 37.8, 0],
                        [-122.5, 37.8, 0],
                        [-122.5, 37.7, 0]
                    ]]
                },
                "properties": {
                    "land_cover": "forest",
                    "quality": 0.8,
                    "resource_density": 0.7,
                    "area_hectares": 100.5
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.4, 37.7, 0],
                        [-122.3, 37.7, 0],
                        [-122.3, 37.8, 0],
                        [-122.4, 37.8, 0],
                        [-122.4, 37.7, 0]
                    ]]
                },
                "properties": {
                    "land_cover": "agricultural",
                    "quality": 0.9,
                    "resource_density": 0.9,
                    "area_hectares": 150.2
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.45, 37.75, 0]
                },
                "properties": {
                    "name": "Bee Colony Location",
                    "type": "hive",
                    "quality": 1.0,
                    "resource_density": 0.8
                }
            }
        ]
    }
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False)
    json.dump(geojson_data, temp_file, indent=2)
    temp_file.close()
    
    return temp_file.name

def create_example_csv_points() -> str:
    """Create example CSV points file for demonstration"""
    
    csv_content = """latitude,longitude,flower_type,quality,resource_density,notes
37.75,-122.45,lavender,0.8,0.7,Dense lavender field
37.76,-122.44,sunflower,0.9,0.9,Large sunflower patch
37.74,-122.46,wildflower,0.6,0.6,Mixed wildflower meadow
37.77,-122.43,clover,0.7,0.8,White clover field
37.73,-122.47,berry,0.5,0.4,Berry bushes with some flowers
"""
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write(csv_content)
    temp_file.close()
    
    return temp_file.name

def run_gis_integration_example():
    """Run complete GIS integration example"""
    
    print("🌍 BSTEW GIS Integration Example")
    print("=" * 40)
    
    try:
        from bstew.core.gis_integration import create_gis_integration_system, DataFormat
        from bstew.core.spatial_integration import create_spatial_integration_system
        
        # Create example data files
        print("📄 Creating example spatial data files...")
        geojson_file = create_example_geojson()
        csv_file = create_example_csv_points()
        
        print(f"   ✓ Created GeoJSON: {geojson_file}")
        print(f"   ✓ Created CSV: {csv_file}")
        
        # Configure GIS integration
        print("\n🗺️  Initializing GIS integration system...")
        gis_manager = create_gis_integration_system(
            source_crs="EPSG:4326",  # WGS84 (GPS coordinates)
            target_crs="LOCAL"       # Local coordinate system
        )
        print("   ✓ GIS integration system initialized")
        
        # Test coordinate transformation
        print("\n📍 Testing coordinate transformations...")
        test_coords = [(-122.45, 37.75), (-122.40, 37.80)]
        transformed_coords = gis_manager.coordinate_transformer.transform_points(test_coords)
        
        print("   Original coordinates (WGS84):")
        for i, (lon, lat) in enumerate(test_coords):
            print(f"     Point {i+1}: {lon:.6f}, {lat:.6f}")
        
        print("   Transformed coordinates (Local):")
        for i, (x, y) in enumerate(transformed_coords):
            print(f"     Point {i+1}: {x:.2f}, {y:.2f}")
        
        # Load vector data
        print("\n📊 Loading vector data...")
        
        # Load GeoJSON
        geojson_features, geojson_metadata = gis_manager.vector_processor.load_vector_data(
            geojson_file, DataFormat.GEOJSON
        )
        print(f"   ✓ Loaded GeoJSON: {geojson_metadata.feature_count} features")
        print(f"     Geometry type: {geojson_metadata.geometry_type}")
        print(f"     Bounds: {geojson_metadata.bounds}")
        
        # Load CSV points
        csv_features, csv_metadata = gis_manager.vector_processor.load_vector_data(
            csv_file, DataFormat.CSV_POINTS
        )
        print(f"   ✓ Loaded CSV points: {csv_metadata.feature_count} features")
        print(f"     Bounds: {csv_metadata.bounds}")
        
        # Convert to spatial patches
        print("\n🏞️  Converting to spatial patches...")
        geojson_patches = gis_manager.vector_processor.convert_to_spatial_patches(
            geojson_features, 
            quality_field="quality",
            resource_field="resource_density"
        )
        
        csv_patches = gis_manager.vector_processor.convert_to_spatial_patches(
            csv_features,
            quality_field="quality", 
            resource_field="resource_density"
        )
        
        all_patches = geojson_patches + csv_patches
        print(f"   ✓ Created {len(all_patches)} spatial patches")
        
        # Analyze patch properties
        total_area = sum(patch.area for patch in all_patches)
        avg_quality = sum(patch.quality for patch in all_patches) / len(all_patches)
        avg_resource = sum(patch.resource_density for patch in all_patches) / len(all_patches)
        
        print(f"     Total area: {total_area:.2f} hectares")
        print(f"     Average quality: {avg_quality:.3f}")
        print(f"     Average resource density: {avg_resource:.3f}")
        
        # Create landscape configuration for simulation
        print("\n🌱 Creating simulation landscape...")
        landscape_config = {
            'use_gis_data': True,
            'gis_config': {
                'source_crs': 'EPSG:4326',
                'target_crs': 'LOCAL'
            },
            'vector_sources': [
                {
                    'file_path': geojson_file,
                    'quality_field': 'quality',
                    'resource_field': 'resource_density'
                },
                {
                    'file_path': csv_file,
                    'quality_field': 'quality',
                    'resource_field': 'resource_density'
                }
            ]
        }
        
        # Initialize spatial integration system
        spatial_env, spatial_bee_manager = create_spatial_integration_system(landscape_config)
        
        patch_count = len(spatial_env.spatial_index._patches)
        print(f"   ✓ Spatial simulation system initialized with {patch_count} patches")
        
        # Test export functionality
        print("\n💾 Testing data export...")
        output_file = tempfile.NamedTemporaryFile(suffix='.geojson', delete=False)
        output_file.close()
        
        exported_path = spatial_env.export_landscape_to_gis(output_file.name)
        print(f"   ✓ Exported landscape to: {exported_path}")
        
        # Analyze landscape
        print("\n📈 Performing landscape analysis...")
        gis_analysis = spatial_env.get_gis_analysis_results()
        
        if 'spatial_statistics' in gis_analysis:
            stats = gis_analysis['spatial_statistics']
            print(f"   Total patches: {stats['total_patches']}")
            print(f"   Total area: {stats['total_area']:.2f} hectares")
            print(f"   Mean patch area: {stats['mean_patch_area']:.2f} hectares")
            print(f"   Mean quality: {stats['mean_quality']:.3f}")
            print(f"   Quality variance: {stats['quality_variance']:.6f}")
        
        if 'coordinate_system' in gis_analysis:
            cs = gis_analysis['coordinate_system']
            print(f"   Source CRS: {cs['source_crs']}")
            print(f"   Target CRS: {cs['target_crs']}")
        
        print("\n✅ GIS integration example completed successfully!")
        print("\n📋 Summary:")
        print(f"   • Processed {len(geojson_features)} GeoJSON features")
        print(f"   • Processed {len(csv_features)} CSV point features")
        print(f"   • Created {len(all_patches)} spatial patches")
        print("   • Integrated with simulation system")
        print("   • Exported results to GIS format")
        
        # Cleanup
        print("\n🧹 Cleaning up temporary files...")
        Path(geojson_file).unlink()
        Path(csv_file).unlink()
        Path(exported_path).unlink()
        print("   ✓ Temporary files removed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   GIS integration requires optional dependencies:")
        print("   pip install pyproj rasterio geopandas shapely")
        
    except Exception as e:
        print(f"❌ Error during GIS integration example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_gis_integration_example()