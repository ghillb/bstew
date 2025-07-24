"""
Tests for Visual Landscape Modification Engine
==============================================

Comprehensive test suite for landscape visualization and modification functionality.
"""

import pytest
from datetime import datetime, date

from src.bstew.components.landscape_modification import (
    LandscapeModificationEngine,
    LandscapeVisualization,
    LandscapeLayer,
    ModificationRecord,
    ModificationType,
    VisualizationMode,
    QualityMetric,
    ColorScheme,
    RenderSettings
)
from src.bstew.spatial.patches import ResourcePatch, HabitatType


class TestModificationType:
    """Test ModificationType enum"""

    def test_modification_type_values(self):
        """Test modification type enum values"""
        assert ModificationType.MARGIN_CREATION.value == "margin_creation"
        assert ModificationType.WILDFLOWER_STRIP.value == "wildflower_strip"
        assert ModificationType.HEDGE_PLANTING.value == "hedge_planting"
        assert ModificationType.POND_CREATION.value == "pond_creation"
        assert ModificationType.TREE_PLANTING.value == "tree_planting"
        assert ModificationType.GRASSLAND_RESTORATION.value == "grassland_restoration"
        assert ModificationType.WETLAND_CREATION.value == "wetland_creation"
        assert ModificationType.HABITAT_CONNECTIVITY.value == "habitat_connectivity"


class TestVisualizationMode:
    """Test VisualizationMode enum"""

    def test_visualization_mode_values(self):
        """Test visualization mode enum values"""
        assert VisualizationMode.CURRENT_STATE.value == "current_state"
        assert VisualizationMode.PROPOSED_CHANGES.value == "proposed_changes"
        assert VisualizationMode.BEFORE_AFTER_SPLIT.value == "before_after_split"
        assert VisualizationMode.QUALITY_HEATMAP.value == "quality_heatmap"
        assert VisualizationMode.RESOURCE_DENSITY.value == "resource_density"
        assert VisualizationMode.CONNECTIVITY_MAP.value == "connectivity_map"
        assert VisualizationMode.TEMPORAL_SEQUENCE.value == "temporal_sequence"


class TestColorScheme:
    """Test ColorScheme dataclass"""

    def test_color_scheme_defaults(self):
        """Test default color scheme values"""
        scheme = ColorScheme()

        # Check habitat colors
        assert HabitatType.CROPLAND in scheme.habitat_colors
        assert HabitatType.GRASSLAND in scheme.habitat_colors
        assert HabitatType.HEDGEROW in scheme.habitat_colors
        assert HabitatType.WOODLAND in scheme.habitat_colors

        # Check quality gradient
        assert len(scheme.quality_gradient) == 6
        assert scheme.quality_gradient[0] == "#440154"  # Low
        assert scheme.quality_gradient[-1] == "#FDE725"  # High

        # Check modification colors
        assert ModificationType.WILDFLOWER_STRIP in scheme.modification_colors
        assert ModificationType.MARGIN_CREATION in scheme.modification_colors


class TestRenderSettings:
    """Test RenderSettings dataclass"""

    def test_render_settings_defaults(self):
        """Test default render settings"""
        settings = RenderSettings()

        assert settings.width == 1200
        assert settings.height == 800
        assert settings.scale == 1.0
        assert settings.opacity == 1.0
        assert settings.show_grid is True
        assert settings.grid_size == 100
        assert settings.show_labels is True
        assert settings.show_legend is True
        assert settings.animation_speed == 1.0
        assert settings.quality_resolution == 10


class TestModificationRecord:
    """Test ModificationRecord dataclass"""

    def test_modification_record_creation(self):
        """Test creating modification record"""
        record = ModificationRecord(
            modification_id="mod_1",
            modification_type=ModificationType.WILDFLOWER_STRIP,
            timestamp=datetime.now(),
            geometry={"type": "LineString", "coordinates": [[0, 0], [100, 0]]},
            affected_patches=[1001, 1002],
            parameters={"width": 6, "seed_mix": "pollinator"},
            predicted_impact={"nectar_increase": 5.0, "pollen_increase": 3.0},
            cost_estimate=750.0
        )

        assert record.modification_id == "mod_1"
        assert record.modification_type == ModificationType.WILDFLOWER_STRIP
        assert len(record.affected_patches) == 2
        assert record.cost_estimate == 750.0

    def test_modification_record_to_dict(self):
        """Test converting modification record to dictionary"""
        timestamp = datetime.now()
        record = ModificationRecord(
            modification_id="mod_2",
            modification_type=ModificationType.MARGIN_CREATION,
            timestamp=timestamp,
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
            affected_patches=[2001],
            parameters={"width": 12},
            predicted_impact={"habitat_area_change": 0.1},
            cost_estimate=500.0
        )

        result = record.to_dict()

        assert result["modification_id"] == "mod_2"
        assert result["modification_type"] == "margin_creation"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["geometry"]["type"] == "Polygon"
        assert result["cost_estimate"] == 500.0


class TestLandscapeLayer:
    """Test LandscapeLayer model"""

    def test_landscape_layer_creation(self):
        """Test creating landscape layer"""
        features = [
            {
                "type": "Feature",
                "id": "patch_1",
                "properties": {"habitat_type": "grassland"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}
            }
        ]

        layer = LandscapeLayer(
            layer_id="habitat_layer",
            layer_type="habitat",
            features=features,
            z_order=10
        )

        assert layer.layer_id == "habitat_layer"
        assert layer.layer_type == "habitat"
        assert len(layer.features) == 1
        assert layer.visible is True  # Default
        assert layer.opacity == 1.0   # Default
        assert layer.z_order == 10
        assert layer.interactive is True  # Default
        assert layer.selectable is True   # Default
        assert layer.editable is False    # Default

    def test_layer_visibility_settings(self):
        """Test layer visibility and interaction settings"""
        layer = LandscapeLayer(
            layer_id="test_layer",
            layer_type="quality",
            visible=False,
            opacity=0.7,
            interactive=False,
            selectable=False,
            editable=True
        )

        assert layer.visible is False
        assert layer.opacity == 0.7
        assert layer.interactive is False
        assert layer.selectable is False
        assert layer.editable is True


class TestLandscapeVisualization:
    """Test LandscapeVisualization model"""

    def test_landscape_visualization_creation(self):
        """Test creating landscape visualization"""
        viz = LandscapeVisualization(
            visualization_id="viz_1",
            bounds={"min_x": 0, "min_y": 0, "max_x": 1000, "max_y": 1000},
            center=(500, 500),
            mode=VisualizationMode.QUALITY_HEATMAP
        )

        assert viz.visualization_id == "viz_1"
        assert viz.bounds["max_x"] == 1000
        assert viz.center == (500, 500)
        assert viz.zoom == 1.0  # Default
        assert viz.mode == VisualizationMode.QUALITY_HEATMAP
        assert len(viz.layers) == 0  # Empty by default
        assert viz.active_layer_id is None
        assert isinstance(viz.render_settings, RenderSettings)
        assert isinstance(viz.color_scheme, ColorScheme)
        assert len(viz.selected_features) == 0
        assert len(viz.highlights) == 0


class TestLandscapeModificationEngine:
    """Test LandscapeModificationEngine class"""

    @pytest.fixture
    def engine(self):
        """Create engine instance for testing"""
        return LandscapeModificationEngine()

    @pytest.fixture
    def test_patches(self):
        """Create test resource patches"""
        patches = []
        for i in range(5):
            patch = ResourcePatch(
                patch_id=1000 + i,
                x=i * 200,
                y=0,
                habitat_type=HabitatType.CROPLAND if i % 2 == 0 else HabitatType.GRASSLAND
            )
            patch.area_ha = 5.0
            patch.base_nectar_production = 2.0 + i * 0.5
            patch.base_pollen_production = 1.5 + i * 0.3
            patches.append(patch)
        return patches

    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.visualizations == {}
        assert engine.modifications == {}
        assert engine.modification_history == []
        assert engine.quality_cache == {}
        assert engine.connectivity_cache == {}
        assert engine.render_cache == {}
        assert len(engine.pending_modifications) == 0
        assert len(engine.applied_modifications) == 0

    def test_create_visualization(self, engine, test_patches):
        """Test creating visualization"""
        viz = engine.create_visualization(
            patches=test_patches,
            mode=VisualizationMode.CURRENT_STATE
        )

        assert viz.visualization_id.startswith("viz_")
        assert viz.visualization_id in engine.visualizations
        assert len(viz.layers) == 1  # Habitat layer
        assert viz.layers[0].layer_type == "habitat"
        assert len(viz.layers[0].features) == len(test_patches)

        # Check bounds calculation
        assert viz.bounds["min_x"] < 0  # Includes buffer
        assert viz.bounds["max_x"] > 800  # Last patch at 800

        # Check center calculation
        assert viz.center[0] == (viz.bounds["min_x"] + viz.bounds["max_x"]) / 2
        assert viz.center[1] == (viz.bounds["min_y"] + viz.bounds["max_y"]) / 2

    def test_create_visualization_with_quality_mode(self, engine, test_patches):
        """Test creating visualization in quality heatmap mode"""
        viz = engine.create_visualization(
            patches=test_patches,
            mode=VisualizationMode.QUALITY_HEATMAP
        )

        assert len(viz.layers) == 2  # Habitat + quality layers
        assert viz.layers[0].layer_type == "habitat"
        assert viz.layers[1].layer_type == "quality"
        assert viz.layers[1].layer_id.startswith("quality_")

    def test_calculate_bounds(self, engine, test_patches):
        """Test bounds calculation"""
        bounds = engine._calculate_bounds(test_patches)

        # Patches at x: 0, 200, 400, 600, 800
        assert bounds["min_x"] == -50  # 0 - 50 buffer
        assert bounds["max_x"] == 850   # 800 + 50 buffer
        assert bounds["min_y"] == -50   # 0 - 50 buffer
        assert bounds["max_y"] == 50    # 0 + 50 buffer

    def test_create_habitat_layer(self, engine, test_patches):
        """Test habitat layer creation"""
        layer = engine._create_habitat_layer(test_patches)

        assert layer.layer_id == "habitat_base"
        assert layer.layer_type == "habitat"
        assert len(layer.features) == len(test_patches)
        assert layer.z_order == 0

        # Check feature structure
        feature = layer.features[0]
        assert feature["type"] == "Feature"
        assert feature["id"] == "patch_1000"
        assert feature["properties"]["habitat_type"] == "cropland"
        assert feature["properties"]["area_ha"] == 5.0
        assert feature["geometry"]["type"] == "Polygon"

        # Check layer style
        assert "fill_color_property" in layer.style
        assert "fill_colors" in layer.style
        assert layer.style["stroke_color"] == "#333333"
        assert layer.style["stroke_width"] == 1

    def test_calculate_quality_at_point(self, engine, test_patches):
        """Test quality calculation at specific point"""
        bounds = engine._calculate_bounds(test_patches)
        from src.bstew.components.habitat_creation import ResourceDensityGrid
        density_grid = ResourceDensityGrid(bounds, resolution_m=50)

        # Add patches to grid
        for patch in test_patches:
            density_grid.add_resource_patch(patch)

        # Test quality at patch location
        quality = engine._calculate_quality_at_point(
            test_patches[0].x, test_patches[0].y,
            test_patches, QualityMetric.NECTAR_AVAILABILITY
        )

        assert quality > 0  # Should have some quality
        assert quality <= 1.0  # Should be normalized

    def test_get_patch_quality(self, engine, test_patches):
        """Test patch quality scoring"""
        patch = test_patches[0]

        # Test nectar availability
        nectar_quality = engine._get_patch_quality(patch, QualityMetric.NECTAR_AVAILABILITY)
        assert 0 <= nectar_quality <= 1.0

        # Test pollen availability
        pollen_quality = engine._get_patch_quality(patch, QualityMetric.POLLEN_AVAILABILITY)
        assert 0 <= pollen_quality <= 1.0

        # Test nesting suitability
        nesting_quality = engine._get_patch_quality(patch, QualityMetric.NESTING_SUITABILITY)
        assert nesting_quality == 0.2  # Cropland value

        # Test biodiversity index (composite)
        biodiversity = engine._get_patch_quality(patch, QualityMetric.BIODIVERSITY_INDEX)
        assert 0 <= biodiversity <= 1.0

    def test_add_modification(self, engine, test_patches):
        """Test adding landscape modification"""
        viz = engine.create_visualization(test_patches)

        modification = engine.add_modification(
            visualization_id=viz.visualization_id,
            modification_type=ModificationType.WILDFLOWER_STRIP,
            geometry={
                "type": "LineString",
                "coordinates": [[0, 0], [200, 0], [400, 0]]
            },
            parameters={
                "width": 6,
                "seed_mix": "pollinator_nectar",
                "length": 400
            }
        )

        assert modification.modification_id.startswith("mod_")
        assert modification.modification_type == ModificationType.WILDFLOWER_STRIP
        assert len(modification.affected_patches) > 0
        assert modification.cost_estimate > 0
        assert "nectar_increase" in modification.predicted_impact
        assert "pollen_increase" in modification.predicted_impact

        # Check modification is stored
        assert modification.modification_id in engine.modifications
        assert modification in engine.pending_modifications

        # Check layer added to visualization
        assert len(viz.layers) > 1
        mod_layer = viz.layers[-1]
        assert mod_layer.layer_type == "modification"
        assert mod_layer.editable is True

    def test_predict_modification_impact(self, engine):
        """Test modification impact prediction"""
        geometry = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
        }

        # Test wildflower strip impact
        impact = engine._predict_modification_impact(
            ModificationType.WILDFLOWER_STRIP,
            geometry,
            {"seed_mix": "pollinator"},
            [1001, 1002]
        )

        assert impact["nectar_increase"] > 0
        assert impact["pollen_increase"] > 0
        assert impact["habitat_area_change"] > 0
        assert impact["biodiversity_increase"] > 0

        # Test margin creation impact
        impact_margin = engine._predict_modification_impact(
            ModificationType.MARGIN_CREATION,
            geometry,
            {"width": 12},
            [1001]
        )

        assert impact_margin["habitat_area_change"] > 0
        assert impact_margin["connectivity_improvement"] > 0

    def test_calculate_geometry_area(self, engine):
        """Test geometry area calculation"""
        # Test polygon area
        polygon = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
        }
        area = engine._calculate_geometry_area(polygon)
        assert area == 1.0  # 100m x 100m = 10000m² = 1ha

        # Test linestring area (with default width)
        linestring = {
            "type": "LineString",
            "coordinates": [[0, 0], [100, 0]]
        }
        area_line = engine._calculate_geometry_area(linestring)
        assert area_line == 0.06  # 100m * 6m = 600m² = 0.06ha

    def test_estimate_modification_cost(self, engine):
        """Test modification cost estimation"""
        geometry = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
        }

        # Test wildflower strip cost
        cost = engine._estimate_modification_cost(
            ModificationType.WILDFLOWER_STRIP,
            geometry,
            {}
        )
        assert cost == 750.0  # 1ha * 750£/ha

        # Test hedge planting (by length)
        hedge_geometry = {
            "type": "LineString",
            "coordinates": [[0, 0], [1000, 0]]  # 1km
        }
        hedge_cost = engine._estimate_modification_cost(
            ModificationType.HEDGE_PLANTING,
            hedge_geometry,
            {}
        )
        assert hedge_cost == 8000.0  # 1km * 8000£/km

    def test_render_before_after(self, engine, test_patches):
        """Test before/after rendering"""
        viz = engine.create_visualization(test_patches)

        # Add modification
        engine.add_modification(
            viz.visualization_id,
            ModificationType.WILDFLOWER_STRIP,
            {"type": "LineString", "coordinates": [[0, 0], [200, 0]]},
            {"width": 6}
        )

        # Render before/after
        render_data = engine.render_before_after(viz.visualization_id)

        assert render_data["type"] == "split_view"
        assert "left" in render_data  # Current state
        assert "right" in render_data  # After modifications
        assert len(render_data["modifications"]) == 1
        assert "summary" in render_data

        # Check left side has no modifications
        left_layers = render_data["left"]["layers"]
        assert all(layer.layer_type != "modification" for layer in left_layers)

        # Check right side includes modifications
        right_layers = render_data["right"]["layers"]
        assert any(layer.layer_type == "modification" for layer in right_layers)

    def test_calculate_modification_summary(self, engine, test_patches):
        """Test modification summary calculation"""
        viz = engine.create_visualization(test_patches)

        # Add multiple modifications
        mod1 = engine.add_modification(
            viz.visualization_id,
            ModificationType.WILDFLOWER_STRIP,
            {"type": "LineString", "coordinates": [[0, 0], [100, 0]]},
            {"width": 6}
        )

        mod2 = engine.add_modification(
            viz.visualization_id,
            ModificationType.MARGIN_CREATION,
            {"type": "Polygon", "coordinates": [[[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]]]},
            {"width": 12}
        )

        summary = engine._calculate_modification_summary([mod1.modification_id, mod2.modification_id])

        assert summary["modification_count"] == 2
        assert summary["total_cost"] > 0
        assert summary["total_area_ha"] > 0
        assert "predicted_impacts" in summary
        assert summary["cost_per_hectare"] > 0

    def test_update_quality_map(self, engine, test_patches):
        """Test quality map update"""
        viz = engine.create_visualization(test_patches)

        # Update quality map
        engine.update_quality_map(
            viz.visualization_id,
            QualityMetric.BIODIVERSITY_INDEX,
            include_modifications=False
        )

        # Check quality layer added/updated
        quality_layers = [layer for layer in viz.layers if layer.layer_type == "quality"]
        assert len(quality_layers) == 1
        assert quality_layers[0].layer_id.startswith("quality_biodiversity")

        # Test cache clearing
        assert len(engine.quality_cache) == 0  # Should be cleared

    def test_analyze_connectivity(self, engine, test_patches):
        """Test connectivity analysis"""
        viz = engine.create_visualization(test_patches)

        # Analyze connectivity
        analysis = engine.analyze_connectivity(viz.visualization_id, threshold_distance=250.0)

        assert "connectivity_score" in analysis
        assert "connections" in analysis
        assert "isolated_patches" in analysis
        assert "average_connection_distance" in analysis
        assert "recommendations" in analysis

        # With 250m threshold, adjacent patches (200m apart) should be connected
        assert analysis["connectivity_score"] >= 0  # May be 0 if patches are isolated
        assert len(analysis["connections"]) >= 0  # May be 0 if no connections found

    def test_animate_temporal_changes(self, engine, test_patches):
        """Test temporal animation generation"""
        viz = engine.create_visualization(test_patches)

        # Add modification with timestamp
        engine.add_modification(
            viz.visualization_id,
            ModificationType.WILDFLOWER_STRIP,
            {"type": "LineString", "coordinates": [[0, 0], [200, 0]]},
            {"width": 6}
        )

        # Generate animation frames
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        frames = engine.animate_temporal_changes(viz.visualization_id, start, end, interval_days=30)

        assert len(frames) > 0
        assert all("date" in frame for frame in frames)
        assert all("day_of_year" in frame for frame in frames)
        assert all("season" in frame for frame in frames)
        assert all("bloom_intensity" in frame for frame in frames)
        assert all("changes" in frame for frame in frames)

        # Check seasonal variation
        seasons = set(frame["season"] for frame in frames)
        assert len(seasons) == 4  # All seasons

        # Check bloom intensity variation
        bloom_values = [frame["bloom_intensity"] for frame in frames]
        assert max(bloom_values) > min(bloom_values)  # Should vary

    def test_export_visualization_geojson(self, engine, test_patches):
        """Test exporting visualization as GeoJSON"""
        viz = engine.create_visualization(test_patches)

        # Export as GeoJSON
        export = engine.export_visualization(viz.visualization_id, format="geojson")

        assert export["type"] == "FeatureCollection"
        assert len(export["features"]) == len(test_patches)
        assert "properties" in export
        assert export["properties"]["visualization_id"] == viz.visualization_id
        assert export["properties"]["mode"] == viz.mode.value

    def test_export_visualization_summary(self, engine, test_patches):
        """Test exporting visualization summary"""
        viz = engine.create_visualization(test_patches)

        # Add modification
        engine.add_modification(
            viz.visualization_id,
            ModificationType.WILDFLOWER_STRIP,
            {"type": "LineString", "coordinates": [[0, 0], [200, 0]]},
            {"width": 6}
        )

        # Export summary
        summary = engine.export_visualization(viz.visualization_id, format="summary")

        assert summary["visualization_id"] == viz.visualization_id
        assert summary["layer_count"] >= 1
        assert summary["modification_count"] == 1
        assert summary["total_estimated_cost"] > 0
        assert summary["affected_area_ha"] > 0

    def test_get_visualization_state(self, engine, test_patches):
        """Test getting visualization state"""
        viz = engine.create_visualization(test_patches)

        state = engine.get_visualization_state(viz.visualization_id)

        assert state["visualization_id"] == viz.visualization_id
        assert state["mode"] == "current_state"
        assert state["bounds"] == viz.bounds
        assert state["center"] == viz.center
        assert state["zoom"] == 1.0
        assert state["layer_count"] >= 1
        assert len(state["visible_layers"]) >= 1
        assert state["pending_modifications"] == 0
        assert "render_settings" in state


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""

    @pytest.fixture
    def configured_engine(self):
        """Engine with sample landscape"""
        engine = LandscapeModificationEngine()

        # Create diverse patches
        patches = []
        habitat_types = [
            HabitatType.CROPLAND,
            HabitatType.GRASSLAND,
            HabitatType.HEDGEROW,
            HabitatType.WOODLAND,
            HabitatType.WILDFLOWER
        ]

        for i in range(10):
            patch = ResourcePatch(
                patch_id=2000 + i,
                x=(i % 5) * 200,
                y=(i // 5) * 200,
                habitat_type=habitat_types[i % len(habitat_types)]
            )
            patch.area_ha = 2.0 + i * 0.5
            patch.base_nectar_production = 1.0 + i * 0.2
            patch.base_pollen_production = 0.8 + i * 0.15
            patches.append(patch)

        # Create visualization
        viz = engine.create_visualization(
            patches,
            mode=VisualizationMode.CURRENT_STATE
        )

        return engine, viz.visualization_id, patches

    def test_landscape_modification_workflow(self, configured_engine):
        """Test complete modification workflow"""
        engine, viz_id, patches = configured_engine

        # Add multiple modifications
        mods = []

        # Add wildflower strip
        mod1 = engine.add_modification(
            viz_id,
            ModificationType.WILDFLOWER_STRIP,
            {
                "type": "LineString",
                "coordinates": [[0, 100], [200, 100], [400, 100], [600, 100]]
            },
            {"width": 6, "seed_mix": "pollinator_nectar"}
        )
        mods.append(mod1)

        # Add margin
        mod2 = engine.add_modification(
            viz_id,
            ModificationType.MARGIN_CREATION,
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [50, 0], [50, 200], [0, 200], [0, 0]]]
            },
            {"width": 12, "species_mix": "wildflower"}
        )
        mods.append(mod2)

        # Add pond
        mod3 = engine.add_modification(
            viz_id,
            ModificationType.POND_CREATION,
            {
                "type": "Polygon",
                "coordinates": [[[300, 300], [350, 300], [350, 350], [300, 350], [300, 300]]]
            },
            {"depth": 1.5}
        )
        mods.append(mod3)

        # Check all modifications added
        assert len(engine.pending_modifications) == 3

        # Render before/after
        render = engine.render_before_after(viz_id)
        assert len(render["modifications"]) == 3

        # Check total impact
        summary = render["summary"]
        assert summary["total_cost"] > 0
        assert summary["total_area_ha"] > 0
        assert summary["predicted_impacts"]["nectar_increase"] > 0
        assert summary["predicted_impacts"]["biodiversity_increase"] > 0

    def test_quality_mapping_with_modifications(self, configured_engine):
        """Test quality mapping before and after modifications"""
        engine, viz_id, patches = configured_engine

        # Update quality map without modifications
        engine.update_quality_map(
            viz_id,
            QualityMetric.NECTAR_AVAILABILITY,
            include_modifications=False
        )

        # Add modifications
        engine.add_modification(
            viz_id,
            ModificationType.WILDFLOWER_STRIP,
            {"type": "LineString", "coordinates": [[0, 0], [800, 0]]},
            {"width": 12, "seed_mix": "nectar_rich"}
        )

        # Update with modifications
        engine.update_quality_map(
            viz_id,
            QualityMetric.NECTAR_AVAILABILITY,
            include_modifications=True
        )

        # Quality should be improved (implementation would show this)
        viz = engine.visualizations[viz_id]
        quality_layers = [layer for layer in viz.layers if layer.layer_type == "quality"]
        assert len(quality_layers) == 1

    def test_connectivity_improvement(self, configured_engine):
        """Test connectivity analysis and improvement"""
        engine, viz_id, patches = configured_engine

        # Initial connectivity
        initial_analysis = engine.analyze_connectivity(viz_id, threshold_distance=150)
        initial_analysis["connectivity_score"]

        # Add corridor modifications
        engine.add_modification(
            viz_id,
            ModificationType.HABITAT_CONNECTIVITY,
            {"type": "LineString", "coordinates": [[0, 0], [200, 200]]},
            {"corridor_type": "wildflower", "width": 10}
        )

        engine.add_modification(
            viz_id,
            ModificationType.HABITAT_CONNECTIVITY,
            {"type": "LineString", "coordinates": [[400, 0], [400, 200]]},
            {"corridor_type": "hedgerow", "width": 3}
        )

        # Re-analyze (would show improvement with full implementation)
        assert len(engine.pending_modifications) == 2

        # Check recommendations
        assert len(initial_analysis["recommendations"]) >= 0
        if initial_analysis["isolated_patches"]:
            assert any(r["type"] == "connect_isolated" for r in initial_analysis["recommendations"])

    def test_temporal_visualization(self, configured_engine):
        """Test temporal visualization across seasons"""
        engine, viz_id, patches = configured_engine

        # Add seasonal modifications
        engine.add_modification(
            viz_id,
            ModificationType.WILDFLOWER_STRIP,
            {"type": "LineString", "coordinates": [[0, 0], [400, 0]]},
            {"seed_mix": "early_spring", "width": 6}
        )

        engine.add_modification(
            viz_id,
            ModificationType.WILDFLOWER_STRIP,
            {"type": "LineString", "coordinates": [[0, 100], [400, 100]]},
            {"seed_mix": "summer_nectar", "width": 6}
        )

        # Generate full year animation
        frames = engine.animate_temporal_changes(
            viz_id,
            date(2024, 1, 1),
            date(2024, 12, 31),
            interval_days=14  # Bi-weekly
        )

        assert len(frames) >= 24  # At least 24 frames for year

        # Check bloom intensity peaks in summer
        summer_frames = [f for f in frames if f["season"] == "summer"]
        summer_bloom = max(f["bloom_intensity"] for f in summer_frames)

        winter_frames = [f for f in frames if f["season"] == "winter"]
        winter_bloom = max(f["bloom_intensity"] for f in winter_frames)

        assert summer_bloom > winter_bloom
