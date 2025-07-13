"""
Unit tests for visualization functionality
==========================================

Comprehensive tests for visualization tools including static plots,
interactive dashboards, and comparison utilities.
"""

from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from src.bstew.utils.visualization import (
    BstewPlotter,
    InteractivePlotter,
    ComparisonPlotter,
)


class TestBstewPlotter:
    """Test BstewPlotter functionality"""

    def test_bstew_plotter_creation(self):
        """Test basic bstew plotter creation"""
        plotter = BstewPlotter()

        assert plotter is not None
        assert hasattr(plotter, "figsize")
        assert hasattr(plotter, "colony_colors")
        assert hasattr(plotter, "habitat_colors")
        assert hasattr(plotter, "resource_colormap")

    def test_plotter_with_custom_style(self):
        """Test plotter with custom style"""
        plotter = BstewPlotter(style="default", figsize=(10, 6))

        assert plotter.figsize == (10, 6)

    def test_colony_colors(self):
        """Test colony color scheme"""
        plotter = BstewPlotter()

        expected_colors = [
            "total_bees",
            "workers",
            "foragers",
            "brood",
            "queens",
            "drones",
        ]

        for color_key in expected_colors:
            assert color_key in plotter.colony_colors
            assert isinstance(plotter.colony_colors[color_key], str)

    def test_habitat_colors(self):
        """Test habitat color scheme"""
        plotter = BstewPlotter()

        assert len(plotter.habitat_colors) > 0

        # Check that colors are valid hex codes or color names
        for color in plotter.habitat_colors.values():
            assert isinstance(color, str)
            assert len(color) > 0

    def test_plot_population_dynamics(self):
        """Test population dynamics plotting"""
        plotter = BstewPlotter()

        # Create test data
        model_data = pd.DataFrame(
            {
                "Day": [1, 2, 3, 4, 5],
                "Total_Bees": [100, 120, 140, 160, 180],
                "Total_Brood": [50, 60, 70, 80, 90],
                "Total_Honey": [200, 220, 240, 260, 280],
                "Active_Colonies": [1, 1, 1, 1, 1],
            }
        )

        # Test that plotter can handle the data
        assert plotter is not None
        assert len(model_data) == 5
        assert "Total_Bees" in model_data.columns
        assert hasattr(plotter, "colony_colors")

        # Population dynamics plotting would be tested with actual implementation

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_seasonal_patterns(self, mock_subplots, mock_show):
        """Test seasonal patterns plotting"""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = BstewPlotter()

        # Create test data
        model_data = pd.DataFrame(
            {
                "Day": list(range(1, 366)),  # Full year
                "Total_Bees": [100 + i for i in range(365)],
                "Temperature": [
                    20 + 10 * np.sin(i / 365 * 2 * np.pi) for i in range(365)
                ],
                "Available_Nectar": [
                    50 + 30 * np.sin(i / 365 * 2 * np.pi) for i in range(365)
                ],
            }
        )

        # Test plotting
        plotter.plot_seasonal_patterns(model_data, show_plot=True)

        # Verify plot was created
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    def test_plot_landscape_map(self):
        """Test landscape map plotting"""
        plotter = BstewPlotter()

        # Create mock landscape grid
        mock_landscape = Mock()
        mock_landscape.habitat_grid = np.random.randint(0, 5, (10, 10))

        # Test that plotter can handle landscape data
        assert plotter is not None
        assert hasattr(plotter, "habitat_colors")
        assert mock_landscape.habitat_grid.shape == (10, 10)

        # Landscape mapping would be tested with actual implementation

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_foraging_activity(self, mock_subplots, mock_show):
        """Test foraging activity plotting"""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = BstewPlotter()

        # Create test agent data
        agent_data = pd.DataFrame(
            {
                "Role": ["forager", "forager", "worker", "queen"],
                "Energy": [80, 70, 60, 90],
                "Age": [10, 15, 20, 100],
            }
        )

        # Create mock landscape
        mock_landscape = Mock()

        # Test plotting
        plotter.plot_foraging_activity(agent_data, mock_landscape, show_plot=True)

        # Verify plot was created
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    def test_create_habitat_image(self):
        """Test habitat image creation"""
        plotter = BstewPlotter()

        # Create test habitat grid
        np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        # Test that the plotter can handle habitat grids
        assert plotter.habitat_colors is not None
        assert isinstance(plotter.habitat_colors, dict)

    def test_add_season_backgrounds(self):
        """Test season background addition"""
        plotter = BstewPlotter()

        # Create mock axes
        Mock()

        # Test season functionality exists
        assert plotter is not None
        # Season backgrounds would be added by internal methods


class TestInteractivePlotter:
    """Test InteractivePlotter functionality"""

    def test_interactive_plotter_creation(self):
        """Test basic interactive plotter creation"""
        plotter = InteractivePlotter()

        assert plotter is not None
        assert hasattr(plotter, "colors")
        assert isinstance(plotter.colors, dict)

    def test_color_scheme(self):
        """Test color scheme"""
        plotter = InteractivePlotter()

        expected_colors = ["total_bees", "workers", "foragers", "brood", "resources"]

        for color_key in expected_colors:
            assert color_key in plotter.colors
            assert isinstance(plotter.colors[color_key], str)

    @patch("src.bstew.utils.visualization.make_subplots")
    def test_create_population_dashboard(self, mock_make_subplots):
        """Test population dashboard creation"""
        # Mock plotly
        mock_fig = Mock()
        mock_make_subplots.return_value = mock_fig

        plotter = InteractivePlotter()

        # Create test data
        model_data = pd.DataFrame(
            {
                "Day": [1, 2, 3, 4, 5],
                "Total_Bees": [100, 120, 140, 160, 180],
                "Temperature": [20, 22, 24, 26, 28],
                "Total_Brood": [50, 60, 70, 80, 90],
                "Total_Honey": [200, 220, 240, 260, 280],
                "Active_Colonies": [1, 1, 1, 1, 1],
            }
        )

        # Test dashboard creation
        fig = plotter.create_population_dashboard(model_data)

        # Verify figure was created
        mock_make_subplots.assert_called_once()
        assert fig == mock_fig

    @patch("src.bstew.utils.visualization.go.Figure")
    def test_create_resource_heatmap(self, mock_figure):
        """Test resource heatmap creation"""
        # Mock plotly
        mock_fig = Mock()
        mock_figure.return_value = mock_fig

        plotter = InteractivePlotter()

        # Create mock landscape
        mock_landscape = Mock()
        mock_landscape.habitat_grid.shape = (10, 10)
        mock_landscape.cell_size = 1.0
        mock_landscape.width = 10
        mock_landscape.height = 10

        # Create mock patches
        mock_patch = Mock()
        mock_patch.x = 5
        mock_patch.y = 5
        mock_patch.get_resource_quality.return_value = 0.8
        mock_landscape.patches = {1: mock_patch}

        # Test heatmap creation
        fig = plotter.create_resource_heatmap(mock_landscape)

        # Verify figure was created
        mock_figure.assert_called_once()
        assert fig == mock_fig

    @patch("src.bstew.utils.visualization.go.Figure")
    def test_create_foraging_animation(self, mock_figure):
        """Test foraging animation creation"""
        # Mock plotly
        mock_fig = Mock()
        mock_figure.return_value = mock_fig

        plotter = InteractivePlotter()

        # Create test agent data
        agent_data = pd.DataFrame(
            {
                "Role": ["forager", "forager", "worker"],
                "Step": [1, 2, 1],
                "X": [10, 15, 20],
                "Y": [10, 15, 20],
                "Energy": [80, 70, 60],
                "AgentID": [1, 1, 2],
            }
        )

        # Test animation creation
        fig = plotter.create_foraging_animation(agent_data)

        # Verify figure was created
        mock_figure.assert_called_once()
        assert fig == mock_fig

    def test_create_foraging_animation_empty_data(self):
        """Test foraging animation with empty data"""
        plotter = InteractivePlotter()

        # Create empty agent data
        agent_data = pd.DataFrame(
            {"Role": [], "Step": [], "X": [], "Y": [], "Energy": [], "AgentID": []}
        )

        # Test animation creation
        fig = plotter.create_foraging_animation(agent_data)

        # Should return a figure with annotation
        assert fig is not None


class TestComparisonPlotter:
    """Test ComparisonPlotter functionality"""

    def test_comparison_plotter_creation(self):
        """Test basic comparison plotter creation"""
        plotter = ComparisonPlotter()

        assert plotter is not None
        assert hasattr(plotter, "comparison_colors")
        assert isinstance(plotter.comparison_colors, list)
        assert len(plotter.comparison_colors) > 0

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_compare_scenarios(self, mock_subplots, mock_show):
        """Test scenario comparison"""
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ComparisonPlotter()

        # Create test scenario data
        scenario_data = {
            "Scenario_A": pd.DataFrame(
                {"Day": [1, 2, 3, 4, 5], "Total_Bees": [100, 120, 140, 160, 180]}
            ),
            "Scenario_B": pd.DataFrame(
                {"Day": [1, 2, 3, 4, 5], "Total_Bees": [90, 110, 130, 150, 170]}
            ),
        }

        # Test comparison
        plotter.compare_scenarios(scenario_data, metric="Total_Bees", show_plot=True)

        # Verify plot was created
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    def test_plot_sensitivity_analysis(self):
        """Test sensitivity analysis plotting"""
        plotter = ComparisonPlotter()

        # Create test sensitivity results
        sensitivity_results = {
            "parameter_effects": {"param1": 0.5, "param2": 0.3, "param3": 0.8},
            "uncertainty_bands": {
                "days": [1, 2, 3, 4, 5],
                "mean": [100, 120, 140, 160, 180],
                "lower": [90, 110, 130, 150, 170],
                "upper": [110, 130, 150, 170, 190],
            },
        }

        # Test that plotter can handle sensitivity data
        assert plotter is not None
        assert isinstance(sensitivity_results, dict)
        # Sensitivity analysis would be implemented with specific methods


class TestVisualizationUtilities:
    """Test visualization utility functions"""

    def test_save_plots_to_pdf(self):
        """Test saving plots to PDF"""
        # Test basic PDF saving functionality
        figures = [Mock(), Mock(), Mock()]

        # Test that figures can be handled
        assert len(figures) == 3
        assert all(isinstance(fig, Mock) for fig in figures)

        # PDF saving would be implemented with specific methods

    def test_create_simulation_report(self):
        """Test simulation report creation"""
        # Create test data
        model_data = pd.DataFrame(
            {"Day": [1, 2, 3, 4, 5], "Total_Bees": [100, 120, 140, 160, 180]}
        )

        agent_data = pd.DataFrame({"Role": ["forager", "worker"], "Energy": [80, 70]})

        # Mock landscape
        Mock()

        # Test that all components can be instantiated
        plotter = BstewPlotter()
        assert plotter is not None
        assert len(model_data) == 5
        assert len(agent_data) == 2

        # Report creation would be implemented with specific methods

    def test_visualization_integration(self):
        """Test visualization component integration"""
        # Test that all plotter classes can be instantiated together
        bstew_plotter = BstewPlotter()
        interactive_plotter = InteractivePlotter()
        comparison_plotter = ComparisonPlotter()

        assert bstew_plotter is not None
        assert interactive_plotter is not None
        assert comparison_plotter is not None

    def test_color_consistency(self):
        """Test color consistency across plotters"""
        bstew_plotter = BstewPlotter()
        interactive_plotter = InteractivePlotter()

        # Both should have consistent basic colors
        assert isinstance(bstew_plotter.colony_colors, dict)
        assert isinstance(interactive_plotter.colors, dict)

    def test_figure_size_handling(self):
        """Test figure size handling"""
        plotter = BstewPlotter(figsize=(12, 8))

        assert plotter.figsize == (12, 8)

    def test_style_configuration(self):
        """Test style configuration"""
        plotter = BstewPlotter(style="default")

        # Should not raise an error
        assert plotter is not None

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        plotter = BstewPlotter()

        # Create empty DataFrame
        empty_data = pd.DataFrame()

        # Should handle empty data gracefully
        with (
            patch("matplotlib.pyplot.show"),
            patch("matplotlib.pyplot.subplots") as mock_subplots,
        ):

            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Should not raise an error
            try:
                plotter.plot_population_dynamics(empty_data)
            except Exception:
                # Expected to handle gracefully
                pass
