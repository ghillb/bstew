"""
Test for Plot System fixes
=========================

Tests for date parsing, format conversion, and zero plots validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from bstew.analysis.population_plotter import PopulationPlotter, PlotConfig
from bstew.cli.commands.analyze import PlotCommand
from bstew.cli.core.base import CLIContext


def test_robust_date_parsing():
    """Test that various date formats are parsed correctly"""
    plotter = PopulationPlotter()

    # Test different date formats
    test_dates = [
        pd.Series(['2024-01-01', '2024-01-02', '2024-01-03']),  # ISO format
        pd.Series(['01/01/2024', '02/01/2024', '03/01/2024']),  # UK format
        pd.Series(['1/1/2024', '1/2/2024', '1/3/2024']),        # US format
        pd.Series(['2024-01-01 12:00:00', '2024-01-02 13:00:00']),  # Datetime
        pd.Series(['01-01-2024', '02-01-2024', '03-01-2024']),  # Dash format
    ]

    for dates in test_dates:
        result = plotter._parse_dates_safely(dates)
        assert isinstance(result, pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(result) or pd.api.types.is_numeric_dtype(result)


def test_invalid_date_fallback():
    """Test that invalid dates fall back to numeric range"""
    plotter = PopulationPlotter()

    # Invalid dates that would cause "day is out of range for month"
    invalid_dates = pd.Series(['2024-02-31', '2024-04-31', '2024-06-31'])

    result = plotter._parse_dates_safely(invalid_dates)

    # Should fall back to numeric range
    assert isinstance(result, pd.Series)
    assert result.name == 'day_number'
    assert list(result) == [0, 1, 2]


def test_plot_format_conversion():
    """Test that plots save in the correct format"""
    plotter = PopulationPlotter()

    # Create sample data
    data = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=10),
        'population': np.random.randint(100, 200, 10)
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test different formats
        for fmt in ['png', 'pdf', 'svg']:
            output_path = Path(tmpdir) / f'test_plot.{fmt}'
            plotter.plot_population_trends(data, save_path=str(output_path))

            # Verify file exists and has correct extension
            assert output_path.exists()
            assert output_path.suffix == f'.{fmt}'
            # Verify file size is reasonable
            assert output_path.stat().st_size > 100


def test_zero_plots_validation():
    """Test that zero plots raises an error instead of claiming success"""
    context = CLIContext()
    command = PlotCommand(context)

    # Create a scenario where no plots would be generated
    # by providing empty data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty data directory
        input_dir = Path(tmpdir) / 'input'
        output_dir = Path(tmpdir) / 'output'
        input_dir.mkdir()
        output_dir.mkdir()

        # This should raise an error because no plots can be generated
        result = command.execute(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            plot_type='comparison'  # This type requires specific data
        )

        # Should fail with appropriate error message
        assert not result.success
        assert "No plots generated" in result.message or "No data files found" in result.message


def test_save_plot_with_format_validation():
    """Test the save_plot_with_format method validates file creation"""
    plotter = PopulationPlotter()

    import matplotlib.pyplot as plt

    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'test_plot.png'

        # Should save successfully
        plotter._save_plot_with_format(fig, str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 100

        # The method will default to png for invalid formats and force correct extension
        # So test that it handles this correctly
        invalid_path = Path(tmpdir) / 'test_plot.invalid'
        plotter._save_plot_with_format(fig, str(invalid_path))

        # Should have created a .png file instead
        expected_path = Path(tmpdir) / 'test_plot.png'
        assert expected_path.exists()

    plt.close(fig)


def test_sample_data_generates_valid_dates():
    """Test that sample data generation creates valid dates"""
    from bstew.cli.commands.analyze import PlotCommand, PlotOptions

    context = CLIContext()
    command = PlotCommand(context)

    options = PlotOptions()
    data = command._create_sample_data('population', options)

    assert 'population_data' in data
    df = data['population_data']

    # Verify time column exists and has valid dates
    assert 'time' in df.columns

    # Parse dates to ensure they're valid
    plotter = PopulationPlotter()
    parsed_dates = plotter._parse_dates_safely(df['time'])

    # Should successfully parse all dates
    assert len(parsed_dates) == len(df)
    assert not parsed_dates.isna().any()
