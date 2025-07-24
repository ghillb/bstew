"""
Test Empty Results Handling
===========================

Tests for Fix #8: Handle empty analysis results with helpful user feedback.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from bstew.cli.commands.analyze import AnalyzeCommand
from bstew.cli.core.base import CLIContext


class TestEmptyResultsHandling:
    """Test empty analysis results handling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.context = CLIContext()
        self.command = AnalyzeCommand(self.context)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)

    def test_handle_empty_results_method(self):
        """Test the handle_empty_analysis_results method directly"""
        # Test with empty results
        empty_results = {}
        handled = self.command.handle_empty_analysis_results(empty_results)

        assert handled['status'] == 'warning'
        assert handled['message'] == 'No analysis results generated'
        assert 'diagnosis' in handled
        assert 'suggestions' in handled
        assert len(handled['suggestions']) > 0

    def test_handle_results_with_empty_dicts(self):
        """Test handling results with empty nested dictionaries"""
        results = {
            'population_trends': {},
            'growth_rates': {},
            'foraging_efficiency': {},
            'summary_statistics': {}
        }

        handled = self.command.handle_empty_analysis_results(results)

        assert handled['status'] == 'warning'
        assert 'Check if input directory contains valid simulation data' in handled['suggestions']

    def test_handle_non_empty_results(self):
        """Test that non-empty results are returned unchanged"""
        results = {
            'population_trends': {'species_1': {'trend': 'increasing'}},
            'summary_statistics': {'total_population': 1000}
        }

        handled = self.command.handle_empty_analysis_results(results)

        # Should return original results unchanged
        assert handled == results
        assert 'status' not in handled

    def test_diagnose_empty_results(self):
        """Test the diagnosis function"""
        diagnosis = self.command._diagnose_empty_results()

        assert "Possible causes for empty results:" in diagnosis
        # Should check for missing files
        assert "population_data.csv" in diagnosis or "Missing expected data files" in diagnosis

    def test_analyze_empty_directory(self):
        """Test analyzing an empty directory"""
        # Create empty directory
        empty_dir = Path(self.temp_dir) / "empty_data"
        empty_dir.mkdir()

        # Run analysis
        result = self.command.execute(
            input_dir=str(empty_dir),
            format_type="table",
            output_file=None
        )

        # Should succeed but with warning output
        assert result.success
        # Check that appropriate warnings would be displayed
        # (actual display depends on context.print_warning being called)

    def test_display_empty_results_warning(self):
        """Test the display warning method"""
        warning_results = {
            'status': 'warning',
            'message': 'No analysis results generated',
            'diagnosis': 'Test diagnosis: Missing data files',
            'suggestions': [
                'Check input directory',
                'Verify data format'
            ],
            'summary_statistics': {'partial_stat': 123}
        }

        # This should not raise any exceptions
        self.command._display_empty_results_warning(warning_results)

    def test_save_not_attempted_for_empty_results(self):
        """Test that save is not attempted when results are empty"""
        empty_dir = Path(self.temp_dir) / "empty_data"
        empty_dir.mkdir()

        output_file = str(Path(self.temp_dir) / "output.csv")

        # Run analysis with output file
        result = self.command.execute(
            input_dir=str(empty_dir),
            format_type="csv",
            output_file=output_file
        )

        # Should succeed without creating output file
        assert result.success
        assert not Path(output_file).exists()

    def test_empty_results_with_sample_data(self):
        """Test that sample data generation handles empty results properly"""
        # The command uses sample data when no files are found
        # Make sure this doesn't bypass empty results handling
        empty_dir = Path(self.temp_dir) / "no_data"
        empty_dir.mkdir()

        # Analyze results
        results = self.command._analyze_results(empty_dir, "comprehensive")

        # Should have some results from sample data
        assert 'summary_statistics' in results
        # Sample data should produce non-empty results
        assert results['summary_statistics']  # Should not be empty


def test_integration_with_cli_context():
    """Test integration with CLI context for proper message display"""
    context = CLIContext()
    command = AnalyzeCommand(context)

    # Create results that will trigger empty handling
    empty_results = {
        'population_trends': {},
        'growth_rates': {},
        'summary_statistics': {}
    }

    # Process through the handler
    handled = command.handle_empty_analysis_results(empty_results)

    # Verify structure
    assert handled['status'] == 'warning'
    assert isinstance(handled['suggestions'], list)
    assert len(handled['suggestions']) >= 3  # Should have multiple suggestions

    # Check specific suggestions are included
    suggestions_text = ' '.join(handled['suggestions'])
    assert 'valid simulation data' in suggestions_text
    assert 'data format' in suggestions_text
    assert '--debug flag' in suggestions_text
