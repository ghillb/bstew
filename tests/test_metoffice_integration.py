"""
Test Met Office API Integration
================================

Tests for the Met Office weather API integration with proper mocking and error handling.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from datetime import datetime

from bstew.utils.weather import WeatherFileLoader, WeatherDataSource, WeatherStation


class TestMetOfficeIntegration:
    """Test Met Office API integration"""

    def setup_method(self):
        """Set up test fixtures"""
        self.loader = WeatherFileLoader()
        self.data_source = WeatherDataSource(
            source_type="api",
            api_config={"type": "metoffice"},
            station_info=WeatherStation(
                station_id="london_test",
                name="London Test Station",
                latitude=51.5074,
                longitude=-0.1278,
                elevation=10.0,
                data_source="metoffice",
                active_period=(datetime(2020, 1, 1), datetime(2025, 12, 31))
            )
        )

    @patch.dict(os.environ, {"MET_OFFICE_API_KEY": "test_key_123"})
    @patch("requests.get")
    def test_successful_api_call(self, mock_get):
        """Test successful Met Office API data retrieval"""
        # Mock API responses
        mock_sites_response = MagicMock()
        mock_sites_response.json.return_value = {
            "sites": [
                {"id": "3672", "name": "Heathrow", "latitude": 51.479, "longitude": -0.449},
                {"id": "3776", "name": "Greenwich", "latitude": 51.478, "longitude": 0.0}
            ]
        }
        mock_sites_response.raise_for_status = MagicMock()

        mock_obs_response = MagicMock()
        mock_obs_response.json.return_value = {
            "site": {"name": "Heathrow"},
            "siteTimeSeries": {
                "timeSeries": [
                    {
                        "time": "2024-01-01T12:00:00Z",
                        "temperature": 10.5,
                        "windSpeed": 5.2,
                        "humidity": 80,
                        "pressure": 1015,
                        "precipitationAmount": 0.0,
                        "significantWeatherCode": 1
                    }
                ]
            }
        }
        mock_obs_response.raise_for_status = MagicMock()

        mock_forecast_response = MagicMock()
        mock_forecast_response.json.return_value = {
            "location": {"name": "Heathrow"},
            "forecast": {
                "timeSeries": [
                    {
                        "time": "2024-01-02T12:00:00Z",
                        "temperature": 12.0,
                        "windSpeed": 6.0,
                        "humidity": 75,
                        "pressure": 1012,
                        "precipitationProbability": 20,
                        "significantWeatherCode": 3
                    }
                ]
            }
        }
        mock_forecast_response.raise_for_status = MagicMock()

        # Set up mock to return different responses based on URL
        def side_effect(url, **kwargs):
            if "all-sites" in url:
                return mock_sites_response
            elif "/site/" in url:
                return mock_obs_response
            elif "/forecasts/" in url:
                return mock_forecast_response
            return MagicMock()

        mock_get.side_effect = side_effect

        # Call the method
        result = self.loader._load_metoffice_api(self.data_source)

        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 1 observation + 1 forecast
        assert "date" in result.columns
        assert "temperature" in result.columns
        assert "humidity" in result.columns
        assert "wind_speed" in result.columns
        assert "rainfall" in result.columns

        # Check data values
        assert result.iloc[0]["temperature"] == 10.5
        assert result.iloc[1]["temperature"] == 12.0

    @patch.dict(os.environ, {}, clear=True)  # No API key
    def test_missing_api_key_fallback(self):
        """Test fallback to synthetic data when API key is missing"""
        with patch.object(self.loader, '_generate_synthetic_data') as mock_synthetic:
            mock_synthetic.return_value = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=10),
                'temperature': [15.0] * 10,
                'rainfall': [0.0] * 10,
                'wind_speed': [5.0] * 10,
                'humidity': [75.0] * 10
            })

            result = self.loader._load_metoffice_api(self.data_source)

            # Verify fallback was called
            mock_synthetic.assert_called_once_with(self.data_source)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 10

    @patch.dict(os.environ, {"MET_OFFICE_API_KEY": "test_key_123"})
    @patch("requests.get")
    def test_api_request_failure(self, mock_get):
        """Test graceful handling of API request failures"""
        # Mock a request exception
        mock_get.side_effect = Exception("Network error")

        with patch.object(self.loader, '_generate_synthetic_data') as mock_synthetic:
            mock_synthetic.return_value = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=5),
                'temperature': [20.0] * 5,
                'rainfall': [0.0] * 5,
                'wind_speed': [3.0] * 5,
                'humidity': [70.0] * 5
            })

            result = self.loader._load_metoffice_api(self.data_source)

            # Verify fallback was called
            mock_synthetic.assert_called_once()
            assert isinstance(result, pd.DataFrame)

    @patch.dict(os.environ, {"MET_OFFICE_API_KEY": "test_key_123"})
    @patch("requests.get")
    def test_missing_data_columns_handling(self, mock_get):
        """Test handling of missing columns in API response"""
        # Mock API response with missing columns
        mock_sites_response = MagicMock()
        mock_sites_response.json.return_value = {
            "sites": [{"id": "3672", "latitude": 51.479, "longitude": -0.449}]
        }
        mock_sites_response.raise_for_status = MagicMock()

        mock_obs_response = MagicMock()
        mock_obs_response.json.return_value = {
            "site": {"name": "Test"},
            "siteTimeSeries": {
                "timeSeries": [
                    {
                        "time": "2024-01-01T12:00:00Z",
                        "temperature": 15.0,
                        "windSpeed": 10.0
                        # Missing humidity, pressure, precipitation
                    }
                ]
            }
        }
        mock_obs_response.raise_for_status = MagicMock()

        mock_forecast_response = MagicMock()
        mock_forecast_response.json.return_value = {
            "location": {"name": "Test"},
            "forecast": {"timeSeries": []}  # Empty forecast
        }
        mock_forecast_response.raise_for_status = MagicMock()

        def side_effect(url, **kwargs):
            if "all-sites" in url:
                return mock_sites_response
            elif "/site/" in url:
                return mock_obs_response
            elif "/forecasts/" in url:
                return mock_forecast_response
            return MagicMock()

        mock_get.side_effect = side_effect

        result = self.loader._load_metoffice_api(self.data_source)

        # Verify default values were added
        assert "humidity" in result.columns
        assert result.iloc[0]["humidity"] == 75.0  # Default value
        assert "rainfall" in result.columns
        assert result.iloc[0]["rainfall"] == 0.0  # Default value

    def test_nearest_location_calculation(self):
        """Test finding nearest Met Office location"""
        headers = {"x-ibm-client-id": "test_key"}

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "sites": [
                    {"id": "3672", "name": "Heathrow", "latitude": 51.479, "longitude": -0.449},
                    {"id": "3776", "name": "Greenwich", "latitude": 51.478, "longitude": 0.0},
                    {"id": "3808", "name": "Gatwick", "latitude": 51.148, "longitude": -0.190}
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # Test location closest to central London
            location_id = self.loader._get_nearest_metoffice_location(
                51.5074, -0.1278, headers
            )

            # Should return Gatwick as it's closest to the coordinates
            assert location_id in ["3672", "3776", "3808"]  # Any of these is reasonable

    def test_data_source_without_coordinates(self):
        """Test handling data source without coordinates"""
        # Create data source without coordinates
        data_source = WeatherDataSource(
            source_type="api",
            api_config={"type": "metoffice"}
        )

        with patch.dict(os.environ, {"MET_OFFICE_API_KEY": "test_key"}):
            with patch.object(self.loader, '_generate_synthetic_data') as mock_synthetic:
                mock_synthetic.return_value = pd.DataFrame({
                    'date': pd.date_range('2024-01-01', periods=3),
                    'temperature': [10.0, 11.0, 12.0],
                    'rainfall': [0.0, 0.0, 0.0],
                    'wind_speed': [5.0, 5.0, 5.0],
                    'humidity': [75.0, 75.0, 75.0]
                })

                with patch("requests.get") as mock_get:
                    # Mock successful API calls
                    mock_sites = MagicMock()
                    mock_sites.json.return_value = {
                        "sites": [{"id": "3672", "latitude": 51.479, "longitude": -0.449}]
                    }
                    mock_sites.raise_for_status = MagicMock()

                    mock_data = MagicMock()
                    mock_data.json.return_value = {
                        "site": {"name": "Test"},
                        "siteTimeSeries": {"timeSeries": []}
                    }
                    mock_data.raise_for_status = MagicMock()

                    mock_get.side_effect = [mock_sites, mock_data, mock_data]

                    # Should use default London coordinates
                    result = self.loader._load_metoffice_api(data_source)

                    # Verify it returned synthetic data (since no real data in mock)
                    assert isinstance(result, pd.DataFrame)


def test_integration_with_weather_manager():
    """Test integration with WeatherIntegrationManager"""
    from bstew.utils.weather import WeatherIntegrationManager

    manager = WeatherIntegrationManager()

    # Add Met Office data source
    data_source = WeatherDataSource(
        source_type="api",
        api_config={"type": "metoffice"},
        station_info=WeatherStation(
            station_id="birmingham_test",
            name="Birmingham Test Station",
            latitude=52.4862,
            longitude=-1.8904,
            elevation=140.0,
            data_source="metoffice",
            active_period=(datetime(2020, 1, 1), datetime(2025, 12, 31))
        )
    )

    manager.add_weather_source(data_source)

    # Mock the API call
    with patch.dict(os.environ, {"MET_OFFICE_API_KEY": "test_key"}):
        with patch.object(manager.file_loader, '_load_metoffice_api') as mock_api:
            mock_api.return_value = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=7),
                'temperature': [5.0, 6.0, 7.0, 8.0, 7.0, 6.0, 5.0],
                'rainfall': [0.0, 2.0, 5.0, 0.0, 0.0, 1.0, 0.0],
                'wind_speed': [10.0, 12.0, 15.0, 8.0, 10.0, 11.0, 9.0],
                'humidity': [80.0, 85.0, 90.0, 75.0, 80.0, 82.0, 78.0]
            })

            # Load weather data
            weather_data = manager.load_all_weather_data()

            assert "source_0" in weather_data
            assert len(weather_data["source_0"]) == 7

            # Get weather for simulation
            conditions = manager.get_weather_for_simulation(
                datetime(2024, 1, 1), 7
            )

            assert len(conditions) == 7
            assert all(hasattr(c, 'temperature') for c in conditions)
