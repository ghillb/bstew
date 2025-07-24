"""
Weather system integration with file loading for BSTEW
======================================================

Handles loading and processing of weather data from various file formats,
integrating with the environmental effects system.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta
import requests
import json

from ..components.environment import (
    WeatherConditions,
    WeatherType,
    SeasonType,
    WeatherGenerator,
)


class WeatherStation(BaseModel):
    """Weather station information"""

    model_config = {"validate_assignment": True}

    station_id: str = Field(description="Unique station identifier")
    name: str = Field(description="Station name")
    latitude: float = Field(ge=-90.0, le=90.0, description="Latitude in degrees")
    longitude: float = Field(ge=-180.0, le=180.0, description="Longitude in degrees")
    elevation: float = Field(description="Elevation in meters")
    data_source: str = Field(description="Data source identifier")
    active_period: Tuple[datetime, datetime] = Field(
        description="Active period start and end"
    )


class WeatherDataSource(BaseModel):
    """Weather data source configuration"""

    model_config = {"validate_assignment": True}

    source_type: str = Field(description="Source type: 'file', 'api', 'synthetic'")
    file_path: Optional[str] = Field(
        default=None, description="Path to weather data file"
    )
    api_config: Optional[Dict[str, Any]] = Field(
        default=None, description="API configuration"
    )
    station_info: Optional[WeatherStation] = Field(
        default=None, description="Weather station information"
    )
    column_mapping: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Column name mappings"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize default column mapping after model creation"""
        if self.column_mapping is None:
            self.column_mapping = self._get_default_column_mapping()

    def _get_default_column_mapping(self) -> Dict[str, List[str]]:
        """Get default column name mappings"""
        return {
            "date": ["date", "Date", "DATE", "time", "Time", "datetime"],
            "temperature": ["temp", "temperature", "Temperature", "TEMP", "air_temp"],
            "rainfall": ["rain", "rainfall", "precipitation", "precip", "RAIN"],
            "wind_speed": ["wind", "wind_speed", "windspeed", "WIND", "wind_mph"],
            "humidity": ["humidity", "relative_humidity", "RH", "HUMIDITY"],
            "pressure": ["pressure", "atmospheric_pressure", "atm_press", "PRESSURE"],
        }


class WeatherFileLoader:
    """
    Loads weather data from various file formats.

    Supports:
    - CSV files
    - Excel files
    - JSON files
    - NetCDF files
    - Custom text formats
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.supported_formats = [".csv", ".xlsx", ".xls", ".json", ".nc", ".txt"]

    def load_weather_data(self, data_source: WeatherDataSource) -> pd.DataFrame:
        """Load weather data from specified source"""

        if data_source.source_type == "file":
            return self._load_from_file(data_source)
        elif data_source.source_type == "api":
            return self._load_from_api(data_source)
        elif data_source.source_type == "synthetic":
            return self._generate_synthetic_data(data_source)
        else:
            raise ValueError(f"Unsupported source type: {data_source.source_type}")

    def _load_from_file(self, data_source: WeatherDataSource) -> pd.DataFrame:
        """Load weather data from file"""

        if data_source.file_path is None:
            raise ValueError("File path is required for file-based weather data source")
        file_path = Path(data_source.file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Weather file not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext == ".csv":
            return self._load_csv(file_path, data_source)
        elif file_ext in [".xlsx", ".xls"]:
            return self._load_excel(file_path, data_source)
        elif file_ext == ".json":
            return self._load_json(file_path, data_source)
        elif file_ext == ".nc":
            return self._load_netcdf(file_path, data_source)
        elif file_ext == ".txt":
            return self._load_text(file_path, data_source)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def _load_csv(
        self, file_path: Path, data_source: WeatherDataSource
    ) -> pd.DataFrame:
        """Load CSV weather data"""

        try:
            # Try different common separators and encodings
            for sep in [",", ";", "\t"]:
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                        if len(df.columns) > 1:
                            break
                    except Exception:
                        continue
                else:
                    continue
                break
            else:
                raise ValueError(
                    "Could not parse CSV file with common separators and encodings"
                )

            return self._standardize_weather_data(df, data_source)

        except Exception as e:
            self.logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    def _load_excel(
        self, file_path: Path, data_source: WeatherDataSource
    ) -> pd.DataFrame:
        """Load Excel weather data"""

        try:
            # Try to detect which sheet contains weather data
            xl_file = pd.ExcelFile(file_path)

            # Look for sheets with common weather-related names
            weather_sheets = []
            for sheet in xl_file.sheet_names:
                if any(
                    keyword in str(sheet).lower()
                    for keyword in ["weather", "data", "daily", "hourly"]
                ):
                    weather_sheets.append(sheet)

            # Use first sheet if no weather-specific sheet found
            sheet_name = weather_sheets[0] if weather_sheets else xl_file.sheet_names[0]

            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return self._standardize_weather_data(df, data_source)

        except Exception as e:
            self.logger.error(f"Error loading Excel file {file_path}: {e}")
            raise

    def _load_json(
        self, file_path: Path, data_source: WeatherDataSource
    ) -> pd.DataFrame:
        """Load JSON weather data"""

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if "data" in data:
                    df = pd.DataFrame(data["data"])
                elif "weather" in data:
                    df = pd.DataFrame(data["weather"])
                else:
                    # Try to flatten nested structure
                    df = pd.json_normalize(data)
            else:
                raise ValueError("Unsupported JSON structure")

            return self._standardize_weather_data(df, data_source)

        except Exception as e:
            self.logger.error(f"Error loading JSON file {file_path}: {e}")
            raise

    def _load_netcdf(
        self, file_path: Path, data_source: WeatherDataSource
    ) -> pd.DataFrame:
        """Load NetCDF weather data"""

        try:
            import xarray as xr

            ds = xr.open_dataset(file_path)

            # Convert to DataFrame
            df = ds.to_dataframe().reset_index()

            return self._standardize_weather_data(df, data_source)

        except ImportError:
            raise ImportError("xarray package required for NetCDF support")
        except Exception as e:
            self.logger.error(f"Error loading NetCDF file {file_path}: {e}")
            raise

    def _load_text(
        self, file_path: Path, data_source: WeatherDataSource
    ) -> pd.DataFrame:
        """Load custom text format weather data"""

        try:
            # Try common delimiters
            for delimiter in ["\t", " ", ",", ";"]:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, comment="#")
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            else:
                raise ValueError("Could not parse text file with common delimiters")

            return self._standardize_weather_data(df, data_source)

        except Exception as e:
            self.logger.error(f"Error loading text file {file_path}: {e}")
            raise

    def _standardize_weather_data(
        self, df: pd.DataFrame, data_source: WeatherDataSource
    ) -> pd.DataFrame:
        """Standardize weather data column names and formats"""

        standardized_df = df.copy()
        column_mapping = data_source.column_mapping

        # Map columns to standard names
        if column_mapping is not None:
            for standard_name, possible_names in column_mapping.items():
                for col in df.columns:
                    if col in possible_names or col.lower() in [
                        name.lower() for name in possible_names
                    ]:
                        standardized_df = standardized_df.rename(
                            columns={col: standard_name}
                        )
                        break

        # Ensure required columns exist
        required_columns = ["date", "temperature"]
        missing_columns = [
            col for col in required_columns if col not in standardized_df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert date column
        if "date" in standardized_df.columns:
            standardized_df["date"] = pd.to_datetime(standardized_df["date"])

        # Set default values for missing optional columns
        if "rainfall" not in standardized_df.columns:
            standardized_df["rainfall"] = 0.0
        if "wind_speed" not in standardized_df.columns:
            standardized_df["wind_speed"] = 5.0  # Default light wind
        if "humidity" not in standardized_df.columns:
            standardized_df["humidity"] = 60.0  # Default humidity
        if "pressure" not in standardized_df.columns:
            standardized_df["pressure"] = 1013.25  # Standard atmospheric pressure

        # Convert numeric columns
        numeric_columns = [
            "temperature",
            "rainfall",
            "wind_speed",
            "humidity",
            "pressure",
        ]
        for col in numeric_columns:
            if col in standardized_df.columns:
                standardized_df[col] = pd.to_numeric(
                    standardized_df[col], errors="coerce"
                )

        # Sort by date
        standardized_df = standardized_df.sort_values("date").reset_index(drop=True)

        return standardized_df

    def _load_from_api(self, data_source: WeatherDataSource) -> pd.DataFrame:
        """Load weather data from API"""

        if not data_source.api_config:
            raise ValueError("API configuration required for API data source")

        api_type = data_source.api_config.get("type", "openweather")

        if api_type == "openweather":
            return self._load_openweather_api(data_source)
        elif api_type == "metoffice":
            return self._load_metoffice_api(data_source)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

    def _load_openweather_api(self, data_source: WeatherDataSource) -> pd.DataFrame:
        """Load data from OpenWeather API"""

        api_config = data_source.api_config
        if api_config is None:
            raise ValueError(
                "API configuration is required for API-based weather data source"
            )
        api_key = api_config.get("api_key")

        if not api_key:
            raise ValueError("OpenWeather API key required")

        station = data_source.station_info
        if not station:
            raise ValueError("Station information required for API access")

        # Get historical weather data
        url = "http://api.openweathermap.org/data/2.5/onecall/timemachine"

        weather_data = []
        start_date = api_config.get("start_date", datetime.now() - timedelta(days=30))
        end_date = api_config.get("end_date", datetime.now())

        current_date = start_date
        while current_date <= end_date:
            params = {
                "lat": station.latitude,
                "lon": station.longitude,
                "dt": int(current_date.timestamp()),
                "appid": api_key,
                "units": "metric",
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if "current" in data:
                    weather_data.append(
                        {
                            "date": current_date,
                            "temperature": data["current"]["temp"],
                            "humidity": data["current"]["humidity"],
                            "pressure": data["current"]["pressure"],
                            "wind_speed": data["current"]["wind_speed"],
                            "rainfall": data.get("current", {})
                            .get("rain", {})
                            .get("1h", 0),
                        }
                    )

            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {current_date}: {e}")

            current_date += timedelta(days=1)

        if not weather_data:
            raise ValueError("No weather data retrieved from API")

        return pd.DataFrame(weather_data)

    def _load_metoffice_api(self, data_source: WeatherDataSource) -> pd.DataFrame:
        """Load data from Met Office API with proper error handling and fallback

        The Met Office DataHub API provides access to weather observations and forecasts.
        API documentation: https://data.hub.api.metoffice.gov.uk/
        """
        import os

        try:
            # API key should come from environment variable for security
            api_key = os.getenv("MET_OFFICE_API_KEY")
            if not api_key:
                self.logger.warning(
                    "Met Office API key not found. Set MET_OFFICE_API_KEY environment variable. "
                    "Falling back to synthetic weather data."
                )
                return self._generate_synthetic_data(data_source)

            headers = {"accept": "application/json", "x-ibm-client-id": api_key}

            # Get location from data source
            if data_source.station_info:
                lat = data_source.station_info.latitude
                lon = data_source.station_info.longitude
            else:
                # Default to London coordinates
                lat, lon = 51.5074, -0.1278
                self.logger.warning(
                    f"No coordinates provided, using London defaults: {lat}, {lon}"
                )

            # Find nearest location ID
            location_id = self._get_nearest_metoffice_location(lat, lon, headers)
            if not location_id:
                self.logger.error("Could not find Met Office location ID")
                return self._generate_synthetic_data(data_source)

            # Fetch both historical observations and forecast data
            weather_data = []

            # Get historical observations (if available)
            try:
                obs_data = self._fetch_metoffice_observations(location_id, headers)
                weather_data.extend(obs_data)
            except Exception as e:
                self.logger.warning(f"Could not fetch historical observations: {e}")

            # Get forecast data
            try:
                forecast_data = self._fetch_metoffice_forecast(location_id, headers)
                weather_data.extend(forecast_data)
            except Exception as e:
                self.logger.warning(f"Could not fetch forecast data: {e}")

            if not weather_data:
                self.logger.warning("No data received from Met Office API")
                return self._generate_synthetic_data(data_source)

            # Convert to DataFrame
            df = pd.DataFrame(weather_data)

            # Ensure required columns exist
            required_columns = [
                "date",
                "temperature",
                "rainfall",
                "wind_speed",
                "humidity",
            ]
            for col in required_columns:
                if col not in df.columns:
                    if col == "humidity" and "relative_humidity" in df.columns:
                        df["humidity"] = df["relative_humidity"]
                    elif col == "rainfall" and "precipitation_amount" in df.columns:
                        df["rainfall"] = df["precipitation_amount"]
                    else:
                        # Add default values for missing columns
                        self.logger.warning(f"Missing column '{col}', using defaults")
                        if col == "humidity":
                            df[col] = 75.0  # Default 75% humidity
                        elif col == "rainfall":
                            df[col] = 0.0
                        else:
                            df[col] = 0.0

            # Sort by date
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            self.logger.info(
                f"Successfully loaded {len(df)} weather records from Met Office API"
            )

            return df

        except requests.RequestException as e:
            self.logger.error(f"Met Office API request failed: {e}")
            self.logger.info("Falling back to synthetic weather data")
            return self._generate_synthetic_data(data_source)

        except Exception as e:
            self.logger.error(f"Unexpected error in Met Office API integration: {e}")
            self.logger.info("Falling back to synthetic weather data")
            return self._generate_synthetic_data(data_source)

    def _get_nearest_metoffice_location(
        self, lat: float, lon: float, headers: dict
    ) -> Optional[str]:
        """Find the nearest Met Office weather station location ID"""
        try:
            # Met Office site list endpoint
            sites_url = (
                "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/all-sites"
            )

            response = requests.get(sites_url, headers=headers, timeout=10)
            response.raise_for_status()

            sites = response.json()

            # Find nearest site
            min_distance = float("inf")
            nearest_id = None

            for site in sites.get("sites", []):
                site_lat = float(site.get("latitude", 0))
                site_lon = float(site.get("longitude", 0))

                # Simple distance calculation (good enough for finding nearest)
                distance = ((lat - site_lat) ** 2 + (lon - site_lon) ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    nearest_id = site.get("id")

            if nearest_id:
                self.logger.info(f"Found nearest Met Office location: {nearest_id}")

            return nearest_id

        except Exception as e:
            self.logger.error(f"Error finding nearest Met Office location: {e}")
            return None

    def _fetch_metoffice_observations(
        self, location_id: str, headers: dict
    ) -> List[Dict[str, Any]]:
        """Fetch historical weather observations from Met Office"""
        try:
            # Observations endpoint (last 24 hours)
            obs_url = f"https://data.hub.api.metoffice.gov.uk/sitespecific/v0/site/{location_id}"
            params = {"res": "hourly", "metricUnit": "metric"}

            response = requests.get(obs_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            observations = []

            # Parse observation data
            location = data.get("site", {})
            for period in data.get("siteTimeSeries", {}).get("timeSeries", []):
                obs = {
                    "date": period.get("time"),
                    "temperature": float(period.get("temperature", 0)),
                    "wind_speed": float(period.get("windSpeed", 0)),
                    "relative_humidity": float(period.get("humidity", 75)),
                    "pressure": float(period.get("pressure", 1013)),
                    "precipitation_amount": float(period.get("precipitationAmount", 0)),
                    "weather_type": period.get("significantWeatherCode", 0),
                    "location_name": location.get("name", "Unknown"),
                    "data_source": "metoffice_observation",
                }
                observations.append(obs)

            return observations

        except Exception as e:
            self.logger.warning(f"Could not fetch Met Office observations: {e}")
            return []

    def _fetch_metoffice_forecast(
        self, location_id: str, headers: dict
    ) -> List[Dict[str, Any]]:
        """Fetch weather forecast data from Met Office"""
        try:
            # Forecast endpoint
            forecast_url = f"https://data.hub.api.metoffice.gov.uk/sitespecific/v0/forecasts/{location_id}"
            params = {"res": "3hourly", "metricUnit": "metric"}

            response = requests.get(
                forecast_url, headers=headers, params=params, timeout=10
            )
            response.raise_for_status()

            data = response.json()
            forecasts = []

            # Parse forecast data
            location = data.get("location", {})
            for period in data.get("forecast", {}).get("timeSeries", []):
                forecast = {
                    "date": period.get("time"),
                    "temperature": float(period.get("temperature", 0)),
                    "wind_speed": float(period.get("windSpeed", 0)),
                    "relative_humidity": float(period.get("humidity", 75)),
                    "pressure": float(period.get("pressure", 1013)),
                    "precipitation_amount": float(
                        period.get("precipitationProbability", 0)
                    )
                    / 100.0
                    * 5.0,  # Estimate
                    "weather_type": period.get("significantWeatherCode", 0),
                    "location_name": location.get("name", "Unknown"),
                    "data_source": "metoffice_forecast",
                }
                forecasts.append(forecast)

            return forecasts

        except Exception as e:
            self.logger.warning(f"Could not fetch Met Office forecast: {e}")
            return []

    def _generate_synthetic_data(self, data_source: WeatherDataSource) -> pd.DataFrame:
        """Generate synthetic weather data"""

        station = data_source.station_info
        if not station:
            raise ValueError(
                "Station information required for synthetic data generation"
            )

        # Use WeatherGenerator to create synthetic data
        generator = WeatherGenerator(station.latitude)

        # Generate data for specified period or default 365 days
        days = 365
        weather_data = []

        for day in range(days):
            weather = generator.generate_weather(day)

            weather_data.append(
                {
                    "date": datetime.now() - timedelta(days=days - day),
                    "temperature": weather.temperature,
                    "rainfall": weather.rainfall,
                    "wind_speed": weather.wind_speed,
                    "humidity": weather.humidity,
                    "pressure": weather.pressure,
                }
            )

        return pd.DataFrame(weather_data)


class WeatherIntegrationManager:
    """
    Manages integration of weather data with BSTEW simulation.

    Handles:
    - Multiple weather data sources
    - Data interpolation and gap filling
    - Real-time weather updates
    - Weather forecast integration
    """

    def __init__(self) -> None:
        self.file_loader = WeatherFileLoader()
        self.weather_sources: List[WeatherDataSource] = []
        self.cached_weather_data: Dict[str, pd.DataFrame] = {}
        self.logger = logging.getLogger(__name__)

    def add_weather_source(self, data_source: WeatherDataSource) -> None:
        """Add a weather data source"""
        self.weather_sources.append(data_source)
        self.logger.info(f"Added weather source: {data_source.source_type}")

    def load_all_weather_data(self) -> Dict[str, pd.DataFrame]:
        """Load weather data from all configured sources"""

        for i, source in enumerate(self.weather_sources):
            source_id = f"source_{i}"

            try:
                self.logger.info(f"Loading weather data from {source.source_type}")
                weather_data = self.file_loader.load_weather_data(source)
                self.cached_weather_data[source_id] = weather_data

                self.logger.info(
                    f"Loaded {len(weather_data)} weather records from {source.source_type}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to load weather data from {source.source_type}: {e}"
                )

        return self.cached_weather_data

    def get_weather_for_simulation(
        self,
        start_date: datetime,
        duration_days: int,
        preferred_source: Optional[str] = None,
    ) -> List[WeatherConditions]:
        """Get weather conditions for simulation period"""

        if not self.cached_weather_data:
            self.load_all_weather_data()

        # Select data source
        if preferred_source and preferred_source in self.cached_weather_data:
            weather_df = self.cached_weather_data[preferred_source]
        else:
            # Use first available source
            if not self.cached_weather_data:
                raise ValueError("No weather data available")
            weather_df = list(self.cached_weather_data.values())[0]

        # Generate weather conditions for simulation period
        weather_conditions = []

        for day in range(duration_days):
            simulation_date = start_date + timedelta(days=day)

            # Find closest weather data
            weather_row = self._find_closest_weather_data(weather_df, simulation_date)

            if weather_row is not None:
                weather_conditions.append(
                    self._create_weather_conditions(weather_row, day)
                )
            else:
                # Generate synthetic weather if no data available
                generator = WeatherGenerator()
                weather_conditions.append(generator.generate_weather(day))

        return weather_conditions

    def _find_closest_weather_data(
        self, weather_df: pd.DataFrame, target_date: datetime
    ) -> Optional[pd.Series]:
        """Find closest weather data to target date"""

        if "date" not in weather_df.columns:
            return None

        # Calculate time differences
        time_diffs = abs(weather_df["date"] - target_date)
        closest_idx = time_diffs.idxmin()

        # Only use data within reasonable time window (e.g., 7 days)
        if time_diffs.iloc[int(closest_idx)] <= timedelta(days=7):
            return weather_df.iloc[int(closest_idx)]
        else:
            return None

    def _create_weather_conditions(
        self, weather_row: pd.Series, day_of_year: int
    ) -> WeatherConditions:
        """Create WeatherConditions from weather data row"""

        # Determine weather type from conditions
        weather_type = self._determine_weather_type(
            weather_row.get("temperature", 15),
            weather_row.get("rainfall", 0),
            weather_row.get("wind_speed", 5),
        )

        # Determine season
        season = self._determine_season(day_of_year)

        # Calculate daylight hours (simplified)
        daylight_hours = 12 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        daylight_hours = max(6, min(18, daylight_hours))

        return WeatherConditions(
            temperature=weather_row.get("temperature", 15),
            rainfall=weather_row.get("rainfall", 0),
            wind_speed=weather_row.get("wind_speed", 5),
            humidity=weather_row.get("humidity", 60),
            pressure=weather_row.get("pressure", 1013.25),
            daylight_hours=daylight_hours,
            weather_type=weather_type,
            season=season,
            day_of_year=day_of_year % 365,
        )

    def _determine_weather_type(
        self, temperature: float, rainfall: float, wind_speed: float
    ) -> WeatherType:
        """Determine weather type from conditions"""

        if temperature < 0 and rainfall > 0:
            return WeatherType.SNOW
        elif rainfall > 10:
            return WeatherType.HEAVY_RAIN
        elif rainfall > 2:
            return WeatherType.LIGHT_RAIN
        elif wind_speed > 25:
            return WeatherType.WINDY
        elif rainfall > 0 and wind_speed > 15:
            return WeatherType.THUNDERSTORM
        elif rainfall == 0 and np.random.random() < 0.3:
            return WeatherType.CLOUDY
        else:
            return WeatherType.CLEAR

    def _determine_season(self, day_of_year: int) -> SeasonType:
        """Determine season from day of year"""
        day_of_year = day_of_year % 365

        if 80 <= day_of_year < 172:
            return SeasonType.SPRING
        elif 172 <= day_of_year < 266:
            return SeasonType.SUMMER
        elif 266 <= day_of_year < 355:
            return SeasonType.AUTUMN
        else:
            return SeasonType.WINTER

    def get_weather_summary(self) -> Dict[str, Any]:
        """Get summary of loaded weather data"""

        summary: Dict[str, Any] = {
            "total_sources": len(self.weather_sources),
            "loaded_sources": len(self.cached_weather_data),
            "source_details": {},
        }

        for source_id, weather_df in self.cached_weather_data.items():
            if len(weather_df) > 0:
                summary["source_details"][source_id] = {
                    "records": len(weather_df),
                    "date_range": {
                        "start": (
                            weather_df["date"].min().isoformat()
                            if "date" in weather_df.columns
                            else None
                        ),
                        "end": (
                            weather_df["date"].max().isoformat()
                            if "date" in weather_df.columns
                            else None
                        ),
                    },
                    "temperature_range": {
                        "min": (
                            weather_df["temperature"].min()
                            if "temperature" in weather_df.columns
                            else None
                        ),
                        "max": (
                            weather_df["temperature"].max()
                            if "temperature" in weather_df.columns
                            else None
                        ),
                    },
                    "total_rainfall": (
                        weather_df["rainfall"].sum()
                        if "rainfall" in weather_df.columns
                        else None
                    ),
                }

        return summary


def create_weather_source_from_config(config: Dict[str, Any]) -> WeatherDataSource:
    """Create weather data source from configuration"""

    source_type = config.get("type", "file")

    station_info = None
    if "station" in config:
        station_config = config["station"]
        station_info = WeatherStation(
            station_id=station_config.get("id", "unknown"),
            name=station_config.get("name", "Unknown Station"),
            latitude=station_config.get("latitude", 0.0),
            longitude=station_config.get("longitude", 0.0),
            elevation=station_config.get("elevation", 0.0),
            data_source=source_type,
            active_period=(
                datetime.fromisoformat(station_config.get("start_date", "2020-01-01")),
                datetime.fromisoformat(station_config.get("end_date", "2023-12-31")),
            ),
        )

    return WeatherDataSource(
        source_type=source_type,
        file_path=config.get("file_path"),
        api_config=config.get("api_config"),
        station_info=station_info,
        column_mapping=config.get("column_mapping"),
    )
