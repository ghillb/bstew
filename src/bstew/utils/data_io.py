"""
Data Input/Output utilities for BSTEW
=====================================

Handles CSV, Excel, image, and other data format processing.
Interfaces with NetLogo data files and external datasets.
"""

import pandas as pd
import numpy as np
from PIL import Image
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from pydantic import BaseModel, Field
import re

from ..spatial.patches import HabitatType, FlowerSpecies


class WeatherData(BaseModel):
    """Weather data structure"""

    model_config = {"validate_assignment": True}

    date: str = Field(description="Date string")
    temperature: float = Field(description="Temperature in °C")
    rainfall: float = Field(ge=0.0, description="Rainfall in mm")
    wind_speed: float = Field(ge=0.0, description="Wind speed in mph")
    humidity: float = Field(ge=0.0, le=100.0, description="Humidity percentage")
    pressure: float = Field(ge=0.0, description="Atmospheric pressure in hPa")
    daylight_hours: float = Field(ge=0.0, le=24.0, description="Daylight hours")


class ParameterSet(BaseModel):
    """Parameter set for species or experiments"""

    model_config = {"validate_assignment": True}

    name: str = Field(description="Parameter set name")
    species: str = Field(description="Species name")
    parameters: Dict[str, Any] = Field(description="Parameter dictionary")
    source: str = Field(description="Data source")
    description: str = Field(description="Parameter set description")


class DataLoader:
    """
    Data loading utilities for various file formats.

    Handles:
    - Weather data from CSV files
    - Landscape maps from images
    - Species parameters from CSV/Excel
    - NetLogo parameter files
    - Experimental data
    """

    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)

        # Supported file formats
        self.image_formats = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        self.data_formats = {".csv", ".xlsx", ".xls", ".json", ".yaml", ".yml"}

    def load_weather_data(self, filepath: str) -> List[WeatherData]:
        """Load weather data from CSV file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Weather file not found: {filepath}")

        try:
            df = pd.read_csv(file_path)

            # Handle different column naming conventions
            column_mapping = self._detect_weather_columns(list(df.columns))

            weather_data = []
            for _, row in df.iterrows():
                weather = WeatherData(
                    date=str(row.get(column_mapping.get("date", "date"), "")),
                    temperature=float(
                        row.get(column_mapping.get("temperature", "temp"), 15.0)
                    ),
                    rainfall=float(
                        row.get(column_mapping.get("rainfall", "rain"), 0.0)
                    ),
                    wind_speed=float(
                        row.get(column_mapping.get("wind_speed", "wind"), 5.0)
                    ),
                    humidity=float(
                        row.get(column_mapping.get("humidity", "humidity"), 60.0)
                    ),
                    pressure=float(
                        row.get(column_mapping.get("pressure", "pressure"), 1013.0)
                    ),
                    daylight_hours=float(
                        row.get(column_mapping.get("daylight", "daylight"), 12.0)
                    ),
                )
                weather_data.append(weather)

            self.logger.info(
                f"Loaded {len(weather_data)} weather records from {filepath}"
            )
            return weather_data

        except Exception as e:
            raise ValueError(f"Error loading weather data from {filepath}: {e}")

    def _detect_weather_columns(self, columns: List[str]) -> Dict[str, str]:
        """Detect weather data column names"""
        column_patterns = {
            "date": ["date", "time", "day", "datetime"],
            "temperature": ["temp", "temperature", "temp_c", "temperature_c"],
            "rainfall": ["rain", "rainfall", "precip", "precipitation"],
            "wind_speed": ["wind", "wind_speed", "windspeed", "wind_mph"],
            "humidity": ["humidity", "humid", "rh", "relative_humidity"],
            "pressure": ["pressure", "press", "hpa", "pressure_hpa"],
            "daylight": ["daylight", "daylight_hours", "sunshine", "sun_hours"],
        }

        mapping = {}
        columns_lower = [col.lower() for col in columns]

        for key, patterns in column_patterns.items():
            for pattern in patterns:
                for i, col in enumerate(columns_lower):
                    if pattern in col:
                        mapping[key] = columns[i]
                        break
                if key in mapping:
                    break

        return mapping

    def load_landscape_image(self, filepath: str) -> np.ndarray:
        """Load landscape image and return RGB array"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Landscape image not found: {filepath}")

        if file_path.suffix.lower() not in self.image_formats:
            raise ValueError(f"Unsupported image format: {file_path.suffix}")

        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                rgb_array = np.array(img)

            self.logger.info(
                f"Loaded landscape image: {rgb_array.shape} from {filepath}"
            )
            return rgb_array

        except Exception as e:
            raise ValueError(f"Error loading landscape image from {filepath}: {e}")

    def load_species_parameters(self, filepath: str) -> List[ParameterSet]:
        """Load species parameters from CSV or Excel file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Species parameter file not found: {filepath}")

        try:
            # Load based on file extension
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(
                    f"Unsupported parameter file format: {file_path.suffix}"
                )

            parameter_sets = []

            for _, row in df.iterrows():
                params = {}
                for col in df.columns:
                    if col not in ["name", "species", "source", "description"]:
                        params[col] = row[col]

                param_set = ParameterSet(
                    name=str(row.get("name", "unknown")),
                    species=str(row.get("species", "unknown")),
                    parameters=params,
                    source=str(row.get("source", filepath)),
                    description=str(row.get("description", "")),
                )
                parameter_sets.append(param_set)

            self.logger.info(
                f"Loaded {len(parameter_sets)} parameter sets from {filepath}"
            )
            return parameter_sets

        except Exception as e:
            raise ValueError(f"Error loading species parameters from {filepath}: {e}")

    def load_flower_species_database(self, filepath: str) -> Dict[str, FlowerSpecies]:
        """Load flower species database from CSV file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Flower species file not found: {filepath}")

        try:
            df = pd.read_csv(file_path)

            species_db = {}

            for _, row in df.iterrows():
                species = FlowerSpecies(
                    name=str(row["name"]),
                    bloom_start=int(row["bloom_start"]),
                    bloom_end=int(row["bloom_end"]),
                    nectar_production=float(row["nectar_production"]),
                    pollen_production=float(row["pollen_production"]),
                    flower_density=float(row["flower_density"]),
                    attractiveness=float(row["attractiveness"]),
                )
                species_db[species.name] = species

            self.logger.info(f"Loaded {len(species_db)} flower species from {filepath}")
            return species_db

        except Exception as e:
            raise ValueError(f"Error loading flower species from {filepath}: {e}")

    def load_netlogo_parameters(self, filepath: str) -> Dict[str, Any]:
        """Load NetLogo-style parameter files"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"NetLogo parameter file not found: {filepath}")

        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)

                # Convert to dictionary format
                params = {}
                for _, row in df.iterrows():
                    if "parameter" in df.columns and "value" in df.columns:
                        params[row["parameter"]] = row["value"]
                    else:
                        # Use first column as key, second as value
                        key = str(row.iloc[0])
                        value = row.iloc[1]
                        params[key] = value

            elif file_path.suffix.lower() in [".txt"]:
                # Handle NetLogo text format
                params = {}
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            key, value = line.split("=", 1)
                            params[key.strip()] = self._parse_netlogo_value(
                                value.strip()
                            )

            else:
                raise ValueError(
                    f"Unsupported NetLogo parameter format: {file_path.suffix}"
                )

            self.logger.info(f"Loaded {len(params)} NetLogo parameters from {filepath}")
            return params

        except Exception as e:
            raise ValueError(f"Error loading NetLogo parameters from {filepath}: {e}")

    def _parse_netlogo_value(self, value_str: str) -> Any:
        """Parse NetLogo parameter values"""
        value_str = value_str.strip()

        # Try to parse as number
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # Try to parse as boolean
        if value_str.lower() in ["true", "false"]:
            return value_str.lower() == "true"

        # Return as string
        return value_str.strip("\"'")

    def load_color_mapping(
        self, filepath: str
    ) -> Dict[Tuple[int, int, int], HabitatType]:
        """Load color to habitat mapping from file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Color mapping file not found: {filepath}")

        try:
            if file_path.suffix.lower() in [".json"]:
                with open(file_path, "r") as f:
                    data = json.load(f)

            elif file_path.suffix.lower() in [".yaml", ".yml"]:
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)

            elif file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
                data = {}
                for _, row in df.iterrows():
                    color_key = f"({row['r']}, {row['g']}, {row['b']})"
                    data[color_key] = row["habitat"]

            else:
                raise ValueError(
                    f"Unsupported color mapping format: {file_path.suffix}"
                )

            # Convert to proper format
            color_mapping = {}
            for color_str, habitat_str in data.items():
                # Parse color tuple from string
                if isinstance(color_str, str):
                    rgb_match = re.match(r"\((\d+),\s*(\d+),\s*(\d+)\)", color_str)
                    if rgb_match:
                        r, g, b = map(int, rgb_match.groups())
                        color = (r, g, b)
                    else:
                        continue
                else:
                    color = tuple(color_str)

                # Convert habitat string to enum
                try:
                    habitat = HabitatType(habitat_str.lower())
                    color_mapping[color] = habitat
                except ValueError:
                    self.logger.warning(f"Unknown habitat type: {habitat_str}")

            self.logger.info(
                f"Loaded {len(color_mapping)} color mappings from {filepath}"
            )
            return color_mapping

        except Exception as e:
            raise ValueError(f"Error loading color mapping from {filepath}: {e}")


class DataExporter:
    """
    Data export utilities for simulation results and analysis.

    Handles:
    - CSV export for analysis
    - Excel reports with multiple sheets
    - JSON data for web interfaces
    - Image export for visualizations
    - NetLogo-compatible formats
    """

    def __init__(self, output_dir: str = "artifacts/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def export_model_data(
        self, model_data: pd.DataFrame, filename: str = "model_data.csv"
    ) -> str:
        """Export model-level data to CSV"""
        output_path = self.output_dir / filename

        try:
            model_data.to_csv(output_path, index=False)
            self.logger.info(f"Exported model data to {output_path}")
            return str(output_path)

        except Exception as e:
            raise ValueError(f"Error exporting model data: {e}")

    def export_agent_data(
        self, agent_data: pd.DataFrame, filename: str = "agent_data.csv"
    ) -> str:
        """Export agent-level data to CSV"""
        output_path = self.output_dir / filename

        try:
            agent_data.to_csv(output_path, index=False)
            self.logger.info(f"Exported agent data to {output_path}")
            return str(output_path)

        except Exception as e:
            raise ValueError(f"Error exporting agent data: {e}")

    def export_to_csv(self, data: pd.DataFrame, output_path: Path) -> str:
        """Export DataFrame to CSV file"""
        try:
            data.to_csv(output_path, index=False)
            self.logger.info(f"Exported data to {output_path}")
            return str(output_path)
        except Exception as e:
            raise ValueError(f"Error exporting to CSV: {e}")

    def export_to_json(self, data: Dict[str, Any], output_path: Path) -> str:
        """Export dictionary data to JSON file"""
        import json

        try:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"Exported data to {output_path}")
            return str(output_path)
        except Exception as e:
            raise ValueError(f"Error exporting to JSON: {e}")

    def export_spatial_data(
        self, landscape_grid: Any, filename: str = "spatial_data.json"
    ) -> str:
        """Export spatial/landscape data to JSON"""
        output_path = self.output_dir / filename

        try:
            spatial_data = landscape_grid.export_to_dict()

            with open(output_path, "w") as f:
                json.dump(spatial_data, f, indent=2)

            self.logger.info(f"Exported spatial data to {output_path}")
            return str(output_path)

        except Exception as e:
            raise ValueError(f"Error exporting spatial data: {e}")

    def export_excel_report(
        self,
        data_dict: Dict[str, pd.DataFrame],
        filename: str = "simulation_report.xlsx",
    ) -> str:
        """Export multi-sheet Excel report"""
        output_path = self.output_dir / filename

        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                for sheet_name, data in data_dict.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)

            self.logger.info(f"Exported Excel report to {output_path}")
            return str(output_path)

        except Exception as e:
            raise ValueError(f"Error exporting Excel report: {e}")

    def export_summary_stats(
        self, model_data: pd.DataFrame, filename: str = "summary_stats.json"
    ) -> str:
        """Export summary statistics to JSON"""
        output_path = self.output_dir / filename

        try:
            # Calculate summary statistics
            stats: Dict[str, Any] = {
                "simulation_duration": len(model_data),
                "final_day": (
                    model_data["Day"].max() if "Day" in model_data.columns else 0
                ),
                "population_stats": {},
                "resource_stats": {},
            }

            # Population statistics
            if "Total_Bees" in model_data.columns:
                stats["population_stats"] = {
                    "final_population": int(model_data["Total_Bees"].iloc[-1]),
                    "max_population": int(model_data["Total_Bees"].max()),
                    "min_population": int(model_data["Total_Bees"].min()),
                    "mean_population": float(model_data["Total_Bees"].mean()),
                }

            # Resource statistics
            if "Total_Honey" in model_data.columns:
                stats["resource_stats"] = {
                    "final_honey": float(model_data["Total_Honey"].iloc[-1]),
                    "max_honey": float(model_data["Total_Honey"].max()),
                    "total_honey_produced": float(model_data["Total_Honey"].sum()),
                }

            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)

            self.logger.info(f"Exported summary statistics to {output_path}")
            return str(output_path)

        except Exception as e:
            raise ValueError(f"Error exporting summary statistics: {e}")

    def export_landscape_image(
        self, landscape_grid: Any, filename: str = "landscape_map.png"
    ) -> str:
        """Export landscape as image"""
        output_path = self.output_dir / filename

        try:
            # Create RGB array from habitat grid
            height, width = landscape_grid.habitat_grid.shape
            rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

            # Color mapping for habitats
            habitat_colors = {
                0: (0, 255, 0),  # Grassland - Green
                1: (255, 255, 0),  # Cropland - Yellow
                2: (255, 0, 255),  # Wildflower - Magenta
                3: (128, 64, 0),  # Woodland - Brown
                4: (0, 128, 0),  # Hedgerow - Dark green
                5: (128, 128, 128),  # Urban - Gray
                6: (0, 0, 255),  # Water - Blue
                7: (255, 255, 255),  # Bare soil - White
                8: (64, 64, 64),  # Road - Dark gray
                9: (255, 0, 0),  # Building - Red
            }

            for habitat_id, color in habitat_colors.items():
                mask = landscape_grid.habitat_grid == habitat_id
                rgb_array[mask] = color

            # Create and save image
            image = Image.fromarray(rgb_array, "RGB")
            image.save(output_path)

            self.logger.info(f"Exported landscape image to {output_path}")
            return str(output_path)

        except Exception as e:
            raise ValueError(f"Error exporting landscape image: {e}")

    def export_netlogo_format(
        self, data: Dict[str, Any], filename: str = "netlogo_export.txt"
    ) -> str:
        """Export data in NetLogo-compatible format"""
        output_path = self.output_dir / filename

        try:
            with open(output_path, "w") as f:
                for key, value in data.items():
                    if isinstance(value, (list, tuple)):
                        value_str = " ".join(map(str, value))
                        f.write(f"{key} = [{value_str}]\n")
                    elif isinstance(value, bool):
                        f.write(f"{key} = {str(value).lower()}\n")
                    else:
                        f.write(f"{key} = {value}\n")

            self.logger.info(f"Exported NetLogo format data to {output_path}")
            return str(output_path)

        except Exception as e:
            raise ValueError(f"Error exporting NetLogo format: {e}")

    def create_data_package(
        self,
        model_data: pd.DataFrame,
        agent_data: pd.DataFrame,
        landscape_grid: Any,
        metadata: Dict[str, Any],
        package_name: str = "simulation_package",
    ) -> str:
        """Create complete data package with all outputs"""
        package_dir = self.output_dir / package_name
        package_dir.mkdir(exist_ok=True)

        try:
            # Export all data formats
            self.export_model_data(model_data, str(package_dir / "model_data.csv"))
            self.export_agent_data(agent_data, str(package_dir / "agent_data.csv"))
            self.export_spatial_data(
                landscape_grid, str(package_dir / "spatial_data.json")
            )
            self.export_summary_stats(
                model_data, str(package_dir / "summary_stats.json")
            )
            self.export_landscape_image(
                landscape_grid, str(package_dir / "landscape_map.png")
            )

            # Create Excel report
            excel_data = {
                "Model_Data": model_data.head(1000),  # Limit size
                "Summary_Stats": pd.DataFrame([metadata]),
            }
            if not agent_data.empty:
                excel_data["Agent_Sample"] = agent_data.head(1000)

            with pd.ExcelWriter(
                package_dir / "report.xlsx", engine="openpyxl"
            ) as writer:
                for sheet_name, data in excel_data.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)

            # Export metadata
            with open(package_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Created data package: {package_dir}")
            return str(package_dir)

        except Exception as e:
            raise ValueError(f"Error creating data package: {e}")


class FileValidator:
    """Validate input files and data formats"""

    @staticmethod
    def validate_weather_file(filepath: str) -> List[str]:
        """Validate weather data file format"""
        errors = []

        try:
            df = pd.read_csv(filepath)

            # Check required columns (flexible naming)
            required_patterns = ["temp", "rain", "wind"]

            columns_lower = [col.lower() for col in df.columns]

            for pattern in required_patterns:
                if not any(pattern in col for col in columns_lower):
                    errors.append(f"Missing weather column matching pattern: {pattern}")

            # Check data ranges
            for col in df.columns:
                col_lower = col.lower()
                if "temp" in col_lower:
                    if df[col].min() < -50 or df[col].max() > 60:
                        errors.append(
                            f"Temperature values out of range: {df[col].min()} to {df[col].max()}"
                        )

                elif "rain" in col_lower:
                    if df[col].min() < 0 or df[col].max() > 500:
                        errors.append(
                            f"Rainfall values out of range: {df[col].min()} to {df[col].max()}"
                        )

        except Exception as e:
            errors.append(f"Error reading weather file: {e}")

        return errors

    @staticmethod
    def validate_landscape_image(filepath: str) -> List[str]:
        """Validate landscape image file"""
        errors = []

        try:
            with Image.open(filepath) as img:
                # Check dimensions
                if img.width < 10 or img.height < 10:
                    errors.append(f"Image too small: {img.width}x{img.height}")

                if img.width > 10000 or img.height > 10000:
                    errors.append(f"Image too large: {img.width}x{img.height}")

                # Check format
                if img.mode not in ["RGB", "RGBA", "L"]:
                    errors.append(f"Unsupported image mode: {img.mode}")

        except Exception as e:
            errors.append(f"Error reading landscape image: {e}")

        return errors
