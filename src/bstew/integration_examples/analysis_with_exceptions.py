"""
Analysis Module with Integrated Exception Handling
==================================================

Example showing how to integrate standardized exceptions into analysis commands.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from bstew.exceptions import (
    AnalysisError,
    DataError,
    FileSystemError,
    ValidationError,
    raise_validation_error,
    raise_data_error,
)


class AnalysisEngineWithExceptions:
    """Enhanced analysis engine with standardized exception handling."""

    def __init__(self, output_dir: str = "artifacts/analysis"):
        """Initialize analysis engine with error handling."""
        try:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FileSystemError(
                "Failed to create analysis output directory",
                {"directory": output_dir, "error": str(e)},
            ) from e

    def analyze_simulation_results(
        self, data_path: str, analysis_type: str = "summary"
    ) -> Dict[str, Any]:
        """Analyze simulation results with comprehensive error handling."""
        # Validate analysis type
        valid_types = ["summary", "detailed", "population", "foraging", "health"]
        if analysis_type not in valid_types:
            raise_validation_error(
                f"Invalid analysis type. Must be one of: {valid_types}",
                field="analysis_type",
                value=analysis_type,
            )

        # Load data with error handling
        try:
            data = self._load_simulation_data(data_path)
        except (DataError, FileSystemError):
            raise  # Re-raise data loading errors
        except Exception as e:
            raise AnalysisError(
                "Unexpected error loading simulation data",
                {"data_path": data_path, "error": str(e)},
            ) from e

        # Validate data before analysis
        self._validate_data_for_analysis(data, analysis_type)

        # Perform analysis
        try:
            if analysis_type == "summary":
                return self._analyze_summary(data)
            elif analysis_type == "population":
                return self._analyze_population(data)
            elif analysis_type == "foraging":
                return self._analyze_foraging(data)
            elif analysis_type == "health":
                return self._analyze_health(data)
            else:
                return self._analyze_detailed(data)

        except AnalysisError:
            raise  # Re-raise analysis errors
        except Exception as e:
            raise AnalysisError(
                f"Analysis failed for type '{analysis_type}'",
                {
                    "analysis_type": analysis_type,
                    "data_shape": data.shape,
                    "error": str(e),
                    "data_columns": list(data.columns),
                },
            ) from e

    def _load_simulation_data(self, data_path: str) -> pd.DataFrame:
        """Load simulation data with error handling."""
        path = Path(data_path)

        # Check file exists
        if not path.exists():
            raise FileSystemError(
                "Simulation data file not found",
                {
                    "file_path": str(data_path),
                    "cwd": str(Path.cwd()),
                    "search_paths": [".", "artifacts/results", "output"],
                },
            )

        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            raise DataError(
                "Simulation data file is empty",
                {"file_path": str(data_path), "size_bytes": 0},
            )

        if file_size > 1e9:  # 1GB
            raise DataError(
                "Simulation data file too large for analysis",
                {
                    "file_path": str(data_path),
                    "size_gb": file_size / 1e9,
                    "max_size_gb": 1.0,
                },
            )

        # Load data based on extension
        try:
            if path.suffix == ".csv":
                return pd.read_csv(path)
            elif path.suffix == ".json":
                return pd.read_json(path)
            elif path.suffix in [".xlsx", ".xls"]:
                return pd.read_excel(path)
            else:
                raise DataError(
                    f"Unsupported file format: {path.suffix}",
                    {
                        "file_path": str(path),
                        "extension": path.suffix,
                        "supported": [".csv", ".json", ".xlsx", ".xls"],
                    },
                )

        except pd.errors.EmptyDataError:
            raise DataError("No data found in file", {"file_path": str(path)})
        except pd.errors.ParserError as e:
            raise_data_error(f"Failed to parse data file: {e}", file_path=str(path))
            # This line is unreachable but satisfies type checker
            return pd.DataFrame()
        except Exception as e:
            raise DataError(
                "Failed to load data file",
                {
                    "file_path": str(path),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

    def _validate_data_for_analysis(
        self, data: pd.DataFrame, analysis_type: str
    ) -> None:
        """Validate data has required columns for analysis type."""
        if data.empty:
            raise AnalysisError(
                "Cannot analyze empty dataset", {"rows": 0, "columns": 0}
            )

        # Define required columns for each analysis type
        required_columns = {
            "summary": ["step", "total_population"],
            "population": ["step", "total_population", "queens", "workers", "drones"],
            "foraging": [
                "step",
                "foragers_active",
                "nectar_collected",
                "pollen_collected",
            ],
            "health": ["step", "colony_health", "disease_level", "mortality_rate"],
            "detailed": ["step"],  # Minimal requirement
        }

        required = required_columns.get(analysis_type, [])
        missing = [col for col in required if col not in data.columns]

        if missing:
            raise AnalysisError(
                f"Missing required columns for {analysis_type} analysis",
                {
                    "analysis_type": analysis_type,
                    "required_columns": required,
                    "missing_columns": missing,
                    "available_columns": list(data.columns),
                },
            )

        # Check for valid data types
        numeric_columns = [col for col in required if col != "step"]
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                raise AnalysisError(
                    f"Column '{col}' must contain numeric data",
                    {
                        "column": col,
                        "actual_dtype": str(data[col].dtype),
                        "sample_values": data[col].head(5).tolist(),
                    },
                )

    def _analyze_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform summary analysis with error handling."""
        try:
            summary = {
                "total_steps": len(data),
                "duration_days": data["step"].max() if "step" in data.columns else 0,
                "population": {
                    "initial": float(data["total_population"].iloc[0]),
                    "final": float(data["total_population"].iloc[-1]),
                    "mean": float(data["total_population"].mean()),
                    "std": float(data["total_population"].std()),
                    "min": float(data["total_population"].min()),
                    "max": float(data["total_population"].max()),
                },
            }

            # Add growth rate if we have enough data
            if len(data) > 1:
                population_data = summary["population"]
                if isinstance(population_data, dict):
                    growth_rate = (
                        population_data["final"] - population_data["initial"]
                    ) / population_data["initial"]
                    population_data["growth_rate"] = float(growth_rate)

            return summary

        except KeyError as e:
            raise AnalysisError(
                f"Missing expected column in data: {e}",
                {"missing_column": str(e), "available_columns": list(data.columns)},
            ) from e
        except Exception as e:
            raise AnalysisError(
                f"Summary analysis failed: {e}",
                {"data_shape": data.shape, "error_type": type(e).__name__},
            ) from e

    def _analyze_population(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze population dynamics with error handling."""
        try:
            # Calculate population metrics
            castes = ["queens", "workers", "drones"]
            population_analysis: Dict[str, Any] = {
                "total_population": self._calculate_time_series_stats(
                    data["total_population"]
                ),
                "by_caste": {},
            }

            for caste in castes:
                if caste in data.columns:
                    caste_stats = self._calculate_time_series_stats(data[caste])
                    population_analysis["by_caste"][caste] = caste_stats
                else:
                    population_analysis["by_caste"][caste] = {
                        "error": f"No data for {caste}"
                    }

            # Detect population crashes
            crashes = self._detect_population_crashes(data["total_population"])
            if crashes:
                population_analysis["crashes_detected"] = crashes

            return population_analysis

        except Exception as e:
            raise AnalysisError(
                "Population analysis failed",
                {"error": str(e), "available_columns": list(data.columns)},
            ) from e

    def _calculate_time_series_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for a time series."""
        try:
            # Handle empty or all-NaN series
            if series.empty or series.isna().all():
                raise AnalysisError(
                    "Cannot calculate statistics for empty series",
                    {"series_name": series.name},
                )

            # Remove NaN values
            clean_series = series.dropna()
            if len(clean_series) == 0:
                raise AnalysisError(
                    "No valid data points after removing NaN values",
                    {"series_name": series.name, "original_length": len(series)},
                )

            stats: Dict[str, Any] = {
                "mean": float(clean_series.mean()),
                "std": float(clean_series.std()),
                "min": float(clean_series.min()),
                "max": float(clean_series.max()),
                "trend": (
                    "increasing"
                    if clean_series.iloc[-1] > clean_series.iloc[0]
                    else "decreasing"
                ),
                "volatility": float(clean_series.std() / clean_series.mean())
                if clean_series.mean() != 0
                else float("inf"),
            }
            return stats

        except Exception as e:
            raise AnalysisError(
                "Failed to calculate time series statistics",
                {"error": str(e), "series_length": len(series)},
            ) from e

    def _detect_population_crashes(
        self, population: pd.Series, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect significant population drops."""
        crashes = []

        try:
            # Calculate day-over-day changes
            changes = population.pct_change()

            # Find crashes (drops greater than threshold)
            crash_indices = changes[changes < -threshold].index

            for idx in crash_indices:
                if idx > 0:  # Skip first index
                    crashes.append(
                        {
                            "step": int(idx),
                            "population_before": float(population.iloc[idx - 1]),
                            "population_after": float(population.iloc[idx]),
                            "percent_drop": float(changes.iloc[idx] * 100),
                        }
                    )

            return crashes

        except Exception as e:
            # Log error but don't fail the entire analysis
            return [{"error": f"Crash detection failed: {e}"}]

    def save_analysis_results(
        self, results: Dict[str, Any], output_filename: str
    ) -> Path:
        """Save analysis results with error handling."""
        import json

        output_path = self.output_dir / output_filename

        try:
            # Ensure results are JSON serializable
            serializable_results = self._make_json_serializable(results)

            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2)

            # Verify file was created
            if not output_path.exists():
                raise FileSystemError(
                    "Analysis results file was not created",
                    {"output_path": str(output_path)},
                )

            return output_path

        except TypeError as e:
            raise AnalysisError(
                "Analysis results contain non-serializable data",
                {"error": str(e), "result_keys": list(results.keys())},
            ) from e
        except OSError as e:
            raise FileSystemError(
                "Failed to save analysis results",
                {
                    "output_path": str(output_path),
                    "error": str(e),
                    "disk_space": self._get_available_disk_space(),
                },
            ) from e

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas objects to JSON-serializable format."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _get_available_disk_space(self) -> str:
        """Get available disk space for error messages."""
        try:
            import shutil

            stat = shutil.disk_usage(self.output_dir)
            return f"{stat.free / (1024**3):.1f}GB"
        except Exception:
            return "unknown"

    def _analyze_foraging(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Placeholder for foraging analysis."""
        raise NotImplementedError("Foraging analysis not yet implemented")

    def _analyze_health(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Placeholder for health analysis."""
        raise NotImplementedError("Health analysis not yet implemented")

    def _analyze_detailed(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Placeholder for detailed analysis."""
        raise NotImplementedError("Detailed analysis not yet implemented")


# Example usage showing exception handling
if __name__ == "__main__":
    analyzer = AnalysisEngineWithExceptions()

    # Example 1: Missing file
    try:
        results = analyzer.analyze_simulation_results("missing_file.csv")
    except FileSystemError as e:
        print(f"File error: {e}")
        print(f"Search paths: {e.details.get('search_paths', [])}")

    # Example 2: Invalid analysis type
    try:
        results = analyzer.analyze_simulation_results("data.csv", "invalid_type")
    except ValidationError as e:
        print(f"Validation error: {e}")
        print(f"Valid types: {e.details.get('value', [])}")

    # Example 3: Empty data file
    try:
        # Create empty file
        Path("empty.csv").touch()
        results = analyzer.analyze_simulation_results("empty.csv")
    except DataError as e:
        print(f"Data error: {e}")
        print(f"File size: {e.details.get('size_bytes', 'unknown')} bytes")
