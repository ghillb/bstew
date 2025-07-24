"""
Exception Integration Examples
==============================

This file demonstrates how to integrate the standardized exception hierarchy
into existing BSTEW modules. These are examples - actual integration would
modify the existing files.
"""

from typing import Dict, Any
import pandas as pd
from pathlib import Path

from bstew.exceptions import (
    ValidationError,
    SimulationError,
    DataError,
    ConfigurationError,
    AnalysisError,
    IntegrationError,
    FileSystemError,
    ParameterError,
    ResourceError,
    StateError,
    raise_validation_error,
    raise_data_error,
    raise_config_error,
)


# Example 1: Configuration validation with proper exceptions
def validate_simulation_config(config: Dict[str, Any]) -> None:
    """
    Example of using ConfigurationError and ValidationError in config validation.

    This would replace generic ValueError/TypeError with specific exceptions.
    """
    # Check required sections
    required_sections = ["simulation", "colony", "environment"]
    missing_sections = [s for s in required_sections if s not in config]

    if missing_sections:
        raise ConfigurationError(
            f"Missing required configuration sections: {missing_sections}",
            {
                "missing_sections": missing_sections,
                "required_sections": required_sections,
            },
        )

    # Validate simulation duration
    duration = config.get("simulation", {}).get("duration_days")
    if duration is None:
        raise_config_error(
            "Missing simulation.duration_days",
            config_key="simulation.duration_days",
            expected_type="int",
        )

    if not isinstance(duration, int) or duration <= 0:
        raise_validation_error(
            "Simulation duration must be a positive integer",
            field="simulation.duration_days",
            value=duration,
        )


# Example 2: Data loading with proper error handling
def load_weather_data(file_path: str) -> pd.DataFrame:
    """
    Example of using DataError and FileSystemError in data loading.

    This would replace generic exceptions with specific ones.
    """
    path = Path(file_path)

    # Check file exists
    if not path.exists():
        raise FileSystemError(
            f"Weather data file not found: {file_path}",
            {"file_path": file_path, "cwd": str(Path.cwd())},
        )

    # Check file permissions
    if not path.is_file() or not path.stat().st_size > 0:
        raise FileSystemError(
            f"Invalid weather data file: {file_path}",
            {
                "file_path": file_path,
                "size": path.stat().st_size if path.is_file() else 0,
            },
        )

    try:
        # Load data
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["date", "temperature", "rainfall"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise_data_error(
                f"Missing required columns in weather data: {missing_columns}",
                file_path=file_path,
            )

        return df

    except pd.errors.ParserError as e:
        raise DataError(
            f"Failed to parse weather data file: {e}",
            {"file_path": file_path, "parser_error": str(e)},
        ) from e


# Example 3: Simulation execution with proper error handling
class BeeModelExample:
    """Example of using exceptions in the main model class."""

    def __init__(self) -> None:
        self.initialized = False
        self.current_step = 0

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize model with proper error handling."""
        try:
            # Validate configuration
            validate_simulation_config(config)

            # Initialize components (these would be real methods in production)
            # self._init_landscape(config["environment"])
            # self._init_colonies(config["colony"])
            self.initialized = True

        except ConfigurationError:
            # Re-raise configuration errors as-is
            raise
        except Exception as e:
            # Wrap other errors in SimulationError
            raise SimulationError(
                f"Model initialization failed: {e}",
                {"config": config, "error_type": type(e).__name__},
            ) from e

    def step(self) -> None:
        """Execute one simulation step with error handling."""
        try:
            # Check model state
            if not hasattr(self, "initialized") or not self.initialized:
                raise SimulationError(
                    "Model not initialized before calling step()",
                    {"state": "uninitialized"},
                )

            # Execute step logic (these would be real methods in production)
            # self._update_weather()
            # self._update_foraging()
            # self._update_colonies()
            self.current_step += 1

        except SimulationError:
            # Re-raise simulation errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise SimulationError(
                f"Simulation step failed at step {self.current_step}: {e}",
                {"step": self.current_step, "error_type": type(e).__name__},
            ) from e


# Example 4: Analysis with proper error handling
def analyze_population_trends(
    data: pd.DataFrame, method: str = "rolling_mean"
) -> Dict[str, Any]:
    """Example of using AnalysisError in analysis functions."""
    # Validate input data
    if data.empty:
        raise AnalysisError("Cannot analyze empty dataset", {"data_shape": data.shape})

    required_columns = ["date", "population"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise AnalysisError(
            f"Missing required columns for population analysis: {missing_columns}",
            {"required": required_columns, "available": list(data.columns)},
        )

    # Validate method
    valid_methods = ["rolling_mean", "exponential_smoothing", "linear_regression"]
    if method not in valid_methods:
        raise_validation_error(
            f"Invalid analysis method. Must be one of: {valid_methods}",
            field="method",
            value=method,
        )

    try:
        # Perform analysis
        if method == "rolling_mean":
            result = data["population"].rolling(window=7).mean()
        elif method == "exponential_smoothing":
            result = data["population"].ewm(span=7).mean()
        else:
            # Linear regression would go here
            result = data["population"]  # Use original as placeholder

        return {
            "method": method,
            "result": result.tolist() if result is not None else [],
            "summary_stats": {
                "mean": float(data["population"].mean()),
                "std": float(data["population"].std()),
                "trend": "increasing"
                if result is not None
                and len(result) > 1
                and result.iloc[-1] > result.iloc[0]
                else "decreasing",
            },
        }

    except Exception as e:
        raise AnalysisError(
            f"Population trend analysis failed: {e}",
            {"method": method, "data_shape": data.shape, "error": str(e)},
        ) from e


# Example 5: External API integration with proper error handling
def fetch_met_office_data(location: tuple, api_key: str) -> Dict[str, Any]:
    """Example of using IntegrationError for external services."""
    import requests

    # Validate inputs
    if not api_key:
        raise IntegrationError(
            "Met Office API key not provided",
            {"service": "met_office", "error_type": "authentication"},
        )

    lat, lon = location
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise_validation_error(
            "Invalid geographic coordinates", field="location", value=location
        )

    try:
        # Make API request
        response = requests.get(
            "https://api.metoffice.gov.uk/data",
            params={"lat": lat, "lon": lon},
            headers={"X-API-Key": api_key},
            timeout=30,
        )

        # Check response
        if response.status_code == 401:
            raise IntegrationError(
                "Met Office API authentication failed",
                {
                    "service": "met_office",
                    "status_code": 401,
                    "error_type": "authentication",
                },
            )
        elif response.status_code == 429:
            raise IntegrationError(
                "Met Office API rate limit exceeded",
                {
                    "service": "met_office",
                    "status_code": 429,
                    "error_type": "rate_limit",
                },
            )
        elif response.status_code != 200:
            raise IntegrationError(
                f"Met Office API request failed: {response.status_code}",
                {
                    "service": "met_office",
                    "status_code": response.status_code,
                    "response": response.text,
                },
            )

        data: Dict[str, Any] = response.json()
        return data

    except requests.RequestException as e:
        raise IntegrationError(
            f"Failed to connect to Met Office API: {e}",
            {"service": "met_office", "error_type": "connection", "error": str(e)},
        ) from e


# Example 6: Parameter loading with proper error handling
def load_parameters(param_file: str) -> Dict[str, Any]:
    """Example of using ParameterError in parameter loading."""
    import yaml

    path = Path(param_file)

    # Check file exists
    if not path.exists():
        raise ParameterError(
            f"Parameter file not found: {param_file}",
            {"file_path": param_file, "search_paths": [".", "configs", "parameters"]},
        )

    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)

        # Validate parameter structure
        if not isinstance(params, dict):
            raise ParameterError(
                "Parameter file must contain a dictionary at top level",
                {"file_path": param_file, "actual_type": type(params).__name__},
            )

        # Validate required parameters
        required_params = ["foraging_range", "colony_size", "simulation_days"]
        missing_params = [p for p in required_params if p not in params]

        if missing_params:
            raise ParameterError(
                f"Missing required parameters: {missing_params}",
                {
                    "file_path": param_file,
                    "missing": missing_params,
                    "available": list(params.keys()),
                },
            )

        return params

    except yaml.YAMLError as e:
        raise ParameterError(
            f"Failed to parse parameter file: {e}",
            {"file_path": param_file, "error_type": "yaml_parse", "error": str(e)},
        ) from e


# Example 7: Resource management with proper error handling
def check_resource_availability(required_memory_gb: float = 4.0) -> None:
    """Example of using ResourceError for resource checks."""
    import psutil

    # Check available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    if available_memory_gb < required_memory_gb:
        raise ResourceError(
            f"Insufficient memory: {available_memory_gb:.1f}GB available, {required_memory_gb:.1f}GB required",
            {
                "resource_type": "memory",
                "available_gb": available_memory_gb,
                "required_gb": required_memory_gb,
                "total_gb": psutil.virtual_memory().total / (1024**3),
            },
        )

    # Check disk space
    disk_usage = psutil.disk_usage("/")
    available_disk_gb = disk_usage.free / (1024**3)
    required_disk_gb = 10.0  # Require 10GB for outputs

    if available_disk_gb < required_disk_gb:
        raise ResourceError(
            f"Insufficient disk space: {available_disk_gb:.1f}GB available, {required_disk_gb:.1f}GB required",
            {
                "resource_type": "disk",
                "available_gb": available_disk_gb,
                "required_gb": required_disk_gb,
                "mount_point": "/",
            },
        )


# Example 8: State management with proper error handling
class SimulationStateManager:
    """Example of using StateError for state management."""

    VALID_STATES = ["initialized", "running", "paused", "completed", "error"]
    VALID_TRANSITIONS = {
        "initialized": ["running"],
        "running": ["paused", "completed", "error"],
        "paused": ["running", "completed"],
        "completed": [],
        "error": [],
    }

    def __init__(self) -> None:
        self.state = "initialized"

    def transition_to(self, new_state: str) -> None:
        """Transition to a new state with validation."""
        # Validate new state
        if new_state not in self.VALID_STATES:
            raise StateError(
                f"Invalid state: {new_state}",
                {"requested_state": new_state, "valid_states": self.VALID_STATES},
            )

        # Check if transition is allowed
        allowed_transitions = self.VALID_TRANSITIONS.get(self.state, [])
        if new_state not in allowed_transitions:
            raise StateError(
                f"Invalid state transition: {self.state} -> {new_state}",
                {
                    "current_state": self.state,
                    "requested_state": new_state,
                    "allowed_transitions": allowed_transitions,
                },
            )

        # Perform transition
        old_state = self.state
        self.state = new_state
        print(f"State transition: {old_state} -> {new_state}")


# Example usage demonstrating error handling patterns
if __name__ == "__main__":
    # Example: Configuration validation
    try:
        config = {"simulation": {"duration_days": -10}}
        validate_simulation_config(config)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        print(f"Details: {e.details}")

    # Example: State management
    try:
        manager = SimulationStateManager()
        manager.transition_to("running")
        manager.transition_to("initialized")  # Invalid transition
    except StateError as e:
        print(f"State error: {e}")
        print(f"Details: {e.details}")

    # Example: Resource checking
    try:
        check_resource_availability(
            required_memory_gb=1000.0
        )  # Unrealistic requirement
    except ResourceError as e:
        print(f"Resource error: {e}")
        print(f"Details: {e.details}")
