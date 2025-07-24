"""
Configuration Module with Integrated Exception Handling
=======================================================

Example showing how to integrate standardized exceptions into the config module.
This demonstrates the pattern for updating existing modules.
"""

from typing import Dict, Any, Union
from pathlib import Path
import yaml

from bstew.exceptions import (
    ConfigurationError,
    ValidationError,
    FileSystemError,
    raise_validation_error,
)
from bstew.utils.config import BstewConfig


class ConfigManagerWithExceptions:
    """
    Enhanced ConfigManager with standardized exception handling.

    This shows how the existing ConfigManager would be updated to use
    the new exception hierarchy.
    """

    def __init__(self, config_dir: str = "configs"):
        """Initialize config manager with exception handling."""
        try:
            self.config_dir = Path(config_dir)
            self.config_dir.mkdir(exist_ok=True)
        except OSError as e:
            raise FileSystemError(
                f"Failed to create config directory: {config_dir}",
                {"path": config_dir, "error": str(e)},
            ) from e

        # Set up paths
        self.default_config_path = self.config_dir / "default.yaml"
        self.species_config_dir = self.config_dir / "species"
        self.scenario_config_dir = self.config_dir / "scenarios"

        # Create subdirectories with error handling
        for subdir in [self.species_config_dir, self.scenario_config_dir]:
            try:
                subdir.mkdir(exist_ok=True)
            except OSError as e:
                raise FileSystemError(
                    f"Failed to create subdirectory: {subdir}",
                    {"path": str(subdir), "parent": str(self.config_dir)},
                ) from e

    def load_config(self, config_path: Union[str, Path]) -> BstewConfig:
        """Load configuration with proper exception handling."""
        path = Path(config_path)

        # Check file exists
        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                {
                    "file_path": str(config_path),
                    "search_paths": [str(self.config_dir), ".", str(Path.cwd())],
                    "available_configs": self._list_available_configs(),
                },
            )

        # Check file is readable
        if not path.is_file():
            raise FileSystemError(
                f"Configuration path is not a file: {config_path}",
                {
                    "path": str(config_path),
                    "path_type": "directory" if path.is_dir() else "other",
                },
            )

        try:
            # Load YAML
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            # Handle empty files
            if data is None:
                data = {}

            # Validate and create config
            return BstewConfig.model_validate(data)

        except yaml.YAMLError as e:
            raise ConfigurationError(
                "Invalid YAML syntax in configuration file",
                {
                    "file_path": str(config_path),
                    "yaml_error": str(e),
                    "line": getattr(e, "problem_mark", {}).get("line", "unknown"),
                },
            ) from e

        except ValidationError as e:
            # Re-raise Pydantic validation errors as ConfigurationError
            raise ConfigurationError(
                "Configuration validation failed",
                {
                    "file_path": str(config_path),
                    "validation_errors": self._format_pydantic_errors(e),
                },
            ) from e

        except OSError as e:
            raise FileSystemError(
                "Failed to read configuration file",
                {"file_path": str(config_path), "error": str(e)},
            ) from e

    def save_config(self, config: BstewConfig, config_path: Union[str, Path]) -> None:
        """Save configuration with proper exception handling."""
        path = Path(config_path)

        # Ensure parent directory exists
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FileSystemError(
                "Failed to create directory for config file",
                {"directory": str(path.parent), "error": str(e)},
            ) from e

        try:
            # Convert to dict and save
            config_dict = config.model_dump()

            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        except OSError as e:
            raise FileSystemError(
                "Failed to write configuration file",
                {
                    "file_path": str(path),
                    "error": str(e),
                    "disk_space_available": self._get_available_disk_space(),
                },
            ) from e

        except Exception as e:
            raise ConfigurationError(
                "Failed to serialize configuration",
                {"file_path": str(path), "error": str(e)},
            ) from e

    def validate_config_for_species(self, config: BstewConfig, species: str) -> None:
        """Validate configuration for specific species with proper exceptions."""
        species_lower = species.lower()

        if species_lower.startswith("bombus"):
            # Bumblebee validations
            if config.colony.initial_population.get("workers", 0) > 1000:
                raise_validation_error(
                    "Bumblebee colonies typically have fewer than 1000 workers",
                    field="colony.initial_population.workers",
                    value=config.colony.initial_population["workers"],
                )

            if config.simulation.duration_days > 200:
                raise_validation_error(
                    "Bumblebee colonies typically last less than 200 days per year",
                    field="simulation.duration_days",
                    value=config.simulation.duration_days,
                )

        elif species_lower.startswith("apis"):
            # Honeybee validations
            if config.colony.initial_population.get("workers", 0) < 1000:
                raise_validation_error(
                    "Honeybee colonies typically have more than 1000 workers",
                    field="colony.initial_population.workers",
                    value=config.colony.initial_population["workers"],
                )
        else:
            raise ConfigurationError(
                f"Unknown species for validation: {species}",
                {
                    "species": species,
                    "supported_species": [
                        "apis_mellifera",
                        "bombus_terrestris",
                        "bombus_lapidarius",
                    ],
                },
            )

    def merge_configs(
        self,
        base_config: BstewConfig,
        override_config: Union[BstewConfig, Dict[str, Any]],
    ) -> BstewConfig:
        """Merge configurations with proper error handling."""
        try:
            # Convert to dicts
            base_dict = base_config.model_dump()

            if isinstance(override_config, BstewConfig):
                override_dict = override_config.model_dump()
            else:
                override_dict = override_config

            # Deep merge
            merged_dict = self._deep_merge_dicts(base_dict, override_dict)

            # Validate merged result
            return BstewConfig.model_validate(merged_dict)

        except ValidationError as e:
            raise ConfigurationError(
                "Merged configuration is invalid",
                {
                    "validation_errors": self._format_pydantic_errors(e),
                    "base_config_valid": True,
                    "merge_attempted": True,
                },
            ) from e

        except Exception as e:
            raise ConfigurationError(
                f"Configuration merge failed: {e}", {"error_type": type(e).__name__}
            ) from e

    def _format_pydantic_errors(self, validation_error: Any) -> list:
        """Format Pydantic validation errors for better readability."""
        errors = []
        if hasattr(validation_error, "errors"):
            for error in validation_error.errors():
                errors.append(
                    {
                        "field": ".".join(str(x) for x in error["loc"]),
                        "message": error["msg"],
                        "type": error["type"],
                    }
                )
        return errors

    def _list_available_configs(self) -> Dict[str, list]:
        """List available configuration files."""
        try:
            return {
                "default": [self.default_config_path.name]
                if self.default_config_path.exists()
                else [],
                "species": [f.name for f in self.species_config_dir.glob("*.yaml")],
                "scenarios": [f.name for f in self.scenario_config_dir.glob("*.yaml")],
            }
        except OSError:
            return {"error": ["Could not list configuration files"]}

    def _get_available_disk_space(self) -> str:
        """Get available disk space for error reporting."""
        try:
            import shutil

            stat = shutil.disk_usage(self.config_dir)
            return f"{stat.free / (1024**3):.1f}GB"
        except Exception:
            return "unknown"

    def _deep_merge_dicts(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value

        return result


# Example usage showing error handling
if __name__ == "__main__":
    # Example 1: Handle missing config file
    try:
        manager = ConfigManagerWithExceptions()
        config = manager.load_config("nonexistent.yaml")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        print(f"Available configs: {e.details.get('available_configs', {})}")

    # Example 2: Handle invalid YAML
    try:
        # Create invalid YAML file
        with open("invalid.yaml", "w") as f:
            f.write("invalid: yaml: content: {broken")

        config = manager.load_config("invalid.yaml")
    except ConfigurationError as e:
        print(f"YAML error: {e}")
        print(f"Line: {e.details.get('line', 'unknown')}")

    # Example 3: Handle validation errors
    # Note: BstewConfig expects SimulationConfig object, not dict
    # This example shows the error handling pattern
