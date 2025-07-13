"""
Input validation utilities for CLI commands
===========================================

Provides validation for command-line arguments, configuration files, and parameters.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import re
from enum import Enum

from ..types import ValidationErrors


class ValidationSeverity(Enum):
    """Validation message severity levels"""
    
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationMessage:
    """Validation message with severity and context"""
    
    def __init__(
        self,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        field: Optional[str] = None,
        value: Optional[Any] = None,
    ) -> None:
        self.message = message
        self.severity = severity
        self.field = field
        self.value = value
    
    def __str__(self) -> str:
        if self.field:
            return f"{self.field}: {self.message}"
        return self.message


class InputValidator:
    """General input validation utilities"""
    
    @staticmethod
    def validate_file_exists(path: Union[str, Path], description: str = "File") -> ValidationErrors:
        """Validate that a file exists"""
        errors = []
        file_path = Path(path)
        
        if not file_path.exists():
            errors.append(f"{description} not found: {path}")
        elif not file_path.is_file():
            errors.append(f"{description} is not a file: {path}")
        
        return errors
    
    @staticmethod
    def validate_directory_writable(path: Union[str, Path], description: str = "Directory") -> ValidationErrors:
        """Validate that a directory is writable"""
        errors = []
        dir_path = Path(path)
        
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = dir_path / ".test_write"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            errors.append(f"{description} is not writable: {path} ({e})")
        
        return errors
    
    @staticmethod
    def validate_positive_integer(value: Any, name: str, min_value: int = 1) -> ValidationErrors:
        """Validate positive integer value"""
        errors = []
        
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                errors.append(f"{name} must be an integer, got: {type(value).__name__}")
                return errors
        
        if value < min_value:
            errors.append(f"{name} must be >= {min_value}, got: {value}")
        
        return errors
    
    @staticmethod
    def validate_float_range(
        value: Any,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> ValidationErrors:
        """Validate float value within range"""
        errors = []
        
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                errors.append(f"{name} must be a number, got: {type(value).__name__}")
                return errors
        
        if min_value is not None and value < min_value:
            errors.append(f"{name} must be >= {min_value}, got: {value}")
        
        if max_value is not None and value > max_value:
            errors.append(f"{name} must be <= {max_value}, got: {value}")
        
        return errors
    
    @staticmethod
    def validate_choice(value: Any, name: str, choices: List[str]) -> ValidationErrors:
        """Validate value is in allowed choices"""
        errors = []
        
        if str(value) not in choices:
            errors.append(f"{name} must be one of {choices}, got: {value}")
        
        return errors
    
    @staticmethod
    def validate_email(email: str) -> ValidationErrors:
        """Validate email format"""
        errors = []
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            errors.append(f"Invalid email format: {email}")
        
        return errors
    
    @staticmethod
    def validate_parameter_name(name: str) -> ValidationErrors:
        """Validate parameter name format"""
        errors = []
        
        # Allow dotted notation like "colony.species"
        pattern = r'^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)*$'
        
        if not re.match(pattern, name):
            errors.append(
                f"Invalid parameter name: {name}. Must start with letter and contain only "
                "letters, numbers, underscores, and dots for nested parameters."
            )
        
        return errors


class ConfigValidator:
    """Configuration validation utilities"""
    
    def __init__(self) -> None:
        self.validators: Dict[str, Callable[[Any], ValidationErrors]] = {}
    
    def register_validator(self, key: str, validator: Callable[[Any], ValidationErrors]) -> None:
        """Register a validator for a specific configuration key"""
        self.validators[key] = validator
    
    def validate_simulation_config(self, config: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate simulation configuration"""
        messages = []
        
        # Validate simulation section
        if "simulation" in config:
            sim_config = config["simulation"]
            
            # Duration
            if "duration_days" in sim_config:
                errors = InputValidator.validate_positive_integer(
                    sim_config["duration_days"], "simulation.duration_days", min_value=1
                )
                messages.extend([ValidationMessage(err) for err in errors])
            
            # Random seed
            if "random_seed" in sim_config and sim_config["random_seed"] is not None:
                errors = InputValidator.validate_positive_integer(
                    sim_config["random_seed"], "simulation.random_seed", min_value=0
                )
                messages.extend([ValidationMessage(err) for err in errors])
        
        # Validate colony section
        if "colony" in config:
            colony_config = config["colony"]
            
            # Species
            if "species" in colony_config:
                valid_species = [
                    "B_terrestris", "B_pascuorum", "B_lapidarius", 
                    "B_hortorum", "B_pratorum", "B_hypnorum", "Psithyrus"
                ]
                errors = InputValidator.validate_choice(
                    colony_config["species"], "colony.species", valid_species
                )
                messages.extend([ValidationMessage(err) for err in errors])
            
            # Initial population
            if "initial_population" in colony_config:
                pop_config = colony_config["initial_population"]
                if isinstance(pop_config, dict):
                    for role, count in pop_config.items():
                        errors = InputValidator.validate_positive_integer(
                            count, f"colony.initial_population.{role}", min_value=0
                        )
                        messages.extend([ValidationMessage(err) for err in errors])
        
        # Validate environment section
        if "environment" in config:
            env_config = config["environment"]
            
            # Landscape dimensions
            for dim in ["landscape_width", "landscape_height"]:
                if dim in env_config:
                    errors = InputValidator.validate_positive_integer(
                        env_config[dim], f"environment.{dim}", min_value=1
                    )
                    messages.extend([ValidationMessage(err) for err in errors])
        
        return messages
    
    def validate_experiment_config(self, config: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate experiment configuration"""
        messages = []
        
        # Validate experiment type
        if "experiment_type" in config:
            valid_types = [
                "parameter_sweep", "monte_carlo", "sensitivity_analysis", 
                "factorial_design", "latin_hypercube", "optimization", "validation"
            ]
            errors = InputValidator.validate_choice(
                config["experiment_type"], "experiment_type", valid_types
            )
            messages.extend([ValidationMessage(err) for err in errors])
        
        # Validate parameters section
        if "parameters" in config:
            for param_name, param_config in config["parameters"].items():
                # Validate parameter name
                errors = InputValidator.validate_parameter_name(param_name)
                messages.extend([ValidationMessage(err, field=param_name) for err in errors])
                
                # Validate parameter configuration
                if isinstance(param_config, dict):
                    # Min/max values
                    if "min_value" in param_config and "max_value" in param_config:
                        min_val = param_config["min_value"]
                        max_val = param_config["max_value"]
                        
                        if min_val >= max_val:
                            messages.append(ValidationMessage(
                                "min_value must be less than max_value",
                                field=param_name
                            ))
                    
                    # Distribution
                    if "distribution" in param_config:
                        valid_distributions = ["uniform", "normal", "log_uniform"]
                        errors = InputValidator.validate_choice(
                            param_config["distribution"], 
                            f"{param_name}.distribution", 
                            valid_distributions
                        )
                        messages.extend([ValidationMessage(err) for err in errors])
        
        # Validate sample sizes
        for field in ["n_samples", "n_replicates"]:
            if field in config:
                errors = InputValidator.validate_positive_integer(
                    config[field], field, min_value=1
                )
                messages.extend([ValidationMessage(err) for err in errors])
        
        return messages
    
    def validate_optimization_config(self, config: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate optimization configuration"""
        messages = []
        
        # Validate optimization method
        if "method" in config:
            valid_methods = [
                "genetic_algorithm", "bayesian_optimization", "particle_swarm",
                "differential_evolution", "nelder_mead", "scipy_minimize"
            ]
            errors = InputValidator.validate_choice(
                config["method"], "method", valid_methods
            )
            messages.extend([ValidationMessage(err) for err in errors])
        
        # Validate optimization parameters
        if "max_iterations" in config:
            errors = InputValidator.validate_positive_integer(
                config["max_iterations"], "max_iterations", min_value=1
            )
            messages.extend([ValidationMessage(err) for err in errors])
        
        if "population_size" in config:
            errors = InputValidator.validate_positive_integer(
                config["population_size"], "population_size", min_value=2
            )
            messages.extend([ValidationMessage(err) for err in errors])
        
        # Validate objective function
        if "objective" in config:
            valid_objectives = ["rmse", "mae", "r_squared", "likelihood", "custom"]
            errors = InputValidator.validate_choice(
                config["objective"], "objective", valid_objectives
            )
            messages.extend([ValidationMessage(err) for err in errors])
        
        return messages