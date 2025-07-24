"""
CSV Parameter Loading System for NetLogo BEE-STEWARD v2 Parity
==============================================================

Advanced parameter loading system matching NetLogo's CSV-based configuration
capabilities for bee behavior parameters, environmental settings, and simulation
configuration.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pydantic import BaseModel, Field, field_validator, PrivateAttr
import csv
import json
import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging
from datetime import datetime
import threading
from copy import deepcopy

from .netlogo_integration import NetLogoDataIntegrator, NetLogoFileType


class ParameterType(Enum):
    """Types of parameters that can be loaded"""

    BEHAVIORAL = "behavioral"
    ENVIRONMENTAL = "environmental"
    GENETIC = "genetic"
    COLONY = "colony"
    FORAGING = "foraging"
    MORTALITY = "mortality"
    SIMULATION = "simulation"


class ParameterFormat(Enum):
    """Supported parameter file formats"""

    CSV = "csv"
    JSON = "json"
    XLSX = "xlsx"
    TXT = "txt"


@dataclass
class ParameterValidationRule:
    """Validation rule for parameter values"""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    data_type: type = float
    required: bool = True
    default_value: Optional[Any] = None


@dataclass
class ParameterChangeEvent:
    """Event representing a parameter change during simulation"""

    timestamp: datetime
    parameter_name: str
    old_value: Any
    new_value: Any
    change_reason: str
    validation_passed: bool
    error_message: Optional[str] = None


class RuntimeParameterManager:
    """Manages runtime parameter modifications during simulation"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.parameter_history: List[ParameterChangeEvent] = []
        self.current_parameters: Dict[str, Any] = {}
        self.validation_rules: Dict[str, ParameterValidationRule] = {}
        self.parameter_lock = threading.Lock()
        self.change_callbacks: Dict[str, List[Callable]] = {}
        self.rollback_stack: List[Dict[str, Any]] = []
        self.max_rollback_depth = 10

    def initialize_parameters(
        self,
        parameters: Dict[str, Any],
        validation_rules: Optional[Dict[str, ParameterValidationRule]] = None,
    ) -> None:
        """Initialize parameters for runtime management"""
        with self.parameter_lock:
            self.current_parameters = deepcopy(parameters)
            if validation_rules:
                self.validation_rules = validation_rules
            self.rollback_stack = [deepcopy(parameters)]

    def register_change_callback(self, parameter_name: str, callback: Callable) -> None:
        """Register callback for parameter changes"""
        if parameter_name not in self.change_callbacks:
            self.change_callbacks[parameter_name] = []
        self.change_callbacks[parameter_name].append(callback)

    def modify_parameter(
        self,
        parameter_name: str,
        new_value: Any,
        change_reason: str = "Manual modification",
    ) -> bool:
        """Modify a parameter during simulation runtime"""
        with self.parameter_lock:
            old_value = self.current_parameters.get(parameter_name)

            # Validate the change
            validation_result = self._validate_parameter_change(
                parameter_name, new_value
            )

            # Create change event
            change_event = ParameterChangeEvent(
                timestamp=datetime.now(),
                parameter_name=parameter_name,
                old_value=old_value,
                new_value=new_value,
                change_reason=change_reason,
                validation_passed=validation_result[0],
                error_message=validation_result[1],
            )

            self.parameter_history.append(change_event)

            if validation_result[0]:
                # Save current state for rollback
                self.rollback_stack.append(deepcopy(self.current_parameters))
                if len(self.rollback_stack) > self.max_rollback_depth:
                    self.rollback_stack.pop(0)

                # Apply the change
                self.current_parameters[parameter_name] = new_value

                # Notify callbacks
                if parameter_name in self.change_callbacks:
                    for callback in self.change_callbacks[parameter_name]:
                        try:
                            callback(parameter_name, old_value, new_value)
                        except Exception as e:
                            self.logger.error(
                                f"Error in parameter change callback: {e}"
                            )

                self.logger.info(
                    f"Parameter {parameter_name} changed from {old_value} to {new_value}"
                )
                return True
            else:
                self.logger.warning(
                    f"Parameter change rejected: {validation_result[1]}"
                )
                return False

    def _validate_parameter_change(
        self, parameter_name: str, new_value: Any
    ) -> Tuple[bool, Optional[str]]:
        """Validate a parameter change"""
        if parameter_name not in self.validation_rules:
            return True, None  # No validation rules, allow change

        rule = self.validation_rules[parameter_name]

        # Type validation
        if not isinstance(new_value, rule.data_type):
            return (
                False,
                f"Invalid type for {parameter_name}: expected {rule.data_type.__name__}, got {type(new_value).__name__}",
            )

        # Range validation
        if (
            rule.min_value is not None
            and isinstance(new_value, (int, float))
            and new_value < rule.min_value
        ):
            return (
                False,
                f"Value {new_value} is below minimum {rule.min_value} for {parameter_name}",
            )

        if (
            rule.max_value is not None
            and isinstance(new_value, (int, float))
            and new_value > rule.max_value
        ):
            return (
                False,
                f"Value {new_value} is above maximum {rule.max_value} for {parameter_name}",
            )

        # Allowed values validation
        if rule.allowed_values is not None and new_value not in rule.allowed_values:
            return (
                False,
                f"Value {new_value} not in allowed values {rule.allowed_values} for {parameter_name}",
            )

        return True, None

    def get_parameter(self, parameter_name: str, default: Any = None) -> Any:
        """Get current parameter value"""
        with self.parameter_lock:
            return self.current_parameters.get(parameter_name, default)

    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all current parameters"""
        with self.parameter_lock:
            return deepcopy(self.current_parameters)

    def rollback_parameter(self, parameter_name: str) -> bool:
        """Rollback a specific parameter to previous value"""
        if not self.parameter_history:
            return False

        # Find the last change for this parameter
        for event in reversed(self.parameter_history):
            if event.parameter_name == parameter_name and event.validation_passed:
                return self.modify_parameter(
                    parameter_name, event.old_value, "Rollback"
                )

        return False

    def rollback_all_parameters(self) -> bool:
        """Rollback all parameters to previous state"""
        if len(self.rollback_stack) < 2:
            return False

        with self.parameter_lock:
            # Remove current state
            self.rollback_stack.pop()
            # Restore previous state
            previous_state = self.rollback_stack[-1]
            self.current_parameters = deepcopy(previous_state)

            # Log rollback
            self.parameter_history.append(
                ParameterChangeEvent(
                    timestamp=datetime.now(),
                    parameter_name="ALL",
                    old_value="current_state",
                    new_value="previous_state",
                    change_reason="Full rollback",
                    validation_passed=True,
                )
            )

            self.logger.info("All parameters rolled back to previous state")
            return True

    def get_parameter_history(
        self, parameter_name: Optional[str] = None
    ) -> List[ParameterChangeEvent]:
        """Get parameter change history"""
        if parameter_name is None:
            return self.parameter_history.copy()
        else:
            return [
                event
                for event in self.parameter_history
                if event.parameter_name == parameter_name
            ]

    def export_parameter_log(self, output_path: str) -> None:
        """Export parameter change log to file"""
        log_data = []
        for event in self.parameter_history:
            log_data.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "parameter_name": event.parameter_name,
                    "old_value": event.old_value,
                    "new_value": event.new_value,
                    "change_reason": event.change_reason,
                    "validation_passed": event.validation_passed,
                    "error_message": event.error_message,
                }
            )

        with open(output_path, "w") as f:
            json.dump(log_data, f, indent=2)

        self.logger.info(f"Parameter change log exported to {output_path}")


class BehavioralParameters(BaseModel):
    """Behavioral parameters for bee agents"""

    model_config = {"validate_assignment": True}

    # Activity state durations (in time steps)
    foraging_duration_min: int = Field(
        default=10, ge=1, description="Minimum foraging duration"
    )
    foraging_duration_max: int = Field(
        default=50, ge=1, description="Maximum foraging duration"
    )
    nursing_duration_min: int = Field(
        default=5, ge=1, description="Minimum nursing duration"
    )
    nursing_duration_max: int = Field(
        default=30, ge=1, description="Maximum nursing duration"
    )
    resting_duration_min: int = Field(
        default=1, ge=1, description="Minimum resting duration"
    )
    resting_duration_max: int = Field(
        default=10, ge=1, description="Maximum resting duration"
    )

    # Energy parameters
    base_energy_consumption: float = Field(
        default=1.0, ge=0.0, description="Base energy consumption per step"
    )
    foraging_energy_multiplier: float = Field(
        default=3.0, ge=1.0, description="Foraging energy multiplier"
    )
    nursing_energy_multiplier: float = Field(
        default=1.5, ge=1.0, description="Nursing energy multiplier"
    )

    # Foraging behavior
    exploration_probability: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Exploration probability"
    )
    dance_following_probability: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Dance following probability"
    )
    memory_decay_rate: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Memory decay rate per day"
    )

    # Social behavior
    dance_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Quality threshold for dancing"
    )
    communication_range: float = Field(
        default=5.0, ge=0.0, description="Communication range in meters"
    )

    # Role transition parameters
    nurse_to_forager_age: int = Field(
        default=200, ge=0, description="Age for nurse to forager transition"
    )
    forager_experience_threshold: int = Field(
        default=5, ge=0, description="Experience threshold for foraging"
    )

    @field_validator("foraging_duration_max")
    @classmethod
    def validate_foraging_duration(cls, v: float, info: Any) -> float:
        if (
            hasattr(info, "data")
            and "foraging_duration_min" in info.data
            and v < info.data["foraging_duration_min"]
        ):
            raise ValueError("foraging_duration_max must be >= foraging_duration_min")
        return v

    @field_validator("nursing_duration_max")
    @classmethod
    def validate_nursing_duration(cls, v: float, info: Any) -> float:
        if (
            hasattr(info, "data")
            and "nursing_duration_min" in info.data
            and v < info.data["nursing_duration_min"]
        ):
            raise ValueError("nursing_duration_max must be >= nursing_duration_min")
        return v

    @field_validator("resting_duration_max")
    @classmethod
    def validate_resting_duration(cls, v: float, info: Any) -> float:
        if (
            hasattr(info, "data")
            and "resting_duration_min" in info.data
            and v < info.data["resting_duration_min"]
        ):
            raise ValueError("resting_duration_max must be >= resting_duration_min")
        return v


class EnvironmentalParameters(BaseModel):
    """Environmental parameters for simulation"""

    model_config = {"validate_assignment": True}

    # Weather parameters
    temperature_min: float = Field(default=5.0, description="Minimum temperature (°C)")
    temperature_max: float = Field(default=35.0, description="Maximum temperature (°C)")
    temperature_optimal: float = Field(
        default=25.0, description="Optimal temperature (°C)"
    )

    # Seasonal parameters
    season_length: int = Field(
        default=365, ge=1, description="Length of season in days"
    )
    flowering_season_start: int = Field(
        default=60, ge=0, description="Start of flowering season"
    )
    flowering_season_end: int = Field(
        default=280, ge=0, description="End of flowering season"
    )

    # Resource availability
    nectar_base_availability: float = Field(
        default=1.0, ge=0.0, description="Base nectar availability"
    )
    pollen_base_availability: float = Field(
        default=1.0, ge=0.0, description="Base pollen availability"
    )
    resource_regeneration_rate: float = Field(
        default=0.1, ge=0.0, description="Resource regeneration rate"
    )

    # Spatial parameters
    foraging_range: float = Field(
        default=1000.0, ge=0.0, description="Maximum foraging range (m)"
    )
    patch_density: float = Field(
        default=0.1, ge=0.0, description="Patch density per unit area"
    )

    # Environmental stress factors
    weather_stress_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weather stress threshold"
    )
    predator_pressure_base: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Base predator pressure"
    )

    @field_validator("temperature_max")
    @classmethod
    def validate_temperature_range(cls, v: float, info: Any) -> float:
        if (
            hasattr(info, "data")
            and "temperature_min" in info.data
            and v < info.data["temperature_min"]
        ):
            raise ValueError("temperature_max must be >= temperature_min")
        return v

    @field_validator("flowering_season_end")
    @classmethod
    def validate_flowering_season(cls, v: int, info: Any) -> int:
        if (
            hasattr(info, "data")
            and "flowering_season_start" in info.data
            and v < info.data["flowering_season_start"]
        ):
            raise ValueError("flowering_season_end must be >= flowering_season_start")
        return v


class ColonyParameters(BaseModel):
    """Colony-level parameters"""

    model_config = {"validate_assignment": True}

    # Initial population
    initial_population: int = Field(
        default=1000, ge=1, description="Initial colony population"
    )
    initial_nurses: int = Field(
        default=600, ge=0, description="Initial number of nurses"
    )
    initial_foragers: int = Field(
        default=300, ge=0, description="Initial number of foragers"
    )
    initial_builders: int = Field(
        default=80, ge=0, description="Initial number of builders"
    )
    initial_guards: int = Field(
        default=20, ge=0, description="Initial number of guards"
    )

    # Queen parameters
    queen_egg_laying_rate: float = Field(
        default=2000.0, ge=0.0, description="Queen egg laying rate per day"
    )
    queen_lifespan: int = Field(
        default=1825, ge=1, description="Queen lifespan in days"
    )

    # Colony energy management
    energy_storage_capacity: float = Field(
        default=10000.0, ge=0.0, description="Colony energy storage capacity"
    )
    energy_reserve_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Energy reserve threshold"
    )

    # Genetic parameters
    genetic_diversity_target: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Target genetic diversity"
    )
    inbreeding_tolerance: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Inbreeding tolerance"
    )

    # Population dynamics
    max_population: int = Field(
        default=50000, ge=1, description="Maximum colony population"
    )
    carrying_capacity_factor: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Carrying capacity factor"
    )

    @field_validator("initial_population")
    @classmethod
    def validate_initial_population(cls, v: int, info: Any) -> int:
        # Check if sum of role populations equals total (only if all role values are provided)
        role_fields = [
            "initial_nurses",
            "initial_foragers",
            "initial_builders",
            "initial_guards",
        ]
        if hasattr(info, "data") and all(field in info.data for field in role_fields):
            role_sum = sum(
                [
                    info.data.get("initial_nurses", 0),
                    info.data.get("initial_foragers", 0),
                    info.data.get("initial_builders", 0),
                    info.data.get("initial_guards", 0),
                ]
            )
            if role_sum != v:
                raise ValueError(
                    f"Sum of role populations ({role_sum}) must equal initial_population ({v})"
                )
        return v


class ParameterLoader(BaseModel):
    """Advanced parameter loading system matching NetLogo capabilities"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    # Private logger attribute (set in __init__ to bypass Pydantic validation)
    _logger: logging.Logger = PrivateAttr()

    # Configuration
    parameter_directory: str = Field(
        default="artifacts/parameters",
        description="Directory containing parameter files",
    )
    validation_enabled: bool = Field(
        default=True, description="Enable parameter validation"
    )
    auto_save_enabled: bool = Field(
        default=True, description="Enable auto-save of validated parameters"
    )

    # Parameter storage
    behavioral_params: Optional[BehavioralParameters] = Field(
        default=None, description="Loaded behavioral parameters"
    )
    environmental_params: Optional[EnvironmentalParameters] = Field(
        default=None, description="Loaded environmental parameters"
    )
    colony_params: Optional[ColonyParameters] = Field(
        default=None, description="Loaded colony parameters"
    )

    # Validation rules
    validation_rules: Dict[str, ParameterValidationRule] = Field(
        default_factory=dict, description="Validation rules"
    )

    # Load history
    load_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Parameter load history"
    )

    # Runtime components (excluded from Pydantic validation)
    runtime_manager: Optional[RuntimeParameterManager] = Field(
        default=None, exclude=True
    )
    netlogo_integrator: Optional[NetLogoDataIntegrator] = Field(
        default=None, exclude=True
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Initialize logger using private attribute to bypass Pydantic validation
        object.__setattr__(self, "_logger", logging.getLogger(__name__))

        # Initialize validation rules
        self._initialize_validation_rules()

        # Initialize runtime parameter manager
        self.runtime_manager = RuntimeParameterManager()

        # Initialize NetLogo data integrator
        self.netlogo_integrator = NetLogoDataIntegrator()

    def _initialize_validation_rules(self) -> None:
        """Initialize default validation rules"""

        # Behavioral parameter rules
        self.validation_rules.update(
            {
                "foraging_duration_min": ParameterValidationRule(
                    min_value=1, max_value=1000, data_type=int
                ),
                "foraging_duration_max": ParameterValidationRule(
                    min_value=1, max_value=1000, data_type=int
                ),
                "exploration_probability": ParameterValidationRule(
                    min_value=0.0, max_value=1.0, data_type=float
                ),
                "dance_following_probability": ParameterValidationRule(
                    min_value=0.0, max_value=1.0, data_type=float
                ),
                "memory_decay_rate": ParameterValidationRule(
                    min_value=0.0, max_value=1.0, data_type=float
                ),
            }
        )

        # Environmental parameter rules
        self.validation_rules.update(
            {
                "temperature_min": ParameterValidationRule(
                    min_value=-50.0, max_value=50.0, data_type=float
                ),
                "temperature_max": ParameterValidationRule(
                    min_value=-50.0, max_value=50.0, data_type=float
                ),
                "nectar_base_availability": ParameterValidationRule(
                    min_value=0.0, max_value=10.0, data_type=float
                ),
                "pollen_base_availability": ParameterValidationRule(
                    min_value=0.0, max_value=10.0, data_type=float
                ),
            }
        )

        # Colony parameter rules
        self.validation_rules.update(
            {
                "initial_population": ParameterValidationRule(
                    min_value=1, max_value=100000, data_type=int
                ),
                "queen_egg_laying_rate": ParameterValidationRule(
                    min_value=0.0, max_value=10000.0, data_type=float
                ),
                "genetic_diversity_target": ParameterValidationRule(
                    min_value=0.0, max_value=1.0, data_type=float
                ),
            }
        )

    def load_parameters_from_csv(
        self, file_path: str, parameter_type: ParameterType
    ) -> Dict[str, Any]:
        """Load parameters from CSV file"""

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parameter file not found: {file_path}")

        parameters = {}

        try:
            with open(file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    param_name = row.get("parameter_name", row.get("name", ""))
                    param_value = row.get("parameter_value", row.get("value", ""))
                    param_type = row.get("parameter_type", row.get("type", "float"))

                    if param_name and param_value:
                        # Convert value to appropriate type
                        converted_value = self._convert_parameter_value(
                            param_value, param_type
                        )

                        # Validate if enabled
                        if self.validation_enabled:
                            self._validate_parameter(param_name, converted_value)

                        parameters[param_name] = converted_value

        except Exception as e:
            raise ValueError(f"Error loading CSV parameters from {file_path}: {str(e)}")

        # Record load history
        self.load_history.append(
            {
                "file_path": file_path,
                "parameter_type": parameter_type.value,
                "parameters_loaded": len(parameters),
                "timestamp": pd.Timestamp.now(),
            }
        )

        return parameters

    def load_parameters_from_json(
        self, file_path: str, parameter_type: ParameterType
    ) -> Dict[str, Any]:
        """Load parameters from JSON file"""

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parameter file not found: {file_path}")

        try:
            with open(file_path, "r") as jsonfile:
                parameters = json.load(jsonfile)

            # Validate if enabled
            if self.validation_enabled:
                for param_name, param_value in parameters.items():
                    self._validate_parameter(param_name, param_value)

        except Exception as e:
            raise ValueError(
                f"Error loading JSON parameters from {file_path}: {str(e)}"
            )

        # Record load history
        self.load_history.append(
            {
                "file_path": file_path,
                "parameter_type": parameter_type.value,
                "parameters_loaded": len(parameters),
                "timestamp": pd.Timestamp.now(),
            }
        )

        return dict(parameters)

    def load_parameters_from_excel(
        self,
        file_path: str,
        parameter_type: ParameterType,
        sheet_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load parameters from Excel file"""

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parameter file not found: {file_path}")

        try:
            # Read Excel file
            df_or_dict = pd.read_excel(file_path, sheet_name=sheet_name)

            # Handle case where multiple sheets are returned as dict
            if isinstance(df_or_dict, dict):
                # Use first sheet if multiple sheets returned
                df = next(iter(df_or_dict.values()))
            else:
                df = df_or_dict

            parameters = {}

            # Assume columns are: parameter_name, parameter_value, parameter_type
            for _, row in df.iterrows():
                param_name = row.get("parameter_name", row.get("name", ""))
                param_value = row.get("parameter_value", row.get("value", ""))
                param_type = row.get("parameter_type", row.get("type", "float"))

                if pd.notna(param_name) and pd.notna(param_value):
                    # Convert value to appropriate type
                    converted_value = self._convert_parameter_value(
                        param_value, param_type
                    )

                    # Validate if enabled
                    if self.validation_enabled:
                        self._validate_parameter(param_name, converted_value)

                    parameters[param_name] = converted_value

        except Exception as e:
            raise ValueError(
                f"Error loading Excel parameters from {file_path}: {str(e)}"
            )

        # Record load history
        self.load_history.append(
            {
                "file_path": file_path,
                "parameter_type": parameter_type.value,
                "parameters_loaded": len(parameters),
                "timestamp": pd.Timestamp.now(),
            }
        )

        return parameters

    def _convert_parameter_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to appropriate type"""

        if param_type.lower() == "int":
            return int(value)
        elif param_type.lower() == "float":
            return float(value)
        elif param_type.lower() == "bool":
            return value.lower() in ["true", "1", "yes", "on"]
        elif param_type.lower() == "str":
            return str(value)
        elif param_type.lower() == "list":
            # Handle list format: "[1, 2, 3]" or "1,2,3"
            if value.startswith("[") and value.endswith("]"):
                return eval(value)  # Safe for controlled input
            else:
                return [x.strip() for x in value.split(",")]
        else:
            # Default to string
            return str(value)

    def _validate_parameter(self, param_name: str, param_value: Any) -> None:
        """Validate parameter value against rules"""

        if param_name not in self.validation_rules:
            return  # No validation rule defined

        rule = self.validation_rules[param_name]

        # Type validation
        if not isinstance(param_value, rule.data_type):
            raise ValueError(
                f"Parameter {param_name} must be of type {rule.data_type.__name__}"
            )

        # Range validation
        if (
            rule.min_value is not None
            and isinstance(param_value, (int, float))
            and param_value < rule.min_value
        ):
            raise ValueError(f"Parameter {param_name} must be >= {rule.min_value}")

        if (
            rule.max_value is not None
            and isinstance(param_value, (int, float))
            and param_value > rule.max_value
        ):
            raise ValueError(f"Parameter {param_name} must be <= {rule.max_value}")

        # Allowed values validation
        if rule.allowed_values is not None and param_value not in rule.allowed_values:
            raise ValueError(
                f"Parameter {param_name} must be one of {rule.allowed_values}"
            )

    def load_behavioral_parameters(
        self, file_path: str, file_format: ParameterFormat = ParameterFormat.CSV
    ) -> BehavioralParameters:
        """Load behavioral parameters from file"""

        if file_format == ParameterFormat.CSV:
            params = self.load_parameters_from_csv(file_path, ParameterType.BEHAVIORAL)
        elif file_format == ParameterFormat.JSON:
            params = self.load_parameters_from_json(file_path, ParameterType.BEHAVIORAL)
        elif file_format == ParameterFormat.XLSX:
            params = self.load_parameters_from_excel(
                file_path, ParameterType.BEHAVIORAL
            )
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Create BehavioralParameters instance
        self.behavioral_params = BehavioralParameters(**params)

        # Auto-save if enabled
        if self.auto_save_enabled:
            self._save_validated_parameters(
                "behavioral", self.behavioral_params.model_dump()
            )

        return self.behavioral_params

    def load_environmental_parameters(
        self, file_path: str, file_format: ParameterFormat = ParameterFormat.CSV
    ) -> EnvironmentalParameters:
        """Load environmental parameters from file"""

        if file_format == ParameterFormat.CSV:
            params = self.load_parameters_from_csv(
                file_path, ParameterType.ENVIRONMENTAL
            )
        elif file_format == ParameterFormat.JSON:
            params = self.load_parameters_from_json(
                file_path, ParameterType.ENVIRONMENTAL
            )
        elif file_format == ParameterFormat.XLSX:
            params = self.load_parameters_from_excel(
                file_path, ParameterType.ENVIRONMENTAL
            )
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Create EnvironmentalParameters instance
        self.environmental_params = EnvironmentalParameters(**params)

        # Auto-save if enabled
        if self.auto_save_enabled:
            self._save_validated_parameters(
                "environmental", self.environmental_params.model_dump()
            )

        return self.environmental_params

    def load_colony_parameters(
        self, file_path: str, file_format: ParameterFormat = ParameterFormat.CSV
    ) -> ColonyParameters:
        """Load colony parameters from file"""

        if file_format == ParameterFormat.CSV:
            params = self.load_parameters_from_csv(file_path, ParameterType.COLONY)
        elif file_format == ParameterFormat.JSON:
            params = self.load_parameters_from_json(file_path, ParameterType.COLONY)
        elif file_format == ParameterFormat.XLSX:
            params = self.load_parameters_from_excel(file_path, ParameterType.COLONY)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Create ColonyParameters instance
        self.colony_params = ColonyParameters(**params)

        # Auto-save if enabled
        if self.auto_save_enabled:
            self._save_validated_parameters("colony", self.colony_params.model_dump())

        return self.colony_params

    def load_all_parameters(
        self, parameter_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load all parameter types from directory with enhanced validation"""

        if parameter_directory is None:
            parameter_directory = self.parameter_directory

        if not os.path.exists(parameter_directory):
            raise FileNotFoundError(
                f"Parameter directory not found: {parameter_directory}"
            )

        all_params: Dict[str, Any] = {}

        # Look for standard parameter files
        file_patterns = {
            "behavioral": ["behavioral.csv", "behavioral.json", "behavioral.xlsx"],
            "environmental": [
                "environmental.csv",
                "environmental.json",
                "environmental.xlsx",
            ],
            "colony": ["colony.csv", "colony.json", "colony.xlsx"],
        }

        for param_type, file_names in file_patterns.items():
            for file_name in file_names:
                file_path = os.path.join(parameter_directory, file_name)
                if os.path.exists(file_path):
                    file_format = ParameterFormat(file_name.split(".")[-1].lower())

                    if param_type == "behavioral":
                        all_params["behavioral"] = self.load_behavioral_parameters(
                            file_path, file_format
                        )
                    elif param_type == "environmental":
                        all_params["environmental"] = (
                            self.load_environmental_parameters(file_path, file_format)
                        )
                    elif param_type == "colony":
                        all_params["colony"] = self.load_colony_parameters(
                            file_path, file_format
                        )

                    break  # Use first found file for each type

        return all_params

    def _save_validated_parameters(
        self, param_type: str, params: Dict[str, Any]
    ) -> None:
        """Save validated parameters to file"""

        output_dir = os.path.join(self.parameter_directory, "validated")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{param_type}_validated.json")

        with open(output_file, "w") as f:
            json.dump(params, f, indent=2, default=str)

    def save_validated_parameters(
        self, param_type: str, params: Dict[str, Any]
    ) -> None:
        """Save validated parameters to file (public alias)"""
        self._save_validated_parameters(param_type, params)

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get summary of loaded parameters"""

        summary: Dict[str, Any] = {
            "behavioral_loaded": self.behavioral_params is not None,
            "environmental_loaded": self.environmental_params is not None,
            "colony_loaded": self.colony_params is not None,
            "validation_enabled": self.validation_enabled,
            "auto_save_enabled": self.auto_save_enabled,
            "load_history": len(self.load_history),
            "validation_rules": len(self.validation_rules),
        }

        if self.behavioral_params:
            summary["behavioral_params"] = {
                "exploration_probability": self.behavioral_params.exploration_probability,
                "dance_following_probability": self.behavioral_params.dance_following_probability,
                "memory_decay_rate": self.behavioral_params.memory_decay_rate,
            }

        if self.environmental_params:
            summary["environmental_params"] = {
                "temperature_range": (
                    self.environmental_params.temperature_min,
                    self.environmental_params.temperature_max,
                ),
                "foraging_range": self.environmental_params.foraging_range,
                "resource_availability": (
                    self.environmental_params.nectar_base_availability,
                    self.environmental_params.pollen_base_availability,
                ),
            }

        if self.colony_params:
            summary["colony_params"] = {
                "initial_population": self.colony_params.initial_population,
                "queen_egg_laying_rate": self.colony_params.queen_egg_laying_rate,
                "max_population": self.colony_params.max_population,
            }

        return summary

    def export_parameters_to_csv(self, output_directory: str) -> None:
        """Export current parameters to CSV files"""

        os.makedirs(output_directory, exist_ok=True)

        # Export behavioral parameters
        if self.behavioral_params:
            self._export_pydantic_to_csv(
                self.behavioral_params,
                os.path.join(output_directory, "behavioral_parameters.csv"),
            )

        # Export environmental parameters
        if self.environmental_params:
            self._export_pydantic_to_csv(
                self.environmental_params,
                os.path.join(output_directory, "environmental_parameters.csv"),
            )

        # Export colony parameters
        if self.colony_params:
            self._export_pydantic_to_csv(
                self.colony_params,
                os.path.join(output_directory, "colony_parameters.csv"),
            )

    def _export_pydantic_to_csv(self, model: BaseModel, file_path: str) -> None:
        """Export Pydantic model to CSV format"""

        data = model.model_dump()

        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["parameter_name", "parameter_value", "parameter_type"])

            for param_name, param_value in data.items():
                param_type = type(param_value).__name__
                writer.writerow([param_name, param_value, param_type])

    def create_parameter_template(
        self, param_type: ParameterType, output_path: str
    ) -> None:
        """Create parameter template file"""

        template_params: Union[
            BehavioralParameters, EnvironmentalParameters, ColonyParameters
        ]
        if param_type == ParameterType.BEHAVIORAL:
            template_params = BehavioralParameters()
        elif param_type == ParameterType.ENVIRONMENTAL:
            template_params = EnvironmentalParameters()
        elif param_type == ParameterType.COLONY:
            template_params = ColonyParameters()
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

        self._export_pydantic_to_csv(template_params, output_path)

    def validate_all_parameters(self) -> Dict[str, List[str]]:
        """Validate all loaded parameters"""

        validation_errors = {}

        # Validate behavioral parameters
        if self.behavioral_params:
            try:
                self.behavioral_params.model_validate(
                    self.behavioral_params.model_dump()
                )
            except Exception as e:
                validation_errors["behavioral"] = [str(e)]

        # Validate environmental parameters
        if self.environmental_params:
            try:
                self.environmental_params.model_validate(
                    self.environmental_params.model_dump()
                )
            except Exception as e:
                validation_errors["environmental"] = [str(e)]

        # Validate colony parameters
        if self.colony_params:
            try:
                self.colony_params.model_validate(self.colony_params.model_dump())
            except Exception as e:
                validation_errors["colony"] = [str(e)]

        return validation_errors

    def enable_runtime_modifications(
        self,
        parameters: Dict[str, Any],
        validation_rules: Optional[Dict[str, ParameterValidationRule]] = None,
    ) -> None:
        """Enable runtime parameter modifications"""
        if self.runtime_manager is None:
            raise RuntimeError("Runtime manager not initialized")
        if validation_rules is None:
            validation_rules = self.validation_rules
        self.runtime_manager.initialize_parameters(parameters, validation_rules)

    def modify_parameter_runtime(
        self,
        parameter_name: str,
        new_value: Any,
        change_reason: str = "Runtime modification",
    ) -> bool:
        """Modify a parameter during runtime"""
        if self.runtime_manager is None:
            raise RuntimeError("Runtime manager not initialized")
        return self.runtime_manager.modify_parameter(
            parameter_name, new_value, change_reason
        )

    def get_runtime_parameter(self, parameter_name: str, default: Any = None) -> Any:
        """Get current runtime parameter value"""
        if self.runtime_manager is None:
            return default
        return self.runtime_manager.get_parameter(parameter_name, default)

    def get_all_runtime_parameters(self) -> Dict[str, Any]:
        """Get all current runtime parameters"""
        if self.runtime_manager is None:
            return {}
        return self.runtime_manager.get_all_parameters()

    def rollback_parameter_runtime(self, parameter_name: str) -> bool:
        """Rollback a parameter to previous value"""
        if self.runtime_manager is None:
            return False
        return self.runtime_manager.rollback_parameter(parameter_name)

    def rollback_all_parameters_runtime(self) -> bool:
        """Rollback all parameters to previous state"""
        if self.runtime_manager is None:
            return False
        return self.runtime_manager.rollback_all_parameters()

    def register_parameter_change_callback(
        self, parameter_name: str, callback: Callable
    ) -> None:
        """Register callback for parameter changes"""
        if self.runtime_manager is None:
            raise RuntimeError("Runtime manager not initialized")
        self.runtime_manager.register_change_callback(parameter_name, callback)

    def get_parameter_change_history(
        self, parameter_name: Optional[str] = None
    ) -> List[ParameterChangeEvent]:
        """Get parameter change history"""
        if self.runtime_manager is None:
            return []
        return self.runtime_manager.get_parameter_history(parameter_name)

    def export_parameter_change_log(self, output_path: str) -> None:
        """Export parameter change log"""
        if self.runtime_manager is None:
            raise RuntimeError("Runtime manager not initialized")
        self.runtime_manager.export_parameter_log(output_path)

    def setup_default_validation_rules(self) -> Dict[str, ParameterValidationRule]:
        """Setup default validation rules for common parameters"""
        rules = {
            "foraging_range": ParameterValidationRule(
                min_value=0.0, max_value=10000.0, data_type=float, required=True
            ),
            "colony_size": ParameterValidationRule(
                min_value=1, max_value=100000, data_type=int, required=True
            ),
            "mortality_rate": ParameterValidationRule(
                min_value=0.0, max_value=1.0, data_type=float, required=True
            ),
            "temperature": ParameterValidationRule(
                min_value=-50.0, max_value=60.0, data_type=float, required=False
            ),
            "dance_threshold": ParameterValidationRule(
                min_value=0.0, max_value=1.0, data_type=float, required=True
            ),
            "energy_cost": ParameterValidationRule(
                min_value=0.0, max_value=1000.0, data_type=float, required=True
            ),
            "bee_species": ParameterValidationRule(
                allowed_values=[
                    "apis_mellifera",
                    "bombus_terrestris",
                    "bombus_lapidarius",
                ],
                data_type=str,
                required=True,
            ),
            "simulation_duration": ParameterValidationRule(
                min_value=1, max_value=365 * 10, data_type=int, required=True
            ),
        }
        return rules

    def load_netlogo_parameters(self, file_path: str) -> Dict[str, Any]:
        """Load parameters directly from NetLogo parameter file"""
        if self.netlogo_integrator is None:
            raise RuntimeError("NetLogo integrator not initialized")
        return self.netlogo_integrator.load_netlogo_parameters(Path(file_path))

    def load_netlogo_species(self, file_path: str) -> List[Dict[str, Any]]:
        """Load species data from NetLogo species file"""
        if self.netlogo_integrator is None:
            raise RuntimeError("NetLogo integrator not initialized")
        return self.netlogo_integrator.load_netlogo_species(Path(file_path))

    def discover_netlogo_files(self, directory: str) -> Dict[str, Any]:
        """Discover NetLogo files in directory"""
        if self.netlogo_integrator is None:
            raise RuntimeError("NetLogo integrator not initialized")
        files_by_type = self.netlogo_integrator.discover_netlogo_files(Path(directory))

        # Convert to serializable format
        result: Dict[str, List[Dict[str, Any]]] = {}
        for file_type, files in files_by_type.items():
            result[file_type.value] = []
            for file_info in files:
                result[file_type.value].append(
                    {
                        "file_path": str(file_info.file_path),
                        "contains_data": file_info.contains_data,
                        "row_count": file_info.row_count,
                        "column_count": file_info.column_count,
                        "columns": file_info.columns,
                        "validation_errors": file_info.validation_errors,
                    }
                )
        return result

    def validate_netlogo_compatibility(self, directory: str) -> Dict[str, Any]:
        """Validate NetLogo data compatibility"""
        if self.netlogo_integrator is None:
            raise RuntimeError("NetLogo integrator not initialized")
        return self.netlogo_integrator.validate_netlogo_compatibility(Path(directory))

    def convert_netlogo_to_bstew(
        self, netlogo_directory: str, output_directory: str
    ) -> None:
        """Convert NetLogo data files to BSTEW format"""
        if self.netlogo_integrator is None:
            raise RuntimeError("NetLogo integrator not initialized")
        self.netlogo_integrator.convert_netlogo_to_bstew(
            Path(netlogo_directory), Path(output_directory)
        )

    def _validate_parameter_ranges(
        self, params: Dict[str, Any], parameter_type: ParameterType
    ) -> Dict[str, Any]:
        """Validate parameter ranges"""
        violations = []

        # Get validation rules for this parameter type
        rules = self.get_parameter_validation_rules()

        if parameter_type == ParameterType.BEHAVIORAL:
            param_rules = rules.get("behavioral_parameters", {})
        elif parameter_type == ParameterType.ENVIRONMENTAL:
            param_rules = rules.get("environmental_parameters", {})
        else:
            param_rules = {}

        for param_name, value in params.items():
            if param_name in param_rules:
                rule = param_rules[param_name]
                try:
                    if "min" in rule and value < rule["min"]:
                        violations.append(f"{param_name}: {value} < {rule['min']}")
                    if "max" in rule and value > rule["max"]:
                        violations.append(f"{param_name}: {value} > {rule['max']}")
                except (TypeError, ValueError):
                    violations.append(f"{param_name}: invalid type")

        return {"violations": violations}

    def load_parameters_with_validation(
        self,
        file_path: str,
        parameter_type: ParameterType,
        validate_ranges: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load parameters with comprehensive validation"""

        validation_result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "parameter_count": 0,
            "range_violations": [],
        }

        try:
            # Load parameters based on file type
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".csv":
                params = self.load_parameters_from_csv(file_path, parameter_type)
            elif file_ext == ".json":
                params = self.load_parameters_from_json(file_path, parameter_type)
            elif file_ext in [".xlsx", ".xls"]:
                params = self.load_parameters_from_excel(file_path, parameter_type)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            validation_result["parameter_count"] = len(params)

            # Validate parameter ranges if requested
            if validate_ranges and parameter_type in [
                ParameterType.BEHAVIORAL,
                ParameterType.ENVIRONMENTAL,
            ]:
                range_validation = self._validate_parameter_ranges(
                    params, parameter_type
                )
                validation_result["range_violations"] = range_validation["violations"]
                if range_validation["violations"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(
                        [
                            f"Range violation: {v}"
                            for v in range_validation["violations"]
                        ]
                    )

            # Record successful load
            self.load_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "file_path": file_path,
                    "parameter_type": parameter_type.value,
                    "parameter_count": len(params),
                    "validation_passed": validation_result["valid"],
                }
            )

            return params, validation_result

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
            return {}, validation_result

    def get_parameter_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameter validation rules for NetLogo compatibility"""

        return {
            "behavioral_parameters": {
                "cropvolume_myl": {
                    "min": 10.0,
                    "max": 150.0,
                    "type": "float",
                    "description": "Crop volume in microliters",
                },
                "glossaLength_mm": {
                    "min": 1.0,
                    "max": 10.0,
                    "type": "float",
                    "description": "Glossa length in millimeters",
                },
                "proboscis_length_mm": {
                    "min": 3.0,
                    "max": 15.0,
                    "type": "float",
                    "description": "Proboscis length in millimeters",
                },
                "mandible_width_mm": {
                    "min": 0.5,
                    "max": 3.0,
                    "type": "float",
                    "description": "Mandible width in millimeters",
                },
                "dance_threshold": {
                    "min": 0.0,
                    "max": 1.0,
                    "type": "float",
                    "description": "Quality threshold for dancing",
                },
                "exploration_probability": {
                    "min": 0.0,
                    "max": 1.0,
                    "type": "float",
                    "description": "Exploration probability",
                },
                "memory_decay_rate": {
                    "min": 0.8,
                    "max": 1.0,
                    "type": "float",
                    "description": "Memory decay rate per day",
                },
            },
            "environmental_parameters": {
                "temperature_min": {
                    "min": -10.0,
                    "max": 40.0,
                    "type": "float",
                    "description": "Minimum temperature in Celsius",
                },
                "temperature_max": {
                    "min": 0.0,
                    "max": 50.0,
                    "type": "float",
                    "description": "Maximum temperature in Celsius",
                },
                "season_length": {
                    "min": 30,
                    "max": 365,
                    "type": "int",
                    "description": "Season length in days",
                },
            },
            "colony_parameters": {
                "initial_population": {
                    "min": 10,
                    "max": 10000,
                    "type": "int",
                    "description": "Initial colony population",
                },
                "max_foraging_range": {
                    "min": 100.0,
                    "max": 5000.0,
                    "type": "float",
                    "description": "Maximum foraging range in meters",
                },
                "energy_threshold": {
                    "min": 1.0,
                    "max": 100.0,
                    "type": "float",
                    "description": "Energy threshold for activities",
                },
            },
        }

    def load_parameters_from_netlogo_directory(self, directory: str) -> Dict[str, Any]:
        """Load all parameters from NetLogo directory"""
        if self.netlogo_integrator is None:
            raise RuntimeError("NetLogo integrator not initialized")
        files_by_type = self.netlogo_integrator.discover_netlogo_files(Path(directory))

        all_parameters = {}

        # Load parameter files
        if NetLogoFileType.PARAMETERS in files_by_type:
            for file_info in files_by_type[NetLogoFileType.PARAMETERS]:
                try:
                    params = self.load_netlogo_parameters(str(file_info.file_path))
                    all_parameters.update(params)
                except Exception as e:
                    self._logger.error(
                        f"Error loading parameters from {file_info.file_path}: {e}"
                    )

        # Load species data
        if NetLogoFileType.SPECIES in files_by_type:
            species_data = []
            for file_info in files_by_type[NetLogoFileType.SPECIES]:
                try:
                    species = self.load_netlogo_species(str(file_info.file_path))
                    species_data.extend(species)
                except Exception as e:
                    self._logger.error(
                        f"Error loading species from {file_info.file_path}: {e}"
                    )
            if species_data:
                all_parameters["species_data"] = species_data

        return all_parameters
