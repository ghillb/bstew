"""
Dynamic Parameter Modification System for NetLogo BEE-STEWARD v2 Parity
======================================================================

Comprehensive system for modifying simulation parameters during runtime,
enabling adaptive simulations, parameter optimization, and dynamic scenario testing.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import logging
import time
import math


class ParameterType(Enum):
    """Types of parameters that can be modified"""

    COLONY = "colony"
    ENVIRONMENT = "environment"
    FORAGING = "foraging"
    COMMUNICATION = "communication"
    DISEASE = "disease"
    REPRODUCTION = "reproduction"
    WEATHER = "weather"
    BEHAVIORAL = "behavioral"
    SPATIAL = "spatial"
    PERFORMANCE = "performance"


class ModificationTrigger(Enum):
    """Triggers for parameter modifications"""

    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    MANUAL = "manual"
    ADAPTIVE = "adaptive"
    SCHEDULED = "scheduled"


class ModificationStrategy(Enum):
    """Strategies for parameter modification"""

    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    STEPPED = "stepped"
    OSCILLATING = "oscillating"
    THRESHOLD_BASED = "threshold_based"
    FEEDBACK_CONTROLLED = "feedback_controlled"


@dataclass
class ParameterConstraint:
    """Constraints for parameter values"""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    validation_function: Optional[str] = (
        None  # String representation of validation logic
    )
    dependencies: List[str] = field(
        default_factory=list
    )  # Other parameters this depends on


@dataclass
class ModificationRule:
    """Rule for parameter modification"""

    rule_id: str
    parameter_path: (
        str  # Dot notation path to parameter (e.g., "colony.foraging.efficiency")
    )
    parameter_type: ParameterType
    trigger: ModificationTrigger
    strategy: ModificationStrategy

    # Trigger conditions
    trigger_condition: Optional[str] = None  # Condition expression
    trigger_time: Optional[float] = None  # Time-based trigger
    trigger_event: Optional[str] = None  # Event type to trigger on

    # Modification parameters
    target_value: Optional[Union[float, str, bool]] = None
    value_function: Optional[str] = None  # Function for calculating new value
    modification_rate: float = 0.1  # Rate of change for gradual modifications
    duration: Optional[float] = None  # Duration of modification

    # Constraints and validation
    constraint: Optional[ParameterConstraint] = None
    enabled: bool = True
    priority: int = 1  # Higher priority rules override lower priority

    # Metadata
    description: str = ""
    created_time: float = field(default_factory=time.time)
    last_applied: Optional[float] = None
    application_count: int = 0


@dataclass
class ParameterHistory:
    """History of parameter changes"""

    parameter_path: str
    timestamp: float
    old_value: Any
    new_value: Any
    rule_id: Optional[str] = None
    trigger_type: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class ParameterMonitor(BaseModel):
    """Monitor for tracking parameter changes and effects"""

    model_config = {"validate_assignment": True}

    parameter_path: str = Field(description="Parameter being monitored")
    current_value: Any = Field(description="Current parameter value")
    baseline_value: Any = Field(description="Original baseline value")

    # Change tracking
    change_history: List[ParameterHistory] = Field(default_factory=list)
    last_change_time: Optional[float] = None
    change_frequency: float = 0.0

    # Impact metrics
    impact_metrics: Dict[str, float] = Field(default_factory=dict)
    performance_correlation: Dict[str, float] = Field(default_factory=dict)

    # Alerts and thresholds
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    alert_history: List[Dict[str, Any]] = Field(default_factory=list)


class DynamicParameterSystem(BaseModel):
    """Comprehensive dynamic parameter modification system"""

    model_config = {"validate_assignment": True}

    # Core components
    modification_rules: Dict[str, ModificationRule] = Field(default_factory=dict)
    parameter_monitors: Dict[str, ParameterMonitor] = Field(default_factory=dict)
    active_modifications: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # System state
    current_simulation_time: float = 0.0
    modification_history: List[ParameterHistory] = Field(default_factory=list)
    system_state: Dict[str, Any] = Field(default_factory=dict)

    # Performance tracking
    modification_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    system_health_metrics: Dict[str, float] = Field(default_factory=dict)

    # Configuration
    enable_adaptive_modifications: bool = True
    max_simultaneous_modifications: int = 10
    modification_cooldown: float = 5.0  # Seconds between modifications

    # Logging
    logger: Any = Field(
        default_factory=lambda: logging.getLogger(__name__), exclude=True
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def add_modification_rule(self, rule: ModificationRule) -> None:
        """Add a new parameter modification rule"""

        if self.logger:
            self.logger.info(f"Adding modification rule: {rule.rule_id}")

        # Validate rule
        validation_result = self._validate_modification_rule(rule)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid modification rule: {validation_result['error']}")

        # Add rule to system
        self.modification_rules[rule.rule_id] = rule

        # Initialize parameter monitor if needed
        if rule.parameter_path not in self.parameter_monitors:
            self._initialize_parameter_monitor(rule.parameter_path)

        self.logger.info(
            f"Rule {rule.rule_id} added successfully for parameter {rule.parameter_path}"
        )

    def remove_modification_rule(self, rule_id: str) -> bool:
        """Remove a modification rule"""

        if rule_id in self.modification_rules:
            del self.modification_rules[rule_id]

            # Clean up any active modifications
            if rule_id in self.active_modifications:
                del self.active_modifications[rule_id]

            self.logger.info(f"Modification rule {rule_id} removed")
            return True

        return False

    def update_system_state(
        self, simulation_time: float, model_state: Dict[str, Any]
    ) -> None:
        """Update system state with current simulation information"""

        self.current_simulation_time = simulation_time
        self.system_state.update(model_state)

        # Update parameter monitors
        self._update_parameter_monitors(model_state)

        # Check for triggered modifications
        triggered_rules = self._check_modification_triggers()

        # Apply triggered modifications
        for rule_id in triggered_rules:
            self._apply_modification_rule(rule_id, model_state)

        # Update active modifications
        self._update_active_modifications()

        # Update system health metrics
        self._update_system_health_metrics()

    def apply_manual_modification(
        self, parameter_path: str, new_value: Any, duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Apply manual parameter modification"""

        self.logger.info(
            f"Applying manual modification to {parameter_path}: {new_value}"
        )

        # Create temporary rule for manual modification
        manual_rule = ModificationRule(
            rule_id=f"manual_{parameter_path}_{time.time()}",
            parameter_path=parameter_path,
            parameter_type=ParameterType.BEHAVIORAL,  # Default type
            trigger=ModificationTrigger.MANUAL,
            strategy=ModificationStrategy.IMMEDIATE,
            target_value=new_value,
            duration=duration,
            description=f"Manual modification to {parameter_path}",
        )

        # Apply modification immediately
        result = self._execute_parameter_modification(manual_rule, new_value)

        # Record in history
        if result["success"]:
            history_entry = ParameterHistory(
                parameter_path=parameter_path,
                timestamp=self.current_simulation_time,
                old_value=result.get("old_value"),
                new_value=new_value,
                rule_id=manual_rule.rule_id,
                trigger_type="manual",
                success=True,
            )
            self.modification_history.append(history_entry)

        return result

    def get_parameter_analytics(
        self, parameter_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get analytics for parameter modifications"""

        if parameter_path:
            # Analytics for specific parameter
            if parameter_path not in self.parameter_monitors:
                return {"error": f"Parameter {parameter_path} not monitored"}

            monitor = self.parameter_monitors[parameter_path]
            parameter_history = [
                h
                for h in self.modification_history
                if h.parameter_path == parameter_path
            ]

            return {
                "parameter_path": parameter_path,
                "current_value": monitor.current_value,
                "baseline_value": monitor.baseline_value,
                "change_count": len(parameter_history),
                "last_change_time": monitor.last_change_time,
                "change_frequency": monitor.change_frequency,
                "impact_metrics": monitor.impact_metrics,
                "performance_correlation": monitor.performance_correlation,
                "recent_changes": parameter_history[-10:] if parameter_history else [],
            }
        else:
            # System-wide analytics
            return {
                "total_rules": len(self.modification_rules),
                "active_modifications": len(self.active_modifications),
                "monitored_parameters": len(self.parameter_monitors),
                "total_modifications": len(self.modification_history),
                "system_health": self.system_health_metrics,
                "modification_performance": self.modification_performance,
                "recent_activity": self.modification_history[-20:]
                if self.modification_history
                else [],
            }

    def optimize_parameters_adaptive(
        self, target_metrics: Dict[str, float], learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Adaptive parameter optimization based on target metrics"""

        self.logger.info(
            f"Starting adaptive parameter optimization with targets: {target_metrics}"
        )

        optimization_results: Dict[str, Any] = {
            "optimization_id": f"adaptive_{time.time()}",
            "target_metrics": target_metrics,
            "parameter_adjustments": {},
            "performance_improvement": {},
            "convergence_status": "in_progress",
        }

        # Analyze current performance vs targets
        current_performance = self._get_current_performance_metrics()

        for metric_name, target_value in target_metrics.items():
            current_value = current_performance.get(metric_name, 0.0)

            if abs(current_value - target_value) > 0.05:  # 5% tolerance
                # Find parameters that influence this metric
                influential_parameters = self._find_parameters_affecting_metric(
                    metric_name
                )

                for param_path, influence_score in influential_parameters.items():
                    if influence_score > 0.1:  # Significant influence
                        # Calculate adjustment direction and magnitude
                        adjustment = self._calculate_parameter_adjustment(
                            param_path,
                            metric_name,
                            current_value,
                            target_value,
                            learning_rate,
                        )

                        if adjustment["magnitude"] > 0.01:  # Meaningful adjustment
                            # Create adaptive modification rule
                            adaptive_rule = ModificationRule(
                                rule_id=f"adaptive_{param_path}_{metric_name}_{time.time()}",
                                parameter_path=param_path,
                                parameter_type=ParameterType.BEHAVIORAL,
                                trigger=ModificationTrigger.ADAPTIVE,
                                strategy=ModificationStrategy.GRADUAL,
                                target_value=adjustment["target_value"],
                                modification_rate=learning_rate,
                                duration=100.0,  # 100 time steps
                                description=f"Adaptive optimization for {metric_name}",
                            )

                            self.add_modification_rule(adaptive_rule)
                            optimization_results["parameter_adjustments"][
                                param_path
                            ] = adjustment

        return optimization_results

    def _validate_modification_rule(self, rule: ModificationRule) -> Dict[str, Any]:
        """Validate modification rule"""

        # Check required fields
        if not rule.parameter_path:
            return {"valid": False, "error": "Parameter path is required"}

        # Check trigger configuration
        if rule.trigger == ModificationTrigger.TIME_BASED and rule.trigger_time is None:
            return {"valid": False, "error": "Time-based trigger requires trigger_time"}

        if rule.trigger == ModificationTrigger.EVENT_BASED and not rule.trigger_event:
            return {
                "valid": False,
                "error": "Event-based trigger requires trigger_event",
            }

        if (
            rule.trigger == ModificationTrigger.CONDITION_BASED
            and not rule.trigger_condition
        ):
            return {
                "valid": False,
                "error": "Condition-based trigger requires trigger_condition",
            }

        # Check modification configuration
        if not rule.target_value and not rule.value_function:
            return {
                "valid": False,
                "error": "Either target_value or value_function is required",
            }

        # Check constraints
        if rule.constraint:
            constraint_validation = self._validate_parameter_constraint(
                rule.constraint, rule.target_value
            )
            if not constraint_validation["valid"]:
                return constraint_validation

        return {"valid": True}

    def _validate_parameter_constraint(
        self, constraint: ParameterConstraint, value: Any
    ) -> Dict[str, Any]:
        """Validate parameter value against constraints"""

        if constraint.min_value is not None and isinstance(value, (int, float)):
            if value < constraint.min_value:
                return {
                    "valid": False,
                    "error": f"Value {value} below minimum {constraint.min_value}",
                }

        if constraint.max_value is not None and isinstance(value, (int, float)):
            if value > constraint.max_value:
                return {
                    "valid": False,
                    "error": f"Value {value} above maximum {constraint.max_value}",
                }

        if constraint.allowed_values and value not in constraint.allowed_values:
            return {
                "valid": False,
                "error": f"Value {value} not in allowed values {constraint.allowed_values}",
            }

        return {"valid": True}

    def _initialize_parameter_monitor(self, parameter_path: str) -> None:
        """Initialize monitor for parameter"""

        monitor = ParameterMonitor(
            parameter_path=parameter_path, current_value=None, baseline_value=None
        )

        self.parameter_monitors[parameter_path] = monitor
        self.logger.info(f"Parameter monitor initialized for {parameter_path}")

    def _update_parameter_monitors(self, model_state: Dict[str, Any]) -> None:
        """Update all parameter monitors with current values"""

        for parameter_path, monitor in self.parameter_monitors.items():
            try:
                current_value = self._get_parameter_value(parameter_path, model_state)

                if monitor.baseline_value is None:
                    monitor.baseline_value = current_value

                # Update change tracking
                if monitor.current_value != current_value:
                    monitor.last_change_time = self.current_simulation_time
                    monitor.change_frequency = len(
                        [
                            h
                            for h in monitor.change_history
                            if h.timestamp > self.current_simulation_time - 100.0
                        ]
                    )

                monitor.current_value = current_value

            except Exception as e:
                self.logger.warning(
                    f"Failed to update monitor for {parameter_path}: {e}"
                )

    def _check_modification_triggers(self) -> List[str]:
        """Check all modification rules for triggers"""

        triggered_rules = []

        for rule_id, rule in self.modification_rules.items():
            if not rule.enabled or rule_id in self.active_modifications:
                continue

            # Check cooldown
            if (
                rule.last_applied
                and self.current_simulation_time - rule.last_applied
                < self.modification_cooldown
            ):
                continue

            trigger_result = self._evaluate_trigger(rule)
            if trigger_result:
                triggered_rules.append(rule_id)

        return triggered_rules

    def _evaluate_trigger(self, rule: ModificationRule) -> bool:
        """Evaluate if rule trigger condition is met"""

        if rule.trigger == ModificationTrigger.TIME_BASED:
            return (
                rule.trigger_time is not None
                and self.current_simulation_time >= rule.trigger_time
            )

        elif rule.trigger == ModificationTrigger.CONDITION_BASED:
            return bool(
                rule.trigger_condition
                and self._evaluate_condition(rule.trigger_condition)
            )

        elif rule.trigger == ModificationTrigger.SCHEDULED:
            # Check if scheduled time has arrived
            return self._check_scheduled_trigger(rule)

        elif rule.trigger == ModificationTrigger.ADAPTIVE:
            # Adaptive triggers are handled separately
            return self._check_adaptive_trigger(rule)

        return False

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate condition expression"""

        try:
            # Safe evaluation of condition using system state
            # This would need proper security measures in production
            context = {
                "simulation_time": self.current_simulation_time,
                "system_state": self.system_state,
                "math": math,
                "abs": abs,
                "min": min,
                "max": max,
            }

            result = eval(condition, {"__builtins__": {}}, context)
            return bool(result)

        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def _apply_modification_rule(
        self, rule_id: str, model_state: Dict[str, Any]
    ) -> None:
        """Apply modification rule"""

        rule = self.modification_rules[rule_id]

        # Calculate new value
        if rule.value_function:
            new_value = self._calculate_value_from_function(
                rule.value_function, model_state
            )
        else:
            new_value = rule.target_value

        # Apply modification based on strategy
        if rule.strategy == ModificationStrategy.IMMEDIATE:
            result = self._execute_parameter_modification(rule, new_value)

        elif rule.strategy == ModificationStrategy.GRADUAL:
            result = self._start_gradual_modification(rule, new_value)

        elif rule.strategy == ModificationStrategy.STEPPED:
            result = self._execute_stepped_modification(rule, new_value)

        else:
            result = {
                "success": False,
                "error": f"Unsupported modification strategy: {rule.strategy}",
            }

        # Update rule application tracking
        if result["success"]:
            rule.last_applied = self.current_simulation_time
            rule.application_count += 1

            # Record in history
            history_entry = ParameterHistory(
                parameter_path=rule.parameter_path,
                timestamp=self.current_simulation_time,
                old_value=result.get("old_value"),
                new_value=new_value,
                rule_id=rule_id,
                trigger_type=rule.trigger.value,
                success=True,
            )
            self.modification_history.append(history_entry)

        self.logger.info(f"Applied modification rule {rule_id}: {result}")

    def _execute_parameter_modification(
        self, rule: ModificationRule, new_value: Any
    ) -> Dict[str, Any]:
        """Execute immediate parameter modification"""

        try:
            # Get current value
            current_value = self._get_parameter_value(
                rule.parameter_path, self.system_state
            )

            # Validate new value against constraints
            if rule.constraint:
                validation = self._validate_parameter_constraint(
                    rule.constraint, new_value
                )
                if not validation["valid"]:
                    return {"success": False, "error": validation["error"]}

            # Apply modification (this would interface with the actual model)
            success = self._set_parameter_value(rule.parameter_path, new_value)

            if success:
                return {
                    "success": True,
                    "old_value": current_value,
                    "new_value": new_value,
                    "parameter_path": rule.parameter_path,
                }
            else:
                return {"success": False, "error": "Failed to set parameter value"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _start_gradual_modification(
        self, rule: ModificationRule, target_value: Any
    ) -> Dict[str, Any]:
        """Start gradual parameter modification"""

        current_value = self._get_parameter_value(
            rule.parameter_path, self.system_state
        )

        # Create active modification tracking
        modification_data = {
            "rule_id": rule.rule_id,
            "parameter_path": rule.parameter_path,
            "start_value": current_value,
            "target_value": target_value,
            "current_value": current_value,
            "start_time": self.current_simulation_time,
            "modification_rate": rule.modification_rate,
            "duration": rule.duration,
            "strategy": rule.strategy.value,
        }

        self.active_modifications[rule.rule_id] = modification_data

        return {
            "success": True,
            "modification_type": "gradual_started",
            "duration": rule.duration,
        }

    def _update_active_modifications(self) -> None:
        """Update all active gradual modifications"""

        completed_modifications = []

        for rule_id, mod_data in self.active_modifications.items():
            # Calculate progress
            elapsed_time = self.current_simulation_time - mod_data["start_time"]
            duration = mod_data.get("duration", 100.0)
            progress = min(1.0, elapsed_time / duration)

            # Calculate new value based on progress
            start_val = mod_data["start_value"]
            target_val = mod_data["target_value"]

            if isinstance(start_val, (int, float)) and isinstance(
                target_val, (int, float)
            ):
                new_value = start_val + (target_val - start_val) * progress
            else:
                # For non-numeric values, switch at 50% progress
                new_value = target_val if progress >= 0.5 else start_val

            # Apply modification
            success = self._set_parameter_value(mod_data["parameter_path"], new_value)

            if success:
                mod_data["current_value"] = new_value

            # Check if modification is complete
            if progress >= 1.0:
                completed_modifications.append(rule_id)

        # Clean up completed modifications
        for rule_id in completed_modifications:
            del self.active_modifications[rule_id]

    def _get_parameter_value(
        self, parameter_path: str, model_state: Dict[str, Any]
    ) -> Any:
        """Get parameter value from model state using dot notation"""

        try:
            parts = parameter_path.split(".")
            value: Any = model_state

            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)

                if value is None:
                    return None

            return value

        except Exception as e:
            self.logger.warning(
                f"Failed to get parameter value for {parameter_path}: {e}"
            )
            return None

    def _set_parameter_value(self, parameter_path: str, new_value: Any) -> bool:
        """Set parameter value in model (interface with actual model)"""

        # This would interface with the actual model to set parameter values
        # For now, just log the modification
        self.logger.info(f"Setting parameter {parameter_path} to {new_value}")

        # Update system state for tracking
        parts = parameter_path.split(".")
        current_dict = self.system_state

        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        current_dict[parts[-1]] = new_value
        return True

    def _get_current_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics from system state"""

        performance_metrics = {}

        # Extract common performance metrics from system state
        if "foraging_efficiency" in self.system_state:
            performance_metrics["foraging_efficiency"] = self.system_state[
                "foraging_efficiency"
            ]

        if "colony_health" in self.system_state:
            performance_metrics["colony_health"] = self.system_state["colony_health"]

        if "population_growth" in self.system_state:
            performance_metrics["population_growth"] = self.system_state[
                "population_growth"
            ]

        # Calculate derived metrics
        if (
            "total_energy_collected" in self.system_state
            and "total_energy_consumed" in self.system_state
        ):
            total_collected = self.system_state["total_energy_collected"]
            total_consumed = self.system_state["total_energy_consumed"]
            if total_consumed > 0:
                performance_metrics["energy_efficiency"] = (
                    total_collected / total_consumed
                )

        return performance_metrics

    def _find_parameters_affecting_metric(self, metric_name: str) -> Dict[str, float]:
        """Find parameters that affect a specific metric"""

        # This would use correlation analysis or learned relationships
        # For now, return some default influential parameters
        parameter_influences = {
            "foraging_efficiency": {
                "colony.foraging.search_radius": 0.8,
                "colony.foraging.energy_threshold": 0.6,
                "environment.flower_density": 0.7,
                "behavioral.decision_threshold": 0.5,
            },
            "colony_health": {
                "colony.nutrition.protein_requirement": 0.7,
                "colony.disease.resistance": 0.9,
                "environment.temperature_stress": 0.6,
            },
            "population_growth": {
                "colony.reproduction.egg_laying_rate": 0.9,
                "colony.mortality.base_rate": 0.8,
                "colony.nutrition.larval_feeding": 0.7,
            },
        }

        return parameter_influences.get(metric_name, {})

    def _calculate_parameter_adjustment(
        self,
        param_path: str,
        metric_name: str,
        current_value: float,
        target_value: float,
        learning_rate: float,
    ) -> Dict[str, Any]:
        """Calculate parameter adjustment for optimization"""

        # Calculate error
        error = target_value - current_value

        # Get parameter's current value
        current_param_value = self._get_parameter_value(param_path, self.system_state)
        if current_param_value is None:
            current_param_value = 1.0  # Default

        # Calculate adjustment magnitude based on error and learning rate
        adjustment_magnitude = abs(error) * learning_rate

        # Determine adjustment direction (simplified heuristic)
        if (
            "efficiency" in metric_name
            or "health" in metric_name
            or "growth" in metric_name
        ):
            # For metrics where higher is better
            direction = 1 if error > 0 else -1
        else:
            # For metrics where lower might be better
            direction = -1 if error > 0 else 1

        # Calculate target parameter value
        if isinstance(current_param_value, (int, float)):
            adjustment = direction * adjustment_magnitude * abs(current_param_value)
            target_param_value = current_param_value + adjustment
        else:
            # For non-numeric parameters
            target_param_value = current_param_value
            adjustment_magnitude = 0.0

        return {
            "current_param_value": current_param_value,
            "target_value": target_param_value,
            "adjustment": adjustment
            if isinstance(current_param_value, (int, float))
            else 0.0,
            "magnitude": adjustment_magnitude,
            "direction": direction,
            "error": error,
        }

    def _calculate_value_from_function(
        self, value_function: str, model_state: Dict[str, Any]
    ) -> Any:
        """Calculate new value from function string"""

        try:
            context = {
                "simulation_time": self.current_simulation_time,
                "model_state": model_state,
                "system_state": self.system_state,
                "math": math,
                "abs": abs,
                "min": min,
                "max": max,
                "sin": math.sin,
                "cos": math.cos,
                "log": math.log,
                "exp": math.exp,
            }

            result = eval(value_function, {"__builtins__": {}}, context)
            return result

        except Exception as e:
            self.logger.warning(
                f"Failed to calculate value from function '{value_function}': {e}"
            )
            return None

    def _check_scheduled_trigger(self, rule: ModificationRule) -> bool:
        """Check if scheduled trigger condition is met"""

        # This would implement more sophisticated scheduling logic
        # For now, just use trigger_time if available
        return (
            rule.trigger_time is not None
            and self.current_simulation_time >= rule.trigger_time
        )

    def _check_adaptive_trigger(self, rule: ModificationRule) -> bool:
        """Check if adaptive trigger condition is met"""

        # Adaptive triggers fire based on performance deviation
        if not self.enable_adaptive_modifications:
            return False

        # Check if parameter has deviated from optimal range
        current_value = self._get_parameter_value(
            rule.parameter_path, self.system_state
        )
        target_value = rule.target_value

        if isinstance(current_value, (int, float)) and isinstance(
            target_value, (int, float)
        ):
            deviation = abs(current_value - target_value) / max(1.0, abs(target_value))
            return deviation > 0.1  # 10% deviation threshold

        return False

    def _execute_stepped_modification(
        self, rule: ModificationRule, new_value: Any
    ) -> Dict[str, Any]:
        """Execute stepped parameter modification"""

        # For stepped modifications, apply immediately but in discrete steps
        current_value = self._get_parameter_value(
            rule.parameter_path, self.system_state
        )

        if isinstance(current_value, (int, float)) and isinstance(
            new_value, (int, float)
        ):
            # Calculate step size
            step_size = (new_value - current_value) * rule.modification_rate
            stepped_value = current_value + step_size
        else:
            stepped_value = new_value

        return self._execute_parameter_modification(rule, stepped_value)

    def _update_system_health_metrics(self) -> None:
        """Update system health metrics"""

        # Calculate system health based on modification activity
        total_modifications = len(self.modification_history)
        recent_modifications = len(
            [
                h
                for h in self.modification_history
                if h.timestamp > self.current_simulation_time - 50.0
            ]
        )

        # Calculate modification success rate
        successful_modifications = len(
            [h for h in self.modification_history if h.success]
        )
        success_rate = successful_modifications / max(1, total_modifications)

        # Calculate stability (lower modification frequency = higher stability)
        stability_score = max(0.0, 1.0 - (recent_modifications / 10.0))

        self.system_health_metrics.update(
            {
                "modification_success_rate": success_rate,
                "system_stability": stability_score,
                "active_modifications": len(self.active_modifications),
                "total_rules": len(self.modification_rules),
                "recent_activity": recent_modifications,
            }
        )
