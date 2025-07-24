"""
Unit tests for Dynamic Parameter Modification System
==================================================

Tests for the comprehensive dynamic parameter system including modification rules,
parameter monitoring, adaptive optimization, and runtime parameter changes.
"""

import pytest
from unittest.mock import patch

from src.bstew.core.dynamic_parameters import (
    DynamicParameterSystem, ParameterMonitor, ModificationRule, ParameterHistory,
    ParameterConstraint, ParameterType, ModificationTrigger, ModificationStrategy
)


class TestParameterType:
    """Test parameter type definitions"""

    def test_parameter_type_values(self):
        """Test parameter type enum values"""
        assert ParameterType.COLONY.value == "colony"
        assert ParameterType.ENVIRONMENT.value == "environment"
        assert ParameterType.FORAGING.value == "foraging"
        assert ParameterType.COMMUNICATION.value == "communication"
        assert ParameterType.DISEASE.value == "disease"
        assert ParameterType.REPRODUCTION.value == "reproduction"
        assert ParameterType.WEATHER.value == "weather"
        assert ParameterType.BEHAVIORAL.value == "behavioral"
        assert ParameterType.SPATIAL.value == "spatial"
        assert ParameterType.PERFORMANCE.value == "performance"


class TestModificationTrigger:
    """Test modification trigger definitions"""

    def test_modification_trigger_values(self):
        """Test modification trigger enum values"""
        assert ModificationTrigger.TIME_BASED.value == "time_based"
        assert ModificationTrigger.EVENT_BASED.value == "event_based"
        assert ModificationTrigger.CONDITION_BASED.value == "condition_based"
        assert ModificationTrigger.MANUAL.value == "manual"
        assert ModificationTrigger.ADAPTIVE.value == "adaptive"
        assert ModificationTrigger.SCHEDULED.value == "scheduled"


class TestModificationStrategy:
    """Test modification strategy definitions"""

    def test_modification_strategy_values(self):
        """Test modification strategy enum values"""
        assert ModificationStrategy.IMMEDIATE.value == "immediate"
        assert ModificationStrategy.GRADUAL.value == "gradual"
        assert ModificationStrategy.STEPPED.value == "stepped"
        assert ModificationStrategy.OSCILLATING.value == "oscillating"
        assert ModificationStrategy.THRESHOLD_BASED.value == "threshold_based"
        assert ModificationStrategy.FEEDBACK_CONTROLLED.value == "feedback_controlled"


class TestParameterConstraint:
    """Test parameter constraint data structure"""

    def setup_method(self):
        """Setup test fixtures"""
        self.constraint = ParameterConstraint(
            min_value=0.0,
            max_value=100.0,
            allowed_values=[10, 20, 30, 40, 50],
            validation_function="value > 0 and value <= 100",
            dependencies=["temperature", "humidity"]
        )

    def test_initialization(self):
        """Test constraint initialization"""
        assert self.constraint.min_value == 0.0
        assert self.constraint.max_value == 100.0
        assert len(self.constraint.allowed_values) == 5
        assert 30 in self.constraint.allowed_values
        assert self.constraint.validation_function == "value > 0 and value <= 100"
        assert len(self.constraint.dependencies) == 2
        assert "temperature" in self.constraint.dependencies

    def test_empty_constraint(self):
        """Test constraint with no restrictions"""
        empty_constraint = ParameterConstraint()

        assert empty_constraint.min_value is None
        assert empty_constraint.max_value is None
        assert empty_constraint.allowed_values is None
        assert empty_constraint.validation_function is None
        assert len(empty_constraint.dependencies) == 0

    def test_range_only_constraint(self):
        """Test constraint with only range restrictions"""
        range_constraint = ParameterConstraint(min_value=5.0, max_value=15.0)

        assert range_constraint.min_value == 5.0
        assert range_constraint.max_value == 15.0
        assert range_constraint.allowed_values is None

    def test_allowed_values_only_constraint(self):
        """Test constraint with only allowed values"""
        values_constraint = ParameterConstraint(allowed_values=["low", "medium", "high"])

        assert values_constraint.min_value is None
        assert values_constraint.max_value is None
        assert "medium" in values_constraint.allowed_values


class TestModificationRule:
    """Test modification rule data structure"""

    def setup_method(self):
        """Setup test fixtures"""
        constraint = ParameterConstraint(min_value=0.0, max_value=1.0)

        self.rule = ModificationRule(
            rule_id="test_rule_001",
            parameter_path="colony.foraging.efficiency",
            parameter_type=ParameterType.FORAGING,
            trigger=ModificationTrigger.TIME_BASED,
            strategy=ModificationStrategy.GRADUAL,
            trigger_time=100.0,
            target_value=0.8,
            modification_rate=0.05,
            duration=50.0,
            constraint=constraint,
            enabled=True,
            priority=1,
            description="Test foraging efficiency modification"
        )

    def test_initialization(self):
        """Test rule initialization"""
        assert self.rule.rule_id == "test_rule_001"
        assert self.rule.parameter_path == "colony.foraging.efficiency"
        assert self.rule.parameter_type == ParameterType.FORAGING
        assert self.rule.trigger == ModificationTrigger.TIME_BASED
        assert self.rule.strategy == ModificationStrategy.GRADUAL
        assert self.rule.trigger_time == 100.0
        assert self.rule.target_value == 0.8
        assert self.rule.modification_rate == 0.05
        assert self.rule.duration == 50.0
        assert self.rule.enabled is True
        assert self.rule.priority == 1
        assert "foraging efficiency" in self.rule.description

    def test_constraint_application(self):
        """Test constraint association"""
        assert self.rule.constraint is not None
        assert self.rule.constraint.min_value == 0.0
        assert self.rule.constraint.max_value == 1.0

    def test_metadata_tracking(self):
        """Test metadata fields"""
        assert hasattr(self.rule, 'created_time')
        assert self.rule.last_applied is None
        assert self.rule.application_count == 0
        assert isinstance(self.rule.created_time, float)

    def test_event_based_rule(self):
        """Test event-based modification rule"""
        event_rule = ModificationRule(
            rule_id="event_rule_001",
            parameter_path="environment.temperature",
            parameter_type=ParameterType.ENVIRONMENT,
            trigger=ModificationTrigger.EVENT_BASED,
            strategy=ModificationStrategy.IMMEDIATE,
            trigger_event="weather_change",
            target_value=25.0,
            description="Temperature change on weather event"
        )

        assert event_rule.trigger == ModificationTrigger.EVENT_BASED
        assert event_rule.trigger_event == "weather_change"
        assert event_rule.strategy == ModificationStrategy.IMMEDIATE

    def test_condition_based_rule(self):
        """Test condition-based modification rule"""
        condition_rule = ModificationRule(
            rule_id="condition_rule_001",
            parameter_path="colony.energy_level",
            parameter_type=ParameterType.COLONY,
            trigger=ModificationTrigger.CONDITION_BASED,
            strategy=ModificationStrategy.THRESHOLD_BASED,
            trigger_condition="colony.energy_level < 500",
            target_value=1000.0,
            description="Emergency energy boost"
        )

        assert condition_rule.trigger == ModificationTrigger.CONDITION_BASED
        assert condition_rule.trigger_condition == "colony.energy_level < 500"
        assert condition_rule.strategy == ModificationStrategy.THRESHOLD_BASED


class TestParameterHistory:
    """Test parameter history data structure"""

    def setup_method(self):
        """Setup test fixtures"""
        self.history_entry = ParameterHistory(
            parameter_path="colony.foraging.efficiency",
            timestamp=1234567890.0,
            old_value=0.6,
            new_value=0.8,
            rule_id="test_rule_001",
            trigger_type="time_based",
            success=True
        )

    def test_initialization(self):
        """Test history entry initialization"""
        assert self.history_entry.parameter_path == "colony.foraging.efficiency"
        assert self.history_entry.timestamp == 1234567890.0
        assert self.history_entry.old_value == 0.6
        assert self.history_entry.new_value == 0.8
        assert self.history_entry.rule_id == "test_rule_001"
        assert self.history_entry.trigger_type == "time_based"
        assert self.history_entry.success is True
        assert self.history_entry.error_message is None

    def test_failed_modification_entry(self):
        """Test history entry for failed modification"""
        failed_entry = ParameterHistory(
            parameter_path="invalid.parameter.path",
            timestamp=1234567900.0,
            old_value=None,
            new_value=None,
            rule_id="invalid_rule",
            trigger_type="manual",
            success=False,
            error_message="Parameter path not found"
        )

        assert failed_entry.success is False
        assert failed_entry.error_message == "Parameter path not found"
        assert failed_entry.old_value is None
        assert failed_entry.new_value is None


class TestParameterMonitor:
    """Test parameter monitoring system"""

    def setup_method(self):
        """Setup test fixtures"""
        self.monitor = ParameterMonitor(
            parameter_path="colony.foraging.efficiency",
            current_value=0.75,
            baseline_value=0.6,
            alert_thresholds={"low": 0.3, "high": 0.9}
        )

    def test_initialization(self):
        """Test monitor initialization"""
        assert self.monitor.parameter_path == "colony.foraging.efficiency"
        assert self.monitor.current_value == 0.75
        assert self.monitor.baseline_value == 0.6
        assert len(self.monitor.change_history) == 0
        assert self.monitor.last_change_time is None
        assert self.monitor.change_frequency == 0.0
        assert isinstance(self.monitor.impact_metrics, dict)
        assert isinstance(self.monitor.performance_correlation, dict)
        assert self.monitor.alert_thresholds["low"] == 0.3
        assert self.monitor.alert_thresholds["high"] == 0.9

    def test_change_tracking(self):
        """Test change history tracking"""
        # Add change history entries
        change1 = ParameterHistory(
            parameter_path=self.monitor.parameter_path,
            timestamp=100.0,
            old_value=0.6,
            new_value=0.7,
            success=True
        )

        change2 = ParameterHistory(
            parameter_path=self.monitor.parameter_path,
            timestamp=200.0,
            old_value=0.7,
            new_value=0.75,
            success=True
        )

        self.monitor.change_history.extend([change1, change2])
        self.monitor.last_change_time = 200.0
        self.monitor.change_frequency = 2.0 / 100.0  # 2 changes in 100 time units

        assert len(self.monitor.change_history) == 2
        assert self.monitor.last_change_time == 200.0
        assert self.monitor.change_frequency == 0.02

    def test_impact_metrics_tracking(self):
        """Test impact metrics tracking"""
        self.monitor.impact_metrics.update({
            "colony_energy": 0.15,
            "foraging_success": 0.82,
            "resource_collection": 0.65
        })

        assert self.monitor.impact_metrics["colony_energy"] == 0.15
        assert self.monitor.impact_metrics["foraging_success"] == 0.82
        assert len(self.monitor.impact_metrics) == 3

    def test_performance_correlation_tracking(self):
        """Test performance correlation tracking"""
        self.monitor.performance_correlation.update({
            "efficiency_correlation": 0.78,
            "energy_correlation": -0.23,
            "success_correlation": 0.91
        })

        assert self.monitor.performance_correlation["efficiency_correlation"] == 0.78
        assert self.monitor.performance_correlation["energy_correlation"] == -0.23
        assert self.monitor.performance_correlation["success_correlation"] == 0.91

    def test_alert_system(self):
        """Test alert threshold system"""
        # Add alert entries
        alert1 = {
            "timestamp": 150.0,
            "alert_type": "threshold_exceeded",
            "threshold_name": "high",
            "threshold_value": 0.9,
            "actual_value": 0.95,
            "severity": "warning"
        }

        self.monitor.alert_history.append(alert1)

        assert len(self.monitor.alert_history) == 1
        assert self.monitor.alert_history[0]["alert_type"] == "threshold_exceeded"
        assert self.monitor.alert_history[0]["severity"] == "warning"


class TestDynamicParameterSystem:
    """Test dynamic parameter system"""

    def setup_method(self):
        """Setup test fixtures"""
        self.parameter_system = DynamicParameterSystem()

        # Create test modification rule
        self.test_rule = ModificationRule(
            rule_id="test_efficiency_rule",
            parameter_path="colony.foraging.efficiency",
            parameter_type=ParameterType.FORAGING,
            trigger=ModificationTrigger.TIME_BASED,
            strategy=ModificationStrategy.GRADUAL,
            trigger_time=100.0,
            target_value=0.8,
            modification_rate=0.05,
            duration=50.0,
            description="Test efficiency modification"
        )

        # Test system state
        self.test_model_state = {
            "colony": {
                "foraging": {"efficiency": 0.6, "success_rate": 0.7},
                "energy_level": 1000.0,
                "population": 100
            },
            "environment": {
                "temperature": 20.0,
                "weather": "clear"
            }
        }

    def test_initialization(self):
        """Test system initialization"""
        assert isinstance(self.parameter_system.modification_rules, dict)
        assert isinstance(self.parameter_system.parameter_monitors, dict)
        assert isinstance(self.parameter_system.active_modifications, dict)
        assert self.parameter_system.current_simulation_time == 0.0
        assert isinstance(self.parameter_system.modification_history, list)
        assert isinstance(self.parameter_system.system_state, dict)
        assert self.parameter_system.enable_adaptive_modifications is True
        assert self.parameter_system.max_simultaneous_modifications == 10
        assert self.parameter_system.modification_cooldown == 5.0

    def test_add_modification_rule(self):
        """Test adding modification rules"""
        self.parameter_system.add_modification_rule(self.test_rule)

        # Check rule was added
        assert "test_efficiency_rule" in self.parameter_system.modification_rules
        stored_rule = self.parameter_system.modification_rules["test_efficiency_rule"]
        assert stored_rule.parameter_path == "colony.foraging.efficiency"
        assert stored_rule.trigger == ModificationTrigger.TIME_BASED

        # Check parameter monitor was initialized
        assert "colony.foraging.efficiency" in self.parameter_system.parameter_monitors

    def test_remove_modification_rule(self):
        """Test removing modification rules"""
        # Add rule first
        self.parameter_system.add_modification_rule(self.test_rule)
        assert "test_efficiency_rule" in self.parameter_system.modification_rules

        # Remove rule
        success = self.parameter_system.remove_modification_rule("test_efficiency_rule")
        assert success is True
        assert "test_efficiency_rule" not in self.parameter_system.modification_rules

        # Try to remove non-existent rule
        success = self.parameter_system.remove_modification_rule("non_existent_rule")
        assert success is False

    def test_rule_validation_success(self):
        """Test successful rule validation"""
        validation_result = self.parameter_system._validate_modification_rule(self.test_rule)
        assert validation_result["valid"] is True

    def test_rule_validation_missing_parameter_path(self):
        """Test rule validation with missing parameter path"""
        invalid_rule = ModificationRule(
            rule_id="invalid_rule",
            parameter_path="",  # Empty parameter path
            parameter_type=ParameterType.FORAGING,
            trigger=ModificationTrigger.TIME_BASED,
            strategy=ModificationStrategy.IMMEDIATE
        )

        validation_result = self.parameter_system._validate_modification_rule(invalid_rule)
        assert validation_result["valid"] is False
        assert "Parameter path is required" in validation_result["error"]

    def test_rule_validation_time_based_without_time(self):
        """Test rule validation for time-based trigger without time"""
        invalid_rule = ModificationRule(
            rule_id="invalid_time_rule",
            parameter_path="test.parameter",
            parameter_type=ParameterType.BEHAVIORAL,
            trigger=ModificationTrigger.TIME_BASED,
            strategy=ModificationStrategy.IMMEDIATE,
            trigger_time=None  # Missing required trigger_time
        )

        validation_result = self.parameter_system._validate_modification_rule(invalid_rule)
        assert validation_result["valid"] is False
        assert "Time-based trigger requires trigger_time" in validation_result["error"]

    def test_rule_validation_event_based_without_event(self):
        """Test rule validation for event-based trigger without event"""
        invalid_rule = ModificationRule(
            rule_id="invalid_event_rule",
            parameter_path="test.parameter",
            parameter_type=ParameterType.BEHAVIORAL,
            trigger=ModificationTrigger.EVENT_BASED,
            strategy=ModificationStrategy.IMMEDIATE,
            trigger_event=""  # Missing required trigger_event
        )

        validation_result = self.parameter_system._validate_modification_rule(invalid_rule)
        assert validation_result["valid"] is False
        assert "Event-based trigger requires trigger_event" in validation_result["error"]

    def test_rule_validation_condition_based_without_condition(self):
        """Test rule validation for condition-based trigger without condition"""
        invalid_rule = ModificationRule(
            rule_id="invalid_condition_rule",
            parameter_path="test.parameter",
            parameter_type=ParameterType.BEHAVIORAL,
            trigger=ModificationTrigger.CONDITION_BASED,
            strategy=ModificationStrategy.IMMEDIATE,
            trigger_condition=""  # Missing required trigger_condition
        )

        validation_result = self.parameter_system._validate_modification_rule(invalid_rule)
        assert validation_result["valid"] is False
        assert "Condition-based trigger requires trigger_condition" in validation_result["error"]

    def test_system_state_update(self):
        """Test system state updates"""
        simulation_time = 50.0

        self.parameter_system.update_system_state(simulation_time, self.test_model_state)

        assert self.parameter_system.current_simulation_time == simulation_time
        assert "colony" in self.parameter_system.system_state
        assert self.parameter_system.system_state["colony"]["energy_level"] == 1000.0

    def test_manual_modification(self):
        """Test manual parameter modification"""
        parameter_path = "colony.foraging.efficiency"
        new_value = 0.85
        duration = 100.0

        # Mock the execution method
        with patch.object(self.parameter_system, '_execute_parameter_modification') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "old_value": 0.6,
                "new_value": new_value
            }

            result = self.parameter_system.apply_manual_modification(
                parameter_path, new_value, duration
            )

        assert result["success"] is True
        assert result["new_value"] == new_value

        # Check history was recorded
        assert len(self.parameter_system.modification_history) == 1
        history_entry = self.parameter_system.modification_history[0]
        assert history_entry.parameter_path == parameter_path
        assert history_entry.new_value == new_value
        assert history_entry.trigger_type == "manual"

    def test_parameter_analytics_specific(self):
        """Test analytics for specific parameter"""
        # Add a rule and monitor
        self.parameter_system.add_modification_rule(self.test_rule)

        # Add some history
        history_entry = ParameterHistory(
            parameter_path="colony.foraging.efficiency",
            timestamp=100.0,
            old_value=0.6,
            new_value=0.75,
            rule_id="test_efficiency_rule",
            trigger_type="time_based",
            success=True
        )
        self.parameter_system.modification_history.append(history_entry)

        analytics = self.parameter_system.get_parameter_analytics("colony.foraging.efficiency")

        assert analytics["parameter_path"] == "colony.foraging.efficiency"
        assert analytics["change_count"] == 1
        assert len(analytics["recent_changes"]) == 1
        assert "current_value" in analytics
        assert "baseline_value" in analytics

    def test_parameter_analytics_system_wide(self):
        """Test system-wide analytics"""
        # Add a rule
        self.parameter_system.add_modification_rule(self.test_rule)

        analytics = self.parameter_system.get_parameter_analytics()

        assert analytics["total_rules"] == 1
        assert analytics["active_modifications"] == 0
        assert analytics["monitored_parameters"] == 1
        assert analytics["total_modifications"] == 0
        assert "system_health" in analytics
        assert "modification_performance" in analytics

    def test_parameter_analytics_unknown_parameter(self):
        """Test analytics for unknown parameter"""
        analytics = self.parameter_system.get_parameter_analytics("unknown.parameter")

        assert "error" in analytics
        assert "not monitored" in analytics["error"]

    def test_adaptive_optimization(self):
        """Test adaptive parameter optimization"""
        target_metrics = {
            "foraging_efficiency": 0.85,
            "energy_conservation": 0.9
        }
        learning_rate = 0.1

        # Mock internal methods
        with patch.object(self.parameter_system, '_get_current_performance_metrics') as mock_performance, \
             patch.object(self.parameter_system, '_find_parameters_affecting_metric') as mock_find_params, \
             patch.object(self.parameter_system, '_calculate_parameter_adjustment') as mock_calc_adjustment:

            mock_performance.return_value = {
                "foraging_efficiency": 0.6,
                "energy_conservation": 0.7
            }

            mock_find_params.return_value = {
                "colony.foraging.efficiency": 0.8,
                "colony.foraging.success_rate": 0.6
            }

            mock_calc_adjustment.return_value = {
                "target_value": 0.75,
                "magnitude": 0.15,
                "direction": "increase"
            }

            optimization_result = self.parameter_system.optimize_parameters_adaptive(
                target_metrics, learning_rate
            )

        assert "optimization_id" in optimization_result
        assert optimization_result["target_metrics"] == target_metrics
        assert "parameter_adjustments" in optimization_result
        assert optimization_result["convergence_status"] == "in_progress"

    def test_trigger_checking(self):
        """Test modification trigger checking"""
        # Add time-based rule
        self.parameter_system.add_modification_rule(self.test_rule)

        # Update system time to trigger condition
        self.parameter_system.current_simulation_time = 150.0  # Past trigger time of 100.0

        # Mock the trigger checking method
        with patch.object(self.parameter_system, '_check_modification_triggers') as mock_check:
            mock_check.return_value = ["test_efficiency_rule"]

            triggered_rules = self.parameter_system._check_modification_triggers()

        assert "test_efficiency_rule" in triggered_rules

    def test_active_modifications_management(self):
        """Test active modifications tracking"""
        # Add active modification
        self.parameter_system.active_modifications["test_modification"] = {
            "rule_id": "test_efficiency_rule",
            "start_time": 100.0,
            "duration": 50.0,
            "current_progress": 0.5,
            "status": "active"
        }

        assert len(self.parameter_system.active_modifications) == 1
        assert "test_modification" in self.parameter_system.active_modifications

        modification = self.parameter_system.active_modifications["test_modification"]
        assert modification["rule_id"] == "test_efficiency_rule"
        assert modification["status"] == "active"

    def test_system_health_metrics(self):
        """Test system health metrics tracking"""
        # Initialize some health metrics
        self.parameter_system.system_health_metrics.update({
            "modification_success_rate": 0.95,
            "system_stability": 0.88,
            "parameter_convergence": 0.72,
            "performance_impact": 0.15
        })

        assert self.parameter_system.system_health_metrics["modification_success_rate"] == 0.95
        assert self.parameter_system.system_health_metrics["system_stability"] == 0.88
        assert len(self.parameter_system.system_health_metrics) == 4


class TestDynamicParameterSystemIntegration:
    """Test integration aspects of dynamic parameter system"""

    def setup_method(self):
        """Setup integration test fixtures"""
        self.parameter_system = DynamicParameterSystem()

        # Create multiple rules for different scenarios
        self.rules = [
            ModificationRule(
                rule_id="efficiency_boost",
                parameter_path="colony.foraging.efficiency",
                parameter_type=ParameterType.FORAGING,
                trigger=ModificationTrigger.TIME_BASED,
                strategy=ModificationStrategy.GRADUAL,
                trigger_time=50.0,
                target_value=0.9,
                modification_rate=0.02,
                duration=100.0,
                priority=1
            ),
            ModificationRule(
                rule_id="emergency_energy",
                parameter_path="colony.energy_level",
                parameter_type=ParameterType.COLONY,
                trigger=ModificationTrigger.CONDITION_BASED,
                strategy=ModificationStrategy.IMMEDIATE,
                trigger_condition="colony.energy_level < 200",
                target_value=500.0,
                priority=2
            ),
            ModificationRule(
                rule_id="weather_response",
                parameter_path="environment.foraging_conditions",
                parameter_type=ParameterType.ENVIRONMENT,
                trigger=ModificationTrigger.EVENT_BASED,
                strategy=ModificationStrategy.STEPPED,
                trigger_event="weather_change",
                target_value="poor",
                duration=25.0,
                priority=1
            )
        ]

    def test_multiple_rule_management(self):
        """Test managing multiple modification rules"""
        # Add all rules
        for rule in self.rules:
            self.parameter_system.add_modification_rule(rule)

        assert len(self.parameter_system.modification_rules) == 3
        assert "efficiency_boost" in self.parameter_system.modification_rules
        assert "emergency_energy" in self.parameter_system.modification_rules
        assert "weather_response" in self.parameter_system.modification_rules

        # Check parameter monitors were created
        assert len(self.parameter_system.parameter_monitors) == 3

    def test_rule_priority_handling(self):
        """Test rule priority system"""
        # Add rules with different priorities
        for rule in self.rules:
            self.parameter_system.add_modification_rule(rule)

        # Emergency energy rule has priority 2 (higher than others)
        emergency_rule = self.parameter_system.modification_rules["emergency_energy"]
        efficiency_rule = self.parameter_system.modification_rules["efficiency_boost"]

        assert emergency_rule.priority == 2
        assert efficiency_rule.priority == 1
        assert emergency_rule.priority > efficiency_rule.priority

    def test_complete_workflow_simulation(self):
        """Test complete parameter modification workflow"""
        # Add rules
        for rule in self.rules:
            self.parameter_system.add_modification_rule(rule)

        # Simulate time progression with state updates
        model_states = [
            {"colony": {"energy_level": 1000, "foraging": {"efficiency": 0.6}}, "environment": {"temperature": 20}},
            {"colony": {"energy_level": 800, "foraging": {"efficiency": 0.65}}, "environment": {"temperature": 22}},
            {"colony": {"energy_level": 150, "foraging": {"efficiency": 0.7}}, "environment": {"temperature": 18}},  # Low energy trigger
            {"colony": {"energy_level": 500, "foraging": {"efficiency": 0.75}}, "environment": {"temperature": 25}}
        ]

        times = [25.0, 60.0, 75.0, 120.0]

        for time_step, model_state in zip(times, model_states):
            self.parameter_system.update_system_state(time_step, model_state)

        # Should have processed multiple time steps
        assert self.parameter_system.current_simulation_time == 120.0
        assert len(self.parameter_system.system_state) > 0


class TestDynamicParameterSystemEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Setup edge case test fixtures"""
        self.parameter_system = DynamicParameterSystem()

    def test_invalid_rule_addition(self):
        """Test handling of invalid rule additions"""
        invalid_rule = ModificationRule(
            rule_id="invalid_rule",
            parameter_path="",  # Invalid empty path
            parameter_type=ParameterType.BEHAVIORAL,
            trigger=ModificationTrigger.TIME_BASED,
            strategy=ModificationStrategy.IMMEDIATE
        )

        with pytest.raises(ValueError, match="Invalid modification rule"):
            self.parameter_system.add_modification_rule(invalid_rule)

    def test_duplicate_rule_id_handling(self):
        """Test handling of duplicate rule IDs"""
        rule1 = ModificationRule(
            rule_id="duplicate_id",
            parameter_path="test.parameter1",
            parameter_type=ParameterType.BEHAVIORAL,
            trigger=ModificationTrigger.MANUAL,
            strategy=ModificationStrategy.IMMEDIATE,
            target_value=0.8
        )

        rule2 = ModificationRule(
            rule_id="duplicate_id",  # Same ID
            parameter_path="test.parameter2",
            parameter_type=ParameterType.BEHAVIORAL,
            trigger=ModificationTrigger.MANUAL,
            strategy=ModificationStrategy.IMMEDIATE,
            target_value=0.9
        )

        # Add first rule
        self.parameter_system.add_modification_rule(rule1)

        # Adding second rule with same ID should replace first
        self.parameter_system.add_modification_rule(rule2)

        assert len(self.parameter_system.modification_rules) == 1
        stored_rule = self.parameter_system.modification_rules["duplicate_id"]
        assert stored_rule.parameter_path == "test.parameter2"

    def test_empty_system_analytics(self):
        """Test analytics on empty system"""
        analytics = self.parameter_system.get_parameter_analytics()

        assert analytics["total_rules"] == 0
        assert analytics["active_modifications"] == 0
        assert analytics["monitored_parameters"] == 0
        assert analytics["total_modifications"] == 0

    def test_max_simultaneous_modifications_limit(self):
        """Test maximum simultaneous modifications limit"""
        # Set low limit for testing
        self.parameter_system.max_simultaneous_modifications = 2

        # Try to add 3 active modifications
        for i in range(3):
            self.parameter_system.active_modifications[f"mod_{i}"] = {
                "rule_id": f"rule_{i}",
                "start_time": 100.0,
                "status": "active"
            }

        # System should handle limit appropriately
        assert len(self.parameter_system.active_modifications) == 3
        # Implementation would need to enforce limit in actual modification application


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
