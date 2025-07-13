#!/usr/bin/env python3
"""
Test Script for Dynamic Parameter Modification System
====================================================

Demonstrates the dynamic parameter system functionality including
rule creation, parameter monitoring, and adaptive optimization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bstew.core.dynamic_parameters import (
    DynamicParameterSystem, ModificationRule, ParameterType, 
    ModificationTrigger, ModificationStrategy, ParameterConstraint
)
import time

def test_dynamic_parameter_system():
    """Test the dynamic parameter modification system"""
    
    print("=== Dynamic Parameter Modification System Test ===\n")
    
    # Initialize the system
    param_system = DynamicParameterSystem(
        enable_adaptive_modifications=True,
        max_simultaneous_modifications=5,
        modification_cooldown=1.0
    )
    
    print("1. Dynamic Parameter System initialized")
    print(f"   - Adaptive modifications: {param_system.enable_adaptive_modifications}")
    print(f"   - Max simultaneous modifications: {param_system.max_simultaneous_modifications}")
    print(f"   - Modification cooldown: {param_system.modification_cooldown}s\n")
    
    # Test 1: Add a time-based modification rule
    print("2. Adding time-based modification rule...")
    
    time_rule = ModificationRule(
        rule_id="test_time_rule",
        parameter_path="foraging.efficiency",
        parameter_type=ParameterType.FORAGING,
        trigger=ModificationTrigger.TIME_BASED,
        strategy=ModificationStrategy.IMMEDIATE,
        trigger_time=5.0,
        target_value=0.8,
        description="Reduce foraging efficiency at time 5"
    )
    
    param_system.add_modification_rule(time_rule)
    print(f"   - Added rule: {time_rule.rule_id}")
    print(f"   - Will trigger at time: {time_rule.trigger_time}")
    print(f"   - Target value: {time_rule.target_value}\n")
    
    # Test 2: Add a condition-based modification rule
    print("3. Adding condition-based modification rule...")
    
    condition_rule = ModificationRule(
        rule_id="test_condition_rule",
        parameter_path="colony.activity_level",
        parameter_type=ParameterType.COLONY,
        trigger=ModificationTrigger.CONDITION_BASED,
        strategy=ModificationStrategy.GRADUAL,
        trigger_condition="system_state.get('temperature', 20) < 10",
        target_value=0.3,
        modification_rate=0.05,
        duration=10.0,
        description="Reduce activity in cold weather"
    )
    
    param_system.add_modification_rule(condition_rule)
    print(f"   - Added rule: {condition_rule.rule_id}")
    print(f"   - Trigger condition: {condition_rule.trigger_condition}")
    print(f"   - Modification strategy: {condition_rule.strategy.value}\n")
    
    # Test 3: Add rule with constraints
    print("4. Adding rule with parameter constraints...")
    
    constraint = ParameterConstraint(
        min_value=0.1,
        max_value=2.0,
        dependencies=["colony.population"]
    )
    
    constrained_rule = ModificationRule(
        rule_id="test_constrained_rule",
        parameter_path="foraging.search_radius",
        parameter_type=ParameterType.FORAGING,
        trigger=ModificationTrigger.CONDITION_BASED,
        strategy=ModificationStrategy.STEPPED,
        trigger_condition="system_state.get('resource_scarcity', 0) > 0.7",
        target_value=1.5,
        modification_rate=0.1,
        constraint=constraint,
        description="Expand search radius when resources are scarce"
    )
    
    param_system.add_modification_rule(constrained_rule)
    print(f"   - Added rule: {constrained_rule.rule_id}")
    print(f"   - Constraints: min={constraint.min_value}, max={constraint.max_value}")
    print(f"   - Dependencies: {constraint.dependencies}\n")
    
    # Test 4: Simulate system updates
    print("5. Simulating system updates over time...")
    
    for sim_time in range(0, 15):
        # Mock model state
        model_state = {
            "temperature": 5.0 if sim_time > 7 else 20.0,  # Cold weather after time 7
            "resource_scarcity": 0.8 if sim_time > 10 else 0.3,  # Scarcity after time 10
            "foraging_efficiency": 1.0,
            "colony_activity_level": 1.0,
            "foraging_search_radius": 1.0,
            "total_population": 100,
            "colony_health": 0.8
        }
        
        # Update parameter system
        param_system.update_system_state(float(sim_time), model_state)
        
        # Check for active modifications
        if param_system.active_modifications:
            print(f"   Time {sim_time}: {len(param_system.active_modifications)} active modifications")
        
        # Check for recent modifications
        recent_modifications = [h for h in param_system.modification_history 
                              if h.timestamp >= sim_time - 1]
        if recent_modifications:
            for mod in recent_modifications:
                print(f"   Time {sim_time}: Applied {mod.rule_id} - {mod.parameter_path} -> {mod.new_value}")
        
        time.sleep(0.1)  # Small delay for demonstration
    
    print()
    
    # Test 5: Manual parameter modification
    print("6. Testing manual parameter modification...")
    
    manual_result = param_system.apply_manual_modification(
        "test.manual_parameter", 42.0, duration=5.0
    )
    
    print(f"   - Manual modification result: {manual_result}")
    print()
    
    # Test 6: Adaptive optimization
    print("7. Testing adaptive parameter optimization...")
    
    target_metrics = {
        "foraging_efficiency": 0.9,
        "colony_health": 0.95,
        "population_growth": 0.1
    }
    
    optimization_result = param_system.optimize_parameters_adaptive(
        target_metrics, learning_rate=0.1
    )
    
    print(f"   - Optimization targets: {target_metrics}")
    print(f"   - Parameter adjustments: {len(optimization_result.get('parameter_adjustments', {}))}")
    print(f"   - Optimization ID: {optimization_result.get('optimization_id', 'N/A')}")
    print()
    
    # Test 7: Get analytics
    print("8. Getting system analytics...")
    
    system_analytics = param_system.get_parameter_analytics()
    
    print(f"   - Total rules: {system_analytics.get('total_rules', 0)}")
    print(f"   - Active modifications: {system_analytics.get('active_modifications', 0)}")
    print(f"   - Total modifications applied: {system_analytics.get('total_modifications', 0)}")
    print(f"   - System health: {system_analytics.get('system_health', {})}")
    print()
    
    # Test 8: Parameter-specific analytics
    print("9. Getting parameter-specific analytics...")
    
    for param_path in ["foraging.efficiency", "colony.activity_level"]:
        param_analytics = param_system.get_parameter_analytics(param_path)
        if "error" not in param_analytics:
            print(f"   - {param_path}:")
            print(f"     Current value: {param_analytics.get('current_value', 'N/A')}")
            print(f"     Change count: {param_analytics.get('change_count', 0)}")
            print(f"     Last change: {param_analytics.get('last_change_time', 'Never')}")
    
    print()
    
    # Summary
    print("=== Test Summary ===")
    print(f"Rules created: {len(param_system.modification_rules)}")
    print(f"Modifications applied: {len(param_system.modification_history)}")
    print(f"Parameters monitored: {len(param_system.parameter_monitors)}")
    print(f"System health score: {param_system.system_health_metrics.get('system_stability', 'N/A')}")
    
    print("\n✅ Dynamic Parameter System test completed successfully!")

if __name__ == "__main__":
    test_dynamic_parameter_system()