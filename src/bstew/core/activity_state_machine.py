"""
Activity State Machine for NetLogo BEE-STEWARD v2 Compatibility
==============================================================

Implements detailed activity state transitions and behaviors matching
NetLogo's complex activity system with 15+ states.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator
import math

from .enums import BeeStatus, ActivityStateCategory


# ActivityStateCategory is now imported from enums module


class StateTransitionRule(BaseModel):
    """Rules for state transitions"""
    
    model_config = {"validate_assignment": True}
    
    from_state: BeeStatus = Field(description="Source state")
    to_state: BeeStatus = Field(description="Target state")
    probability: float = Field(ge=0.0, le=1.0, description="Transition probability")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Transition conditions")
    min_duration: int = Field(default=0, ge=0, description="Minimum steps in current state")
    max_duration: Optional[int] = Field(default=None, ge=0, description="Maximum steps in current state")
    
    @field_validator("max_duration")
    @classmethod
    def validate_max_duration(cls, v: Optional[int]) -> Optional[int]:
        # Note: In Pydantic v2, we can't access other field values during validation
        # This validation would need to be done at the model level if needed
        return v


class ActivityStateConfig(BaseModel):
    """Configuration for activity states"""
    
    model_config = {"validate_assignment": True}
    
    energy_consumption: float = Field(default=1.0, ge=0.0, description="Energy consumption per step")
    duration_range: Tuple[int, int] = Field(default=(1, 10), description="Duration range in steps")
    required_conditions: Dict[str, Any] = Field(default_factory=dict, description="Required conditions")
    energy_cost_multiplier: float = Field(default=1.0, ge=0.0, description="Energy cost multiplier")
    success_probability: float = Field(default=1.0, ge=0.0, le=1.0, description="Activity success probability")
    
    # Physiological parameters
    flight_speed: float = Field(default=5.0, ge=0.0, description="Flight speed in m/s")
    metabolic_rate: float = Field(default=1.0, ge=0.0, description="Metabolic rate multiplier")
    stress_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Stress level (0-1)")
    thermoregulation_cost: float = Field(default=0.0, ge=0.0, description="Thermoregulation energy cost")
    
    @field_validator("duration_range")
    @classmethod
    def validate_duration_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("Duration range minimum must be <= maximum")
        return v


class PersonalTimeTracker(BaseModel):
    """Personal time tracking for individual bees with enhanced context-awareness"""
    
    model_config = {"validate_assignment": True}
    
    current_state: BeeStatus = Field(default=BeeStatus.ALIVE, description="Current activity state")
    state_start_time: int = Field(default=0, ge=0, description="Time when current state began")
    state_duration: int = Field(default=0, ge=0, description="Duration in current state")
    daily_activities: Dict[str, int] = Field(default_factory=dict, description="Daily activity tracking")
    total_foraging_time: int = Field(default=0, ge=0, description="Total foraging time")
    total_nursing_time: int = Field(default=0, ge=0, description="Total nursing time")
    last_activity_change: int = Field(default=0, ge=0, description="Last activity change time")
    
    # Enhanced context-awareness fields
    state_transition_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent state transitions")
    success_history: List[bool] = Field(default_factory=list, description="Recent success/failure history")
    energy_history: List[float] = Field(default_factory=list, description="Recent energy levels")
    environmental_context: Dict[str, Any] = Field(default_factory=dict, description="Environmental context memory")
    behavioral_preferences: Dict[str, float] = Field(default_factory=dict, description="Learned behavioral preferences")
    adaptation_factors: Dict[str, float] = Field(default_factory=dict, description="Adaptation factors for decision making")
    
    def update_time(self, current_step: int) -> None:
        """Update time tracking with enhanced context awareness"""
        self.state_duration = current_step - self.state_start_time
        
        # Update daily activities
        activity_key = self.current_state.value
        if activity_key not in self.daily_activities:
            self.daily_activities[activity_key] = 0
        self.daily_activities[activity_key] += 1
        
        # Update specialized counters
        if self.current_state in [BeeStatus.NECTAR_FORAGING, BeeStatus.POLLEN_FORAGING, 
                                 BeeStatus.COLLECT_NECTAR, BeeStatus.COLLECT_POLLEN]:
            self.total_foraging_time += 1
        elif self.current_state == BeeStatus.NURSING:
            self.total_nursing_time += 1
    
    def transition_to_state(self, new_state: BeeStatus, current_step: int) -> None:
        """Transition to new state with context tracking"""
        if new_state != self.current_state:
            # Record transition in history
            transition_record = {
                "from_state": self.current_state.value,
                "to_state": new_state.value,
                "step": current_step,
                "duration_in_previous": self.state_duration
            }
            self.state_transition_history.append(transition_record)
            
            # Maintain history size (keep last 50 transitions)
            if len(self.state_transition_history) > 50:
                self.state_transition_history.pop(0)
            
            # Update current state
            self.current_state = new_state
            self.state_start_time = current_step
            self.state_duration = 0
            self.last_activity_change = current_step
    
    def record_activity_success(self, success: bool) -> None:
        """Record success/failure of current activity"""
        self.success_history.append(success)
        
        # Maintain history size (keep last 20 attempts)
        if len(self.success_history) > 20:
            self.success_history.pop(0)
    
    def record_energy_level(self, energy: float) -> None:
        """Record energy level for trend analysis"""
        self.energy_history.append(energy)
        
        # Maintain history size (keep last 30 readings)
        if len(self.energy_history) > 30:
            self.energy_history.pop(0)
    
    def update_environmental_context(self, context: Dict[str, Any]) -> None:
        """Update environmental context memory"""
        self.environmental_context.update(context)
    
    def get_recent_success_rate(self, window: int = 10) -> float:
        """Get recent success rate for adaptive behavior"""
        if not self.success_history:
            return 0.5  # Default neutral rate
        
        recent_attempts = self.success_history[-window:]
        if not recent_attempts:
            return 0.5
        
        return sum(recent_attempts) / len(recent_attempts)
    
    def get_energy_trend(self) -> str:
        """Get energy trend (increasing, decreasing, stable)"""
        if len(self.energy_history) < 3:
            return "stable"
        
        recent_values = self.energy_history[-3:]
        if recent_values[-1] > recent_values[0] + 5:
            return "increasing"
        elif recent_values[-1] < recent_values[0] - 5:
            return "decreasing"
        else:
            return "stable"
    
    def get_preferred_state_sequence(self) -> List[str]:
        """Get most common state transition sequence"""
        if len(self.state_transition_history) < 2:
            return []
        
        # Find most common 2-state sequences
        sequences = {}
        for i in range(len(self.state_transition_history) - 1):
            seq = (self.state_transition_history[i]["to_state"], 
                  self.state_transition_history[i + 1]["to_state"])
            sequences[seq] = sequences.get(seq, 0) + 1
        
        if not sequences:
            return []
        
        # Return most common sequence
        most_common = max(sequences.items(), key=lambda x: x[1])
        return list(most_common[0])
    
    def adapt_behavior_preferences(self, state: BeeStatus, success: bool, 
                                 environmental_factors: Dict[str, Any]) -> None:
        """Adapt behavioral preferences based on experience"""
        state_key = state.value
        
        # Initialize preference if not exists
        if state_key not in self.behavioral_preferences:
            self.behavioral_preferences[state_key] = 0.5
        
        # Adjust preference based on success
        adjustment = 0.1 if success else -0.05
        self.behavioral_preferences[state_key] += adjustment
        
        # Keep preferences within bounds
        self.behavioral_preferences[state_key] = max(0.0, min(1.0, self.behavioral_preferences[state_key]))
        
        # Environmental adaptation
        weather = environmental_factors.get('weather', 'clear')
        temp = environmental_factors.get('temperature', 20)
        
        # Create adaptation factors for environmental conditions
        env_key = f"{weather}_{temp}"
        if env_key not in self.adaptation_factors:
            self.adaptation_factors[env_key] = 1.0
        
        # Adjust based on success in these conditions
        env_adjustment = 0.05 if success else -0.02
        self.adaptation_factors[env_key] += env_adjustment
        self.adaptation_factors[env_key] = max(0.5, min(1.5, self.adaptation_factors[env_key]))
    
    def get_contextual_preference(self, state: BeeStatus, current_conditions: Dict[str, Any]) -> float:
        """Get preference for a state considering current environmental context"""
        state_key = state.value
        base_preference = self.behavioral_preferences.get(state_key, 0.5)
        
        # Environmental adjustment
        weather = current_conditions.get('weather', 'clear')
        temp = current_conditions.get('temperature', 20)
        env_key = f"{weather}_{temp}"
        
        env_factor = self.adaptation_factors.get(env_key, 1.0)
        
        return min(1.0, max(0.0, base_preference * env_factor))


class ActivityStateMachine(BaseModel):
    """Activity state machine for NetLogo BEE-STEWARD v2 compatibility"""
    
    model_config = {"validate_assignment": True}
    
    # State configurations
    state_configs: Dict[BeeStatus, ActivityStateConfig] = Field(default_factory=dict)
    
    # State transition rules
    transition_rules: List[StateTransitionRule] = Field(default_factory=list)
    
    # State categories
    state_categories: Dict[BeeStatus, ActivityStateCategory] = Field(default_factory=dict)
    
    # Personal time trackers for individual bees
    personal_trackers: Dict[int, PersonalTimeTracker] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_default_configurations()
        self._initialize_state_categories()
        self._initialize_transition_rules()
    
    def _initialize_default_configurations(self) -> None:
        """Initialize default state configurations"""
        # Dormancy states
        self.state_configs[BeeStatus.HIBERNATING] = ActivityStateConfig(
            energy_consumption=0.1,
            duration_range=(100, 1000),
            energy_cost_multiplier=0.1
        )
        
        # Construction states
        self.state_configs[BeeStatus.NEST_CONSTRUCTION] = ActivityStateConfig(
            energy_consumption=3.0,
            duration_range=(5, 20),
            energy_cost_multiplier=1.5
        )
        
        # Foraging states
        self.state_configs[BeeStatus.SEARCHING] = ActivityStateConfig(
            energy_consumption=2.5,
            duration_range=(1, 5),
            energy_cost_multiplier=1.2,
            flight_speed=6.0,
            metabolic_rate=1.3,
            stress_level=0.4
        )
        
        self.state_configs[BeeStatus.NECTAR_FORAGING] = ActivityStateConfig(
            energy_consumption=3.0,
            duration_range=(10, 30),
            energy_cost_multiplier=1.3,
            flight_speed=4.5,
            metabolic_rate=1.5,
            stress_level=0.2
        )
        
        self.state_configs[BeeStatus.POLLEN_FORAGING] = ActivityStateConfig(
            energy_consumption=3.5,
            duration_range=(10, 30),
            energy_cost_multiplier=1.4
        )
        
        self.state_configs[BeeStatus.COLLECT_NECTAR] = ActivityStateConfig(
            energy_consumption=2.0,
            duration_range=(1, 3),
            energy_cost_multiplier=1.1
        )
        
        self.state_configs[BeeStatus.COLLECT_POLLEN] = ActivityStateConfig(
            energy_consumption=2.5,
            duration_range=(1, 3),
            energy_cost_multiplier=1.2
        )
        
        self.state_configs[BeeStatus.BRINGING_NECTAR] = ActivityStateConfig(
            energy_consumption=2.0,
            duration_range=(5, 15),
            energy_cost_multiplier=1.1
        )
        
        self.state_configs[BeeStatus.BRINGING_POLLEN] = ActivityStateConfig(
            energy_consumption=2.0,
            duration_range=(5, 15),
            energy_cost_multiplier=1.1
        )
        
        self.state_configs[BeeStatus.RETURNING_EMPTY] = ActivityStateConfig(
            energy_consumption=1.5,
            duration_range=(3, 10),
            energy_cost_multiplier=1.0
        )
        
        self.state_configs[BeeStatus.RETURNING_UNHAPPY_NECTAR] = ActivityStateConfig(
            energy_consumption=1.8,
            duration_range=(3, 10),
            energy_cost_multiplier=1.0
        )
        
        self.state_configs[BeeStatus.RETURNING_UNHAPPY_POLLEN] = ActivityStateConfig(
            energy_consumption=1.8,
            duration_range=(3, 10),
            energy_cost_multiplier=1.0
        )
        
        self.state_configs[BeeStatus.EXPERIMENTAL_FORAGING_NECTAR] = ActivityStateConfig(
            energy_consumption=4.0,
            duration_range=(20, 60),
            energy_cost_multiplier=1.6
        )
        
        self.state_configs[BeeStatus.EXPERIMENTAL_FORAGING_POLLEN] = ActivityStateConfig(
            energy_consumption=4.0,
            duration_range=(20, 60),
            energy_cost_multiplier=1.6
        )
        
        # Reproductive states
        self.state_configs[BeeStatus.EGG_LAYING] = ActivityStateConfig(
            energy_consumption=2.0,
            duration_range=(1, 5),
            energy_cost_multiplier=1.2
        )
        
        # Maintenance states
        self.state_configs[BeeStatus.NURSING] = ActivityStateConfig(
            energy_consumption=1.5,
            duration_range=(5, 20),
            energy_cost_multiplier=1.1
        )
        
        # Communication states
        self.state_configs[BeeStatus.DANCING] = ActivityStateConfig(
            energy_consumption=2.0,
            duration_range=(2, 8),
            energy_cost_multiplier=1.2
        )
        
        # General states
        self.state_configs[BeeStatus.RESTING] = ActivityStateConfig(
            energy_consumption=0.5,
            duration_range=(1, 10),
            energy_cost_multiplier=0.5
        )
        
        self.state_configs[BeeStatus.FORAGING] = ActivityStateConfig(
            energy_consumption=3.0,
            duration_range=(10, 30),
            energy_cost_multiplier=1.3
        )
    
    def _initialize_state_categories(self) -> None:
        """Initialize state categories"""
        self.state_categories.update({
            BeeStatus.HIBERNATING: ActivityStateCategory.DORMANT,
            BeeStatus.NEST_CONSTRUCTION: ActivityStateCategory.CONSTRUCTION,
            BeeStatus.SEARCHING: ActivityStateCategory.FORAGING,
            BeeStatus.NECTAR_FORAGING: ActivityStateCategory.FORAGING,
            BeeStatus.POLLEN_FORAGING: ActivityStateCategory.FORAGING,
            BeeStatus.COLLECT_NECTAR: ActivityStateCategory.FORAGING,
            BeeStatus.COLLECT_POLLEN: ActivityStateCategory.FORAGING,
            BeeStatus.BRINGING_NECTAR: ActivityStateCategory.FORAGING,
            BeeStatus.BRINGING_POLLEN: ActivityStateCategory.FORAGING,
            BeeStatus.RETURNING_EMPTY: ActivityStateCategory.FORAGING,
            BeeStatus.RETURNING_UNHAPPY_NECTAR: ActivityStateCategory.FORAGING,
            BeeStatus.RETURNING_UNHAPPY_POLLEN: ActivityStateCategory.FORAGING,
            BeeStatus.EXPERIMENTAL_FORAGING_NECTAR: ActivityStateCategory.FORAGING,
            BeeStatus.EXPERIMENTAL_FORAGING_POLLEN: ActivityStateCategory.FORAGING,
            BeeStatus.EGG_LAYING: ActivityStateCategory.REPRODUCTION,
            BeeStatus.NURSING: ActivityStateCategory.MAINTENANCE,
            BeeStatus.DANCING: ActivityStateCategory.COMMUNICATION,
            BeeStatus.RESTING: ActivityStateCategory.GENERAL,
            BeeStatus.FORAGING: ActivityStateCategory.FORAGING,
        })
    
    def _initialize_transition_rules(self) -> None:
        """Initialize state transition rules"""
        # Foraging state transitions
        self.transition_rules.extend([
            # Searching -> Foraging
            StateTransitionRule(
                from_state=BeeStatus.SEARCHING,
                to_state=BeeStatus.NECTAR_FORAGING,
                probability=0.6,
                conditions={"patch_found": True, "resource_type": "nectar"}
            ),
            StateTransitionRule(
                from_state=BeeStatus.SEARCHING,
                to_state=BeeStatus.POLLEN_FORAGING,
                probability=0.4,
                conditions={"patch_found": True, "resource_type": "pollen"}
            ),
            StateTransitionRule(
                from_state=BeeStatus.SEARCHING,
                to_state=BeeStatus.RETURNING_EMPTY,
                probability=0.3,
                conditions={"patch_found": False}
            ),
            
            # Foraging -> Collection
            StateTransitionRule(
                from_state=BeeStatus.NECTAR_FORAGING,
                to_state=BeeStatus.COLLECT_NECTAR,
                probability=0.8,
                conditions={"at_patch": True, "resources_available": True}
            ),
            StateTransitionRule(
                from_state=BeeStatus.POLLEN_FORAGING,
                to_state=BeeStatus.COLLECT_POLLEN,
                probability=0.8,
                conditions={"at_patch": True, "resources_available": True}
            ),
            
            # Collection -> Bringing
            StateTransitionRule(
                from_state=BeeStatus.COLLECT_NECTAR,
                to_state=BeeStatus.BRINGING_NECTAR,
                probability=0.9,
                conditions={"collection_successful": True}
            ),
            StateTransitionRule(
                from_state=BeeStatus.COLLECT_POLLEN,
                to_state=BeeStatus.BRINGING_POLLEN,
                probability=0.9,
                conditions={"collection_successful": True}
            ),
            
            # Bringing -> Resting/Dancing
            StateTransitionRule(
                from_state=BeeStatus.BRINGING_NECTAR,
                to_state=BeeStatus.DANCING,
                probability=0.7,
                conditions={"at_hive": True, "resource_quality": "high"}
            ),
            StateTransitionRule(
                from_state=BeeStatus.BRINGING_POLLEN,
                to_state=BeeStatus.DANCING,
                probability=0.7,
                conditions={"at_hive": True, "resource_quality": "high"}
            ),
            StateTransitionRule(
                from_state=BeeStatus.BRINGING_NECTAR,
                to_state=BeeStatus.RESTING,
                probability=0.3,
                conditions={"at_hive": True}
            ),
            StateTransitionRule(
                from_state=BeeStatus.BRINGING_POLLEN,
                to_state=BeeStatus.RESTING,
                probability=0.3,
                conditions={"at_hive": True}
            ),
            
            # Dancing -> Resting/Foraging
            StateTransitionRule(
                from_state=BeeStatus.DANCING,
                to_state=BeeStatus.RESTING,
                probability=0.6,
                min_duration=2
            ),
            StateTransitionRule(
                from_state=BeeStatus.DANCING,
                to_state=BeeStatus.SEARCHING,
                probability=0.4,
                min_duration=2
            ),
            
            # Resting transitions
            StateTransitionRule(
                from_state=BeeStatus.RESTING,
                to_state=BeeStatus.SEARCHING,
                probability=0.3,
                conditions={"role": "forager", "energy": "sufficient"}
            ),
            StateTransitionRule(
                from_state=BeeStatus.RESTING,
                to_state=BeeStatus.NURSING,
                probability=0.5,
                conditions={"role": "nurse", "brood_present": True}
            ),
            StateTransitionRule(
                from_state=BeeStatus.RESTING,
                to_state=BeeStatus.NEST_CONSTRUCTION,
                probability=0.2,
                conditions={"role": "builder", "construction_needed": True}
            ),
        ])
    
    def get_personal_tracker(self, bee_id: int) -> PersonalTimeTracker:
        """Get or create personal time tracker for bee"""
        if bee_id not in self.personal_trackers:
            self.personal_trackers[bee_id] = PersonalTimeTracker()
        return self.personal_trackers[bee_id]
    
    def update_bee_activity(self, bee_id: int, current_step: int) -> None:
        """Update bee's activity tracking"""
        tracker = self.get_personal_tracker(bee_id)
        tracker.update_time(current_step)
    
    def transition_bee_state(self, bee_id: int, new_state: BeeStatus, current_step: int) -> None:
        """Transition bee to new state"""
        tracker = self.get_personal_tracker(bee_id)
        tracker.transition_to_state(new_state, current_step)
    
    # Energy consumption method defined below with enhanced logic
    
    def get_state_duration_range(self, state: BeeStatus) -> Tuple[int, int]:
        """Get duration range for a state"""
        if state in self.state_configs:
            return self.state_configs[state].duration_range
        return (1, 10)  # Default range
    
    def should_transition_state(self, bee_id: int, current_conditions: Dict[str, Any]) -> Optional[BeeStatus]:
        """Check if bee should transition to new state with dynamic context-awareness"""
        tracker = self.get_personal_tracker(bee_id)
        current_state = tracker.current_state
        
        # Enhanced dynamic transition logic
        
        # 1. Check for forced transitions due to critical conditions
        forced_transition = self._check_forced_transitions(current_state, current_conditions, tracker)
        if forced_transition:
            return forced_transition
        
        # 2. Check for environmental transitions
        environmental_transition = self._check_environmental_transitions(current_state, current_conditions, tracker)
        if environmental_transition:
            return environmental_transition
        
        # 3. Check for energy-based transitions
        energy_transition = self._check_energy_based_transitions(current_state, current_conditions, tracker)
        if energy_transition:
            return energy_transition
        
        # 4. Check for role-specific transitions
        role_transition = self._check_role_specific_transitions(current_state, current_conditions, tracker)
        if role_transition:
            return role_transition
        
        # 5. Check for time-based transitions
        time_transition = self._check_time_based_transitions(current_state, current_conditions, tracker)
        if time_transition:
            return time_transition
        
        # 6. Check standard rule-based transitions
        rule_transition = self._check_rule_based_transitions(current_state, current_conditions, tracker)
        if rule_transition:
            return rule_transition
        
        # 7. Check for adaptive transitions based on success history
        adaptive_transition = self._check_adaptive_transitions(current_state, current_conditions, tracker)
        if adaptive_transition:
            return adaptive_transition
        
        return None
    
    # ============================================
    # DYNAMIC TRANSITION LOGIC METHODS
    # ============================================
    
    def _check_forced_transitions(self, current_state: BeeStatus, conditions: Dict[str, Any], 
                                tracker: PersonalTimeTracker) -> Optional[BeeStatus]:
        """Check for forced transitions due to critical conditions"""
        
        # Critical energy states
        energy_level = conditions.get('energy', 'sufficient')
        if energy_level == 'critical' and current_state != BeeStatus.RESTING:
            return BeeStatus.RESTING
        
        # Death conditions
        if conditions.get('health_status') == 'dying':
            return BeeStatus.DEAD
        
        # Maximum duration exceeded for any state
        if current_state in self.state_configs:
            duration_range = self.state_configs[current_state].duration_range
            max_duration = duration_range[1] * 2  # Allow 2x normal duration before forced transition
            if tracker.state_duration > max_duration:
                # Force transition based on state category
                category = self.get_state_category(current_state)
                if category == ActivityStateCategory.FORAGING:
                    return BeeStatus.RETURNING_EMPTY
                elif category == ActivityStateCategory.COMMUNICATION:
                    return BeeStatus.RESTING
                else:
                    return BeeStatus.RESTING
        
        # Stuck in incompatible states for role
        role = conditions.get('role', 'worker')
        if role == 'queen' and current_state in [BeeStatus.NECTAR_FORAGING, BeeStatus.POLLEN_FORAGING]:
            return BeeStatus.EGG_LAYING
        elif role == 'drone' and current_state in [BeeStatus.NURSING, BeeStatus.NEST_CONSTRUCTION]:
            return BeeStatus.RESTING
        
        return None
    
    def _check_environmental_transitions(self, current_state: BeeStatus, conditions: Dict[str, Any], 
                                       tracker: PersonalTimeTracker) -> Optional[BeeStatus]:
        """Check for environment-driven transitions"""
        
        # Weather-based transitions
        weather = conditions.get('weather', 'clear')
        if weather in ['storm', 'heavy_rain'] and current_state in self.get_foraging_states():
            return BeeStatus.RETURNING_EMPTY
        
        # Temperature-based transitions
        temperature = conditions.get('temperature', 20)
        if isinstance(temperature, (int, float)):
            if temperature < 5 and current_state != BeeStatus.HIBERNATING:
                return BeeStatus.HIBERNATING
            elif temperature > 5 and current_state == BeeStatus.HIBERNATING:
                return BeeStatus.RESTING
        elif temperature == 'cold' and current_state != BeeStatus.HIBERNATING:
            return BeeStatus.HIBERNATING
        elif temperature == 'warm' and current_state == BeeStatus.HIBERNATING:
            return BeeStatus.RESTING
        
        # Daylight-based transitions
        time_of_day = conditions.get('time_of_day', 12)
        if isinstance(time_of_day, int):
            if (time_of_day < 6 or time_of_day > 20) and current_state in self.get_foraging_states():
                # Night time - return to hive
                if current_state in [BeeStatus.NECTAR_FORAGING, BeeStatus.POLLEN_FORAGING, BeeStatus.SEARCHING]:
                    return BeeStatus.RETURNING_EMPTY
        
        # Seasonal transitions
        season = conditions.get('season', 'spring')
        if season == 'winter' and current_state in self.get_foraging_states():
            return BeeStatus.HIBERNATING
        
        return None
    
    def _check_energy_based_transitions(self, current_state: BeeStatus, conditions: Dict[str, Any], 
                                      tracker: PersonalTimeTracker) -> Optional[BeeStatus]:
        """Check for energy-level driven transitions"""
        
        energy_level = conditions.get('energy', 'sufficient')
        energy_value = conditions.get('energy_value', 50)
        
        # Low energy transitions
        if energy_level == 'low' or (isinstance(energy_value, (int, float)) and energy_value < 30):
            if current_state in self.get_foraging_states():
                if current_state in [BeeStatus.COLLECT_NECTAR, BeeStatus.COLLECT_POLLEN]:
                    return BeeStatus.RETURNING_EMPTY
                elif current_state in [BeeStatus.NECTAR_FORAGING, BeeStatus.POLLEN_FORAGING, BeeStatus.SEARCHING]:
                    return BeeStatus.RETURNING_EMPTY
            elif current_state in [BeeStatus.DANCING, BeeStatus.NURSING, BeeStatus.NEST_CONSTRUCTION]:
                return BeeStatus.RESTING
        
        # Very low energy - emergency rest
        if energy_level == 'critical' or (isinstance(energy_value, (int, float)) and energy_value < 10):
            if current_state != BeeStatus.RESTING:
                return BeeStatus.RESTING
        
        # High energy transitions
        if energy_level == 'high' or (isinstance(energy_value, (int, float)) and energy_value > 80):
            if current_state == BeeStatus.RESTING:
                # Choose activity based on role
                role = conditions.get('role', 'forager')
                if role in ['forager', 'worker']:
                    return BeeStatus.SEARCHING
                elif role == 'nurse' and conditions.get('brood_present', False):
                    return BeeStatus.NURSING
                elif role == 'queen':
                    return BeeStatus.EGG_LAYING
                elif role == 'builder' and conditions.get('construction_needed', False):
                    return BeeStatus.NEST_CONSTRUCTION
        
        return None
    
    def _check_role_specific_transitions(self, current_state: BeeStatus, conditions: Dict[str, Any], 
                                       tracker: PersonalTimeTracker) -> Optional[BeeStatus]:
        """Check for role-specific behavioral transitions"""
        
        role = conditions.get('role', 'worker')
        age = conditions.get('age', 10)
        
        # Queen-specific transitions
        if role == 'queen':
            if current_state == BeeStatus.RESTING and conditions.get('colony_needs_eggs', True):
                return BeeStatus.EGG_LAYING
            elif current_state == BeeStatus.EGG_LAYING and tracker.state_duration > 5:
                # Take breaks from egg laying
                import random
                if random.random() < 0.3:
                    return BeeStatus.RESTING
        
        # Nurse-specific transitions
        elif role == 'nurse':
            if current_state == BeeStatus.RESTING and conditions.get('brood_present', False):
                if conditions.get('brood_needs_care', False):
                    return BeeStatus.NURSING
            elif current_state == BeeStatus.NURSING and not conditions.get('brood_needs_care', True):
                # No more brood to care for
                if age > 15 and conditions.get('energy', 'sufficient') in ['sufficient', 'high']:
                    # Mature nurse can become forager
                    return BeeStatus.SEARCHING
                else:
                    return BeeStatus.RESTING
        
        # Forager-specific transitions
        elif role == 'forager':
            if current_state == BeeStatus.RESTING and conditions.get('energy', 'sufficient') in ['sufficient', 'high']:
                if conditions.get('patches_available', True):
                    return BeeStatus.SEARCHING
            elif current_state == BeeStatus.DANCING and tracker.state_duration > 8:
                # End dancing after reasonable time
                if conditions.get('energy', 'sufficient') in ['sufficient', 'high']:
                    return BeeStatus.SEARCHING  # Go foraging again
                else:
                    return BeeStatus.RESTING
        
        # Builder-specific transitions
        elif role == 'builder':
            if current_state == BeeStatus.RESTING and conditions.get('construction_needed', False):
                return BeeStatus.NEST_CONSTRUCTION
            elif current_state == BeeStatus.NEST_CONSTRUCTION and not conditions.get('construction_needed', True):
                return BeeStatus.RESTING
        
        # Age-based role transitions
        if age > 20 and role == 'nurse' and current_state == BeeStatus.RESTING:
            # Older nurses become foragers
            return BeeStatus.SEARCHING
        
        return None
    
    def _check_time_based_transitions(self, current_state: BeeStatus, conditions: Dict[str, Any], 
                                    tracker: PersonalTimeTracker) -> Optional[BeeStatus]:
        """Check for time-duration based transitions"""
        
        # Check state-specific duration limits
        if current_state in self.state_configs:
            config = self.state_configs[current_state]
            min_duration, max_duration = config.duration_range
            
            # Minimum duration check
            if tracker.state_duration < min_duration:
                return None  # Stay in current state
            
            # Probabilistic transition based on duration
            if tracker.state_duration >= max_duration:
                # Force transition after maximum duration
                return self._get_natural_transition_target(current_state, conditions)
            elif tracker.state_duration > min_duration:
                # Increasing probability of transition as duration increases
                excess_duration = tracker.state_duration - min_duration
                duration_range = max_duration - min_duration
                transition_probability = min(0.8, excess_duration / duration_range)
                
                import random
                if random.random() < transition_probability:
                    return self._get_natural_transition_target(current_state, conditions)
        
        # Circadian rhythm effects
        time_of_day = conditions.get('time_of_day', 12)
        if isinstance(time_of_day, int):
            # Morning activity increase
            if 6 <= time_of_day <= 8 and current_state == BeeStatus.RESTING:
                role = conditions.get('role', 'forager')
                if role in ['forager', 'worker'] and conditions.get('energy', 'sufficient') in ['sufficient', 'high']:
                    return BeeStatus.SEARCHING
            
            # Evening wind-down
            elif 18 <= time_of_day <= 20 and current_state in self.get_foraging_states():
                if current_state not in [BeeStatus.BRINGING_NECTAR, BeeStatus.BRINGING_POLLEN]:
                    return BeeStatus.RETURNING_EMPTY
        
        return None
    
    def _check_rule_based_transitions(self, current_state: BeeStatus, conditions: Dict[str, Any], 
                                    tracker: PersonalTimeTracker) -> Optional[BeeStatus]:
        """Check standard rule-based transitions (original logic)"""
        
        # Check applicable transition rules
        applicable_rules = [
            rule for rule in self.transition_rules 
            if rule.from_state == current_state
        ]
        
        for rule in applicable_rules:
            # Check minimum duration
            if tracker.state_duration < rule.min_duration:
                continue
            
            # Check maximum duration
            if rule.max_duration is not None and tracker.state_duration > rule.max_duration:
                return rule.to_state  # Force transition
            
            # Check conditions
            if self._check_transition_conditions(rule.conditions, conditions):
                # Apply context-aware probability adjustment
                adjusted_probability = self._adjust_transition_probability(rule.probability, conditions, tracker)
                
                import random
                if random.random() < adjusted_probability:
                    return rule.to_state
        
        return None
    
    def _check_adaptive_transitions(self, current_state: BeeStatus, conditions: Dict[str, Any], 
                                  tracker: PersonalTimeTracker) -> Optional[BeeStatus]:
        """Check for adaptive transitions based on learning and success history"""
        
        # Success-based adaptations
        success_rate = conditions.get('recent_success_rate', 0.5)
        conditions.get('foraging_efficiency', 1.0)
        
        # Poor success rate adaptations
        if success_rate < 0.3:
            if current_state == BeeStatus.SEARCHING and tracker.state_duration > 3:
                # Try experimental foraging if normal searching fails
                resource_preference = conditions.get('resource_preference', 'nectar')
                if resource_preference == 'nectar':
                    return BeeStatus.EXPERIMENTAL_FORAGING_NECTAR
                else:
                    return BeeStatus.EXPERIMENTAL_FORAGING_POLLEN
            elif current_state in [BeeStatus.COLLECT_NECTAR, BeeStatus.COLLECT_POLLEN] and tracker.state_duration > 2:
                # Give up on poor patches faster
                return BeeStatus.SEARCHING
        
        # High success rate adaptations
        elif success_rate > 0.8:
            if current_state == BeeStatus.RESTING and conditions.get('energy', 'sufficient') in ['sufficient', 'high']:
                # Quick return to successful foraging
                return BeeStatus.SEARCHING
        
        # Social learning adaptations
        dance_followers = conditions.get('recent_dance_followers', 0)
        if dance_followers > 3 and current_state == BeeStatus.DANCING and tracker.state_duration > 3:
            # Successful dancers continue dancing longer
            if tracker.state_duration < 10:  # But not too long
                return None  # Stay dancing
            else:
                return BeeStatus.RESTING
        
        # Colony need-based adaptations
        colony_energy_level = conditions.get('colony_energy_level', 'sufficient')
        if colony_energy_level == 'low':
            if current_state == BeeStatus.RESTING and conditions.get('energy', 'sufficient') in ['sufficient', 'high']:
                # Colony needs resources, prioritize foraging
                return BeeStatus.SEARCHING
            elif current_state == BeeStatus.DANCING and tracker.state_duration > 2:
                # Shorter dances when colony needs resources urgently
                return BeeStatus.SEARCHING
        
        return None
    
    def _get_natural_transition_target(self, current_state: BeeStatus, conditions: Dict[str, Any]) -> BeeStatus:
        """Get natural transition target for a state"""
        
        # State-specific natural transitions
        if current_state == BeeStatus.SEARCHING:
            patch_found = conditions.get('patch_found', False)
            if patch_found:
                resource_type = conditions.get('resource_type', 'nectar')
                return BeeStatus.NECTAR_FORAGING if resource_type == 'nectar' else BeeStatus.POLLEN_FORAGING
            else:
                return BeeStatus.RETURNING_EMPTY
        
        elif current_state in [BeeStatus.NECTAR_FORAGING, BeeStatus.POLLEN_FORAGING]:
            at_patch = conditions.get('at_patch', False)
            if at_patch:
                return BeeStatus.COLLECT_NECTAR if current_state == BeeStatus.NECTAR_FORAGING else BeeStatus.COLLECT_POLLEN
            else:
                return BeeStatus.SEARCHING
        
        elif current_state in [BeeStatus.COLLECT_NECTAR, BeeStatus.COLLECT_POLLEN]:
            collection_successful = conditions.get('collection_successful', False)
            if collection_successful:
                resource_quality = conditions.get('resource_quality', 'medium')
                if resource_quality == 'high':
                    return BeeStatus.BRINGING_NECTAR if current_state == BeeStatus.COLLECT_NECTAR else BeeStatus.BRINGING_POLLEN
                else:
                    return BeeStatus.RETURNING_UNHAPPY_NECTAR if current_state == BeeStatus.COLLECT_NECTAR else BeeStatus.RETURNING_UNHAPPY_POLLEN
            else:
                return BeeStatus.SEARCHING
        
        elif current_state in [BeeStatus.BRINGING_NECTAR, BeeStatus.BRINGING_POLLEN, 
                              BeeStatus.RETURNING_EMPTY, BeeStatus.RETURNING_UNHAPPY_NECTAR, 
                              BeeStatus.RETURNING_UNHAPPY_POLLEN]:
            at_hive = conditions.get('at_hive', False)
            if at_hive:
                if current_state in [BeeStatus.BRINGING_NECTAR, BeeStatus.BRINGING_POLLEN]:
                    return BeeStatus.DANCING
                else:
                    return BeeStatus.RESTING
            else:
                return current_state  # Continue returning
        
        elif current_state == BeeStatus.DANCING:
            return BeeStatus.RESTING
        
        elif current_state == BeeStatus.NURSING:
            brood_present = conditions.get('brood_present', True)
            if brood_present:
                return BeeStatus.NURSING  # Continue nursing
            else:
                return BeeStatus.RESTING
        
        elif current_state == BeeStatus.EGG_LAYING:
            return BeeStatus.RESTING
        
        elif current_state == BeeStatus.NEST_CONSTRUCTION:
            construction_needed = conditions.get('construction_needed', True)
            if construction_needed:
                return BeeStatus.NEST_CONSTRUCTION  # Continue construction
            else:
                return BeeStatus.RESTING
        
        elif current_state == BeeStatus.RESTING:
            # Choose next activity based on role and conditions
            role = conditions.get('role', 'forager')
            energy = conditions.get('energy', 'sufficient')
            
            if energy in ['sufficient', 'high']:
                if role == 'queen':
                    return BeeStatus.EGG_LAYING
                elif role == 'nurse' and conditions.get('brood_present', False):
                    return BeeStatus.NURSING
                elif role in ['forager', 'worker']:
                    return BeeStatus.SEARCHING
                elif role == 'builder' and conditions.get('construction_needed', False):
                    return BeeStatus.NEST_CONSTRUCTION
            
            return BeeStatus.RESTING
        
        # Default transition
        return BeeStatus.RESTING
    
    def _adjust_transition_probability(self, base_probability: float, conditions: Dict[str, Any], 
                                     tracker: PersonalTimeTracker) -> float:
        """Adjust transition probability based on context"""
        
        adjusted_probability = base_probability
        
        # Energy level adjustments
        energy_level = conditions.get('energy', 'sufficient')
        if energy_level == 'high':
            adjusted_probability *= 1.2  # More likely to be active
        elif energy_level == 'low':
            adjusted_probability *= 0.7  # Less likely to transition to energy-intensive states
        elif energy_level == 'critical':
            adjusted_probability *= 0.3  # Much less likely to be active
        
        # Time of day adjustments
        time_of_day = conditions.get('time_of_day', 12)
        if isinstance(time_of_day, int):
            if 6 <= time_of_day <= 18:  # Daytime
                adjusted_probability *= 1.1
            else:  # Night time
                adjusted_probability *= 0.6
        
        # Weather adjustments
        weather = conditions.get('weather', 'clear')
        if weather == 'clear':
            adjusted_probability *= 1.0
        elif weather in ['cloudy', 'light_rain']:
            adjusted_probability *= 0.8
        elif weather in ['heavy_rain', 'storm']:
            adjusted_probability *= 0.3
        
        # Colony need adjustments
        colony_energy_level = conditions.get('colony_energy_level', 'sufficient')
        if colony_energy_level == 'low':
            adjusted_probability *= 1.3  # More urgent transitions
        elif colony_energy_level == 'high':
            adjusted_probability *= 0.9  # Less urgent
        
        # Success rate adjustments
        success_rate = conditions.get('recent_success_rate', 0.5)
        if success_rate > 0.7:
            adjusted_probability *= 1.1  # Successful bees are more active
        elif success_rate < 0.3:
            adjusted_probability *= 0.8  # Unsuccessful bees are more cautious
        
        # Ensure probability stays within bounds
        return max(0.0, min(1.0, adjusted_probability))
    
    def _check_transition_conditions(self, required_conditions: Dict[str, Any], 
                                   current_conditions: Dict[str, Any]) -> bool:
        """Check if transition conditions are met"""
        for key, required_value in required_conditions.items():
            if key not in current_conditions:
                return False
            
            current_value = current_conditions[key]
            
            # Handle different condition types
            if isinstance(required_value, bool):
                if current_value != required_value:
                    return False
            elif isinstance(required_value, str):
                if current_value != required_value:
                    return False
            elif isinstance(required_value, (int, float)):
                if current_value < required_value:
                    return False
        
        return True
    
    def get_activity_summary(self, bee_id: int) -> Dict[str, Any]:
        """Get activity summary for bee"""
        if bee_id not in self.personal_trackers:
            return {}
        
        tracker = self.personal_trackers[bee_id]
        return {
            "current_state": tracker.current_state.value,
            "state_duration": tracker.state_duration,
            "total_foraging_time": tracker.total_foraging_time,
            "total_nursing_time": tracker.total_nursing_time,
            "daily_activities": tracker.daily_activities.copy()
        }
    
    def validate_personal_tracker(self, bee_id: int) -> List[str]:
        """Validate personal time tracker data"""
        errors = []
        
        if bee_id not in self.personal_trackers:
            errors.append(f"No personal tracker found for bee {bee_id}")
            return errors
        
        tracker = self.personal_trackers[bee_id]
        
        # Validate state duration consistency
        if tracker.state_duration < 0:
            errors.append(f"Negative state duration: {tracker.state_duration}")
        
        # Validate time tracking counters
        if tracker.total_foraging_time < 0:
            errors.append(f"Negative foraging time: {tracker.total_foraging_time}")
        
        if tracker.total_nursing_time < 0:
            errors.append(f"Negative nursing time: {tracker.total_nursing_time}")
        
        # Validate daily activities
        for activity, count in tracker.daily_activities.items():
            if count < 0:
                errors.append(f"Negative activity count for {activity}: {count}")
        
        # Validate time sequence
        if tracker.state_start_time > tracker.last_activity_change:
            errors.append(f"Invalid time sequence: start {tracker.state_start_time} > last change {tracker.last_activity_change}")
        
        return errors
    
    def get_state_efficiency_metrics(self, bee_id: int) -> Dict[str, float]:
        """Get state efficiency metrics for bee"""
        if bee_id not in self.personal_trackers:
            return {}
        
        tracker = self.personal_trackers[bee_id]
        total_time = sum(tracker.daily_activities.values())
        
        if total_time == 0:
            return {}
        
        metrics = {}
        
        # Calculate time allocation percentages
        for activity, time_spent in tracker.daily_activities.items():
            percentage = (time_spent / total_time) * 100
            metrics[f"{activity}_percentage"] = percentage
        
        # Calculate efficiency ratios
        if tracker.total_foraging_time > 0:
            metrics["foraging_efficiency"] = tracker.total_foraging_time / total_time
        
        if tracker.total_nursing_time > 0:
            metrics["nursing_efficiency"] = tracker.total_nursing_time / total_time
        
        return metrics
    
    def get_state_category(self, state: BeeStatus) -> ActivityStateCategory:
        """Get category for a state"""
        return self.state_categories.get(state, ActivityStateCategory.GENERAL)
    
    def get_state_energy_consumption(self, state: BeeStatus) -> float:
        """Get energy consumption for a state"""
        if state not in self.state_configs:
            return 1.0
        
        config = self.state_configs[state]
        return config.energy_consumption * config.energy_cost_multiplier
    
    def get_foraging_states(self) -> List[BeeStatus]:
        """Get all foraging-related states"""
        return [
            state for state, category in self.state_categories.items()
            if category == ActivityStateCategory.FORAGING
        ]
    
    def get_maintenance_states(self) -> List[BeeStatus]:
        """Get all maintenance-related states"""
        return [
            state for state, category in self.state_categories.items()
            if category == ActivityStateCategory.MAINTENANCE
        ]
    
    def remove_bee_tracker(self, bee_id: int) -> None:
        """Remove personal tracker when bee dies"""
        if bee_id in self.personal_trackers:
            del self.personal_trackers[bee_id]
    
    def get_state_physiological_effects(self, state: BeeStatus) -> Dict[str, float]:
        """Get physiological effects for a state"""
        if state not in self.state_configs:
            return {}
        
        config = self.state_configs[state]
        return {
            "flight_speed": config.flight_speed,
            "metabolic_rate": config.metabolic_rate,
            "stress_level": config.stress_level,
            "thermoregulation_cost": config.thermoregulation_cost
        }
    
    def calculate_state_specific_energy_cost(self, state: BeeStatus, base_energy: float, 
                                           environmental_factors: Dict[str, float]) -> float:
        """Calculate energy cost with state-specific physiological adjustments"""
        if state not in self.state_configs:
            return base_energy
        
        config = self.state_configs[state]
        
        # Base energy cost
        adjusted_energy = base_energy * config.energy_cost_multiplier
        
        # Metabolic rate adjustment
        adjusted_energy *= config.metabolic_rate
        
        # Environmental stress adjustment
        if "temperature" in environmental_factors:
            temp = environmental_factors["temperature"]
            if temp < 10 or temp > 35:  # Extreme temperatures
                adjusted_energy += config.thermoregulation_cost
        
        # Stress level adjustment
        stress_multiplier = 1.0 + (config.stress_level * 0.5)
        adjusted_energy *= stress_multiplier
        
        return adjusted_energy
    
    def get_state_behavioral_parameters(self, state: BeeStatus) -> Dict[str, Any]:
        """Get behavioral parameters for state"""
        if state not in self.state_configs:
            return {}
        
        config = self.state_configs[state]
        category = self.get_state_category(state)
        
        parameters = {
            "category": category.value,
            "success_probability": config.success_probability,
            "duration_range": config.duration_range,
            "energy_efficiency": 1.0 / config.energy_cost_multiplier if config.energy_cost_multiplier > 0 else 1.0
        }
        
        # Add category-specific parameters
        if category == ActivityStateCategory.FORAGING:
            parameters.update({
                "flight_speed": config.flight_speed,
                "search_efficiency": 1.0 - config.stress_level,
                "resource_detection_range": config.flight_speed * 2.0
            })
        elif category == ActivityStateCategory.MAINTENANCE:
            parameters.update({
                "care_efficiency": 1.0 + config.metabolic_rate * 0.2,
                "attention_span": config.duration_range[1] - config.duration_range[0]
            })
        elif category == ActivityStateCategory.COMMUNICATION:
            parameters.update({
                "dance_vigor": config.metabolic_rate,
                "information_accuracy": 1.0 - config.stress_level
            })
        
        return parameters
    
    # ============================================
    # ENHANCED DYNAMIC STATE TRANSITION METHODS
    # ============================================
    
    def predict_next_state(self, bee_id: int, current_conditions: Dict[str, Any], 
                          lookahead_steps: int = 5) -> List[Tuple[BeeStatus, float]]:
        """Predict likely future states for a bee"""
        
        tracker = self.get_personal_tracker(bee_id)
        current_state = tracker.current_state
        
        predictions = []
        
        # Simulate future steps
        simulated_conditions = current_conditions.copy()
        simulated_tracker = PersonalTimeTracker(
            current_state=current_state,
            state_start_time=tracker.state_start_time,
            state_duration=tracker.state_duration
        )
        
        for step in range(lookahead_steps):
            # Check what transition would happen
            next_state = self.should_transition_state(bee_id, simulated_conditions)
            
            if next_state:
                # Calculate confidence based on transition probability
                confidence = self._calculate_transition_confidence(
                    simulated_tracker.current_state, next_state, simulated_conditions
                )
                predictions.append((next_state, confidence))
                
                # Update simulated tracker
                simulated_tracker.transition_to_state(next_state, step)
                
                # Update conditions for next prediction
                simulated_conditions = self._update_predicted_conditions(
                    simulated_conditions, next_state, step
                )
            else:
                # Stay in current state
                predictions.append((simulated_tracker.current_state, 0.8))
                simulated_tracker.update_time(step)
        
        return predictions
    
    def get_state_transition_matrix(self) -> Dict[BeeStatus, Dict[BeeStatus, float]]:
        """Get transition probability matrix between states"""
        
        matrix = {}
        
        for from_state in BeeStatus:
            matrix[from_state] = {}
            
            # Initialize all transitions to 0
            for to_state in BeeStatus:
                matrix[from_state][to_state] = 0.0
            
            # Fill in rule-based probabilities
            for rule in self.transition_rules:
                if rule.from_state == from_state:
                    matrix[from_state][rule.to_state] = rule.probability
        
        return matrix
    
    def analyze_state_patterns(self, bee_id: int) -> Dict[str, Any]:
        """Analyze behavioral patterns for a bee"""
        
        if bee_id not in self.personal_trackers:
            return {}
        
        tracker = self.personal_trackers[bee_id]
        
        analysis = {
            "primary_activities": [],
            "activity_preferences": {},
            "efficiency_metrics": {},
            "behavioral_consistency": 0.0,
            "role_adherence": 0.0
        }
        
        # Calculate activity preferences
        total_time = sum(tracker.daily_activities.values())
        if total_time > 0:
            for activity, time_spent in tracker.daily_activities.items():
                preference = time_spent / total_time
                analysis["activity_preferences"][activity] = preference
                
                # Identify primary activities (>20% time allocation)
                if preference > 0.2:
                    analysis["primary_activities"].append(activity)
        
        # Calculate efficiency metrics
        analysis["efficiency_metrics"] = self.get_state_efficiency_metrics(bee_id)
        
        # Calculate behavioral consistency (how predictable the bee's patterns are)
        analysis["behavioral_consistency"] = self._calculate_behavioral_consistency(tracker)
        
        return analysis
    
    def optimize_state_transitions(self, environmental_factors: Dict[str, Any]) -> None:
        """Optimize state transition rules based on environmental factors"""
        
        # Adjust transition probabilities based on environmental conditions
        weather = environmental_factors.get('weather', 'clear')
        temperature = environmental_factors.get('temperature', 20)
        season = environmental_factors.get('season', 'spring')
        
        # Weather adjustments
        if weather in ['storm', 'heavy_rain']:
            # Reduce foraging transition probabilities
            for rule in self.transition_rules:
                if rule.to_state in self.get_foraging_states():
                    rule.probability *= 0.3
        elif weather == 'clear':
            # Restore normal probabilities
            self._initialize_transition_rules()  # Reset to defaults
        
        # Temperature adjustments
        if isinstance(temperature, (int, float)):
            if temperature < 10:
                # Cold weather - promote hibernation
                for rule in self.transition_rules:
                    if rule.to_state == BeeStatus.HIBERNATING:
                        rule.probability *= 2.0
            elif temperature > 30:
                # Hot weather - reduce activity
                for rule in self.transition_rules:
                    if rule.to_state in [BeeStatus.NECTAR_FORAGING, BeeStatus.POLLEN_FORAGING]:
                        rule.probability *= 0.7
        
        # Seasonal adjustments
        if season == 'winter':
            # Promote hibernation and reduce foraging
            for rule in self.transition_rules:
                if rule.to_state == BeeStatus.HIBERNATING:
                    rule.probability *= 3.0
                elif rule.to_state in self.get_foraging_states():
                    rule.probability *= 0.2
        elif season == 'spring':
            # Increase activity and foraging
            for rule in self.transition_rules:
                if rule.to_state in self.get_foraging_states():
                    rule.probability *= 1.3
                elif rule.to_state == BeeStatus.EGG_LAYING:
                    rule.probability *= 1.5  # Spring breeding
    
    def get_context_aware_recommendations(self, bee_id: int, 
                                        current_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get context-aware behavioral recommendations for a bee"""
        
        tracker = self.get_personal_tracker(bee_id)
        current_state = tracker.current_state
        
        recommendations = []
        
        # Energy-based recommendations
        energy_level = current_conditions.get('energy', 'sufficient')
        if energy_level == 'low':
            recommendations.append({
                "type": "energy_management",
                "priority": "high",
                "action": "transition_to_resting",
                "reason": "Low energy detected - rest recommended"
            })
        
        # Role-specific recommendations
        role = current_conditions.get('role', 'worker')
        age = current_conditions.get('age', 10)
        
        if role == 'nurse' and age > 20:
            recommendations.append({
                "type": "role_transition",
                "priority": "medium", 
                "action": "consider_forager_role",
                "reason": "Mature nurse bee could transition to foraging"
            })
        
        # Environmental recommendations
        weather = current_conditions.get('weather', 'clear')
        if weather in ['storm', 'heavy_rain'] and current_state in self.get_foraging_states():
            recommendations.append({
                "type": "environmental_safety",
                "priority": "high",
                "action": "return_to_hive",
                "reason": "Dangerous weather conditions detected"
            })
        
        # Efficiency recommendations
        success_rate = current_conditions.get('recent_success_rate', 0.5)
        if success_rate < 0.3 and current_state == BeeStatus.SEARCHING:
            recommendations.append({
                "type": "efficiency_improvement",
                "priority": "medium",
                "action": "try_experimental_foraging",
                "reason": "Poor success rate suggests need for new foraging areas"
            })
        
        # Time-based recommendations
        time_of_day = current_conditions.get('time_of_day', 12)
        if isinstance(time_of_day, int) and (time_of_day < 6 or time_of_day > 20):
            if current_state in self.get_foraging_states():
                recommendations.append({
                    "type": "circadian_rhythm",
                    "priority": "medium",
                    "action": "end_foraging_activities",
                    "reason": "Night time - foraging not optimal"
                })
        
        return recommendations
    
    def _calculate_transition_confidence(self, from_state: BeeStatus, to_state: BeeStatus, 
                                       conditions: Dict[str, Any]) -> float:
        """Calculate confidence in a state transition prediction"""
        
        # Find matching transition rule
        for rule in self.transition_rules:
            if rule.from_state == from_state and rule.to_state == to_state:
                base_confidence = rule.probability
                
                # Adjust based on how well conditions match
                condition_match = self._calculate_condition_match(rule.conditions, conditions)
                
                return base_confidence * condition_match
        
        # If no rule found, use heuristic confidence
        return 0.5
    
    def _calculate_condition_match(self, required_conditions: Dict[str, Any], 
                                 current_conditions: Dict[str, Any]) -> float:
        """Calculate how well current conditions match required conditions"""
        
        if not required_conditions:
            return 1.0
        
        matches = 0
        total_conditions = len(required_conditions)
        
        for key, required_value in required_conditions.items():
            if key in current_conditions:
                current_value = current_conditions[key]
                
                if isinstance(required_value, bool) and current_value == required_value:
                    matches += 1
                elif isinstance(required_value, str) and current_value == required_value:
                    matches += 1
                elif isinstance(required_value, (int, float)) and isinstance(current_value, (int, float)):
                    # For numeric values, use tolerance
                    if abs(current_value - required_value) / max(abs(required_value), 1) < 0.2:
                        matches += 1
        
        return matches / total_conditions if total_conditions > 0 else 0.0
    
    def _update_predicted_conditions(self, conditions: Dict[str, Any], 
                                   new_state: BeeStatus, step: int) -> Dict[str, Any]:
        """Update conditions for future state prediction"""
        
        updated_conditions = conditions.copy()
        
        # Update energy based on state
        if new_state in self.state_configs:
            energy_consumption = self.state_configs[new_state].energy_consumption
            current_energy = updated_conditions.get('energy_value', 50)
            if isinstance(current_energy, (int, float)):
                new_energy = max(0, current_energy - energy_consumption)
                updated_conditions['energy_value'] = new_energy
                
                # Update energy level category
                if new_energy < 10:
                    updated_conditions['energy'] = 'critical'
                elif new_energy < 30:
                    updated_conditions['energy'] = 'low'
                elif new_energy > 80:
                    updated_conditions['energy'] = 'high'
                else:
                    updated_conditions['energy'] = 'sufficient'
        
        # Update location-based conditions
        if new_state in [BeeStatus.NECTAR_FORAGING, BeeStatus.POLLEN_FORAGING]:
            updated_conditions['at_patch'] = True
            updated_conditions['at_hive'] = False
        elif new_state in [BeeStatus.BRINGING_NECTAR, BeeStatus.BRINGING_POLLEN, 
                          BeeStatus.RETURNING_EMPTY, BeeStatus.RESTING]:
            updated_conditions['at_hive'] = True
            updated_conditions['at_patch'] = False
        
        return updated_conditions
    
    def _calculate_behavioral_consistency(self, tracker: PersonalTimeTracker) -> float:
        """Calculate behavioral consistency score for a bee"""
        
        if not tracker.daily_activities:
            return 0.0
        
        # Calculate entropy of activity distribution
        total_time = sum(tracker.daily_activities.values())
        if total_time == 0:
            return 0.0
        
        entropy = 0.0
        for time_spent in tracker.daily_activities.values():
            if time_spent > 0:
                probability = time_spent / total_time
                entropy -= probability * math.log2(probability)
        
        # Normalize entropy to 0-1 scale (lower entropy = higher consistency)
        max_entropy = math.log2(len(tracker.daily_activities)) if tracker.daily_activities else 1
        consistency = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        return max(0.0, min(1.0, consistency))
    
    def get_state_transition_history(self, bee_id: int, steps: int = 10) -> List[Dict[str, Any]]:
        """Get recent state transition history for analysis"""
        
        # This would require storing transition history in the tracker
        # For now, return current state info
        if bee_id not in self.personal_trackers:
            return []
        
        tracker = self.personal_trackers[bee_id]
        
        return [{
            "step": tracker.last_activity_change,
            "state": tracker.current_state.value,
            "duration": tracker.state_duration,
            "total_foraging_time": tracker.total_foraging_time,
            "total_nursing_time": tracker.total_nursing_time
        }]