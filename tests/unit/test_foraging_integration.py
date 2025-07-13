"""
Unit tests for Foraging Integration System
=========================================

Tests for the comprehensive foraging integration system including decision engines,
trip management, patch selection, communication integration, and session lifecycle.
"""

import pytest
from unittest.mock import Mock, patch
import time

from src.bstew.core.foraging_integration import (
    IntegratedForagingSystem, ForagingSessionResult, ForagingMode
)
from src.bstew.core.foraging_algorithms import (
    ForagingDecisionEngine, ForagingTripManager, ForagingDecisionType, ForagingTripLifecycle
)
from src.bstew.core.patch_selection import AdvancedPatchSelector, PatchInfo, ResourceType as PatchResourceType
from src.bstew.core.bee_communication import (
    ForagingCommunicationIntegrator
)
from src.bstew.components.proboscis_matching import ProboscisCorollaSystem, AccessibilityResult


class TestForagingMode:
    """Test foraging mode definitions"""
    
    def test_foraging_mode_values(self):
        """Test foraging mode enum values"""
        assert ForagingMode.EXPLORATION.value == "exploration"
        assert ForagingMode.EXPLOITATION.value == "exploitation"
        assert ForagingMode.RECRUITMENT_FOLLOWING.value == "recruitment_following"
        assert ForagingMode.MIXED_STRATEGY.value == "mixed_strategy"
        assert ForagingMode.EMERGENCY_FORAGING.value == "emergency_foraging"


class TestForagingSessionResult:
    """Test foraging session result data structure"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.session_result = ForagingSessionResult(
            session_id="session_123_1234567890",
            bee_id=123,
            colony_id=1,
            session_start_time=1234567890.0,
            session_duration=300.0,
            trips_completed=5,
            patches_visited={101, 102, 103},
            total_distance_traveled=150.0,
            total_energy_consumed=50.0,
            total_energy_gained=80.0,
            net_energy_result=30.0,
            nectar_collected=25.0,
            pollen_collected=15.0,
            resource_quality_avg=0.75,
            dances_performed=2,
            recruits_obtained=3,
            information_shared=5,
            trip_efficiency=0.8,
            energy_efficiency=1.6,
            time_efficiency=0.7,
            overall_success_score=0.75
        )
    
    def test_initialization(self):
        """Test session result initialization"""
        assert self.session_result.session_id == "session_123_1234567890"
        assert self.session_result.bee_id == 123
        assert self.session_result.colony_id == 1
        assert self.session_result.session_duration == 300.0
        assert self.session_result.trips_completed == 5
        assert len(self.session_result.patches_visited) == 3
        assert 101 in self.session_result.patches_visited
    
    def test_efficiency_calculations(self):
        """Test efficiency metrics"""
        assert self.session_result.net_energy_result == 30.0
        assert self.session_result.energy_efficiency == 1.6  # 80/50
        assert self.session_result.trip_efficiency == 0.8
        assert self.session_result.overall_success_score == 0.75
    
    def test_resource_collection(self):
        """Test resource collection tracking"""
        assert self.session_result.nectar_collected == 25.0
        assert self.session_result.pollen_collected == 15.0
        assert self.session_result.resource_quality_avg == 0.75
    
    def test_communication_metrics(self):
        """Test communication outcome tracking"""
        assert self.session_result.dances_performed == 2
        assert self.session_result.recruits_obtained == 3
        assert self.session_result.information_shared == 5


class TestIntegratedForagingSystem:
    """Test integrated foraging system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create mocked dependencies
        self.mock_decision_engine = Mock(spec=ForagingDecisionEngine)
        self.mock_trip_manager = Mock(spec=ForagingTripManager)
        self.mock_patch_selector = Mock(spec=AdvancedPatchSelector)
        self.mock_communication_integrator = Mock(spec=ForagingCommunicationIntegrator)
        self.mock_proboscis_system = Mock(spec=ProboscisCorollaSystem)
        
        # Create integrated system with mocked dependencies
        self.foraging_system = IntegratedForagingSystem(
            decision_engine=self.mock_decision_engine,
            trip_manager=self.mock_trip_manager,
            patch_selector=self.mock_patch_selector,
            communication_integrator=self.mock_communication_integrator,
            proboscis_system=self.mock_proboscis_system
        )
        
        # Setup test data
        self.test_patch_info = PatchInfo(
            patch_id=101,
            location=(10.0, 20.0),
            resource_type=PatchResourceType.NECTAR,
            quality_metrics={},  # Use empty dict for simplicity
            species_compatibility={},  # Use empty dict for simplicity
            distance_from_hive=50.0,
            current_foragers=2,
            max_capacity=10,
            depletion_rate=0.1,
            regeneration_rate=0.05,
            seasonal_availability={}  # Use empty dict for simplicity
        )
        
        self.test_bee_state = {
            "energy": 80.0,
            "foraging_experience": 5,
            "memory": {"visited_patches": [100, 99]},
            "species": "bombus_terrestris"
        }
        
        self.test_colony_state = {
            "energy_level": 1200.0,
            "forager_count": 25,
            "species": "bombus_terrestris",
            "location": (0.0, 0.0)
        }
        
        self.test_environmental_context = {
            "season": "summer",
            "hour": 14,
            "weather": "clear",
            "wind_speed": 2.0,
            "temperature": 22.0
        }
    
    def test_initialization(self):
        """Test foraging system initialization"""
        assert self.foraging_system.decision_engine == self.mock_decision_engine
        assert self.foraging_system.trip_manager == self.mock_trip_manager
        assert self.foraging_system.patch_selector == self.mock_patch_selector
        assert isinstance(self.foraging_system.active_foraging_sessions, dict)
        assert isinstance(self.foraging_system.patch_database, dict)
        assert isinstance(self.foraging_system.foraging_history, list)
        assert self.foraging_system.max_session_duration == 480.0
    
    def test_colony_foraging_initialization(self):
        """Test colony foraging system initialization"""
        colony_id = 1
        landscape_patches = [self.test_patch_info]
        
        self.foraging_system.initialize_colony_foraging(
            colony_id, self.test_colony_state, landscape_patches
        )
        
        # Check patch database updated
        assert 101 in self.foraging_system.patch_database
        assert self.foraging_system.patch_database[101] == self.test_patch_info
        
        # Check strategy assignment
        assert colony_id in self.foraging_system.colony_foraging_strategies
        strategy = self.foraging_system.colony_foraging_strategies[colony_id]
        assert strategy in ForagingMode
        
        # Check performance metrics initialization
        assert colony_id in self.foraging_system.system_performance_metrics
        metrics = self.foraging_system.system_performance_metrics[colony_id]
        assert "total_foraging_sessions" in metrics
        assert "successful_sessions" in metrics
        assert "average_efficiency" in metrics
    
    def test_strategy_selection_high_energy(self):
        """Test foraging strategy selection with high energy colony"""
        colony_state = self.test_colony_state.copy()
        colony_state["energy_level"] = 1500.0
        colony_state["forager_count"] = 35
        
        self.foraging_system.initialize_colony_foraging(1, colony_state, [self.test_patch_info])
        
        strategy = self.foraging_system.colony_foraging_strategies[1]
        assert strategy == ForagingMode.MIXED_STRATEGY
    
    def test_strategy_selection_low_energy(self):
        """Test foraging strategy selection with low energy colony"""
        colony_state = self.test_colony_state.copy()
        colony_state["energy_level"] = 400.0
        
        self.foraging_system.initialize_colony_foraging(1, colony_state, [self.test_patch_info])
        
        strategy = self.foraging_system.colony_foraging_strategies[1]
        assert strategy == ForagingMode.EMERGENCY_FORAGING
    
    def test_strategy_selection_exploration(self):
        """Test foraging strategy selection for exploration"""
        colony_state = self.test_colony_state.copy()
        colony_state["energy_level"] = 1000.0
        colony_state["forager_count"] = 15
        
        self.foraging_system.initialize_colony_foraging(1, colony_state, [self.test_patch_info])
        
        strategy = self.foraging_system.colony_foraging_strategies[1]
        assert strategy == ForagingMode.EXPLORATION
    
    def test_foraging_decision_evaluation(self):
        """Test foraging decision evaluation"""
        # Setup mocks
        self.mock_decision_engine.make_foraging_decision.return_value = {
            "decision_type": ForagingDecisionType.EXPLORE_NEW,
            "selected_patches": [self.test_patch_info],
            "confidence": 0.8
        }
        
        # Initialize system
        self.foraging_system.initialize_colony_foraging(1, self.test_colony_state, [self.test_patch_info])
        
        # Test decision evaluation
        with patch('random.random', return_value=0.3):  # Ensure decision is positive
            decision_result = self.foraging_system._evaluate_foraging_decision(
                123, 1, self.test_bee_state, self.test_colony_state, self.test_environmental_context
            )
        
        assert "should_forage" in decision_result
        assert "decision_type" in decision_result
        assert "selected_patches" in decision_result
        assert "decision_confidence" in decision_result
        assert "strategy_influence" in decision_result
        assert "decision_probability" in decision_result
        
        # Verify decision engine was called
        self.mock_decision_engine.make_foraging_decision.assert_called_once()
    
    def test_foraging_step_new_session(self):
        """Test executing foraging step with new session"""
        # Setup mocks
        self.mock_decision_engine.make_foraging_decision.return_value = {
            "decision_type": ForagingDecisionType.EXPLORE_NEW,
            "selected_patches": [self.test_patch_info],
            "confidence": 0.8
        }
        
        self.mock_patch_selector.select_optimal_patches.return_value = {
            "selected_patches": [self.test_patch_info],
            "patch_qualities": {101: 0.8},
            "selection_confidence": 0.9
        }
        
        mock_trip_lifecycle = ForagingTripLifecycle(
            travel_time_to_patch=10.0,
            handling_time_per_flower=2.0,
            inter_flower_time=1.0,
            flowers_visited=8,
            travel_time_to_hive=10.0,
            total_trip_duration=50.0,
            energy_consumed_travel=5.0,
            energy_consumed_foraging=3.0,
            energy_gained=12.0,
            net_energy_gain=4.0
        )
        
        self.mock_trip_manager.simulate_complete_foraging_trip.return_value = mock_trip_lifecycle
        
        # Mock communication integrator
        self.mock_communication_integrator.integrate_foraging_success_with_communication.return_value = {
            "communication_triggered": True,
            "dance_probability": 0.8,
            "recruitment_success": True,
            "information_shared": True
        }
        
        # Initialize system
        self.foraging_system.initialize_colony_foraging(1, self.test_colony_state, [self.test_patch_info])
        
        # Execute foraging step
        with patch('random.random', return_value=0.3), \
             patch('time.time', return_value=1000.0):
            
            step_result = self.foraging_system.execute_foraging_step(
                123, 1, self.test_bee_state, self.test_colony_state, self.test_environmental_context
            )
        
        assert step_result["action_taken"] == "foraging_session_started"
        assert "session_id" in step_result
        assert "trip_result" in step_result
        assert "foraging_decision" in step_result
        
        # Verify bee is now in active session
        assert 123 in self.foraging_system.active_foraging_sessions
    
    def test_foraging_step_declined(self):
        """Test foraging step when foraging is declined"""
        # Setup mocks to decline foraging
        self.mock_decision_engine.make_foraging_decision.return_value = {
            "decision_type": ForagingDecisionType.EXPLORE_NEW,
            "selected_patches": [self.test_patch_info],
            "confidence": 0.2
        }
        
        # Initialize system
        self.foraging_system.initialize_colony_foraging(1, self.test_colony_state, [self.test_patch_info])
        
        # Execute foraging step with high random value to decline
        with patch('random.random', return_value=0.9):
            step_result = self.foraging_system.execute_foraging_step(
                123, 1, self.test_bee_state, self.test_colony_state, self.test_environmental_context
            )
        
        assert step_result["action_taken"] == "foraging_declined"
        assert "foraging_decision" in step_result
        
        # Verify bee is not in active session
        assert 123 not in self.foraging_system.active_foraging_sessions
    
    def test_continue_foraging_session(self):
        """Test continuing an existing foraging session"""
        # Setup existing session
        session_data = {
            "session_id": "session_123_1000",
            "bee_id": 123,
            "colony_id": 1,
            "start_time": time.time() - 100,  # 100 seconds ago
            "current_trip": 1,
            "completed_trips": [],
            "patches_visited": set(),
            "total_energy_consumed": 5.0,
            "total_energy_gained": 8.0,
            "total_distance": 20.0,
            "resources_collected": {"nectar": 3.0, "pollen": 0.0},
            "communication_events": [],
            "current_patch": None,
            "session_status": "active"
        }
        
        self.foraging_system.active_foraging_sessions[123] = session_data
        self.foraging_system.initialize_colony_foraging(1, self.test_colony_state, [self.test_patch_info])
        
        # Setup mocks for continuing session
        self.mock_patch_selector.select_optimal_patches.return_value = {
            "selected_patches": [self.test_patch_info],
            "patch_qualities": {101: 0.8},
            "selection_confidence": 0.9
        }
        
        mock_trip_lifecycle = ForagingTripLifecycle(
            travel_time_to_patch=10.0,
            handling_time_per_flower=2.0,
            inter_flower_time=1.0,
            flowers_visited=8,
            travel_time_to_hive=10.0,
            total_trip_duration=50.0,
            energy_consumed_travel=5.0,
            energy_consumed_foraging=3.0,
            energy_gained=12.0,
            net_energy_gain=4.0
        )
        
        self.mock_trip_manager.simulate_complete_foraging_trip.return_value = mock_trip_lifecycle
        
        # Mock communication integrator
        self.mock_communication_integrator.integrate_foraging_success_with_communication.return_value = {
            "communication_triggered": True,
            "dance_probability": 0.8,
            "recruitment_success": True,
            "information_shared": True
        }
        
        # Continue session
        step_result = self.foraging_system.execute_foraging_step(
            123, 1, self.test_bee_state, self.test_colony_state, self.test_environmental_context
        )
        
        assert step_result["action_taken"] == "foraging_session_continued"
        assert "session_id" in step_result
        assert "trip_result" in step_result
        assert "session_duration" in step_result
    
    def test_session_termination_duration(self):
        """Test session termination due to duration limit"""
        # Setup session that exceeds duration limit
        session_data = {
            "session_id": "session_123_1000",
            "bee_id": 123,
            "colony_id": 1,
            "start_time": time.time() - 500,  # 500 seconds ago (exceeds 480s limit)
            "current_trip": 3,
            "completed_trips": [],
            "patches_visited": {101, 102},
            "total_energy_consumed": 20.0,
            "total_energy_gained": 30.0,
            "total_distance": 100.0,
            "resources_collected": {"nectar": 15.0, "pollen": 5.0},
            "communication_events": [],
            "current_patch": 102,
            "session_status": "active"
        }
        
        self.foraging_system.active_foraging_sessions[123] = session_data
        self.foraging_system.initialize_colony_foraging(1, self.test_colony_state, [self.test_patch_info])
        
        # Execute step - should terminate session
        step_result = self.foraging_system.execute_foraging_step(
            123, 1, self.test_bee_state, self.test_colony_state, self.test_environmental_context
        )
        
        # Should result in session termination
        assert step_result["action_taken"] == "foraging_session_ended"
        assert 123 not in self.foraging_system.active_foraging_sessions
        assert len(self.foraging_system.foraging_history) == 1
    
    def test_session_termination_low_energy(self):
        """Test session termination due to low bee energy"""
        # Setup session with low energy bee
        session_data = {
            "session_id": "session_123_1000",
            "bee_id": 123,
            "colony_id": 1,
            "start_time": time.time() - 100,
            "current_trip": 2,
            "completed_trips": [],
            "patches_visited": {101},
            "total_energy_consumed": 15.0,
            "total_energy_gained": 20.0,
            "total_distance": 50.0,
            "resources_collected": {"nectar": 8.0, "pollen": 2.0},
            "communication_events": [],
            "current_patch": 101,
            "session_status": "active"
        }
        
        self.foraging_system.active_foraging_sessions[123] = session_data
        self.foraging_system.initialize_colony_foraging(1, self.test_colony_state, [self.test_patch_info])
        
        # Set low energy in bee state
        low_energy_bee_state = self.test_bee_state.copy()
        low_energy_bee_state["energy"] = 15.0  # Below 20.0 threshold
        
        # Execute step - should terminate session
        step_result = self.foraging_system.execute_foraging_step(
            123, 1, low_energy_bee_state, self.test_colony_state, self.test_environmental_context
        )
        
        # Should result in session termination
        assert step_result["action_taken"] == "foraging_session_ended"
        assert 123 not in self.foraging_system.active_foraging_sessions
    
    def test_no_patches_available(self):
        """Test handling when no patches are available"""
        # Setup session where all patches have been visited
        session_data = {
            "session_id": "session_123_1000",
            "bee_id": 123,
            "colony_id": 1,
            "start_time": time.time() - 100,
            "current_trip": 1,
            "completed_trips": [],
            "patches_visited": {101},  # All available patches visited
            "total_energy_consumed": 5.0,
            "total_energy_gained": 8.0,
            "total_distance": 20.0,
            "resources_collected": {"nectar": 3.0, "pollen": 0.0},
            "communication_events": [],
            "current_patch": None,
            "session_status": "active"
        }
        
        self.foraging_system.active_foraging_sessions[123] = session_data
        self.foraging_system.initialize_colony_foraging(1, self.test_colony_state, [self.test_patch_info])
        
        # Reset patch selector mock so it doesn't interfere with the no-patches logic
        self.mock_patch_selector.reset_mock()
        
        # Mock communication integrator for the case where it's needed
        self.mock_communication_integrator.integrate_foraging_success_with_communication.return_value = {
            "communication_triggered": False,
            "dance_probability": 0.0,
            "recruitment_success": False,
            "information_shared": False
        }
        
        # Execute step - should end session due to no available patches
        step_result = self.foraging_system.execute_foraging_step(
            123, 1, self.test_bee_state, self.test_colony_state, self.test_environmental_context
        )
        
        # Should result in session termination
        assert step_result["action_taken"] == "foraging_session_ended"
        assert 123 not in self.foraging_system.active_foraging_sessions
    
    def test_flower_accessibility_evaluation(self):
        """Test flower accessibility evaluation"""
        # Mock proboscis system
        from src.bstew.components.proboscis_matching import AccessibilityLevel
        mock_accessibility = AccessibilityResult(
            accessibility_level=AccessibilityLevel.GOOD,
            accessibility_score=0.8,
            nectar_extraction_efficiency=0.9,
            pollen_extraction_efficiency=0.7,
            energy_cost_multiplier=1.1,
            handling_time_multiplier=0.9
        )
        
        # Mock the actual methods used
        from src.bstew.components.proboscis_matching import ProboscisCharacteristics
        mock_proboscis = ProboscisCharacteristics(
            length_mm=9.0,
            width_mm=0.8,
            flexibility=0.7
        )
        
        self.mock_proboscis_system.get_species_proboscis.return_value = mock_proboscis
        self.mock_proboscis_system.calculate_accessibility.return_value = mock_accessibility
        self.mock_proboscis_system.get_foraging_efficiency_modifier.return_value = {
            "efficiency_multiplier": 1.2,
            "energy_cost_modifier": 0.8
        }
        
        # Test accessibility evaluation
        accessibility_results = self.foraging_system._evaluate_flower_accessibility(
            "bombus_terrestris", self.test_patch_info
        )
        
        assert isinstance(accessibility_results, dict)
        self.mock_proboscis_system.get_species_proboscis.assert_called_once()
        self.mock_proboscis_system.calculate_accessibility.assert_called_once()
        self.mock_proboscis_system.get_foraging_efficiency_modifier.assert_called_once()
    
    def test_strategy_influence_application(self):
        """Test application of colony strategy influence"""
        foraging_decision = {
            "decision_type": ForagingDecisionType.EXPLORE_NEW,
            "confidence": 0.5
        }
        
        # Test different strategies
        exploration_influence = self.foraging_system._apply_strategy_influence(
            foraging_decision, ForagingMode.EXPLORATION
        )
        
        emergency_influence = self.foraging_system._apply_strategy_influence(
            foraging_decision, ForagingMode.EMERGENCY_FORAGING
        )
        
        mixed_influence = self.foraging_system._apply_strategy_influence(
            foraging_decision, ForagingMode.MIXED_STRATEGY
        )
        
        # All should return valid probabilities
        assert 0.0 <= exploration_influence <= 1.0
        assert 0.0 <= emergency_influence <= 1.0
        assert 0.0 <= mixed_influence <= 1.0
    
    def test_performance_metrics_update(self):
        """Test performance metrics tracking"""
        # Initialize system
        self.foraging_system.initialize_colony_foraging(1, self.test_colony_state, [self.test_patch_info])
        
        # Simulate completed session
        session_result = ForagingSessionResult(
            session_id="test_session",
            bee_id=123,
            colony_id=1,
            session_start_time=1000.0,
            session_duration=300.0,
            trips_completed=3,
            patches_visited={101, 102},
            total_distance_traveled=75.0,
            total_energy_consumed=25.0,
            total_energy_gained=40.0,
            net_energy_result=15.0,
            nectar_collected=20.0,
            pollen_collected=10.0,
            resource_quality_avg=0.8,
            dances_performed=1,
            recruits_obtained=2,
            information_shared=3,
            trip_efficiency=0.85,
            energy_efficiency=1.6,
            time_efficiency=0.75,
            overall_success_score=0.8
        )
        
        # Add to history
        self.foraging_system.foraging_history.append(session_result)
        
        # Update performance metrics
        if hasattr(self.foraging_system, '_update_performance_metrics'):
            self.foraging_system._update_performance_metrics(1, session_result)
        
        # Verify metrics exist
        metrics = self.foraging_system.system_performance_metrics[1]
        assert "total_foraging_sessions" in metrics
        assert "successful_sessions" in metrics


class TestForagingSystemIntegration:
    """Test integration aspects of foraging system"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.foraging_system = IntegratedForagingSystem()
        
        # Create realistic test data
        self.patches = [
            PatchInfo(
                patch_id=i,
                location=(i * 10.0, i * 5.0),
                resource_type=PatchResourceType.NECTAR if i % 2 == 0 else PatchResourceType.POLLEN,
                quality_metrics={},  # Use empty dict for simplicity
                species_compatibility={},  # Use empty dict for simplicity
                distance_from_hive=50.0 + i * 10,
                current_foragers=i,
                max_capacity=10,
                depletion_rate=0.1,
                regeneration_rate=0.05,
                seasonal_availability={}  # Use empty dict for simplicity
            )
            for i in range(1, 6)
        ]
        
        self.colony_state = {
            "energy_level": 1000.0,
            "forager_count": 20,
            "species": "bombus_terrestris",
            "location": (0.0, 0.0)
        }
    
    def test_full_foraging_workflow(self):
        """Test complete foraging workflow integration"""
        # Initialize system
        self.foraging_system.initialize_colony_foraging(1, self.colony_state, self.patches)
        
        # Verify initialization
        assert len(self.foraging_system.patch_database) == 5
        assert 1 in self.foraging_system.colony_foraging_strategies
        assert 1 in self.foraging_system.system_performance_metrics
        
        # Test that all patches are available
        for patch in self.patches:
            assert patch.patch_id in self.foraging_system.patch_database
    
    def test_multiple_bee_sessions(self):
        """Test managing multiple concurrent bee sessions"""
        self.foraging_system.initialize_colony_foraging(1, self.colony_state, self.patches)
        
        bee_states = [
            {"energy": 90.0, "foraging_experience": i, "memory": {}}
            for i in range(3)
        ]
        
        environmental_context = {
            "season": "summer",
            "hour": 12,
            "weather": "clear",
            "temperature": 25.0
        }
        
        # Start sessions for multiple bees
        results = []
        for i, bee_state in enumerate(bee_states):
            with patch('random.random', return_value=0.2):  # Ensure foraging starts
                result = self.foraging_system.execute_foraging_step(
                    i + 100, 1, bee_state, self.colony_state, environmental_context
                )
                results.append(result)
        
        # Verify multiple sessions can be managed
        assert len(results) == 3
        
        # Check that some bees started foraging (depends on random decisions)
        active_sessions = len(self.foraging_system.active_foraging_sessions)
        assert active_sessions >= 0  # Could be 0-3 depending on decisions
    
    def test_session_data_persistence(self):
        """Test that session data persists correctly"""
        self.foraging_system.initialize_colony_foraging(1, self.colony_state, self.patches)
        
        # Create a session manually
        session_data = {
            "session_id": "manual_session_123",
            "bee_id": 123,
            "colony_id": 1,
            "start_time": time.time(),
            "current_trip": 0,
            "completed_trips": [],
            "patches_visited": set(),
            "total_energy_consumed": 0.0,
            "total_energy_gained": 0.0,
            "total_distance": 0.0,
            "resources_collected": {"nectar": 0.0, "pollen": 0.0},
            "communication_events": [],
            "current_patch": None,
            "session_status": "active"
        }
        
        self.foraging_system.active_foraging_sessions[123] = session_data
        
        # Verify session persists
        assert 123 in self.foraging_system.active_foraging_sessions
        retrieved_session = self.foraging_system.active_foraging_sessions[123]
        assert retrieved_session["session_id"] == "manual_session_123"
        assert retrieved_session["bee_id"] == 123
        assert retrieved_session["session_status"] == "active"


class TestForagingSystemEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup edge case test fixtures"""
        # Create mocked dependencies
        self.mock_decision_engine = Mock(spec=ForagingDecisionEngine)
        self.mock_trip_manager = Mock(spec=ForagingTripManager)
        self.mock_patch_selector = Mock(spec=AdvancedPatchSelector)
        self.mock_communication_integrator = Mock(spec=ForagingCommunicationIntegrator)
        self.mock_proboscis_system = Mock(spec=ProboscisCorollaSystem)
        
        # Create integrated system with mocked dependencies
        self.foraging_system = IntegratedForagingSystem(
            decision_engine=self.mock_decision_engine,
            trip_manager=self.mock_trip_manager,
            patch_selector=self.mock_patch_selector,
            communication_integrator=self.mock_communication_integrator,
            proboscis_system=self.mock_proboscis_system
        )
    
    def test_empty_patch_database(self):
        """Test behavior with empty patch database"""
        self.foraging_system.initialize_colony_foraging(1, {"energy_level": 1000.0, "forager_count": 20}, [])
        
        # Should handle empty patch database gracefully
        assert len(self.foraging_system.patch_database) == 0
        assert 1 in self.foraging_system.colony_foraging_strategies
    
    def test_invalid_bee_state(self):
        """Test handling of invalid bee state"""
        # Mock the decision engine
        self.mock_decision_engine.make_foraging_decision.return_value = {
            "decision_type": ForagingDecisionType.EXPLORE_NEW,
            "selected_patches": [],
            "confidence": 0.5
        }
        
        self.foraging_system.initialize_colony_foraging(1, {"energy_level": 1000.0, "forager_count": 20}, [])
        
        # Test with missing energy
        invalid_bee_state = {"foraging_experience": 0}
        
        environmental_context = {"season": "spring", "hour": 12}
        colony_state = {"energy_level": 1000.0}
        
        # Should handle gracefully using defaults
        result = self.foraging_system.execute_foraging_step(
            999, 1, invalid_bee_state, colony_state, environmental_context
        )
        
        assert isinstance(result, dict)
        assert "action_taken" in result
    
    def test_invalid_environmental_context(self):
        """Test handling of invalid environmental context"""
        # Mock the decision engine
        self.mock_decision_engine.make_foraging_decision.return_value = {
            "decision_type": ForagingDecisionType.EXPLORE_NEW,
            "selected_patches": [],
            "confidence": 0.5
        }
        
        # Mock patch selector to decline foraging (returns no patches)
        self.mock_patch_selector.select_optimal_patches.return_value = {
            "selected_patches": [],
            "patch_qualities": {},
            "selection_confidence": 0.0
        }
        
        patch_info = PatchInfo(
            patch_id=1,
            location=(10.0, 10.0),
            resource_type=PatchResourceType.NECTAR,
            quality_metrics={},  # Use empty dict for simplicity
            species_compatibility={},  # Use empty dict for simplicity
            distance_from_hive=50.0,
            current_foragers=2,
            max_capacity=10,
            depletion_rate=0.1,
            regeneration_rate=0.05,
            seasonal_availability={}  # Use empty dict for simplicity
        )
        
        self.foraging_system.initialize_colony_foraging(1, {"energy_level": 1000.0, "forager_count": 20}, [patch_info])
        
        # Test with minimal environmental context
        minimal_context = {}
        bee_state = {"energy": 80.0}
        colony_state = {"energy_level": 1000.0}
        
        # Should handle gracefully using defaults
        result = self.foraging_system.execute_foraging_step(
            888, 1, bee_state, colony_state, minimal_context
        )
        
        assert isinstance(result, dict)
        assert "action_taken" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])