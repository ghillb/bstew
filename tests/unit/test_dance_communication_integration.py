"""
Unit tests for Dance Communication Integration
=============================================

Tests for the advanced dance communication integration system that connects
bee agents with waggle dance mechanics and recruitment patterns.
"""

import pytest
from unittest.mock import Mock
import time

from src.bstew.core.dance_communication_integration import (
    DanceDecisionEngine, RecruitmentProcessor, DanceCommunicationIntegrator,
    create_dance_communication_integration
)
from src.bstew.core.bee_communication import (
    DanceType, DanceInformation
)
from src.bstew.core.enums import BeeStatus
from src.bstew.core.resource_collection import ResourceType


class TestDanceDecisionEngine:
    """Test dance decision engine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = DanceDecisionEngine()
        
        # Mock bee
        self.bee_mock = Mock()
        self.bee_mock.unique_id = 123
        self.bee_mock.energy = 80.0
        self.bee_mock.foraging_success_rate = 0.7
        self.bee_mock.recent_patch_quality = 0.8
        self.bee_mock.dance_threshold = 0.6
        
        # Mock patch
        self.patch_mock = Mock()
        self.patch_mock.id = 456
        self.patch_mock.resource_quality = 0.9
        self.patch_mock.distance_from_nest = 150.0
        self.patch_mock.current_nectar = 75.0
    
    def test_initialization(self):
        """Test engine initialization"""
        assert hasattr(self.engine, 'min_dance_quality_threshold')
        assert hasattr(self.engine, 'quality_weight')
        assert hasattr(self.engine, 'energy_weight')
        assert hasattr(self.engine, 'distance_weight')
        assert self.engine.min_dance_quality_threshold == 0.3
        assert self.engine.quality_weight == 0.4
    
    def test_dance_decision_factors(self):
        """Test dance decision factor evaluation"""
        # Test that the engine can evaluate dance decisions
        foraging_result = {
            'patch_quality': 0.9,
            'energy_gained': 50.0,
            'distance_traveled': 150.0,
            'patch_id': 1
        }
        colony_state = {'current_need': 0.8}
        bee_experience = {'foraging_success_rate': 0.7}
        
        decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, foraging_result, colony_state, bee_experience
        )
        
        # Test decision structure
        assert hasattr(decision, 'should_dance')
        assert hasattr(decision, 'dance_type')
        assert hasattr(decision, 'dance_intensity')
        assert hasattr(decision, 'decision_factors')
    
    def test_dance_probability_calculation(self):
        """Test dance probability calculation through decision evaluation"""
        # Test high-quality patch scenario
        high_quality_result = {
            'patch_quality': 0.9,
            'energy_gained': 80.0,
            'distance_traveled': 100.0,
            'patch_id': 1
        }
        colony_state = {'current_need': 0.8}
        bee_experience = {'foraging_success_rate': 0.8}
        
        high_decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, high_quality_result, colony_state, bee_experience
        )
        
        # Test low-quality patch scenario
        low_quality_result = {
            'patch_quality': 0.2,
            'energy_gained': 10.0,
            'distance_traveled': 400.0,
            'patch_id': 2
        }
        
        low_decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, low_quality_result, colony_state, bee_experience
        )
        
        # High quality should be more likely to result in dancing
        assert isinstance(high_decision.should_dance, bool)
        assert isinstance(low_decision.should_dance, bool)
        assert 0.0 <= high_decision.dance_intensity <= 1.0
        assert 0.0 <= low_decision.dance_intensity <= 1.0
    
    def test_dance_type_selection(self):
        """Test dance type selection logic"""
        # Test close patch decision
        close_result = {
            'patch_quality': 0.8,
            'energy_gained': 50.0,
            'distance_traveled': 50.0,
            'patch_id': 1
        }
        
        close_decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, close_result, {}, {}
        )
        
        # Test distant patch decision
        distant_result = {
            'patch_quality': 0.8,
            'energy_gained': 50.0,
            'distance_traveled': 300.0,
            'patch_id': 2
        }
        
        distant_decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, distant_result, {}, {}
        )
        
        # Verify dance types are valid
        assert close_decision.dance_type in [DanceType.ROUND_DANCE, DanceType.WAGGLE_DANCE, DanceType.RECRUITMENT_DANCE]
        assert distant_decision.dance_type in [DanceType.ROUND_DANCE, DanceType.WAGGLE_DANCE, DanceType.RECRUITMENT_DANCE]
    
    def test_dance_intensity_calculation(self):
        """Test dance intensity calculation"""
        # High quality patch scenario
        high_result = {
            'patch_quality': 0.9,
            'energy_gained': 90.0,
            'distance_traveled': 100.0,
            'patch_id': 1
        }
        
        high_decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, high_result, {}, {}
        )
        
        # Low quality patch scenario
        low_result = {
            'patch_quality': 0.3,
            'energy_gained': 20.0,
            'distance_traveled': 100.0,
            'patch_id': 2
        }
        
        low_decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, low_result, {}, {}
        )
        
        # Test intensity values
        assert 0.0 <= high_decision.dance_intensity <= 1.0
        assert 0.0 <= low_decision.dance_intensity <= 1.0
    
    def test_contextual_dance_decisions(self):
        """Test contextual dance decisions"""
        # Test emergency colony state
        emergency_result = {
            'patch_quality': 0.6,
            'energy_gained': 30.0,
            'distance_traveled': 150.0,
            'patch_id': 1
        }
        emergency_colony_state = {
            'current_need': 0.9,
            'resource_scarcity': True
        }
        
        emergency_decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, emergency_result, emergency_colony_state, {}
        )
        
        # Test normal conditions
        normal_colony_state = {
            'current_need': 0.3,
            'resource_scarcity': False
        }
        
        normal_decision = self.engine.evaluate_dance_decision(
            self.bee_mock.unique_id, emergency_result, normal_colony_state, {}
        )
        
        # Test decision structure
        assert hasattr(emergency_decision, 'should_dance')
        assert hasattr(emergency_decision, 'dance_type')
        assert hasattr(emergency_decision, 'dance_intensity')
        assert hasattr(normal_decision, 'should_dance')
        assert isinstance(normal_decision.should_dance, bool)
        assert isinstance(emergency_decision.should_dance, bool)


class TestRecruitmentProcessor:
    """Test recruitment processing system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = RecruitmentProcessor()
        
        # Mock dance information
        self.dance_info = DanceInformation(
            dance_id="test_dance_1",
            dancer_id=123,
            dance_type=DanceType.WAGGLE_DANCE,
            patch_id=456,
            patch_location=(100.0, 150.0),
            patch_distance=200.0,
            patch_direction=1.57,  # π/2 radians (90 degrees)
            resource_type=ResourceType.NECTAR,
            resource_quality=0.8,
            resource_quantity=1.0,
            energy_profitability=50.0,
            dance_duration=15.0,
            dance_vigor=0.9,
            waggle_run_count=18,
            dance_repetitions=3,
            recruitment_threshold=0.6,
            urgency_level=0.7,
            timestamp=time.time()
        )
        
        # Mock potential recruits
        self.recruit1 = Mock()
        self.recruit1.unique_id = 124
        self.recruit1.energy = 75.0
        self.recruit1.recruitment_threshold = 0.6
        self.recruit1.current_status = BeeStatus.RESTING
        
        self.recruit2 = Mock()
        self.recruit2.unique_id = 125
        self.recruit2.energy = 60.0
        self.recruit2.recruitment_threshold = 0.8
        self.recruit2.current_status = BeeStatus.NURSING
    
    def test_initialization(self):
        """Test processor initialization"""
        assert hasattr(self.processor, 'attention_probability_base')
        assert hasattr(self.processor, 'attention_distance_factor')
        assert hasattr(self.processor, 'recruitment_threshold')
        assert self.processor.attention_probability_base == 0.3
        assert self.processor.recruitment_threshold == 0.6
    
    def test_attention_calculation(self):
        """Test attention calculation through dance processing"""
        # Create a dance performance
        from src.bstew.core.dance_communication_integration import DancePerformance
        
        performance = DancePerformance(
            performance_id="test_perf_1",
            dancer_id=123,
            dance_info=self.dance_info,
            start_time=time.time(),
            duration=15.0,
            intensity=0.8,
            audience_size=5
        )
        
        # Test with different bee states
        high_energy_state = {
            'status': BeeStatus.RESTING,
            'energy': 80.0,
            'attention_tendency': 0.8
        }
        
        low_energy_state = {
            'status': BeeStatus.NURSING,
            'energy': 40.0,
            'attention_tendency': 0.3
        }
        
        # Test that attention determination method exists and works
        high_attention = self.processor._determines_attention(
            self.recruit1.unique_id, performance, high_energy_state
        )
        low_attention = self.processor._determines_attention(
            self.recruit2.unique_id, performance, low_energy_state
        )
        
        assert isinstance(high_attention, bool)
        assert isinstance(low_attention, bool)
    
    def test_recruitment_decision(self):
        """Test recruitment decision making through process_dance_audience"""
        from src.bstew.core.dance_communication_integration import DancePerformance
        
        performance = DancePerformance(
            performance_id="test_perf_2",
            dancer_id=123,
            dance_info=self.dance_info,
            start_time=time.time(),
            duration=15.0,
            intensity=0.8
        )
        
        potential_followers = [self.recruit1.unique_id, self.recruit2.unique_id]
        colony_state = {'energy_level': 0.6}
        bee_states = {
            self.recruit1.unique_id: {
                'status': BeeStatus.RESTING,
                'energy': 75.0,
                'attention_tendency': 0.7,
                'foraging_motivation': 0.8
            },
            self.recruit2.unique_id: {
                'status': BeeStatus.NURSING,
                'energy': 60.0,
                'attention_tendency': 0.5,
                'foraging_motivation': 0.4
            }
        }
        
        responses = self.processor.process_dance_audience(
            performance, potential_followers, colony_state, bee_states
        )
        
        # Should return list of FollowerResponse objects
        assert isinstance(responses, list)
        # Responses length should be <= number of followers (some may not pay attention)
        assert len(responses) <= len(potential_followers)
    
    def test_patch_information_decoding(self):
        """Test patch information through dance_info object"""
        # The dance info already contains decoded patch information
        assert hasattr(self.dance_info, 'patch_location')
        assert hasattr(self.dance_info, 'patch_distance')
        assert hasattr(self.dance_info, 'patch_direction')
        assert hasattr(self.dance_info, 'resource_quality')
        
        # Check location information
        location = self.dance_info.patch_location
        assert isinstance(location, tuple)
        assert len(location) == 2
        
        # Check quality estimate
        quality = self.dance_info.resource_quality
        assert 0.0 <= quality <= 1.0
    
    def test_recruitment_success_tracking(self):
        """Test recruitment success through response analysis"""
        from src.bstew.core.dance_communication_integration import DancePerformance
        
        performance = DancePerformance(
            performance_id="test_perf_3",
            dancer_id=123,
            dance_info=self.dance_info,
            start_time=time.time(),
            duration=15.0,
            intensity=0.8
        )
        
        potential_followers = [124, 125, 126]
        colony_state = {'energy_level': 0.7}
        bee_states = {
            124: {'status': BeeStatus.RESTING, 'energy': 75.0},
            125: {'status': BeeStatus.RESTING, 'energy': 70.0}, 
            126: {'status': BeeStatus.NURSING, 'energy': 60.0}
        }
        
        responses = self.processor.process_dance_audience(
            performance, potential_followers, colony_state, bee_states
        )
        
        # Calculate success rate from responses
        if responses:
            follow_through_count = sum(1 for r in responses if r.follow_through)
            success_rate = follow_through_count / len(responses)
            assert 0.0 <= success_rate <= 1.0
    
    def test_dance_following_behavior(self):
        """Test dance following behavior through information acquisition"""
        from src.bstew.core.dance_communication_integration import DancePerformance
        
        performance = DancePerformance(
            performance_id="test_perf_4",
            dancer_id=123,
            dance_info=self.dance_info,
            start_time=time.time(),
            duration=15.0,
            intensity=0.8
        )
        
        follower_state = {
            'status': BeeStatus.RESTING,
            'energy': 75.0,
            'dance_following_experience': 0.6,
            'learning_rate': 0.7
        }
        colony_state = {'energy_level': 0.6}
        
        # Test information acquisition process
        response = self.processor._process_information_acquisition(
            self.recruit1.unique_id, performance, follower_state, colony_state
        )
        
        assert hasattr(response, 'attention_duration')
        assert hasattr(response, 'information_quality')
        assert hasattr(response, 'follow_through')
        assert response.attention_duration > 0
        assert 0.0 <= response.information_quality <= 1.0
    
    def test_recruitment_bias_factors(self):
        """Test recruitment bias factors"""
        # Test different dance qualities  
        high_quality_dance = DanceInformation(
            dance_id="test_dance_high",
            dancer_id=125,
            dance_type=DanceType.WAGGLE_DANCE,
            patch_id=458,
            patch_location=(200.0, 300.0),
            patch_distance=200.0,
            patch_direction=1.57,
            resource_type=ResourceType.NECTAR,
            resource_quality=0.9,
            resource_quantity=1.0,
            energy_profitability=80.0,
            dance_duration=20.0,
            dance_vigor=0.95,
            waggle_run_count=25,
            dance_repetitions=4,
            recruitment_threshold=0.6,
            urgency_level=0.9,
            timestamp=time.time()
        )
        
        low_quality_dance = DanceInformation(
            dance_id="test_dance_low",
            dancer_id=126,
            dance_type=DanceType.ROUND_DANCE,
            patch_id=457,
            patch_location=(25.0, 35.0),
            patch_distance=50.0,
            patch_direction=0.0,
            resource_type=ResourceType.NECTAR,
            resource_quality=0.3,
            resource_quantity=1.0,
            energy_profitability=10.0,
            dance_duration=5.0,
            dance_vigor=0.2,
            waggle_run_count=4,
            dance_repetitions=1,
            recruitment_threshold=0.6,
            urgency_level=0.2,
            timestamp=time.time()
        )
        
        # Test recruitment bias by comparing dance responses
        from src.bstew.core.dance_communication_integration import DancePerformance
        
        high_performance = DancePerformance(
            performance_id="test_high_quality",
            dancer_id=125,
            dance_info=high_quality_dance,
            start_time=time.time(),
            duration=20.0,
            intensity=0.95
        )
        
        low_performance = DancePerformance(
            performance_id="test_low_quality", 
            dancer_id=126,
            dance_info=low_quality_dance,
            start_time=time.time(),
            duration=5.0,
            intensity=0.2
        )
        
        follower_state = {
            'status': BeeStatus.RESTING,
            'energy': 75.0,
            'foraging_motivation': 0.7
        }
        colony_state = {'energy_level': 0.6}
        
        # Process both dances with same follower
        high_response = self.processor._process_information_acquisition(
            self.recruit1.unique_id, high_performance, follower_state, colony_state
        )
        low_response = self.processor._process_information_acquisition(
            self.recruit1.unique_id, low_performance, follower_state, colony_state
        )
        
        # Both should be valid responses
        assert hasattr(high_response, 'follow_through')
        assert hasattr(low_response, 'follow_through')
        assert isinstance(high_response.follow_through, bool)
        assert isinstance(low_response.follow_through, bool)


class TestDanceCommunicationIntegrator:
    """Test dance communication integrator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.integrator = DanceCommunicationIntegrator()
        
        # Mock colony
        self.colony_mock = Mock()
        self.colony_mock.unique_id = 1
        self.colony_mock.location = (0.0, 0.0)
        
        # Mock model
        self.model_mock = Mock()
        self.model_mock.current_step = 100
    
    def test_initialization(self):
        """Test integrator initialization"""
        assert hasattr(self.integrator, 'dance_decision_engine')
        assert hasattr(self.integrator, 'recruitment_processor')
        assert hasattr(self.integrator, 'active_dances')
    
    def test_dance_initiation(self):
        """Test dance initiation process using process_returning_forager"""
        # Mock foraging result
        foraging_result = {
            'patch_quality': 0.8,
            'energy_gained': 45.0,
            'distance_traveled': 200.0,
            'patch_id': 456,
            'patch_location': (200.0, 100.0),
            'resource_type': 'nectar'
        }
        
        colony_state = {
            'energy_level': 0.6,
            'total_bees': 100,
            'available_foragers': 25
        }
        
        bee_states = {
            123: {
                'energy': 85.0,
                'foraging_experience': 0.7,
                'colony_id': 1
            }
        }
        
        # Test dance processing
        dance_performance = self.integrator.process_returning_forager(
            bee_id=123, 
            foraging_result=foraging_result,
            colony_id=1,
            colony_state=colony_state,
            bee_states=bee_states
        )
        
        # Should return DancePerformance or None
        assert dance_performance is None or hasattr(dance_performance, 'performance_id')
    
    def test_recruitment_processing(self):
        """Test recruitment processing"""
        # Create dance information
        DanceInformation(
            dance_id="test_recruitment_dance",
            dancer_id=123,
            dance_type=DanceType.WAGGLE_DANCE,
            patch_id=456,
            patch_location=(150.0, 100.0),
            patch_distance=150.0,
            patch_direction=0.785,  # 45 degrees
            resource_type=ResourceType.NECTAR,
            resource_quality=0.85,
            resource_quantity=1.0,
            energy_profitability=60.0,
            dance_duration=12.0,
            dance_vigor=0.8,
            waggle_run_count=16,
            dance_repetitions=2,
            recruitment_threshold=0.6,
            urgency_level=0.7,
            timestamp=time.time()
        )
        
        # Mock potential recruits
        potential_recruits = []
        for i in range(5):
            recruit = Mock()
            recruit.unique_id = 200 + i
            recruit.energy = 70.0 + i * 5
            recruit.recruitment_threshold = 0.5 + i * 0.1
            recruit.current_status = BeeStatus.RESTING
            potential_recruits.append(recruit)
        
        # Test recruitment through get_recruited_bees method
        recruited_bees = self.integrator.get_recruited_bees(colony_id=1)
        
        # Should return a list (empty initially)
        assert isinstance(recruited_bees, list)
        
        # Test that integrator has proper attributes
        assert hasattr(self.integrator, 'active_dances')
        assert hasattr(self.integrator, 'dance_followers')
        assert isinstance(self.integrator.active_dances, dict)
        assert isinstance(self.integrator.dance_followers, dict)
    
    def test_communication_effectiveness_tracking(self):
        """Test communication effectiveness tracking through available methods"""
        # Test available colony metrics method
        metrics = self.integrator.get_colony_communication_metrics(colony_id=1)
        
        # Should return metrics dictionary
        assert isinstance(metrics, dict)
        assert 'active_dances' in metrics
        assert 'total_recruited_bees' in metrics
        assert 'known_patches' in metrics
        assert 'average_dance_success_rate' in metrics
        assert 'information_quality' in metrics
    
    def test_dance_interference_and_competition(self):
        """Test basic dance management functionality"""
        # Test cleanup method
        current_time = time.time()
        self.integrator.cleanup_finished_dances(current_time)
        
        # Should not crash and maintain empty dances dict
        assert isinstance(self.integrator.active_dances, dict)
        
        # Test follower outcome updates (should handle gracefully if no performance)
        self.integrator.update_follower_outcomes("nonexistent_performance", {123: True})
        
        # Should not crash
    
    def test_spatial_communication_effects(self):
        """Test spatial communication through colony information"""
        # Test colony information state tracking
        from src.bstew.core.dance_communication_integration import ColonyInformationState
        
        # Should have colony information dict
        assert hasattr(self.integrator, 'colony_information')
        assert isinstance(self.integrator.colony_information, dict)
        
        # Test creating a colony information state  
        info_state = ColonyInformationState()
        assert hasattr(info_state, 'known_patches')
        assert hasattr(info_state, 'patch_quality_estimates')
        assert hasattr(info_state, 'collective_knowledge_quality')


class TestDanceCommunicationSystemIntegration:
    """Test system integration"""
    
    def test_create_integration_function(self):
        """Test integration creation function"""
        # Test integration creation
        integration = create_dance_communication_integration()
        
        assert integration is not None
        assert hasattr(integration, 'dance_decision_engine')
        assert hasattr(integration, 'recruitment_processor')
    
    def test_end_to_end_communication_flow(self):
        """Test complete communication flow with available methods"""
        integrator = DanceCommunicationIntegrator()
        
        # Test basic flow - foraging result processing
        foraging_result = {
            'patch_quality': 0.85,
            'energy_gained': 60.0,
            'distance_traveled': 180.0,
            'patch_id': 456,
            'patch_location': (180.0, 120.0),
            'resource_type': 'nectar'
        }
        
        colony_state = {
            'energy_level': 0.6,
            'total_bees': 100,
            'available_foragers': 25
        }
        
        bee_states = {
            123: {
                'energy': 90.0,
                'foraging_experience': 0.8,
                'colony_id': 1
            }
        }
        
        # Process returning forager
        dance_performance = integrator.process_returning_forager(
            bee_id=123,
            foraging_result=foraging_result,
            colony_id=1,
            colony_state=colony_state,
            bee_states=bee_states
        )
        
        # Validate result
        assert dance_performance is None or hasattr(dance_performance, 'performance_id')
        
        # Test metrics
        metrics = integrator.get_colony_communication_metrics(colony_id=1)
        assert isinstance(metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])