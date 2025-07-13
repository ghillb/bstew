"""
Dance Communication Integration for NetLogo BEE-STEWARD v2 Parity
================================================================

Advanced integration of dance communication with individual bee agents,
providing realistic waggle dance mechanics, recruitment patterns, and
information flow dynamics within bee colonies.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import logging
import random
import time
from collections import deque

from .bee_communication import (
    DanceType, DanceInformation
)
from .enums import BeeStatus

class DanceDecisionFactor(Enum):
    """Factors influencing dance decisions"""
    RESOURCE_QUALITY = "resource_quality"
    ENERGY_GAIN = "energy_gain"
    DISTANCE_EFFICIENCY = "distance_efficiency"
    PATCH_DEPLETION = "patch_depletion"
    COLONY_NEED = "colony_need"
    SOCIAL_INFLUENCE = "social_influence"
    EXPERIENCE_LEVEL = "experience_level"

class RecruitmentStage(Enum):
    """Stages of recruitment process"""
    DANCE_ATTENTION = "dance_attention"
    INFORMATION_ACQUISITION = "information_acquisition"
    DECISION_MAKING = "decision_making"
    DEPARTURE_PREPARATION = "departure_preparation"
    FLIGHT_INITIATION = "flight_initiation"
    PATCH_DISCOVERY = "patch_discovery"
    SUCCESS_EVALUATION = "success_evaluation"

@dataclass
class IndividualDanceDecision:
    """Individual bee's dance decision process"""
    bee_id: int
    should_dance: bool
    dance_type: DanceType
    dance_intensity: float  # 0.0 to 1.0
    dance_duration: float  # seconds
    decision_factors: Dict[DanceDecisionFactor, float] = field(default_factory=dict)
    confidence_level: float = 0.5
    expected_recruits: int = 0

@dataclass
class DancePerformance:
    """Complete dance performance by a bee"""
    performance_id: str
    dancer_id: int
    dance_info: DanceInformation
    start_time: float
    duration: float
    intensity: float
    audience_size: int = 0
    successful_recruits: int = 0
    total_followers: int = 0
    interruptions: int = 0
    energy_cost: float = 0.0

@dataclass
class FollowerResponse:
    """Follower bee's response to dance"""
    follower_id: int
    dancer_id: int
    dance_id: str
    attention_duration: float
    information_quality: float  # How well they understood
    recruitment_probability: float
    decision_time: float
    follow_through: bool = False
    success_outcome: Optional[bool] = None

@dataclass
class ColonyInformationState:
    """Current information state of the colony"""
    known_patches: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    patch_quality_estimates: Dict[int, float] = field(default_factory=dict)
    patch_distance_estimates: Dict[int, float] = field(default_factory=dict)
    patch_last_reported: Dict[int, float] = field(default_factory=dict)
    information_reliability: Dict[int, float] = field(default_factory=dict)
    collective_knowledge_quality: float = 0.5

class DanceDecisionEngine(BaseModel):
    """Engine for making dance decisions based on multiple factors"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Decision thresholds
    min_dance_quality_threshold: float = 0.3
    min_energy_gain_threshold: float = 10.0
    max_dance_distance: float = 1000.0
    
    # Decision weights
    quality_weight: float = 0.4
    energy_weight: float = 0.3
    distance_weight: float = 0.2
    colony_need_weight: float = 0.1
    
    # Experience modifiers
    experience_modifier_range: float = 0.2
    social_influence_strength: float = 0.15
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def evaluate_dance_decision(self, bee_id: int, foraging_result: Dict[str, Any], 
                              colony_state: Dict[str, Any], 
                              bee_experience: Dict[str, Any]) -> IndividualDanceDecision:
        """Evaluate whether a bee should dance and how"""
        
        # Extract foraging information
        patch_quality = foraging_result.get('patch_quality', 0.5)
        energy_gained = foraging_result.get('energy_gained', 0.0)
        patch_distance = foraging_result.get('distance_traveled', 100.0)
        foraging_result.get('patch_id', 0)
        
        # Calculate decision factors
        decision_factors = self._calculate_decision_factors(
            patch_quality, energy_gained, patch_distance, colony_state, bee_experience
        )
        
        # Determine if bee should dance
        should_dance = self._should_perform_dance(decision_factors)
        
        # Determine dance type and characteristics
        dance_type = self._determine_dance_type(patch_distance, patch_quality)
        dance_intensity = self._calculate_dance_intensity(decision_factors)
        dance_duration = self._calculate_dance_duration(dance_intensity, patch_quality)
        
        # Calculate confidence and expected recruits
        confidence = self._calculate_confidence_level(decision_factors, bee_experience)
        expected_recruits = self._estimate_expected_recruits(dance_intensity, colony_state)
        
        return IndividualDanceDecision(
            bee_id=bee_id,
            should_dance=should_dance,
            dance_type=dance_type,
            dance_intensity=dance_intensity,
            dance_duration=dance_duration,
            decision_factors=decision_factors,
            confidence_level=confidence,
            expected_recruits=expected_recruits
        )
    
    def _calculate_decision_factors(self, patch_quality: float, energy_gained: float,
                                  patch_distance: float, colony_state: Dict[str, Any],
                                  bee_experience: Dict[str, Any]) -> Dict[DanceDecisionFactor, float]:
        """Calculate individual decision factors"""
        
        factors = {}
        
        # Resource quality factor
        factors[DanceDecisionFactor.RESOURCE_QUALITY] = min(1.0, patch_quality)
        
        # Energy gain factor
        normalized_energy = min(1.0, energy_gained / 100.0)
        factors[DanceDecisionFactor.ENERGY_GAIN] = normalized_energy
        
        # Distance efficiency factor
        distance_efficiency = max(0.0, 1.0 - (patch_distance / self.max_dance_distance))
        factors[DanceDecisionFactor.DISTANCE_EFFICIENCY] = distance_efficiency
        
        # Colony need factor
        colony_energy_level = colony_state.get('energy_level', 0.5)
        colony_need = max(0.0, 1.0 - colony_energy_level)
        factors[DanceDecisionFactor.COLONY_NEED] = colony_need
        
        # Experience level factor
        experience_level = bee_experience.get('foraging_experience', 0.5)
        factors[DanceDecisionFactor.EXPERIENCE_LEVEL] = experience_level
        
        # Social influence factor (simplified)
        recent_dances = colony_state.get('recent_dance_count', 0)
        social_influence = min(1.0, recent_dances / 10.0) * self.social_influence_strength
        factors[DanceDecisionFactor.SOCIAL_INFLUENCE] = social_influence
        
        return factors
    
    def _should_perform_dance(self, decision_factors: Dict[DanceDecisionFactor, float]) -> bool:
        """Determine if bee should perform a dance"""
        
        # Calculate weighted decision score
        score = (
            decision_factors.get(DanceDecisionFactor.RESOURCE_QUALITY, 0) * self.quality_weight +
            decision_factors.get(DanceDecisionFactor.ENERGY_GAIN, 0) * self.energy_weight +
            decision_factors.get(DanceDecisionFactor.DISTANCE_EFFICIENCY, 0) * self.distance_weight +
            decision_factors.get(DanceDecisionFactor.COLONY_NEED, 0) * self.colony_need_weight
        )
        
        # Add experience and social modifiers
        experience_modifier = (decision_factors.get(DanceDecisionFactor.EXPERIENCE_LEVEL, 0.5) - 0.5) * self.experience_modifier_range
        social_modifier = decision_factors.get(DanceDecisionFactor.SOCIAL_INFLUENCE, 0)
        
        final_score = score + experience_modifier + social_modifier
        
        # Probabilistic decision with threshold
        dance_probability = max(0.0, min(1.0, final_score))
        
        return random.random() < dance_probability
    
    def _determine_dance_type(self, distance: float, quality: float) -> DanceType:
        """Determine appropriate dance type based on distance and quality"""
        
        if distance < 50.0:  # Close patches
            return DanceType.ROUND_DANCE
        elif quality > 0.8:  # High quality distant patches
            return DanceType.WAGGLE_DANCE
        elif quality > 0.5:  # Moderate quality
            return DanceType.RECRUITMENT_DANCE
        else:
            return DanceType.ROUND_DANCE
    
    def _calculate_dance_intensity(self, decision_factors: Dict[DanceDecisionFactor, float]) -> float:
        """Calculate dance intensity based on decision factors"""
        
        # Base intensity from quality and energy
        base_intensity = (
            decision_factors.get(DanceDecisionFactor.RESOURCE_QUALITY, 0) * 0.5 +
            decision_factors.get(DanceDecisionFactor.ENERGY_GAIN, 0) * 0.3 +
            decision_factors.get(DanceDecisionFactor.COLONY_NEED, 0) * 0.2
        )
        
        # Modify by experience
        experience_factor = decision_factors.get(DanceDecisionFactor.EXPERIENCE_LEVEL, 0.5)
        intensity_modifier = 0.8 + (experience_factor * 0.4)  # 0.8 to 1.2 range
        
        final_intensity = base_intensity * intensity_modifier
        
        return max(0.1, min(1.0, final_intensity))
    
    def _calculate_dance_duration(self, intensity: float, quality: float) -> float:
        """Calculate dance duration in seconds"""
        
        # Base duration between 10-120 seconds
        base_duration = 30.0 + (quality * 60.0)
        
        # Modify by intensity
        duration = base_duration * (0.5 + intensity * 0.5)
        
        # Add some variability
        variability = random.uniform(0.8, 1.2)
        
        return max(10.0, min(120.0, duration * variability))
    
    def _calculate_confidence_level(self, decision_factors: Dict[DanceDecisionFactor, float],
                                  bee_experience: Dict[str, Any]) -> float:
        """Calculate bee's confidence in the dance decision"""
        
        # Base confidence from quality and experience
        quality_confidence = decision_factors.get(DanceDecisionFactor.RESOURCE_QUALITY, 0)
        experience_confidence = bee_experience.get('foraging_experience', 0.5)
        
        # Combine factors
        confidence = (quality_confidence * 0.6 + experience_confidence * 0.4)
        
        # Add some variability
        confidence += random.uniform(-0.1, 0.1)
        
        return max(0.1, min(1.0, confidence))
    
    def _estimate_expected_recruits(self, intensity: float, colony_state: Dict[str, Any]) -> int:
        """Estimate expected number of recruits"""
        
        # Base recruits from intensity
        base_recruits = intensity * 8.0  # 0-8 base recruits
        
        # Modify by colony size
        colony_size = colony_state.get('total_bees', 100)
        size_factor = min(2.0, colony_size / 100.0)
        
        # Modify by available foragers
        available_foragers = colony_state.get('available_foragers', 20)
        availability_factor = min(1.5, available_foragers / 20.0)
        
        expected = base_recruits * size_factor * availability_factor
        
        return max(0, int(expected))

class RecruitmentProcessor(BaseModel):
    """Process recruitment of follower bees through dance communication"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Attention parameters
    attention_probability_base: float = 0.3
    attention_distance_factor: float = 0.1
    attention_duration_range: Tuple[float, float] = (5.0, 30.0)
    
    # Information processing parameters
    information_processing_rate: float = 0.8
    distance_encoding_accuracy: float = 0.9
    direction_encoding_accuracy: float = 0.85
    
    # Decision parameters
    recruitment_threshold: float = 0.6
    social_facilitation_factor: float = 0.2
    individual_variation: float = 0.15
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def process_dance_audience(self, dance_performance: DancePerformance,
                             potential_followers: List[int],
                             colony_state: Dict[str, Any],
                             bee_states: Dict[int, Dict[str, Any]]) -> List[FollowerResponse]:
        """Process recruitment of potential followers"""
        
        responses = []
        
        for follower_id in potential_followers:
            # Check if bee pays attention to dance
            if self._determines_attention(follower_id, dance_performance, bee_states.get(follower_id, {})):
                
                # Process information acquisition
                response = self._process_information_acquisition(
                    follower_id, dance_performance, bee_states.get(follower_id, {}), colony_state
                )
                
                responses.append(response)
        
        return responses
    
    def _determines_attention(self, follower_id: int, dance_performance: DancePerformance,
                            follower_state: Dict[str, Any]) -> bool:
        """Determine if a bee pays attention to a dance"""
        
        # Base attention probability
        attention_prob = self.attention_probability_base
        
        # Modify by bee's current state
        bee_status = follower_state.get('status', BeeStatus.RESTING)
        if bee_status in [BeeStatus.RESTING, BeeStatus.ALIVE]:
            attention_prob += 0.3
        elif bee_status in [BeeStatus.FORAGING, BeeStatus.SEARCHING]:
            attention_prob -= 0.2
        
        # Modify by dance intensity
        attention_prob += dance_performance.intensity * 0.3
        
        # Modify by spatial proximity (simplified)
        attention_prob += self.attention_distance_factor
        
        # Individual variation
        individual_factor = follower_state.get('attention_tendency', 0.5)
        attention_prob += (individual_factor - 0.5) * self.individual_variation
        
        return random.random() < max(0.0, min(1.0, attention_prob))
    
    def _process_information_acquisition(self, follower_id: int, dance_performance: DancePerformance,
                                       follower_state: Dict[str, Any], 
                                       colony_state: Dict[str, Any]) -> FollowerResponse:
        """Process how well a follower acquires information from the dance"""
        
        # Determine attention duration
        base_duration = random.uniform(*self.attention_duration_range)
        intensity_modifier = 0.5 + (dance_performance.intensity * 0.5)
        attention_duration = base_duration * intensity_modifier
        
        # Calculate information quality acquired
        info_quality = self._calculate_information_quality(
            dance_performance, follower_state, attention_duration
        )
        
        # Calculate recruitment probability
        recruitment_prob = self._calculate_recruitment_probability(
            info_quality, dance_performance, follower_state, colony_state
        )
        
        # Decision time (time to decide whether to follow)
        decision_time = self._calculate_decision_time(info_quality, recruitment_prob)
        
        # Determine if bee will follow through
        follow_through = random.random() < recruitment_prob
        
        return FollowerResponse(
            follower_id=follower_id,
            dancer_id=dance_performance.dancer_id,
            dance_id=dance_performance.performance_id,
            attention_duration=attention_duration,
            information_quality=info_quality,
            recruitment_probability=recruitment_prob,
            decision_time=decision_time,
            follow_through=follow_through
        )
    
    def _calculate_information_quality(self, dance_performance: DancePerformance,
                                     follower_state: Dict[str, Any],
                                     attention_duration: float) -> float:
        """Calculate quality of information acquired by follower"""
        
        # Base information quality from dance intensity and duration
        base_quality = dance_performance.intensity * 0.6
        
        # Modify by attention duration (longer attention = better info)
        duration_factor = min(1.0, attention_duration / 20.0)
        base_quality += duration_factor * 0.3
        
        # Modify by follower's experience and learning ability
        experience_level = follower_state.get('dance_following_experience', 0.5)
        learning_ability = follower_state.get('learning_rate', 0.5)
        
        experience_bonus = experience_level * 0.2
        learning_bonus = learning_ability * 0.1
        
        # Add encoding accuracy limitations
        distance_accuracy = self.distance_encoding_accuracy
        direction_accuracy = self.direction_encoding_accuracy
        
        encoding_factor = (distance_accuracy + direction_accuracy) / 2.0
        
        final_quality = (base_quality + experience_bonus + learning_bonus) * encoding_factor
        
        # Add some noise
        noise = random.uniform(-0.05, 0.05)
        
        return max(0.0, min(1.0, final_quality + noise))
    
    def _calculate_recruitment_probability(self, info_quality: float, 
                                         dance_performance: DancePerformance,
                                         follower_state: Dict[str, Any],
                                         colony_state: Dict[str, Any]) -> float:
        """Calculate probability that follower will be recruited"""
        
        # Base probability from information quality
        base_prob = info_quality * 0.7
        
        # Modify by dance intensity
        intensity_bonus = dance_performance.intensity * 0.2
        
        # Modify by follower's motivation state
        motivation = follower_state.get('foraging_motivation', 0.5)
        motivation_bonus = motivation * 0.2
        
        # Social facilitation (if other bees are also following)
        audience_factor = min(1.0, dance_performance.audience_size / 5.0)
        social_bonus = audience_factor * self.social_facilitation_factor
        
        # Colony need factor
        colony_energy = colony_state.get('energy_level', 0.5)
        colony_need = 1.0 - colony_energy
        need_bonus = colony_need * 0.15
        
        # Individual variation
        individual_factor = follower_state.get('recruitment_tendency', 0.5)
        individual_modifier = (individual_factor - 0.5) * self.individual_variation
        
        recruitment_prob = (base_prob + intensity_bonus + motivation_bonus + 
                          social_bonus + need_bonus + individual_modifier)
        
        return max(0.0, min(1.0, recruitment_prob))
    
    def _calculate_decision_time(self, info_quality: float, recruitment_prob: float) -> float:
        """Calculate time for bee to make recruitment decision"""
        
        # Base decision time (10-60 seconds)
        base_time = 35.0
        
        # Faster decision with higher information quality
        quality_factor = 1.0 - (info_quality * 0.4)
        
        # Faster decision with higher recruitment probability
        prob_factor = 1.0 - (recruitment_prob * 0.3)
        
        decision_time = base_time * quality_factor * prob_factor
        
        # Add individual variation
        variability = random.uniform(0.7, 1.3)
        
        return max(5.0, min(90.0, decision_time * variability))

class DanceCommunicationIntegrator(BaseModel):
    """Main integrator for dance communication with bee agents"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Core components
    dance_decision_engine: DanceDecisionEngine
    recruitment_processor: RecruitmentProcessor
    
    # Optional recruitment mechanism manager
    recruitment_manager: Optional[Any] = Field(default=None)
    
    # Active dances and followers
    active_dances: Dict[str, DancePerformance] = Field(default_factory=dict)
    dance_followers: Dict[str, List[FollowerResponse]] = Field(default_factory=dict)
    colony_information: Dict[int, ColonyInformationState] = Field(default_factory=dict)
    
    # Performance tracking
    dance_success_rates: Dict[int, float] = Field(default_factory=dict)  # bee_id -> success rate
    recruitment_history: deque = Field(default_factory=lambda: deque(maxlen=1000))
    
    # Configuration
    max_concurrent_dances: int = 10
    information_decay_rate: float = 0.05
    success_rate_learning_rate: float = 0.1
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        if 'dance_decision_engine' not in data:
            data['dance_decision_engine'] = DanceDecisionEngine()
        if 'recruitment_processor' not in data:
            data['recruitment_processor'] = RecruitmentProcessor()
        
        super().__init__(**data)
    
    def process_returning_forager(self, bee_id: int, foraging_result: Dict[str, Any],
                                colony_id: int, colony_state: Dict[str, Any],
                                bee_states: Dict[int, Dict[str, Any]]) -> Optional[DancePerformance]:
        """Process a returning forager and determine dance behavior"""
        
        # Get bee's experience and state
        bee_experience = bee_states.get(bee_id, {})
        
        # Make dance decision
        dance_decision = self.dance_decision_engine.evaluate_dance_decision(
            bee_id, foraging_result, colony_state, bee_experience
        )
        
        # If bee decides to dance, create dance performance
        if dance_decision.should_dance:
            return self._initiate_dance_performance(
                bee_id, dance_decision, foraging_result, colony_id, colony_state, bee_states
            )
        
        return None
    
    def _initiate_dance_performance(self, bee_id: int, dance_decision: IndividualDanceDecision,
                                  foraging_result: Dict[str, Any], colony_id: int,
                                  colony_state: Dict[str, Any], 
                                  bee_states: Dict[int, Dict[str, Any]]) -> DancePerformance:
        """Initiate a dance performance"""
        
        # Create dance information
        dance_info = DanceInformation(
            dance_id=f"dance_{bee_id}_{int(time.time())}",
            dancer_id=bee_id,
            dance_type=dance_decision.dance_type,
            patch_id=foraging_result.get('patch_id', 0),
            patch_location=foraging_result.get('patch_location', (0.0, 0.0)),
            patch_distance=foraging_result.get('distance_traveled', 100.0),
            patch_direction=foraging_result.get('patch_direction', 0.0),
            resource_type=foraging_result.get('resource_type', 'nectar'),
            resource_quality=foraging_result.get('patch_quality', 0.5),
            resource_quantity=foraging_result.get('resource_quantity', 1.0),
            energy_profitability=foraging_result.get('energy_gained', 0.0),
            dance_duration=dance_decision.dance_duration,
            dance_vigor=dance_decision.dance_intensity,
            waggle_run_count=int(dance_decision.dance_intensity * 20),
            dance_repetitions=int(dance_decision.dance_duration / 10),
            recruitment_threshold=0.6,
            urgency_level=dance_decision.confidence_level,
            timestamp=time.time()
        )
        
        # Create dance performance
        performance = DancePerformance(
            performance_id=dance_info.dance_id,
            dancer_id=bee_id,
            dance_info=dance_info,
            start_time=time.time(),
            duration=dance_decision.dance_duration,
            intensity=dance_decision.dance_intensity
        )
        
        # Find potential audience
        potential_followers = self._find_potential_followers(colony_id, bee_states)
        performance.audience_size = len(potential_followers)
        
        # Process recruitment
        followers = self.recruitment_processor.process_dance_audience(
            performance, potential_followers, colony_state, bee_states
        )
        
        # Update performance with follower information
        performance.total_followers = len(followers)
        performance.successful_recruits = sum(1 for f in followers if f.follow_through)
        
        # Store active dance
        self.active_dances[performance.performance_id] = performance
        self.dance_followers[performance.performance_id] = followers
        
        # Update colony information state
        self._update_colony_information(colony_id, dance_info)
        
        # Process recruitment through advanced mechanism manager
        if self.recruitment_manager:
            recruitment_events = self.recruitment_manager.process_recruitment_event(
                performance, followers, colony_id, bee_states
            )
            self.logger.debug(f"Created {len(recruitment_events)} recruitment events")
        
        # Record recruitment event
        self.recruitment_history.append({
            'timestamp': time.time(),
            'dancer_id': bee_id,
            'colony_id': colony_id,
            'recruits': performance.successful_recruits,
            'intensity': performance.intensity,
            'patch_quality': dance_info.resource_quality
        })
        
        self.logger.info(f"Bee {bee_id} initiated dance {performance.performance_id} "
                        f"with {performance.successful_recruits} recruits")
        
        return performance
    
    def _find_potential_followers(self, colony_id: int, 
                                bee_states: Dict[int, Dict[str, Any]]) -> List[int]:
        """Find bees that could potentially follow a dance"""
        
        potential_followers = []
        
        for bee_id, bee_state in bee_states.items():
            # Check if bee belongs to same colony
            if bee_state.get('colony_id') != colony_id:
                continue
            
            # Check if bee is in appropriate state to follow dances
            bee_status = bee_state.get('status', BeeStatus.RESTING)
            if bee_status in [BeeStatus.RESTING, BeeStatus.ALIVE]:
                potential_followers.append(bee_id)
            elif bee_status == BeeStatus.FORAGING and random.random() < 0.2:
                # Small chance for active foragers to switch
                potential_followers.append(bee_id)
        
        return potential_followers
    
    def _update_colony_information(self, colony_id: int, dance_info: DanceInformation) -> None:
        """Update colony's collective information state"""
        
        if colony_id not in self.colony_information:
            self.colony_information[colony_id] = ColonyInformationState()
        
        info_state = self.colony_information[colony_id]
        patch_id = dance_info.patch_id
        
        # Update patch information
        info_state.known_patches[patch_id] = {
            'location': dance_info.patch_location,
            'resource_type': dance_info.resource_type,
            'last_dancer': dance_info.dancer_id,
            'dance_count': info_state.known_patches.get(patch_id, {}).get('dance_count', 0) + 1
        }
        
        # Update quality estimate (weighted average)
        current_quality = info_state.patch_quality_estimates.get(patch_id, dance_info.resource_quality)
        new_quality = 0.7 * current_quality + 0.3 * dance_info.resource_quality
        info_state.patch_quality_estimates[patch_id] = new_quality
        
        # Update distance estimate
        info_state.patch_distance_estimates[patch_id] = dance_info.patch_distance
        
        # Update last reported time
        info_state.patch_last_reported[patch_id] = dance_info.timestamp
        
        # Update information reliability based on dancer's success rate
        dancer_success_rate = self.dance_success_rates.get(dance_info.dancer_id, 0.5)
        info_state.information_reliability[patch_id] = dancer_success_rate
    
    def update_follower_outcomes(self, performance_id: str, 
                                follower_outcomes: Dict[int, bool]) -> None:
        """Update outcomes of followers who attempted to find the danced patch"""
        
        if performance_id not in self.dance_followers:
            return
        
        followers = self.dance_followers[performance_id]
        
        for follower in followers:
            if follower.follower_id in follower_outcomes:
                follower.success_outcome = follower_outcomes[follower.follower_id]
        
        # Update dancer's success rate
        if performance_id in self.active_dances:
            dancer_id = self.active_dances[performance_id].dancer_id
            success_count = sum(1 for outcome in follower_outcomes.values() if outcome)
            total_attempts = len(follower_outcomes)
            
            if total_attempts > 0:
                success_rate = success_count / total_attempts
                self._update_dancer_success_rate(dancer_id, success_rate)
    
    def _update_dancer_success_rate(self, dancer_id: int, observed_success_rate: float) -> None:
        """Update dancer's historical success rate"""
        
        current_rate = self.dance_success_rates.get(dancer_id, 0.5)
        
        # Exponential moving average
        new_rate = ((1 - self.success_rate_learning_rate) * current_rate + 
                   self.success_rate_learning_rate * observed_success_rate)
        
        self.dance_success_rates[dancer_id] = new_rate
    
    def get_recruited_bees(self, colony_id: int) -> List[Dict[str, Any]]:
        """Get list of bees that have been recruited and their target information"""
        
        recruited_bees = []
        
        for performance_id, followers in self.dance_followers.items():
            if performance_id not in self.active_dances:
                continue
            
            performance = self.active_dances[performance_id]
            
            for follower in followers:
                if follower.follow_through:
                    recruited_bees.append({
                        'bee_id': follower.follower_id,
                        'target_patch_id': performance.dance_info.patch_id,
                        'target_location': performance.dance_info.patch_location,
                        'target_distance': performance.dance_info.patch_distance,
                        'information_quality': follower.information_quality,
                        'recruitment_time': time.time(),
                        'dancer_id': follower.dancer_id
                    })
        
        return recruited_bees
    
    def cleanup_finished_dances(self, current_time: float) -> None:
        """Clean up finished dance performances"""
        
        finished_dances = []
        
        for performance_id, performance in self.active_dances.items():
            if current_time > performance.start_time + performance.duration:
                finished_dances.append(performance_id)
        
        for performance_id in finished_dances:
            del self.active_dances[performance_id]
            if performance_id in self.dance_followers:
                del self.dance_followers[performance_id]
    
    def get_colony_communication_metrics(self, colony_id: int) -> Dict[str, Any]:
        """Get communication metrics for a colony"""
        
        metrics = {
            'active_dances': len([p for p in self.active_dances.values() 
                                if p.dance_info.dancer_id in self._get_colony_bees(colony_id)]),
            'total_recruited_bees': len(self.get_recruited_bees(colony_id)),
            'known_patches': len(self.colony_information.get(colony_id, ColonyInformationState()).known_patches),
            'average_dance_success_rate': 0.0,
            'information_quality': 0.0
        }
        
        # Calculate average success rate for colony dancers
        colony_bees = self._get_colony_bees(colony_id)
        success_rates = [self.dance_success_rates.get(bee_id, 0.5) for bee_id in colony_bees 
                        if bee_id in self.dance_success_rates]
        
        if success_rates:
            metrics['average_dance_success_rate'] = sum(success_rates) / len(success_rates)
        
        # Calculate information quality
        if colony_id in self.colony_information:
            info_state = self.colony_information[colony_id]
            if info_state.information_reliability:
                avg_reliability = sum(info_state.information_reliability.values()) / len(info_state.information_reliability)
                metrics['information_quality'] = avg_reliability
        
        # Add recruitment mechanism metrics if available
        if self.recruitment_manager:
            recruitment_metrics = self.recruitment_manager.get_colony_recruitment_metrics(colony_id)
            metrics.update({
                'recruitment_' + k: v for k, v in recruitment_metrics.items()
            })
        
        return metrics
    
    def _get_colony_bees(self, colony_id: int) -> List[int]:
        """Get list of bee IDs belonging to a colony (simplified)"""
        # This would normally query the actual colony structure
        # For now, return empty list as placeholder
        return []


def create_dance_communication_integration(include_recruitment_manager: bool = True) -> DanceCommunicationIntegrator:
    """Factory function to create dance communication integrator"""
    
    integrator_data = {}
    
    if include_recruitment_manager:
        try:
            from .recruitment_mechanisms import create_recruitment_mechanism_manager
            integrator_data['recruitment_manager'] = create_recruitment_mechanism_manager()
        except ImportError:
            # Recruitment manager not available, continue without it
            pass
    
    return DanceCommunicationIntegrator(**integrator_data)