"""
Recruitment Mechanisms and Information Flow for NetLogo BEE-STEWARD v2 Parity
============================================================================

Advanced recruitment system implementing realistic bee recruitment dynamics,
information flow patterns, and social learning mechanisms within bee colonies.
"""

from typing import Dict, List, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import logging
import math
import random
from collections import deque
import time

from .bee_communication import (
    DanceDecisionModel, RecruitmentModel, InformationSharingModel,
    DanceInformation
)
from .dance_communication_integration import (
    FollowerResponse, DancePerformance
)

class RecruitmentPhase(Enum):
    """Phases of recruitment process"""
    DISCOVERY = "discovery"
    ATTENTION = "attention"
    FOLLOWING = "following"
    EXPLORATION = "exploration"
    ASSESSMENT = "assessment"
    COMMITMENT = "commitment"
    RETURN = "return"

class InformationFlowType(Enum):
    """Types of information flow"""
    DANCE_COMMUNICATION = "dance_communication"
    DIRECT_OBSERVATION = "direct_observation"
    SOCIAL_LEARNING = "social_learning"
    EXPERIENCE_SHARING = "experience_sharing"
    COLLECTIVE_MEMORY = "collective_memory"

@dataclass
class RecruitmentEvent:
    """Individual recruitment event record"""
    event_id: str
    timestamp: float
    recruiter_id: int
    recruit_id: int
    patch_id: int
    recruitment_phase: RecruitmentPhase
    success: bool = False
    information_quality: float = 0.5
    distance_accuracy: float = 0.8
    direction_accuracy: float = 0.8
    resource_assessment: float = 0.5
    follow_through_probability: float = 0.5
    
    def calculate_recruitment_success_score(self) -> float:
        """Calculate overall recruitment success score"""
        return (self.information_quality * 0.4 + 
                self.distance_accuracy * 0.2 + 
                self.direction_accuracy * 0.2 + 
                self.resource_assessment * 0.2)

@dataclass
class InformationFlow:
    """Information flow between bees"""
    flow_id: str
    source_bee: int
    target_bee: int
    information_type: InformationFlowType
    patch_id: int
    information_content: Dict[str, Any] = field(default_factory=dict)
    transmission_accuracy: float = 0.8
    received_timestamp: float = 0.0
    processing_time: float = 0.0
    behavioral_impact: float = 0.0
    
    def get_information_degradation(self, time_elapsed: float, decay_rate: float = 0.02) -> float:
        """Calculate information degradation over time"""
        return max(0.1, self.transmission_accuracy * math.exp(-decay_rate * time_elapsed))

@dataclass
class ColonyInformationNetwork:
    """Colony-wide information network state"""
    colony_id: int
    information_flows: List[InformationFlow] = field(default_factory=list)
    active_recruitments: Dict[str, RecruitmentEvent] = field(default_factory=dict)
    collective_knowledge: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    information_reliability: Dict[int, float] = field(default_factory=dict)
    social_network: Dict[int, Set[int]] = field(default_factory=dict)
    learning_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update_collective_knowledge(self, patch_id: int, new_info: Dict[str, Any], 
                                   source_reliability: float) -> None:
        """Update colony's collective knowledge about a patch"""
        if patch_id not in self.collective_knowledge:
            self.collective_knowledge[patch_id] = {}
        
        current_info = self.collective_knowledge[patch_id]
        
        # Weighted update based on source reliability
        for key, value in new_info.items():
            if key in current_info:
                # Weighted average with reliability as weight
                current_weight = self.information_reliability.get(patch_id, 0.5)
                new_weight = source_reliability
                total_weight = current_weight + new_weight
                
                if isinstance(value, (int, float)):
                    updated_value = ((current_info[key] * current_weight + 
                                    value * new_weight) / total_weight)
                    current_info[key] = updated_value
                else:
                    current_info[key] = value  # Replace non-numeric values
            else:
                current_info[key] = value
        
        # Update reliability
        if patch_id in self.information_reliability:
            self.information_reliability[patch_id] = min(1.0, 
                self.information_reliability[patch_id] * 0.9 + source_reliability * 0.1)
        else:
            self.information_reliability[patch_id] = source_reliability

class RecruitmentMechanismManager(BaseModel):
    """Manages recruitment mechanisms and information flow"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Core models
    dance_decision_model: DanceDecisionModel
    recruitment_model: RecruitmentModel
    information_sharing_model: InformationSharingModel
    
    # Colony information networks
    colony_networks: Dict[int, ColonyInformationNetwork] = Field(default_factory=dict)
    
    # Active recruitment tracking
    active_recruitments: Dict[str, RecruitmentEvent] = Field(default_factory=dict)
    recruitment_history: deque = Field(default_factory=lambda: deque(maxlen=5000))
    
    # Configuration
    max_recruitment_distance: float = 150.0
    information_decay_rate: float = 0.02
    social_learning_rate: float = 0.1
    network_update_interval: int = 100
    
    # Performance metrics
    recruitment_success_rates: Dict[int, float] = Field(default_factory=dict)
    information_flow_efficiency: Dict[int, float] = Field(default_factory=dict)
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        if 'dance_decision_model' not in data:
            data['dance_decision_model'] = DanceDecisionModel()
        if 'recruitment_model' not in data:
            data['recruitment_model'] = RecruitmentModel()
        if 'information_sharing_model' not in data:
            data['information_sharing_model'] = InformationSharingModel()
        
        super().__init__(**data)
    
    def initialize_colony_network(self, colony_id: int, bee_ids: List[int]) -> None:
        """Initialize information network for a colony"""
        
        network = ColonyInformationNetwork(colony_id=colony_id)
        
        # Initialize social network (who communicates with whom)
        for bee_id in bee_ids:
            # Each bee can potentially communicate with 3-8 other bees
            network_size = random.randint(3, 8)
            potential_connections = [b for b in bee_ids if b != bee_id]
            
            if len(potential_connections) >= network_size:
                connections = set(random.sample(potential_connections, network_size))
            else:
                connections = set(potential_connections)
            
            network.social_network[bee_id] = connections
        
        self.colony_networks[colony_id] = network
        self.logger.info(f"Initialized information network for colony {colony_id} with {len(bee_ids)} bees")
    
    def process_recruitment_event(self, dance_performance: DancePerformance,
                                follower_responses: List[FollowerResponse],
                                colony_id: int, bee_states: Dict[int, Dict[str, Any]]) -> List[RecruitmentEvent]:
        """Process recruitment events from dance performance"""
        
        recruitment_events = []
        current_time = time.time()
        
        for response in follower_responses:
            if response.follow_through:
                # Create recruitment event
                event = RecruitmentEvent(
                    event_id=f"recruit_{response.follower_id}_{int(current_time)}",
                    timestamp=current_time,
                    recruiter_id=response.dancer_id,
                    recruit_id=response.follower_id,
                    patch_id=dance_performance.dance_info.patch_id,
                    recruitment_phase=RecruitmentPhase.FOLLOWING,
                    information_quality=response.information_quality,
                    distance_accuracy=self._calculate_distance_accuracy(response),
                    direction_accuracy=self._calculate_direction_accuracy(response),
                    resource_assessment=dance_performance.dance_info.resource_quality,
                    follow_through_probability=response.recruitment_probability
                )
                
                # Add to active recruitments
                self.active_recruitments[event.event_id] = event
                recruitment_events.append(event)
                
                # Create information flow
                self._create_information_flow(
                    response.dancer_id, response.follower_id, 
                    dance_performance.dance_info, colony_id, InformationFlowType.DANCE_COMMUNICATION
                )
        
        # Update recruitment success rates
        self._update_recruitment_success_rates(dance_performance.dancer_id, len(recruitment_events))
        
        return recruitment_events
    
    def _calculate_distance_accuracy(self, response: FollowerResponse) -> float:
        """Calculate distance encoding accuracy for follower"""
        base_accuracy = 0.85
        quality_factor = response.information_quality * 0.15
        attention_factor = min(0.1, response.attention_duration / 30.0) * 0.1
        
        return min(1.0, base_accuracy + quality_factor + attention_factor)
    
    def _calculate_direction_accuracy(self, response: FollowerResponse) -> float:
        """Calculate direction encoding accuracy for follower"""
        base_accuracy = 0.80
        quality_factor = response.information_quality * 0.2
        
        return min(1.0, base_accuracy + quality_factor)
    
    def _create_information_flow(self, source_bee: int, target_bee: int,
                               dance_info: DanceInformation, colony_id: int,
                               flow_type: InformationFlowType) -> None:
        """Create information flow between bees"""
        
        if colony_id not in self.colony_networks:
            return
        
        network = self.colony_networks[colony_id]
        
        # Create information content
        information_content = {
            'patch_location': dance_info.patch_location,
            'patch_distance': dance_info.patch_distance,
            'patch_direction': dance_info.patch_direction,
            'resource_quality': dance_info.resource_quality,
            'resource_type': dance_info.resource_type,
            'energy_profitability': dance_info.energy_profitability,
            'dance_vigor': dance_info.dance_vigor,
            'transmission_time': time.time()
        }
        
        # Calculate transmission accuracy
        transmission_accuracy = self._calculate_transmission_accuracy(
            source_bee, target_bee, flow_type, dance_info
        )
        
        # Create flow
        flow = InformationFlow(
            flow_id=f"flow_{source_bee}_{target_bee}_{int(time.time())}",
            source_bee=source_bee,
            target_bee=target_bee,
            information_type=flow_type,
            patch_id=dance_info.patch_id,
            information_content=information_content,
            transmission_accuracy=transmission_accuracy,
            received_timestamp=time.time()
        )
        
        network.information_flows.append(flow)
        
        # Update social network connections
        if source_bee not in network.social_network:
            network.social_network[source_bee] = set()
        network.social_network[source_bee].add(target_bee)
        
        if target_bee not in network.social_network:
            network.social_network[target_bee] = set()
        network.social_network[target_bee].add(source_bee)
    
    def _calculate_transmission_accuracy(self, source_bee: int, target_bee: int,
                                       flow_type: InformationFlowType,
                                       dance_info: DanceInformation) -> float:
        """Calculate information transmission accuracy"""
        
        base_accuracy = 0.8
        
        # Flow type modifiers
        type_modifiers = {
            InformationFlowType.DANCE_COMMUNICATION: 0.0,
            InformationFlowType.DIRECT_OBSERVATION: 0.1,
            InformationFlowType.SOCIAL_LEARNING: -0.1,
            InformationFlowType.EXPERIENCE_SHARING: 0.05,
            InformationFlowType.COLLECTIVE_MEMORY: -0.05
        }
        
        accuracy = base_accuracy + type_modifiers.get(flow_type, 0.0)
        
        # Dance quality influence
        if flow_type == InformationFlowType.DANCE_COMMUNICATION:
            dance_quality_bonus = dance_info.dance_vigor * 0.1
            accuracy += dance_quality_bonus
        
        # Source reliability
        source_reliability = self.recruitment_success_rates.get(source_bee, 0.5)
        reliability_bonus = (source_reliability - 0.5) * 0.1
        accuracy += reliability_bonus
        
        return max(0.3, min(1.0, accuracy))
    
    def update_recruitment_outcomes(self, recruitment_events: List[RecruitmentEvent],
                                  foraging_outcomes: Dict[int, Dict[str, Any]]) -> None:
        """Update recruitment events with foraging outcomes"""
        
        for event in recruitment_events:
            if event.recruit_id in foraging_outcomes:
                outcome = foraging_outcomes[event.recruit_id]
                
                # Update event success
                event.success = outcome.get('found_patch', False)
                
                if event.success:
                    # Update assessment accuracy
                    actual_quality = outcome.get('actual_patch_quality', 0.5)
                    expected_quality = event.resource_assessment
                    
                    assessment_error = abs(actual_quality - expected_quality)
                    event.resource_assessment = 1.0 - assessment_error
                    
                    # Update colony collective knowledge
                    self._update_colony_knowledge_from_recruitment(event, outcome)
                
                # Move to completion phase
                event.recruitment_phase = RecruitmentPhase.RETURN
                
                # Add to history
                self.recruitment_history.append(event)
        
        # Clean up completed events
        completed_events = [e.event_id for e in recruitment_events if e.recruitment_phase == RecruitmentPhase.RETURN]
        for event_id in completed_events:
            if event_id in self.active_recruitments:
                del self.active_recruitments[event_id]
    
    def _update_colony_knowledge_from_recruitment(self, event: RecruitmentEvent,
                                                outcome: Dict[str, Any]) -> None:
        """Update colony knowledge based on recruitment outcome"""
        
        # Find colony network
        colony_network = None
        for network in self.colony_networks.values():
            if event.recruit_id in network.social_network:
                colony_network = network
                break
        
        if not colony_network:
            return
        
        # Update collective knowledge
        new_info = {
            'quality': outcome.get('actual_patch_quality', 0.5),
            'resource_density': outcome.get('resource_density', 0.5),
            'accessibility': outcome.get('accessibility', 1.0),
            'last_visited': time.time(),
            'visitor_count': outcome.get('visitor_count', 1),
            'success_rate': 1.0 if event.success else 0.0
        }
        
        # Calculate source reliability
        recruiter_reliability = self.recruitment_success_rates.get(event.recruiter_id, 0.5)
        
        colony_network.update_collective_knowledge(
            event.patch_id, new_info, recruiter_reliability
        )
        
        # Record learning event
        learning_event = {
            'timestamp': time.time(),
            'learner_id': event.recruit_id,
            'teacher_id': event.recruiter_id,
            'patch_id': event.patch_id,
            'success': event.success,
            'information_quality': event.information_quality
        }
        
        colony_network.learning_history.append(learning_event)
    
    def _update_recruitment_success_rates(self, recruiter_id: int, recruitment_count: int) -> None:
        """Update recruitment success rates for recruiters"""
        
        current_rate = self.recruitment_success_rates.get(recruiter_id, 0.5)
        
        # Simple success rate update (would be more sophisticated in reality)
        if recruitment_count > 0:
            # Positive feedback for successful recruitment
            new_rate = current_rate * 0.9 + 0.1 * (recruitment_count / 5.0)
        else:
            # Slight decrease for no recruitment
            new_rate = current_rate * 0.95
        
        self.recruitment_success_rates[recruiter_id] = max(0.1, min(1.0, new_rate))
    
    def propagate_information_through_network(self, colony_id: int, timestep: int) -> None:
        """Propagate information through social network"""
        
        if colony_id not in self.colony_networks:
            return
        
        network = self.colony_networks[colony_id]
        
        # Process recent information flows
        recent_flows = [f for f in network.information_flows 
                       if time.time() - f.received_timestamp < 3600]  # Last hour
        
        for flow in recent_flows:
            # Check for secondary propagation
            if flow.target_bee in network.social_network:
                connections = network.social_network[flow.target_bee]
                
                # Propagate to connected bees with degraded accuracy
                for connected_bee in connections:
                    if random.random() < 0.3:  # 30% chance of sharing
                        degraded_accuracy = flow.transmission_accuracy * 0.8
                        
                        if degraded_accuracy > 0.4:  # Minimum threshold for useful information
                            # Create secondary flow
                            secondary_flow = InformationFlow(
                                flow_id=f"secondary_{flow.target_bee}_{connected_bee}_{int(time.time())}",
                                source_bee=flow.target_bee,
                                target_bee=connected_bee,
                                information_type=InformationFlowType.SOCIAL_LEARNING,
                                patch_id=flow.patch_id,
                                information_content=flow.information_content.copy(),
                                transmission_accuracy=degraded_accuracy,
                                received_timestamp=time.time()
                            )
                            
                            network.information_flows.append(secondary_flow)
        
        # Update information flow efficiency
        if recent_flows:
            avg_accuracy = sum(f.transmission_accuracy for f in recent_flows) / len(recent_flows)
            self.information_flow_efficiency[colony_id] = avg_accuracy
    
    def get_colony_recruitment_metrics(self, colony_id: int) -> Dict[str, Any]:
        """Get recruitment metrics for a colony"""
        
        if colony_id not in self.colony_networks:
            return {}
        
        network = self.colony_networks[colony_id]
        
        # Calculate metrics from recent history
        recent_history = [event for event in self.recruitment_history 
                         if time.time() - event.timestamp < 86400]  # Last 24 hours
        
        colony_events = [e for e in recent_history 
                        if e.recruiter_id in network.social_network or 
                           e.recruit_id in network.social_network]
        
        metrics = {
            'total_recruitment_events': len(colony_events),
            'successful_recruitments': sum(1 for e in colony_events if e.success),
            'recruitment_success_rate': 0.0,
            'average_information_quality': 0.0,
            'information_flow_count': len(network.information_flows),
            'known_patches': len(network.collective_knowledge),
            'network_connectivity': 0.0,
            'information_reliability': 0.0
        }
        
        if colony_events:
            metrics['recruitment_success_rate'] = metrics['successful_recruitments'] / len(colony_events)
            metrics['average_information_quality'] = sum(e.information_quality for e in colony_events) / len(colony_events)
        
        # Network connectivity
        if network.social_network:
            total_connections = sum(len(connections) for connections in network.social_network.values())
            max_possible = len(network.social_network) * (len(network.social_network) - 1)
            if max_possible > 0:
                metrics['network_connectivity'] = total_connections / max_possible
        
        # Information reliability
        if network.information_reliability:
            metrics['information_reliability'] = sum(network.information_reliability.values()) / len(network.information_reliability)
        
        return metrics
    
    def get_bee_recruitment_performance(self, bee_id: int) -> Dict[str, Any]:
        """Get recruitment performance metrics for individual bee"""
        
        # Get recent recruitment events for this bee
        recent_events = [event for event in self.recruitment_history 
                        if (event.recruiter_id == bee_id or event.recruit_id == bee_id) and 
                           time.time() - event.timestamp < 86400]
        
        recruiter_events = [e for e in recent_events if e.recruiter_id == bee_id]
        recruit_events = [e for e in recent_events if e.recruit_id == bee_id]
        
        performance = {
            'as_recruiter': {
                'recruitment_attempts': len(recruiter_events),
                'successful_recruitments': sum(1 for e in recruiter_events if e.success),
                'success_rate': 0.0,
                'average_information_quality': 0.0
            },
            'as_recruit': {
                'recruitment_responses': len(recruit_events),
                'successful_follows': sum(1 for e in recruit_events if e.success),
                'follow_success_rate': 0.0,
                'average_assessment_accuracy': 0.0
            },
            'overall_reputation': self.recruitment_success_rates.get(bee_id, 0.5)
        }
        
        # Calculate recruiter metrics
        if recruiter_events:
            performance['as_recruiter']['success_rate'] = performance['as_recruiter']['successful_recruitments'] / len(recruiter_events)
            performance['as_recruiter']['average_information_quality'] = sum(e.information_quality for e in recruiter_events) / len(recruiter_events)
        
        # Calculate recruit metrics
        if recruit_events:
            performance['as_recruit']['follow_success_rate'] = performance['as_recruit']['successful_follows'] / len(recruit_events)
            performance['as_recruit']['average_assessment_accuracy'] = sum(e.resource_assessment for e in recruit_events) / len(recruit_events)
        
        return performance

def create_recruitment_mechanism_manager() -> RecruitmentMechanismManager:
    """Factory function to create recruitment mechanism manager"""
    
    return RecruitmentMechanismManager()