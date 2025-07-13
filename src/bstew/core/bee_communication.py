"""
Bee Communication System for NetLogo BEE-STEWARD v2 Parity
==========================================================

Advanced communication system linking foraging success to dance probability,
patch information sharing, recruitment success tracking, and dance-based
patch discovery.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import random
import time
from collections import defaultdict

from .resource_collection import ResourceType


class DanceType(Enum):
    """Types of bee dances"""
    WAGGLE_DANCE = "waggle_dance"
    ROUND_DANCE = "round_dance"
    TREMBLE_DANCE = "tremble_dance"
    STOP_SIGNAL = "stop_signal"
    RECRUITMENT_DANCE = "recruitment_dance"


class CommunicationEvent(Enum):
    """Types of communication events"""
    DANCE_PERFORMANCE = "dance_performance"
    DANCE_FOLLOWING = "dance_following"
    INFORMATION_SHARING = "information_sharing"
    RECRUITMENT_SUCCESS = "recruitment_success"
    PATCH_DISCOVERY = "patch_discovery"
    SOCIAL_LEARNING = "social_learning"


@dataclass
class DanceInformation:
    """Information encoded in a dance"""
    dance_id: str
    dancer_id: int
    dance_type: DanceType
    
    # Patch information
    patch_id: int
    patch_location: Tuple[float, float]
    patch_distance: float
    patch_direction: float  # Radians from hive
    
    # Resource information
    resource_type: ResourceType
    resource_quality: float
    resource_quantity: float
    energy_profitability: float
    
    # Dance characteristics
    dance_duration: float
    dance_vigor: float
    waggle_run_count: int
    dance_repetitions: int
    
    # Recruitment information
    recruitment_threshold: float
    urgency_level: float
    
    # Metadata
    timestamp: float
    success_rate: float = 0.0
    followers: Set[int] = field(default_factory=set)
    
    def calculate_dance_vigor(self) -> float:
        """Calculate dance vigor based on resource quality and profitability"""
        base_vigor = self.resource_quality * 0.5
        profitability_bonus = self.energy_profitability * 0.3
        urgency_bonus = self.urgency_level * 0.2
        
        return min(1.0, base_vigor + profitability_bonus + urgency_bonus)
    
    def should_recruit(self) -> bool:
        """Determine if dance should recruit followers"""
        return (self.resource_quality > self.recruitment_threshold and 
                self.energy_profitability > 0.5 and
                self.dance_vigor > 0.6)


@dataclass
class CommunicationRecord:
    """Record of communication event"""
    event_id: str
    event_type: CommunicationEvent
    timestamp: float
    
    # Participants
    sender_id: int
    receiver_id: Optional[int] = None
    audience_ids: Set[int] = field(default_factory=set)
    
    # Content
    information_content: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    
    # Effectiveness
    information_accuracy: float = 0.0
    transmission_success: float = 0.0
    behavioral_change: bool = False


class DanceDecisionModel(BaseModel):
    """Model for dance decision making"""
    
    model_config = {"validate_assignment": True}
    
    # Dance thresholds
    min_quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum quality to dance")
    min_profitability_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="Minimum profitability to dance")
    distance_factor: float = Field(default=0.8, ge=0.0, le=1.0, description="Distance impact on dance probability")
    
    # Dance intensity parameters
    vigor_quality_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality weight in vigor calculation")
    vigor_profitability_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Profitability weight in vigor")
    vigor_urgency_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Urgency weight in vigor")
    
    # Dance duration parameters
    base_duration: float = Field(default=30.0, ge=0.0, description="Base dance duration (seconds)")
    quality_duration_multiplier: float = Field(default=2.0, ge=1.0, description="Quality impact on duration")
    
    def should_dance(self, patch_quality: float, energy_profitability: float, 
                    distance: float, individual_threshold: float) -> bool:
        """Determine if bee should dance for a patch"""
        
        # Basic quality check
        if patch_quality < self.min_quality_threshold:
            return False
        
        # Profitability check
        if energy_profitability < self.min_profitability_threshold:
            return False
        
        # Distance adjustment
        distance_adjusted_quality = patch_quality * (1.0 - (distance / 2000.0) * (1.0 - self.distance_factor))
        
        # Individual threshold (bee-specific)
        if distance_adjusted_quality < individual_threshold:
            return False
        
        return True
    
    def calculate_dance_probability(self, patch_quality: float, energy_profitability: float,
                                  distance: float, recent_success_rate: float) -> float:
        """Calculate probability of dancing"""
        
        # Base probability from quality
        base_prob = patch_quality * 0.6
        
        # Profitability influence
        profitability_bonus = energy_profitability * 0.3
        
        # Distance penalty
        distance_penalty = min(0.3, distance / 1000.0 * 0.1)
        
        # Recent success influence
        success_bonus = recent_success_rate * 0.1
        
        probability = base_prob + profitability_bonus - distance_penalty + success_bonus
        
        return max(0.0, min(1.0, probability))
    
    def calculate_dance_duration(self, patch_quality: float, energy_profitability: float) -> float:
        """Calculate dance duration based on patch characteristics"""
        
        quality_factor = patch_quality * self.quality_duration_multiplier
        profitability_factor = energy_profitability * 1.5
        
        duration = self.base_duration * (quality_factor + profitability_factor) / 2.0
        
        return max(10.0, min(300.0, duration))  # 10 seconds to 5 minutes


class RecruitmentModel(BaseModel):
    """Model for recruitment success and following behavior"""
    
    model_config = {"validate_assignment": True}
    
    # Following behavior parameters
    base_following_probability: float = Field(default=0.3, ge=0.0, le=1.0, description="Base probability of following")
    dance_vigor_influence: float = Field(default=0.4, ge=0.0, le=1.0, description="Dance vigor influence")
    dancer_reputation_influence: float = Field(default=0.2, ge=0.0, le=1.0, description="Dancer reputation influence")
    
    # Recruitment success parameters
    information_accuracy_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Information accuracy threshold")
    recruitment_success_rate: float = Field(default=0.6, ge=0.0, le=1.0, description="Base recruitment success rate")
    
    # Social learning parameters
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Social learning rate")
    experience_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Personal experience weight")
    
    def calculate_following_probability(self, dance_vigor: float, dancer_reputation: float,
                                      follower_experience: float, colony_need: float) -> float:
        """Calculate probability of following a dance"""
        
        # Base probability
        base_prob = self.base_following_probability
        
        # Dance vigor influence
        vigor_bonus = dance_vigor * self.dance_vigor_influence
        
        # Dancer reputation influence
        reputation_bonus = dancer_reputation * self.dancer_reputation_influence
        
        # Follower experience (less experienced bees follow more)
        experience_factor = max(0.1, 1.0 - follower_experience) * 0.2
        
        # Colony need factor
        need_factor = colony_need * 0.3
        
        probability = base_prob + vigor_bonus + reputation_bonus + experience_factor + need_factor
        
        return max(0.0, min(1.0, probability))
    
    def calculate_recruitment_success(self, dance_info: DanceInformation, 
                                    follower_characteristics: Dict[str, float]) -> float:
        """Calculate recruitment success probability"""
        
        # Information accuracy
        accuracy_factor = dance_info.success_rate * 0.4
        
        # Dance quality
        dance_quality = dance_info.dance_vigor * 0.3
        
        # Follower capability
        follower_capability = follower_characteristics.get('foraging_efficiency', 0.5) * 0.2
        
        # Resource quality
        resource_factor = dance_info.resource_quality * 0.1
        
        success_prob = accuracy_factor + dance_quality + follower_capability + resource_factor
        
        return max(0.0, min(1.0, success_prob))
    
    def update_bee_reputation(self, bee_id: int, recruitment_success: bool,
                            current_reputation: float) -> float:
        """Update bee reputation based on recruitment success"""
        
        if recruitment_success:
            reputation_change = 0.1 * (1.0 - current_reputation)  # Diminishing returns
        else:
            reputation_change = -0.05 * current_reputation  # Proportional decline
        
        new_reputation = current_reputation + reputation_change
        
        return max(0.0, min(1.0, new_reputation))


class InformationSharingModel(BaseModel):
    """Model for information sharing and patch discovery"""
    
    model_config = {"validate_assignment": True}
    
    # Information sharing parameters
    information_decay_rate: float = Field(default=0.95, ge=0.0, le=1.0, description="Information decay per time step")
    sharing_range: float = Field(default=10.0, ge=0.0, description="Information sharing range (meters)")
    
    # Patch discovery parameters
    discovery_bonus: float = Field(default=0.3, ge=0.0, le=1.0, description="Bonus for patch discovery")
    novelty_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for novel information")
    
    # Information quality parameters
    accuracy_improvement_rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Information accuracy improvement")
    
    def calculate_information_value(self, patch_info: Dict[str, Any], 
                                  colony_knowledge: Dict[int, Dict[str, Any]]) -> float:
        """Calculate value of shared information"""
        
        patch_id = patch_info.get('patch_id', 0)
        
        # Novelty value
        if patch_id not in colony_knowledge:
            novelty_value = self.novelty_weight
        else:
            # Information update value
            known_info = colony_knowledge[patch_id]
            quality_difference = abs(patch_info.get('quality', 0.5) - known_info.get('quality', 0.5))
            novelty_value = quality_difference * 0.2
        
        # Resource value
        resource_value = patch_info.get('quality', 0.5) * 0.4
        
        # Distance value (closer patches are more valuable)
        distance = patch_info.get('distance', 1000.0)
        distance_value = max(0.1, 1.0 - distance / 2000.0) * 0.3
        
        # Discovery bonus
        discovery_value = self.discovery_bonus if patch_id not in colony_knowledge else 0.0
        
        return novelty_value + resource_value + distance_value + discovery_value
    
    def share_patch_information(self, sender_id: int, receiver_ids: List[int],
                              patch_info: Dict[str, Any]) -> List[CommunicationRecord]:
        """Share patch information between bees"""
        
        records = []
        
        for receiver_id in receiver_ids:
            # Calculate information transmission success
            transmission_success = self._calculate_transmission_success(sender_id, receiver_id, patch_info)
            
            # Create communication record
            record = CommunicationRecord(
                event_id=f"info_share_{sender_id}_{receiver_id}_{time.time()}",
                event_type=CommunicationEvent.INFORMATION_SHARING,
                timestamp=time.time(),
                sender_id=sender_id,
                receiver_id=receiver_id,
                information_content=patch_info.copy(),
                success=transmission_success > 0.5,
                transmission_success=transmission_success
            )
            
            records.append(record)
        
        return records
    
    def _calculate_transmission_success(self, sender_id: int, receiver_id: int,
                                      patch_info: Dict[str, Any]) -> float:
        """Calculate success probability of information transmission"""
        
        # Base success rate
        base_success = 0.7
        
        # Information complexity (simpler information transmits better)
        complexity_factor = 1.0 / (1.0 + len(patch_info) * 0.1)
        
        # Information quality (higher quality information is shared more accurately)
        quality_factor = patch_info.get('quality', 0.5) * 0.3
        
        # Distance factor (closer sharing is more successful)
        distance_factor = 0.9  # Simplified - assume close proximity
        
        success_prob = base_success * complexity_factor + quality_factor * distance_factor
        
        return max(0.0, min(1.0, success_prob))


class BeeCommunicationSystem(BaseModel):
    """Comprehensive bee communication system"""
    
    model_config = {"validate_assignment": True}
    
    # Component models
    dance_model: DanceDecisionModel = Field(default_factory=DanceDecisionModel)
    recruitment_model: RecruitmentModel = Field(default_factory=RecruitmentModel)
    information_model: InformationSharingModel = Field(default_factory=InformationSharingModel)
    
    # System state
    active_dances: Dict[str, DanceInformation] = Field(default_factory=dict, description="Active dances")
    communication_records: List[CommunicationRecord] = Field(default_factory=list, description="Communication history")
    bee_reputations: Dict[int, float] = Field(default_factory=dict, description="Bee reputations")
    colony_knowledge: Dict[int, Dict[str, Any]] = Field(default_factory=dict, description="Colony patch knowledge")
    
    # Performance tracking
    recruitment_success_rates: Dict[int, float] = Field(default_factory=dict, description="Recruitment success by bee")
    information_sharing_success: Dict[int, float] = Field(default_factory=dict, description="Information sharing success")
    
    def evaluate_dance_decision(self, bee_id: int, patch_info: Dict[str, Any],
                               foraging_success: Dict[str, Any]) -> bool:
        """Evaluate whether bee should dance for a patch"""
        
        # Extract patch characteristics
        patch_quality = patch_info.get('quality', 0.5)
        energy_profitability = foraging_success.get('energy_efficiency', 0.5)
        distance = patch_info.get('distance', 1000.0)
        
        # Get bee's individual threshold
        individual_threshold = self._get_bee_dance_threshold(bee_id)
        
        # Make dance decision
        should_dance = self.dance_model.should_dance(
            patch_quality, energy_profitability, distance, individual_threshold
        )
        
        return should_dance
    
    def perform_dance(self, bee_id: int, patch_info: Dict[str, Any],
                     foraging_success: Dict[str, Any]) -> Optional[DanceInformation]:
        """Perform waggle dance for a patch"""
        
        # Calculate dance characteristics
        patch_quality = patch_info.get('quality', 0.5)
        energy_profitability = foraging_success.get('energy_efficiency', 0.5)
        distance = patch_info.get('distance', 1000.0)
        
        # Determine dance type
        if distance < 50.0:
            dance_type = DanceType.ROUND_DANCE
        else:
            dance_type = DanceType.WAGGLE_DANCE
        
        # Calculate dance parameters
        dance_duration = self.dance_model.calculate_dance_duration(patch_quality, energy_profitability)
        dance_vigor = min(1.0, patch_quality * 0.7 + energy_profitability * 0.3)
        
        # Create dance information
        dance_info = DanceInformation(
            dance_id=f"dance_{bee_id}_{time.time()}",
            dancer_id=bee_id,
            dance_type=dance_type,
            patch_id=patch_info.get('patch_id', 0),
            patch_location=patch_info.get('location', (0.0, 0.0)),
            patch_distance=distance,
            patch_direction=patch_info.get('direction', 0.0),
            resource_type=ResourceType.NECTAR,  # Simplified
            resource_quality=patch_quality,
            resource_quantity=patch_info.get('resource_amount', 50.0),
            energy_profitability=energy_profitability,
            dance_duration=dance_duration,
            dance_vigor=dance_vigor,
            waggle_run_count=int(dance_duration / 5.0),  # Simplified
            dance_repetitions=int(dance_vigor * 10),
            recruitment_threshold=0.6,
            urgency_level=min(1.0, patch_quality + energy_profitability - 0.5),
            timestamp=time.time()
        )
        
        # Store active dance
        self.active_dances[dance_info.dance_id] = dance_info
        
        # Record communication event
        self._record_communication_event(
            CommunicationEvent.DANCE_PERFORMANCE,
            bee_id,
            information_content={
                'dance_id': dance_info.dance_id,
                'patch_id': dance_info.patch_id,
                'quality': dance_info.resource_quality,
                'vigor': dance_info.dance_vigor
            }
        )
        
        return dance_info
    
    def follow_dance(self, follower_id: int, dance_id: str) -> bool:
        """Attempt to follow a dance"""
        
        if dance_id not in self.active_dances:
            return False
        
        dance_info = self.active_dances[dance_id]
        
        # Get follower characteristics
        follower_experience = self._get_bee_experience(follower_id)
        dancer_reputation = self.bee_reputations.get(dance_info.dancer_id, 0.5)
        colony_need = self._assess_colony_need()
        
        # Calculate following probability
        following_prob = self.recruitment_model.calculate_following_probability(
            dance_info.dance_vigor, dancer_reputation, follower_experience, colony_need
        )
        
        # Make following decision
        will_follow = random.random() < following_prob
        
        if will_follow:
            # Add to followers
            dance_info.followers.add(follower_id)
            
            # Record communication event
            self._record_communication_event(
                CommunicationEvent.DANCE_FOLLOWING,
                follower_id,
                receiver_id=dance_info.dancer_id,
                information_content={
                    'dance_id': dance_id,
                    'patch_id': dance_info.patch_id,
                    'following_probability': following_prob
                }
            )
            
            # Share patch information with follower
            self._share_dance_information(dance_info, follower_id)
        
        return will_follow
    
    def update_recruitment_success(self, dance_id: str, follower_id: int,
                                 recruitment_successful: bool) -> None:
        """Update recruitment success tracking"""
        
        if dance_id not in self.active_dances:
            return
        
        dance_info = self.active_dances[dance_id]
        dancer_id = dance_info.dancer_id
        
        # Update recruitment success rates
        if dancer_id not in self.recruitment_success_rates:
            self.recruitment_success_rates[dancer_id] = 0.5
        
        # Running average
        current_rate = self.recruitment_success_rates[dancer_id]
        new_rate = current_rate * 0.9 + (1.0 if recruitment_successful else 0.0) * 0.1
        self.recruitment_success_rates[dancer_id] = new_rate
        
        # Update bee reputation
        current_reputation = self.bee_reputations.get(dancer_id, 0.5)
        new_reputation = self.recruitment_model.update_bee_reputation(
            dancer_id, recruitment_successful, current_reputation
        )
        self.bee_reputations[dancer_id] = new_reputation
        
        # Record recruitment success
        self._record_communication_event(
            CommunicationEvent.RECRUITMENT_SUCCESS,
            dancer_id,
            receiver_id=follower_id,
            information_content={
                'dance_id': dance_id,
                'success': recruitment_successful,
                'new_reputation': new_reputation
            },
            success=recruitment_successful
        )
    
    def discover_patch_through_dance(self, bee_id: int, dance_id: str) -> Optional[Dict[str, Any]]:
        """Discover new patch through dance information"""
        
        if dance_id not in self.active_dances:
            return None
        
        dance_info = self.active_dances[dance_id]
        
        # Create patch discovery information
        patch_discovery = {
            'patch_id': dance_info.patch_id,
            'location': dance_info.patch_location,
            'distance': dance_info.patch_distance,
            'direction': dance_info.patch_direction,
            'quality': dance_info.resource_quality,
            'resource_type': dance_info.resource_type.value,
            'discovered_through_dance': True,
            'source_dancer': dance_info.dancer_id,
            'information_accuracy': min(1.0, dance_info.dance_vigor + 0.2)
        }
        
        # Update colony knowledge
        self.colony_knowledge[dance_info.patch_id] = patch_discovery
        
        # Record patch discovery
        self._record_communication_event(
            CommunicationEvent.PATCH_DISCOVERY,
            bee_id,
            information_content=patch_discovery
        )
        
        return patch_discovery
    
    def _get_bee_dance_threshold(self, bee_id: int) -> float:
        """Get bee's individual dance threshold"""
        # Simplified - could be based on genetic, experience, or role factors
        return 0.6 + (bee_id % 10) * 0.03  # Slight individual variation
    
    def _get_bee_experience(self, bee_id: int) -> float:
        """Get bee's foraging experience level"""
        # Simplified - could track actual foraging history
        return 0.5 + (bee_id % 20) * 0.025  # Individual variation
    
    def _assess_colony_need(self) -> float:
        """Assess colony's need for resources"""
        # Simplified - could be based on actual colony energy levels
        return 0.6  # Moderate need
    
    def _share_dance_information(self, dance_info: DanceInformation, follower_id: int) -> None:
        """Share dance information with follower"""
        
        patch_info = {
            'patch_id': dance_info.patch_id,
            'location': dance_info.patch_location,
            'distance': dance_info.patch_distance,
            'direction': dance_info.patch_direction,
            'quality': dance_info.resource_quality,
            'resource_type': dance_info.resource_type.value,
            'energy_profitability': dance_info.energy_profitability,
            'information_source': 'dance',
            'dancer_id': dance_info.dancer_id,
            'accuracy': dance_info.dance_vigor
        }
        
        # Update follower's knowledge
        self.colony_knowledge[dance_info.patch_id] = patch_info
        
        # Record information sharing
        records = self.information_model.share_patch_information(
            dance_info.dancer_id, [follower_id], patch_info
        )
        
        self.communication_records.extend(records)
    
    def _record_communication_event(self, event_type: CommunicationEvent, sender_id: int,
                                   receiver_id: Optional[int] = None,
                                   information_content: Optional[Dict[str, Any]] = None,
                                   success: bool = True) -> None:
        """Record communication event"""
        
        record = CommunicationRecord(
            event_id=f"comm_{sender_id}_{time.time()}",
            event_type=event_type,
            timestamp=time.time(),
            sender_id=sender_id,
            receiver_id=receiver_id,
            information_content=information_content or {},
            success=success
        )
        
        self.communication_records.append(record)
        
        # Maintain record history size
        if len(self.communication_records) > 10000:
            self.communication_records.pop(0)
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get comprehensive communication statistics"""
        
        if not self.communication_records:
            return {"total_events": 0}
        
        # Event type distribution
        event_counts = defaultdict(int)
        for record in self.communication_records:
            event_counts[record.event_type.value] += 1
        
        # Success rates
        successful_events = sum(1 for record in self.communication_records if record.success)
        success_rate = successful_events / len(self.communication_records)
        
        # Dance statistics
        dance_count = len(self.active_dances)
        total_followers = sum(len(dance.followers) for dance in self.active_dances.values())
        
        return {
            "total_events": len(self.communication_records),
            "event_distribution": dict(event_counts),
            "overall_success_rate": success_rate,
            "active_dances": dance_count,
            "total_followers": total_followers,
            "average_followers_per_dance": total_followers / dance_count if dance_count > 0 else 0,
            "patches_in_colony_knowledge": len(self.colony_knowledge),
            "bee_reputations": len(self.bee_reputations),
            "recruitment_success_rates": dict(self.recruitment_success_rates)
        }


class ForagingCommunicationIntegrator(BaseModel):
    """Integrates communication system with foraging algorithms for comprehensive recruitment"""
    
    model_config = {"validate_assignment": True}
    
    # Core systems
    communication_system: BeeCommunicationSystem = Field(default_factory=BeeCommunicationSystem)
    effectiveness_tracker: "CommunicationEffectivenessTracker" = Field(default_factory=lambda: CommunicationEffectivenessTracker())
    
    # Foraging-communication parameters
    foraging_success_dance_probability: float = Field(default=0.8, description="Probability to dance after successful foraging")
    recruitment_efficiency_threshold: float = Field(default=0.3, description="Minimum efficiency for recruitment")
    patch_information_sharing_rate: float = Field(default=0.6, description="Rate of patch information sharing")
    
    # Adaptive behavior
    communication_learning_rate: float = Field(default=0.1, description="Rate of communication adaptation")
    social_learning_influence: float = Field(default=0.4, description="Influence of social learning on decisions")
    
    # State tracking
    recruitment_attempts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    foraging_communication_history: List[Dict[str, Any]] = Field(default_factory=list)
    patch_recruitment_outcomes: Dict[int, List[float]] = Field(default_factory=dict)
    
    def integrate_foraging_success_with_communication(self, 
                                                    bee_id: int, 
                                                    foraging_trip_result: Dict[str, Any],
                                                    patch_info: Dict[str, Any],
                                                    colony_state: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate foraging trip results with communication decisions"""
        
        integration_result = {
            "communication_triggered": False,
            "dance_performed": False,
            "recruitment_initiated": False,
            "information_shared": False,
            "communication_details": {}
        }
        
        # Extract foraging results
        energy_gain = foraging_trip_result.get("net_energy_gain", 0.0)
        trip_efficiency = foraging_trip_result.get("trip_efficiency", 0.0)
        patch_quality = foraging_trip_result.get("patch_quality", 0.0)
        resource_abundance = foraging_trip_result.get("resource_abundance", 0.0)
        
        # Calculate communication triggers
        success_threshold = self._calculate_adaptive_success_threshold(patch_info["patch_id"])
        communication_probability = self._calculate_communication_probability(
            energy_gain, trip_efficiency, patch_quality, resource_abundance
        )
        
        # Decide on communication actions
        if energy_gain > 0 and trip_efficiency > success_threshold:
            integration_result["communication_triggered"] = True
            
            # Share patch information
            if random.random() < self.patch_information_sharing_rate:
                self._share_patch_information(bee_id, patch_info, foraging_trip_result)
                integration_result["information_shared"] = True
            
            # Consider dancing
            if random.random() < communication_probability:
                dance_info = self.communication_system.create_dance(
                    bee_id, patch_info, patch_quality, 
                    energy_gain / max(1.0, foraging_trip_result.get("energy_cost", 1.0))
                )
                integration_result["dance_performed"] = True
                integration_result["communication_details"]["dance_id"] = dance_info.dance_id
                
                # Initiate recruitment if dance is vigorous enough
                if dance_info.should_recruit():
                    recruitment_result = self._initiate_recruitment(
                        bee_id, dance_info, colony_state
                    )
                    integration_result["recruitment_initiated"] = True
                    integration_result["communication_details"]["recruitment"] = recruitment_result
        
        # Update communication effectiveness based on outcome
        self._update_communication_effectiveness(bee_id, integration_result, foraging_trip_result)
        
        # Record communication-foraging event
        self.foraging_communication_history.append({
            "timestamp": time.time(),
            "bee_id": bee_id,
            "patch_id": patch_info["patch_id"],
            "foraging_result": foraging_trip_result,
            "communication_result": integration_result
        })
        
        return integration_result
    
    def _calculate_communication_probability(self, energy_gain: float, efficiency: float, 
                                           quality: float, abundance: float) -> float:
        """Calculate probability of initiating communication"""
        
        # Base probability from foraging success
        base_prob = min(1.0, (energy_gain / 50.0) * 0.4)  # Scale by typical energy gain
        
        # Efficiency bonus
        efficiency_bonus = efficiency * 0.3
        
        # Quality bonus
        quality_bonus = quality * 0.2
        
        # Abundance bonus
        abundance_bonus = min(abundance / 100.0, 0.1)  # Cap abundance influence
        
        total_probability = base_prob + efficiency_bonus + quality_bonus + abundance_bonus
        
        # Apply learning adjustments
        historical_success = self.effectiveness_tracker.dance_success_rates.get(DanceType.WAGGLE_DANCE, 0.5)
        learning_adjustment = (historical_success - 0.5) * self.communication_learning_rate
        
        return max(0.0, min(1.0, total_probability + learning_adjustment))
    
    def _calculate_adaptive_success_threshold(self, patch_id: int) -> float:
        """Calculate adaptive success threshold for patch communication"""
        
        # Base threshold
        base_threshold = 0.5
        
        # Adjust based on patch recruitment history
        if patch_id in self.patch_recruitment_outcomes:
            recent_outcomes = self.patch_recruitment_outcomes[patch_id][-10:]  # Last 10 outcomes
            if recent_outcomes:
                avg_outcome = sum(recent_outcomes) / len(recent_outcomes)
                # Lower threshold for historically successful patches
                adjustment = (avg_outcome - 0.5) * 0.3
                base_threshold -= adjustment
        
        return max(0.2, min(0.8, base_threshold))
    
    def _share_patch_information(self, bee_id: int, patch_info: Dict[str, Any], 
                               foraging_result: Dict[str, Any]) -> None:
        """Share patch information with colony knowledge base"""
        
        # Update colony knowledge through communication system
        self.communication_system.update_colony_knowledge(patch_info["patch_id"], {
            "location": patch_info.get("location", (0.0, 0.0)),
            "quality": foraging_result.get("patch_quality", 0.0),
            "resource_abundance": foraging_result.get("resource_abundance", 0.0),
            "last_visited": time.time(),
            "visitor_id": bee_id,
            "energy_profitability": foraging_result.get("net_energy_gain", 0.0),
            "trip_efficiency": foraging_result.get("trip_efficiency", 0.0)
        })
        
        # Record information sharing event
        self.communication_system._record_communication_event(
            CommunicationEvent.INFORMATION_SHARING,
            bee_id,
            information_content={
                "patch_id": patch_info["patch_id"],
                "shared_quality": foraging_result.get("patch_quality", 0.0),
                "shared_abundance": foraging_result.get("resource_abundance", 0.0)
            }
        )
    
    def _initiate_recruitment(self, dancer_id: int, dance_info: DanceInformation, 
                            colony_state: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate recruitment process for promising patch"""
        
        recruitment_id = f"recruitment_{dancer_id}_{time.time()}"
        
        # Determine recruitment parameters
        max_recruits = self._calculate_optimal_recruitment_size(dance_info, colony_state)
        recruitment_vigor = dance_info.calculate_dance_vigor()
        
        # Start recruitment tracking
        recruitment_attempt = {
            "recruitment_id": recruitment_id,
            "dance_id": dance_info.dance_id,
            "dancer_id": dancer_id,
            "patch_id": dance_info.patch_id,
            "target_recruits": max_recruits,
            "recruitment_vigor": recruitment_vigor,
            "start_time": time.time(),
            "recruited_bees": set(),
            "recruitment_status": "active"
        }
        
        self.recruitment_attempts[recruitment_id] = recruitment_attempt
        
        # Record recruitment initiation
        self.communication_system._record_communication_event(
            CommunicationEvent.RECRUITMENT_SUCCESS,
            dancer_id,
            information_content={
                "recruitment_id": recruitment_id,
                "patch_id": dance_info.patch_id,
                "target_recruits": max_recruits,
                "recruitment_vigor": recruitment_vigor
            }
        )
        
        return {
            "recruitment_id": recruitment_id,
            "target_recruits": max_recruits,
            "recruitment_vigor": recruitment_vigor,
            "patch_id": dance_info.patch_id
        }
    
    def _calculate_optimal_recruitment_size(self, dance_info: DanceInformation, 
                                          colony_state: Dict[str, Any]) -> int:
        """Calculate optimal number of recruits for patch"""
        
        # Base recruitment size from patch quality and resource abundance
        base_size = int(dance_info.resource_quality * dance_info.resource_quantity / 10.0)
        
        # Adjust for colony state
        available_foragers = colony_state.get("available_foragers", 50)
        colony_energy_level = colony_state.get("energy_level", 1000.0)
        
        # Scale by available workforce
        workforce_factor = min(1.0, available_foragers / 20.0)  # Assume 20 is typical forager count
        
        # Scale by colony energy needs
        energy_factor = 1.0 if colony_energy_level > 800.0 else 1.5  # Recruit more if energy is low
        
        optimal_size = int(base_size * workforce_factor * energy_factor)
        
        # Apply constraints
        return max(1, min(optimal_size, available_foragers // 3))  # Don't recruit more than 1/3 of foragers
    
    def _update_communication_effectiveness(self, bee_id: int, 
                                          communication_result: Dict[str, Any],
                                          foraging_result: Dict[str, Any]) -> None:
        """Update communication effectiveness metrics"""
        
        if communication_result["dance_performed"]:
            # Track dance effectiveness
            followers = len(self.communication_system.active_dances.get(
                communication_result["communication_details"].get("dance_id", ""), 
                DanceInformation("", 0, DanceType.WAGGLE_DANCE, 0, (0,0), 0, 0, ResourceType.NECTAR, 0, 0, 0, 0, 0, 0, 0, 0, 0, time.time())
            ).followers)
            
            # Estimate successful recruitments (simplified)
            successful_recruitments = int(followers * 0.7)  # Assume 70% success rate
            
            self.effectiveness_tracker.update_dance_effectiveness(
                DanceType.WAGGLE_DANCE, followers, successful_recruitments
            )
        
        # Update patch recruitment outcomes
        patch_id = foraging_result.get("patch_id", 0)
        if patch_id not in self.patch_recruitment_outcomes:
            self.patch_recruitment_outcomes[patch_id] = []
        
        outcome_score = foraging_result.get("trip_efficiency", 0.0)
        self.patch_recruitment_outcomes[patch_id].append(outcome_score)
        
        # Maintain history size
        if len(self.patch_recruitment_outcomes[patch_id]) > 50:
            self.patch_recruitment_outcomes[patch_id].pop(0)
    
    def get_recruitment_analytics(self) -> Dict[str, Any]:
        """Get comprehensive recruitment analytics"""
        
        active_recruitments = sum(1 for r in self.recruitment_attempts.values() 
                                if r["recruitment_status"] == "active")
        
        total_recruitments = len(self.recruitment_attempts)
        
        # Calculate average recruitment success
        successful_recruitments = sum(1 for r in self.recruitment_attempts.values() 
                                    if len(r["recruited_bees"]) > 0)
        
        recruitment_success_rate = (successful_recruitments / total_recruitments 
                                  if total_recruitments > 0 else 0.0)
        
        # Patch recruitment effectiveness
        patch_effectiveness = {}
        for patch_id, outcomes in self.patch_recruitment_outcomes.items():
            if outcomes:
                patch_effectiveness[patch_id] = {
                    "average_efficiency": sum(outcomes) / len(outcomes),
                    "recruitment_count": len(outcomes),
                    "success_rate": sum(1 for o in outcomes if o > 0.5) / len(outcomes)
                }
        
        return {
            "recruitment_summary": {
                "active_recruitments": active_recruitments,
                "total_recruitments": total_recruitments,
                "recruitment_success_rate": recruitment_success_rate,
                "average_recruits_per_attempt": self._calculate_average_recruits_per_attempt()
            },
            "communication_effectiveness": self.effectiveness_tracker.get_communication_metrics(),
            "patch_recruitment_effectiveness": patch_effectiveness,
            "foraging_communication_events": len(self.foraging_communication_history),
            "recent_communication_rate": self._calculate_recent_communication_rate()
        }
    
    def _calculate_average_recruits_per_attempt(self) -> float:
        """Calculate average number of recruits per recruitment attempt"""
        
        if not self.recruitment_attempts:
            return 0.0
        
        total_recruits = sum(len(r["recruited_bees"]) for r in self.recruitment_attempts.values())
        return total_recruits / len(self.recruitment_attempts)
    
    def _calculate_recent_communication_rate(self) -> float:
        """Calculate recent communication rate (last 100 foraging events)"""
        
        recent_events = self.foraging_communication_history[-100:]
        if not recent_events:
            return 0.0
        
        communication_events = sum(1 for event in recent_events 
                                 if event["communication_result"]["communication_triggered"])
        
        return communication_events / len(recent_events)


@dataclass
class RecruitmentResult:
    """Result of recruitment attempt"""
    recruitment_id: str
    dance_info: DanceInformation
    recruited_bees: Set[int]
    recruitment_success_rate: float
    patch_visitation_outcome: Dict[int, Dict[str, Any]]  # bee_id -> outcome
    total_energy_gained: float
    colony_benefit_score: float
    recruitment_efficiency: float  # benefit per recruited bee


class CommunicationEffectivenessTracker(BaseModel):
    """Tracks effectiveness of different communication strategies"""
    
    model_config = {"validate_assignment": True}
    
    # Dance effectiveness tracking
    dance_success_rates: Dict[DanceType, float] = Field(default_factory=dict)
    dance_follow_rates: Dict[DanceType, float] = Field(default_factory=dict)
    
    # Recruitment effectiveness
    recruitment_outcomes: Dict[str, RecruitmentResult] = Field(default_factory=dict)
    patch_discovery_rates: Dict[int, float] = Field(default_factory=dict)
    
    # Information accuracy
    distance_prediction_accuracy: float = Field(default=0.8)
    quality_prediction_accuracy: float = Field(default=0.7)
    direction_prediction_accuracy: float = Field(default=0.9)
    
    # Communication network metrics
    information_flow_efficiency: float = Field(default=0.6)
    social_learning_rate: float = Field(default=0.5)
    
    def update_dance_effectiveness(self, dance_type: DanceType, 
                                 followers: int, successful_recruitments: int) -> None:
        """Update dance effectiveness metrics"""
        if dance_type not in self.dance_success_rates:
            self.dance_success_rates[dance_type] = 0.5
            self.dance_follow_rates[dance_type] = 0.3
        
        # Update follow rate
        current_follow_rate = self.dance_follow_rates[dance_type]
        new_follow_rate = followers / max(1, followers + 5)  # Assume 5 potential observers
        self.dance_follow_rates[dance_type] = (current_follow_rate * 0.8 + new_follow_rate * 0.2)
        
        # Update success rate
        if followers > 0:
            success_rate = successful_recruitments / followers
            current_success_rate = self.dance_success_rates[dance_type]
            self.dance_success_rates[dance_type] = (current_success_rate * 0.8 + success_rate * 0.2)
    
    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get comprehensive communication effectiveness metrics"""
        return {
            "dance_effectiveness": {
                "success_rates": self.dance_success_rates,
                "follow_rates": self.dance_follow_rates,
                "average_success": (sum(self.dance_success_rates.values()) / 
                                   len(self.dance_success_rates) if self.dance_success_rates else 0.0)
            },
            "recruitment_metrics": {
                "total_recruitments": len(self.recruitment_outcomes),
                "average_recruitment_efficiency": self._calculate_avg_recruitment_efficiency(),
                "patch_discovery_success": (sum(self.patch_discovery_rates.values()) / 
                                           len(self.patch_discovery_rates) if self.patch_discovery_rates else 0.0)
            },
            "information_accuracy": {
                "distance": self.distance_prediction_accuracy,
                "quality": self.quality_prediction_accuracy,
                "direction": self.direction_prediction_accuracy,
                "overall": (self.distance_prediction_accuracy + self.quality_prediction_accuracy + self.direction_prediction_accuracy) / 3
            },
            "network_efficiency": {
                "flow_efficiency": self.information_flow_efficiency,
                "learning_rate": self.social_learning_rate
            }
        }
    
    def _calculate_avg_recruitment_efficiency(self) -> float:
        """Calculate average recruitment efficiency"""
        if not self.recruitment_outcomes:
            return 0.0
        
        efficiencies = [result.recruitment_efficiency for result in self.recruitment_outcomes.values()]
        return sum(efficiencies) / len(efficiencies)