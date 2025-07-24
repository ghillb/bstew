"""
Honey Bee Dance Communication System
===================================

Implements the sophisticated waggle dance communication system used by
honey bees (Apis mellifera) for sharing patch location information and
coordinating foraging activities.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import random
import time

from .resource_collection import ResourceType


class DanceType(Enum):
    """Types of honey bee dances"""

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
    """Information encoded in a honey bee dance"""

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
        return (
            self.resource_quality > self.recruitment_threshold
            and self.energy_profitability > 0.5
            and self.dance_vigor > 0.6
        )


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
    """Model for honey bee dance decision making"""

    model_config = {"validate_assignment": True}

    # Dance thresholds
    min_quality_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum quality to dance"
    )
    min_profitability_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Minimum profitability to dance"
    )
    distance_factor: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Distance impact on dance probability"
    )

    # Dance intensity parameters
    vigor_quality_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Quality weight in vigor calculation"
    )
    vigor_profitability_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Profitability weight in vigor"
    )
    vigor_urgency_weight: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Urgency weight in vigor"
    )

    # Dance duration parameters
    base_duration: float = Field(
        default=30.0, ge=0.0, description="Base dance duration (seconds)"
    )
    quality_duration_multiplier: float = Field(
        default=2.0, ge=1.0, description="Quality impact on duration"
    )

    def should_dance(
        self,
        patch_quality: float,
        energy_profitability: float,
        distance: float,
        individual_threshold: float,
    ) -> bool:
        """Determine if honey bee should dance for a patch"""

        # Basic quality check
        if patch_quality < self.min_quality_threshold:
            return False

        # Profitability check
        if energy_profitability < self.min_profitability_threshold:
            return False

        # Distance adjustment
        distance_adjusted_quality = patch_quality * (
            1.0 - (distance / 2000.0) * (1.0 - self.distance_factor)
        )

        # Individual threshold (bee-specific)
        if distance_adjusted_quality < individual_threshold:
            return False

        return True

    def calculate_dance_probability(
        self,
        patch_quality: float,
        energy_profitability: float,
        distance: float,
        recent_success_rate: float,
    ) -> float:
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

    def calculate_dance_duration(
        self, patch_quality: float, energy_profitability: float
    ) -> float:
        """Calculate dance duration based on patch characteristics"""

        quality_factor = patch_quality * self.quality_duration_multiplier
        profitability_factor = energy_profitability * 1.5

        duration = self.base_duration * (quality_factor + profitability_factor) / 2.0

        return max(10.0, min(300.0, duration))  # 10 seconds to 5 minutes


class RecruitmentModel(BaseModel):
    """Model for honey bee recruitment success and following behavior"""

    model_config = {"validate_assignment": True}

    # Following behavior parameters
    base_following_probability: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Base probability of following"
    )
    dance_vigor_influence: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Dance vigor influence"
    )
    dancer_reputation_influence: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Dancer reputation influence"
    )

    # Recruitment success parameters
    information_accuracy_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Information accuracy threshold"
    )
    recruitment_success_rate: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Base recruitment success rate"
    )

    # Social learning parameters
    learning_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Social learning rate"
    )
    experience_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Personal experience weight"
    )

    def calculate_following_probability(
        self,
        dance_vigor: float,
        dancer_reputation: float,
        follower_experience: float,
        colony_need: float,
    ) -> float:
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

        probability = (
            base_prob + vigor_bonus + reputation_bonus + experience_factor + need_factor
        )

        return max(0.0, min(1.0, probability))

    def calculate_recruitment_success(
        self, dance_info: DanceInformation, follower_characteristics: Dict[str, float]
    ) -> float:
        """Calculate recruitment success probability"""

        # Information accuracy
        accuracy_factor = dance_info.success_rate * 0.4

        # Dance quality
        dance_quality = dance_info.dance_vigor * 0.3

        # Follower capability
        follower_capability = (
            follower_characteristics.get("foraging_efficiency", 0.5) * 0.2
        )

        # Resource quality
        resource_factor = dance_info.resource_quality * 0.1

        success_prob = (
            accuracy_factor + dance_quality + follower_capability + resource_factor
        )

        return max(0.0, min(1.0, success_prob))

    def update_bee_reputation(
        self, bee_id: int, recruitment_success: bool, current_reputation: float
    ) -> float:
        """Update bee reputation based on recruitment success"""

        if recruitment_success:
            reputation_change = 0.1 * (1.0 - current_reputation)  # Diminishing returns
        else:
            reputation_change = -0.05 * current_reputation  # Proportional decline

        new_reputation = current_reputation + reputation_change

        return max(0.0, min(1.0, new_reputation))


class InformationSharingModel(BaseModel):
    """Model for honey bee information sharing and patch discovery"""

    model_config = {"validate_assignment": True}

    # Information sharing parameters
    information_decay_rate: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Information decay per time step"
    )
    sharing_range: float = Field(
        default=10.0, ge=0.0, description="Information sharing range (meters)"
    )

    # Patch discovery parameters
    discovery_bonus: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Bonus for patch discovery"
    )
    novelty_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for novel information"
    )

    # Information quality parameters
    accuracy_improvement_rate: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Information accuracy improvement"
    )

    def calculate_information_value(
        self, patch_info: Dict[str, Any], colony_knowledge: Dict[int, Dict[str, Any]]
    ) -> float:
        """Calculate value of shared information"""

        patch_id = patch_info.get("patch_id", 0)

        # Novelty value
        if patch_id not in colony_knowledge:
            novelty_value = self.novelty_weight
        else:
            # Information update value
            known_info = colony_knowledge[patch_id]
            quality_difference = abs(
                float(patch_info.get("quality", 0.5))
                - float(known_info.get("quality", 0.5))
            )
            novelty_value = quality_difference * 0.2

        # Resource value
        resource_value = float(patch_info.get("quality", 0.5)) * 0.4

        # Distance value (closer patches are more valuable)
        distance = float(patch_info.get("distance", 1000.0))
        distance_value = max(0.1, 1.0 - distance / 2000.0) * 0.3

        # Discovery bonus
        discovery_value = (
            self.discovery_bonus if patch_id not in colony_knowledge else 0.0
        )

        return novelty_value + resource_value + distance_value + discovery_value


class HoneyBeeCommunicationSystem(BaseModel):
    """Comprehensive honey bee dance communication system"""

    model_config = {"validate_assignment": True}

    # Component models
    dance_model: DanceDecisionModel = Field(default_factory=DanceDecisionModel)
    recruitment_model: RecruitmentModel = Field(default_factory=RecruitmentModel)
    information_model: InformationSharingModel = Field(
        default_factory=InformationSharingModel
    )

    # System state
    active_dances: Dict[str, DanceInformation] = Field(
        default_factory=dict, description="Active dances"
    )
    communication_records: List[CommunicationRecord] = Field(
        default_factory=list, description="Communication history"
    )
    bee_reputations: Dict[int, float] = Field(
        default_factory=dict, description="Bee reputations"
    )
    colony_knowledge: Dict[int, Dict[str, Any]] = Field(
        default_factory=dict, description="Colony patch knowledge"
    )

    # Performance tracking
    recruitment_success_rates: Dict[int, float] = Field(
        default_factory=dict, description="Recruitment success by bee"
    )
    information_sharing_success: Dict[int, float] = Field(
        default_factory=dict, description="Information sharing success"
    )

    def evaluate_dance_decision(
        self, bee_id: int, patch_info: Dict[str, Any], foraging_success: Dict[str, Any]
    ) -> bool:
        """Evaluate whether honey bee should dance for a patch"""

        # Extract patch characteristics
        patch_quality = patch_info.get("quality", 0.5)
        energy_profitability = foraging_success.get("energy_efficiency", 0.5)
        distance = patch_info.get("distance", 1000.0)

        # Get bee's individual threshold
        individual_threshold = self._get_bee_dance_threshold(bee_id)

        # Make dance decision
        should_dance = self.dance_model.should_dance(
            patch_quality, energy_profitability, distance, individual_threshold
        )

        return should_dance

    def perform_dance(
        self, bee_id: int, patch_info: Dict[str, Any], foraging_success: Dict[str, Any]
    ) -> Optional[DanceInformation]:
        """Perform waggle dance for a patch"""

        # Calculate dance characteristics
        patch_quality = patch_info.get("quality", 0.5)
        energy_profitability = foraging_success.get("energy_efficiency", 0.5)
        distance = patch_info.get("distance", 1000.0)

        # Determine dance type
        if distance < 50.0:
            dance_type = DanceType.ROUND_DANCE
        else:
            dance_type = DanceType.WAGGLE_DANCE

        # Calculate dance parameters
        dance_duration = self.dance_model.calculate_dance_duration(
            patch_quality, energy_profitability
        )
        dance_vigor = min(1.0, patch_quality * 0.7 + energy_profitability * 0.3)

        # Create dance information
        dance_info = DanceInformation(
            dance_id=f"dance_{bee_id}_{time.time()}",
            dancer_id=bee_id,
            dance_type=dance_type,
            patch_id=patch_info.get("patch_id", 0),
            patch_location=patch_info.get("location", (0.0, 0.0)),
            patch_distance=distance,
            patch_direction=patch_info.get("direction", 0.0),
            resource_type=ResourceType.NECTAR,  # Simplified
            resource_quality=patch_quality,
            resource_quantity=patch_info.get("resource_amount", 50.0),
            energy_profitability=energy_profitability,
            dance_duration=dance_duration,
            dance_vigor=dance_vigor,
            waggle_run_count=int(dance_duration / 5.0),  # Simplified
            dance_repetitions=int(dance_vigor * 10),
            recruitment_threshold=0.6,
            urgency_level=min(1.0, patch_quality + energy_profitability - 0.5),
            timestamp=time.time(),
        )

        # Store active dance
        self.active_dances[dance_info.dance_id] = dance_info

        # Record communication event
        self._record_communication_event(
            CommunicationEvent.DANCE_PERFORMANCE,
            bee_id,
            information_content={
                "dance_id": dance_info.dance_id,
                "patch_id": dance_info.patch_id,
                "quality": dance_info.resource_quality,
                "vigor": dance_info.dance_vigor,
            },
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
                    "dance_id": dance_id,
                    "patch_id": dance_info.patch_id,
                    "following_probability": following_prob,
                },
            )

            # Share patch information with follower
            self._share_dance_information(dance_info, follower_id)

        return will_follow

    def update_recruitment_success(
        self, dance_id: str, follower_id: int, recruitment_successful: bool
    ) -> None:
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
                "dance_id": dance_id,
                "success": recruitment_successful,
                "new_reputation": new_reputation,
            },
            success=recruitment_successful,
        )

    def discover_patch_through_dance(
        self, bee_id: int, dance_id: str
    ) -> Optional[Dict[str, Any]]:
        """Discover new patch through dance information"""

        if dance_id not in self.active_dances:
            return None

        dance_info = self.active_dances[dance_id]

        # Create patch discovery information
        patch_discovery = {
            "patch_id": dance_info.patch_id,
            "location": dance_info.patch_location,
            "distance": dance_info.patch_distance,
            "direction": dance_info.patch_direction,
            "quality": dance_info.resource_quality,
            "resource_type": dance_info.resource_type.value,
            "discovered_through_dance": True,
            "source_dancer": dance_info.dancer_id,
            "information_accuracy": min(1.0, dance_info.dance_vigor + 0.2),
        }

        # Update colony knowledge
        self.colony_knowledge[dance_info.patch_id] = patch_discovery

        # Record patch discovery
        self._record_communication_event(
            CommunicationEvent.PATCH_DISCOVERY,
            bee_id,
            information_content=patch_discovery,
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

    def _share_dance_information(
        self, dance_info: DanceInformation, follower_id: int
    ) -> None:
        """Share dance information with follower"""

        patch_info = {
            "patch_id": dance_info.patch_id,
            "location": dance_info.patch_location,
            "distance": dance_info.patch_distance,
            "direction": dance_info.patch_direction,
            "quality": dance_info.resource_quality,
            "resource_type": dance_info.resource_type.value,
            "energy_profitability": dance_info.energy_profitability,
            "information_source": "dance",
            "dancer_id": dance_info.dancer_id,
            "accuracy": dance_info.dance_vigor,
        }

        # Update follower's knowledge
        self.colony_knowledge[dance_info.patch_id] = patch_info

    def _record_communication_event(
        self,
        event_type: CommunicationEvent,
        sender_id: int,
        receiver_id: Optional[int] = None,
        information_content: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> None:
        """Record communication event"""

        record = CommunicationRecord(
            event_id=f"comm_{sender_id}_{time.time()}",
            event_type=event_type,
            timestamp=time.time(),
            sender_id=sender_id,
            receiver_id=receiver_id,
            information_content=information_content or {},
            success=success,
        )

        self.communication_records.append(record)


# Backward compatibility alias
BeeCommunicationSystem = HoneyBeeCommunicationSystem
