"""
Advanced Dance Communication Integration System
==============================================

Integrates bee agents with waggle dance mechanics and recruitment patterns
for honey bee communication systems. This module provides the dance decision
engine, recruitment processing, and communication integration functionality.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DanceType(Enum):
    """Types of waggle dances"""

    ROUND_DANCE = "round_dance"
    WAGGLE_DANCE = "waggle_dance"
    RECRUITMENT_DANCE = "recruitment_dance"


class ResourceType(Enum):
    """Types of resources"""

    NECTAR = "nectar"
    POLLEN = "pollen"
    WATER = "water"


class BeeStatus(Enum):
    """Bee status types"""

    RESTING = "resting"
    NURSING = "nursing"
    FORAGING = "foraging"
    DANCING = "dancing"


@dataclass
class DanceDecision:
    """Decision result for whether a bee should dance"""

    should_dance: bool
    dance_type: DanceType
    dance_intensity: float
    decision_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class DanceInformation:
    """Information encoded in a waggle dance"""

    dance_id: str
    dancer_id: int
    dance_type: DanceType
    patch_id: int
    patch_location: Tuple[float, float]
    patch_distance: float
    patch_direction: float  # Radians
    resource_type: ResourceType
    resource_quality: float
    resource_quantity: float
    energy_profitability: float
    dance_duration: float
    dance_vigor: float
    waggle_run_count: int
    dance_repetitions: int
    recruitment_threshold: float
    urgency_level: float
    timestamp: float


@dataclass
class DancePerformance:
    """A dance performance event"""

    performance_id: str
    dancer_id: int
    dance_info: DanceInformation
    start_time: float
    duration: float
    intensity: float
    audience_size: int = 0


@dataclass
class FollowerResponse:
    """Response of a bee following a dance"""

    follower_id: int
    performance_id: str
    attention_duration: float
    information_quality: float
    follow_through: bool
    confidence: float = 0.0


@dataclass
class ColonyInformationState:
    """Colony's collective information state"""

    known_patches: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    patch_quality_estimates: Dict[int, float] = field(default_factory=dict)
    collective_knowledge_quality: float = 0.0
    information_freshness: Dict[int, float] = field(default_factory=dict)


class DanceDecisionEngine:
    """Engine for making dance decisions based on foraging results"""

    def __init__(self):
        self.min_dance_quality_threshold = 0.3
        self.quality_weight = 0.4
        self.energy_weight = 0.3
        self.distance_weight = 0.3
        self.urgency_weight = 0.2

    def evaluate_dance_decision(
        self,
        bee_id: int,
        foraging_result: Dict[str, Any],
        colony_state: Dict[str, Any],
        bee_experience: Dict[str, Any],
    ) -> DanceDecision:
        """Evaluate whether a bee should dance based on foraging results"""

        patch_quality = foraging_result.get("patch_quality", 0.0)
        energy_gained = foraging_result.get("energy_gained", 0.0)
        distance = foraging_result.get("distance_traveled", 0.0)

        # Calculate decision factors
        quality_factor = patch_quality * self.quality_weight
        energy_factor = min(1.0, energy_gained / 100.0) * self.energy_weight
        distance_factor = max(0.0, 1.0 - distance / 1000.0) * self.distance_weight

        colony_need = colony_state.get("current_need", 0.5)
        urgency_factor = colony_need * self.urgency_weight

        total_score = quality_factor + energy_factor + distance_factor + urgency_factor

        # Determine dance decision
        should_dance = total_score > self.min_dance_quality_threshold

        # Determine dance type based on distance
        if distance < 50:
            dance_type = DanceType.ROUND_DANCE
        elif distance < 200:
            dance_type = DanceType.WAGGLE_DANCE
        else:
            dance_type = DanceType.RECRUITMENT_DANCE

        # Calculate dance intensity
        dance_intensity = min(1.0, total_score)

        decision_factors = {
            "quality_factor": quality_factor,
            "energy_factor": energy_factor,
            "distance_factor": distance_factor,
            "urgency_factor": urgency_factor,
            "total_score": total_score,
        }

        return DanceDecision(
            should_dance=should_dance,
            dance_type=dance_type,
            dance_intensity=dance_intensity,
            decision_factors=decision_factors,
        )


class RecruitmentProcessor:
    """Processes dance recruitment and follower responses"""

    def __init__(self):
        self.attention_probability_base = 0.3
        self.attention_distance_factor = 0.1
        self.recruitment_threshold = 0.6

    def process_dance_audience(
        self,
        performance: DancePerformance,
        potential_followers: List[int],
        colony_state: Dict[str, Any],
        bee_states: Dict[int, Dict[str, Any]],
    ) -> List[FollowerResponse]:
        """Process dance audience and determine follower responses"""

        responses = []

        for follower_id in potential_followers:
            if follower_id not in bee_states:
                continue

            bee_state = bee_states[follower_id]

            # Determine if bee pays attention
            if self._determines_attention(follower_id, performance, bee_state):
                response = self._process_information_acquisition(
                    follower_id, performance, bee_state, colony_state
                )
                responses.append(response)

        return responses

    def _determines_attention(
        self, bee_id: int, performance: DancePerformance, bee_state: Dict[str, Any]
    ) -> bool:
        """Determine if a bee pays attention to a dance"""

        base_prob = self.attention_probability_base
        energy_factor = bee_state.get("energy", 50.0) / 100.0
        status_factor = 1.0 if bee_state.get("status") == BeeStatus.RESTING else 0.5

        attention_prob = base_prob * energy_factor * status_factor

        import random

        return random.random() < attention_prob

    def _process_information_acquisition(
        self,
        follower_id: int,
        performance: DancePerformance,
        follower_state: Dict[str, Any],
        colony_state: Dict[str, Any],
    ) -> FollowerResponse:
        """Process information acquisition from dance"""

        # Calculate attention duration based on dance quality
        base_duration = performance.duration * 0.3
        quality_factor = performance.dance_info.resource_quality
        attention_duration = base_duration * (0.5 + quality_factor * 0.5)

        # Calculate information quality based on follower's learning ability
        learning_rate = follower_state.get("learning_rate", 0.5)
        dance_vigor = performance.intensity
        information_quality = min(1.0, learning_rate * dance_vigor)

        # Determine follow-through decision
        motivation = follower_state.get("foraging_motivation", 0.5)
        colony_need = colony_state.get("energy_level", 0.5)

        follow_through_prob = information_quality * motivation * colony_need
        import random

        follow_through = random.random() < follow_through_prob

        return FollowerResponse(
            follower_id=follower_id,
            performance_id=performance.performance_id,
            attention_duration=attention_duration,
            information_quality=information_quality,
            follow_through=follow_through,
            confidence=information_quality * 0.8,
        )


class DanceCommunicationIntegrator:
    """Main integrator for dance communication system"""

    def __init__(self):
        self.dance_decision_engine = DanceDecisionEngine()
        self.recruitment_processor = RecruitmentProcessor()
        self.active_dances: Dict[str, DancePerformance] = {}
        self.dance_followers: Dict[str, List[FollowerResponse]] = {}
        self.colony_information: Dict[int, ColonyInformationState] = {}

    def process_returning_forager(
        self,
        bee_id: int,
        foraging_result: Dict[str, Any],
        colony_id: int,
        colony_state: Dict[str, Any],
        bee_states: Dict[int, Dict[str, Any]],
    ) -> Optional[DancePerformance]:
        """Process a returning forager and potentially initiate dance"""

        bee_experience = bee_states.get(bee_id, {})

        # Make dance decision
        decision = self.dance_decision_engine.evaluate_dance_decision(
            bee_id, foraging_result, colony_state, bee_experience
        )

        if not decision.should_dance:
            return None

        # Create dance information
        dance_info = DanceInformation(
            dance_id=f"dance_{bee_id}_{time.time()}",
            dancer_id=bee_id,
            dance_type=decision.dance_type,
            patch_id=foraging_result.get("patch_id", 0),
            patch_location=foraging_result.get("patch_location", (0.0, 0.0)),
            patch_distance=foraging_result.get("distance_traveled", 0.0),
            patch_direction=0.0,  # Simplified
            resource_type=ResourceType.NECTAR,
            resource_quality=foraging_result.get("patch_quality", 0.0),
            resource_quantity=1.0,
            energy_profitability=foraging_result.get("energy_gained", 0.0),
            dance_duration=15.0,
            dance_vigor=decision.dance_intensity,
            waggle_run_count=int(15 * decision.dance_intensity),
            dance_repetitions=3,
            recruitment_threshold=0.6,
            urgency_level=colony_state.get("current_need", 0.5),
            timestamp=time.time(),
        )

        # Create dance performance
        performance = DancePerformance(
            performance_id=f"perf_{dance_info.dance_id}",
            dancer_id=bee_id,
            dance_info=dance_info,
            start_time=time.time(),
            duration=dance_info.dance_duration,
            intensity=decision.dance_intensity,
        )

        # Store active dance
        self.active_dances[performance.performance_id] = performance

        return performance

    def get_recruited_bees(self, colony_id: int) -> List[int]:
        """Get list of recruited bees for a colony"""
        recruited = []

        for performance_id, responses in self.dance_followers.items():
            for response in responses:
                if response.follow_through:
                    recruited.append(response.follower_id)

        return recruited

    def get_colony_communication_metrics(self, colony_id: int) -> Dict[str, Any]:
        """Get communication effectiveness metrics for a colony"""

        active_dance_count = len(
            [d for d in self.active_dances.values() if d.dancer_id in range(1000)]
        )  # Simplified colony membership

        total_recruited = len(self.get_recruited_bees(colony_id))

        if colony_id not in self.colony_information:
            self.colony_information[colony_id] = ColonyInformationState()

        colony_info = self.colony_information[colony_id]

        return {
            "active_dances": active_dance_count,
            "total_recruited_bees": total_recruited,
            "known_patches": len(colony_info.known_patches),
            "average_dance_success_rate": 0.5,  # Simplified
            "information_quality": colony_info.collective_knowledge_quality,
        }

    def cleanup_finished_dances(self, current_time: float) -> None:
        """Clean up finished dance performances"""

        finished = []
        for perf_id, performance in self.active_dances.items():
            if current_time - performance.start_time > performance.duration:
                finished.append(perf_id)

        for perf_id in finished:
            del self.active_dances[perf_id]
            if perf_id in self.dance_followers:
                del self.dance_followers[perf_id]

    def update_follower_outcomes(
        self, performance_id: str, outcomes: Dict[int, bool]
    ) -> None:
        """Update follower outcomes for a dance performance"""

        if performance_id in self.dance_followers:
            for response in self.dance_followers[performance_id]:
                if response.follower_id in outcomes:
                    response.follow_through = outcomes[response.follower_id]


def create_dance_communication_integration() -> DanceCommunicationIntegrator:
    """Create and configure dance communication integration system"""
    return DanceCommunicationIntegrator()
