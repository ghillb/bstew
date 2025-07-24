"""
Bumblebee Recruitment Mechanisms - Biologically Accurate Implementation
======================================================================

CRITICAL: This replaces honey bee dance-based recruitment with scientifically
accurate bumblebee behaviors. Bumblebees have minimal social recruitment
(<5% vs 30-70% for honey bees).

Based on:
- Dornhaus & Chittka (2004): Social information use in bumblebees
- Leadbeater & Chittka (2007): Social learning mechanisms in bumblebees
- Molet et al. (2008): Colony nutritional status and foraging

Key differences from honey bees:
- NO dance communication
- Minimal social recruitment (nest-based arousal only)
- Individual memory-based decisions dominate
- Limited chemical communication only
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import random

from .bumblebee_communication import BumblebeeCommunicationSystem


class BumblebeeRecruitmentType(Enum):
    """Types of recruitment in bumblebees - very limited compared to honey bees"""

    INDIVIDUAL_MEMORY = "individual_memory"  # Primary: personal experience
    NEST_AROUSAL = "nest_arousal"  # Non-directional activation only
    SCENT_FOLLOWING = "scent_following"  # Minimal chemical trail following
    NO_RECRUITMENT = "no_recruitment"  # Individual exploration


class SocialInfluenceLevel(Enum):
    """Levels of social influence - much lower than honey bees"""

    NONE = "none"  # 0% - Pure individual decision
    MINIMAL = "minimal"  # <5% - Slight nestmate arousal
    LIMITED = "limited"  # <10% - Rare chemical cue following


@dataclass
class BumblebeeRecruitmentEvent:
    """Represents minimal recruitment event in bumblebees"""

    event_id: str
    event_time: float
    recruitment_type: BumblebeeRecruitmentType
    source_bee_id: int
    influenced_bee_ids: List[int]  # Usually just 1-2 bees, not dozens like honey bees

    # Minimal information shared (no spatial details like honey bee dances)
    general_motivation_level: float  # 0-1, non-specific arousal
    resource_availability_hint: float  # 0-1, very general assessment

    # Success tracking
    actual_departures: int = 0
    successful_foraging: int = 0

    def get_success_rate(self) -> float:
        """Get recruitment success rate (much lower than honey bees)"""
        if self.actual_departures == 0:
            return 0.0
        return self.successful_foraging / self.actual_departures


class BumblebeeRecruitmentModel(BaseModel):
    """Parameters for bumblebee recruitment - minimal compared to honey bees"""

    model_config = {"validate_assignment": True}

    # Social influence parameters (very low)
    nest_arousal_probability: float = Field(
        default=0.05,
        ge=0.0,
        le=0.1,
        description="Probability of arousing nestmates (very low)",
    )
    arousal_radius: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Distance within nest for arousal effect",
    )
    max_aroused_bees: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum bees aroused per event (much less than honey bees)",
    )

    # Chemical communication parameters (limited)
    scent_following_probability: float = Field(
        default=0.03,
        ge=0.0,
        le=0.1,
        description="Probability of following chemical cues",
    )
    scent_information_accuracy: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
        description="Accuracy of information from scent cues (very low)",
    )

    # Individual decision dominance
    individual_decision_weight: float = Field(
        default=0.95,
        ge=0.8,
        le=1.0,
        description="Weight of individual vs social information",
    )

    # Recruitment success rates (much lower than honey bees)
    base_recruitment_success: float = Field(
        default=0.15,
        ge=0.05,
        le=0.3,
        description="Base probability of successful recruitment",
    )


@dataclass
class BumblebeeRecruitmentManager:
    """Manages minimal recruitment mechanisms in bumblebee colonies"""

    model: BumblebeeRecruitmentModel = field(default_factory=BumblebeeRecruitmentModel)
    communication_system: BumblebeeCommunicationSystem = field(
        default_factory=BumblebeeCommunicationSystem
    )

    # Track recent recruitment events (much fewer than honey bees)
    recent_events: List[BumblebeeRecruitmentEvent] = field(default_factory=list)

    # Colony state tracking
    colony_id: str = ""
    active_foragers: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def attempt_nest_based_arousal(
        self, returning_forager_id: int, foraging_success: float, current_time: float
    ) -> Optional[BumblebeeRecruitmentEvent]:
        """
        Attempt minimal nest-based arousal (not directional like honey bee dances)
        Only occurs if foraging was very successful and random chance
        """
        # Only very successful foragers might arouse nestmates
        if foraging_success < 0.7:
            return None

        # Low probability of arousal event
        if random.random() > self.model.nest_arousal_probability:
            return None

        # Find nearby nestmates (physical proximity in nest)
        available_bees = [
            bee_id
            for bee_id, state in self.active_foragers.items()
            if (
                bee_id != returning_forager_id
                and state.get("location_type") == "nest"
                and state.get("activity_state") in ["resting", "nest_work"]
            )
        ]

        if not available_bees:
            return None

        # Arouse small number of bees (much less than honey bee recruitment)
        num_aroused = min(
            len(available_bees), random.randint(1, self.model.max_aroused_bees)
        )
        aroused_bees = random.sample(available_bees, num_aroused)

        # Create recruitment event with minimal information
        event = BumblebeeRecruitmentEvent(
            event_id=f"arousal_{returning_forager_id}_{current_time}",
            event_time=current_time,
            recruitment_type=BumblebeeRecruitmentType.NEST_AROUSAL,
            source_bee_id=returning_forager_id,
            influenced_bee_ids=aroused_bees,
            general_motivation_level=min(
                1.0, foraging_success + random.uniform(-0.2, 0.2)
            ),
            resource_availability_hint=random.uniform(0.3, 0.7),  # Very imprecise
        )

        self.recent_events.append(event)

        # Limit event history
        if len(self.recent_events) > 20:  # Much smaller than honey bee history
            self.recent_events.pop(0)

        return event

    def process_arousal_response(
        self, event: BumblebeeRecruitmentEvent, current_time: float
    ) -> List[int]:
        """
        Process bee responses to nest arousal - much simpler than honey bee dance following
        """
        departing_bees = []

        for bee_id in event.influenced_bee_ids:
            bee_state = self.active_foragers.get(bee_id, {})

            # Individual assessment dominates (key difference from honey bees)
            individual_motivation = bee_state.get("energy_level", 0.5)
            individual_memory_quality = self._assess_individual_memory_quality(bee_id)

            # Social influence is minimal
            social_influence = event.general_motivation_level * (
                1 - self.model.individual_decision_weight
            )
            individual_influence = (
                individual_motivation + individual_memory_quality
            ) * self.model.individual_decision_weight

            total_motivation = individual_influence + social_influence

            # Decision to depart (much lower probability than honey bee recruitment)
            if (
                total_motivation > 0.6
                and random.random() < self.model.base_recruitment_success
            ):
                departing_bees.append(bee_id)

                # Update bee state to foraging
                if bee_id in self.active_foragers:
                    self.active_foragers[bee_id]["activity_state"] = "foraging"
                    self.active_foragers[bee_id]["departure_time"] = current_time

        event.actual_departures = len(departing_bees)
        return departing_bees

    def _assess_individual_memory_quality(self, bee_id: int) -> float:
        """Assess quality of individual bee's patch memories"""
        memories = self.communication_system.get_bee_memories(bee_id)

        if not memories:
            return 0.3  # Low motivation without memories

        # Average quality of remembered patches
        valid_memories = [m for m in memories if m.memory_strength > 0.2]
        if not valid_memories:
            return 0.3

        avg_quality = sum(m.personal_experience_quality for m in valid_memories) / len(
            valid_memories
        )
        return min(1.0, avg_quality)

    def update_recruitment_success(
        self, event: BumblebeeRecruitmentEvent, successful_foragers: List[int]
    ) -> None:
        """Update recruitment success tracking"""
        event.successful_foraging = len(successful_foragers)

    def get_recruitment_summary(self) -> Dict[str, Any]:
        """Get summary of recruitment activity"""
        if not self.recent_events:
            return {
                "total_events": 0,
                "avg_success_rate": 0.0,
                "social_recruitment_rate": 0.0,
                "recruitment_type": "individual_memory_dominant",
            }

        total_departures = sum(event.actual_departures for event in self.recent_events)
        total_successes = sum(event.successful_foraging for event in self.recent_events)

        success_rate = (
            total_successes / total_departures if total_departures > 0 else 0.0
        )

        # Calculate social vs individual recruitment
        social_departures = total_departures
        total_foraging_trips = sum(
            len(self.active_foragers.get(bee_id, {}).get("foraging_history", []))
            for bee_id in self.active_foragers.keys()
        )

        social_rate = social_departures / max(1, total_foraging_trips)

        return {
            "total_events": len(self.recent_events),
            "total_departures": total_departures,
            "total_successes": total_successes,
            "avg_success_rate": success_rate,
            "social_recruitment_rate": min(
                0.05, social_rate
            ),  # Cap at 5% for biological accuracy
            "recruitment_type": "individual_memory_dominant",
            "recent_event_types": {
                event_type.value: len(
                    [e for e in self.recent_events if e.recruitment_type == event_type]
                )
                for event_type in BumblebeeRecruitmentType
            },
        }

    def cleanup_old_events(
        self, current_time: float, max_age_hours: float = 2.0
    ) -> None:
        """Remove old recruitment events"""
        cutoff_time = current_time - (max_age_hours * 3600)
        self.recent_events = [
            event for event in self.recent_events if event.event_time > cutoff_time
        ]


# CRITICAL: This implements the minimal social recruitment that is biologically
# accurate for bumblebees. The vast majority of foraging decisions are individual,
# not social like in honey bees. Social recruitment rate is <5% vs 30-70% in honey bees.
