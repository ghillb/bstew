"""
Bumblebee Communication System - Biologically Accurate Implementation
====================================================================

CRITICAL: This replaces honey bee dance communication with scientifically
accurate bumblebee behaviors. Bumblebees DO NOT perform waggle dances.

Based on:
- Dornhaus & Chittka (2004): Information flow and regulation of foraging
- Leadbeater & Chittka (2007): Social learning in insects
- Molet et al. (2008): Colony nutritional status and foraging behavior

Bumblebee communication features:
- Individual memory-based foraging (PRIMARY)
- Limited scent marking at flowers
- Nest-based arousal (non-directional)
- No spatial dance communication
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import random
from collections import defaultdict

from .resource_collection import ResourceType


class BumblebeeCommunicationType(Enum):
    """Biologically accurate bumblebee communication types"""

    INDIVIDUAL_MEMORY = (
        "individual_memory"  # Primary behavior: personal patch knowledge
    )
    SCENT_MARKING = "scent_marking"  # Limited chemical communication at flowers
    NEST_BASED_AROUSAL = "nest_arousal"  # Non-directional social activity increase
    INDIVIDUAL_EXPLORATION = (
        "exploration"  # Personal patch discovery without social info
    )
    CUTICULAR_HYDROCARBON = "cuticular_cue"  # Chemical cues from nestmate contact


class CommunicationEvent(Enum):
    """Types of communication events - bumblebee specific"""

    INDIVIDUAL_LEARNING = "individual_learning"
    SCENT_DETECTION = "scent_detection"
    NESTMATE_AROUSAL = "nestmate_arousal"
    PATCH_DISCOVERY = "patch_discovery"
    MEMORY_REINFORCEMENT = "memory_reinforcement"


@dataclass
class IndividualMemoryEntry:
    """Individual bumblebee's memory of a patch"""

    patch_id: str
    location: Tuple[float, float]
    resource_type: ResourceType
    last_visit_time: float
    visit_count: int
    average_reward: float
    energy_cost: float  # Travel + handling costs
    success_rate: float  # Proportion of successful visits
    memory_strength: float  # Decays over time
    personal_experience_quality: float  # Individual assessment

    def update_memory(
        self, reward: float, energy_cost: float, current_time: float
    ) -> None:
        """Update memory based on personal foraging experience"""
        self.last_visit_time = current_time
        self.visit_count += 1

        # Update running average of reward
        alpha = 0.3  # Learning rate for reward updating
        self.average_reward = (1 - alpha) * self.average_reward + alpha * reward
        self.energy_cost = (1 - alpha) * self.energy_cost + alpha * energy_cost

        # Update success rate
        success = 1.0 if reward > 0 else 0.0
        self.success_rate = (1 - alpha) * self.success_rate + alpha * success

        # Update memory strength (successful visits strengthen memory)
        if reward > 0:
            self.memory_strength = min(1.0, self.memory_strength + 0.1)

        # Update personal quality assessment
        profitability = max(0, self.average_reward - self.energy_cost)
        self.personal_experience_quality = profitability * self.success_rate


@dataclass
class ScentMark:
    """Simple scent mark left by bumblebee - very limited information"""

    mark_id: str
    marker_id: int
    location: Tuple[float, float]
    mark_time: float
    strength: float = 1.0  # Decays over time
    mark_type: str = "foraging_trace"  # Very basic mark type

    def decay(self, current_time: float, decay_rate: float = 0.1) -> float:
        """Scent marks decay quickly - not long-lasting like honey bee info"""
        time_elapsed = current_time - self.mark_time
        self.strength = max(0, 1.0 - decay_rate * time_elapsed)
        return self.strength


class BumblebeeCommunicationModel(BaseModel):
    """Biologically accurate bumblebee communication model"""

    model_config = {"validate_assignment": True}

    # Individual memory parameters
    memory_decay_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Rate at which patch memories decay per time step",
    )
    memory_capacity: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Maximum number of patches remembered (literature: 8-20)",
    )

    # Scent marking parameters (very limited compared to honey bees)
    scent_mark_probability: float = Field(
        default=0.15,
        ge=0.0,
        le=0.3,
        description="Low probability of leaving scent marks",
    )
    scent_detection_range: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Range for detecting scent marks (meters)",
    )
    scent_decay_rate: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Fast decay of chemical marks"
    )

    # Social arousal parameters (non-directional)
    nestmate_arousal_probability: float = Field(
        default=0.05,
        ge=0.0,
        le=0.1,
        description="Very low probability of arousing nestmates",
    )
    arousal_radius: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Distance within which arousal can occur",
    )

    # Exploration parameters (primary foraging mode)
    individual_exploration_rate: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="High rate of individual exploration vs social info use",
    )
    patch_fidelity_strength: float = Field(
        default=0.7,
        ge=0.3,
        le=0.9,
        description="Tendency to return to known productive patches",
    )


@dataclass
class BumblebeeCommunicationSystem:
    """Manages all bumblebee communication - primarily individual memory"""

    model: BumblebeeCommunicationModel = field(
        default_factory=BumblebeeCommunicationModel
    )

    # Individual memory storage (key: bee_id)
    individual_memories: Dict[int, List[IndividualMemoryEntry]] = field(
        default_factory=dict
    )

    # Limited environmental scent marks
    active_scent_marks: List[ScentMark] = field(default_factory=list)

    # Simple communication events log
    communication_events: List[Dict[str, Any]] = field(default_factory=list)

    def get_bee_memories(self, bee_id: int) -> List[IndividualMemoryEntry]:
        """Get individual bee's patch memories"""
        if bee_id not in self.individual_memories:
            self.individual_memories[bee_id] = []
        return self.individual_memories[bee_id]

    def update_bee_memory(
        self,
        bee_id: int,
        patch_id: str,
        location: Tuple[float, float],
        resource_type: ResourceType,
        reward: float,
        energy_cost: float,
        current_time: float,
    ) -> None:
        """Update individual bee's memory based on foraging experience"""
        memories = self.get_bee_memories(bee_id)

        # Find existing memory or create new one
        existing_memory = None
        for memory in memories:
            if memory.patch_id == patch_id:
                existing_memory = memory
                break

        if existing_memory:
            existing_memory.update_memory(reward, energy_cost, current_time)
        else:
            # Create new memory entry
            new_memory = IndividualMemoryEntry(
                patch_id=patch_id,
                location=location,
                resource_type=resource_type,
                last_visit_time=current_time,
                visit_count=1,
                average_reward=reward,
                energy_cost=energy_cost,
                success_rate=1.0 if reward > 0 else 0.0,
                memory_strength=0.8,
                personal_experience_quality=max(0, reward - energy_cost),
            )

            memories.append(new_memory)

            # Limit memory capacity (forget oldest/weakest memories)
            if len(memories) > self.model.memory_capacity:
                memories.sort(key=lambda m: m.memory_strength)
                memories.pop(0)  # Remove weakest memory

        # Log individual learning event
        self.communication_events.append(
            {
                "event_type": CommunicationEvent.INDIVIDUAL_LEARNING,
                "bee_id": bee_id,
                "patch_id": patch_id,
                "reward": reward,
                "time": current_time,
            }
        )

    def potentially_leave_scent_mark(
        self, bee_id: int, location: Tuple[float, float], current_time: float
    ) -> bool:
        """Bumblebee may leave limited scent mark (low probability)"""
        if random.random() < self.model.scent_mark_probability:
            mark = ScentMark(
                mark_id=f"scent_{bee_id}_{current_time}",
                marker_id=bee_id,
                location=location,
                mark_time=current_time,
            )
            self.active_scent_marks.append(mark)

            self.communication_events.append(
                {
                    "event_type": CommunicationEvent.SCENT_DETECTION,
                    "bee_id": bee_id,
                    "location": location,
                    "time": current_time,
                }
            )
            return True
        return False

    def detect_nearby_scent_marks(
        self, bee_location: Tuple[float, float]
    ) -> List[ScentMark]:
        """Detect scent marks within detection range"""
        nearby_marks = []
        for mark in self.active_scent_marks:
            if mark.strength > 0:
                distance = (
                    (bee_location[0] - mark.location[0]) ** 2
                    + (bee_location[1] - mark.location[1]) ** 2
                ) ** 0.5
                if distance <= self.model.scent_detection_range:
                    nearby_marks.append(mark)
        return nearby_marks

    def select_foraging_target(
        self,
        bee_id: int,
        current_location: Tuple[float, float],
        available_patches: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Select foraging target based on individual memory and limited social cues"""
        memories = self.get_bee_memories(bee_id)

        # Primary strategy: Use individual memory (patch fidelity)
        if memories and random.random() < self.model.patch_fidelity_strength:
            # Sort memories by personal experience quality
            valid_memories = [m for m in memories if m.memory_strength > 0.1]
            if valid_memories:
                valid_memories.sort(
                    key=lambda m: m.personal_experience_quality, reverse=True
                )
                return valid_memories[0].patch_id

        # Secondary strategy: Individual exploration of new patches
        if random.random() < self.model.individual_exploration_rate:
            # Explore based on individual assessment, not social information
            if available_patches:
                return str(random.choice(available_patches)["patch_id"])

        # Minimal social influence: Check for nearby scent marks (rarely used)
        nearby_scents = self.detect_nearby_scent_marks(current_location)
        if nearby_scents and random.random() < 0.1:  # Very low social influence
            # This represents minimal chemical cue following
            mark = random.choice(nearby_scents)
            # Try to find patch near the scent mark
            for patch in available_patches:
                patch_loc = patch.get("location", (0, 0))
                distance = (
                    (mark.location[0] - patch_loc[0]) ** 2
                    + (mark.location[1] - patch_loc[1]) ** 2
                ) ** 0.5
                if distance < 5.0:  # Close to scent mark
                    return str(patch["patch_id"])

        return None

    def decay_memories_and_scents(self, current_time: float) -> None:
        """Decay individual memories and scent marks over time"""
        # Decay individual memories
        for bee_id, memories in self.individual_memories.items():
            for memory in memories:
                memory.memory_strength = max(
                    0, memory.memory_strength - self.model.memory_decay_rate
                )

        # Decay and remove old scent marks
        for mark in self.active_scent_marks:
            mark.decay(current_time, self.model.scent_decay_rate)

        self.active_scent_marks = [
            mark for mark in self.active_scent_marks if mark.strength > 0.01
        ]

    def get_communication_summary(self) -> Dict[str, Any]:
        """Get summary of bumblebee communication activity"""
        total_memories = sum(
            len(memories) for memories in self.individual_memories.values()
        )
        active_scents = len([m for m in self.active_scent_marks if m.strength > 0.1])

        event_counts: Dict[CommunicationEvent, int] = defaultdict(int)
        for event in self.communication_events[-100:]:  # Recent events
            event_counts[event["event_type"]] += 1

        return {
            "total_individual_memories": total_memories,
            "active_scent_marks": active_scents,
            "recent_events": dict(event_counts),
            "communication_type": "individual_memory_based",  # Key difference from honey bees
            "social_recruitment_rate": 0.05,  # <5% vs 30-70% for honey bees
        }


# CRITICAL: No dance classes, no waggle dance, no spatial information sharing
# This is biologically accurate for bumblebees (Bombus spp.)
