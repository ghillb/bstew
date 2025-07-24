"""
Unified Bee Communication System Interface
=========================================

Provides a configurable interface that routes to the appropriate
communication system based on bee species - honey bee dance communication
or bumblebee scent-based communication.
"""

from typing import Dict, List, Any
from enum import Enum
import logging

from .bee_species_config import (
    BeeSpeciesType,
    get_species_config,
    get_communication_system,
)


class CommunicationEvent(Enum):
    """Unified communication event types"""

    # Honey bee events
    DANCE_PERFORMANCE = "dance_performance"
    DANCE_FOLLOWING = "dance_following"
    RECRUITMENT_SUCCESS = "recruitment_success"

    # Bumblebee events
    SCENT_MARKING = "scent_marking"
    SCENT_FOLLOWING = "scent_following"
    NEST_AROUSAL = "nest_arousal"

    # Common events
    INFORMATION_SHARING = "information_sharing"
    PATCH_DISCOVERY = "patch_discovery"
    SOCIAL_LEARNING = "social_learning"


class UnifiedBeeCommunicationSystem:
    """
    Unified communication system that adapts behavior based on bee species
    """

    def __init__(self, species_type: BeeSpeciesType):
        self.species_type = species_type
        self.species_config = get_species_config(species_type)
        self.communication_system = get_communication_system(species_type)
        self.logger = logging.getLogger(__name__)

        # Track which system type we're using
        self.is_honey_bee = species_type == BeeSpeciesType.APIS_MELLIFERA
        self.is_bumblebee = species_type.value.startswith("Bombus_")

    def integrate_foraging_success_with_communication(
        self,
        bee_id: int,
        foraging_result: Dict[str, Any],
        colony_state: Dict[str, Any],
        environmental_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Integrate foraging success with appropriate communication system
        """

        if self.is_honey_bee:
            return self._honey_bee_integration(
                bee_id, foraging_result, colony_state, environmental_context
            )
        elif self.is_bumblebee:
            return self._bumblebee_integration(
                bee_id, foraging_result, colony_state, environmental_context
            )
        else:
            raise ValueError(f"Unsupported species type: {self.species_type}")

    def _honey_bee_integration(
        self,
        bee_id: int,
        foraging_result: Dict[str, Any],
        colony_state: Dict[str, Any],
        environmental_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle honey bee dance communication integration"""

        honey_system = self.communication_system

        # Evaluate dance decision
        should_dance = honey_system.evaluate_dance_decision(
            bee_id, foraging_result, foraging_result
        )

        result = {
            "communication_type": "dance",
            "should_communicate": should_dance,
            "communication_vigor": 0.0,
            "expected_followers": 0,
            "information_shared": False,
        }

        if should_dance:
            # Perform dance
            dance_info = honey_system.perform_dance(
                bee_id, foraging_result, foraging_result
            )

            if dance_info:
                result.update(
                    {
                        "communication_vigor": dance_info.dance_vigor,
                        "expected_followers": len(dance_info.followers),
                        "information_shared": True,
                        "dance_id": dance_info.dance_id,
                        "dance_type": dance_info.dance_type.value,
                        "dance_duration": dance_info.dance_duration,
                    }
                )

        return result

    def _bumblebee_integration(
        self,
        bee_id: int,
        foraging_result: Dict[str, Any],
        colony_state: Dict[str, Any],
        environmental_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle bumblebee scent communication integration"""

        bumblebee_system = self.communication_system

        # Evaluate scent marking decision using bumblebee system
        import time

        should_mark = bumblebee_system.potentially_leave_scent_mark(
            bee_id, foraging_result.get("location", (0.0, 0.0)), time.time()
        )

        result = {
            "communication_type": "scent",
            "should_communicate": should_mark,
            "communication_vigor": 0.0,
            "expected_followers": 0,
            "information_shared": False,
            "scent_marked": False,
            "scent_strength": 0.0,
        }

        if should_mark:
            # Scent was marked
            result.update(
                {
                    "scent_marked": True,
                    "scent_strength": 0.5,  # Default strength for successful marking
                    "information_shared": True,
                    "communication_vigor": 0.5,
                    "expected_followers": 1,
                }
            )

        return result

    def potentially_leave_scent_mark(
        self, bee_id: int, patch_quality: float, energy_gained: float, distance: float
    ) -> Dict[str, Any]:
        """
        Potentially leave a scent mark (bumblebee-specific method for compatibility)
        """

        if self.is_bumblebee:
            bumblebee_system = self.communication_system
            import time

            scent_marked = bumblebee_system.potentially_leave_scent_mark(
                bee_id, (0.0, 0.0), time.time()
            )

            if scent_marked:
                return {
                    "scent_marked": True,
                    "scent_strength": 0.5,
                    "memory_updated": True,
                }

        # Default for honey bees or when not marking
        return {"scent_marked": False, "scent_strength": 0.0, "memory_updated": False}

    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get metrics appropriate for the species"""

        if self.is_honey_bee:
            honey_system = self.communication_system
            return {
                "active_dances": len(honey_system.active_dances),
                "total_recruited_bees": sum(
                    len(dance.followers)
                    for dance in honey_system.active_dances.values()
                ),
                "average_dance_success_rate": sum(
                    honey_system.recruitment_success_rates.values()
                )
                / max(1, len(honey_system.recruitment_success_rates)),
                "colony_knowledge_patches": len(honey_system.colony_knowledge),
                "communication_events": len(honey_system.communication_records),
            }
        elif self.is_bumblebee:
            bumblebee_system = self.communication_system
            return {
                "active_scents": len(bumblebee_system.active_scents),
                "recent_arousal_events": len(bumblebee_system.recent_arousal_events),
                "scent_following_rate": getattr(
                    bumblebee_system, "scent_following_success_rate", 0.0
                ),
                "memory_patches": len(getattr(bumblebee_system, "patch_memory", {})),
                "communication_events": len(
                    getattr(bumblebee_system, "communication_log", [])
                ),
            }
        else:
            return {}

    def get_species_info(self) -> Dict[str, Any]:
        """Get information about the configured species"""

        return {
            "species_type": self.species_type.value,
            "common_name": self.species_config.common_name,
            "scientific_name": self.species_config.scientific_name,
            "communication_type": self.species_config.communication_type.value,
            "uses_dance_communication": self.species_config.uses_dance_communication,
            "uses_scent_communication": self.species_config.uses_scent_communication,
            "social_information_sharing": self.species_config.social_information_sharing,
            "individual_decision_weight": self.species_config.individual_decision_weight,
            "max_foraging_distance": self.species_config.max_foraging_distance_m,
            "colony_size_range": f"{self.species_config.min_colony_size}-{self.species_config.max_colony_size}",
        }


def create_communication_system(
    species_type: BeeSpeciesType,
) -> UnifiedBeeCommunicationSystem:
    """Create a communication system for the specified species"""
    return UnifiedBeeCommunicationSystem(species_type)


def create_multi_species_communication(
    species_types: List[BeeSpeciesType],
) -> Dict[BeeSpeciesType, UnifiedBeeCommunicationSystem]:
    """Create communication systems for multiple species"""

    systems = {}
    for species_type in species_types:
        systems[species_type] = create_communication_system(species_type)

    return systems


# Backward compatibility aliases
BeeCommunicationSystem = UnifiedBeeCommunicationSystem
