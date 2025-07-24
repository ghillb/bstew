"""
Core Enums for BSTEW - Bee Steward Simulation
=============================================

Centralized enum definitions to avoid circular imports.
"""

from enum import Enum


class BeeStatus(Enum):
    """Enumeration of bee status states - expanded to match NetLogo BEE-STEWARD v2"""

    # Basic life status
    ALIVE = "alive"
    DEAD = "dead"

    # Hibernation and dormancy
    HIBERNATING = "hibernating"

    # Nest construction and maintenance
    NEST_CONSTRUCTION = "nest_construction"

    # Foraging activities
    SEARCHING = "searching"
    RETURNING_EMPTY = "returning_empty"
    RETURNING_UNHAPPY_NECTAR = "returning_unhappy_nectar"
    RETURNING_UNHAPPY_POLLEN = "returning_unhappy_pollen"
    NECTAR_FORAGING = "nectar_foraging"
    COLLECT_NECTAR = "collect_nectar"
    BRINGING_NECTAR = "bringing_nectar"
    EXPERIMENTAL_FORAGING_NECTAR = "experimental_foraging_nectar"
    POLLEN_FORAGING = "pollen_foraging"
    COLLECT_POLLEN = "collect_pollen"
    BRINGING_POLLEN = "bringing_pollen"
    EXPERIMENTAL_FORAGING_POLLEN = "experimental_foraging_pollen"

    # Reproductive activities
    EGG_LAYING = "egg_laying"

    # Colony maintenance
    NURSING = "nursing"

    # Communication
    DANCING = "dancing"

    # General states
    RESTING = "resting"
    FORAGING = "foraging"  # Kept for backward compatibility


class BeeRole(Enum):
    """Enumeration of bee roles within the colony"""

    QUEEN = "queen"
    NURSE = "nurse"
    FORAGER = "forager"
    GUARD = "guard"
    BUILDER = "builder"
    DRONE = "drone"
    WORKER = "worker"


class ActivityStateCategory(Enum):
    """Categories of activity states"""

    FORAGING = "foraging"
    COMMUNICATION = "communication"
    MAINTENANCE = "maintenance"
    CARE = "care"
    CONSTRUCTION = "construction"
    GENERAL = "general"
    DORMANT = "dormant"
    HIVE_WORK = "hive_work"
    PHYSIOLOGICAL = "physiological"
    REPRODUCTIVE = "reproductive"
    EXPLORATION = "exploration"
    REPRODUCTION = "reproduction"
