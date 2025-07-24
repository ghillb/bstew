"""
Recruitment Mechanisms - Redirects to Biologically Accurate Implementation
=========================================================================

CRITICAL CHANGE: This module now redirects to bumblebee_recruitment_mechanisms.py
which implements scientifically accurate bumblebee behaviors.

REMOVED: All honey bee dance-based recruitment (biologically invalid for bumblebees)
REASON: Bumblebees have <5% social recruitment vs 30-70% for honey bees
"""

# Import biologically accurate bumblebee recruitment
from .bumblebee_recruitment_mechanisms import (
    BumblebeeRecruitmentType,
    SocialInfluenceLevel,
    BumblebeeRecruitmentEvent,
    BumblebeeRecruitmentModel,
    BumblebeeRecruitmentManager,
)

# Import additional classes needed for tests
from .information_sharing import (
    InformationSharingModel,
    ColonyInformationNetwork,
    RecruitmentMechanismManager,
)

# Legacy compatibility exports (for existing code that expects these names)
# These now redirect to biologically accurate implementations
RecruitmentType = BumblebeeRecruitmentType
RecruitmentEvent = BumblebeeRecruitmentEvent
RecruitmentModel = BumblebeeRecruitmentModel
RecruitmentManager = BumblebeeRecruitmentManager

# REMOVED CLASSES (honey bee behaviors that don't exist in bumblebees):
# - DanceDecisionModel (no dance decisions)
# - InformationSharingModel (no spatial information sharing)
# - FollowerResponse (no dance following)
# - DancePerformance (no dances performed)

__all__ = [
    "BumblebeeRecruitmentType",
    "SocialInfluenceLevel",
    "BumblebeeRecruitmentEvent",
    "BumblebeeRecruitmentModel",
    "BumblebeeRecruitmentManager",
    # Legacy compatibility
    "RecruitmentType",
    "RecruitmentEvent",
    "RecruitmentModel",
    "RecruitmentManager",
    # Additional classes for tests
    "InformationSharingModel",
    "ColonyInformationNetwork",
    "RecruitmentMechanismManager",
]
