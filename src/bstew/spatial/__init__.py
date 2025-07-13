"""
Spatial modeling system for BSTEW
=================================

Handles landscape representation, resource distribution, and spatial queries.
Replaces NetLogo's patch-based system with NumPy grids.
"""

from .landscape import LandscapeGrid
from .patches import ResourcePatch, HabitatType
from .resources import ResourceDistribution

__all__ = ["LandscapeGrid", "ResourcePatch", "HabitatType", "ResourceDistribution"]
