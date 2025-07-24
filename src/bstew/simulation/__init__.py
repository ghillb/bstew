"""
BSTEW Simulation Engine
======================

Simulation engine module providing high-level interfaces for running
BSTEW simulations, managing configurations, and collecting results.

This module wraps the existing BeeModel and BstewModel implementations
to provide a consistent simulation API.
"""

from .simulation_engine import SimulationEngine, SimulationResults
from ..utils.config import SimulationConfig

__all__ = ["SimulationEngine", "SimulationConfig", "SimulationResults"]
