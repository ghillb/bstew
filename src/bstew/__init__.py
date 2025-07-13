"""
BSTEW - BeeSteward v2 Python Transpilation
===========================================

A high-performance Python implementation of the BeeSteward v2 NetLogo model
for agent-based pollinator population modeling and decision support.

Core Components:
- Agent-based modeling framework (Mesa)
- Mathematical foundations (SciPy, NumPy)
- Spatial modeling system
- Configuration management
- CLI interface
"""

__version__ = "0.1.0"
__author__ = "BSTEW Team"

from .core.agents import BeeAgent, Queen, Worker, Forager
from .core.colony import Colony
from .core.environment import Environment
from .core.mathematics import ColonyDynamics
from .core.scheduler import BeeScheduler

__all__ = [
    "BeeAgent",
    "Queen",
    "Worker",
    "Forager",
    "Colony",
    "Environment",
    "ColonyDynamics",
    "BeeScheduler",
]
