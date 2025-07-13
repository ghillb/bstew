"""
Scheduler system for BSTEW
==========================

Mesa-based scheduler for managing bee agent activation and time steps.
"""

import mesa
from typing import List, Dict, Any, Set
from collections import defaultdict
import logging

from .agents import BeeAgent, BeeRole, BeeStatus


class BeeScheduler(mesa.time.BaseScheduler):
    """
    Custom scheduler for bee agents with role-based activation.

    Implements:
    - Role-based activation order
    - Stochastic activation within roles
    - Activity-based filtering
    - Performance optimization
    """

    def __init__(self, model: Any) -> None:
        super().__init__(model)

        # Role activation order (queens first, then workers, foragers, drones)
        self.role_order = [
            BeeRole.QUEEN,
            BeeRole.NURSE,
            BeeRole.FORAGER,
            BeeRole.GUARD,
            BeeRole.BUILDER,
            BeeRole.DRONE,
        ]

        # Optimized agent tracking with sets for O(1) operations
        self.agents_by_role: Dict[BeeRole, Set[BeeAgent]] = defaultdict(set)
        self.agents_by_status: Dict[BeeStatus, Set[BeeAgent]] = defaultdict(set)
        
        # Pre-filtered active agents by role for fast iteration
        self.active_agents_by_role: Dict[BeeRole, Set[BeeAgent]] = defaultdict(set)
        
        # Activity-based filtering (converted to sets for performance)
        self.active_agents: Set[BeeAgent] = set()
        self.inactive_agents: Set[BeeAgent] = set()
        
        # Fast lookup dictionaries
        self.agent_status_lookup: Dict[int, BeeStatus] = {}  # agent_id -> status
        self.agent_role_lookup: Dict[int, BeeRole] = {}     # agent_id -> role
        self.agent_energy_lookup: Dict[int, float] = {}     # agent_id -> energy
        
        # Dirty flags for incremental updates
        self.status_dirty_agents: Set[BeeAgent] = set()
        self.role_dirty_agents: Set[BeeAgent] = set()
        
        # Performance counters
        self.activation_counts: Dict[str, int] = defaultdict(int)
        
        # Performance logging
        self.logger = logging.getLogger(__name__)

    def add(self, agent: Any) -> None:
        """Add agent to scheduler"""
        super().add(agent)

        # Add to optimized role-based grouping (only for BeeAgent instances)
        if hasattr(agent, "role"):
            self.agents_by_role[agent.role].add(agent)
            # Update lookup dictionaries
            self.agent_role_lookup[agent.unique_id] = agent.role
            
        # Add to status tracking
        if hasattr(agent, "status"):
            self.agents_by_status[agent.status].add(agent)
            self.agent_status_lookup[agent.unique_id] = agent.status
            
        # Add energy tracking
        if hasattr(agent, "energy"):
            self.agent_energy_lookup[agent.unique_id] = agent.energy

        # Add to active agents initially (only for BeeAgent instances)
        if hasattr(agent, "role"):
            self.active_agents.add(agent)
            # Update pre-filtered active agents by role
            if agent.status != BeeStatus.DEAD:
                self.active_agents_by_role[agent.role].add(agent)

    def remove(self, agent: Any) -> None:
        """Remove agent from scheduler"""
        super().remove(agent)

        # Remove from optimized role-based grouping (only for BeeAgent instances)
        if hasattr(agent, "role") and agent in self.agents_by_role[agent.role]:
            self.agents_by_role[agent.role].discard(agent)
            self.active_agents_by_role[agent.role].discard(agent)
            
        # Remove from status tracking
        if hasattr(agent, "status") and agent in self.agents_by_status[agent.status]:
            self.agents_by_status[agent.status].discard(agent)
            
        # Remove from lookup dictionaries
        agent_id = agent.unique_id
        self.agent_role_lookup.pop(agent_id, None)
        self.agent_status_lookup.pop(agent_id, None)
        self.agent_energy_lookup.pop(agent_id, None)

        # Remove from activity sets (only for BeeAgent instances)
        self.active_agents.discard(agent)
        self.inactive_agents.discard(agent)
        
        # Remove from dirty flags
        self.status_dirty_agents.discard(agent)
        self.role_dirty_agents.discard(agent)

    def step(self) -> None:
        """Execute one scheduler step"""
        self.steps += 1
        self.time += 1

        # Update activity status and dirty flags
        self._update_agent_activity()
        self._process_dirty_flags()

        # Activate agents by role in order using optimized data structures
        for role in self.role_order:
            # Use pre-filtered active agents for this role
            active_role_agents = self.active_agents_by_role[role]

            if not active_role_agents:
                continue

            # Convert to list for activation (still need list for random sampling)
            active_role_agents_list = list(active_role_agents)

            # Activate agents (with optional randomization)
            self._activate_role_agents(role, active_role_agents_list)

        # Clean up dead agents
        self._cleanup_dead_agents()

    def _update_agent_activity(self) -> None:
        """Update agent activity status based on conditions using optimized sets"""
        new_active = set()
        new_inactive = set()

        for agent in self.agents:
            if self._is_agent_active(agent):
                new_active.add(agent)
            else:
                new_inactive.add(agent)

        # Update active/inactive sets
        self.active_agents = new_active
        self.inactive_agents = new_inactive
        
        # Update pre-filtered active agents by role
        self.active_agents_by_role.clear()
        for agent in new_active:
            if hasattr(agent, "role") and agent.status != BeeStatus.DEAD:
                self.active_agents_by_role[agent.role].add(agent)

    def _is_agent_active(self, agent: BeeAgent) -> bool:
        """Determine if agent should be active this step"""
        # Handle Colony objects differently from BeeAgent objects
        from .colony import Colony

        if isinstance(agent, Colony):
            # Colonies are always active unless they have collapsed
            return hasattr(agent, "health") and agent.health != "collapsed"

        if agent.status == BeeStatus.DEAD:
            return False

        # Energy-based activity
        if agent.energy <= 10.0:
            return False

        # Weather-based activity (for foragers)
        if hasattr(agent, "role") and agent.role == BeeRole.FORAGER:
            weather = self.model.current_weather
            if (
                self.model.resource_distribution.landscape.calculate_weather_factor(
                    weather
                )
                < 0.5
            ):
                return False

        # Seasonal activity (reduced in winter)
        seasonal_factor = self.model.get_seasonal_factor()
        if seasonal_factor < 0.5 and self.model.random.random() > seasonal_factor:
            return False

        # Age-based activity (very old or very young bees less active)
        if agent.age < 3 or agent.age > 60:
            return bool(self.model.random.random() < 0.7)

        return True

    def _process_dirty_flags(self) -> None:
        """Process dirty flags for incremental updates"""
        
        # Update status lookups for dirty agents
        for agent in self.status_dirty_agents:
            old_status = self.agent_status_lookup.get(agent.unique_id)
            new_status = agent.status
            
            if old_status != new_status:
                # Remove from old status set
                if old_status is not None:
                    self.agents_by_status[old_status].discard(agent)
                
                # Add to new status set
                self.agents_by_status[new_status].add(agent)
                self.agent_status_lookup[agent.unique_id] = new_status
        
        # Update role lookups for dirty agents  
        for agent in self.role_dirty_agents:
            old_role = self.agent_role_lookup.get(agent.unique_id)
            new_role = agent.role
            
            if old_role != new_role:
                # Remove from old role set
                if old_role is not None:
                    self.agents_by_role[old_role].discard(agent)
                    self.active_agents_by_role[old_role].discard(agent)
                
                # Add to new role set
                self.agents_by_role[new_role].add(agent)
                if agent in self.active_agents and agent.status != BeeStatus.DEAD:
                    self.active_agents_by_role[new_role].add(agent)
                    
                self.agent_role_lookup[agent.unique_id] = new_role
        
        # Update energy lookups for all agents (fast operation)
        for agent in self.agents:
            if hasattr(agent, "energy"):
                self.agent_energy_lookup[agent.unique_id] = agent.energy
        
        # Clear dirty flags
        self.status_dirty_agents.clear()
        self.role_dirty_agents.clear()

    def _activate_role_agents(self, role: BeeRole, agents: List[BeeAgent]) -> None:
        """Activate agents of specific role"""
        if not agents:
            return

        # Role-specific activation strategies
        if role == BeeRole.QUEEN:
            # Queens always activate deterministically
            for agent in agents:
                agent.step()
                self.activation_counts[role.value] += 1

        elif role == BeeRole.FORAGER:
            # Foragers activate based on colony needs and weather
            colony_needs_foraging = self._assess_foraging_needs()

            if colony_needs_foraging:
                # Activate all foragers when colony needs resources
                for agent in agents:
                    agent.step()
                    self.activation_counts[role.value] += 1
            else:
                # Activate subset of foragers for maintenance
                num_to_activate = max(1, len(agents) // 3)
                selected_agents = self.model.random.sample(agents, num_to_activate)

                for agent in selected_agents:
                    agent.step()
                    self.activation_counts[role.value] += 1

        elif role in [BeeRole.NURSE, BeeRole.GUARD, BeeRole.BUILDER]:
            # House bees activate based on colony tasks
            task_urgency = self._assess_house_bee_tasks(role)

            # Stochastic activation based on task urgency
            activation_probability = 0.5 + 0.5 * task_urgency

            for agent in agents:
                if self.model.random.random() < activation_probability:
                    agent.step()
                    self.activation_counts[role.value] += 1

        elif role == BeeRole.DRONE:
            # Drones activate less frequently except during mating season
            if self.model.is_mating_season():
                activation_probability = 0.8
            else:
                activation_probability = 0.3

            for agent in agents:
                if self.model.random.random() < activation_probability:
                    agent.step()
                    self.activation_counts[role.value] += 1

    def _assess_foraging_needs(self) -> bool:
        """Assess if colony needs more foraging activity"""
        if not self.model.colonies:
            return False

        colony = self.model.colonies[0]  # Assuming single colony for now

        # Check resource levels
        resource_adequacy = colony.get_resource_adequacy()

        # Check population needs
        needs = colony.assess_needs()

        return bool(resource_adequacy < 0.7 or needs.food > 0.5)

    def _assess_house_bee_tasks(self, role: BeeRole) -> float:
        """Assess urgency of house bee tasks (0-1)"""
        if not self.model.colonies:
            return 0.0

        colony = self.model.colonies[0]
        colony.assess_needs()

        if role == BeeRole.NURSE:
            # Urgency based on brood-to-nurse ratio
            brood_count = colony.get_brood_count()
            nurse_count = len(colony.get_bees_by_role(BeeRole.NURSE))

            if nurse_count == 0:
                return 1.0

            brood_per_nurse = brood_count / nurse_count
            return float(
                min(1.0, brood_per_nurse / 50.0)
            )  # Optimal ~50 brood per nurse

        elif role == BeeRole.GUARD:
            # Guards needed more during high activity periods
            season = self.model.get_current_season()
            if season in ["spring", "summer"]:
                return 0.7
            else:
                return 0.3

        elif role == BeeRole.BUILDER:
            # Builders needed when colony is growing
            if colony.health.value in ["thriving", "healthy"]:
                return 0.6
            else:
                return 0.2

        return 0.5

    def _cleanup_dead_agents(self) -> None:
        """Remove dead agents from scheduler"""
        from .agents import BeeAgent

        dead_agents = [
            agent
            for agent in self.agents
            if isinstance(agent, BeeAgent) and agent.status == BeeStatus.DEAD
        ]

        for agent in dead_agents:
            self.remove(agent)

    def get_agent_count_by_role(self) -> Dict[BeeRole, int]:
        """Get count of agents by role"""
        counts = {}

        for role in BeeRole:
            counts[role] = len(self.agents_by_role[role])

        return counts

    def get_activation_statistics(self) -> Dict[str, Any]:
        """Get activation statistics"""
        total_activations = sum(self.activation_counts.values())

        stats: Dict[str, Any] = {
            "total_agents": len(self.agents),
            "active_agents": len(self.active_agents),
            "inactive_agents": len(self.inactive_agents),
            "total_activations": total_activations,
            "activations_by_role": dict(self.activation_counts),
            "activation_rates": {},
        }

        # Calculate activation rates
        for role_str, count in self.activation_counts.items():
            role_enum = BeeRole(role_str)
            role_agents = len(self.agents_by_role[role_enum])
            if role_agents > 0:
                stats["activation_rates"][role_str] = count / (role_agents * self.steps)
            else:
                stats["activation_rates"][role_str] = 0.0

        return stats

    def reset_activation_counts(self) -> None:
        """Reset activation counters"""
        self.activation_counts.clear()

    def get_agents_by_status(self, status: BeeStatus) -> List[BeeAgent]:
        """Get agents with specific status using optimized lookup"""
        return list(self.agents_by_status[status])

    def mark_agent_status_dirty(self, agent: Any) -> None:
        """Mark agent as having dirty status for next update cycle"""
        if hasattr(agent, "status"):
            self.status_dirty_agents.add(agent)
    
    def mark_agent_role_dirty(self, agent: Any) -> None:
        """Mark agent as having dirty role for next update cycle"""
        if hasattr(agent, "role"):
            self.role_dirty_agents.add(agent)
    
    def get_fast_agent_count_by_role(self) -> Dict[BeeRole, int]:
        """Get fast agent count by role using optimized sets"""
        return {role: len(agents) for role, agents in self.agents_by_role.items()}
    
    def get_fast_agent_count_by_status(self) -> Dict[BeeStatus, int]:
        """Get fast agent count by status using optimized sets"""
        return {status: len(agents) for status, agents in self.agents_by_status.items()}

    def get_most_active_agents(self, n: int = 10) -> List[BeeAgent]:
        """Get most active agents (placeholder for detailed tracking)"""
        # In full implementation, would track individual agent activity
        return list(self.active_agents)[:n]

    def optimize_scheduling(self) -> None:
        """Optimize scheduling performance"""
        # Remove empty role groups
        empty_roles = [
            role for role, agents in self.agents_by_role.items() if not agents
        ]
        for role in empty_roles:
            del self.agents_by_role[role]

        # Update role order based on current population
        role_counts = self.get_agent_count_by_role()
        self.role_order = sorted(
            role_counts.keys(), key=lambda r: role_counts[r], reverse=True
        )

    def __repr__(self) -> str:
        return (
            f"BeeScheduler(agents={len(self.agents)}, "
            f"active={len(self.active_agents)}, "
            f"step={self.steps})"
        )
