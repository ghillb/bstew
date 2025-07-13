"""
Unit tests for BeeScheduler class
==================================

Tests for the custom scheduler managing bee agent activation and time steps.
"""

from unittest.mock import Mock
from collections import defaultdict

from src.bstew.core.scheduler import BeeScheduler
from src.bstew.core.agents import Queen, Worker, Forager, BeeRole, BeeStatus


class TestBeeSchedulerInitialization:
    """Test BeeScheduler initialization"""

    def setup_method(self):
        """Setup test fixtures"""
        self.model = Mock()
        self.model.running = True
        self.model.current_step = 0
        # Create mock colony for house bee task assessment
        mock_colony = Mock()
        mock_colony.assess_needs = Mock(return_value=Mock(food=0.5))
        mock_colony.get_brood_count = Mock(return_value=100)
        mock_colony.get_bees_by_role = Mock(return_value=[Mock(), Mock()])  # 2 nurses
        mock_colony.health = Mock()
        mock_colony.health.value = "healthy"

        self.model.colonies = [mock_colony]
        self.model.current_weather = Mock()
        self.model.resource_distribution = Mock()
        self.model.resource_distribution.landscape = Mock()
        self.model.resource_distribution.landscape.calculate_weather_factor = Mock(
            return_value=0.8
        )
        self.model.get_seasonal_factor = Mock(return_value=0.8)
        self.model.random = Mock()
        self.model.random.random = Mock(return_value=0.5)
        self.model.is_mating_season = Mock(return_value=False)
        self.model.get_current_season = Mock(return_value="spring")

        self.scheduler = BeeScheduler(self.model)

    def test_scheduler_initialization(self):
        """Test basic scheduler initialization"""
        assert self.scheduler.model == self.model
        assert self.scheduler.time == 0
        assert len(self.scheduler.agents) == 0
        assert isinstance(self.scheduler.agents_by_role, dict)
        assert isinstance(self.scheduler.active_agents, set)
        assert isinstance(self.scheduler.inactive_agents, set)

    def test_role_order_initialization(self):
        """Test role activation order initialization"""
        expected_order = [
            BeeRole.QUEEN,
            BeeRole.NURSE,
            BeeRole.FORAGER,
            BeeRole.GUARD,
            BeeRole.BUILDER,
            BeeRole.DRONE,
        ]

        assert self.scheduler.role_order == expected_order

    def test_agents_by_role_initialization(self):
        """Test agents by role dictionary initialization"""
        # Should have empty lists for each role when accessed
        for role in self.scheduler.role_order:
            assert len(self.scheduler.agents_by_role[role]) == 0

    def test_performance_counters_initialization(self):
        """Test performance counters initialization"""
        assert hasattr(self.scheduler, "activation_counts")
        assert isinstance(self.scheduler.activation_counts, defaultdict)


class TestBeeSchedulerAgentManagement:
    """Test BeeScheduler agent management"""

    def setup_method(self):
        """Setup test fixtures"""
        self.model = Mock()
        self.model.running = True
        self.model.current_step = 0
        # Create mock colony for house bee task assessment
        mock_colony = Mock()
        mock_colony.assess_needs = Mock(return_value=Mock(food=0.5))
        mock_colony.get_brood_count = Mock(return_value=100)
        mock_colony.get_bees_by_role = Mock(return_value=[Mock(), Mock()])  # 2 nurses
        mock_colony.health = Mock()
        mock_colony.health.value = "healthy"

        self.model.colonies = [mock_colony]
        self.model.current_weather = Mock()
        self.model.resource_distribution = Mock()
        self.model.resource_distribution.landscape = Mock()
        self.model.resource_distribution.landscape.calculate_weather_factor = Mock(
            return_value=0.8
        )
        self.model.get_seasonal_factor = Mock(return_value=0.8)
        self.model.random = Mock()
        self.model.random.random = Mock(return_value=0.5)
        self.model.is_mating_season = Mock(return_value=False)
        self.model.get_current_season = Mock(return_value="spring")

        self.scheduler = BeeScheduler(self.model)

        # Create mock agents with required properties
        self.queen = Mock(spec=Queen)
        self.queen.unique_id = 1
        self.queen.role = BeeRole.QUEEN
        self.queen.status = BeeStatus.ALIVE
        self.queen.energy = 50.0
        self.queen.age = 30
        self.queen.step = Mock()

        self.worker = Mock(spec=Worker)
        self.worker.unique_id = 2
        self.worker.role = BeeRole.NURSE
        self.worker.status = BeeStatus.ALIVE
        self.worker.energy = 50.0
        self.worker.age = 30
        self.worker.step = Mock()

        self.forager = Mock(spec=Forager)
        self.forager.unique_id = 3
        self.forager.role = BeeRole.FORAGER
        self.forager.status = BeeStatus.ALIVE
        self.forager.energy = 50.0
        self.forager.age = 30
        self.forager.step = Mock()

    def test_add_agent(self):
        """Test adding agents to scheduler"""
        # Add agents
        self.scheduler.add(self.queen)
        self.scheduler.add(self.worker)
        self.scheduler.add(self.forager)

        # Check agents were added
        assert len(self.scheduler.agents) == 3
        assert self.queen in self.scheduler.agents
        assert self.worker in self.scheduler.agents
        assert self.forager in self.scheduler.agents

    def test_agents_by_role_organization(self):
        """Test agents are organized by role"""
        # Add agents
        self.scheduler.add(self.queen)
        self.scheduler.add(self.worker)
        self.scheduler.add(self.forager)

        # Check role organization
        assert self.queen in self.scheduler.agents_by_role[BeeRole.QUEEN]
        assert self.worker in self.scheduler.agents_by_role[BeeRole.NURSE]
        assert self.forager in self.scheduler.agents_by_role[BeeRole.FORAGER]

    def test_active_agent_tracking(self):
        """Test active agent tracking"""
        # Add active agents
        self.scheduler.add(self.queen)
        self.scheduler.add(self.worker)

        # Check active agents
        assert self.queen in self.scheduler.active_agents
        assert self.worker in self.scheduler.active_agents

    def test_remove_agent(self):
        """Test removing agents from scheduler"""
        # Add and remove agent
        self.scheduler.add(self.worker)
        self.scheduler.remove(self.worker)

        # Check agent was removed
        assert self.worker not in self.scheduler.agents
        assert self.worker not in self.scheduler.agents_by_role[BeeRole.NURSE]
        assert self.worker not in self.scheduler.active_agents

    def test_agent_count_by_role(self):
        """Test agent count by role"""
        # Add agents
        self.scheduler.add(self.queen)
        self.scheduler.add(self.worker)
        self.scheduler.add(self.forager)

        # Check counts
        assert len(self.scheduler.agents_by_role[BeeRole.QUEEN]) == 1
        assert len(self.scheduler.agents_by_role[BeeRole.NURSE]) == 1
        assert len(self.scheduler.agents_by_role[BeeRole.FORAGER]) == 1
        assert len(self.scheduler.agents_by_role[BeeRole.DRONE]) == 0

    def test_total_agent_count(self):
        """Test total agent count"""
        # Add agents
        self.scheduler.add(self.queen)
        self.scheduler.add(self.worker)
        self.scheduler.add(self.forager)

        # Check total count
        assert len(self.scheduler.agents) == 3


class TestBeeSchedulerStepExecution:
    """Test BeeScheduler step execution"""

    def setup_method(self):
        """Setup test fixtures"""
        self.model = Mock()
        self.model.running = True
        self.model.current_step = 0
        # Create mock colony for house bee task assessment
        mock_colony = Mock()
        mock_colony.assess_needs = Mock(return_value=Mock(food=0.5))
        mock_colony.get_brood_count = Mock(return_value=100)
        mock_colony.get_bees_by_role = Mock(return_value=[Mock(), Mock()])  # 2 nurses
        mock_colony.health = Mock()
        mock_colony.health.value = "healthy"

        self.model.colonies = [mock_colony]
        self.model.current_weather = Mock()
        self.model.resource_distribution = Mock()
        self.model.resource_distribution.landscape = Mock()
        self.model.resource_distribution.landscape.calculate_weather_factor = Mock(
            return_value=0.8
        )
        self.model.get_seasonal_factor = Mock(return_value=0.8)
        self.model.random = Mock()
        self.model.random.random = Mock(return_value=0.5)
        self.model.is_mating_season = Mock(return_value=False)
        self.model.get_current_season = Mock(return_value="spring")

        self.scheduler = BeeScheduler(self.model)

        # Create mock agents with step methods
        self.queen = Mock(spec=Queen)
        self.queen.unique_id = 1
        self.queen.role = BeeRole.QUEEN
        self.queen.status = BeeStatus.ALIVE
        self.queen.energy = 50.0
        self.queen.age = 30
        self.queen.step = Mock()

        self.worker = Mock(spec=Worker)
        self.worker.unique_id = 2
        self.worker.role = BeeRole.NURSE
        self.worker.status = BeeStatus.ALIVE
        self.worker.energy = 50.0
        self.worker.age = 30
        self.worker.step = Mock()

    def test_basic_step_execution(self):
        """Test basic step execution"""
        # Add agents
        self.scheduler.add(self.queen)
        self.scheduler.add(self.worker)

        # Execute step
        self.scheduler.step()

        # Check queen was activated (deterministic)
        self.queen.step.assert_called_once()

        # Worker activation depends on probability, so check it was called at least once over multiple steps
        self.worker.step.reset_mock()
        for _ in range(10):
            self.scheduler.step()

        # Should have been called at least once in 10 steps
        assert self.worker.step.call_count >= 1

    def test_dead_agent_exclusion(self):
        """Test dead agents are excluded from activation"""
        # Create dead agent
        dead_agent = Mock(spec=Worker)
        dead_agent.unique_id = 5
        dead_agent.role = BeeRole.NURSE
        dead_agent.status = BeeStatus.DEAD
        dead_agent.energy = 50.0
        dead_agent.age = 30
        dead_agent.step = Mock()

        # Add dead agent
        self.scheduler.add(dead_agent)

        # Execute step
        self.scheduler.step()

        # Dead agent should not be activated
        dead_agent.step.assert_not_called()

    def test_time_increment(self):
        """Test time increment during step"""
        initial_time = self.scheduler.time

        # Execute step
        self.scheduler.step()

        # Check time incremented
        assert self.scheduler.time == initial_time + 1

    def test_activation_count_tracking(self):
        """Test activation count tracking"""
        # Add agents
        self.scheduler.add(self.queen)
        self.scheduler.add(self.worker)

        # Execute step
        self.scheduler.step()

        # Check activation counts
        assert self.scheduler.activation_counts[BeeRole.QUEEN.value] == 1
        assert (
            self.scheduler.activation_counts[BeeRole.NURSE.value] >= 0
        )  # May vary based on probability

    def test_multiple_steps(self):
        """Test multiple step execution"""
        # Add agents
        self.scheduler.add(self.queen)
        self.scheduler.add(self.worker)

        # Execute multiple steps
        for _ in range(3):
            self.scheduler.step()

        # Check time incremented
        assert self.scheduler.time == 3

        # Check queens were activated (deterministic)
        assert self.queen.step.call_count == 3


class TestBeeSchedulerUtilityMethods:
    """Test BeeScheduler utility methods"""

    def setup_method(self):
        """Setup test fixtures"""
        self.model = Mock()
        self.model.running = True
        self.model.current_step = 0
        # Create mock colony for house bee task assessment
        mock_colony = Mock()
        mock_colony.assess_needs = Mock(return_value=Mock(food=0.5))
        mock_colony.get_brood_count = Mock(return_value=100)
        mock_colony.get_bees_by_role = Mock(return_value=[Mock(), Mock()])  # 2 nurses
        mock_colony.health = Mock()
        mock_colony.health.value = "healthy"

        self.model.colonies = [mock_colony]
        self.model.current_weather = Mock()
        self.model.resource_distribution = Mock()
        self.model.resource_distribution.landscape = Mock()
        self.model.resource_distribution.landscape.calculate_weather_factor = Mock(
            return_value=0.8
        )
        self.model.get_seasonal_factor = Mock(return_value=0.8)
        self.model.random = Mock()
        self.model.random.random = Mock(return_value=0.5)
        self.model.is_mating_season = Mock(return_value=False)
        self.model.get_current_season = Mock(return_value="spring")

        self.scheduler = BeeScheduler(self.model)

    def test_get_agent_count_by_role(self):
        """Test getting agent count by role"""
        # Create and add agents
        queen = Mock(spec=Queen)
        queen.unique_id = 1
        queen.role = BeeRole.QUEEN
        queen.status = BeeStatus.ALIVE
        queen.energy = 50.0
        queen.age = 30

        worker = Mock(spec=Worker)
        worker.unique_id = 2
        worker.role = BeeRole.NURSE
        worker.status = BeeStatus.ALIVE
        worker.energy = 50.0
        worker.age = 30

        self.scheduler.add(queen)
        self.scheduler.add(worker)

        # Get counts
        counts = self.scheduler.get_agent_count_by_role()

        # Check counts
        assert counts[BeeRole.QUEEN] == 1
        assert counts[BeeRole.NURSE] == 1
        assert counts[BeeRole.FORAGER] == 0

    def test_get_activation_statistics(self):
        """Test getting activation statistics"""
        # Create and add agents
        queen = Mock(spec=Queen)
        queen.unique_id = 1
        queen.role = BeeRole.QUEEN
        queen.status = BeeStatus.ALIVE
        queen.energy = 50.0
        queen.age = 30
        queen.step = Mock()

        self.scheduler.add(queen)

        # Execute step
        self.scheduler.step()

        # Get statistics
        stats = self.scheduler.get_activation_statistics()

        # Check statistics
        assert "total_agents" in stats
        assert "active_agents" in stats
        assert "total_activations" in stats
        assert stats["total_agents"] == 1

    def test_reset_activation_counts(self):
        """Test resetting activation counts"""
        # Create and add agents
        queen = Mock(spec=Queen)
        queen.unique_id = 1
        queen.role = BeeRole.QUEEN
        queen.status = BeeStatus.ALIVE
        queen.energy = 50.0
        queen.age = 30
        queen.step = Mock()

        self.scheduler.add(queen)

        # Execute step
        self.scheduler.step()

        # Reset counts
        self.scheduler.reset_activation_counts()

        # Check counts are reset
        assert len(self.scheduler.activation_counts) == 0

    def test_get_agents_by_status(self):
        """Test getting agents by status"""
        # Create agents with different statuses
        alive_agent = Mock(spec=Worker)
        alive_agent.unique_id = 1
        alive_agent.role = BeeRole.NURSE
        alive_agent.status = BeeStatus.ALIVE
        alive_agent.energy = 50.0
        alive_agent.age = 30

        dead_agent = Mock(spec=Worker)
        dead_agent.unique_id = 2
        dead_agent.role = BeeRole.NURSE
        dead_agent.status = BeeStatus.DEAD
        dead_agent.energy = 50.0
        dead_agent.age = 30

        self.scheduler.add(alive_agent)
        self.scheduler.add(dead_agent)

        # Get agents by status
        alive_agents = self.scheduler.get_agents_by_status(BeeStatus.ALIVE)
        dead_agents = self.scheduler.get_agents_by_status(BeeStatus.DEAD)

        # Check results
        assert alive_agent in alive_agents
        assert dead_agent in dead_agents
        assert len(alive_agents) == 1
        assert len(dead_agents) == 1

    def test_scheduler_repr(self):
        """Test scheduler string representation"""
        repr_str = repr(self.scheduler)

        assert "BeeScheduler" in repr_str
        assert "agents=" in repr_str
        assert "active=" in repr_str
        assert "step=" in repr_str
