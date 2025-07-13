"""
Unit tests for bee agent classes
================================

Comprehensive tests for agent classes and their functionality.
"""

from unittest.mock import Mock

from src.bstew.core.agents import (
    BeeAgent,
    Queen,
    Worker,
    Forager,
    Drone,
    BeeRole,
    BeeStatus,
)


class TestBeeAgentBasics:
    """Test basic BeeAgent functionality"""

    def test_bee_agent_creation(self):
        """Test basic bee agent creation"""
        model = Mock()
        agent = BeeAgent(unique_id=1, model=model)

        assert agent.unique_id == 1
        assert agent.model == model
        assert agent.pos is None

    def test_bee_agent_with_age(self):
        """Test bee agent creation with age"""
        model = Mock()
        agent = BeeAgent(unique_id=1, model=model, age=5)

        assert agent.age == 5

    def test_bee_agent_energy_system(self):
        """Test bee agent energy system"""
        model = Mock()
        agent = BeeAgent(unique_id=1, model=model)

        # Should have initial energy
        assert hasattr(agent, "energy")
        assert agent.energy >= 0

    def test_bee_agent_step_method(self):
        """Test bee agent step method exists"""
        model = Mock()
        agent = BeeAgent(unique_id=1, model=model)

        # Should have step method
        assert hasattr(agent, "step")
        assert callable(agent.step)


class TestQueenBasics:
    """Test basic Queen functionality"""

    def test_queen_creation(self):
        """Test basic queen creation"""
        model = Mock()
        colony = Mock()

        queen = Queen(unique_id=1, model=model, colony=colony)

        assert queen.unique_id == 1
        assert queen.model == model
        assert queen.colony == colony

    def test_queen_egg_laying(self):
        """Test queen egg laying capability"""
        model = Mock()
        colony = Mock()

        queen = Queen(unique_id=1, model=model, colony=colony)

        # Should have egg laying method
        assert hasattr(queen, "lay_eggs")
        assert callable(queen.lay_eggs)

    def test_queen_pheromone_production(self):
        """Test queen pheromone production"""
        model = Mock()
        colony = Mock()

        queen = Queen(unique_id=1, model=model, colony=colony)

        # Should have pheromone production
        assert hasattr(queen, "produce_pheromones")
        assert callable(queen.produce_pheromones)

    def test_queen_spermatheca(self):
        """Test queen spermatheca system"""
        model = Mock()
        colony = Mock()

        queen = Queen(unique_id=1, model=model, colony=colony)

        # Should have spermatheca system
        assert hasattr(queen, "spermatheca")


class TestWorkerBasics:
    """Test basic Worker functionality"""

    def test_worker_creation(self):
        """Test basic worker creation"""
        model = Mock()
        colony = Mock()

        worker = Worker(unique_id=1, model=model, colony=colony)

        assert worker.unique_id == 1
        assert worker.model == model
        assert worker.colony == colony

    def test_worker_task_performance(self):
        """Test worker task performance"""
        model = Mock()
        colony = Mock()

        worker = Worker(unique_id=1, model=model, colony=colony)

        # Should have task execution capability
        assert hasattr(worker, "execute_role_behavior")
        assert callable(worker.execute_role_behavior)


class TestForagerBasics:
    """Test basic Forager functionality"""

    def test_forager_creation(self):
        """Test basic forager creation"""
        model = Mock()
        colony = Mock()

        forager = Forager(unique_id=1, model=model, colony=colony)

        assert forager.unique_id == 1
        assert forager.model == model
        assert forager.colony == colony

    def test_forager_distance_calculation(self):
        """Test forager distance calculation"""
        model = Mock()
        colony = Mock()

        forager = Forager(unique_id=1, model=model, colony=colony)

        # Should have distance calculation
        assert hasattr(forager, "get_distance_to")
        assert callable(forager.get_distance_to)


class TestDroneBasics:
    """Test basic Drone functionality"""

    def test_drone_creation(self):
        """Test basic drone creation"""
        model = Mock()
        colony = Mock()

        drone = Drone(unique_id=1, model=model, colony=colony)

        assert drone.unique_id == 1
        assert drone.model == model
        assert drone.colony == colony


class TestBeeRoleEnum:
    """Test BeeRole enumeration"""

    def test_bee_role_values(self):
        """Test BeeRole enum values"""
        assert BeeRole.QUEEN.value == "queen"
        assert BeeRole.NURSE.value == "nurse"
        assert BeeRole.FORAGER.value == "forager"
        assert BeeRole.GUARD.value == "guard"
        assert BeeRole.BUILDER.value == "builder"
        assert BeeRole.DRONE.value == "drone"

    def test_bee_role_members(self):
        """Test BeeRole enum members"""
        roles = list(BeeRole)
        assert len(roles) == 7
        assert BeeRole.QUEEN in roles
        assert BeeRole.DRONE in roles
        assert BeeRole.WORKER in roles


class TestBeeStatusEnum:
    """Test BeeStatus enumeration"""

    def test_bee_status_values(self):
        """Test BeeStatus enum values"""
        assert BeeStatus.ALIVE.value == "alive"
        assert BeeStatus.DEAD.value == "dead"
        assert BeeStatus.FORAGING.value == "foraging"
        assert BeeStatus.NURSING.value == "nursing"
        assert BeeStatus.DANCING.value == "dancing"
        assert BeeStatus.RESTING.value == "resting"

    def test_bee_status_members(self):
        """Test BeeStatus enum members"""
        statuses = list(BeeStatus)
        assert len(statuses) == 21  # Updated with 15+ activity states
        assert BeeStatus.ALIVE in statuses
        assert BeeStatus.DEAD in statuses
        # Check some of the activity states
        assert BeeStatus.HIBERNATING in statuses
        assert BeeStatus.NECTAR_FORAGING in statuses
        assert BeeStatus.POLLEN_FORAGING in statuses
        assert BeeStatus.EGG_LAYING in statuses
        assert BeeStatus.NURSING in statuses


class TestAgentInteractions:
    """Test basic agent interactions"""

    def test_agent_model_reference(self):
        """Test agents maintain model reference"""
        model = Mock()

        queen = Queen(unique_id=1, model=model, colony=Mock())
        worker = Worker(unique_id=2, model=model, colony=Mock())

        assert queen.model == model
        assert worker.model == model
        assert queen.model == worker.model

    def test_agent_colony_reference(self):
        """Test agents maintain colony reference"""
        model = Mock()
        colony = Mock()

        queen = Queen(unique_id=1, model=model, colony=colony)
        worker = Worker(unique_id=2, model=model, colony=colony)

        assert queen.colony == colony
        assert worker.colony == colony
        assert queen.colony == worker.colony

    def test_unique_id_assignment(self):
        """Test unique ID assignment"""
        model = Mock()
        colony = Mock()

        agents = [
            Queen(unique_id=1, model=model, colony=colony),
            Worker(unique_id=2, model=model, colony=colony),
            Forager(unique_id=3, model=model, colony=colony),
            Drone(unique_id=4, model=model, colony=colony),
        ]

        unique_ids = [agent.unique_id for agent in agents]
        assert len(set(unique_ids)) == 4  # All IDs should be unique
        assert unique_ids == [1, 2, 3, 4]

    def test_agent_step_methods(self):
        """Test all agents have step methods"""
        model = Mock()
        colony = Mock()

        agents = [
            Queen(unique_id=1, model=model, colony=colony),
            Worker(unique_id=2, model=model, colony=colony),
            Forager(unique_id=3, model=model, colony=colony),
            Drone(unique_id=4, model=model, colony=colony),
        ]

        for agent in agents:
            assert hasattr(agent, "step")
            assert callable(agent.step)

    def test_agent_energy_systems(self):
        """Test all agents have energy systems"""
        model = Mock()
        colony = Mock()

        agents = [
            Queen(unique_id=1, model=model, colony=colony),
            Worker(unique_id=2, model=model, colony=colony),
            Forager(unique_id=3, model=model, colony=colony),
            Drone(unique_id=4, model=model, colony=colony),
        ]

        for agent in agents:
            assert hasattr(agent, "energy")
            assert agent.energy >= 0

    def test_agent_mortality_system(self):
        """Test all agents have mortality system"""
        model = Mock()
        colony = Mock()

        agents = [
            Queen(unique_id=1, model=model, colony=colony),
            Worker(unique_id=2, model=model, colony=colony),
            Forager(unique_id=3, model=model, colony=colony),
            Drone(unique_id=4, model=model, colony=colony),
        ]

        for agent in agents:
            assert hasattr(agent, "age")
            assert agent.age >= 0
