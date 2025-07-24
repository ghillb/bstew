"""
Information Sharing Models
==========================

Models for information sharing and social learning in bee colonies,
including information decay, sharing range, and accuracy improvement.
"""

from typing import Dict, Any, List
from collections import defaultdict, deque


class InformationSharingModel:
    """Model for information sharing between bees"""

    def __init__(
        self,
        information_decay_rate: float = 0.95,
        sharing_range: float = 10.0,
        discovery_bonus: float = 0.3,
        novelty_weight: float = 0.4,
        accuracy_improvement_rate: float = 0.05,
    ):
        self.information_decay_rate = information_decay_rate
        self.sharing_range = sharing_range
        self.discovery_bonus = discovery_bonus
        self.novelty_weight = novelty_weight
        self.accuracy_improvement_rate = accuracy_improvement_rate


class ColonyInformationNetwork:
    """Network model for colony information flow"""

    def __init__(
        self, colony_id: int, bee_ids: List[int] = None, max_history: int = 1000
    ):
        self.colony_id = colony_id
        self.information_flows: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.active_recruitments: Dict[str, Any] = {}
        self.collective_knowledge: Dict[int, Dict[str, Any]] = {}
        self.information_reliability: Dict[int, float] = {}
        self.social_network: Dict[int, set] = defaultdict(set)
        self.learning_history = deque(maxlen=max_history)

        # Initialize social network for provided bees
        if bee_ids:
            for bee_id in bee_ids:
                self.social_network[bee_id] = set()


class RecruitmentMechanismManager:
    """Manager for recruitment mechanisms"""

    def __init__(self):
        # Import dance model from honey bee communication
        try:
            from .honey_bee_communication import DanceDecisionModel

            self.dance_decision_model = DanceDecisionModel()
        except ImportError:
            # Create mock for testing
            self.dance_decision_model = type(
                "MockDanceModel",
                (),
                {
                    "min_quality_threshold": 0.5,
                    "min_profitability_threshold": 0.3,
                    "distance_factor": 0.8,
                    "vigor_quality_weight": 0.6,
                    "vigor_profitability_weight": 0.3,
                    "vigor_urgency_weight": 0.1,
                },
            )()

        try:
            from .honey_bee_communication import RecruitmentModel

            self.recruitment_model = RecruitmentModel()
        except ImportError:
            # Create mock for testing
            self.recruitment_model = type("MockRecruitmentModel", (), {})()

        self.information_sharing_model = InformationSharingModel()
        self.colony_networks: Dict[int, ColonyInformationNetwork] = {}
        self.active_recruitments: Dict[str, Any] = {}
        self.recruitment_history = deque(maxlen=5000)
        self.recruitment_success_rates: Dict[int, float] = defaultdict(float)
        self.information_flow_efficiency: Dict[int, float] = defaultdict(float)

        # Configuration parameters
        self.max_recruitment_distance = 150.0
        self.information_decay_rate = 0.02
        self.social_learning_rate = 0.1
        self.network_update_interval = 100

    def initialize_colony_network(self, colony_id: int, bee_ids: List[int]):
        """Initialize a colony information network"""
        self.colony_networks[colony_id] = ColonyInformationNetwork(colony_id, bee_ids)
