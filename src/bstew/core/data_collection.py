"""
Comprehensive Data Collection System for NetLogo BEE-STEWARD v2 Parity
=====================================================================

Advanced data collection matching NetLogo's comprehensive tracking capabilities
including bee-level metrics, colony-level statistics, and environmental monitoring.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from dataclasses import dataclass, field
import json
import csv
import time
import logging
from collections import defaultdict, deque
import statistics
import tempfile

from .enums import BeeStatus
from .foraging_integration import IntegratedForagingSystem


class DataCollectionFrequency(Enum):
    """Data collection frequency options"""
    EVERY_STEP = "every_step"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    ON_EVENT = "on_event"


class MetricType(Enum):
    """Types of metrics collected"""
    POPULATION = "population"
    ACTIVITY = "activity"
    FORAGING = "foraging"
    ENERGY = "energy"
    MORTALITY = "mortality"
    DEVELOPMENT = "development"
    GENETIC = "genetic"
    ENVIRONMENTAL = "environmental"
    PERFORMANCE = "performance"
    COMMUNICATION = "communication"
    SPATIAL = "spatial"
    BEHAVIORAL = "behavioral"
    EFFICIENCY = "efficiency"


@dataclass
class DataPoint:
    """Individual data point with metadata"""
    timestamp: int
    metric_name: str
    value: Union[int, float, str, bool]
    agent_id: Optional[int] = None
    colony_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_type: Optional[str] = None
    aggregation_level: str = "individual"  # individual, colony, model


@dataclass
class SimulationEvent:
    """Simulation event for detailed tracking"""
    event_id: str
    event_type: str
    timestamp: float
    source_agent_id: Optional[int] = None
    target_agent_id: Optional[int] = None
    colony_id: Optional[int] = None
    event_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    duration: Optional[float] = None
    energy_impact: Optional[float] = None
    spatial_location: Optional[Tuple[float, float]] = None


@dataclass
class AggregationRule:
    """Rule for data aggregation"""
    metric_name: str
    aggregation_function: str  # mean, sum, count, min, max, std, median
    time_window: int  # steps to aggregate over
    grouping: List[str]  # fields to group by
    condition: Optional[str] = None  # filter condition
    weight_field: Optional[str] = None  # field to use for weighted aggregation


class BeeMetrics(BaseModel):
    """Individual bee metrics tracking"""
    
    model_config = {"validate_assignment": True}
    
    bee_id: int = Field(description="Unique bee identifier")
    
    # Basic demographics
    age: int = Field(default=0, ge=0, description="Age in time steps")
    role: str = Field(default="nurse", description="Current role")
    status: str = Field(default="alive", description="Current status")
    
    # Activity tracking
    total_steps: int = Field(default=0, ge=0, description="Total simulation steps lived")
    activity_time: Dict[str, int] = Field(default_factory=dict, description="Time spent in each activity")
    state_transitions: int = Field(default=0, ge=0, description="Number of state transitions")
    
    # Foraging metrics
    foraging_trips: int = Field(default=0, ge=0, description="Total foraging trips")
    successful_foraging: int = Field(default=0, ge=0, description="Successful foraging attempts")
    energy_collected: float = Field(default=0.0, ge=0.0, description="Total energy collected")
    patches_discovered: int = Field(default=0, ge=0, description="New patches discovered")
    
    # Energy and physiology
    energy_history: List[float] = Field(default_factory=list, description="Energy level history")
    metabolic_rate: float = Field(default=1.0, ge=0.0, description="Current metabolic rate")
    stress_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Current stress level")
    
    # Social interactions
    dances_performed: int = Field(default=0, ge=0, description="Waggle dances performed")
    dances_followed: int = Field(default=0, ge=0, description="Dances followed")
    communication_events: int = Field(default=0, ge=0, description="Total communication events")
    recruitment_attempts: int = Field(default=0, ge=0, description="Recruitment attempts initiated")
    recruits_obtained: int = Field(default=0, ge=0, description="Successful recruits obtained")
    information_shared: int = Field(default=0, ge=0, description="Information sharing events")
    
    # Enhanced foraging metrics
    foraging_sessions: int = Field(default=0, ge=0, description="Total foraging sessions")
    foraging_session_duration_avg: float = Field(default=0.0, ge=0.0, description="Average session duration")
    patch_visits: int = Field(default=0, ge=0, description="Total patch visits")
    unique_patches_visited: int = Field(default=0, ge=0, description="Number of unique patches visited")
    foraging_efficiency_avg: float = Field(default=0.0, ge=0.0, description="Average foraging efficiency")
    travel_distance_total: float = Field(default=0.0, ge=0.0, description="Total travel distance")
    
    # Detailed foraging trip metrics
    foraging_trip_durations: List[float] = Field(default_factory=list, description="All foraging trip durations")
    foraging_trip_energies: List[float] = Field(default_factory=list, description="Energy gained per trip")
    foraging_trip_distances: List[float] = Field(default_factory=list, description="Distance traveled per trip")
    flowers_visited_per_trip: List[int] = Field(default_factory=list, description="Flowers visited per trip")
    patches_visited_per_trip: List[int] = Field(default_factory=list, description="Patches visited per trip")
    
    # Success rate metrics
    successful_trips: int = Field(default=0, ge=0, description="Number of successful foraging trips")
    failed_trips: int = Field(default=0, ge=0, description="Number of failed foraging trips")
    foraging_success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall foraging success rate")
    
    # Resource collection metrics
    nectar_collected_total: float = Field(default=0.0, ge=0.0, description="Total nectar collected (mg)")
    pollen_collected_total: float = Field(default=0.0, ge=0.0, description="Total pollen collected (mg)")
    nectar_per_trip_avg: float = Field(default=0.0, ge=0.0, description="Average nectar per trip")
    pollen_per_trip_avg: float = Field(default=0.0, ge=0.0, description="Average pollen per trip")
    
    # Behavioral adaptation
    strategy_changes: int = Field(default=0, ge=0, description="Number of strategy changes")
    learning_events: int = Field(default=0, ge=0, description="Learning events")
    adaptation_score: float = Field(default=0.0, ge=0.0, description="Adaptation capability score")
    
    # Mortality data
    death_cause: Optional[str] = Field(default=None, description="Cause of death if applicable")
    lifespan: Optional[int] = Field(default=None, description="Total lifespan if dead")
    
    # Performance metrics
    efficiency_score: float = Field(default=0.0, ge=0.0, description="Overall efficiency rating")
    contribution_score: float = Field(default=0.0, ge=0.0, description="Colony contribution score")


class ColonyMetrics(BaseModel):
    """Colony-level metrics tracking"""
    
    model_config = {"validate_assignment": True}
    
    colony_id: int = Field(description="Unique colony identifier")
    
    # Population dynamics
    total_population: int = Field(default=0, ge=0, description="Current total population")
    population_by_role: Dict[str, int] = Field(default_factory=dict, description="Population by role")
    population_by_age: Dict[int, int] = Field(default_factory=dict, description="Population by age group")
    
    # Enhanced population tracking by life stage
    population_by_life_stage: Dict[str, int] = Field(default_factory=dict, description="Population by life stage")
    egg_count: int = Field(default=0, ge=0, description="Current egg count")
    larva_count: int = Field(default=0, ge=0, description="Current larva count")  
    pupa_count: int = Field(default=0, ge=0, description="Current pupa count")
    adult_count: int = Field(default=0, ge=0, description="Current adult count")
    
    # Daily population history lists
    population_size_day_list: List[int] = Field(default_factory=list, description="Daily population history")
    egg_count_day_list: List[int] = Field(default_factory=list, description="Daily egg count history") 
    larva_count_day_list: List[int] = Field(default_factory=list, description="Daily larva count history")
    pupa_count_day_list: List[int] = Field(default_factory=list, description="Daily pupa count history")
    adult_count_day_list: List[int] = Field(default_factory=list, description="Daily adult count history")
    
    # Birth and death rates
    births_today: int = Field(default=0, ge=0, description="Births in current day")
    deaths_today: int = Field(default=0, ge=0, description="Deaths in current day")
    birth_rate: float = Field(default=0.0, ge=0.0, description="Current birth rate")
    death_rate: float = Field(default=0.0, ge=0.0, description="Current death rate")
    
    # Colony production counting
    total_queens_produced: int = Field(default=0, ge=0, description="Total queens produced to date")
    total_drones_produced: int = Field(default=0, ge=0, description="Total drones produced to date")
    total_workers_produced: int = Field(default=0, ge=0, description="Total workers produced to date")
    queens_produced_today: int = Field(default=0, ge=0, description="Queens produced today")
    drones_produced_today: int = Field(default=0, ge=0, description="Drones produced today")
    workers_produced_today: int = Field(default=0, ge=0, description="Workers produced today")
    
    # Production history lists
    queens_produced_day_list: List[int] = Field(default_factory=list, description="Daily queen production history")
    drones_produced_day_list: List[int] = Field(default_factory=list, description="Daily drone production history")
    workers_produced_day_list: List[int] = Field(default_factory=list, description="Daily worker production history")
    
    # Current population counts by role
    queen_count: int = Field(default=0, ge=0, description="Current number of queens")
    worker_count: int = Field(default=0, ge=0, description="Current number of workers")  
    drone_count: int = Field(default=0, ge=0, description="Current number of drones")
    forager_count: int = Field(default=0, ge=0, description="Current number of foragers")
    nurse_count: int = Field(default=0, ge=0, description="Current number of nurses")
    brood_count: int = Field(default=0, ge=0, description="Current total brood count")
    
    # Daily metrics
    daily_productivity: float = Field(default=0.0, ge=0.0, description="Daily colony productivity")
    foraging_efficiency_today: float = Field(default=0.0, ge=0.0, description="Today's foraging efficiency")
    
    # Activity distribution
    activity_distribution: Dict[str, int] = Field(default_factory=dict, description="Bees by activity")
    foraging_efficiency: float = Field(default=0.0, ge=0.0, description="Colony foraging efficiency")
    
    # Resource management
    total_energy: float = Field(default=0.0, ge=0.0, description="Total colony energy")
    energy_reserves: float = Field(default=0.0, ge=0.0, description="Energy reserves")
    daily_energy_collection: float = Field(default=0.0, ge=0.0, description="Energy collected today")
    daily_energy_consumption: float = Field(default=0.0, ge=0.0, description="Energy consumed today")
    
    # Genetic diversity
    genetic_diversity: float = Field(default=0.0, ge=0.0, le=1.0, description="Genetic diversity index")
    inbreeding_coefficient: float = Field(default=0.0, ge=0.0, le=1.0, description="Inbreeding coefficient")
    
    # Health metrics
    average_stress: float = Field(default=0.0, ge=0.0, le=1.0, description="Average stress level")
    disease_prevalence: float = Field(default=0.0, ge=0.0, le=1.0, description="Disease prevalence")
    
    # Performance indicators
    colony_fitness: float = Field(default=0.0, ge=0.0, description="Overall colony fitness")
    survival_probability: float = Field(default=1.0, ge=0.0, le=1.0, description="Colony survival probability")


class EnvironmentalMetrics(BaseModel):
    """Environmental conditions tracking"""
    
    model_config = {"validate_assignment": True}
    
    # Weather conditions
    temperature: float = Field(default=20.0, description="Current temperature (°C)")
    humidity: float = Field(default=50.0, ge=0.0, le=100.0, description="Humidity percentage")
    wind_speed: float = Field(default=0.0, ge=0.0, description="Wind speed (m/s)")
    precipitation: float = Field(default=0.0, ge=0.0, description="Precipitation (mm)")
    weather_condition: str = Field(default="clear", description="Weather condition")
    
    # Seasonal information
    season: str = Field(default="spring", description="Current season")
    day_of_year: int = Field(default=1, ge=1, le=365, description="Day of year")
    photoperiod: float = Field(default=12.0, ge=0.0, le=24.0, description="Hours of daylight")
    
    # Resource availability
    nectar_availability: float = Field(default=1.0, ge=0.0, description="Nectar availability index")
    pollen_availability: float = Field(default=1.0, ge=0.0, description="Pollen availability index")
    patch_quality: Dict[int, float] = Field(default_factory=dict, description="Quality by patch")
    
    # Ecological factors
    predator_pressure: float = Field(default=0.0, ge=0.0, le=1.0, description="Predator pressure index")
    competition_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Inter-colony competition")
    habitat_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Habitat quality index")


class DataCollectionRule(BaseModel):
    """Rule for automatic data collection"""
    
    model_config = {"validate_assignment": True}
    
    metric_name: str = Field(description="Name of metric to collect")
    frequency: DataCollectionFrequency = Field(description="Collection frequency")
    condition: Optional[str] = Field(default=None, description="Condition for collection")
    aggregation: str = Field(default="none", description="Aggregation method")
    targets: List[str] = Field(default_factory=list, description="Target entities")
    
    @field_validator('aggregation')
    @classmethod
    def validate_aggregation(cls, v: str) -> str:
        valid_methods = ['none', 'sum', 'mean', 'median', 'min', 'max', 'count']
        if v not in valid_methods:
            raise ValueError(f"Aggregation must be one of: {valid_methods}")
        return v


class ComprehensiveDataCollector(BaseModel):
    """Comprehensive data collection system matching NetLogo capabilities"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Configuration
    max_history_length: int = Field(default=10000, ge=100, description="Maximum history length")
    collection_rules: List[DataCollectionRule] = Field(default_factory=list, description="Collection rules")
    
    # Data storage
    bee_metrics: Dict[int, Any] = Field(default_factory=dict)
    colony_metrics: Dict[int, Any] = Field(default_factory=dict)
    environmental_metrics: Any = Field(default_factory=lambda: None)
    
    # Time series data
    time_series_data: Dict[str, Any] = Field(default_factory=dict)
    event_log: List[Any] = Field(default_factory=list)
    
    # Aggregated statistics
    daily_summaries: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    weekly_summaries: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    
    # Performance tracking
    collection_stats: Dict[str, int] = Field(default_factory=dict)
    
    # Event tracking
    simulation_events: List[Any] = Field(default_factory=list)
    event_handlers: Dict[str, List[Any]] = Field(default_factory=dict)
    
    # Aggregation system
    aggregation_rules: List[Any] = Field(default_factory=list)
    aggregated_data: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Real-time streaming data
    live_data_streams: Dict[str, Any] = Field(default_factory=dict)
    
    # Event integration
    foraging_system_integration: Any = Field(default=None)
    
    # Logging field
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        
        # Initialize with actual types after Pydantic validation
        if not self.environmental_metrics:
            self.environmental_metrics = EnvironmentalMetrics()
        
        if not self.time_series_data:
            self.time_series_data = defaultdict(lambda: deque(maxlen=self.max_history_length))
        
        if not self.collection_stats:
            self.collection_stats = defaultdict(int)
            
        if not self.event_handlers:
            self.event_handlers = defaultdict(list)
            
        if not self.aggregated_data:
            self.aggregated_data = defaultdict(dict)
            
        if not self.live_data_streams:
            self.live_data_streams = defaultdict(lambda: deque(maxlen=1000))
        
        # Event integration - now properly declared as field above
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize default collection rules
        self._initialize_default_rules()
        self._initialize_aggregation_rules()
        self._setup_event_handlers()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default data collection rules"""
        
        default_rules = [
            # Population tracking
            DataCollectionRule(
                metric_name="total_population",
                frequency=DataCollectionFrequency.EVERY_STEP,
                aggregation="sum",
                targets=["colony"]
            ),
            
            # Activity distribution
            DataCollectionRule(
                metric_name="activity_distribution",
                frequency=DataCollectionFrequency.HOURLY,
                aggregation="count",
                targets=["bee"]
            ),
            
            # Energy tracking
            DataCollectionRule(
                metric_name="colony_energy",
                frequency=DataCollectionFrequency.EVERY_STEP,
                aggregation="sum",
                targets=["colony"]
            ),
            
            # Foraging success
            DataCollectionRule(
                metric_name="foraging_success_rate",
                frequency=DataCollectionFrequency.DAILY,
                aggregation="mean",
                targets=["bee"]
            ),
            
            # Mortality tracking
            DataCollectionRule(
                metric_name="mortality_rate",
                frequency=DataCollectionFrequency.DAILY,
                aggregation="count",
                targets=["bee"],
                condition="status == 'dead'"
            )
        ]
        
        self.collection_rules.extend(default_rules)
    
    def _initialize_aggregation_rules(self) -> None:
        """Initialize data aggregation rules"""
        
        aggregation_rules = [
            # Population aggregations
            AggregationRule(
                metric_name="population_by_role",
                aggregation_function="count",
                time_window=1,
                grouping=["colony_id", "role"]
            ),
            
            # Foraging efficiency aggregations
            AggregationRule(
                metric_name="foraging_efficiency_trend",
                aggregation_function="mean",
                time_window=24,  # Daily average
                grouping=["colony_id"],
                condition="foraging_efficiency_avg > 0"
            ),
            
            # Communication network aggregations
            AggregationRule(
                metric_name="communication_network_stats",
                aggregation_function="sum",
                time_window=24,
                grouping=["colony_id"],
                condition="communication_events > 0"
            ),
            
            # Energy flow aggregations
            AggregationRule(
                metric_name="energy_flow_analysis",
                aggregation_function="sum",
                time_window=24,
                grouping=["colony_id"],
                weight_field="energy_collected"
            ),
            
            # Behavioral adaptation aggregations
            AggregationRule(
                metric_name="adaptation_metrics",
                aggregation_function="mean",
                time_window=168,  # Weekly average
                grouping=["colony_id", "role"]
            )
        ]
        
        self.aggregation_rules.extend(aggregation_rules)
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for different simulation events"""
        
        # Foraging event handlers
        self.register_event_handler("foraging_session_started", self._handle_foraging_session_start)
        self.register_event_handler("foraging_session_ended", self._handle_foraging_session_end)
        self.register_event_handler("foraging_trip_completed", self._handle_foraging_trip)
        
        # Communication event handlers
        self.register_event_handler("dance_performed", self._handle_dance_event)
        self.register_event_handler("recruitment_initiated", self._handle_recruitment_event)
        self.register_event_handler("information_shared", self._handle_information_sharing)
        
        # State transition handlers
        self.register_event_handler("state_transition", self._handle_state_transition)
        self.register_event_handler("role_change", self._handle_role_change)
        
        # Energy and resource handlers
        self.register_event_handler("energy_change", self._handle_energy_change)
        self.register_event_handler("resource_collection", self._handle_resource_collection)
        
        # Mortality and lifecycle handlers
        self.register_event_handler("bee_death", self._handle_bee_death)
        self.register_event_handler("bee_emergence", self._handle_bee_emergence)
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register event handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def emit_event(self, event: SimulationEvent) -> None:
        """Emit simulation event and trigger handlers"""
        
        # Store event
        self.simulation_events.append(event)
        
        # Update live streams
        stream_key = f"{event.event_type}_events"
        if stream_key not in self.live_data_streams:
            self.live_data_streams[stream_key] = deque(maxlen=1000)
        self.live_data_streams[stream_key].append({
            "timestamp": event.timestamp,
            "event_id": event.event_id,
            "success": event.success,
            "colony_id": event.colony_id,
            "metadata": event.event_data
        })
        
        # Trigger event handlers
        for handler in self.event_handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in event handler for {event.event_type}: {e}")
        
        # Update collection stats
        stats_key = f"events_{event.event_type}"
        if stats_key not in self.collection_stats:
            self.collection_stats[stats_key] = 0
        self.collection_stats[stats_key] += 1
    
    def integrate_with_foraging_system(self, foraging_system: IntegratedForagingSystem) -> None:
        """Integrate with foraging system for comprehensive data collection"""
        
        self.foraging_system_integration = foraging_system
        if self.logger:
            self.logger.info("Data collector integrated with foraging system")
    
    def collect_bee_data(self, bee_agent: Any, current_step: int) -> None:
        """Collect comprehensive data for individual bee"""
        
        bee_id = bee_agent.unique_id
        
        # Initialize metrics if not exists
        if bee_id not in self.bee_metrics:
            self.bee_metrics[bee_id] = BeeMetrics(bee_id=bee_id)
        
        metrics = self.bee_metrics[bee_id]
        
        # Update basic demographics
        metrics.age = bee_agent.age
        metrics.role = bee_agent.role.value if hasattr(bee_agent.role, 'value') else str(bee_agent.role)
        metrics.status = bee_agent.status.value if hasattr(bee_agent.status, 'value') else str(bee_agent.status)
        metrics.total_steps += 1
        
        # Update activity tracking
        current_activity = metrics.status
        if current_activity not in metrics.activity_time:
            metrics.activity_time[current_activity] = 0
        metrics.activity_time[current_activity] += 1
        
        # Track state transitions
        if hasattr(bee_agent, 'activity_state_machine') and bee_agent.activity_state_machine:
            if bee_id in bee_agent.activity_state_machine.personal_trackers:
                tracker = bee_agent.activity_state_machine.personal_trackers[bee_id]
                if tracker.last_activity_change == current_step:
                    metrics.state_transitions += 1
        
        # Update foraging metrics
        if hasattr(bee_agent, 'foraging_decision_engine') and bee_agent.foraging_decision_engine:
            foraging_summary = bee_agent.get_foraging_summary()
            if 'memory_count' in foraging_summary:
                metrics.patches_discovered = foraging_summary['memory_count']
        
        # Update energy tracking
        if hasattr(bee_agent, 'energy'):
            metrics.energy_history.append(bee_agent.energy)
            if len(metrics.energy_history) > 100:  # Keep last 100 readings
                metrics.energy_history.pop(0)
        
        # Update physiological metrics
        if hasattr(bee_agent, 'activity_state_machine') and bee_agent.activity_state_machine:
            effects = bee_agent.activity_state_machine.get_state_physiological_effects(bee_agent.status)
            if effects:
                metrics.metabolic_rate = effects.get('metabolic_rate', 1.0)
                metrics.stress_level = effects.get('stress_level', 0.0)
        
        # Calculate performance metrics
        self._calculate_bee_performance(metrics, bee_agent)
        
        # Log data point
        self._log_data_point(current_step, "bee_update", bee_id, bee_id=bee_id)
    
    def collect_colony_data(self, colony: Any, current_step: int) -> None:
        """Collect comprehensive data for colony"""
        
        colony_id = colony.unique_id if hasattr(colony, 'unique_id') else 0
        
        # Initialize metrics if not exists
        if colony_id not in self.colony_metrics:
            self.colony_metrics[colony_id] = ColonyMetrics(colony_id=colony_id)
        
        metrics = self.colony_metrics[colony_id]
        
        # Update population metrics
        if hasattr(colony, 'agents'):
            agents = colony.agents
            metrics.total_population = len([a for a in agents if a.status != BeeStatus.DEAD])
            
            # Population by role
            role_counts: Dict[str, int] = defaultdict(int)
            activity_counts: Dict[str, int] = defaultdict(int)
            age_counts: Dict[str, int] = defaultdict(int)
            life_stage_counts: Dict[str, int] = defaultdict(int)  # Life stage tracking
            
            total_energy = 0.0
            total_stress = 0.0
            alive_count = 0
            
            for agent in agents:
                if agent.status != BeeStatus.DEAD:
                    alive_count += 1
                    role_counts[agent.role.value] += 1
                    activity_counts[agent.status.value] += 1
                    age_group = agent.age // 10  # Group by decades
                    age_counts[age_group] += 1
                    
                    # Track life stages based on age
                    life_stage = self._determine_life_stage(agent)
                    life_stage_counts[life_stage] += 1
                    
                    if hasattr(agent, 'energy'):
                        total_energy += agent.energy
                    
                    if hasattr(agent, 'activity_state_machine') and agent.activity_state_machine:
                        effects = agent.activity_state_machine.get_state_physiological_effects(agent.status)
                        if effects:
                            total_stress += effects.get('stress_level', 0.0)
            
            metrics.population_by_role = dict(role_counts)
            metrics.activity_distribution = dict(activity_counts)
            metrics.population_by_age = dict(age_counts)
            metrics.population_by_life_stage = dict(life_stage_counts)
            metrics.total_energy = total_energy
            
            # Update individual life stage counts (NetLogo style)
            metrics.egg_count = life_stage_counts.get('egg', 0)
            metrics.larva_count = life_stage_counts.get('larva', 0)
            metrics.pupa_count = life_stage_counts.get('pupa', 0)
            metrics.adult_count = life_stage_counts.get('adult', 0)
            
            if alive_count > 0:
                metrics.average_stress = total_stress / alive_count
        
        # Calculate colony fitness
        self._calculate_colony_fitness(metrics)
        
        # Update daily history lists if it's a new day
        self._update_daily_history_lists(metrics, current_step)
        
        # Log data point
        self._log_data_point(current_step, "colony_update", colony_id, colony_id=colony_id)
    
    def _determine_life_stage(self, agent: Any) -> str:
        """Determine bee life stage based on age and development"""
        
        # Default age-based life stage determination
        # This should be enhanced based on actual bee development model
        age = getattr(agent, 'age', 0)
        
        if age < 3:  # Egg stage (first 3 days)
            return 'egg'
        elif age < 9:  # Larva stage (days 3-9) 
            return 'larva'
        elif age < 12:  # Pupa stage (days 9-12)
            return 'pupa'
        else:  # Adult stage (day 12+)
            return 'adult'
    
    def _update_daily_history_lists(self, metrics: "ColonyMetrics", current_step: int) -> None:
        """Update daily history lists (NetLogo style tracking)"""
        
        # Check if it's a new day (assuming 24 steps per day)
        current_day = current_step // 24
        
        # Only update once per day
        if len(metrics.population_size_day_list) <= current_day:
            # Population history
            metrics.population_size_day_list.append(metrics.total_population)
            metrics.egg_count_day_list.append(metrics.egg_count)
            metrics.larva_count_day_list.append(metrics.larva_count)
            metrics.pupa_count_day_list.append(metrics.pupa_count)
            metrics.adult_count_day_list.append(metrics.adult_count)
            
            # Production history (reset daily counters)
            metrics.queens_produced_day_list.append(metrics.queens_produced_today)
            metrics.drones_produced_day_list.append(metrics.drones_produced_today)
            metrics.workers_produced_day_list.append(metrics.workers_produced_today)
            
            # Reset daily counters for next day
            metrics.queens_produced_today = 0
            metrics.drones_produced_today = 0
            metrics.workers_produced_today = 0
    
    def track_bee_production(self, colony_id: int, role: str) -> None:
        """Track production of new bees by role (NetLogo compatibility)"""
        
        if colony_id not in self.colony_metrics:
            self.colony_metrics[colony_id] = ColonyMetrics(colony_id=colony_id)
        
        metrics = self.colony_metrics[colony_id]
        
        # Update daily production counters
        if role.lower() == 'queen':
            metrics.queens_produced_today += 1
            metrics.total_queens_produced += 1
        elif role.lower() == 'drone':
            metrics.drones_produced_today += 1
            metrics.total_drones_produced += 1
        elif role.lower() in ['worker', 'forager', 'nurse']:
            metrics.workers_produced_today += 1
            metrics.total_workers_produced += 1
    
    def update_colony_metrics(self, colony: Any, current_step: int) -> None:
        """Update colony metrics for system integration (fixes missing method error)"""
        try:
            # Use existing collect_colony_data method
            self.collect_colony_data(colony, current_step)
            
            # Additional update operations for system integration
            colony_id = colony.unique_id if hasattr(colony, 'unique_id') else 0
            
            if colony_id in self.colony_metrics:
                metrics = self.colony_metrics[colony_id]
                
                # Update time series with current metrics
                current_metrics = {
                    'population': metrics.total_population,
                    'queens': metrics.queen_count,
                    'workers': metrics.worker_count,
                    'drones': metrics.drone_count,
                    'foragers': metrics.forager_count,
                    'nurses': metrics.nurse_count,
                    'brood': metrics.brood_count,
                    'productivity': metrics.daily_productivity,
                    'foraging_efficiency': metrics.foraging_efficiency_today
                }
                
                # Store in time series data
                if 'colony_metrics' not in self.time_series_data:
                    self.time_series_data['colony_metrics'] = {}
                
                if colony_id not in self.time_series_data['colony_metrics']:
                    self.time_series_data['colony_metrics'][colony_id] = []
                
                self.time_series_data['colony_metrics'][colony_id].append({
                    'step': current_step,
                    'timestamp': time.time(),
                    'metrics': current_metrics
                })
                
                # Keep only last 1000 entries per colony for memory efficiency
                if len(self.time_series_data['colony_metrics'][colony_id]) > 1000:
                    self.time_series_data['colony_metrics'][colony_id] = \
                        self.time_series_data['colony_metrics'][colony_id][-1000:]
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating colony metrics: {e}")

    def track_foraging_trip_details(self, bee_id: int, trip_data: Dict[str, Any]) -> None:
        """Track detailed foraging trip metrics (NetLogo compatibility)"""
        
        if bee_id not in self.bee_metrics:
            self.bee_metrics[bee_id] = BeeMetrics(bee_id=bee_id)
        
        metrics = self.bee_metrics[bee_id]
        
        # Extract trip details
        duration = trip_data.get('duration', 0.0)
        energy_gained = trip_data.get('energy_gained', 0.0)
        distance = trip_data.get('distance', 0.0)
        flowers_visited = trip_data.get('flowers_visited', 0)
        patches_visited = trip_data.get('patches_visited', 1)
        success = trip_data.get('success', False)
        nectar_collected = trip_data.get('nectar_collected', 0.0)
        pollen_collected = trip_data.get('pollen_collected', 0.0)
        
        # Update detailed trip metrics
        metrics.foraging_trip_durations.append(duration)
        metrics.foraging_trip_energies.append(energy_gained)
        metrics.foraging_trip_distances.append(distance)
        metrics.flowers_visited_per_trip.append(flowers_visited)
        metrics.patches_visited_per_trip.append(patches_visited)
        
        # Update success tracking
        if success:
            metrics.successful_trips += 1
        else:
            metrics.failed_trips += 1
        
        # Update success rate
        total_trips = metrics.successful_trips + metrics.failed_trips
        if total_trips > 0:
            metrics.foraging_success_rate = metrics.successful_trips / total_trips
        
        # Update resource collection totals
        metrics.nectar_collected_total += nectar_collected
        metrics.pollen_collected_total += pollen_collected
        
        # Update averages
        if metrics.foraging_trips > 0:
            metrics.nectar_per_trip_avg = metrics.nectar_collected_total / metrics.foraging_trips
            metrics.pollen_per_trip_avg = metrics.pollen_collected_total / metrics.foraging_trips
    
    def collect_environmental_data(self, environment: Any, current_step: int) -> None:
        """Collect environmental data"""
        
        # Update environmental metrics
        if hasattr(environment, 'temperature'):
            self.environmental_metrics.temperature = environment.temperature
        
        if hasattr(environment, 'weather_condition'):
            self.environmental_metrics.weather_condition = environment.weather_condition
        
        if hasattr(environment, 'resource_availability'):
            resource_info = environment.resource_availability
            self.environmental_metrics.nectar_availability = resource_info.get('nectar', 1.0)
            self.environmental_metrics.pollen_availability = resource_info.get('pollen', 1.0)
        
        # Log environmental data
        self._log_data_point(current_step, "environment_update", None)
    
    def _calculate_bee_performance(self, metrics: BeeMetrics, bee_agent: Any) -> None:
        """Calculate performance metrics for individual bee"""
        
        # Efficiency score based on energy collection vs consumption
        if metrics.total_steps > 0:
            foraging_efficiency = metrics.successful_foraging / max(1, metrics.foraging_trips)
            activity_efficiency = sum(metrics.activity_time.values()) / metrics.total_steps
            
            metrics.efficiency_score = (foraging_efficiency + activity_efficiency) / 2
        
        # Contribution score based on role performance
        if metrics.role == "forager":
            metrics.contribution_score = metrics.energy_collected / max(1, metrics.total_steps)
        elif metrics.role == "nurse":
            metrics.contribution_score = metrics.communication_events / max(1, metrics.total_steps)
        else:
            metrics.contribution_score = 0.5  # Default for other roles
    
    def _calculate_colony_fitness(self, metrics: ColonyMetrics) -> None:
        """Calculate colony fitness metrics"""
        
        # Basic fitness based on population stability
        population_stability = min(1.0, metrics.total_population / 100)  # Assume 100 is optimal
        
        # Energy efficiency
        energy_efficiency = 1.0
        if metrics.daily_energy_consumption > 0:
            energy_efficiency = min(1.0, metrics.daily_energy_collection / metrics.daily_energy_consumption)
        
        # Health factor
        health_factor = 1.0 - metrics.average_stress
        
        # Genetic diversity factor
        diversity_factor = metrics.genetic_diversity
        
        # Calculate overall fitness
        metrics.colony_fitness = (
            population_stability * 0.3 +
            energy_efficiency * 0.3 +
            health_factor * 0.2 +
            diversity_factor * 0.2
        )
        
        # Survival probability based on fitness
        metrics.survival_probability = max(0.1, min(1.0, metrics.colony_fitness))
    
    def _log_data_point(self, timestamp: int, metric_name: str, value: Any, 
                       bee_id: Optional[int] = None, colony_id: Optional[int] = None) -> None:
        """Log individual data point"""
        
        data_point = DataPoint(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            agent_id=bee_id,
            colony_id=colony_id
        )
        
        self.event_log.append(data_point)
        
        # Maintain log size
        if len(self.event_log) > self.max_history_length:
            self.event_log.pop(0)
        
        # Update collection statistics
        if metric_name not in self.collection_stats:
            self.collection_stats[metric_name] = 0
        self.collection_stats[metric_name] += 1
    
    def get_time_series(self, metric_name: str, time_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, Any]]:
        """Get time series data for specific metric"""
        
        if metric_name not in self.time_series_data:
            return []
        
        data = list(self.time_series_data[metric_name])
        
        if time_range:
            start_time, end_time = time_range
            data = [(t, v) for t, v in data if start_time <= t <= end_time]
        
        return data
    
    def get_aggregated_data(self, metric_name: str, aggregation_method: str, 
                           time_window: int = 100) -> Dict[str, Any]:
        """Get aggregated data for metric"""
        
        time_series = self.get_time_series(metric_name)
        if not time_series:
            return {}
        
        # Get recent data
        recent_data = time_series[-time_window:]
        values = [v for _, v in recent_data if isinstance(v, (int, float))]
        
        if not values:
            return {}
        
        result = {}
        
        if aggregation_method == "mean":
            result["mean"] = statistics.mean(values)
        elif aggregation_method == "median":
            result["median"] = statistics.median(values)
        elif aggregation_method == "min":
            result["min"] = min(values)
        elif aggregation_method == "max":
            result["max"] = max(values)
        elif aggregation_method == "sum":
            result["sum"] = sum(values)
        elif aggregation_method == "count":
            result["count"] = len(values)
        
        if len(values) > 1:
            result["std_dev"] = statistics.stdev(values)
        
        return result
    
    def export_data(self, filename: str, format: str = "csv") -> None:
        """Export collected data to file"""
        
        if format == "csv":
            self._export_to_csv(filename)
        elif format == "json":
            self._export_to_json(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_csv(self, filename: str) -> None:
        """Export data to CSV format"""
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'metric_name', 'value', 'agent_id', 'colony_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data_point in self.event_log:
                writer.writerow({
                    'timestamp': data_point.timestamp,
                    'metric_name': data_point.metric_name,
                    'value': data_point.value,
                    'agent_id': data_point.agent_id,
                    'colony_id': data_point.colony_id
                })
    
    def _export_to_json(self, filename: str) -> None:
        """Export data to JSON format"""
        
        export_data = {
            'bee_metrics': {k: v.model_dump() for k, v in self.bee_metrics.items()},
            'colony_metrics': {k: v.model_dump() for k, v in self.colony_metrics.items()},
            'environmental_metrics': self.environmental_metrics.model_dump(),
            'event_log': [
                {
                    'timestamp': dp.timestamp,
                    'metric_name': dp.metric_name,
                    'value': dp.value,
                    'agent_id': dp.agent_id,
                    'colony_id': dp.colony_id
                }
                for dp in self.event_log
            ]
        }
        
        with open(filename, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        
        return {
            "data_collection_stats": dict(self.collection_stats),
            "total_bees_tracked": len(self.bee_metrics),
            "total_colonies_tracked": len(self.colony_metrics),
            "total_data_points": len(self.event_log),
            "collection_rules": len(self.collection_rules),
            "memory_usage": {
                "event_log_size": len(self.event_log),
                "time_series_count": len(self.time_series_data),
                "max_history_length": self.max_history_length
            },
            "environmental_conditions": {
                "temperature": self.environmental_metrics.temperature,
                "season": self.environmental_metrics.season,
                "weather": self.environmental_metrics.weather_condition
            }
        }
    
    def reset_data(self) -> None:
        """Reset all collected data"""
        
        self.bee_metrics.clear()
        self.colony_metrics.clear()
        self.environmental_metrics = EnvironmentalMetrics()
        self.time_series_data.clear()
        self.event_log.clear()
        self.daily_summaries.clear()
        self.weekly_summaries.clear()
        self.collection_stats.clear()
        self.simulation_events.clear()
        self.aggregated_data.clear()
        self.live_data_streams.clear()
    
    # Event Handler Methods
    def _handle_foraging_session_start(self, event: SimulationEvent) -> None:
        """Handle foraging session start event"""
        
        if event.source_agent_id and event.colony_id:
            # Update bee metrics
            if event.source_agent_id in self.bee_metrics:
                self.bee_metrics[event.source_agent_id].foraging_sessions += 1
            
            # Create data point
            data_point = DataPoint(
                timestamp=int(event.timestamp),
                metric_name="foraging_session_started",
                value=1,
                agent_id=event.source_agent_id,
                colony_id=event.colony_id,
                metadata=event.event_data,
                event_type="foraging",
                aggregation_level="individual"
            )
            
            self.event_log.append(data_point)
    
    def _handle_foraging_session_end(self, event: SimulationEvent) -> None:
        """Handle foraging session end event"""
        
        if event.source_agent_id and event.colony_id:
            # Extract session result from event data
            session_result = event.event_data.get("session_result")
            
            if session_result and event.source_agent_id in self.bee_metrics:
                metrics = self.bee_metrics[event.source_agent_id]
                
                # Update detailed foraging metrics
                metrics.energy_collected += session_result.get("net_energy_result", 0.0)
                metrics.patch_visits += len(session_result.get("patches_visited", []))
                metrics.unique_patches_visited = max(
                    metrics.unique_patches_visited,
                    len(session_result.get("patches_visited", []))
                )
                metrics.travel_distance_total += session_result.get("total_distance_traveled", 0.0)
                
                # Update session duration average
                current_avg = metrics.foraging_session_duration_avg
                session_count = metrics.foraging_sessions
                new_duration = session_result.get("session_duration", 0.0) / 60.0  # Convert to minutes
                
                if session_count > 0:
                    metrics.foraging_session_duration_avg = (
                        (current_avg * (session_count - 1) + new_duration) / session_count
                    )
                else:
                    metrics.foraging_session_duration_avg = new_duration
                
                # Update efficiency average
                session_efficiency = session_result.get("overall_success_score", 0.0)
                if session_count > 0:
                    metrics.foraging_efficiency_avg = (
                        (metrics.foraging_efficiency_avg * (session_count - 1) + session_efficiency) / session_count
                    )
                else:
                    metrics.foraging_efficiency_avg = session_efficiency
            
            # Create aggregated data point
            data_point = DataPoint(
                timestamp=int(event.timestamp),
                metric_name="foraging_session_completed",
                value=event.event_data.get("session_result", {}).get("overall_success_score", 0.0),
                agent_id=event.source_agent_id,
                colony_id=event.colony_id,
                metadata=event.event_data,
                event_type="foraging",
                aggregation_level="individual"
            )
            
            self.event_log.append(data_point)
    
    def _handle_foraging_trip(self, event: SimulationEvent) -> None:
        """Handle individual foraging trip completion"""
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            metrics.foraging_trips += 1
            
            # Track detailed trip data if available
            if event.event_data:
                self.track_foraging_trip_details(event.source_agent_id, event.event_data)
            
            trip_success = event.event_data.get("success", False)
            if trip_success:
                metrics.successful_foraging += 1
            
            # Update patch discovery
            new_patches = event.event_data.get("new_patches_discovered", 0)
            metrics.patches_discovered += new_patches
    
    def _handle_dance_event(self, event: SimulationEvent) -> None:
        """Handle dance performance event"""
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            metrics.dances_performed += 1
            metrics.communication_events += 1
            
            # Track recruitment attempts
            if event.event_data.get("recruitment_initiated", False):
                metrics.recruitment_attempts += 1
    
    def _handle_recruitment_event(self, event: SimulationEvent) -> None:
        """Handle recruitment event"""
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            
            recruited_count = event.event_data.get("recruited_count", 0)
            metrics.recruits_obtained += recruited_count
    
    def _handle_information_sharing(self, event: SimulationEvent) -> None:
        """Handle information sharing event"""
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            metrics.information_shared += 1
            metrics.communication_events += 1
    
    def _handle_state_transition(self, event: SimulationEvent) -> None:
        """Handle bee state transition event"""
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            metrics.state_transitions += 1
            
            # Track adaptation events
            if event.event_data.get("adaptive_transition", False):
                metrics.learning_events += 1
    
    def _handle_role_change(self, event: SimulationEvent) -> None:
        """Handle bee role change event"""
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            metrics.strategy_changes += 1
    
    def _handle_energy_change(self, event: SimulationEvent) -> None:
        """Handle energy change event"""
        
        energy_change = event.event_data.get("energy_delta", 0.0)
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            
            if energy_change > 0:
                metrics.energy_collected += energy_change
    
    def _handle_resource_collection(self, event: SimulationEvent) -> None:
        """Handle resource collection event"""
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            
            resource_amount = event.event_data.get("resource_amount", 0.0)
            if resource_amount > 0:
                metrics.energy_collected += resource_amount
    
    def _handle_bee_death(self, event: SimulationEvent) -> None:
        """Handle bee death event"""
        
        if event.source_agent_id and event.source_agent_id in self.bee_metrics:
            metrics = self.bee_metrics[event.source_agent_id]
            metrics.death_cause = event.event_data.get("cause", "unknown")
            metrics.lifespan = event.event_data.get("age", metrics.age)
    
    def _handle_bee_emergence(self, event: SimulationEvent) -> None:
        """Handle bee emergence event"""
        
        # This would be called when new bees are created
        if event.source_agent_id:
            # Initialize metrics for new bee
            if event.source_agent_id not in self.bee_metrics:
                self.bee_metrics[event.source_agent_id] = BeeMetrics(bee_id=event.source_agent_id)
    
    def aggregate_data(self, current_step: int) -> Dict[str, Any]:
        """Perform data aggregation based on defined rules"""
        
        aggregation_results = {}
        
        for rule in self.aggregation_rules:
            try:
                result = self._apply_aggregation_rule(rule, current_step)
                aggregation_results[rule.metric_name] = result
                
                # Store in aggregated data cache
                if rule.metric_name not in self.aggregated_data:
                    self.aggregated_data[rule.metric_name] = {}
                
                self.aggregated_data[rule.metric_name][str(current_step)] = result
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error applying aggregation rule {rule.metric_name}: {e}")
        
        return aggregation_results
    
    def _apply_aggregation_rule(self, rule: AggregationRule, current_step: int) -> Any:
        """Apply individual aggregation rule"""
        
        # Get data within time window
        start_step = max(0, current_step - rule.time_window)
        
        # Filter data points based on rule criteria
        relevant_points = []
        
        for point in self.event_log:
            if (start_step <= point.timestamp <= current_step and
                (not rule.condition or self._evaluate_condition(point, rule.condition))):
                relevant_points.append(point)
        
        if not relevant_points:
            return None
        
        # Group data by specified fields
        grouped_data = defaultdict(list)
        
        for point in relevant_points:
            group_key = tuple(
                getattr(point, field, None) or point.metadata.get(field, "unknown")
                for field in rule.grouping
            )
            grouped_data[group_key].append(point)
        
        # Apply aggregation function
        results = {}
        
        for group_key, points in grouped_data.items():
            values = [point.value for point in points if isinstance(point.value, (int, float))]
            
            if values:
                if rule.aggregation_function == "mean":
                    result = statistics.mean(values)
                elif rule.aggregation_function == "sum":
                    result = sum(values)
                elif rule.aggregation_function == "count":
                    result = len(values)
                elif rule.aggregation_function == "min":
                    result = min(values)
                elif rule.aggregation_function == "max":
                    result = max(values)
                elif rule.aggregation_function == "std":
                    result = statistics.stdev(values) if len(values) > 1 else 0.0
                elif rule.aggregation_function == "median":
                    result = statistics.median(values)
                else:
                    result = values
                
                # Create readable group key
                group_label = "_".join(str(k) for k in group_key)
                results[group_label] = result
        
        return results
    
    def _evaluate_condition(self, point: DataPoint, condition: str) -> bool:
        """Evaluate condition string against data point"""
        
        try:
            # Simple condition evaluation (can be extended)
            if ">" in condition:
                field, threshold = condition.split(">")
                field = field.strip()
                threshold = float(threshold.strip())
                
                value = getattr(point, field, None) or point.metadata.get(field, 0)
                return float(value) > threshold
                
            elif "==" in condition:
                field, expected = condition.split("==")
                field = field.strip()
                expected = expected.strip().strip("'\"")
                
                value = getattr(point, field, None) or point.metadata.get(field, "")
                return str(value) == expected
            
            # Add more condition types as needed
            return True
            
        except Exception:
            return True
    
    def get_live_data_stream(self, stream_name: str) -> List[Dict[str, Any]]:
        """Get live data stream"""
        
        return list(self.live_data_streams.get(stream_name, []))
    
    def get_aggregated_metric_data(self, metric_name: str, time_range: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Get aggregated data for specific metric"""
        
        if metric_name not in self.aggregated_data:
            return {}
        
        data = self.aggregated_data[metric_name]
        
        if time_range:
            start_time, end_time = time_range
            data = {k: v for k, v in data.items() if start_time <= k <= end_time}
        
        return data
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics across all collected data"""
        
        analytics = {
            "collection_summary": {
                "total_bees_tracked": len(self.bee_metrics),
                "total_colonies_tracked": len(self.colony_metrics),
                "total_events_recorded": len(self.simulation_events),
                "collection_stats": dict(self.collection_stats)
            },
            "foraging_analytics": {},
            "communication_analytics": {},
            "behavioral_analytics": {},
            "performance_analytics": {}
        }
        
        # Aggregate foraging analytics
        total_foraging_sessions = sum(m.foraging_sessions for m in self.bee_metrics.values())
        total_energy_collected = sum(m.energy_collected for m in self.bee_metrics.values())
        avg_foraging_efficiency = statistics.mean([
            m.foraging_efficiency_avg for m in self.bee_metrics.values() if m.foraging_efficiency_avg > 0
        ]) if any(m.foraging_efficiency_avg > 0 for m in self.bee_metrics.values()) else 0.0
        
        analytics["foraging_analytics"] = {
            "total_sessions": total_foraging_sessions,
            "total_energy_collected": total_energy_collected,
            "average_efficiency": avg_foraging_efficiency,
            "total_patches_discovered": sum(m.patches_discovered for m in self.bee_metrics.values())
        }
        
        # Aggregate communication analytics
        total_dances = sum(m.dances_performed for m in self.bee_metrics.values())
        total_communication_events = sum(m.communication_events for m in self.bee_metrics.values())
        
        analytics["communication_analytics"] = {
            "total_dances_performed": total_dances,
            "total_communication_events": total_communication_events,
            "total_recruitment_attempts": sum(m.recruitment_attempts for m in self.bee_metrics.values()),
            "total_information_shared": sum(m.information_shared for m in self.bee_metrics.values())
        }
        
        # Behavioral analytics
        total_state_transitions = sum(m.state_transitions for m in self.bee_metrics.values())
        total_learning_events = sum(m.learning_events for m in self.bee_metrics.values())
        
        analytics["behavioral_analytics"] = {
            "total_state_transitions": total_state_transitions,
            "total_learning_events": total_learning_events,
            "total_strategy_changes": sum(m.strategy_changes for m in self.bee_metrics.values()),
            "average_adaptation_score": statistics.mean([
                m.adaptation_score for m in self.bee_metrics.values() if m.adaptation_score > 0
            ]) if any(m.adaptation_score > 0 for m in self.bee_metrics.values()) else 0.0
        }
        
        return analytics
    
    def generate_excel_report(self, report_type: str = "summary", 
                            output_dir: Optional[str] = None,
                            simulation_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive Excel report from collected data"""
        
        try:
            # Import Excel integration
            from ..reports.excel_integration import ExcelReportGenerator, ReportType
            
            # Create Excel report generator
            excel_generator = ExcelReportGenerator(
                output_directory=output_dir or tempfile.gettempdir(),
                include_charts=True,
                include_conditional_formatting=True
            )
            
            # Convert string to ReportType enum
            report_type_map = {
                "summary": ReportType.SUMMARY,
                "detailed": ReportType.DETAILED,
                "health": ReportType.HEALTH_MONITORING,
                "foraging": ReportType.FORAGING_ANALYSIS,
                "temporal": ReportType.TEMPORAL_ANALYSIS,
                "comparative": ReportType.COMPARATIVE
            }
            
            report_enum = report_type_map.get(report_type.lower(), ReportType.SUMMARY)
            
            # Set up output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp()
            
            # Prepare simulation metadata
            if simulation_metadata is None:
                simulation_metadata = {
                    "simulation_id": f"sim_{int(time.time())}",
                    "duration_days": max(self.aggregated_metrics.keys()) if self.aggregated_metrics else 100,
                    "start_date": min(m.timestamp for m in self.simulation_events) if self.simulation_events else time.time(),
                    "end_date": max(m.timestamp for m in self.simulation_events) if self.simulation_events else time.time()
                }
            
            # Generate report
            report_path = excel_generator.generate_comprehensive_report(
                data_collector=self,
                report_type=report_enum,
                simulation_metadata=simulation_metadata
            )
            
            if self.logger:
                self.logger.info(f"Excel report generated: {report_path}")
            return report_path
            
        except ImportError:
            if self.logger:
                self.logger.error("Excel integration not available - install openpyxl")
            raise RuntimeError("Excel integration requires openpyxl library")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to generate Excel report: {e}")
            raise
    
    def get_population_data_for_excel(self) -> List[Dict[str, Any]]:
        """Get population data formatted for Excel export"""
        
        population_data = []
        
        for day, metrics in self.aggregated_metrics.items():
            if isinstance(metrics, dict) and 'population' in metrics:
                # Legacy dict format
                pop_data = metrics['population']
                population_data.append({
                    'day': day,
                    'total_population': pop_data.get('total', 0),
                    'adult_population': pop_data.get('adults', 0),
                    'larval_population': pop_data.get('larvae', 0),
                    'colony_count': pop_data.get('colonies', 1),
                    'growth_rate': pop_data.get('growth_rate', 0.0)
                })
            elif isinstance(metrics, dict) and metrics and any(hasattr(m, 'population_size') for m in metrics.values() if hasattr(m, '__dict__')):
                # New colony metrics format
                total_pop = sum(getattr(colony, 'population_size', 0) for colony in metrics.values())
                total_foragers = sum(getattr(colony, 'active_foragers', 0) for colony in metrics.values())
                total_nurses = sum(getattr(colony, 'nurse_bees', 0) for colony in metrics.values())
                
                population_data.append({
                    'day': day,
                    'total_population': total_pop,
                    'active_foragers': total_foragers,
                    'nurse_bees': total_nurses,
                    'colony_count': len(metrics),
                    'avg_fitness': statistics.mean([getattr(colony, 'colony_fitness', 0) for colony in metrics.values()]) if metrics else 0.0
                })
        
        return population_data
    
    def get_health_data_for_excel(self) -> List[Dict[str, Any]]:
        """Get health data formatted for Excel export"""
        
        health_data = []
        
        # Aggregate health data by day and colony
        health_by_day = defaultdict(lambda: defaultdict(list))
        
        for colony_id, metrics in self.colony_metrics.items():
            for i, health_score in enumerate(metrics.health_scores):
                day = i + 1  # Assuming daily collection
                health_by_day[day][colony_id].append(health_score)
        
        for day, colonies in health_by_day.items():
            for colony_id, scores in colonies.items():
                avg_score = statistics.mean(scores) if scores else 0.0
                min_score = min(scores) if scores else 0.0
                max_score = max(scores) if scores else 0.0
                
                # Determine status based on score
                if avg_score >= 0.8:
                    status = "excellent"
                elif avg_score >= 0.6:
                    status = "good"
                elif avg_score >= 0.4:
                    status = "fair"
                elif avg_score >= 0.2:
                    status = "poor"
                else:
                    status = "critical"
                
                health_data.append({
                    'day': day,
                    'colony_id': colony_id,
                    'health_score': avg_score,
                    'min_health_score': min_score,
                    'max_health_score': max_score,
                    'status': status,
                    'alerts_count': 0  # Would be populated from actual alert data
                })
        
        return health_data
    
    def get_foraging_data_for_excel(self) -> List[Dict[str, Any]]:
        """Get foraging data formatted for Excel export"""
        
        foraging_data = []
        
        for day, metrics in self.aggregated_metrics.items():
            if isinstance(metrics, dict) and 'foraging' in metrics:
                # Legacy format
                forage_data = metrics['foraging']
                foraging_data.append({
                    'day': day,
                    'efficiency': forage_data.get('avg_efficiency', 0.0),
                    'energy_collected': forage_data.get('total_energy', 0.0),
                    'trips_completed': forage_data.get('total_trips', 0),
                    'success_rate': forage_data.get('success_rate', 0.0),
                    'avg_trip_duration': forage_data.get('avg_duration', 0.0),
                    'patches_visited': forage_data.get('patches_visited', 0)
                })
            elif isinstance(metrics, dict) and metrics and any(hasattr(m, 'foraging_efficiency') for m in metrics.values() if hasattr(m, '__dict__')):
                # New colony metrics format
                total_energy = sum(getattr(colony, 'daily_energy_collection', 0) for colony in metrics.values())
                avg_efficiency = statistics.mean([getattr(colony, 'foraging_efficiency', 0) for colony in metrics.values()]) if metrics else 0.0
                total_foragers = sum(getattr(colony, 'active_foragers', 0) for colony in metrics.values())
                
                foraging_data.append({
                    'day': day,
                    'efficiency': avg_efficiency,
                    'energy_collected': total_energy,
                    'active_foragers': total_foragers,
                    'avg_energy_per_forager': total_energy / max(total_foragers, 1),
                    'colony_count': len(metrics)
                })
        
        return foraging_data
    
    def get_event_log_for_excel(self) -> List[Dict[str, Any]]:
        """Get event log formatted for Excel export"""
        
        events = []
        
        for event in self.simulation_events[-1000:]:  # Last 1000 events
            events.append({
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'source_agent_id': event.source_agent_id,
                'target_agent_id': event.target_agent_id,
                'colony_id': event.colony_id,
                'success': event.success,
                'energy_impact': event.energy_impact or 0.0,
                'data_summary': str(event.event_data)[:100] if event.event_data else ""
            })
        
        return events
    
    def get_summary_statistics_for_excel(self) -> Dict[str, Any]:
        """Get summary statistics for Excel dashboard"""
        
        stats = {}
        
        # Basic collection stats
        stats.update({
            "Total Simulation Days": len(self.aggregated_metrics),
            "Total Bees Tracked": len(self.bee_metrics),
            "Total Colonies Tracked": len(self.colony_metrics),
            "Total Events Recorded": len(self.simulation_events),
            "Data Collection Points": sum(self.collection_stats.values())
        })
        
        # Population stats
        if self.aggregated_metrics:
            population_data = [metrics.get('population', {}).get('total', 0) 
                             for metrics in self.aggregated_metrics.values()]
            if population_data:
                stats.update({
                    "Average Population": statistics.mean(population_data),
                    "Peak Population": max(population_data),
                    "Min Population": min(population_data)
                })
        
        # Health stats
        all_health_scores = []
        for colony_metrics in self.colony_metrics.values():
            all_health_scores.extend(colony_metrics.health_scores)
        
        if all_health_scores:
            stats.update({
                "Average Health Score": statistics.mean(all_health_scores),
                "Min Health Score": min(all_health_scores),
                "Health Score Std Dev": statistics.stdev(all_health_scores) if len(all_health_scores) > 1 else 0
            })
        
        # Foraging stats
        all_efficiency_scores = []
        for bee_metrics in self.bee_metrics.values():
            if bee_metrics.foraging_efficiency_avg > 0:
                all_efficiency_scores.append(bee_metrics.foraging_efficiency_avg)
        
        if all_efficiency_scores:
            stats.update({
                "Average Foraging Efficiency": statistics.mean(all_efficiency_scores),
                "Peak Foraging Efficiency": max(all_efficiency_scores)
            })
        
        return stats
    
    def get_key_findings_for_excel(self) -> List[str]:
        """Get key findings for Excel dashboard"""
        
        findings = []
        
        # Analyze population trends
        if self.aggregated_metrics:
            population_data = [metrics.get('population', {}).get('total', 0) 
                             for metrics in self.aggregated_metrics.values()]
            if len(population_data) > 1:
                growth_rate = (population_data[-1] - population_data[0]) / population_data[0] if population_data[0] > 0 else 0
                if growth_rate > 0.1:
                    findings.append(f"Population grew by {growth_rate:.1%} over simulation period")
                elif growth_rate < -0.1:
                    findings.append(f"Population declined by {abs(growth_rate):.1%} over simulation period")
                else:
                    findings.append("Population remained stable throughout simulation")
        
        # Analyze health trends
        all_health_scores = []
        for colony_metrics in self.colony_metrics.values():
            all_health_scores.extend(colony_metrics.health_scores)
        
        if all_health_scores:
            avg_health = statistics.mean(all_health_scores)
            if avg_health >= 0.8:
                findings.append("Colony health maintained at excellent levels")
            elif avg_health >= 0.6:
                findings.append("Colony health remained good throughout simulation")
            else:
                findings.append("Colony health showed concerning decline patterns")
        
        # Analyze foraging performance
        total_energy = sum(m.energy_collected for m in self.bee_metrics.values())
        if total_energy > 0:
            findings.append(f"Total energy collected: {total_energy:.0f} units")
        
        # Event analysis
        event_types = defaultdict(int)
        for event in self.simulation_events:
            event_types[event.event_type] += 1
        
        if event_types:
            most_common = max(event_types, key=event_types.get)
            findings.append(f"Most frequent event type: {most_common} ({event_types[most_common]} occurrences)")
        
        # Collection performance
        total_collections = sum(self.collection_stats.values())
        if total_collections > 1000:
            findings.append(f"High-frequency data collection: {total_collections:,} data points")
        
        return findings[:5]  # Return top 5 findings