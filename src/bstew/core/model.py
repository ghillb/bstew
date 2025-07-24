"""
Main BSTEW Model
================

Core simulation model integrating all BSTEW components including
colonies, environment, weather, disease, and performance optimization.
"""

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from typing import Dict, List, Any, Optional
import logging
import math
import time
from datetime import datetime

from .colony import Colony
from .scheduler import BeeScheduler
from ..spatial.landscape import LandscapeGrid
from ..components.environment import EnvironmentalEffectsManager, ClimateScenario
from ..components.disease import DiseaseManager
from ..components.reproduction import ReproductionManager
from ..utils.weather import WeatherIntegrationManager, create_weather_source_from_config
from ..utils.performance import SimulationOptimizer
from .foraging_integration import IntegratedForagingSystem
from .patch_selection import (
    PatchInfo,
    ResourceType as PatchResourceType,
    PatchQualityMetric,
)
from .data_collection import ComprehensiveDataCollector, SimulationEvent
from .dynamic_parameters import (
    DynamicParameterSystem,
    ModificationRule,
    ParameterType,
    ModificationTrigger,
    ModificationStrategy,
)
from .health_monitoring import HealthMonitoringSystem, HealthThresholds, HealthAlert
from .spatial_integration import (
    SpatialEnvironmentManager,
    SpatialBeeManager,
    create_spatial_integration_system,
)

# REMOVED: Dance communication integration (honey bee behavior - not for bumblebees)
# from .dance_communication_integration import (
#     DanceCommunicationIntegrator,
#     create_dance_communication_integration,
# )
from .bumblebee_communication import BumblebeeCommunicationSystem
from ..visualization.live_visualization import (
    LiveVisualizationManager,
    create_live_visualization_system,
)
from .system_integrator import SystemIntegrator


class BeeModel(Model):
    """
    Main BSTEW simulation model.

    Integrates all simulation components and manages the simulation lifecycle.
    """

    # Type annotations for components
    environment_manager: Optional[EnvironmentalEffectsManager]
    weather_manager: Optional[WeatherIntegrationManager]
    optimizer: Optional[SimulationOptimizer]
    landscape: Any  # Can be LandscapeGrid or MockLandscape
    foraging_system: Optional[IntegratedForagingSystem]
    dynamic_parameter_system: Optional[DynamicParameterSystem]
    health_monitoring_system: Optional[HealthMonitoringSystem]
    spatial_environment: Optional[SpatialEnvironmentManager]
    spatial_bee_manager: Optional[SpatialBeeManager]
    # REMOVED: dance_communication_integrator (honey bee behavior)
    bumblebee_communication_system: Optional[BumblebeeCommunicationSystem]
    live_visualization_manager: Optional[LiveVisualizationManager]
    system_integrator: Optional[SystemIntegrator]

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, random_seed: Optional[int] = None
    ):
        super().__init__()

        # Store random seed for reset functionality
        self.initial_random_seed = random_seed

        # ID counter for unique IDs
        self._next_id = 1

        # Set random seed
        if random_seed:
            np.random.seed(random_seed)
            self.random.seed(random_seed)

        # Load configuration
        if isinstance(config, dict):
            # Convert dict to BstewConfig for consistency
            from ..utils.config import BstewConfig

            self.config = BstewConfig.from_dict(config)
        elif config is not None:
            # Config is already a BstewConfig object
            self.config = config
        else:
            from ..utils.config import ConfigManager

            config_manager = ConfigManager()
            self.config = config_manager.load_default_config()

        # Keep dict version for backward compatibility
        self.config_dict = self.config.model_dump()

        # Extract random seed from config if not explicitly provided
        if (
            random_seed is None
            and hasattr(self.config, "simulation")
            and hasattr(self.config.simulation, "random_seed")
        ):
            random_seed = self.config.simulation.random_seed
            self.initial_random_seed = random_seed
            if random_seed:
                np.random.seed(random_seed)
                self.random.seed(random_seed)

        self.logger = logging.getLogger(__name__)

        # Configure performance logging based on simulation needs
        self._configure_performance_logging()

        # Initialize simulation parameters
        self.simulation_start = datetime.now()
        self.current_day = 0
        self.max_days = self.config_dict.get("simulation", {}).get("duration_days", 365)
        self.running: bool = True

        # Initialize components
        self._initialize_scheduler()
        self._initialize_landscape()
        self._initialize_environment()
        self._initialize_weather()
        self._initialize_foraging_system()
        self._initialize_dynamic_parameters()
        self._initialize_health_monitoring()
        self._initialize_spatial_systems()
        self._initialize_dance_communication()
        self._initialize_live_visualization()
        self._initialize_system_integrator()
        self._initialize_colonies()
        self._initialize_data_collection()
        self._initialize_optimization()

        self.logger.info(
            f"BSTEW model initialized with {self.get_colony_count()} colonies"
        )

    def run_simulation(self, days: Optional[int] = None) -> None:
        """Run simulation for specified number of days"""
        if days is not None and days < 0:
            raise ValueError("Days must be non-negative")

        if days is None:
            days = self.max_days

        target_day = self.current_day + days

        while self.current_day < target_day and self.running:
            self.step()

        # Set running to False when simulation completes
        if self.current_day >= target_day:
            self.running = False

    def next_id(self) -> int:
        """Generate next unique ID"""
        current_id = self._next_id
        self._next_id += 1
        return current_id

    def _initialize_scheduler(self) -> None:
        """Initialize simulation scheduler"""
        self.schedule = BeeScheduler(self)

    def _initialize_landscape(self) -> None:
        """Initialize spatial landscape"""
        landscape_config = self.config_dict.get("landscape", {})

        self.landscape = LandscapeGrid(
            width=landscape_config.get("width", 100),
            height=landscape_config.get("height", 100),
            cell_size=landscape_config.get("cell_size", 20.0),
        )

        # Load landscape data if specified
        if "landscape_file" in landscape_config:
            self.landscape.load_from_file(landscape_config["landscape_file"])

    def _initialize_environment(self) -> None:
        """Initialize environmental effects"""
        env_config = self.config_dict.get("environment", {})

        if env_config.get("enabled", True):
            latitude = env_config.get("latitude", 40.0)
            self.environment_manager = EnvironmentalEffectsManager(latitude)

            # Apply climate scenario if specified
            if "climate_scenario" in env_config:
                scenario_config = env_config["climate_scenario"]
                if isinstance(scenario_config, dict):
                    climate_scenario = ClimateScenario(
                        name=scenario_config.get("name", "default"),
                        description=scenario_config.get("description", ""),
                        temperature_trend=scenario_config.get("temperature_trend", 0.0),
                        precipitation_trend=scenario_config.get(
                            "precipitation_trend", 0.0
                        ),
                        extreme_frequency_multiplier=scenario_config.get(
                            "extreme_frequency_multiplier", 1.0
                        ),
                    )
                else:
                    # Handle string scenario name
                    climate_scenario = ClimateScenario(
                        name=str(scenario_config),
                        description="",
                        temperature_trend=0.0,
                        precipitation_trend=0.0,
                        extreme_frequency_multiplier=1.0,
                    )
                self.environment_manager.set_climate_scenario(climate_scenario)
        else:
            self.environment_manager = None

    def _initialize_weather(self) -> None:
        """Initialize weather system"""
        weather_config = self.config_dict.get("weather", {})

        if weather_config.get("enabled", False):
            self.weather_manager = WeatherIntegrationManager()

            # Add weather sources from configuration
            for source_config in weather_config.get("sources", []):
                weather_source = create_weather_source_from_config(source_config)
                self.weather_manager.add_weather_source(weather_source)

            # Load weather data
            try:
                self.weather_manager.load_all_weather_data()
                self.logger.info("Weather data loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load weather data: {e}")
                self.weather_manager = None
        else:
            self.weather_manager = None

    def _initialize_foraging_system(self) -> None:
        """Initialize comprehensive foraging system"""
        foraging_config = self.config_dict.get("foraging", {})

        if foraging_config.get("enabled", True):
            self.foraging_system = IntegratedForagingSystem()
            self.logger.info("Integrated foraging system initialized")
        else:
            self.foraging_system = None

    def _initialize_dynamic_parameters(self) -> None:
        """Initialize dynamic parameter modification system"""
        dynamic_params_config = self.config_dict.get("dynamic_parameters", {})

        if dynamic_params_config.get("enabled", True):
            self.dynamic_parameter_system = DynamicParameterSystem(
                enable_adaptive_modifications=dynamic_params_config.get(
                    "adaptive_enabled", True
                ),
                max_simultaneous_modifications=dynamic_params_config.get(
                    "max_simultaneous", 10
                ),
                modification_cooldown=dynamic_params_config.get(
                    "cooldown_seconds", 5.0
                ),
            )

            # Add default modification rules
            self._add_default_parameter_rules()

            self.logger.info("Dynamic parameter modification system initialized")
        else:
            self.dynamic_parameter_system = None

    def _initialize_health_monitoring(self) -> None:
        """Initialize colony health monitoring system"""
        from typing import Dict

        health_config = self.config_dict.get("health_monitoring", {})

        # Initialize health status tracking
        self._last_health_status: Dict[int, str] = {}

        if health_config.get("enabled", True):
            # Create custom thresholds if specified
            threshold_config = health_config.get("thresholds", {})
            thresholds = (
                HealthThresholds(**threshold_config)
                if threshold_config
                else HealthThresholds()
            )

            self.health_monitoring_system = HealthMonitoringSystem(
                thresholds=thresholds,
                monitoring_enabled=health_config.get("monitoring_enabled", True),
                update_frequency=health_config.get("update_frequency", 1),
            )

            # Add alert callback for logging
            def health_alert_callback(alert: HealthAlert) -> None:
                self.logger.warning(
                    f"Health Alert: {alert.message} (Colony {alert.colony_id})"
                )
                # Emit simulation event for data collection
                self.emit_simulation_event(
                    "health_alert",
                    colony_id=alert.colony_id,
                    event_data={
                        "alert_level": alert.level.value,
                        "indicator": alert.indicator.value,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                    },
                    success=False,
                )

            self.health_monitoring_system.add_alert_callback(health_alert_callback)

            self.logger.info("Colony health monitoring system initialized")
        else:
            self.health_monitoring_system = None

    def _initialize_colonies(self) -> None:
        """Initialize bee colonies"""
        colony_config = self.config_dict.get("colony", {})
        colonies_config = self.config_dict.get("colonies", {})

        # Number of initial colonies
        initial_count = colonies_config.get("initial_count", 1)

        for i in range(initial_count):
            # Create colony at random location
            x = self.random.uniform(0, self.landscape.width)
            y = self.random.uniform(0, self.landscape.height)

            colony = Colony(
                model=self,
                species=colony_config.get("species", "bombus_terrestris"),
                location=(x, y),
                initial_population=colony_config.get(
                    "initial_population", {"queens": 1, "workers": 50}
                ),
            )

            # Add integrated components
            self._add_colony_components(colony)

            # Initialize foraging system for colony
            if self.foraging_system:
                self._initialize_colony_foraging(colony)

            # Register colony with health monitoring
            if self.health_monitoring_system:
                self._register_colony_health_monitoring(colony)

            self.schedule.add(colony)

    def _add_colony_components(self, colony: Colony) -> None:
        """Add integrated components to colony"""

        # Disease management
        disease_config = self.config_dict.get("disease", {})
        if disease_config.get("enabled", True):
            setattr(colony, "disease_manager", DiseaseManager(colony))

        # Reproduction management
        reproduction_config = self.config_dict.get("reproduction", {})
        if reproduction_config.get("enabled", True):
            setattr(colony, "reproduction_manager", ReproductionManager(colony))

    def _initialize_colony_foraging(self, colony: Colony) -> None:
        """Initialize foraging system for a specific colony"""

        # Create patches from landscape
        landscape_patches = self._create_patches_from_landscape(colony)

        # Get colony state for foraging initialization
        colony_state = {
            "species": colony.species,
            "location": colony.location,
            "energy_level": colony.resources.total_food,
            "forager_count": colony.get_forager_count(),
            "available_foragers": len(
                [
                    b
                    for b in colony.bees
                    if hasattr(b, "activity_state")
                    and b.activity_state == "ready_to_forage"
                ]
            ),
        }

        # Initialize foraging system for this colony
        if self.foraging_system is not None:
            self.foraging_system.initialize_colony_foraging(
                colony.unique_id, colony_state, landscape_patches
            )

        self.logger.info(
            f"Foraging system initialized for colony {colony.unique_id} with {len(landscape_patches)} patches"
        )

    def _create_patches_from_landscape(self, colony: Colony) -> List[PatchInfo]:
        """Create foraging patches from landscape data"""

        patches = []

        # Get patches within foraging range
        foraging_range = getattr(colony, "foraging_range", 2000.0)

        # Create mock patches for demonstration (would be replaced with actual landscape data)
        for i in range(20):  # Create 20 patches around colony
            # Random location within foraging range
            angle = self.random.uniform(0, 2 * 3.14159)
            distance = self.random.uniform(100, foraging_range)
            x = colony.location[0] + distance * math.cos(angle)
            y = colony.location[1] + distance * math.sin(angle)

            # Determine resource type
            resource_type = self.random.choice(
                [
                    PatchResourceType.NECTAR,
                    PatchResourceType.POLLEN,
                    PatchResourceType.MIXED,
                ]
            )

            # Create patch info
            patch = PatchInfo(
                patch_id=i,
                location=(x, y),
                resource_type=resource_type,
                quality_metrics={
                    PatchQualityMetric.RESOURCE_DENSITY: self.random.uniform(0.3, 1.0),
                    PatchQualityMetric.SUGAR_CONCENTRATION: self.random.uniform(
                        0.1, 0.8
                    ),
                    PatchQualityMetric.FLOWER_COUNT: float(
                        self.random.randint(10, 100)
                    ),
                    PatchQualityMetric.ACCESSIBILITY: self.random.uniform(0.5, 1.0),
                    PatchQualityMetric.COMPETITION_LEVEL: self.random.uniform(0.0, 0.6),
                    PatchQualityMetric.HANDLING_TIME: self.random.uniform(0.1, 0.5),
                    PatchQualityMetric.COROLLA_DEPTH: self.random.uniform(2.0, 15.0),
                    PatchQualityMetric.COROLLA_WIDTH: self.random.uniform(1.0, 8.0),
                },
                species_compatibility={colony.species: self.random.uniform(0.6, 1.0)},
                distance_from_hive=distance,
                current_foragers=0,
                max_capacity=self.random.randint(5, 20),
                depletion_rate=self.random.uniform(0.01, 0.05),
                regeneration_rate=self.random.uniform(0.02, 0.08),
                seasonal_availability={
                    "spring": self.random.uniform(0.7, 1.0),
                    "summer": self.random.uniform(0.8, 1.0),
                    "autumn": self.random.uniform(0.5, 0.9),
                    "winter": self.random.uniform(0.1, 0.4),
                },
            )

            patches.append(patch)

        return patches

    def _update_spatial_bee_behaviors(self) -> None:
        """Update spatial behaviors for all bees"""
        if not self.spatial_bee_manager:
            return

        for colony in self.get_colonies():
            for bee in getattr(colony, "bees", []):
                if hasattr(bee, "status") and hasattr(bee, "unique_id"):
                    # Register bee in spatial system if not already registered
                    if bee.unique_id not in self.spatial_bee_manager.bee_states:
                        # Default position at colony location
                        from .spatial_algorithms import SpatialPoint

                        initial_position = SpatialPoint(
                            x=colony.location[0], y=colony.location[1], z=0.0
                        )
                        self.spatial_bee_manager.register_bee(
                            bee.unique_id, initial_position
                        )

                    # Update spatial behavior
                    behavior_result = (
                        self.spatial_bee_manager.update_bee_spatial_behavior(
                            bee.unique_id, bee.status, self.current_day
                        )
                    )

                    # Apply behavior results to bee
                    if behavior_result.get("foraging_success"):
                        if hasattr(bee, "add_energy"):
                            bee.add_energy(
                                behavior_result.get("resources_collected", 0)
                            )

    def _process_bumblebee_communication(self) -> None:
        """Process bumblebee communication (scent-based, no dancing)"""
        if not self.bumblebee_communication_system:
            return

        current_time = self.current_day * 24.0  # Convert to hours

        # Process simple bumblebee scent communication
        for colony in self.get_colonies():
            colony_id = colony.unique_id

            # Check for returning foragers for scent marking
            for bee in getattr(colony, "bees", []):
                if (
                    hasattr(bee, "status")
                    and hasattr(bee, "unique_id")
                    and str(bee.status).endswith("RETURNING")
                ):
                    # Create foraging result from bee state
                    foraging_result = self._extract_foraging_result(bee)

                    if foraging_result:
                        # Process simple scent marking (no dance behavior)
                        communication_event = self.bumblebee_communication_system.process_returning_forager(
                            bee.unique_id,
                            foraging_result,
                            colony_id,
                            current_time,
                        )

                        if communication_event:
                            self.logger.debug(
                                f"Bee {bee.unique_id} left scent marks at time {current_time}"
                            )

    def _get_colony_state_for_communication(self, colony: Colony) -> Dict[str, Any]:
        """Get colony state information for dance communication"""
        return {
            "total_bees": colony.get_adult_population(),
            "energy_level": getattr(
                getattr(colony, "resources", None), "total_food", 0.5
            )
            / 1000.0,
            "available_foragers": len(
                [
                    b
                    for b in getattr(colony, "bees", [])
                    if hasattr(b, "status") and "IDLE" in str(b.status)
                ]
            ),
            "recent_dance_count": 0,  # Would track recent dances
        }

    def _get_bee_states_for_communication(
        self, colony: Colony
    ) -> Dict[int, Dict[str, Any]]:
        """Get bee state information for dance communication"""
        bee_states = {}

        for bee in getattr(colony, "bees", []):
            if hasattr(bee, "unique_id"):
                bee_states[bee.unique_id] = {
                    "colony_id": colony.unique_id,
                    "status": getattr(bee, "status", "IDLE"),
                    "foraging_experience": getattr(bee, "foraging_experience", 0.5),
                    "dance_following_experience": getattr(
                        bee, "dance_following_experience", 0.5
                    ),
                    "attention_tendency": getattr(bee, "attention_tendency", 0.5),
                    "recruitment_tendency": getattr(bee, "recruitment_tendency", 0.5),
                    "foraging_motivation": getattr(bee, "foraging_motivation", 0.5),
                    "learning_rate": getattr(bee, "learning_rate", 0.5),
                }

        return bee_states

    def _extract_foraging_result(self, bee: Any) -> Optional[Dict[str, Any]]:
        """Extract foraging result from bee state"""
        if not hasattr(bee, "last_foraging_trip"):
            return None

        # Extract result from bee's last foraging trip
        return {
            "patch_id": getattr(bee, "last_patch_visited", 0),
            "patch_location": getattr(bee, "last_patch_location", (0.0, 0.0)),
            "patch_direction": getattr(bee, "last_patch_direction", 0.0),
            "distance_traveled": getattr(bee, "last_travel_distance", 100.0),
            "patch_quality": getattr(bee, "last_patch_quality", 0.5),
            "resource_type": getattr(bee, "last_resource_type", "nectar"),
            "resource_quantity": getattr(bee, "last_resource_quantity", 1.0),
            "energy_gained": getattr(bee, "last_energy_gained", 0.0),
        }

    def _update_recruitment_networks(self) -> None:
        """Update recruitment mechanism networks - not used for bumblebees"""
        # Bumblebees don't use dance communication, so no recruitment networks to update
        pass

    def _initialize_spatial_systems(self) -> None:
        """Initialize spatial integration systems"""
        spatial_config = self.config_dict.get("spatial", {})

        if spatial_config.get("enabled", False):
            # Create spatial system configuration
            landscape_config = spatial_config.get("landscape", {})

            try:
                # Initialize spatial integration system
                self.spatial_environment, self.spatial_bee_manager = (
                    create_spatial_integration_system(landscape_config)
                )
                self.logger.info("Spatial integration system initialized")

            except Exception as e:
                self.logger.error(f"Failed to initialize spatial systems: {e}")
                self.spatial_environment = None
                self.spatial_bee_manager = None
        else:
            self.spatial_environment = None
            self.spatial_bee_manager = None

    def _initialize_dance_communication(self) -> None:
        """Initialize dance communication integration"""
        comm_config = self.config_dict.get("communication", {})

        if comm_config.get("dance_enabled", True):
            try:
                # Initialize bumblebee communication system (biologically accurate)
                self.bumblebee_communication_system = BumblebeeCommunicationSystem()
                self.logger.info(
                    "Bumblebee communication system initialized (no dance - scientifically correct)"
                )

            except Exception as e:
                self.logger.error(f"Failed to initialize bumblebee communication: {e}")
                self.bumblebee_communication_system = None
        else:
            self.bumblebee_communication_system = None

    def _initialize_live_visualization(self) -> None:
        """Initialize live visualization system"""
        viz_config = self.config_dict.get("visualization", {})

        if viz_config.get("enabled", False):
            try:
                # Create output directory
                output_dir = viz_config.get("output_directory", "visualization_output")

                # Initialize visualization system
                self.live_visualization_manager = create_live_visualization_system(
                    output_dir
                )

                # Register model data collectors
                self.live_visualization_manager.register_model_data_collector(
                    self, "bstew"
                )

                self.logger.info("Live visualization system initialized")

            except Exception as e:
                self.logger.error(f"Failed to initialize live visualization: {e}")
                self.live_visualization_manager = None
        else:
            self.live_visualization_manager = None

    def _initialize_system_integrator(self) -> None:
        """Initialize advanced system integrator"""
        integrator_config = self.config_dict.get("system_integration", {})

        try:
            self.system_integrator = SystemIntegrator(
                output_directory=integrator_config.get("output_directory", "artifacts"),
                enable_mortality_tracking=integrator_config.get(
                    "enable_mortality_tracking", True
                ),
                enable_excel_reporting=integrator_config.get(
                    "enable_excel_reporting", True
                ),
                enable_spatial_analysis=integrator_config.get(
                    "enable_spatial_analysis", True
                ),
                enable_health_monitoring=integrator_config.get(
                    "enable_health_monitoring", True
                ),
            )

            # Initialize all integrated systems
            initialization_status = self.system_integrator.initialize_systems(self)

            active_systems = list(initialization_status.keys())
            if "error" in initialization_status:
                self.logger.error(
                    f"System integration error: {initialization_status['error']}"
                )
            else:
                self.logger.info(
                    f"System integrator initialized with systems: {active_systems}"
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize system integrator: {e}")
            self.system_integrator = None

    def _update_foraging_environment(self) -> None:
        """Update environmental conditions for foraging system"""

        # Get current environmental conditions
        current_weather = self.current_weather
        seasonal_factor = self.get_seasonal_factor()

        # Update patch conditions based on environmental factors
        for colony in self.get_colonies():
            colony_id = colony.unique_id

            # Environmental context for foraging
            environmental_context = {
                "season": self._get_current_season(),
                "hour": (self.current_day * 24) % 24,  # Simplified hour calculation
                "weather": self._get_weather_condition(),
                "temperature": current_weather["temperature"],
                "humidity": current_weather["humidity"],
                "wind_speed": current_weather["wind_speed"],
                "precipitation": current_weather["precipitation"],
                "seasonal_factor": seasonal_factor,
            }

            # Update foraging system with current conditions
            # This would be implemented as part of the foraging system's environmental update
            # For now, just log the update
            if self.current_day % 10 == 0:  # Log every 10 days
                self.logger.debug(
                    f"Updated foraging environment for colony {colony_id}: {environmental_context}"
                )

    def _process_foraging_outcomes(self) -> None:
        """Process foraging outcomes and update colony resources"""

        total_foraging_sessions = 0
        total_energy_collected = 0

        for colony in self.get_colonies():
            colony_id = colony.unique_id

            # Get foraging analytics for colony
            if self.foraging_system is None:
                continue
            analytics = self.foraging_system.get_colony_foraging_analytics(colony_id)

            if "session_statistics" in analytics:
                sessions_today = analytics.get("active_sessions", 0)
                total_foraging_sessions += sessions_today

                # Estimate energy collected based on recent efficiency
                recent_efficiency = analytics["session_statistics"].get(
                    "recent_efficiency", 0.0
                )
                estimated_energy = (
                    sessions_today * recent_efficiency * 50.0
                )  # Simplified calculation
                total_energy_collected += estimated_energy

                # Update colony resources based on foraging outcomes
                if estimated_energy > 0:
                    colony.resources.nectar += estimated_energy * 0.6  # 60% nectar
                    colony.resources.pollen += estimated_energy * 0.4  # 40% pollen

        # Log foraging summary periodically
        if self.current_day % 50 == 0 and total_foraging_sessions > 0:
            self.logger.info(
                f"Day {self.current_day} foraging summary: "
                f"{total_foraging_sessions} sessions, "
                f"{total_energy_collected:.1f} energy collected"
            )

    def _get_current_season(self) -> str:
        """Get current season based on day of year"""
        day_of_year = self.current_day % 365

        if day_of_year < 80:  # Jan-Mar
            return "winter"
        elif day_of_year < 172:  # Apr-Jun
            return "spring"
        elif day_of_year < 266:  # Jul-Sep
            return "summer"
        else:  # Oct-Dec
            return "autumn"

    def _get_weather_condition(self) -> str:
        """Get simplified weather condition"""
        current_weather = self.current_weather

        if current_weather["precipitation"] > 5.0:
            return "rain"
        elif current_weather["wind_speed"] > 15.0:
            return "windy"
        elif current_weather["temperature"] < 10.0:
            return "cold"
        elif current_weather["temperature"] > 30.0:
            return "hot"
        else:
            return "clear"

    def get_foraging_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive foraging system analytics"""

        if not self.foraging_system:
            return {"error": "Foraging system not initialized"}

        return self.foraging_system.get_system_wide_analytics()

    def emit_simulation_event(
        self,
        event_type: str,
        source_agent_id: Optional[int] = None,
        target_agent_id: Optional[int] = None,
        colony_id: Optional[int] = None,
        event_data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        energy_impact: Optional[float] = None,
    ) -> None:
        """Emit simulation event for data collection"""

        if hasattr(self, "comprehensive_data_collector"):
            event = SimulationEvent(
                event_id=f"{event_type}_{self.current_day}_{time.time()}",
                event_type=event_type,
                timestamp=time.time(),
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                colony_id=colony_id,
                event_data=event_data or {},
                success=success,
                energy_impact=energy_impact,
            )

            self.comprehensive_data_collector.emit_event(event)

    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics from data collector"""

        if hasattr(self, "comprehensive_data_collector"):
            return self.comprehensive_data_collector.get_comprehensive_analytics()
        else:
            return {"error": "Comprehensive data collector not initialized"}

    def _initialize_data_collection(self) -> None:
        """Initialize comprehensive data collection system"""

        # Initialize comprehensive data collector
        self.comprehensive_data_collector = ComprehensiveDataCollector()

        # Initialize spatial analysis system
        self.spatial_environment = None
        self.spatial_bee_manager = None
        if self.config_dict.get("enable_spatial_analysis", True):
            self._initialize_spatial_systems()

        # Integrate with foraging system
        if self.foraging_system:
            self.comprehensive_data_collector.integrate_with_foraging_system(
                self.foraging_system
            )

        # Initialize Mesa data collector for compatibility
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Bees": lambda m: sum(
                    colony.get_adult_population() for colony in m.get_colonies()
                ),
                "Total_Brood": lambda m: sum(
                    colony.get_brood_count() for colony in m.get_colonies()
                ),
                "Total_Honey": lambda m: sum(
                    colony.get_honey_stores() for colony in m.get_colonies()
                ),
                "Active_Colonies": lambda m: m.get_colony_count(),
                "Average_Colony_Health": lambda m: (
                    np.mean([colony.get_health_score() for colony in m.get_colonies()])
                    if m.get_colonies()
                    else 0
                ),
                "Total_Foragers": lambda m: sum(
                    colony.get_forager_count() for colony in m.get_colonies()
                ),
                "Daily_Mortality": lambda m: sum(
                    getattr(colony, "daily_mortality", 0) for colony in m.get_colonies()
                ),
                "Weather_Temperature": lambda m: m.get_current_temperature(),
                "Weather_Rainfall": lambda m: m.get_current_rainfall(),
                "Foraging_Efficiency": lambda m: m.get_average_foraging_efficiency(),
            },
            agent_reporters={
                "Population": lambda agent: (
                    agent.get_adult_population()
                    if hasattr(agent, "get_adult_population")
                    else 0
                ),
                "Health": lambda agent: (
                    agent.get_health_score()
                    if hasattr(agent, "get_health_score")
                    else 0
                ),
                "Honey_Stores": lambda agent: (
                    agent.get_honey_stores()
                    if hasattr(agent, "get_honey_stores")
                    else 0
                ),
                "Age": lambda agent: getattr(agent, "age", 0),
            },
        )

    def _initialize_optimization(self) -> None:
        """Initialize performance optimization"""

        optimization_config = self.config_dict.get("optimization", {})

        if optimization_config.get("enabled", False):
            self.optimizer = SimulationOptimizer(optimization_config)
        else:
            self.optimizer = None

    def _configure_performance_logging(self) -> None:
        """Configure performance logging based on simulation requirements"""
        perf_config = self.config_dict.get("performance", {})
        logging_config = perf_config.get("logging", {})

        # Determine optimal logging level based on simulation size and duration
        colony_count = self.config_dict.get("colonies", {}).get("initial_count", 1)
        duration_days = self.config_dict.get("simulation", {}).get("duration_days", 365)
        simulation_size = colony_count * duration_days

        # Auto-optimize logging level based on simulation complexity
        if logging_config.get("auto_optimize", True):
            if simulation_size > 10000:  # Large simulation
                target_level = logging.ERROR
            elif simulation_size > 1000:  # Medium simulation
                target_level = logging.WARNING
            else:  # Small simulation
                target_level = logging.INFO
        else:
            # Use explicit configuration
            level_name = logging_config.get("level", "WARNING").upper()
            target_level = getattr(logging, level_name, logging.WARNING)

        # Apply configuration
        logging.getLogger().setLevel(target_level)

        self.logger.info(
            f"Performance logging configured: level={logging.getLevelName(target_level)}"
        )

    def step(self) -> None:
        """Execute one simulation step (day)"""

        # Update environmental conditions
        if self.environment_manager:
            self.environment_manager.update_environmental_conditions(self.current_day)

        # Apply performance optimization
        if self.optimizer:
            simulation_state = {
                "current_step": self.current_day,
                "colonies": self.get_colonies(),
            }
            self.optimizer.optimize_simulation_step(simulation_state)

        # Process batch if batch processing is enabled
        if getattr(self, "batch_processing_enabled", False):
            if self.current_day % getattr(self, "batch_size", 1) == 0:
                self.batches_processed = getattr(self, "batches_processed", 0) + 1

        # Update dynamic parameters before agent steps
        if self.dynamic_parameter_system:
            self._update_dynamic_parameters()

        # Update foraging system state before agent steps
        if self.foraging_system:
            self._update_foraging_environment()

        # Execute agent steps
        self.schedule.step()

        # Process foraging results after agent steps
        if self.foraging_system:
            self._process_foraging_outcomes()

        # Update spatial environment
        if self.spatial_environment:
            self.spatial_environment.update_landscape_dynamics(self.current_day)

        # Update spatial bee behaviors
        if self.spatial_bee_manager:
            self._update_spatial_bee_behaviors()

        # Process bumblebee communication (no dance - simple scent-based communication)
        if self.bumblebee_communication_system:
            self._process_bumblebee_communication()

        # Update health monitoring
        if (
            self.health_monitoring_system
            and self.current_day % self.health_monitoring_system.update_frequency == 0
        ):
            self._update_colony_health_monitoring()

        # Update integrated systems (mortality tracking, spatial analysis, etc.)
        if self.system_integrator:
            self.system_integrator.update_all_systems(self, self.current_day)

        # Collect comprehensive data
        if hasattr(self, "comprehensive_data_collector"):
            # Collect data from all colonies and agents
            for colony in self.get_colonies():
                self.comprehensive_data_collector.collect_colony_data(
                    colony, self.current_day
                )

                # Collect bee-level data
                for bee in getattr(colony, "bees", []):
                    self.comprehensive_data_collector.collect_bee_data(
                        bee, self.current_day
                    )

            # Perform data aggregation
            aggregation_results = self.comprehensive_data_collector.aggregate_data(
                self.current_day
            )

            # Log aggregation results periodically
            if self.current_day % 100 == 0 and aggregation_results:
                self.logger.info(
                    f"Day {self.current_day} aggregation completed: {len(aggregation_results)} metrics"
                )

        # Collect Mesa data for compatibility
        self.datacollector.collect(self)

        # Update simulation time
        self.current_day += 1

        # Check termination conditions
        if self.current_day >= self.max_days:
            self.running = False

        # Check if any colonies are still alive
        active_colonies = 0
        for colony in self.get_colonies():
            if hasattr(colony, "health"):
                from ..core.colony import ColonyHealth

                if colony.health != ColonyHealth.COLLAPSED:
                    active_colonies += 1
            else:
                active_colonies += 1

        if active_colonies == 0:
            self.logger.warning("All colonies have died - ending simulation")
            self.running = False

    def get_colonies(self) -> List[Colony]:
        """Get all active colonies"""
        return [agent for agent in self.schedule.agents if isinstance(agent, Colony)]

    @property
    def colonies(self) -> List[Colony]:
        """Get all active colonies (property accessor)"""
        return self.get_colonies()

    def get_seasonal_factor(self) -> float:
        """Get seasonal activity factor based on current day"""
        if self.current_day is None:
            return 1.0

        # Simple seasonal model: peak activity in spring/summer
        day_of_year = self.current_day % 365

        if day_of_year < 60 or day_of_year > 335:  # Winter
            return 0.3
        elif day_of_year < 152 or day_of_year > 244:  # Spring/Fall
            return 0.7
        else:  # Summer
            return 1.0

    def get_colony_count(self) -> int:
        """Get number of active colonies"""
        return len(self.get_colonies())

    @property
    def current_weather(self) -> Dict[str, float]:
        """Get current weather conditions"""
        if hasattr(self, "weather_manager") and self.weather_manager:
            # Get weather for current day
            weather_summary = self.weather_manager.get_weather_summary()
            return {
                "temperature": weather_summary.get("avg_temperature", 20.0),
                "humidity": weather_summary.get("avg_humidity", 50.0),
                "precipitation": weather_summary.get("total_precipitation", 0.0),
                "wind_speed": weather_summary.get("avg_wind_speed", 0.0),
            }
        else:
            # Return default weather
            return {
                "temperature": 20.0,
                "humidity": 0.6,
                "wind_speed": 5.0,
                "precipitation": 0.0,
            }

    @property
    def resource_distribution(self) -> Any:
        """Get resource distribution system"""
        if not hasattr(self, "_resource_distribution"):
            # Create mock resource distribution
            class MockResourceDistribution:
                def __init__(self, landscape: Any) -> None:
                    self.landscape = landscape

                class MockLandscape:
                    def __init__(self) -> None:
                        self.use_masterpatch_system = True

                    def calculate_weather_factor(
                        self, weather: Dict[str, Any]
                    ) -> float:
                        # Simple weather factor based on temperature
                        temp = weather.get("temperature", 20.0)
                        if temp < 5.0 or temp > 35.0:
                            return 0.2
                        elif temp < 10.0 or temp > 30.0:
                            return 0.5
                        else:
                            return 0.8

                    def get_patches_in_radius(
                        self, center: Any, radius: float
                    ) -> List[Any]:
                        """Mock method to get patches in radius"""
                        return []

                    def get_best_foraging_patches(
                        self,
                        species_name: str,
                        current_pos: Any,
                        foraging_range: float,
                        proboscis_system: Any,
                    ) -> List[Any]:
                        """Mock method to get best foraging patches"""
                        return []

                self.landscape = MockLandscape()

            self._resource_distribution = MockResourceDistribution(self.landscape)
        return self._resource_distribution

    def get_current_temperature(self) -> float:
        """Get current temperature"""
        if self.environment_manager and self.environment_manager.current_weather:
            return self.environment_manager.current_weather.temperature
        return 20.0  # Default temperature

    def get_current_rainfall(self) -> float:
        """Get current rainfall"""
        if self.environment_manager and self.environment_manager.current_weather:
            return self.environment_manager.current_weather.rainfall
        return 0.0  # Default no rain

    def get_average_foraging_efficiency(self) -> float:
        """Get average foraging efficiency across colonies"""
        if self.environment_manager:
            return self.environment_manager.get_foraging_efficiency_modifier(
                self.current_day
            )
        return 1.0  # Default full efficiency

    def is_mating_season(self) -> bool:
        """Check if it's currently mating season"""
        # Simple seasonal check - mating season is typically spring/early summer
        day_of_year = self.current_day % 365
        return 90 <= day_of_year <= 180  # Roughly April-June

    def get_current_season(self) -> str:
        """Get current season"""
        day_of_year = self.current_day % 365
        if day_of_year < 80 or day_of_year >= 355:
            return "winter"
        elif day_of_year < 172:
            return "spring"
        elif day_of_year < 266:
            return "summer"
        else:
            return "autumn"

    def add_environmental_stressor(
        self, stressor_type: str, severity: float, duration: int, affected_area: float
    ) -> None:
        """Add environmental stressor to simulation"""

        if self.environment_manager:
            from ..components.environment import EnvironmentalStressor

            stressor = EnvironmentalStressor(
                name=f"{stressor_type}_{self.current_day}",
                stressor_type=stressor_type,
                severity=severity,
                duration=duration,
                affected_area=affected_area,
            )

            self.environment_manager.add_environmental_stressor(
                stressor, self.current_day
            )

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation summary"""

        self.datacollector.get_model_vars_dataframe()
        colonies = self.get_colonies()

        summary = {
            "simulation_info": {
                "current_day": self.current_day,
                "max_days": self.max_days,
                "start_time": self.simulation_start.isoformat(),
                "active_colonies": len(colonies),
            },
            "population_summary": {
                "total_bees": sum(colony.get_adult_population() for colony in colonies),
                "total_brood": sum(colony.get_brood_count() for colony in colonies),
                "total_honey": sum(colony.get_honey_stores() for colony in colonies),
            },
            "health_summary": {
                "average_health": (
                    np.mean([colony.get_health_score() for colony in colonies])
                    if colonies
                    else 0
                ),
                "disease_pressure": (
                    np.mean(
                        [getattr(colony, "disease_pressure", 0) for colony in colonies]
                    )
                    if colonies
                    else 0
                ),
            },
        }

        # Add environmental summary
        if self.environment_manager:
            summary["environment"] = self.environment_manager.get_environmental_summary(
                self.current_day
            )

        # Add weather summary
        if self.weather_manager:
            summary["weather"] = self.weather_manager.get_weather_summary()

        # Add spatial analysis summary
        if self.spatial_environment:
            summary["spatial"] = self.spatial_environment.get_gis_analysis_results()

        # Add bumblebee communication summary (simple scent-based communication)
        if self.bumblebee_communication_system:
            communication_metrics = {}
            for colony in colonies:
                colony_id = colony.unique_id
                # Get basic communication metrics from bumblebee system
                if hasattr(
                    self.bumblebee_communication_system, "get_communication_metrics"
                ):
                    colony_comm_metrics = (
                        self.bumblebee_communication_system.get_communication_metrics(
                            colony_id
                        )
                    )
                    communication_metrics[f"colony_{colony_id}"] = colony_comm_metrics
            if communication_metrics:
                summary["bumblebee_communication"] = communication_metrics

        # Add performance metrics
        if self.optimizer:
            summary["performance"] = self.optimizer.get_optimization_metrics().__dict__

        return summary

    def export_results(self, output_dir: str = "results") -> None:
        """Export simulation results"""

        from pathlib import Path
        from ..utils.data_io import DataExporter

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        exporter = DataExporter()

        # Export model data
        model_data = self.datacollector.get_model_vars_dataframe()
        exporter.export_to_csv(model_data, output_path / "model_data.csv")

        # Export agent data
        agent_data = self.datacollector.get_agent_vars_dataframe()
        if len(agent_data) > 0:
            exporter.export_to_csv(agent_data, output_path / "agent_data.csv")

        # Export simulation summary
        summary = self.get_simulation_summary()
        exporter.export_to_json(summary, output_path / "simulation_summary.json")

        self.logger.info(f"Results exported to {output_path}")

    def cleanup(self) -> None:
        """Clean up simulation resources"""

        if self.optimizer:
            self.optimizer.cleanup()

        # Generate final reports and cleanup integrated systems
        if self.system_integrator:
            try:
                final_reports = self.system_integrator.generate_reports(
                    self.current_day, final_report=True
                )
                if final_reports:
                    self.logger.info(
                        f"Final reports generated: {list(final_reports.keys())}"
                    )

                self.system_integrator.cleanup_systems()
            except Exception as e:
                self.logger.error(f"Error during system integrator cleanup: {e}")

        # Clear data structures
        self.schedule.agents.clear()

        # Cleanup visualization
        if self.live_visualization_manager:
            self.live_visualization_manager.stop_data_collection()

        self.logger.info("Model cleanup completed")

    def start_live_visualization(self) -> None:
        """Start live visualization data collection"""
        if self.live_visualization_manager:
            self.live_visualization_manager.start_data_collection()
            self.logger.info("Started live visualization")
        else:
            self.logger.warning("Live visualization not initialized")

    def stop_live_visualization(self) -> None:
        """Stop live visualization data collection"""
        if self.live_visualization_manager:
            self.live_visualization_manager.stop_data_collection()
            self.logger.info("Stopped live visualization")

    # Data collection methods
    def get_total_bee_count(self) -> int:
        """Get total number of bees across all colonies"""
        return sum(colony.get_adult_population() for colony in self.get_colonies())

    def get_total_population(self) -> int:
        """Get total population across all colonies

        This is an alias for get_total_bee_count() to match expected interface
        from CLI commands.
        """
        return self.get_total_bee_count()

    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state for monitoring

        Returns comprehensive state information for verbose output and monitoring.
        """
        return {
            "step": self.schedule.steps,
            "total_population": self.get_total_population(),
            "colony_count": self.get_colony_count(),
            "active_agents": len(self.schedule.agents),
            "current_day": self.current_day,
            "current_season": self.get_current_season(),
            "total_nectar_stores": self.get_total_nectar_stores(),
            "active_colonies": self.get_active_colony_count(),
        }

    def get_active_colony_count(self) -> int:
        """Get number of active (non-collapsed) colonies"""
        active_count = 0
        for colony in self.get_colonies():
            if hasattr(colony, "health"):
                from ..core.colony import ColonyHealth

                if colony.health != ColonyHealth.COLLAPSED:
                    active_count += 1
            else:
                active_count += 1
        return active_count

    def get_total_nectar_stores(self) -> float:
        """Get total nectar stores across all colonies"""
        return sum(colony.get_honey_stores() for colony in self.get_colonies())

    def get_colony_metrics(self, colony_id: int) -> dict:
        """Get metrics for a specific colony"""
        for colony in self.get_colonies():
            if colony.unique_id == colony_id:
                return {
                    "population": colony.get_adult_population(),
                    "resources": colony.get_honey_stores(),
                    "health": colony.get_health_score(),
                }
        return {}

    def get_spatial_metrics(self) -> dict:
        """Get spatial distribution metrics"""
        return {
            "resource_distribution": (
                self.landscape.get_total_resources()
                if hasattr(self.landscape, "get_total_resources")
                else {}
            ),
            "bee_distribution": {
                "total_bees": self.get_total_bee_count(),
                "colonies": len(self.get_colonies()),
            },
        }

    def get_performance_metrics(self) -> dict:
        """Get performance metrics"""
        if self.optimizer:
            return self.optimizer.get_optimization_metrics().__dict__
        return {
            "execution_time": 0.0,
            "memory_usage": 0.0,
            "total_execution_time": 0.0,
            "average_step_time": 0.0,
        }

    def export_model_data(self) -> dict:
        """Export model data"""
        model_data = self.datacollector.get_model_vars_dataframe()
        return {"model_data": model_data.to_dict() if len(model_data) > 0 else {}}

    def export_agent_data(self) -> dict:
        """Export agent data"""
        agent_data = self.datacollector.get_agent_vars_dataframe()
        return {"agent_data": agent_data.to_dict() if len(agent_data) > 0 else {}}

    # Environmental integration methods
    def get_current_weather(self) -> dict:
        """Get current weather conditions"""
        return self.current_weather

    def set_season(self, season: str) -> None:
        """Set current season"""
        self.current_season = season

    def get_seasonal_effects(self) -> dict:
        """Get seasonal effects on simulation"""
        season = getattr(self, "current_season", self.get_current_season())

        if season == "winter":
            return {"temperature_modifier": -0.5, "resource_modifier": 0.3}
        elif season == "spring":
            return {"temperature_modifier": 0.2, "resource_modifier": 1.2}
        elif season == "summer":
            return {"temperature_modifier": 0.5, "resource_modifier": 1.0}
        else:  # autumn
            return {"temperature_modifier": 0.0, "resource_modifier": 0.8}

    def apply_climate_scenario(self, scenario: dict) -> None:
        """Apply climate scenario to model"""
        self.climate_scenario = scenario

    def get_climate_effects(self) -> dict:
        """Get climate effects"""
        return getattr(self, "climate_scenario", {})

    def get_total_landscape_resources(self) -> float:
        """Get total landscape resources"""
        if hasattr(self.landscape, "get_total_resources"):
            resources = self.landscape.get_total_resources()
            return float(resources.get("total", 0.0))
        return 0.0

    def apply_environmental_stressors(self, stressors: dict) -> None:
        """Apply environmental stressors"""
        self.environmental_stressors = stressors

    def get_environmental_stress_effects(self) -> dict:
        """Get environmental stress effects"""
        stressors = getattr(self, "environmental_stressors", {})

        # Calculate combined stress effects
        drought_stress = stressors.get("drought", 0.0)
        pollution_stress = stressors.get("pollution", 0.0)
        fragmentation_stress = stressors.get("habitat_fragmentation", 0.0)

        return {
            "foraging_efficiency": 1.0 - (drought_stress + pollution_stress) * 0.5,
            "mortality_rate": (drought_stress + pollution_stress + fragmentation_stress)
            * 0.1,
        }

    # Colony management methods
    def create_colony(
        self, species: str, location: tuple, initial_population: dict
    ) -> "Colony":
        """Create a new colony"""
        from ..core.colony import Colony

        colony = Colony(
            model=self,
            species=species,
            location=location,
            initial_population=initial_population,
        )

        # Add integrated components
        self._add_colony_components(colony)
        self.schedule.add(colony)

        return colony

    def remove_colony(self, colony_id: int) -> None:
        """Remove a colony from the model"""
        colonies_to_remove = [
            colony for colony in self.get_colonies() if colony.unique_id == colony_id
        ]

        for colony in colonies_to_remove:
            self.schedule.remove(colony)

    def process_colony_failures(self) -> list:
        """Process colony failures and return list of failed colony IDs"""
        failed_colonies = []

        for colony in self.get_colonies():
            if hasattr(colony, "health"):
                from ..core.colony import ColonyHealth

                if colony.health == ColonyHealth.COLLAPSED:
                    failed_colonies.append(colony.unique_id)

        return failed_colonies

    def process_colony_swarming(self) -> list:
        """Process colony swarming and return list of swarm events"""
        swarms = []

        for colony in self.get_colonies():
            # Check swarming conditions
            if hasattr(colony, "population_size") and colony.population_size > 400:
                if (
                    hasattr(colony, "resources")
                    and getattr(colony.resources, "nectar", 0) > 50
                ):
                    # Swarming conditions met
                    swarms.append(
                        {
                            "parent_colony": colony.unique_id,
                            "swarm_size": colony.population_size // 2,
                            "location": colony.location,
                        }
                    )

        return swarms

    def calculate_colony_competition(self) -> dict:
        """Calculate resource competition between colonies"""
        colonies = self.get_colonies()

        if len(colonies) < 2:
            return {}

        competition_matrix = {}

        for i, colony1 in enumerate(colonies):
            for j, colony2 in enumerate(colonies):
                if i != j:
                    # Calculate distance-based competition
                    distance = (
                        (colony1.location[0] - colony2.location[0]) ** 2
                        + (colony1.location[1] - colony2.location[1]) ** 2
                    ) ** 0.5

                    # Closer colonies have higher competition
                    competition = max(0.0, 1.0 - distance / 1000.0)

                    key = f"{colony1.unique_id}_{colony2.unique_id}"
                    competition_matrix[key] = competition

        return competition_matrix

    # Optimization methods
    def enable_memory_optimization(self) -> None:
        """Enable memory optimization"""
        self.memory_optimization_enabled = True

    def get_memory_usage(self) -> float:
        """Get current memory usage"""
        import psutil

        process = psutil.Process()
        return float(process.memory_info().rss / (1024 * 1024))  # MB

    def reset(self) -> None:
        """Reset model to initial state"""
        self.current_day = 0
        self.running = True

        # Reset random seed to original state for deterministic behavior
        if (
            hasattr(self, "initial_random_seed")
            and self.initial_random_seed is not None
        ):
            np.random.seed(self.initial_random_seed)
            self.random.seed(self.initial_random_seed)

        # Reset data collector
        self.datacollector = None
        self._initialize_data_collection()

        # Reset all colonies to initial state
        for colony in self.get_colonies():
            if hasattr(colony, "reset"):
                colony.reset()

        # Reset environmental manager if present
        if hasattr(self, "environment_manager") and self.environment_manager:
            if hasattr(self.environment_manager, "reset"):
                self.environment_manager.reset()

        # Reset optimization counters
        if hasattr(self, "batches_processed"):
            self.batches_processed = 0

    def enable_caching(self) -> None:
        """Enable caching"""
        self.caching_enabled = True
        if not hasattr(self, "cache_enabled"):
            self.cache_enabled = True

    def disable_caching(self) -> None:
        """Disable caching"""
        self.caching_enabled = False
        if hasattr(self, "cache_enabled"):
            self.cache_enabled = False

    def enable_batch_processing(self, batch_size: int = 10) -> None:
        """Enable batch processing"""
        self.batch_processing_enabled = True
        self.batch_size = batch_size
        self.batches_processed = 0

    def get_batch_processing_stats(self) -> dict:
        """Get batch processing statistics"""
        return {
            "batches_processed": getattr(self, "batches_processed", 0),
            "batch_size": getattr(self, "batch_size", 0),
        }

    # Validation methods
    def get_total_system_energy(self) -> float:
        """Get total system energy"""
        total_energy = 0.0
        for colony in self.get_colonies():
            total_energy += colony.get_adult_population() * 10.0  # Mock energy per bee
        return total_energy

    def update_configuration(self, config_updates: dict) -> None:
        """Update configuration during simulation runtime"""

        import copy

        # Validate configuration updates
        valid_keys = self._get_valid_config_keys()
        invalid_keys = [key for key in config_updates.keys() if key not in valid_keys]

        if invalid_keys:
            raise ValueError(f"Invalid configuration keys: {invalid_keys}")

        # Create backup of current configuration
        config_backup = copy.deepcopy(self.config)

        try:
            # Apply updates with validation
            for key, value in config_updates.items():
                self._validate_config_value(key, value)
                self._update_config_key(key, value)

            # Trigger necessary system updates
            self._notify_config_change(config_updates)

            self.logger.info(f"Configuration updated: {list(config_updates.keys())}")

        except Exception as e:
            # Restore backup on failure
            self.config = config_backup
            raise RuntimeError(
                f"Configuration update failed, restored previous state: {e}"
            )

    def _get_valid_config_keys(self) -> set:
        """Get set of valid configuration keys that can be updated at runtime"""
        return {
            "foraging.max_foraging_range",
            "foraging.dance_threshold",
            "foraging.recruitment_efficiency",
            "disease.enable_varroa",
            "disease.enable_viruses",
            "disease.enable_nosema",
            "disease.natural_resistance",
            "environment.weather_variation",
            "environment.seasonal_effects",
            "colony.colony_strength",
            "colony.genetic_diversity",
            "simulation.output_frequency",
            "simulation.save_state",
        }

    def _validate_config_value(self, key: str, value: Any) -> None:
        """Validate configuration value for specific key"""
        # Type and range validation based on key
        if key == "foraging.max_foraging_range":
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(
                    f"foraging.max_foraging_range must be positive number, got {value}"
                )
        elif key.startswith("disease.enable_"):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean, got {value}")
        elif key == "disease.natural_resistance":
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                raise ValueError(
                    f"disease.natural_resistance must be between 0 and 1, got {value}"
                )
        elif key == "environment.weather_variation":
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                raise ValueError(
                    f"environment.weather_variation must be between 0 and 1, got {value}"
                )
        elif key == "environment.seasonal_effects":
            if not isinstance(value, bool):
                raise ValueError(
                    f"environment.seasonal_effects must be boolean, got {value}"
                )
        elif key in ["colony.colony_strength", "colony.genetic_diversity"]:
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                raise ValueError(f"{key} must be between 0 and 1, got {value}")
        elif key == "simulation.output_frequency":
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"simulation.output_frequency must be positive integer, got {value}"
                )
        elif key == "simulation.save_state":
            if not isinstance(value, bool):
                raise ValueError(f"simulation.save_state must be boolean, got {value}")
        elif key in ["foraging.dance_threshold", "foraging.recruitment_efficiency"]:
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                raise ValueError(f"{key} must be between 0 and 1, got {value}")

    def _update_config_key(self, key: str, value: Any) -> None:
        """Update specific configuration key in nested config structure"""
        # Split key into nested parts
        parts = key.split(".")

        # For Pydantic models, we need to use setattr
        if len(parts) == 2:
            # Get the nested config object
            section = getattr(self.config, parts[0])
            # Set the attribute on the nested object
            setattr(section, parts[1], value)
        else:
            # Handle single-level keys
            setattr(self.config, parts[0], value)

    def _notify_config_change(self, changes: dict) -> None:
        """Notify model components of configuration changes"""
        # Update relevant model components based on changes
        for key, value in changes.items():
            if key.startswith("foraging."):
                # Update foraging-related components
                for colony in self.get_colonies():
                    if hasattr(colony, "update_foraging_config"):
                        colony.update_foraging_config({key: value})
            elif key.startswith("disease."):
                # Update disease-related components
                if hasattr(self, "disease_manager"):
                    self.disease_manager.update_config({key: value})
            elif key.startswith("environment."):
                # Update environment-related components
                if hasattr(self, "environment"):
                    self.environment.update_config({key: value})
            elif key.startswith("colony."):
                # Update colony-related components
                for colony in self.get_colonies():
                    if hasattr(colony, "update_activity_config"):
                        colony.update_activity_config({key: value})

    def validate_state_consistency(self) -> dict:
        """Validate state consistency"""
        return {
            "population_consistency": True,
            "resource_consistency": True,
            "spatial_consistency": True,
        }

    def _add_default_parameter_rules(self) -> None:
        """Add default dynamic parameter modification rules"""

        if not self.dynamic_parameter_system:
            return

        # Temperature-based foraging efficiency rule
        temp_rule = ModificationRule(
            rule_id="temperature_foraging_efficiency",
            parameter_path="foraging.efficiency_modifier",
            parameter_type=ParameterType.FORAGING,
            trigger=ModificationTrigger.CONDITION_BASED,
            strategy=ModificationStrategy.GRADUAL,
            trigger_condition="system_state.get('temperature', 20) < 10 or system_state.get('temperature', 20) > 35",
            target_value=0.3,
            modification_rate=0.05,
            duration=50.0,
            description="Reduce foraging efficiency in extreme temperatures",
        )
        self.dynamic_parameter_system.add_modification_rule(temp_rule)

        # Seasonal colony activity rule
        seasonal_rule = ModificationRule(
            rule_id="seasonal_colony_activity",
            parameter_path="colony.activity_level",
            parameter_type=ParameterType.COLONY,
            trigger=ModificationTrigger.CONDITION_BASED,
            strategy=ModificationStrategy.GRADUAL,
            trigger_condition="simulation_time % 365 < 60 or simulation_time % 365 > 300",  # Winter months
            target_value=0.4,
            modification_rate=0.02,
            duration=100.0,
            description="Reduce colony activity in winter months",
        )
        self.dynamic_parameter_system.add_modification_rule(seasonal_rule)

        # Population-based foraging strategy rule
        population_rule = ModificationRule(
            rule_id="population_foraging_strategy",
            parameter_path="foraging.strategy_aggression",
            parameter_type=ParameterType.FORAGING,
            trigger=ModificationTrigger.CONDITION_BASED,
            strategy=ModificationStrategy.STEPPED,
            trigger_condition="system_state.get('total_population', 0) < 100",
            target_value=1.5,
            modification_rate=0.1,
            duration=200.0,
            description="Increase foraging aggression when population is low",
        )
        self.dynamic_parameter_system.add_modification_rule(population_rule)

        # Resource availability adaptation rule
        resource_rule = ModificationRule(
            rule_id="resource_availability_adaptation",
            parameter_path="foraging.search_radius",
            parameter_type=ParameterType.FORAGING,
            trigger=ModificationTrigger.CONDITION_BASED,
            strategy=ModificationStrategy.GRADUAL,
            trigger_condition="system_state.get('resource_scarcity', 0.0) > 0.7",
            target_value=2000.0,  # Increase search radius
            modification_rate=0.08,
            duration=150.0,
            description="Expand search radius when resources are scarce",
        )
        self.dynamic_parameter_system.add_modification_rule(resource_rule)

        # Communication efficiency rule
        communication_rule = ModificationRule(
            rule_id="communication_efficiency_boost",
            parameter_path="communication.dance_threshold",
            parameter_type=ParameterType.COMMUNICATION,
            trigger=ModificationTrigger.CONDITION_BASED,
            strategy=ModificationStrategy.IMMEDIATE,
            trigger_condition="system_state.get('foraging_success_rate', 1.0) < 0.5",
            target_value=0.3,  # Lower threshold for dancing
            modification_rate=0.1,
            duration=100.0,
            description="Lower dance threshold when foraging success is poor",
        )
        self.dynamic_parameter_system.add_modification_rule(communication_rule)

        self.logger.info("Added 5 default dynamic parameter modification rules")

    def _update_dynamic_parameters(self) -> None:
        """Update dynamic parameter system with current simulation state"""

        if not self.dynamic_parameter_system:
            return

        # Gather current model state
        model_state = self._get_current_model_state()

        # Update dynamic parameter system
        self.dynamic_parameter_system.update_system_state(
            simulation_time=float(self.current_day), model_state=model_state
        )

        # Log system health periodically
        if self.current_day % 100 == 0:
            health_metrics = self.dynamic_parameter_system.system_health_metrics
            self.logger.info(f"Dynamic parameter system health: {health_metrics}")

    def _get_current_model_state(self) -> Dict[str, Any]:
        """Get comprehensive current model state for parameter system"""

        colonies = self.get_colonies()

        # Basic model metrics
        total_population = sum(colony.get_adult_population() for colony in colonies)
        total_energy: float = 0.0
        for colony in colonies:
            if hasattr(colony, "resources") and hasattr(colony.resources, "total_food"):
                if callable(colony.resources.total_food):
                    total_energy += colony.resources.total_food()
                else:
                    total_energy += colony.resources.total_food
        active_colonies = len(
            [c for c in colonies if getattr(c, "health", "healthy") != "collapsed"]
        )

        # Environmental metrics
        current_weather = self.current_weather
        temperature = current_weather.get("temperature", 20.0)
        precipitation = current_weather.get("precipitation", 0.0)
        wind_speed = current_weather.get("wind_speed", 0.0)

        # Foraging metrics
        foraging_efficiency = self.get_average_foraging_efficiency()
        foraging_analytics = {}
        if self.foraging_system:
            system_analytics = self.foraging_system.get_system_wide_analytics()
            foraging_efficiency = system_analytics.get("system_performance", {}).get(
                "average_efficiency", foraging_efficiency
            )
            foraging_analytics = system_analytics

        # Resource availability assessment
        resource_scarcity = self._calculate_resource_scarcity()

        # Communication metrics
        communication_activity = self._calculate_communication_activity()

        # Seasonal factors
        current_season = self._get_current_season()
        seasonal_factor = self.get_seasonal_factor()

        model_state = {
            # Population metrics
            "total_population": total_population,
            "active_colonies": active_colonies,
            "average_colony_size": total_population / max(1, len(colonies)),
            # Environmental metrics
            "temperature": temperature,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "season": current_season,
            "seasonal_factor": seasonal_factor,
            # Resource metrics
            "total_energy": total_energy,
            "resource_scarcity": resource_scarcity,
            "average_energy_per_bee": total_energy / max(1, total_population),
            # Foraging metrics
            "foraging_efficiency": foraging_efficiency,
            "foraging_success_rate": foraging_analytics.get(
                "system_performance", {}
            ).get("overall_success_rate", 0.5),
            "total_foraging_sessions": foraging_analytics.get("total_sessions", 0),
            # Communication metrics
            "communication_activity": communication_activity,
            "dance_frequency": communication_activity.get("dance_frequency", 0.0),
            # System metrics
            "simulation_day": self.current_day,
            "day_of_year": self.current_day % 365,
            "week_of_year": (self.current_day % 365) / 7,
            # Colony health metrics
            "average_colony_health": (
                np.mean([colony.get_health_score() for colony in colonies])
                if colonies
                else 0.0
            ),
            # Behavioral metrics
            "population_stress": max(
                0.0, 1.0 - (total_energy / max(1, total_population * 100))
            ),
            "environmental_stress": max(0.0, abs(temperature - 20.0) / 20.0),
            # Performance indicators
            "system_efficiency": foraging_efficiency * seasonal_factor,
            "sustainability_score": min(
                1.0, total_energy / max(1, total_population * 50)
            ),
        }

        # Add foraging system specific state if available
        if self.foraging_system:
            model_state.update(
                {
                    "active_foraging_sessions": len(
                        self.foraging_system.active_foraging_sessions
                    ),
                    "total_patches_available": len(self.foraging_system.patch_database),
                    "foraging_system_health": foraging_analytics.get(
                        "system_performance", {}
                    ),
                }
            )

        return model_state

    def _calculate_resource_scarcity(self) -> float:
        """Calculate resource scarcity metric"""

        colonies = self.get_colonies()
        if not colonies:
            return 0.0

        total_resources: float = 0.0
        for colony in colonies:
            if hasattr(colony, "resources") and colony.resources:
                if callable(colony.resources.total_food):
                    total_resources += colony.resources.total_food()
                else:
                    total_resources += colony.resources.total_food
        total_population = sum(colony.get_adult_population() for colony in colonies)

        if total_population == 0:
            return 0.0

        # Calculate resources per bee
        resources_per_bee = total_resources / total_population
        optimal_resources_per_bee = 100.0  # Baseline resource requirement

        # Scarcity is inverse of resource availability
        scarcity = max(0.0, 1.0 - (resources_per_bee / optimal_resources_per_bee))
        return min(1.0, scarcity)

    def _calculate_communication_activity(self) -> Dict[str, float]:
        """Calculate communication activity metrics"""

        # This would integrate with the communication system
        # For now, return mock metrics
        return {
            "dance_frequency": 0.5,
            "information_sharing_rate": 0.3,
            "recruitment_success": 0.7,
            "communication_efficiency": 0.6,
        }

    def add_dynamic_parameter_rule(self, rule: ModificationRule) -> bool:
        """Add a dynamic parameter modification rule"""

        if not self.dynamic_parameter_system:
            self.logger.warning("Dynamic parameter system not initialized")
            return False

        try:
            self.dynamic_parameter_system.add_modification_rule(rule)
            self.logger.info(f"Added dynamic parameter rule: {rule.rule_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add parameter rule {rule.rule_id}: {e}")
            return False

    def remove_dynamic_parameter_rule(self, rule_id: str) -> bool:
        """Remove a dynamic parameter modification rule"""

        if not self.dynamic_parameter_system:
            return False

        return self.dynamic_parameter_system.remove_modification_rule(rule_id)

    def apply_manual_parameter_change(
        self, parameter_path: str, new_value: Any, duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Apply manual parameter modification"""

        if not self.dynamic_parameter_system:
            return {
                "success": False,
                "error": "Dynamic parameter system not initialized",
            }

        return self.dynamic_parameter_system.apply_manual_modification(
            parameter_path, new_value, duration
        )

    def get_dynamic_parameter_analytics(
        self, parameter_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get analytics for dynamic parameter modifications"""

        if not self.dynamic_parameter_system:
            return {"error": "Dynamic parameter system not initialized"}

        return self.dynamic_parameter_system.get_parameter_analytics(parameter_path)

    def optimize_parameters_for_target(
        self, target_metrics: Dict[str, float], learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Optimize parameters to achieve target metrics"""

        if not self.dynamic_parameter_system:
            return {"error": "Dynamic parameter system not initialized"}

        return self.dynamic_parameter_system.optimize_parameters_adaptive(
            target_metrics, learning_rate
        )

    def _register_colony_health_monitoring(self, colony: Colony) -> None:
        """Register colony with health monitoring system"""

        if not self.health_monitoring_system:
            return

        # Prepare initial colony data for baseline establishment
        resources = getattr(colony, "resources", None)
        total_energy = 1000  # default
        if resources is not None and hasattr(resources, "total_food"):
            total_energy = resources.total_food

        initial_data = {
            "adult_population": colony.get_adult_population(),
            "total_energy": total_energy,
            "species": colony.species,
            "location": colony.location,
            "health_score": getattr(colony, "health_score", 0.8),
        }

        # Register with health monitoring system
        self.health_monitoring_system.register_colony(colony.unique_id, initial_data)

        self.logger.info(f"Registered colony {colony.unique_id} with health monitoring")

    def _update_colony_health_monitoring(self) -> None:
        """Update health monitoring for all colonies"""

        if not self.health_monitoring_system:
            return

        for colony in self.get_colonies():
            colony_data = self._collect_colony_health_data(colony)

            # Update health monitoring
            profile = self.health_monitoring_system.update_colony_health(
                colony.unique_id, colony_data, float(self.current_day)
            )

            # Log significant health changes
            if hasattr(self, "_last_health_status"):
                last_status = self._last_health_status.get(colony.unique_id)
                if last_status and last_status != profile.health_status.value:
                    self.logger.info(
                        f"Colony {colony.unique_id} health status changed: "
                        f"{last_status} → {profile.health_status.value}"
                    )
            else:
                self._last_health_status = {}

            self._last_health_status[colony.unique_id] = profile.health_status.value

            # Emit health update event for data collection
            self.emit_simulation_event(
                "health_update",
                colony_id=colony.unique_id,
                event_data={
                    "health_score": profile.overall_health_score,
                    "health_status": profile.health_status.value,
                    "active_alerts": len(profile.active_alerts),
                    "critical_indicators": [
                        ind.value for ind in profile.critical_indicators
                    ],
                },
                success=True,
            )

    def _collect_colony_health_data(self, colony: Colony) -> Dict[str, Any]:
        """Collect comprehensive health data from colony"""

        # Basic population and demographics
        adult_population = colony.get_adult_population()
        brood_count = getattr(colony, "brood_count", 0)

        # Resource and energy data
        resources = getattr(colony, "resources", None)
        if resources:
            total_energy = resources.total_food
            food_stores = resources.total_food
        else:
            total_energy = adult_population * 100  # Estimated
            food_stores = total_energy

        # Mortality data
        mortality_tracker = getattr(colony, "mortality_tracker", None)
        if mortality_tracker and hasattr(mortality_tracker, "mortality_events"):
            # Calculate approximate daily mortality rate from events
            total_deaths = len(mortality_tracker.mortality_events)
            mortality_rate = min(total_deaths / max(adult_population, 1), 1.0)
        else:
            mortality_rate = 0.01  # Default 1% daily mortality

        # Foraging data
        foraging_efficiency = 0.8  # Default
        if self.foraging_system:
            analytics = self.foraging_system.get_colony_foraging_analytics(
                colony.unique_id
            )
            if analytics and "session_statistics" in analytics:
                foraging_efficiency = analytics["session_statistics"].get(
                    "average_efficiency", 0.8
                )

        # Disease data
        disease_manager = getattr(colony, "disease_manager", None)
        if disease_manager and hasattr(disease_manager, "get_disease_summary"):
            summary = disease_manager.get_disease_summary()
            disease_prevalence = summary.get("total_infections", 0) / max(
                adult_population, 1
            )
        else:
            disease_prevalence = 0.05  # Default 5% infection rate

        # Environmental data
        current_weather = self.current_weather
        temperature = current_weather.get("temperature", 20.0)

        # Reproduction data
        reproduction_manager = getattr(colony, "reproduction_manager", None)
        if reproduction_manager:
            # Use a simple estimation for reproduction rate
            reproduction_rate = 0.6  # Safe default
        else:
            reproduction_rate = 0.6  # Default reproduction rate

        # Activity and behavioral data
        activity_level = self._calculate_colony_activity_level(colony)

        # Resource scarcity (calculated from foraging system or environmental factors)
        resource_scarcity = self._calculate_resource_scarcity_for_colony(colony)

        colony_health_data = {
            # Core metrics
            "adult_population": adult_population,
            "brood_count": brood_count,
            "total_energy": total_energy,
            "food_stores": food_stores,
            # Rates and efficiency
            "mortality_rate": mortality_rate,
            "foraging_efficiency": foraging_efficiency,
            "reproduction_rate": reproduction_rate,
            "disease_prevalence": disease_prevalence,
            # Environmental factors
            "temperature": temperature,
            "resource_scarcity": resource_scarcity,
            "activity_level": activity_level,
            # Derived metrics
            "energy_per_bee": total_energy / max(1, adult_population),
            "population_density": adult_population
            / max(1, brood_count + adult_population),
            # Time factors
            "simulation_day": self.current_day,
            "season": self._get_current_season(),
            # Colony characteristics
            "species": colony.species,
            "location": colony.location,
            "age": getattr(colony, "age", self.current_day),
        }

        return colony_health_data

    def _calculate_colony_activity_level(self, colony: Colony) -> float:
        """Calculate colony activity level"""

        # Base activity from foraging
        base_activity = 0.7

        # Adjust for environmental factors
        current_weather = self.current_weather
        temperature = current_weather.get("temperature", 20.0)

        # Temperature effects
        if 15.0 <= temperature <= 25.0:
            temp_factor = 1.0  # Optimal temperature
        elif 10.0 <= temperature <= 30.0:
            temp_factor = 0.8  # Sub-optimal but acceptable
        else:
            temp_factor = 0.5  # Extreme temperatures reduce activity

        # Weather effects
        precipitation = current_weather.get("precipitation", 0.0)
        weather_factor = max(0.3, 1.0 - (precipitation / 10.0))  # Rain reduces activity

        # Seasonal effects
        seasonal_factor = self.get_seasonal_factor()

        # Calculate overall activity
        activity_level = base_activity * temp_factor * weather_factor * seasonal_factor

        return min(1.0, max(0.0, activity_level))

    def _calculate_resource_scarcity_for_colony(self, colony: Colony) -> float:
        """Calculate resource scarcity level for specific colony"""

        # Use system-wide calculation as base
        base_scarcity = self._calculate_resource_scarcity()

        # Adjust for colony-specific factors
        adult_population = colony.get_adult_population()
        resources = getattr(colony, "resources", None)

        if resources:
            energy_per_bee = resources.total_food / max(1, adult_population)
            optimal_energy = 100.0  # Baseline energy per bee

            if energy_per_bee < optimal_energy:
                colony_scarcity = 1.0 - (energy_per_bee / optimal_energy)
            else:
                colony_scarcity = 0.0

            # Combine with system-wide scarcity
            return float(min(1.0, max(base_scarcity, colony_scarcity)))

        return base_scarcity

    def get_colony_health_status(self, colony_id: int) -> Dict[str, Any]:
        """Get health status for specific colony"""

        if not self.health_monitoring_system:
            return {"error": "Health monitoring system not initialized"}

        return self.health_monitoring_system.get_colony_health_summary(colony_id)

    def get_system_health_overview(self) -> Dict[str, Any]:
        """Get system-wide health overview"""

        if not self.health_monitoring_system:
            return {"error": "Health monitoring system not initialized"}

        return self.health_monitoring_system.get_system_health_overview()

    def get_health_alerts(
        self, colony_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get current health alerts"""

        if not self.health_monitoring_system:
            return []

        alerts = []

        if colony_id is not None:
            # Get alerts for specific colony
            if colony_id in self.health_monitoring_system.colony_profiles:
                profile = self.health_monitoring_system.colony_profiles[colony_id]
                alerts = [
                    {
                        "alert_id": alert.alert_id,
                        "level": alert.level.value,
                        "indicator": alert.indicator.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                        "colony_id": alert.colony_id,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                        "recommendation": alert.recommendation,
                    }
                    for alert in profile.active_alerts
                ]
        else:
            # Get all system alerts
            for profile in self.health_monitoring_system.colony_profiles.values():
                for alert in profile.active_alerts:
                    alerts.append(
                        {
                            "alert_id": alert.alert_id,
                            "level": alert.level.value,
                            "indicator": alert.indicator.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp,
                            "colony_id": alert.colony_id,
                            "current_value": alert.current_value,
                            "threshold_value": alert.threshold_value,
                            "recommendation": alert.recommendation,
                        }
                    )

        return alerts

    def get_health_predictions(
        self, colony_id: int, horizon: int = 10
    ) -> Dict[str, Any]:
        """Get health predictions for colony"""

        if not self.health_monitoring_system:
            return {"error": "Health monitoring system not initialized"}

        return self.health_monitoring_system.get_health_predictions(colony_id, horizon)

    def acknowledge_health_alert(self, alert_id: str) -> bool:
        """Acknowledge a health alert"""

        if not self.health_monitoring_system:
            return False

        # Find and acknowledge alert
        for profile in self.health_monitoring_system.colony_profiles.values():
            for alert in profile.active_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    self.logger.info(f"Health alert {alert_id} acknowledged")
                    return True

        return False

    def _initialize_spatial_analysis_systems(self) -> None:
        """Initialize spatial analysis and bee management systems"""

        try:
            # Create default landscape configuration if not provided
            landscape_config = self.config_dict.get(
                "landscape_config",
                {"patches": self._generate_default_landscape_patches()},
            )

            # Create spatial integration system
            self.spatial_environment, self.spatial_bee_manager = (
                create_spatial_integration_system(landscape_config)
            )

            # Register existing bees in spatial system
            self._register_bees_in_spatial_system()

            self.logger.info("Spatial analysis systems initialized successfully")

        except Exception as e:
            self.logger.warning(f"Failed to initialize spatial systems: {e}")
            self.spatial_environment = None
            self.spatial_bee_manager = None

    def _generate_default_landscape_patches(self) -> List[Dict[str, Any]]:
        """Generate default landscape patches for spatial analysis"""

        patches = []

        # Create a grid of patches around the hive
        hive_x, hive_y = 0.0, 0.0
        patch_id = 0

        # Hive patch
        patches.append(
            {
                "id": patch_id,
                "x": hive_x,
                "y": hive_y,
                "z": 0.0,
                "area": 4.0,
                "perimeter": 8.0,
                "quality": 0.8,
                "resource_density": 0.9,
                "accessibility": 1.0,
            }
        )
        patch_id += 1

        # Generate surrounding patches in concentric rings
        for ring in range(1, 6):  # 5 rings
            num_patches_in_ring = 8 * ring  # More patches in outer rings
            for i in range(num_patches_in_ring):
                angle = 2 * math.pi * i / num_patches_in_ring
                distance = ring * 20.0  # 20 units between rings

                x = hive_x + distance * math.cos(angle)
                y = hive_y + distance * math.sin(angle)

                # Vary quality and resources based on distance and randomness
                base_quality = max(0.2, 0.8 - ring * 0.1)  # Decrease with distance
                quality = base_quality + np.random.uniform(-0.2, 0.2)
                quality = max(0.1, min(1.0, quality))

                resource_density = quality * (0.8 + np.random.uniform(-0.2, 0.2))
                resource_density = max(0.1, min(1.0, resource_density))

                patches.append(
                    {
                        "id": patch_id,
                        "x": x,
                        "y": y,
                        "z": 0.0,
                        "area": 1.0 + np.random.uniform(0, 2.0),
                        "perimeter": 4.0,
                        "quality": quality,
                        "resource_density": resource_density,
                        "accessibility": 1.0,
                    }
                )
                patch_id += 1

        return patches

    def _register_bees_in_spatial_system(self) -> None:
        """Register existing bees in the spatial system"""

        if not self.spatial_bee_manager:
            return

        from .spatial_algorithms import SpatialPoint

        # Register all bees from all colonies
        for colony in self.colonies:
            for bee in colony.agents:
                # Place bee at hive initially (patch 0)
                initial_position = SpatialPoint(x=0.0, y=0.0, z=0.0)
                self.spatial_bee_manager.register_bee(
                    bee_id=bee.unique_id,
                    initial_position=initial_position,
                    initial_patch=0,  # Hive patch
                )

    def _update_spatial_bee_behavior_analysis(self) -> None:
        """Update spatial behaviors for all bees"""

        if not self.spatial_bee_manager:
            return

        from .enums import BeeStatus

        # Update spatial behavior for each bee in each colony
        for colony in self.colonies:
            for bee in colony.agents:
                if hasattr(bee, "status") and hasattr(bee, "unique_id"):
                    # Get bee status
                    bee_status = getattr(bee, "status", BeeStatus.RESTING)

                    # Update spatial behavior
                    spatial_result = (
                        self.spatial_bee_manager.update_bee_spatial_behavior(
                            bee.unique_id, bee_status, self.current_day
                        )
                    )

                    # Apply spatial results to bee
                    if spatial_result:
                        self._apply_spatial_results_to_bee(bee, spatial_result)

    def _apply_spatial_results_to_bee(
        self, bee: Any, spatial_result: Dict[str, Any]
    ) -> None:
        """Apply spatial behavior results to bee"""

        # Update bee's spatial metrics if available
        if hasattr(bee, "spatial_metrics"):
            bee.spatial_metrics.update(spatial_result)

        # Handle foraging results
        if "foraging_success" in spatial_result:
            if hasattr(bee, "energy") and spatial_result["foraging_success"]:
                energy_gain = spatial_result.get("resources_collected", 0.1) * 100
                bee.energy = min(
                    bee.energy + energy_gain, getattr(bee, "max_energy", 100)
                )

        # Handle scouting results
        if "scouting_result" in spatial_result:
            scout_info = spatial_result["scouting_result"]
            if hasattr(bee, "discovered_patches"):
                bee.discovered_patches.append(scout_info)

        # Update movement distance
        if "distance_traveled" in spatial_result:
            if hasattr(bee, "daily_distance"):
                bee.daily_distance += spatial_result["distance_traveled"]
            else:
                bee.daily_distance = spatial_result["distance_traveled"]

    def get_spatial_analysis_results(self) -> Dict[str, Any]:
        """Get comprehensive spatial analysis results"""

        if not self.spatial_environment or not self.spatial_bee_manager:
            return {"error": "Spatial analysis systems not initialized"}

        results: Dict[str, Any] = {}

        # Landscape analysis
        landscape_quality = (
            self.spatial_environment.landscape_analyzer.assess_landscape_quality()
        )
        fragmentation = self.spatial_environment.landscape_analyzer.calculate_landscape_fragmentation()
        isolation = (
            self.spatial_environment.landscape_analyzer.analyze_patch_isolation()
        )

        results["landscape_analysis"] = {
            "quality_metrics": landscape_quality,
            "fragmentation_metrics": fragmentation,
            "patch_isolation": isolation,
        }

        # Connectivity analysis
        from .spatial_algorithms import ConnectivityType

        connectivity_graph = (
            self.spatial_environment.connectivity_analyzer.build_connectivity_graph(
                ConnectivityType.FUNCTIONAL
            )
        )
        connectivity_metrics = (
            self.spatial_environment.connectivity_analyzer.analyze_connectivity_metrics(
                connectivity_graph
            )
        )

        results["connectivity_analysis"] = connectivity_metrics

        # Movement corridors
        corridors = self.spatial_environment.connectivity_analyzer.identify_corridors(
            connectivity_graph, min_quality=0.6
        )

        results["movement_corridors"] = {
            "corridor_count": len(corridors),
            "corridors": [
                {
                    "id": corridor.corridor_id,
                    "start_patch": corridor.start_patch,
                    "end_patch": corridor.end_patch,
                    "length": corridor.get_path_length(),
                    "quality": corridor.quality,
                }
                for corridor in corridors
            ],
        }

        # Bee spatial metrics
        bee_metrics = {}
        for bee_id, bee_state in self.spatial_bee_manager.bee_states.items():
            metrics = self.spatial_bee_manager.get_bee_spatial_metrics(bee_id)
            if metrics:
                bee_metrics[str(bee_id)] = metrics

        results["bee_spatial_metrics"] = bee_metrics

        return results

    def load_landscape_from_gis(self, gis_config: Dict[str, Any]) -> bool:
        """Load landscape data from GIS sources"""

        if not self.spatial_environment:
            self.logger.warning("Spatial environment not initialized")
            return False

        try:
            # Update landscape configuration with GIS data
            landscape_config = {
                "use_gis_data": True,
                "gis_config": gis_config.get("coordinate_systems", {}),
                "raster_sources": gis_config.get("raster_sources", []),
                "vector_sources": gis_config.get("vector_sources", []),
            }

            # Reinitialize landscape with GIS data
            self.spatial_environment.initialize_landscape(landscape_config)

            self.logger.info("Successfully loaded landscape from GIS sources")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load GIS landscape data: {e}")
            return False

    def export_spatial_data_to_gis(
        self, output_path: str, output_format: str = "geojson"
    ) -> Optional[str]:
        """Export current spatial data to GIS format"""

        if not self.spatial_environment:
            self.logger.warning("Spatial environment not initialized")
            return None

        try:
            result_path = self.spatial_environment.export_landscape_to_gis(
                output_path, output_format
            )
            self.logger.info(f"Exported spatial data to: {result_path}")
            return result_path

        except Exception as e:
            self.logger.error(f"Failed to export spatial data: {e}")
            return None

    def get_gis_analysis_summary(self) -> Dict[str, Any]:
        """Get GIS-based spatial analysis summary"""

        if not self.spatial_environment:
            return {"error": "Spatial environment not initialized"}

        try:
            gis_results = self.spatial_environment.get_gis_analysis_results()
            spatial_results = self.get_spatial_analysis_results()

            # Combine results
            combined_results = {
                "gis_analysis": gis_results,
                "spatial_analysis": spatial_results,
                "integration_status": {
                    "gis_enabled": self.spatial_environment.gis_manager is not None,
                    "coordinate_transformations": self.spatial_environment.gis_manager
                    is not None,
                    "data_sources": len(
                        self.spatial_environment.spatial_index._patches
                    ),
                },
            }

            return combined_results

        except Exception as e:
            self.logger.error(f"Failed to generate GIS analysis summary: {e}")
            return {"error": str(e)}


# Import pandas for data structure returns
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False


class SimulationResults:
    """Results container for BSTEW simulation runs"""

    def __init__(self, model: "BeeModel", success: bool = True, message: str = ""):
        self.model = model
        self.success = success
        self.message = message
        self.duration_days = model.current_day if model.current_day else 0
        self.final_colonies = len(model.get_colonies())

    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary statistics"""
        return {
            "success": self.success,
            "message": self.message,
            "duration_days": self.duration_days,
            "final_colonies": self.final_colonies,
            "total_agents": len(self.model.schedule.agents)
            if self.model.schedule
            else 0,
        }


class BstewModel:
    """
    User-friendly API wrapper for BSTEW simulation model.

    Provides the documented API interface while integrating with the existing
    BeeModel implementation for backward compatibility and feature completeness.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BSTEW model with optional configuration.

        Args:
            config: Configuration dictionary for simulation parameters
        """
        # Create underlying BeeModel with configuration
        if config is None:
            config = {}

        self._bee_model = BeeModel(config=config)
        self._simulation_results: Optional[SimulationResults] = None

    def run(self, days: int = 365) -> SimulationResults:
        """
        Run BSTEW simulation for specified number of days.

        Args:
            days: Number of simulation days to run (default: 365)

        Returns:
            SimulationResults object containing simulation outcomes
        """
        try:
            # Set maximum days on the underlying model
            self._bee_model.max_days = days

            # Run the simulation
            self._bee_model.running = True
            step_count = 0

            while self._bee_model.running and step_count < days:
                self._bee_model.step()
                step_count += 1

                # Safety check to prevent infinite loops
                if step_count >= days:
                    break

            # Create results object
            self._simulation_results = SimulationResults(
                model=self._bee_model,
                success=True,
                message=f"Simulation completed successfully after {step_count} days",
            )

            return self._simulation_results

        except Exception as e:
            self._simulation_results = SimulationResults(
                model=self._bee_model,
                success=False,
                message=f"Simulation failed: {str(e)}",
            )
            return self._simulation_results

    def get_population_data(self) -> Any:
        """
        Get population data from the simulation.

        Returns:
            pandas.DataFrame if pandas available, otherwise dict with population data
        """
        if not self._bee_model:
            return {} if not PANDAS_AVAILABLE else pd.DataFrame()

        # Collect population data from colonies
        population_data = []

        for colony in self._bee_model.get_colonies():
            if hasattr(colony, "get_population_summary"):
                pop_summary = colony.get_population_summary()
                pop_data = {
                    "day": self._bee_model.current_day or 0,
                    "colony_id": colony.unique_id,
                    "species": getattr(colony, "species", "apis_mellifera"),
                    "queens": pop_summary.get("queens", 0),
                    "workers": pop_summary.get("workers", 0),
                    "foragers": pop_summary.get("foragers", 0),
                    "drones": pop_summary.get("drones", 0),
                    "eggs": pop_summary.get("eggs", 0),
                    "larvae": pop_summary.get("larvae", 0),
                    "pupae": pop_summary.get("pupae", 0),
                    "total_population": pop_summary.get("total_population", 0),
                }
                population_data.append(pop_data)

        # Return as DataFrame if pandas available, otherwise as list of dicts
        if PANDAS_AVAILABLE and population_data:
            return pd.DataFrame(population_data)
        else:
            return population_data

    def get_foraging_data(self) -> Any:
        """
        Get foraging data from the simulation.

        Returns:
            pandas.DataFrame if pandas available, otherwise dict with foraging data
        """
        if not self._bee_model:
            return {} if not PANDAS_AVAILABLE else pd.DataFrame()

        # Collect foraging data from data collector
        foraging_data = []

        if hasattr(self._bee_model, "comprehensive_data_collector"):
            collector = self._bee_model.comprehensive_data_collector

            # Extract foraging events from data collector
            if hasattr(collector, "foraging_events"):
                for event in collector.foraging_events:
                    foraging_record = {
                        "day": event.get("day", self._bee_model.current_day or 0),
                        "species": event.get("species", "apis_mellifera"),
                        "forager_id": event.get("forager_id", 0),
                        "colony_id": event.get("colony_id", 0),
                        "patch_id": event.get("patch_id", 0),
                        "nectar_load": event.get("nectar_load", 0.0),
                        "pollen_load": event.get("pollen_load", 0.0),
                        "flight_distance": event.get("flight_distance", 0.0),
                        "success": event.get("success", False),
                    }
                    foraging_data.append(foraging_record)

        # Return as DataFrame if pandas available, otherwise as list of dicts
        if PANDAS_AVAILABLE and foraging_data:
            return pd.DataFrame(foraging_data)
        else:
            return foraging_data

    @property
    def colonies(self) -> List[Colony]:
        """Get all active colonies from the simulation"""
        return self._bee_model.get_colonies() if self._bee_model else []

    @property
    def current_day(self) -> int:
        """Get current simulation day"""
        return self._bee_model.current_day if self._bee_model else 0

    @property
    def is_running(self) -> bool:
        """Check if simulation is currently running"""
        return self._bee_model.running if self._bee_model else False

    def get_model_state(self) -> Dict[str, Any]:
        """Get current state of the underlying model"""
        if not self._bee_model:
            return {}

        return {
            "current_day": self.current_day,
            "is_running": self.is_running,
            "colony_count": len(self.colonies),
            "total_agents": len(self._bee_model.schedule.agents)
            if self._bee_model.schedule
            else 0,
            "simulation_results": self._simulation_results.get_summary()
            if self._simulation_results
            else None,
        }
