"""
BSTEW Simulation Engine
======================

High-level simulation engine that wraps the core BeeModel/BstewModel
to provide a consistent API for running simulations, managing parameters,
and collecting results.

This engine is designed to be used by:
- Interactive dashboards
- Validation and benchmarking systems
- Batch processing tools
- Performance testing frameworks
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..core.model import BeeModel, BstewModel
from ..utils.config import SimulationConfig as BaseSimulationConfig


@dataclass
class SimulationResults:
    """Container for simulation results and metadata."""

    simulation_id: str
    config: BaseSimulationConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Core results
    final_data: Dict[str, Any] = field(default_factory=dict)
    time_series: Dict[str, List[Any]] = field(default_factory=dict)

    # Performance metrics
    performance_stats: Dict[str, float] = field(default_factory=dict)

    # Status tracking
    status: str = "initialized"  # initialized, running, completed, failed, cancelled
    error_message: Optional[str] = None

    # Model-specific data
    model_data: Optional[Dict[str, Any]] = None

    def mark_completed(self, final_data: Dict[str, Any]) -> None:
        """Mark simulation as completed with final results."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = "completed"
        self.final_data = final_data

    def mark_failed(self, error: str) -> None:
        """Mark simulation as failed with error message."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = "failed"
        self.error_message = error


class SimulationEngine:
    """
    High-level simulation engine for BSTEW.

    Provides a consistent interface for running simulations while wrapping
    the existing BeeModel and BstewModel implementations.
    """

    def __init__(self, config: Optional[BaseSimulationConfig] = None):
        """Initialize the simulation engine."""
        self.config = config or BaseSimulationConfig()
        self.logger = logging.getLogger(__name__)

        # Current simulation state
        self.current_simulation: Optional[SimulationResults] = None
        self.model: Optional[Union[BeeModel, BstewModel]] = None

        # Simulation history
        self.simulation_history: List[SimulationResults] = []

        # Performance monitoring
        self.enable_monitoring = True
        self.monitoring_interval = 1.0  # seconds

    def create_simulation(
        self,
        config: Optional[BaseSimulationConfig] = None,
        simulation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SimulationResults:
        """
        Create a new simulation with the given configuration.

        Args:
            config: Simulation configuration. Uses engine default if None.
            simulation_id: Unique ID for this simulation. Auto-generated if None.
            **kwargs: Additional configuration parameters to override

        Returns:
            SimulationResults object for tracking the simulation
        """
        # Use provided config or engine default
        sim_config = config or self.config

        # Override config with any provided kwargs
        if kwargs:
            config_dict = sim_config.model_dump()
            config_dict.update(kwargs)
            sim_config = BaseSimulationConfig(**config_dict)

        # Generate simulation ID if not provided
        if simulation_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            simulation_id = f"bstew_sim_{timestamp}"

        # Create simulation results container
        simulation = SimulationResults(
            simulation_id=simulation_id, config=sim_config, start_time=datetime.now()
        )

        return simulation

    def run_simulation(
        self,
        simulation: Optional[SimulationResults] = None,
        config: Optional[BaseSimulationConfig] = None,
        **kwargs: Any,
    ) -> SimulationResults:
        """
        Run a complete simulation from start to finish.

        Args:
            simulation: Pre-created simulation results object
            config: Configuration for new simulation if simulation is None
            **kwargs: Additional configuration overrides

        Returns:
            SimulationResults with completed simulation data
        """
        # Create simulation if not provided
        if simulation is None:
            simulation = self.create_simulation(config, **kwargs)

        self.current_simulation = simulation

        try:
            # Mark as running
            simulation.status = "running"
            self.logger.info(f"Starting simulation: {simulation.simulation_id}")

            # Initialize the model
            self.model = self._create_model(simulation.config)

            # Run the simulation
            results = self._run_model(self.model, simulation.config)

            # Collect final results
            final_data = self._collect_results(self.model)

            # Mark as completed
            simulation.mark_completed(final_data)
            simulation.model_data = results

            # Add to history
            self.simulation_history.append(simulation)

            self.logger.info(
                f"Simulation {simulation.simulation_id} completed in "
                f"{simulation.duration_seconds:.2f}s"
            )

        except Exception as e:
            error_msg = f"Simulation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            simulation.mark_failed(error_msg)

        finally:
            self.current_simulation = None

        return simulation

    def run_step(self) -> Optional[Dict[str, Any]]:
        """
        Run a single simulation step.

        Used for interactive simulations or step-by-step execution.

        Returns:
            Current step data if simulation is running, None otherwise
        """
        if not self.current_simulation or not self.model:
            return None

        try:
            # This would advance the model by one step
            # For now, return mock data
            if hasattr(self.model, "step"):
                self.model.step()
                return self._get_current_state()
            else:
                return None

        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            if self.current_simulation:
                self.current_simulation.mark_failed(str(e))
            return None

    def stop_simulation(self) -> None:
        """Stop the currently running simulation."""
        if self.current_simulation:
            self.current_simulation.status = "cancelled"
            self.current_simulation.end_time = datetime.now()
            if self.current_simulation.start_time:
                self.current_simulation.duration_seconds = (
                    self.current_simulation.end_time
                    - self.current_simulation.start_time
                ).total_seconds()
            self.logger.info(
                f"Simulation {self.current_simulation.simulation_id} cancelled"
            )

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current simulation state for monitoring."""
        return self._get_current_state()

    def get_simulation_history(self) -> List[SimulationResults]:
        """Get list of all completed simulations."""
        return self.simulation_history.copy()

    def _create_model(
        self, config: BaseSimulationConfig
    ) -> Union[BeeModel, BstewModel]:
        """Create and initialize the simulation model."""
        # For now, create a BstewModel (the user-friendly wrapper)
        # This would be configured based on the config parameters

        # Create BstewModel with proper configuration
        model = BstewModel(config.model_dump())
        return model

    def _create_mock_model(self, config: BaseSimulationConfig) -> Any:
        """Create a mock model for testing purposes."""

        class MockModel:
            def __init__(self, config: BaseSimulationConfig) -> None:
                self.config = config
                self.step_count = 0
                self.data: Dict[str, Any] = {}

            def step(self) -> None:
                self.step_count += 1

            def run_model(self) -> None:
                for _ in range(self.config.duration_days):
                    self.step()

        return MockModel(config)

    def _run_model(self, model: Any, config: BaseSimulationConfig) -> Dict[str, Any]:
        """Run the model for the specified duration."""
        start_time = time.time()

        # Check if model has run_model method
        if hasattr(model, "run_model"):
            model.run_model()
        else:
            # Run step-by-step
            for step in range(config.duration_days):
                if hasattr(model, "step"):
                    model.step()

                # Optional: collect intermediate data
                if step % config.output_frequency == 0:
                    self._collect_step_data(model, step)

        end_time = time.time()

        return {
            "execution_time": end_time - start_time,
            "steps_completed": getattr(model, "step_count", config.duration_days),
            "model_type": type(model).__name__,
        }

    def _collect_results(self, model: Any) -> Dict[str, Any]:
        """Collect final results from the completed model."""
        results = {
            "simulation_completed": True,
            "model_type": type(model).__name__,
            "final_step": getattr(model, "step_count", 0),
        }

        # Collect model-specific data
        if hasattr(model, "datacollector"):
            try:
                results["model_data"] = model.datacollector.get_model_vars_dataframe()
                results["agent_data"] = model.datacollector.get_agent_vars_dataframe()
            except Exception as e:
                self.logger.warning(f"Could not collect datacollector data: {e}")

        # Add any other available data
        if hasattr(model, "data"):
            results["additional_data"] = model.data

        return results

    def _collect_step_data(self, model: Any, step: int) -> None:
        """Collect data for a single simulation step."""
        if not self.current_simulation:
            return

        # This would collect step-specific data for time series
        step_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }

        # Add model-specific step data
        if hasattr(model, "get_current_data"):
            step_data.update(model.get_current_data())

        # Store in time series
        if "steps" not in self.current_simulation.time_series:
            self.current_simulation.time_series["steps"] = []

        self.current_simulation.time_series["steps"].append(step_data)

    def _get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current model state for monitoring."""
        if not self.model:
            return None

        state = {
            "current_step": getattr(self.model, "step_count", 0),
            "model_type": type(self.model).__name__,
            "timestamp": datetime.now().isoformat(),
        }

        # Add model-specific state info
        if hasattr(self.model, "get_current_data"):
            state.update(self.model.get_current_data())

        return state


# Re-export SimulationConfig from utils.config for convenience
SimulationConfig = BaseSimulationConfig
