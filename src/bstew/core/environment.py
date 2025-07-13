"""
Environment system for BSTEW
============================

Main environment/model class integrating all components.
Mesa-based model with scheduler and data collection.
"""

import mesa
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .colony import Colony
from .scheduler import BeeScheduler
from ..spatial.landscape import LandscapeGrid
from ..spatial.resources import ResourceDistribution
from ..utils.config import BstewConfig


class Environment(mesa.Model):
    """
    Main BSTEW environment model.

    Integrates:
    - Agent-based bee populations
    - Mathematical colony dynamics
    - Spatial landscape system
    - Weather and seasonal effects
    - Data collection and output
    """

    def __init__(self, config: BstewConfig):
        super().__init__()

        self.config = config
        self.random_seed = config.simulation.random_seed

        # Initialize random number generator
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Time tracking
        self.step_count = 0
        self.current_day = 0
        self.start_date = datetime.strptime(config.simulation.start_date, "%Y-%m-%d")
        self.current_date = self.start_date

        # Initialize scheduler
        self.schedule = BeeScheduler(self)

        # Agent ID counter (must be before colony initialization)
        self.agent_id_counter = 0

        # Logging (must be before colony initialization)
        self.logger = logging.getLogger(__name__)

        # Initialize landscape
        self.landscape = LandscapeGrid(
            config.environment.landscape_width,
            config.environment.landscape_height,
            config.environment.cell_size,
        )

        # Load landscape from file if specified
        if config.environment.landscape_file:
            try:
                self.landscape.load_from_image(config.environment.landscape_file)
            except Exception as e:
                self.logger.warning(f"Could not load landscape file: {e}")

        # Initialize resource distribution
        self.resource_distribution = ResourceDistribution(self.landscape)

        # Initialize colonies
        self.colonies: List[Colony] = []
        self._initialize_colonies()

        # Weather system
        self.weather_data: Dict[int, Dict[str, float]] = {}
        self.current_weather = self._get_default_weather()

        if config.environment.weather_file:
            self._load_weather_data(config.environment.weather_file)

        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Total_Bees": lambda m: sum(
                    c.get_adult_population() for c in m.colonies
                ),
                "Total_Brood": lambda m: sum(
                    c.population_counts["brood"] for c in m.colonies
                ),
                "Total_Honey": lambda m: sum(c.resources.honey for c in m.colonies),
                "Active_Colonies": lambda m: len(
                    [c for c in m.colonies if c.health.value != "collapsed"]
                ),
                "Day": lambda m: m.current_day,
                "Temperature": lambda m: m.current_weather.get("temperature", 15.0),
                "Available_Nectar": lambda m: m.landscape.get_total_resources()[
                    "nectar"
                ],
            },
            agent_reporters={
                "Age": "age",
                "Energy": "energy",
                "Role": lambda a: a.role.value if hasattr(a, "role") else None,
                "Colony_ID": lambda a: (
                    a.colony.id if hasattr(a, "colony") and a.colony else None
                ),
            },
        )

        self.logger.info("Environment initialized successfully")

    def _initialize_colonies(self) -> None:
        """Initialize bee colonies"""
        colony = Colony(
            model=self,
            species=self.config.colony.species,
            location=(
                float(self.config.colony.location[0]),
                float(self.config.colony.location[1]),
            ),
            initial_population=self.config.colony.initial_population,
        )

        # Set colony ID
        colony.unique_id = 0

        self.colonies.append(colony)
        self.logger.info(
            f"Initialized colony with {colony.get_adult_population()} bees"
        )

    def _load_weather_data(self, weather_file: str) -> None:
        """Load weather data from file"""
        try:
            import pandas as pd

            df = pd.read_csv(weather_file)

            # Convert to dictionary indexed by day of year
            for _, row in df.iterrows():
                day = int(row.get("day", 0))
                self.weather_data[day] = {
                    "temperature": row.get("temperature", 15.0),
                    "rainfall": row.get("rainfall", 0.0),
                    "wind_speed": row.get("wind_speed", 5.0),
                    "humidity": row.get("humidity", 60.0),
                }

            self.logger.info(f"Loaded weather data for {len(self.weather_data)} days")

        except Exception as e:
            self.logger.warning(f"Could not load weather data: {e}")

    def _get_default_weather(self) -> Dict[str, float]:
        """Get default weather conditions"""
        return {
            "temperature": 15.0,  # °C
            "rainfall": 0.0,  # mm
            "wind_speed": 5.0,  # mph
            "humidity": 60.0,  # %
        }

    def get_current_weather(self) -> Dict[str, float]:
        """Get weather for current day"""
        day_of_year = self.current_date.timetuple().tm_yday

        if day_of_year in self.weather_data:
            return self.weather_data[day_of_year]
        else:
            # Generate seasonal weather if no data
            return self._generate_seasonal_weather(day_of_year)

    def _generate_seasonal_weather(self, day_of_year: int) -> Dict[str, float]:
        """Generate seasonal weather patterns"""
        # Simple seasonal temperature model
        base_temp = 10.0 + 15.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Add daily variation
        daily_variation = self.random.gauss(0, 3)
        temperature = base_temp + daily_variation

        # Seasonal rainfall
        if 90 <= day_of_year <= 270:  # Growing season
            rainfall = max(0, self.random.gauss(1.0, 2.0))
        else:
            rainfall = max(0, self.random.gauss(2.0, 3.0))

        # Wind and humidity
        wind_speed = max(0, self.random.gauss(8.0, 4.0))
        humidity = np.clip(self.random.gauss(65.0, 15.0), 20, 100)

        return {
            "temperature": temperature,
            "rainfall": rainfall,
            "wind_speed": wind_speed,
            "humidity": humidity,
        }

    def get_current_season(self) -> str:
        """Get current season based on date"""
        day_of_year = self.current_date.timetuple().tm_yday

        if 80 <= day_of_year < 172:  # Mar 21 - Jun 20
            return "spring"
        elif 172 <= day_of_year < 266:  # Jun 21 - Sep 22
            return "summer"
        elif 266 <= day_of_year < 355:  # Sep 23 - Dec 20
            return "autumn"
        else:  # Dec 21 - Mar 20
            return "winter"

    def get_seasonal_factor(self) -> float:
        """Get seasonal activity factor"""
        season = self.get_current_season()

        factors = {"spring": 1.2, "summer": 1.0, "autumn": 0.8, "winter": 0.3}

        return factors.get(season, 1.0)

    def is_mating_season(self) -> bool:
        """Check if it's mating season for drones"""
        day_of_year = self.current_date.timetuple().tm_yday
        return 120 <= day_of_year <= 210  # May-July

    def next_id(self) -> int:
        """Generate next unique agent ID"""
        self.agent_id_counter += 1
        return self.agent_id_counter

    def step(self) -> None:
        """Execute one simulation step"""
        self.step_count += 1
        self.current_day = self.step_count
        self.current_date = self.start_date + timedelta(days=self.current_day - 1)

        # Update weather
        self.current_weather = self.get_current_weather()

        # Update landscape resources
        day_of_year = self.current_date.timetuple().tm_yday
        self.resource_distribution.update_landscape_resources(
            day_of_year, self.current_weather
        )

        # Apply agricultural management
        self.resource_distribution.simulate_agricultural_management(day_of_year)

        # Step all colonies
        for colony in self.colonies[:]:  # Copy list to allow removal
            colony.step()

            # Remove collapsed colonies
            if colony.health.value == "collapsed":
                self.logger.info(f"Colony {colony.unique_id} has collapsed")
                self.colonies.remove(colony)

        # Step all agents (handled by scheduler)
        self.schedule.step()

        # Collect data
        self.datacollector.collect(self)

        # Log progress
        if self.step_count % 30 == 0:  # Monthly logging
            self.logger.info(
                f"Day {self.current_day}: {len(self.colonies)} active colonies, "
                f"{sum(c.get_adult_population() for c in self.colonies)} total bees"
            )

    def run_simulation(self, steps: Optional[int] = None) -> mesa.DataCollector:
        """Run complete simulation"""
        if steps is None:
            steps = self.config.simulation.duration_days

        self.logger.info(f"Starting simulation for {steps} days")

        for i in range(steps):
            self.step()

            # Check termination conditions
            if not self.colonies:
                self.logger.warning("All colonies have collapsed - ending simulation")
                break

        self.logger.info(f"Simulation completed after {self.step_count} days")

        return self.datacollector

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get simulation summary statistics"""
        if not self.colonies:
            return {"status": "All colonies collapsed", "final_day": self.current_day}

        total_bees = sum(c.get_adult_population() for c in self.colonies)
        total_brood = sum(c.population_counts["brood"] for c in self.colonies)
        total_honey = sum(c.resources.honey for c in self.colonies)

        return {
            "status": "Active colonies present",
            "final_day": self.current_day,
            "active_colonies": len(self.colonies),
            "total_adult_bees": total_bees,
            "total_brood": total_brood,
            "total_honey_mg": total_honey,
            "landscape_connectivity": self.landscape.calculate_landscape_connectivity(),
            "total_landscape_resources": self.landscape.get_total_resources(),
        }

    def export_data(self, output_dir: str) -> None:
        """Export simulation data to files"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export model data
        model_data = self.datacollector.get_model_vars_dataframe()
        model_data.to_csv(output_path / "model_data.csv")

        # Export agent data
        agent_data = self.datacollector.get_agent_vars_dataframe()
        if not agent_data.empty:
            agent_data.to_csv(output_path / "agent_data.csv")

        # Export landscape data
        landscape_data = self.landscape.export_to_dict()

        import json

        with open(output_path / "landscape_data.json", "w") as f:
            json.dump(landscape_data, f, indent=2)

        # Export colony summaries
        colony_summaries = []
        for colony in self.colonies:
            summary = {
                "id": colony.id,  # type: ignore[attr-defined]
                "species": colony.species,
                "location": colony.location,
                "health": colony.health.value,
                "population": colony.population_counts,
                "resources": {
                    "pollen": colony.resources.pollen,
                    "nectar": colony.resources.nectar,
                    "honey": colony.resources.honey,
                },
            }
            colony_summaries.append(summary)

        with open(output_path / "colony_summaries.json", "w") as f:
            json.dump(colony_summaries, f, indent=2)

        self.logger.info(f"Data exported to {output_path}")

    def save_state(self, filepath: str) -> None:
        """Save simulation state for restart"""
        import pickle

        state = {
            "step_count": self.step_count,
            "current_day": self.current_day,
            "current_date": self.current_date,
            "colonies": self.colonies,
            "landscape": self.landscape,
            "weather_data": self.weather_data,
            "config": self.config,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        self.logger.info(f"State saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load simulation state from file"""
        import pickle

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.step_count = state["step_count"]
        self.current_day = state["current_day"]
        self.current_date = state["current_date"]
        self.colonies = state["colonies"]
        self.landscape = state["landscape"]
        self.weather_data = state["weather_data"]

        self.logger.info(f"State loaded from {filepath}")

    def __repr__(self) -> str:
        return (
            f"Environment(day={self.current_day}, colonies={len(self.colonies)}, "
            f"total_bees={sum(c.get_adult_population() for c in self.colonies)})"
        )
