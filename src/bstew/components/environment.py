"""
Advanced environmental effects system for BSTEW
===============================================

Implements weather impacts, climate change scenarios, seasonal dynamics,
and environmental stressors affecting bee colonies.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from collections import deque
import logging

from ..spatial.patches import ResourcePatch


class WeatherType(Enum):
    """Weather condition types"""

    CLEAR = "clear"
    CLOUDY = "cloudy"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    SNOW = "snow"
    FOG = "fog"
    WINDY = "windy"


class SeasonType(Enum):
    """Season types"""

    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class WeatherConditions(BaseModel):
    """Current weather conditions"""

    model_config = {"validate_assignment": True}

    temperature: float = Field(description="Temperature in Celsius")
    rainfall: float = Field(ge=0.0, description="Rainfall in mm/day")
    wind_speed: float = Field(ge=0.0, description="Wind speed in mph")
    humidity: float = Field(ge=0.0, le=100.0, description="Humidity percentage")
    pressure: float = Field(
        ge=800.0, le=1200.0, description="Atmospheric pressure in hPa"
    )
    daylight_hours: float = Field(ge=0.0, le=24.0, description="Hours of daylight")
    weather_type: WeatherType = Field(description="Current weather type")
    season: SeasonType = Field(description="Current season")
    day_of_year: int = Field(ge=0, le=365, description="Day of the year")

    def get_foraging_suitability(self) -> float:
        """Calculate foraging suitability (0-1 scale)"""

        # Temperature suitability (optimal 15-25°C)
        if 15 <= self.temperature <= 25:
            temp_factor = 1.0
        elif 10 <= self.temperature < 15 or 25 < self.temperature <= 30:
            temp_factor = 0.7
        elif 5 <= self.temperature < 10 or 30 < self.temperature <= 35:
            temp_factor = 0.3
        else:
            temp_factor = 0.0

        # Rain factor
        if self.rainfall == 0:
            rain_factor = 1.0
        elif self.rainfall <= 2:
            rain_factor = 0.8
        elif self.rainfall <= 5:
            rain_factor = 0.4
        else:
            rain_factor = 0.1

        # Wind factor
        if self.wind_speed <= 10:
            wind_factor = 1.0
        elif self.wind_speed <= 20:
            wind_factor = 0.6
        else:
            wind_factor = 0.2

        # Daylight factor
        daylight_factor = min(1.0, self.daylight_hours / 12.0)

        return temp_factor * rain_factor * wind_factor * daylight_factor

    def get_colony_stress_factor(self) -> float:
        """Calculate environmental stress factor for colonies"""

        stress_factors = []

        # Temperature stress
        if self.temperature < 5 or self.temperature > 35:
            stress_factors.append(0.8)
        elif self.temperature < 10 or self.temperature > 30:
            stress_factors.append(0.4)
        else:
            stress_factors.append(0.0)

        # Weather stress
        weather_stress = {
            WeatherType.CLEAR: 0.0,
            WeatherType.CLOUDY: 0.1,
            WeatherType.LIGHT_RAIN: 0.3,
            WeatherType.HEAVY_RAIN: 0.6,
            WeatherType.THUNDERSTORM: 0.8,
            WeatherType.SNOW: 0.9,
            WeatherType.FOG: 0.2,
            WeatherType.WINDY: 0.4,
        }
        stress_factors.append(weather_stress.get(self.weather_type, 0.0))

        # Seasonal stress
        seasonal_stress = {
            SeasonType.SPRING: 0.0,
            SeasonType.SUMMER: 0.1,
            SeasonType.AUTUMN: 0.2,
            SeasonType.WINTER: 0.6,
        }
        stress_factors.append(seasonal_stress.get(self.season, 0.0))

        return min(1.0, sum(stress_factors) / len(stress_factors))


class ClimateScenario(BaseModel):
    """Climate change scenario parameters"""

    model_config = {"validate_assignment": True}

    name: str = Field(description="Scenario name")
    description: str = Field(description="Scenario description")
    temperature_trend: float = Field(description="Temperature change in °C per year")
    precipitation_trend: float = Field(description="Precipitation change % per year")
    extreme_frequency_multiplier: float = Field(
        ge=0.0, description="Multiplier for extreme events"
    )
    start_year: int = Field(default=0, ge=0, description="Year scenario starts")

    def apply_to_weather(
        self, weather: WeatherConditions, current_year: int
    ) -> WeatherConditions:
        """Apply climate scenario to weather conditions"""
        if current_year < self.start_year:
            return weather

        years_elapsed = current_year - self.start_year

        # Apply temperature trend
        weather.temperature += self.temperature_trend * years_elapsed

        # Apply precipitation trend
        precip_multiplier = 1.0 + (self.precipitation_trend * years_elapsed / 100.0)
        weather.rainfall *= precip_multiplier

        # Increase extreme events
        if weather.weather_type in [WeatherType.THUNDERSTORM, WeatherType.HEAVY_RAIN]:
            # Make extreme events more frequent/intense
            if random.random() < 0.1 * self.extreme_frequency_multiplier:
                weather.rainfall *= 1.5
                weather.wind_speed *= 1.3

        return weather


class WeatherGenerator:
    """
    Procedural weather generation system.

    Generates realistic weather patterns with:
    - Seasonal variations
    - Multi-day weather systems
    - Extreme events
    - Climate change scenarios
    """

    def __init__(self, location_latitude: float = 40.0):
        self.latitude = location_latitude
        self.weather_history: deque = deque(maxlen=30)  # 30-day history
        self.current_weather_system: Optional[str] = None
        self.system_duration = 0

        # Weather generation parameters
        self.base_temperatures = {
            SeasonType.SPRING: (10, 20),  # (min, max) °C
            SeasonType.SUMMER: (18, 28),
            SeasonType.AUTUMN: (8, 18),
            SeasonType.WINTER: (0, 8),
        }

        self.rainfall_patterns = {
            SeasonType.SPRING: 3.0,  # mm/day average
            SeasonType.SUMMER: 2.0,
            SeasonType.AUTUMN: 4.0,
            SeasonType.WINTER: 2.5,
        }

    def generate_weather(
        self, day_of_year: int, climate_scenario: Optional[ClimateScenario] = None
    ) -> WeatherConditions:
        """Generate weather for a specific day"""

        season = self._get_season(day_of_year)

        # Generate base weather
        weather = self._generate_base_weather(day_of_year, season)

        # Apply weather systems
        weather = self._apply_weather_systems(weather)

        # Apply climate scenario if provided
        if climate_scenario:
            current_year = day_of_year // 365
            weather = climate_scenario.apply_to_weather(weather, current_year)

        # Store in history
        self.weather_history.append(weather)

        return weather

    def _get_season(self, day_of_year: int) -> SeasonType:
        """Determine season from day of year"""
        day_of_year = day_of_year % 365

        if 80 <= day_of_year < 172:  # ~March 21 - June 20
            return SeasonType.SPRING
        elif 172 <= day_of_year < 266:  # ~June 21 - September 22
            return SeasonType.SUMMER
        elif 266 <= day_of_year < 355:  # ~September 23 - December 20
            return SeasonType.AUTUMN
        else:  # ~December 21 - March 20
            return SeasonType.WINTER

    def _generate_base_weather(
        self, day_of_year: int, season: SeasonType
    ) -> WeatherConditions:
        """Generate base weather conditions"""

        # Temperature with seasonal variation
        temp_min, temp_max = self.base_temperatures[season]

        # Add daily temperature variation (sinusoidal)
        seasonal_factor = math.sin(2 * math.pi * (day_of_year % 365) / 365)
        daily_temp_range = temp_max - temp_min
        base_temp = temp_min + (daily_temp_range * (0.5 + 0.3 * seasonal_factor))

        # Add random variation
        temperature = base_temp + random.gauss(0, 3)

        # Rainfall
        base_rainfall = self.rainfall_patterns[season]
        rainfall = max(
            0, random.expovariate(1 / base_rainfall) if random.random() < 0.3 else 0
        )

        # Wind speed
        wind_speed = max(0, random.gauss(8, 4))

        # Humidity (higher in winter, varies with rainfall)
        base_humidity = 65 if season == SeasonType.WINTER else 55
        humidity = min(
            100, max(20, base_humidity + random.gauss(0, 10) + (rainfall * 5))
        )

        # Pressure
        pressure = random.gauss(1013, 15)

        # Daylight hours
        daylight_hours = self._calculate_daylight_hours(day_of_year)

        # Weather type
        weather_type = self._determine_weather_type(temperature, rainfall, wind_speed)

        return WeatherConditions(
            temperature=temperature,
            rainfall=rainfall,
            wind_speed=wind_speed,
            humidity=humidity,
            pressure=pressure,
            daylight_hours=daylight_hours,
            weather_type=weather_type,
            season=season,
            day_of_year=day_of_year % 365,
        )

    def _calculate_daylight_hours(self, day_of_year: int) -> float:
        """Calculate daylight hours based on latitude and day of year"""

        # Solar declination
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

        # Hour angle
        lat_rad = math.radians(self.latitude)
        decl_rad = math.radians(declination)

        hour_angle = math.acos(-math.tan(lat_rad) * math.tan(decl_rad))

        # Daylight hours
        daylight_hours = 2 * hour_angle * 12 / math.pi

        return max(0, min(24, daylight_hours))

    def _determine_weather_type(
        self, temperature: float, rainfall: float, wind_speed: float
    ) -> WeatherType:
        """Determine weather type from conditions"""

        if temperature < 0 and rainfall > 0:
            return WeatherType.SNOW
        elif rainfall > 10:
            return WeatherType.HEAVY_RAIN
        elif rainfall > 2:
            return WeatherType.LIGHT_RAIN
        elif wind_speed > 25:
            return WeatherType.WINDY
        elif rainfall > 0 and wind_speed > 15:
            return WeatherType.THUNDERSTORM
        elif rainfall == 0 and random.random() < 0.3:
            return WeatherType.CLOUDY
        else:
            return WeatherType.CLEAR

    def _apply_weather_systems(self, weather: WeatherConditions) -> WeatherConditions:
        """Apply multi-day weather system persistence"""

        # Check if current system should continue
        if self.current_weather_system and self.system_duration > 0:
            # Modify weather to be similar to system

            if self.current_weather_system == "high_pressure":
                weather.temperature += 2
                weather.rainfall *= 0.3
                weather.pressure += 10
                weather.weather_type = WeatherType.CLEAR

            elif self.current_weather_system == "low_pressure":
                weather.temperature -= 1
                weather.rainfall *= 2.0
                weather.pressure -= 10
                if weather.rainfall > 5:
                    weather.weather_type = WeatherType.HEAVY_RAIN

            elif self.current_weather_system == "frontal":
                weather.wind_speed *= 1.5
                weather.rainfall *= 1.5
                weather.weather_type = WeatherType.THUNDERSTORM

            self.system_duration -= 1

        else:
            # Start new weather system
            if random.random() < 0.2:  # 20% chance of new system
                systems = ["high_pressure", "low_pressure", "frontal"]
                self.current_weather_system = random.choice(systems)
                self.system_duration = random.randint(2, 7)  # 2-7 day systems

        return weather


class EnvironmentalStressor:
    """
    Individual environmental stressor that affects colonies.

    Examples: drought, flood, pesticide application, habitat loss
    """

    def __init__(
        self,
        name: str,
        stressor_type: str,
        severity: float,
        duration: int,
        affected_area: float,
    ):
        self.name = name
        self.stressor_type = stressor_type
        self.severity = severity  # 0-1 scale
        self.duration = duration  # days
        self.affected_area = affected_area  # km²
        self.start_day = 0
        self.active = False

    def activate(self, start_day: int) -> None:
        """Activate stressor on given day"""
        self.start_day = start_day
        self.active = True

    def is_active(self, current_day: int) -> bool:
        """Check if stressor is currently active"""
        if not self.active:
            return False
        return current_day < self.start_day + self.duration

    def get_impact_factor(self, current_day: int, distance_from_center: float) -> float:
        """Calculate impact factor for given day and distance"""
        if not self.is_active(current_day):
            return 0.0

        # Distance decay
        if distance_from_center > self.affected_area:
            return 0.0
        distance_factor = 1.0 - (distance_from_center / self.affected_area)

        # Temporal variation
        days_active = current_day - self.start_day
        temporal_factor = self._get_temporal_pattern(days_active)

        return self.severity * distance_factor * temporal_factor

    def _get_temporal_pattern(self, days_active: int) -> float:
        """Get temporal pattern of stressor intensity"""

        if self.stressor_type == "drought":
            # Gradually increases
            return min(1.0, days_active / (self.duration * 0.7))
        elif self.stressor_type == "flood":
            # Peak at beginning, then declines
            return max(0.1, 1.0 - (days_active / self.duration))
        elif self.stressor_type == "pesticide":
            # Sharp peak then rapid decline
            if days_active <= 3:
                return 1.0
            else:
                return max(0.0, 1.0 - ((days_active - 3) / (self.duration - 3)))
        else:
            # Default constant pattern
            return 1.0


class PestControlEvent:
    """
    Pesticide application event affecting bee health.
    """

    def __init__(
        self,
        pesticide_type: str,
        application_area: float,
        toxicity: float,
        persistence: int,
    ):
        self.pesticide_type = pesticide_type
        self.application_area = application_area  # km²
        self.toxicity = toxicity  # 0-1 scale
        self.persistence = persistence  # days
        self.application_day = 0

    def get_exposure_risk(
        self, current_day: int, distance_from_application: float
    ) -> float:
        """Calculate exposure risk for bees"""

        days_since_application = current_day - self.application_day

        if days_since_application < 0 or days_since_application > self.persistence:
            return 0.0

        # Distance factor
        if distance_from_application > self.application_area:
            return 0.0
        distance_factor = 1.0 - (distance_from_application / self.application_area)

        # Persistence factor (exponential decay)
        persistence_factor = math.exp(-days_since_application / (self.persistence / 3))

        return self.toxicity * distance_factor * persistence_factor


class EnvironmentalEffectsManager:
    """
    Comprehensive environmental effects management system.

    Integrates weather, climate, stressors, and seasonal effects.
    """

    def __init__(self, latitude: float = 40.0):
        self.weather_generator = WeatherGenerator(latitude)
        self.climate_scenario: Optional[ClimateScenario] = None
        self.active_stressors: List[EnvironmentalStressor] = []
        self.pest_control_events: List[PestControlEvent] = []
        self.current_weather: Optional[WeatherConditions] = None

        self.logger = logging.getLogger(__name__)

    def set_climate_scenario(self, scenario: ClimateScenario) -> None:
        """Set climate change scenario"""
        self.climate_scenario = scenario
        self.logger.info(f"Applied climate scenario: {scenario.name}")

    def add_environmental_stressor(
        self, stressor: EnvironmentalStressor, start_day: int
    ) -> None:
        """Add environmental stressor"""
        stressor.activate(start_day)
        self.active_stressors.append(stressor)
        self.logger.info(f"Added stressor: {stressor.name} starting day {start_day}")

    def add_pest_control_event(
        self, event: PestControlEvent, application_day: int
    ) -> None:
        """Add pesticide application event"""
        event.application_day = application_day
        self.pest_control_events.append(event)
        self.logger.info(
            f"Added pesticide application: {event.pesticide_type} on day {application_day}"
        )

    def update_environmental_conditions(self, current_day: int) -> None:
        """Update all environmental conditions for current day"""

        # Generate weather
        self.current_weather = self.weather_generator.generate_weather(
            current_day, self.climate_scenario
        )

        # Clean up expired stressors and events
        self.active_stressors = [
            s for s in self.active_stressors if s.is_active(current_day)
        ]
        self.pest_control_events = [
            e
            for e in self.pest_control_events
            if current_day - e.application_day <= e.persistence
        ]

    def get_resource_production_modifier(
        self, patch: ResourcePatch, current_day: int
    ) -> float:
        """Calculate resource production modifier for patch"""

        modifier = 1.0

        # Weather effects
        if self.current_weather:
            # Temperature effects on flowering
            temp = self.current_weather.temperature
            if 15 <= temp <= 25:
                temp_modifier = 1.0
            elif 10 <= temp < 15 or 25 < temp <= 30:
                temp_modifier = 0.8
            elif 5 <= temp < 10 or 30 < temp <= 35:
                temp_modifier = 0.5
            else:
                temp_modifier = 0.2

            # Rainfall effects
            if self.current_weather.rainfall > 20:  # Extreme rainfall
                rain_modifier = 0.6
            elif self.current_weather.rainfall > 10:
                rain_modifier = 0.8
            else:
                rain_modifier = 1.0

            modifier *= temp_modifier * rain_modifier

        # Environmental stressor effects
        for stressor in self.active_stressors:
            distance = self._calculate_distance(
                patch.location, (0, 0)
            )  # Assuming stressor at origin
            impact = stressor.get_impact_factor(current_day, distance)

            if stressor.stressor_type == "drought":
                modifier *= 1.0 - impact * 0.8
            elif stressor.stressor_type == "flood":
                modifier *= 1.0 - impact * 0.6
            elif stressor.stressor_type == "habitat_loss":
                modifier *= 1.0 - impact * 0.9

        return max(0.0, modifier)

    def get_bee_mortality_modifier(
        self, bee_location: Tuple[float, float], current_day: int
    ) -> float:
        """Calculate additional mortality risk for bees"""

        mortality_risk = 0.0

        # Weather-related mortality
        if self.current_weather:
            if (
                self.current_weather.temperature < -5
                or self.current_weather.temperature > 40
            ):
                mortality_risk += 0.1
            elif (
                self.current_weather.temperature < 0
                or self.current_weather.temperature > 35
            ):
                mortality_risk += 0.05

            if self.current_weather.weather_type == WeatherType.THUNDERSTORM:
                mortality_risk += 0.02
            elif self.current_weather.weather_type == WeatherType.HEAVY_RAIN:
                mortality_risk += 0.01

        # Pesticide exposure
        for event in self.pest_control_events:
            distance = self._calculate_distance(
                bee_location, (0, 0)
            )  # Assuming application at origin
            exposure_risk = event.get_exposure_risk(current_day, distance)
            mortality_risk += exposure_risk * 0.3  # 30% mortality at full exposure

        # Environmental stressor effects
        for stressor in self.active_stressors:
            distance = self._calculate_distance(bee_location, (0, 0))
            impact = stressor.get_impact_factor(current_day, distance)

            if stressor.stressor_type == "pesticide":
                mortality_risk += impact * 0.4

        return min(1.0, mortality_risk)

    def get_foraging_efficiency_modifier(self, current_day: int) -> float:
        """Calculate foraging efficiency modifier"""

        if not self.current_weather:
            return 1.0

        base_efficiency = self.current_weather.get_foraging_suitability()

        # Environmental stressor effects
        stressor_penalty = 0.0
        for stressor in self.active_stressors:
            if stressor.stressor_type in ["habitat_loss", "pesticide"]:
                # Assuming colony at origin for simplicity
                impact = stressor.get_impact_factor(current_day, 0)
                stressor_penalty += impact * 0.3

        return max(0.1, base_efficiency - stressor_penalty)

    def get_colony_stress_level(
        self, colony_location: Tuple[float, float], current_day: int
    ) -> float:
        """Calculate overall colony stress level"""

        stress_level = 0.0

        # Weather stress
        if self.current_weather:
            stress_level += self.current_weather.get_colony_stress_factor()

        # Environmental stressor effects
        for stressor in self.active_stressors:
            distance = self._calculate_distance(colony_location, (0, 0))
            impact = stressor.get_impact_factor(current_day, distance)
            stress_level += impact * 0.5

        return min(1.0, stress_level)

    def _calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_environmental_summary(self, current_day: int) -> Dict[str, Any]:
        """Get comprehensive environmental status summary"""

        summary: Dict[str, Any] = {
            "current_day": current_day,
            "weather": None,
            "active_stressors": len(self.active_stressors),
            "pest_control_events": len(self.pest_control_events),
            "climate_scenario": None,
        }

        if self.current_weather:
            summary["weather"] = {
                "temperature": self.current_weather.temperature,
                "rainfall": self.current_weather.rainfall,
                "wind_speed": self.current_weather.wind_speed,
                "weather_type": self.current_weather.weather_type.value,
                "season": self.current_weather.season.value,
                "foraging_suitability": self.current_weather.get_foraging_suitability(),
                "colony_stress_factor": self.current_weather.get_colony_stress_factor(),
            }

        if self.climate_scenario:
            summary["climate_scenario"] = {
                "name": self.climate_scenario.name,
                "temperature_trend": self.climate_scenario.temperature_trend,
                "precipitation_trend": self.climate_scenario.precipitation_trend,
            }

        summary["stressor_details"] = [
            {
                "name": s.name,
                "type": s.stressor_type,
                "severity": s.severity,
                "active": s.is_active(current_day),
                "days_remaining": max(0, s.start_day + s.duration - current_day),
            }
            for s in self.active_stressors
        ]

        return summary
