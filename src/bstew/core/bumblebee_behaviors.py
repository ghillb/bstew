"""
Bumblebee-Specific Behaviors: Buzz Pollination and Thermoregulation
==================================================================

CRITICAL: This implements biologically unique bumblebee behaviors that
distinguish them from honey bees for accurate conservation modeling.

Key behaviors:
1. Buzz pollination: Sonication of flowers (tomatoes, blueberries) that honey bees cannot perform
2. Thermoregulation: Cold weather foraging capabilities enabling activity at 2-8°C vs honey bee 12°C minimum

Based on:
- Buchmann (1983): Buzz pollination in angiosperms
- De Luca & Vallejo-Marín (2013): What's the 'buzz' about? The ecology and evolutionary significance of buzz-pollination
- Heinrich (1979): Bumblebee Economics: Thermoregulation and energetics
- Stone et al. (1999): Thermoregulation in four species of tropical solitary bees
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging

from .species_parameters import LiteratureValidatedSpeciesParameters


class SonicationFlowerType(Enum):
    """Flower types requiring buzz pollination"""

    TOMATO = "Solanum_lycopersicum"
    BLUEBERRY = "Vaccinium_corymbosum"
    EGGPLANT = "Solanum_melongena"
    CRANBERRY = "Vaccinium_macrocarpon"
    KIWI = "Actinidia_deliciosa"
    POTATO = "Solanum_tuberosum"


class BuzzPollinationEfficiency(BaseModel):
    """Buzz pollination efficiency metrics"""

    model_config = {"validate_assignment": True}

    vibration_frequency_hz: float = Field(
        ge=100.0, le=1000.0, description="Wing beat frequency for sonication (Hz)"
    )
    amplitude_force_g: float = Field(
        ge=0.1, le=30.0, description="Vibration amplitude in g-force"
    )
    duration_ms: float = Field(
        ge=10.0, le=500.0, description="Sonication duration per flower visit (ms)"
    )
    pollen_release_efficiency: float = Field(
        ge=0.0, le=1.0, description="Proportion of available pollen released"
    )
    energy_cost_j: float = Field(
        ge=0.001, le=0.1, description="Energy cost per sonication event (Joules)"
    )


class ThermoregulationCapacity(BaseModel):
    """Thermoregulation capacity metrics"""

    model_config = {"validate_assignment": True}

    min_flight_temperature_c: float = Field(
        ge=-2.0, le=15.0, description="Minimum air temperature for flight activity"
    )
    thoracic_heat_production_w: float = Field(
        ge=0.001, le=0.1, description="Maximum thoracic heat production (Watts)"
    )
    preflight_warmup_time_s: float = Field(
        ge=0.0, le=600.0, description="Time required for pre-flight warm-up (seconds)"
    )
    thermal_efficiency: float = Field(
        ge=0.0, le=1.0, description="Thermal regulation efficiency (0-1)"
    )
    cold_tolerance_advantage: float = Field(
        ge=0.0, le=20.0, description="Temperature advantage over honey bees (°C)"
    )


class BuzzPollinationBehavior:
    """
    Implements buzz pollination behavior unique to bumblebees.

    Buzz pollination (sonication) involves grabbing flowers and vibrating
    flight muscles to shake pollen from poricidal anthers. This behavior
    is critical for crops like tomatoes and blueberries that honey bees
    cannot effectively pollinate.

    Key features:
    - Species-specific sonication frequencies and amplitudes
    - Energy cost calculations for foraging decisions
    - Flower type compatibility assessment
    - Pollen collection efficiency modeling
    """

    def __init__(
        self, species_parameters: LiteratureValidatedSpeciesParameters
    ) -> None:
        self.species_parameters = species_parameters
        self.logger = logging.getLogger(__name__)

        # Initialize species-specific buzz pollination parameters
        self.buzz_efficiency = self._initialize_buzz_parameters()

        # Compatible flower types for this species
        self.compatible_flowers = self._determine_compatible_flowers()

        # Performance tracking
        self.sonication_events = 0
        self.total_energy_expended = 0.0
        self.pollen_collected_mg = 0.0

    def _initialize_buzz_parameters(self) -> BuzzPollinationEfficiency:
        """Initialize species-specific buzz pollination parameters"""

        # Base parameters on species body size
        body_size = self.species_parameters.body_size_mm

        # Larger bees can generate more force but at lower frequencies
        base_frequency = 250.0 + (20.0 - body_size) * 10.0  # Hz
        base_amplitude = (body_size / 20.0) * 15.0  # g-force

        # Species-specific adjustments based on literature
        species_name = self.species_parameters.species_name

        if species_name == "Bombus_terrestris":
            # Most efficient buzz pollinator
            frequency_hz = min(400.0, base_frequency * 1.2)
            amplitude_g = min(25.0, base_amplitude * 1.3)
            efficiency = 0.85
            duration_ms = 150.0

        elif species_name == "Bombus_pascuorum":
            # Medium efficiency, longer duration
            frequency_hz = base_frequency * 0.9
            amplitude_g = base_amplitude * 0.8
            efficiency = 0.65
            duration_ms = 180.0

        elif species_name == "Bombus_lapidarius":
            # High frequency, shorter bursts
            frequency_hz = min(450.0, base_frequency * 1.4)
            amplitude_g = base_amplitude * 0.9
            efficiency = 0.75
            duration_ms = 120.0

        else:
            # Default parameters
            frequency_hz = base_frequency
            amplitude_g = base_amplitude
            efficiency = 0.7
            duration_ms = 150.0

        # Energy cost scales with amplitude and duration
        energy_cost = (amplitude_g / 10.0) * (duration_ms / 1000.0) * 0.02

        return BuzzPollinationEfficiency(
            vibration_frequency_hz=frequency_hz,
            amplitude_force_g=amplitude_g,
            duration_ms=duration_ms,
            pollen_release_efficiency=efficiency,
            energy_cost_j=energy_cost,
        )

    def _determine_compatible_flowers(self) -> List[SonicationFlowerType]:
        """Determine which sonication flowers this species can effectively pollinate"""

        compatible = []
        body_size = self.species_parameters.body_size_mm

        # All bumblebees can buzz pollinate basic crops
        compatible.extend(
            [
                SonicationFlowerType.TOMATO,
                SonicationFlowerType.EGGPLANT,
                SonicationFlowerType.POTATO,
            ]
        )

        # Larger species can handle bigger flowers
        if body_size >= 18.0:
            compatible.extend(
                [SonicationFlowerType.BLUEBERRY, SonicationFlowerType.CRANBERRY]
            )

        # Very large species can handle kiwi
        if body_size >= 22.0:
            compatible.append(SonicationFlowerType.KIWI)

        return compatible

    def can_buzz_pollinate(self, flower_type: SonicationFlowerType) -> bool:
        """Check if this species can buzz pollinate given flower type"""
        return flower_type in self.compatible_flowers

    def perform_buzz_pollination(
        self,
        flower_type: SonicationFlowerType,
        flower_characteristics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Perform buzz pollination on a flower.

        Returns:
            Dict containing sonication results:
            - success: Whether pollination was successful
            - pollen_collected_mg: Amount of pollen collected
            - energy_expended_j: Energy cost of the behavior
            - sonication_duration_ms: Actual duration of sonication
        """

        if not self.can_buzz_pollinate(flower_type):
            return {
                "success": False,
                "pollen_collected_mg": 0.0,
                "energy_expended_j": 0.0,
                "sonication_duration_ms": 0.0,
                "reason": f"Species cannot buzz pollinate {flower_type.value}",
            }

        # Calculate effectiveness based on flower characteristics
        if flower_characteristics is None:
            flower_characteristics = self._get_default_flower_characteristics(
                flower_type
            )

        # Sonication effectiveness
        frequency_match = self._calculate_frequency_effectiveness(
            flower_characteristics.get("optimal_frequency_hz", 300.0)
        )

        size_match = self._calculate_size_compatibility(
            flower_characteristics.get("anther_size_mm", 3.0)
        )

        # Overall effectiveness
        effectiveness = (
            frequency_match
            * size_match
            * self.buzz_efficiency.pollen_release_efficiency
        )

        # Calculate pollen collection
        available_pollen = flower_characteristics.get("available_pollen_mg", 0.5)
        pollen_collected = available_pollen * effectiveness

        # Adjust duration based on effectiveness (less effective = longer tries)
        base_duration = self.buzz_efficiency.duration_ms
        actual_duration = base_duration * (1.0 + (1.0 - effectiveness) * 0.5)

        # Calculate energy cost
        energy_cost = self.buzz_efficiency.energy_cost_j * (
            actual_duration / base_duration
        )

        # Update tracking
        self.sonication_events += 1
        self.total_energy_expended += energy_cost
        self.pollen_collected_mg += pollen_collected

        return {
            "success": effectiveness > 0.3,  # Minimum 30% effectiveness for success
            "pollen_collected_mg": pollen_collected,
            "energy_expended_j": energy_cost,
            "sonication_duration_ms": actual_duration,
            "effectiveness": effectiveness,
            "frequency_match": frequency_match,
            "size_match": size_match,
        }

    def _calculate_frequency_effectiveness(self, optimal_frequency_hz: float) -> float:
        """Calculate how well bee's frequency matches flower's optimal frequency"""

        frequency_diff = abs(
            self.buzz_efficiency.vibration_frequency_hz - optimal_frequency_hz
        )
        max_effective_diff = 100.0  # Hz

        if frequency_diff <= max_effective_diff:
            return 1.0 - (frequency_diff / max_effective_diff) * 0.5
        else:
            return 0.5  # Minimum effectiveness

    def _calculate_size_compatibility(self, anther_size_mm: float) -> float:
        """Calculate size compatibility between bee and flower anthers"""

        bee_size = self.species_parameters.body_size_mm

        # Optimal size ratio - bees work best with anthers ~1/4 to 1/2 their body size
        optimal_anther_size = bee_size / 4.0  # Target anther size
        size_difference = abs(anther_size_mm - optimal_anther_size)
        max_tolerance = optimal_anther_size  # Can handle up to 100% size difference

        if size_difference <= max_tolerance:
            return (
                1.0 - (size_difference / max_tolerance) * 0.2
            )  # 80-100% compatibility
        else:
            return (
                0.8 - (size_difference - max_tolerance) / optimal_anther_size * 0.3
            )  # Decreasing compatibility

    def _get_default_flower_characteristics(
        self, flower_type: SonicationFlowerType
    ) -> Dict[str, float]:
        """Get default characteristics for common sonication flowers"""

        characteristics = {
            SonicationFlowerType.TOMATO: {
                "optimal_frequency_hz": 300.0,
                "anther_size_mm": 2.5,
                "available_pollen_mg": 0.8,
            },
            SonicationFlowerType.BLUEBERRY: {
                "optimal_frequency_hz": 350.0,
                "anther_size_mm": 1.8,
                "available_pollen_mg": 0.3,
            },
            SonicationFlowerType.EGGPLANT: {
                "optimal_frequency_hz": 280.0,
                "anther_size_mm": 3.2,
                "available_pollen_mg": 1.2,
            },
            SonicationFlowerType.CRANBERRY: {
                "optimal_frequency_hz": 400.0,
                "anther_size_mm": 1.5,
                "available_pollen_mg": 0.2,
            },
            SonicationFlowerType.KIWI: {
                "optimal_frequency_hz": 250.0,
                "anther_size_mm": 4.0,
                "available_pollen_mg": 2.0,
            },
            SonicationFlowerType.POTATO: {
                "optimal_frequency_hz": 320.0,
                "anther_size_mm": 2.8,
                "available_pollen_mg": 0.6,
            },
        }

        return characteristics.get(
            flower_type,
            {
                "optimal_frequency_hz": 300.0,
                "anther_size_mm": 3.0,
                "available_pollen_mg": 0.5,
            },
        )

    def get_buzz_pollination_summary(self) -> Dict[str, Any]:
        """Get summary of buzz pollination capabilities and performance"""

        return {
            "species_name": self.species_parameters.species_name,
            "buzz_parameters": {
                "frequency_hz": self.buzz_efficiency.vibration_frequency_hz,
                "amplitude_g": self.buzz_efficiency.amplitude_force_g,
                "duration_ms": self.buzz_efficiency.duration_ms,
                "efficiency": self.buzz_efficiency.pollen_release_efficiency,
                "energy_cost_j": self.buzz_efficiency.energy_cost_j,
            },
            "compatible_flowers": [f.value for f in self.compatible_flowers],
            "performance_metrics": {
                "total_sonication_events": self.sonication_events,
                "total_energy_expended_j": self.total_energy_expended,
                "total_pollen_collected_mg": self.pollen_collected_mg,
                "average_energy_per_event": (
                    self.total_energy_expended / self.sonication_events
                    if self.sonication_events > 0
                    else 0.0
                ),
                "average_pollen_per_event": (
                    self.pollen_collected_mg / self.sonication_events
                    if self.sonication_events > 0
                    else 0.0
                ),
            },
        }


class ThermalRegulationBehavior:
    """
    Implements thermoregulation behavior unique to bumblebees.

    Bumblebees can regulate their thoracic temperature through flight muscle
    contractions, enabling foraging at temperatures as low as 2-8°C where
    honey bees cannot fly (minimum 12°C). This gives bumblebees a significant
    ecological advantage in cool climates and early/late seasons.

    Key features:
    - Pre-flight warm-up behavior
    - Species-specific thermal thresholds
    - Energy cost calculations for thermoregulation
    - Activity decisions based on temperature
    """

    def __init__(
        self, species_parameters: LiteratureValidatedSpeciesParameters
    ) -> None:
        self.species_parameters = species_parameters
        self.logger = logging.getLogger(__name__)

        # Initialize species-specific thermal regulation parameters
        self.thermal_capacity = self._initialize_thermal_parameters()

        # Current thermal state
        self.thoracic_temperature_c = 20.0  # Default thoracic temperature
        self.is_warmed_up = False
        self.warmup_energy_expended = 0.0

        # Performance tracking
        self.cold_weather_flights = 0
        self.total_warmup_time = 0.0
        self.total_thermal_energy = 0.0

    def _initialize_thermal_parameters(self) -> ThermoregulationCapacity:
        """Initialize species-specific thermoregulation parameters"""

        # Base parameters on species characteristics
        body_size = self.species_parameters.body_size_mm
        min_temp_spec = self.species_parameters.temperature_tolerance_min_c

        # Larger bees have better thermal regulation
        heat_production = (body_size / 20.0) * 0.05  # Watts

        # Species-specific thermal capabilities based on literature
        species_name = self.species_parameters.species_name

        if species_name == "Bombus_terrestris":
            # Excellent thermoregulation, very cold tolerant
            min_flight_temp = min_temp_spec  # Use species-specific value
            warmup_time = 30.0 + (10.0 - min_flight_temp) * 5.0
            efficiency = 0.85
            cold_advantage = 12.0 - min_flight_temp  # vs honey bees

        elif species_name == "Bombus_pascuorum":
            # Good thermoregulation, slightly less cold tolerant
            min_flight_temp = min_temp_spec
            warmup_time = 45.0 + (10.0 - min_flight_temp) * 6.0
            efficiency = 0.75
            cold_advantage = 12.0 - min_flight_temp

        elif species_name == "Bombus_lapidarius":
            # Heat-adapted, less cold tolerant but still better than honey bees
            min_flight_temp = min_temp_spec
            warmup_time = 60.0 + (10.0 - min_flight_temp) * 8.0
            efficiency = 0.65
            cold_advantage = 12.0 - min_flight_temp

        else:
            # Default parameters
            min_flight_temp = min_temp_spec
            warmup_time = 45.0
            efficiency = 0.7
            cold_advantage = 5.0

        return ThermoregulationCapacity(
            min_flight_temperature_c=min_flight_temp,
            thoracic_heat_production_w=heat_production,
            preflight_warmup_time_s=warmup_time,
            thermal_efficiency=efficiency,
            cold_tolerance_advantage=cold_advantage,
        )

    def can_fly_at_temperature(self, air_temperature_c: float) -> bool:
        """Check if bee can fly at given air temperature"""
        return air_temperature_c >= self.thermal_capacity.min_flight_temperature_c

    def requires_warmup(self, air_temperature_c: float) -> bool:
        """Check if pre-flight warm-up is required at given temperature"""
        warmup_threshold = self.thermal_capacity.min_flight_temperature_c + 5.0
        return air_temperature_c < warmup_threshold

    def perform_preflight_warmup(
        self, air_temperature_c: float, target_activity: str = "foraging"
    ) -> Dict[str, Any]:
        """
        Perform pre-flight warm-up behavior.

        Args:
            air_temperature_c: Current air temperature
            target_activity: Planned activity ("foraging", "nest_maintenance", etc.)

        Returns:
            Dict containing warm-up results:
            - success: Whether warm-up was successful
            - warmup_time_s: Time spent warming up
            - energy_expended_j: Energy cost of warm-up
            - final_thoracic_temp_c: Achieved thoracic temperature
        """

        if not self.can_fly_at_temperature(air_temperature_c):
            return {
                "success": False,
                "warmup_time_s": 0.0,
                "energy_expended_j": 0.0,
                "final_thoracic_temp_c": air_temperature_c,
                "reason": f"Temperature {air_temperature_c}°C below minimum {self.thermal_capacity.min_flight_temperature_c}°C",
            }

        if not self.requires_warmup(air_temperature_c):
            # No warm-up needed
            self.is_warmed_up = True
            self.thoracic_temperature_c = max(30.0, air_temperature_c + 10.0)
            return {
                "success": True,
                "warmup_time_s": 0.0,
                "energy_expended_j": 0.0,
                "final_thoracic_temp_c": self.thoracic_temperature_c,
                "reason": "No warm-up required",
            }

        # Calculate required warm-up
        temp_deficit = max(0.0, 30.0 - air_temperature_c)  # Target 30°C thoracic temp

        # Warm-up time increases with temperature deficit
        base_warmup_time = self.thermal_capacity.preflight_warmup_time_s
        actual_warmup_time = base_warmup_time * (1.0 + temp_deficit * 0.1)

        # Energy cost calculation (reduced to more realistic values)
        heat_production_rate = (
            self.thermal_capacity.thoracic_heat_production_w * 0.1
        )  # Scale down heat production
        energy_cost = heat_production_rate * actual_warmup_time

        # Account for thermal efficiency
        energy_cost /= self.thermal_capacity.thermal_efficiency

        # Calculate final thoracic temperature
        heat_generated = (
            heat_production_rate
            * actual_warmup_time
            * self.thermal_capacity.thermal_efficiency
        )
        temp_increase = heat_generated * 200.0  # Conversion factor for bee mass

        final_thoracic_temp = air_temperature_c + temp_increase
        final_thoracic_temp = min(final_thoracic_temp, 40.0)  # Max physiological limit

        # Update state
        self.is_warmed_up = final_thoracic_temp >= 25.0  # Minimum for flight
        self.thoracic_temperature_c = final_thoracic_temp
        self.warmup_energy_expended += energy_cost

        # Update tracking
        self.total_warmup_time += actual_warmup_time
        self.total_thermal_energy += energy_cost

        if air_temperature_c < 10.0:
            self.cold_weather_flights += 1

        return {
            "success": self.is_warmed_up,
            "warmup_time_s": actual_warmup_time,
            "energy_expended_j": energy_cost,
            "final_thoracic_temp_c": final_thoracic_temp,
            "thermal_efficiency": self.thermal_capacity.thermal_efficiency,
        }

    def calculate_flight_energy_cost(
        self,
        air_temperature_c: float,
        flight_duration_s: float,
        activity_intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate energy cost of flight including thermoregulation.

        Args:
            air_temperature_c: Air temperature during flight
            flight_duration_s: Duration of flight activity
            activity_intensity: Activity intensity multiplier (0.5-2.0)

        Returns:
            Dict with energy cost breakdown
        """

        if not self.can_fly_at_temperature(air_temperature_c):
            return {
                "total_energy_j": float("inf"),
                "flight_energy_j": 0.0,
                "thermal_energy_j": 0.0,
                "feasible": False,
            }

        # Base flight energy cost (from literature: ~0.1-0.3 J/s for bumblebees)
        base_flight_cost = 0.2 * activity_intensity  # J/s
        flight_energy = base_flight_cost * flight_duration_s

        # Additional thermoregulatory cost in cold conditions
        thermal_energy = 0.0

        if air_temperature_c < 15.0:  # Need active thermoregulation
            temp_deficit = 15.0 - air_temperature_c
            thermal_cost_rate = (
                self.thermal_capacity.thoracic_heat_production_w
                * 0.1
                * (temp_deficit / 10.0)
            )  # Scale down
            thermal_energy = thermal_cost_rate * flight_duration_s
            thermal_energy /= self.thermal_capacity.thermal_efficiency

        total_energy = flight_energy + thermal_energy

        return {
            "total_energy_j": total_energy,
            "flight_energy_j": flight_energy,
            "thermal_energy_j": thermal_energy,
            "feasible": True,
            "thermal_cost_ratio": thermal_energy / total_energy
            if total_energy > 0
            else 0.0,
        }

    def compare_to_honey_bee_capability(
        self, air_temperature_c: float
    ) -> Dict[str, Any]:
        """Compare bumblebee thermal capability to honey bees at given temperature"""

        honey_bee_min_temp = 12.0  # °C - honey bee minimum flight temperature

        bumblebee_can_fly = self.can_fly_at_temperature(air_temperature_c)
        honey_bee_can_fly = air_temperature_c >= honey_bee_min_temp

        temperature_advantage = (
            honey_bee_min_temp - self.thermal_capacity.min_flight_temperature_c
        )

        return {
            "air_temperature_c": air_temperature_c,
            "bumblebee_can_fly": bumblebee_can_fly,
            "honey_bee_can_fly": honey_bee_can_fly,
            "bumblebee_advantage": bumblebee_can_fly and not honey_bee_can_fly,
            "temperature_advantage_c": temperature_advantage,
            "exclusive_foraging_opportunity": bumblebee_can_fly
            and not honey_bee_can_fly,
            "min_flight_temps": {
                "bumblebee": self.thermal_capacity.min_flight_temperature_c,
                "honey_bee": honey_bee_min_temp,
            },
        }

    def get_thermal_regulation_summary(self) -> Dict[str, Any]:
        """Get summary of thermal regulation capabilities and performance"""

        return {
            "species_name": self.species_parameters.species_name,
            "thermal_parameters": {
                "min_flight_temperature_c": self.thermal_capacity.min_flight_temperature_c,
                "heat_production_w": self.thermal_capacity.thoracic_heat_production_w,
                "warmup_time_s": self.thermal_capacity.preflight_warmup_time_s,
                "thermal_efficiency": self.thermal_capacity.thermal_efficiency,
                "cold_advantage_over_honey_bees_c": self.thermal_capacity.cold_tolerance_advantage,
            },
            "current_state": {
                "thoracic_temperature_c": self.thoracic_temperature_c,
                "is_warmed_up": self.is_warmed_up,
                "warmup_energy_expended_j": self.warmup_energy_expended,
            },
            "performance_metrics": {
                "cold_weather_flights": self.cold_weather_flights,
                "total_warmup_time_s": self.total_warmup_time,
                "total_thermal_energy_j": self.total_thermal_energy,
                "average_warmup_time": (
                    self.total_warmup_time / max(1, self.cold_weather_flights)
                ),
                "thermal_energy_efficiency": (
                    self.total_thermal_energy / max(1.0, self.total_warmup_time)  # J/s
                ),
            },
        }


# Export for use in other modules
__all__ = [
    "SonicationFlowerType",
    "BuzzPollinationEfficiency",
    "ThermoregulationCapacity",
    "BuzzPollinationBehavior",
    "ThermalRegulationBehavior",
]
