"""
Mathematical foundations for BSTEW
==================================

Implements the differential equation system and mathematical models
derived from Khoury et al. (2011, 2013) for bee colony dynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional, Any
from pydantic import BaseModel, Field
import math


class ColonyParameters(BaseModel):
    """Parameters for colony dynamics calculations"""

    model_config = {"validate_assignment": True}

    development_time: float = Field(
        default=21.0, ge=0.0, description="Days from egg to adult"
    )
    pupae_development_time: float = Field(
        default=12.0, ge=0.0, description="Pupae development time in days"
    )
    hive_bee_lifespan: float = Field(
        default=30.0, ge=0.0, description="Hive bee lifespan in days"
    )
    forager_lifespan: float = Field(
        default=7.0, ge=0.0, description="Forager lifespan in days"
    )

    # Saturation constants
    K1: float = Field(
        default=1000.0, ge=0.0, description="Hive bee saturation constant"
    )
    K2: float = Field(
        default=500.0, ge=0.0, description="Pollen saturation constant (mg)"
    )
    K3: float = Field(
        default=1000.0, ge=0.0, description="Nectar saturation constant (mg)"
    )
    Kv: float = Field(
        default=100.0, ge=0.0, description="Varroa mite saturation constant"
    )

    # Recruitment parameters
    recruitment_rate: float = Field(
        default=0.1, ge=0.0, description="Recruitment rate delta"
    )
    max_foragers: float = Field(
        default=5000.0, ge=0.0, description="Maximum foragers Fmax"
    )
    social_inhibition: float = Field(
        default=0.001, ge=0.0, description="Social inhibition alpha"
    )

    # Mortality rates
    hive_bee_mortality: float = Field(
        default=0.033, ge=0.0, description="Hive bee mortality rate (1/30 days)"
    )
    forager_mortality: float = Field(
        default=0.14, ge=0.0, description="Forager mortality rate (1/7 days)"
    )

    # Consumption rates
    pollen_consumption: float = Field(
        default=0.5, ge=0.0, description="Pollen consumption mg per bee per day"
    )
    nectar_consumption: float = Field(
        default=1.0, ge=0.0, description="Nectar consumption mg per bee per day"
    )

    # Decay rates
    pollen_decay: float = Field(
        default=0.1, ge=0.0, description="Pollen decay rate per day"
    )
    nectar_decay: float = Field(
        default=0.05, ge=0.0, description="Nectar decay rate per day"
    )

    # Collection rates
    pollen_collection_rate: float = Field(
        default=10.0, ge=0.0, description="Pollen collection mg per forager per day"
    )
    nectar_collection_rate: float = Field(
        default=50.0, ge=0.0, description="Nectar collection mg per forager per day"
    )

    # Queen egg laying parameters
    max_egg_laying_rate: float = Field(
        default=2000.0, ge=0.0, description="Maximum eggs per day that queen can lay"
    )


class ColonyDynamics:
    """
    Core colony dynamics implementation based on Khoury et al. mathematical framework.

    State variables:
    - Bo: Uncapped brood
    - Bc: Capped brood
    - H: Hive bees
    - F: Foragers
    - fp: Pollen stores
    - fn: Nectar stores
    - V: Varroa mites
    """

    def __init__(self, params: ColonyParameters):
        self.params = params

    def survival_function(self, H: float, fp: float, fn: float) -> float:
        """
        Colony survival function: S(H, fp, fn) = (H/(H + K₁)) * (fp/(fp + K₂)) * (fn/(fn + K₃))

        Args:
            H: Hive bee population
            fp: Pollen stores (mg)
            fn: Nectar stores (mg)

        Returns:
            Survival rate (0-1)
        """
        h_term = H / (H + self.params.K1)
        p_term = fp / (fp + self.params.K2)
        n_term = fn / (fn + self.params.K3)
        return h_term * p_term * n_term

    def brood_development(
        self, Bo: float, L: float, H: float, fp: float, fn: float
    ) -> float:
        """
        Brood development: dBo/dt = L - (Bo/τ) * S(H, fp, fn)

        Args:
            Bo: Current uncapped brood
            L: Daily egg laying rate
            H: Hive bee population
            fp: Pollen stores
            fn: Nectar stores

        Returns:
            Rate of change in uncapped brood
        """
        survival = self.survival_function(H, fp, fn)
        return L - (Bo / self.params.development_time) * survival

    def adult_population_dynamics(
        self, H: float, Bc: float, F: float, resource_stress: float = 0.0
    ) -> float:
        """
        Adult bee population: dH/dt = Bc/τc - H * (1/τh + μh) - γ(H, F)

        Args:
            H: Current hive bee population
            Bc: Capped brood population
            F: Forager population
            resource_stress: Additional mortality from resource stress

        Returns:
            Rate of change in hive bee population
        """
        emergence = Bc / self.params.pupae_development_time
        natural_mortality = H * (
            1 / self.params.hive_bee_lifespan + self.params.hive_bee_mortality
        )
        transition_to_foragers = self.forager_recruitment(H, F)
        stress_mortality = H * resource_stress

        return emergence - natural_mortality - transition_to_foragers - stress_mortality

    def forager_recruitment(self, H: float, F: float) -> float:
        """
        Forager recruitment: γ(H, F) = δ * H * (1 - F/Fmax) * (1 - exp(-αF))

        Args:
            H: Hive bee population
            F: Current forager population

        Returns:
            Rate of transition from hive bees to foragers
        """
        capacity_factor = 1 - (F / self.params.max_foragers)
        social_factor = 1 - math.exp(-self.params.social_inhibition * F)

        return self.params.recruitment_rate * H * capacity_factor * social_factor

    def forager_dynamics(
        self, F: float, H: float, environmental_mortality: float = 0.0
    ) -> float:
        """
        Forager population dynamics: dF/dt = γ(H, F) - F * (1/τf + μf)

        Args:
            F: Current forager population
            H: Hive bee population
            environmental_mortality: Additional mortality from weather/environment

        Returns:
            Rate of change in forager population
        """
        recruitment = self.forager_recruitment(H, F)
        natural_mortality = F * (
            1 / self.params.forager_lifespan + self.params.forager_mortality
        )
        environmental_mortality = F * environmental_mortality

        return recruitment - natural_mortality - environmental_mortality

    def resource_dynamics(
        self,
        resource_stores: float,
        collection_rate: float,
        forager_count: float,
        consumption_rate: float,
        total_bees: float,
        decay_rate: float,
    ) -> float:
        """
        Resource dynamics: dr/dt = collection - consumption - decay

        Args:
            resource_stores: Current resource level
            collection_rate: Collection rate per forager
            forager_count: Number of foragers
            consumption_rate: Consumption rate per bee
            total_bees: Total bee population
            decay_rate: Resource decay rate

        Returns:
            Rate of change in resource stores
        """
        collection = collection_rate * forager_count
        consumption = consumption_rate * total_bees
        decay = decay_rate * resource_stores

        return collection - consumption - decay

    def pollen_dynamics(self, fp: float, Fp: float, total_bees: float) -> float:
        """Pollen store dynamics"""
        return self.resource_dynamics(
            fp,
            self.params.pollen_collection_rate,
            Fp,
            self.params.pollen_consumption,
            total_bees,
            self.params.pollen_decay,
        )

    def nectar_dynamics(self, fn: float, Fn: float, total_bees: float) -> float:
        """Nectar store dynamics"""
        return self.resource_dynamics(
            fn,
            self.params.nectar_collection_rate,
            Fn,
            self.params.nectar_consumption,
            total_bees,
            self.params.nectar_decay,
        )

    def varroa_dynamics(self, V: float, Bc: float) -> float:
        """
        Varroa mite dynamics: dV/dt = β * V * (Bc/(Bc + Kv)) - δv * V

        Args:
            V: Current varroa mite population
            Bc: Capped brood population

        Returns:
            Rate of change in varroa mite population
        """
        reproduction_rate = 0.02  # β
        mortality_rate = 0.01  # δv

        reproduction = reproduction_rate * V * (Bc / (Bc + self.params.Kv))
        mortality = mortality_rate * V

        return reproduction - mortality

    def egg_laying_rate(
        self, H: float, fp: float, fn: float, queen_age: float = 100.0
    ) -> float:
        """
        Calculate queen egg laying rate based on colony conditions

        Args:
            H: Hive bee population
            fp: Pollen stores
            fn: Nectar stores
            queen_age: Queen age in days

        Returns:
            Daily egg laying rate
        """
        # Base laying rate depends on colony size and resources
        base_rate = 2000.0  # eggs per day at optimal conditions

        # Resource limitation
        resource_factor = self.survival_function(H, fp, fn)

        # Queen age factor (productivity decreases with age)
        age_factor = max(0.1, 1.0 - (queen_age / 365.0) * 0.5)

        # Seasonal factor (simplified - could be more complex)
        seasonal_factor = 1.0  # Would be modified by day of year

        return base_rate * resource_factor * age_factor * seasonal_factor


class ODEIntegrator:
    """
    Numerical integration for colony dynamics system
    """

    def __init__(self, dynamics: ColonyDynamics):
        self.dynamics = dynamics

    def colony_system(
        self, t: float, y: List[float], environmental_params: Dict[str, Any]
    ) -> List[float]:
        """
        Complete colony dynamics system for ODE integration

        Args:
            t: Time
            y: State vector [Bo, Bc, H, F, fp, fn, V]
            environmental_params: Environmental conditions

        Returns:
            Derivatives [dBo/dt, dBc/dt, dH/dt, dF/dt, dfp/dt, dfn/dt, dV/dt]
        """
        Bo, Bc, H, F, fp, fn, V = y

        # Calculate egg laying rate
        L = self.dynamics.egg_laying_rate(H, fp, fn)

        # Calculate derivatives
        dBo_dt = self.dynamics.brood_development(Bo, L, H, fp, fn)
        dBc_dt = (
            Bo / self.dynamics.params.development_time
        ) * self.dynamics.survival_function(H, fp, fn) - (
            Bc / self.dynamics.params.pupae_development_time
        )
        dH_dt = self.dynamics.adult_population_dynamics(H, Bc, F)
        dF_dt = self.dynamics.forager_dynamics(
            F, H, environmental_params.get("mortality", 0.0)
        )

        # Assume half foragers collect pollen, half nectar
        Fp = F / 2.0
        Fn = F / 2.0
        total_bees = Bo + Bc + H + F

        dfp_dt = self.dynamics.pollen_dynamics(fp, Fp, total_bees)
        dfn_dt = self.dynamics.nectar_dynamics(fn, Fn, total_bees)
        dV_dt = self.dynamics.varroa_dynamics(V, Bc)

        return [dBo_dt, dBc_dt, dH_dt, dF_dt, dfp_dt, dfn_dt, dV_dt]

    def integrate_euler(
        self,
        y0: List[float],
        t_span: Tuple[float, float],
        dt: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple Euler integration for fast simulation

        Args:
            y0: Initial conditions
            t_span: (t_start, t_end)
            dt: Time step

        Returns:
            (time_points, solution_array)
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        t = np.linspace(t_start, t_end, n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        y[0] = y0

        environmental_params = kwargs.get("environmental_params", {})

        for i in range(n_steps):
            dy_dt = self.colony_system(t[i], y[i], environmental_params)
            y[i + 1] = y[i] + dt * np.array(dy_dt)

            # Ensure non-negative populations
            y[i + 1] = np.maximum(y[i + 1], 0.0)

        return t, y

    def integrate_rk4(
        self,
        y0: List[float],
        t_span: Tuple[float, float],
        dt: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runge-Kutta 4th order integration for accurate simulation

        Args:
            y0: Initial conditions
            t_span: (t_start, t_end)
            dt: Time step

        Returns:
            (time_points, solution_array)
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        t = np.linspace(t_start, t_end, n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        y[0] = y0

        environmental_params = kwargs.get("environmental_params", {})

        for i in range(n_steps):
            k1 = np.array(self.colony_system(t[i], y[i], environmental_params))
            k2 = np.array(
                self.colony_system(
                    t[i] + dt / 2, y[i] + dt * k1 / 2, environmental_params
                )
            )
            k3 = np.array(
                self.colony_system(
                    t[i] + dt / 2, y[i] + dt * k2 / 2, environmental_params
                )
            )
            k4 = np.array(
                self.colony_system(t[i] + dt, y[i] + dt * k3, environmental_params)
            )

            y[i + 1] = y[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            # Ensure non-negative populations
            y[i + 1] = np.maximum(y[i + 1], 0.0)

        return t, y

    def integrate_scipy(
        self,
        y0: List[float],
        t_span: Tuple[float, float],
        method: str = "RK45",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use SciPy's adaptive ODE solvers

        Args:
            y0: Initial conditions
            t_span: (t_start, t_end)
            method: Integration method

        Returns:
            (time_points, solution_array)
        """
        environmental_params = kwargs.get("environmental_params", {})

        def system_wrapper(t: float, y: np.ndarray) -> np.ndarray:
            return np.array(self.colony_system(t, y.tolist(), environmental_params))

        t_eval = kwargs.get("t_eval", None)
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1] - t_span[0]) + 1)

        # Call solve_ivp with proper overload matching
        from typing import cast, Literal

        # Cast method to expected literal type for MyPy
        scipy_method = cast(
            Literal["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"], method
        )

        if t_eval is not None:
            sol = solve_ivp(
                system_wrapper,
                t_span,
                np.array(y0),
                method=scipy_method,
                t_eval=np.array(t_eval),
            )
        else:
            sol = solve_ivp(
                system_wrapper,
                t_span,
                np.array(y0),
                method=scipy_method,
            )

        return sol.t, sol.y.T


class StochasticElements:
    """
    Stochastic components for realistic variation
    """

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.RandomState(random_seed)

    def environmental_noise(self, base_value: float, noise_level: float = 0.1) -> float:
        """Add environmental noise to parameters"""
        return base_value * (1 + self.rng.normal(0, noise_level))

    def mortality_event(self, population: float, probability: float = 0.001) -> float:
        """Stochastic mortality events"""
        if self.rng.random() < probability:
            return population * self.rng.uniform(0.1, 0.3)  # 10-30% mortality
        return 0.0

    def resource_variation(
        self, base_production: float, weather_factor: float = 1.0
    ) -> float:
        """Resource production variation"""
        seasonal_noise = self.rng.normal(1.0, 0.2)
        return base_production * weather_factor * seasonal_noise
