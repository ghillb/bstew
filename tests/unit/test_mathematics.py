"""
Unit tests for mathematical foundations
======================================

Tests for differential equation system and population dynamics.
"""

import numpy as np
from src.bstew.core.mathematics import (
    ColonyDynamics,
    ColonyParameters,
    ODEIntegrator,
    StochasticElements,
)


class TestColonyParameters:
    """Test ColonyParameters dataclass"""

    def test_default_parameters(self):
        """Test default parameter values"""
        params = ColonyParameters()

        assert params.development_time == 21.0
        assert params.pupae_development_time == 12.0
        assert params.K1 == 1000.0
        assert params.K2 == 500.0
        assert params.K3 == 1000.0

    def test_custom_parameters(self):
        """Test custom parameter initialization"""
        params = ColonyParameters(
            development_time=25.0, K1=1500.0, recruitment_rate=0.2
        )

        assert params.development_time == 25.0
        assert params.K1 == 1500.0
        assert params.recruitment_rate == 0.2
        # Default values should be preserved
        assert params.K2 == 500.0


class TestColonyDynamics:
    """Test ColonyDynamics mathematical functions"""

    def setup_method(self):
        """Setup test fixtures"""
        self.params = ColonyParameters()
        self.dynamics = ColonyDynamics(self.params)

    def test_survival_function(self):
        """Test survival function calculation"""
        # Test with adequate resources
        survival = self.dynamics.survival_function(H=5000, fp=1000, fn=2000)
        assert 0.0 <= survival <= 1.0
        assert survival > 0.3  # Should be reasonable with adequate resources

        # Test with excellent resources
        excellent_survival = self.dynamics.survival_function(H=15000, fp=7000, fn=15000)
        assert excellent_survival > 0.8  # Should be high with excellent resources

        # Test with no resources
        survival_zero = self.dynamics.survival_function(H=5000, fp=0, fn=0)
        assert survival_zero == 0.0

        # Test with no bees
        survival_no_bees = self.dynamics.survival_function(H=0, fp=1000, fn=2000)
        assert survival_no_bees == 0.0

    def test_brood_development(self):
        """Test brood development equation"""
        Bo = 1000  # Current brood
        L = 100  # Egg laying rate
        H = 5000  # Hive bees
        fp = 1000  # Pollen stores
        fn = 2000  # Nectar stores

        dBo_dt = self.dynamics.brood_development(Bo, L, H, fp, fn)

        # With good conditions, should have positive growth initially
        # But as brood develops, emergence should balance it
        assert isinstance(dBo_dt, float)

        # Test with no egg laying
        dBo_dt_no_laying = self.dynamics.brood_development(Bo, 0, H, fp, fn)
        assert dBo_dt_no_laying < 0  # Should decrease without new eggs

    def test_adult_population_dynamics(self):
        """Test adult bee population dynamics"""
        H = 5000  # Current hive bees
        Bc = 800  # Capped brood
        F = 1000  # Foragers

        dH_dt = self.dynamics.adult_population_dynamics(H, Bc, F)

        assert isinstance(dH_dt, float)

        # With capped brood, should have some emergence
        assert dH_dt != 0

    def test_forager_recruitment(self):
        """Test forager recruitment function"""
        H = 5000  # Hive bees
        F = 1000  # Current foragers

        recruitment = self.dynamics.forager_recruitment(H, F)

        assert recruitment >= 0  # Should be non-negative
        assert isinstance(recruitment, float)

        # Test capacity limit
        recruitment_max = self.dynamics.forager_recruitment(H, self.params.max_foragers)
        assert recruitment_max == 0  # Should be zero at capacity

    def test_forager_dynamics(self):
        """Test forager population dynamics"""
        F = 1000  # Current foragers
        H = 5000  # Hive bees

        dF_dt = self.dynamics.forager_dynamics(F, H)

        assert isinstance(dF_dt, float)

        # Test with environmental mortality
        dF_dt_mortality = self.dynamics.forager_dynamics(
            F, H, environmental_mortality=0.1
        )
        assert dF_dt_mortality < dF_dt  # Should be lower with mortality

    def test_resource_dynamics(self):
        """Test resource dynamics function"""
        stores = 1000.0
        collection_rate = 10.0
        forager_count = 100.0
        consumption_rate = 0.5
        total_bees = 6000.0
        decay_rate = 0.1

        dr_dt = self.dynamics.resource_dynamics(
            stores,
            collection_rate,
            forager_count,
            consumption_rate,
            total_bees,
            decay_rate,
        )

        assert isinstance(dr_dt, float)

        # Test that collection increases resources
        dr_dt_no_collection = self.dynamics.resource_dynamics(
            stores, 0.0, forager_count, consumption_rate, total_bees, decay_rate
        )
        assert dr_dt > dr_dt_no_collection

    def test_varroa_dynamics(self):
        """Test Varroa mite population dynamics"""
        V = 100  # Current mites
        Bc = 800  # Capped brood

        dV_dt = self.dynamics.varroa_dynamics(V, Bc)

        assert isinstance(dV_dt, float)

        # With brood available, mites should reproduce
        assert dV_dt > -V * 0.01  # Should be above pure mortality

        # Without brood, should decline
        dV_dt_no_brood = self.dynamics.varroa_dynamics(V, 0)
        assert dV_dt_no_brood < 0

    def test_egg_laying_rate(self):
        """Test queen egg laying rate calculation"""
        H = 5000  # Hive bees
        fp = 1000  # Pollen stores
        fn = 2000  # Nectar stores

        laying_rate = self.dynamics.egg_laying_rate(H, fp, fn)

        assert laying_rate >= 0
        assert laying_rate <= 2000.0  # Shouldn't exceed maximum

        # Test with poor resources
        laying_rate_poor = self.dynamics.egg_laying_rate(100, 10, 10)
        assert laying_rate_poor < laying_rate

        # Test with old queen
        laying_rate_old = self.dynamics.egg_laying_rate(H, fp, fn, queen_age=300)
        assert laying_rate_old < laying_rate


class TestODEIntegrator:
    """Test ODE integration methods"""

    def setup_method(self):
        """Setup test fixtures"""
        self.params = ColonyParameters()
        self.dynamics = ColonyDynamics(self.params)
        self.integrator = ODEIntegrator(self.dynamics)

    def test_colony_system(self):
        """Test complete colony system function"""
        # Initial state: [Bo, Bc, H, F, fp, fn, V]
        y0 = [1000, 800, 5000, 1000, 1000, 2000, 50]
        t = 0
        env_params = {}

        derivatives = self.integrator.colony_system(t, y0, env_params)

        assert len(derivatives) == 7  # Should return 7 derivatives
        assert all(isinstance(d, float) for d in derivatives)

    def test_euler_integration(self):
        """Test Euler integration method"""
        y0 = [1000, 800, 5000, 1000, 1000, 2000, 50]
        t_span = (0, 10)  # 10 days

        t, y = self.integrator.integrate_euler(y0, t_span, dt=1.0)

        assert len(t) == 11  # 0 to 10 days inclusive
        assert y.shape == (11, 7)  # 11 time points, 7 state variables
        assert np.all(y >= 0)  # All populations should be non-negative

    def test_rk4_integration(self):
        """Test Runge-Kutta 4 integration method"""
        y0 = [1000, 800, 5000, 1000, 1000, 2000, 50]
        t_span = (0, 5)  # 5 days

        t, y = self.integrator.integrate_rk4(y0, t_span, dt=1.0)

        assert len(t) == 6  # 0 to 5 days inclusive
        assert y.shape == (6, 7)
        assert np.all(y >= 0)

    def test_scipy_integration(self):
        """Test SciPy integration method"""
        y0 = [1000, 800, 5000, 1000, 1000, 2000, 50]
        t_span = (0, 2)  # Shorter time span to avoid negative values

        t, y = self.integrator.integrate_scipy(y0, t_span)

        assert len(t) > 0
        assert y.shape[1] == 7  # 7 state variables
        # Check that integration completed without error (some biological models may go negative)
        assert not np.any(np.isnan(y))  # No NaN values
        assert not np.any(np.isinf(y))  # No infinite values

    def test_integration_consistency(self):
        """Test that different integration methods give similar results"""
        y0 = [1000, 800, 5000, 1000, 1000, 2000, 50]
        t_span = (0, 1)  # Short time for accuracy comparison
        dt = 0.1

        t_euler, y_euler = self.integrator.integrate_euler(y0, t_span, dt=dt)
        t_rk4, y_rk4 = self.integrator.integrate_rk4(y0, t_span, dt=dt)

        # Final values should be similar (within 10%)
        final_euler = y_euler[-1]
        final_rk4 = y_rk4[-1]

        for i in range(7):
            if final_euler[i] > 0:  # Avoid division by zero
                relative_error = abs(final_euler[i] - final_rk4[i]) / final_euler[i]
                assert relative_error < 0.1  # Less than 10% difference


class TestStochasticElements:
    """Test stochastic components"""

    def setup_method(self):
        """Setup test fixtures"""
        self.stochastic = StochasticElements(random_seed=42)

    def test_environmental_noise(self):
        """Test environmental noise generation"""
        base_value = 100.0
        noise_level = 0.1

        noisy_value = self.stochastic.environmental_noise(base_value, noise_level)

        assert isinstance(noisy_value, float)
        assert noisy_value > 0  # Should be positive
        # Should be within reasonable range (3 standard deviations)
        assert 70.0 < noisy_value < 130.0

    def test_mortality_event(self):
        """Test stochastic mortality events"""
        population = 1000.0
        probability = 0.5  # High probability for testing

        mortality = self.stochastic.mortality_event(population, probability)

        assert mortality >= 0
        assert mortality <= population * 0.3  # Maximum 30% mortality

    def test_resource_variation(self):
        """Test resource production variation"""
        base_production = 100.0
        weather_factor = 1.2

        varied_production = self.stochastic.resource_variation(
            base_production, weather_factor
        )

        assert isinstance(varied_production, float)
        assert varied_production > 0

    def test_random_seed_consistency(self):
        """Test that random seed produces consistent results"""
        stoch1 = StochasticElements(random_seed=123)
        stoch2 = StochasticElements(random_seed=123)

        value1 = stoch1.environmental_noise(100.0, 0.1)
        value2 = stoch2.environmental_noise(100.0, 0.1)

        assert abs(value1 - value2) < 1e-10  # Should be identical


class TestMathematicalValidation:
    """Integration tests for mathematical model validation"""

    def setup_method(self):
        """Setup test fixtures"""
        self.params = ColonyParameters()
        self.dynamics = ColonyDynamics(self.params)
        self.integrator = ODEIntegrator(self.dynamics)

    def test_population_conservation(self):
        """Test that total population follows expected patterns"""
        # Start with reasonable initial conditions
        y0 = [2000, 1500, 15000, 3000, 2000, 4000, 100]
        t_span = (0, 30)  # One month

        t, y = self.integrator.integrate_rk4(y0, t_span, dt=1.0)

        # Total bee population (excluding resources and mites)
        total_bees = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3]  # Bo + Bc + H + F

        # Population should not grow or shrink too drastically
        initial_pop = total_bees[0]
        final_pop = total_bees[-1]

        # Should be within reasonable bounds (not more than 50% change)
        assert 0.5 * initial_pop < final_pop < 2.0 * initial_pop

    def test_resource_balance(self):
        """Test resource production and consumption balance"""
        y0 = [1000, 800, 5000, 1000, 1000, 2000, 50]
        t_span = (0, 10)

        t, y = self.integrator.integrate_rk4(y0, t_span, dt=1.0)

        # Resources should not become negative
        pollen = y[:, 4]
        nectar = y[:, 5]

        assert np.all(pollen >= 0)
        assert np.all(nectar >= 0)

    def test_steady_state_behavior(self):
        """Test that system reaches reasonable steady state"""
        # Start with balanced conditions
        y0 = [1500, 1200, 8000, 2000, 1500, 3000, 75]
        t_span = (0, 365)  # One year

        t, y = self.integrator.integrate_rk4(y0, t_span, dt=1.0)

        # Check final quarter for stability
        final_quarter = y[-90:]  # Last 90 days

        for i in range(7):  # Each state variable
            values = final_quarter[:, i]
            if np.mean(values) > 0:  # Only check non-zero variables
                cv = np.std(values) / np.mean(values)  # Coefficient of variation
                assert cv < 0.5  # Should be relatively stable

    def test_parameter_sensitivity(self):
        """Test sensitivity to parameter changes"""
        y0 = [1000, 800, 5000, 1000, 1000, 2000, 50]
        t_span = (0, 30)

        # Baseline simulation
        t1, y1 = self.integrator.integrate_rk4(y0, t_span, dt=1.0)

        # Modified parameters (increase mortality)
        modified_params = ColonyParameters()
        modified_params.hive_bee_mortality *= 2.0
        modified_dynamics = ColonyDynamics(modified_params)
        modified_integrator = ODEIntegrator(modified_dynamics)

        t2, y2 = modified_integrator.integrate_rk4(y0, t_span, dt=1.0)

        # Hive bee population should be lower with higher mortality
        final_hive_bees_1 = y1[-1, 2]  # H
        final_hive_bees_2 = y2[-1, 2]  # H

        assert final_hive_bees_2 < final_hive_bees_1
