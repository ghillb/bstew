"""
Pytest configuration and fixtures for BSTEW tests
=================================================
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
from typing import Dict, Any, Generator

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bstew.utils.config import ConfigManager


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def config_manager() -> ConfigManager:
    """Provide a ConfigManager instance"""
    return ConfigManager()


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide a test configuration"""
    return {
        "simulation": {"duration_days": 30, "random_seed": 42},
        "colony": {"initial_population": 10000, "initial_brood": 2000},
        "biology": {
            "max_lifespan": 45,
            "egg_laying_rate": 1000,
            "base_mortality": 0.02,
        },
        "foraging": {"max_range": 2000.0, "efficiency": 0.8},
        "disease": {"varroa_enabled": False, "virus_enabled": False},
        "environment": {"weather_enabled": False},
    }


@pytest.fixture
def large_test_config() -> Dict[str, Any]:
    """Provide a larger test configuration for performance tests"""
    return {
        "simulation": {"duration_days": 365, "random_seed": 42},
        "colony": {"initial_population": 25000, "initial_brood": 5000},
        "biology": {
            "max_lifespan": 45,
            "egg_laying_rate": 1500,
            "base_mortality": 0.02,
        },
    }


@pytest.fixture(autouse=True)
def cleanup_after_test() -> Generator[None, None, None]:
    """Cleanup after each test"""
    yield
    # Force garbage collection
    import gc

    gc.collect()


# Test markers
def pytest_configure(config: Any) -> None:
    """Configure pytest markers"""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark integration tests
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)

        # Mark slow tests
        if any(
            keyword in item.name
            for keyword in ["seasonal", "full_year", "large", "complex"]
        ):
            item.add_marker(pytest.mark.slow)
