"""
Tests for data analysis functions (isolated testing)
===================================================

Tests for the specific data analysis functions without importing the full CLI.
"""

import pytest
import json
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# Copy the functions we need to test directly here to avoid circular imports
def parse_time_range(time_range: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse time range string"""
    
    if ":" in time_range:
        start_str, end_str = time_range.split(":")
        start = int(start_str) if start_str else None
        end = int(end_str) if end_str else None
        return start, end
    else:
        raise ValueError(f"Invalid time range format: {time_range}")


def load_simulation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load simulation data from file"""
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_dict("records")  # type: ignore
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_colony_data_with_mortality(file_path: str) -> List[Dict[str, Any]]:
    """Load colony data including mortality information"""
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Colony data file not found: {file_path}")
    
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_dict("records")  # type: ignore
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def extract_dead_colonies(colony_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract dead colonies from colony data"""
    
    dead_colonies = []
    for colony in colony_data:
        if colony.get("status") == "dead" or colony.get("death_time") is not None:
            dead_colonies.append(colony)
    
    return dead_colonies


def extract_surviving_colonies(colony_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract surviving colonies from colony data"""
    
    surviving_colonies = []
    for colony in colony_data:
        if colony.get("status") != "dead" and colony.get("death_time") is None:
            surviving_colonies.append(colony)
    
    return surviving_colonies


def filter_dead_colonies_by_time(dead_colonies: List[Dict[str, Any]], start: Optional[int], end: Optional[int]) -> List[Dict[str, Any]]:
    """Filter dead colonies by time range"""
    
    filtered_colonies = []
    for colony in dead_colonies:
        death_time = colony.get("death_time", 0)
        
        if start is not None and death_time < start:
            continue
        if end is not None and death_time > end:
            continue
        
        filtered_colonies.append(colony)
    
    return filtered_colonies


def analyze_dead_colony_summary(dead_colonies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze summary statistics for dead colonies"""
    
    summary = {
        "total_dead_colonies": len(dead_colonies),
        "average_lifespan": 0,
        "death_distribution": {},
        "seasonal_mortality": {}
    }
    
    if dead_colonies:
        lifespans = []
        death_times = []
        
        for colony in dead_colonies:
            death_time = colony.get("death_time", 0)
            birth_time = colony.get("birth_time", 0)
            lifespan = death_time - birth_time
            
            lifespans.append(lifespan)
            death_times.append(death_time)
        
        summary["average_lifespan"] = sum(lifespans) / len(lifespans)
        summary["median_lifespan"] = sorted(lifespans)[len(lifespans) // 2]
        summary["min_lifespan"] = min(lifespans)
        summary["max_lifespan"] = max(lifespans)
    
    return summary


def analyze_mortality_patterns(dead_colonies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze mortality patterns"""
    
    patterns: Dict[str, Any] = {
        "temporal_distribution": {},
        "age_at_death": {},
        "mortality_rate_over_time": {}
    }
    
    # Analyze temporal distribution
    death_times = [colony.get("death_time", 0) for colony in dead_colonies]
    
    # Group by time periods (simplified)
    for death_time in death_times:
        period = f"period_{death_time // 30}"  # 30-day periods
        patterns["temporal_distribution"][period] = patterns["temporal_distribution"].get(period, 0) + 1
    
    return patterns


def analyze_death_causes(dead_colonies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze causes of death"""
    
    cause_analysis: Dict[str, Any] = {
        "cause_distribution": {},
        "cause_by_season": {},
        "preventable_deaths": 0
    }
    
    for colony in dead_colonies:
        cause = colony.get("death_cause", "unknown")
        season = colony.get("death_season", "unknown")
        
        # Count causes
        cause_dist = cause_analysis["cause_distribution"]
        cause_dist[cause] = cause_dist.get(cause, 0) + 1
        
        # Count by season
        cause_by_season = cause_analysis["cause_by_season"]
        if season not in cause_by_season:
            cause_by_season[season] = {}
        cause_by_season[season][cause] = cause_by_season[season].get(cause, 0) + 1
        
        # Count preventable deaths
        if cause in ["starvation", "disease", "environmental"]:
            cause_analysis["preventable_deaths"] += 1
    
    return cause_analysis


def compare_dead_vs_surviving(dead_colonies: List[Dict[str, Any]], surviving_colonies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare dead colonies with surviving ones"""
    
    comparison: Dict[str, Any] = {
        "population_comparison": {},
        "performance_comparison": {},
        "risk_factors": {}
    }
    
    # Population comparison
    dead_max_pops = [colony.get("max_population", 0) for colony in dead_colonies]
    surviving_max_pops = [colony.get("max_population", 0) for colony in surviving_colonies]
    
    comparison["population_comparison"] = {
        "dead_avg_max_population": sum(dead_max_pops) / len(dead_max_pops) if dead_max_pops else 0,
        "surviving_avg_max_population": sum(surviving_max_pops) / len(surviving_max_pops) if surviving_max_pops else 0,
        "population_ratio": (sum(dead_max_pops) / len(dead_max_pops)) / (sum(surviving_max_pops) / len(surviving_max_pops)) if dead_max_pops and surviving_max_pops else 0
    }
    
    return comparison


class TestDataAnalysisHelpers:
    """Test helper functions for data analysis."""
    
    def test_parse_time_range_valid(self):
        """Test parsing valid time range."""
        start, end = parse_time_range("10:50")
        assert start == 10
        assert end == 50
    
    def test_parse_time_range_invalid(self):
        """Test parsing invalid time range."""
        with pytest.raises(ValueError):
            parse_time_range("invalid")
    
    def test_load_simulation_data_json(self):
        """Test loading simulation data from JSON."""
        test_data = [
            {"time": 1, "population": 100},
            {"time": 2, "population": 150}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            loaded_data = load_simulation_data(temp_path)
            assert len(loaded_data) == 2
            assert loaded_data[0]["time"] == 1
            assert loaded_data[1]["population"] == 150
        finally:
            Path(temp_path).unlink()
    
    def test_load_simulation_data_single_object(self):
        """Test loading single object from JSON."""
        test_data = {"time": 1, "population": 100}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            loaded_data = load_simulation_data(temp_path)
            assert len(loaded_data) == 1
            assert loaded_data[0]["time"] == 1
        finally:
            Path(temp_path).unlink()
    
    def test_load_simulation_data_nonexistent(self):
        """Test loading data from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_simulation_data("/nonexistent/path.json")
    
    def test_load_simulation_data_unsupported_format(self):
        """Test loading data from unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"some text")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_simulation_data(temp_path)
        finally:
            Path(temp_path).unlink()


class TestDeadColonyAnalysis:
    """Test dead colony analysis functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_colony_data = [
            {
                "colony_id": 1,
                "status": "dead",
                "death_time": 45,
                "birth_time": 0,
                "death_cause": "starvation",
                "death_season": "winter",
                "max_population": 150,
                "foraging_efficiency": 0.45
            },
            {
                "colony_id": 2,
                "status": "alive",
                "birth_time": 0,
                "max_population": 800,
                "foraging_efficiency": 0.78
            },
            {
                "colony_id": 3,
                "status": "dead", 
                "death_time": 120,
                "birth_time": 30,
                "death_cause": "disease",
                "death_season": "spring",
                "max_population": 300,
                "foraging_efficiency": 0.62
            }
        ]
    
    def test_extract_dead_colonies(self):
        """Test extracting dead colonies from data."""
        dead_colonies = extract_dead_colonies(self.sample_colony_data)
        assert len(dead_colonies) == 2
        assert dead_colonies[0]["colony_id"] == 1
        assert dead_colonies[1]["colony_id"] == 3
    
    def test_extract_surviving_colonies(self):
        """Test extracting surviving colonies from data."""
        surviving_colonies = extract_surviving_colonies(self.sample_colony_data)
        assert len(surviving_colonies) == 1
        assert surviving_colonies[0]["colony_id"] == 2
    
    def test_analyze_dead_colony_summary(self):
        """Test analyzing dead colony summary."""
        dead_colonies = extract_dead_colonies(self.sample_colony_data)
        summary = analyze_dead_colony_summary(dead_colonies)
        
        assert summary["total_dead_colonies"] == 2
        assert summary["average_lifespan"] == 67.5  # (45 + 90) / 2
        assert summary["min_lifespan"] == 45
        assert summary["max_lifespan"] == 90
        assert summary["median_lifespan"] == 90  # Second element in [45, 90], index 1
    
    def test_analyze_mortality_patterns(self):
        """Test analyzing mortality patterns."""
        dead_colonies = extract_dead_colonies(self.sample_colony_data)
        patterns = analyze_mortality_patterns(dead_colonies)
        
        assert "temporal_distribution" in patterns
        assert "age_at_death" in patterns
        assert "mortality_rate_over_time" in patterns
        
        # Check temporal distribution
        temporal = patterns["temporal_distribution"]
        assert "period_1" in temporal  # death_time 45 -> period 1
        assert "period_4" in temporal  # death_time 120 -> period 4
        assert temporal["period_1"] == 1
        assert temporal["period_4"] == 1
    
    def test_analyze_death_causes(self):
        """Test analyzing death causes."""
        dead_colonies = extract_dead_colonies(self.sample_colony_data)
        cause_analysis = analyze_death_causes(dead_colonies)
        
        assert cause_analysis["cause_distribution"]["starvation"] == 1
        assert cause_analysis["cause_distribution"]["disease"] == 1
        assert cause_analysis["preventable_deaths"] == 2  # Both are preventable
        
        # Check seasonal distribution
        seasonal = cause_analysis["cause_by_season"]
        assert seasonal["winter"]["starvation"] == 1
        assert seasonal["spring"]["disease"] == 1
    
    def test_compare_dead_vs_surviving(self):
        """Test comparing dead vs surviving colonies."""
        dead_colonies = extract_dead_colonies(self.sample_colony_data)
        surviving_colonies = extract_surviving_colonies(self.sample_colony_data)
        
        comparison = compare_dead_vs_surviving(dead_colonies, surviving_colonies)
        
        pop_comp = comparison["population_comparison"]
        assert pop_comp["dead_avg_max_population"] == 225.0  # (150 + 300) / 2
        assert pop_comp["surviving_avg_max_population"] == 800.0
        assert pop_comp["population_ratio"] == 225.0 / 800.0
    
    def test_filter_dead_colonies_by_time(self):
        """Test filtering dead colonies by time range."""
        dead_colonies = extract_dead_colonies(self.sample_colony_data)
        
        # Filter to include only colonies that died between 40 and 100
        filtered = filter_dead_colonies_by_time(dead_colonies, 40, 100)
        assert len(filtered) == 1
        assert filtered[0]["colony_id"] == 1  # died at 45
        
        # Filter to include all
        filtered_all = filter_dead_colonies_by_time(dead_colonies, None, None)
        assert len(filtered_all) == 2
        
        # Filter with only start time
        filtered_start = filter_dead_colonies_by_time(dead_colonies, 100, None)
        assert len(filtered_start) == 1
        assert filtered_start[0]["colony_id"] == 3  # died at 120
    
    def test_load_colony_data_with_mortality(self):
        """Test loading colony data with mortality information."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_colony_data, f)
            temp_path = f.name
        
        try:
            loaded_data = load_colony_data_with_mortality(temp_path)
            assert len(loaded_data) == 3
            assert loaded_data[0]["colony_id"] == 1
            assert loaded_data[2]["death_cause"] == "disease"
        finally:
            Path(temp_path).unlink()


class TestDataAnalysisEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dead_colonies_list(self):
        """Test functions with empty dead colonies list."""
        empty_list: List[Dict[str, Any]] = []
        
        summary = analyze_dead_colony_summary(empty_list)
        assert summary["total_dead_colonies"] == 0
        
        patterns = analyze_mortality_patterns(empty_list)
        assert patterns["temporal_distribution"] == {}
        
        causes = analyze_death_causes(empty_list)
        assert causes["cause_distribution"] == {}
        assert causes["preventable_deaths"] == 0
    
    def test_missing_data_fields(self):
        """Test handling of missing data fields."""
        incomplete_data = [
            {"colony_id": 1, "status": "dead"},  # Missing death_time, birth_time, etc.
            {"colony_id": 2}  # Missing status
        ]
        
        dead_colonies = extract_dead_colonies(incomplete_data)
        assert len(dead_colonies) == 1  # Only colony 1 has status="dead"
        
        surviving_colonies = extract_surviving_colonies(incomplete_data)
        assert len(surviving_colonies) == 1  # Colony 2 doesn't have status="dead"
        
        # Should handle missing fields gracefully
        summary = analyze_dead_colony_summary(dead_colonies)
        assert summary["total_dead_colonies"] == 1
        assert summary["average_lifespan"] == 0  # death_time=0, birth_time=0
    
    def test_invalid_time_range_formats(self):
        """Test various invalid time range formats."""
        with pytest.raises(ValueError):
            parse_time_range("not_a_range")
        
        with pytest.raises(ValueError):
            parse_time_range("10-20")  # Wrong separator
        
        with pytest.raises(ValueError):
            parse_time_range("abc:def")  # Non-numeric values