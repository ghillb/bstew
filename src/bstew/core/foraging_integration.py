"""
Foraging Systems Integration for NetLogo BEE-STEWARD v2 Parity
==============================================================

Comprehensive integration of all foraging systems with the main simulation loop,
providing seamless coordination between patch selection, communication, proboscis
matching, trip lifecycle management, and colony dynamics.
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass
import logging
import time
import random
from collections import defaultdict

from .foraging_algorithms import (
    ForagingDecisionEngine, ForagingTripManager, ForagingContext,
    ForagingDecisionType, ForagingTripLifecycle
)
from .patch_selection import AdvancedPatchSelector, PatchInfo, ResourceType as PatchResourceType
from .bee_communication import (
    ForagingCommunicationIntegrator
)
from ..components.proboscis_matching import ProboscisCorollaSystem
from ..spatial.patches import FlowerSpecies


class ForagingMode(Enum):
    """Foraging operation modes"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    RECRUITMENT_FOLLOWING = "recruitment_following"
    MIXED_STRATEGY = "mixed_strategy"
    EMERGENCY_FORAGING = "emergency_foraging"


@dataclass
class ForagingSessionResult:
    """Result of complete foraging session"""
    session_id: str
    bee_id: int
    colony_id: int
    session_start_time: float
    session_duration: float
    
    # Trip details
    trips_completed: int
    patches_visited: Set[int]
    total_distance_traveled: float
    total_energy_consumed: float
    total_energy_gained: float
    net_energy_result: float
    
    # Resource collection
    nectar_collected: float
    pollen_collected: float
    resource_quality_avg: float
    
    # Communication outcomes
    dances_performed: int
    recruits_obtained: int
    information_shared: int
    
    # Efficiency metrics
    trip_efficiency: float
    energy_efficiency: float
    time_efficiency: float
    overall_success_score: float


class IntegratedForagingSystem(BaseModel):
    """Comprehensive foraging system integrating all components"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Core foraging systems
    decision_engine: ForagingDecisionEngine = Field(default_factory=ForagingDecisionEngine)
    trip_manager: ForagingTripManager = Field(default_factory=ForagingTripManager)
    patch_selector: AdvancedPatchSelector = Field(default_factory=AdvancedPatchSelector)
    communication_integrator: ForagingCommunicationIntegrator = Field(default_factory=ForagingCommunicationIntegrator)
    proboscis_system: ProboscisCorollaSystem = Field(default_factory=ProboscisCorollaSystem)
    
    # System state
    active_foraging_sessions: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    patch_database: Dict[int, PatchInfo] = Field(default_factory=dict)
    foraging_history: List[ForagingSessionResult] = Field(default_factory=list)
    
    # Performance tracking
    system_performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    colony_foraging_strategies: Dict[int, ForagingMode] = Field(default_factory=dict)
    
    # Configuration
    max_session_duration: float = Field(default=480.0, description="Maximum foraging session duration (minutes)")
    performance_update_interval: int = Field(default=100, description="Steps between performance updates")
    
    # Logging
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
    
    def initialize_colony_foraging(self, colony_id: int, colony_state: Dict[str, Any],
                                 landscape_patches: List[PatchInfo]) -> None:
        """Initialize foraging system for a colony"""
        
        self.logger.info(f"Initializing foraging system for colony {colony_id}")
        
        # Update patch database with landscape information
        for patch in landscape_patches:
            self.patch_database[patch.patch_id] = patch
        
        # Set initial foraging strategy based on colony state
        energy_level = colony_state.get("energy_level", 1000.0)
        forager_count = colony_state.get("forager_count", 20)
        
        if energy_level < 500.0:
            strategy = ForagingMode.EMERGENCY_FORAGING
        elif forager_count > 30:
            strategy = ForagingMode.MIXED_STRATEGY
        else:
            strategy = ForagingMode.EXPLORATION
        
        self.colony_foraging_strategies[colony_id] = strategy
        
        # Initialize performance tracking
        self.system_performance_metrics[colony_id] = {
            "total_foraging_sessions": 0,
            "successful_sessions": 0,
            "average_efficiency": 0.0,
            "communication_events": 0,
            "patch_discoveries": 0,
            "recruitment_success_rate": 0.0
        }
    
    def execute_foraging_step(self, bee_id: int, colony_id: int, bee_state: Dict[str, Any],
                            colony_state: Dict[str, Any], environmental_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step of integrated foraging for a bee"""
        
        step_result = {
            "action_taken": "none",
            "foraging_decision": None,
            "trip_result": None,
            "communication_result": None,
            "state_changes": {},
            "efficiency_metrics": {}
        }
        
        # Check if bee is already in active foraging session
        if bee_id in self.active_foraging_sessions:
            # Continue existing session
            step_result = self._continue_foraging_session(
                bee_id, colony_id, bee_state, colony_state, environmental_context
            )
        else:
            # Decide whether to start new foraging session
            decision_result = self._evaluate_foraging_decision(
                bee_id, colony_id, bee_state, colony_state, environmental_context
            )
            
            if decision_result["should_forage"]:
                # Start new foraging session
                step_result = self._start_foraging_session(
                    bee_id, colony_id, bee_state, colony_state, environmental_context, decision_result
                )
            else:
                step_result["action_taken"] = "foraging_declined"
                step_result["foraging_decision"] = decision_result
        
        return step_result
    
    def _evaluate_foraging_decision(self, bee_id: int, colony_id: int, bee_state: Dict[str, Any],
                                   colony_state: Dict[str, Any], environmental_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether bee should start foraging"""
        
        # Create foraging context
        foraging_context = ForagingContext(
            current_season=environmental_context.get("season", "spring"),
            current_hour=environmental_context.get("hour", 12),
            current_weather=environmental_context.get("weather", "clear"),
            colony_energy_level=colony_state.get("energy_level", 1000.0),
            patch_competition={},  # Will be populated from patch database
            dance_information=[],  # Will be populated from communication system
            scent_trails=[],
            wind_speed=environmental_context.get("wind_speed", 0.0),
            temperature=environmental_context.get("temperature", 20.0),
            flower_density={},
            patch_distances={}
        )
        
        # Get bee's current state and capabilities
        bee_energy = bee_state.get("energy", 100.0)
        bee_state.get("foraging_experience", 0)
        colony_state.get("species", "bombus_terrestris")
        
        # Use decision engine to evaluate foraging opportunity
        available_patches = list(self.patch_database.values())
        decision_result = self.decision_engine.make_foraging_decision(
            bee_id, foraging_context, bee_energy, available_patches
        )
        
        # Handle both mock (dictionary) and real (tuple) return types
        if isinstance(decision_result, dict):
            # Mock return - dictionary format
            foraging_decision = decision_result
        else:
            # Real implementation - tuple format
            decision_type, selected_patch_id = decision_result
            
            # Convert the tuple result to a dictionary-like structure for compatibility
            selected_patches = []
            if selected_patch_id is not None:
                selected_patch = next((p for p in available_patches if p.patch_id == selected_patch_id), None)
                if selected_patch:
                    selected_patches = [selected_patch]
            
            foraging_decision = {
                "decision_type": decision_type,
                "selected_patches": selected_patches,
                "confidence": 0.7 if selected_patches else 0.3  # Default confidence based on whether patch was found
            }
        
        # Apply colony strategy influence
        colony_strategy = self.colony_foraging_strategies.get(colony_id, ForagingMode.MIXED_STRATEGY)
        decision_probability = self._apply_strategy_influence(foraging_decision, colony_strategy)
        
        should_forage = random.random() < decision_probability
        
        return {
            "should_forage": should_forage,
            "decision_type": foraging_decision["decision_type"],
            "selected_patches": foraging_decision.get("selected_patches", []),
            "decision_confidence": foraging_decision.get("confidence", 0.5),
            "strategy_influence": colony_strategy.value,
            "decision_probability": decision_probability
        }
    
    def _start_foraging_session(self, bee_id: int, colony_id: int, bee_state: Dict[str, Any],
                              colony_state: Dict[str, Any], environmental_context: Dict[str, Any],
                              decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """Start new foraging session"""
        
        session_id = f"session_{bee_id}_{time.time()}"
        
        # Initialize session tracking
        session_data = {
            "session_id": session_id,
            "bee_id": bee_id,
            "colony_id": colony_id,
            "start_time": time.time(),
            "current_trip": 0,
            "completed_trips": [],
            "patches_visited": set(),
            "total_energy_consumed": 0.0,
            "total_energy_gained": 0.0,
            "total_distance": 0.0,
            "resources_collected": {"nectar": 0.0, "pollen": 0.0},
            "communication_events": [],
            "current_patch": None,
            "session_status": "active"
        }
        
        self.active_foraging_sessions[bee_id] = session_data
        
        # Execute first trip
        trip_result = self._execute_foraging_trip(
            bee_id, colony_id, bee_state, colony_state, environmental_context, decision_result
        )
        
        return {
            "action_taken": "foraging_session_started",
            "session_id": session_id,
            "trip_result": trip_result,
            "foraging_decision": decision_result
        }
    
    def _continue_foraging_session(self, bee_id: int, colony_id: int, bee_state: Dict[str, Any],
                                 colony_state: Dict[str, Any], environmental_context: Dict[str, Any]) -> Dict[str, Any]:
        """Continue existing foraging session"""
        
        session_data = self.active_foraging_sessions[bee_id]
        
        # Check session termination conditions
        session_duration = time.time() - session_data["start_time"]
        if (session_duration > self.max_session_duration or 
            bee_state.get("energy", 100.0) < 20.0 or
            session_data["session_status"] != "active"):
            
            return self._end_foraging_session(bee_id, colony_id, bee_state, colony_state)
        
        # Continue with next trip
        trip_result = self._execute_foraging_trip(
            bee_id, colony_id, bee_state, colony_state, environmental_context, {}
        )
        
        # If the trip resulted in session termination, return that result directly
        if trip_result.get("action_taken") == "foraging_session_ended":
            return trip_result
        
        return {
            "action_taken": "foraging_session_continued",
            "session_id": session_data["session_id"],
            "trip_result": trip_result,
            "session_duration": session_duration
        }
    
    def _execute_foraging_trip(self, bee_id: int, colony_id: int, bee_state: Dict[str, Any],
                             colony_state: Dict[str, Any], environmental_context: Dict[str, Any],
                             decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual foraging trip"""
        
        session_data = self.active_foraging_sessions[bee_id]
        bee_species = colony_state.get("species", "bombus_terrestris")
        
        # Select target patch using advanced patch selector
        current_conditions = {
            "weather": environmental_context.get("weather", "clear"),
            "hour": environmental_context.get("hour", 12),
            "season": environmental_context.get("season", "spring"),
            "temperature": environmental_context.get("temperature", 20.0)
        }
        
        available_patches = [p for p in self.patch_database.values() 
                           if p.patch_id not in session_data["patches_visited"]]
        
        if not available_patches:
            # No new patches available, end session
            session_data["session_status"] = "no_patches_available"
            return self._end_foraging_session(bee_id, colony_id, bee_state, colony_state)
        
        # Use patch selector to choose optimal patch
        selection_result = self.patch_selector.select_optimal_patches(
            available_patches, bee_species, bee_state.get("energy", 100.0),
            bee_state.get("memory", {}), current_conditions
        )
        
        # Handle both mock (dictionary) and real (list) return types
        if isinstance(selection_result, dict):
            # Mock return - dictionary format
            selected_patches = selection_result.get("selected_patches", [])
            patch_qualities = selection_result.get("patch_qualities", {})
        else:
            # Real implementation - list format
            selected_patches = selection_result
            patch_qualities = {p.patch_id: p.success_rate for p in selected_patches}
        
        if not selected_patches:
            # No suitable patches found
            session_data["session_status"] = "no_suitable_patches"
            return self._end_foraging_session(bee_id, colony_id, bee_state, colony_state)
        
        target_patch = selected_patches[0]
        
        # Check proboscis-corolla compatibility
        accessibility_results = self._evaluate_flower_accessibility(bee_species, target_patch)
        
        # Calculate patch quality 
        patch_quality = patch_qualities.get(target_patch.patch_id, target_patch.success_rate)
        
        # Simulate complete foraging trip
        trip_lifecycle = self.trip_manager.simulate_complete_foraging_trip(
            hive_location=colony_state.get("location", (0.0, 0.0)),
            patch_location=target_patch.location,
            patch_quality=patch_quality,
            bee_energy=bee_state.get("energy", 100.0),
            environmental_conditions=environmental_context,
            accessibility_results=accessibility_results
        )
        
        # Update session data
        session_data["completed_trips"].append(trip_lifecycle)
        session_data["patches_visited"].add(target_patch.patch_id)
        session_data["current_trip"] += 1
        session_data["total_energy_consumed"] += trip_lifecycle.energy_consumed_travel + trip_lifecycle.energy_consumed_foraging
        session_data["total_energy_gained"] += trip_lifecycle.energy_gained
        session_data["total_distance"] += trip_lifecycle.travel_time_to_patch + trip_lifecycle.travel_time_to_hive
        
        # Estimate resource collection based on trip efficiency
        resource_amount = trip_lifecycle.flowers_visited * 2.0  # Simplified calculation
        if target_patch.resource_type == PatchResourceType.NECTAR:
            session_data["resources_collected"]["nectar"] += resource_amount
        else:
            session_data["resources_collected"]["pollen"] += resource_amount
        
        # Process communication events
        communication_result = self._process_trip_communication(
            bee_id, colony_id, target_patch, trip_lifecycle, accessibility_results, colony_state
        )
        
        if communication_result["communication_triggered"]:
            session_data["communication_events"].append(communication_result)
        
        trip_result = {
            "patch_id": target_patch.patch_id,
            "trip_lifecycle": trip_lifecycle,
            "accessibility_results": accessibility_results,
            "communication_result": communication_result,
            "selection_analytics": selection_result.get("selection_summary", {}),
            "trip_efficiency": trip_lifecycle.net_energy_gain / max(1.0, trip_lifecycle.total_trip_duration),
            "success": trip_lifecycle.net_energy_gain > 0
        }
        
        return trip_result
    
    def _evaluate_flower_accessibility(self, bee_species: str, patch: PatchInfo) -> Dict[str, Any]:
        """Evaluate flower accessibility for bee species"""
        
        # Create mock flower species from patch info
        mock_flower = FlowerSpecies(
            name=f"Patch_{patch.patch_id}_flower",
            bloom_start=120,  # May (day 120)
            bloom_end=240,    # August (day 240)
            corolla_depth_mm=patch.quality_metrics.get("corolla_depth", 5.0),
            corolla_width_mm=patch.quality_metrics.get("corolla_width", 3.0),
            nectar_production=50.0,
            pollen_production=30.0,
            flower_density=10.0,
            attractiveness=0.8,
            nectar_accessibility=0.8
        )
        
        # Get proboscis characteristics and calculate accessibility
        proboscis = self.proboscis_system.get_species_proboscis(bee_species)
        accessibility_result = self.proboscis_system.calculate_accessibility(proboscis, mock_flower)
        efficiency_modifiers = self.proboscis_system.get_foraging_efficiency_modifier(bee_species, mock_flower)
        
        return {
            "accessibility_result": accessibility_result,
            "efficiency_modifiers": efficiency_modifiers,
            "proboscis_characteristics": proboscis.model_dump(),
            "flower_compatibility": accessibility_result.accessibility_score
        }
    
    def _process_trip_communication(self, bee_id: int, colony_id: int, patch: PatchInfo,
                                  trip_lifecycle: ForagingTripLifecycle, accessibility_results: Dict[str, Any],
                                  colony_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process communication events related to foraging trip"""
        
        # Prepare foraging trip result for communication system
        foraging_trip_result = {
            "net_energy_gain": trip_lifecycle.net_energy_gain,
            "trip_efficiency": trip_lifecycle.net_energy_gain / max(1.0, trip_lifecycle.total_trip_duration),
            "patch_quality": trip_lifecycle.net_energy_gain / max(1.0, trip_lifecycle.energy_gained),
            "resource_abundance": trip_lifecycle.flowers_visited * 10.0,  # Simplified
            "energy_cost": trip_lifecycle.energy_consumed_travel + trip_lifecycle.energy_consumed_foraging,
            "patch_id": patch.patch_id
        }
        
        patch_info = {
            "patch_id": patch.patch_id,
            "location": patch.location,
            "distance": patch.distance_from_hive,
            "resource_type": patch.resource_type.value
        }
        
        # Use communication integrator to process foraging success
        communication_result = self.communication_integrator.integrate_foraging_success_with_communication(
            bee_id, foraging_trip_result, patch_info, colony_state
        )
        
        return communication_result
    
    def _end_foraging_session(self, bee_id: int, colony_id: int, bee_state: Dict[str, Any],
                            colony_state: Dict[str, Any]) -> Dict[str, Any]:
        """End foraging session and compile results"""
        
        if bee_id not in self.active_foraging_sessions:
            return {"action_taken": "session_not_found"}
        
        session_data = self.active_foraging_sessions.pop(bee_id)
        
        # Compile session results
        session_duration = time.time() - session_data["start_time"]
        total_trips = len(session_data["completed_trips"])
        
        if total_trips > 0:
            trip_efficiency = sum(t.net_energy_gain for t in session_data["completed_trips"]) / total_trips
            energy_efficiency = session_data["total_energy_gained"] / max(1.0, session_data["total_energy_consumed"])
            time_efficiency = session_data["total_energy_gained"] / max(1.0, session_duration)
        else:
            trip_efficiency = energy_efficiency = time_efficiency = 0.0
        
        # Calculate overall success score
        success_score = (trip_efficiency * 0.4 + energy_efficiency * 0.3 + time_efficiency * 0.3)
        
        session_result = ForagingSessionResult(
            session_id=session_data["session_id"],
            bee_id=bee_id,
            colony_id=colony_id,
            session_start_time=session_data["start_time"],
            session_duration=session_duration,
            trips_completed=total_trips,
            patches_visited=session_data["patches_visited"],
            total_distance_traveled=session_data["total_distance"],
            total_energy_consumed=session_data["total_energy_consumed"],
            total_energy_gained=session_data["total_energy_gained"],
            net_energy_result=session_data["total_energy_gained"] - session_data["total_energy_consumed"],
            nectar_collected=session_data["resources_collected"]["nectar"],
            pollen_collected=session_data["resources_collected"]["pollen"],
            resource_quality_avg=success_score,  # Simplified
            dances_performed=len([e for e in session_data["communication_events"] if e.get("dance_performed", False)]),
            recruits_obtained=0,  # Would be calculated from communication system
            information_shared=len([e for e in session_data["communication_events"] if e.get("information_shared", False)]),
            trip_efficiency=trip_efficiency,
            energy_efficiency=energy_efficiency,
            time_efficiency=time_efficiency,
            overall_success_score=success_score
        )
        
        # Store session result
        self.foraging_history.append(session_result)
        
        # Update performance metrics
        self._update_performance_metrics(colony_id, session_result)
        
        return {
            "action_taken": "foraging_session_ended",
            "session_result": session_result,
            "session_summary": {
                "duration_minutes": session_duration / 60.0,
                "trips_completed": total_trips,
                "patches_visited": len(session_data["patches_visited"]),
                "net_energy": session_result.net_energy_result,
                "success_score": success_score
            }
        }
    
    def _apply_strategy_influence(self, foraging_decision: Dict[str, Any], strategy: ForagingMode) -> float:
        """Apply colony foraging strategy influence on decision probability"""
        
        base_probability = foraging_decision.get("confidence", 0.5)
        decision_type = foraging_decision.get("decision_type", ForagingDecisionType.EXPLORE_NEW)
        
        # Strategy-specific adjustments
        if strategy == ForagingMode.EXPLORATION:
            if decision_type == ForagingDecisionType.EXPLORE_NEW:
                return min(1.0, base_probability * 1.3)
            elif decision_type == ForagingDecisionType.EXPLOIT_KNOWN:
                return base_probability * 0.7
        
        elif strategy == ForagingMode.EXPLOITATION:
            if decision_type == ForagingDecisionType.EXPLOIT_KNOWN:
                return min(1.0, base_probability * 1.4)
            elif decision_type == ForagingDecisionType.EXPLORE_NEW:
                return base_probability * 0.6
        
        elif strategy == ForagingMode.EMERGENCY_FORAGING:
            # More aggressive foraging regardless of decision type
            return min(1.0, base_probability * 1.5)
        
        elif strategy == ForagingMode.RECRUITMENT_FOLLOWING:
            if decision_type == ForagingDecisionType.FOLLOW_DANCE:
                return min(1.0, base_probability * 1.6)
            else:
                return base_probability * 0.8
        
        # MIXED_STRATEGY or default
        return base_probability
    
    def _update_performance_metrics(self, colony_id: int, session_result: ForagingSessionResult) -> None:
        """Update colony foraging performance metrics"""
        
        if str(colony_id) not in self.system_performance_metrics:
            self.system_performance_metrics[str(colony_id)] = {
                "total_foraging_sessions": 0,
                "successful_sessions": 0,
                "average_efficiency": 0.0,
                "communication_events": 0,
                "patch_discoveries": 0,
                "recruitment_success_rate": 0.0
            }
        
        metrics = self.system_performance_metrics[str(colony_id)]
        
        # Update counters
        metrics["total_foraging_sessions"] += 1
        if session_result.overall_success_score > 0.5:
            metrics["successful_sessions"] += 1
        
        # Update rolling average efficiency
        current_avg = metrics["average_efficiency"]
        session_count = metrics["total_foraging_sessions"]
        metrics["average_efficiency"] = ((current_avg * (session_count - 1)) + session_result.overall_success_score) / session_count
        
        # Update communication metrics
        metrics["communication_events"] += session_result.dances_performed + session_result.information_shared
        metrics["patch_discoveries"] += len(session_result.patches_visited)
    
    def get_colony_foraging_analytics(self, colony_id: int) -> Dict[str, Any]:
        """Get comprehensive foraging analytics for colony"""
        
        # Filter history for this colony
        colony_sessions = [s for s in self.foraging_history if s.colony_id == colony_id]
        
        if not colony_sessions:
            return {"message": "No foraging data available for colony"}
        
        # Calculate analytics
        total_sessions = len(colony_sessions)
        successful_sessions = sum(1 for s in colony_sessions if s.overall_success_score > 0.5)
        
        avg_efficiency = sum(s.overall_success_score for s in colony_sessions) / total_sessions
        avg_energy_gain = sum(s.net_energy_result for s in colony_sessions) / total_sessions
        total_resources = sum(s.nectar_collected + s.pollen_collected for s in colony_sessions)
        
        # Recent performance (last 20 sessions)
        recent_sessions = colony_sessions[-20:]
        recent_avg_efficiency = (sum(s.overall_success_score for s in recent_sessions) / len(recent_sessions)
                               if recent_sessions else 0.0)
        
        return {
            "colony_id": colony_id,
            "foraging_strategy": self.colony_foraging_strategies.get(colony_id, ForagingMode.MIXED_STRATEGY).value,
            "session_statistics": {
                "total_sessions": total_sessions,
                "successful_sessions": successful_sessions,
                "success_rate": successful_sessions / total_sessions,
                "average_efficiency": avg_efficiency,
                "recent_efficiency": recent_avg_efficiency
            },
            "resource_collection": {
                "total_resources_collected": total_resources,
                "average_energy_gain": avg_energy_gain,
                "nectar_vs_pollen_ratio": self._calculate_resource_ratio(colony_sessions)
            },
            "patch_utilization": {
                "unique_patches_visited": len(set().union(*[s.patches_visited for s in colony_sessions])),
                "avg_patches_per_session": sum(len(s.patches_visited) for s in colony_sessions) / total_sessions
            },
            "communication_activity": {
                "total_dances": sum(s.dances_performed for s in colony_sessions),
                "total_information_sharing": sum(s.information_shared for s in colony_sessions),
                "communication_rate": sum(s.dances_performed + s.information_shared for s in colony_sessions) / total_sessions
            },
            "performance_metrics": self.system_performance_metrics.get(colony_id, {}),
            "active_sessions": len([s for s in self.active_foraging_sessions.values() if s["colony_id"] == colony_id])
        }
    
    def _calculate_resource_ratio(self, sessions: List[ForagingSessionResult]) -> Dict[str, float]:
        """Calculate nectar vs pollen collection ratio"""
        
        total_nectar = sum(s.nectar_collected for s in sessions)
        total_pollen = sum(s.pollen_collected for s in sessions)
        total_resources = total_nectar + total_pollen
        
        if total_resources > 0:
            return {
                "nectar_percentage": total_nectar / total_resources,
                "pollen_percentage": total_pollen / total_resources,
                "nectar_pollen_ratio": total_nectar / max(1.0, total_pollen)
            }
        else:
            return {"nectar_percentage": 0.0, "pollen_percentage": 0.0, "nectar_pollen_ratio": 0.0}
    
    def get_system_wide_analytics(self) -> Dict[str, Any]:
        """Get system-wide foraging analytics across all colonies"""
        
        all_colonies = set(s.colony_id for s in self.foraging_history)
        
        system_analytics = {
            "total_colonies": len(all_colonies),
            "total_sessions": len(self.foraging_history),
            "total_patches_in_database": len(self.patch_database),
            "active_foraging_sessions": len(self.active_foraging_sessions),
            "colony_analytics": {},
            "system_performance": {}
        }
        
        # Collect analytics for each colony
        for colony_id in all_colonies:
            system_analytics["colony_analytics"][colony_id] = self.get_colony_foraging_analytics(colony_id)
        
        # System-wide performance metrics
        if self.foraging_history:
            system_analytics["system_performance"] = {
                "overall_success_rate": sum(1 for s in self.foraging_history if s.overall_success_score > 0.5) / len(self.foraging_history),
                "average_efficiency": sum(s.overall_success_score for s in self.foraging_history) / len(self.foraging_history),
                "total_energy_collected": sum(s.net_energy_result for s in self.foraging_history),
                "total_resources_collected": sum(s.nectar_collected + s.pollen_collected for s in self.foraging_history),
                "communication_effectiveness": self.communication_integrator.get_recruitment_analytics(),
                "patch_utilization": self._calculate_patch_utilization_stats()
            }
        
        return system_analytics
    
    def _calculate_patch_utilization_stats(self) -> Dict[str, Any]:
        """Calculate patch utilization statistics"""
        
        patch_visit_counts = defaultdict(int)
        patch_success_rates = defaultdict(list)
        
        for session in self.foraging_history:
            for patch_id in session.patches_visited:
                patch_visit_counts[patch_id] += 1
                patch_success_rates[patch_id].append(session.overall_success_score)
        
        if patch_visit_counts:
            most_visited_patch = max(patch_visit_counts.items(), key=lambda x: x[1])
            avg_success_by_patch = {
                patch_id: sum(scores) / len(scores) 
                for patch_id, scores in patch_success_rates.items()
            }
            most_successful_patch = max(avg_success_by_patch.items(), key=lambda x: x[1])
            
            return {
                "total_unique_patches_visited": len(patch_visit_counts),
                "most_visited_patch": {"patch_id": most_visited_patch[0], "visit_count": most_visited_patch[1]},
                "most_successful_patch": {"patch_id": most_successful_patch[0], "success_rate": most_successful_patch[1]},
                "average_visits_per_patch": sum(patch_visit_counts.values()) / len(patch_visit_counts),
                "patch_utilization_rate": len(patch_visit_counts) / len(self.patch_database)
            }
        else:
            return {"message": "No patch utilization data available"}