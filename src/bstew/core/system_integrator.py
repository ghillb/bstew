"""
System Integration Manager for BSTEW NetLogo Parity
=================================================

Manages activation and integration of advanced systems including dead colony
tracking, Excel reporting, and spatial analysis for comprehensive simulation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
from pathlib import Path
import os

from .data_collection import ComprehensiveDataCollector
from .colony_mortality import DeadColonyTracker
from ..reports.excel_integration import ExcelReportGenerator
from .spatial_analysis import PatchConnectivityAnalyzer
from .parameter_loader import ParameterLoader
from .health_monitoring import HealthMonitoringSystem


class SystemIntegrator(BaseModel):
    """Manages integration and activation of advanced BSTEW systems"""
    
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}
    
    # Core systems
    data_collector: Optional[ComprehensiveDataCollector] = Field(default=None)
    mortality_tracker: Optional[DeadColonyTracker] = Field(default=None)
    excel_reporter: Optional[ExcelReportGenerator] = Field(default=None)
    spatial_analyzer: Optional[PatchConnectivityAnalyzer] = Field(default=None)
    parameter_loader: Optional[ParameterLoader] = Field(default=None)
    health_monitor: Optional[HealthMonitoringSystem] = Field(default=None)
    
    # Configuration
    output_directory: str = Field(default="artifacts", description="Base output directory")
    enable_mortality_tracking: bool = Field(default=True, description="Enable dead colony tracking")
    enable_excel_reporting: bool = Field(default=True, description="Enable Excel report generation")
    enable_spatial_analysis: bool = Field(default=True, description="Enable spatial analysis")
    enable_health_monitoring: bool = Field(default=True, description="Enable health monitoring")
    
    # Integration status
    systems_initialized: bool = Field(default=False, description="Systems initialization status")
    active_systems: List[str] = Field(default_factory=list, description="Currently active systems")
    
    # Logging
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = logging.getLogger(__name__)
        
    def initialize_systems(self, model: Any = None) -> Dict[str, bool]:
        """Initialize and activate all integrated systems"""
        
        initialization_status = {}
        
        try:
            # Ensure output directories exist
            self._create_output_directories()
            
            # Initialize data collector
            if not self.data_collector:
                self.data_collector = ComprehensiveDataCollector()
                initialization_status["data_collector"] = True
                self.active_systems.append("data_collector")
                self.logger.info("Data collector initialized")
            
            # Initialize mortality tracker
            if self.enable_mortality_tracking and not self.mortality_tracker:
                self.mortality_tracker = DeadColonyTracker()
                initialization_status["mortality_tracker"] = True
                self.active_systems.append("mortality_tracker")
                self.logger.info("Dead colony tracker initialized")
            
            # Initialize Excel reporter
            if self.enable_excel_reporting and not self.excel_reporter:
                try:
                    self.excel_reporter = ExcelReportGenerator(
                        output_directory=os.path.join(self.output_directory, "reports")
                    )
                    initialization_status["excel_reporter"] = True
                    self.active_systems.append("excel_reporter")
                    self.logger.info("Excel reporter initialized")
                except Exception as e:
                    self.logger.warning(f"Excel reporter initialization failed: {e}")
                    initialization_status["excel_reporter_error"] = str(e)
            
            # Initialize spatial analyzer
            if self.enable_spatial_analysis and not self.spatial_analyzer:
                self.spatial_analyzer = PatchConnectivityAnalyzer()
                initialization_status["spatial_analyzer"] = True
                self.active_systems.append("spatial_analyzer")
                self.logger.info("Spatial analyzer initialized")
            
            # Initialize health monitor
            if self.enable_health_monitoring and not self.health_monitor:
                self.health_monitor = HealthMonitoringSystem()
                initialization_status["health_monitor"] = True
                self.active_systems.append("health_monitor")
                self.logger.info("Health monitor initialized")
            
            # Initialize parameter loader if not provided
            if not self.parameter_loader:
                self.parameter_loader = ParameterLoader()
                initialization_status["parameter_loader"] = True
                self.active_systems.append("parameter_loader")
                self.logger.info("Parameter loader initialized")
            
            self.systems_initialized = True
            self.logger.info(f"System integration completed. Active systems: {self.active_systems}")
            
        except Exception as e:
            self.logger.error(f"Error during system initialization: {e}")
            initialization_status["error"] = str(e)
        
        return initialization_status
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories"""
        
        directories = [
            self.output_directory,
            os.path.join(self.output_directory, "reports"),
            os.path.join(self.output_directory, "data"),
            os.path.join(self.output_directory, "spatial"),
            os.path.join(self.output_directory, "mortality"),
            os.path.join(self.output_directory, "health")
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def update_all_systems(self, model: Any, current_step: int) -> None:
        """Update all active systems with current simulation state"""
        
        if not self.systems_initialized:
            return
        
        try:
            # Get colonies from model
            colonies = getattr(model, 'colonies', [])
            if not colonies:
                # Fallback: try to get agents and group by colony
                agents = getattr(model, 'schedule', {})
                if hasattr(agents, 'agents'):
                    colonies = self._group_agents_by_colony(agents.agents)
            
            # Update data collection
            if self.data_collector and "data_collector" in self.active_systems:
                for colony in colonies:
                    self.data_collector.update_colony_metrics(colony, current_step)
                
                # Collect environmental data if available
                if hasattr(model, 'environment'):
                    self.data_collector.collect_environmental_data(model.environment, current_step)
            
            # Update mortality tracking
            if (self.mortality_tracker and "mortality_tracker" in self.active_systems and 
                self.data_collector):
                for colony in colonies:
                    colony_id = getattr(colony, 'colony_id', getattr(colony, 'unique_id', 0))
                    if colony_id in self.data_collector.colony_metrics:
                        metrics = self.data_collector.colony_metrics[colony_id]
                        self.mortality_tracker.update_colony_health(colony_id, metrics, current_step)
            
            # Update health monitoring
            if (self.health_monitor and "health_monitor" in self.active_systems):
                for colony in colonies:
                    self.health_monitor.update_colony_health(colony, current_step)
            
            # Update spatial analysis (less frequent)
            if (self.spatial_analyzer and "spatial_analyzer" in self.active_systems and 
                current_step % 24 == 0):  # Update daily
                self._update_spatial_analysis(model, current_step)
                
        except Exception as e:
            self.logger.error(f"Error updating systems at step {current_step}: {e}")
    
    def _group_agents_by_colony(self, agents: List[Any]) -> List[Any]:
        """Group agents by colony for processing"""
        
        colony_agents = {}
        for agent in agents:
            colony_id = getattr(agent, 'colony_id', 0)
            if colony_id not in colony_agents:
                colony_agents[colony_id] = []
            colony_agents[colony_id].append(agent)
        
        # Create pseudo-colony objects
        colonies = []
        for colony_id, agent_list in colony_agents.items():
            # Create a simple colony-like object
            class PseudoColony:
                def __init__(self, colony_id, agents):
                    self.colony_id = colony_id
                    self.unique_id = colony_id
                    self.agents = agents
            
            colonies.append(PseudoColony(colony_id, agent_list))
        
        return colonies
    
    def _update_spatial_analysis(self, model: Any, current_step: int) -> None:
        """Update spatial analysis with current model state"""
        
        try:
            # Get spatial data from model
            if hasattr(model, 'space') and hasattr(model.space, 'get_all_cell_contents'):
                # Mesa space-based model
                all_agents = model.space.get_all_cell_contents()
                positions = [(agent.pos[0], agent.pos[1]) for agent in all_agents if hasattr(agent, 'pos')]
                
                if positions:
                    self.spatial_analyzer.update_agent_positions(positions, current_step)
            
        except Exception as e:
            self.logger.error(f"Error updating spatial analysis: {e}")
    
    def generate_reports(self, current_step: int, final_report: bool = False) -> Dict[str, str]:
        """Generate reports from all active systems"""
        
        report_paths = {}
        
        try:
            # Generate Excel reports
            if (self.excel_reporter and "excel_reporter" in self.active_systems):
                if final_report or current_step % 240 == 0:  # Every 10 days or final
                    report_path = self.excel_reporter.generate_comprehensive_report(
                        f"simulation_report_step_{current_step}.xlsx"
                    )
                    report_paths["excel_comprehensive"] = report_path
            
            # Generate mortality reports
            if (self.mortality_tracker and "mortality_tracker" in self.active_systems):
                if final_report:
                    mortality_summary = self.mortality_tracker.generate_survival_report()
                    report_paths["mortality_summary"] = mortality_summary
            
            # Generate spatial analysis reports
            if (self.spatial_analyzer and "spatial_analyzer" in self.active_systems):
                if final_report or current_step % 480 == 0:  # Every 20 days or final
                    spatial_report = self._generate_spatial_report(current_step)
                    report_paths["spatial_analysis"] = spatial_report
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            report_paths["error"] = str(e)
        
        return report_paths
    
    def _generate_spatial_report(self, current_step: int) -> str:
        """Generate spatial analysis report"""
        
        try:
            output_path = os.path.join(
                self.output_directory, "spatial", 
                f"spatial_analysis_step_{current_step}.json"
            )
            
            # Get spatial analysis summary
            if hasattr(self.spatial_analyzer, 'get_analysis_summary'):
                summary = self.spatial_analyzer.get_analysis_summary()
                
                import json
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating spatial report: {e}")
        
        return ""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all integrated systems"""
        
        status = {
            "systems_initialized": self.systems_initialized,
            "active_systems": self.active_systems,
            "system_details": {}
        }
        
        # Check each system
        for system_name in self.active_systems:
            system_obj = getattr(self, system_name, None)
            if system_obj:
                status["system_details"][system_name] = {
                    "initialized": True,
                    "type": type(system_obj).__name__
                }
                
                # Add system-specific status if available
                if hasattr(system_obj, 'get_status'):
                    status["system_details"][system_name]["status"] = system_obj.get_status()
        
        return status
    
    def cleanup_systems(self) -> None:
        """Cleanup and finalize all systems"""
        
        try:
            # Generate final reports
            final_reports = self.generate_reports(0, final_report=True)
            self.logger.info(f"Final reports generated: {list(final_reports.keys())}")
            
            # Cleanup individual systems
            if self.data_collector and hasattr(self.data_collector, 'finalize'):
                self.data_collector.finalize()
            
            if self.mortality_tracker and hasattr(self.mortality_tracker, 'finalize'):
                self.mortality_tracker.finalize()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during system cleanup: {e}")