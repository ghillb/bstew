"""
Configuration management command implementation
==============================================

Handles configuration creation, validation, display, and management.
"""

from typing import Any, Optional, List
from pathlib import Path
from rich.syntax import Syntax
from rich.panel import Panel
from rich.tree import Tree

from ..core.base import BaseCLICommand
from ..types import CLIResult


class ConfigCommand(BaseCLICommand):
    """Command for configuration management"""
    
    def execute(
        self,
        action: str,
        name: Optional[str] = None,
        template: str = "basic",
        output: Optional[str] = None,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute configuration management action"""
        
        try:
            if action == "create":
                return self._create_config(name, template, output)
            elif action == "validate":
                return self._validate_config(name)
            elif action == "show":
                return self._show_config(name)
            elif action == "list":
                return self._list_configs()
            else:
                valid_actions = ["create", "validate", "show", "list"]
                self.context.print_error(f"Unknown action: {action}")
                self.context.print_info(f"Valid actions: {', '.join(valid_actions)}")
                return CLIResult(
                    success=False,
                    message=f"Unknown action: {action}",
                    exit_code=1,
                )
                
        except Exception as e:
            return self.handle_exception(e, f"Configuration {action}")
    
    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate command inputs"""
        errors = []
        
        action = kwargs.get("action")
        if not action:
            errors.append("Action is required")
            return errors
        
        valid_actions = ["create", "validate", "show", "list"]
        if action not in valid_actions:
            errors.append(f"Action must be one of {valid_actions}")
        
        # Action-specific validation
        if action == "create":
            name = kwargs.get("name")
            if not name:
                errors.append("Configuration name is required for create action")
            
            template = kwargs.get("template", "basic")
            if not self._is_valid_template(template):
                errors.append(f"Invalid template: {template}")
        
        elif action == "validate":
            name = kwargs.get("name")
            if not name:
                errors.append("Configuration file path is required for validate action")
            elif not Path(name).exists():
                errors.append(f"Configuration file not found: {name}")
        
        return errors
    
    def _create_config(
        self,
        name: Optional[str],
        template: str,
        output: Optional[str],
    ) -> CLIResult:
        """Create a new configuration file"""
        
        if not name:
            return CLIResult(
                success=False,
                message="Configuration name is required for create action",
                exit_code=1,
            )
        
        try:
            # Get template configuration
            config = self.config_manager.get_config_template(template)
            
            # Determine output path
            output_path = output or f"configs/{name}.yaml"
            
            # Save configuration
            self.config_manager.save_config(config, output_path)
            
            self.context.print_success(f"Created configuration: {output_path}")
            
            return CLIResult(
                success=True,
                message=f"Configuration created: {output_path}",
                data={"output_path": output_path, "template": template},
            )
            
        except Exception as e:
            self.context.print_error(f"Failed to create configuration: {e}")
            return CLIResult(
                success=False,
                message=f"Failed to create configuration: {e}",
                exit_code=1,
            )
    
    def _validate_config(self, config_path: Optional[str]) -> CLIResult:
        """Validate a configuration file"""
        
        if not config_path:
            return CLIResult(
                success=False,
                message="Configuration file path is required for validate action",
                exit_code=1,
            )
        
        try:
            # Load configuration
            config = self.config_manager.load_config(config_path)
            
            # Validate configuration
            errors = self.config_manager.validate_config(config)
            
            if errors:
                self.context.print_error("Configuration validation failed:")
                for error in errors:
                    self.context.print_error(f"  • {error}")
                
                return CLIResult(
                    success=False,
                    message="Configuration validation failed",
                    data={"errors": errors},
                    exit_code=1,
                )
            
            self.context.print_success("Configuration is valid")
            
            return CLIResult(
                success=True,
                message="Configuration is valid",
                data={"config_path": config_path},
            )
            
        except Exception as e:
            self.context.print_error(f"Validation error: {e}")
            return CLIResult(
                success=False,
                message=f"Validation error: {e}",
                exit_code=1,
            )
    
    def _show_config(self, config_path: Optional[str]) -> CLIResult:
        """Display configuration file with syntax highlighting"""
        
        # Use default if no path provided
        if not config_path:
            config_path = "configs/default.yaml"
        
        try:
            # Read configuration file
            path = Path(config_path)
            if not path.exists():
                self.context.print_error(f"Configuration file not found: {config_path}")
                return CLIResult(
                    success=False,
                    message=f"Configuration file not found: {config_path}",
                    exit_code=1,
                )
            
            content = path.read_text()
            
            # Display with syntax highlighting
            syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
            panel = Panel(syntax, title=f"Configuration: {config_path}")
            self.console.print(panel)
            
            return CLIResult(
                success=True,
                message=f"Displayed configuration: {config_path}",
                data={"config_path": config_path, "content": content},
            )
            
        except Exception as e:
            self.context.print_error(f"Failed to read configuration: {e}")
            return CLIResult(
                success=False,
                message=f"Failed to read configuration: {e}",
                exit_code=1,
            )
    
    def _list_configs(self) -> CLIResult:
        """List available configurations"""
        
        try:
            configs = self.config_manager.list_available_configs()
            
            # Create tree display
            tree = Tree("📁 Available Configurations")
            
            if configs.get("scenarios"):
                scenarios_branch = tree.add("🎯 Scenarios")
                for scenario in configs["scenarios"]:
                    scenarios_branch.add(scenario)
            
            if configs.get("species"):
                species_branch = tree.add("🐝 Species")
                for species in configs["species"]:
                    species_branch.add(species)
            
            if configs.get("templates"):
                templates_branch = tree.add("📋 Templates")
                for template in configs["templates"]:
                    templates_branch.add(template)
            
            # Show basic configs in current directory
            configs_dir = Path("configs")
            if configs_dir.exists():
                basic_branch = tree.add("📄 Current Directory")
                for config_file in configs_dir.glob("*.yaml"):
                    basic_branch.add(config_file.name)
            
            self.console.print(tree)
            
            return CLIResult(
                success=True,
                message="Listed available configurations",
                data={"configs": configs},
            )
            
        except Exception as e:
            self.context.print_error(f"Failed to list configurations: {e}")
            return CLIResult(
                success=False,
                message=f"Failed to list configurations: {e}",
                exit_code=1,
            )
    
    def _is_valid_template(self, template: str) -> bool:
        """Check if template is valid"""
        valid_templates = ["basic", "honeybee", "bumblebee", "minimal", "research"]
        return template in valid_templates