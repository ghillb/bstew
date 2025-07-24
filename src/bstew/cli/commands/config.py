"""
Configuration management command implementation
==============================================

Handles configuration creation, validation, display, and management.
"""

from typing import Any, Optional, List, Dict
from pathlib import Path
from rich.syntax import Syntax
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table

from ..core.base import BaseCLICommand
from ..types import CLIResult
from ...core.bee_species_config import BeeSpeciesType, BeeSpeciesManager


class ConfigCommand(BaseCLICommand):
    """Command for configuration management"""

    def execute(
        self,
        action: str,
        name: Optional[str] = None,
        template: str = "basic",
        output: Optional[str] = None,
        species: Optional[str] = None,
        **kwargs: Any,
    ) -> CLIResult:
        """Execute configuration management action"""

        try:
            if action == "create":
                return self._create_config(name, template, output, species)
            elif action == "validate":
                return self._validate_config(name)
            elif action == "show":
                return self._show_config(name)
            elif action == "list":
                return self._list_configs()
            elif action == "diff":
                return self._diff_configs(name, template)
            elif action == "species":
                return self._show_species_info(species)
            else:
                valid_actions = [
                    "create",
                    "validate",
                    "show",
                    "list",
                    "diff",
                    "species",
                ]
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

        valid_actions = ["create", "validate", "show", "list", "diff", "species"]
        if action not in valid_actions:
            errors.append(f"Action must be one of {valid_actions}")

        # Action-specific validation
        if action == "create":
            name = kwargs.get("name")
            if not name:
                errors.append(
                    "Configuration name is required for create action. "
                    "Usage: bstew config create <name> --template <template>"
                )
            # Only validate template if name is provided (avoid confusing double errors)
            elif name:
                template = kwargs.get("template", "basic")
                if not self._is_valid_template(template):
                    valid_templates = [
                        "basic",
                        "honeybee",
                        "bumblebee",
                        "minimal",
                        "research",
                    ]
                    errors.append(
                        f"Invalid template '{template}'. Available templates: {', '.join(valid_templates)}"
                    )

        elif action == "validate":
            name = kwargs.get("name")
            if not name:
                errors.append("Configuration file path is required for validate action")
            elif not Path(name).exists():
                errors.append(f"Configuration file not found: {name}")

        elif action == "diff":
            name = kwargs.get("name")  # First config file
            template = kwargs.get(
                "template"
            )  # Second config file (reusing template param)
            if not name:
                errors.append("First configuration file is required for diff action")
            elif not template:
                errors.append("Second configuration file is required for diff action")
            else:
                if not Path(name).exists():
                    errors.append(f"First configuration file not found: {name}")
                if not Path(template).exists():
                    errors.append(f"Second configuration file not found: {template}")

        return errors

    def _create_config(
        self,
        name: Optional[str],
        template: str,
        output: Optional[str],
        species: Optional[str] = None,
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

    def _diff_configs(
        self,
        config1_path: Optional[str],
        config2_path: Optional[str],
    ) -> CLIResult:
        """Compare two configuration files and show differences"""

        if not config1_path or not config2_path:
            return CLIResult(
                success=False,
                message="Two configuration files are required for diff",
                exit_code=1,
            )

        try:
            # Load both configurations as dictionaries
            import yaml

            with open(config1_path) as f:
                config1 = yaml.safe_load(f)
            with open(config2_path) as f:
                config2 = yaml.safe_load(f)

            # Perform recursive diff
            differences = self._recursive_diff(config1, config2)

            if not differences:
                self.context.print_success("Configuration files are identical")
                return CLIResult(
                    success=True,
                    message="Configuration files are identical",
                    data={"differences": []},
                )

            # Display differences with Rich formatting
            from rich.table import Table
            from rich.text import Text

            self.console.print(
                f"\n[bold]Configuration Diff:[/bold] {config1_path} vs {config2_path}\n"
            )

            # Create a table for differences
            table = Table(show_header=True, header_style="bold")
            table.add_column("Path", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Value 1", style="yellow")
            table.add_column("Value 2", style="green")

            for diff in differences:
                status_color = {
                    "added": "green",
                    "removed": "red",
                    "modified": "yellow",
                }[diff["status"]]

                status_text = Text(diff["status"].upper(), style=status_color)

                value1 = (
                    str(diff.get("value1", ""))
                    if diff.get("value1") is not None
                    else ""
                )
                value2 = (
                    str(diff.get("value2", ""))
                    if diff.get("value2") is not None
                    else ""
                )

                table.add_row(
                    diff["path"],
                    status_text,
                    value1,
                    value2,
                )

            self.console.print(table)

            # Summary
            summary = {
                "added": len([d for d in differences if d["status"] == "added"]),
                "removed": len([d for d in differences if d["status"] == "removed"]),
                "modified": len([d for d in differences if d["status"] == "modified"]),
            }

            self.console.print("\n[bold]Summary:[/bold]")
            self.console.print(f"  [green]Added:[/green] {summary['added']}")
            self.console.print(f"  [red]Removed:[/red] {summary['removed']}")
            self.console.print(f"  [yellow]Modified:[/yellow] {summary['modified']}")

            return CLIResult(
                success=True,
                message="Configuration diff completed",
                data={"differences": differences, "summary": summary},
            )

        except Exception as e:
            return self.handle_exception(e, "Configuration diff")

    def _recursive_diff(
        self,
        obj1: Any,
        obj2: Any,
        path: str = "",
    ) -> List[Dict[str, Any]]:
        """Recursively compare two objects and return differences"""

        differences = []

        if type(obj1) is not type(obj2):
            differences.append(
                {
                    "path": path or "root",
                    "status": "modified",
                    "value1": obj1,
                    "value2": obj2,
                }
            )
            return differences

        if isinstance(obj1, dict):
            # Compare dictionaries
            all_keys = set(obj1.keys()) | set(obj2.keys())

            for key in all_keys:
                new_path = f"{path}.{key}" if path else key

                if key not in obj1:
                    differences.append(
                        {
                            "path": new_path,
                            "status": "added",
                            "value1": None,
                            "value2": obj2[key],
                        }
                    )
                elif key not in obj2:
                    differences.append(
                        {
                            "path": new_path,
                            "status": "removed",
                            "value1": obj1[key],
                            "value2": None,
                        }
                    )
                else:
                    # Recursively compare values
                    sub_diffs = self._recursive_diff(obj1[key], obj2[key], new_path)
                    differences.extend(sub_diffs)

        elif isinstance(obj1, list):
            # Compare lists
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                new_path = f"{path}[{i}]"
                sub_diffs = self._recursive_diff(item1, item2, new_path)
                differences.extend(sub_diffs)

            # Handle different list lengths
            if len(obj1) > len(obj2):
                for i in range(len(obj2), len(obj1)):
                    differences.append(
                        {
                            "path": f"{path}[{i}]",
                            "status": "removed",
                            "value1": obj1[i],
                            "value2": None,
                        }
                    )
            elif len(obj2) > len(obj1):
                for i in range(len(obj1), len(obj2)):
                    differences.append(
                        {
                            "path": f"{path}[{i}]",
                            "status": "added",
                            "value1": None,
                            "value2": obj2[i],
                        }
                    )

        else:
            # Compare scalar values
            if obj1 != obj2:
                differences.append(
                    {
                        "path": path or "root",
                        "status": "modified",
                        "value1": obj1,
                        "value2": obj2,
                    }
                )

        return differences

    def _show_species_info(self, species: Optional[str] = None) -> CLIResult:
        """Show information about available species and their characteristics"""

        try:
            species_manager = BeeSpeciesManager()

            if species:
                # Show info for specific species
                try:
                    species_type = BeeSpeciesType[species.upper()]
                    config = species_manager.get_species_config(species_type)

                    # Create table for species details
                    table = Table(title=f"Species Information: {species_type.value}")
                    table.add_column("Property", style="cyan")
                    table.add_column("Value", style="yellow")

                    table.add_row("Scientific Name", config.scientific_name)
                    table.add_row("Common Name", config.common_name)
                    table.add_row(
                        "Communication Type", config.communication_type.value.title()
                    )
                    table.add_row(
                        "Typical Colony Size", f"{config.typical_colony_size:,}"
                    )
                    table.add_row(
                        "Max Foraging Range", f"{config.max_foraging_range_m:,}m"
                    )
                    table.add_row(
                        "Foraging Season", f"{config.foraging_season_months} months"
                    )
                    table.add_row(
                        "Social Structure",
                        config.social_structure.value.title().replace("_", " "),
                    )

                    if hasattr(config, "optimal_temperature_range"):
                        table.add_row(
                            "Temperature Range",
                            f"{config.optimal_temperature_range[0]}°C - {config.optimal_temperature_range[1]}°C",
                        )

                    self.console.print(table)

                    # Show example config path
                    config_path = f"configs/species/{species.lower()}.yaml"
                    self.context.print_info(f"\nExample configuration: {config_path}")

                    return CLIResult(
                        success=True,
                        message=f"Species information displayed for {species_type.value}",
                        data={"species": species_type.value, "config": config},
                    )

                except KeyError:
                    available_species = [
                        s.value for s in species_manager.get_available_species()
                    ]
                    self.context.print_error(f"Unknown species: {species}")
                    self.context.print_info(
                        f"Available species: {', '.join(available_species)}"
                    )
                    return CLIResult(
                        success=False,
                        message=f"Unknown species: {species}",
                        exit_code=1,
                    )
            else:
                # Show all available species
                available_species = species_manager.get_available_species()

                # Create overview table
                table = Table(title="Available Bee Species")
                table.add_column("Species", style="cyan")
                table.add_column("Common Name", style="yellow")
                table.add_column("Communication", style="green")
                table.add_column("Colony Size", style="magenta", justify="right")
                table.add_column("Foraging Range", style="blue", justify="right")

                for species_type in available_species:
                    config = species_manager.get_species_config(species_type)
                    table.add_row(
                        species_type.value,
                        config.common_name,
                        config.communication_type.value.title(),
                        f"{config.typical_colony_size:,}",
                        f"{config.max_foraging_range_m:,}m",
                    )

                self.console.print(table)

                # Show configuration examples
                self.context.print_info("\nConfiguration examples:")
                self.context.print_info(
                    "  configs/species/honey_bee.yaml - Honey bee optimized"
                )
                self.context.print_info(
                    "  configs/species/bumblebee_terrestris.yaml - Bumblebee optimized"
                )
                self.context.print_info(
                    "  configs/species/multi_species.yaml - Multi-species study"
                )

                self.context.print_info(
                    "\nUse 'bstew config species <SPECIES_NAME>' for detailed info"
                )

                return CLIResult(
                    success=True,
                    message="Species overview displayed",
                    data={"available_species": [s.value for s in available_species]},
                )

        except Exception as e:
            return self.handle_exception(e, "Species information")
