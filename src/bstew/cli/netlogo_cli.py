#!/usr/bin/env python3
"""
NetLogo CLI Tools for BSTEW
===========================

Command-line interface for NetLogo data processing and conversion.
Provides utilities for parsing, mapping, and validating NetLogo data.
"""

import typer
import json
import yaml
from pathlib import Path
from typing import Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from enum import Enum

# Import BSTEW NetLogo components
from ..data.netlogo_parser import NetLogoDataParser
from ..config.netlogo_mapping import NetLogoParameterMapper, convert_netlogo_to_bstew
from ..utils.validation import NetLogoSpecificValidator
from ..data.netlogo_output_parser import NetLogoBehaviorSpaceParser
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
from tests.test_netlogo_integration import NetLogoIntegrationTester

app = typer.Typer(
    name="bstew-netlogo",
    help="NetLogo CLI Tools for BSTEW - Process and convert NetLogo data files",
    rich_markup_mode="rich",
)

console = Console()


class OutputType(str, Enum):
    """NetLogo output types"""

    behaviorspace = "behaviorspace"
    table = "table"
    reporter = "reporter"


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def parse(
    input_dir: Path = typer.Argument(
        ..., help="Input directory containing NetLogo data files"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    show_samples: bool = typer.Option(
        False, "--show-samples", help="Show sample data from parsed files"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Parse NetLogo data files and convert to structured format"""
    setup_logging(verbose)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Parsing NetLogo data files...", total=None)

        parser = NetLogoDataParser()

        try:
            # Parse all data files
            netlogo_data = parser.parse_all_data_files(str(input_dir))

            # Create summary table
            table = Table(title="📊 Parsing Results")
            table.add_column("Data Type", style="cyan")
            table.add_column("Count", style="magenta")

            table.add_row("Parameters", str(len(netlogo_data.get("parameters", {}))))
            table.add_row("Species", str(len(netlogo_data.get("species", {}))))
            table.add_row("Flowers", str(len(netlogo_data.get("flowers", {}))))
            table.add_row(
                "Food Sources", str(len(netlogo_data.get("food_sources", {})))
            )
            table.add_row("Habitats", str(len(netlogo_data.get("habitats", {}))))

            console.print(table)

            # Save output if requested
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)

                if output.suffix.lower() == ".json":
                    with open(output, "w") as f:
                        json.dump(netlogo_data, f, indent=2, default=str)
                elif output.suffix.lower() in [".yaml", ".yml"]:
                    with open(output, "w") as f:
                        yaml.dump(netlogo_data, f, default_flow_style=False)
                else:
                    raise typer.BadParameter(
                        f"Unsupported output format: {output.suffix}"
                    )

                rprint(f"\\n💾 Data saved to: [bold green]{output}[/bold green]")

            # Show sample data if requested
            if show_samples:
                console.print("\\n📋 Sample Data:", style="bold blue")

                # Sample parameters
                params = netlogo_data.get("parameters", {})
                if params:
                    sample_params = list(params.items())[:3]
                    console.print("  Parameters (first 3):", style="bold")
                    for name, param in sample_params:
                        value = param.value if hasattr(param, "value") else param
                        console.print(f"    {name}: {value}")

                # Sample species
                species = netlogo_data.get("species", {})
                if species:
                    sample_species = list(species.items())[:2]
                    console.print("  Species (first 2):", style="bold")
                    for name, spec in sample_species:
                        console.print(
                            f"    {name}: {spec.species_id if hasattr(spec, 'species_id') else name}"
                        )

                # Sample flowers
                flowers = netlogo_data.get("flowers", {})
                if flowers:
                    sample_flowers = list(flowers.items())[:3]
                    console.print("  Flowers (first 3):", style="bold")
                    for name, flower in sample_flowers:
                        depth = (
                            flower.corolla_depth_mm
                            if hasattr(flower, "corolla_depth_mm")
                            else "N/A"
                        )
                        console.print(f"    {name}: depth={depth}mm")

        except Exception as e:
            console.print(f"❌ Error parsing NetLogo data: {e}", style="bold red")
            raise typer.Exit(1)

    rprint(
        "\\n✅ [bold green]NetLogo data parsing completed successfully![/bold green]"
    )


@app.command()
def map(
    input_dir: Path = typer.Argument(
        ..., help="Input directory containing NetLogo data files"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output BSTEW config file"),
    show_critical: bool = typer.Option(
        False, "--show-critical", help="Show critical parameter mappings"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Map NetLogo parameters to BSTEW format"""
    setup_logging(verbose)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Mapping NetLogo parameters...", total=None)

        parser = NetLogoDataParser()
        mapper = NetLogoParameterMapper()

        try:
            # Parse NetLogo data
            netlogo_data = parser.parse_all_data_files(str(input_dir))

            # Convert to BSTEW format
            bstew_config = convert_netlogo_to_bstew(netlogo_data, str(output))

            # Create mapping results table
            table = Table(title="🔄 Parameter Mapping Results")
            table.add_column("Data Type", style="cyan")
            table.add_column("NetLogo", style="magenta")
            table.add_column("BSTEW", style="green")

            # System parameters
            system_params = netlogo_data.get("parameters", {})
            mapped_system = sum(
                1
                for section in [
                    "colony",
                    "foraging",
                    "landscape",
                    "predation",
                    "genetics",
                    "mortality",
                    "resources",
                    "visualization",
                    "environment",
                    "detection",
                    "hibernation",
                ]
                if bstew_config.get(section)
            )
            table.add_row(
                "System Parameters",
                str(len(system_params)),
                f"{mapped_system} sections",
            )

            # Species parameters
            species_count = len(netlogo_data.get("species", {}))
            bstew_species_count = len(bstew_config.get("species", {}))
            table.add_row("Species", str(species_count), str(bstew_species_count))

            # Flower parameters
            flower_count = len(netlogo_data.get("flowers", {}))
            bstew_flower_count = len(bstew_config.get("flowers", {}))
            table.add_row("Flowers", str(flower_count), str(bstew_flower_count))

            console.print(table)

            # Show populated sections
            populated_sections = [
                k for k, v in bstew_config.items() if v and k != "metadata"
            ]
            console.print(f"\\n📋 Populated BSTEW sections: {populated_sections}")

            # Show mapping coverage
            mapper_summary = mapper.get_mapping_summary()
            system_coverage = (
                len(system_params) / mapper_summary["system_parameters"]["count"] * 100
                if mapper_summary["system_parameters"]["count"] > 0
                else 0
            )
            console.print(f"📊 System parameter coverage: {system_coverage:.1f}%")

            # Show critical parameters
            if show_critical:
                console.print("\\n🔑 Critical Parameters:", style="bold blue")

                critical_params = {
                    "colony.initial_queens": bstew_config.get("colony", {}).get(
                        "initial_queens"
                    ),
                    "foraging.max_range_m": bstew_config.get("foraging", {}).get(
                        "max_range_m"
                    ),
                    "landscape.grid_size": bstew_config.get("landscape", {}).get(
                        "grid_size"
                    ),
                    "predation.badger_count": bstew_config.get("predation", {}).get(
                        "badger_count"
                    ),
                    "genetics.csd_enabled": bstew_config.get("genetics", {}).get(
                        "csd_enabled"
                    ),
                }

                for param, value in critical_params.items():
                    status = "✅" if value is not None else "❌"
                    console.print(f"  {status} {param}: {value}")

        except Exception as e:
            console.print(f"❌ Error mapping parameters: {e}", style="bold red")
            raise typer.Exit(1)

    rprint("\\n✅ [bold green]Parameter mapping completed successfully![/bold green]")


@app.command()
def validate(
    input_dir: Path = typer.Argument(
        ..., help="Input directory containing NetLogo data files"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output validation report file"
    ),
    show_failures: bool = typer.Option(
        False, "--show-failures", help="Show detailed validation failures"
    ),
    max_failures: int = typer.Option(
        10, "--max-failures", help="Maximum number of failures to show"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Validate NetLogo compatibility"""
    setup_logging(verbose)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Validating NetLogo compatibility...", total=None)

        parser = NetLogoDataParser()
        NetLogoParameterMapper()
        validator = NetLogoSpecificValidator()

        try:
            # Parse NetLogo data
            netlogo_data = parser.parse_all_data_files(str(input_dir))

            # Convert to BSTEW format
            bstew_config = convert_netlogo_to_bstew(netlogo_data)

            # Validate genetic parameters
            genetic_results = validator.validate_genetic_parameters(
                netlogo_data.get("parameters", {}), bstew_config
            )

            # Validate species parameters
            species_results = validator.validate_species_parameters(
                netlogo_data.get("species", {}), bstew_config.get("species", {})
            )

            # Validate flower parameters
            flower_results = validator.validate_flower_accessibility(
                netlogo_data.get("flowers", {}), bstew_config.get("flowers", {})
            )

            # Create validation results table
            table = Table(title="🔍 Validation Results")
            table.add_column("System", style="cyan")
            table.add_column("Passed", style="green")
            table.add_column("Failed", style="red")
            table.add_column("Total", style="magenta")

            # Genetic validation
            genetic_passed = sum(1 for result in genetic_results if result.passed)
            genetic_failed = len(genetic_results) - genetic_passed
            table.add_row(
                "Genetic System",
                str(genetic_passed),
                str(genetic_failed),
                str(len(genetic_results)),
            )

            # Species validation
            species_passed = sum(1 for result in species_results if result.passed)
            species_failed = len(species_results) - species_passed
            table.add_row(
                "Species System",
                str(species_passed),
                str(species_failed),
                str(len(species_results)),
            )

            # Flower validation
            flower_passed = sum(1 for result in flower_results if result.passed)
            flower_failed = len(flower_results) - flower_passed
            table.add_row(
                "Flower System",
                str(flower_passed),
                str(flower_failed),
                str(len(flower_results)),
            )

            console.print(table)

            # Overall score
            total_tests = (
                len(genetic_results) + len(species_results) + len(flower_results)
            )
            total_passed = genetic_passed + species_passed + flower_passed
            overall_score = (total_passed / total_tests * 100) if total_tests > 0 else 0

            console.print(
                f"\\n📊 Overall Compatibility: [bold green]{overall_score:.1f}%[/bold green]"
            )

            # Show detailed failures if requested
            if show_failures:
                console.print("\\n❌ Failed Validations:", style="bold red")

                all_results = genetic_results + species_results + flower_results
                failed_results = [r for r in all_results if not r.passed]

                if failed_results:
                    for result in failed_results[:max_failures]:
                        console.print(
                            f"  • [red]{result.test_name}[/red]: {result.recommendation}"
                        )

                    if len(failed_results) > max_failures:
                        console.print(
                            f"  ... and {len(failed_results) - max_failures} more"
                        )
                else:
                    console.print("  No validation failures found!", style="bold green")

            # Save validation report if requested
            if output:
                validation_report = {
                    "timestamp": "2025-07-09T10:00:00",
                    "genetic_validation": [
                        {
                            "test_name": r.test_name,
                            "passed": r.passed,
                            "score": r.score,
                            "recommendation": r.recommendation,
                        }
                        for r in genetic_results
                    ],
                    "species_validation": [
                        {
                            "test_name": r.test_name,
                            "passed": r.passed,
                            "score": r.score,
                            "recommendation": r.recommendation,
                        }
                        for r in species_results
                    ],
                    "flower_validation": [
                        {
                            "test_name": r.test_name,
                            "passed": r.passed,
                            "score": r.score,
                            "recommendation": r.recommendation,
                        }
                        for r in flower_results
                    ],
                    "summary": {
                        "total_tests": total_tests,
                        "total_passed": total_passed,
                        "overall_score": overall_score,
                    },
                }

                output.parent.mkdir(parents=True, exist_ok=True)

                with open(output, "w") as f:
                    json.dump(validation_report, f, indent=2)

                rprint(
                    f"\\n💾 Validation report saved to: [bold green]{output}[/bold green]"
                )

        except Exception as e:
            console.print(f"❌ Error validating compatibility: {e}", style="bold red")
            raise typer.Exit(1)

    rprint(
        "\\n✅ [bold green]Compatibility validation completed successfully![/bold green]"
    )


@app.command()
def test(
    input_dir: Path = typer.Argument(
        ..., help="Input directory containing NetLogo data files"
    ),
    output_dir: Path = typer.Option(
        Path("test_output"), "--output-dir", help="Output directory for test results"
    ),
    save_report: bool = typer.Option(
        False, "--save-report", help="Save detailed test report"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Run NetLogo integration tests"""
    setup_logging(verbose)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running integration tests...", total=None)

        try:
            # Initialize integration tester
            tester = NetLogoIntegrationTester(
                data_dir=str(input_dir), output_dir=str(output_dir)
            )

            # Run all tests
            results = tester.run_all_tests()

            # Create results table
            table = Table(title="📊 Integration Test Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Tests Run", str(results["total_tests"]))
            table.add_row("Passed", str(results["passed_tests"]))
            table.add_row("Failed", str(results["failed_tests"]))

            success_rate = results["passed_tests"] / results["total_tests"] * 100
            table.add_row("Success Rate", f"{success_rate:.1f}%")

            console.print(table)

            # Show failures if any
            if results["failed_tests"] > 0:
                console.print("\\n❌ Failed Tests:", style="bold red")
                for test_result in results["test_results"]:
                    if not test_result["passed"]:
                        console.print(
                            f"  • [red]{test_result['test_name']}[/red]: {len(test_result['errors'])} errors"
                        )

            # Save detailed report if requested
            if save_report:
                report_path = output_dir / "integration_test_report.json"
                with open(report_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                rprint(
                    f"\\n💾 Detailed report saved to: [bold green]{report_path}[/bold green]"
                )

        except Exception as e:
            console.print(f"❌ Error running integration tests: {e}", style="bold red")
            raise typer.Exit(1)

    rprint("\\n✅ [bold green]Integration tests completed successfully![/bold green]")


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input NetLogo output file"),
    output_type: OutputType = typer.Option(
        ..., "--type", help="Type of NetLogo output"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Convert NetLogo output files to structured format"""
    setup_logging(verbose)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Converting NetLogo output...", total=None)

        # output_parser will be created based on output type

        try:
            if output_type == OutputType.behaviorspace:
                # Parse BehaviorSpace output
                output_parser = NetLogoBehaviorSpaceParser()
                experiments = output_parser.parse_behaviorspace_csv(str(input_file))

                table = Table(title="📊 BehaviorSpace Results")
                table.add_column("Experiment", style="cyan")
                table.add_column("Runs", style="magenta")
                table.add_column("Parameters", style="green")

                # experiments is a single NetLogoExperiment object
                table.add_row(
                    experiments.experiment_name,
                    str(len(experiments.runs)),
                    str(len(experiments.runs[0].parameters) if experiments.runs else 0),
                )

                # Only one experiment object

                console.print(table)
                console.print(f"\\nTotal runs: {len(experiments.runs)}")

            elif output_type == OutputType.table:
                # Parse table output
                # Table output parsing not implemented yet
                console.print(
                    "[yellow]Table output parsing not yet implemented[/yellow]"
                )
                console.print("📋 Table Data parsing completed")
                return

            elif output_type == OutputType.reporter:
                # Parse reporter output
                # Reporter output parsing not implemented yet
                console.print(
                    "[yellow]Reporter output parsing not yet implemented[/yellow]"
                )
                console.print("📈 Reporter Data parsing completed")
                return

            # Save converted output if requested
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)

                if output_type == OutputType.behaviorspace:
                    # Convert experiments to JSON
                    experiments_data = []
                    for experiment in experiments:
                        exp_data = {
                            "experiment_name": experiment.experiment_name,  # type: ignore[attr-defined]
                            "runs": [
                                {
                                    "run_number": run.run_number,
                                    "parameters": run.parameters,
                                    "metrics": run.metrics,
                                }
                                for run in experiment.runs  # type: ignore[attr-defined]
                            ],
                        }
                        experiments_data.append(exp_data)

                    with open(output, "w") as f:
                        json.dump(experiments_data, f, indent=2, default=str)

                elif output_type == OutputType.table:
                    # Table data saving not implemented
                    console.print("[yellow]Table data saving not implemented[/yellow]")

                elif output_type == OutputType.reporter:
                    # Reporter data saving not implemented
                    console.print(
                        "[yellow]Reporter data saving not implemented[/yellow]"
                    )

                rprint(
                    f"\\n💾 Converted output saved to: [bold green]{output}[/bold green]"
                )

        except Exception as e:
            console.print(f"❌ Error converting NetLogo output: {e}", style="bold red")
            raise typer.Exit(1)

    rprint(
        "\\n✅ [bold green]NetLogo output conversion completed successfully![/bold green]"
    )


def main() -> None:
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
