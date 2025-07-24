"""
Benchmark CLI Commands for BSTEW
================================

CLI commands for running performance benchmarks and validation tests.
"""

import typer
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from ...benchmarks.benchmark_runner import BenchmarkRunner
from ...benchmarks.netlogo_parity_benchmarks import NetLogoParityBenchmarks

# Initialize console for rich output
console = Console()

# Create Typer app for benchmark commands
app = typer.Typer(
    name="benchmark",
    help="Performance benchmarking and validation commands",
    add_completion=False,
)


@app.command()
def run(
    output_dir: Path = typer.Option(
        "artifacts/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        "-q",
        help="Run quick validation instead of full benchmark suite",
    ),
    netlogo_only: bool = typer.Option(
        False, "--netlogo-only", help="Run only NetLogo parity benchmarks"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Run BSTEW performance benchmarks and validation tests"""

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    console.print("🚀 [bold blue]Starting BSTEW Benchmark Suite[/bold blue]")

    try:
        # Initialize benchmark runner
        runner = BenchmarkRunner(str(output_dir))

        if quick:
            console.print("⚡ Running quick validation suite...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running quick validation...", total=None)
                results = runner.run_quick_validation()
                progress.update(task, completed=True)

            _display_quick_results(results)

        elif netlogo_only:
            console.print("🧪 Running NetLogo parity benchmarks...")
            netlogo_benchmarks = NetLogoParityBenchmarks(str(output_dir))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running NetLogo benchmarks...", total=None)
                results = netlogo_benchmarks.run_complete_benchmark_suite()
                progress.update(task, completed=True)

            _display_netlogo_results(results)

        else:
            console.print("🔬 Running complete validation suite...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running complete validation...", total=None)
                results = runner.run_complete_validation_suite()
                progress.update(task, completed=True)

            _display_complete_results(results)

        console.print(f"✅ [green]Benchmark results saved to {output_dir}[/green]")

    except Exception as e:
        console.print(f"❌ [red]Benchmark execution failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def compare(
    baseline_file: Path = typer.Argument(
        ..., help="Baseline benchmark results file (JSON)"
    ),
    current_file: Optional[Path] = typer.Option(
        None,
        "--current",
        help="Current benchmark results file (if not provided, runs new benchmark)",
    ),
    output_dir: Path = typer.Option(
        "artifacts/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for comparison results",
    ),
    threshold: float = typer.Option(
        0.1, "--threshold", "-t", help="Performance regression threshold (default: 10%)"
    ),
) -> None:
    """Compare benchmark results for performance regression analysis"""

    console.print("📊 [bold blue]Running Benchmark Comparison[/bold blue]")

    try:
        # Load baseline results
        if not baseline_file.exists():
            console.print(f"❌ [red]Baseline file not found: {baseline_file}[/red]")
            raise typer.Exit(1)

        with open(baseline_file, "r") as f:
            baseline_results = json.load(f)

        console.print(f"📁 Loaded baseline from {baseline_file}")

        # Get current results
        if current_file:
            if not current_file.exists():
                console.print(
                    f"❌ [red]Current results file not found: {current_file}[/red]"
                )
                raise typer.Exit(1)

            with open(current_file, "r") as f:
                current_results = json.load(f)

            console.print(f"📁 Loaded current results from {current_file}")
        else:
            console.print("🚀 Running new benchmark for comparison...")
            runner = BenchmarkRunner(str(output_dir))
            validation_results = runner.run_complete_validation_suite()

            # Extract benchmark results
            current_results = validation_results.get("performance_benchmarks", {}).get(
                "results", {}
            )

        # Perform comparison
        comparison_results = _compare_benchmark_results(
            baseline_results, current_results, threshold
        )

        # Display comparison
        _display_comparison_results(comparison_results, threshold)

        # Save comparison results
        from datetime import datetime

        comparison_file = (
            output_dir
            / f"benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(comparison_file, "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)

        console.print(f"💾 Comparison results saved to {comparison_file}")

    except Exception as e:
        console.print(f"❌ [red]Benchmark comparison failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def profile(
    scenario: str = typer.Option(
        "medium_colony",
        "--scenario",
        "-s",
        help="Profiling scenario: small_colony, medium_colony, large_colony",
    ),
    steps: int = typer.Option(
        100, "--steps", help="Number of simulation steps to profile"
    ),
    output_dir: Path = typer.Option(
        "artifacts/profiling",
        "--output-dir",
        "-o",
        help="Output directory for profiling results",
    ),
) -> None:
    """Run detailed performance profiling of BSTEW simulation"""

    console.print(
        f"🔍 [bold blue]Starting Performance Profiling: {scenario}[/bold blue]"
    )

    try:
        from ...utils.performance import PerformanceProfiler

        # Define scenarios
        scenarios: Dict[str, Dict[str, Any]] = {
            "small_colony": {
                "simulation": {"steps": steps, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 70,
                        "foragers": 20,
                        "drones": 5,
                        "brood": 5,
                    }
                },
                "environment": {"patches": 20, "flower_density": 0.3},
            },
            "medium_colony": {
                "simulation": {"steps": steps, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 350,
                        "foragers": 100,
                        "drones": 25,
                        "brood": 25,
                    }
                },
                "environment": {"patches": 50, "flower_density": 0.3},
            },
            "large_colony": {
                "simulation": {"steps": steps, "random_seed": 42},
                "colony": {
                    "initial_population": {
                        "queens": 1,
                        "workers": 700,
                        "foragers": 200,
                        "drones": 50,
                        "brood": 50,
                    }
                },
                "environment": {"patches": 100, "flower_density": 0.3},
            },
        }

        if scenario not in scenarios:
            console.print(f"❌ [red]Unknown scenario: {scenario}[/red]")
            console.print(f"Available scenarios: {', '.join(scenarios.keys())}")
            raise typer.Exit(1)

        config = scenarios[scenario]

        # Setup profiling
        profiler = PerformanceProfiler()
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(
            f"⚙️ Configuration: {config['colony']['initial_population']} bees, {steps} steps"
        )

        # Run profiled simulation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running profiled simulation...", total=None)

            profiling_results = profiler.profile_simulation(
                config=config, steps=steps, output_directory=str(output_dir)
            )

            progress.update(task, completed=True)

        # Display profiling results
        _display_profiling_results(profiling_results)

        console.print(f"✅ [green]Profiling results saved to {output_dir}[/green]")

    except Exception as e:
        console.print(f"❌ [red]Performance profiling failed: {e}[/red]")
        raise typer.Exit(1)


def _display_quick_results(results: dict) -> None:
    """Display quick validation results"""

    overall_status = results.get("overall_status", "unknown")
    status_color = "green" if overall_status == "passed" else "red"

    console.print("\n📋 [bold]Quick Validation Results[/bold]")
    console.print(
        f"Overall Status: [{status_color}]{overall_status.upper()}[/{status_color}]"
    )

    # Create results table
    table = Table(title="Component Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")

    for component, result in results.items():
        if isinstance(result, dict) and "success" in result:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            details = ""

            if "steps_per_second" in result:
                details = f"{result['steps_per_second']:.1f} steps/sec"
            elif "error" in result:
                details = (
                    result["error"][:50] + "..."
                    if len(result["error"]) > 50
                    else result["error"]
                )

            table.add_row(component.replace("_", " ").title(), status, details)

    console.print(table)


def _display_netlogo_results(results: dict) -> None:
    """Display NetLogo benchmark results"""

    console.print("\n🧪 [bold]NetLogo Parity Benchmark Results[/bold]")

    # Simulation speed results
    if "simulation_speed" in results:
        speed_table = Table(title="Simulation Speed Benchmarks")
        speed_table.add_column("Colony Size", style="cyan")
        speed_table.add_column("Steps/Second", style="white")
        speed_table.add_column("Execution Time", style="dim")
        speed_table.add_column("Memory Peak", style="dim")

        for size_name, result in results["simulation_speed"].items():
            if hasattr(result, "success") and result.success:
                speed_table.add_row(
                    size_name.replace("_", " ").title(),
                    f"{result.steps_per_second:.1f}",
                    f"{result.execution_time:.2f}s",
                    f"{result.memory_peak:.1f} MB",
                )

        console.print(speed_table)

    # NetLogo comparison
    if "netlogo_comparison" in results:
        comparison_table = Table(title="NetLogo Performance Comparison")
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("BSTEW", style="white")
        comparison_table.add_column("NetLogo Ref", style="white")
        comparison_table.add_column("Ratio", style="white")
        comparison_table.add_column("Status", style="white")

        for comparison in results["netlogo_comparison"]:
            status = "✅ PASS" if comparison.passes_benchmark else "❌ FAIL"
            comparison_table.add_row(
                comparison.metric_name.replace("_", " ").title(),
                f"{comparison.bstew_value:.1f}",
                f"{comparison.netlogo_reference:.1f}",
                f"{comparison.performance_ratio:.2f}x",
                status,
            )

        console.print(comparison_table)


def _display_complete_results(results: dict) -> None:
    """Display complete validation results"""

    summary = results.get("validation_summary", {})
    overall_status = summary.get("overall_status", "unknown")

    status_colors = {
        "excellent": "green",
        "good": "yellow",
        "needs_attention": "red",
        "unknown": "dim",
    }
    status_color = status_colors.get(overall_status, "dim")

    # Overall status panel
    status_panel = Panel(
        f"[{status_color}]{overall_status.upper()}[/{status_color}]",
        title="Overall Validation Status",
        border_style=status_color,
    )
    console.print(status_panel)

    # Component status table
    component_table = Table(title="Component Status Summary")
    component_table.add_column("Component", style="cyan")
    component_table.add_column("Status", style="white")

    components = [
        ("Performance Benchmarks", summary.get("performance_status", "unknown")),
        ("Integration Tests", summary.get("integration_status", "unknown")),
        ("End-to-End Tests", summary.get("end_to_end_status", "unknown")),
        ("Stress Tests", summary.get("stress_status", "unknown")),
    ]

    for component, status in components:
        if status in ["excellent", "good", "passed", "completed"]:
            status_display = f"✅ {status.upper()}"
        else:
            status_display = f"❌ {status.upper()}"

        component_table.add_row(component, status_display)

    console.print(component_table)

    # Recommendations
    recommendations = summary.get("recommendations", [])
    if recommendations:
        console.print("\n💡 [bold yellow]Recommendations:[/bold yellow]")
        for rec in recommendations:
            console.print(f"  • {rec}")


def _display_profiling_results(results: dict) -> None:
    """Display performance profiling results"""

    console.print("\n🔍 [bold]Performance Profiling Results[/bold]")

    # Overall metrics
    if "performance_metrics" in results:
        metrics = results["performance_metrics"]

        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")

        metric_items = [
            ("Execution Time", f"{metrics.get('simulation_time', 0):.2f}s"),
            ("Steps per Second", f"{metrics.get('steps_per_second', 0):.1f}"),
            ("Memory Peak", f"{metrics.get('memory_peak', 0):.1f} MB"),
            ("CPU Usage", f"{metrics.get('cpu_usage', 0):.1f}%"),
            ("Efficiency Score", f"{metrics.get('efficiency_score', 0):.2f}"),
        ]

        for metric, value in metric_items:
            metrics_table.add_row(metric, value)

        console.print(metrics_table)

    # Function timings
    if "function_timings" in results:
        console.print("\n⏱️ [bold]Top Function Timings[/bold]")
        timings = results["function_timings"]

        # Sort by total time and show top 10
        sorted_timings = sorted(
            timings.items(), key=lambda x: x[1].get("total_time", 0), reverse=True
        )[:10]

        timings_table = Table()
        timings_table.add_column("Function", style="cyan")
        timings_table.add_column("Total Time", style="white")
        timings_table.add_column("Calls", style="dim")
        timings_table.add_column("Avg Time", style="dim")

        for func_name, timing_data in sorted_timings:
            total_time = timing_data.get("total_time", 0)
            call_count = timing_data.get("call_count", 0)
            avg_time = total_time / call_count if call_count > 0 else 0

            timings_table.add_row(
                func_name, f"{total_time:.3f}s", str(call_count), f"{avg_time:.6f}s"
            )

        console.print(timings_table)


def _compare_benchmark_results(
    baseline: Dict[str, Any], current: Dict[str, Any], threshold: float
) -> Dict[str, Any]:
    """Compare benchmark results for regression analysis"""

    from datetime import datetime

    comparison: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "regressions": [],
        "improvements": [],
        "unchanged": [],
        "summary": {},
    }

    # Compare simulation speed benchmarks
    if "simulation_speed" in baseline and "simulation_speed" in current:
        baseline_speed = baseline["simulation_speed"]
        current_speed = current["simulation_speed"]

        for size_name in baseline_speed:
            if size_name in current_speed:
                baseline_result = baseline_speed[size_name]
                current_result = current_speed[size_name]

                if hasattr(baseline_result, "steps_per_second") and hasattr(
                    current_result, "steps_per_second"
                ):
                    baseline_sps = baseline_result.steps_per_second
                    current_sps = current_result.steps_per_second

                    change = (current_sps - baseline_sps) / baseline_sps

                    comparison_item = {
                        "metric": f"simulation_speed_{size_name}",
                        "baseline_value": baseline_sps,
                        "current_value": current_sps,
                        "change_percent": change * 100,
                        "change_absolute": current_sps - baseline_sps,
                    }

                    if change < -threshold:  # Regression
                        comparison["regressions"].append(comparison_item)
                    elif change > threshold:  # Improvement
                        comparison["improvements"].append(comparison_item)
                    else:  # Unchanged
                        comparison["unchanged"].append(comparison_item)

    # Generate summary
    comparison["summary"] = {
        "total_metrics": len(comparison["regressions"])
        + len(comparison["improvements"])
        + len(comparison["unchanged"]),
        "regressions_count": len(comparison["regressions"]),
        "improvements_count": len(comparison["improvements"]),
        "unchanged_count": len(comparison["unchanged"]),
        "has_regressions": len(comparison["regressions"]) > 0,
    }

    return comparison


def _display_comparison_results(comparison: dict, threshold: float) -> None:
    """Display benchmark comparison results"""

    summary = comparison["summary"]
    has_regressions = summary["has_regressions"]

    # Overall status
    status_color = "red" if has_regressions else "green"
    status_text = "REGRESSIONS DETECTED" if has_regressions else "NO REGRESSIONS"

    status_panel = Panel(
        f"[{status_color}]{status_text}[/{status_color}]",
        title="Regression Analysis Results",
        border_style=status_color,
    )
    console.print(status_panel)

    # Summary table
    summary_table = Table(title="Comparison Summary")
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Count", style="white")

    summary_table.add_row("Total Metrics", str(summary["total_metrics"]))
    summary_table.add_row("Regressions", f"[red]{summary['regressions_count']}[/red]")
    summary_table.add_row(
        "Improvements", f"[green]{summary['improvements_count']}[/green]"
    )
    summary_table.add_row("Unchanged", str(summary["unchanged_count"]))

    console.print(summary_table)

    # Detailed results
    if comparison["regressions"]:
        console.print("\n❌ [bold red]Performance Regressions:[/bold red]")
        for reg in comparison["regressions"]:
            console.print(
                f"  • {reg['metric']}: {reg['change_percent']:.1f}% slower ({reg['baseline_value']:.1f} → {reg['current_value']:.1f})"
            )

    if comparison["improvements"]:
        console.print("\n✅ [bold green]Performance Improvements:[/bold green]")
        for imp in comparison["improvements"]:
            console.print(
                f"  • {imp['metric']}: {imp['change_percent']:.1f}% faster ({imp['baseline_value']:.1f} → {imp['current_value']:.1f})"
            )


if __name__ == "__main__":
    app()
