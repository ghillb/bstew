"""
Progress reporting and Rich integration for CLI
===============================================

Provides progress bars, status indicators, and visual feedback for CLI operations.
"""

from typing import Dict, Optional, Any, ContextManager, Iterator
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID,
)
from rich.console import Console
from rich.table import Table
import contextlib


class RichProgressReporter:
    """Rich-based progress reporter implementation"""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._progress: Optional[Progress] = None
        self._tasks: Dict[str, TaskID] = {}
        self._task_counter = 0

    def start_progress(self) -> Progress:
        """Start progress tracking"""
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
        return self._progress

    def start_task(self, description: str, total: Optional[int] = None) -> str:
        """Start a new progress task"""
        if self._progress is None:
            self.start_progress()

        assert self._progress is not None
        task_id = self._progress.add_task(description, total=total)
        task_name = f"task_{self._task_counter}"
        self._tasks[task_name] = task_id
        self._task_counter += 1

        return task_name

    def update_task(
        self,
        task_name: str,
        advance: int = 1,
        description: Optional[str] = None,
        total: Optional[int] = None,
    ) -> None:
        """Update progress task"""
        if self._progress is None or task_name not in self._tasks:
            return

        task_id = self._tasks[task_name]
        kwargs: Dict[str, Any] = {"advance": advance}

        if description is not None:
            kwargs["description"] = description
        if total is not None:
            kwargs["total"] = total

        self._progress.update(task_id, **kwargs)

    def finish_task(self, task_name: str, description: Optional[str] = None) -> None:
        """Mark task as finished"""
        if self._progress is None or task_name not in self._tasks:
            return

        task_id = self._tasks[task_name]

        if description:
            self._progress.update(task_id, description=f"✅ {description}")
        else:
            current_desc = str(self._progress.tasks[task_id].description)
            if not current_desc.startswith("✅"):
                self._progress.update(task_id, description=f"✅ {current_desc}")

    def stop_progress(self) -> None:
        """Stop progress tracking"""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._tasks.clear()


class ProgressManager:
    """Context manager for progress operations"""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._reporter = RichProgressReporter(console)

    @contextlib.contextmanager
    def progress_context(self) -> Iterator[RichProgressReporter]:
        """Context manager for progress operations"""
        progress = self._reporter.start_progress()

        try:
            with progress:
                yield self._reporter
        finally:
            self._reporter.stop_progress()

    def simple_progress(
        self,
        items: list,
        description: str = "Processing...",
        show_progress: bool = True,
    ) -> ContextManager[Any]:
        """Simple progress context for iterating over items"""
        return self._simple_progress_context(items, description, show_progress)

    @contextlib.contextmanager
    def _simple_progress_context(
        self,
        items: list,
        description: str,
        show_progress: bool,
    ) -> Iterator[Any]:
        """Internal context manager for simple progress"""
        if not show_progress:
            yield items
            return

        with self.progress_context() as progress:
            task = progress.start_task(description, total=len(items))

            def progress_iterator() -> Iterator[Any]:
                for i, item in enumerate(items):
                    yield item
                    progress.update_task(task, advance=1)
                progress.finish_task(task)

            yield progress_iterator()


class StatusDisplay:
    """Rich status display utilities"""

    def __init__(self, console: Console) -> None:
        self.console = console

    def show_simulation_info(self, config: Dict[str, Any]) -> None:
        """Display simulation configuration info"""
        info_table = Table(title="Simulation Configuration")
        info_table.add_column("Parameter", style="cyan")
        info_table.add_column("Value", style="yellow")

        # Extract common configuration values safely
        simulation = config.get("simulation", {})
        colony = config.get("colony", {})
        environment = config.get("environment", {})

        info_table.add_row("Duration", f"{simulation.get('duration_days', 'N/A')} days")
        info_table.add_row("Random Seed", str(simulation.get("random_seed", "None")))
        info_table.add_row("Colony Species", str(colony.get("species", "N/A")))

        # Handle initial population safely
        initial_pop = colony.get("initial_population", {})
        if isinstance(initial_pop, dict):
            workers = initial_pop.get("workers", "N/A")
        else:
            workers = "N/A"
        info_table.add_row("Initial Workers", str(workers))

        info_table.add_row(
            "Landscape Size",
            f"{environment.get('landscape_width', 'N/A')}x{environment.get('landscape_height', 'N/A')}",
        )

        self.console.print(info_table)

    def show_results_summary(self, results: Dict[str, Any]) -> None:
        """Display results summary table"""
        results_table = Table(title="Results Summary")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="yellow")
        results_table.add_column("Unit", style="green")

        # Add results with safe access
        for key, value in results.items():
            if isinstance(value, (int, float)):
                if "population" in key.lower():
                    unit = "bees"
                elif "honey" in key.lower():
                    unit = "kg"
                elif "efficiency" in key.lower() or "ratio" in key.lower():
                    unit = "ratio"
                elif "survival" in key.lower():
                    unit = "boolean"
                else:
                    unit = "units"

                formatted_value = (
                    f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                )
                results_table.add_row(
                    key.replace("_", " ").title(), formatted_value, unit
                )

        self.console.print(results_table)

    def show_banner(self) -> None:
        """Display BSTEW banner"""
        banner = """
    ╭─────────────────────────────────────────────────────────╮
    │                                                         │
    │   ██████╗ ███████╗████████╗███████╗██╗    ██╗          │
    │   ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██║    ██║          │
    │   ██████╔╝███████╗   ██║   █████╗  ██║ █╗ ██║          │
    │   ██╔══██╗╚════██║   ██║   ██╔══╝  ██║███╗██║          │
    │   ██████╔╝███████║   ██║   ███████╗╚███╔███╔╝          │
    │   ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚══╝╚══╝           │
    │                                                         │
    │        BeeSteward v2 Python Transpilation               │
    │        Agent-based Pollinator Population Modeling       │
    │                                                         │
    ╰─────────────────────────────────────────────────────────╯
        """
        self.console.print(banner, style="bold blue")


def create_experiment_progress() -> Progress:
    """Create progress bar for experiments"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=Console(),
        transient=False,
    )
