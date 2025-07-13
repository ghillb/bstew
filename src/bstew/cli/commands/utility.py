"""
Utility command implementations
===============================

Handles version information, project initialization, and other utility functions.
"""

from typing import Dict, Any, List
from pathlib import Path
import sys

from ..core.base import BaseCLICommand, CLIContext
from ..core.progress import ProgressManager
from ..types import CLIResult


class VersionCommand(BaseCLICommand):
    """Command for displaying version information"""
    
    def execute(self, **kwargs: Any) -> CLIResult:
        """Execute version display"""
        
        try:
            self._display_version_info()
            
            return CLIResult(
                success=True,
                message="Version information displayed",
                data=self._get_version_data(),
            )
            
        except Exception as e:
            return self.handle_exception(e, "Version display")
    
    def _display_version_info(self) -> None:
        """Display version information table"""
        
        from rich.table import Table
        
        version_info = Table(title="BSTEW Version Information")
        version_info.add_column("Component", style="cyan")
        version_info.add_column("Version", style="yellow")
        
        version_info.add_row("BSTEW", "0.1.0")
        version_info.add_row(
            "Python",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
        version_info.add_row("Platform", sys.platform)
        
        # Add dependency information
        try:
            import mesa
            version_info.add_row("Mesa", mesa.__version__)
        except ImportError:
            version_info.add_row("Mesa", "Not installed")
        
        try:
            import numpy
            version_info.add_row("NumPy", numpy.__version__)
        except ImportError:
            version_info.add_row("NumPy", "Not installed")
        
        try:
            import pandas
            version_info.add_row("Pandas", pandas.__version__)
        except ImportError:
            version_info.add_row("Pandas", "Not installed")
        
        try:
            import typer
            version_info.add_row("Typer", typer.__version__)
        except ImportError:
            version_info.add_row("Typer", "Not installed")
        
        try:
            import rich
            version_info.add_row("Rich", getattr(rich, '__version__', 'unknown'))
        except ImportError:
            version_info.add_row("Rich", "Not installed")
        
        self.console.print(version_info)
    
    def _get_version_data(self) -> Dict[str, str]:
        """Get version data as dictionary"""
        
        data = {
            "bstew": "0.1.0",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
        }
        
        # Add dependency versions
        dependencies = ["mesa", "numpy", "pandas", "typer", "rich"]
        for dep in dependencies:
            try:
                module = __import__(dep)
                data[dep] = getattr(module, "__version__", "unknown")
            except ImportError:
                data[dep] = "not_installed"
        
        return data


class InitCommand(BaseCLICommand):
    """Command for initializing new BSTEW projects"""
    
    def __init__(self, context: CLIContext) -> None:
        super().__init__(context)
        self.progress_manager = ProgressManager(self.console)
    
    def execute(
        self,
        directory: str = ".",
        template: str = "basic",
        **kwargs: Any,
    ) -> CLIResult:
        """Execute project initialization"""
        
        try:
            project_path = Path(directory)
            
            # Check if directory is empty
            if project_path.exists() and any(project_path.iterdir()):
                self.context.print_warning(f"Directory {directory} is not empty")
                # In real implementation, would ask for confirmation
            
            project_path.mkdir(exist_ok=True)
            
            self.context.print_info(
                f"Initializing BSTEW project in: {project_path.absolute()}"
            )
            
            # Initialize project with progress tracking
            self._initialize_project(project_path, template)
            
            self.context.print_success("BSTEW project initialized successfully!")
            self.context.print_info(f"Project directory: {project_path.absolute()}")
            
            # Display next steps
            self._display_next_steps()
            
            return CLIResult(
                success=True,
                message="Project initialized successfully",
                data={"project_path": str(project_path), "template": template},
            )
            
        except Exception as e:
            return self.handle_exception(e, "Project initialization")
    
    def validate_inputs(self, **kwargs: Any) -> List[str]:
        """Validate command inputs"""
        errors = []
        
        # Validate template
        template = kwargs.get("template", "basic")
        valid_templates = ["basic", "honeybee", "bumblebee", "minimal", "research"]
        if template not in valid_templates:
            errors.append(f"Invalid template: {template}. Valid: {valid_templates}")
        
        return errors
    
    def _initialize_project(self, project_path: Path, template: str) -> None:
        """Initialize project structure with progress tracking"""
        
        with self.progress_manager.progress_context() as progress:
            task = progress.start_task("Creating project structure...", total=6)
            
            # Create directories
            directories = ["configs", "data", "results", "scripts", "experiments"]
            for directory in directories:
                (project_path / directory).mkdir(exist_ok=True)
            progress.update_task(task, advance=1)
            
            # Create subdirectories
            (project_path / "data" / "landscapes").mkdir(exist_ok=True)
            (project_path / "data" / "weather").mkdir(exist_ok=True)
            (project_path / "data" / "parameters").mkdir(exist_ok=True)
            progress.update_task(task, advance=1)
            
            # Create default configuration
            config = self.config_manager.get_config_template(template)
            config_path = project_path / "configs" / "config.yaml"
            self.config_manager.save_config(config, str(config_path))
            progress.update_task(task, advance=1)
            
            # Create README
            readme_content = self._generate_readme_content(template)
            with open(project_path / "README.md", "w") as f:
                f.write(readme_content)
            progress.update_task(task, advance=1)
            
            # Create example scripts
            self._create_example_scripts(project_path)
            progress.update_task(task, advance=1)
            
            # Create .gitignore
            gitignore_content = self._generate_gitignore_content()
            with open(project_path / ".gitignore", "w") as f:
                f.write(gitignore_content)
            progress.update_task(task, advance=1)
            
            progress.finish_task(task, "Project structure created")
    
    def _generate_readme_content(self, template: str) -> str:
        """Generate README content for the project"""
        
        return f"""# BSTEW Project ({template} template)

This is a BSTEW (BeeSteward v2 Python) simulation project.

## Quick Start

1. Configure your simulation:
   ```bash
   bstew config show configs/config.yaml
   ```

2. Run the simulation:
   ```bash
   bstew run --config configs/config.yaml
   ```

3. Analyze results:
   ```bash
   bstew analyze results/
   ```

## Project Structure

- `configs/` - Configuration files
  - `config.yaml` - Main simulation configuration
- `data/` - Input data
  - `landscapes/` - Landscape maps and spatial data
  - `weather/` - Weather data files
  - `parameters/` - Parameter files for experiments
- `results/` - Simulation output
- `scripts/` - Custom analysis scripts
- `experiments/` - Experiment designs and results

## Advanced Usage

### Parameter Sweeps
```bash
bstew sweep colony.initial_population.workers 1000 10000 --steps 10
```

### Optimization
```bash
bstew optimize-parameters field_data.csv --method genetic_algorithm
```

### Validation
```bash
bstew validate results/ field_data.csv
```

## Configuration

Edit `configs/config.yaml` to customize:
- Colony parameters (species, initial population)
- Environment settings (landscape, weather)
- Simulation duration and output options
- Disease and mortality parameters

## Data Requirements

- **Landscape data**: RGB images or GIS files defining habitat types
- **Weather data**: CSV files with daily temperature, rainfall, wind
- **Species parameters**: CSV files with species-specific traits
- **Field data**: Observational data for validation

## Getting Help

- Run `bstew --help` for command documentation
- Use `bstew config validate` to check configuration
- Check the [BSTEW documentation](https://github.com/your-org/bstew) for detailed guides
"""
    
    def _create_example_scripts(self, project_path: Path) -> None:
        """Create example analysis scripts"""
        
        scripts_dir = project_path / "scripts"
        
        # Example analysis script
        analysis_script = """#!/usr/bin/env python3
\"\"\"
Example analysis script for BSTEW results
\"\"\"

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_population_dynamics(results_dir):
    \"\"\"Analyze population dynamics from simulation results\"\"\"
    
    # Load model data
    model_data_path = Path(results_dir) / "model_data.csv"
    if not model_data_path.exists():
        self.console.print(f"❌ Model data not found: {model_data_path}", style="red")
        return
    
    df = pd.read_csv(model_data_path)
    
    # Plot population over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['Step'], df['Total_Bees'])
    plt.title('Total Bee Population')
    plt.xlabel('Day')
    plt.ylabel('Number of Bees')
    
    plt.subplot(2, 2, 2)
    plt.plot(df['Step'], df['Total_Honey'])
    plt.title('Honey Production')
    plt.xlabel('Day')
    plt.ylabel('Honey (kg)')
    
    plt.subplot(2, 2, 3)
    plt.plot(df['Step'], df['Active_Colonies'])
    plt.title('Active Colonies')
    plt.xlabel('Day')
    plt.ylabel('Number of Colonies')
    
    plt.subplot(2, 2, 4)
    if 'Foraging_Efficiency' in df.columns:
        plt.plot(df['Step'], df['Foraging_Efficiency'])
        plt.title('Foraging Efficiency')
        plt.xlabel('Day')
        plt.ylabel('Efficiency')
    
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "population_analysis.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results"
    
    analyze_population_dynamics(results_dir)
"""
        
        with open(scripts_dir / "analyze_results.py", "w") as f:
            f.write(analysis_script)
        
        # Make script executable
        (scripts_dir / "analyze_results.py").chmod(0o755)
    
    def _generate_gitignore_content(self) -> str:
        """Generate .gitignore content"""
        
        return """# BSTEW specific
results/
experiments/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Data files (large datasets)
data/landscapes/*.tif
data/weather/*.nc
"""
    
    def _display_next_steps(self) -> None:
        """Display next steps for the user"""
        
        self.console.print("\nNext steps:", style="bold")
        self.console.print("1. Edit configs/config.yaml to customize your simulation")
        self.console.print("2. Add landscape and weather data to data/")
        self.console.print("3. Run: bstew run --config configs/config.yaml")
        self.console.print("4. Use scripts/analyze_results.py to analyze outputs")