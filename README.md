# BSTEW - BeeSteward v2 Python Implementation

**Production-ready Python implementation of the NetLogo BEE-STEWARD model with comprehensive enhancements for modern research workflows.**

## Overview

BSTEW provides high-performance agent-based modeling of bee populations with advanced features for stewardship assessment, economic analysis, and landscape management. The system maintains 100% NetLogo compatibility while offering 3-5x performance improvements.

### Key Features

- **Multi-Species Support**: Configurable for both bumblebees (default) and honey bees
- **Species-Specific Communication**: Scent-based (bumblebees) or dance-based (honey bees)
- **Proboscis-Corolla Matching**: 80-species morphological database with realistic foraging constraints
- **Economic Assessment**: ROI calculations, yield impacts, and subsidy optimization
- **CSS Components**: Enhanced margins, wildflower strips, and habitat creation algorithms

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Git for cloning the repository
- 4GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/ghillb/bstew.git
cd bstew

# Install with uv (recommended) - installs all dependencies
uv sync

# Alternative: Install in development mode with pip to make bstew available globally
uv pip install -e .

# Verify installation
uv run bstew version
uv run bstew --help
```

See the [official UV installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you need to install UV.

### Basic Usage

```bash
# Run simulation with default settings (bumblebees)
uv run bstew run

# Run with custom configuration
uv run bstew run --config configs/my-config.yaml --days 180

# Configure species
uv run bstew config species  # Show available species
uv run bstew config create myconfig --species APIS_MELLIFERA  # Create honey bee config

# Analyze results
uv run bstew analyze results/ --type population

# Launch interactive dashboard
uv run bstew visualize dashboard results/
```

## Species Configuration

BSTEW defaults to bumblebees (`BOMBUS_TERRESTRIS`) but supports multiple species:

```yaml
# In config file
colony:
  species: "BOMBUS_TERRESTRIS"  # or "APIS_MELLIFERA" for honey bees
```

See [Species Configuration Guide](docs/SPECIES_CONFIGURATION.md) for details.

## Documentation

📖 **[Complete User Guide](docs/guide.md)** - Comprehensive documentation for all features, architecture, development, and more.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{bstew2025,
  title={BSTEW: BeeSteward v2 Python Implementation},
  author={BSTEW Development Team},
  year={2025},
  url={https://github.com/ghillb/bstew}
}
```