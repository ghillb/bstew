"""
Path utilities for BSTEW - Centralized path management
=====================================================

Utilities for managing file paths and ensuring all artifacts 
are stored in the correct directories.
"""

import os
from pathlib import Path
from typing import Union

# Base artifacts directory
ARTIFACTS_DIR = "artifacts"

def get_artifacts_path(subdir: str = "") -> str:
    """
    Get the path to the artifacts directory or a subdirectory.
    
    Args:
        subdir: Optional subdirectory within artifacts
        
    Returns:
        Path to artifacts directory or subdirectory
        
    Examples:
        >>> get_artifacts_path()
        'artifacts'
        >>> get_artifacts_path("results")
        'artifacts/results'
        >>> get_artifacts_path("cache")
        'artifacts/cache'
    """
    if subdir:
        return os.path.join(ARTIFACTS_DIR, subdir)
    return ARTIFACTS_DIR

def ensure_artifacts_dir(subdir: str = "") -> Path:
    """
    Ensure the artifacts directory (or subdirectory) exists.
    
    Args:
        subdir: Optional subdirectory within artifacts
        
    Returns:
        Path object for the directory
    """
    path = Path(get_artifacts_path(subdir))
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_output_path(filename: str, subdir: str = "outputs") -> str:
    """
    Get a standardized output path for generated files.
    
    Args:
        filename: Name of the file
        subdir: Subdirectory within artifacts (default: "outputs")
        
    Returns:
        Full path to the output file
    """
    return os.path.join(get_artifacts_path(subdir), filename)

def get_cache_path(filename: str = "") -> str:
    """
    Get a path in the cache directory.
    
    Args:
        filename: Optional filename
        
    Returns:
        Path to cache directory or specific cache file
    """
    cache_dir = get_artifacts_path("cache")
    if filename:
        return os.path.join(cache_dir, filename)
    return cache_dir

def get_results_path(filename: str = "") -> str:
    """
    Get a path in the results directory.
    
    Args:
        filename: Optional filename
        
    Returns:
        Path to results directory or specific results file
    """
    results_dir = get_artifacts_path("results")
    if filename:
        return os.path.join(results_dir, filename)
    return results_dir

def get_parameters_path(filename: str = "") -> str:
    """
    Get a path in the parameters directory.
    
    Args:
        filename: Optional filename
        
    Returns:
        Path to parameters directory or specific parameters file
    """
    params_dir = get_artifacts_path("parameters")
    if filename:
        return os.path.join(params_dir, filename)
    return params_dir

def validate_artifact_path(path: Union[str, Path]) -> bool:
    """
    Validate that a path is within the artifacts directory.
    
    Args:
        path: Path to validate
        
    Returns:
        True if path is within artifacts directory
    """
    path_str = str(path)
    return path_str.startswith(ARTIFACTS_DIR) or path_str.startswith(f"./{ARTIFACTS_DIR}")

# Predefined common paths
COMMON_PATHS = {
    "cache": get_artifacts_path("cache"),
    "results": get_artifacts_path("results"),
    "parameters": get_artifacts_path("parameters"),
    "reports": get_artifacts_path("reports"),
    "outputs": get_artifacts_path("outputs"),
    "experiments": get_artifacts_path("experiments"),
    "validation": get_artifacts_path("validation"),
    "optimization": get_artifacts_path("optimization"),
    "performance": get_artifacts_path("performance"),
    "coverage": get_artifacts_path("coverage"),
}

def get_common_path(name: str) -> str:
    """
    Get a common artifact path by name.
    
    Args:
        name: Name of the common path
        
    Returns:
        Path to the common directory
        
    Raises:
        KeyError: If the path name is not recognized
    """
    if name not in COMMON_PATHS:
        raise KeyError(f"Unknown common path: {name}. Available: {list(COMMON_PATHS.keys())}")
    return COMMON_PATHS[name]