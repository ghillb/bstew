"""
NetLogo Data File Integration for BSTEW
======================================

Direct NetLogo .csv file compatibility with automatic data discovery,
file format validation, and NetLogo-BSTEW parameter mapping.
"""

import csv
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import re
import logging
from dataclasses import dataclass
from enum import Enum

# Import moved to avoid circular import - using TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class NetLogoFileType(Enum):
    """NetLogo file types supported by BSTEW"""
    PARAMETERS = "parameters"
    SPECIES = "species"
    FLOWERSPECIES = "flowerspecies"
    HABITATS = "habitats"
    FOODSOURCES = "foodsources"
    LANDSCAPES = "landscapes"
    REPORTER = "reporter"
    EXPERIMENT = "experiment"
    UNKNOWN = "unknown"


@dataclass
class NetLogoFileInfo:
    """Information about a NetLogo file"""
    file_path: Path
    file_type: NetLogoFileType
    contains_data: bool
    row_count: int
    column_count: int
    columns: List[str]
    validation_errors: List[str]
    bstew_mapping: Optional[Dict[str, str]] = None


class NetLogoFileDetector:
    """Automatically detect and classify NetLogo files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # File type detection patterns
        self.file_patterns = {
            NetLogoFileType.PARAMETERS: [
                r".*parameters.*\.csv$",
                r".*_SYSTEM_Parameters.*\.csv$",
                r".*param.*\.csv$"
            ],
            NetLogoFileType.SPECIES: [
                r".*species.*\.csv$",
                r".*_SYSTEM_.*Species.*\.csv$",
                r".*BumbleSpecies.*\.csv$"
            ],
            NetLogoFileType.FLOWERSPECIES: [
                r".*flower.*species.*\.csv$",
                r".*_SYSTEM_Flowerspecies.*\.csv$",
                r".*floral.*\.csv$"
            ],
            NetLogoFileType.HABITATS: [
                r".*habitat.*\.csv$",
                r".*_SYSTEM_Habitats.*\.csv$",
                r".*land.*use.*\.csv$"
            ],
            NetLogoFileType.FOODSOURCES: [
                r".*food.*source.*\.csv$",
                r".*_SYSTEM_.*Food.*\.csv$",
                r".*resource.*\.csv$"
            ],
            NetLogoFileType.LANDSCAPES: [
                r".*landscape.*\.csv$",
                r".*_SYSTEM_Landscape.*\.csv$",
                r".*spatial.*\.csv$"
            ],
            NetLogoFileType.REPORTER: [
                r".*reporter.*\.csv$",
                r".*output.*\.csv$",
                r".*results.*\.csv$"
            ],
            NetLogoFileType.EXPERIMENT: [
                r".*experiment.*\.csv$",
                r".*behavioral.*space.*\.csv$",
                r".*bs.*\.csv$"
            ]
        }
        
        # Column header detection patterns
        self.column_patterns = {
            NetLogoFileType.PARAMETERS: [
                "parameter", "value", "min", "max", "description", "type"
            ],
            NetLogoFileType.SPECIES: [
                "species", "name", "proboscis", "length", "weight", "foraging"
            ],
            NetLogoFileType.FLOWERSPECIES: [
                "flower", "species", "corolla", "depth", "nectar", "pollen"
            ],
            NetLogoFileType.HABITATS: [
                "habitat", "type", "quality", "area", "management"
            ],
            NetLogoFileType.FOODSOURCES: [
                "food", "source", "location", "quality", "quantity"
            ]
        }
        
    def detect_file_type(self, file_path: Path) -> NetLogoFileType:
        """Detect the type of NetLogo file"""
        file_name = file_path.name.lower()
        
        # Check filename patterns
        for file_type, patterns in self.file_patterns.items():
            for pattern in patterns:
                if re.match(pattern, file_name, re.IGNORECASE):
                    return file_type
                    
        # Check column headers if filename doesn't match
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                
            if headers:
                headers_lower = [h.lower() for h in headers]
                for file_type, keywords in self.column_patterns.items():
                    matches = sum(1 for keyword in keywords if any(keyword in h for h in headers_lower))
                    if matches >= 2:  # At least 2 keyword matches
                        return file_type
                        
        except Exception as e:
            self.logger.warning(f"Could not read headers from {file_path}: {e}")
            
        return NetLogoFileType.UNKNOWN
        
    def scan_directory(self, directory: Path) -> List[NetLogoFileInfo]:
        """Scan directory for NetLogo files"""
        netlogo_files = []
        
        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return netlogo_files
            
        # Find all CSV files
        csv_files = list(directory.rglob("*.csv"))
        
        for csv_file in csv_files:
            try:
                file_info = self.analyze_file(csv_file)
                netlogo_files.append(file_info)
            except Exception as e:
                self.logger.error(f"Error analyzing file {csv_file}: {e}")
                
        return netlogo_files
        
    def analyze_file(self, file_path: Path) -> NetLogoFileInfo:
        """Analyze a NetLogo file and extract information"""
        file_type = self.detect_file_type(file_path)
        validation_errors = []
        
        try:
            # Read file to analyze structure
            df = pd.read_csv(file_path)
            row_count = len(df)
            column_count = len(df.columns)
            columns = df.columns.tolist()
            contains_data = row_count > 0
            
            # Validate file structure
            if file_type != NetLogoFileType.UNKNOWN:
                validation_errors = self._validate_file_structure(df, file_type)
                
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            row_count = 0
            column_count = 0
            columns = []
            contains_data = False
            validation_errors = [f"File read error: {str(e)}"]
            
        return NetLogoFileInfo(
            file_path=file_path,
            file_type=file_type,
            contains_data=contains_data,
            row_count=row_count,
            column_count=column_count,
            columns=columns,
            validation_errors=validation_errors
        )
        
    def _validate_file_structure(self, df: pd.DataFrame, file_type: NetLogoFileType) -> List[str]:
        """Validate NetLogo file structure"""
        errors = []
        
        if file_type == NetLogoFileType.PARAMETERS:
            required_columns = ["parameter", "value"]
            for col in required_columns:
                if col not in df.columns:
                    errors.append(f"Missing required column: {col}")
                    
        elif file_type == NetLogoFileType.SPECIES:
            required_columns = ["species"]
            for col in required_columns:
                if col not in df.columns:
                    errors.append(f"Missing required column: {col}")
                    
        elif file_type == NetLogoFileType.FLOWERSPECIES:
            required_columns = ["flower_species"]
            for col in required_columns:
                if col not in df.columns:
                    errors.append(f"Missing required column: {col}")
                    
        # Check for empty data
        if len(df) == 0:
            errors.append("File contains no data rows")
            
        return errors


class NetLogoParameterMapper:
    """Maps NetLogo parameters to BSTEW parameters"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # NetLogo to BSTEW parameter mapping
        self.parameter_mapping = {
            # Behavioral parameters
            "foraging_duration": "foraging_duration_min",
            "max_foraging_distance": "foraging_range",
            "dance_probability": "dance_following_probability",
            "memory_decay": "memory_decay_rate",
            "exploration_rate": "exploration_probability",
            
            # Colony parameters
            "initial_workers": "initial_population",
            "colony_size": "initial_population",
            "queen_count": "initial_queens",
            "drone_count": "initial_drones",
            
            # Environmental parameters
            "temperature": "temperature_min",
            "temp_variation": "temperature_variation",
            "weather_factor": "weather_variation",
            "seasonal_effect": "seasonal_effects",
            
            # Foraging parameters
            "nectar_collection_rate": "nectar_collection_efficiency",
            "pollen_collection_rate": "pollen_collection_efficiency",
            "handling_time": "handling_time_nectar",
            "travel_time": "travel_time_base",
            
            # Mortality parameters
            "mortality_rate": "daily_mortality_rate",
            "disease_mortality": "disease_mortality_rate",
            "starvation_mortality": "starvation_mortality_rate",
            
            # Species parameters
            "proboscis_length": "proboscis_length_mm",
            "body_weight": "body_weight_mg",
            "crop_capacity": "crop_capacity_ul",
            "flight_speed": "flight_speed_ms"
        }
        
        # Reverse mapping for BSTEW to NetLogo
        self.reverse_mapping = {v: k for k, v in self.parameter_mapping.items()}
        
    def map_netlogo_to_bstew(self, netlogo_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map NetLogo parameters to BSTEW format"""
        bstew_params = {}
        unmapped_params = {}
        
        for netlogo_param, value in netlogo_params.items():
            if netlogo_param in self.parameter_mapping:
                bstew_param = self.parameter_mapping[netlogo_param]
                bstew_params[bstew_param] = value
            else:
                # Try fuzzy matching
                bstew_param = self._fuzzy_match_parameter(netlogo_param)
                if bstew_param:
                    bstew_params[bstew_param] = value
                else:
                    unmapped_params[netlogo_param] = value
                    
        if unmapped_params:
            self.logger.warning(f"Unmapped NetLogo parameters: {list(unmapped_params.keys())}")
            
        return bstew_params
        
    def map_bstew_to_netlogo(self, bstew_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map BSTEW parameters to NetLogo format"""
        netlogo_params = {}
        unmapped_params = {}
        
        for bstew_param, value in bstew_params.items():
            if bstew_param in self.reverse_mapping:
                netlogo_param = self.reverse_mapping[bstew_param]
                netlogo_params[netlogo_param] = value
            else:
                unmapped_params[bstew_param] = value
                
        if unmapped_params:
            self.logger.warning(f"Unmapped BSTEW parameters: {list(unmapped_params.keys())}")
            
        return netlogo_params
        
    def _fuzzy_match_parameter(self, param_name: str) -> Optional[str]:
        """Attempt fuzzy matching for parameter names"""
        param_lower = param_name.lower()
        
        # Common variations
        variations = {
            "temp": "temperature",
            "mort": "mortality", 
            "prob": "probability",
            "dist": "distance",
            "dur": "duration",
            "eff": "efficiency",
            "var": "variation"
        }
        
        for abbrev, full in variations.items():
            if abbrev in param_lower:
                # Look for matching BSTEW parameter
                for bstew_param in self.parameter_mapping.values():
                    if full in bstew_param.lower():
                        return bstew_param
                        
        return None


class NetLogoDataIntegrator:
    """Integrates NetLogo data files with BSTEW"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_detector = NetLogoFileDetector()
        self.parameter_mapper = NetLogoParameterMapper()
        
    def discover_netlogo_files(self, directory: Path) -> Dict[NetLogoFileType, List[NetLogoFileInfo]]:
        """Discover and classify NetLogo files"""
        all_files = self.file_detector.scan_directory(directory)
        
        # Group by file type
        files_by_type = {}
        for file_info in all_files:
            if file_info.file_type not in files_by_type:
                files_by_type[file_info.file_type] = []
            files_by_type[file_info.file_type].append(file_info)
            
        return files_by_type
        
    def load_netlogo_parameters(self, file_path: Path) -> Dict[str, Any]:
        """Load parameters from NetLogo parameter file"""
        try:
            df = pd.read_csv(file_path)
            
            # Handle different NetLogo parameter file formats
            if 'parameter' in df.columns and 'value' in df.columns:
                # Standard format: parameter, value
                params = dict(zip(df['parameter'], df['value']))
            elif 'Parameter' in df.columns and 'Value' in df.columns:
                # Capitalized format
                params = dict(zip(df['Parameter'], df['Value']))
            elif len(df.columns) >= 2:
                # Assume first two columns are parameter and value
                params = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            else:
                raise ValueError("Unable to determine parameter file format")
                
            # Map to BSTEW format
            bstew_params = self.parameter_mapper.map_netlogo_to_bstew(params)
            
            return bstew_params
            
        except Exception as e:
            self.logger.error(f"Error loading NetLogo parameters from {file_path}: {e}")
            raise
            
    def load_netlogo_species(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load species data from NetLogo species file"""
        try:
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to list of dictionaries
            species_data = df.to_dict('records')
            
            # Map column names to BSTEW format if needed
            mapped_species = []
            for species in species_data:
                mapped_species.append(self._map_species_data(species))
                
            return mapped_species
            
        except Exception as e:
            self.logger.error(f"Error loading NetLogo species from {file_path}: {e}")
            raise
            
    def _map_species_data(self, species_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map NetLogo species data to BSTEW format"""
        mapping = {
            "species": "species_name",
            "proboscis_length": "proboscis_length_mm",
            "body_weight": "body_weight_mg",
            "crop_capacity": "crop_capacity_ul",
            "flight_speed": "flight_speed_ms",
            "foraging_range": "max_foraging_distance",
            "nectar_preference": "nectar_preference",
            "pollen_preference": "pollen_preference"
        }
        
        mapped_data = {}
        for netlogo_key, bstew_key in mapping.items():
            if netlogo_key in species_data:
                mapped_data[bstew_key] = species_data[netlogo_key]
            elif netlogo_key.title() in species_data:
                mapped_data[bstew_key] = species_data[netlogo_key.title()]
                
        # Add unmapped data as-is
        for key, value in species_data.items():
            if key not in mapping and key.title() not in mapping:
                mapped_data[key] = value
                
        return mapped_data
        
    def validate_netlogo_compatibility(self, directory: Path) -> Dict[str, Any]:
        """Validate NetLogo data compatibility with BSTEW"""
        validation_report = {
            "compatible": True,
            "files_found": {},
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "parameter_validation": {},
            "data_quality": {}
        }
        
        # Discover files
        files_by_type = self.discover_netlogo_files(directory)
        
        for file_type, files in files_by_type.items():
            validation_report["files_found"][file_type.value] = len(files)
            
            for file_info in files:
                if file_info.validation_errors:
                    validation_report["errors"].extend(file_info.validation_errors)
                    validation_report["compatible"] = False
                    
        # Check for required files
        required_files = [NetLogoFileType.PARAMETERS]
        for required_type in required_files:
            if required_type not in files_by_type or not files_by_type[required_type]:
                validation_report["errors"].append(f"Missing required file type: {required_type.value}")
                validation_report["compatible"] = False
        
        # Enhanced NetLogo BEE-STEWARD v2 parameter validation
        if NetLogoFileType.PARAMETERS in files_by_type:
            param_validation = self._validate_netlogo_parameters(files_by_type[NetLogoFileType.PARAMETERS])
            validation_report["parameter_validation"] = param_validation
            if not param_validation["valid"]:
                validation_report["compatible"] = False
                validation_report["errors"].extend(param_validation["errors"])
        
        # Species file validation
        if NetLogoFileType.SPECIES in files_by_type:
            species_validation = self._validate_netlogo_species(files_by_type[NetLogoFileType.SPECIES])
            validation_report["data_quality"]["species"] = species_validation
            if not species_validation["valid"]:
                validation_report["warnings"].extend(species_validation["warnings"])
        
        # Flower species validation  
        if NetLogoFileType.FLOWERSPECIES in files_by_type:
            flower_validation = self._validate_netlogo_flower_species(files_by_type[NetLogoFileType.FLOWERSPECIES])
            validation_report["data_quality"]["flower_species"] = flower_validation
            if not flower_validation["valid"]:
                validation_report["warnings"].extend(flower_validation["warnings"])
                
        # Add recommendations
        if NetLogoFileType.SPECIES not in files_by_type:
            validation_report["recommendations"].append("Consider adding species files for better compatibility")
            
        if NetLogoFileType.FLOWERSPECIES not in files_by_type:
            validation_report["recommendations"].append("Consider adding flower species files for resource modeling")
            
        # Check for BEE-STEWARD v2 specific requirements
        self._validate_bee_steward_v2_requirements(validation_report)
            
        return validation_report
    
    def _validate_netlogo_parameters(self, parameter_files: List[NetLogoFileInfo]) -> Dict[str, Any]:
        """Validate NetLogo BEE-STEWARD v2 parameter files"""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "required_parameters_found": {},
            "parameter_ranges_valid": {}
        }
        
        # Required parameters for NetLogo BEE-STEWARD v2 compatibility
        required_params = {
            # Physiological parameters
            "cropvolume_myl": {"min": 10.0, "max": 150.0, "type": float},
            "glossaLength_mm": {"min": 1.0, "max": 10.0, "type": float},
            "proboscis_length_mm": {"min": 3.0, "max": 15.0, "type": float},
            "mandible_width_mm": {"min": 0.5, "max": 3.0, "type": float},
            
            # Colony parameters
            "initial_population": {"min": 10, "max": 10000, "type": int},
            "max_foraging_range": {"min": 100.0, "max": 5000.0, "type": float},
            "energy_threshold": {"min": 1.0, "max": 100.0, "type": float},
            
            # Behavioral parameters
            "dance_threshold": {"min": 0.0, "max": 1.0, "type": float},
            "exploration_probability": {"min": 0.0, "max": 1.0, "type": float},
            "memory_decay_rate": {"min": 0.8, "max": 1.0, "type": float},
            
            # Environmental parameters
            "season_length": {"min": 30, "max": 365, "type": int},
            "temperature_min": {"min": -10.0, "max": 40.0, "type": float},
            "temperature_max": {"min": 0.0, "max": 50.0, "type": float}
        }
        
        found_parameters = {}
        
        # Check each parameter file
        for file_info in parameter_files:
            try:
                params = self.load_netlogo_parameters(file_info.file_path)
                found_parameters.update(params)
            except Exception as e:
                validation["errors"].append(f"Error reading parameter file {file_info.file_path}: {e}")
                validation["valid"] = False
                continue
        
        # Validate required parameters
        for param_name, constraints in required_params.items():
            if param_name in found_parameters:
                validation["required_parameters_found"][param_name] = True
                
                # Validate parameter value range
                try:
                    value = constraints["type"](found_parameters[param_name])
                    if "min" in constraints and value < constraints["min"]:
                        validation["errors"].append(f"Parameter {param_name} value {value} below minimum {constraints['min']}")
                        validation["valid"] = False
                    elif "max" in constraints and value > constraints["max"]:
                        validation["errors"].append(f"Parameter {param_name} value {value} above maximum {constraints['max']}")
                        validation["valid"] = False
                    else:
                        validation["parameter_ranges_valid"][param_name] = True
                except (ValueError, TypeError) as e:
                    validation["errors"].append(f"Parameter {param_name} has invalid type: {e}")
                    validation["valid"] = False
            else:
                validation["required_parameters_found"][param_name] = False
                validation["warnings"].append(f"Required parameter {param_name} not found - using default value")
        
        return validation
    
    def _validate_netlogo_species(self, species_files: List[NetLogoFileInfo]) -> Dict[str, Any]:
        """Validate NetLogo species files"""
        validation = {
            "valid": True,
            "warnings": [],
            "species_count": 0,
            "required_columns_found": {}
        }
        
        required_columns = [
            "SpeciesName", "BodyLength_mm", "Wings", "SocialStatus", 
            "FlightRange_m", "ActivePeriod", "ForagingPeriod"
        ]
        
        for file_info in species_files:
            try:
                df = pd.read_csv(file_info.file_path)
                validation["species_count"] += len(df)
                
                # Check for required columns
                for col in required_columns:
                    if col in df.columns:
                        validation["required_columns_found"][col] = True
                    else:
                        validation["required_columns_found"][col] = False
                        validation["warnings"].append(f"Species file missing column: {col}")
                        
            except Exception as e:
                validation["warnings"].append(f"Error reading species file {file_info.file_path}: {e}")
        
        return validation
    
    def _validate_netlogo_flower_species(self, flower_files: List[NetLogoFileInfo]) -> Dict[str, Any]:
        """Validate NetLogo flower species files"""
        validation = {
            "valid": True,
            "warnings": [],
            "flower_species_count": 0,
            "required_columns_found": {}
        }
        
        required_columns = [
            "FlowerSpecies", "CorollaDepth_mm", "CorollaWidth_mm", 
            "NectarVolume_myl", "PollenMass_mg", "BloomPeriod"
        ]
        
        for file_info in flower_files:
            try:
                df = pd.read_csv(file_info.file_path)
                validation["flower_species_count"] += len(df)
                
                # Check for required columns
                for col in required_columns:
                    if col in df.columns:
                        validation["required_columns_found"][col] = True
                    else:
                        validation["required_columns_found"][col] = False
                        validation["warnings"].append(f"Flower species file missing column: {col}")
                        
            except Exception as e:
                validation["warnings"].append(f"Error reading flower species file {file_info.file_path}: {e}")
        
        return validation
    
    def _validate_bee_steward_v2_requirements(self, validation_report: Dict[str, Any]) -> None:
        """Validate specific BEE-STEWARD v2 requirements"""
        
        # Check for BEE-STEWARD v2 specific file naming patterns
        if not any("BEE-STEWARD" in str(files) for files in validation_report["files_found"]):
            validation_report["recommendations"].append(
                "Consider using BEE-STEWARD v2 naming conventions for better compatibility"
            )
        
        # Check for minimum required data completeness
        param_validation = validation_report.get("parameter_validation", {})
        if param_validation:
            required_found = param_validation.get("required_parameters_found", {})
            found_count = sum(1 for found in required_found.values() if found)
            total_count = len(required_found)
            
            if total_count > 0:
                completeness = found_count / total_count
                if completeness < 0.8:  # Less than 80% of required parameters
                    validation_report["warnings"].append(
                        f"Only {completeness:.1%} of required BEE-STEWARD v2 parameters found"
                    )
                elif completeness == 1.0:
                    validation_report["recommendations"].append(
                        "Excellent! All required BEE-STEWARD v2 parameters found"
                    )
        
        return validation_report
        
    def convert_netlogo_to_bstew(self, netlogo_directory: Path, output_directory: Path) -> None:
        """Convert NetLogo data files to BSTEW format"""
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover files
        files_by_type = self.discover_netlogo_files(netlogo_directory)
        
        converted_files = []
        
        for file_type, files in files_by_type.items():
            for file_info in files:
                try:
                    if file_type == NetLogoFileType.PARAMETERS:
                        params = self.load_netlogo_parameters(file_info.file_path)
                        output_file = output_dir / f"bstew_parameters_{file_info.file_path.stem}.json"
                        with open(output_file, 'w') as f:
                            json.dump(params, f, indent=2)
                        converted_files.append(output_file)
                        
                    elif file_type == NetLogoFileType.SPECIES:
                        species = self.load_netlogo_species(file_info.file_path)
                        output_file = output_dir / f"bstew_species_{file_info.file_path.stem}.json"
                        with open(output_file, 'w') as f:
                            json.dump(species, f, indent=2)
                        converted_files.append(output_file)
                        
                    else:
                        # Copy other files as-is for now
                        output_file = output_dir / f"bstew_{file_info.file_path.name}"
                        df = pd.read_csv(file_info.file_path)
                        df.to_csv(output_file, index=False)
                        converted_files.append(output_file)
                        
                except Exception as e:
                    self.logger.error(f"Error converting file {file_info.file_path}: {e}")
                    
        self.logger.info(f"Converted {len(converted_files)} files to BSTEW format")
        
        # Create conversion report
        report = {
            "conversion_date": pd.Timestamp.now().isoformat(),
            "source_directory": str(netlogo_directory),
            "output_directory": str(output_directory),
            "converted_files": [str(f) for f in converted_files],
            "file_types_processed": list(files_by_type.keys())
        }
        
        report_file = output_dir / "conversion_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Conversion report saved to {report_file}")