"""
NetLogo CSV Compatibility and Validation System
===============================================

Comprehensive system for reading, writing, and validating NetLogo-compatible
CSV files, ensuring seamless data exchange between BSTEW and NetLogo models.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import math


class NetLogoDataType(Enum):
    """NetLogo data types"""

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"
    AGENT = "agent"
    AGENTSET = "agentset"
    PATCH = "patch"
    TURTLE = "turtle"
    LINK = "link"


class CSVValidationLevel(Enum):
    """CSV validation strictness levels"""

    STRICT = "strict"  # Full NetLogo compatibility required
    MODERATE = "moderate"  # Some flexibility allowed
    LENIENT = "lenient"  # Basic validation only


@dataclass
class ColumnSchema:
    """Schema definition for CSV columns"""

    name: str
    data_type: NetLogoDataType
    required: bool = True
    default_value: Optional[Any] = None
    valid_range: Optional[Tuple[float, float]] = None
    valid_values: Optional[List[Any]] = None
    description: str = ""
    netlogo_variable: Optional[str] = None  # Corresponding NetLogo variable name


@dataclass
class ValidationResult:
    """Result of CSV validation"""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    netlogo_compatible: bool = True
    suggested_fixes: List[str] = field(default_factory=list)


class NetLogoCSVReader(BaseModel):
    """Reader for NetLogo-compatible CSV files"""

    model_config = {"validate_assignment": True}

    # Configuration
    validation_level: CSVValidationLevel = CSVValidationLevel.MODERATE
    auto_convert_types: bool = True
    handle_missing_values: bool = True
    default_missing_value: str = "0"

    # Schema definitions
    agent_schema: Dict[str, ColumnSchema] = Field(default_factory=dict)
    patch_schema: Dict[str, ColumnSchema] = Field(default_factory=dict)
    global_schema: Dict[str, ColumnSchema] = Field(default_factory=dict)

    # State tracking
    last_validation_result: Optional[ValidationResult] = None

    # Logging
    logger: Any = Field(
        default_factory=lambda: logging.getLogger(__name__), exclude=True
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

        # Initialize default schemas
        self._initialize_default_schemas()

    def _initialize_default_schemas(self) -> None:
        """Initialize default CSV schemas for NetLogo compatibility"""

        # Agent (bee) schema
        self.agent_schema = {
            "who": ColumnSchema(
                "who", NetLogoDataType.NUMBER, True, description="Agent ID"
            ),
            "breed": ColumnSchema(
                "breed", NetLogoDataType.STRING, True, description="Agent breed"
            ),
            "xcor": ColumnSchema(
                "xcor", NetLogoDataType.NUMBER, True, description="X coordinate"
            ),
            "ycor": ColumnSchema(
                "ycor", NetLogoDataType.NUMBER, True, description="Y coordinate"
            ),
            "heading": ColumnSchema(
                "heading",
                NetLogoDataType.NUMBER,
                True,
                valid_range=(0, 360),
                description="Agent heading",
            ),
            "color": ColumnSchema(
                "color",
                NetLogoDataType.NUMBER,
                False,
                default_value=15,
                description="Agent color",
            ),
            "size": ColumnSchema(
                "size",
                NetLogoDataType.NUMBER,
                False,
                default_value=1.0,
                valid_range=(0, 10),
                description="Agent size",
            ),
            "energy": ColumnSchema(
                "energy",
                NetLogoDataType.NUMBER,
                False,
                default_value=100.0,
                valid_range=(0, 1000),
                description="Agent energy",
            ),
            "age": ColumnSchema(
                "age",
                NetLogoDataType.NUMBER,
                False,
                default_value=0,
                valid_range=(0, 365),
                description="Agent age",
            ),
            "status": ColumnSchema(
                "status",
                NetLogoDataType.STRING,
                False,
                valid_values=[
                    "foraging",
                    "resting",
                    "nursing",
                    "guarding",
                    "building",
                    "dead",
                ],
                description="Agent status",
            ),
        }

        # Patch schema
        self.patch_schema = {
            "pxcor": ColumnSchema(
                "pxcor", NetLogoDataType.NUMBER, True, description="Patch X coordinate"
            ),
            "pycor": ColumnSchema(
                "pycor", NetLogoDataType.NUMBER, True, description="Patch Y coordinate"
            ),
            "pcolor": ColumnSchema(
                "pcolor",
                NetLogoDataType.NUMBER,
                False,
                default_value=0,
                description="Patch color",
            ),
            "flower-density": ColumnSchema(
                "flower-density",
                NetLogoDataType.NUMBER,
                False,
                default_value=0.0,
                valid_range=(0, 100),
                description="Flower density",
            ),
            "nectar-amount": ColumnSchema(
                "nectar-amount",
                NetLogoDataType.NUMBER,
                False,
                default_value=0.0,
                valid_range=(0, 1000),
                description="Nectar amount",
            ),
            "pollen-amount": ColumnSchema(
                "pollen-amount",
                NetLogoDataType.NUMBER,
                False,
                default_value=0.0,
                valid_range=(0, 1000),
                description="Pollen amount",
            ),
            "resource-quality": ColumnSchema(
                "resource-quality",
                NetLogoDataType.NUMBER,
                False,
                default_value=0.5,
                valid_range=(0, 1),
                description="Resource quality",
            ),
        }

        # Global variables schema
        self.global_schema = {
            "ticks": ColumnSchema(
                "ticks", NetLogoDataType.NUMBER, True, description="Simulation time"
            ),
            "total-bees": ColumnSchema(
                "total-bees",
                NetLogoDataType.NUMBER,
                False,
                description="Total bee count",
            ),
            "total-energy": ColumnSchema(
                "total-energy",
                NetLogoDataType.NUMBER,
                False,
                description="Total system energy",
            ),
            "temperature": ColumnSchema(
                "temperature",
                NetLogoDataType.NUMBER,
                False,
                valid_range=(-20, 50),
                description="Temperature",
            ),
            "season": ColumnSchema(
                "season",
                NetLogoDataType.STRING,
                False,
                valid_values=["spring", "summer", "autumn", "winter"],
                description="Current season",
            ),
        }

    def read_csv(
        self, file_path: str, csv_type: str = "agents"
    ) -> Tuple[pd.DataFrame, ValidationResult]:
        """Read and validate NetLogo-compatible CSV file"""

        self.logger.info(f"Reading NetLogo CSV file: {file_path} (type: {csv_type})")

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Validate the CSV
            validation_result = self.validate_csv_data(df, csv_type)
            self.last_validation_result = validation_result

            if (
                validation_result.is_valid
                or self.validation_level == CSVValidationLevel.LENIENT
            ):
                # Apply data conversions and fixes
                df = self._apply_data_conversions(df, csv_type, validation_result)

                self.logger.info(
                    f"Successfully read CSV: {validation_result.row_count} rows, {validation_result.column_count} columns"
                )

                return df, validation_result
            else:
                self.logger.error(
                    f"CSV validation failed: {len(validation_result.errors)} errors"
                )
                for error in validation_result.errors:
                    self.logger.error(f"  - {error}")

                return pd.DataFrame(), validation_result

        except Exception as e:
            error_result = ValidationResult(
                is_valid=False,
                errors=[f"Failed to read CSV file: {str(e)}"],
                netlogo_compatible=False,
            )
            return pd.DataFrame(), error_result

    def validate_csv_data(self, df: pd.DataFrame, csv_type: str) -> ValidationResult:
        """Validate CSV data against NetLogo schemas"""

        result = ValidationResult(
            is_valid=True, row_count=len(df), column_count=len(df.columns)
        )

        # Get appropriate schema
        schema = self._get_schema_for_type(csv_type)

        if not schema:
            result.errors.append(f"Unknown CSV type: {csv_type}")
            result.is_valid = False
            return result

        # Validate column presence
        self._validate_required_columns(df, schema, result)

        # Validate data types
        self._validate_data_types(df, schema, result)

        # Validate value ranges and constraints
        self._validate_value_constraints(df, schema, result)

        # Check NetLogo naming conventions
        self._validate_netlogo_conventions(df, result)

        # Validate data consistency
        self._validate_data_consistency(df, csv_type, result)

        # Determine overall validity
        result.is_valid = len(result.errors) == 0
        result.netlogo_compatible = result.is_valid and len(result.warnings) <= 2

        # Generate suggested fixes
        if not result.is_valid:
            result.suggested_fixes = self._generate_suggested_fixes(result, schema)

        return result

    def _get_schema_for_type(self, csv_type: str) -> Optional[Dict[str, ColumnSchema]]:
        """Get schema for CSV type"""

        schema_map = {
            "agents": self.agent_schema,
            "turtles": self.agent_schema,  # NetLogo alias
            "bees": self.agent_schema,  # Domain-specific alias
            "patches": self.patch_schema,
            "globals": self.global_schema,
            "world": self.global_schema,  # NetLogo alias
        }

        return schema_map.get(csv_type.lower())

    def _validate_required_columns(
        self,
        df: pd.DataFrame,
        schema: Dict[str, ColumnSchema],
        result: ValidationResult,
    ) -> None:
        """Validate presence of required columns"""

        required_columns = [col.name for col in schema.values() if col.required]
        missing_columns = [col for col in required_columns if col not in df.columns]

        for col in missing_columns:
            result.errors.append(f"Required column missing: {col}")

        # Check for unknown columns
        unknown_columns = [col for col in df.columns if col not in schema]
        for col in unknown_columns:
            result.warnings.append(f"Unknown column found: {col}")

    def _validate_data_types(
        self,
        df: pd.DataFrame,
        schema: Dict[str, ColumnSchema],
        result: ValidationResult,
    ) -> None:
        """Validate data types for each column"""

        for col_name, col_schema in schema.items():
            if col_name not in df.columns:
                continue

            column = df[col_name]
            expected_type = col_schema.data_type

            # Check for type compatibility
            if expected_type == NetLogoDataType.NUMBER:
                non_numeric = column.apply(
                    lambda x: not self._is_numeric(x) if pd.notna(x) else False
                )
                if non_numeric.any():
                    invalid_count = non_numeric.sum()
                    result.errors.append(
                        f"Column '{col_name}' contains {invalid_count} non-numeric values"
                    )

            elif expected_type == NetLogoDataType.BOOLEAN:
                non_boolean = column.apply(
                    lambda x: not self._is_boolean(x) if pd.notna(x) else False
                )
                if non_boolean.any():
                    invalid_count = non_boolean.sum()
                    result.errors.append(
                        f"Column '{col_name}' contains {invalid_count} non-boolean values"
                    )

            elif expected_type == NetLogoDataType.STRING:
                # Strings are generally flexible, just check for extreme values
                if column.dtype == "object":
                    very_long_strings = column.apply(
                        lambda x: len(str(x)) > 1000 if pd.notna(x) else False
                    )
                    if very_long_strings.any():
                        result.warnings.append(
                            f"Column '{col_name}' contains very long string values"
                        )

    def _validate_value_constraints(
        self,
        df: pd.DataFrame,
        schema: Dict[str, ColumnSchema],
        result: ValidationResult,
    ) -> None:
        """Validate value ranges and constraints"""

        for col_name, col_schema in schema.items():
            if col_name not in df.columns:
                continue

            column = df[col_name]

            # Check valid range
            if (
                col_schema.valid_range
                and col_schema.data_type == NetLogoDataType.NUMBER
            ):
                min_val, max_val = col_schema.valid_range
                numeric_column = pd.to_numeric(column, errors="coerce")

                out_of_range = (numeric_column < min_val) | (numeric_column > max_val)
                out_of_range = out_of_range & numeric_column.notna()

                if out_of_range.any():
                    count = out_of_range.sum()
                    result.warnings.append(
                        f"Column '{col_name}' has {count} values outside range [{min_val}, {max_val}]"
                    )

            # Check valid values
            if col_schema.valid_values:
                invalid_values = ~column.isin(
                    col_schema.valid_values + [col_schema.default_value]
                )
                invalid_values = invalid_values & column.notna()

                if invalid_values.any():
                    count = invalid_values.sum()
                    unique_invalid = column[invalid_values].unique()[
                        :5
                    ]  # Show first 5 invalid values
                    result.warnings.append(
                        f"Column '{col_name}' has {count} invalid values. Examples: {list(unique_invalid)}"
                    )

    def _validate_netlogo_conventions(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Validate NetLogo naming and formatting conventions"""

        # Check column naming conventions
        for col in df.columns:
            # NetLogo typically uses lowercase with hyphens
            if not re.match(r"^[a-z][a-z0-9\-]*$", col):
                result.warnings.append(
                    f"Column '{col}' doesn't follow NetLogo naming convention (lowercase with hyphens)"
                )

            # Check for reserved NetLogo words
            reserved_words = [
                "breed",
                "color",
                "heading",
                "hidden?",
                "label",
                "label-color",
                "pen-mode",
                "pen-size",
                "shape",
                "size",
                "who",
                "xcor",
                "ycor",
            ]
            if col in reserved_words:
                # This is actually good for compatibility
                pass

        # Check for NetLogo-specific data patterns
        if "who" in df.columns:
            who_column = df["who"]
            if who_column.dtype != "int64" and who_column.dtype != "float64":
                result.errors.append("'who' column should contain numeric agent IDs")

            # Check for duplicate IDs
            if who_column.duplicated().any():
                duplicate_count = who_column.duplicated().sum()
                result.errors.append(
                    f"'who' column contains {duplicate_count} duplicate agent IDs"
                )

        # Check coordinate consistency
        if "xcor" in df.columns and "ycor" in df.columns:
            xcor = pd.to_numeric(df["xcor"], errors="coerce")
            ycor = pd.to_numeric(df["ycor"], errors="coerce")

            if xcor.isna().any() or ycor.isna().any():
                result.errors.append(
                    "Coordinate columns (xcor, ycor) contain non-numeric values"
                )

    def _validate_data_consistency(
        self, df: pd.DataFrame, csv_type: str, result: ValidationResult
    ) -> None:
        """Validate internal data consistency"""

        if csv_type.lower() in ["agents", "turtles", "bees"]:
            # Agent-specific consistency checks

            # Energy should be non-negative
            if "energy" in df.columns:
                energy = pd.to_numeric(df["energy"], errors="coerce")
                negative_energy = energy < 0
                if negative_energy.any():
                    count = negative_energy.sum()
                    result.warnings.append(
                        f"{count} agents have negative energy values"
                    )

            # Age consistency
            if "age" in df.columns:
                age = pd.to_numeric(df["age"], errors="coerce")
                unrealistic_age = age > 365  # Bees don't live longer than a year
                if unrealistic_age.any():
                    count = unrealistic_age.sum()
                    result.warnings.append(
                        f"{count} agents have unrealistic age values (>365)"
                    )

            # Status consistency
            if "status" in df.columns and "energy" in df.columns:
                dead_agents = df["status"] == "dead"
                dead_with_energy = dead_agents & (
                    pd.to_numeric(df["energy"], errors="coerce") > 0
                )
                if dead_with_energy.any():
                    count = dead_with_energy.sum()
                    result.warnings.append(f"{count} dead agents still have energy")

        elif csv_type.lower() == "patches":
            # Patch-specific consistency checks

            # Resource amounts should be non-negative
            resource_columns = ["nectar-amount", "pollen-amount"]
            for col in resource_columns:
                if col in df.columns:
                    amounts = pd.to_numeric(df[col], errors="coerce")
                    negative_amounts = amounts < 0
                    if negative_amounts.any():
                        count = negative_amounts.sum()
                        result.warnings.append(f"{count} patches have negative {col}")

    def _apply_data_conversions(
        self, df: pd.DataFrame, csv_type: str, validation_result: ValidationResult
    ) -> pd.DataFrame:
        """Apply data type conversions and handle missing values"""

        if not self.auto_convert_types:
            return df

        schema = self._get_schema_for_type(csv_type)
        if not schema:
            return df

        df_converted = df.copy()

        for col_name, col_schema in schema.items():
            if col_name not in df_converted.columns:
                # Add missing column with default value
                if col_schema.default_value is not None:
                    df_converted[col_name] = col_schema.default_value
                    self.logger.info(
                        f"Added missing column '{col_name}' with default value: {col_schema.default_value}"
                    )
                continue

            column = df_converted[col_name]

            # Handle missing values
            if self.handle_missing_values and column.isna().any():
                if col_schema.default_value is not None:
                    df_converted[col_name] = column.fillna(col_schema.default_value)
                elif col_schema.data_type == NetLogoDataType.NUMBER:
                    df_converted[col_name] = column.fillna(0)
                elif col_schema.data_type == NetLogoDataType.STRING:
                    df_converted[col_name] = column.fillna("")
                elif col_schema.data_type == NetLogoDataType.BOOLEAN:
                    df_converted[col_name] = column.fillna(False)

            # Convert data types
            if col_schema.data_type == NetLogoDataType.NUMBER:
                df_converted[col_name] = pd.to_numeric(
                    df_converted[col_name], errors="coerce"
                )
            elif col_schema.data_type == NetLogoDataType.BOOLEAN:
                df_converted[col_name] = df_converted[col_name].apply(
                    self._convert_to_boolean
                )
            elif col_schema.data_type == NetLogoDataType.STRING:
                df_converted[col_name] = df_converted[col_name].astype(str)

        return df_converted

    def _generate_suggested_fixes(
        self, result: ValidationResult, schema: Dict[str, ColumnSchema]
    ) -> List[str]:
        """Generate suggested fixes for validation errors"""

        fixes = []

        for error in result.errors:
            if "Required column missing" in error:
                col_name = error.split(": ")[-1]
                if col_name in schema:
                    col_schema = schema[col_name]
                    if col_schema.default_value is not None:
                        fixes.append(
                            f"Add column '{col_name}' with default value: {col_schema.default_value}"
                        )
                    else:
                        fixes.append(
                            f"Add required column '{col_name}' ({col_schema.description})"
                        )

            elif "non-numeric values" in error:
                # Try to extract column name from error message
                if "'" in error:
                    try:
                        col_name = error.split("'")[1]
                        fixes.append(
                            f"Convert non-numeric values in column '{col_name}' to numbers or remove invalid rows"
                        )
                    except IndexError:
                        fixes.append(
                            "Convert non-numeric values to numbers or remove invalid rows"
                        )
                else:
                    fixes.append(
                        "Convert non-numeric values to numbers or remove invalid rows"
                    )

            elif "duplicate agent IDs" in error:
                fixes.append(
                    "Remove duplicate agent IDs from 'who' column or reassign unique IDs"
                )

            elif "non-numeric values" in error and "coordinate" in error:
                fixes.append(
                    "Ensure xcor and ycor columns contain only numeric coordinate values"
                )

        return fixes

    def _is_numeric(self, value: Any) -> bool:
        """Check if value can be converted to a number"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _is_boolean(self, value: Any) -> bool:
        """Check if value is boolean or boolean-like"""
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            return value.lower() in ["true", "false", "t", "f", "1", "0", "yes", "no"]
        if isinstance(value, (int, float)):
            return value in [0, 1]
        return False

    def _convert_to_boolean(self, value: Any) -> bool:
        """Convert value to boolean"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ["true", "t", "1", "yes"]
        if isinstance(value, (int, float)):
            return bool(value)
        return False


class NetLogoCSVWriter(BaseModel):
    """Writer for NetLogo-compatible CSV files"""

    model_config = {"validate_assignment": True}

    # Configuration
    include_header: bool = True
    netlogo_precision: int = 6  # Decimal precision for numbers
    boolean_format: str = "true/false"  # or "1/0"
    coordinate_precision: int = 3

    # Formatting options
    date_format: str = "%Y-%m-%d %H:%M:%S"
    null_representation: str = ""

    # Logging
    logger: Any = Field(
        default_factory=lambda: logging.getLogger(__name__), exclude=True
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    def write_csv(
        self, df: pd.DataFrame, file_path: str, csv_type: str = "agents"
    ) -> ValidationResult:
        """Write DataFrame to NetLogo-compatible CSV format"""

        self.logger.info(f"Writing NetLogo CSV file: {file_path} (type: {csv_type})")

        try:
            # Prepare DataFrame for NetLogo format
            df_formatted = self._format_for_netlogo(df, csv_type)

            # Validate before writing
            reader = NetLogoCSVReader()
            validation_result = reader.validate_csv_data(df_formatted, csv_type)

            if (
                validation_result.is_valid
                or reader.validation_level == CSVValidationLevel.LENIENT
            ):
                # Write to CSV
                df_formatted.to_csv(
                    file_path,
                    index=False,
                    header=self.include_header,
                    float_format=f"%.{self.netlogo_precision}f",
                    na_rep=self.null_representation,
                )

                self.logger.info(
                    f"Successfully wrote CSV: {len(df_formatted)} rows, {len(df_formatted.columns)} columns"
                )

                return validation_result
            else:
                self.logger.error(
                    f"Cannot write invalid CSV data: {len(validation_result.errors)} errors"
                )
                return validation_result

        except Exception as e:
            error_result = ValidationResult(
                is_valid=False,
                errors=[f"Failed to write CSV file: {str(e)}"],
                netlogo_compatible=False,
            )
            return error_result

    def _format_for_netlogo(self, df: pd.DataFrame, csv_type: str) -> pd.DataFrame:
        """Format DataFrame for NetLogo compatibility"""

        df_formatted = df.copy()

        # Format numeric columns
        for col in df_formatted.select_dtypes(include=[np.number]).columns:
            if col in ["xcor", "ycor", "pxcor", "pycor"]:
                # Coordinate precision
                df_formatted[col] = df_formatted[col].round(self.coordinate_precision)
            else:
                # General numeric precision
                df_formatted[col] = df_formatted[col].round(self.netlogo_precision)

        # Format boolean columns
        bool_columns = df_formatted.select_dtypes(include=[bool]).columns
        for col in bool_columns:
            if self.boolean_format == "1/0":
                df_formatted[col] = df_formatted[col].astype(int)
            else:
                df_formatted[col] = df_formatted[col].map(
                    {True: "true", False: "false"}
                )

        # Ensure column order follows NetLogo conventions
        df_formatted = self._reorder_columns_for_netlogo(df_formatted, csv_type)

        return df_formatted

    def _reorder_columns_for_netlogo(
        self, df: pd.DataFrame, csv_type: str
    ) -> pd.DataFrame:
        """Reorder columns to match NetLogo conventions"""

        if csv_type.lower() in ["agents", "turtles", "bees"]:
            # Standard NetLogo agent column order
            standard_order = [
                "who",
                "breed",
                "color",
                "heading",
                "xcor",
                "ycor",
                "shape",
                "label",
                "label-color",
                "size",
                "pen-size",
                "pen-mode",
                "hidden?",
            ]

            # Add custom columns
            custom_columns = [col for col in df.columns if col not in standard_order]
            desired_order = [
                col for col in standard_order if col in df.columns
            ] + sorted(custom_columns)

            return df[desired_order]

        elif csv_type.lower() == "patches":
            # Standard NetLogo patch column order
            standard_order = ["pxcor", "pycor", "pcolor", "plabel", "plabel-color"]

            # Add custom columns
            custom_columns = [col for col in df.columns if col not in standard_order]
            desired_order = [
                col for col in standard_order if col in df.columns
            ] + sorted(custom_columns)

            return df[desired_order]

        return df


class NetLogoDataConverter(BaseModel):
    """Converter between BSTEW and NetLogo data formats"""

    model_config = {"validate_assignment": True}

    # Mapping configurations
    breed_mapping: Dict[str, str] = Field(default_factory=dict)
    status_mapping: Dict[str, str] = Field(default_factory=dict)
    coordinate_scale: float = 1.0
    coordinate_offset: Tuple[float, float] = (0.0, 0.0)

    # Logging
    logger: Any = Field(
        default_factory=lambda: logging.getLogger(__name__), exclude=True
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

        # Initialize default mappings
        self._initialize_default_mappings()

    def _initialize_default_mappings(self) -> None:
        """Initialize default value mappings"""

        # Bee breed mapping
        self.breed_mapping = {
            "bombus_terrestris": "bumblebees",
            "apis_mellifera": "honeybees",
            "osmia_rufa": "mason-bees",
            "megachile_rotundata": "leafcutter-bees",
        }

        # Status mapping
        self.status_mapping = {
            "foraging": "foraging",
            "at_hive": "resting",
            "nursing": "nursing",
            "guarding": "guarding",
            "building": "building",
            "dead": "dead",
            "emerging": "emerging",
        }

    def convert_bstew_to_netlogo(
        self, data: Dict[str, Any], data_type: str = "agent"
    ) -> Dict[str, Any]:
        """Convert BSTEW data format to NetLogo format"""

        if data_type == "agent":
            return self._convert_agent_to_netlogo(data)
        elif data_type == "patch":
            return self._convert_patch_to_netlogo(data)
        elif data_type == "global":
            return self._convert_global_to_netlogo(data)
        else:
            self.logger.warning(f"Unknown data type for conversion: {data_type}")
            return data

    def convert_netlogo_to_bstew(
        self, data: Dict[str, Any], data_type: str = "agent"
    ) -> Dict[str, Any]:
        """Convert NetLogo data format to BSTEW format"""

        if data_type == "agent":
            return self._convert_agent_from_netlogo(data)
        elif data_type == "patch":
            return self._convert_patch_from_netlogo(data)
        elif data_type == "global":
            return self._convert_global_from_netlogo(data)
        else:
            self.logger.warning(f"Unknown data type for conversion: {data_type}")
            return data

    def _convert_agent_to_netlogo(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert BSTEW agent data to NetLogo format"""

        netlogo_data = {}

        # Basic NetLogo agent properties
        netlogo_data["who"] = agent_data.get("unique_id", agent_data.get("id", 0))

        # Convert breed
        species = agent_data.get("species", "unknown")
        netlogo_data["breed"] = self.breed_mapping.get(species, species)

        # Convert coordinates
        pos = agent_data.get("pos", agent_data.get("position", (0, 0)))
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = pos[0], pos[1]
            # Apply scaling and offset
            netlogo_data["xcor"] = (x * self.coordinate_scale) + self.coordinate_offset[
                0
            ]
            netlogo_data["ycor"] = (y * self.coordinate_scale) + self.coordinate_offset[
                1
            ]
        else:
            netlogo_data["xcor"] = 0
            netlogo_data["ycor"] = 0

        # Convert heading (BSTEW might use radians, NetLogo uses degrees)
        heading = agent_data.get("heading", 0)
        if heading > 2 * math.pi:  # Assume degrees
            netlogo_data["heading"] = heading % 360
        else:  # Assume radians, convert to degrees
            netlogo_data["heading"] = (heading * 180 / math.pi) % 360

        # Agent properties
        netlogo_data["color"] = agent_data.get("color", 15)  # NetLogo default
        netlogo_data["size"] = agent_data.get("size", 1.0)
        netlogo_data["energy"] = agent_data.get("energy", 100.0)
        netlogo_data["age"] = agent_data.get("age", 0)

        # Convert status
        status = agent_data.get("status", agent_data.get("state", "unknown"))
        netlogo_data["status"] = self.status_mapping.get(status, status)

        # Custom BSTEW properties
        netlogo_data["foraging-experience"] = agent_data.get("foraging_experience", 0)
        netlogo_data["load-nectar"] = agent_data.get("nectar_load", 0.0)
        netlogo_data["load-pollen"] = agent_data.get("pollen_load", 0.0)
        netlogo_data["home-colony"] = agent_data.get("colony_id", 0)

        return netlogo_data

    def _convert_patch_to_netlogo(self, patch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert BSTEW patch data to NetLogo format"""

        netlogo_data = {}

        # Patch coordinates
        pos = patch_data.get("pos", patch_data.get("position", (0, 0)))
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            netlogo_data["pxcor"] = int(pos[0])
            netlogo_data["pycor"] = int(pos[1])
        else:
            netlogo_data["pxcor"] = 0
            netlogo_data["pycor"] = 0

        # Patch properties
        netlogo_data["pcolor"] = patch_data.get("color", 0)
        netlogo_data["flower-density"] = patch_data.get("flower_density", 0.0)
        netlogo_data["nectar-amount"] = patch_data.get("nectar_amount", 0.0)
        netlogo_data["pollen-amount"] = patch_data.get("pollen_amount", 0.0)
        netlogo_data["resource-quality"] = patch_data.get("resource_quality", 0.5)

        # Environmental factors
        netlogo_data["elevation"] = patch_data.get("elevation", 0.0)
        netlogo_data["slope"] = patch_data.get("slope", 0.0)
        netlogo_data["shelter"] = patch_data.get("shelter", 0.0)

        return netlogo_data

    def _convert_global_to_netlogo(self, global_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert BSTEW global data to NetLogo format"""

        netlogo_data = {}

        # Time
        netlogo_data["ticks"] = global_data.get(
            "current_step", global_data.get("current_day", 0)
        )

        # Population metrics
        netlogo_data["total-bees"] = global_data.get("total_population", 0)
        netlogo_data["total-energy"] = global_data.get("total_energy", 0.0)
        netlogo_data["active-colonies"] = global_data.get("active_colonies", 0)

        # Environmental
        netlogo_data["temperature"] = global_data.get("temperature", 20.0)
        netlogo_data["season"] = global_data.get("season", "spring")
        netlogo_data["day-of-year"] = global_data.get("day_of_year", 1)

        # System metrics
        netlogo_data["foraging-efficiency"] = global_data.get(
            "foraging_efficiency", 1.0
        )
        netlogo_data["resource-scarcity"] = global_data.get("resource_scarcity", 0.0)

        return netlogo_data

    def _convert_agent_from_netlogo(
        self, netlogo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert NetLogo agent data to BSTEW format"""

        bstew_data = {}

        # Basic properties
        bstew_data["unique_id"] = netlogo_data.get("who", 0)

        # Convert breed back to species
        breed = netlogo_data.get("breed", "unknown")
        reverse_breed_mapping = {v: k for k, v in self.breed_mapping.items()}
        bstew_data["species"] = reverse_breed_mapping.get(breed, breed)

        # Convert coordinates back
        xcor = netlogo_data.get("xcor", 0)
        ycor = netlogo_data.get("ycor", 0)
        # Reverse scaling and offset
        x = (xcor - self.coordinate_offset[0]) / self.coordinate_scale
        y = (ycor - self.coordinate_offset[1]) / self.coordinate_scale
        bstew_data["pos"] = (x, y)

        # Convert heading back to radians
        heading_degrees = netlogo_data.get("heading", 0)
        bstew_data["heading"] = heading_degrees * math.pi / 180

        # Other properties
        bstew_data["color"] = netlogo_data.get("color", 15)
        bstew_data["size"] = netlogo_data.get("size", 1.0)
        bstew_data["energy"] = netlogo_data.get("energy", 100.0)
        bstew_data["age"] = netlogo_data.get("age", 0)

        # Convert status back
        status = netlogo_data.get("status", "unknown")
        reverse_status_mapping = {v: k for k, v in self.status_mapping.items()}
        bstew_data["status"] = reverse_status_mapping.get(status, status)

        # Custom properties
        bstew_data["foraging_experience"] = netlogo_data.get("foraging-experience", 0)
        bstew_data["nectar_load"] = netlogo_data.get("load-nectar", 0.0)
        bstew_data["pollen_load"] = netlogo_data.get("load-pollen", 0.0)
        bstew_data["colony_id"] = netlogo_data.get("home-colony", 0)

        return bstew_data

    def _convert_patch_from_netlogo(
        self, netlogo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert NetLogo patch data to BSTEW format"""

        bstew_data = {}

        # Coordinates
        pxcor = netlogo_data.get("pxcor", 0)
        pycor = netlogo_data.get("pycor", 0)
        bstew_data["pos"] = (pxcor, pycor)

        # Properties
        bstew_data["color"] = netlogo_data.get("pcolor", 0)
        bstew_data["flower_density"] = netlogo_data.get("flower-density", 0.0)
        bstew_data["nectar_amount"] = netlogo_data.get("nectar-amount", 0.0)
        bstew_data["pollen_amount"] = netlogo_data.get("pollen-amount", 0.0)
        bstew_data["resource_quality"] = netlogo_data.get("resource-quality", 0.5)

        # Environmental
        bstew_data["elevation"] = netlogo_data.get("elevation", 0.0)
        bstew_data["slope"] = netlogo_data.get("slope", 0.0)
        bstew_data["shelter"] = netlogo_data.get("shelter", 0.0)

        return bstew_data

    def _convert_global_from_netlogo(
        self, netlogo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert NetLogo global data to BSTEW format"""

        bstew_data = {}

        # Time
        bstew_data["current_step"] = netlogo_data.get("ticks", 0)
        bstew_data["current_day"] = netlogo_data.get("ticks", 0)

        # Population
        bstew_data["total_population"] = netlogo_data.get("total-bees", 0)
        bstew_data["total_energy"] = netlogo_data.get("total-energy", 0.0)
        bstew_data["active_colonies"] = netlogo_data.get("active-colonies", 0)

        # Environment
        bstew_data["temperature"] = netlogo_data.get("temperature", 20.0)
        bstew_data["season"] = netlogo_data.get("season", "spring")
        bstew_data["day_of_year"] = netlogo_data.get("day-of-year", 1)

        # Metrics
        bstew_data["foraging_efficiency"] = netlogo_data.get("foraging-efficiency", 1.0)
        bstew_data["resource_scarcity"] = netlogo_data.get("resource-scarcity", 0.0)

        return bstew_data


def create_netlogo_export_package(
    model_data: Dict[str, Any], output_directory: str
) -> Dict[str, Any]:
    """Create complete NetLogo-compatible export package"""

    logger = logging.getLogger(__name__)
    logger.info(f"Creating NetLogo export package in: {output_directory}")

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Initialize components
    NetLogoCSVReader()
    writer = NetLogoCSVWriter()
    converter = NetLogoDataConverter()

    export_results: Dict[str, Any] = {
        "success": True,
        "files_created": [],
        "validation_results": {},
        "errors": [],
    }

    try:
        # Export agents/bees
        if "agents" in model_data:
            agents_data = []
            for agent in model_data["agents"]:
                netlogo_agent = converter.convert_bstew_to_netlogo(agent, "agent")
                agents_data.append(netlogo_agent)

            agents_df = pd.DataFrame(agents_data)
            agents_file = os.path.join(output_directory, "agents.csv")

            validation_result = writer.write_csv(agents_df, agents_file, "agents")
            export_results["validation_results"]["agents"] = validation_result

            if validation_result.is_valid:
                export_results["files_created"].append(agents_file)
            else:
                export_results["errors"].extend(validation_result.errors)

        # Export patches
        if "patches" in model_data:
            patches_data = []
            for patch in model_data["patches"]:
                netlogo_patch = converter.convert_bstew_to_netlogo(patch, "patch")
                patches_data.append(netlogo_patch)

            patches_df = pd.DataFrame(patches_data)
            patches_file = os.path.join(output_directory, "patches.csv")

            validation_result = writer.write_csv(patches_df, patches_file, "patches")
            export_results["validation_results"]["patches"] = validation_result

            if validation_result.is_valid:
                export_results["files_created"].append(patches_file)
            else:
                export_results["errors"].extend(validation_result.errors)

        # Export globals
        if "globals" in model_data:
            netlogo_globals = converter.convert_bstew_to_netlogo(
                model_data["globals"], "global"
            )
            globals_df = pd.DataFrame([netlogo_globals])
            globals_file = os.path.join(output_directory, "globals.csv")

            validation_result = writer.write_csv(globals_df, globals_file, "globals")
            export_results["validation_results"]["globals"] = validation_result

            if validation_result.is_valid:
                export_results["files_created"].append(globals_file)
            else:
                export_results["errors"].extend(validation_result.errors)

        # Create metadata file
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "bstew_version": "2.0",
            "netlogo_compatibility": "6.4",
            "coordinate_scale": converter.coordinate_scale,
            "coordinate_offset": converter.coordinate_offset,
            "files_included": [
                os.path.basename(f) for f in export_results["files_created"]
            ],
            "validation_summary": {
                filename: {"valid": result.is_valid, "error_count": len(result.errors)}
                for filename, result in export_results["validation_results"].items()
            },
        }

        metadata_file = os.path.join(output_directory, "export_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        export_results["files_created"].append(metadata_file)

        # Determine overall success
        export_results["success"] = len(export_results["errors"]) == 0

        logger.info(
            f"Export package created: {len(export_results['files_created'])} files"
        )

    except Exception as e:
        export_results["success"] = False
        export_results["errors"].append(f"Export failed: {str(e)}")
        logger.error(f"Failed to create export package: {e}")

    return export_results
