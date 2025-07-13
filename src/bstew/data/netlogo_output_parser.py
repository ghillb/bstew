"""
NetLogo Output Parser for BSTEW
===============================

Parses NetLogo simulation output files and converts them to BSTEW-compatible formats.
Handles BehaviorSpace experiments, reporter outputs, and time-series data.
"""

import pandas as pd
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from pathlib import Path
import logging
from datetime import datetime


class NetLogoRun(BaseModel):
    """Single NetLogo simulation run data"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    run_id: str = Field(description="Unique run identifier")
    experiment_name: str = Field(description="Name of experiment")
    parameters: Dict[str, Any] = Field(description="Run parameters")
    time_series: pd.DataFrame = Field(description="Time series data")
    final_values: Dict[str, Any] = Field(description="Final output values")
    metadata: Dict[str, Any] = Field(description="Run metadata")


class NetLogoExperiment(BaseModel):
    """NetLogo BehaviorSpace experiment data"""

    model_config = {"validate_assignment": True}

    experiment_name: str = Field(description="Experiment name")
    setup_commands: List[str] = Field(description="Setup commands")
    go_commands: List[str] = Field(description="Go commands")
    final_commands: List[str] = Field(description="Final commands")
    time_limit: Optional[int] = Field(
        default=None, description="Time limit for experiment"
    )
    parameters: Dict[str, Any] = Field(description="Experiment parameters")
    reporters: List[str] = Field(description="Reporter variables")
    runs: List[NetLogoRun] = Field(description="Individual simulation runs")
    metadata: Dict[str, Any] = Field(description="Experiment metadata")


class NetLogoReporter(BaseModel):
    """NetLogo reporter data"""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    reporter_name: str = Field(description="Reporter variable name")
    data_type: str = Field(description="Data type of reporter")
    description: str = Field(description="Reporter description")
    time_series: pd.DataFrame = Field(description="Time series data")
    summary_stats: Dict[str, float] = Field(description="Summary statistics")


class NetLogoBehaviorSpaceParser:
    """
    Parses NetLogo BehaviorSpace experiment output files.

    Handles CSV and table format outputs from NetLogo BehaviorSpace experiments.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def parse_behaviorspace_csv(self, filepath: str) -> NetLogoExperiment:
        """Parse BehaviorSpace CSV output file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"BehaviorSpace file not found: {filepath}")

        try:
            # Read the full CSV file
            df = pd.read_csv(filepath)

            # Extract experiment metadata from first rows
            experiment_name = self._extract_experiment_name(filepath)

            # Identify parameter columns vs. reporter columns
            param_cols, reporter_cols = self._identify_columns(df)

            # Group by run
            runs = self._group_by_run(df, param_cols, reporter_cols)

            # Create experiment object
            experiment = NetLogoExperiment(
                experiment_name=experiment_name,
                setup_commands=["setup"],  # Default commands
                go_commands=["go"],
                final_commands=[],
                time_limit=None,
                parameters=self._extract_parameter_space(df, param_cols),
                reporters=reporter_cols,
                runs=runs,
                metadata={
                    "source_file": str(file_path),
                    "total_runs": len(runs),
                    "parsed_date": datetime.now().isoformat(),
                },
            )

            self.logger.info(
                f"Parsed BehaviorSpace experiment with {len(runs)} runs from {filepath}"
            )
            return experiment

        except Exception as e:
            raise ValueError(f"Error parsing BehaviorSpace file {filepath}: {e}")

    def _extract_experiment_name(self, filepath: str) -> str:
        """Extract experiment name from file path"""
        file_path = Path(filepath)

        # Try to extract from filename
        filename = file_path.stem

        # Common patterns for experiment names
        if "experiment" in filename.lower():
            return filename
        elif "behaviorspace" in filename.lower():
            return filename
        else:
            return f"experiment_{filename}"

    def _identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify parameter columns vs. reporter columns"""
        param_cols = []
        reporter_cols = []

        # Common parameter column patterns
        param_patterns = [
            r".*param.*",
            r".*setting.*",
            r".*config.*",
            r".*initial.*",
            r".*max.*",
            r".*min.*",
            r".*rate.*",
            r".*probability.*",
            r".*threshold.*",
        ]

        # Common reporter column patterns
        reporter_patterns = [
            r".*count.*",
            r".*total.*",
            r".*average.*",
            r".*mean.*",
            r".*sum.*",
            r".*population.*",
            r".*colonies.*",
            r".*bees.*",
            r".*honey.*",
            r".*pollen.*",
            r".*step.*",
            r".*day.*",
            r".*time.*",
        ]

        for col in df.columns:
            col_lower = col.lower()

            # Check if it's a parameter column
            is_param = any(re.match(pattern, col_lower) for pattern in param_patterns)

            # Check if it's a reporter column
            is_reporter = any(
                re.match(pattern, col_lower) for pattern in reporter_patterns
            )

            # Special handling for known columns
            if col_lower in [
                "run number",
                "run_number",
                "experiment",
                "step",
                "day",
                "time",
            ]:
                if col_lower in ["step", "day", "time"]:
                    reporter_cols.append(col)
                # Skip run number and experiment columns
                continue

            if is_param and not is_reporter:
                param_cols.append(col)
            elif is_reporter and not is_param:
                reporter_cols.append(col)
            else:
                # Default to reporter if ambiguous
                reporter_cols.append(col)

        return param_cols, reporter_cols

    def _group_by_run(
        self, df: pd.DataFrame, param_cols: List[str], reporter_cols: List[str]
    ) -> List[NetLogoRun]:
        """Group data by simulation run"""
        runs = []

        # Check for run identifier column
        run_col = None
        for col in df.columns:
            if col.lower() in ["run number", "run_number", "run"]:
                run_col = col
                break

        if run_col:
            # Group by run number
            grouped = df.groupby(run_col)

            for run_id, group in grouped:
                run = self._create_run_from_group(
                    str(run_id), group, param_cols, reporter_cols
                )
                runs.append(run)

        else:
            # Assume entire file is one run
            run = self._create_run_from_group("1", df, param_cols, reporter_cols)
            runs.append(run)

        return runs

    def _create_run_from_group(
        self,
        run_id: str,
        group: pd.DataFrame,
        param_cols: List[str],
        reporter_cols: List[str],
    ) -> NetLogoRun:
        """Create NetLogoRun object from grouped data"""

        # Extract parameters (should be constant within run)
        parameters = {}
        for col in param_cols:
            if col in group.columns:
                # Take first value (should be constant)
                parameters[col] = group[col].iloc[0]

        # Extract time series data
        time_series_data = {}
        for col in reporter_cols:
            if col in group.columns:
                time_series_data[col] = group[col].tolist()

        time_series = pd.DataFrame(time_series_data)

        # Extract final values
        final_values = {}
        for col in reporter_cols:
            if col in group.columns and len(group[col]) > 0:
                final_values[f"final_{col}"] = group[col].iloc[-1]

        return NetLogoRun(
            run_id=run_id,
            experiment_name="behaviorspace_experiment",
            parameters=parameters,
            time_series=time_series,
            final_values=final_values,
            metadata={
                "steps": len(group),
                "start_step": group.index[0] if len(group) > 0 else 0,
                "end_step": group.index[-1] if len(group) > 0 else 0,
            },
        )

    def _extract_parameter_space(
        self, df: pd.DataFrame, param_cols: List[str]
    ) -> Dict[str, Any]:
        """Extract parameter space from experiment"""
        parameter_space = {}

        for col in param_cols:
            if col in df.columns:
                unique_values = df[col].unique()
                parameter_space[col] = {
                    "values": unique_values.tolist(),
                    "count": len(unique_values),
                    "type": (
                        str(type(unique_values[0]).__name__)
                        if len(unique_values) > 0
                        else "unknown"
                    ),
                }

        return parameter_space


class NetLogoTableParser:
    """
    Parses NetLogo table format output files.

    Handles simple table outputs from NetLogo simulations.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def parse_table_file(self, filepath: str) -> pd.DataFrame:
        """Parse NetLogo table format file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Table file not found: {filepath}")

        try:
            # Try different separators
            separators = ["\t", ",", " ", ";"]

            for sep in separators:
                try:
                    df = pd.read_csv(filepath, sep=sep)

                    # Check if parsing was successful
                    if len(df.columns) > 1:
                        self.logger.info(
                            f"Parsed table file with separator '{sep}' from {filepath}"
                        )
                        return df

                except Exception:
                    continue

            # If all separators failed, try reading as raw text
            with open(filepath, "r") as f:
                lines = f.readlines()

            # Parse manually
            data = []
            headers = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Split by whitespace
                parts = line.split()

                if headers is None:
                    headers = parts
                else:
                    if len(parts) == len(headers):
                        data.append(parts)

            if headers and data:
                df = pd.DataFrame(data, columns=headers)

                # Try to convert numeric columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except ValueError:
                        pass

                self.logger.info(f"Parsed table file manually from {filepath}")
                return df

            raise ValueError("Could not parse table file with any method")

        except Exception as e:
            raise ValueError(f"Error parsing table file {filepath}: {e}")


class NetLogoReporterParser:
    """
    Parses NetLogo reporter output files.

    Handles individual reporter outputs and time-series data.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def parse_reporter_file(
        self, filepath: str, reporter_name: Optional[str] = None
    ) -> NetLogoReporter:
        """Parse NetLogo reporter output file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Reporter file not found: {filepath}")

        if reporter_name is None:
            reporter_name = file_path.stem

        try:
            # Try to read as CSV first
            try:
                df = pd.read_csv(filepath)
            except Exception:
                # Try as simple text file
                with open(filepath, "r") as f:
                    lines = f.readlines()

                # Parse as simple time series
                values = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            values.append(float(line))
                        except ValueError:
                            pass

                if values:
                    df = pd.DataFrame(
                        {"step": range(len(values)), reporter_name: values}
                    )
                else:
                    raise ValueError("No numeric values found in file")

            # Calculate summary statistics
            if reporter_name in df.columns:
                data_col = df[reporter_name]
                summary_stats = {
                    "count": len(data_col),
                    "mean": float(data_col.mean()),
                    "std": float(data_col.std()),
                    "min": float(data_col.min()),
                    "max": float(data_col.max()),
                    "median": float(data_col.median()),
                    "final_value": (
                        float(data_col.iloc[-1]) if len(data_col) > 0 else 0.0
                    ),
                }
            else:
                summary_stats = {}

            # Determine data type
            if reporter_name in df.columns:
                data_type = str(df[reporter_name].dtype)
            else:
                data_type = "unknown"

            reporter = NetLogoReporter(
                reporter_name=reporter_name,
                data_type=data_type,
                description=f"Reporter data from {file_path.name}",
                time_series=df,
                summary_stats=summary_stats,
            )

            self.logger.info(f"Parsed reporter '{reporter_name}' from {filepath}")
            return reporter

        except Exception as e:
            raise ValueError(f"Error parsing reporter file {filepath}: {e}")


class NetLogoLogParser:
    """
    Parses NetLogo log files and extracts runtime information.

    Handles NetLogo console output and error logs.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def parse_log_file(self, filepath: str) -> Dict[str, Any]:
        """Parse NetLogo log file"""
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Log file not found: {filepath}")

        try:
            log_data: Dict[str, Any] = {
                "file_path": str(file_path),
                "commands": [],
                "errors": [],
                "warnings": [],
                "info": [],
                "runtime_stats": {},
                "metadata": {},
            }

            with open(filepath, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    if not line:
                        continue

                    # Categorize log entries
                    if "error" in line.lower():
                        log_data["errors"].append({"line": line_num, "message": line})
                    elif "warning" in line.lower():
                        log_data["warnings"].append({"line": line_num, "message": line})
                    elif line.startswith("> "):
                        # Command
                        log_data["commands"].append(
                            {"line": line_num, "command": line[2:]}
                        )
                    else:
                        # Info
                        log_data["info"].append({"line": line_num, "message": line})

            # Extract runtime statistics
            log_data["runtime_stats"] = {
                "total_lines": line_num,
                "command_count": len(log_data["commands"]),
                "error_count": len(log_data["errors"]),
                "warning_count": len(log_data["warnings"]),
            }

            self.logger.info(f"Parsed log file from {filepath}")
            return log_data

        except Exception as e:
            raise ValueError(f"Error parsing log file {filepath}: {e}")


class NetLogoOutputParser:
    """
    Main NetLogo output parser that coordinates all sub-parsers.

    Provides a unified interface for parsing all types of NetLogo output files.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Initialize sub-parsers
        self.behaviorspace_parser = NetLogoBehaviorSpaceParser()
        self.table_parser = NetLogoTableParser()
        self.reporter_parser = NetLogoReporterParser()
        self.log_parser = NetLogoLogParser()

    def parse_output_directory(self, output_dir: str) -> Dict[str, Any]:
        """Parse all NetLogo output files in directory"""
        output_path = Path(output_dir)

        if not output_path.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        results: Dict[str, Any] = {
            "experiments": {},
            "tables": {},
            "reporters": {},
            "logs": {},
            "metadata": {
                "source_directory": str(output_path),
                "parsed_date": datetime.now().isoformat(),
                "files_processed": [],
            },
        }

        try:
            # Parse BehaviorSpace files
            bs_files = list(output_path.glob("**/*.csv"))
            for bs_file in bs_files:
                if (
                    "behaviorspace" in bs_file.name.lower()
                    or "experiment" in bs_file.name.lower()
                ):
                    try:
                        experiment = self.behaviorspace_parser.parse_behaviorspace_csv(
                            str(bs_file)
                        )
                        results["experiments"][bs_file.stem] = experiment
                        results["metadata"]["files_processed"].append(str(bs_file))
                    except Exception as e:
                        self.logger.error(
                            f"Error parsing BehaviorSpace file {bs_file}: {e}"
                        )

            # Parse table files
            table_files = list(output_path.glob("**/*.txt"))
            for table_file in table_files:
                if "table" in table_file.name.lower():
                    try:
                        table_data = self.table_parser.parse_table_file(str(table_file))
                        results["tables"][table_file.stem] = table_data
                        results["metadata"]["files_processed"].append(str(table_file))
                    except Exception as e:
                        self.logger.error(f"Error parsing table file {table_file}: {e}")

            # Parse reporter files
            reporter_files = list(output_path.glob("**/*.dat"))
            for reporter_file in reporter_files:
                try:
                    reporter_data = self.reporter_parser.parse_reporter_file(
                        str(reporter_file)
                    )
                    results["reporters"][reporter_file.stem] = reporter_data
                    results["metadata"]["files_processed"].append(str(reporter_file))
                except Exception as e:
                    self.logger.error(
                        f"Error parsing reporter file {reporter_file}: {e}"
                    )

            # Parse log files
            log_files = list(output_path.glob("**/*.log"))
            for log_file in log_files:
                try:
                    log_data = self.log_parser.parse_log_file(str(log_file))
                    results["logs"][log_file.stem] = log_data
                    results["metadata"]["files_processed"].append(str(log_file))
                except Exception as e:
                    self.logger.error(f"Error parsing log file {log_file}: {e}")

            self.logger.info(
                f"Parsed {len(results['metadata']['files_processed'])} output files from {output_dir}"
            )

        except Exception as e:
            self.logger.error(f"Error parsing output directory {output_dir}: {e}")
            raise

        return results

    def convert_to_bstew_format(
        self, netlogo_output: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Convert NetLogo output to BSTEW-compatible format"""
        bstew_format = {}

        # Convert experiments to time-series DataFrames
        for exp_name, experiment in netlogo_output.get("experiments", {}).items():
            for run in experiment.runs:
                # Create standardized column names
                df = run.time_series.copy()

                # Rename columns to BSTEW standard
                column_mapping = {
                    "step": "Day",
                    "day": "Day",
                    "time": "Day",
                    "total_bees": "Total_Bees",
                    "total_colonies": "Total_Colonies",
                    "total_honey": "Total_Honey",
                    "total_pollen": "Total_Pollen",
                    "total_brood": "Total_Brood",
                    "active_colonies": "Active_Colonies",
                    "dead_colonies": "Dead_Colonies",
                }

                # Apply column mapping
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df = df.rename(columns={old_name: new_name})

                # Add metadata columns
                df["Run_ID"] = run.run_id
                df["Experiment"] = exp_name

                # Add to results
                bstew_format[f"{exp_name}_run_{run.run_id}"] = df

        # Convert reporter data
        for reporter_name, reporter in netlogo_output.get("reporters", {}).items():
            df = reporter.time_series.copy()

            # Ensure standard time column
            if "step" not in df.columns and "Day" not in df.columns:
                df["Day"] = range(len(df))

            bstew_format[f"reporter_{reporter_name}"] = df

        return bstew_format

    def export_parsed_output(
        self, parsed_output: Dict[str, Any], output_dir: str
    ) -> str:
        """Export parsed output data to JSON format"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable_data = self._make_serializable(parsed_output)

        output_file = output_path / "netlogo_output_parsed.json"

        try:
            with open(output_file, "w") as f:
                json.dump(serializable_data, f, indent=2, default=str)

            self.logger.info(f"Exported parsed output to {output_file}")
            return str(output_file)

        except Exception as e:
            raise ValueError(f"Error exporting parsed output: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert complex objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


def parse_netlogo_output(
    output_dir: str, export_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to parse NetLogo output directory.

    Args:
        output_dir: Directory containing NetLogo output files
        export_dir: Optional directory to export parsed data

    Returns:
        Dictionary containing all parsed output data
    """
    parser = NetLogoOutputParser()

    # Parse all output files
    parsed_output = parser.parse_output_directory(output_dir)

    # Export if output directory specified
    if export_dir:
        parser.export_parsed_output(parsed_output, export_dir)

    return parsed_output
