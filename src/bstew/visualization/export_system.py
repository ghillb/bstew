"""
Visualization Export System for BSTEW
====================================

Advanced visualization export capabilities with animation generation,
report generation with visuals, and publication-ready figure export.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime
import json
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from jinja2 import Template
import zipfile
import shutil
from dataclasses import dataclass
from enum import Enum

from .display_system import DisplayStateManager
from .agent_visualization import AgentVisualizationEngine, VisualizationFrame
from .interactive_analysis import DataExplorationEngine


class ExportFormat(Enum):
    """Export format options"""

    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    MP4 = "mp4"
    GIF = "gif"
    ZIP = "zip"


class ExportQuality(Enum):
    """Export quality settings"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PUBLICATION = "publication"


@dataclass
class ExportConfiguration:
    """Configuration for export operations"""

    format: ExportFormat
    quality: ExportQuality
    dpi: int = 300
    width: int = 1200
    height: int = 800
    include_metadata: bool = True
    include_timestamp: bool = True
    compress: bool = False
    transparent_background: bool = False
    custom_styling: Optional[Dict[str, Any]] = None


class VisualizationExporter:
    """Main visualization export system"""

    def __init__(
        self,
        display_manager: DisplayStateManager,
        viz_engine: AgentVisualizationEngine,
        data_engine: DataExplorationEngine,
    ):
        self.display_manager = display_manager
        self.viz_engine = viz_engine
        self.data_engine = data_engine
        self.logger = logging.getLogger(__name__)

        # Export configurations by quality
        self.quality_configs = {
            ExportQuality.LOW: {
                "dpi": 72,
                "width": 800,
                "height": 600,
                "compression": True,
            },
            ExportQuality.MEDIUM: {
                "dpi": 150,
                "width": 1200,
                "height": 800,
                "compression": True,
            },
            ExportQuality.HIGH: {
                "dpi": 300,
                "width": 1920,
                "height": 1080,
                "compression": False,
            },
            ExportQuality.PUBLICATION: {
                "dpi": 600,
                "width": 3000,
                "height": 2000,
                "compression": False,
            },
        }

        # Style configurations
        self.publication_style = {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "lines.linewidth": 2,
            "patch.linewidth": 0.5,
            "axes.linewidth": 1.5,
        }

    def export_static_visualization(
        self,
        agents: List[Any],
        patches: Optional[List[Any]] = None,
        output_path: Union[str, Path] = "artifacts/exports/visualization",
        config: Optional[ExportConfiguration] = None,
    ) -> Dict[str, str]:
        """Export static visualization in multiple formats"""

        if config is None:
            config = ExportConfiguration(
                format=ExportFormat.PNG, quality=ExportQuality.HIGH
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply quality settings
        quality_settings = self.quality_configs[config.quality]

        # Create visualization
        fig = self.viz_engine.create_static_visualization(
            agents=agents, patches=patches, title="BSTEW Agent Visualization"
        )

        # Apply styling
        self._apply_export_styling(fig, config)

        exported_files = {}

        # Export in requested format
        if config.format == ExportFormat.PNG:
            png_path = output_path.with_suffix(".png")
            fig.savefig(
                png_path,
                dpi=quality_settings["dpi"],
                bbox_inches="tight",
                transparent=config.transparent_background,
                facecolor="white" if not config.transparent_background else "none",
            )
            exported_files["png"] = str(png_path)

        elif config.format == ExportFormat.SVG:
            svg_path = output_path.with_suffix(".svg")
            fig.savefig(
                svg_path,
                format="svg",
                bbox_inches="tight",
                transparent=config.transparent_background,
            )
            exported_files["svg"] = str(svg_path)

        elif config.format == ExportFormat.PDF:
            pdf_path = output_path.with_suffix(".pdf")
            fig.savefig(
                pdf_path,
                format="pdf",
                bbox_inches="tight",
                transparent=config.transparent_background,
            )
            exported_files["pdf"] = str(pdf_path)

        # Export metadata if requested
        if config.include_metadata:
            metadata = self._generate_metadata(agents, patches, config)
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            exported_files["metadata"] = str(metadata_path)

        plt.close(fig)

        self.logger.info(
            f"Static visualization exported to {len(exported_files)} files"
        )
        return exported_files

    def export_animation(
        self,
        frame_data: List[VisualizationFrame],
        output_path: Union[str, Path] = "artifacts/exports/animation",
        config: Optional[ExportConfiguration] = None,
    ) -> Dict[str, str]:
        """Export animation in multiple formats"""

        if config is None:
            config = ExportConfiguration(
                format=ExportFormat.MP4, quality=ExportQuality.MEDIUM
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Create animation
        anim = self.viz_engine.create_animated_visualization(frame_data)

        if config.format == ExportFormat.MP4:
            mp4_path = output_path.with_suffix(".mp4")
            anim.save(str(mp4_path), writer="ffmpeg", fps=5, bitrate=1800)
            exported_files["mp4"] = str(mp4_path)

        elif config.format == ExportFormat.GIF:
            gif_path = output_path.with_suffix(".gif")
            anim.save(str(gif_path), writer="pillow", fps=2)
            exported_files["gif"] = str(gif_path)

        # Export individual frames if requested
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frame_data):
            frame_fig = self.viz_engine.create_static_visualization(
                agents=frame.agents,
                patches=frame.patches,
                title=f"Frame {i + 1} - {frame.timestamp.strftime('%H:%M:%S')}",
            )

            frame_path = frames_dir / f"frame_{i:04d}.png"
            frame_fig.savefig(frame_path, dpi=150, bbox_inches="tight")
            plt.close(frame_fig)

        exported_files["frames_directory"] = str(frames_dir)

        # Export frame metadata
        frame_metadata = []
        for i, frame in enumerate(frame_data):
            frame_metadata.append(
                {
                    "frame_index": i,
                    "timestamp": frame.timestamp.isoformat(),
                    "n_agents": len(frame.agents),
                    "n_patches": len(frame.patches),
                    "metadata": frame.metadata,
                }
            )

        metadata_path = output_path.parent / f"{output_path.stem}_frames.json"
        with open(metadata_path, "w") as f:
            json.dump(frame_metadata, f, indent=2)
        exported_files["frame_metadata"] = str(metadata_path)

        self.logger.info(f"Animation exported to {len(exported_files)} files")
        return exported_files

    def export_interactive_plots(
        self,
        data_sources: Dict[str, Any],
        output_path: Union[str, Path] = "artifacts/exports/interactive",
        config: Optional[ExportConfiguration] = None,
    ) -> Dict[str, str]:
        """Export interactive plots"""

        if config is None:
            config = ExportConfiguration(
                format=ExportFormat.HTML, quality=ExportQuality.HIGH
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load data into engine
        self.data_engine.load_data(data_sources)

        exported_files = {}

        # Create various interactive plots
        for source_name, data in data_sources.items():
            if isinstance(data, (list, pd.DataFrame)):
                # Convert to DataFrame if needed
                df = pd.DataFrame(data) if isinstance(data, list) else data

                # Create different plot types
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                if len(numeric_cols) >= 2:
                    # Scatter plot
                    scatter_fig = self.data_engine.create_interactive_plot(
                        source_name,
                        "scatter",
                        numeric_cols[0],
                        numeric_cols[1],
                        categorical_cols[0] if categorical_cols else None,
                    )

                    # Time series plot if timestamp column exists
                    time_cols = [
                        col
                        for col in df.columns
                        if "time" in col.lower() or "date" in col.lower()
                    ]
                    if time_cols and len(numeric_cols) >= 1:
                        ts_fig = self.data_engine.create_time_series_plot(
                            source_name, time_cols[0], numeric_cols[:3]
                        )

                        ts_path = output_path.parent / f"{source_name}_timeseries.html"
                        pio.write_html(ts_fig, str(ts_path))
                        exported_files[f"{source_name}_timeseries"] = str(ts_path)

                    # Correlation matrix
                    if len(numeric_cols) > 2:
                        corr_fig = self.data_engine.create_correlation_matrix(
                            source_name
                        )
                        corr_path = (
                            output_path.parent / f"{source_name}_correlation.html"
                        )
                        pio.write_html(corr_fig, str(corr_path))
                        exported_files[f"{source_name}_correlation"] = str(corr_path)

                    # Export main scatter plot
                    scatter_path = output_path.parent / f"{source_name}_scatter.html"
                    pio.write_html(scatter_fig, str(scatter_path))
                    exported_files[f"{source_name}_scatter"] = str(scatter_path)

        # Create combined dashboard HTML
        dashboard_html = self._create_dashboard_html(exported_files, data_sources)
        dashboard_path = output_path.with_suffix(".html")
        with open(dashboard_path, "w") as f:
            f.write(dashboard_html)
        exported_files["dashboard"] = str(dashboard_path)

        self.logger.info(f"Interactive plots exported to {len(exported_files)} files")
        return exported_files

    def generate_comprehensive_report(
        self,
        agents: List[Any],
        patches: Optional[List[Any]] = None,
        analysis_data: Optional[Dict[str, Any]] = None,
        output_path: str = "artifacts/exports/report",
        config: Optional[ExportConfiguration] = None,
    ) -> str:
        """Generate comprehensive analysis report with visualizations"""

        if config is None:
            config = ExportConfiguration(
                format=ExportFormat.PDF, quality=ExportQuality.PUBLICATION
            )

        output_path_obj = Path(output_path).with_suffix(".pdf")
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Create PDF document
        doc = SimpleDocTemplate(str(output_path_obj), pagesize=A4)
        styles = getSampleStyleSheet()
        story: List[Any] = []

        # Title page
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
        )

        story.append(Paragraph("BSTEW Analysis Report", title_style))
        story.append(Spacer(1, 20))

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
        story.append(Spacer(1, 40))

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles["Heading2"]))

        summary_text = f"""
        This report presents a comprehensive analysis of the BSTEW simulation results.
        The analysis includes {len(agents)} agents across various species and roles.
        """

        if patches:
            summary_text += f" The landscape contains {len(patches)} patches with varying habitat qualities."

        story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 20))

        # Agent Statistics
        story.append(Paragraph("Agent Statistics", styles["Heading2"]))

        # Create statistics table
        stats_data = self._calculate_agent_statistics(agents)
        table_data = [["Metric", "Value"]]
        for key, value in stats_data.items():
            table_data.append([key.replace("_", " ").title(), str(value)])

        table = Table(table_data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 20))

        # Generate and add visualizations
        story.append(Paragraph("Visualizations", styles["Heading2"]))

        # Main visualization
        main_fig = self.viz_engine.create_static_visualization(
            agents=agents, patches=patches, title="Agent Distribution"
        )

        # Apply publication styling
        plt.rcParams.update(self.publication_style)

        # Save figure to buffer
        img_buffer = io.BytesIO()
        main_fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)

        # Add to report
        img = RLImage(img_buffer, width=6 * inch, height=4 * inch)
        story.append(img)
        story.append(Spacer(1, 20))

        plt.close(main_fig)

        # Species distribution chart
        species_fig = self._create_species_distribution_chart(agents)
        species_buffer = io.BytesIO()
        species_fig.savefig(species_buffer, format="png", dpi=300, bbox_inches="tight")
        species_buffer.seek(0)

        story.append(Paragraph("Species Distribution", styles["Heading3"]))
        species_img = RLImage(species_buffer, width=5 * inch, height=3 * inch)
        story.append(species_img)
        story.append(Spacer(1, 20))

        plt.close(species_fig)

        # Add analysis data if provided
        if analysis_data:
            story.append(Paragraph("Additional Analysis", styles["Heading2"]))

            for analysis_name, data in analysis_data.items():
                story.append(
                    Paragraph(
                        analysis_name.replace("_", " ").title(), styles["Heading3"]
                    )
                )

                if isinstance(data, dict):
                    # Create table for dictionary data
                    analysis_table_data = [["Parameter", "Value"]]
                    for key, value in data.items():
                        analysis_table_data.append([str(key), str(value)])

                    analysis_table = Table(analysis_table_data)
                    analysis_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ]
                        )
                    )

                    story.append(analysis_table)
                    story.append(Spacer(1, 10))

        # Build PDF
        doc.build(story)

        self.logger.info(f"Comprehensive report generated: {output_path_obj}")
        return str(output_path_obj)

    def export_publication_figures(
        self,
        agents: List[Any],
        patches: Optional[List[Any]] = None,
        output_directory: Union[str, Path] = "artifacts/exports/publication",
        formats: Optional[List[ExportFormat]] = None,
    ) -> Dict[str, List[str]]:
        """Export publication-ready figures in multiple formats"""

        if formats is None:
            formats = [ExportFormat.PNG, ExportFormat.SVG, ExportFormat.PDF]

        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Apply publication styling
        plt.rcParams.update(self.publication_style)

        exported_files: Dict[str, List[str]] = {fmt.value: [] for fmt in formats}

        # Figure 1: Agent distribution overview
        fig1 = self.viz_engine.create_static_visualization(
            agents=agents, patches=patches, title="Spatial Distribution of Bee Agents"
        )

        for fmt in formats:
            fig1_path = output_dir / f"figure1_agent_distribution.{fmt.value}"
            if fmt == ExportFormat.PNG:
                fig1.savefig(fig1_path, dpi=600, bbox_inches="tight", facecolor="white")
            elif fmt == ExportFormat.SVG:
                fig1.savefig(fig1_path, format="svg", bbox_inches="tight")
            elif fmt == ExportFormat.PDF:
                fig1.savefig(fig1_path, format="pdf", bbox_inches="tight")
            exported_files[fmt.value].append(str(fig1_path))

        plt.close(fig1)

        # Figure 2: Species composition
        fig2 = self._create_species_distribution_chart(agents)
        fig2.suptitle(
            "Species Composition in BSTEW Simulation", fontsize=16, fontweight="bold"
        )

        for fmt in formats:
            fig2_path = output_dir / f"figure2_species_composition.{fmt.value}"
            if fmt == ExportFormat.PNG:
                fig2.savefig(fig2_path, dpi=600, bbox_inches="tight", facecolor="white")
            elif fmt == ExportFormat.SVG:
                fig2.savefig(fig2_path, format="svg", bbox_inches="tight")
            elif fmt == ExportFormat.PDF:
                fig2.savefig(fig2_path, format="pdf", bbox_inches="tight")
            exported_files[fmt.value].append(str(fig2_path))

        plt.close(fig2)

        # Figure 3: Activity distribution
        fig3 = self._create_activity_distribution_chart(agents)
        fig3.suptitle(
            "Activity Distribution Across Agent Population",
            fontsize=16,
            fontweight="bold",
        )

        for fmt in formats:
            fig3_path = output_dir / f"figure3_activity_distribution.{fmt.value}"
            if fmt == ExportFormat.PNG:
                fig3.savefig(fig3_path, dpi=600, bbox_inches="tight", facecolor="white")
            elif fmt == ExportFormat.SVG:
                fig3.savefig(fig3_path, format="svg", bbox_inches="tight")
            elif fmt == ExportFormat.PDF:
                fig3.savefig(fig3_path, format="pdf", bbox_inches="tight")
            exported_files[fmt.value].append(str(fig3_path))

        plt.close(fig3)

        # Create figure captions
        captions = {
            "figure1_agent_distribution": "Spatial distribution of bee agents across the landscape. Different colors represent different species, with point size indicating colony size.",
            "figure2_species_composition": "Relative abundance of different bee species in the simulation. Data shows the proportion of each species in the total population.",
            "figure3_activity_distribution": "Distribution of activities across the agent population. Activities include foraging, nursing, building, and other behavioral states.",
        }

        captions_path = output_dir / "figure_captions.json"
        with open(captions_path, "w") as f:
            json.dump(captions, f, indent=2)

        # Reset matplotlib parameters
        plt.rcdefaults()

        total_files = sum(len(files) for files in exported_files.values())
        self.logger.info(
            f"Publication figures exported: {total_files} files across {len(formats)} formats"
        )

        return exported_files

    def create_export_package(
        self,
        agents: List[Any],
        patches: Optional[List[Any]] = None,
        frame_data: Optional[List[VisualizationFrame]] = None,
        analysis_data: Optional[Dict[str, Any]] = None,
        output_path: str = "artifacts/exports/complete_package.zip",
    ) -> str:
        """Create complete export package with all visualizations and data"""

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for package contents
        temp_dir = output_path_obj.parent / "temp_export"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Export static visualizations
            static_dir = temp_dir / "static_visualizations"
            static_dir.mkdir(exist_ok=True)

            for quality in [ExportQuality.MEDIUM, ExportQuality.HIGH]:
                quality_dir = static_dir / quality.value
                quality_dir.mkdir(exist_ok=True)

                config = ExportConfiguration(format=ExportFormat.PNG, quality=quality)
                self.export_static_visualization(
                    agents, patches, str(quality_dir / "visualization"), config
                )

            # Export publication figures
            pub_dir = temp_dir / "publication_figures"
            self.export_publication_figures(agents, patches, str(pub_dir))

            # Export animations if frame data provided
            if frame_data:
                anim_dir = temp_dir / "animations"
                anim_dir.mkdir(exist_ok=True)

                self.export_animation(
                    frame_data, str(anim_dir / "simulation_animation")
                )

            # Export interactive plots
            interactive_dir = temp_dir / "interactive_plots"
            data_sources = {"agents": agents}
            if patches:
                data_sources["patches"] = patches

            self.export_interactive_plots(
                data_sources, str(interactive_dir / "dashboard")
            )

            # Generate comprehensive report
            _report_path = self.generate_comprehensive_report(
                agents, patches, analysis_data, str(temp_dir / "comprehensive_report")
            )

            # Export raw data
            data_dir = temp_dir / "raw_data"
            data_dir.mkdir(exist_ok=True)

            # Agents data
            agents_data = []
            for agent in agents:
                agent_dict = {}
                for attr in ["x", "y", "species", "role", "status", "health"]:
                    if hasattr(agent, attr):
                        agent_dict[attr] = getattr(agent, attr)
                agents_data.append(agent_dict)

            agents_df = pd.DataFrame(agents_data)
            agents_df.to_csv(data_dir / "agents_data.csv", index=False)

            # Patches data
            if patches:
                patches_data = []
                for patch in patches:
                    patch_dict = {}
                    for attr in ["x", "y", "patch_type", "area", "habitat_quality"]:
                        if hasattr(patch, attr):
                            value = getattr(patch, attr)
                            if hasattr(value, "value"):  # Handle enums
                                value = value.value
                            patch_dict[attr] = value
                    patches_data.append(patch_dict)

                patches_df = pd.DataFrame(patches_data)
                patches_df.to_csv(data_dir / "patches_data.csv", index=False)

            # Analysis data
            if analysis_data:
                with open(data_dir / "analysis_data.json", "w") as f:
                    json.dump(analysis_data, f, indent=2, default=str)

            # Create package metadata
            metadata = {
                "package_created": datetime.now().isoformat(),
                "n_agents": len(agents),
                "n_patches": len(patches) if patches else 0,
                "n_frames": len(frame_data) if frame_data else 0,
                "contents": {
                    "static_visualizations": True,
                    "publication_figures": True,
                    "animations": frame_data is not None,
                    "interactive_plots": True,
                    "comprehensive_report": True,
                    "raw_data": True,
                },
                "display_configuration": {
                    "color_scheme": self.display_manager.current_color_scheme.value,
                    "active_toggles": {
                        toggle.value: self.display_manager.get_toggle_state(toggle)
                        for toggle in self.display_manager.current_toggles
                    },
                },
            }

            with open(temp_dir / "package_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Create README
            readme_content = self._generate_package_readme(metadata)
            with open(temp_dir / "README.md", "w") as f:
                f.write(readme_content)

            # Create ZIP package
            with zipfile.ZipFile(output_path_obj, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.logger.info(f"Complete export package created: {output_path_obj}")
        return str(output_path_obj)

    def _apply_export_styling(
        self, fig: plt.Figure, config: ExportConfiguration
    ) -> None:
        """Apply styling for export"""

        if config.quality == ExportQuality.PUBLICATION:
            plt.rcParams.update(self.publication_style)

        if config.custom_styling:
            plt.rcParams.update(config.custom_styling)

    def _generate_metadata(
        self,
        agents: List[Any],
        patches: Optional[List[Any]],
        config: ExportConfiguration,
    ) -> Dict[str, Any]:
        """Generate metadata for exported visualization"""

        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "export_configuration": {
                "format": config.format.value,
                "quality": config.quality.value,
                "dpi": config.dpi,
                "dimensions": [config.width, config.height],
            },
            "data_summary": {
                "n_agents": len(agents),
                "n_patches": len(patches) if patches else 0,
            },
            "display_settings": {
                "color_scheme": self.display_manager.current_color_scheme.value,
                "active_toggles": {
                    toggle.value: self.display_manager.get_toggle_state(toggle)
                    for toggle in self.display_manager.current_toggles
                },
            },
        }

        # Add agent statistics
        if agents:
            metadata["agent_statistics"] = self._calculate_agent_statistics(agents)

        return metadata

    def _calculate_agent_statistics(self, agents: List[Any]) -> Dict[str, Any]:
        """Calculate statistics for agents"""

        stats: Dict[str, Any] = {
            "total_agents": len(agents),
            "species_counts": {},
            "role_counts": {},
            "status_counts": {},
        }

        for agent in agents:
            # Species
            species = getattr(agent, "species", "unknown")
            stats["species_counts"][species] = (
                stats["species_counts"].get(species, 0) + 1
            )

            # Role
            role = getattr(agent, "role", "unknown")
            stats["role_counts"][role] = stats["role_counts"].get(role, 0) + 1

            # Status
            status = str(getattr(agent, "status", "unknown"))
            stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1

        return stats

    def _create_species_distribution_chart(self, agents: List[Any]) -> plt.Figure:
        """Create species distribution chart"""

        # Calculate species counts
        species_counts: Dict[str, int] = {}
        for agent in agents:
            species = getattr(agent, "species", "unknown")
            species_counts[species] = species_counts.get(species, 0) + 1

        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))

        species = list(species_counts.keys())
        counts = list(species_counts.values())

        pie_result = ax.pie(counts, labels=species, autopct="%1.1f%%", startangle=90)
        # Unpack safely - pie() may return 2 or 3 values depending on autopct
        if len(pie_result) == 3:
            wedges, texts, _autotexts = pie_result
        else:
            wedges, texts = pie_result

        ax.set_title("Species Distribution", fontsize=14, fontweight="bold")

        return fig

    def _create_activity_distribution_chart(self, agents: List[Any]) -> plt.Figure:
        """Create activity distribution chart"""

        # Calculate activity counts
        activity_counts: Dict[str, int] = {}
        for agent in agents:
            activity = str(getattr(agent, "status", "unknown"))
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        activities = list(activity_counts.keys())
        counts = list(activity_counts.values())

        _bars = ax.bar(activities, counts)

        ax.set_xlabel("Activity", fontsize=12)
        ax.set_ylabel("Number of Agents", fontsize=12)
        ax.set_title("Activity Distribution", fontsize=14, fontweight="bold")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        return fig

    def _create_dashboard_html(
        self, exported_files: Dict[str, str], data_sources: Dict[str, Any]
    ) -> str:
        """Create HTML dashboard combining all interactive plots"""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BSTEW Interactive Analysis Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .plot-container { margin: 20px 0; }
                .plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                iframe { width: 100%; height: 500px; border: 1px solid #ccc; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BSTEW Interactive Analysis Dashboard</h1>
                <p>Generated on {{ timestamp }}</p>
            </div>

            <div class="plot-grid">
                {% for name, path in plots.items() %}
                <div class="plot-container">
                    <h3>{{ name|title }}</h3>
                    <iframe src="{{ path|basename }}"></iframe>
                </div>
                {% endfor %}
            </div>

            <div class="summary">
                <h2>Data Summary</h2>
                <ul>
                    {% for source, data in data_summary.items() %}
                    <li><strong>{{ source }}:</strong> {{ data|length }} records</li>
                    {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """

        template = Template(html_template)

        # Filter HTML files
        html_plots = {k: v for k, v in exported_files.items() if v.endswith(".html")}

        return template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            plots=html_plots,
            data_summary=data_sources,
        )

    def _generate_package_readme(self, metadata: Dict[str, Any]) -> str:
        """Generate README for export package"""

        readme_content = f"""# BSTEW Visualization Export Package

This package contains comprehensive visualizations and analysis results from BSTEW simulation.

## Package Contents

- **static_visualizations/**: High-quality static plots in multiple resolutions
- **publication_figures/**: Publication-ready figures in PNG, SVG, and PDF formats
- **interactive_plots/**: Interactive HTML visualizations
- **raw_data/**: CSV files with simulation data
- **comprehensive_report.pdf**: Complete analysis report

## Data Summary

- **Agents**: {metadata["n_agents"]} agents analyzed
- **Patches**: {metadata["n_patches"]} landscape patches
- **Created**: {metadata["package_created"]}

## Usage

1. View static visualizations in the `static_visualizations/` directory
2. Open interactive plots by opening HTML files in a web browser
3. Use publication figures for research papers and presentations
4. Refer to the comprehensive report for detailed analysis

## Display Configuration

- **Color Scheme**: {metadata["display_configuration"]["color_scheme"]}
- **Active Display Toggles**: {", ".join(k for k, v in metadata["display_configuration"]["active_toggles"].items() if v)}

Generated by BSTEW Visualization Export System
"""

        return readme_content
