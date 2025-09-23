#!/usr/bin/env python3
"""
Batch Plot Generator for ProteoGyver

This module generates plots from batch pipeline output using the same plotting functions
as the GUI, producing figures in identical folder structure with multiple export formats.

The module can process both proteomics and interactomics batch outputs and generate:
- PNG files for web viewing
- PDF files for publications
- HTML files for interactive viewing
- PDF-only for special plots like supervenn

Key Features:
- Uses the same plotting functions as the GUI for consistency
- Generates plots in the same directory structure as GUI export
- Supports both proteomics and interactomics workflows
- Handles multiple export formats based on plot type
- Maintains the same figure quality and styling
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from io import StringIO

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import plotting modules
from components.figures import (
    bar_graph, histogram, volcano_plot, scatter, tic_graph,
    heatmaps, comparative_plot, cvplot, before_after_plot,
    reproducibility_graph, imputation_histogram, commonality_graph,
    network_plot
)
from components.infra import figure_export_directories
from components import figure_functions
import plotly.graph_objects as go
import plotly.io as pio
from plotly import express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Configure logging
logger = logging.getLogger(__name__)

# Default figure settings (matching GUI defaults)
DEFAULT_FIGURE_SETTINGS = {
    'height': 600,
    'width': 800,
    'config': {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
}

# Export formats for different plot types
EXPORT_FORMATS = {
    'plotly': ['html', 'pdf', 'png'],  # Standard plotly figures
    'matplotlib': ['pdf'],              # Matplotlib figures (supervenn)
    'image': ['png']                   # Pre-rendered images
}


class BatchPlotGenerator:
    """Generates plots from batch pipeline output using GUI plotting functions."""
    
    def __init__(self, batch_output_dir: str, export_base_dir: str, 
                 session_name: str = None, figure_settings: Dict = None):
        """Initialize the plot generator.
        
        Args:
            batch_output_dir: Directory containing batch pipeline output
            export_base_dir: Base directory for figure export
            session_name: Optional session name for export directory
            figure_settings: Custom figure settings (uses defaults if None)
        """
        self.batch_output_dir = batch_output_dir
        self.export_base_dir = export_base_dir
        self.session_name = session_name or f"batch-plots-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        self.figure_settings = figure_settings or DEFAULT_FIGURE_SETTINGS.copy()
        
        # Create export directory structure
        self.figure_export_dir = os.path.join(export_base_dir, "Figures")
        os.makedirs(self.figure_export_dir, exist_ok=True)
        
        # Load batch data
        self.data = self._load_batch_data()
        self.workflow = self._detect_workflow()
        
        logger.info(f"Initialized BatchPlotGenerator for {self.workflow} workflow")
        logger.info(f"Export directory: {self.figure_export_dir}")
    
    def _load_batch_data(self) -> Dict[str, Any]:
        """Load all JSON data files from batch output directory."""
        data = {}
        
        for filename in os.listdir(self.batch_output_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.batch_output_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        file_data = json.load(f)
                        # Store with both filename and clean key
                        data[filename] = file_data
                        clean_key = filename.replace('.json', '')
                        data[clean_key] = file_data
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    
        logger.info(f"Loaded {len(data)//2} data files from batch output")
        return data
    
    def _detect_workflow(self) -> str:
        """Detect workflow type from available data files."""
        if any('saint' in key for key in self.data.keys()):
            return 'interactomics'
        elif any('volcano' in key for key in self.data.keys()):
            return 'proteomics'
        else:
            # Fallback to checking data dictionary
            data_dict = self.data.get('01_data_dictionary', {})
            if 'data tables' in data_dict:
                if any('saint' in key.lower() for key in data_dict['data tables'].keys()):
                    return 'interactomics'
            return 'proteomics'  # Default assumption
    
    def _save_figure(self, figure: Any, fig_name: str, subdir: str, 
                    fig_type: str = 'plotly', formats: List[str] = None) -> Dict[str, str]:
        """Save figure in multiple formats.
        
        Args:
            figure: The figure object (plotly Figure, matplotlib figure, or image data)
            fig_name: Name for the saved figure
            subdir: Subdirectory name for organization
            fig_type: Type of figure ('plotly', 'matplotlib', 'image')
            formats: List of formats to save (uses defaults if None)
            
        Returns:
            Dictionary mapping format to saved file path
        """
        if formats is None:
            formats = EXPORT_FORMATS.get(fig_type, ['png'])
            
        # Create target directory
        target_dir = os.path.join(self.figure_export_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
        
        saved_files = {}
        
        for fmt in formats:
            filename = f"{fig_name}.{fmt}"
            filepath = os.path.join(target_dir, filename)
            
            try:
                if fig_type == 'plotly' and isinstance(figure, go.Figure):
                    if fmt == 'html':
                        # Save as standalone HTML
                        figure.write_html(filepath, include_plotlyjs='cdn')
                    else:
                        # Save as image (PDF/PNG)
                        figure.write_image(filepath, engine='kaleido')
                        
                elif fig_type == 'matplotlib':
                    if fmt == 'pdf' and hasattr(figure, 'savefig'):
                        figure.savefig(filepath, format='pdf', bbox_inches='tight')
                    elif fmt == 'pdf' and isinstance(figure, bytes):
                        # Pre-encoded PDF data
                        with open(filepath, 'wb') as f:
                            f.write(figure)
                            
                elif fig_type == 'image':
                    if fmt == 'png' and isinstance(figure, str):
                        # Base64 encoded image
                        img_data = base64.b64decode(figure.replace('data:image/png;base64,', ''))
                        with open(filepath, 'wb') as f:
                            f.write(img_data)
                            
                saved_files[fmt] = filepath
                logger.debug(f"Saved {fig_name}.{fmt} to {subdir}")
                
            except Exception as e:
                logger.error(f"Failed to save {fig_name}.{fmt}: {e}")
                # Save error info
                error_path = filepath.replace(f'.{fmt}', f'_ERROR.txt')
                with open(error_path, 'w') as f:
                    f.write(f"Error saving {fmt} format:\n{str(e)}")
                    
        return saved_files
    
    def generate_all_plots(self) -> Dict[str, Dict[str, str]]:
        """Generate all available plots for the detected workflow.
        
        Returns:
            Dictionary mapping plot names to saved file paths
        """
        all_plots = {}
        
        logger.info(f"Generating plots for {self.workflow} workflow")
        
        # Generate QC plots (common to both workflows)
        qc_plots = self._generate_qc_plots()
        all_plots.update(qc_plots)
        
        if self.workflow == 'proteomics':
            proteomics_plots = self._generate_proteomics_plots()
            all_plots.update(proteomics_plots)
        elif self.workflow == 'interactomics':
            interactomics_plots = self._generate_interactomics_plots()
            all_plots.update(interactomics_plots)
            
        logger.info(f"Generated {len(all_plots)} plots total")
        return all_plots
    
    def _generate_qc_plots(self) -> Dict[str, Dict[str, str]]:
        """Generate Quality Control plots common to both workflows."""
        plots = {}
        
        # Load QC data
        qc_data = self.data.get('03_qc_artifacts', {})
        data_dict = self.data.get('01_data_dictionary', {})
        
        if not qc_data:
            logger.warning("No QC data found for plot generation")
            return plots
            
        try:
            # 1. Missing values per sample
            if 'missing_values_per_sample' in qc_data:
                mv_data = pd.DataFrame(qc_data['missing_values_per_sample'])
                if not mv_data.empty:
                    fig = bar_graph.bar_plot(
                        defaults=self.figure_settings,
                        value_df=mv_data,
                        title="Missing values per sample",
                        y_label="Number of missing values"
                    )
                    plots['Missing values per sample'] = self._save_figure(
                        fig, "Missing values per sample", "QC figures"
                    )
                    
            # 2. Proteins per sample  
            if 'proteins_per_sample' in qc_data:
                pps_data = pd.DataFrame(qc_data['proteins_per_sample'])
                if not pps_data.empty:
                    fig = bar_graph.bar_plot(
                        defaults=self.figure_settings,
                        value_df=pps_data,
                        title="Proteins per sample",
                        y_label="Number of proteins"
                    )
                    plots['Proteins per sample'] = self._save_figure(
                        fig, "Proteins per sample", "QC figures"
                    )
                    
            # 3. Sum of values per sample
            if 'sum_values_per_sample' in qc_data:
                sum_data = pd.DataFrame(qc_data['sum_values_per_sample'])
                if not sum_data.empty:
                    fig = bar_graph.bar_plot(
                        defaults=self.figure_settings,
                        value_df=sum_data,
                        title="Sum of values per sample",
                        y_label="Sum of intensity values"
                    )
                    plots['Sum of values per sample'] = self._save_figure(
                        fig, "Sum of values per sample", "QC figures"
                    )
                    
            # 4. Value distribution histograms
            if 'value_distributions' in qc_data:
                for sample, distribution in qc_data['value_distributions'].items():
                    if distribution and 'values' in distribution:
                        dist_df = pd.DataFrame({'values': distribution['values']})
                        fig = histogram.make_figure(
                            data_table=dist_df,
                            x_column='values',
                            title=f"Value distribution - {sample}",
                            defaults=self.figure_settings
                        )
                        plots[f'Value distribution - {sample}'] = self._save_figure(
                            fig, f"Value distribution - {sample}", "QC figures"
                        )
                        
            # 5. Shared identifications (supervenn plot)
            if 'shared_proteins' in qc_data:
                shared_data = qc_data['shared_proteins']
                if shared_data and isinstance(shared_data, dict):
                    # Convert to format expected by supervenn
                    group_sets = {}
                    for group, proteins in shared_data.items():
                        if isinstance(proteins, list):
                            group_sets[group] = set(proteins)
                    
                    if group_sets:
                        try:
                            # Generate supervenn plot
                            img_obj, pdf_data = commonality_graph.supervenn(
                                group_sets, "shared-proteins-plot"
                            )
                            # Save PDF data
                            plots['Shared identifications'] = self._save_figure(
                                pdf_data, "Shared identifications", "QC figures", 
                                fig_type='matplotlib', formats=['pdf']
                            )
                        except Exception as e:
                            logger.error(f"Failed to generate supervenn plot: {e}")
                            
        except Exception as e:
            logger.error(f"Error generating QC plots: {e}")
            
        return plots
    
    def _generate_proteomics_plots(self) -> Dict[str, Dict[str, str]]:
        """Generate proteomics-specific plots."""
        plots = {}
        
        try:
            # 1. Volcano plots
            volcano_data = self.data.get('14_volcano', {})
            if volcano_data:
                if isinstance(volcano_data, str):
                    # Data is stored as JSON string - parse it as DataFrame
                    try:
                        df = pd.read_json(StringIO(volcano_data), orient='split')
                        if not df.empty and 'fold_change' in df.columns:
                            # Group by comparison if available
                            if 'Sample' in df.columns and 'Control' in df.columns:
                                comparisons = df.groupby(['Sample', 'Control'])
                                for (sample, control), group_df in comparisons:
                                    comparison_name = f"{sample} vs {control}"
                                    result = volcano_plot.volcano_plot(
                                        data_table=group_df,
                                        defaults=self.figure_settings,
                                        title=f"Volcano plot - {comparison_name}"
                                    )
                                    # Handle both single figure and tuple returns
                                    fig = result[0] if isinstance(result, tuple) else result
                                    plots[f'Volcano plot - {comparison_name}'] = self._save_figure(
                                        fig, f"Volcano plot - {comparison_name}", "Proteomics figures"
                                    )
                            else:
                                # Single comparison
                                result = volcano_plot.volcano_plot(
                                    data_table=df,
                                    defaults=self.figure_settings,
                                    title="Volcano plot"
                                )
                                fig = result[0] if isinstance(result, tuple) else result
                                plots['Volcano plot'] = self._save_figure(
                                    fig, "Volcano plot", "Proteomics figures"
                                )
                    except Exception as e:
                        logger.error(f"Failed to parse volcano data: {e}")
                elif isinstance(volcano_data, dict):
                    # Data is stored as dictionary of comparisons
                    for comparison, comp_data in volcano_data.items():
                            if isinstance(comp_data, dict) and 'data' in comp_data:
                                df = pd.DataFrame(comp_data['data'])
                                if not df.empty and 'fold_change' in df.columns:
                                    result = volcano_plot.volcano_plot(
                                        data_table=df,
                                        defaults=self.figure_settings,
                                        title=f"Volcano plot - {comparison}"
                                    )
                                    fig = result[0] if isinstance(result, tuple) else result
                                    plots[f'Volcano plot - {comparison}'] = self._save_figure(
                                        fig, f"Volcano plot - {comparison}", "Proteomics figures"
                                    )
                            
            # 2. PCA plots
            pca_data = self.data.get('13_pca', {})
            if pca_data:
                try:
                    if isinstance(pca_data, str):
                        # Parse JSON string
                        pca_dict = json.loads(pca_data)
                    else:
                        pca_dict = pca_data
                        
                    if 'plot_data' in pca_dict:
                        pca_df = pd.DataFrame(pca_dict['plot_data'])
                        if not pca_df.empty and 'PC1' in pca_df.columns:
                            fig = scatter.make_figure(
                                defaults=self.figure_settings,
                                data_table=pca_df,
                                x='PC1',
                                y='PC2',
                                color_col='Color',
                                name_col='Sample',
                                title="PCA Analysis"
                            )
                            plots['PCA'] = self._save_figure(
                                fig, "PCA", "Proteomics figures"
                            )
                except Exception as e:
                    logger.error(f"Failed to parse PCA data: {e}")
                    
            # 3. Imputation histogram
            imputed_data = self.data.get('12_imputed', {})
            na_filtered_data = self.data.get('10_na_filtered', {})
            
            if imputed_data and na_filtered_data:
                try:
                    # Compare before/after imputation
                    imputed_df = pd.read_json(StringIO(imputed_data), orient='split')
                    na_filtered_df = pd.read_json(StringIO(na_filtered_data), orient='split')
                    
                    graph_obj = imputation_histogram.make_graph(
                        non_imputed=na_filtered_df,
                        imputed=imputed_df,
                        defaults=self.figure_settings,
                        title="Imputation Analysis"
                    )
                    
                    if hasattr(graph_obj, 'figure'):
                        plots['Imputation'] = self._save_figure(
                            graph_obj.figure, "Imputation", "Proteomics figures"
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to generate imputation plot: {e}")
                    
            # 4. Normalization before/after
            normalized_data = self.data.get('11_normalized', {})
            if normalized_data and na_filtered_data:
                try:
                    normalized_df = pd.read_json(StringIO(normalized_data), orient='split')
                    na_filtered_df = pd.read_json(StringIO(na_filtered_data), orient='split')
                    
                    # Calculate means for comparison
                    before_means = na_filtered_df.mean()
                    after_means = normalized_df.mean()
                    
                    graph_obj = before_after_plot.make_graph(
                        defaults=self.figure_settings,
                        before=before_means,
                        after=after_means,
                        graph_id="normalization-plot",
                        title="Normalization Effect"
                    )
                    
                    if hasattr(graph_obj, 'figure'):
                        plots['Normalization'] = self._save_figure(
                            graph_obj.figure, "Normalization", "Proteomics figures"
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to generate normalization plot: {e}")
                    
        except Exception as e:
            logger.error(f"Error generating proteomics plots: {e}")
            
        return plots
    
    def _generate_interactomics_plots(self) -> Dict[str, Dict[str, str]]:
        """Generate interactomics-specific plots."""
        plots = {}
        
        try:
            # 1. SAINT BFDR distribution
            saint_data = self.data.get('21_saint_output_raw', {})
            if saint_data and isinstance(saint_data, list):
                saint_df = pd.DataFrame(saint_data)
                if not saint_df.empty and 'BFDR' in saint_df.columns:
                    fig = histogram.make_figure(
                        data_table=saint_df,
                        x_column='BFDR',
                        title="SAINT BFDR value distribution",
                        defaults=self.figure_settings,
                        nbins=30
                    )
                    plots['SAINT BFDR value distribution'] = self._save_figure(
                        fig, "SAINT BFDR value distribution", "Interactomics figures"
                    )
                    
            # 2. Filtered prey counts per bait
            filtered_data = self.data.get('23_saint_filtered', {})
            if filtered_data:
                if isinstance(filtered_data, list):
                    filtered_df = pd.DataFrame(filtered_data)
                elif isinstance(filtered_data, str):
                    filtered_df = pd.read_json(StringIO(filtered_data), orient='split')
                else:
                    filtered_df = pd.DataFrame()
                    
                if not filtered_df.empty and 'Bait' in filtered_df.columns:
                    # Count prey per bait
                    bait_counts = filtered_df.groupby('Bait').size().reset_index(name='Prey_Count')
                    bait_counts['Color'] = '#1f77b4'  # Default blue color
                    
                    fig = bar_graph.bar_plot(
                        defaults=self.figure_settings,
                        value_df=bait_counts,
                        title="Filtered Prey counts per bait",
                        x_name='Bait',
                        y_name='Prey_Count',
                        y_label="Number of prey proteins"
                    )
                    plots['Filtered Prey counts per bait'] = self._save_figure(
                        fig, "Filtered Prey counts per bait", "Interactomics figures"
                    )
                    
            # 3. PCA plot for interactomics
            pca_data = self.data.get('25_pca', {})
            if pca_data and 'plot_data' in pca_data:
                pca_df = pd.DataFrame(pca_data['plot_data'])
                if not pca_df.empty and 'PC1' in pca_df.columns:
                    fig = scatter.make_figure(
                        defaults=self.figure_settings,
                        data_table=pca_df,
                        x='PC1',
                        y='PC2',
                        color_col='Color',
                        name_col='Sample',
                        title="SPC PCA"
                    )
                    plots['SPC PCA'] = self._save_figure(
                        fig, "SPC PCA", "Interactomics figures"
                    )
                    
            # 4. Network elements visualization
            network_data = self.data.get('24_network_elements', {})
            interactions_data = self.data.get('24_interactions', {})
            
            if network_data and interactions_data:
                try:
                    # This would require the network plotting functionality
                    # For now, we'll skip this complex visualization
                    logger.info("Network visualization skipped - requires complex implementation")
                except Exception as e:
                    logger.error(f"Failed to generate network plot: {e}")
                    
        except Exception as e:
            logger.error(f"Error generating interactomics plots: {e}")
            
        return plots


def generate_plots_from_batch(batch_output_dir: str, export_dir: str = None, 
                            session_name: str = None) -> Dict[str, Dict[str, str]]:
    """Main function to generate plots from batch output directory.
    
    Args:
        batch_output_dir: Directory containing batch pipeline output files
        export_dir: Directory to save figures (uses batch_output_dir/../figures if None)
        session_name: Optional session name for organization
        
    Returns:
        Dictionary mapping plot names to saved file paths
    """
    if export_dir is None:
        export_dir = os.path.join(os.path.dirname(batch_output_dir), "figures")
        
    generator = BatchPlotGenerator(
        batch_output_dir=batch_output_dir,
        export_base_dir=export_dir,
        session_name=session_name
    )
    
    return generator.generate_all_plots()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots from ProteoGyver batch output")
    parser.add_argument("batch_dir", help="Directory containing batch output JSON files")
    parser.add_argument("--export-dir", help="Directory to save figures")
    parser.add_argument("--session-name", help="Session name for organization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Generate plots
    try:
        plots = generate_plots_from_batch(
            args.batch_dir, 
            args.export_dir, 
            args.session_name
        )
        
        print(f"\nGenerated {len(plots)} plots:")
        for plot_name, files in plots.items():
            print(f"  {plot_name}: {', '.join(files.keys())}")
            
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        sys.exit(1)
