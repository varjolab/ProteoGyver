#!/usr/bin/env python3
"""
Batch Figure Builder for ProteoGyver

This module constructs analysis_divs in the exact format expected by the GUI's
infra.save_figures function, allowing the batch pipeline to use the same
figure export infrastructure as the interactive GUI.

The module converts batch pipeline results into Dash Div components that
contain Graph and Image components with the same structure as GUI components.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from io import StringIO
import logging
import os
import sys

# Add path for components import
sys.path.insert(0, os.path.dirname(__file__))

from components.figures import (
    volcano_plot, tic_graph, bar_graph, histogram, 
    imputation_histogram, color_tools, reproducibility_graph
)
from components import qc_analysis

logger = logging.getLogger(__name__)


def create_graph_component(figure, graph_id: str) -> Dict:
    """Create a Dash Graph component from a Plotly figure.
    
    Args:
        figure: Plotly figure object
        graph_id: Unique ID for the graph component
        
    Returns:
        Dict: Dash Graph component structure
    """
    return {
        'props': {
            'id': graph_id,
            'figure': figure,
            'config': {'displayModeBar': True}
        },
        'type': 'Graph',
        'namespace': 'dash_core_components'
    }


def create_header_component(text: str, header_id: str, level: str = 'h4') -> Dict:
    """Create a Dash header component.
    
    Args:
        text: Header text
        header_id: Unique ID for the header
        level: Header level ('h4' or 'h5')
        
    Returns:
        Dict: Dash header component structure
    """
    return {
        'props': {
            'id': header_id,
            'children': text
        },
        'type': level.upper(),
        'namespace': 'dash_html_components'
    }


def create_image_component(image_src: str, image_id: str) -> Dict:
    """Create a Dash Image component.
    
    Args:
        image_src: Base64 encoded image string or URL
        image_id: Unique ID for the image
        
    Returns:
        Dict: Dash Image component structure
    """
    return {
        'props': {
            'id': image_id,
            'src': image_src
        },
        'type': 'Img',
        'namespace': 'dash_html_components'
    }


def create_div_component(children: List[Dict], div_id: str) -> Dict:
    """Create a Dash Div component containing other components.
    
    Args:
        children: List of child components
        div_id: Unique ID for the div
        
    Returns:
        Dict: Dash Div component structure
    """
    return {
        'props': {
            'id': div_id,
            'children': children
        },
        'type': 'Div',
        'namespace': 'dash_html_components'
    }


def build_qc_figures(batch_output_dir: str, workflow: str, svenn: bool) -> List[Dict]:
    """Build QC figures from batch output.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        workflow: Either 'proteomics' or 'interactomics'
        svenn: Whether to force the use of supervenn for commonality plot
    Returns:
        List[Dict]: List of figure components (headers + graphs)
    """
    components = []
    additional_rets = {}
    
    try:
        # Load QC data
        qc_data_path = f"{batch_output_dir}/03_qc_artifacts.json"
        with open(qc_data_path, 'r') as f:
            qc_data = json.load(f)
        
        # Load data dictionary for colors
        data_dict_path = f"{batch_output_dir}/01_data_dictionary.json"
        with open(data_dict_path, 'r') as f:
            data_dict = json.load(f)
        
        # Extract replicate colors
        replicate_colors = {}
        if 'sample colors' in data_dict:
            replicate_colors = {
                'samples': data_dict['sample colors'],
                'sample groups': data_dict.get('sample group colors', {})
            }
        
        # Build each QC plot
        qc_plots = [
            ('Sample run TICs', 'tic', 'tic-graph'),
            ('Proteins per sample', 'counts', 'count-graph'),
            ('Common proteins in data (qc)', 'common_proteins', 'common-protein-graph'),
            ('Protein identification coverage', 'coverage', 'coverage-graph'),
            ('Sample reproducibility', 'reproducibility', 'reproducibility-graph'),
            ('Missing values per sample', 'missing', 'missing-graph'),
            ('Sum of values per sample', 'sum', 'sum-graph'),
            ('Value mean', 'mean', 'mean-graph'),
            ('Value distribution per sample', 'distribution', 'distribution-graph'),
            ('Shared identifications', 'commonality', 'shared-graph')
        ]
        
        for plot_name, data_key, graph_id in qc_plots:
            if data_key in qc_data:
                # Add header
                components.append(create_header_component(
                    plot_name, f"qc-{data_key}-header", 'h4'
                ))
                
                # Create figure based on data type
                try:
                    defaults = {'template': 'plotly_white', 'height': 400, 'width': 600, 'config': {}}
                    
                    if data_key == 'tic':
                        # TIC data is a dict, create TIC graph
                        if isinstance(qc_data[data_key], dict):
                            fig = tic_graph.tic_figure(
                                defaults=defaults,
                                traces=qc_data[data_key]
                            )
                            components.append(create_graph_component(fig, graph_id))
                    elif data_key == 'reproducibility':
                        # Reproducibility data is a dict structure, not DataFrame
                        if isinstance(qc_data[data_key], str):
                            # Try to parse as JSON if it's a string
                            try:
                                repro_data = json.loads(qc_data[data_key])
                                graph_component = reproducibility_graph.make_graph(
                                    graph_id=graph_id,
                                    defaults=defaults,
                                    plot_data=repro_data,
                                    title=plot_name,
                                    table_type="intensity"
                                )
                                # Extract the figure from the Graph component
                                fig = graph_component.figure
                                components.append(create_graph_component(fig, graph_id))
                            except:
                                continue
                        elif isinstance(qc_data[data_key], dict):
                            graph_component = reproducibility_graph.make_graph(
                                graph_id=graph_id,
                                defaults=defaults,
                                plot_data=qc_data[data_key],
                                title=plot_name,
                                table_type="intensity"
                            )
                            fig = graph_component.figure
                            components.append(create_graph_component(fig, graph_id))
                    else:
                        # Other data are JSON strings of DataFrames
                        if isinstance(qc_data[data_key], str):
                            df = pd.read_json(StringIO(qc_data[data_key]), orient='split')
                            
                            if data_key in ['counts', 'coverage', 'missing', 'sum', 'mean']:
                                # Use bar_graph.bar_plot for these
                                if data_key == 'counts':
                                    y_col = 'Protein count' if 'Protein count' in df.columns else df.columns[-1]
                                elif data_key == 'coverage':
                                    y_col = 'Coverage' if 'Coverage' in df.columns else df.columns[-1]
                                elif data_key == 'missing':
                                    y_col = 'Missing count' if 'Missing count' in df.columns else df.columns[-1]
                                elif data_key == 'sum':
                                    y_col = 'Sum' if 'Sum' in df.columns else df.columns[-1]
                                elif data_key == 'mean':
                                    y_col = 'Mean' if 'Mean' in df.columns else df.columns[-1]
                                
                                # Add Color column for plotting
                                if 'Color' not in df.columns and 'samples' in replicate_colors:
                                    # Map sample names to colors - use index as sample names if available
                                    df = df.reset_index()
                                    sample_col = df.columns[0]  # First column after reset_index
                                    df['Color'] = df[sample_col].map(replicate_colors['samples'])
                                    # Fill any missing colors with a default
                                    df['Color'] = df['Color'].fillna('rgba(128,128,128,1)')
                                
                                fig = bar_graph.bar_plot(
                                    defaults=defaults,
                                    value_df=df,
                                    title=plot_name,
                                    y_name=y_col,
                                    color=True if 'Color' in df.columns else False,
                                    color_discrete_map=True if 'Color' in df.columns else False,
                                    color_discrete_map_dict=None  # Let it use the Color column directly
                                )
                            elif data_key == 'distribution':
                                # Use histogram.make_figure
                                value_col = df.columns[0] if len(df.columns) > 0 else 'value'
                                fig = histogram.make_figure(
                                    data_table=df,
                                    x_column=value_col,
                                    title=plot_name,
                                    defaults=defaults
                                )
                            elif data_key == 'common_proteins':
                                # Use bar_graph for common proteins
                                fig = bar_graph.bar_plot(
                                    defaults=defaults,
                                    value_df=df,
                                    title=plot_name,
                                    y_name=df.columns[-1],
                                    color=True if 'Color' in df.columns else False,
                                    color_discrete_map=True if 'Color' in df.columns else False,
                                    color_discrete_map_dict=None
                                )
                            elif data_key == 'commonality':
                                graph_area, common_str, image_str = get_commonality_pdf_data(batch_output_dir, workflow, svenn)
                                components.append(graph_area)
                                additional_rets['commonality-pdf-str'] = image_str
                                additional_rets['commonality-file-str'] = common_str
                                # Skip shared proteins for now - needs supervenn
                                continue
                            else:
                                continue
                            
                            components.append(create_graph_component(fig, graph_id))
                            
                except Exception as e:
                    logger.warning(f"Failed to create {plot_name}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error building QC figures: {e}")
    
    return components, additional_rets

def build_proteomics_figures(batch_output_dir: str) -> List[Dict]:
    """Build proteomics-specific figures from batch output.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        
    Returns:
        List[Dict]: List of proteomics figure components
    """
    import json  # Move import to top to avoid scope issues
    components = []
    
    try:
        # Load data dictionary for settings
        data_dict_path = f"{batch_output_dir}/01_data_dictionary.json"
        with open(data_dict_path, 'r') as f:
            data_dict = json.load(f)
        
        figure_settings = {
            'template': 'plotly_white', 
            'height': 800, 
            'width': 1200, 
            'config': {},
            'full-height': 800,
            'half-height': 400
        }
        
        # 1. Volcano plots - these will go to "Volcano plots" directory automatically
        volcano_path = f"{batch_output_dir}/14_volcano.json"
        try:
            with open(volcano_path, 'r') as f:
                volcano_data = f.read()
            
            # Parse volcano data as DataFrame (double parse - file contains JSON string)
            volcano_json_string = json.loads(volcano_data)
            df = pd.read_json(StringIO(volcano_json_string), orient='split')
            if not df.empty and 'fold_change' in df.columns:
                # Group by comparison if available
                if 'Sample' in df.columns and 'Control' in df.columns:
                    comparisons = df.groupby(['Sample', 'Control'])
                    for (sample, control), group_df in comparisons:
                        # Create individual volcano plot
                        volcano_title = f"Volcano {sample} vs {control}"
                        components.append(create_header_component(
                            volcano_title, f"volcano-{sample}-{control}-header", 'h4'
                        ))
                        
                        try:
                            result = volcano_plot.volcano_plot(
                                data_table=group_df,
                                defaults=figure_settings,
                                title=volcano_title
                            )
                            fig = result[0] if isinstance(result, tuple) else result
                            components.append(create_graph_component(
                                fig, f"volcano-{sample}-{control}-graph"
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to create volcano plot for {volcano_title}: {e}")
                            continue
                        
                        # Create "All significant differences" plot for controls
                        if control in ['SG2', 'SG3']:  # Only for certain controls
                            sig_title = f"All significant differences vs {control}"
                            components.append(create_header_component(
                                sig_title, f"sig-diff-{control}-header", 'h4'
                            ))
                            
                            try:
                                # Filter for significant differences
                                sig_df = group_df[group_df['Significant'] == True]
                                if not sig_df.empty:
                                    result = volcano_plot.volcano_plot(
                                        data_table=sig_df,
                                        defaults=figure_settings,
                                        title=sig_title
                                    )
                                    fig = result[0] if isinstance(result, tuple) else result
                                    components.append(create_graph_component(
                                        fig, f"sig-diff-{control}-graph"
                                    ))
                            except Exception as e:
                                logger.warning(f"Failed to create significant differences plot vs {control}: {e}")
                                continue
                    
        except FileNotFoundError:
            logger.warning(f"Volcano data not found: {volcano_path}")
        except Exception as e:
            logger.error(f"Failed to create volcano plots: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. PCA plot
        pca_path = f"{batch_output_dir}/13_pca.json"
        try:
            with open(pca_path, 'r') as f:
                pca_data_str = f.read()
            
            # Double parse PCA data (it's a DataFrame in JSON split format)
            pca_json_string = json.loads(pca_data_str)
            pca_df = pd.read_json(StringIO(pca_json_string), orient='split')
            
            if not pca_df.empty:
                components.append(create_header_component(
                    "PCA", "pca-header", 'h4'
                ))
                
                # Find PC1 and PC2 columns (they contain percentages)
                pc_cols = [col for col in pca_df.columns if col.startswith('PC') and '%' in col]
                if len(pc_cols) >= 2:
                    pc1, pc2 = pc_cols[0], pc_cols[1]
                    
                    # Create PCA scatter plot using scatter.make_graph
                    from components.figures.scatter import make_graph as scatter_make_graph
                    
                    # Load replicate colors
                    replicate_colors_path = f"{batch_output_dir}/02_replicate_colors.json"
                    with open(replicate_colors_path, 'r') as f:
                        replicate_colors = json.load(f)
                    
                    # Add sample group colors to DataFrame
                    pca_df['Sample group color'] = pca_df['Sample group'].map(
                        replicate_colors.get('sample groups', {})
                    )
                    
                    pca_graph = scatter_make_graph(
                        id_name='proteomics-pca-plot',
                        defaults=figure_settings,
                        data_table=pca_df,
                        x=pc1,
                        y=pc2,
                        color_col='Sample group color',
                        name_col='Sample group',
                        hover_data=['Sample group', 'Sample name']
                    )
                    
                    fig = pca_graph.figure
                    components.append(create_graph_component(fig, "pca-graph"))
                
        except FileNotFoundError:
            logger.warning(f"PCA data not found: {pca_path}")
        except Exception as e:
            logger.error(f"Failed to create PCA plot: {e}")
        
        # 3. Imputation histogram
        imputed_path = f"{batch_output_dir}/12_imputed.json"
        na_filtered_path = f"{batch_output_dir}/10_na_filtered.json"
        try:
            with open(imputed_path, 'r') as f:
                imputed_data = f.read()
            with open(na_filtered_path, 'r') as f:
                na_filtered_data = f.read()
            
            # Double parse for both files
            imputed_json_string = json.loads(imputed_data)
            na_filtered_json_string = json.loads(na_filtered_data)
            imputed_df = pd.read_json(StringIO(imputed_json_string), orient='split')
            na_filtered_df = pd.read_json(StringIO(na_filtered_json_string), orient='split')
            
            if not imputed_df.empty and not na_filtered_df.empty:
                components.append(create_header_component(
                    "Imputation", "imputation-header", 'h4'
                ))
                
                # Use imputation_histogram.make_graph
                graph_component = imputation_histogram.make_graph(
                    non_imputed=na_filtered_df,
                    imputed=imputed_df,
                    defaults=figure_settings,
                    id_name="imputation-graph",
                    title="Imputation"
                )
                fig = graph_component.figure
                components.append(create_graph_component(fig, "imputation-graph"))
                
        except FileNotFoundError:
            logger.warning(f"Imputation data not found: {imputed_path} or {na_filtered_path}")
        except Exception as e:
            logger.error(f"Failed to create imputation histogram: {e}")
        
        # 4. Normalization histogram
        normalized_path = f"{batch_output_dir}/11_normalized.json"
        try:
            with open(normalized_path, 'r') as f:
                normalized_data = f.read()
            
            # Double parse the normalized data
            normalized_json_string = json.loads(normalized_data)
            df = pd.read_json(StringIO(normalized_json_string), orient='split')
            if not df.empty:
                components.append(create_header_component(
                    "Normalization", "normalization-header", 'h4'
                ))
                
                # Use histogram.make_figure for distribution
                value_col = df.columns[0] if len(df.columns) > 0 else 'value'
                fig = histogram.make_figure(
                    data_table=df.melt(),
                    x_column='value',
                    title="Normalization",
                    defaults=figure_settings
                )
                components.append(create_graph_component(fig, "normalization-graph"))
                
        except FileNotFoundError:
            logger.warning(f"Normalization data not found: {normalized_path}")
        except Exception as e:
            logger.error(f"Failed to create normalization histogram: {e}")
            
        # 5. CV (Coefficients of variation) plot
        cv_path = f"{batch_output_dir}/13_cv.json"
        try:
            with open(cv_path, 'r') as f:
                cv_data = json.load(f)
            
            if cv_data and 'group_cvs' in cv_data:
                components.append(create_header_component(
                    "Coefficients of variation", "cv-header", 'h4'
                ))
                
                # Create CV violin plot
                from components.figures.cvplot import make_graph as cv_make_graph
                # Reconstruct DataFrame for CV plot (we need original imputed data)
                imputed_path = f"{batch_output_dir}/12_imputed.json"
                with open(imputed_path, 'r') as f:
                    imputed_data = f.read()
                # Double parse
                imputed_json_string = json.loads(imputed_data)
                imputed_df = pd.read_json(StringIO(imputed_json_string), orient='split')
                
                # Load replicate colors and get sample groups from data dict
                replicate_colors_path = f"{batch_output_dir}/02_replicate_colors.json"
                with open(replicate_colors_path, 'r') as f:
                    replicate_colors = json.load(f)
                
                cv_graph = cv_make_graph(
                    imputed_df,
                    data_dict.get('sample groups', {}).get('norm', {}),
                    replicate_colors,
                    figure_settings,
                    "proteomics-cv-plot"
                )
                
                # Extract figure from Graph component
                if isinstance(cv_graph, tuple):
                    # cv_make_graph returns a tuple, extract the Graph component
                    if len(cv_graph) > 0 and hasattr(cv_graph[0], 'figure'):
                        fig = cv_graph[0].figure
                        components.append(create_graph_component(fig, "cv-graph"))
                    else:
                        components.append(cv_graph[0])
                elif hasattr(cv_graph, 'figure'):
                    fig = cv_graph.figure
                    components.append(create_graph_component(fig, "cv-graph"))
                else:
                    # cv_make_graph returns a Graph component, use it directly
                    components.append(cv_graph)
                
        except FileNotFoundError:
            logger.warning(f"CV data not found: {cv_path}")
        except Exception as e:
            logger.error(f"Failed to create CV plot: {e}")
            
        # 6. Clustermap/Sample correlation clustering
        clustermap_path = f"{batch_output_dir}/13_clustermap.json"
        try:
            with open(clustermap_path, 'r') as f:
                clustermap_data = f.read()
            
            # Double parse the clustermap data
            clustermap_json_string = json.loads(clustermap_data)
            df = pd.read_json(StringIO(clustermap_json_string), orient='split')
            if not df.empty:
                components.append(create_header_component(
                    "Sample correlation clustering", "clustermap-header", 'h4'
                ))
                
                # Create clustermap using heatmaps module
                from components.figures.heatmaps import draw_clustergram
                fig = draw_clustergram(df, figure_settings, center_values=False)
                components.append(create_graph_component(fig, "clustermap-graph"))
                
        except FileNotFoundError:
            logger.warning(f"Clustermap data not found: {clustermap_path}")
        except Exception as e:
            logger.error(f"Failed to create clustermap: {e}")
            
        # 7. Missing value filtering plot
        na_filtered_path = f"{batch_output_dir}/10_na_filtered.json"
        try:
            with open(na_filtered_path, 'r') as f:
                na_filtered_data = f.read()
            
            # Compare with original intensity data to show filtering effect
            if 'data tables' in data_dict and 'intensity' in data_dict['data tables']:
                # The intensity data from data_dict is already parsed, but na_filtered_data needs double parse
                original_df = pd.read_json(StringIO(data_dict['data tables']['intensity']), orient='split')
                na_filtered_json_string = json.loads(na_filtered_data)
                filtered_df = pd.read_json(StringIO(na_filtered_json_string), orient='split')
                
                components.append(create_header_component(
                    "Missing value filtering", "na-filter-header", 'h4'
                ))
                
                # Create before/after count comparison
                from components.figures.before_after_plot import make_graph as before_after_make_graph
                original_counts = original_df.count()
                filtered_counts = filtered_df.count()
                
                before_after_graph = before_after_make_graph(
                    defaults=figure_settings,
                    before=original_counts,
                    after=filtered_counts,
                    graph_id="na-filter-graph",
                    title="Missing value filtering"
                )
                
                fig = before_after_graph.figure
                components.append(create_graph_component(fig, "na-filter-graph"))
                
        except (FileNotFoundError, KeyError):
            logger.warning(f"NA filtered data not found: {na_filtered_path}")
        except Exception as e:
            logger.error(f"Failed to create NA filter plot: {e}")
            
        # 8. Intensity of proteins with missing values in other samples
        try:
            # This plot shows the intensity distribution of proteins that have missing values
            na_filtered_json_string = json.loads(na_filtered_data)
            filtered_df = pd.read_json(StringIO(na_filtered_json_string), orient='split')
            
            # Find proteins with some missing values
            missing_mask = filtered_df.isnull()
            proteins_with_missing = missing_mask.any(axis=1)
            
            if proteins_with_missing.any():
                components.append(create_header_component(
                    "Intensity of proteins with missing values in other samples", 
                    "missing-intensity-header", 'h4'
                ))
                
                # Get intensity values for proteins that have missing values somewhere
                missing_proteins_df = filtered_df[proteins_with_missing]
                
                # Create histogram of non-missing values for these proteins
                from components.figures.histogram import make_figure as histogram_make_figure
                
                # Melt the DataFrame to get all intensity values
                melted_df = missing_proteins_df.melt(var_name='Sample', value_name='Intensity')
                melted_df = melted_df.dropna()  # Remove the actual missing values
                
                if not melted_df.empty:
                    fig = histogram_make_figure(
                        data_table=melted_df,
                        x_column='Intensity',
                        title="Intensity of proteins with missing values in other samples",
                        defaults=figure_settings
                    )
                    components.append(create_graph_component(fig, "missing-intensity-graph"))
                
        except Exception as e:
            logger.error(f"Failed to create missing intensity plot: {e}")
            
    except Exception as e:
        logger.error(f"Error building proteomics figures: {e}")
    
    return components


def build_interactomics_figures(batch_output_dir: str) -> List[Dict]:
    """Build interactomics-specific figures from batch output.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        
    Returns:
        List[Dict]: List of interactomics figure components
    """
    components = []
    
    try:
        figure_settings = {'template': 'plotly_white', 'height': 800, 'width': 1200, 'config': {}}
        
        # 1. PCA plot
        pca_path = f"{batch_output_dir}/25_pca.json"
        try:
            with open(pca_path, 'r') as f:
                pca_data = json.load(f)
            
            if 'figure' in pca_data:
                components.append(create_header_component(
                    "Interactomics PCA plot", "interactomics-pca-header", 'h4'
                ))
                components.append(create_graph_component(
                    pca_data['figure'], "interactomics-pca-graph"
                ))
                
        except FileNotFoundError:
            logger.warning(f"Interactomics PCA data not found: {pca_path}")
        except Exception as e:
            logger.error(f"Failed to create interactomics PCA plot: {e}")
        
        # 2. SAINT filtered results histogram (if available)
        saint_filtered_path = f"{batch_output_dir}/23_saint_filtered.json"
        try:
            with open(saint_filtered_path, 'r') as f:
                saint_data = json.load(f)
            
            df = pd.read_json(StringIO(saint_data), orient='split')
            if not df.empty:
                components.append(create_header_component(
                    "SAINT Filtered Results", "saint-filtered-header", 'h4'
                ))
                
                # Create a simple histogram of BFDR values if available
                if 'BFDR' in df.columns:
                    fig = histogram.make_figure(df, 'BFDR','', figure_settings)
                    components.append(create_graph_component(fig, "saint-filtered-graph"))
                
        except FileNotFoundError:
            logger.warning(f"SAINT filtered data not found: {saint_filtered_path}")
        except Exception as e:
            logger.error(f"Failed to create SAINT filtered histogram: {e}")
            
    except Exception as e:
        logger.error(f"Error building interactomics figures: {e}")
    
    return components


def get_commonality_pdf_data(batch_output_dir: str, workflow: str, svenn: bool) -> Optional[str]:
    """Generate commonality PDF data using supervenn if shared proteins data is available.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        workflow: Either 'proteomics' or 'interactomics'
        svenn: Whether to force the use of supervenn for commonality plot
    Returns:
        Optional[str]: Base64 encoded PDF data or None
    """
    try:
        figure_settings = {'template': 'plotly_white', 'height': 800, 'width': 1200, 'config': {}}
        # Load QC data to check for shared proteins
        qc_data_path = f"{batch_output_dir}/03_qc_artifacts.json"
        with open(qc_data_path, 'r') as f:
            qc_data = json.load(f)
        
        # Check if we have shared proteins data for supervenn
        if 'commonality' in qc_data:
            # Load sample group data for grouping
            data_dict_path = f"{batch_output_dir}/01_data_dictionary.json"
            with open(data_dict_path, 'r') as f:
                data_dict = json.load(f)
            
            rev_sample_groups = data_dict.get('sample groups', {}).get('rev', {})
            data_table = data_dict['data tables'][data_dict['data tables']['table to use']]
            if rev_sample_groups:
                # Call qc_analysis.commonality_plot to generate supervenn PDF
                return qc_analysis.commonality_plot(
                    data_table, rev_sample_groups, figure_settings, svenn
                )
    except Exception as e:
        logger.warning(f"Failed to generate commonality PDF data: {e}")
    
    return None


def build_analysis_divs_from_batch_output(batch_output_dir: str, workflow: str, svenn: bool) -> List[Dict]:
    """Build complete analysis_divs from batch output directory.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        workflow: Either 'proteomics' or 'interactomics'
        svenn: Whether to force the use of supervenn for commonality plot
    Returns:
        List[Dict]: Complete analysis_divs ready for infra.save_figures
    """
    all_components = []
    
    # Add main header
    all_components.append(create_header_component(
        "QC and Data Analysis", "qc-main-header", 'h4'
    ))
    
    # Add QC figures (common to both workflows)
    qc_components, additional_rets = build_qc_figures(batch_output_dir, workflow, svenn)
    all_components.extend(qc_components)
    
    # Add workflow-specific figures
    if workflow == 'proteomics':
        proteomics_components = build_proteomics_figures(batch_output_dir)
        all_components.extend(proteomics_components)
    elif workflow == 'interactomics':
        interactomics_components = build_interactomics_figures(batch_output_dir)
        all_components.extend(interactomics_components)
    
    # Wrap everything in a main div (like the GUI does)
    analysis_divs = [create_div_component(all_components, "qc-area")]
    
    return analysis_divs, additional_rets


def save_batch_figures_using_infra(batch_output_dir: str, export_dir: str, workflow: str,
                                 output_formats: List[str] = None, svenn: bool = False) -> Dict[str, Any]:
    """Save batch figures using the GUI's infra.save_figures function.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        export_dir: Directory to save exported figures
        workflow: Either 'proteomics' or 'interactomics'
        output_formats: List of formats ['html', 'pdf', 'png'] (default all)
        svenn: Whether to force the use of supervenn for commonality plot
    Returns:
        Dict: Summary of figure export operation
    """
    if output_formats is None:
        output_formats = ['html', 'pdf', 'png']
    
    from components.infra import save_figures
    
    # Build analysis_divs from batch output
    analysis_divs, additional_rets = build_analysis_divs_from_batch_output(batch_output_dir, workflow, svenn)
    shared = ''
    commonality_pdf_data = ''
    if 'commonality-pdf-str' in additional_rets:
        commonality_pdf_data = additional_rets['commonality-pdf-str']
        shared = additional_rets['commonality-file-str']
    # Use GUI's save_figures function
    save_figures(
        analysis_divs=analysis_divs,
        export_dir=export_dir,
        output_formats=output_formats,
        commonality_pdf_data=commonality_pdf_data,
        workflow=workflow
    )
    
    return ({
        'analysis_divs_count': len(analysis_divs[0]['props']['children']),
        'export_directory': export_dir,
        'workflow': workflow,
        'output_formats': output_formats,
        'commonality_pdf_generated': ((commonality_pdf_data is not None) and (len(commonality_pdf_data) > 0)),
        'infrastructure_used': 'GUI (infra.save_figures)'
    }, shared)

if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Add app directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser(description="Build figures from batch output and export using GUI infrastructure")
    parser.add_argument("batch_dir", help="Directory containing batch output JSON files")
    parser.add_argument("export_dir", help="Directory to save exported figures")
    parser.add_argument("workflow", choices=['proteomics', 'interactomics'], help="Workflow type")
    parser.add_argument("--formats", nargs="+", choices=['html', 'pdf', 'png'], 
                       default=['html', 'pdf', 'png'], help="Output formats")
    parser.add_argument("--svenn", action="store_true", help="Force the use of supervenn for commonality plot")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        result = save_batch_figures_using_infra(args.batch_dir, args.export_dir, args.workflow, args.formats, args.svenn)
        print(f"✅ Successfully exported {result['analysis_divs_count']} figure components")
        print(f"📁 Export directory: {result['export_directory']}")
        print(f"🎨 Formats: {', '.join(result['output_formats'])}")
        print(f"🔧 Infrastructure: {result['infrastructure_used']}")
        print(f"🔧 Supervenn: {args.svenn}")
        if result['commonality_pdf_generated']:
            print("📊 Commonality PDF generated")
        
    except Exception as e:
        logger.error(f"Figure export failed: {e}")
        sys.exit(1)
