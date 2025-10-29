#!/usr/bin/env python3
"""
Batch Data Store Builder for ProteoGyver

This module constructs data stores in the exact format expected by the GUI,
allowing the batch pipeline to use the same infra.save_data_stores function
as the interactive GUI.

The module converts batch pipeline results into Dash Store components that
match the structure and content expected by the GUI export system.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from io import StringIO
import logging
from components import infra

logger = logging.getLogger(__name__)


def create_data_store_component(store_id: str, data: Any, timestamp: Optional[float] = None) -> Dict:
    """Create a Dash data store component.
    
    Args:
        store_id: The data store ID (e.g., 'proteomics-volcano-data-store')
        data: The data to store (can be dict, DataFrame as JSON string, etc.)
        timestamp: Optional timestamp, uses current time if None
        
    Returns:
        Dict: Dash Store component structure
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp() * 1000  # Convert to milliseconds
        
    return {
        'props': {
            'id': {'type': 'data-store', 'name': store_id},
            'data': data,
            'modified_timestamp': timestamp
        },
        'type': 'Store',
        'namespace': 'dash_core_components'
    }


def build_upload_data_store(data_dict: Dict) -> Dict:
    """Build the main upload data store from batch data dictionary.
    
    Args:
        data_dict: The data dictionary from batch output
        
    Returns:
        Dict: Data store component for upload-data-store
    """
    upload_data = {
        'sample groups': {},
        'data tables': {},
        'info': {},
        'file info': {},
        'other': {}
    }
    
    # Extract data tables
    if 'data tables' in data_dict:
        upload_data['data tables'] = data_dict['data tables']
    
    # Extract sample groups
    if 'sample groups' in data_dict:
        upload_data['sample groups'] = data_dict['sample groups']
        
    # Extract other information
    if 'workflow' in data_dict:
        upload_data['info']['workflow'] = data_dict['workflow']
        
    return create_data_store_component('upload-data-store', upload_data)


def build_replicate_colors_stores(data_dict: Dict) -> List[Dict]:
    """Build replicate color data stores.
    
    Args:
        data_dict: The data dictionary from batch output
        
    Returns:
        List[Dict]: List containing replicate color data stores
    """
    stores = []
    
    # Regular replicate colors
    if 'sample colors' in data_dict:
        color_data = {
            'samples': data_dict['sample colors'],
            'sample groups': data_dict.get('sample group colors', {})
        }
        stores.append(create_data_store_component('replicate-colors-data-store', color_data))
    
    # With contaminants colors (if available)
    if 'sample colors with contaminants' in data_dict:
        color_data_cont = {
            'samples': data_dict['sample colors with contaminants'],
            'sample groups': data_dict.get('sample group colors', {})
        }
        stores.append(create_data_store_component('replicate-colors-with-contaminants-data-store', color_data_cont))
    
    return stores


def build_qc_data_stores(qc_data: Dict) -> List[Dict]:
    """Build QC-related data stores.
    
    Args:
        qc_data: QC artifacts data from batch output
        
    Returns:
        List[Dict]: List of QC data store components
    """
    stores = []
    
    # TIC data store
    if 'tic' in qc_data:
        stores.append(create_data_store_component('tic-data-store', qc_data['tic']))
    
    # Count data store
    if 'counts' in qc_data:
        stores.append(create_data_store_component('count-data-store', qc_data['counts']))
    
    # Common protein data store
    if 'common_proteins' in qc_data:
        stores.append(create_data_store_component('common-protein-data-store', qc_data['common_proteins']))
    
    # Coverage data store
    if 'coverage' in qc_data:
        stores.append(create_data_store_component('coverage-data-store', qc_data['coverage']))
    
    # Reproducibility data store
    if 'reproducibility' in qc_data:
        stores.append(create_data_store_component('reproducibility-data-store', qc_data['reproducibility']))
    
    # Missing data store
    if 'missing' in qc_data:
        stores.append(create_data_store_component('missing-data-store', qc_data['missing']))
    
    # Sum data store
    if 'sum' in qc_data:
        stores.append(create_data_store_component('sum-data-store', qc_data['sum']))
    
    # Mean data store
    if 'mean' in qc_data:
        stores.append(create_data_store_component('mean-data-store', qc_data['mean']))
    
    # Distribution data store
    if 'distribution' in qc_data:
        stores.append(create_data_store_component('distribution-data-store', qc_data['distribution']))
    
    # Common proteins
    if 'common_proteins' in qc_data:
        stores.append(create_data_store_component('common-protein-data-store', qc_data['common_proteins']))
    
    # Commonality data store (for supervenn plots)
    if 'commonality' in qc_data:
        stores.append(create_data_store_component('commonality-data-store', qc_data['commonality']))
    
    return stores


def build_proteomics_data_stores(batch_output_dir: str) -> List[Dict]:
    """Build proteomics-specific data stores from batch output.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        
    Returns:
        List[Dict]: List of proteomics data store components
    """
    stores = []
    
    try:
        # NA filtered data store
        na_filtered_path = f"{batch_output_dir}/10_na_filtered.json"
        try:
            with open(na_filtered_path, 'r') as f:
                na_filtered_data = json.load(f)
            stores.append(create_data_store_component('proteomics-na-filtered-data-store', na_filtered_data))
        except FileNotFoundError:
            logger.warning(f"NA filtered data not found: {na_filtered_path}")
        
        # Normalized data store
        normalized_path = f"{batch_output_dir}/11_normalized.json"
        try:
            with open(normalized_path, 'r') as f:
                normalized_data = json.load(f)
            stores.append(create_data_store_component('proteomics-normalization-data-store', normalized_data))
        except FileNotFoundError:
            logger.warning(f"Normalized data not found: {normalized_path}")
        
        # Imputed data store
        imputed_path = f"{batch_output_dir}/12_imputed.json"
        try:
            with open(imputed_path, 'r') as f:
                imputed_data = json.load(f)
            stores.append(create_data_store_component('proteomics-imputation-data-store', imputed_data))
        except FileNotFoundError:
            logger.warning(f"Imputed data not found: {imputed_path}")
        
        # PCA data store
        pca_path = f"{batch_output_dir}/13_pca.json"
        try:
            with open(pca_path, 'r') as f:
                pca_data = json.load(f)
            stores.append(create_data_store_component('proteomics-pca-data-store', pca_data))
        except FileNotFoundError:
            logger.warning(f"PCA data not found: {pca_path}")
        
        # Volcano data store
        volcano_path = f"{batch_output_dir}/14_volcano.json"
        try:
            with open(volcano_path, 'r') as f:
                volcano_data = json.load(f)
            stores.append(create_data_store_component('proteomics-volcano-data-store', volcano_data))
        except FileNotFoundError:
            logger.warning(f"Volcano data not found: {volcano_path}")
            
        # CV data store
        cv_path = f"{batch_output_dir}/13_cv.json"
        try:
            with open(cv_path, 'r') as f:
                cv_data = json.load(f)
            stores.append(create_data_store_component('proteomics-cv-data-store', cv_data))
        except FileNotFoundError:
            logger.warning(f"CV data not found: {cv_path}")
            
        # Clustermap data store
        clustermap_path = f"{batch_output_dir}/13_clustermap.json"
        try:
            with open(clustermap_path, 'r') as f:
                clustermap_data = json.load(f)
            stores.append(create_data_store_component('proteomics-clustermap-data-store', clustermap_data))
        except FileNotFoundError:
            logger.warning(f"Clustermap data not found: {clustermap_path}")
            
        # Perturbation data store (if available)
        perturbation_path = f"{batch_output_dir}/13_perturbation.json"
        try:
            with open(perturbation_path, 'r') as f:
                perturbation_data = json.load(f)
            stores.append(create_data_store_component('proteomics-pertubation-data-store', perturbation_data))
        except FileNotFoundError:
            logger.debug(f"Perturbation data not found: {perturbation_path}")
            
    except Exception as e:
        logger.error(f"Error building proteomics data stores: {e}")
    
    return stores


def build_interactomics_data_stores(batch_output_dir: str) -> List[Dict]:
    """Build interactomics-specific data stores from batch output.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        
    Returns:
        List[Dict]: List of interactomics data store components
    """
    stores = []
    
    try:
        # CRAPome data store
        crapome_path = f"{batch_output_dir}/20_crapome_data.json"
        try:
            with open(crapome_path, 'r') as f:
                crapome_data = json.load(f)
            stores.append(create_data_store_component('interactomics-saint-crapome-data-store', crapome_data))
        except FileNotFoundError:
            logger.warning(f"CRAPome data not found: {crapome_path}")
        
        # SAINT input data store
        saint_dict_path = f"{batch_output_dir}/20_saint_dict.json"
        try:
            with open(saint_dict_path, 'r') as f:
                saint_dict = json.load(f)
            stores.append(create_data_store_component('interactomics-saint-input-data-store', saint_dict))
        except FileNotFoundError:
            logger.warning(f"SAINT dict not found: {saint_dict_path}")
        
        # SAINT raw output data store
        saint_raw_path = f"{batch_output_dir}/21_saint_output_raw.json"
        try:
            with open(saint_raw_path, 'r') as f:
                saint_raw = json.load(f)
            stores.append(create_data_store_component('interactomics-saint-output-data-store', saint_raw))
        except FileNotFoundError:
            logger.warning(f"SAINT raw output not found: {saint_raw_path}")
        
        # SAINT with CRAPome data store
        saint_crapome_path = f"{batch_output_dir}/22_saint_with_crapome.json"
        try:
            with open(saint_crapome_path, 'r') as f:
                saint_crapome = json.load(f)
            stores.append(create_data_store_component('interactomics-saint-final-output-data-store', saint_crapome))
        except FileNotFoundError:
            logger.warning(f"SAINT with CRAPome not found: {saint_crapome_path}")
        
        # SAINT filtered data store
        saint_filtered_path = f"{batch_output_dir}/23_saint_filtered.json"
        try:
            with open(saint_filtered_path, 'r') as f:
                saint_filtered = json.load(f)
            stores.append(create_data_store_component('interactomics-saint-filtered-output-data-store', saint_filtered))
        except FileNotFoundError:
            logger.warning(f"SAINT filtered not found: {saint_filtered_path}")

        # SAINT filtered and intensity mapped data store
        saint_filtered_and_intensity_mapped_path = f"{batch_output_dir}/23_saint_filtered_and_intensity_mapped.json"
        try:
            with open(saint_filtered_and_intensity_mapped_path, 'r') as f:
                saint_filtered_and_intensity_mapped = json.load(f)
            stores.append(create_data_store_component('interactomics-saint-filtered-and-intensity-mapped-output-data-store', saint_filtered_and_intensity_mapped))
        except FileNotFoundError:
            logger.warning(f"SAINT filtered and intensity mapped not found: {saint_filtered_and_intensity_mapped_path}")
        
        # SAINT filtered and intensity mapped with knowns data store
        saint_filtered_and_intensity_mapped_with_knowns_path = f"{batch_output_dir}/23_saint_filtered_and_intensity_mapped_with_knowns.json"
        try:
            with open(saint_filtered_and_intensity_mapped_with_knowns_path, 'r') as f:
                saint_filtered_and_intensity_mapped_with_knowns = json.load(f)
            stores.append(create_data_store_component('interactomics-saint-filt-int-known-data-store', saint_filtered_and_intensity_mapped_with_knowns))
        except FileNotFoundError:
            logger.warning(f"SAINT filtered and intensity mapped with knowns not found: {saint_filtered_and_intensity_mapped_with_knowns_path}")

        # Network interactions data store
        interactions_path = f"{batch_output_dir}/24_interactions.json"
        try:
            with open(interactions_path, 'r') as f:
                interactions = json.load(f)
            stores.append(create_data_store_component('interactomics-network-interactions-data-store', interactions))
        except FileNotFoundError:
            logger.warning(f"Interactions not found: {interactions_path}")
        
        # Network elements data store
        network_path = f"{batch_output_dir}/24_network_elements.json"
        try:
            with open(network_path, 'r') as f:
                network_elements = json.load(f)
            stores.append(create_data_store_component('interactomics-network-data-store', network_elements))
        except FileNotFoundError:
            logger.warning(f"Network elements not found: {network_path}")
        
        # PCA data store
        pca_path = f"{batch_output_dir}/25_pca.json"
        try:
            with open(pca_path, 'r') as f:
                pca_data = json.load(f)
            stores.append(create_data_store_component('interactomics-pca-data-store', pca_data))
        except FileNotFoundError:
            logger.warning(f"Interactomics PCA not found: {pca_path}")
        
        # Enrichment data store
        enrichment_path = f"{batch_output_dir}/26_enrichment_data.json"
        try:
            with open(enrichment_path, 'r') as f:
                enrichment_data = json.load(f)
            stores.append(create_data_store_component('interactomics-enrichment-data-store', enrichment_data))
        except FileNotFoundError:
            logger.info(f"Enrichment data not found: {enrichment_path}")
        
        # Enrichment information data store
        enrichment_info_path = f"{batch_output_dir}/26_enrichment_info.json"
        try:
            with open(enrichment_info_path, 'r') as f:
                enrichment_info = json.load(f)
            stores.append(create_data_store_component('interactomics-enrichment-information-data-store', enrichment_info))
        except FileNotFoundError:
            logger.info(f"Enrichment info not found: {enrichment_info_path}")
        
        # MS microscopy data store (if available)
        msmic_path = f"{batch_output_dir}/27_msmic_data.json"
        try:
            with open(msmic_path, 'r') as f:
                msmic_data = json.load(f)
            stores.append(create_data_store_component('interactomics-msmic-data-store', msmic_data))
        except FileNotFoundError:
            logger.debug(f"MS microscopy data not found: {msmic_path}")
            
    except Exception as e:
        logger.error(f"Error building interactomics data stores: {e}")
    
    return stores


def build_data_stores_from_batch_output(batch_output_dir: str, workflow: str) -> List[Dict]:
    """Build complete data stores list from batch output directory.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        workflow: Either 'proteomics' or 'interactomics'
        
    Returns:
        List[Dict]: Complete list of data store components ready for infra.save_data_stores
    """
    data_stores = []
    
    try:
        # Load core data
        data_dict_path = f"{batch_output_dir}/01_data_dictionary.json"
        with open(data_dict_path, 'r') as f:
            data_dict = json.load(f)
        
        qc_data_path = f"{batch_output_dir}/03_qc_artifacts.json"
        with open(qc_data_path, 'r') as f:
            qc_data = json.load(f)
        
        # Build common data stores
        data_stores.append(build_upload_data_store(data_dict))
        data_stores.extend(build_replicate_colors_stores(data_dict))
        data_stores.extend(build_qc_data_stores(qc_data))
        
        # Add uploaded data table stores expected by the GUI
        if 'input_data_tables' in data_dict:
            # Create upload split format for data tables
            upload_tables = {}
            #skip_tables = ['experimental design', 'table to use', 'with-contaminants']
            for table_name, table_data in data_dict['input_data_tables'].items():
             #   if table_name not in skip_tables and isinstance(table_data, str):
                upload_tables[table_name] = table_data
            
            if upload_tables:
                data_stores.append(create_data_store_component('uploaded-data-table-data-store', upload_tables))
            
            # Sample table store
            if 'input_sample_table' in data_dict:
                data_stores.append(create_data_store_component('uploaded-sample-table-data-store', 
                                                             data_dict['input_sample_table']))
        
        # Build workflow-specific data stores
        if workflow == 'proteomics':
            data_stores.extend(build_proteomics_data_stores(batch_output_dir))
        elif workflow == 'interactomics':
            data_stores.extend(build_interactomics_data_stores(batch_output_dir))
        
        # Add info data stores
        data_stores.append(create_data_store_component('uploaded-data-table-info-data-store', {
            'Modified time': data_dict['file info']['Data']['File modified'],
            'File name': data_dict['file info']['Data']['File name'],
            'Data type': data_dict['info']['Data type'],
            'Data source guess': data_dict['info']['Data source guess']
        }))
        data_stores.append(create_data_store_component('uploaded-sample-table-info-data-store', {
            'Modified time': data_dict['file info']['Sample table']['File modified'],
            'File name': data_dict['file info']['Sample table']['File name']
        }))
        
        logger.info(f"Built {len(data_stores)} data stores for {workflow} workflow")
        
    except Exception as e:
        logger.error(f"Error building data stores: {e}")
        raise
    
    return data_stores


def save_batch_data_using_infra(batch_output_dir: str, export_dir: str, workflow: str) -> Dict[str, Any]:
    """Save batch data using the GUI's infra.save_data_stores function.
    
    Args:
        batch_output_dir: Directory containing batch output JSON files
        export_dir: Directory to save exported data
        workflow: Either 'proteomics' or 'interactomics'
        
    Returns:
        Dict: Summary of export operation
    """
    
    # Build data stores from batch output
    data_stores = build_data_stores_from_batch_output(batch_output_dir, workflow)
    
    # Use GUI's save function
    result = infra.save_data_stores(data_stores, export_dir)
    
    return {
        'data_stores_count': len(data_stores),
        'export_directory': export_dir,
        'workflow': workflow,
        'infra_result': result
    }

if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Add app directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser(description="Build data stores from batch output and export using GUI infrastructure")
    parser.add_argument("batch_dir", help="Directory containing batch output JSON files")
    parser.add_argument("export_dir", help="Directory to save exported data")
    parser.add_argument("workflow", choices=['proteomics', 'interactomics'], help="Workflow type")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        result = save_batch_data_using_infra(args.batch_dir, args.export_dir, args.workflow)
        print(f"Successfully exported {result['data_stores_count']} data stores")
        print(f"Export directory: {result['export_directory']}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)
