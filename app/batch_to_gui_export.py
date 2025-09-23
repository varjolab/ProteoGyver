#!/usr/bin/env python3
"""
Adapter script to convert batch pipeline output to GUI export format.

This script takes the JSON output files from the batch pipeline and converts them
into the data store format expected by the GUI export functions in infra.py.
"""

import json
import os
from datetime import datetime
from components.infra import save_data_stores, save_figures
from typing import Dict, List, Any


def create_data_store_component(store_id: str, data: Any, modified_timestamp: float = None) -> Dict:
    """Create a data store component in the format expected by save_data_stores.
    
    Args:
        store_id: The data store ID (must match keys in data_store_export_configuration)
        data: The data to store (string, dict, or other format depending on store type)
        modified_timestamp: Optional timestamp, defaults to current time
        
    Returns:
        dict: Data store component in GUI format
    """
    if modified_timestamp is None:
        modified_timestamp = datetime.now().timestamp()
    
    return {
        'props': {
            'id': {'name': store_id},
            'data': data,
            'modified_timestamp': modified_timestamp
        }
    }


def convert_batch_output_to_gui_format(batch_output_dir: str, session_name: str = None) -> List[Dict]:
    """Convert batch pipeline output files to GUI data store format.
    
    Args:
        batch_output_dir: Directory containing batch pipeline JSON output files
        session_name: Optional session name for the analysis
        
    Returns:
        list: List of data store components in GUI format
    """
    data_stores = []
    
    # Mapping from batch output files to GUI data store IDs
    file_to_store_mapping = {
        '01_data_dictionary.json': None,  # This contains multiple data stores
        '02_replicate_colors.json': 'replicate-colors-data-store',
        '02_replicate_colors_with_cont.json': 'replicate-colors-with-contaminants-data-store',
        '03_qc_artifacts.json': None,  # This contains multiple QC data stores
        # Proteomics-specific files
        '10_na_filtered.json': 'proteomics-na-filtered-data-store',
        '11_normalized.json': 'proteomics-normalization-data-store',
        '12_imputed.json': 'proteomics-imputation-data-store',
        '13_pca.json': 'proteomics-pca-data-store',
        '14_volcano.json': 'proteomics-volcano-data-store',
        # Interactomics-specific files
        '20_crapome_data.json': 'interactomics-saint-crapome-data-store',
        '20_saint_dict.json': 'interactomics-saint-input-data-store',
        '21_saint_output_raw.json': 'interactomics-saint-output-data-store',
        '22_saint_with_crapome.json': 'interactomics-saint-final-output-data-store',
        '23_saint_filtered.json': 'interactomics-saint-filtered-output-data-store',
        '24_interactions.json': 'interactomics-network-interactions-data-store',
        '24_network_elements.json': 'interactomics-network-data-store',
        '25_pca.json': 'interactomics-pca-data-store',
        '26_enrichment_data.json': 'interactomics-enrichment-data-store',
        '26_enrichment_info.json': 'interactomics-enrichment-information-data-store',
        '27_msmic_data.json': 'interactomics-saint-filtered-and-intensity-mapped-output-data-store',
    }
    
    # Process each output file
    for filename, store_id in file_to_store_mapping.items():
        filepath = os.path.join(batch_output_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if filename == '01_data_dictionary.json':
            # Extract multiple data stores from data dictionary
            data_stores.extend(_extract_data_dictionary_stores(data))
        elif filename == '03_qc_artifacts.json':
            # Extract QC data stores
            data_stores.extend(_extract_qc_stores(data))
        elif store_id:
            # Direct mapping to single data store
            data_stores.append(create_data_store_component(store_id, data))
    
    # Add session information if available
    if session_name:
        # Add upload information (mock data for batch processing)
        upload_info = {
            'Modified time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'File name': 'Batch pipeline input',
            'Data type': 'Proteomics',
        }
        data_stores.append(create_data_store_component(
            'uploaded-data-table-info-data-store', upload_info))
        data_stores.append(create_data_store_component(
            'uploaded-sample-table-info-data-store', upload_info))
    
    return data_stores


def _extract_data_dictionary_stores(data_dict: Dict) -> List[Dict]:
    """Extract data stores from the data dictionary.
    
    Args:
        data_dict: The data dictionary from batch output
        
    Returns:
        list: List of data store components
    """
    stores = []
    
    # Extract data tables
    if 'data tables' in data_dict:
        data_tables = data_dict['data tables']
        
        # Create upload data store (format expected: dict with table names as keys and JSON strings as values)
        # The upload-split format expects each table to be a separate JSON string
        upload_data = {}
        for table_name, table_data in data_tables.items():
            if table_name == 'experimental design':  # Skip experimental design as it's handled separately
                continue
            if table_name == 'table to use':  # Skip simple string values that aren't DataFrames
                continue
            if table_name == 'with-contaminants':  # This is a nested structure, extract the actual data tables
                if isinstance(table_data, dict):
                    # Extract individual data tables from the with-contaminants structure
                    for sub_table_name, sub_table_data in table_data.items():
                        if isinstance(sub_table_data, str):
                            upload_data[f"with-contaminants-{sub_table_name}"] = sub_table_data
                continue
            
            if isinstance(table_data, str):
                upload_data[table_name] = table_data
            else:
                upload_data[table_name] = json.dumps(table_data)
        
        stores.append(create_data_store_component(
            'uploaded-data-table-data-store', upload_data))
        
        # Sample table
        if 'experimental design' in data_tables:
            stores.append(create_data_store_component(
                'uploaded-sample-table-data-store', data_tables['experimental design']))
    
    # Extract other useful information
    if 'sample groups' in data_dict:
        # This is used by various components but not directly exported
        pass
    
    return stores


def _extract_qc_stores(qc_data: Dict) -> List[Dict]:
    """Extract QC data stores from QC artifacts.
    
    Args:
        qc_data: QC artifacts dictionary from batch output
        
    Returns:
        list: List of QC data store components
    """
    stores = []
    
    # Mapping from QC artifact keys to data store IDs
    qc_mapping = {
        'tic': 'tic-data-store',
        'counts': 'count-data-store',
        'common_proteins': 'common-protein-data-store',
        'coverage': 'coverage-data-store',
        'reproducibility': 'reproducibility-data-store',
        'missing': 'missing-data-store',
        'sum': 'sum-data-store',
        'mean': 'mean-data-store',
        'distribution': 'distribution-data-store',
    }
    
    for qc_key, store_id in qc_mapping.items():
        if qc_key in qc_data and qc_data[qc_key] is not None:
            stores.append(create_data_store_component(store_id, qc_data[qc_key]))
    
    return stores


def export_batch_results(batch_output_dir: str, export_dir: str, session_name: str = None) -> Dict:
    """Export batch pipeline results using GUI export functions.
    
    Args:
        batch_output_dir: Directory containing batch pipeline JSON output files
        export_dir: Directory to export results to
        session_name: Optional session name for the analysis
        
    Returns:
        dict: Export timestamps and status information
    """
    print(f"Converting batch output from {batch_output_dir} to GUI format...")
    
    # Convert batch output to GUI data store format
    data_stores = convert_batch_output_to_gui_format(batch_output_dir, session_name)
    
    print(f"Created {len(data_stores)} data store components")
    for store in data_stores:
        print(f"  - {store['props']['id']['name']}")
    
    # Create export directory
    os.makedirs(export_dir, exist_ok=True)
    
    # Use GUI export function to save data stores
    print(f"Exporting data to {export_dir}...")
    timestamps = save_data_stores(data_stores, export_dir)
    
    return {
        'export_dir': export_dir,
        'data_stores_exported': len(data_stores),
        'timestamps': timestamps,
        'session_name': session_name or 'batch_pipeline'
    }


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export batch pipeline results using GUI export format")
    parser.add_argument("batch_output_dir", help="Directory containing batch pipeline JSON output")
    parser.add_argument("--export-dir", default="batch_gui_export", help="Export directory (default: batch_gui_export)")
    parser.add_argument("--session-name", help="Session name for the analysis")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.batch_output_dir):
        print(f"Error: Batch output directory {args.batch_output_dir} does not exist")
        return 1
    
    try:
        result = export_batch_results(args.batch_output_dir, args.export_dir, args.session_name)
        print(f"\nExport completed successfully!")
        print(f"Export directory: {result['export_dir']}")
        print(f"Data stores exported: {result['data_stores_exported']}")
        print(f"Session name: {result['session_name']}")
        return 0
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
