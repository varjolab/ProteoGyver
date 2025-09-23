#!/usr/bin/env python3
"""
Complete batch pipeline runner with GUI-compatible export.

This script runs the full batch pipeline and then exports the results
using the same directory structure and format as the GUI export.
"""

import os
import sys
import json
import tempfile
from datetime import datetime

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from pipeline_batch import BatchConfig, run_pipeline
from batch_to_gui_export import export_batch_results


def run_batch_pipeline_with_export(toml_path: str, export_dir: str = None) -> dict:
    """Run the batch pipeline and export results in GUI format.
    
    Args:
        toml_path: Path to the TOML configuration file
        export_dir: Optional custom export directory
        
    Returns:
        dict: Summary of pipeline execution and export
    """
    # Import here to avoid path issues
    from pipeline_from_toml import _load_config
    
    print(f"Loading configuration from {toml_path}...")
    config = _load_config(toml_path)
    
    print(f"Running {config.workflow} pipeline...")
    
    # Use temporary directory for batch output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the output directory to use temp directory
        config.outdir = temp_dir
        
        # Run the pipeline
        pipeline_result = run_pipeline(config)
        
        # Determine export directory
        if export_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"{config.workflow}_export_{timestamp}"
        
        print(f"Pipeline completed. Exporting results to {export_dir}...")
        
        # Export using GUI format
        export_result = export_batch_results(
            temp_dir, 
            export_dir, 
            pipeline_result.get('session_name', 'batch_pipeline')
        )
        
        return {
            'pipeline_result': pipeline_result,
            'export_result': export_result,
            'toml_config': toml_path,
            'workflow': config.workflow
        }


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ProteoGyver batch pipeline with GUI-compatible export"
    )
    parser.add_argument("toml_file", help="Path to TOML configuration file")
    parser.add_argument("--export-dir", help="Export directory (default: auto-generated)")
    parser.add_argument("--keep-batch-output", action="store_true", 
                       help="Keep the intermediate batch JSON files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.toml_file):
        print(f"Error: TOML file {args.toml_file} does not exist")
        return 1
    
    # Change to app directory for proper execution
    original_dir = os.getcwd()
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
    os.chdir(app_dir)
    
    try:
        result = run_batch_pipeline_with_export(
            os.path.join(original_dir, args.toml_file),
            args.export_dir
        )
        
        print("\n" + "="*60)
        print("BATCH PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Workflow: {result['workflow']}")
        print(f"Configuration: {result['toml_config']}")
        print(f"Export directory: {result['export_result']['export_dir']}")
        print(f"Data stores exported: {result['export_result']['data_stores_exported']}")
        print(f"Session name: {result['export_result']['session_name']}")
        
        # Print pipeline summary
        pipeline_summary = result['pipeline_result']
        print(f"\nPipeline Summary:")
        for key, value in pipeline_summary.items():
            if key not in ['artifacts', 'outdir']:
                print(f"  {key}: {value}")
        
        print(f"\nExported files can be found in: {result['export_result']['export_dir']}")
        print("Directory structure matches GUI export format.")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    exit(main())
