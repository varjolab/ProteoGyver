#!/usr/bin/env python3
"""
ProteoGyver Batch Pipeline

This script runs the complete batch pipeline using the same infrastructure
as the GUI, ensuring identical behavior and maintainability.

"""

import os
import sys
import argparse
import tempfile
from datetime import datetime
import logging
from pathlib import Path
from pipeline_module import pipeline_batch
from pipeline_module import pipeline_from_toml
from pipeline_module import batch_data_store_builder
from pipeline_module import batch_figure_builder_from_divs
from components import infra
from components import parsing

logger = logging.getLogger(__name__)


def run_batch_pipeline(toml_path: str) -> dict:
    """Run the complete batch pipeline using GUI infrastructure.
    
    Args:
        toml_path: Path to the TOML configuration file
    Returns:
        dict: Summary of pipeline execution, export, and plot generation
    """
    script_dir = Path(__file__).resolve().parent
    parameters_file = script_dir / 'parameters.toml'
    parameters = parsing.parse_parameters(parameters_file)
    
    input_dir = os.path.dirname(os.path.realpath(toml_path))
    config = pipeline_from_toml.load_config(toml_path, default_toml_dir=Path(*parameters['Pipeline module']['Default toml files directory']))
    plot_formats = config.plot_formats
    keep_batch_output = config.keep_batch_output

    export_dir = os.path.join(input_dir, 'PG output')
    os.makedirs(export_dir, exist_ok=True)

    # Generate session name from timestamp
    session_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}--pipeline"
    logger.info(f"Session name: {session_name}, output directory: {export_dir}")
    # Determine whether to use temporary directory or keep output. The pipeline
    # writes artifacts to config.outdir; ensure downstream readers use the same path.
    temp_context = None
    if keep_batch_output:
        # Ensure configured outdir exists and use it directly
        os.makedirs(config.outdir, exist_ok=True)
        batch_output_dir = config.outdir
    else:
        # Use a temporary directory and point config.outdir to it
        temp_context = tempfile.TemporaryDirectory(prefix="proteogyver_pipeline_")
        config.outdir = temp_context.name
        batch_output_dir = config.outdir
    try:
        if temp_context:
            logger.info(f"Running batch pipeline with temporary output: {batch_output_dir}")
        else:
            logger.info(f"Running batch pipeline with permanent output: {batch_output_dir}")
        
        # Step 1: Run the batch pipeline
        summary = {}
        try:
            logger.info("Step 1: Running batch pipeline...")
            
            summary = pipeline_batch.run_pipeline(config, parameters)
            
            # Check if pipeline returned error due to warnings
            if "error" in summary and "warnings" in summary:
                error_msg = f"{summary['error']}"
                logger.error(error_msg)
                logger.error(f"Warnings: {summary['warnings']}")
                
                # Write error file to input directory
                input_dir = os.path.dirname(os.path.realpath(toml_path))
                error_file = os.path.join(input_dir, "ERRORS.txt")
                from datetime import datetime as _dt
                ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(error_file, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] Errors:\n")
                    for warning in summary['warnings']:
                        f.write(f"[{ts}] - {warning}\n")
                logger.info(f"Warnings written to {error_file}")
                
                raise ValueError(error_msg)
            
            logger.info(f"Batch pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Batch pipeline failed: {e}")
            raise
            
            # Step 2: Build data stores and export using GUI infrastructure
        try:
            logger.info("Step 2: Building data stores and exporting using GUI infrastructure...")
            
            # Detect workflow from config
            workflow = config.workflow
            
            # Build data stores from batch output
            data_stores = batch_data_store_builder.build_data_stores_from_batch_output(batch_output_dir, workflow)
            logger.info(f"Built {len(data_stores)} data stores")
            
            data_export_result = infra.save_data_stores(data_stores, export_dir)
            logger.info(f"Data export completed using GUI infrastructure")
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise
        
        figure_summary = {}
        try:
            logger.info("Step 3: Generating figures using GUI infrastructure...")
            figures_export_dir = export_dir
            
            figure_summary = batch_figure_builder_from_divs.save_batch_figures_using_saved_divs(
                batch_output_dir=batch_output_dir,
                export_dir=figures_export_dir,
                workflow=workflow,
                parameters=parameters,
                output_formats=plot_formats
            )
            logger.info(f"Figure generation completed using GUI infrastructure")
            
        except Exception as e:
            logger.error(f"Figure generation failed: {e}")
            # Don't raise - figures are optional
            figure_summary = {"error": str(e)}
    
        guide_path = os.path.join(os.path.dirname(__file__), 'data', 'output_guide.md')
        infra.write_README(export_dir, guide_path)
    finally:
        # Clean up temporary directory if used
        if temp_context:
            temp_context.cleanup()
    
    # Final summary
    result = {
        "pipeline_summary": summary,
        "export_directory": export_dir,
        "session_name": session_name,
        "workflow": workflow,
        "data_stores_built": len(data_stores),
        "batch_output_directory": batch_output_dir if keep_batch_output else "temporary",
        "figures_generated": figure_summary.get("analysis_divs_count", 0) if isinstance(figure_summary, dict) and "error" not in figure_summary else 0,
        "figure_details": figure_summary,
    }
    
    logger.info(f"pipeline finished. Export directory: {export_dir}")
    return result


def main():
    """Command line interface for the batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Run ProteoGyver batch pipeline using GUI infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Examples:
  # Run proteomics pipeline
  python run_as_pipeline.py proteomics_pipeline.toml
  
  # Run interactomics pipeline with custom export directory
  python run_as_pipeline.py interactomics_pipeline.toml --export-dir my_results
  
  # Keep intermediate batch files for debugging
  python run_as_pipeline.py config.toml --keep-batch-output
  
  # Run without plot generation
  python run_as_pipeline.py config.toml --no-plots
        """
    )
    
    parser.add_argument("toml_file", help="TOML configuration file for the pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
        # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    toml_dir = Path(args.toml_file).resolve().parent
    log_file = toml_dir / f"{timestamp}_pipeline.log"

    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file))
        ]
    )

    # Validate inputs
    if not os.path.exists(args.toml_file):
        logger.error(f"TOML file not found: {args.toml_file}")
        sys.exit(1)
    try:
        # Run the pipeline
        result = run_batch_pipeline(
            toml_path=args.toml_file,
        )
        
        # Print summary
        print(f"\nPipeline Complete!")
        print(f"Workflow: {result['workflow']}")
        
        if "error" in result.get('figure_details', {}):
            print(f"ERROR: Figure generation failed: {result['figure_details']['error']}")
        
        print(f"\nSession name: {result['session_name']}")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        print(f"\nERROR: Pipeline Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
