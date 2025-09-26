#!/usr/bin/env python3
"""
Refactored ProteoGyver Batch Pipeline with GUI Infrastructure

This script runs the complete batch pipeline using the same infrastructure
as the GUI, ensuring identical behavior and maintainability.

Key improvements over the original batch system:
- Uses infra.save_data_stores for data export (same as GUI)
- Uses infra.save_figures for figure export (same as GUI) 
- Constructs data stores in the exact format expected by GUI
- Maintains identical directory structure and file formats
- Ensures complete compatibility with GUI export system
"""

import os
import sys
import argparse
import tempfile
from datetime import datetime
import logging

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from pipeline_batch import run_pipeline
from pipeline_from_toml import _load_config
from batch_data_store_builder import build_data_stores_from_batch_output, create_data_store_component
from batch_figure_builder_from_divs import save_batch_figures_using_saved_divs
from components.infra import save_data_stores, write_README
from components import parsing

logger = logging.getLogger(__name__)


def run_refactored_batch_pipeline(toml_path: str, export_dir: str = None, 
                                generate_plots: bool = True,
                                plot_formats: list = None,
                                keep_batch_output: bool = False,
                                batch_output_dir: str = None,
                                rerun: bool = False) -> dict:
    """Run the complete refactored batch pipeline using GUI infrastructure.
    
    Args:
        toml_path: Path to the TOML configuration file
        export_dir: Directory for final export (auto-generated if None)
        generate_plots: Whether to generate plots (default True)
        plot_formats: List of plot formats ['html', 'pdf', 'png'] (default all)
        keep_batch_output: Whether to keep intermediate batch JSON files
        batch_output_dir: Directory for intermediate batch JSON files
        rerun: Whether to rerun the pipeline. If the batch_output_dir is provided and exists, the pipeline will rerun only, if it's empty.
    Returns:
        dict: Summary of pipeline execution, export, and plot generation
    """
    if plot_formats is None:
        plot_formats = ['html', 'pdf', 'png']
    parameters_file = 'app/parameters.toml'
    parameters = parsing.parse_parameters(parameters_file)
    # Generate session name from timestamp
    session_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}--refactored-batch"
    
    if export_dir is None:
        export_dir = f"refactored_export_{session_name}"
    
    # Determine whether to use temporary directory or keep output
    if keep_batch_output:
        # Use permanent directory in project
        if batch_output_dir is None:
            if export_dir is not None:
                batch_output_dir = os.path.join(export_dir, "batch_output")
            else:
                batch_output_dir = f"batch_output_{session_name}"
        os.makedirs(batch_output_dir, exist_ok=True)
        # Convert to absolute path to avoid issues when changing directories
        batch_output_dir = os.path.abspath(batch_output_dir)
        temp_context = None
    else:
        # Use temporary directory
        batch_output_dir = None
        temp_context = tempfile.TemporaryDirectory(prefix="proteogyver_refactored_batch_")
    
    try:
        if temp_context:
            batch_output_dir = temp_context.name
            logger.info(f"Running batch pipeline with temporary output: {batch_output_dir}")
        else:
            logger.info(f"Running batch pipeline with permanent output: {batch_output_dir}")
        
        # Step 1: Run the batch pipeline
        summary = {}
        try:
            logger.info("Step 1: Running batch pipeline...")
            
            # Change to app directory for pipeline execution
            original_cwd = os.getcwd()
            app_dir = os.path.join(os.path.dirname(__file__), 'app')
            os.chdir(app_dir)
            
            # Adjust TOML path to be relative to app directory
            abs_toml_path = os.path.abspath(os.path.join(original_cwd, toml_path))
            
            try:
                config = _load_config(abs_toml_path)
                
                # Update config to use batch output directory
                original_outdir = config.outdir
                config.outdir = batch_output_dir
                
                summary = run_pipeline(config)
                logger.info(f"Batch pipeline completed successfully")
                
            finally:
                # Always restore original directory
                os.chdir(original_cwd)
            
        except Exception as e:
            logger.error(f"Batch pipeline failed: {e}")
            raise
            
            # Step 2: Build data stores and export using GUI infrastructure
        try:
            logger.info("Step 2: Building data stores and exporting using GUI infrastructure...")
            
            # Detect workflow from config
            workflow = config.workflow
            
            # Build data stores from batch output
            data_stores = build_data_stores_from_batch_output(batch_output_dir, workflow)
            logger.info(f"Built {len(data_stores)} data stores")
            
            # Use GUI's save_data_stores function
            data_export_result = save_data_stores(data_stores, export_dir)
            logger.info(f"Data export completed using GUI infrastructure")
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise
        
        # Step 3: Generate figures using GUI infrastructure (optional)
        figure_summary = {}
        if generate_plots:
            try:
                logger.info("Step 3: Generating figures using GUI infrastructure...")
                figures_export_dir = export_dir
                
                figure_summary = save_batch_figures_using_saved_divs(
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
    
        write_README(export_dir, os.path.join('app', 'data','output_guide.md'))
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
        "infrastructure_used": "GUI (infra.save_data_stores + infra.save_figures)"
    }
    
    logger.info(f"Refactored pipeline finished. Export directory: {export_dir}")
    return result


def main():
    """Command line interface for the refactored batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Run refactored ProteoGyver batch pipeline using GUI infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This refactored pipeline uses the same infrastructure as the GUI for:
- Data export (infra.save_data_stores)
- Figure export (infra.save_figures) 
- Data store construction

This ensures identical behavior and easier maintenance.

Examples:
  # Run proteomics pipeline
  python run_batch_refactored.py proteomics_pipeline.toml
  
  # Run interactomics pipeline with custom export directory
  python run_batch_refactored.py interactomics_pipeline.toml --export-dir my_results
  
  # Keep intermediate batch files for debugging
  python run_batch_refactored.py config.toml --keep-batch-output
  
  # Run without plot generation
  python run_batch_refactored.py config.toml --no-plots
        """
    )
    
    parser.add_argument("toml_file", help="TOML configuration file for the pipeline")
    parser.add_argument("--export-dir", help="Directory for final export (auto-generated if not specified)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--plot-formats", nargs="+", choices=["html", "pdf", "png"], 
                       help="Plot formats to generate (default: all)")
    parser.add_argument("--keep-batch-output", action="store_true", 
                       help="Keep intermediate batch JSON files (useful for debugging)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"refactored_batch_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    # Validate inputs
    if not os.path.exists(args.toml_file):
        logger.error(f"TOML file not found: {args.toml_file}")
        sys.exit(1)
    if args.keep_batch_output:
        batch_output_dir = f"{args.export_dir}/batch_output"
    try:
        # Run the refactored pipeline
        result = run_refactored_batch_pipeline(
            toml_path=args.toml_file,
            export_dir=args.export_dir,
            generate_plots=not args.no_plots,
            plot_formats=args.plot_formats,
            keep_batch_output=args.keep_batch_output
        )
        
        # Print summary
        print(f"\\n✅ Refactored Pipeline Complete!")
        print(f"🔧 Infrastructure: {result['infrastructure_used']}")
        print(f"📁 Export Directory: {result['export_directory']}")
        print(f"📊 Data Stores Built: {result['data_stores_built']}")
        print(f"🧬 Workflow: {result['workflow']}")
        
        if result['figures_generated'] > 0:
            print(f"📈 Figures Generated: {result['figures_generated']}")
            print(f"🎨 Output Formats: {', '.join(result['figure_details'].get('output_formats', []))}")
            if result['figure_details'].get('commonality_pdf_generated', False):
                print("📊 Commonality PDF generated")
        elif "error" in result.get('figure_details', {}):
            print(f"⚠️  Figure generation failed: {result['figure_details']['error']}")
        
        if args.keep_batch_output:
            print(f"📂 Batch Output Kept: {result['batch_output_directory']}")
        
        print(f"\\n🎯 Session: {result['session_name']}")
        print("\\n🚀 This pipeline uses the same infrastructure as the GUI!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\\n❌ Pipeline Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
