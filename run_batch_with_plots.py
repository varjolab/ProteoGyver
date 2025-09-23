#!/usr/bin/env python3
"""
Complete ProteoGyver Batch Pipeline with Plots

This script runs the complete batch pipeline including:
1. Data processing using the batch pipeline
2. Export to GUI-compatible format 
3. Generation of all plots in multiple formats

The output matches the GUI export structure exactly, including:
- Data tables in TSV format
- Figures in HTML, PDF, and PNG formats
- Same directory structure as GUI export
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
from batch_to_gui_export import export_batch_results
from batch_plot_generator import generate_plots_from_batch

logger = logging.getLogger(__name__)


def run_complete_batch_pipeline(toml_path: str, export_dir: str = None, 
                              generate_plots: bool = True, 
                              plot_formats: list = None) -> dict:
    """Run the complete batch pipeline with data export and plot generation.
    
    Args:
        toml_path: Path to the TOML configuration file
        export_dir: Directory for final export (auto-generated if None)
        generate_plots: Whether to generate plots (default True)
        plot_formats: List of plot formats ['html', 'pdf', 'png'] (default all)
        
    Returns:
        dict: Summary of pipeline execution, export, and plot generation
    """
    if plot_formats is None:
        plot_formats = ['html', 'pdf', 'png']
        
    # Generate session name from timestamp
    session_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}--batch"
    
    if export_dir is None:
        export_dir = f"batch_export_{session_name}"
    
    # Create temporary directory for batch output
    with tempfile.TemporaryDirectory(prefix="proteogyver_batch_") as temp_dir:
        logger.info(f"Running batch pipeline with temporary output: {temp_dir}")
        
        # Step 1: Run the batch pipeline
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
                
                # Update config to use temporary directory
                original_outdir = config.outdir
                config.outdir = temp_dir
                
                summary = run_pipeline(config)
                logger.info(f"Batch pipeline completed: {summary}")
                
            finally:
                # Always restore original directory
                os.chdir(original_cwd)
            
        except Exception as e:
            logger.error(f"Batch pipeline failed: {e}")
            raise
        
        # Step 2: Export to GUI format
        try:
            logger.info("Step 2: Exporting to GUI format...")
            data_export_dir = os.path.join(export_dir, "Data")
            
            export_summary = export_batch_results(
                batch_output_dir=temp_dir,
                export_dir=data_export_dir,
                session_name=session_name
            )
            logger.info(f"Data export completed: {len(export_summary)} data stores exported")
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise
        
        # Step 3: Generate plots
        plot_summary = {}
        if generate_plots:
            try:
                logger.info("Step 3: Generating plots...")
                figures_export_dir = os.path.join(export_dir, "Figures")
                
                plot_summary = generate_plots_from_batch(
                    batch_output_dir=temp_dir,
                    export_dir=figures_export_dir,
                    session_name=session_name
                )
                logger.info(f"Plot generation completed: {len(plot_summary)} plots generated")
                
            except Exception as e:
                logger.error(f"Plot generation failed: {e}")
                # Don't raise - plots are optional
                plot_summary = {"error": str(e)}
    
    # Final summary
    result = {
        "pipeline_summary": summary,
        "export_directory": export_dir,
        "session_name": session_name,
        "data_stores_exported": len(export_summary) if export_summary else 0,
        "plots_generated": len(plot_summary) if isinstance(plot_summary, dict) and "error" not in plot_summary else 0,
        "plot_details": plot_summary
    }
    
    logger.info(f"Complete pipeline finished. Export directory: {export_dir}")
    return result


def main():
    """Command line interface for the complete batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Run complete ProteoGyver batch pipeline with data export and plot generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run proteomics pipeline
  python run_batch_with_plots.py proteomics_pipeline.toml
  
  # Run interactomics pipeline with custom export directory
  python run_batch_with_plots.py interactomics_pipeline.toml --export-dir my_results
  
  # Run without plot generation
  python run_batch_with_plots.py config.toml --no-plots
  
  # Generate only specific plot formats
  python run_batch_with_plots.py config.toml --plot-formats pdf png
        """
    )
    
    parser.add_argument("toml_file", help="TOML configuration file for the pipeline")
    parser.add_argument("--export-dir", help="Directory for final export (auto-generated if not specified)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--plot-formats", nargs="+", choices=["html", "pdf", "png"], 
                       help="Plot formats to generate (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"batch_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    # Validate inputs
    if not os.path.exists(args.toml_file):
        logger.error(f"TOML file not found: {args.toml_file}")
        sys.exit(1)
    
    try:
        # Run the complete pipeline
        result = run_complete_batch_pipeline(
            toml_path=args.toml_file,
            export_dir=args.export_dir,
            generate_plots=not args.no_plots,
            plot_formats=args.plot_formats
        )
        
        # Print summary
        print(f"\\n✅ Pipeline Complete!")
        print(f"📁 Export Directory: {result['export_directory']}")
        print(f"📊 Data Stores Exported: {result['data_stores_exported']}")
        
        if result['plots_generated'] > 0:
            print(f"📈 Plots Generated: {result['plots_generated']}")
            print("\\nGenerated Plots:")
            for plot_name, formats in result['plot_details'].items():
                if isinstance(formats, dict):
                    print(f"  • {plot_name}: {', '.join(formats.keys())}")
        elif "error" in result.get('plot_details', {}):
            print(f"⚠️  Plot generation failed: {result['plot_details']['error']}")
        
        print(f"\\n🎯 Session: {result['session_name']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\\n❌ Pipeline Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
