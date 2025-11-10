"""API endpoint and Celery task for accepting and saving pipeline input files.

This module provides a Flask API endpoint that accepts file uploads (data_table,
sample_table, pipeline_toml, and optionally proteomics_comparisons) and a Celery
background task that saves these files to a specified directory.

The API runs continuously as part of the Flask server, and file saving is handled
asynchronously by Celery workers.
"""

import os
import logging
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from flask import request, jsonify, send_file
from werkzeug.utils import secure_filename
from celery import shared_task
from uuid import uuid4
from components.tools.utils import save_toml, read_toml, load_toml

logger = logging.getLogger("file_upload_api")


@shared_task
def save_uploaded_files(
    data_table_content: bytes,
    data_table_filename: str,
    sample_table_content: bytes,
    sample_table_filename: str,
    pipeline_toml_content: bytes,
    pipeline_toml_filename: str,
    proteomics_comparisons_content: Optional[bytes] = None,
    proteomics_comparisons_filename: Optional[str] = None,
    output_directory: Optional[str] = None,
    upload_dir_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Save uploaded files to the specified directory.

    This Celery task saves the uploaded files (data_table, sample_table,
    pipeline_toml, and optionally proteomics_comparisons) to a specified
    output directory. If no output directory is provided, it uses the
    default from parameters.toml.

    :param data_table_content: Binary content of the data table file.
    :param data_table_filename: Original filename of the data table.
    :param sample_table_content: Binary content of the sample table file.
    :param sample_table_filename: Original filename of the sample table.
    :param pipeline_toml_content: Binary content of the pipeline TOML file.
    :param pipeline_toml_filename: Original filename of the pipeline TOML.
    :param proteomics_comparisons_content: Optional binary content of proteomics comparisons file.
    :param proteomics_comparisons_filename: Optional original filename of proteomics comparisons.
    :param output_directory: Optional output directory path. If None, uses default from parameters.toml.
    :param upload_dir_name: Optional upload directory name. If None, generates a new one.
    :returns: Dictionary with status, message, and saved file paths.
    """
    try:
        # Load parameters to get default output directory if not provided
        root_dir = Path(__file__).resolve().parents[1]
        parameters_path = os.path.join(root_dir, 'config','parameters.toml')
        parameters = read_toml(Path(parameters_path))
        
        if output_directory is None:
            pipeline_path = parameters.get('Pipeline module', {}).get('Input watch directory', [])
            if isinstance(pipeline_path, list):
                # Filter out None values and ensure all are strings
                path_parts = [str(p) for p in pipeline_path if p is not None]
                output_directory = os.path.join(*path_parts) if path_parts else str(pipeline_path)
            else:
                output_directory = str(pipeline_path)
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Create timestamped subdirectory for this upload
        if upload_dir_name is None:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            upload_dir_name = f'{timestamp}--api_upload_{uuid4()}'
        
        upload_dir = os.path.join(output_directory, upload_dir_name)
        os.makedirs(upload_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save data table
        data_table_path = os.path.join(upload_dir, secure_filename(data_table_filename))
        with open(data_table_path, 'wb') as f:
            f.write(data_table_content)
        saved_files['data_table'] = data_table_path
        logger.info(f"Saved data table to {data_table_path}")
        
        # Save sample table
        sample_table_path = os.path.join(upload_dir, secure_filename(sample_table_filename))
        with open(sample_table_path, 'wb') as f:
            f.write(sample_table_content)
        saved_files['sample_table'] = sample_table_path
        logger.info(f"Saved sample table to {sample_table_path}")
        
        # Save pipeline TOML
        pipeline_toml_path = os.path.join(upload_dir, secure_filename(pipeline_toml_filename))
        with open(pipeline_toml_path, 'wb') as f:
            f.write(pipeline_toml_content)
        saved_files['pipeline_toml'] = pipeline_toml_path
        logger.info(f"Saved pipeline TOML to {pipeline_toml_path}")
        
        # Save proteomics comparisons if provided
        if proteomics_comparisons_content is not None and proteomics_comparisons_filename:
            proteomics_comparisons_path = os.path.join(
                upload_dir, secure_filename(proteomics_comparisons_filename)
            )
            with open(proteomics_comparisons_path, 'wb') as f:
                f.write(proteomics_comparisons_content)
            saved_files['proteomics_comparisons'] = proteomics_comparisons_path
            logger.info(f"Saved proteomics comparisons to {proteomics_comparisons_path}")
        
        # Modify pipeline TOML to update file paths
        try:
            # Parse TOML
            doc = load_toml(Path(pipeline_toml_path))
            
            # Ensure 'general' section exists
            if 'general' not in doc:
                doc['general'] = {}
            
            # Update or create 'data' key in general section
            data_filename = secure_filename(data_table_filename)
            doc['general']['data'] = data_filename
            logger.info(f"Updated pipeline TOML: general.data = {data_filename}")
            
            # Update or create 'sample table' key in general section
            sample_table_filename_secure = secure_filename(sample_table_filename)
            doc['general']['sample table'] = sample_table_filename_secure
            logger.info(f"Updated pipeline TOML: general['sample table'] = {sample_table_filename_secure}")
            
            # Update proteomics comparisons if provided
            if proteomics_comparisons_content is not None and proteomics_comparisons_filename:
                # Ensure 'proteomics' section exists
                if 'proteomics' not in doc:
                    doc['proteomics'] = {}
                
                # Update or create 'comparison_file' key in proteomics section
                proteomics_comparisons_filename_secure = secure_filename(proteomics_comparisons_filename)
                doc['proteomics']['comparison_file'] = proteomics_comparisons_filename_secure
                logger.info(f"Updated pipeline TOML: proteomics.comparison_file = {proteomics_comparisons_filename_secure}")
            
            # Write the modified TOML back
            save_toml(doc, Path(pipeline_toml_path))
            
            logger.info(f"Successfully modified pipeline TOML at {pipeline_toml_path}")
        
        except Exception as e:
            # Log error but don't fail the task - files are already saved
            error_msg = f"Error modifying pipeline TOML: {str(e)}"
            logger.warning(error_msg, exc_info=True)
        
        
        return {
            'status': 'success',
            'message': f'Files saved successfully to {upload_dir}',
            'upload_directory': upload_dir,
            'upload_directory_name': upload_dir_name,
            'saved_files': saved_files
        }
    
    except Exception as e:
        error_msg = f"Error saving uploaded files: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'status': 'error',
            'message': error_msg,
            'upload_directory': None,
            'saved_files': {}
        }


def register_file_upload_api(server, celery_app_instance=None):
    """Register the file upload API endpoint with the Flask server.

    This function sets up a POST endpoint at '/api/upload-pipeline-files' that
    accepts multipart/form-data file uploads. The endpoint accepts:
    - data_table (required): Data table file
    - sample_table (required): Sample table file
    - pipeline_toml (required): Pipeline configuration TOML file
    - proteomics_comparisons (optional): Proteomics comparisons file

    Files are processed asynchronously via a Celery task.

    :param server: Flask server instance (typically app.server from Dash).
    :param celery_app_instance: Optional Celery app instance to use for task execution.
    :returns: None
    """
    # Store celery_app instance for use in the endpoint
    celery_app_for_task = celery_app_instance
    
    @server.route('/api/upload-pipeline-files', methods=['POST'])
    def upload_pipeline_files():
        """API endpoint for uploading pipeline input files.

        Accepts POST requests with multipart/form-data containing:
        - data_table (required): File upload
        - sample_table (required): File upload
        - pipeline_toml (required): File upload
        - proteomics_comparisons (optional): File upload
        - output_directory (optional): Custom output directory path

        :returns: JSON response with task ID and status.
        """
        try:
            # Check for required files
            if 'data_table' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required file: data_table'
                }), 400
            
            if 'sample_table' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required file: sample_table'
                }), 400
            
            if 'pipeline_toml' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required file: pipeline_toml'
                }), 400
            
            # Get required files
            data_table_file = request.files['data_table']
            sample_table_file = request.files['sample_table']
            pipeline_toml_file = request.files['pipeline_toml']
            
            # Check if files are actually uploaded (not empty)
            if data_table_file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'data_table file is empty'
                }), 400
            
            if sample_table_file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'sample_table file is empty'
                }), 400
            
            if pipeline_toml_file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'pipeline_toml file is empty'
                }), 400
            
            # Read file contents
            data_table_content = data_table_file.read()
            sample_table_content = sample_table_file.read()
            pipeline_toml_content = pipeline_toml_file.read()
            
            # Get optional proteomics_comparisons file
            proteomics_comparisons_content = None
            proteomics_comparisons_filename = None
            if 'proteomics_comparisons' in request.files:
                proteomics_comparisons_file = request.files['proteomics_comparisons']
                if proteomics_comparisons_file.filename != '':
                    proteomics_comparisons_content = proteomics_comparisons_file.read()
                    proteomics_comparisons_filename = proteomics_comparisons_file.filename
            
            # Get optional output directory
            output_directory = request.form.get('output_directory', None)
            
            # Generate upload directory name synchronously so we can return it immediately
            # This matches what the task will create
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            upload_dir_name = f'{timestamp}--api_upload_{uuid4()}'
            
            # Trigger Celery task to save files
            try:
                # Use the celery_app instance if provided, otherwise try to import it
                app_instance = celery_app_for_task
                if app_instance is None:
                    try:
                        from app import celery_app
                        app_instance = celery_app
                    except ImportError:
                        app_instance = None
                
                if app_instance is not None:
                    # Use send_task with the configured celery_app instance
                    task = app_instance.send_task(
                        'components.file_upload_api.save_uploaded_files',
                        args=[],
                        kwargs={
                            'data_table_content': data_table_content,
                            'data_table_filename': data_table_file.filename,
                            'sample_table_content': sample_table_content,
                            'sample_table_filename': sample_table_file.filename,
                            'pipeline_toml_content': pipeline_toml_content,
                            'pipeline_toml_filename': pipeline_toml_file.filename,
                            'proteomics_comparisons_content': proteomics_comparisons_content,
                            'proteomics_comparisons_filename': proteomics_comparisons_filename,
                            'output_directory': output_directory,
                            'upload_dir_name': upload_dir_name
                        }
                    )
                else:
                    # Fallback to using shared_task directly
                    task = save_uploaded_files.delay(
                        data_table_content=data_table_content,
                        data_table_filename=data_table_file.filename,
                        sample_table_content=sample_table_content,
                        sample_table_filename=sample_table_file.filename,
                        pipeline_toml_content=pipeline_toml_content,
                        pipeline_toml_filename=pipeline_toml_file.filename,
                        proteomics_comparisons_content=proteomics_comparisons_content,
                        proteomics_comparisons_filename=proteomics_comparisons_filename,
                        output_directory=output_directory,
                        upload_dir_name=upload_dir_name
                    )
                
                logger.info(f"File upload task queued: {task.id}, upload directory: {upload_dir_name}")
                
                return jsonify({
                    'status': 'accepted',
                    'message': 'Files uploaded successfully, processing in background',
                    'task_id': task.id,
                    'upload_directory_name': upload_dir_name
                }), 202
            
            except Exception as celery_error:
                # Check if it's a Redis connection error
                error_str = str(celery_error)
                if 'Connection refused' in error_str or '111' in error_str or 'ConnectionError' in str(type(celery_error).__name__):
                    logger.error(f"Celery/Redis connection error: {celery_error}", exc_info=True)
                    return jsonify({
                        'status': 'error',
                        'message': 'Cannot connect to Celery/Redis. Please ensure Redis is running and Celery workers are started.'
                    }), 503  # Service Unavailable
                else:
                    # Re-raise other errors to be caught by outer exception handler
                    raise
        
        except Exception as e:
            error_msg = f"Error processing file upload request: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500
    
    @server.route('/api/pipeline-status', methods=['GET'])
    def check_pipeline_status():
        """API endpoint for checking pipeline processing status.
        
        Accepts GET requests with query parameter:
        - upload_directory_name (required): Name of the upload directory
        
        Returns the status of the pipeline run:
        - 'processing': Pipeline is currently running (.pg_analyzing.lock exists)
        - 'success': Pipeline completed successfully (pipeline.success.txt exists)
        - 'error': Pipeline failed (pipeline.failure.txt exists)
        - 'not_found': Upload directory not found
        - 'unknown': No status files found (may not have started yet)
        
        If status is 'error', also returns the contents of ERRORS.txt.
        
        :returns: JSON response with status and optional error message.
        """
        try:
            # Get upload directory name from query parameter
            upload_dir_name = request.args.get('upload_directory_name')
            if not upload_dir_name:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required parameter: upload_directory_name'
                }), 400
            
            # Load parameters to get the base output directory
            root_dir = Path(__file__).resolve().parents[1]
            parameters_path = os.path.join(root_dir, 'config','parameters.toml')
            parameters = read_toml(Path(parameters_path))
            pipeline_path = parameters.get('Pipeline module', {}).get('Input watch directory', [])
            
            if isinstance(pipeline_path, list):
                # Filter out None values and ensure all are strings
                path_parts = [str(p) for p in pipeline_path if p is not None]
                base_directory = os.path.join(*path_parts) if path_parts else str(pipeline_path)
            else:
                base_directory = str(pipeline_path)
            
            # Construct full path to upload directory
            upload_dir = os.path.join(base_directory, upload_dir_name)
            upload_dir_path = Path(upload_dir)
            
            # Check if directory exists
            if not upload_dir_path.exists() or not upload_dir_path.is_dir():
                return jsonify({
                    'status': 'not_found',
                    'message': f'Upload directory not found: {upload_dir_name}',
                    'upload_directory_name': upload_dir_name
                }), 404
            
            # Check for status files
            lock_file = upload_dir_path / '.pg_analyzing.lock'
            watcher_log = upload_dir_path / 'watcher.log'
            success_file = upload_dir_path / 'pipeline.success.txt'
            failure_file = upload_dir_path / 'pipeline.failure.txt'
            errors_file = upload_dir_path / 'ERRORS.txt'
            
            # Determine status based on file presence
            if lock_file.exists():
                # Pipeline is currently processing
                return jsonify({
                    'status': 'processing',
                    'message': 'Pipeline is currently running',
                    'upload_directory_name': upload_dir_name
                }), 200
            
            if success_file.exists():
                # Pipeline completed successfully
                return jsonify({
                    'status': 'success',
                    'message': 'Pipeline completed successfully',
                    'upload_directory_name': upload_dir_name,
                    'success_timestamp': success_file.read_text(encoding='utf-8').strip() if success_file.exists() else None
                }), 200
            
            if failure_file.exists():
                # Pipeline failed
                error_content = None
                if errors_file.exists():
                    try:
                        error_content = errors_file.read_text(encoding='utf-8')
                    except Exception as e:
                        logger.warning(f"Error reading ERRORS.txt: {e}")
                        error_content = f"Error reading ERRORS.txt: {str(e)}"
                
                return jsonify({
                    'status': 'error',
                    'message': 'Pipeline execution failed',
                    'upload_directory_name': upload_dir_name,
                    'error_message': error_content,
                    'failure_timestamp': failure_file.read_text(encoding='utf-8').strip() if failure_file.exists() else None
                }), 200
            
            # Pipeline being watched, not yet processing. watcher.log will remain after it's done, so we will check it last.
            if watcher_log.exists():
                return jsonify({
                    'status': 'processing',
                    'message': 'Pipeline is currently running',
                    'upload_directory_name': upload_dir_name
                }), 200

            
            return jsonify({
                'status': 'unknown',
                'message': 'No status files found. Pipeline may not have started yet.',
                'upload_directory_name': upload_dir_name
            }), 200
    
        except Exception as e:
            error_msg = f"Error checking pipeline status: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500
    
    @server.route('/api/download-output', methods=['GET'])
    def download_output():
        """API endpoint for downloading the PG output directory as a zip file.
        
        Accepts GET requests with query parameter:
        - upload_directory_name (required): Name of the upload directory
        
        Returns the "PG output" directory as a zip file named "PG output.zip".
        If the PG output directory doesn't exist, returns a 404 error.
        
        :returns: ZIP file download or error response.
        """
        try:
            # Get upload directory name from query parameter
            upload_dir_name = request.args.get('upload_directory_name')
            if not upload_dir_name:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required parameter: upload_directory_name'
                }), 400
            
            # Load parameters to get the base output directory
            root_dir = Path(__file__).resolve().parents[1]
            parameters_path = os.path.join(root_dir, 'config','parameters.toml')
            parameters = read_toml(Path(parameters_path))
            pipeline_path = parameters.get('Pipeline module', {}).get('Input watch directory', [])
            
            if isinstance(pipeline_path, list):
                # Filter out None values and ensure all are strings
                path_parts = [str(p) for p in pipeline_path if p is not None]
                base_directory = os.path.join(*path_parts) if path_parts else str(pipeline_path)
            else:
                base_directory = str(pipeline_path)
            
            # Construct full path to upload directory
            upload_dir = os.path.join(base_directory, upload_dir_name)
            upload_dir_path = Path(upload_dir)
            
            # Check if directory exists
            if not upload_dir_path.exists() or not upload_dir_path.is_dir():
                return jsonify({
                    'status': 'not_found',
                    'message': f'Upload directory not found: {upload_dir_name}'
                }), 404
            
            # Find PG output directory
            pg_output_dir = upload_dir_path / 'PG output'
            
            if not pg_output_dir.exists() or not pg_output_dir.is_dir():
                return jsonify({
                    'status': 'not_found',
                    'message': 'PG output directory not found. Pipeline may not have completed yet.'
                }), 404
            
            # Create a temporary zip file
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip_path = Path(temp_zip.name)
            temp_zip.close()
            
            try:
                # Create zip file
                with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Walk through the PG output directory and add all files
                    for root, dirs, files in os.walk(upload_dir_path):
                        for file in files:
                            file_path = Path(root) / file
                            # Create archive name relative to PG output directory
                            arcname = file_path.relative_to(upload_dir_path)
                            zipf.write(file_path, arcname)
                
                logger.info(f"Created zip file for PG output: {temp_zip_path}")
                
                # Send the zip file
                return send_file(
                    str(temp_zip_path),
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name='PG output.zip'
                )
            
            except Exception as zip_error:
                # Clean up temp file on error
                if temp_zip_path.exists():
                    temp_zip_path.unlink()
                logger.error(f"Error creating zip file: {zip_error}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': f'Error creating zip file: {str(zip_error)}'
                }), 500
        
        except Exception as e:
            error_msg = f"Error downloading output: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500

