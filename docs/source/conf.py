# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
 
# Add project root to path so we can import app as a package
sys.path.insert(0, os.path.abspath('../..'))  # Path to project root

# Create a custom import hook to redirect 'components' imports to 'app.components'
# This allows modules to use 'from components import ...' while still
# allowing 'app' to be imported as a package
import importlib
import importlib.util
import importlib.abc

class AppImportFinder(importlib.abc.MetaPathFinder):
    """Custom finder that redirects top-level imports to 'app.*' modules."""
    
    # Map of top-level module names to their app.* equivalents
    MODULE_MAP = {
        'components': 'app.components',
        'pipeline_module': 'app.pipeline_module',
        'element_styles': 'app.element_styles',
        'database_updater': 'app.database_updater',
        '_version': 'app._version',
    }
    
    def find_spec(self, name, path, target=None):
        # Check if this is a top-level module that should be redirected
        if name in self.MODULE_MAP:
            app_name = self.MODULE_MAP[name]
            app_spec = importlib.util.find_spec(app_name)
            if app_spec is not None:
                return self._create_redirect_spec(name, app_name, app_spec)
        
        # Check if this is a submodule of a redirected module (e.g., components.tools)
        for top_level, app_prefix in self.MODULE_MAP.items():
            if name.startswith(top_level + '.'):
                app_name = name.replace(top_level, app_prefix, 1)
                app_spec = importlib.util.find_spec(app_name)
                if app_spec is not None:
                    return self._create_redirect_spec(name, app_name, app_spec)
        
        return None
    
    def _create_redirect_spec(self, name, app_name, app_spec):
        """Create a spec that redirects name to app_name."""
        class RedirectLoader(importlib.abc.Loader):
            def __init__(self, app_module_name):
                self.app_module_name = app_module_name
            
            def create_module(self, spec):
                # Import the app module
                app_module = importlib.import_module(self.app_module_name)
                # Register the app module with the original name too
                sys.modules[name] = app_module
                return app_module
            
            def exec_module(self, module):
                # Module is already loaded, nothing to execute
                pass
        
        # Create a new spec with our custom loader
        spec = importlib.util.spec_from_loader(name, RedirectLoader(app_name))
        if spec and app_spec and hasattr(app_spec, 'submodule_search_locations') and app_spec.submodule_search_locations:
            spec.submodule_search_locations = app_spec.submodule_search_locations
        return spec

# Install the custom import finder
sys.meta_path.insert(0, AppImportFinder())  

# Copy example files directory to build output so links work
def copy_example_files(app, exception):
    """Copy example files directory to build output after build.
    
    This allows the link ../app/data/PG example files/ to work
    from the built HTML files in docs/build/html/
    """
    if exception:
        return
    import shutil
    # app.outdir is docs/build/html, so ../app/data is docs/build/app/data
    build_dir = os.path.join(app.outdir, '..', 'app', 'data')
    # Source is at project root: app/data/PG example files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(project_root, 'app', 'data', 'PG example files')
    target_dir = os.path.join(build_dir, 'PG example files')
    if os.path.exists(source_dir):
        os.makedirs(build_dir, exist_ok=True)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

def setup(app):
    """Sphinx setup function to register event handlers."""
    app.connect('build-finished', copy_example_files)
    return {'version': '1.0', 'parallel_read_safe': True}

project = 'ProteoGyver'
copyright = '2025, Kari Salokas'
author = 'Kari Salokas'
release = '1.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [  
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings  
    'sphinx.ext.napoleon',       # Support Google/NumPy-style docstrings  
    'sphinx_autodoc_typehints',  # Include type hints in docs  
    'sphinx_rtd_theme',          # Use Read the Docs theme  
    "myst_parser",
    "sphinx.ext.viewcode",
]  
templates_path = ['_templates']
exclude_patterns = []

# -- Options for autodoc -------------------------------------------------
# Suppress warnings for modules that can't be imported and MyST cross-reference warnings
suppress_warnings = ['myst.xref_missing']

# Autodoc settings
autodoc_mock_imports = []  # List of modules to mock if needed
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'ProteoGyver Documentation'  # Title in browser tabs  
copyright = '2025, Kari Salokas'            # Copyright notice  
author = 'Kari Salokas'  