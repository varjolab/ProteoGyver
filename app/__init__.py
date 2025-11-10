"""ProteoGyver - A web-based platform for proteomics and interactomics data analysis."""

try:
    from app._version import __version__
except ImportError:
    # Fallback for when running from within the app directory
    from _version import __version__

__all__ = ["__version__"]

