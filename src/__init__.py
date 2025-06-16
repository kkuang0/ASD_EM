# src/__init__.py - Minimal version for debugging

"""
EM Axon Classification Package
"""

__version__ = "1.0.0"

# For now, just make this an importable package
# Individual modules will be imported directly as needed

# Optional: Add this for debugging
import sys
import os

# Add current directory to path (helps with imports)
_current_dir = os.path.dirname(__file__)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)