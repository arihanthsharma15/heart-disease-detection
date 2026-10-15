"""
pytest configuration — add src/ to sys.path so tests can import heart_disease.
"""
import sys
from pathlib import Path

# Make `src/` importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
