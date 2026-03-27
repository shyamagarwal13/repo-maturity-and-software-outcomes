"""Root conftest — ensures project root is on sys.path and .env is loaded."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
