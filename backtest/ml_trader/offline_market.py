"""Offline market data utilities"""

from pathlib import Path

# Default output directory for MT5 historical data
# Adjust this path based on your system setup
DEFAULT_EXNESS_STRUCTURED_HISTORY_ROOT = Path.home() / ".mt5_history" / "exness" / "structured" / "history"
