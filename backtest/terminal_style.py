"""Terminal styling utilities"""

def format_status_line(message: str, status: str = "info") -> str:
    """Format a status message for terminal output"""
    status_symbols = {
        "error": "[X]",
        "warning": "[!]",
        "info": "[*]",
        "success": "[+]",
    }
    symbol = status_symbols.get(status, "[*]")
    return f"{symbol} {message}"
