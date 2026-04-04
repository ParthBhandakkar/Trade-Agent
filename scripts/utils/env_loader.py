from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_root_env(repo_root: Path, *, override: bool = False) -> Path | None:
    """Load the repository root `.env` if it exists."""
    env_path = repo_root / ".env"
    if not env_path.exists():
        return None
    load_dotenv(env_path, override=override)
    return env_path
