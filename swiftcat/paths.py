"""
Module for defining paths used by swiftcat.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATA_DIR = Path.home() / "swiftcat_data"


def get_data_dir() -> Path:
    """
    Get the base data directory for swiftcat, from the SWIFTDATADIR
    environment variable (or .env) if set, otherwise a default directory
    in the user's home directory

    :return: Path to the data directory
    """
    load_dotenv()
    data_dir = Path(os.getenv("SWIFTDATADIR") or DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
