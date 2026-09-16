"""
Module for defining paths used by swiftcat.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_IMAGE_DIR = Path.home() / "swift_images"
DEFAULT_DATA_DIR = Path.home() / "swiftcat_data"


def _get_directory_from_env(env_var: str, default: Path) -> Path:
    """
    Function to resolve a directory from an environment variable (or
    .env), falling back to a default if unset/blank, creating it if it
    does not yet exist

    :param env_var: Name of the environment variable to read
    :param default: Fallback directory if the environment variable is
        unset or blank
    :return: Path to the directory
    """
    load_dotenv()
    directory = Path(os.getenv(env_var) or default)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_image_dir() -> Path:
    """
    Get the directory where raw/processed Swift UVOT images live, from
    the SWIFT_IMAGE_DIR environment variable (or .env) if set, otherwise
    a default directory in the user's home directory

    :return: Path to the image directory
    """
    return _get_directory_from_env("SWIFT_IMAGE_DIR", DEFAULT_IMAGE_DIR)


def get_data_dir() -> Path:
    """
    Get the directory where swiftcat writes its own output data, from
    the SWIFTCAT_DATA_DIR environment variable (or .env) if set,
    otherwise a default directory in the user's home directory

    :return: Path to the data directory
    """
    return _get_directory_from_env("SWIFTCAT_DATA_DIR", DEFAULT_DATA_DIR)
