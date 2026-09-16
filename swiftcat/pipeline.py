"""
Module to run the full per-image pipeline: detect, then tag against PS1
"""

from pathlib import Path

import pandas as pd
from blastwave.query.boom import BoomClient

from swiftcat.crossmatch import crossmatch_ps1
from swiftcat.detect import find_sources


def detect_and_classify(
    image_path: Path,
    raw_subexposures: Path | None = None,
    client: BoomClient | None = None,
) -> pd.DataFrame:
    """
    Function to detect every source in a UVOT image, restricted to the
    overlap region, and tag each as known_star / known_galaxy /
    known_unclear / new against PS1

    :param image_path: Path to a UVOT image (any filter, any epoch)
    :param raw_subexposures: Raw multi-extension sky image the summed
        image was created from; if given, restricts detection to the
        region covered by every sub-exposure
    :param client: Optional shared BoomClient
    :return: Source table with `is_point_source` and `category` columns
    """
    cat = find_sources(image_path, raw_subexposures=raw_subexposures)
    return crossmatch_ps1(cat, image_path, client=client)
