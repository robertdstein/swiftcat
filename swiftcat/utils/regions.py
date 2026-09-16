"""
Module for writing DS9 region files that overlay classified sources on
an image, colour-coded by category
"""

from pathlib import Path

import pandas as pd

DEFAULT_RADIUS_ARCSEC = 5.0
DEFAULT_COLOR = "magenta"
CATEGORY_COLORS = {
    "new": "red",
    "known_star": "green",
    "known_galaxy": "blue",
    "known_unclear": "yellow",
}

RA_COL = "ALPHA_J2000"
DEC_COL = "DELTA_J2000"
CATEGORY_COL = "category"


def write_region_file(
    image_path: Path,
    sources: pd.DataFrame,
    out_path: Path | None = None,
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
) -> Path:
    """
    Function to write a DS9 region file overlaying each source's
    position, colour-coded by its classification, for visual inspection
    alongside an image

    :param image_path: Path to the image the sources belong to (used
        only to name the output region file when out_path isn't given)
    :param sources: Table of sources, as produced by crossmatch_ps1 -
        needs ALPHA_J2000/DELTA_J2000 (degrees) and category columns
    :param out_path: Path to write the region file to; defaults to
        image_path with a .reg extension
    :param radius_arcsec: Circle radius to draw around each source
    :return: Path the region file was written to
    """
    if out_path is None:
        out_path = image_path.with_suffix(".reg")

    lines = ["# Region file format: DS9 version 4.1", "fk5"]
    for _, source in sources.iterrows():
        category = source[CATEGORY_COL]
        color = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
        lines.append(
            f'circle({source[RA_COL]},{source[DEC_COL]},{radius_arcsec}")'
            f" # color={color} text={{{category}}}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return out_path
