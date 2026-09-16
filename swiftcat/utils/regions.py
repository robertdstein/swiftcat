"""
Module for writing DS9 region files that overlay classified sources on
an image, colour-coded by category
"""

from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

DEFAULT_COLOR = "orange"
CATEGORY_COLORS = {
    "new": "green",
    "known_star": "cyan",
    "known_galaxy": "magenta",
    "known_unclear": "yellow",
}

RA_COL = "ALPHA_J2000"
DEC_COL = "DELTA_J2000"
CATEGORY_COL = "category"

# SExtractor's own half-light radius column, scaled up for visibility -
# used directly, each source's marker is barely a few pixels across.
RADIUS_COL = "FLUX_RADIUS"
RADIUS_SCALE = 6.0


def get_pixel_scale_arcsec(image_path: Path) -> float:
    """
    Function to get the (mean) pixel scale of a UVOT image's first
    data-bearing extension

    :param image_path: Path to the image
    :return: Pixel scale, in arcsec/pixel
    """
    with fits.open(image_path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        wcs = WCS(hdu.header)
    return proj_plane_pixel_scales(wcs).mean() * 3600.0


def write_region_file(
    image_path: Path,
    sources: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """
    Function to write a DS9 region file overlaying each source's
    position, sized by its own FLUX_RADIUS and colour-coded by its
    classification, for visual inspection alongside an image

    :param image_path: Path to the image the sources belong to - used to
        convert each source's pixel radius to sky-coordinate arcsec, and
        (when out_path isn't given) to name the output region file
    :param sources: Table of sources, as produced by find_sources -
        needs ALPHA_J2000/DELTA_J2000 (degrees), FLUX_RADIUS (pixels),
        and category columns
    :param out_path: Path to write the region file to; defaults to
        image_path with a .reg extension
    :return: Path the region file was written to
    """
    if out_path is None:
        out_path = image_path.with_suffix(".reg")

    pixel_scale_arcsec = get_pixel_scale_arcsec(image_path)

    lines = ["# Region file format: DS9 version 4.1", "fk5"]
    for _, source in sources.iterrows():
        category = source[CATEGORY_COL]
        color = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
        radius_arcsec = source[RADIUS_COL] * RADIUS_SCALE * pixel_scale_arcsec
        lines.append(
            f'circle({source[RA_COL]},{source[DEC_COL]},{radius_arcsec:.3f}")'
            f" # color={color} text={{{category}}}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return out_path
