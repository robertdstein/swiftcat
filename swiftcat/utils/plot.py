"""
Module for rendering a quicklook raster image (e.g. JPEG) of a UVOT
image with classified sources overlaid, colour-coded by category
"""

from pathlib import Path

import matplotlib

# Must run before importing pyplot, so a non-interactive backend is
# picked up before matplotlib locks one in - needed for headless
# pipeline/test runs with no display.
matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.colors import to_rgba

from swiftcat.detect import overlap_mask
from swiftcat.utils.regions import (
    CATEGORY_COL,
    CATEGORY_COLORS,
    DEC_COL,
    DEFAULT_COLOR,
    DEFAULT_RADIUS_ARCSEC,
    RA_COL,
)

# pylint: enable=wrong-import-position

EXCLUDED_REGION_COLOR = "black"
EXCLUDED_REGION_ALPHA = 0.6


def load_image_data_and_wcs(image_path: Path) -> tuple[np.ndarray, WCS]:
    """
    Function to load the pixel data and WCS of a UVOT image's first
    data-bearing extension

    :param image_path: Path to the image
    :return: (pixel data, WCS)
    """
    with fits.open(image_path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        return hdu.data.astype(float), WCS(hdu.header)


def get_source_pixel_positions(
    sources: pd.DataFrame, wcs: WCS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to convert each source's sky position to pixel coordinates
    in the given WCS

    :param sources: Table of sources, needs ALPHA_J2000/DELTA_J2000
    :param wcs: WCS to project into
    :return: (x pixel positions, y pixel positions)
    """
    coords = SkyCoord(
        ra=sources[RA_COL].to_numpy(), dec=sources[DEC_COL].to_numpy(), unit="deg"
    )
    return wcs.world_to_pixel(coords)


def draw_source_circles(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    categories: pd.Series,
    radius_pix: float,
) -> None:
    """
    Function to draw a colour-coded circle at each source's pixel
    position, colour-coded by category (matching write_region_file's
    colour scheme)

    :param ax: Axes to draw on
    :param xs: x pixel positions
    :param ys: y pixel positions
    :param categories: Classification of each source
    :param radius_pix: Circle radius, in pixels
    :return: None
    """
    for x, y, category in zip(xs, ys, categories):
        color = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
        ax.add_patch(
            plt.Circle(
                (x, y), radius_pix, edgecolor=color, facecolor="none", linewidth=1.2
            )
        )


def shade_excluded_region(ax: plt.Axes, mask: np.ndarray) -> None:
    """
    Function to shade, with a semi-transparent overlay, the part of an
    image NOT covered by every sub-exposure - the region find_sources
    drops detections from when given raw_subexposures

    :param ax: Axes to draw on
    :param mask: Boolean coverage mask, True where covered by every
        sub-exposure (as returned by overlap_mask)
    :return: None
    """
    overlay = np.zeros((*mask.shape, 4))
    overlay[..., :3] = to_rgba(EXCLUDED_REGION_COLOR)[:3]
    overlay[..., 3] = np.where(mask, 0.0, EXCLUDED_REGION_ALPHA)
    ax.imshow(overlay, origin="lower")


def plot_image_with_sources(
    image_path: Path,
    sources: pd.DataFrame,
    raw_subexposures: Path | None = None,
    out_path: Path | None = None,
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
) -> Path:
    """
    Function to render a quicklook image of a UVOT observation with each
    classified source circled, colour-coded by category (matching
    write_region_file's colour scheme), and save it as a raster image -
    format is inferred from out_path's extension (e.g. .jpg, .png)

    :param image_path: Path to the image
    :param sources: Table of sources, as produced by crossmatch_ps1 -
        needs ALPHA_J2000/DELTA_J2000 (degrees) and category columns
    :param raw_subexposures: Raw multi-extension sky image the summed
        image was created from; if given, the region outside every
        sub-exposure's coverage (as excluded by find_sources) is shaded
    :param out_path: Path to save the plot to; defaults to image_path
        with a .jpg extension
    :param radius_arcsec: Circle radius to draw around each source
    :return: Path the plot was saved to
    """
    if out_path is None:
        out_path = image_path.with_suffix(".jpg")

    data, wcs = load_image_data_and_wcs(image_path)
    pixel_scale_deg = proj_plane_pixel_scales(wcs).mean()
    radius_pix = (radius_arcsec / 3600.0) / pixel_scale_deg
    xs, ys = get_source_pixel_positions(sources, wcs)

    fig, ax = plt.subplots()
    norm = ImageNormalize(data, interval=ZScaleInterval())
    ax.imshow(data, origin="lower", cmap="gray", norm=norm)
    if raw_subexposures is not None:
        shade_excluded_region(ax, overlap_mask(wcs, data.shape, raw_subexposures))
    draw_source_circles(ax, xs, ys, sources[CATEGORY_COL], radius_pix)
    ax.set_axis_off()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
