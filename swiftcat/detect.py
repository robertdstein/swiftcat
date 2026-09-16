"""
Module to detect sources in a UVOT image with SExtractor
"""

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import maximum_filter

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"

# Quality cuts, tuned against real PS1 ground truth on AT2025abcr UVOT data
# (see uvot-sextractor's README for the full investigation each came from).
DETECT_THRESH = 4.0
DETECT_MINAREA = 3
MIN_FLUX_MAX = 5.0  # raw-count floor, on top of DETECT_THRESH sigma
CLASS_STAR_MIN = 0.05
ELLIPTICITY_MAX = 0.3
EDGE_BUFFER_PIX = 10


def run_sextractor(image_path: Path, out_cat: Path) -> tuple[pd.DataFrame, str]:
    """
    Function to run SExtractor on a UVOT image

    :param image_path: Path to the image
    :param out_cat: Path to write the output catalog
    :return: (catalog, combined stdout+stderr log text)
    """
    result = subprocess.run(
        [
            "sex",
            str(image_path),
            "-c",
            str(CONFIG_DIR / "uvot.sex"),
            "-CATALOG_NAME",
            str(out_cat),
            "-PARAMETERS_NAME",
            str(CONFIG_DIR / "default.param"),
            "-FILTER_NAME",
            str(CONFIG_DIR / "gauss_2.5_5x5.conv"),
            "-STARNNW_NAME",
            str(CONFIG_DIR / "default.nnw"),
            "-DETECT_THRESH",
            str(DETECT_THRESH),
            "-ANALYSIS_THRESH",
            str(DETECT_THRESH),
            "-DETECT_MINAREA",
            str(DETECT_MINAREA),
            "-VERBOSE_TYPE",
            "FULL",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    log_text = (result.stdout or "") + (result.stderr or "")
    cat = Table.read(out_cat, format="ascii.sextractor").to_pandas()
    return cat, log_text


def overlap_mask(  # pylint: disable=too-many-locals
    wcs: WCS, shape: tuple[int, int], raw_subexposures: Path
) -> np.ndarray:
    """
    Function to build a mask of the region covered by every sub-exposure in
    a raw, multi-extension UVOT sky image - mosaic edges, covered by only
    some sub-exposures, are where spurious detections cluster

    :param wcs: WCS of the (summed) image being detected on
    :param shape: (ny, nx) of that image
    :param raw_subexposures: Path to the raw multi-extension sky image the
        summed image was created from
    :return: Boolean mask, True where every sub-exposure has coverage
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    sky = wcs.pixel_to_world(xx.ravel(), yy.ravel())

    coverage = np.zeros(ny * nx, dtype=int)
    n_sub = 0
    with fits.open(raw_subexposures) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            n_sub += 1
            sub_wcs = WCS(hdu.header)
            px, py = sub_wcs.world_to_pixel(sky)
            ix, iy = np.round(px).astype(int), np.round(py).astype(int)
            in_bounds = (
                (ix >= 0)
                & (ix < hdu.data.shape[1])
                & (iy >= 0)
                & (iy < hdu.data.shape[0])
            )
            exposed = np.zeros(ny * nx, dtype=bool)
            idx = np.flatnonzero(in_bounds)
            exposed[idx] = hdu.data[iy[idx], ix[idx]] != 0
            coverage += exposed.astype(int)

    if n_sub == 0:
        raise ValueError(f"No image extensions found in {raw_subexposures}")
    return coverage.reshape(ny, nx) == n_sub


def check_failure(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    data: np.ndarray,
    cat: pd.DataFrame,
    log_text: str,
    n_check: int = 8,
    match_radius_px: float = 15.0,
    min_missing: int = 2,
) -> bool:
    """
    Function to flag a likely SExtractor detection failure - a hard-coded
    limit in SExtractor's deblender (NSONMAX=1024, see src/refine.c) can
    silently drop or mis-position bright, real sources at UVOT's very low
    per-pixel count levels. Combines the "Deblending overflow" log warning
    with a direct check that several of the brightest real pixel-data
    peaks have no matching catalog entry

    :param data: Image data SExtractor was run on
    :param cat: SExtractor output catalog
    :param log_text: Combined stdout+stderr from run_sextractor
    :param n_check: How many of the brightest local maxima to check
    :param match_radius_px: A local max within this many pixels of a
        catalog entry counts as found
    :param min_missing: Flag only if at least this many of the n_check
        brightest sources are missing (a single miss could be a hot pixel)
    :return: True if this image's catalog looks unreliable
    """
    if "Deblending overflow" in log_text or "Pixel stack overflow" in log_text:
        return True

    local_max = maximum_filter(data, size=9)
    ys, xs = np.where((data == local_max) & (data > 0))
    order = np.argsort(-data[ys, xs])[:n_check]
    cat_x, cat_y = np.asarray(cat["X_IMAGE"]) - 1, np.asarray(cat["Y_IMAGE"]) - 1

    n_missing = 0
    for i in order:
        if (
            len(cat_x) == 0
            or np.min(np.hypot(cat_x - xs[i], cat_y - ys[i])) > match_radius_px
        ):
            n_missing += 1
    return n_missing >= min_missing


def find_sources(  # pylint: disable=too-many-locals
    image_path: Path, raw_subexposures: Path | None = None
) -> pd.DataFrame:
    """
    Function to detect and shape-classify every source in a UVOT image

    :param image_path: Path to a UVOT image (any filter, any epoch)
    :param raw_subexposures: Raw multi-extension sky image the summed
        image was created from; if given, detections outside the region
        covered by every sub-exposure are dropped
    :return: Source table with `is_point_source` and
        `sextractor_likely_failed` columns added
    """
    with fits.open(image_path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        data = hdu.data.astype(float)
        wcs = WCS(hdu.header)
    ny, nx = data.shape

    cat, log_text = run_sextractor(image_path, image_path.with_suffix(".cat"))
    likely_failed = check_failure(data, cat, log_text)
    if likely_failed:
        logger.warning(
            f"{image_path}: SExtractor detection likely failed, "
            "treat catalog with caution"
        )

    keep = (
        (cat["X_IMAGE"] >= EDGE_BUFFER_PIX)
        & (cat["X_IMAGE"] < nx - EDGE_BUFFER_PIX)
        & (cat["Y_IMAGE"] >= EDGE_BUFFER_PIX)
        & (cat["Y_IMAGE"] < ny - EDGE_BUFFER_PIX)
        & (cat["FLUX_MAX"] >= MIN_FLUX_MAX)
    )
    if raw_subexposures is not None:
        mask = overlap_mask(wcs, (ny, nx), raw_subexposures)
        ix = np.round(cat["X_IMAGE"]).astype(int) - 1
        iy = np.round(cat["Y_IMAGE"]).astype(int) - 1
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        in_region = np.zeros(len(cat), dtype=bool)
        in_region[valid] = mask[iy[valid], ix[valid]]
        keep &= in_region
    cat = cat.loc[keep].reset_index(drop=True)

    cat["is_point_source"] = (cat["CLASS_STAR"] >= CLASS_STAR_MIN) & (
        cat["ELLIPTICITY"] <= ELLIPTICITY_MAX
    )
    cat["sextractor_likely_failed"] = likely_failed
    # Also stored in .attrs (unlike the column, readable even on a 0-row
    # table - a failed epoch typically has few or no surviving detections).
    cat.attrs["sextractor_likely_failed"] = likely_failed
    return cat
