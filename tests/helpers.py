"""
Shared test fixtures for swiftcat's test suite (not itself a test module -
unittest's discovery only picks up test*.py files).
"""

import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from blastwave.errors import BOOMCredentialsError
from blastwave.query.boom import BoomClient
from dotenv import load_dotenv

from swiftcat.paths import get_image_dir

# A real, already-reduced UVOT observation available locally (from
# uvotredux's own output) - used for genuine end-to-end runs rather than
# a synthetic/mocked catalog. If a machine doesn't have it (e.g. a future
# CI runner), tests relying on it skip rather than fail.
REAL_OBS_DIR = get_image_dir() / "AT2025mwm" / "00019851001" / "uvot" / "image"
REAL_IMAGE = REAL_OBS_DIR / "UW1.fits"
REAL_SUBEXPOSURES = REAL_OBS_DIR / "sw00019851001uw1_sk.img"

HAS_REAL_DATA = REAL_IMAGE.is_file() and REAL_SUBEXPOSURES.is_file()
HAS_SEXTRACTOR = shutil.which("sex") is not None


def tan_wcs(crpix: list[float], crval: list[float], cdelt: float = 0.001) -> WCS:
    """
    Function to build a simple real tangent-plane WCS for tests

    :param crpix: Reference pixel (x, y)
    :param crval: Reference sky position (ra, dec) in degrees
    :param cdelt: Pixel scale in degrees/pixel
    :return: WCS
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = crpix
    wcs.wcs.cdelt = [-cdelt, cdelt]
    wcs.wcs.crval = crval
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def write_single_extension_fits(
    image_path: Path, wcs: WCS, data: np.ndarray | None = None
) -> None:
    """
    Function to write a minimal single-extension FITS file for tests

    :param image_path: Path to write the FITS file to
    :param wcs: WCS to embed in the extension header
    :param data: Pixel data; defaults to a 10x10 zero array
    :return: None
    """
    if data is None:
        data = np.zeros((10, 10))
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=wcs.to_header())]
    ).writeto(image_path)


def boom_reachable(ra: float = 250.0767333333, dec: float = 26.9258638889) -> bool:
    """
    Function to check whether a real BOOM query can be made right now,
    so tests can skip cleanly instead of failing when it can't.

    :param ra: RA to query, degrees
    :param dec: Dec to query, degrees
    :return: True if a real cone_search succeeds
    """
    # BoomClient reads its credentials via its own load_dotenv() call,
    # which resolves relative to wherever blastwave itself is installed
    # (e.g. site-packages) rather than this project, so it can't find
    # this project's .env on its own - load it here first instead.
    load_dotenv()
    try:
        BoomClient().cone_search(
            ra=ra, dec=dec, radius_arcsec=5, catalog="PS1_DR2", limit=1
        )
        return True
    except (BOOMCredentialsError, OSError):
        return False
