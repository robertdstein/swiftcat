"""
Module to tag SExtractor detections as known PS1 stars/galaxies or new
(uncatalogued) sources, using PS1's own ps_score (Tachibana & Miller 2018)
star/galaxy classification - not the STRM classifier
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from blastwave.query.boom import BoomClient

PS1_CATALOG = "PS1_DR2"
MATCH_RADIUS = 1.5 * u.arcsec  # pylint: disable=no-member
PS_SCORE_POINT_SOURCE_THRESHOLD = 0.83
# BOOM_QUERY_LIMIT truncation risk beyond this
MAX_QUERY_RADIUS = 800 * u.arcsec  # pylint: disable=no-member


def field_center_and_radius(
    wcs: WCS, shape: tuple[int, int]
) -> tuple[SkyCoord, u.Quantity]:
    """
    Function to get a cone-search center and radius covering an image

    :param wcs: WCS of the image
    :param shape: (ny, nx) of the image
    :return: (field center, cone-search radius)
    """
    ny, nx = shape
    center = wcs.pixel_to_world(nx / 2, ny / 2)
    corner = wcs.pixel_to_world(0, 0)
    radius = min(center.separation(corner) * 1.1, MAX_QUERY_RADIUS)
    return center, radius


def get_field_wcs_and_shape(image_path: Path) -> tuple[WCS, tuple[int, int]]:
    """
    Function to get the WCS and pixel shape of a UVOT image's first
    data-bearing extension

    :param image_path: Path to the image
    :return: (WCS, (ny, nx))
    """
    with fits.open(image_path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        return WCS(hdu.header), hdu.data.shape


def query_ps1_catalog(
    wcs: WCS, shape: tuple[int, int], client: BoomClient
) -> tuple[SkyCoord, np.ndarray]:
    """
    Function to cone-search PS1 around the field covered by an image

    :param wcs: WCS of the image
    :param shape: (ny, nx) of the image
    :param client: BoomClient to query with
    :return: (PS1 source positions, matching ps_score array)
    """
    center, radius = field_center_and_radius(wcs, shape)
    results = client.cone_search(
        ra=center.ra.deg,
        dec=center.dec.deg,
        radius_arcsec=radius.to_value(u.arcsec),  # pylint: disable=no-member
        catalog=PS1_CATALOG,
        limit=200_000,
    )
    ps1_coords = SkyCoord(
        ra=[r["ra"] for r in results], dec=[r["dec"] for r in results], unit="deg"
    )
    ps_score = np.array(
        [
            r.get("ps_score") if r.get("ps_score") is not None else np.nan
            for r in results
        ]
    )
    return ps1_coords, ps_score


def crossmatch_ps1(
    cat: pd.DataFrame, image_path: Path, client: BoomClient | None = None
) -> pd.DataFrame:
    """
    Function to tag each detection as known_star / known_galaxy /
    known_unclear / new, from a fresh PS1 pull for this image's own field

    :param cat: SExtractor catalog (needs ALPHA_J2000, DELTA_J2000)
    :param image_path: Path to the image the catalog was detected on
    :param client: Optional shared BoomClient
    :return: cat with an added `category` column
    """
    wcs, shape = get_field_wcs_and_shape(image_path)
    client = client or BoomClient()
    ps1_coords, ps_score = query_ps1_catalog(wcs, shape, client)

    category = np.full(len(cat), "new", dtype=object)
    if len(cat) > 0 and len(ps1_coords) > 0:
        det_coords = SkyCoord(ra=cat["ALPHA_J2000"], dec=cat["DELTA_J2000"], unit="deg")
        idx, sep, _ = det_coords.match_to_catalog_sky(ps1_coords)
        is_known = sep < MATCH_RADIUS
        matched_score = ps_score[idx]
        has_score = is_known & ~np.isnan(matched_score)
        category[has_score & (matched_score >= PS_SCORE_POINT_SOURCE_THRESHOLD)] = (
            "known_star"
        )
        category[has_score & (matched_score < PS_SCORE_POINT_SOURCE_THRESHOLD)] = (
            "known_galaxy"
        )
        category[is_known & np.isnan(matched_score)] = "known_unclear"
    cat["category"] = category
    return cat
