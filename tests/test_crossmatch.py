"""
Module for testing swiftcat.crossmatch
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from blastwave.query.boom import BoomClient
from helpers import boom_reachable, tan_wcs

from swiftcat.crossmatch import (
    MAX_QUERY_RADIUS,
    crossmatch_ps1,
    field_center_and_radius,
    get_field_wcs_and_shape,
    query_ps1_catalog,
)

# A real field with well-established PS1 stars, used for genuine
# end-to-end BOOM/PS1 queries below rather than a mocked client - PS1's
# catalog for a fixed field is effectively static.
TEST_RA, TEST_DEC = 250.0767333333, 26.9258638889

HAS_BOOM = boom_reachable(TEST_RA, TEST_DEC)


class TestFieldCenterAndRadius(unittest.TestCase):
    """
    Class for testing field_center_and_radius
    """

    def test_center_and_radius_cover_the_corner(self):
        """
        :return: None
        """
        wcs = tan_wcs(crpix=[50, 50], crval=[10, 20])
        shape = (100, 100)

        center, radius = field_center_and_radius(wcs, shape)
        corner = wcs.pixel_to_world(0, 0)

        self.assertIsInstance(center, SkyCoord)
        self.assertGreaterEqual(radius, center.separation(corner))

    def test_radius_is_capped(self):
        """
        :return: None
        """
        wcs = tan_wcs(crpix=[5000, 5000], crval=[10, 20], cdelt=0.01)
        shape = (10000, 10000)

        _, radius = field_center_and_radius(wcs, shape)

        self.assertEqual(radius, MAX_QUERY_RADIUS)


class TestGetFieldWcsAndShape(unittest.TestCase):
    """
    Class for testing get_field_wcs_and_shape against a genuine FITS file
    """

    def test_reads_wcs_and_shape_from_first_data_extension(self):
        """
        :return: None
        """
        wcs_in = tan_wcs(crpix=[1, 1], crval=[10, 20])
        shape = (20, 30)

        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(data=np.zeros(shape), header=wcs_in.to_header()),
                ]
            ).writeto(image_path)

            wcs_out, shape_out = get_field_wcs_and_shape(image_path)

        self.assertEqual(shape_out, shape)
        self.assertEqual(wcs_out.wcs.ctype[0], "RA---TAN")


@unittest.skipUnless(HAS_BOOM, "BOOM API not reachable with configured credentials")
class TestQueryPs1CatalogReal(unittest.TestCase):
    """
    Class for testing query_ps1_catalog against the real, live BOOM/PS1
    service - a fixed real field's PS1 catalog is effectively static, so
    this is more reliable than mocking the client's response shape.
    """

    def test_returns_real_ps1_sources(self):
        """
        :return: None
        """
        wcs = tan_wcs(crpix=[50, 50], crval=[TEST_RA, TEST_DEC])
        shape = (100, 100)

        ps1_coords, ps_score = query_ps1_catalog(wcs, shape, BoomClient())

        self.assertGreater(len(ps1_coords), 0)
        self.assertEqual(len(ps1_coords), len(ps_score))


@unittest.skipUnless(HAS_BOOM, "BOOM API not reachable with configured credentials")
class TestCrossmatchPs1Real(unittest.TestCase):
    """
    Class for testing crossmatch_ps1 end-to-end against the real BOOM/PS1
    service, using a genuine FITS image with a known-star field.
    """

    def test_classifies_known_and_new_sources(self):
        """
        A detection right on a known bright star should be classified as
        known (star or galaxy); a detection far from any catalogued
        source should be "new".

        :return: None
        """
        wcs = tan_wcs(crpix=[50, 50], crval=[TEST_RA, TEST_DEC])
        shape = (100, 100)

        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(data=np.zeros(shape), header=wcs.to_header()),
                ]
            ).writeto(image_path)

            cat = pd.DataFrame(
                {
                    "ALPHA_J2000": [TEST_RA, TEST_RA + 1.0],
                    "DELTA_J2000": [TEST_DEC, TEST_DEC],
                }
            )
            result = crossmatch_ps1(cat, image_path, client=BoomClient())

        self.assertIn(result["category"].iloc[0], ["known_star", "known_galaxy"])
        self.assertEqual(result["category"].iloc[1], "new")

    def test_empty_catalog_returns_empty_with_column(self):
        """
        :return: None
        """
        wcs = tan_wcs(crpix=[50, 50], crval=[TEST_RA, TEST_DEC])
        shape = (100, 100)

        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(data=np.zeros(shape), header=wcs.to_header()),
                ]
            ).writeto(image_path)

            cat = pd.DataFrame({"ALPHA_J2000": [], "DELTA_J2000": []})
            result = crossmatch_ps1(cat, image_path, client=BoomClient())

        self.assertEqual(len(result), 0)
        self.assertIn("category", result.columns)


if __name__ == "__main__":
    unittest.main()
