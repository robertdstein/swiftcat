"""
Module for testing swiftcat.detect
"""

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from astropy.io import fits
from helpers import (
    HAS_REAL_DATA,
    HAS_SEXTRACTOR,
    REAL_IMAGE,
    REAL_OBS_DIR,
    REAL_SUBEXPOSURES,
    tan_wcs,
)

from swiftcat.detect import (
    check_failure,
    find_sources,
    get_detections_in_overlap_region,
    get_sub_exposure_coverage,
    overlap_mask,
)


class TestGetSubExposureCoverage(unittest.TestCase):
    """
    Class for testing get_sub_exposure_coverage
    """

    def test_flags_only_in_bounds_nonzero_pixels(self):
        """
        :return: None
        """
        wcs = tan_wcs(crpix=[5, 5], crval=[10, 20])
        sky = wcs.pixel_to_world(np.array([0, 4, 100]), np.array([0, 4, 100]))
        sub_data = np.ones((10, 10))
        sub_data[0, 0] = 0  # in bounds, but no signal

        covered = get_sub_exposure_coverage(wcs, sky, sub_data)

        np.testing.assert_array_equal(covered, [False, True, False])


class TestGetDetectionsInOverlapRegion(unittest.TestCase):
    """
    Class for testing get_detections_in_overlap_region
    """

    def test_flags_detections_inside_mask(self):
        """
        :return: None
        """
        cat = pd.DataFrame({"X_IMAGE": [1.0, 50.0], "Y_IMAGE": [1.0, 50.0]})
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True

        result = get_detections_in_overlap_region(cat, mask, (10, 10))

        np.testing.assert_array_equal(result, [True, False])


class TestOverlapMask(unittest.TestCase):
    """
    Class for testing overlap_mask against a genuine two-extension FITS
    file - no mocking needed, a small multi-extension FITS with real WCS
    headers is straightforward to construct directly.
    """

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.tmp_dir.cleanup)

    def test_only_doubly_covered_region_is_true(self):
        """
        Two 10x10 sub-exposures, offset by 5 pixels in x, overlap in a
        5-column-wide strip.

        :return: None
        """
        wcs_a = tan_wcs(crpix=[1, 1], crval=[10, 20])
        wcs_b = tan_wcs(crpix=[6, 1], crval=[10, 20])

        raw_path = Path(self.tmp_dir.name) / "raw.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=np.ones((10, 10)), header=wcs_a.to_header()),
                fits.ImageHDU(data=np.ones((10, 10)), header=wcs_b.to_header()),
            ]
        ).writeto(raw_path)

        mask = overlap_mask(wcs_a, (10, 10), raw_path)

        self.assertEqual(mask.sum(), 50)

    def test_raises_if_no_data_extensions(self):
        """
        :return: None
        """
        raw_path = Path(self.tmp_dir.name) / "empty.fits"
        fits.HDUList([fits.PrimaryHDU(data=None)]).writeto(raw_path)

        with self.assertRaises(ValueError):
            overlap_mask(tan_wcs(crpix=[1, 1], crval=[10, 20]), (10, 10), raw_path)


class TestCheckFailure(unittest.TestCase):
    """
    Class for testing check_failure
    """

    def test_log_warning_flags_failure(self):
        """
        :return: None
        """
        data = np.zeros((20, 20))
        cat = pd.DataFrame({"X_IMAGE": [], "Y_IMAGE": []})

        self.assertTrue(check_failure(data, cat, "Deblending overflow detected"))

    def test_two_missing_bright_peaks_flags_failure(self):
        """
        :return: None
        """
        data = np.zeros((20, 20))
        data[5, 5] = 100
        data[15, 15] = 90
        cat = pd.DataFrame({"X_IMAGE": [], "Y_IMAGE": []})

        self.assertTrue(check_failure(data, cat, ""))

    def test_single_missing_bright_peak_is_not_enough(self):
        """
        A single miss could just be a hot pixel, so it shouldn't alone
        flag a failure.

        :return: None
        """
        data = np.zeros((20, 20))
        data[5, 5] = 100
        cat = pd.DataFrame({"X_IMAGE": [], "Y_IMAGE": []})

        self.assertFalse(check_failure(data, cat, ""))

    def test_matched_bright_peaks_do_not_flag_failure(self):
        """
        :return: None
        """
        data = np.zeros((20, 20))
        data[5, 5] = 100
        data[15, 15] = 90
        cat = pd.DataFrame({"X_IMAGE": [6.0, 16.0], "Y_IMAGE": [6.0, 16.0]})

        self.assertFalse(check_failure(data, cat, ""))


@unittest.skipUnless(HAS_REAL_DATA, f"real test data not found under {REAL_OBS_DIR}")
@unittest.skipUnless(HAS_SEXTRACTOR, "SExtractor ('sex') not found on PATH")
class TestFindSourcesRealData(unittest.TestCase):
    """
    Class for testing find_sources end-to-end against a real, local UVOT
    observation - no synthetic data or mocking. SExtractor's own
    background estimation and deblending are not worth faking, so this
    runs the real binary against a real image.
    """

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.tmp_dir.cleanup)
        self.image_path = Path(self.tmp_dir.name) / REAL_IMAGE.name
        shutil.copy(REAL_IMAGE, self.image_path)

    def test_detects_real_sources(self):
        """
        :return: None
        """
        cat = find_sources(self.image_path)

        self.assertGreater(len(cat), 0)
        self.assertIn("is_point_source", cat.columns)
        self.assertFalse(cat.attrs["sextractor_likely_failed"])

    def test_overlap_restriction_drops_edge_detections(self):
        """
        Restricting to the region covered by every sub-exposure should
        never find more sources than detecting on the full image.

        :return: None
        """
        unrestricted = find_sources(self.image_path)
        restricted = find_sources(self.image_path, raw_subexposures=REAL_SUBEXPOSURES)

        self.assertLessEqual(len(restricted), len(unrestricted))
        self.assertGreater(len(restricted), 0)


if __name__ == "__main__":
    unittest.main()
