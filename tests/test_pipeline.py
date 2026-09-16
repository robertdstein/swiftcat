"""
Module for testing swiftcat.pipeline
"""

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from helpers import (
    HAS_REAL_DATA,
    HAS_SEXTRACTOR,
    REAL_IMAGE,
    REAL_OBS_DIR,
    REAL_SUBEXPOSURES,
    boom_reachable,
)

from swiftcat.pipeline import detect_and_classify

HAS_BOOM = boom_reachable()

EXPECTED_CSV = Path(__file__).parent / "test_data" / "at2025abcr_expected.csv"


@unittest.skipUnless(HAS_REAL_DATA, f"real test data not found under {REAL_OBS_DIR}")
@unittest.skipUnless(HAS_SEXTRACTOR, "SExtractor ('sex') not found on PATH")
@unittest.skipUnless(HAS_BOOM, "BOOM API not reachable with configured credentials")
class TestDetectAndClassifyRealData(unittest.TestCase):
    """
    Class for testing the full detect_and_classify pipeline end-to-end,
    against a real local UVOT observation and the real BOOM/PS1 service -
    no synthetic data or mocking anywhere in this path.
    """

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.tmp_dir.cleanup)
        self.image_path = Path(self.tmp_dir.name) / REAL_IMAGE.name
        shutil.copy(REAL_IMAGE, self.image_path)

    def test_full_pipeline_produces_classified_sources(self):
        """
        :return: None
        """
        result = detect_and_classify(
            self.image_path, raw_subexposures=REAL_SUBEXPOSURES
        )

        self.assertGreater(len(result), 0)
        self.assertIn("category", result.columns)
        self.assertIn("is_point_source", result.columns)
        self.assertTrue(
            set(result["category"]).issubset(
                {"known_star", "known_galaxy", "known_unclear", "new"}
            )
        )


@unittest.skipUnless(HAS_REAL_DATA, f"real test data not found under {REAL_OBS_DIR}")
@unittest.skipUnless(HAS_SEXTRACTOR, "SExtractor ('sex') not found on PATH")
@unittest.skipUnless(HAS_BOOM, "BOOM API not reachable with configured credentials")
class TestDetectAndClassifyMatchesRecordedReference(unittest.TestCase):
    """
    Class for testing that detect_and_classify's output for this fixed
    real observation matches a recorded reference (tests/test_data/
    at2025abcr_expected.csv) - a regression guard against silent changes
    in detection or classification. NUMBER and category must match
    exactly, since they're discrete and this exact SExtractor/BOOM setup
    reproduced them bit-for-bit across repeated runs when the reference
    was recorded. RA/Dec/MAG_AUTO/MAGERR_AUTO only need to match
    closely, since a future SExtractor or dependency version could shift
    these very slightly without that being an actual regression.
    """

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.tmp_dir.cleanup)
        self.image_path = Path(self.tmp_dir.name) / REAL_IMAGE.name
        shutil.copy(REAL_IMAGE, self.image_path)

    def test_matches_recorded_reference(self):
        """
        :return: None
        """
        expected = pd.read_csv(EXPECTED_CSV)
        result = detect_and_classify(
            self.image_path, raw_subexposures=REAL_SUBEXPOSURES
        ).reset_index(drop=True)

        self.assertEqual(len(result), len(expected))
        pd.testing.assert_series_equal(
            result["NUMBER"], expected["NUMBER"], check_dtype=False
        )
        pd.testing.assert_series_equal(result["category"], expected["category"])

        for col in ("ALPHA_J2000", "DELTA_J2000"):
            pd.testing.assert_series_equal(
                result[col], expected[col], check_exact=False, atol=1e-5
            )
        for col in ("MAG_AUTO", "MAGERR_AUTO"):
            pd.testing.assert_series_equal(
                result[col], expected[col], check_exact=False, atol=1e-3
            )


if __name__ == "__main__":
    unittest.main()
