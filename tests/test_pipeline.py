"""
Module for testing swiftcat.pipeline
"""

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()
