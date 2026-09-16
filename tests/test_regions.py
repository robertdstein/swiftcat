"""
Module for testing swiftcat.utils.regions
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import regions as regions_lib

from swiftcat.utils.regions import CATEGORY_COLORS, DEFAULT_COLOR, write_region_file


class TestWriteRegionFile(unittest.TestCase):
    """
    Class for testing write_region_file, verified by round-tripping the
    written file back through the real `regions` DS9 parser rather than
    just checking the raw text - a genuinely parseable file is a
    stronger guarantee than a string match.
    """

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.tmp_dir.cleanup)
        self.image_path = Path(self.tmp_dir.name) / "image.fits"

    def test_defaults_to_image_stem_with_reg_extension(self):
        """
        :return: None
        """
        sources = pd.DataFrame(
            {"ALPHA_J2000": [10.0], "DELTA_J2000": [20.0], "category": ["new"]}
        )

        out_path = write_region_file(self.image_path, sources)

        self.assertEqual(out_path, self.image_path.with_suffix(".reg"))
        self.assertTrue(out_path.is_file())

    def test_writes_to_explicit_out_path(self):
        """
        :return: None
        """
        sources = pd.DataFrame(
            {"ALPHA_J2000": [10.0], "DELTA_J2000": [20.0], "category": ["new"]}
        )
        explicit_path = Path(self.tmp_dir.name) / "custom.reg"

        out_path = write_region_file(self.image_path, sources, out_path=explicit_path)

        self.assertEqual(out_path, explicit_path)
        self.assertTrue(explicit_path.is_file())

    def test_round_trips_positions_and_colors_through_ds9_parser(self):
        """
        :return: None
        """
        sources = pd.DataFrame(
            {
                "ALPHA_J2000": [10.0, 20.0, 30.0, 40.0],
                "DELTA_J2000": [1.0, 2.0, 3.0, 4.0],
                "category": ["new", "known_star", "known_galaxy", "known_unclear"],
            }
        )

        out_path = write_region_file(self.image_path, sources)
        parsed = regions_lib.Regions.read(out_path, format="ds9")

        self.assertEqual(len(parsed), len(sources))
        for region, (_, source) in zip(parsed, sources.iterrows()):
            self.assertAlmostEqual(region.center.ra.deg, source["ALPHA_J2000"])
            self.assertAlmostEqual(region.center.dec.deg, source["DELTA_J2000"])
            expected_color = CATEGORY_COLORS[source["category"]]
            self.assertEqual(region.visual["facecolor"], expected_color)

    def test_unrecognized_category_falls_back_to_default_color(self):
        """
        :return: None
        """
        sources = pd.DataFrame(
            {
                "ALPHA_J2000": [10.0],
                "DELTA_J2000": [20.0],
                "category": ["something_unexpected"],
            }
        )

        out_path = write_region_file(self.image_path, sources)
        parsed = regions_lib.Regions.read(out_path, format="ds9")

        self.assertEqual(parsed[0].visual["facecolor"], DEFAULT_COLOR)

    def test_empty_sources_writes_header_only(self):
        """
        :return: None
        """
        sources = pd.DataFrame({"ALPHA_J2000": [], "DELTA_J2000": [], "category": []})

        out_path = write_region_file(self.image_path, sources)
        parsed = regions_lib.Regions.read(out_path, format="ds9")

        self.assertEqual(len(parsed), 0)

    def test_custom_radius_is_applied(self):
        """
        :return: None
        """
        sources = pd.DataFrame(
            {"ALPHA_J2000": [10.0], "DELTA_J2000": [20.0], "category": ["new"]}
        )

        out_path = write_region_file(self.image_path, sources, radius_arcsec=12.5)
        parsed = regions_lib.Regions.read(out_path, format="ds9")

        self.assertAlmostEqual(parsed[0].radius.to_value("arcsec"), 12.5)


if __name__ == "__main__":
    unittest.main()
