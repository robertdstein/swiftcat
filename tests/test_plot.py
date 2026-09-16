"""
Module for testing swiftcat.utils.plot
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from helpers import tan_wcs
from PIL import Image

from swiftcat.utils.plot import (
    draw_source_circles,
    get_source_pixel_positions,
    load_image_data_and_wcs,
    plot_image_with_sources,
)


class TestLoadImageDataAndWcs(unittest.TestCase):
    """
    Class for testing load_image_data_and_wcs against a genuine FITS file
    """

    def test_reads_data_and_wcs_from_first_data_extension(self):
        """
        :return: None
        """
        wcs_in = tan_wcs(crpix=[1, 1], crval=[10, 20])
        data_in = np.arange(20).reshape(4, 5).astype(float)

        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(data=data_in, header=wcs_in.to_header()),
                ]
            ).writeto(image_path)

            data_out, wcs_out = load_image_data_and_wcs(image_path)

        np.testing.assert_array_equal(data_out, data_in)
        self.assertEqual(wcs_out.wcs.ctype[0], "RA---TAN")


class TestGetSourcePixelPositions(unittest.TestCase):
    """
    Class for testing get_source_pixel_positions
    """

    def test_matches_direct_world_to_pixel(self):
        """
        :return: None
        """
        wcs = tan_wcs(crpix=[50, 50], crval=[10, 20])
        sources = pd.DataFrame(
            {"ALPHA_J2000": [10.0, 10.01], "DELTA_J2000": [20.0, 20.0]}
        )

        xs, ys = get_source_pixel_positions(sources, wcs)

        self.assertAlmostEqual(xs[0], 49.0, places=3)
        self.assertAlmostEqual(ys[0], 49.0, places=3)
        self.assertEqual(len(xs), len(sources))


class TestDrawSourceCircles(unittest.TestCase):
    """
    Class for testing draw_source_circles
    """

    def test_draws_one_circle_per_source_with_correct_colors(self):
        """
        :return: None
        """
        _, ax = plt.subplots()
        xs = np.array([1.0, 2.0])
        ys = np.array([3.0, 4.0])
        categories = pd.Series(["new", "known_star"])

        draw_source_circles(ax, xs, ys, categories, radius_pix=5.0)

        patches = ax.patches
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches[0].get_edgecolor()[:3], (1.0, 0.0, 0.0))  # red
        self.assertEqual(
            patches[1].get_edgecolor()[:3], (0.0, 0.50196078431372548, 0.0)
        )
        plt.close("all")


class TestPlotImageWithSources(unittest.TestCase):
    """
    Class for testing plot_image_with_sources end-to-end against a
    genuine (if synthetic) FITS image - real image data and real WCS,
    no mocking.
    """

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.tmp_dir.cleanup)
        wcs = tan_wcs(crpix=[25, 25], crval=[10, 20])
        data = np.random.default_rng(0).normal(size=(50, 50))
        self.image_path = Path(self.tmp_dir.name) / "image.fits"
        fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=wcs.to_header())]
        ).writeto(self.image_path)
        self.sources = pd.DataFrame(
            {
                "ALPHA_J2000": [10.0, 10.001],
                "DELTA_J2000": [20.0, 20.001],
                "category": ["new", "known_star"],
            }
        )

    def test_defaults_to_image_stem_with_jpg_extension(self):
        """
        :return: None
        """
        out_path = plot_image_with_sources(self.image_path, self.sources)

        self.assertEqual(out_path, self.image_path.with_suffix(".jpg"))
        self.assertTrue(out_path.is_file())

    def test_writes_a_real_readable_image_of_the_expected_size(self):
        """
        :return: None
        """
        out_path = plot_image_with_sources(self.image_path, self.sources)

        with Image.open(out_path) as img:
            img.verify()
            self.assertGreater(img.size[0], 0)
            self.assertGreater(img.size[1], 0)

    def test_writes_to_explicit_out_path_and_format(self):
        """
        :return: None
        """
        explicit_path = Path(self.tmp_dir.name) / "custom.png"

        out_path = plot_image_with_sources(
            self.image_path, self.sources, out_path=explicit_path
        )

        self.assertEqual(out_path, explicit_path)
        with Image.open(explicit_path) as img:
            self.assertEqual(img.format, "PNG")


if __name__ == "__main__":
    unittest.main()
