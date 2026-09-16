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
from helpers import tan_wcs, write_single_extension_fits
from PIL import Image

from swiftcat.utils.plot import (
    EXCLUDED_REGION_ALPHA,
    add_category_legend,
    draw_source_circles,
    get_object_and_target_id,
    get_source_pixel_positions,
    load_image_data_and_wcs,
    plot_image_with_sources,
    shade_excluded_region,
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


class TestGetObjectAndTargetId(unittest.TestCase):
    """
    Class for testing get_object_and_target_id against a genuine FITS
    file
    """

    def test_reads_object_and_targ_id(self):
        """
        :return: None
        """
        wcs = tan_wcs(crpix=[1, 1], crval=[10, 20])
        header = wcs.to_header()
        header["OBJECT"] = "AT2025abcr"
        header["TARG_ID"] = 3000183

        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(data=np.zeros((10, 10)), header=header),
                ]
            ).writeto(image_path)

            obj, targ_id = get_object_and_target_id(image_path)

        self.assertEqual(obj, "AT2025abcr")
        self.assertEqual(targ_id, "3000183")

    def test_missing_keywords_fall_back_to_unknown(self):
        """
        :return: None
        """
        wcs = tan_wcs(crpix=[1, 1], crval=[10, 20])

        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.fits"
            write_single_extension_fits(image_path, wcs)

            obj, targ_id = get_object_and_target_id(image_path)

        self.assertEqual(obj, "unknown")
        self.assertEqual(targ_id, "unknown")


class TestAddCategoryLegend(unittest.TestCase):
    """
    Class for testing add_category_legend
    """

    def test_legend_labels_include_counts(self):
        """
        :return: None
        """
        _, ax = plt.subplots()
        categories = pd.Series(
            ["known_star", "known_star", "new", "known_galaxy", "known_star"]
        )

        add_category_legend(ax, categories)

        labels = {t.get_text() for t in ax.get_legend().get_texts()}
        self.assertEqual(labels, {"known_star (3)", "new (1)", "known_galaxy (1)"})
        plt.close("all")


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

    def test_draws_one_circle_per_source_with_correct_colors_and_radii(self):
        """
        :return: None
        """
        _, ax = plt.subplots()
        xs = np.array([1.0, 2.0])
        ys = np.array([3.0, 4.0])
        categories = pd.Series(["new", "known_star"])
        radii_pix = np.array([5.0, 8.0])

        draw_source_circles(ax, xs, ys, categories, radii_pix)

        patches = ax.patches
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches[0].get_edgecolor()[:3], (1.0, 0.0, 0.0))  # red
        self.assertEqual(patches[1].get_edgecolor()[:3], (0.0, 1.0, 1.0))  # cyan
        self.assertEqual(patches[0].radius, 5.0)
        self.assertEqual(patches[1].radius, 8.0)
        plt.close("all")


class TestShadeExcludedRegion(unittest.TestCase):
    """
    Class for testing shade_excluded_region
    """

    def test_shades_only_excluded_pixels(self):
        """
        :return: None
        """
        _, ax = plt.subplots()
        mask = np.array([[True, False], [False, True]])

        shade_excluded_region(ax, mask)

        overlay = ax.images[-1].get_array()
        self.assertEqual(overlay[0, 0, 3], 0.0)
        self.assertEqual(overlay[1, 1, 3], 0.0)
        self.assertEqual(overlay[0, 1, 3], EXCLUDED_REGION_ALPHA)
        self.assertEqual(overlay[1, 0, 3], EXCLUDED_REGION_ALPHA)
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
                "FLUX_RADIUS": [2.0, 3.0],
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

    def test_accepts_raw_subexposures_and_still_produces_a_valid_image(self):
        """
        :return: None
        """
        wcs_a = tan_wcs(crpix=[1, 1], crval=[10, 20])
        wcs_b = tan_wcs(crpix=[26, 1], crval=[10, 20])
        raw_path = Path(self.tmp_dir.name) / "raw.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=np.ones((50, 50)), header=wcs_a.to_header()),
                fits.ImageHDU(data=np.ones((50, 50)), header=wcs_b.to_header()),
            ]
        ).writeto(raw_path)

        out_path = plot_image_with_sources(
            self.image_path, self.sources, raw_subexposures=raw_path
        )

        with Image.open(out_path) as img:
            img.verify()


if __name__ == "__main__":
    unittest.main()
