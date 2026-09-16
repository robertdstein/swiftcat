"""
Module for testing swiftcat.paths
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from swiftcat.paths import get_data_dir, get_directory_from_env, get_image_dir


class TestGetDirectoryFromEnv(unittest.TestCase):
    """
    Class for testing get_directory_from_env
    """

    def test_uses_env_var_when_set(self):
        """
        :return: None
        """
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "from_env"
            default = Path(tmp_dir) / "default"

            with patch.dict(os.environ, {"SWIFTCAT_TEST_DIR": str(target)}):
                result = get_directory_from_env("SWIFTCAT_TEST_DIR", default)

            self.assertEqual(result, target)
            self.assertTrue(result.is_dir())

    def test_falls_back_to_default_when_unset(self):
        """
        :return: None
        """
        with TemporaryDirectory() as tmp_dir:
            default = Path(tmp_dir) / "default"

            with patch.dict(os.environ):
                os.environ.pop("SWIFTCAT_TEST_DIR", None)
                result = get_directory_from_env("SWIFTCAT_TEST_DIR", default)

            self.assertEqual(result, default)
            self.assertTrue(result.is_dir())

    def test_falls_back_to_default_when_blank(self):
        """
        A blank environment variable (e.g. an unfilled-in .env entry)
        should fall back to the default, not resolve to an empty path.

        :return: None
        """
        with TemporaryDirectory() as tmp_dir:
            default = Path(tmp_dir) / "default"

            with patch.dict(os.environ, {"SWIFTCAT_TEST_DIR": ""}):
                result = get_directory_from_env("SWIFTCAT_TEST_DIR", default)

            self.assertEqual(result, default)

    def test_creates_directory_if_missing(self):
        """
        :return: None
        """
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "nested" / "does_not_exist_yet"

            with patch.dict(os.environ, {"SWIFTCAT_TEST_DIR": str(target)}):
                result = get_directory_from_env("SWIFTCAT_TEST_DIR", target)

            self.assertTrue(result.is_dir())


class TestGetImageAndDataDir(unittest.TestCase):
    """
    Class for testing that get_image_dir and get_data_dir read from the
    correct, distinct environment variables
    """

    def test_get_image_dir_uses_swift_image_dir(self):
        """
        :return: None
        """
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "images"
            with patch.dict(os.environ, {"SWIFT_IMAGE_DIR": str(target)}):
                self.assertEqual(get_image_dir(), target)

    def test_get_data_dir_uses_swiftcat_data_dir(self):
        """
        :return: None
        """
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "output"
            with patch.dict(os.environ, {"SWIFTCAT_DATA_DIR": str(target)}):
                self.assertEqual(get_data_dir(), target)


if __name__ == "__main__":
    unittest.main()
