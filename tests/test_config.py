import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nimslo_cli import configured_range_paths, load_dotenv


class ConfigTests(unittest.TestCase):
    def test_dotenv_loads_values_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "NAP_INPUT_DIR='~/from-file'\n"
                "NAP_GIF_OUTPUT_DIR=~/gifs\n"
                "export NAP_MP4_OUTPUT_DIR=~/videos\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"NAP_INPUT_DIR": "/from-environment"},
                clear=True,
            ):
                load_dotenv(env_file)
                self.assertEqual(os.environ["NAP_INPUT_DIR"], "/from-environment")
                self.assertEqual(os.environ["NAP_GIF_OUTPUT_DIR"], "~/gifs")
                self.assertEqual(os.environ["NAP_MP4_OUTPUT_DIR"], "~/videos")

    def test_configured_paths_expand_environment_variables(self):
        with patch.dict(
            os.environ,
            {
                "NAP_ROOT": "/example",
                "NAP_INPUT_DIR": "$NAP_ROOT/input",
                "NAP_GIF_OUTPUT_DIR": "$NAP_ROOT/gif",
                "NAP_MP4_OUTPUT_DIR": "$NAP_ROOT/mp4",
            },
            clear=True,
        ):
            self.assertEqual(
                configured_range_paths(),
                (
                    Path("/example/input"),
                    Path("/example/gif"),
                    Path("/example/mp4"),
                ),
            )

    def test_configured_paths_report_missing_variables(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "NAP_INPUT_DIR"):
                configured_range_paths()


if __name__ == "__main__":
    unittest.main()
