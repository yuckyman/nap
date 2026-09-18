import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nimslo_cli import configured_range_paths, load_dotenv, main


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

    def test_numeric_interactive_mode_uses_both_configured_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            gif_root = root / "gif"
            mp4_root = root / "mp4"
            batch = input_root / "132"
            batch.mkdir(parents=True)
            environment = {
                "NAP_INPUT_DIR": str(input_root),
                "NAP_GIF_OUTPUT_DIR": str(gif_root),
                "NAP_MP4_OUTPUT_DIR": str(mp4_root),
            }

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("sys.argv", ["nap", "--interactive", "132"]),
                patch(
                    "nimslo_cli.process_single_batch",
                    return_value={"success": True},
                ) as process,
            ):
                main()

            process.assert_called_once_with(
                batch,
                gif_root / "132",
                output_format="both",
                quality="best",
                show_masks=False,
                interactive=True,
                preview=False,
                mp4_loops=None,
                mp4_output_path=mp4_root / "132",
            )
            self.assertTrue(gif_root.is_dir())
            self.assertTrue(mp4_root.is_dir())


if __name__ == "__main__":
    unittest.main()
