"""CPU tests for split-isolated video teachers and safe resumable publishing."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from tools import run_video_gbuffers as video


class VideoGBufferTests(unittest.TestCase):
    def test_default_segments_isolate_train_and_test_before_padding(self):
        ranges = video.parse_holdout_ranges("20:30,60:70", 100)
        segments = video.split_segments(100, 100, ranges)
        self.assertEqual(
            [(segment["role"], segment["frame_ids"][0], segment["frame_ids"][-1]) for segment in segments],
            [("train", 100, 119), ("test", 120, 129), ("train", 130, 159),
             ("test", 160, 169), ("train", 170, 199)],
        )
        heldout = set(range(120, 130)) | set(range(160, 170))
        for segment in segments:
            padded = video.padded_frame_ids(segment["frame_ids"])
            self.assertEqual(len(padded), 57)
            self.assertEqual(padded[:len(segment["frame_ids"])], segment["frame_ids"])
            self.assertEqual(set(padded), set(segment["frame_ids"]))
            self.assertEqual(padded[-1], segment["frame_ids"][-1])
            self.assertEqual(bool(set(padded) & heldout), segment["role"] == "test")
        self.assertEqual(video.output_filename(129, "basecolor"), "0000.0129.basecolor.png")
        self.assertEqual(video.padded_frame_ids(list(range(60))), list(range(60)))

    def test_ranges_validate_and_adjacent_holdouts_merge(self):
        for value in ("-1:5", "0:101", "2:2", "5:3", "a:b", "0:5:6", "0:5,4:8"):
            with self.assertRaises(ValueError):
                video.parse_holdout_ranges(value, 100)
        segments = video.split_segments(0, 10, video.parse_holdout_ranges("2:4,4:6", 10))
        self.assertEqual([segment["frame_ids"] for segment in segments],
                         [[0, 1], [2, 3, 4, 5], [6, 7, 8, 9]])
        with self.assertRaises(ValueError):
            video.split_segments(-1, 10, [])
        with self.assertRaises(ValueError):
            video.padded_frame_ids([])
        self.assertEqual(video.parse_holdout_ranges("", 5), [])

    @contextlib.contextmanager
    def fixture(self):
        """Tiny real images, fake executable paths, and deterministic stub inference."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            source = repository / "data/waymo/processed/training/788/images/raw"
            source.mkdir(parents=True)
            (repository / "tools").mkdir()
            (repository / "tools/prepare_tiled_dr_inputs.py").write_text("# fixed test crop code\n")
            for frame in range(4):
                Image.new("RGB", (12, 8), (20 + frame * 20, 30, 40)).save(source / f"{frame:03d}_0.jpg")
            cosmos = root / "cosmos"
            inverse = cosmos / "cosmos_predict1/diffusion/inference/inference_inverse_renderer_long.py"
            inverse.parent.mkdir(parents=True)
            inverse.write_text("# fake Cosmos entrypoint\n")
            checkpoint = cosmos / "checkpoints/Diffusion_Renderer_Inverse_Cosmos_7B/model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fake-model")
            conda = root / "conda"
            conda.write_text("# fake conda\n")
            environment = root / "cosmos-env"
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").write_text("# fake python\n")
            args = video.make_parser().parse_args([
                "--frames", "4", "--holdout-ranges", "1:2", "--work-dir", str(root / "work"),
                "--output-root", str(root / "output"), "--cosmos-root", str(cosmos),
                "--conda", str(conda), "--cosmos-env", str(environment),
            ])
            def restore(top, bottom, pass_name):
                with Image.open(top) as image:
                    return np.array(image.resize((12, 8)), dtype=np.uint8)

            def backend(command, **kwargs):
                self.assertTrue(kwargs["check"])
                dataset = Path(command[command.index("--dataset_path") + 1])
                results = Path(command[command.index("--video_save_folder") + 1])
                for folder in sorted(dataset.iterdir()):
                    target = results / "gbuffer_frames" / folder.name
                    target.mkdir(parents=True)
                    for local, path in enumerate(sorted(folder.glob("*.png"))):
                        with Image.open(path) as image:
                            for pass_name in video.PASSES:
                                image.save(target / f"full.{local:05d}.{pass_name}.png")
                return subprocess.CompletedProcess(command, 0)

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(video, "ROOT", repository))
                stack.enter_context(mock.patch.object(video, "WINDOW", 3))
                stack.enter_context(mock.patch.object(video, "STRIDE", 1))
                stack.enter_context(mock.patch.object(video, "SOURCE_SIZE", (12, 8)))
                stack.enter_context(mock.patch.object(video, "DR_SIZE", (8, 6)))
                stack.enter_context(mock.patch.object(video, "CROPS", {
                    "top": (0, 0, 12, 7), "bottom": (0, 1, 12, 8),
                }))
                stack.enter_context(mock.patch.object(video, "restore_pair", side_effect=restore))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                yield args, root, backend

    def test_preview_validates_paths_and_writes_nothing(self):
        with self.fixture() as (args, root, backend):
            with mock.patch.object(video.subprocess, "run") as run:
                self.assertEqual(video.run(args), 0)
                run.assert_not_called()
            self.assertFalse(args.work_dir.exists())
            self.assertFalse(args.output_root.exists())
            (args.cosmos_env / "bin/python").unlink()
            with self.assertRaises(FileNotFoundError):
                video.run(args)

    def test_external_command_matches_installed_long_video_interface(self):
        with self.fixture() as (args, root, backend):
            command = video.inverse_command(args, root / "inputs", root / "results")
            for flag, expected in (
                ("--pin_mode", "hybrid"), ("--num_steps", "15"), ("--seed", "1000"),
                ("--group_mode", "folder"), ("--save_image_format", "png"),
                ("--save_video", "false"), ("--normalize_normal", "false"),
            ):
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertIn("inference_inverse_renderer_long.py", " ".join(command))
            self.assertNotIn("inference_inverse_renderer.py", " ".join(command))

    def test_end_to_end_provenance_mapping_and_completed_resume(self):
        with self.fixture() as (args, root, backend):
            args.execute = True
            with mock.patch.object(video.subprocess, "run", side_effect=backend) as run:
                video.run(args)
                self.assertEqual(run.call_count, 1)
            manifest_path = args.work_dir / "generation.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["stage"], "completed")
            self.assertEqual(len(manifest["final_sha256"]), 16)
            self.assertEqual(len(manifest["identity"]["clips"]), 6)
            self.assertEqual([item["role"] for item in manifest["identity"]["frame_provenance"]],
                             ["train", "test", "train", "train"])
            output = args.output_root / "gbuffer_frames/raw"
            for frame in range(4):
                target = output / video.output_filename(frame, "basecolor")
                self.assertTrue(video.valid_png(target, (12, 8)))
                with Image.open(target) as image:
                    self.assertAlmostEqual(np.array(image)[4, 6, 0], 20 + frame * 20, delta=3)
            args.resume = True
            with mock.patch.object(video.subprocess, "run") as run:
                video.run(args)
                run.assert_not_called()
            # Bytes, not only PNG dimensions, guard the completed result.
            Image.new("RGB", (12, 8), (99, 99, 99)).save(output / video.output_filename(0, "basecolor"))
            with self.assertRaisesRegex(ValueError, "changed"):
                video.run(args)

    def test_resume_rejects_changed_source_split_or_recipe(self):
        with self.fixture() as (args, root, backend):
            plan = video.build_plan(args)
            args.work_dir.mkdir()
            video.atomic_json(args.work_dir / "generation.json", plan)
            args.resume = True
            args.holdout_ranges = "2:3"
            with self.assertRaisesRegex(ValueError, "differs"):
                video.run(args)
            args.holdout_ranges = "1:2"
            source = Path(plan["identity"]["source"]) / "000_0.jpg"
            Image.new("RGB", (12, 8), (99, 0, 0)).save(source)
            with self.assertRaisesRegex(ValueError, "differs"):
                video.run(args)

    def test_failed_attempt_resumes_only_complete_clips_and_preserves_partial(self):
        with self.fixture() as (args, root, backend):
            args.execute = True
            def partial(command, **kwargs):
                backend(command, **kwargs)
                results = Path(command[command.index("--video_save_folder") + 1])
                folders = sorted((results / "gbuffer_frames").iterdir())
                for folder in folders[1:]:
                    # Retain partial output for every other clip.
                    next(folder.glob("*.png")).unlink()
                raise subprocess.CalledProcessError(1, command)
            with mock.patch.object(video.subprocess, "run", side_effect=partial):
                with self.assertRaises(subprocess.CalledProcessError):
                    video.run(args)
            first = json.loads((args.work_dir / "generation.json").read_text())
            self.assertEqual(first["stage"], "failed")
            first_path = Path(first["attempts"][0]["results"])
            self.assertEqual(len(video.discover_completed(first)), 1)
            args.resume = True
            with mock.patch.object(video.subprocess, "run", side_effect=backend) as run:
                video.run(args)
                command = run.call_args.args[0]
                dataset = Path(command[command.index("--dataset_path") + 1])
                self.assertEqual(len(list(dataset.iterdir())), 5)
            final = json.loads((args.work_dir / "generation.json").read_text())
            self.assertEqual(len(final["attempts"]), 2)
            self.assertTrue(first_path.exists())
            self.assertEqual(final["stage"], "completed")

    def test_existing_output_is_never_overwritten(self):
        with self.fixture() as (args, root, backend):
            output = args.output_root / "gbuffer_frames/raw"
            output.mkdir(parents=True)
            keep = output / "keep.txt"
            keep.write_text("keep")
            args.execute = True
            with self.assertRaises(FileExistsError):
                video.run(args)
            self.assertEqual(keep.read_text(), "keep")
            self.assertFalse(args.work_dir.exists())


if __name__ == "__main__":
    unittest.main()
