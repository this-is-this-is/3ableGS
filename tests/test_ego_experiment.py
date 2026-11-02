"""CPU checks for the experiment's data separation and reported quantities."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tests.test_ego_temporal import packet
from tools.ego_motion_experiment import (
    FrameCache, MODEL_KEYS, diagnose, digest, evaluate, history_frames,
    make_head, metric_summary, model_inputs, motion_statistics, read_motion,
    region_masks, verify_cache_source,
)


def cached_packet(frame):
    result = packet(h=8, w=10, frame_id=frame)
    result.update(target_rgb=torch.full((8, 10, 3), .4),
                  evaluation_static=torch.ones(8, 10, dtype=torch.bool),
                  image_infos={"img_idx": torch.tensor([frame])},
                  background_linear=torch.zeros(8, 10, 3), forward_ms=2.)
    return result


def write_cache(directory, frames=range(8)):
    directory = Path(directory)
    (directory / "frames").mkdir(parents=True)
    source = directory / "source"
    source.mkdir()
    (source / "checkpoint_final.pth").write_bytes(b"unit-test-checkpoint")
    (source / "config.yaml").write_text("unit_test: true\n")
    identity = {"checkpoint": str(source / "checkpoint_final.pth"),
                "checkpoint_sha256": digest(source / "checkpoint_final.pth"),
                "config_sha256": digest(source / "config.yaml"),
                "downscale": 2., "antialiased": False, "split": "all"}
    hashes = {}
    for frame in frames:
        path = directory / "frames" / f"{frame:06d}.pt"
        torch.save(cached_packet(frame), path)
        hashes[str(frame)] = digest(path)
    metadata = {"status": "completed", "frames": list(frames), "hashes": hashes,
                "identity": identity, "scene": 788, "train_frames": [0, 1, 4, 5],
                "val_frames": [2, 3], "test_frames": [6, 7]}
    (directory / "manifest.json").write_text(json.dumps(metadata))
    return metadata


class ExperimentMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_identity_camera_and_analytic_translation(self):
        current, previous = cached_packet(1), cached_packet(0)
        stationary = motion_statistics(current, previous)
        self.assertAlmostEqual(stationary["displacement_p90"], 0.)
        self.assertEqual(stationary["frustum_overlap"], 1.)
        self.assertEqual(stationary["rotation_deg"], 0.)
        current["camera_to_world"][0, 3] = .5
        # fx=4 pixels, Z=2 metres: .5-m translation gives one pixel.
        moving = motion_statistics(current, previous)
        self.assertAlmostEqual(moving["displacement_p50"], 1.)
        self.assertAlmostEqual(moving["displacement_p90"], 1.)
        self.assertAlmostEqual(moving["translation_m"], .5)
        self.assertAlmostEqual(moving["frustum_overlap"], .9, places=6)
        current["target_rgb"].fill_(999)
        self.assertEqual(moving, motion_statistics(current, previous))

    def test_motion_empty_static_surface_has_no_displacement_observation(self):
        current, previous = cached_packet(1), cached_packet(0)
        current["static_mask"].zero_()
        result = motion_statistics(current, previous)
        self.assertIsNone(result["displacement_p50"])
        self.assertIsNone(result["displacement_p90"])
        self.assertEqual(result["frustum_overlap"], 0.)

    def test_history_never_jumps_a_held_out_interval(self):
        train = [0, 1, 2, 5, 6, 7, 8, 9]
        self.assertEqual(history_frames(5, train), [])
        self.assertEqual(history_frames(6, train), [5])
        self.assertEqual(history_frames(9, train), [8, 7, 5])
        self.assertEqual(history_frames(4, train), [])
        # Independent held-out blocks reset even though a long gap reaches
        # an earlier allowed held-out frame.
        self.assertEqual(history_frames(6, [2, 3, 6, 7]), [])

    def test_features_exclude_all_reference_and_shading_only_keys(self):
        value = cached_packet(1)
        value.update(pixels=torch.ones(8, 10, 3), sky_masks=torch.ones(8, 10),
                     gbuffer_albedo=torch.ones(8, 10, 3), lidar_depth_map=torch.ones(8, 10))
        features = model_inputs(value)
        self.assertEqual(set(features), MODEL_KEYS)
        for key in ("target_rgb", "evaluation_static", "pixels", "sky_masks",
                    "gbuffer_albedo", "lidar_depth_map", "image_infos", "background_linear"):
            self.assertNotIn(key, features)

    def test_current_control_preserves_full_model_capacity_and_ignores_history(self):
        current = make_head("current", width=8)
        full = make_head("full", width=8)
        self.assertEqual(current.guides, "full")
        self.assertEqual(sum(value.numel() for value in current.parameters()),
                         sum(value.numel() for value in full.parameters()))
        self.assertEqual({name: tuple(value.shape) for name, value in current.state_dict().items()},
                         {name: tuple(value.shape) for name, value in full.state_dict().items()})
        frame, previous = cached_packet(1), cached_packet(0)
        previous["albedo"].fill_(.9)
        with torch.no_grad():
            without = current(model_inputs(frame), [])
            supplied = current(model_inputs(frame), [model_inputs(previous)])
        torch.testing.assert_close(without["albedo"], supplied["albedo"], rtol=0, atol=0)
        self.assertEqual(supplied["history_weight"].sum(), 0)

    def test_quality_uses_pixel_weighted_error_and_pair_weighted_temporal_error(self):
        records = [
            {"prediction_full_squares": 99, "prediction_full_squared_error": 1.,
             "prediction_full_values": 100, "prediction_temporal_residual_l1_sum": 2.,
             "prediction_temporal_residual_l1_count": 10, "temporal_has_pair": True,
             "prediction_edge_strength_ratio_numerator": 2.,
             "prediction_edge_strength_ratio_denominator": 1., "lpips": .1},
            {"prediction_full_squared_error": 9., "prediction_full_values": 300,
             "prediction_temporal_residual_l1_sum": 3.,
             "prediction_temporal_residual_l1_count": 30, "temporal_has_pair": True,
             "prediction_edge_strength_ratio_numerator": 1.,
             "prediction_edge_strength_ratio_denominator": 2., "lpips": .3},
        ]
        summary = metric_summary(records)
        self.assertAlmostEqual(summary["prediction_full_psnr"], -10 * math.log10(10 / 400))
        self.assertAlmostEqual(summary["prediction_temporal_residual_l1"], 5 / 40)
        self.assertEqual(summary["prediction_edge_strength_ratio"], 1.)
        self.assertEqual(summary["temporal_pairs"], 2)
        self.assertAlmostEqual(summary["lpips"], .2)
        self.assertEqual(metric_summary([])["frames"], 0)

    def test_region_masks_use_reference_static_only_for_metrics(self):
        value = cached_packet(1)
        value["evaluation_static"][2, 2] = False
        value["depth"][:, 5:] *= 2
        masks = region_masks(value)
        self.assertTrue(masks["depth_boundary"][:, 4:6].all())
        self.assertFalse(masks["depth_boundary"][:, :4].any())
        for mask in masks.values():
            self.assertFalse(mask[2, 2])
        self.assertFalse(masks["geometry_supported"].any())
        self.assertTrue(masks["unsupported_proxy"][value["evaluation_static"]].all())


class CacheAndEvaluationTests(unittest.TestCase):
    def test_changed_source_checkpoint_or_config_cannot_reshade_cached_materials(self):
        for name in ("checkpoint_final.pth", "config.yaml"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                metadata = write_cache(Path(temporary))
                verify_cache_source(metadata["identity"])
                path = Path(temporary) / "source" / name
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "changed since render-cache"):
                    verify_cache_source(metadata["identity"])

    def test_motion_stratification_cannot_come_from_a_variant_depth_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.json"
            base_identity = {"checkpoint_sha256": "base"}
            path.write_text(json.dumps({"cache_identity": base_identity,
                                        "frames": [{"frame_id": 1, "motion_bin": "high"}]}))
            self.assertEqual(read_motion(path, base_identity), {1: "high"})
            with self.assertRaisesRegex(ValueError, "frozen base cache"):
                read_motion(path, {"checkpoint_sha256": "modified"})

    def test_corrupt_frame_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_cache(directory)
            cache = FrameCache(directory, capacity=1)
            self.assertEqual(cache.get(0)["frame_id"], 0)
            path = directory / "frames/000001.pt"
            path.write_bytes(b"corrupted serialized frame")
            with mock.patch("torch.load", side_effect=AssertionError("must verify before loading")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    cache.get(1)

    def test_diagnosis_resets_region_support_at_each_split_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_cache(directory / "cache")
            args = SimpleNamespace(cache=directory / "cache", output_dir=directory / "diagnostics")
            diagnose(args)
            def support(frame):
                return torch.load(args.output_dir / "regions" / f"{frame:06d}.pt",
                                  weights_only=True)["geometry_supported"]
            for frame in (0, 2, 4, 6):
                self.assertFalse(support(frame).any(), f"frame {frame} crossed a split")
            for frame in (1, 3, 5, 7):
                self.assertTrue(support(frame)[2:-2, 2:-2].all())
            data = json.loads((args.output_dir / "motion.json").read_text())
            self.assertEqual(data["thresholds"], [0., 0.])

    def test_evaluation_resets_lpips_and_flow_and_uses_unchanged_references(self):
        class LPIPS:
            def __init__(self):
                self.reset_count = 0
                self.inputs = []

            def reset(self):
                self.reset_count += 1

            def __call__(self, predicted, reference):
                self.assert_shapes = (tuple(predicted.shape), tuple(reference.shape))
                self.inputs.append((predicted.clone(), reference.clone()))
                return (predicted - reference).abs().mean()

        evaluators = []
        class Flow:
            def __init__(self, *args, **kwargs):
                self.frames = []
                evaluators.append(self)

            def evaluate(self, reference, predictions, static, frame):
                self.frames.append(frame)
                return {"temporal_has_pair": len(self.frames) > 1}

        lpips = LPIPS()
        fake_runtime = SimpleNamespace(device=torch.device("cpu"), trainer=SimpleNamespace(lpips=lpips))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_cache(directory / "cache")
            diagnose(SimpleNamespace(cache=directory / "cache", output_dir=directory / "diagnostics"))
            args = SimpleNamespace(cache=directory / "cache", base_cache=directory / "cache",
                head=None, guide_ablation="none", no_video=True, output_dir=directory / "evaluation",
                motion_manifest=directory / "diagnostics/motion.json", flow_cache=directory / "flow")
            with mock.patch("tools.ego_motion_runtime.SceneRuntime", return_value=fake_runtime), \
                 mock.patch("utils.temporal_metrics.TemporalEvaluator", Flow), \
                 mock.patch("torch.cuda.synchronize"):
                evaluate(args)
            self.assertEqual([e.frames for e in evaluators], [[2, 3], [6, 7]])
            self.assertEqual(lpips.reset_count, 4)
            self.assertEqual(lpips.assert_shapes, ((1, 3, 8, 10), (1, 3, 8, 10)))
            for predicted, reference in lpips.inputs:
                torch.testing.assert_close(predicted, torch.full_like(predicted, .35))
                torch.testing.assert_close(reference, torch.full_like(reference, .4))
            summary = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(summary["val"]["temporal_pairs"], 1)
            self.assertEqual(summary["test"]["temporal_pairs"], 1)
            self.assertAlmostEqual(summary["val"]["prediction_full_psnr"], -20 * math.log10(.05), places=5)


if __name__ == "__main__":
    unittest.main()
