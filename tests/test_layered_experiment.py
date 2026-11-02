"""CPU orchestration checks for the frozen layered G-buffer experiment."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tools.layered_gbuffer_experiment import (
    assert_conservation, diagnostic_summary, make_parser, parse_frames, run_experiment, select_frames,
    validate_args, variant_configs,
    trained_variant_configs,
)


class FakeRuntime:
    def __init__(self, checkpoint, **kwargs):
        self.checkpoint, self.kwargs = checkpoint, kwargs
        self.frame_ids = list(range(8))
        self.train_frames, self.val_frames, self.test_frames = [0, 1, 4, 5], [2, 3], [6, 7]
        self.configuration, self.calls = None, []
        self.trainer = SimpleNamespace(configure_layered_gbuffer=self.configure)
        self.resets = 0

    def configure(self, value):
        self.configuration = value

    def reset(self):
        self.resets += 1

    def render(self, frame_id):
        self.calls.append((frame_id, self.configuration))
        reference = np.full((12, 16, 3), .4 + frame_id * .01, dtype=np.float32)
        dynamic = np.zeros(reference.shape[:2], dtype=bool)
        dynamic[:, :8] = True
        valid = np.ones_like(dynamic)
        valid[-1] = False
        return {
            "rgb": reference + .05,
            "target_rgb": reference,
            "evaluation_dynamic": dynamic,
            "evaluation_valid": valid,
            "forward_ms": 1.5,
            "layered_diagnostics": {"group_count": np.array(3), "mass_reconstruction_max_abs": 0.,
                                    "feature_reconstruction_max_abs": 0., "raster_mass_max_abs": 0.},
        }


class FakeEvaluator:
    instances = []

    def __init__(self):
        self.calls = []
        self.previous = None
        self.instances.append(self)

    def evaluate(self, reference, predictions, dynamic_mask, frame_id, evaluation_mask=None, depth=None):
        self.calls.append((reference.copy(), predictions, dynamic_mask.copy(), frame_id,
                           evaluation_mask.copy(), depth))
        paired = self.previous is not None and frame_id == self.previous + 1
        self.previous = frame_id
        return {"frame_id": frame_id, "temporal_has_pair": paired, "methods": len(predictions)}


def fake_summary(records):
    return {"frames": len(records), "temporal_pairs": sum(row["temporal_has_pair"] for row in records)}


class LayeredExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source"
        self.source.mkdir()
        self.checkpoint = self.source / "checkpoint_final.pth"
        self.checkpoint.write_bytes(b"fake checkpoint, never torch loaded")
        (self.source / "config.yaml").write_text("unit_test: true\n")
        self.output = self.directory / "output"
        FakeEvaluator.instances = []

    def args(self, *extra):
        args = make_parser().parse_args([
            "--checkpoint", str(self.checkpoint), "--output", str(self.output), "--no-perceptual", *extra])
        return validate_args(args)

    def run_fake(self, args, factory=FakeRuntime):
        return run_experiment(args, runtime_factory=factory,
                              evaluator_factory=FakeEvaluator, summarize=fake_summary)

    def test_default_matrix_controls_colorspace_without_changing_grouping(self):
        matrix = variant_configs()
        self.assertEqual(list(matrix), ["B0", "B1", "L0", "L1"])
        self.assertIsNone(matrix["B0"])
        self.assertEqual(matrix["B1"]["grouping"], "single")
        self.assertEqual(matrix["B1"]["color_space"], "linear")
        self.assertEqual(matrix["L0"]["grouping"], matrix["L1"]["grouping"])
        self.assertEqual(matrix["L0"]["color_space"], "srgb")
        self.assertEqual(matrix["L1"]["color_space"], "linear")
        expanded = variant_configs("dynamic", [0, 2, 4], ["normal", "brdf", "albedo"])
        self.assertEqual(len(expanded), 17)
        self.assertEqual(expanded["L1_k2_global_normal"]["max_layers"], 2)
        self.assertEqual(expanded["L1_k4_global_brdf"]["ablation"], "brdf")

    def test_trained_only_recovers_saved_renderer_and_evaluates_one_method(self):
        (self.source / "config.yaml").write_text(
            "trainer:\n  layered_gbuffer:\n    grouping: instance\n    color_space: srgb\n"
            "    max_layers: 0\n    ablation: none\n    diagnostics: false\n")
        result = self.run_fake(self.args("--trained-only"))
        self.assertEqual(list(result["identity"]["variants"]), ["trained"])
        self.assertTrue(result["identity"]["variants"]["trained"]["diagnostics"])
        self.assertEqual(result["identity"]["renderer_policy"], "saved_trained_renderer")
        for call in FakeEvaluator.instances[-1].calls:
            self.assertEqual(list(call[1]), ["trained"])

    def test_trained_only_requires_metadata_and_does_not_override_it(self):
        with self.assertRaisesRegex(ValueError, "explicit saved"):
            trained_variant_configs(self.checkpoint)
        (self.source / "config.yaml").write_text("trainer:\n  layered_gbuffer: null\n")
        self.assertEqual(trained_variant_configs(self.checkpoint), {"trained": None})
        with self.assertRaisesRegex(ValueError, "saved renderer"):
            self.args("--trained-only", "--layer-budgets", "2")

    def test_frames_cannot_silently_overlap_leave_split_or_use_negative_ids(self):
        self.assertEqual(parse_frames("6:8,2:4"), [2, 3, 6, 7])
        for value in ("1:3,2", "", "-1", "4:2", "1:2:3", "2,2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_frames(value)
        runtime = FakeRuntime(self.checkpoint)
        with self.assertRaisesRegex(ValueError, "outside.*test split"):
            select_frames(runtime, self.args("--split", "test", "--frames", "2"))
        self.assertEqual(select_frames(runtime, self.args("--max-frames", "2")), [2, 3])

    def test_all_requires_explicit_diagnostic_authorization(self):
        with self.assertRaisesRegex(ValueError, "requires --diagnostic-all"):
            self.args("--split", "all")
        args = self.args("--split", "all", "--diagnostic-all")
        instances = []
        def factory(*values, **kwargs):
            instances.append(FakeRuntime(*values, **kwargs))
            return instances[-1]
        summary = self.run_fake(args, factory)
        self.assertFalse(instances[0].kwargs["require_holdout"])
        self.assertIn("not_clean_holdout", summary["evaluation_claim"])
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["selected_training_frames"], [0, 1, 4, 5])

    def test_invalid_cli_options_fail_before_runtime(self):
        invalid = [("--layer-budgets", "2,2"), ("--layer-budgets", "1"),
                   ("--ablations", "normal,unknown"), ("--downscale", "nan"),
                   ("--downscale", ".5"), ("--max-frames", "0"), ("--save-every", "-1")]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.args(*options)

    def test_every_variant_uses_same_frame_reference_and_reference_masks(self):
        instances = []
        def factory(*values, **kwargs):
            instances.append(FakeRuntime(*values, **kwargs))
            return instances[-1]
        summary = self.run_fake(self.args(), factory)
        runtime = instances[0]
        self.assertEqual(runtime.kwargs, {"downscale": 1., "antialiased": False, "require_holdout": True})
        self.assertEqual([frame for frame, _ in runtime.calls], [2] * 4 + [3] * 4 + [6] * 4 + [7] * 4)
        self.assertEqual(summary["metrics"]["temporal_pairs"], 2)
        self.assertIsNone(runtime.configuration)
        evaluator = FakeEvaluator.instances[-1]
        for reference, predictions, dynamic, frame, valid, depth in evaluator.calls:
            self.assertEqual(list(predictions), ["B0", "B1", "L0", "L1"])
            self.assertEqual(int(dynamic.sum()), 12 * 8)
            self.assertEqual(int(valid.sum()), 11 * 16)
            self.assertTrue(dynamic[-1].any())  # Invalidity is a separate mask.
            for value in predictions.values():
                np.testing.assert_allclose(value, reference + .05)
        self.assertEqual(summary["status"], "completed")
        self.assertIn("shared_mean", summary["identity"]["affine_policy"])
        self.assertEqual(len(list((self.output / "frames").glob("*.json"))), 4)
        self.assertAlmostEqual(summary["rendering"]["L1"]["preclip_global_mse"], .05 ** 2, places=7)

    def test_reference_depth_is_passed_only_to_evaluator(self):
        class WithDepth(FakeRuntime):
            def render(self, frame):
                packet = super().render(frame)
                packet["evaluation_depth"] = np.ones((12, 16), np.float32) * 2
                return packet
        self.run_fake(self.args("--max-frames", "1"), WithDepth)
        depth = FakeEvaluator.instances[-1].calls[0][-1]
        np.testing.assert_array_equal(depth, np.ones((12, 16), np.float32) * 2)

    def test_missing_dynamic_mask_fails_and_records_failure(self):
        class MissingMask(FakeRuntime):
            def render(self, frame):
                packet = super().render(frame)
                packet["evaluation_dynamic"] = None
                return packet
        with self.assertRaisesRegex(ValueError, "Missing reference evaluation_dynamic"):
            self.run_fake(self.args(), MissingMask)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_frames"], 0)

    def test_display_clipping_is_common_and_logged(self):
        class Saturated(FakeRuntime):
            def render(self, frame):
                packet = super().render(frame)
                packet["rgb"][0, 0] = [-.5, 2., .5]
                return packet
        summary = self.run_fake(self.args("--max-frames", "1"), Saturated)
        predictions = FakeEvaluator.instances[-1].calls[0][1]
        for value in predictions.values():
            np.testing.assert_array_equal(value[0, 0], [0, 1, .5])
        self.assertEqual(summary["rendering"]["B0"]["unclipped_rgb_max"]["max"], 2.)
        self.assertGreater(summary["rendering"]["B0"]["clipped_rgb_fraction"]["mean"], 0)

    def test_existing_outputs_need_explicit_matching_restart(self):
        args = self.args("--max-frames", "1")
        self.run_fake(args)
        with self.assertRaises(FileExistsError):
            self.run_fake(args)
        restarted = self.run_fake(self.args("--max-frames", "1", "--restart"))
        self.assertEqual(restarted["frames"], 1)
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            self.run_fake(self.args("--max-frames", "2", "--restart"))
        self.checkpoint.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            self.run_fake(self.args("--max-frames", "1", "--restart"))

    def test_source_mutation_during_run_is_not_marked_completed(self):
        source = self.source
        class Mutating(FakeRuntime):
            def render(self, frame):
                (source / "config.yaml").write_text("changed: true\n")
                return super().render(frame)
        with self.assertRaisesRegex(RuntimeError, "changed during evaluation"):
            self.run_fake(self.args("--max-frames", "1"), Mutating)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")

    def test_preclip_mse_is_pooled_not_mean_of_frame_mses(self):
        records = [
            {"rendering": {"B0": {"preclip_squared_error_sum": 1., "preclip_value_count": 10}}},
            {"rendering": {"B0": {"preclip_squared_error_sum": 9., "preclip_value_count": 90}}},
        ]
        self.assertAlmostEqual(diagnostic_summary(records, ["B0"])["B0"]["preclip_global_mse"], .1)

    def test_mass_feature_and_raster_conservation_fail_closed(self):
        good = {"feature_reconstruction_max_abs": 1e-6, "mass_reconstruction_max_abs": 1e-6,
                "raster_mass_max_abs": 1e-6}
        assert_conservation(good)
        for name in good:
            for value in (1e-3, float("nan"), -1e-6):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "conservation failed"):
                    assert_conservation({**good, name: value})
        with self.assertRaisesRegex(ValueError, "Missing conservation diagnostic"):
            assert_conservation({})


if __name__ == "__main__":
    unittest.main()
