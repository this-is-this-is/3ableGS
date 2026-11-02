"""CPU integration checks for frozen edits and the RGB comparison branch."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_temporal_material_integration import SyntheticTrainer
from tools.render_temporal_material import build_surface_editor, render_inputs


class TemporalSurfaceIntegrationTests(unittest.TestCase):
    def make_bank(self):
        trainer = SyntheticTrainer()
        trainer.configure_temporal_material({"history_weight": 0, "erode_radius": 0})
        trainer._block_holdout_train_ids = (0, 2, 3)
        frames_read, forwarded = [], []

        class Camera:
            def get_image(self, frame, weather):
                frames_read.append(frame)
                images, cameras = trainer.inputs(frame)
                images["pixels"] = torch.ones(trainer.size, trainer.size, 3)
                images["sky_masks"] = torch.ones(trainer.size, trainer.size)
                images["gbuf_normal"] = torch.full((trainer.size, trainer.size, 3), 999.)
                return images, cameras

        original_forward = trainer.forward

        def capture(image_infos, camera_infos, *args, **kwargs):
            forwarded.append(set(image_infos))
            return original_forward(image_infos, camera_infos, *args, **kwargs)

        args = SimpleNamespace(edit_keyframes=2, edit_affinity="gbuffer",
                               edit_roi=[.2, .2, .8, .8])
        with patch.object(trainer, "forward", side_effect=capture):
            summary = build_surface_editor(
                trainer, Camera(), SimpleNamespace(train_timesteps=[0, 2, 3]),
                args, torch.device("cpu"), torch)
        return trainer, summary, frames_read, forwarded

    def test_bank_reads_training_keyframes_only_and_never_forwards_references(self):
        trainer, summary, frames, forwarded = self.make_bank()
        self.assertEqual(frames, [0, 3])
        self.assertEqual(summary["keyframe_ids"], [0, 3])
        self.assertTrue(summary["frozen"])
        for keys in forwarded:
            self.assertTrue(keys.isdisjoint({"pixels", "sky_masks", "dynamic_masks", "gbuf_normal"}))
        self.assertFalse(trainer._export_temporal_inputs)
        self.assertIsNone(trainer._temporal_material._history)
        self.assertEqual(trainer.models["Background"].train_frame_ids, (0, 2, 3))

    def test_surface_edit_changes_material_before_current_shading_and_display(self):
        trainer, _, _, _ = self.make_bank()
        with torch.no_grad():
            images, cameras = trainer.inputs(1)
            result = trainer(render_inputs(images, trainer.device, torch), cameras)
        self.assertGreater(float(result["edit_weight"].max()), .9)
        before, after = trainer.env.shade_inputs[-2:]
        self.assertLess(float(after["albedo"].min()), float(before["albedo"].min()))
        self.assertLess(float(after["roughness"].min()), float(before["roughness"].min()))
        torch.testing.assert_close(after["normal"], before["normal"])
        torch.testing.assert_close(after["viewdir"], before["viewdir"])
        expected = result["albedo"] * .5 + result["roughness"].expand_as(result["albedo"]) * .2
        expected += trainer.normal.abs() * .05 + .1
        torch.testing.assert_close(result["rgb"], expected)
        self.assertGreater(float((result["rgb"] - result["rgb_unfiltered"]).abs().max()), 0)
        self.assertEqual(torch.count_nonzero(result["temporal_weight"]), 0)

    def test_evaluation_order_does_not_change_frozen_edit_bank_or_outputs(self):
        trainer, _, _, _ = self.make_bank()
        bank = trainer._surface_editor
        points, strengths = bank._points.copy(), bank._strength.copy()
        with torch.no_grad():
            first = trainer(*trainer.inputs(1))
            trainer(*trainer.inputs(2))
            second = trainer(*trainer.inputs(1))
        np.testing.assert_array_equal(points, bank._points)
        np.testing.assert_array_equal(strengths, bank._strength)
        torch.testing.assert_close(first["rgb"], second["rgb"], rtol=0, atol=0)
        torch.testing.assert_close(first["edit_weight"], second["edit_weight"], rtol=0, atol=0)

    def test_loading_checkpoint_detaches_old_geometry_edit_bank(self):
        trainer, _, _, _ = self.make_bank()
        self.assertIsNotNone(trainer._surface_editor)
        trainer.load_state_dict(trainer.state_dict())
        self.assertIsNone(trainer._surface_editor)

    def test_rgb_history_blends_linear_radiance_and_keeps_current_material(self):
        trainer = SyntheticTrainer()
        trainer.configure_temporal_material({"mode": "depth", "erode_radius": 0}, domain="rgb")
        with torch.no_grad():
            previous = trainer(*trainer.inputs(0))
            trainer.albedo.fill_(.8)
            trainer.roughness.fill_(.9)
            current = trainer(*trainer.inputs(1))
        torch.testing.assert_close(current["albedo"], trainer.albedo)
        torch.testing.assert_close(current["roughness"], trainer.roughness)
        torch.testing.assert_close(current["normal"], trainer.normal)
        expected = .5 * (previous["rgb_unfiltered"] + current["rgb_unfiltered"])
        torch.testing.assert_close(current["rgb"], expected)
        self.assertNotIn("temporal_signal", current)

    def test_explicit_train_ids_propagate_on_training_forward(self):
        trainer = SyntheticTrainer()
        trainer.train()
        trainer._block_holdout_train_ids = (0, 2, 3)
        trainer(*trainer.inputs(0))
        self.assertEqual(trainer.models["Background"].train_frame_ids, (0, 2, 3))


if __name__ == "__main__":
    unittest.main()
