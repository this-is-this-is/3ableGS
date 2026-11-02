"""CPU checks of the temporal hook in the real PBR forward pipeline.

Only rasterization, lighting and camera construction are replaced by small
deterministic stand-ins. The trainer's material unpack, transport, affine,
forward and checkpoint methods are the implementation under test.
"""
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from omegaconf import OmegaConf

from models.trainers.scene_graph_pbr import MultiTrainer_pbr


class FakeEnvironment(nn.Module):
    external = None

    def __init__(self):
        super().__init__()
        self.shade_inputs = []

    def shade(self, albedo, metallic, roughness, normal, viewdir, **kwargs):
        self.shade_inputs.append({
            "albedo": albedo.clone(), "roughness": roughness.clone(),
            "normal": normal.clone(), "viewdir": viewdir.clone(),
        })
        diffuse = albedo * .5
        specular = roughness.expand_as(albedo) * .2 + normal.abs() * .05
        return {
            "rgb": diffuse + specular, "diffuse": diffuse,
            "specular": specular, "normal_oriented": normal,
        }

    def background(self, viewdirs, **kwargs):
        return torch.zeros_like(viewdirs)


class FakeAffine(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_test_set = False
        self.calls = []

    def forward(self, image_infos):
        self.calls.append((self.in_test_set, "img_idx" in image_infos))
        h, w = image_infos["viewdirs"].shape[:2]
        result = torch.eye(3, 4).expand(h, w, 3, 4).clone()
        if self.in_test_set or "img_idx" not in image_infos:
            result[..., 3] = .1
        else:
            result[..., 3] = image_infos["img_idx"][..., None].float() * .03
        return result


class SyntheticTrainer(MultiTrainer_pbr):
    def __init__(self, size=8):
        # Avoid creating GPU-only geometry, HDR and LPIPS resources.
        nn.Module.__init__(self)
        self.size = size
        self.device = torch.device("cpu")
        self.normalized_timestamps = torch.linspace(0, 1, 4)
        self.cur_frame = torch.tensor(0)
        self.test_set_indices = [1]
        self.step = 0
        self.models = {"EnvLight": FakeEnvironment(), "Affine": FakeAffine()}
        self.gaussian_classes = {"Background": 0}
        self.models["Background"] = nn.Identity()
        self.model_config = OmegaConf.create({key: {} for key in self.models})
        self.render_cfg = OmegaConf.create({"near_plane": .01, "far_plane": 100.})
        self.pts_labels = torch.tensor([0, 1])
        self.render_each_class = False
        self.render_dynamic_mask = False
        self._suppress_eval_layers = True
        self.auxiliary_renders = 0
        self.normal = torch.tensor([0., 0., -1.]).expand(size, size, 3).clone()
        self.albedo = torch.full((size, size, 3), .4)
        self.roughness = torch.full((size, size, 1), .5)
        self.opacity = torch.ones(size, size, 1)
        self.dynamic_opacity = torch.zeros(size, size, 1)
        self.depth = torch.full((size, size, 1), 4.)
        self.eval()

    def process_camera(self, camera_infos, image_ids, novel_view=False):
        return SimpleNamespace(
            camtoworlds=camera_infos["camera_to_world"],
            Ks=camera_infos["intrinsics"], H=self.size, W=self.size,
        )

    def collect_gaussians(self, cam, weather, img_idx):
        return None

    def render_gaussians(self, gs, cam, **kwargs):
        feat = torch.cat((
            self.albedo, torch.full_like(self.roughness, .1),
            self.roughness, self.normal,
        ), -1) * self.opacity

        def render_fn(opacity_mask=None, **unused):
            self.auxiliary_renders += 1
            # Dynamic exclusion must originate from the renderer, regardless
            # of deliberately wrong ground-truth masks supplied in the input.
            torch.testing.assert_close(opacity_mask, torch.tensor([False, True]))
            return feat.clone(), self.depth.clone(), self.dynamic_opacity.clone()

        return {
            "rgb_gaussians": feat, "depth": self.depth.clone(),
            "opacity": self.opacity.clone(),
        }, render_fn

    def display_transform(self, linear_rgb):
        return linear_rgb

    def inputs(self, frame):
        size = self.size
        images = {
            "normed_time": torch.full((size, size), float(frame) / 3),
            "img_idx": torch.full((size, size), frame, dtype=torch.long),
            "frame_idx": torch.full((size, size), frame, dtype=torch.long),
            "viewdirs": torch.tensor([0., 0., 1.]).expand(size, size, 3).clone(),
            "dynamic_masks": torch.ones(size, size),
        }
        cameras = {
            "camera_to_world": torch.eye(4),
            "intrinsics": torch.tensor([[8., 0., size / 2],
                                         [0., 8., size / 2], [0., 0., 1.]]),
            "cam_id": torch.zeros(size, size, dtype=torch.long),
            "height": size, "width": size,
        }
        return images, cameras


class TemporalMaterialIntegrationTests(unittest.TestCase):
    def test_rgb_control_reuses_linear_radiance_but_preserves_current_material(self):
        trainer = SyntheticTrainer()
        trainer.configure_temporal_material(
            {"mode": "normal", "erode_radius": 0, "history_weight": .3}, domain="rgb",
        )
        with torch.no_grad():
            first = trainer(*trainer.inputs(0))
            trainer.albedo.fill_(.8)
            trainer.roughness.fill_(.7)
            current = trainer(*trainer.inputs(1))
        torch.testing.assert_close(current["albedo"], trainer.albedo)
        torch.testing.assert_close(current["roughness"], trainer.roughness)
        weight = current["temporal_weight"]
        expected = current["rgb_unfiltered"] * (1 - weight) + first["rgb"] * weight
        torch.testing.assert_close(current["rgb"], expected)
        self.assertGreater(float(weight[2, 2]), 0)

    def test_block_train_ids_reach_nodes_in_training_and_evaluation(self):
        trainer = SyntheticTrainer()
        trainer._block_holdout_train_ids = (0, 3)
        for training in (True, False):
            trainer.train(training)
            trainer(*trainer.inputs(0))
            self.assertEqual(trainer.models["Background"].train_frame_ids, (0, 3))

    def test_disabled_forward_keeps_original_contract_and_render_cost(self):
        trainer = SyntheticTrainer()
        images, cameras = trainer.inputs(0)
        with torch.no_grad():
            original = trainer(images, cameras)
            trainer.configure_temporal_material(None)
            disabled = trainer(images, cameras)
        self.assertEqual(trainer.auxiliary_renders, 0)
        self.assertEqual(set(original), set(disabled))
        self.assertNotIn("rgb_unfiltered", disabled)
        for name in original:
            torch.testing.assert_close(original[name], disabled[name], rtol=0, atol=0)

    def test_transport_changes_material_before_shading_and_excludes_dynamic_pixels(self):
        trainer = SyntheticTrainer()
        trainer.configure_temporal_material({"mode": "normal", "erode_radius": 0})
        trainer.dynamic_opacity[3, 3] = .8
        with torch.no_grad():
            first = trainer(*trainer.inputs(0))
            torch.testing.assert_close(first["rgb"], first["rgb_unfiltered"])
            self.assertEqual(torch.count_nonzero(first["temporal_weight"]), 0)
            trainer.albedo.fill_(.6)
            trainer.roughness.fill_(.7)
            trainer.normal[..., 0] = .1
            trainer.normal = torch.nn.functional.normalize(trainer.normal, dim=-1)
            images, cameras = trainer.inputs(1)
            images["viewdirs"][..., 1] = .1
            second = trainer(images, cameras)
        self.assertEqual(trainer.auxiliary_renders, 2)
        self.assertGreater(float(second["temporal_weight"][2, 2]), 0.)
        self.assertLess(float(second["albedo"][2, 2, 0]), .6)
        self.assertLess(float(second["roughness"][2, 2, 0]), .7)
        self.assertGreater(float(
            (second["rgb"] - second["rgb_unfiltered"]).abs().max()), 0.)
        self.assertFalse(bool(second["temporal_static_mask"][3, 3]))
        self.assertEqual(float(second["temporal_weight"][3, 3]), 0.)
        torch.testing.assert_close(second["rgb"][3, 3], second["rgb_unfiltered"][3, 3])
        torch.testing.assert_close(second["normal"], trainer.normal)
        # Both shading calls must see this frame's orientation and view.
        unfiltered, transported = trainer.env.shade_inputs[-2:]
        torch.testing.assert_close(unfiltered["normal"], trainer.normal)
        torch.testing.assert_close(transported["normal"], trainer.normal)
        torch.testing.assert_close(unfiltered["viewdir"], transported["viewdir"])
        expected_view = -torch.nn.functional.normalize(images["viewdirs"], dim=-1)
        torch.testing.assert_close(transported["viewdir"], expected_view)
        expected_rgb = (second["albedo"] * .5
                        + second["roughness"].expand_as(second["albedo"]) * .2
                        + trainer.normal.abs() * .05 + .1)
        torch.testing.assert_close(second["rgb"], expected_rgb)

    def test_paired_affine_is_shared_across_train_test_boundary_without_input_mutation(self):
        trainer = SyntheticTrainer()
        trainer.configure_temporal_material({})
        snapshots = []
        with torch.no_grad():
            for frame in (0, 1):
                images, cameras = trainer.inputs(frame)
                original_keys = set(images)
                original_ids = images["img_idx"].clone()
                result = trainer(images, cameras)
                snapshots.append(result["rgb_unfiltered"])
                self.assertEqual(set(images), original_keys)
                torch.testing.assert_close(images["img_idx"], original_ids)
                self.assertTrue(all(not has_id for _, has_id in trainer.models["Affine"].calls))
        torch.testing.assert_close(snapshots[0], snapshots[1])
        # Train/test state is still propagated to other scene components.
        self.assertTrue(trainer.models["Affine"].in_test_set)

    def test_inference_and_sequence_guards_fail_before_rendering(self):
        trainer = SyntheticTrainer()
        trainer.configure_temporal_material({})
        images, cameras = trainer.inputs(0)
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            trainer(images, cameras)
        trainer.train()
        with torch.no_grad(), self.assertRaises(RuntimeError):
            trainer(images, cameras)
        trainer.eval()
        for kwargs in ({"novel_view": True}, {"weather": "rain"}):
            with torch.no_grad(), self.assertRaises(ValueError):
                trainer(images, cameras, **kwargs)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "frame_idx"):
            trainer({k: v for k, v in images.items() if k != "frame_idx"}, cameras)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "cam_id"):
            trainer(images, {k: v for k, v in cameras.items() if k != "cam_id"})
        self.assertEqual(trainer.auxiliary_renders, 0)
        self.assertEqual(trainer.env.shade_inputs, [])

    def test_static_scene_needs_no_dynamic_raster_and_camera_switch_is_cold(self):
        trainer = SyntheticTrainer()
        trainer.pts_labels.zero_()
        trainer.configure_temporal_material({"erode_radius": 0})
        with torch.no_grad():
            trainer(*trainer.inputs(0))
            second = trainer(*trainer.inputs(1))
            self.assertGreater(float(second["temporal_weight"].max()), 0.)
            images, cameras = trainer.inputs(2)
            cameras["cam_id"].fill_(1)
            switched = trainer(images, cameras)
        self.assertEqual(trainer.auxiliary_renders, 0)
        self.assertEqual(torch.count_nonzero(switched["temporal_weight"]), 0)

    def test_history_is_ephemeral_and_checkpoint_load_resets_it(self):
        trainer = SyntheticTrainer()
        checkpoint = trainer.state_dict()
        trainer.configure_temporal_material({"erode_radius": 0})
        with torch.no_grad():
            trainer(*trainer.inputs(0))
            warm = trainer(*trainer.inputs(1))
            self.assertGreater(float(warm["temporal_weight"].max()), 0.)
            self.assertEqual(set(trainer.state_dict()), set(checkpoint))
            self.assertEqual(set(trainer.state_dict()["models"]), set(checkpoint["models"]))
            trainer.load_state_dict(checkpoint, strict=True)
            cold = trainer(*trainer.inputs(2))
        self.assertEqual(torch.count_nonzero(cold["temporal_weight"]), 0)
        # The loader must not consume the caller's envelope, and an old
        # checkpoint does not enable history on a fresh trainer.
        self.assertIn("models", checkpoint)
        self.assertIn("step", checkpoint)
        clone = SyntheticTrainer()
        clone.load_state_dict(checkpoint, strict=True)
        with torch.no_grad():
            result = clone(*clone.inputs(0))
        self.assertNotIn("rgb_unfiltered", result)
        self.assertEqual(clone.auxiliary_renders, 0)

    def test_explicit_reset_and_disable_release_history(self):
        trainer = SyntheticTrainer()
        trainer.configure_temporal_material({"erode_radius": 0}, shared_affine=False)
        with torch.no_grad():
            trainer(*trainer.inputs(0))
            warm = trainer(*trainer.inputs(1))
            self.assertGreater(float(warm["temporal_weight"].max()), 0.)
            trainer.reset_temporal_material()
            cold = trainer(*trainer.inputs(2))
            self.assertEqual(torch.count_nonzero(cold["temporal_weight"]), 0)
            trainer.configure_temporal_material(None)
            renders_before = trainer.auxiliary_renders
            disabled = trainer(*trainer.inputs(3))
        self.assertNotIn("rgb_unfiltered", disabled)
        self.assertEqual(trainer.auxiliary_renders, renders_before)
        self.assertIsNone(trainer._temporal_material)


if __name__ == "__main__":
    unittest.main()
