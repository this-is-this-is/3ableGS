"""CPU contract checks for split integrity and differentiable PBR packets."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from tools.ego_motion_runtime import (
    SceneRuntime, detach_packet, experiment_splits, packet_to,
    select_material_parameters,
)


class RuntimeSplitTests(unittest.TestCase):
    def test_saved_blocks_define_three_disjoint_splits(self):
        train, validation, test = experiment_splits(
            12, {"test_frame_ranges": [[2, 4], [7, 9]], "test_image_stride": 1})
        self.assertEqual(validation, [2, 3])
        self.assertEqual(test, [7, 8])
        self.assertEqual(train, [0, 1, 4, 5, 6, 9, 10, 11])
        self.assertEqual(set(train) | set(validation) | set(test), set(range(12)))

    def test_never_relabels_stride_training_as_a_new_block_holdout(self):
        for config in ({"test_image_stride": 2}, {"test_frame_ranges": [[2, 4]]},
                       {"test_frame_ranges": [[2, 4], [4, 6]]},
                       {"test_frame_ranges": [[2, 3], [7, 9]]}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                experiment_splits(12, config)


class FakeMaterialNode(nn.Module):
    def __init__(self, learned_normal):
        super().__init__()
        self._features = nn.Parameter(torch.ones(4, 5))
        self._means = nn.Parameter(torch.ones(4, 3))
        self._quats = nn.Parameter(torch.ones(4, 4))
        self.mlp_material = nn.ModuleDict({"raw": nn.Linear(5, 5)})
        self.raw_normal_residual = nn.Linear(5, 3) if learned_normal else None


class MaterialFreezingTests(unittest.TestCase):
    def test_features_shared_with_normal_remain_frozen_after_checkpoint_reload(self):
        trainer = nn.Module()
        trainer.models = {
            "Background": FakeMaterialNode(True), "RigidNodes": FakeMaterialNode(False),
            "EnvLight": nn.Linear(2, 2),
        }
        trainer.gaussian_classes = {"Background": 0, "RigidNodes": 1}
        selected, summary = select_material_parameters(trainer)
        names = {name for name, _ in selected}
        self.assertNotIn("Background._features", names)
        self.assertIn("RigidNodes._features", names)
        before = trainer.models["Background"].raw_normal_residual(
            trainer.models["Background"]._features).detach().clone()
        optimizer = torch.optim.SGD([parameter for _, parameter in selected], lr=.1)
        loss = sum(node.mlp_material["raw"](node._features).sum()
                   for name, node in trainer.models.items() if name in trainer.gaussian_classes)
        loss.backward()
        optimizer.step()
        reloaded = FakeMaterialNode(True)
        reloaded.load_state_dict(trainer.models["Background"].state_dict())
        after = reloaded.raw_normal_residual(reloaded._features)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertFalse(trainer.models["Background"]._means.requires_grad)
        self.assertFalse(trainer.models["Background"].raw_normal_residual.weight.requires_grad)
        self.assertFalse(trainer.models["EnvLight"].weight.requires_grad)
        self.assertEqual(summary["nodes"][0]["features_frozen_reason"], "shared_with_learned_normal")


class PacketRuntimeTests(unittest.TestCase):
    @staticmethod
    def runtime():
        value = nn.Parameter(torch.tensor(.4))
        calls = []

        class FakeTrainer:
            def set_eval(self):
                calls.append("eval")

            def __call__(self, infos, cameras, weather):
                calls.append(set(infos))
                rgb = value.expand(3, 4, 3) * 1.
                scalar = rgb[..., :1]
                return {
                    "rgb": rgb, "albedo": rgb, "roughness": scalar,
                    "metallic": scalar, "depth": scalar, "opacity": scalar,
                    "normal": -torch.ones_like(rgb),
                    "ego_normal": torch.ones_like(rgb),
                    "ego_static_mask": torch.ones(3, 4, dtype=torch.bool),
                    "ego_camera_to_world": torch.eye(4), "ego_intrinsics": torch.eye(3),
                    "ego_viewdirs": torch.ones_like(rgb),
                    "ego_background_linear": torch.full_like(rgb, .2),
                }

            def shade_gbuffer(self, materials, viewdirs, weather):
                return {"rgb": materials["albedo"] * materials["roughness"]}

            def display_transform(self, linear):
                return linear

            def output_transform(self, display, infos):
                return display

        runtime = SceneRuntime.__new__(SceneRuntime)
        runtime.device = torch.device("cpu")
        runtime.trainer = FakeTrainer()
        runtime.frame_ids = list(range(4))
        runtime.train_frames = [0, 1]
        infos = {
            "pixels": torch.zeros(3, 4, 3), "viewdirs": torch.ones(3, 4, 3),
            "sky_masks": torch.zeros(3, 4), "dynamic_masks": torch.zeros(3, 4),
            "gbuffer_albedo": torch.full((3, 4, 3), .9), "lidar_depth_map": torch.ones(3, 4),
            "img_idx": torch.tensor([0]), "frame_idx": torch.tensor([0]),
        }
        runtime.camera = SimpleNamespace(get_image=lambda frame, weather: (infos, {}))
        return runtime, value, calls

    def test_renderer_receives_no_reference_fields_and_exports_world_normal(self):
        runtime, value, calls = self.runtime()
        packet = runtime.render(0)
        self.assertEqual(calls[-1], {"viewdirs", "img_idx", "frame_idx"})
        self.assertTrue((packet["normal"] == 1).all())
        self.assertFalse(packet["rgb"].requires_grad)
        self.assertIn("target_rgb", packet)
        self.assertIn("evaluation_static", packet)
        self.assertNotIn("pixels", packet["image_infos"])
        self.assertTrue(packet["static_mask"].all())
        self.assertGreaterEqual(packet["forward_ms"], 0.)

    def test_grad_render_and_reshading_preserve_material_gradient_only_for_train(self):
        runtime, value, calls = self.runtime()
        packet = runtime.render(0, grad=True)
        packet["rgb"].sum().backward()
        self.assertEqual(value.grad.item(), 36)
        detached = detach_packet(packet)
        albedo = detached["albedo"].clone().requires_grad_()
        shaded = runtime.shade(detached, albedo)
        shaded.sum().backward()
        torch.testing.assert_close(albedo.grad, torch.full_like(albedo, .16))
        torch.testing.assert_close(shaded, torch.full_like(shaded, .184))
        with self.assertRaises(ValueError):
            runtime.render(2, grad=True)

    def test_detached_cache_has_independent_storage_and_preserves_nested_inputs(self):
        runtime, _, _ = self.runtime()
        packet = runtime.render(0, grad=True)
        cached = detach_packet(packet)
        self.assertFalse(cached["albedo"].requires_grad)
        self.assertNotEqual(cached["albedo"].data_ptr(), packet["albedo"].data_ptr())
        moved = packet_to(cached, "cpu")
        self.assertEqual(moved["frame_id"], 0)
        self.assertTrue(torch.equal(moved["image_infos"]["viewdirs"], packet["image_infos"]["viewdirs"]))

    def test_training_loss_packet_keeps_references_out_of_renderer_and_holdout(self):
        runtime, _, calls = self.runtime()
        packet = runtime.render(0, grad=True, return_outputs=True)
        self.assertIn("training_outputs", packet)
        self.assertIn("pixels", packet["training_image_infos"])
        self.assertNotIn("pixels", calls[-1])
        self.assertNotIn("gbuffer_albedo", calls[-1])
        self.assertTrue(packet["evaluation_valid"].all())
        self.assertFalse(packet["evaluation_dynamic"].any())
        with self.assertRaisesRegex(ValueError, "Training loss packets"):
            runtime.render(2, return_outputs=True)

    def test_original_dynamic_loss_gets_auxiliary_layers_during_training_packet(self):
        runtime, _, _ = self.runtime()
        old_trainer = runtime.trainer
        observed = []
        class DynamicTrainer(type(old_trainer)):
            def __call__(self, *args):
                observed.append(self._suppress_eval_layers)
                return super().__call__(*args)
        runtime.trainer = DynamicTrainer()
        runtime.trainer.losses_dict = {"dynamic_region": {"w": .2, "start_from": 0}}
        runtime.trainer.step = 30000
        runtime.trainer._suppress_eval_layers = True
        runtime.render(0, grad=True, return_outputs=True)
        self.assertEqual(observed, [False])
        self.assertTrue(runtime.trainer._suppress_eval_layers)


if __name__ == "__main__":
    unittest.main()
