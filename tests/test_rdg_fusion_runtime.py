"""CPU-only checks of sparse history loading, state isolation and persistence."""
import copy
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from models.rdg_fusion_runtime import RDGFusionMixin


def inputs(index, height=25, width=29, num_cams=2):
    images = {
        "viewdirs": torch.tensor([0., 0., 1.]).expand(height, width, 3).clone(),
        "normed_time": torch.full((height, width), (index // num_cams) / 6),
        "img_idx": torch.full((height, width), index, dtype=torch.long),
        "frame_idx": torch.full((height, width), index // num_cams, dtype=torch.long),
        "pixels": torch.full((height, width, 3), float("nan")),
        "dynamic_masks": torch.ones(height, width),
        "gbuf_basecolor": torch.full((height, width, 3), -99.),
    }
    camera = {
        "camera_to_world": torch.eye(4),
        "intrinsics": torch.tensor([[8., 0., width / 2], [0., 8., height / 2], [0., 0., 1.]]),
        "cam_id": torch.full((height, width), (2, 5)[index % num_cams], dtype=torch.long),
        "height": height, "width": width,
    }
    return images, camera


class Sequence:
    def __init__(self, trainer):
        self.trainer = trainer
        self.reads = []
        self.split_indices = list(range(14))
        self.corrupt = None

    def __len__(self):
        return len(self.split_indices)

    def get_image(self, index, downscale, weathers):
        self.reads.append((index, downscale, tuple(weathers)))
        source = self.trainer.dataset.pixel_source
        source._old_downscale_factor.append(source._downscale_factor)
        source._downscale_factor = 1 / downscale
        source.camera_data[2].downscale_factor = 1 / downscale
        if self.corrupt == "loader_exception":
            raise RuntimeError("synthetic load failure")
        images, camera = inputs(index, int(25 / downscale), int(29 / downscale))
        if self.corrupt == "frame":
            images["frame_idx"].add_(1)
        elif self.corrupt == "time":
            images["normed_time"].fill_(.987)
        elif self.corrupt == "camera":
            camera["cam_id"].fill_(3)
        elif self.corrupt == "size":
            images, camera = inputs(index)
        return {"raw": images}, {"raw": camera}


class MinimalBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.step = 6000
        self.normalized_timestamps = torch.linspace(0, 1, 7)
        self.scene_value = nn.Parameter(torch.tensor(.4))
        self.cur_frame = 91
        self.info = {"owner": "current"}
        self.pts_labels = torch.tensor([0])
        self.models = {"Background": SimpleNamespace(
            cur_frame=91, in_test_set=False, _gs_cache={"owner": "current"},
            _raw_normal_residual_penalty=self.scene_value.square())}
        self.dataset = SimpleNamespace(
            train_indices=[0, 1, 2, 3, 6, 7, 10, 11, 12, 13],
            pixel_source=SimpleNamespace(
                num_cams=2, camera_list=[2, 5], _downscale_factor=1.,
                _old_downscale_factor=[.75], camera_data={
                    2: SimpleNamespace(unique_cam_idx=0, downscale_factor=1.),
                    5: SimpleNamespace(unique_cam_idx=1, downscale_factor=1.),
                }))
        self.dataset.full_image_set = Sequence(self)
        self.seen = []
        self.corrupt = None

    def _get_downscale_factor(self):
        return 1

    def forward(self, image_infos, camera_infos, weather):
        self.seen.append((set(image_infos), torch.is_grad_enabled(), self._rdg_context))
        self.cur_frame = int(image_infos["frame_idx"][0, 0])
        self.info = {"owner": "history"}
        self.pts_labels = torch.tensor([1])
        self._render_c2w = camera_infos["camera_to_world"]
        self.models["Background"]._gs_cache = {"owner": "history"}
        self.models["Background"]._raw_normal_residual_penalty = torch.zeros(())
        torch.rand(4)
        random.random()
        np.random.rand()
        if self.corrupt == "forward_exception":
            raise RuntimeError("synthetic render failure")
        height, width = image_infos["viewdirs"].shape[:2]
        albedo = self.scene_value.expand(height, width, 3).clone()
        packet = {
            "albedo": albedo, "rgb": albedo, "normal": -image_infos["viewdirs"],
            "roughness": torch.full((height, width, 1), .5),
            "metallic": torch.full((height, width, 1), .1),
            "opacity": torch.ones(height, width, 1), "depth": torch.full((height, width, 1), 4.),
            "viewdirs": image_infos["viewdirs"], "static_mask": torch.ones(height, width, dtype=torch.bool),
            "camera_to_world": camera_infos["camera_to_world"], "intrinsics": camera_infos["intrinsics"],
            "image_id": int(image_infos["img_idx"][0, 0]), "frame_id": self.cur_frame,
            "stream_id": (weather, int(camera_infos["cam_id"][0, 0])),
        }
        if self.corrupt == "packet":
            packet["frame_id"] += 1
        return {"_rdg_packet": packet}

    def initialize_optimizer(self):
        self.param_groups = {"Scene#all": [self.scene_value]}
        self.optimizer = torch.optim.Adam([{
            "params": [self.scene_value], "name": "Scene#all", "lr": .01}])

    def optimizer_step(self):
        self.clipped_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.norm() for parameter in self.rdg_fusion_head.parameters()
            if parameter.grad is not None]))
        self.optimizer.step()

    def state_dict(self, only_model=True):
        return {**super().state_dict(), "step": self.step}

    def load_state_dict(self, state_dict, load_only_model=True, strict=True):
        state = dict(state_dict)
        self.step = state.pop("step")
        return super().load_state_dict(state, strict=strict)


class Trainer(RDGFusionMixin, MinimalBase):
    def configure(self, **kwargs):
        self._configure_rdg_fusion({"enabled": True, "width": 4, **kwargs})


class RDGRuntimeTests(unittest.TestCase):
    def test_disabled_and_schedule_skip_all_history_work(self):
        trainer = Trainer()
        trainer._configure_rdg_fusion()
        self.assertIsNone(trainer.rdg_fusion_head)
        self.assertIsNone(trainer.rdg_history(inputs(10)[0]))
        self.assertNotIn("_rdg_fusion", trainer.state_dict())
        trainer.configure()
        for step in (4999, 5000, 5001, 5999):
            trainer.step = step
            self.assertEqual(trainer.rdg_strength(), 0.)
            self.assertIsNone(trainer.rdg_history(inputs(10)[0]))
        self.assertEqual(trainer.dataset.full_image_set.reads, [])
        trainer.step = 5008
        self.assertAlmostEqual(trainer.rdg_strength(), .008)
        trainer.eval()
        trainer.step = 5501
        self.assertAlmostEqual(trainer.rdg_strength(), .501)
        trainer.step = 4999
        self.assertEqual(trainer.rdg_strength(), 0.)

    def test_one_nearest_train_history_and_actual_previous_inference_timestamp(self):
        trainer = Trainer()
        trainer.configure()
        previous = trainer.rdg_history(inputs(11)[0])
        self.assertEqual(previous["image_id"], 7)
        self.assertEqual(previous["stream_id"], ("raw", 5))
        self.assertEqual(tuple(previous["albedo"].shape), (6, 7, 3))
        self.assertEqual(trainer.dataset.full_image_set.reads, [(7, 4, ("raw",))])
        self.assertIsNone(trainer.rdg_history(inputs(1)[0]))
        trainer.eval()
        previous = trainer.rdg_history(inputs(11)[0])
        self.assertEqual(previous["image_id"], 9)
        self.assertEqual(previous["frame_id"], 4)
        self.assertEqual(len(trainer.dataset.full_image_set.reads), 2)
        # Independent random access regenerates raw t-1; there is no history cache.
        trainer.rdg_history(inputs(11)[0])
        self.assertEqual(len(trainer.dataset.full_image_set.reads), 3)

    def test_train_split_gap_and_recursive_context_are_bypassed(self):
        trainer = Trainer()
        trainer.configure(max_gap=1)
        self.assertIsNone(trainer.rdg_history(inputs(10)[0]))
        with self.assertRaisesRegex(ValueError, "train split"):
            trainer.rdg_history(inputs(8)[0])
        trainer._rdg_context = True
        self.assertIsNone(trainer.rdg_history(inputs(10)[0]))
        self.assertEqual(trainer.dataset.full_image_set.reads, [])

    def test_generated_only_inputs_detachment_state_and_rng(self):
        trainer = Trainer()
        rng = torch.get_rng_state().clone()
        trainer.configure()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        py_rng, np_rng = random.getstate(), np.random.get_state()
        old_info, old_labels = trainer.info, trainer.pts_labels
        old_cache = trainer.models["Background"]._gs_cache
        old_penalty = trainer.models["Background"]._raw_normal_residual_penalty
        packet = trainer.rdg_history(inputs(10)[0])
        self.assertEqual(trainer.seen[0][1:], (False, True))
        self.assertFalse({"pixels", "gbuf_basecolor", "dynamic_masks"} & trainer.seen[0][0])
        self.assertTrue(torch.isfinite(packet["albedo"]).all())
        self.assertFalse(any(value.requires_grad for value in packet.values() if torch.is_tensor(value)))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(random.getstate(), py_rng)
        np.testing.assert_equal(np.random.get_state(), np_rng)
        self.assertIs(trainer.info, old_info)
        self.assertIs(trainer.pts_labels, old_labels)
        self.assertIs(trainer.models["Background"]._gs_cache, old_cache)
        self.assertIs(trainer.models["Background"]._raw_normal_residual_penalty, old_penalty)
        self.assertEqual(trainer.cur_frame, 91)
        self.assertFalse(hasattr(trainer, "_rdg_context"))
        self.assertFalse(hasattr(trainer, "_render_c2w"))
        self.assertEqual(trainer.dataset.pixel_source._old_downscale_factor, [.75])
        self.assertEqual(trainer.dataset.pixel_source.camera_data[2].downscale_factor, 1.)

    def test_failed_loading_or_rendering_restores_flags_rng_and_downscale(self):
        for failure in ("loader_exception", "forward_exception"):
            with self.subTest(failure=failure):
                trainer = Trainer()
                trainer.configure()
                trainer._rdg_context = False
                trainer._suppress_eval_layers = False
                trainer.corrupt = failure
                trainer.dataset.full_image_set.corrupt = failure
                rng = torch.get_rng_state().clone()
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    trainer.rdg_history(inputs(10)[0])
                self.assertFalse(trainer._rdg_context)
                self.assertFalse(trainer._suppress_eval_layers)
                self.assertEqual(trainer.cur_frame, 91)
                self.assertEqual(trainer.dataset.pixel_source._downscale_factor, 1.)
                self.assertEqual(trainer.dataset.pixel_source._old_downscale_factor, [.75])
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_wrong_source_or_packet_identity_and_resolution_are_rejected(self):
        for corruption in ("frame", "time", "camera", "size", "packet"):
            with self.subTest(corruption=corruption):
                trainer = Trainer()
                trainer.configure()
                trainer.corrupt = corruption
                trainer.dataset.full_image_set.corrupt = corruption
                with self.assertRaises(ValueError):
                    trainer.rdg_history(inputs(10)[0])
        trainer = Trainer()
        trainer.configure()
        image, _ = inputs(10)
        image["frame_idx"].fill_(99)
        with self.assertRaisesRegex(ValueError, "frame/image IDs"):
            trainer.rdg_history(image)
        image, _ = inputs(10)
        image["normed_time"].fill_(.123)
        with self.assertRaisesRegex(ValueError, "normalized time/frame ID mismatch"):
            trainer.rdg_history(image)

    def test_optimizer_registers_head_and_clips_before_step(self):
        trainer = Trainer()
        trainer.configure(lr=.002, weight_decay=.03, grad_clip=.2)
        trainer.initialize_optimizer()
        head_group = next(group for group in trainer.optimizer.param_groups if group["name"] == "RDGFusion#all")
        self.assertEqual(head_group["lr"], .002)
        self.assertEqual(head_group["weight_decay"], .03)
        self.assertEqual({id(value) for value in head_group["params"]},
                         {id(value) for value in trainer.rdg_fusion_head.parameters()})
        for parameter in trainer.rdg_fusion_head.parameters():
            parameter.grad = torch.ones_like(parameter)
        trainer.optimizer_step()
        self.assertLessEqual(float(trainer.clipped_norm), .20001)

    def test_checkpoint_restores_existing_head_and_requires_matching_metadata(self):
        trainer = Trainer()
        trainer.configure()
        saved = copy.deepcopy(trainer.state_dict())
        self.assertEqual(saved["_rdg_fusion"]["rdg_step"], 6000)
        clone = Trainer()
        clone.configure()
        clone.step = 0
        head_identity = id(clone.rdg_fusion_head)
        clone.load_state_dict(saved)
        self.assertEqual(id(clone.rdg_fusion_head), head_identity)
        self.assertEqual(clone.step, 6000)
        for key, value in trainer.rdg_fusion_head.state_dict().items():
            torch.testing.assert_close(value, clone.rdg_fusion_head.state_dict()[key])
        disabled = Trainer()
        disabled._configure_rdg_fusion()
        with self.assertRaisesRegex(ValueError, "saved trainer.rdg_fusion"):
            disabled.load_state_dict(saved, strict=False)
        clone.configure(width=8)
        with self.assertRaisesRegex(ValueError, "checkpoint/config mismatch"):
            clone.load_state_dict(saved)
        clone.configure()
        for corruption in ("step", "missing", "nan", "metadata"):
            broken = copy.deepcopy(saved)
            first = next(key for key in broken if key.startswith("rdg_fusion_head."))
            if corruption == "step":
                broken["_rdg_fusion"]["rdg_step"] = 13
            elif corruption == "missing":
                del broken[first]
            elif corruption == "nan":
                broken[first].fill_(float("nan"))
            else:
                del broken["_rdg_fusion"]
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                clone.load_state_dict(broken, strict=False)

    def test_losses_are_renamed_without_double_weighting_and_conflicts_rejected(self):
        trainer = Trainer()
        trainer.configure()
        anchor, temporal = torch.tensor(.125), torch.tensor(.25)
        self.assertEqual(trainer.rdg_losses({"_rdg_loss_anchor": anchor, "_rdg_loss_temporal": temporal}),
                         {"rdg_anchor": anchor, "rdg_temporal": temporal})
        trainer.eval()
        self.assertEqual(trainer.rdg_losses({"_rdg_loss_anchor": anchor}), {})
        for attr, value in (("gbuffer_attention_config", {"variant": "A2"}),
                            ("_temporal_material", {}), ("_layered_gbuffer", {}),
                            ("rgb_joint_config", {"enabled": True}),
                            ("teacher_calibration_config", {"enabled": True}),
                            ("teacher_confidence_config", {"enabled": True}),
                            ("_diffreg_config", {"enabled": True}),
                            ("losses_dict", {"temporal_material": {"w": .01}})):
            with self.subTest(conflict=attr):
                trainer = Trainer()
                setattr(trainer, attr, value)
                with self.assertRaisesRegex(ValueError, "without other experiments"):
                    trainer.configure()
        trainer = Trainer()
        trainer.configure()
        trainer.losses_dict = {"temporal_material": {"w": .01}}
        with self.assertRaisesRegex(ValueError, "without other experiments"):
            trainer.assert_rdg_compatible()


if __name__ == "__main__":
    unittest.main()
