"""Synthetic cache preprocessing and checkpoint tests; no real scene jobs."""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.gbuffer_attention_runtime import RENDER_IMAGE_KEYS
from models.teacher_confidence import TeacherConfidenceMoments
from models.teacher_confidence_runtime import TeacherConfidenceMixin, validate_teacher_confidence_state


class StateBase(nn.Module):
    def state_dict(self, only_model=True):
        return {**super().state_dict(), "step": self.step}

    def load_state_dict(self, state, load_only_model=True, strict=True):
        state = dict(state)
        self.step = state.pop("step")
        return super().load_state_dict(state, strict=strict)


class CacheTrainer(TeacherConfidenceMixin, StateBase):
    def __init__(self, config=None, crop_frac=0., inference_crop=False, train_indices=(0, 2, 3)):
        super().__init__()
        self.device = torch.device("cpu")
        self.value = nn.Parameter(torch.tensor(.4))
        self.step = 3
        self.crop_frac = 0. if inference_crop else crop_frac
        self.models = {"Background": SimpleNamespace(_gs_cache="original")}
        self.cur_frame = "original"
        self.info = {"main": "original"}
        self._render_c2w = torch.eye(4)
        self._render_intrinsics = torch.eye(3)
        self.loaded, self.rendered = [], []
        self.fail_index = None
        self.wrong_index = None
        self.teacher_height = round(32 * (1 - 2 * crop_frac))
        pixel = SimpleNamespace(
            num_cams=1, _downscale_factor=1., _old_downscale_factor=[.5],
            camera_data={0: SimpleNamespace(downscale_factor=1., gbuffer_crop_frac=crop_frac)})
        self.dataset = SimpleNamespace(train_indices=list(train_indices), pixel_source=pixel)
        self.dataset.train_image_set = SimpleNamespace(get_image=self.get_image)
        self.configure_teacher_confidence({"enabled": True, "build_step": 3, "ramp_steps": 2,
                                           **(config or {})})

    def image(self, index):
        h, w, gh, gw = 8, 12, self.teacher_height, 48
        return {
            "img_idx": torch.full((h, w), index, dtype=torch.long),
            "frame_idx": torch.full((h, w), index, dtype=torch.long),
            "viewdirs": torch.tensor([0., 0., 1.]).expand(h, w, 3),
            "sky_masks": torch.zeros(h, w), "dynamic_masks": torch.zeros(h, w),
            "egocar_masks": torch.zeros(h, w), "pixels": torch.full((h, w, 3), .8),
            "gbuf_basecolor": torch.full((gh, gw, 3), .4 + index * .02),
            "gbuf_roughness": torch.full((gh, gw, 1), .4),
            "gbuf_metallic": torch.full((gh, gw, 1), .1),
        }

    def get_image(self, position, downscale, weathers):
        index = self.dataset.train_indices[position]
        self.loaded.append((index, downscale, weathers))
        random.random()
        np.random.rand()
        torch.rand(1)
        pixel = self.dataset.pixel_source
        pixel._downscale_factor = .25
        pixel._old_downscale_factor.append(1.)
        pixel.camera_data[0].downscale_factor = .25
        if index == self.fail_index:
            raise RuntimeError("synthetic load failure")
        image = self.image(self.wrong_index if self.wrong_index is not None else index)
        camera = {"camera_to_world": torch.eye(4),
                  "intrinsics": torch.tensor([[10., 0., 6.], [0., 10., 4.], [0., 0., 1.]])}
        return {"raw": image}, {"raw": camera}

    def forward(self, images, cameras, weather):
        index = int(images["img_idx"][0, 0])
        self.rendered.append((index, set(images), torch.is_grad_enabled()))
        torch.rand(2)
        self.cur_frame, self.info = index, {"cache_frame": index}
        self.models["Background"]._gs_cache = {"cache_frame": index}
        self._render_c2w = cameras["camera_to_world"]
        self._render_intrinsics = cameras["intrinsics"]
        return {"depth": torch.full((8, 12, 1), 2.), "opacity": torch.ones(8, 12, 1),
                "normal": torch.tensor([0., 0., 1.]).expand(8, 12, 3)}

    def _crop_and_resize(self, value, target_hw):
        top = round(value.shape[0] * self.crop_frac)
        return self._resize_down(value[top:value.shape[0] - top], target_hw)

    @staticmethod
    def _resize_down(value, target_hw):
        if value.shape[:2] == target_hw:
            return value
        return F.interpolate(value.permute(2, 0, 1)[None].float(), size=target_hw,
                             mode="bilinear", antialias=True, align_corners=False)[0].permute(1, 2, 0)


class TeacherConfidenceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def build(self, trainer=None):
        trainer = trainer or CacheTrainer()
        trainer.step = 3
        trainer.prepare_teacher_confidence(3)
        return trainer

    def validate(self, trainer, saved, step=3):
        return validate_teacher_confidence_state(
            saved, config=trainer.teacher_confidence_config,
            train_indices=trainer.dataset.train_indices, num_cams=1,
            crop_frac=trainer._teacher_confidence_crop_frac, checkpoint_step=step)

    def test_build_uses_only_train_positions_and_only_renderer_inputs(self):
        trainer = self.build()
        self.assertEqual([call[0] for call in trainer.loaded], [0, 2, 3])
        self.assertTrue(all(call[1:] == (4, ["raw"]) for call in trainer.loaded))
        self.assertTrue(all(keys <= RENDER_IMAGE_KEYS and not grad for _, keys, grad in trainer.rendered))
        cache = trainer._teacher_confidence_cache
        self.assertEqual(cache["built_step"], 3)
        self.assertEqual(set(cache["frames"]), {0, 2, 3})
        self.assertEqual(cache["stats"]["render_count"], 3)
        self.assertEqual(cache["stats"]["cache_bytes"], 3 * 8 * 12 * 2)
        for value in cache["frames"].values():
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.dtype, torch.float16)
            self.assertFalse(value.requires_grad)
        trainer.prepare_teacher_confidence(4)
        self.assertEqual(len(trainer.loaded), 3)

    def test_disabled_warmup_and_eval_do_not_construct_or_apply_cache(self):
        for trainer, step in ((CacheTrainer({"enabled": False}), 3),
                              (CacheTrainer(), 2), (CacheTrainer().eval(), 3)):
            trainer.step = step
            trainer.prepare_teacher_confidence(step)
            self.assertFalse(trainer.loaded)
            self.assertIsNone(trainer.teacher_confidence_weights({}, (8, 12), torch.ones(8, 12, 1)))

    def test_missing_boundary_cache_fails_instead_of_using_later_geometry(self):
        trainer = CacheTrainer()
        with self.assertRaisesRegex(ValueError, "restart this arm"):
            trainer.prepare_teacher_confidence(4)
        trainer.step = 5
        with self.assertRaisesRegex(ValueError, "prepare_teacher_confidence"):
            trainer.teacher_confidence_weights({}, (8, 12), torch.ones(8, 12, 1))

    def test_late_model_cannot_be_relabelled_and_camera_interpretation_cannot_change(self):
        trainer = CacheTrainer()
        trainer.step = 4
        with self.assertRaisesRegex(ValueError, "current model step"):
            trainer.prepare_teacher_confidence(3)
        trainer.step = 3
        trainer.dataset.pixel_source.num_cams = 2
        with self.assertRaisesRegex(ValueError, "training split changed"):
            trainer.prepare_teacher_confidence(3)
        self.assertFalse(trainer.loaded)
        self.assertIsNone(trainer._teacher_confidence_cache)

    def test_success_and_exception_restore_rng_node_camera_and_downscale(self):
        for fail in (False, True):
            trainer = CacheTrainer()
            original = (trainer.cur_frame, trainer.info, trainer._render_c2w,
                        trainer._render_intrinsics, trainer.models["Background"]._gs_cache)
            random.seed(13)
            np.random.seed(13)
            torch.manual_seed(13)
            states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            trainer.fail_index = 2 if fail else None
            if fail:
                with self.assertRaisesRegex(RuntimeError, "synthetic load"):
                    trainer.prepare_teacher_confidence(3)
                self.assertIsNone(trainer._teacher_confidence_cache)
            else:
                trainer.prepare_teacher_confidence(3)
            for actual, before in zip((trainer.cur_frame, trainer.info, trainer._render_c2w,
                                       trainer._render_intrinsics, trainer.models["Background"]._gs_cache), original):
                self.assertIs(actual, before)
            pixel = trainer.dataset.pixel_source
            self.assertEqual(pixel._downscale_factor, 1.)
            self.assertEqual(pixel._old_downscale_factor, [.5])
            self.assertEqual(pixel.camera_data[0].downscale_factor, 1.)
            self.assertFalse(hasattr(trainer, "_suppress_eval_layers"))
            self.assertEqual(random.getstate(), states[0])
            np.testing.assert_equal(np.random.get_state(), states[1])
            torch.testing.assert_close(torch.get_rng_state(), states[2], atol=0, rtol=0)

    def test_crop_and_resizing_adjust_camera_intrinsics_in_teacher_coordinates(self):
        trainer = CacheTrainer(crop_frac=.25)
        trainer.teacher_height = 20  # Exercises unequal horizontal/vertical resize.
        captured = []
        update = TeacherConfidenceMoments.update

        def capture(tracker, packet, frame_id, camera_id):
            captured.append(packet)
            return update(tracker, packet, frame_id, camera_id)

        with patch.object(TeacherConfidenceMoments, "update", capture):
            self.build(trainer)
        expected = torch.tensor([[10., 0., 6.], [0., 12.5, 2.5], [0., 0., 1.]])
        torch.testing.assert_close(captured[0]["intrinsics"], expected)
        self.assertEqual(captured[0]["depth"].shape, (5, 12, 1))
        self.assertEqual(captured[0]["gbuffer"]["albedo"].shape, (5, 12, 3))

    def test_budget_and_wrong_frame_failure_never_publish_partial_cache(self):
        trainer = CacheTrainer({"max_cache_mb": .0002})
        with self.assertRaisesRegex(ValueError, "max_cache_mb"):
            trainer.prepare_teacher_confidence(3)
        self.assertIsNone(trainer._teacher_confidence_cache)
        trainer = CacheTrainer()
        trainer.wrong_index = 1
        with self.assertRaisesRegex(ValueError, "different training image"):
            trainer.prepare_teacher_confidence(3)
        self.assertIsNone(trainer._teacher_confidence_cache)

    def test_weights_are_detached_normalized_and_dynamic_values_stay_one(self):
        trainer = self.build()
        trainer.step = 5
        trainer._teacher_confidence_cache["frames"][2][:, :6] = .25
        image = trainer.image(2)
        image["dynamic_masks"][:, 8:] = 1
        surface = torch.ones(8, 12, 1, requires_grad=True)
        weights = trainer.teacher_confidence_weights(image, (8, 12), surface)
        self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights[:, 8:], torch.ones_like(weights[:, 8:]), atol=0, rtol=0)
        self.assertAlmostEqual(float(weights[:, :8].mean()), 1., places=6)
        self.assertLess(float(weights[:, :6].mean()), float(weights[:, 6:8].mean()))
        self.assertIn("render_count", trainer._last_teacher_confidence_stats)
        image["img_idx"].fill_(1)
        with self.assertRaisesRegex(ValueError, "held-out"):
            trainer.teacher_confidence_weights(image, (8, 12), surface)

    def test_checkpoint_roundtrip_restores_cache_without_loading_or_rendering(self):
        trainer = self.build()
        trainer.step = 5
        state = trainer.state_dict()
        clone = CacheTrainer().eval()
        clone.load_state_dict(state)
        self.assertFalse(clone.loaded)
        self.assertFalse(clone.rendered)
        for index in trainer._teacher_confidence_cache["frames"]:
            torch.testing.assert_close(clone._teacher_confidence_cache["frames"][index],
                                       trainer._teacher_confidence_cache["frames"][index], atol=0, rtol=0)
        state["_teacher_confidence"]["frames"][0].fill_(.25)
        self.assertTrue(bool((clone._teacher_confidence_cache["frames"][0] == 1).all()))
        baseline = CacheTrainer({"enabled": False})
        self.assertNotIn("_teacher_confidence", baseline.state_dict())
        baseline.load_state_dict(baseline.state_dict())

    def test_nonzero_crop_checkpoint_loads_when_eval_skips_teacher_loading(self):
        trainer = self.build(CacheTrainer(crop_frac=.25))
        state = trainer.state_dict()
        clone = CacheTrainer(crop_frac=.25, inference_crop=True).eval()
        self.assertEqual(clone.crop_frac, 0.)
        self.assertEqual(clone._teacher_confidence_crop_frac, .25)
        clone.load_state_dict(state)
        self.assertFalse(clone.loaded)
        self.assertEqual(clone._teacher_confidence_cache["crop_frac"], .25)

    def test_unbuilt_checkpoint_allowed_only_before_boundary_and_metadata_strict(self):
        trainer = CacheTrainer()
        trainer.step = 0
        early = trainer.state_dict()
        clone = CacheTrainer()
        clone.load_state_dict(early)
        self.assertIsNone(clone._teacher_confidence_cache)
        late = copy.deepcopy(early)
        late["step"] = 3
        with self.assertRaisesRegex(ValueError, "missing its warmup"):
            clone.load_state_dict(late)
        state = self.build().state_dict()
        for change in ("missing", "split", "shape", "nan", "range", "step", "budget", "bool_id"):
            altered = copy.deepcopy(state["_teacher_confidence"])
            if change == "missing":
                altered["frames"].pop(2)
            elif change == "split":
                altered["train_indices"] = [0, 1, 3]
            elif change == "shape":
                altered["frames"][0] = torch.ones(1, dtype=torch.float16)
            elif change == "nan":
                altered["frames"][0][0, 0] = float("nan")
            elif change == "range":
                altered["frames"][0][0, 0] = .1
            elif change == "step":
                altered["built_step"] = 4
            elif change == "budget":
                altered["stats"]["cache_bytes"] += 1
            else:
                altered["frames"][False] = altered["frames"].pop(0)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(trainer, altered)
        small_floor = self.build(CacheTrainer({"confidence_floor": .0001}))
        altered = small_floor.state_dict()["_teacher_confidence"]
        altered["frames"][0][0, 0] = 0.
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.validate(small_floor, altered)


if __name__ == "__main__":
    unittest.main()
