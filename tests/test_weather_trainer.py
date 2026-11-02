"""Paired weather supervision, frozen stage selection, and CUDA integration."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.trainers.scene_graph_pbr_weather import (
    MultiTrainer_pbr_weather, _bounded_logit, _erode_binary_mask,
    _masked_charbonnier, _roughness_logit,
)


def weather_config(stage="material", weather="rain"):
    return OmegaConf.create({
        "enabled": True, "weather": weather, "stage": stage,
        "material_classes": ["Background"], "weather_strength": 1.,
        "material_lr": 1e-4, "environment_lr": 1e-3,
        "logit_eps": 1e-4, "charbonnier_eps": 1e-3,
        "paired_target_mode": "gt_relative",
        "target_albedo_delta_cap": 4., "target_roughness_delta_cap": 5.,
        "material_decoder": {"hidden_dim": 64, "num_hidden_layers": 2,
                             "detach_inputs": True, "world_up": [0, 0, 1]},
        "mask": {"exclude_dynamic": True, "min_opacity": .5,
                 "normal_cos_min": .5, "base_albedo_error_max": .25,
                 "base_roughness_error_max": .35, "erode_px": 1,
                 "min_valid_pixels": 64},
        "losses": {name: {"w": weight} for name, weight in {
            "paired_albedo": 1., "paired_roughness": .5,
            "absolute_albedo": .1, "absolute_roughness": .05,
            "delta_albedo_reg": .001, "delta_roughness_reg": .001,
            "rgb_surface": .8, "rgb_sky": .1, "ssim": .02,
        }.items()},
    })


def bare_trainer(stage="material"):
    trainer = object.__new__(MultiTrainer_pbr_weather)
    torch.nn.Module.__init__(trainer)
    trainer.weather_training_cfg = weather_config(stage)
    trainer.weather_cfg = trainer.weather_training_cfg
    trainer.weather_stage = stage
    trainer.weather_name = "rain"
    trainer.material_classes = ["Background"]
    trainer.crop_frac = 0.
    trainer.device = torch.device("cpu")
    trainer.models = {}
    return trainer


def paired_maps(size=12):
    a = torch.full((size, size, 3), .5)
    r = torch.full((size, size, 1), .505)
    info = {
        "sky_masks": torch.zeros(size, size),
        "egocar_masks": torch.zeros(size, size),
        "dynamic_masks": torch.zeros(size, size),
        "gbuf_basecolor": a.clone(), "gbuf_roughness": r.clone(),
        "gbuf_normal": torch.tensor([0., 0., 1.]).expand(size, size, 3).clone(),
        "pixels": a.clone(),
    }
    outputs = {"albedo": a, "roughness": r, "opacity": torch.ones(size, size, 1)}
    return outputs, info


class WeatherLossTests(unittest.TestCase):
    def test_robust_channel_reduction_zero_and_masked_gradient(self):
        residual = torch.tensor([[[1., 2., 3.], [8., 8., 8.]]], requires_grad=True)
        mask = torch.tensor([[[1.], [0.]]])
        value = _masked_charbonnier(residual, mask, .001)
        expected = (torch.sqrt(residual[0, 0].square() + 1e-6) - .001).mean()
        self.assertTrue(torch.allclose(value, expected))
        value.backward()
        self.assertTrue(torch.equal(residual.grad[0, 1], torch.zeros(3)))
        self.assertEqual(_masked_charbonnier(torch.zeros_like(residual), mask, .001).item(), 0.)
        self.assertTrue(torch.isfinite(_bounded_logit(torch.tensor([0., 1.]), 1e-4)).all())

    def test_confidence_mask_filters_each_condition_and_erosion(self):
        trainer = bare_trainer()
        trainer.weather_cfg.mask.erode_px = 0
        raw, info = paired_maps()
        weather = copy.deepcopy(info)
        info["sky_masks"][1, 1] = 1
        info["egocar_masks"][2, 2] = 1
        info["dynamic_masks"][3, 3] = 1
        raw["opacity"][4, 4] = .49
        weather["gbuf_normal"][5, 5] = torch.tensor([1., 0., 0.])
        raw["albedo"][6, 6] = .76
        raw["roughness"][7, 7] = .9
        mask = trainer._weather_material_mask(raw, info, weather, (12, 12))
        self.assertEqual(mask.sum().item(), 144 - 7)
        self.assertTrue(torch.equal(mask[1:8, 1:8, 0].diagonal(), torch.zeros(7)))
        raw["opacity"][8, 8] = .5
        self.assertEqual(trainer._weather_material_mask(raw, info, weather, (12, 12))[8, 8].item(), 1.)
        one_hole = torch.ones(9, 9, 1)
        one_hole[4, 4] = 0
        eroded = _erode_binary_mask(one_hole, 1)
        self.assertEqual(eroded.sum().item(), 81 - 9)
        self.assertEqual(eroded[0, 0].item(), 1.)

    def test_gt_relative_logit_difference_preserves_base_error(self):
        trainer = bare_trainer()
        raw, info = paired_maps()
        weather = copy.deepcopy(info)
        raw["albedo"].fill_(.3)
        raw["roughness"].fill_(.406)
        weather["gbuf_basecolor"].fill_(.7)
        weather["gbuf_roughness"].fill_(.703)
        delta_a = _bounded_logit(weather["gbuf_basecolor"], 1e-4) - _bounded_logit(info["gbuf_basecolor"], 1e-4)
        delta_r = _roughness_logit(weather["gbuf_roughness"], 1e-4) - _roughness_logit(info["gbuf_roughness"], 1e-4)
        prediction = {
            "albedo": torch.sigmoid(_bounded_logit(raw["albedo"], 1e-4) + delta_a).requires_grad_(),
            "roughness": (.99 * torch.sigmoid(_roughness_logit(raw["roughness"], 1e-4) + delta_r) + .01).requires_grad_(),
        }
        losses = trainer._material_losses(prediction, raw, weather, info)
        self.assertLess(losses["weather_paired_albedo_loss"].item(), 1e-8)
        self.assertLess(losses["weather_paired_roughness_loss"].item(), 1e-8)
        self.assertGreater(losses["weather_absolute_albedo_loss"].item(), .01)
        trainer.weather_cfg.paired_target_mode = "reconstruction"
        changed = trainer._material_losses(prediction, raw, weather, info)
        self.assertGreater(changed["weather_paired_albedo_loss"].item(), .5)

    def test_target_caps_and_minimum_valid_pixels(self):
        trainer = bare_trainer()
        raw, info = paired_maps()
        weather = copy.deepcopy(info)
        weather["gbuf_basecolor"].fill_(1.)
        weather["gbuf_roughness"].fill_(1.)
        pred = {"albedo": torch.sigmoid(torch.full((12, 12, 3), 4.)).requires_grad_(),
                "roughness": (.99 * torch.sigmoid(torch.full((12, 12, 1), 5.)) + .01).requires_grad_()}
        losses = trainer._material_losses(pred, raw, weather, info)
        self.assertLess(losses["weather_paired_albedo_loss"].item(), 1e-8)
        self.assertLess(losses["weather_paired_roughness_loss"].item(), 1e-7)
        info["dynamic_masks"].fill_(1.)
        with self.assertRaisesRegex(RuntimeError, "too few valid pixels"):
            trainer._material_losses(pred, raw, weather, info)

    def test_rgb_surface_sky_and_ssim_contract(self):
        trainer = bare_trainer("environment")
        _, info = paired_maps()
        info["pixels"].zero_()
        info["sky_masks"][:4] = 1.
        info["dynamic_masks"][4:6] = 1.
        info["egocar_masks"][0] = 1.
        pred = torch.ones_like(info["pixels"], requires_grad=True)
        class SSIMCapture(torch.nn.Module):
            def forward(self, gt, output):
                self.seen = (gt.detach().clone(), output.detach().clone())
                return output.mean() * 0 + .75
        trainer.ssim = SSIMCapture()
        losses = trainer._rgb_losses({"rgb": pred}, info)
        self.assertAlmostEqual(losses["weather_rgb_surface_loss"].item(), .8 * ((1 + 1e-6)**.5 - .001), places=6)
        self.assertAlmostEqual(losses["weather_rgb_sky_loss"].item(), .1 * ((1 + 1e-6)**.5 - .001), places=6)
        self.assertAlmostEqual(losses["weather_ssim_loss"].item(), .005, places=7)
        sum(losses.values()).backward()
        self.assertEqual(pred.grad[0].abs().sum().item(), 0.)
        self.assertEqual(pred.grad[4:6].abs().sum().item(), 0.)
        self.assertEqual(trainer.ssim.seen[1][0, :, 4:6].min().item(), 1.)


class StageSelectionTests(unittest.TestCase):
    def test_preload_inspection_with_uninitialized_rigid_and_smpl_nodes(self):
        from models.nodes import RigidNodes_feature, SMPLNodes_feature
        from models.trainers.scene_graph_pbr import MultiTrainer_pbr
        from tests.test_base_v3_material import make_model
        trainer = bare_trainer()
        trainer.models = {
            "Background": torch.nn.Linear(1, 1), "EnvLight": torch.nn.Linear(1, 1),
            "RigidNodes": make_model("RigidNodes", RigidNodes_feature),
            "SMPLNodes": make_model("SMPLNodes", SMPLNodes_feature),
        }
        trainer.gaussian_classes = {"Background": 0, "RigidNodes": 1, "SMPLNodes": 2}
        for name in ("RigidNodes", "SMPLNodes"):
            with self.assertRaises(AttributeError):
                trainer.models[name].state_dict()  # point_ids only exists after loading.
        checkpoint = {"step": 30000, "models": {
            name: torch.nn.Module.state_dict(model) for name, model in trainer.models.items()
        }}
        checkpoint["models"]["RigidNodes"]["points_ids"] = torch.zeros(2, 1, dtype=torch.long)
        checkpoint["weather_optimizer_state"] = {"version": 1}
        def load_parent(state, **kwargs):
            self.assertNotIn("weather_optimizer_state", state)
            # Simulate the existing node loader consuming custom metadata.
            state["models"]["RigidNodes"].pop("points_ids")
        with patch.object(MultiTrainer_pbr, "load_state_dict", side_effect=load_parent) as loader, \
                patch.object(trainer, "configure_weather_stage"):
            trainer.load_state_dict(checkpoint)
            self.assertEqual(trainer.base_model_step, 30000)
            checkpoint["weather_training_state"] = {
                "version": 1, "weather": "rain", "base_model_step": 30000,
            }
            trainer.load_state_dict(checkpoint)
            self.assertEqual(loader.call_count, 2)
        self.assertIn("points_ids", checkpoint["models"]["RigidNodes"])

    def test_background_weather_and_dynamic_raw_collection(self):
        class Node:
            def __init__(self, required_weather):
                self.required_weather = required_weather
            def get_gaussians(self, camera, weather):
                if weather != self.required_weather:
                    raise AssertionError(f"Expected {self.required_weather}, got {weather}")
                return {"_means": torch.zeros(1, 3), "_scales": torch.ones(1, 3),
                        "_quats": torch.tensor([[1., 0., 0., 0.]]),
                        "_rgbs": torch.zeros(1, 8), "_opacities": torch.ones(1, 1)}
        trainer = bare_trainer()
        trainer.gaussian_classes = {"Background": 0, "RigidNodes": 1, "SMPLNodes": 2,
                                    "DeformableNodes": 3}
        trainer.render_dynamic_mask = False
        trainer.models = {name: Node("rain" if name == "Background" else "raw")
                          for name in trainer.gaussian_classes}
        result = trainer.collect_gaussians(None, "rain", torch.tensor(0))
        self.assertEqual(result.means.shape, (4, 3))

    def test_optimizer_freezes_base_and_refinement_is_never_called(self):
        class Gaussian(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.ones(3))
                self.residual = torch.nn.Parameter(torch.ones(4))
                self.step = 30000
            def get_weather_material_parameters(self, weather):
                return [self.residual]
            def preprocess_per_train_step(self, step):
                raise AssertionError("Frozen base preprocessing must not run")
            def postprocess_per_train_step(self, step, *args):
                raise AssertionError("Frozen base densification must not run")
        class Env(Gaussian):
            def get_weather_parameters(self, weather):
                return [self.residual]
        for stage in ("material", "environment", "joint"):
            with self.subTest(stage=stage):
                trainer = bare_trainer(stage)
                trainer.models = {"Background": Gaussian(), "EnvLight": Env(),
                                  "CameraResponse": torch.nn.Linear(1, 1),
                                  "Affine": torch.nn.Linear(1, 1)}
                trainer.gaussian_classes = {"Background": 0}
                trainer.base_model_step = 30000
                trainer.viewer = None
                trainer.optim_general = {"use_grad_scaler": False}
                trainer.initialize_optimizer()
                self.assertEqual(trainer.models["Background"].residual.requires_grad, stage != "environment")
                self.assertEqual(trainer.models["EnvLight"].residual.requires_grad, stage != "material")
                self.assertTrue(all(not p.requires_grad for p in trainer.models["CameraResponse"].parameters()))
                self.assertFalse(trainer.models["Background"].base.requires_grad)
                self.assertFalse(trainer.models["EnvLight"].base.requires_grad)
                before = trainer.models["Background"].base.detach().clone()
                trainer.optimizer.zero_grad()
                sum(p.sum() for group in trainer.optimizer.param_groups for p in group["params"]).backward()
                trainer.optimizer.step()
                self.assertTrue(torch.equal(before, trainer.models["Background"].base))
                trainer.preprocess_per_train_step(1)
                trainer.postprocess_per_train_step(1)
                self.assertEqual(trainer.models["Background"].step, 30000)
                self.assertEqual(trainer.step, 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for real PBR rendering")
class WeatherCUDAIntegrationTests(unittest.TestCase):
    def test_three_stage_gradients_frozen_base_and_checkpoint_round_trip(self):
        from tools.check_base_v3_reconstruction import make_trainer, initialize_background, synthetic_inputs
        from utils.config import load_training_config
        cfg = load_training_config("configs/base_v4/scene788_30f.yaml")
        base = make_trainer(cfg)
        initialize_background(base)
        base.step = 30000
        checkpoint = copy.deepcopy(base.state_dict())
        raw, cam = synthetic_inputs()
        raw["sky_masks"].zero_()
        raw["dynamic_masks"] = torch.zeros_like(raw["sky_masks"])
        base.set_eval()
        with torch.no_grad():
            reference = base(raw, cam, "raw")
        cfg.trainer.type = "models.trainers.scene_graph_pbr_weather.MultiTrainer_pbr_weather"
        cfg.trainer.weather_training = weather_config()
        cfg.trainer.weather_training.mask.min_opacity = 0.
        cfg.model.EnvLight.reg.weather_residual = {"magnitude": {"w": .001}, "tv": {"w": .001}}
        raw["gbuf_basecolor"] = reference["albedo"].clone()
        raw["gbuf_roughness"] = reference["roughness"].clone()
        weather = copy.deepcopy(raw)
        weather["gbuf_basecolor"] = (raw["gbuf_basecolor"] + .06).clamp(0, 1)
        weather["gbuf_roughness"] = (raw["gbuf_roughness"] + .08).clamp(.01, 1)
        weather["pixels"] = (reference["rgb"] + .08).clamp(0, 1)
        for stage in ("material", "environment", "joint"):
            cfg.trainer.weather_training.stage = stage
            trainer = make_trainer(cfg)
            trainer.init_from_base_state(checkpoint)
            self.assertEqual(trainer.step, 0)
            self.assertEqual(trainer.base_model_step, 30000)
            trainer.initialize_optimizer()
            frozen = {f"{name}.{key}": value.detach().clone()
                      for name, model in trainer.models.items()
                      for key, value in model.named_parameters() if not value.requires_grad}
            trainer.set_train()
            trainer.preprocess_per_train_step(1)
            raw_outputs = trainer.render_raw_reference(raw, cam) if trainer.requires_paired_raw else None
            outputs = trainer(weather, cam, "rain")
            losses = trainer.compute_weather_losses(outputs, raw_outputs, weather, raw, cam, "rain")
            self.assertFalse(any("boundary" in name or "vgg" in name for name in losses))
            sum(losses.values()).backward()
            for group in trainer.optimizer.param_groups:
                gradients = [p.grad for p in group["params"] if p.grad is not None]
                self.assertTrue(gradients, group["name"])
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
            trainer.optimizer.step()
            trainer.postprocess_per_train_step(1)
            for name, model in trainer.models.items():
                for key, value in model.named_parameters():
                    if f"{name}.{key}" in frozen:
                        self.assertTrue(torch.equal(frozen[f"{name}.{key}"], value), f"{stage}: {name}.{key}")
            trainer.set_eval()
            trainer.set_render_weather_strengths(0., 0.)
            with torch.no_grad():
                zero = trainer(weather, cam, "rain")
                raw_out = trainer(raw, cam, "raw")
            self.assertTrue(torch.equal(zero["rgb"], raw_out["rgb"]))
            self.assertTrue(torch.allclose(raw_out["rgb"], reference["rgb"], atol=2e-6, rtol=0.))
            trainer.set_render_weather_strengths()
            checkpoint = copy.deepcopy(trainer.state_dict())
            loaded = make_trainer(cfg)
            loaded.load_state_dict(checkpoint)
            self.assertEqual(loaded.step, 1)
            self.assertEqual(loaded.base_model_step, 30000)
            loaded.set_eval()
            with torch.no_grad():
                expected = trainer(weather, cam, "rain")["rgb"]
                actual = loaded(weather, cam, "rain")["rgb"]
            self.assertTrue(torch.allclose(expected, actual, atol=2e-6, rtol=0.),
                            (stage, (expected - actual).abs().max().item()))
            for name, model in loaded.models.items():
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, checkpoint["models"][name][key]),
                                    f"checkpoint changed {name}.{key}")
            corrupt = copy.deepcopy(checkpoint)
            del corrupt["models"]["EnvLight"]["weather_log_residuals.rain"]
            with self.assertRaisesRegex(ValueError, "missing residual weights"):
                loaded.load_state_dict(corrupt)
            wrong = copy.deepcopy(checkpoint)
            wrong["weather_training_state"]["weather"] = "snow"
            with self.assertRaisesRegex(ValueError, "checkpoint/config mismatch"):
                loaded.load_state_dict(wrong)
            missing = copy.deepcopy(checkpoint)
            del missing["models"]["Background"]
            with self.assertRaisesRegex(ValueError, "missing required base models"):
                loaded.load_state_dict(missing)


if __name__ == "__main__":
    unittest.main()
