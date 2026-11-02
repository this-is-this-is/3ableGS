"""CPU checks for held-out isolation and resumable material-only experiments."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from tools.ego_motion_runtime import select_material_parameters
from tools.finetune_ego_material import (
    TrainFrameSampler, anchor_loss, assert_frozen_parameters, atomic_torch_save,
    build_material_anchors, capture_rng, decoded_material, make_parser,
    material_inputs, motion_bins, pose_motion_records, restore_rng,
    temporal_loss, training_pairs, validate_args,
)


class ToyNode(nn.Module):
    def __init__(self, normal=True, spatial=False):
        super().__init__()
        self.pbr_enabled = True
        self._features = nn.Parameter(torch.randn(12, 4))
        self._means = nn.Parameter(torch.randn(12, 3))
        self.raw_spatial_xyz_enabled = spatial
        self.scene_origin = torch.tensor([1., 2., 3.])
        self.scene_scale = 3.
        self.mlp_material = nn.ModuleDict({"raw": nn.Sequential(
            nn.Linear(7 if spatial else 4, 5), nn.Sigmoid(), nn.Linear(5, 5))})
        self.raw_normal_residual = nn.Linear(4, 3) if normal else None


class ToyTrainer(nn.Module):
    def __init__(self, normal=True, spatial=False):
        super().__init__()
        self.register_parameter("exposure", nn.Parameter(torch.ones(1)))
        self.models = {"Background": ToyNode(normal, spatial), "EnvLight": nn.Linear(3, 3)}
        self.gaussian_classes = {"Background": 0}
        self.eval()
        for module in self.models.values():
            module.eval()


class PairAndSamplerTests(unittest.TestCase):
    def test_never_crosses_a_single_frame_holdout(self):
        pairs = training_pairs([0, 1, 3, 4, 5, 8, 9, 10, 11, 12])
        self.assertEqual(pairs[0], ())
        self.assertEqual(pairs[3], ())
        self.assertEqual(pairs[4], (3,))
        self.assertEqual(pairs[5], (4, 3))
        self.assertEqual(pairs[12], (11, 10, 8))
        for current, previous_frames in pairs.items():
            for previous in previous_frames:
                self.assertTrue(set(range(previous, current + 1)) <= set(pairs))

    def test_manifest_unknown_is_low_and_other_labels_unchanged(self):
        records = [{"frame_id": 0, "motion_bin": "unknown", "displacement_p50": None},
                   {"frame_id": 1, "motion_bin": "high", "displacement_p50": 2},
                   {"frame_id": 2, "motion_bin": "medium", "displacement_p50": 100}]
        self.assertEqual(motion_bins([0, 1, 2], records), {0: "low", 1: "high", 2: "medium"})

    def test_quantile_fallback_uses_only_training_frames(self):
        records = [{"frame_id": frame, "displacement_p50": float(frame)} for frame in range(6)]
        records.append({"frame_id": 100, "displacement_p50": 100000.})
        self.assertEqual(motion_bins(list(range(6)), records),
                         {0: "low", 1: "low", 2: "medium", 3: "medium", 4: "high", 5: "high"})
        with self.assertRaises(ValueError):
            motion_bins([0, 7], records)
        with self.assertRaises(ValueError):
            motion_bins([0], records + [records[0]])

    def test_equal_motion_is_not_arbitrarily_promoted(self):
        records = [{"frame_id": frame, "displacement_p50": 0.} for frame in range(5)]
        self.assertEqual(set(motion_bins(range(5), records).values()), {"low"})

    def test_uniform_current_sequence_matches_across_temporal_gaps(self):
        frames = list(range(20)) + list(range(25, 40))
        bins = {frame: "low" for frame in frames}
        continued = TrainFrameSampler(frames, bins, balanced=False, seed=19)
        legacy = TrainFrameSampler(frames, bins, balanced=False, seed=19, gaps=(1,))
        self.assertEqual([continued.sample()[0] for _ in range(100)],
                         [legacy.sample()[0] for _ in range(100)])

    def test_balancing_and_rng_resume(self):
        frames = list(range(30))
        bins = {frame: "low" if frame < 20 else "medium" if frame < 29 else "high" for frame in frames}
        sampler = TrainFrameSampler(frames, bins, balanced=True, seed=5)
        counts = {key: 0 for key in ("low", "medium", "high")}
        for _ in range(3000):
            current, _ = sampler.sample()
            counts[bins[current]] += 1
        self.assertTrue(all(900 < value < 1100 for value in counts.values()), counts)
        state = sampler.state_dict()
        expected = [sampler.sample() for _ in range(100)]
        resumed = TrainFrameSampler(frames, bins, balanced=True, seed=888)
        resumed.load_state_dict(state)
        self.assertEqual(expected, [resumed.sample() for _ in range(100)])
        wrong = TrainFrameSampler(frames, bins, balanced=False, seed=5)
        with self.assertRaises(ValueError):
            wrong.load_state_dict(state)

    def test_pose_proxy_does_not_access_images_or_holdout_pairs(self):
        poses = torch.eye(4).repeat(6, 1, 1)
        poses[:, 0, 3] = torch.tensor([0., 1., 100., 200., 201., 203.])
        runtime = SimpleNamespace(dataset=SimpleNamespace(pixel_source=SimpleNamespace(
            camera_data={0: SimpleNamespace(cam_to_worlds=poses)})))
        records = {row["frame_id"]: row for row in pose_motion_records(runtime, [0, 1, 4, 5])}
        self.assertEqual(records[4]["translation_m"], 0.)
        self.assertEqual(records[1]["translation_m"], 1.)
        self.assertEqual(records[5]["translation_m"], 2.)


class MaterialAndResumeTests(unittest.TestCase):
    def test_freeze_assertion_catches_geometry_and_training_mode(self):
        trainer = ToyTrainer()
        selected, _ = select_material_parameters(trainer)
        assert_frozen_parameters(trainer, selected)
        trainer.models["Background"]._means.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            assert_frozen_parameters(trainer, selected)
        trainer.models["Background"]._means.requires_grad_(False)
        trainer.train()
        with self.assertRaises(RuntimeError):
            assert_frozen_parameters(trainer, selected)

    def test_frozen_normal_survives_optimizer_and_standard_state_reload(self):
        trainer = ToyTrainer(normal=True)
        selected, summary = select_material_parameters(trainer)
        node = trainer.models["Background"]
        before = node.raw_normal_residual(node._features).detach().clone()
        anchors = build_material_anchors(trainer)
        optimizer = torch.optim.Adam([parameter for _, parameter in selected], lr=.1)
        decoded_material(node, slice(None)).sum().backward()
        assert_frozen_parameters(trainer, selected)
        optimizer.step()
        self.assertFalse(summary["nodes"][0]["features_trainable"])
        reloaded = ToyNode(normal=True)
        reloaded.load_state_dict(node.state_dict(), strict=True)
        torch.testing.assert_close(reloaded.raw_normal_residual(reloaded._features), before, atol=0, rtol=0)
        self.assertGreater(anchor_loss(trainer, anchors, 12, torch.Generator().manual_seed(3)).item(), 0.)

    def test_anchor_has_zero_initial_loss_and_only_material_gradients(self):
        trainer = ToyTrainer(normal=False, spatial=True)
        selected, _ = select_material_parameters(trainer)
        anchors = build_material_anchors(trainer, chunk_size=5)
        node = trainer.models["Background"]
        indices = torch.tensor([1, 4])
        inputs = material_inputs(node, indices)
        torch.testing.assert_close(inputs[:, 4:], ((node._means[indices] - node.scene_origin) / 3).clamp(-1, 1))
        generator = torch.Generator().manual_seed(1)
        loss = anchor_loss(trainer, anchors, 12, generator)
        self.assertAlmostEqual(loss.item(), 0., places=7)
        with torch.no_grad():
            node.mlp_material["raw"][-1].bias.add_(.5)
        loss = anchor_loss(trainer, anchors, 12, generator)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertIsNotNone(node._features.grad)
        self.assertIsNone(node._means.grad)
        assert_frozen_parameters(trainer, selected)

    def test_resume_preserves_optimizer_sampler_anchor_and_global_rng(self):
        def setup():
            torch.manual_seed(13)
            model = nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            sampler = TrainFrameSampler(range(12), {frame: "low" for frame in range(12)},
                                        balanced=False, seed=5)
            anchor_rng = torch.Generator().manual_seed(17)
            return model, optimizer, sampler, anchor_rng

        def update(model, optimizer, sampler, anchor_rng):
            frame, previous = sampler.sample()
            x = torch.randn(3, 2) + torch.rand(3, 2, generator=anchor_rng)
            target = random.random() + np.random.rand() + frame + (previous or 0)
            optimizer.zero_grad()
            (model(x) - target).square().mean().backward()
            optimizer.step()

        random.seed(8)
        np.random.seed(9)
        model, optimizer, sampler, anchor_rng = setup()
        for _ in range(4):
            update(model, optimizer, sampler, anchor_rng)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"
            atomic_torch_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "sampler": sampler.state_dict(), "anchor_rng": anchor_rng.get_state(),
                                    "rng": capture_rng()})
            for _ in range(5):
                update(model, optimizer, sampler, anchor_rng)
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model, optimizer, sampler, anchor_rng = setup()
            state = torch.load(path, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            sampler.load_state_dict(state["sampler"])
            anchor_rng.set_state(state["anchor_rng"])
            restore_rng(state["rng"])
            for _ in range(5):
                update(model, optimizer, sampler, anchor_rng)
            for name, actual in model.state_dict().items():
                torch.testing.assert_close(actual, expected[name], atol=0, rtol=0)
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_no_temporal_modes_have_zero_differentiable_loss(self):
        for mode in ("continue", "balanced", "legacy", "albedo", "albedo_roughness"):
            current = {"albedo": torch.ones(3, 4, 3, requires_grad=True)}
            value = temporal_loss(mode, current, None, .25)
            self.assertEqual(float(value.detach()), 0.)
            value.backward()
            torch.testing.assert_close(current["albedo"].grad, torch.zeros_like(current["albedo"]))

    def test_temporal_modes_select_channels_and_detach_previous(self):
        from tests.test_ego_temporal import packet
        for mode in ("legacy", "albedo", "albedo_roughness"):
            current, previous = packet(), packet(frame_id=0)
            current["albedo"].add_(.2).requires_grad_()
            current["roughness"].add_(.2).requires_grad_()
            current["metallic"].add_(.2).requires_grad_()
            for key in ("albedo", "roughness", "metallic"):
                previous[key].requires_grad_()
            loss = temporal_loss(mode, current, previous, .25)
            self.assertGreater(float(loss.detach()), 0.)
            loss.backward()
            self.assertGreater(float(current["albedo"].grad.abs().sum()), 0.)
            if mode == "albedo":
                self.assertIsNone(current["roughness"].grad)
            else:
                self.assertGreater(float(current["roughness"].grad.abs().sum()), 0.)
            if mode != "legacy":
                self.assertIsNone(current["metallic"].grad)
            for key in ("albedo", "roughness", "metallic"):
                self.assertIsNone(previous[key].grad)

    def test_validation_rejects_invalid_arguments_and_nonempty_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "source.pth"
            checkpoint.touch()
            (root / "config.yaml").write_text("{}")
            args = ["--checkpoint", str(checkpoint), "--output-dir", str(root / "out"), "--mode", "albedo"]
            self.assertEqual(validate_args(make_parser().parse_args(args)).steps, 2000)
            for extra in (["--steps", "0"], ["--lr", "nan"], ["--anchor-weight", "-1"],
                          ["--seed", "-1"], ["--downscale", ".5"], ["--resume"]):
                with self.subTest(extra=extra), self.assertRaises((ValueError, FileNotFoundError)):
                    validate_args(make_parser().parse_args(args + extra))
            (root / "out").mkdir()
            (root / "out/existing").touch()
            with self.assertRaises(FileExistsError):
                validate_args(make_parser().parse_args(args))


if __name__ == "__main__":
    unittest.main()
