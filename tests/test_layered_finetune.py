"""CPU contracts for paired, original-loss, material-decoder-only adaptation."""

import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tools.ego_motion_runtime import select_material_parameters
from tools.finetune_ego_material import all_named_parameters
from tools.finetune_layered_gbuffer import (
    checked_train_frames, make_parser, train_step, training_input_digest,
    validate_args, variant_config, restore_model_for_resume, main,
)


class FakeNode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._features = torch.nn.Parameter(torch.tensor([[.5]]))
        self._means = torch.nn.Parameter(torch.tensor([[2., 3., 4.]]))
        self.normal = torch.nn.Parameter(torch.tensor([0., 0., 1.]))
        self.mlp_material = torch.nn.ModuleDict({"raw": torch.nn.Linear(1, 1, bias=False)})
        torch.nn.init.constant_(self.mlp_material["raw"].weight, .2)


class FakeTrainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.light = torch.nn.Parameter(torch.tensor(.1))
        self.models = {"Background": FakeNode()}
        self.gaussian_classes = {"Background": 0}
        self.step = 30000
        self.loss_calls = []
        self.eval()
        for node in self.models.values():
            node.eval()

    def compute_losses(self, outputs, infos, cameras, weather):
        self.loss_calls.append((infos["frame_id"], weather))
        self.last_loss_inputs = (outputs, infos, cameras)
        return {
            "rgb_loss": .3 * (outputs["rgb"] - infos["pixels"]).square().mean(),
            "teacher_loss": .7 * (outputs["albedo"] - infos["gbuf_basecolor"]).square().mean(),
            "frozen_reg": self.light.square() * .01,
        }


class FakeCamera:
    def __init__(self):
        self.calls = []
        self.offset = 0.

    def get_image(self, frame, weather):
        self.calls.append(frame)
        infos = {"pixels": torch.full((2, 3, 3), .3 + self.offset), "frame_id": frame}
        for kind in ("basecolor", "metallic", "roughness", "normal"):
            infos["gbuf_" + kind] = torch.full((2, 3, 3), .4)
        return infos, {"camera_to_world": torch.eye(4), "intrinsics": torch.eye(3)}


class FakeRuntime:
    def __init__(self):
        self.trainer = FakeTrainer()
        self.train_frames, self.val_frames, self.test_frames = [0, 1, 4, 5], [2], [3]
        self.camera = FakeCamera()
        self.render_calls = []

    def render(self, frame, grad=False, return_outputs=False):
        self.render_calls.append((frame, grad, return_outputs))
        node = self.trainer.models["Background"]
        albedo = node.mlp_material["raw"](node._features)
        outputs = {"rgb": albedo + self.trainer.light, "albedo": albedo}
        # References only become available after the prediction is constructed.
        infos = {"pixels": torch.tensor([[.3]]), "gbuf_basecolor": torch.tensor([[.4]]),
                 "frame_id": frame}
        return {"training_outputs": outputs, "training_image_infos": infos,
                "training_camera_infos": {"camera_id": 0}}


class LayeredFinetuneTests(unittest.TestCase):
    def prepare(self):
        runtime = FakeRuntime()
        selected, _ = select_material_parameters(runtime.trainer, update_features=False)
        optimizer = torch.optim.Adam([parameter for _, parameter in selected], lr=.01)
        return runtime, selected, optimizer

    def test_four_renderer_variants_are_exactly_separated(self):
        self.assertIsNone(variant_config("B0"))
        self.assertEqual(variant_config("B1", "instance", 4)["grouping"], "single")
        self.assertEqual(variant_config("B1")["color_space"], "linear")
        self.assertEqual(variant_config("B1", max_layers=4)["max_layers"], 0)
        self.assertEqual(variant_config("L0")["color_space"], "srgb")
        self.assertEqual(variant_config("L1")["color_space"], "linear")
        self.assertEqual(variant_config("L1", "dynamic", 4)["max_layers"], 4)
        with self.assertRaises(ValueError):
            variant_config("L1", max_layers=1)

    def test_original_weighted_losses_update_only_decoder(self):
        runtime, selected, optimizer = self.prepare()
        trainer = runtime.trainer
        before = {name: parameter.detach().clone() for name, parameter in all_named_parameters(trainer)}
        record = train_step(runtime, 0, optimizer, selected)
        expected = .3 * (.2 - .3) ** 2 + .7 * (.1 - .4) ** 2 + .1 ** 2 * .01
        self.assertAlmostEqual(record["loss"], expected, places=6)
        self.assertEqual(set(record["losses"]), {"rgb_loss", "teacher_loss", "frozen_reg"})
        self.assertEqual(trainer.loss_calls, [(0, "raw")])
        self.assertEqual(runtime.render_calls, [(0, True, True)])
        self.assertEqual(trainer.step, 30000)
        selected_ids = {id(parameter) for _, parameter in selected}
        for name, parameter in all_named_parameters(trainer):
            if id(parameter) in selected_ids:
                self.assertFalse(torch.equal(parameter, before[name]))
            else:
                self.assertTrue(torch.equal(parameter, before[name]), name)
                self.assertIsNone(parameter.grad)
        self.assertEqual(len(selected), 1)
        self.assertIn("mlp_material.raw", selected[0][0])

    def test_holdout_cannot_be_rendered_or_passed_to_loss(self):
        runtime, selected, optimizer = self.prepare()
        for frame in (2, 3, 8):
            with self.assertRaises(ValueError):
                train_step(runtime, frame, optimizer, selected)
        self.assertEqual(runtime.render_calls, [])
        self.assertEqual(runtime.trainer.loss_calls, [])
        runtime.train_frames.append(2)
        with self.assertRaises(ValueError):
            checked_train_frames(runtime)

    def test_nonfinite_loss_rejected_before_parameter_update(self):
        runtime, selected, optimizer = self.prepare()
        original = selected[0][1].detach().clone()
        with mock.patch.object(runtime.trainer, "compute_losses", return_value={"bad": torch.tensor(float("nan"))}):
            with self.assertRaises(FloatingPointError):
                train_step(runtime, 0, optimizer, selected)
        self.assertTrue(torch.equal(selected[0][1], original))

    def test_unfrozen_geometry_or_train_mode_are_rejected(self):
        runtime, selected, optimizer = self.prepare()
        runtime.trainer.models["Background"]._means.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            train_step(runtime, 0, optimizer, selected)
        runtime.trainer.models["Background"]._means.requires_grad_(False)
        runtime.trainer.models["Background"].train()
        with self.assertRaises(RuntimeError):
            train_step(runtime, 0, optimizer, selected)
        self.assertEqual(runtime.render_calls, [])

    def test_resume_sampler_and_optimizer_reproduce_updates(self):
        runtime, selected, optimizer = self.prepare()
        sampler = random.Random(10)
        frames = checked_train_frames(runtime)
        train_step(runtime, sampler.choice(frames), optimizer, selected)
        state = {"parameter": selected[0][1].detach().clone(),
                 "optimizer": copy.deepcopy(optimizer.state_dict()), "sampler": sampler.getstate()}
        expected_frames = [sampler.choice(frames) for _ in range(4)]
        for frame in expected_frames:
            train_step(runtime, frame, optimizer, selected)
        expected_parameter = selected[0][1].detach().clone()
        resumed, resumed_selected, resumed_optimizer = self.prepare()
        with torch.no_grad():
            resumed_selected[0][1].copy_(state["parameter"])
        resumed_optimizer.load_state_dict(state["optimizer"])
        resumed_sampler = random.Random(100)
        resumed_sampler.setstate(state["sampler"])
        resumed_frames = [resumed_sampler.choice(frames) for _ in range(4)]
        self.assertEqual(expected_frames, resumed_frames)
        for frame in resumed_frames:
            train_step(resumed, frame, resumed_optimizer, resumed_selected)
        self.assertTrue(torch.equal(expected_parameter, resumed_selected[0][1]))

    def test_training_input_fingerprint_is_train_only_and_content_sensitive(self):
        runtime = FakeRuntime()
        frames = checked_train_frames(runtime)
        with mock.patch("builtins.print"):
            first = training_input_digest(runtime, frames)
            self.assertEqual(runtime.camera.calls, frames)
            again = training_input_digest(runtime, frames)
            self.assertEqual(first, again)
            runtime.camera.offset = .01
            self.assertNotEqual(first, training_input_digest(runtime, frames))

    def test_missing_teacher_is_not_silently_rgb_only(self):
        runtime = FakeRuntime()
        with mock.patch.object(runtime.camera, "get_image", return_value=({"pixels": torch.zeros(2, 3, 3)}, {})):
            with self.assertRaisesRegex(ValueError, "misses saved targets"):
                training_input_digest(runtime, checked_train_frames(runtime))

    def test_cli_defaults_validation_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "checkpoint.pth").touch()
            (source / "config.yaml").write_text("saved: true\n")
            argv = ["--checkpoint", str(source / "checkpoint.pth"), "--output-dir", str(source / "out"),
                    "--variant", "L1"]
            args = validate_args(make_parser().parse_args(argv))
            self.assertEqual(args.steps, 3000)
            self.assertEqual(args.downscale, 1.)
            self.assertFalse(args.antialiased)
            self.assertFalse(args.allow_nonblock_diagnostic)
            self.assertEqual(args.max_layers, 0)
            for option, value in (("--steps", "0"), ("--lr", "nan"), ("--max-layers", "1"),
                                  ("--save-every", "0"), ("--downscale", ".5")):
                with self.assertRaises(ValueError):
                    validate_args(make_parser().parse_args(argv + [option, value]))
            args.output_dir.mkdir()
            (args.output_dir / "existing.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                validate_args(make_parser().parse_args(argv))
            with self.assertRaises(FileNotFoundError):
                validate_args(make_parser().parse_args(argv + ["--resume"]))

    def test_resume_moves_plain_instance_tensors_with_model(self):
        state = {"models": {"RigidNodes": {"point_ids": torch.tensor([[0], [1]]),
                                            "instances_fv": torch.ones(2, 2)}}, "step": 30000}
        trainer = mock.Mock()
        runtime = SimpleNamespace(trainer=trainer, device=torch.device("meta"))
        restore_model_for_resume(runtime, state)
        passed = trainer.load_state_dict.call_args.args[0]
        self.assertEqual(passed["models"]["RigidNodes"]["point_ids"].device.type, "meta")
        self.assertEqual(passed["models"]["RigidNodes"]["instances_fv"].device.type, "meta")
        self.assertEqual(state["models"]["RigidNodes"]["point_ids"].device.type, "cpu")

    def test_output_is_revalidated_after_acquiring_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "checkpoint.pth").touch()
            (source / "config.yaml").write_text("saved: true\n")
            out = source / "out"
            argv = ["--checkpoint", str(source / "checkpoint.pth"), "--output-dir", str(out),
                    "--variant", "L1"]
            def competing_run_finished(*unused):
                (out / "manifest.json").write_text('{"status":"completed"}')
            with mock.patch("torch.cuda.is_available", return_value=True), \
                 mock.patch("tools.finetune_layered_gbuffer.fcntl.flock", side_effect=competing_run_finished), \
                 mock.patch("tools.finetune_layered_gbuffer._run") as run:
                with self.assertRaises(FileExistsError):
                    main(argv)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
