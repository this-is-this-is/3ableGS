"""Behavior checks for the historically recovered Base v3 material contract."""

import io
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.gaussians.feature import FeatureGaussians
from models.gaussians.basics import gaussian_normals, quat_to_rotmat
from models.nodes import RigidNodes_feature, DeformableNodes_feature, SMPLNodes_feature


def make_model(class_name="Background", cls=FeatureGaussians, pbr=True, normal=True):
    raw = {
        "spatial_xyz": {"enabled": class_name == "Background"},
        "decoder": {"activation": "leaky_relu", "negative_slope": 0.01},
        "normal_residual": {
            "enabled": normal, "space": "gaussian_local_tangent",
            "max_tangent": 0.5, "regularization_weight": 0.01, "seed": 15015,
        },
    }
    model = cls(
        class_name=class_name,
        ctrl=OmegaConf.create({
            "sh_degree": 3, "use_voxel_deformer": True,
            "use_deformgs_for_nonrigid": True, "use_deformgs_after": 0,
            "stop_optimizing_canonical_xyz": False,
        }),
        reg=OmegaConf.create({}), feature=OmegaConf.create({"feature_dim": 32}),
        networks=OmegaConf.create({"D": 2, "W": 16, "embed_dim": 4,
                                  "x_multires": 1, "t_multires": 1,
                                  "deform_quat": True, "deform_scale": False}),
        raw_material=OmegaConf.create(raw) if pbr else None,
        scene_origin=torch.tensor([1., 2., 3.]), scene_scale=2., device="cpu",
    )
    model._means = torch.nn.Parameter(torch.tensor([[1., 2., 3.], [8., -4., 4.]]))
    model._features = torch.nn.Parameter(torch.ones(2, 32) * 0.1)
    model._scales = torch.nn.Parameter(torch.tensor([[0.1, 1., 2.], [0.1, 1., 2.]]).log())
    model._quats = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1))
    model._opacities = torch.nn.Parameter(torch.zeros(2, 1))
    return model


def attach_instance(model):
    model.point_ids = torch.zeros(2, 1, dtype=torch.long)
    model.instances_fv = torch.ones(1, 1, dtype=torch.bool)
    model.instances_size = torch.ones(1, 3) * 10
    model.instances_trans = torch.nn.Parameter(torch.zeros(1, 1, 3))
    model.instances_quats = torch.nn.Parameter(torch.tensor(
        [[[2 ** -0.5, 0., 0., 2 ** -0.5]]]))
    model.cur_frame = 0
    model.step = 1
    return model


class BaseV3MaterialTests(unittest.TestCase):
    def test_historical_dimensions_activation_and_zero_normal_initialization(self):
        for name in ("Background", "RigidNodes", "DeformableNodes", "SMPLNodes"):
            model = make_model(name)
            self.assertEqual(model.mlp_material["raw"][0].in_features,
                             35 if name == "Background" else 32)
            self.assertEqual(model.mlp_material["raw"][0].out_features, 32)
            self.assertEqual(model.mlp_material["raw"][-1].out_features, 5)
            self.assertIsInstance(model.mlp_material["raw"][1], torch.nn.LeakyReLU)
            self.assertIsInstance(model.raw_normal_residual[1], torch.nn.ReLU)
            self.assertEqual(torch.count_nonzero(model.raw_normal_residual[-1].weight), 0)
            before = gaussian_normals(model.get_quats, model.get_scaling)
            after = model._apply_raw_normal_residual(before, model.get_quats, model.get_scaling)
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_spatial_input_detaches_geometry_and_clamps_normalized_xyz(self):
        model = make_model()
        inputs = model._raw_material_input()
        torch.testing.assert_close(inputs[:, -3:], torch.tensor([[0., 0., 0.], [1., -1., .5]]))
        sum(x.sum() for x in model.materials()).backward()
        self.assertIsNone(model._means.grad)
        self.assertGreater(model._features.grad.abs().sum(), 0)

    def test_roughness_mapping_has_historical_floor_and_half_value(self):
        logits = torch.zeros(3, 5)
        logits[:, 4] = torch.tensor([-100., 0., 100.])
        albedo, metallic, roughness = FeatureGaussians._activate_material_logits(logits)
        torch.testing.assert_close(roughness[:, 0], torch.tensor([.01, .505, 1.]))
        torch.testing.assert_close(albedo, torch.full((3, 3), .5))
        torch.testing.assert_close(metallic, torch.full((3, 1), .5))

    def test_isolated_normal_initialization_preserves_material_and_global_rng(self):
        torch.manual_seed(91)
        without = make_model(normal=False)
        after_without = torch.rand(8)
        torch.manual_seed(91)
        with_head = make_model(normal=True)
        after_with = torch.rand(8)
        torch.testing.assert_close(after_without, after_with, rtol=0, atol=0)
        for key, value in without.mlp_material.state_dict().items():
            torch.testing.assert_close(value, with_head.mlp_material.state_dict()[key], rtol=0, atol=0)

    def test_local_tangent_rotates_with_gaussian_and_is_bounded(self):
        model = make_model("RigidNodes")
        with torch.no_grad():
            model.raw_normal_residual[-1].bias.copy_(torch.tensor([5., 20., 0.]))
        q0 = model.get_quats
        q1 = torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5]]).repeat(2, 1).requires_grad_()
        scales = model.get_scaling
        n0 = gaussian_normals(q0, scales)
        n1 = gaussian_normals(q1, scales)
        corrected0 = model._apply_raw_normal_residual(n0, q0, scales)
        corrected1 = model._apply_raw_normal_residual(n1, q1, scales)
        expected = torch.bmm(quat_to_rotmat(q1), corrected0[..., None])[..., 0]
        torch.testing.assert_close(corrected1, expected)
        angles = torch.acos((corrected0 * n0).sum(-1).clamp(-1, 1))
        self.assertLessEqual(float(angles.max().detach()), math.atan(.5) + 1e-6)
        regularizer = model.compute_reg_loss("raw")["normal_residual"]
        self.assertGreater(float(regularizer.detach()), 0.)
        (corrected1[:, 0].sum() + regularizer).backward()
        self.assertGreater(model.raw_normal_residual[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(q1.grad.abs().sum(), 0)

    def test_camera_facing_is_applied_after_normal_correction(self):
        model = make_model()
        with torch.no_grad():
            model.raw_normal_residual[-1].bias[1] = 2.
        camera = torch.eye(4)
        camera[:3, 3] = torch.tensor([-100., -100., 3.])
        packed = model.gbuffer("raw", model.get_quats, model.get_scaling,
                               model._means, SimpleNamespace(camtoworlds=camera))
        self.assertEqual(packed.shape, (2, 8))
        towards = camera[:3, 3] - model._means
        self.assertTrue(torch.all((packed[:, 5:] * towards).sum(-1) > 0))
        torch.testing.assert_close(packed[:, 5:].norm(dim=-1), torch.ones(2))

    def test_next_step_clears_normal_graph_for_absent_nodes(self):
        model = make_model()
        model.gbuffer("raw", model.get_quats, model.get_scaling)
        self.assertIn("normal_residual", model.compute_reg_loss("raw"))
        model.preprocess_per_train_step(2)
        self.assertNotIn("normal_residual", model.compute_reg_loss("raw"))

    def test_nodes_supply_final_world_and_deformed_rotations(self):
        rigid = attach_instance(make_model("RigidNodes", RigidNodes_feature))
        deform = attach_instance(make_model("DeformableNodes", DeformableNodes_feature))
        smpl = attach_instance(make_model("SMPLNodes", SMPLNodes_feature))
        for model in (rigid, deform, smpl):
            with torch.no_grad():
                model.raw_normal_residual[-1].bias[2] = 1.
        delta = torch.tensor([[0., 0., .25, 0.]]).repeat(2, 1)
        with patch.object(deform, "get_deformation", return_value=(torch.zeros(2, 3), delta, None)):
            deformed = deform.get_gaussians(None, "raw")
        rigid_result = rigid.get_gaussians(None, "raw")
        posed_quats = torch.tensor([[2 ** -.5, 0., 2 ** -.5, 0.]]).repeat(2, 1)
        with patch.object(smpl, "transform_means_and_quats", return_value=(smpl._means, posed_quats)):
            smpl_result = smpl.get_gaussians(None, "raw")
        for model, result in ((rigid, rigid_result), (deform, deformed), (smpl, smpl_result)):
            expected = model._apply_raw_normal_residual(
                gaussian_normals(result["_quats"], result["_scales"]),
                result["_quats"], result["_scales"])
            self.assertEqual(result["_rgbs"].shape[-1], 8)
            torch.testing.assert_close(result["_rgbs"][:, 5:], expected)
            self.assertFalse(torch.allclose(result["_quats"], model.get_quats))

    def test_material_and_normal_checkpoint_roundtrip_and_parameter_coverage(self):
        model = make_model()
        with torch.no_grad():
            model.raw_normal_residual[-1].bias[1] = .7
        groups = model.get_param_groups()
        parameters = [p for group in groups.values() for p in group]
        self.assertEqual(len({id(p) for p in parameters}), len(parameters))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters()})
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = make_model()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(model.get_gaussians(None, "raw")["_rgbs"],
                                   restored.get_gaussians(None, "raw")["_rgbs"], rtol=0, atol=0)

    def test_split_clone_cull_keep_32d_features_and_adam_state_aligned(self):
        # Exercise the real refinement orchestration, including one split, one
        # clone, one opacity cull, and a subsequent optimizer update.
        for cls, name in ((FeatureGaussians, "Background"),
                          (RigidNodes_feature, "RigidNodes"),
                          (DeformableNodes_feature, "DeformableNodes")):
            with self.subTest(node=name):
                model = make_model(name, cls)
                if cls is not FeatureGaussians:
                    attach_instance(model)
                    model.point_ids = torch.zeros(3, 1, dtype=torch.long)
                model.ctrl_cfg = OmegaConf.merge(model.ctrl_cfg, {
                    "warmup_steps": 500, "reset_alpha_interval": 3000,
                    "refine_interval": 100, "stop_split_at": 15000,
                    "densify_grad_thresh": .0005, "densify_size_thresh": .003,
                    "stop_screen_size_at": 4000, "split_screen_size": .05,
                    "n_split_samples": 2, "cull_alpha_thresh": .005,
                    "cull_out_of_bound": False,
                })
                model.num_train_images = 1
                model._means = torch.nn.Parameter(torch.zeros(3, 3))
                model._features = torch.nn.Parameter(
                    torch.tensor([.1, .2, .3])[:, None].repeat(1, 32))
                model._scales = torch.nn.Parameter(torch.tensor(
                    [[.02] * 3, [.001] * 3, [.002] * 3]).log())
                model._quats = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1))
                model._opacities = torch.nn.Parameter(torch.tensor([[.1], [.1], [.001]]).logit())
                optimizer = torch.optim.Adam([
                    {"params": list(parameters), "name": group, "lr": 0.}
                    for group, parameters in model.get_gaussian_param_groups().items()])
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                original_head = model.raw_normal_residual[-1].weight
                head_moment = optimizer.state[original_head]["exp_avg"].clone()
                feature_moment = optimizer.state[model._features]["exp_avg"].clone()
                model.step = 600
                model.xys_grad_norm = torch.tensor([.001, .001, 0.])
                model.vis_counts = torch.ones(3)
                model.max_2Dsize = torch.zeros(3)
                model.refinement_after(600, optimizer)
                self.assertEqual(model.num_points, 5)
                torch.testing.assert_close(model._features[:, 0], torch.tensor([.1, .2, .1, .1, .2]))
                self.assertEqual(model._features.shape, (5, 32))
                feature_state = optimizer.state[model._features]
                torch.testing.assert_close(feature_state["exp_avg"][:2], feature_moment[:2])
                torch.testing.assert_close(feature_state["exp_avg"][2:], torch.zeros(3, 32))
                torch.testing.assert_close(optimizer.state[original_head]["exp_avg"], head_moment)
                expected = {key: list(value) for key, value in model.get_gaussian_param_groups().items()}
                for group in optimizer.param_groups:
                    self.assertIs(group["params"][0], expected[group["name"]][0])
                    for parameter in group["params"]:
                        self.assertEqual(optimizer.state[parameter]["exp_avg"].shape, parameter.shape)
                        self.assertEqual(optimizer.state[parameter]["exp_avg_sq"].shape, parameter.shape)
                        parameter.grad = torch.ones_like(parameter)
                optimizer.step()  # Raises if any state retained the pre-refinement shape.
                if cls is not FeatureGaussians:
                    self.assertEqual(model.point_ids.shape, (5, 1))

    def test_smpl_knn_regularizer_uses_material_albedo_and_backpropagates(self):
        model = attach_instance(make_model("SMPLNodes", SMPLNodes_feature))
        model.smpl_points_num = 2
        model.ctrl_cfg = OmegaConf.merge(model.ctrl_cfg, {
            "knn_neighbors": 2, "freeze_feature": False, "freeze_o": False,
            "freeze_s": False, "freeze_q": False,
        })
        model.reg_cfg = OmegaConf.create({"knn_reg": {
            "lambda_std_shs_dc": .001, "lambda_std_o": .001,
            "lambda_std_s": .001, "lambda_std_q": .001,
        }})
        model.nn_ind = torch.tensor([[[0, 1], [1, 0]]])
        with torch.no_grad():
            model._features[1].add_(.3)
        losses = model.compute_reg_loss("raw")
        self.assertIn("knn_reg_dc", losses)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["knn_reg_dc"].backward()
        self.assertGreater(model._features.grad.abs().sum(), 0)
        self.assertGreater(model.mlp_material["raw"][-1].weight.grad.abs().sum(), 0)

    def test_existing_rgb_configuration_preserves_three_channel_contract(self):
        model = make_model(pbr=False)
        output = model.get_gaussians(None, "raw")["_rgbs"]
        torch.testing.assert_close(output, model.mlp_color["raw"](model._features))
        self.assertEqual(output.shape, (2, 3))
        self.assertFalse(any(k.startswith("mlp_material") for k in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
