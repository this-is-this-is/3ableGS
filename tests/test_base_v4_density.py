"""Exercise Base v4's historical density schedule and Adam state transitions."""

import unittest

import torch
from omegaconf import OmegaConf

from models.gaussians.feature import FeatureGaussians
from models.nodes import RigidNodes_feature, DeformableNodes_feature, SMPLNodes_feature


def make_population(cls=FeatureGaussians, stop_split_at=25000, **overrides):
    config = {
        "warmup_steps": 500, "reset_alpha_interval": 3000,
        "refine_interval": 100, "stop_split_at": stop_split_at,
        "densify_grad_thresh": .0005, "densify_size_thresh": .003,
        "stop_screen_size_at": 4000, "split_screen_size": .05,
        "n_split_samples": 2, "cull_alpha_thresh": .005,
        "cull_scale_thresh": .5, "cull_screen_size": .15,
        "reset_alpha_value": .01, "cull_out_of_bound": False,
        "use_voxel_deformer": True,
    }
    config.update(overrides)
    model = cls(
        class_name="Background" if cls is FeatureGaussians else cls.__name__,
        ctrl=OmegaConf.create(config), reg=OmegaConf.create({}),
        feature=OmegaConf.create({"feature_dim": 32}),
        networks=OmegaConf.create({"D": 2, "W": 16, "embed_dim": 4,
                                  "x_multires": 1, "t_multires": 1,
                                  "deform_quat": True, "deform_scale": False}),
        raw_material=OmegaConf.create({}), scene_scale=1.,
        num_train_images=27, device="cpu",
    )
    model._means = torch.nn.Parameter(torch.zeros(3, 3))
    model._features = torch.nn.Parameter(torch.arange(3.)[:, None].repeat(1, 32))
    model._scales = torch.nn.Parameter(torch.tensor(
        [[.02] * 3, [.001] * 3, [.002] * 3]).log())
    model._quats = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1))
    model._opacities = torch.nn.Parameter(torch.tensor([[.1], [.1], [.1]]).logit())
    if cls is not FeatureGaussians:
        model.point_ids = torch.zeros(3, 1, dtype=torch.long)
    optimizer = torch.optim.Adam([
        {"params": list(parameters), "name": name, "lr": 0.}
        for name, parameters in model.get_gaussian_param_groups().items()])
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def refine(model, optimizer, step):
    model.step = step
    model.xys_grad_norm = torch.tensor([.001, 0., 0.])
    model.vis_counts = torch.ones(3)
    model.max_2Dsize = torch.zeros(3)
    model.refinement_after(step, optimizer)


class BaseV4DensityTests(unittest.TestCase):
    def test_reset_before_20k_caps_opacity_and_clears_only_opacity_moments(self):
        model, optimizer = make_population(stop_reset_at=20000)
        model._opacities.data.copy_(torch.tensor([[.1], [.004], [.2]]).logit())
        feature_moment = optimizer.state[model._features]["exp_avg"].clone()
        adam_step = optimizer.state[model._opacities]["step"].clone()
        # With the historical 100-step offset, 18100 is the final reset < 20k.
        refine(model, optimizer, 18100)
        torch.testing.assert_close(model.get_opacity, torch.tensor([[.01], [.004], [.01]]))
        self.assertEqual(model.num_points, 3)
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(torch.count_nonzero(optimizer.state[model._opacities][key]), 0)
        torch.testing.assert_close(optimizer.state[model._opacities]["step"], adam_step)
        torch.testing.assert_close(optimizer.state[model._features]["exp_avg"], feature_moment)

    def test_reset_after_20k_preserves_opacities_and_moments(self):
        model, optimizer = make_population(stop_reset_at=20000)
        before = model._opacities.detach().clone()
        moments = {key: value.clone() for key, value in optimizer.state[model._opacities].items()}
        refine(model, optimizer, 21100)
        torch.testing.assert_close(model._opacities, before, rtol=0, atol=0)
        for key, value in moments.items():
            torch.testing.assert_close(optimizer.state[model._opacities][key], value, rtol=0, atol=0)
        self.assertEqual(model.num_points, 3)

    def test_reset_cutoff_is_strict_and_none_retains_base_v3(self):
        model, _ = make_population(stop_reset_at=20000)
        for step, expected in ((19999, True), (20000, False), (30000, False)):
            model.step = step
            self.assertEqual(model.opacity_reset_is_active(), expected)
        for optional in ({}, {"stop_reset_at": None}):
            with self.subTest(optional=optional):
                model, optimizer = make_population(stop_split_at=15000, **optional)
                refine(model, optimizer, 27100)
                torch.testing.assert_close(model.get_opacity, torch.full((3, 1), .01))
                self.assertEqual(torch.count_nonzero(optimizer.state[model._opacities]["exp_avg"]), 0)

    def test_background_splits_at_24900_but_not_25000(self):
        for step, expected_count in ((24900, 5), (25000, 3)):
            with self.subTest(step=step):
                model, optimizer = make_population(stop_reset_at=20000)
                refine(model, optimizer, step)
                self.assertEqual(model.num_points, expected_count)
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        self.assertEqual(optimizer.state[parameter]["exp_avg"].shape, parameter.shape)
                        parameter.grad = torch.ones_like(parameter)
                optimizer.step()

    def test_reset_window_is_retained_after_reset_cutoff(self):
        # Disabling resets does not remove the existing modulo guard on splitting.
        for step, expected_count in ((24000, 3), (24100, 3), (24200, 5)):
            with self.subTest(step=step):
                model, optimizer = make_population(stop_reset_at=20000)
                refine(model, optimizer, step)
                self.assertEqual(model.num_points, expected_count)

    def test_pruning_continues_after_both_cutoffs(self):
        model, optimizer = make_population(stop_reset_at=20000)
        model._opacities.data[1] = torch.tensor(.001).logit()
        refine(model, optimizer, 26000)
        self.assertEqual(model.num_points, 2)
        torch.testing.assert_close(model._features[:, 0], torch.tensor([0., 2.]))
        self.assertEqual(optimizer.state[model._opacities]["exp_avg"].shape, (2, 1))

    def test_dynamic_nodes_keep_20k_split_cutoff_and_later_resets(self):
        for cls in (RigidNodes_feature, DeformableNodes_feature):
            with self.subTest(cls=cls.__name__):
                for step, expected_count in ((19800, 5), (20000, 3)):
                    model, optimizer = make_population(cls, stop_split_at=20000)
                    refine(model, optimizer, step)
                    self.assertEqual(model.num_points, expected_count)
                    self.assertEqual(model.point_ids.shape, (expected_count, 1))
                model, optimizer = make_population(cls, stop_split_at=20000)
                refine(model, optimizer, 21100)
                torch.testing.assert_close(model.get_opacity, torch.full((3, 1), .01))
                self.assertEqual(torch.count_nonzero(optimizer.state[model._opacities]["exp_avg"]), 0)

    def test_smpl_training_postprocess_does_not_densify_or_reset(self):
        # SMPL's normal training hook bypasses inherited refinement altogether.
        # Its inherited stop_split_at=15000 therefore does not drive topology.
        for step in (14800, 18100, 21100):
            with self.subTest(step=step):
                model, optimizer = make_population(SMPLNodes_feature, stop_split_at=15000)
                before = model._opacities.detach().clone()
                moment = optimizer.state[model._opacities]["exp_avg"].clone()
                model.preprocess_per_train_step(step)
                model.postprocess_per_train_step(
                    step, optimizer, torch.ones(3), torch.ones(3, 2), 128)
                self.assertEqual(model.num_points, 3)
                torch.testing.assert_close(model._opacities, before, rtol=0, atol=0)
                torch.testing.assert_close(optimizer.state[model._opacities]["exp_avg"], moment)


if __name__ == "__main__":
    unittest.main()
