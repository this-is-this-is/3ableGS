"""Bounded RGB composition and Gaussian feature lifetime contracts."""
import copy
import unittest

import torch
from omegaconf import OmegaConf

from models.rgb_residual import RGBResidual, normalized_rgb_residual_penalties
from models.gaussians.feature import FeatureGaussians
from models.nodes import RigidNodes_feature, DeformableNodes_feature, SMPLNodes_feature
from tests.test_base_v4_density import make_population, refine


def inputs(size=5):
    return [torch.full((size, size, channels), .4, requires_grad=True)
            for channels in (8, 3, 1, 1, 1, 3, 3)]


class RGBResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_initial_identity_and_construction_preserves_rng(self):
        before = torch.get_rng_state().clone()
        head = RGBResidual()
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        values = inputs()
        output = head(*values)
        self.assertTrue(torch.equal(output["rgb_hybrid"], values[-1]))
        self.assertTrue(torch.equal(output["rgb_gain"], torch.ones_like(values[-1])))
        self.assertEqual(int(output["rgb_additive"].count_nonzero()), 0)
        output["rgb_hybrid"].sum().backward()
        self.assertGreater(float(head.net[-1].weight.grad.abs().sum()), 0)
        torch.testing.assert_close(values[-1].grad, torch.ones_like(values[-1]))

    def test_only_features_condition_cnn_but_pbr_composition_is_trainable(self):
        head = RGBResidual()
        with torch.no_grad():
            head.net[-1].weight.fill_(.05)
            head.net[-1].bias.fill_(.3)
        values = inputs()
        output = head(*values, strength=.5)
        output["rgb_hybrid"].sum().backward()
        expected = 1 + .5 * values[4].detach() * output["rgb_log_gain"].detach().expm1()
        torch.testing.assert_close(values[-1].grad, expected)
        self.assertGreater(float(values[0].grad.abs().sum()), 0)
        for value in values[1:-1]:
            self.assertIsNone(value.grad)

    def test_bounds_zero_alpha_and_fractional_strength(self):
        head = RGBResidual()
        with torch.no_grad():
            head.net[-1].bias.copy_(torch.tensor([-20., 0., 20., -20., 0., 20.]))
        values = inputs()
        values[4].data[0] = 0
        full = head(*values)
        half = head(*values, strength=.5)
        torch.testing.assert_close(half["rgb_residual"], .5 * full["rgb_residual"])
        torch.testing.assert_close(full["rgb_hybrid"][0], values[-1][0])
        self.assertLessEqual(float(full["rgb_log_gain"].abs().max()), .300001)
        self.assertLessEqual(float(full["rgb_additive"].abs().max()), .100001)
        self.assertTrue(torch.isfinite(full["rgb_hybrid"]).all())

    def test_disabled_additive_and_zero_bounds_have_finite_penalties(self):
        head = RGBResidual(additive_enabled=False, max_log_gain=0)
        with torch.no_grad():
            head.net[-1].bias.fill_(1.)
        output = head(*inputs())
        self.assertEqual(int(output["rgb_additive"].count_nonzero()), 0)
        self.assertEqual(int(output["rgb_residual"].count_nonzero()), 0)
        log = torch.full((2, 2, 3), .15, requires_grad=True)
        additive = torch.full_like(log, .025, requires_grad=True)
        gain, add = normalized_rgb_residual_penalties(log, additive, .3, .1)
        torch.testing.assert_close(gain, torch.full((2, 2, 1), .25))
        torch.testing.assert_close(add, torch.full((2, 2, 1), .0625))
        zeros = normalized_rgb_residual_penalties(log, additive, 0, 0)
        sum(x.sum() for x in zeros).backward()
        self.assertEqual(int(log.grad.count_nonzero()), 0)
        self.assertEqual(int(additive.grad.count_nonzero()), 0)

    def test_invalid_bounds_dimensions_and_strength(self):
        for kwargs in ({"max_log_gain": -1}, {"max_additive": float("nan")},
                       {"feature_dim": 0}, {"hidden_dim": True}, {"additive_enabled": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RGBResidual(**kwargs)
        head = RGBResidual()
        for strength in (-1, 1.1, float("nan"), True):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                head(*inputs(), strength=strength)


def enabled_population(cls=FeatureGaussians):
    model, optimizer = make_population(cls, stop_split_at=25000, stop_reset_at=20000)
    model.rgb_residual_enabled = True
    model._initialize_correction_features()
    model._correction_features.data.copy_(torch.arange(24.).reshape(3, 8))
    optimizer.add_param_group({"params": [model._correction_features],
                              "name": model.class_prefix + "correction_feature", "lr": 0.})
    model._correction_features.grad = torch.ones_like(model._correction_features)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


class CorrectionGaussianTests(unittest.TestCase):
    def assert_alignment(self, model, optimizer):
        self.assertEqual(model._correction_features.shape, (model.num_points, 8))
        group = next(g for g in optimizer.param_groups if g["name"].endswith("#correction_feature"))
        self.assertIs(group["params"][0], model._correction_features)
        state = optimizer.state[model._correction_features]
        self.assertEqual(state["exp_avg"].shape, model._correction_features.shape)
        self.assertEqual(state["exp_avg_sq"].shape, model._correction_features.shape)
        model._correction_features.grad = torch.ones_like(model._correction_features)
        optimizer.step()

    def test_feature_config_is_opt_in_and_constructs_zero_seed_rows(self):
        kwargs = dict(class_name="Background", ctrl=OmegaConf.create({}), reg=OmegaConf.create({}),
                      feature=OmegaConf.create({"feature_dim": 32}), raw_material=OmegaConf.create({}), device="cpu")
        torch.manual_seed(28)
        old = FeatureGaussians(**kwargs)
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(28)
        model = FeatureGaussians(**kwargs, rgb_residual=OmegaConf.create({"enabled": True, "feature_dim": 8}))
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertFalse(hasattr(old, "_correction_features"))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        seeds = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        model.create_from_pcd(seeds, torch.ones_like(seeds))
        self.assertEqual(model._correction_features.shape, (4, 8))
        self.assertEqual(int(model._correction_features.count_nonzero()), 0)
        self.assertIn("Background#correction_feature", model.get_gaussian_param_groups())
        payload = {}
        model._add_correction_payload(payload, torch.tensor([True, False, True, False]))
        self.assertEqual(payload["_correction_features"].shape, (2, 8))

    def test_split_clone_prune_and_adam_state_for_background_and_actors(self):
        for cls in (FeatureGaussians, RigidNodes_feature, DeformableNodes_feature):
            for operation in ("split", "clone", "prune"):
                with self.subTest(cls=cls.__name__, operation=operation):
                    model, optimizer = enabled_population(cls)
                    old = model._correction_features.detach().clone()
                    if operation == "prune":
                        model._opacities.data[1] = torch.tensor(.001).logit()
                        refine(model, optimizer, 26000)
                        indices = [0, 2]
                    elif operation == "split":
                        refine(model, optimizer, 19800)
                        indices = [0, 1, 2, 0, 0]
                    else:
                        model._scales.data[0] = torch.tensor(.001).log()
                        refine(model, optimizer, 19800)
                        indices = [0, 1, 2, 0]
                    torch.testing.assert_close(model._correction_features, old[indices])
                    self.assert_alignment(model, optimizer)

    def test_full_checkpoint_requires_matching_feature_rows_and_restores_adam(self):
        source, optim = enabled_population()
        refine(source, optim, 19800)
        state = copy.deepcopy(source.state_dict())
        restored, _ = enabled_population()
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored._correction_features, source._correction_features)
        resumed = torch.optim.Adam([{"params": list(params), "name": name, "lr": 0.}
                                    for name, params in restored.get_gaussian_param_groups().items()])
        # Group order must match the saved optimizer, whose correction group was appended.
        resumed.param_groups.sort(key=lambda group: [g["name"] for g in optim.param_groups].index(group["name"]))
        resumed.load_state_dict(optim.state_dict())
        self.assert_alignment(restored, resumed)
        state.pop("_correction_features")
        with self.assertRaisesRegex(ValueError, "matching correction feature"):
            restored.load_state_dict(state, strict=False)

    def test_actor_instance_removal_and_replacement_carry_features(self):
        model, _ = enabled_population(RigidNodes_feature)
        model.point_ids = torch.tensor([[0], [1], [1]])
        original = model._correction_features.detach().clone()
        model.replace_instances({0: 1})
        torch.testing.assert_close(model._correction_features, original[[1, 2, 1, 2]])
        self.assertEqual(model._correction_features.shape[0], model.num_points)

    def test_smpl_inherits_feature_parameter_and_preserves_it_without_refinement(self):
        model, optimizer = enabled_population(SMPLNodes_feature)
        before = model._correction_features.detach().clone()
        model.preprocess_per_train_step(19800)
        model.postprocess_per_train_step(19800, optimizer, torch.ones(3), torch.ones(3, 2), 128)
        torch.testing.assert_close(model._correction_features, before)
        self.assert_alignment(model, optimizer)


if __name__ == "__main__":
    unittest.main()
