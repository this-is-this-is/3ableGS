"""D1's visibility, instance balancing and detached confidence contracts."""
import unittest

import torch

from models.dynamic_gbuffer import (
    VisibleInstanceAccumulator, balanced_instance_loss, dynamic_gbuffer_weight,
    validate_dynamic_gbuffer_config,
)


ACTIVE = {"enabled": True, "mode": "balanced", "min_pixels": 1,
          "w": 1., "start_step": 0, "ramp_steps": 0}


def ownership(ids, purity=1., *, grad=False):
    ids = torch.as_tensor(ids, dtype=torch.long)
    return {"owner": ids,
            "purity": torch.full(ids.shape, purity, dtype=torch.float64, requires_grad=grad),
            "opacity": torch.ones(ids.shape, dtype=torch.float64, requires_grad=grad),
            "valid": torch.ones(ids.shape, dtype=torch.bool)}


class DynamicGBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_validation_rejects_typos_and_malformed_values(self):
        bad = [False, [], 5, {"enabled": 1}, {"unknown": True}, {"mode": "static"},
               {"w": -1}, {"w": True}, {"w": float("nan")}, {"w": float("inf")},
               {"start_step": -1}, {"ramp_steps": True}, {"ramp_steps": 2.5},
               {"interval": 0}, {"chunk_size": -1}, {"min_pixels": 0},
               {"visibility_downscale": .5}, {"visibility_downscale": 0},
               {"min_opacity": 1.1}, {"min_opacity": -1},
               {"purity_floor": 0}, {"purity_floor": 1.1}, {"purity_power": 0}]
        for cfg in bad:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_dynamic_gbuffer_config(cfg)
        source = {"enabled": True}
        cfg = validate_dynamic_gbuffer_config(source)
        cfg["enabled"] = False
        self.assertTrue(source["enabled"])
        self.assertEqual(cfg["visibility_downscale"], 2)
        with self.assertRaises(ValueError):
            validate_dynamic_gbuffer_config({"enabled": False, "typo": 1})

    def test_schedule_and_disabled_zero_need_no_maps(self):
        cfg = {"enabled": True}
        for step, value in ((4999, 0), (5000, 0), (5500, .125), (6000, .25), (30000, .25)):
            self.assertEqual(dynamic_gbuffer_weight(step, cfg), value)
        self.assertEqual(dynamic_gbuffer_weight(6000), 0)
        self.assertEqual(dynamic_gbuffer_weight(5000, {**cfg, "ramp_steps": 0}), .25)
        for step in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                dynamic_gbuffer_weight(step, cfg)
        self.assertEqual(dynamic_gbuffer_weight(5500, {**cfg, "interval": 3}), 0)
        self.assertAlmostEqual(dynamic_gbuffer_weight(5501, {**cfg, "interval": 3}), .12525)
        loss, diagnostics = balanced_instance_loss(None, None, None)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(diagnostics, {})
        value = torch.ones(2, 3, requires_grad=True)
        loss, _ = balanced_instance_loss(value, None, None, {"enabled": False})
        loss.backward()
        torch.testing.assert_close(value.grad, torch.zeros_like(value))

    def test_full_scene_transmittance_rejects_occluded_actor(self):
        # Ray0: background alpha .8, then object alpha .9 => visible mass .18.
        # Ray1: object alpha .8, then background alpha .9 => visible mass .8.
        # Isolated object opacity would incorrectly favor the actor on ray0.
        alpha = torch.tensor([[.98, .98]], dtype=torch.float64)
        masses = torch.tensor([[[.18], [.8]]], dtype=torch.float64)
        result = VisibleInstanceAccumulator(alpha).add(masses, [7]).finalize()
        self.assertEqual(result["owner"].tolist(), [[0, 7]])
        torch.testing.assert_close(result["purity"], torch.tensor([[0., .8 / .98]], dtype=torch.float64))
        self.assertTrue(result["valid"].all())

    def test_chunk_order_and_id_ties_are_deterministic(self):
        alpha = torch.ones(1, 4, dtype=torch.float64)
        masses = torch.tensor([[[.4, .4], [.1, .7], [.5, 0], [.5, .25]]], dtype=torch.float64)
        all_at_once = VisibleInstanceAccumulator(alpha).add(masses, [17, 3]).finalize()
        reverse_chunks = VisibleInstanceAccumulator(alpha)
        reverse_chunks.add(masses[..., 1:], [3]).add(masses[..., :1], [17])
        reverse_chunks = reverse_chunks.finalize()
        # Equal dynamic masses favor ID3; equal background/object favor bg0.
        self.assertEqual(all_at_once["owner"].tolist(), [[3, 3, 0, 17]])
        for name in all_at_once:
            torch.testing.assert_close(all_at_once[name], reverse_chunks[name])

    def test_zero_objects_and_nonphysical_mass_are_safe(self):
        empty = VisibleInstanceAccumulator(torch.ones(2, 2)).finalize()
        self.assertFalse(empty["owner"].any())
        self.assertFalse(empty["purity"].any())
        alpha = torch.tensor([[1., 1., 1., float("nan")]])
        masses = torch.tensor([[[.9, .8], [-.1, .8], [float("nan"), .8], [.8, .1]]])
        result = VisibleInstanceAccumulator(alpha).add(masses, [1, 2]).finalize()
        self.assertFalse(result["valid"].any())
        self.assertFalse(result["owner"].any())
        self.assertTrue(torch.isfinite(result["purity"]).all())

    def test_accumulator_detaches_inputs_and_rejects_duplicate_channels(self):
        alpha = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
        masses = torch.tensor([[[.8]]], dtype=torch.float64, requires_grad=True)
        acc = VisibleInstanceAccumulator(alpha).add(masses, [1])
        for ids in ([1], [0], [-1], [1.2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                acc.add(masses, ids)
        result = acc.finalize()
        self.assertTrue(all(not value.requires_grad for value in result.values()))
        with self.assertRaises(RuntimeError):
            acc.add(masses, [2])
        with self.assertRaises(RuntimeError):
            acc.finalize()
        with self.assertRaises(ValueError):
            VisibleInstanceAccumulator(alpha).add(torch.ones(1, 1, 2), [2, 2])

    def test_instance_balancing_is_invariant_to_other_actor_area(self):
        # One small actor has error2, another error6; both contribute equally.
        for large_area in (1, 5, 100):
            ids = [[1] + [2] * large_area]
            residual = torch.tensor([[2.] + [6.] * large_area], dtype=torch.float64, requires_grad=True)
            loss, stats = balanced_instance_loss(residual, ownership(ids), torch.ones_like(residual), ACTIVE)
            self.assertAlmostEqual(loss.item(), 4.)
            self.assertEqual(stats["eligible_instances"].item(), 2)
            loss.backward()
            self.assertAlmostEqual(residual.grad[0, 0].item(), .5)
            self.assertAlmostEqual(residual.grad[0, 1:].sum().item(), .5)

    def test_purity_reduces_uniform_uncertainty_without_normalization_cancellation(self):
        residual = torch.full((2, 4), 2., dtype=torch.float64)
        owned = ownership([[1] * 4] * 2, purity=.3)
        valid = torch.ones_like(residual)
        balanced, _ = balanced_instance_loss(residual, owned, valid, ACTIVE)
        weighted, stats = balanced_instance_loss(residual, owned, valid, {**ACTIVE, "mode": "purity"})
        self.assertEqual(balanced.item(), 2.)
        self.assertAlmostEqual(weighted.item(), .6)
        self.assertAlmostEqual(stats["mean_weight"].item(), .3)
        floor, _ = balanced_instance_loss(residual, owned, valid,
                                         {**ACTIVE, "mode": "purity", "purity_power": 3})
        self.assertAlmostEqual(floor.item(), .2)

    def test_rgb_channels_average_before_instance_reduction(self):
        residual = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], requires_grad=True)
        owned = ownership([[1, 2]])
        loss, _ = balanced_instance_loss(residual, owned, torch.ones(1, 2), ACTIVE)
        self.assertAlmostEqual(loss.item(), 3.5)
        loss.backward()
        torch.testing.assert_close(residual.grad, torch.full_like(residual, 1. / 6))

    def test_small_lowopacity_and_masked_actors_are_rejected(self):
        owned = ownership([[1, 1, 1, 2, 2, 3]])
        owned["opacity"][0, 3:5] = .05
        residual = torch.tensor([[2., 4., 500., 100., 100., 500.]], requires_grad=True)
        valid = torch.tensor([[1., 1., 0., 1., 1., 1.]])
        loss, stats = balanced_instance_loss(residual, owned, valid, {**ACTIVE, "min_pixels": 2})
        self.assertAlmostEqual(loss.item(), 3.)
        self.assertEqual(stats["eligible_instances"].item(), 1)
        self.assertEqual(stats["eligible_pixels"].item(), 2)
        self.assertEqual(stats["rejected_small_instances"].item(), 1)
        loss.backward()
        torch.testing.assert_close(residual.grad, torch.tensor([[.5, .5, 0., 0., 0., 0.]]))

    def test_no_eligible_instance_returns_graph_connected_zero(self):
        for ids, cfg in (([[0, 0]], ACTIVE), ([[1, 1]], {**ACTIVE, "min_pixels": 3})):
            residual = torch.ones(1, 2, 3, requires_grad=True)
            loss, stats = balanced_instance_loss(residual, ownership(ids), torch.ones(1, 2), cfg)
            self.assertTrue(loss.requires_grad)
            self.assertEqual(loss.item(), 0)
            self.assertEqual(stats["eligible_instances"].item(), 0)
            loss.backward()
            torch.testing.assert_close(residual.grad, torch.zeros_like(residual))

    def test_invalid_residuals_are_excluded_without_nan_gradients(self):
        residual = torch.tensor([[[2., 2., 2.], [float("nan"), 4., 5.]]], requires_grad=True)
        loss, stats = balanced_instance_loss(residual, ownership([[1, 1]]), torch.ones(1, 2), ACTIVE)
        self.assertEqual(loss.item(), 2)
        self.assertEqual(stats["eligible_pixels"].item(), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        torch.testing.assert_close(residual.grad[0, 1], torch.zeros(3))

    def test_weights_and_mask_have_no_gradients(self):
        residual = torch.ones(1, 2, requires_grad=True)
        owned = ownership([[1, 1]], .7, grad=True)
        valid = torch.ones(1, 2, requires_grad=True)
        loss, stats = balanced_instance_loss(residual, owned, valid, {**ACTIVE, "mode": "purity"})
        loss.backward()
        self.assertGreater(residual.grad.abs().sum().item(), 0)
        self.assertIsNone(owned["opacity"].grad)
        self.assertIsNone(owned["purity"].grad)
        self.assertIsNone(valid.grad)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))

    def test_helpers_do_not_advance_rng(self):
        before = torch.random.get_rng_state().clone()
        result = VisibleInstanceAccumulator(torch.ones(1, 2)).add(torch.full((1, 2, 1), .8), [1]).finalize()
        balanced_instance_loss(torch.ones(1, 2), result, torch.ones(1, 2), ACTIVE)
        torch.testing.assert_close(before, torch.random.get_rng_state())


if __name__ == "__main__":
    unittest.main()
