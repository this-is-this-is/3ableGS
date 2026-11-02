"""Actual feature-node CPU paths for contiguous heldout pose interpolation."""

import math
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.gaussians import FeatureGaussians
from models.nodes.rigid_feature import RigidNodes_feature
from models.nodes.smpl_feature import SMPLNodes_feature


class TemporalPoseInterpolationTests(unittest.TestCase):
    @staticmethod
    def node(cls=RigidNodes_feature, instances=3):
        node = cls.__new__(cls)
        torch.nn.Module.__init__(node)
        node.instances_fv = torch.ones(10, instances, dtype=torch.bool)
        node.instances_trans = torch.zeros(10, instances, 3)
        node.instances_trans[..., 0] = torch.arange(10.)[:, None]
        node.instances_quats = torch.zeros(10, instances, 4)
        node.instances_quats[..., 0] = 1
        node.point_ids = torch.arange(instances)[:, None]
        node.cur_frame = 4
        node.in_test_set = True
        node.train_frame_ids = (0, 2, 8, 9)
        node.device = torch.device("cpu")
        return node

    def test_uses_actual_train_brackets_and_frame_distance(self):
        node = self.node()
        node.instances_trans[3:8] = 999
        result = node.transform_means(torch.zeros(3, 3))
        torch.testing.assert_close(result[:, 0], torch.full((3,), 4.))
        left, right, alpha, known = node._training_pose_brackets()
        self.assertEqual(left.tolist(), [2, 2, 2])
        self.assertEqual(right.tolist(), [8, 8, 8])
        torch.testing.assert_close(alpha, torch.full((3,), 1 / 3))
        self.assertTrue(known.all())

    def test_each_instance_uses_its_own_valid_training_frames(self):
        node = self.node()
        node.instances_fv[:, 1:] = False
        node.instances_fv[8, 1] = True
        node.instances_fv[4, 1:] = True
        node.instances_trans[4, 2] = 999
        result = node.transform_means(torch.zeros(3, 3))
        torch.testing.assert_close(result[:, 0], torch.tensor([4., 8., 0.]))
        self.assertEqual(node.get_pts_valid_mask().tolist(), [True, True, False])
        quats = node._pose_at_current_frame(node.instances_quats, quaternion=True)
        torch.testing.assert_close(quats[2], torch.tensor([1., 0., 0., 0.]))

    def test_quaternion_and_means_share_the_same_interpolated_rotation(self):
        node = self.node(instances=1)
        angle = math.pi / 2
        node.instances_quats[8, 0] = torch.tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
        node.instances_quats[4, 0] = torch.tensor([0., 1., 0., 0.])  # forbidden heldout rotation
        point = torch.tensor([[1., 0., 0.]])
        result = node.transform_means(point)
        expected = torch.tensor([[4 + math.cos(angle / 3), math.sin(angle / 3), 0.]])
        torch.testing.assert_close(result, expected)
        world_q = node.transform_quats(torch.tensor([[1., 0., 0., 0.]]))
        expected_q = torch.tensor([[math.cos(angle / 6), 0., 0., math.sin(angle / 6)]])
        torch.testing.assert_close(world_q, expected_q)

    def test_endpoint_holds_nearest_known_pose_and_cache_tracks_visibility_change(self):
        node = self.node(instances=1)
        node.train_frame_ids = (2, 8)
        node.cur_frame = 0
        self.assertEqual(node._pose_at_current_frame(node.instances_trans)[0, 0], 2)
        node.cur_frame = 9
        self.assertEqual(node._pose_at_current_frame(node.instances_trans)[0, 0], 8)
        node.instances_fv[8, 0] = False
        self.assertEqual(node._pose_at_current_frame(node.instances_trans)[0, 0], 2)

    def test_train_frames_keep_direct_parameter_access(self):
        node = self.node()
        node.in_test_set = False
        node.instances_trans.requires_grad_(True)
        result = node._pose_at_current_frame(node.instances_trans)
        result.sum().backward()
        self.assertEqual(node.instances_trans.grad[4].sum(), 9)
        self.assertEqual(node.instances_trans.grad[:4].sum(), 0)

    def test_legacy_callers_keep_adjacent_interpolation_without_train_ids(self):
        node = self.node()
        del node.train_frame_ids
        node.instances_trans[3] = 2
        node.instances_trans[5] = 8
        torch.testing.assert_close(node._pose_at_current_frame(node.instances_trans), torch.full((3, 3), 5.))

    def test_smpl_root_joints_and_translation_use_train_only_poses(self):
        node = self.node(SMPLNodes_feature, instances=2)
        node.instances_quats = node.instances_quats.unsqueeze(2)
        node.smpl_qauts = node.instances_quats.expand(-1, -1, 23, -1).clone()
        node.smpl_qauts[3:8] = 0
        node.smpl_qauts[3:8, ..., 1] = 1
        node.instances_trans[3:8] = 999
        node.instances_fv[:, 1] = False
        node.instances_fv[4, 1] = True
        node.smpl_points_num = 2
        node.point_ids = torch.tensor([[0], [0], [1], [1]])
        node.use_voxel_deformer = False
        captured = []

        def template(masked_theta, instances_mask, xyz_canonical):
            captured.append(masked_theta.clone())
            self.assertEqual(instances_mask.tolist(), [True, False])
            return torch.ones(1, 2, 1), torch.eye(4).reshape(1, 1, 4, 4)

        node.template = template
        means = node.transform_means(torch.zeros(4, 3))
        torch.testing.assert_close(means[:, 0], torch.tensor([4., 4., 0., 0.]))
        torch.testing.assert_close(captured[0][..., 0], torch.ones(1, 24))
        quats = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
        means_b, quats_b = node.transform_means_and_quats(torch.zeros(4, 3), quats)
        torch.testing.assert_close(means_b, means)
        torch.testing.assert_close(quats_b, quats)

    def test_regularizer_does_not_read_heldout_pose_endpoints(self):
        node = self.node()
        node.in_test_set = False
        node.train_frame_ids = (0, 2, 4, 8, 9)
        node.reg_cfg = OmegaConf.create({"temporal_smooth_reg": {"trans": {"smooth_range": 5, "w": 1.}}})
        node.instances_trans[6] = 1000
        with patch.object(FeatureGaussians, "compute_reg_loss", return_value={}), \
                patch("models.nodes.rigid_feature.random.randint", return_value=2):
            self.assertNotIn("trans_temporal_smooth", node.compute_reg_loss("raw"))
            node.train_frame_ids += (6,)
            self.assertGreater(node.compute_reg_loss("raw")["trans_temporal_smooth"], 0)


if __name__ == "__main__":
    unittest.main()
