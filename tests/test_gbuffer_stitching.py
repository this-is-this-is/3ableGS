"""Check recovered overlap geometry and invalid-normal behavior without a model."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from tools import stitch_tiled_gbuffers as stitch


class GbufferStitchingTests(unittest.TestCase):
    def setUp(self):
        # Keep the same overlap topology with tiny arrays for behavior checks.
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in {"FULL_SIZE": (4, 6), "CROP_SIZE": (4, 5),
                            "BOTTOM_Y": 1, "OVERLAP_START": 1, "OVERLAP_END": 5}.items():
            self.stack.enter_context(patch.object(stitch, name, value))

    def test_feather_preserves_nonoverlap_and_has_no_reversed_seam(self):
        top = np.zeros((5, 4, 3), dtype=np.float32)
        bottom = np.ones_like(top)
        result = stitch.blend(top, bottom, "basecolor")
        np.testing.assert_array_equal(result[0], 0)
        np.testing.assert_array_equal(result[-1], 255)
        self.assertTrue(np.all(np.diff(result[:, 0, 0].astype(int)) >= 0))
        self.assertGreater(result[2, 0, 0], 0)
        self.assertLess(result[3, 0, 0], 255)

    def test_neutral_quantized_normals_do_not_turn_into_sky_directions(self):
        invalid = np.full((5, 4, 3), 128 / 255, dtype=np.float32)
        result = stitch.blend(invalid, invalid, "normal")
        np.testing.assert_array_equal(result, 128)

    def test_invalid_crop_does_not_dilute_valid_overlap_normal(self):
        top = np.full((5, 4, 3), 0.5, dtype=np.float32)
        top[..., 2] = 1.0
        invalid = np.full_like(top, 128 / 255)
        result = stitch.blend(top, invalid, "normal")
        np.testing.assert_array_equal(result[2, 0], [128, 128, 255])
        np.testing.assert_array_equal(result[-1, 0], [128, 128, 128])


if __name__ == "__main__":
    unittest.main()
