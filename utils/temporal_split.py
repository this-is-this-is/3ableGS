"""Validated temporal holdouts and train-only frame correspondences.

Frame IDs here are relative to the loaded sequence. Ranges use an exclusive
end, independently of the dataset's inclusive ``end_timestep`` setting.
"""

from bisect import bisect_left
from numbers import Integral


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def resolve_test_frames(num_frames, test_frame_ranges=None, test_image_stride=0):
    """Return sorted held-out IDs; explicit ranges override the legacy stride."""
    num_frames = _integer(num_frames, "num_frames", 1)
    if test_frame_ranges is None:
        stride = _integer(test_image_stride, "test_image_stride")
        return list(range(stride, num_frames, stride)) if stride else []
    if isinstance(test_frame_ranges, (str, bytes, dict)):
        raise ValueError("test_frame_ranges must be a sequence of [start, end] ranges.")
    try:
        ranges = list(test_frame_ranges)
    except TypeError as exc:
        raise ValueError("test_frame_ranges must be a sequence of [start, end] ranges.") from exc
    result = []
    previous_end = 0
    for index, span in enumerate(ranges):
        try:
            if isinstance(span, (str, bytes, dict)) or len(span) != 2:
                raise ValueError
            start = _integer(span[0], f"test_frame_ranges[{index}].start")
            end = _integer(span[1], f"test_frame_ranges[{index}].end", 1)
        except (TypeError, ValueError) as exc:
            raise ValueError("Each test frame range must contain two integer endpoints.") from exc
        if start >= end or end > num_frames:
            raise ValueError(f"Invalid test frame range [{start}, {end}) for {num_frames} frames.")
        if index and start < previous_end:
            raise ValueError("test_frame_ranges must be sorted and disjoint.")
        result.extend(range(start, end))
        previous_end = end
    if len(result) == num_frames:
        raise ValueError("test_frame_ranges must leave at least one training frame.")
    return result


def training_bracket(frame, train_frame_ids):
    """Return (left, right, alpha) using actual train IDs only.

    Outside the training interval, return the nearest train endpoint twice.
    Within a holdout, alpha accounts for the full distance between train
    observations, including a multi-frame held-out interval.
    """
    frame = _integer(frame, "frame")
    train_ids = sorted(set(_integer(value, "train_frame_id") for value in train_frame_ids))
    if not train_ids:
        raise ValueError("At least one train frame is required.")
    index = bisect_left(train_ids, frame)
    if index == len(train_ids):
        return train_ids[-1], train_ids[-1], 0.
    if train_ids[index] == frame or index == 0:
        return train_ids[index], train_ids[index], 0.
    left, right = train_ids[index - 1], train_ids[index]
    return left, right, (frame - left) / (right - left)


def previous_train_image_index(image_idx, train_indices, num_cams, max_gap=1):
    """Pick the nearest earlier train image of the same camera, or None."""
    image_idx = _integer(image_idx, "image_idx")
    num_cams = _integer(num_cams, "num_cams", 1)
    max_gap = _integer(max_gap, "max_gap", 1)
    allowed = train_indices if isinstance(train_indices, (set, frozenset)) else set(train_indices)
    if image_idx not in allowed:
        raise ValueError("Temporal training cannot use a held-out current image.")
    for gap in range(1, max_gap + 1):
        previous = image_idx - gap * num_cams
        if previous < 0:
            break
        if previous in allowed:
            return previous
    return None
