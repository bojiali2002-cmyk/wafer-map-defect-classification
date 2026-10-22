
"""Preprocessing utilities for WM-811K wafer maps."""

import numpy as np
import torch
import torch.nn.functional as F


TARGET_SIZE = 64


def preprocess_wafer_map(wafer_map, target_size=TARGET_SIZE):
    """Pad, resize, normalize, and convert a wafer map to a tensor."""

    wafer_map = np.asarray(wafer_map, dtype=np.float32)

    height, width = wafer_map.shape
    square_size = max(height, width)

    padded_map = np.zeros(
        (square_size, square_size),
        dtype=np.float32,
    )

    top = (square_size - height) // 2
    left = (square_size - width) // 2

    padded_map[
        top:top + height,
        left:left + width,
    ] = wafer_map

    tensor = torch.from_numpy(padded_map)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    resized_tensor = F.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="nearest",
    )

    normalized_tensor = resized_tensor.squeeze(0) / 2.0

    return normalized_tensor
