"""Picklable transform functions used inside the saved model pipelines."""
import numpy as np


def log1p_clip(x):
    """log(1 + x) for non-negative counts and money."""
    return np.log1p(np.clip(x, 0, None))


def log_roi(x):
    """log(1 + ROI): turns Revenue = Spend x (1 + ROI) into a sum of logs."""
    return np.log1p(np.clip(x, -0.999, None))
