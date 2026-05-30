import numpy as np


def normalize_scores(scores):
    scores = np.array(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores
    min_v = scores.min()
    max_v = scores.max()
    if max_v - min_v < 1e-6:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)
