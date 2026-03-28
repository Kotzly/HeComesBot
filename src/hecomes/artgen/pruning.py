import numpy as np


def image_entropy(frame: np.ndarray) -> float:
    """Shannon entropy of a rendered uint8 frame (all channels flattened)."""
    pixels = frame.reshape(-1)
    counts = np.bincount(pixels, minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def entropy_sensitivity(original: np.ndarray, perturbed: np.ndarray, threshold: float) -> bool:
    """Return True if the child should be pruned (entropy change below threshold)."""
    return abs(image_entropy(perturbed) - image_entropy(original)) < threshold


PRUNE_METHODS: dict = {
    "entropy_sensitivity": entropy_sensitivity,
}
