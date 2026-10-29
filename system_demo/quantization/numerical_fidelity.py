"""Matched-input numerical fidelity metrics; distinct from task recognition accuracy."""
import numpy as np


def unit_normalize(embeddings):
    """Explicitly match a unit-output contract; never silently alter raw metrics."""
    values = np.asarray(embeddings, np.float64)
    if values.ndim != 2 or not values.shape[0] or not np.isfinite(values).all():
        raise ValueError("finite nonempty embedding matrix required")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero-norm embedding")
    return values / norms


def summarize(reference, candidate):
    reference = np.asarray(reference, np.float64)
    candidate = np.asarray(candidate, np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[0] == 0:
        raise ValueError("nonempty matched embedding matrices required")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite embedding")
    norms = np.linalg.norm(reference, axis=1)
    candidate_norms = np.linalg.norm(candidate, axis=1)
    if (norms == 0).any() or (candidate_norms == 0).any():
        raise ValueError("zero-norm embedding")
    difference = candidate - reference
    cosine = np.clip(np.sum(reference * candidate, axis=1) / (norms * candidate_norms), -1., 1.)
    error_energy = np.sum(difference ** 2, axis=1)
    signal_energy = np.sum(reference ** 2, axis=1)
    relative_l2 = np.sqrt(error_energy / signal_energy)
    normalized_l2 = np.linalg.norm(reference / norms[:, None] - candidate / candidate_norms[:, None], axis=1)
    squared_error = float(error_energy.sum())
    sqnr = float(10 * np.log10(signal_energy.sum() / squared_error)) if squared_error else "inf"
    return {"count": len(reference), "embedding_dim": reference.shape[1],
        "cosine_mean": float(cosine.mean()), "cosine_median": float(np.median(cosine)),
        "cosine_min": float(cosine.min()), "cosine_p05": float(np.percentile(cosine, 5)),
        "raw_mae": float(np.abs(difference).mean()), "raw_rmse": float(np.sqrt(np.mean(difference ** 2))),
        "raw_max_abs_error": float(np.abs(difference).max()),
        "nrmse_l2_global": float(np.sqrt(error_energy.sum() / signal_energy.sum())),
        "relative_l2_mean": float(relative_l2.mean()), "relative_l2_p95": float(np.percentile(relative_l2, 95)),
        "normalized_embedding_l2_mean": float(normalized_l2.mean()), "sqnr_db_global": sqnr,
        "norm_ratio_mean": float((candidate_norms / norms).mean()),
        "per_sample": [{"cosine": float(cosine[i]), "raw_rmse": float(np.sqrt(np.mean(difference[i] ** 2))),
            "relative_l2": float(relative_l2[i]), "normalized_l2": float(normalized_l2[i])}
            for i in range(len(reference))]}
