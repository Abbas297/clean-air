import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def fit_cqr_calibration(
    y_true_val: np.ndarray,
    q10_pred_val: np.ndarray,
    q90_pred_val: np.ndarray,
    coverage_level: float = 0.80,
) -> float:
    """
    Fits Split-Conformal Quantile Regression (CQR) correction on validation data.
    Nonconformity score: E_i = max(q10_i - y_i, y_i - q90_i)
    Returns conformal correction factor Q.
    """
    valid_mask = (~np.isnan(y_true_val)) & (~np.isnan(q10_pred_val)) & (~np.isnan(q90_pred_val))
    y = y_true_val[valid_mask]
    q10 = q10_pred_val[valid_mask]
    q90 = q90_pred_val[valid_mask]

    n = len(y)
    if n < 20:
        return 0.0

    # Nonconformity scores
    scores = np.maximum(q10 - y, y - q90)

    # Conformal quantile level
    p_level = np.ceil((n + 1) * coverage_level) / n
    p_level = min(1.0, max(0.0, p_level))

    q_correction = float(np.quantile(scores, p_level, method="higher"))
    return q_correction

def apply_cqr_adjustment(
    q10_preds: np.ndarray,
    q90_preds: np.ndarray,
    q_correction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies conformal correction factor Q to prediction bands:
    [q10_adj, q90_adj] = [q10 - Q, q90 + Q], clipped to non-negative physical values.
    """
    q10_adj = np.clip(q10_preds - q_correction, 0.0, None)
    q90_adj = np.clip(q90_preds + q_correction, 0.0, None)
    return q10_adj, q90_adj

def evaluate_interval_coverage(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> Dict[str, float]:
    """Computes empirical coverage rate and mean interval width."""
    valid_mask = (~np.isnan(y_true)) & (~np.isnan(q_lower)) & (~np.isnan(q_upper))
    y = y_true[valid_mask]
    lo = q_lower[valid_mask]
    hi = q_upper[valid_mask]

    if len(y) == 0:
        return {"coverage": 0.0, "mean_width": 0.0, "n_samples": 0}

    covered = (y >= lo) & (y <= hi)
    coverage = float(np.mean(covered))
    mean_width = float(np.mean(hi - lo))

    return {
        "coverage": round(coverage, 4),
        "mean_width": round(mean_width, 2),
        "n_samples": int(len(y)),
    }
