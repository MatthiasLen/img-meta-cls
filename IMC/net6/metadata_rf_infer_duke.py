"""
Metadata RF inference utilities for Duke dataset.
"""

import joblib
import numpy as np
import torch


def load_rf_model(path: str):
    return joblib.load(path)


def rf_predict_proba(rf_model, meta_row: np.ndarray, class_names: list[str]) -> torch.Tensor:
    """Return probability vector for one sample as torch tensor (1, C)."""
    # rf_model is a pipeline with scaler+rf
    probs = rf_model.predict_proba(meta_row.reshape(1, -1))[0]
    # Some sklearn RFs return list-of-arrays for multi-label; we assume single-label task here
    return torch.tensor(probs, dtype=torch.float32).unsqueeze(0)
