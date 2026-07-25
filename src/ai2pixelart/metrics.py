"""Evaluation metrics for pixel-art restoration.

All metrics compare a predicted true-resolution image against a ground-truth
true-resolution image (both (H, W, 3) uint8). ΔE values are CIE76.

The headline metrics:
- cell accuracy: fraction of cells with the right color (exact / ΔE-tolerant)
- palette fidelity: Hungarian-matched palette ΔE + palette size delta
- detail retention: survival rate of isolated single-cell features (the
  "one-pixel white eye" problem) — the metric this project lives or dies by.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ai2pixelart.palette import delta_e, image_palette, rgb_to_lab


def cell_accuracy(pred: np.ndarray, gt: np.ndarray, tol_de: float = 2.0) -> dict:
    _check_shapes(pred, gt)
    exact = float((pred == gt).all(axis=-1).mean())
    de = delta_e(rgb_to_lab(pred), rgb_to_lab(gt))
    return {"exact": exact, "tolerant": float((de <= tol_de).mean()), "mean_de": float(de.mean())}


def palette_fidelity(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Match palettes with the Hungarian algorithm and report the mean ΔE of
    matched pairs plus the size mismatch. Works on differently sized images."""
    pal_pred, pal_gt = image_palette(pred), image_palette(gt)
    cost = delta_e(
        rgb_to_lab(pal_pred).reshape(-1, 1, 3), rgb_to_lab(pal_gt).reshape(1, -1, 3)
    )
    rows, cols = linear_sum_assignment(cost)
    return {
        "size_pred": int(len(pal_pred)),
        "size_gt": int(len(pal_gt)),
        "matched_mean_de": float(cost[rows, cols].mean()),
        "unmatched": int(abs(len(pal_pred) - len(pal_gt))),
    }


def isolated_details(gt: np.ndarray, isolation_de: float = 10.0) -> np.ndarray:
    """Boolean mask of GT cells that differ from ALL existing 8-neighbors by
    more than `isolation_de` — single-cell details like a 1-px eye highlight."""
    lab = rgb_to_lab(gt)
    h, w = lab.shape[:2]
    isolated = np.ones((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            de = np.full((h, w), np.inf)
            ys = slice(max(dy, 0), h + min(dy, 0))
            yd = slice(max(-dy, 0), h + min(-dy, 0))
            xs = slice(max(dx, 0), w + min(dx, 0))
            xd = slice(max(-dx, 0), w + min(-dx, 0))
            de[yd, xd] = delta_e(lab[yd, xd], lab[ys, xs])
            isolated &= de > isolation_de
    return isolated


def detail_retention(
    pred: np.ndarray, gt: np.ndarray, isolation_de: float = 10.0, match_de: float = 5.0
) -> dict:
    """Of the isolated single-cell details in GT, how many survive in pred?"""
    _check_shapes(pred, gt)
    mask = isolated_details(gt, isolation_de=isolation_de)
    n = int(mask.sum())
    if n == 0:
        return {"n_details": 0, "retained": 0, "rate": float("nan")}
    de = delta_e(rgb_to_lab(pred), rgb_to_lab(gt))
    retained = int((de[mask] <= match_de).sum())
    return {"n_details": n, "retained": retained, "rate": retained / n}


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Aggregate metric dict. A wrong output size is a failure mode we want
    recorded, not an exception, so cell metrics are skipped on mismatch."""
    result: dict = {
        "size_match": pred.shape == gt.shape,
        "shape_pred": list(pred.shape[:2]),
        "shape_gt": list(gt.shape[:2]),
        "palette": palette_fidelity(pred, gt),
    }
    if result["size_match"]:
        result["cells"] = cell_accuracy(pred, gt)
        result["details"] = detail_retention(pred, gt)
    return result


def _check_shapes(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
