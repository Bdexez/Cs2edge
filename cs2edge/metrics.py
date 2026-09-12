"""Metriques d'evaluation. Le log-loss prime : on veut des probabilites, pas des verdicts."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) >= 0.5).astype(float) == np.asarray(y)))


def baseline_log_loss(y: np.ndarray) -> float:
    """Log-loss du modele 'pile ou face'. C'est le seuil a battre."""
    return log_loss(y, np.full(len(y), 0.5))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / max(1, len(y)))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Courbe de fiabilite : proba annoncee vs frequence observee."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(m.sum()),
                "predicted": float(p[m].mean()),
                "observed": float(y[m].mean()),
            }
        )
    return rows


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    ll = log_loss(y, p)
    base = baseline_log_loss(y)
    return {
        "n": int(len(y)),
        "log_loss": ll,
        "baseline_log_loss": base,
        # Part de l'incertitude expliquee par rapport a une piece equilibree.
        "skill_score": float(1.0 - ll / base) if base > 0 else 0.0,
        "brier": brier(y, p),
        "accuracy": accuracy(y, p),
        "ece": expected_calibration_error(y, p),
    }


def format_summary(name: str, s: dict) -> str:
    return (
        f"{name:<22} n={s['n']:<6} logloss={s['log_loss']:.4f} "
        f"(base {s['baseline_log_loss']:.4f}, skill {s['skill_score']:+.1%})  "
        f"acc={s['accuracy']:.1%}  brier={s['brier']:.4f}  ece={s['ece']:.3f}"
    )
