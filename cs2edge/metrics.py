"""Metriques d'evaluation. Le log-loss prime : on veut des probabilites, pas des verdicts."""

from __future__ import annotations

import datetime as dt

import numpy as np

from .validation import positive_integer, probability_pair

EPS = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y, p = probability_pair(y, p)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = probability_pair(y, p)
    return float(np.mean((p - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y, p = probability_pair(y, p)
    return float(np.mean((p >= 0.5).astype(float) == y))


def baseline_log_loss(y: np.ndarray) -> float:
    """Log-loss du modele 'pile ou face'. C'est le seuil a battre."""
    return log_loss(y, np.full(len(y), 0.5))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = probability_pair(y, p)
    positive_integer("bins", bins)
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
    y, p = probability_pair(y, p)
    positive_integer("bins", bins)
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


def paired_block_bootstrap(y, p, reference, dates: list[dt.date], *,
                           samples: int = 1000, block_days: int = 7, seed: int = 0) -> dict:
    """IC percentile apparie du gain de log-loss, par blocs calendaires.

    Un bloc est tire entier : les matchs proches ne sont pas traites comme
    autant de preuves independantes. Les deux modeles voient les memes tirages.
    L'IC reste conditionnel a la periode et aux modeles deja evalues.
    """
    y, p = probability_pair(y, p)
    _, reference = probability_pair(y, reference)
    positive_integer("samples", samples)
    positive_integer("block_days", block_days)
    if len(dates) != len(y) or any(type(d) is not dt.date for d in dates):
        raise ValueError("une date calendaire est requise pour chaque resultat")
    p = np.clip(p, EPS, 1 - EPS)
    reference = np.clip(reference, EPS, 1 - EPS)
    gain = y * np.log(p / reference) + (1 - y) * np.log((1 - p) / (1 - reference))
    _, blocks = np.unique([d.toordinal() // block_days for d in dates], return_inverse=True)
    counts = np.bincount(blocks)
    totals = np.bincount(blocks, weights=gain)
    out = dict(gain=float(gain.mean()), low=None, high=None, blocks=len(counts),
               block_days=block_days, samples=samples, confidence=.95, seed=seed)
    if len(counts) < 2:
        return out
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, samples, 64):
        chosen = rng.integers(0, len(counts), size=(min(64, samples-start), len(counts)))
        draws.extend((totals[chosen].sum(axis=1) / counts[chosen].sum(axis=1)).tolist())
    out["low"], out["high"] = map(float, np.quantile(draws, [.025, .975]))
    return out
