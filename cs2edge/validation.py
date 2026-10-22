"""Contrats numeriques communs aux modeles et a leur evaluation."""

from __future__ import annotations

import math

import numpy as np


def finite_number(name: str, value: float, *, minimum: float = 0.0,
                  strict: bool = False, maximum: float | None = None) -> None:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < minimum
            or (strict and value == minimum) or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} : valeur hors limites ou non finie")


def positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} doit etre un entier strictement positif")


def binary_labels(y) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.all((y == 0) | (y == 1)):
        raise ValueError("y doit etre un vecteur non vide de resultats binaires (0 ou 1)")
    return y


def probability_pair(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = binary_labels(y)
    p = np.asarray(p, dtype=float)
    if (p.shape != y.shape or not np.all(np.isfinite(p))
            or np.any((p < 0) | (p > 1))):
        raise ValueError("p doit etre un vecteur de probabilites dans [0, 1], de meme taille que y")
    return y, p


def feature_matrix(X, *, rows: int | None = None, columns: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if (X.ndim != 2 or 0 in X.shape or not np.all(np.isfinite(X))
            or (rows is not None and X.shape[0] != rows)
            or (columns is not None and X.shape[1] != columns)):
        raise ValueError("X doit etre une matrice finie non vide, de dimensions compatibles")
    return X


def observation_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=float)
    if (w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0)
            or not np.isfinite(w.sum()) or w.sum() <= 0):
        raise ValueError("sample_weight doit etre un vecteur positif ou nul, fini, de somme positive")
    return w
