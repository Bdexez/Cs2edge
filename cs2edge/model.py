"""Modele de prediction.

Coeur : regression logistique sans constante, entrainee par IRLS avec ridge,
sur des features antisymetriques. Sans constante et avec augmentation miroir
(X, y) -> (-X, 1-y), le modele verifie exactement p(A|B) = 1 - p(B|A) : aucun
biais d'ordre des equipes, ce qui est la premiere source d'erreur silencieuse
dans ce genre de systeme.

Option : un GBDT (scikit-learn) capte les non-linearites ; on melange les deux
en espace logit, puis on recalibre par temperature sur un holdout temporel.
Une probabilite mal calibree est pire qu'inutile des qu'on la compare a une cote.
"""

from __future__ import annotations

import datetime as dt
import json
import hashlib
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import log_loss
from .temporal import calibration_boundary, validate_dates

RECENCY_HALFLIFE_DAYS = 540.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def recency_weights(dates: list[dt.date], halflife: float = RECENCY_HALFLIFE_DAYS) -> np.ndarray:
    """Le meta CS2 bouge : un match d'il y a deux ans en dit moins que celui d'hier."""
    if not dates:
        return np.ones(0)
    last = max(dates)
    ages = np.array([(last - d).days for d in dates], dtype=float)
    return 0.5 ** (ages / halflife)


@dataclass
class LogisticModel:
    """Regression logistique ridge, sans constante, resolue par IRLS."""

    l2: float = 2.0
    max_iter: int = 60
    tol: float = 1e-8
    residualize: bool = True
    weights: np.ndarray | None = None
    scales: np.ndarray | None = None
    resid: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()

    def _fit_residualizer(self, X: np.ndarray) -> None:
        """Projette chaque colonne hors de la colonne 0 (l'ecart de niveau).

        Toutes les features sont fortement correlees a `elo_diff` : forme, pool,
        calendrier disent en grande partie la meme chose. Sans decorrelation, le
        ridge repartit ce signal commun au hasard entre les colonnes et les
        coefficients deviennent ininterpretables (`elo_diff` peut meme ressortir
        negatif). La transformation est lineaire et inversible : elle ne change
        pas ce que le modele peut representer, seulement la facon dont la
        penalite repartit le credit. Ce qui reste dans chaque colonne est sa
        contribution *nette* de l'ecart de niveau.
        """
        if not self.residualize or X.shape[1] < 2:
            self.resid = np.zeros(max(0, X.shape[1] - 1))
            return
        base = X[:, 0]
        denom = float(base @ base)
        if denom < 1e-12:
            self.resid = np.zeros(X.shape[1] - 1)
            return
        self.resid = np.array([float(base @ X[:, j]) / denom for j in range(1, X.shape[1])])

    def _apply_residual(self, X: np.ndarray) -> np.ndarray:
        if self.resid is None or len(self.resid) == 0:
            return X
        out = np.array(X, dtype=float, copy=True)
        out[:, 1:] -= np.outer(out[:, 0], self.resid)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X doit etre une matrice non vide")
        self._fit_residualizer(X)
        X = self._apply_residual(X)
        # Mise a l'echelle sans centrage : centrer casserait l'antisymetrie.
        self.scales = np.maximum(np.sqrt(np.mean(X**2, axis=0)), 1e-6)
        Xs = X / self.scales

        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        # Augmentation miroir : impose la symetrie et double l'echantillon.
        Xa = np.vstack([Xs, -Xs])
        ya = np.concatenate([y, 1.0 - y])
        wa = np.concatenate([w, w])

        beta = np.zeros(Xa.shape[1])
        for _ in range(self.max_iter):
            p = sigmoid(Xa @ beta)
            s = wa * np.clip(p * (1 - p), 1e-9, None)
            grad = Xa.T @ (wa * (ya - p)) - self.l2 * beta
            hess = (Xa.T * s) @ Xa + self.l2 * np.eye(Xa.shape[1])
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            beta += step
            if np.max(np.abs(step)) < self.tol:
                break
        self.weights = beta
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.scales is None:
            raise RuntimeError("modele non entraine")
        Xr = self._apply_residual(np.asarray(X, dtype=float))
        return (Xr / self.scales) @ self.weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision(X))

    def coefficients(self) -> list[tuple[str, float]]:
        """Poids ramenes a l'echelle des features standardisees (comparables entre eux)."""
        if self.weights is None:
            return []
        names = self.feature_names or tuple(f"f{i}" for i in range(len(self.weights)))
        return sorted(zip(names, self.weights.tolist()), key=lambda kv: -abs(kv[1]))


def _fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Calibration par temperature : preserve l'ordre et la symetrie, 1 parametre."""
    if len(y) < 30 or len(np.unique(y)) < 2:
        return 1.0
    lo, hi = 0.3, 4.0
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if log_loss(y, sigmoid(z / m1)) < log_loss(y, sigmoid(z / m2)):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


@dataclass
class MatchPredictor:
    """Ensemble logistique (+ GBDT optionnel) avec calibration temperature."""

    l2: float = 2.0
    use_gbdt: bool = True
    gbdt_weight: float = 0.35
    calibration_frac: float = 0.2
    recency_halflife: float = RECENCY_HALFLIFE_DAYS
    logistic: LogisticModel = field(default_factory=LogisticModel)
    gbdt: object | None = None
    temperature: float = 1.0
    feature_names: tuple[str, ...] = ()
    trained_on: int = 0
    last_date: str = ""

    # ------------------------------------------------------------------- train

    def fit(self, X: np.ndarray, y: np.ndarray, dates: list[dt.date] | None = None,
            feature_names: tuple[str, ...] = ()) -> "MatchPredictor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(y) < 50:
            raise ValueError(f"trop peu de matchs pour entrainer ({len(y)}), il en faut ~50+")
        if dates is not None:
            if len(dates) != len(y):
                raise ValueError("une date est requise pour chaque resultat")
            validate_dates(dates)
        if not 0 <= self.calibration_frac < 1:
            raise ValueError("calibration_frac doit etre dans [0, 1[")
        self.feature_names = feature_names or self.feature_names
        self.trained_on = len(y)
        if dates:
            self.last_date = max(dates).isoformat()

        # Holdout temporel (les derniers matchs) pour la calibration.
        n_cal = int(len(y) * self.calibration_frac)
        n_cal = 0 if n_cal < 40 else n_cal
        n_fit = len(y) - n_cal
        if dates and n_cal:
            n_fit = calibration_boundary(dates, n_fit)
            n_cal = len(y) - n_fit
        Xf, yf = X[:n_fit], y[:n_fit]
        w = recency_weights(dates[:n_fit], self.recency_halflife) if dates else None

        self.logistic = LogisticModel(l2=self.l2, feature_names=self.feature_names)
        self.logistic.fit(Xf, yf, sample_weight=w)

        self.gbdt = self._fit_gbdt(Xf, yf, w) if self.use_gbdt else None

        if n_cal:
            z = self._raw_decision(X[n_fit:])
            self.temperature = _fit_temperature(z, y[n_fit:])
        else:
            self.temperature = 1.0
        return self

    def _fit_gbdt(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None):
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError:
            return None
        if len(y) < 400:  # sous ce volume le GBDT sur-apprend plus qu'il n'apporte
            return None
        Xa = np.vstack([X, -X])
        ya = np.concatenate([y, 1.0 - y])
        wa = np.concatenate([w, w]) if w is not None else None
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=250, learning_rate=0.05,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )
        clf.fit(Xa, ya, sample_weight=wa)
        return clf

    # ----------------------------------------------------------------- predict

    def _gbdt_proba(self, X: np.ndarray) -> np.ndarray:
        # Symetrisation explicite : le GBDT n'est pas antisymetrique par nature.
        p_fwd = self.gbdt.predict_proba(X)[:, 1]
        p_rev = self.gbdt.predict_proba(-X)[:, 1]
        return 0.5 * (p_fwd + (1.0 - p_rev))

    def _raw_decision(self, X: np.ndarray) -> np.ndarray:
        z = self.logistic.decision(X)
        if self.gbdt is not None:
            z = (1 - self.gbdt_weight) * z + self.gbdt_weight * logit(self._gbdt_proba(X))
        return z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return sigmoid(self._raw_decision(X) / max(self.temperature, 1e-6))

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))[0])

    def explain(self, x: np.ndarray, top: int = 6) -> list[tuple[str, float]]:
        """Contribution de chaque feature au logit, en points de logit."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        x = self.logistic._apply_residual(x).ravel()
        contrib = (x / self.logistic.scales) * self.logistic.weights
        contrib = contrib * (1 - self.gbdt_weight if self.gbdt is not None else 1.0)
        contrib = contrib / max(self.temperature, 1e-6)
        names = self.feature_names or tuple(f"f{i}" for i in range(len(contrib)))
        pairs = sorted(zip(names, contrib.tolist()), key=lambda kv: -abs(kv[1]))
        return pairs[:top]

    # -------------------------------------------------------------------- I/O

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "version": 1,
            "l2": self.l2,
            "gbdt_weight": self.gbdt_weight if self.gbdt is not None else 0.0,
            "temperature": self.temperature,
            "feature_names": list(self.feature_names),
            "weights": self.logistic.weights.tolist(),
            "scales": self.logistic.scales.tolist(),
            "resid": self.logistic.resid.tolist() if self.logistic.resid is not None else [],
            "trained_on": self.trained_on,
            "last_date": self.last_date,
            "has_gbdt": self.gbdt is not None,
        }
        if self.gbdt is not None:
            blob = pickle.dumps(self.gbdt)
            path.with_suffix(".gbdt.pkl").write_bytes(blob)
            payload["gbdt_sha256"] = hashlib.sha256(blob).hexdigest()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MatchPredictor":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("version de modele non prise en charge")
        weights = np.asarray(payload["weights"], dtype=float)
        scales = np.asarray(payload["scales"], dtype=float)
        resid = np.asarray(payload.get("resid") or [], dtype=float)
        if (weights.ndim != 1 or len(weights) == 0 or scales.shape != weights.shape
                or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(scales))
                or np.any(scales <= 0) or resid.shape not in ((0,), (len(weights)-1,))
                or not np.all(np.isfinite(resid))
                or len(payload["feature_names"]) not in (0, len(weights))
                or not math.isfinite(payload["temperature"]) or payload["temperature"] <= 0):
            raise ValueError("parametres de modele invalides")
        m = cls(l2=payload["l2"], gbdt_weight=payload["gbdt_weight"])
        m.temperature = payload["temperature"]
        m.feature_names = tuple(payload["feature_names"])
        m.trained_on = payload.get("trained_on", 0)
        m.last_date = payload.get("last_date", "")
        m.logistic = LogisticModel(l2=payload["l2"], feature_names=m.feature_names)
        m.logistic.weights, m.logistic.scales, m.logistic.resid = weights, scales, resid
        blob = path.with_suffix(".gbdt.pkl")
        if payload.get("has_gbdt"):
            if not blob.is_file():
                raise ValueError(f"composant GBDT manquant : {blob}")
            data = blob.read_bytes()
            if payload.get("gbdt_sha256") and hashlib.sha256(data).hexdigest() != payload["gbdt_sha256"]:
                raise ValueError("composant GBDT incoherent avec le modele JSON")
            m.gbdt = pickle.loads(data)
        else:
            m.gbdt = None
            m.gbdt_weight = 0.0
        return m


def map_predictor(**kwargs) -> MatchPredictor:
    """Configuration map commune au service et aux deux backtests."""
    return MatchPredictor(**{**kwargs, "use_gbdt": False})


def series_win_probability(p_map: float, best_of: int) -> float:
    """Passe d'une proba par map a une proba de serie (maps supposees i.i.d.).

    Approximation volontairement simple : l'independance entre maps est fausse
    (momentum, lecture d'adversaire), mais l'erreur reste petite devant
    l'incertitude sur p_map lui-meme.
    """
    p = min(max(float(p_map), 1e-6), 1 - 1e-6)
    wins_needed = best_of // 2 + 1
    total = 0.0
    for losses in range(wins_needed):
        games = wins_needed + losses
        total += math.comb(games - 1, losses) * (p**wins_needed) * ((1 - p) ** losses)
    return float(total)
