"""Predictions historiques hors entrainement pour les calibrations derivees."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np

from .model import map_predictor
from .temporal import day_end, validate_dates
from .validation import positive_integer


@dataclass
class CalibrationHistory:
    pools: np.ndarray
    maps: dict[tuple[str, str], float]
    trained_through: list[str]


def prequential_maps(match_ds, map_ds, *, refit_every: int = 150, **model_kwargs) -> CalibrationHistory:
    """Chaque bloc est predit par un modele n'ayant vu que les jours precedents.

    Les frontieres sont ancrees au debut de l'historique : prolonger le fichier
    ne change pas les anciennes predictions. Les premieres rencontres, sans
    au moins 50 maps anterieures, restent explicitement indisponibles.
    """
    positive_integer("refit_every", refit_every)
    validate_dates(match_ds.dates)
    validate_dates(map_ds.dates)
    if match_ds.pool_X is None:
        raise ValueError("les features du pool sont requises pour la calibration")
    n, pool_size, feature_count = match_ds.pool_X.shape
    pools = np.full((n, pool_size), np.nan)
    maps = {}
    trained_through = [""] * n
    start = 0
    while start < n:
        cutoff = match_ds.dates[start]
        k = bisect_left(map_ds.dates, cutoff)
        if k < 50:
            start = day_end(match_ds.dates, start+1)
            continue
        end = day_end(match_ds.dates, min(n, start+refit_every))
        model = map_predictor(**model_kwargs).fit(
            map_ds.X[:k], map_ds.y[:k], map_ds.dates[:k], map_ds.feature_names
        )
        pools[start:end] = model.predict_proba(
            match_ds.pool_X[start:end].reshape(-1, feature_count)
        ).reshape(end-start, pool_size)
        trained_through[start:end] = [model.last_date] * (end-start)
        stop = bisect_right(map_ds.dates, match_ds.dates[end-1])
        ps = model.predict_proba(map_ds.X[k:stop])
        for meta, probability in zip(map_ds.meta[k:stop], ps):
            maps[meta["match_id"], meta["map"]] = float(probability)
        start = end
    return CalibrationHistory(pools, maps, trained_through)
