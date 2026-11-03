"""Predictions historiques hors entrainement pour les calibrations derivees."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import datetime as dt

import numpy as np

from .model import map_predictor
from .temporal import day_end, validate_dates
from .validation import finite_number, positive_integer


@dataclass
class CalibrationHistory:
    pools: np.ndarray
    maps: dict[tuple[str, str], float]
    trained_through: list[str]

    def to_dict(self) -> dict:
        return dict(pools=[[float(p) if np.isfinite(p) else None for p in row] for row in self.pools],
                    maps=[[key[0], key[1], p] for key, p in sorted(self.maps.items())],
                    trained_through=self.trained_through)

    @classmethod
    def from_dict(cls, payload: dict, match_ds, map_ds) -> "CalibrationHistory":
        pools = np.asarray(payload["pools"], dtype=float)
        through = payload["trained_through"]
        if pools.shape != (len(match_ds), match_ds.pool_X.shape[1]) or len(through) != len(match_ds):
            raise ValueError("dimensions d'historique de calibration incompatibles")
        available = set()
        for i, (row, last) in enumerate(zip(pools, through)):
            if last:
                if (not np.isfinite(row).all() or np.any((row < 0) | (row > 1))
                        or dt.date.fromisoformat(last) >= match_ds.dates[i]):
                    raise ValueError("prediction de calibration invalide ou non anterieure")
                available.add(match_ds.meta[i]["match_id"])
            elif not np.isnan(row).all():
                raise ValueError("calibration sans provenance temporelle")
        maps = {}
        for match_id, map_name, p in payload["maps"]:
            finite_number("prediction map de calibration", p, maximum=1.0)
            key = match_id, map_name
            if key in maps:
                raise ValueError("map de calibration dupliquee")
            maps[key] = p
        expected = {(meta["match_id"], meta["map"]) for meta in map_ds.meta
                    if meta["match_id"] in available}
        if set(maps) != expected:
            raise ValueError("inventaire de maps de calibration incompatible")
        return cls(pools, maps, list(through))


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
