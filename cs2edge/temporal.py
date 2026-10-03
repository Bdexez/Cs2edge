"""Frontieres temporelles communes : une date ne traverse jamais un decoupage."""

from __future__ import annotations

from bisect import bisect_left
import datetime as dt
import math
from numbers import Integral


def validate_dates(dates) -> None:
    if any(a > b for a, b in zip(dates, dates[1:])):
        raise ValueError("les dates doivent etre triees chronologiquement")


def day_end(dates, index: int) -> int:
    while 0 < index < len(dates) and dates[index - 1] == dates[index]:
        index += 1
    return index


def forward_splits(dates, initial_frac: float, min_train: int, refit_every: int,
                   evaluation_start: dt.date | None = None) -> list[tuple[int, int]]:
    validate_dates(dates)
    if not math.isfinite(initial_frac) or not 0 < initial_frac < 1:
        raise ValueError("initial_frac doit etre strictement entre 0 et 1")
    if not isinstance(refit_every, Integral) or refit_every < 1:
        raise ValueError("refit_every doit etre un entier strictement positif")
    if not isinstance(min_train, Integral) or min_train < 50:
        raise ValueError("min_train doit etre un entier d'au moins 50")
    start = (max(min_train, int(len(dates) * initial_frac)) if evaluation_start is None
             else bisect_left(dates, evaluation_start))
    start = day_end(dates, start)
    if start < min_train or len(dates) - start < 50:
        raise ValueError(f"historique trop court ({len(dates)} matchs) : "
                         f"il faut {min_train} matchs passes et 50 a evaluer sur des jours distincts")
    splits = []
    while start < len(dates):
        end = day_end(dates, min(len(dates), start + refit_every))
        splits.append((start, end))
        start = end
    return splits


def calibration_boundary(dates, proposed: int, minimum: int = 50) -> int:
    """Recule au debut de la date de calibration ; renonce si le passe est trop court."""
    if proposed >= len(dates):
        return len(dates)
    while proposed > 0 and dates[proposed - 1] == dates[proposed]:
        proposed -= 1
    return proposed if proposed >= minimum else len(dates)
