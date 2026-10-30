"""Confrontation du modele au marche : retrait de la marge, valeur, mise.

Un modele qui "a raison a 65%" ne vaut rien en soi : le marche aussi. Ce qui
compte est l'ecart entre la probabilite du modele et la probabilite du marche
une fois la marge du bookmaker retiree, et la fiabilite de cet ecart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .validation import finite_number


def _probability(p: float) -> None:
    finite_number("probabilite", p, maximum=1.0)


def implied(odds: float) -> float:
    odds = float(odds)
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError("une cote decimale doit etre finie et > 1.0")
    return 1.0 / odds


def overround(odds_a: float, odds_b: float) -> float:
    """Marge du bookmaker. 1.05 = 5% de vig."""
    return implied(odds_a) + implied(odds_b)


def devig_proportional(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Retrait proportionnel : simple, surestime un peu les gros favoris."""
    ia, ib = implied(odds_a), implied(odds_b)
    tot = ia + ib
    return ia / tot, ib / tot


def devig_power(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Retrait par puissance : p_i ∝ implied_i**k, k tel que la somme fasse 1.

    Corrige mieux le biais favori/outsider que le retrait proportionnel.
    """
    ia, ib = implied(odds_a), implied(odds_b)
    # La racine est positive, sans borne universelle a 0.5 ou 2 : une
    # surcote comme une marge tres forte peut sortir de cet ancien intervalle.
    lo, hi = 0.0, 1.0
    while ia**hi + ib**hi > 1.0:
        hi *= 2.0
    for _ in range(80):
        k = (lo + hi) / 2.0
        s = ia**k + ib**k
        if s > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2.0
    s = ia**k + ib**k
    return ia**k / s, ib**k / s


def expected_value(p: float, odds: float) -> float:
    """Rendement attendu par unite misee."""
    _probability(p)
    implied(odds)
    return float(p * (odds - 1.0) - (1.0 - p))


def kelly_fraction(p: float, odds: float, cap: float = 0.05, scale: float = 0.25) -> float:
    """Kelly fractionnaire et plafonne.

    Kelly plein suppose que `p` est exact. Il ne l'est jamais ici : on prend
    un quart de Kelly, plafonne, ce qui divise la variance sans effondrer la
    croissance esperee.
    """
    _probability(p)
    implied(odds)
    finite_number("cap", cap, maximum=1.0)
    finite_number("scale", scale, maximum=1.0)
    b = float(odds) - 1.0
    f = (p * b - (1.0 - p)) / b
    return float(max(0.0, min(cap, f * scale)))


@dataclass
class ValueBet:
    side: str
    model_p: float
    market_p: float
    odds: float
    edge: float
    ev: float
    stake: float

    def __str__(self) -> str:
        return (
            f"{self.side}: modele {self.model_p:.1%} vs marche {self.market_p:.1%} "
            f"@ {self.odds:.2f} -> edge {self.edge:+.1%}, EV {self.ev:+.1%}/unite, "
            f"mise {self.stake:.2%} bankroll"
        )


def find_value(
    p_a: float,
    odds_a: float,
    odds_b: float,
    team_a: str = "A",
    team_b: str = "B",
    min_edge: float = 0.03,
) -> ValueBet | None:
    """Retourne le cote qui offre de la valeur, ou None si le marche est correct."""
    _probability(p_a)
    finite_number("min_edge", min_edge, maximum=1.0)
    ma, mb = devig_power(odds_a, odds_b)
    cands = [
        ValueBet(team_a, p_a, ma, odds_a, p_a - ma, expected_value(p_a, odds_a),
                 kelly_fraction(p_a, odds_a)),
        ValueBet(team_b, 1 - p_a, mb, odds_b, (1 - p_a) - mb, expected_value(1 - p_a, odds_b),
                 kelly_fraction(1 - p_a, odds_b)),
    ]
    best = max(cands, key=lambda c: c.edge)
    return best if best.edge >= min_edge and best.ev > 0 else None


def simulate_flat(results: list[tuple[float, float, int]], stake: float = 1.0) -> dict:
    """ROI a mise fixe. `results` = [(proba modele, cote prise, gagne 0/1)]."""
    finite_number("stake", stake, strict=True)
    for p, odds, won in results:
        _probability(p)
        implied(odds)
        if won not in (0, 1):
            raise ValueError("le resultat d'un pari doit etre 0 ou 1")
    if not results:
        return {"bets": 0, "roi": 0.0, "profit": 0.0, "turnover": 0.0, "hit_rate": 0.0}
    profit = sum((odds - 1.0) * stake if won else -stake for _p, odds, won in results)
    turnover = stake * len(results)
    return {
        "bets": len(results),
        "profit": float(profit),
        "turnover": float(turnover),
        "roi": float(profit / turnover),
        "hit_rate": float(np.mean([w for *_x, w in results])),
        "avg_odds": float(np.mean([o for _p, o, _w in results])),
    }
