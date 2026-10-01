"""Entrees de donnees.

Le projet ne scrape rien tout seul : il consomme un CSV au niveau map, ce qui
le rend agnostique de la source (export HLTV, bo3.gg, Liquipedia, dataset
Kaggle, base maison). `from_records` est le point d'entree a brancher sur
n'importe quel client d'API.

Schema attendu (une ligne = une map jouee) :
    match_id,date,team_a,team_b,map,rounds_a,rounds_b[,best_of,event,tier,lan,map_index,roster_a,roster_b]

`tier` : s / a / b / c / d (poids du resultat dans l'Elo).
`roster_a`/`roster_b` : pseudos separes par des virgules, pour detecter les
changements d'effectif — tres utile, souvent absent des exports publics.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import ACTIVE_DUTY, MapResult, map_from_record, read_csv

__all__ = ["read_csv", "from_records", "simulate_history", "simulate_with_truth",
           "map_win_probability", "Regime", "REGIMES"]


def from_records(records: Iterable[dict]) -> list[MapResult]:
    """Convertit des dicts bruts (sortie d'API) en `MapResult` normalises."""
    out: list[MapResult] = []
    for i, row in enumerate(records, 1):
        try:
            out.append(map_from_record(row))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"enregistrement {i} invalide : {exc}") from exc
    out.sort(key=lambda m: (m.date, m.match_id, m.map_index))
    return out


# --------------------------------------------------------------------------
# Generateur synthetique : permet de faire tourner et tester toute la chaine
# sans donnees reelles. Les scores sont produits round par round, donc les
# distributions de scores ressemblent a du vrai CS.
# --------------------------------------------------------------------------

_TEAM_NAMES = [
    "vitality", "spirit", "navi", "faze", "g2", "mouz", "falcons", "liquid",
    "astralis", "heroic", "furia", "complexity", "virtuspro", "eternalfire",
    "cloud9", "big", "gamerlegion", "nip", "ence", "apeks", "9z", "imperial",
    "mongolz", "fnatic", "sangal", "betboom", "amkal", "monte", "saw", "rebels",
    "passion", "fluxo", "legacy", "case", "nrg", "wildcard", "m80", "party",
    "bad news", "aurora",
]


def _simulate_rounds(p_round: float, rng: random.Random) -> tuple[int, int]:
    """Simule une map en format MR12 avec prolongations."""
    a = b = 0
    while True:
        if rng.random() < p_round:
            a += 1
        else:
            b += 1
        if a >= 13 and a - b >= 2:
            return a, b
        if b >= 13 and b - a >= 2:
            return a, b
        if a == 12 and b == 12:
            break
    # Prolongations MR3 : premier a 4, ecart de 2, sinon on rejoue.
    while True:
        oa = ob = 0
        while True:
            if rng.random() < p_round:
                oa += 1
            else:
                ob += 1
            if oa >= 4 and oa - ob >= 2:
                return a + oa, b + ob
            if ob >= 4 and ob - oa >= 2:
                return a + oa, b + ob
            if oa == 3 and ob == 3:
                a, b = a + 3, b + 3
                break


@dataclass
class Regime:
    """Parametres du monde simule.

    Le regime par defaut ("facile") produit un CS2 ou les ecarts de niveau sont
    larges et stables : un modele correct y atteint ~79% de reussite. Le vrai
    CS2 professionnel plafonne vers 65%. Mesurer une amelioration dans un monde
    trop previsible enseigne les mauvaises lecons - typiquement, cela recompense
    la confiance excessive, puisque la confiance y est presque toujours justifiee.

    Le regime "realiste" resserre les niveaux, accelere leur derive et augmente
    le bruit par round, jusqu'a ramener le plafond de reussite pres du reel.
    """

    skill_spread: float = 1.0       # ecart-type des niveaux latents
    map_spread: float = 0.35        # ecart-type des forces par map
    drift: float = 0.035            # derive du niveau par jour de calendrier
    reshuffle_rate: float = 0.0022  # probabilite quotidienne de remaniement
    reshuffle_shock: float = 0.55   # ampleur d'un remaniement
    round_edge: float = 0.28        # conversion ecart de niveau -> avantage par round
    round_clip: float = 0.80        # borne sur la probabilite de gagner un round


REGIMES = {
    "facile": Regime(),
    # Calibre contre le plafond atteignable, pas contre la performance du modele :
    # un observateur connaissant parfaitement les niveaux latents y atteint 68.8%
    # de reussite (log-loss 0.576), soit l'ordre de grandeur du vrai circuit.
    # C'est le regime a utiliser pour juger une amelioration.
    "realiste": Regime(
        skill_spread=0.30, map_spread=0.18, drift=0.060, reshuffle_rate=0.0040,
        reshuffle_shock=0.60, round_edge=0.13, round_clip=0.62,
    ),
}


def simulate_history(
    n_teams: int = 32,
    years: float = 3.0,
    seed: int = 7,
    start: dt.date | None = None,
    regime: "str | Regime" = "facile",
) -> list[MapResult]:
    """Genere un historique credible : niveaux latents qui derivent, forces par
    map, changements de roster, tiers d'evenements, calendrier irregulier.

    `regime` : "facile" (defaut, lisible) ou "realiste" (aussi imprevisible que
    le vrai circuit). Voir `Regime`.
    """
    cfg = REGIMES[regime] if isinstance(regime, str) else regime
    rng = random.Random(seed)
    names = _TEAM_NAMES[:n_teams]
    start = start or (dt.date.today() - dt.timedelta(days=int(365 * years)))

    skill = {t: rng.gauss(0, cfg.skill_spread) for t in names}
    map_skill = {t: {m: rng.gauss(0, cfg.map_spread) for m in ACTIVE_DUTY} for t in names}
    pool = {t: rng.sample(list(ACTIVE_DUTY), 6) for t in names}

    rows: list[MapResult] = []
    day = start
    match_no = 0
    end = start + dt.timedelta(days=int(365 * years))

    while day < end:
        # Derive lente des niveaux (meta, forme, coaching).
        for t in names:
            skill[t] += rng.gauss(0, cfg.drift)
            if rng.random() < cfg.reshuffle_rate:  # remaniement d'effectif
                skill[t] += rng.gauss(0, cfg.reshuffle_shock)
                for m in ACTIVE_DUTY:
                    map_skill[t][m] = 0.6 * map_skill[t][m] + rng.gauss(0, cfg.map_spread * 0.85)

        tier = rng.choices(["s", "a", "b", "c"], weights=[0.10, 0.25, 0.40, 0.25])[0]
        lan = tier in ("s", "a") and rng.random() < 0.7
        # Les gros evenements reunissent les meilleures equipes.
        ranked = sorted(names, key=lambda t: -skill[t])
        span = {"s": 12, "a": 20, "b": n_teams, "c": n_teams}[tier]
        candidates = ranked[:span] if tier in ("s", "a") else names
        n_matches = rng.randint(2, 7)

        for _ in range(n_matches):
            a, b = rng.sample(candidates, 2)
            best_of = rng.choices([1, 3, 5], weights=[0.32, 0.63, 0.05])[0]
            needed = best_of // 2 + 1
            # Veto : chacun impose une map de son pool, la decider est neutre.
            picks: list[str] = []
            for k in range(best_of):
                owner = a if k % 2 == 0 else b
                picks.append(rng.choice(pool[owner]) if k < best_of - 1 else rng.choice(ACTIVE_DUTY))
            match_no += 1
            mid = f"sim{match_no:06d}"
            wa = wb = 0
            for idx, mp in enumerate(picks):
                if wa == needed or wb == needed:
                    break
                edge = (skill[a] + map_skill[a][mp]) - (skill[b] + map_skill[b][mp])
                # ~0.5 + 0.055 par point d'ecart : un ecart de 2 donne ~62% par round,
                # soit ~90% par map, ordre de grandeur realiste entre tier 1 et tier 3.
                p_round = 1.0 / (1.0 + math.exp(-edge * cfg.round_edge))
                p_round = min(cfg.round_clip, max(1.0 - cfg.round_clip, p_round))
                ra, rb = _simulate_rounds(p_round, rng)
                wa += int(ra > rb)
                wb += int(rb > ra)
                rows.append(
                    MapResult(
                        match_id=mid, date=day, team_a=a, team_b=b, map_name=mp,
                        rounds_a=ra, rounds_b=rb, best_of=best_of,
                        event=f"sim {tier.upper()} {day.year}", tier=tier, lan=lan,
                        map_index=idx,
                    )
                )
        day += dt.timedelta(days=rng.randint(1, 3))

    rows.sort(key=lambda m: (m.date, m.match_id, m.map_index))
    return rows


# --------------------------------------------------------------------------
# Verite terrain : le plafond atteignable.
#
# Sur des donnees simulees, on connait les niveaux latents. On peut donc
# calculer la probabilite *vraie* de chaque match - celle qu'un modele parfait
# annoncerait - et mesurer l'ecart qui reste entre le modele et ce plafond.
#
# C'est le diagnostic qui manque a la plupart des projets de ce type. Sans lui,
# on ne sait pas si un log-loss de 0.43 est excellent ou mediocre : il n'existe
# aucune reference. Avec lui, on sait combien de signal reste sur la table, et
# quand il faut arreter d'optimiser parce qu'il n'en reste plus.
# --------------------------------------------------------------------------


def map_win_probability(p_round: float) -> float:
    """Probabilite de gagner une map en MR12, prolongations comprises.

    Forme fermee : course a 13 gagnee par k rounds d'ecart, plus le cas 12-12
    qui bascule en prolongations (course a 4, avec la meme structure recursive).
    """
    p = min(max(float(p_round), 1e-9), 1 - 1e-9)
    q = 1.0 - p

    # Prolongations : premier a 4, ecart de 2, et 3-3 relance une prolongation.
    a = sum(math.comb(3 + k, k) * p**4 * q**k for k in range(3))
    repeat = math.comb(6, 3) * p**3 * q**3
    p_ot = a / (1.0 - repeat) if repeat < 1.0 else 0.5

    regulation = sum(math.comb(12 + k, k) * p**13 * q**k for k in range(12))
    tie = math.comb(24, 12) * p**12 * q**12
    return regulation + tie * p_ot


def simulate_with_truth(
    n_teams: int = 32,
    years: float = 3.0,
    seed: int = 7,
    start: dt.date | None = None,
    regime: "str | Regime" = "facile",
    veto_samples: int = 400,
) -> tuple[list[MapResult], dict[str, float]]:
    """Comme `simulate_history`, plus la probabilite vraie de chaque match.

    La probabilite vraie marginalise sur le veto (tire au hasard dans les pools,
    comme dans la simulation) : c'est bien la probabilite qu'un observateur
    connaissant parfaitement les niveaux annoncerait *avant* le veto.

    N'a de sens que sur des donnees simulees, ou les niveaux latents existent.
    """
    cfg = REGIMES[regime] if isinstance(regime, str) else regime
    rng = random.Random(seed)
    names = _TEAM_NAMES[:n_teams]
    start = start or (dt.date.today() - dt.timedelta(days=int(365 * years)))

    skill = {t: rng.gauss(0, cfg.skill_spread) for t in names}
    map_skill = {t: {m: rng.gauss(0, cfg.map_spread) for m in ACTIVE_DUTY} for t in names}
    pool = {t: rng.sample(list(ACTIVE_DUTY), 6) for t in names}

    rows: list[MapResult] = []
    truth: dict[str, float] = {}
    day = start
    match_no = 0
    end = start + dt.timedelta(days=int(365 * years))
    truth_rng = random.Random(seed + 99991)

    def p_map(a: str, b: str, mp: str) -> float:
        edge = (skill[a] + map_skill[a][mp]) - (skill[b] + map_skill[b][mp])
        pr = 1.0 / (1.0 + math.exp(-edge * cfg.round_edge))
        pr = min(cfg.round_clip, max(1.0 - cfg.round_clip, pr))
        return map_win_probability(pr)

    while day < end:
        for t in names:
            skill[t] += rng.gauss(0, cfg.drift)
            if rng.random() < cfg.reshuffle_rate:
                skill[t] += rng.gauss(0, cfg.reshuffle_shock)
                for m in ACTIVE_DUTY:
                    map_skill[t][m] = 0.6 * map_skill[t][m] + rng.gauss(0, cfg.map_spread * 0.85)

        tier = rng.choices(["s", "a", "b", "c"], weights=[0.10, 0.25, 0.40, 0.25])[0]
        lan = tier in ("s", "a") and rng.random() < 0.7
        ranked = sorted(names, key=lambda t: -skill[t])
        span = {"s": 12, "a": 20, "b": n_teams, "c": n_teams}[tier]
        candidates = ranked[:span] if tier in ("s", "a") else names

        for _ in range(rng.randint(2, 7)):
            a, b = rng.sample(candidates, 2)
            best_of = rng.choices([1, 3, 5], weights=[0.32, 0.63, 0.05])[0]
            needed = best_of // 2 + 1
            picks = [
                rng.choice(pool[a if k % 2 == 0 else b]) if k < best_of - 1
                else rng.choice(ACTIVE_DUTY)
                for k in range(best_of)
            ]
            match_no += 1
            mid = f"sim{match_no:06d}"

            # Probabilite vraie, en re-tirant le veto comme le fait la simulation.
            wins = 0
            for _s in range(veto_samples):
                sample_picks = [
                    truth_rng.choice(pool[a if k % 2 == 0 else b]) if k < best_of - 1
                    else truth_rng.choice(ACTIVE_DUTY)
                    for k in range(best_of)
                ]
                wa = wb = 0
                for mp in sample_picks:
                    if wa == needed or wb == needed:
                        break
                    if truth_rng.random() < p_map(a, b, mp):
                        wa += 1
                    else:
                        wb += 1
                wins += int(wa > wb)
            truth[mid] = wins / veto_samples

            wa = wb = 0
            for idx, mp in enumerate(picks):
                if wa == needed or wb == needed:
                    break
                edge = (skill[a] + map_skill[a][mp]) - (skill[b] + map_skill[b][mp])
                pr = 1.0 / (1.0 + math.exp(-edge * cfg.round_edge))
                pr = min(cfg.round_clip, max(1.0 - cfg.round_clip, pr))
                ra, rb = _simulate_rounds(pr, rng)
                wa += int(ra > rb)
                wb += int(rb > ra)
                rows.append(
                    MapResult(
                        match_id=mid, date=day, team_a=a, team_b=b, map_name=mp,
                        rounds_a=ra, rounds_b=rb, best_of=best_of,
                        event=f"sim {tier.upper()} {day.year}", tier=tier, lan=lan,
                        map_index=idx,
                    )
                )
        day += dt.timedelta(days=rng.randint(1, 3))

    rows.sort(key=lambda m: (m.date, m.match_id, m.map_index))
    return rows, truth
