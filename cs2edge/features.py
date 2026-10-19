"""Construction des features, sans fuite temporelle.

Regle d'or : chaque ligne du dataset est calculee avec l'etat des equipes
*avant* le match, puis l'etat est mis a jour. Un seul passage chronologique.
Le meme objet sert a l'entrainement et a la prediction d'un match a venir,
ce qui elimine les divergences train/serve.

Toutes les features sont des differences (A - B). Le modele est donc
antisymetrique par construction : p(A bat B) = 1 - p(B bat A).
"""

from __future__ import annotations

import datetime as dt
import math
from itertools import groupby
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .ratings import EloConfig, EloEngine
from .schema import ACTIVE_DUTY, Match, MapResult

FORM_HALFLIFE_DAYS = 75.0
FORM_WINDOW = 30
H2H_HALFLIFE_DAYS = 300.0
H2H_PRIOR = 4.0
WORKLOAD_WINDOW_DAYS = 21


@dataclass
class _Recent:
    """Historique court d'une equipe, pour la forme et la charge de travail."""

    maps: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    per_map: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=12)))
    streak: int = 0
    roster: tuple[str, ...] = ()
    maps_with_roster: int = 0

    def add(self, date: dt.date, won: int, round_diff: int, opp_rating: float,
            map_name: str, own_rating: float = 1500.0) -> None:
        self.maps.append((date, won, round_diff, opp_rating, own_rating))
        self.per_map[map_name].append((date, won, round_diff))
        if won:
            self.streak = max(0, self.streak) + 1
        else:
            self.streak = min(0, self.streak) - 1
        self.maps_with_roster += 1


def _decay(days: float, halflife: float) -> float:
    return 0.5 ** (max(0.0, days) / halflife)


class FeatureExtractor:
    """Etat + extraction. Alimenter via `observe_match` dans l'ordre du temps."""

    MATCH_FEATURES = (
        "elo_diff",
        "pool_mean_diff",
        "pool_top3_diff",
        "form_diff",
        "round_form_diff",
        "h2h_diff",
        "rest_diff",
        "workload_diff",
        "sos_diff",
        "streak_diff",
        "experience_diff",
        "roster_tenure_diff",
        "elo_diff_x_lan",
        "elo_diff_x_bo3",
        # --- ajouts mesures ---
        "elo_diff_x_certainty",   # l'ecart de niveau vaut-il qu'on s'y fie
        "trajectory_diff",        # qui monte, qui descend
        "perf_residual_diff",     # performance au-dessus / en dessous de son niveau
    )

    MAP_FEATURES = (
        "map_elo_diff",
        "elo_diff",
        "map_form_diff",
        "map_experience_diff",
        "form_diff",
        "h2h_map_diff",
    )

    def __init__(self, elo_config: EloConfig | None = None) -> None:
        self.elo = EloEngine(elo_config)
        self.recent: dict[str, _Recent] = defaultdict(_Recent)
        # h2h[(min,max)] = deque de (date, gagnant)
        self.h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=40))
        self.h2h_map: dict[tuple[str, str, str], deque] = defaultdict(lambda: deque(maxlen=20))

    # ------------------------------------------------------------- statistiques

    def _form(self, team: str, on: dt.date) -> tuple[float, float]:
        """Taux de victoire pondere par recence, et ecart de rounds moyen."""
        rec = self.recent.get(team)
        if not rec or not rec.maps:
            return 0.5, 0.0
        num = den = rd = 0.0
        for date, won, round_diff, _opp, _own in rec.maps:
            w = _decay((on - date).days, FORM_HALFLIFE_DAYS)
            num += w * won
            den += w
            rd += w * round_diff
        if den <= 1e-9:
            return 0.5, 0.0
        return num / den, rd / den

    def _map_form(self, team: str, map_name: str, on: dt.date) -> tuple[float, int]:
        rec = self.recent.get(team)
        if not rec or map_name not in rec.per_map or not rec.per_map[map_name]:
            return 0.5, 0
        entries = rec.per_map[map_name]
        num = den = 0.0
        for date, won, _rd in entries:
            w = _decay((on - date).days, FORM_HALFLIFE_DAYS * 2)
            num += w * won
            den += w
        return (num / den if den > 1e-9 else 0.5), len(entries)

    def _sos(self, team: str, on: dt.date) -> float:
        """Force du calendrier : note moyenne des adversaires recents."""
        rec = self.recent.get(team)
        if not rec or not rec.maps:
            return 1500.0
        num = den = 0.0
        for date, _w, _rd, opp, _own in rec.maps:
            w = _decay((on - date).days, FORM_HALFLIFE_DAYS)
            num += w * opp
            den += w
        return num / den if den > 1e-9 else 1500.0

    def _perf_residual(self, team: str, on: dt.date) -> float:
        """Ecart de rounds realise, moins celui que l'ecart de niveau laissait attendre.

        Le differentiel de rounds brut recompense un calendrier facile : ecraser
        des equipes faibles gonfle la statistique sans rien prouver. Le residu
        mesure autre chose - jouer au-dessus ou en dessous de son niveau - et
        c'est cette part-la qui porte de l'information sur la forme du moment.
        """
        rec = self.recent.get(team)
        if not rec or not rec.maps:
            return 0.0
        scale = self.elo.rounds_per_100_elo()
        num = den = 0.0
        for date, _w, round_diff, opp, own in rec.maps:
            w = _decay((on - date).days, FORM_HALFLIFE_DAYS)
            expected = (own - opp) / 100.0 * scale
            num += w * (round_diff - expected)
            den += w
        return num / den if den > 1e-9 else 0.0

    def _rest_days(self, team: str, on: dt.date) -> float:
        tr = self.elo.teams.get(team)
        if tr is None or tr.last_date is None:
            return 30.0
        return float(min(90, max(0, (on - tr.last_date).days)))

    def _workload(self, team: str, on: dt.date) -> float:
        rec = self.recent.get(team)
        if not rec:
            return 0.0
        cutoff = on - dt.timedelta(days=WORKLOAD_WINDOW_DAYS)
        return float(sum(1 for d, *_ in rec.maps if d >= cutoff))

    def _h2h(self, team_a: str, team_b: str, on: dt.date) -> float:
        """Historique direct, ramene dans [-1, 1] et retreci vers 0 si peu de data."""
        key = (min(team_a, team_b), max(team_a, team_b))
        entries = self.h2h.get(key)
        if not entries:
            return 0.0
        num = den = 0.0
        for date, winner in entries:
            w = _decay((on - date).days, H2H_HALFLIFE_DAYS)
            num += w * (1.0 if winner == team_a else 0.0)
            den += w
        if den <= 1e-9:
            return 0.0
        return (2.0 * num - den) / (den + H2H_PRIOR)

    def _h2h_on_map(self, team_a: str, team_b: str, map_name: str, on: dt.date) -> float:
        key = (min(team_a, team_b), max(team_a, team_b), map_name)
        entries = self.h2h_map.get(key)
        if not entries:
            return 0.0
        num = den = 0.0
        for date, winner in entries:
            w = _decay((on - date).days, H2H_HALFLIFE_DAYS)
            num += w * (1.0 if winner == team_a else 0.0)
            den += w
        if den <= 1e-9:
            return 0.0
        return (2.0 * num - den) / (den + 2.0)

    def _roster_tenure(self, team: str) -> float:
        rec = self.recent.get(team)
        return float(rec.maps_with_roster) if rec else 0.0

    # ----------------------------------------------------------------- features

    def map_pool_edges(self, team_a: str, team_b: str, on: dt.date) -> dict[str, float]:
        """Avantage de A sur B, map par map, en points Elo."""
        return {
            m: self.elo.map_rating(team_a, m, on) - self.elo.map_rating(team_b, m, on)
            for m in ACTIVE_DUTY
        }

    def match_features(
        self,
        team_a: str,
        team_b: str,
        on: dt.date,
        best_of: int = 3,
        lan: bool = False,
    ) -> np.ndarray:
        elo_diff = (self.elo.rating(team_a, on) - self.elo.rating(team_b, on)) / 100.0
        edges = self.map_pool_edges(team_a, team_b, on)
        vals = sorted(edges.values(), reverse=True)
        pool_mean = float(np.mean(vals)) / 100.0
        # Un bo3 se joue sur les 3 maps que chacun choisit : le haut de pool compte.
        pool_top3 = (float(np.mean(vals[:3])) + float(np.mean(vals[-3:]))) / 2.0 / 100.0

        form_a, rd_a = self._form(team_a, on)
        form_b, rd_b = self._form(team_b, on)
        rest = (math.log1p(self._rest_days(team_a, on)) - math.log1p(self._rest_days(team_b, on)))
        work = (self._workload(team_a, on) - self._workload(team_b, on)) / 10.0
        sos = (self._sos(team_a, on) - self._sos(team_b, on)) / 100.0
        streak_a = max(-5, min(5, self.recent[team_a].streak)) if team_a in self.recent else 0
        streak_b = max(-5, min(5, self.recent[team_b].streak)) if team_b in self.recent else 0
        exp = (
            math.log1p(self.elo.maps_played(team_a)) - math.log1p(self.elo.maps_played(team_b))
        )
        tenure = math.log1p(self._roster_tenure(team_a)) - math.log1p(self._roster_tenure(team_b))

        # Ces colonnes sont volontairement brutes : la decorrelation vis-a-vis de
        # elo_diff est faite dans le modele (MatchPredictor.residualize), ou le
        # coefficient de projection peut etre estime sur les donnees.
        return np.array(
            [
                elo_diff,
                pool_mean,
                pool_top3,
                form_a - form_b,
                (rd_a - rd_b) / 5.0,
                self._h2h(team_a, team_b, on),
                rest,
                work,
                sos,
                (streak_a - streak_b) / 3.0,
                exp,
                tenure,
                # Interactions centrees (+/-0.5 au lieu de 0/1) : sans cela elles
                # sont colineaires a elo_diff et le ridge repartit le poids
                # n'importe comment entre les trois, ce qui rend `explain` faux.
                elo_diff * (0.5 if lan else -0.5),
                elo_diff * (0.5 if best_of >= 3 else -0.5),
                # `certainty` est symetrique dans les deux equipes, donc ce
                # produit reste antisymetrique : la propriete p(A|B) = 1-p(B|A)
                # tient toujours.
                elo_diff * self.elo.certainty(team_a, team_b, on),
                (self.elo.trajectory(team_a, on) - self.elo.trajectory(team_b, on)) / 100.0,
                (self._perf_residual(team_a, on) - self._perf_residual(team_b, on)) / 5.0,
            ],
            dtype=float,
        )

    def map_features(self, team_a: str, team_b: str, on: dt.date, map_name: str) -> np.ndarray:
        map_elo = (
            self.elo.map_rating(team_a, map_name, on) - self.elo.map_rating(team_b, map_name, on)
        ) / 100.0
        elo_diff = (self.elo.rating(team_a, on) - self.elo.rating(team_b, on)) / 100.0
        fa, na = self._map_form(team_a, map_name, on)
        fb, nb = self._map_form(team_b, map_name, on)
        form_a, _ = self._form(team_a, on)
        form_b, _ = self._form(team_b, on)
        return np.array(
            [
                map_elo,
                elo_diff,
                fa - fb,
                (math.log1p(na) - math.log1p(nb)),
                form_a - form_b,
                self._h2h_on_map(team_a, team_b, map_name, on),
            ],
            dtype=float,
        )

    # ------------------------------------------------------------ mise a jour

    def observe_map(self, result: MapResult) -> None:
        a, b = result.team_a, result.team_b
        ra = self.elo.rating(a, result.date)
        rb = self.elo.rating(b, result.date)
        self.elo.observe(result)
        self.recent[a].add(result.date, result.a_won, result.round_diff, rb, result.map_name, ra)
        self.recent[b].add(result.date, 1 - result.a_won, -result.round_diff, ra, result.map_name, rb)
        key = (min(a, b), max(a, b))
        self.h2h[key].append((result.date, result.winner))
        self.h2h_map[(key[0], key[1], result.map_name)].append((result.date, result.winner))
        for team, roster in ((a, result.roster_a), (b, result.roster_b)):
            if roster:
                rec = self.recent[team]
                if rec.roster and set(roster) != set(rec.roster):
                    rec.maps_with_roster = 1
                rec.roster = roster

    def observe_match(self, match: Match) -> None:
        for m in sorted(match.maps, key=lambda x: x.map_index):
            self.observe_map(m)

    def match_context(self, a: str, b: str, on: dt.date) -> dict:
        """Meme contexte de fiabilite pour les features, le service et les relectures."""
        ta, tb = self.elo._team(a), self.elo._team(b)
        h2h = len(self.h2h.get((min(a, b), max(a, b)), ()))
        return {
            "maps_a": ta.maps_played, "maps_b": tb.maps_played,
            "last_a": ta.last_date.isoformat() if ta.last_date else "jamais",
            "last_b": tb.last_date.isoformat() if tb.last_date else "jamais",
            "maps_played_min": min(ta.maps_played, tb.maps_played),
            "h2h": h2h, "h2h_n": h2h,
            "certainty": self.elo.certainty(a, b, on),
            "days_idle_max": max((on - ta.last_date).days if ta.last_date else 999,
                                 (on - tb.last_date).days if tb.last_date else 999),
        }


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    dates: list[dt.date]
    meta: list[dict]
    feature_names: tuple[str, ...]
    # (n_matchs, 7, n_features_map) : features des 7 maps de l'active duty, avant
    # veto. Permet de backtester la lecture "par maps" exactement comme elle est
    # servie en production, ou aucune map n'est encore connue.
    pool_X: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)

    def slice(self, idx) -> "Dataset":
        idx = np.asarray(idx)
        return Dataset(
            X=self.X[idx],
            y=self.y[idx],
            dates=[self.dates[i] for i in idx],
            meta=[self.meta[i] for i in idx],
            feature_names=self.feature_names,
            pool_X=None if self.pool_X is None else self.pool_X[idx],
        )


def build_datasets(
    matches: list[Match],
    elo_config: EloConfig | None = None,
    warmup_maps: int = 12,
) -> tuple[Dataset, Dataset, FeatureExtractor]:
    """Passe chronologique unique -> (dataset match, dataset map, etat final).

    `warmup_maps` ecarte de l'entrainement les matchs ou une equipe n'a pas
    encore assez d'historique : ses features sont du bruit.
    """
    fx = FeatureExtractor(elo_config)
    mX, my, mdates, mmeta, mpool = [], [], [], [], []
    pX, py, pdates, pmeta = [], [], [], []

    if warmup_maps < 0:
        raise ValueError("warmup_maps doit etre positif ou nul")
    ordered = sorted(matches, key=lambda m: (m.date, m.match_id))
    for _, day_matches in groupby(ordered, key=lambda m: m.date):
        day_matches = list(day_matches)
        # Sans horaires, toutes les predictions du jour utilisent la veille.
        for match in day_matches:
            a, b = match.team_a, match.team_b
            ready = min(fx.elo.maps_played(a), fx.elo.maps_played(b)) >= warmup_maps
            if ready and match.is_complete:
                mX.append(fx.match_features(a, b, match.date, match.best_of, match.lan))
                my.append(match.a_won)
                mdates.append(match.date)
                mmeta.append(
                    {
                        "match_id": match.match_id, "team_a": a, "team_b": b,
                        "event": match.event, "tier": match.tier, "best_of": match.best_of,
                        "lan": match.lan, "score": f"{match.maps_a}-{match.maps_b}",
                        **fx.match_context(a, b, match.date),
                    }
                )
                mpool.append([fx.map_features(a, b, match.date, m) for m in ACTIVE_DUTY])
                for mp in match.maps:
                    pX.append(fx.map_features(a, b, match.date, mp.map_name))
                    py.append(mp.a_won)
                    pdates.append(match.date)
                    pmeta.append(
                        {
                            "match_id": match.match_id, "team_a": a, "team_b": b,
                            "map": mp.map_name, "score": f"{mp.rounds_a}-{mp.rounds_b}",
                        }
                    )
        for match in day_matches:
            fx.observe_match(match)

    match_ds = Dataset(
        X=np.array(mX, dtype=float).reshape(len(my), len(FeatureExtractor.MATCH_FEATURES)),
        y=np.array(my, dtype=float),
        dates=mdates, meta=mmeta, feature_names=FeatureExtractor.MATCH_FEATURES,
        pool_X=np.array(mpool, dtype=float).reshape(
            len(my), len(ACTIVE_DUTY), len(FeatureExtractor.MAP_FEATURES)
        ),
    )
    map_ds = Dataset(
        X=np.array(pX, dtype=float).reshape(len(py), len(FeatureExtractor.MAP_FEATURES)),
        y=np.array(py, dtype=float),
        dates=pdates, meta=pmeta, feature_names=FeatureExtractor.MAP_FEATURES,
    )
    return match_ds, map_ds, fx
