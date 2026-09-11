"""Systeme de notation Elo : global + specifique a chaque map.

Deux notes cohabitent :
  - une note globale, mise a jour a chaque map jouee ;
  - une note par (equipe, map), initialisee sur la note globale de l'equipe
    puis affinee, ce qui permet d'evaluer un avantage de pool avant le veto.

Trois raffinements par rapport a un Elo scolaire :
  - marge de victoire (ecart de rounds) amortie pour les gros favoris ;
  - K module par le tier de l'evenement ;
  - regression vers la moyenne en cas d'inactivite prolongee ou de changement
    de roster (une equipe qui change 2 joueurs n'est plus la meme equipe).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, fields

from .schema import MapResult

BASE_RATING = 1500.0
SCALE = 400.0
# Constante de Glicko : convertit l'echelle Elo en echelle logistique naturelle.
Q = math.log(10.0) / SCALE


@dataclass
class EloConfig:
    #  Pas de mise a jour, en points Elo, pour une equipe d'incertitude
    #  "normale" (voir rd_reference). L'incertitude module ce pas autour de
    #  cette valeur : une equipe mal connue bouge davantage, une equipe etablie
    #  moins. Exprimer le parametre en points Elo le rend lisible et comparable
    #  au K d'un Elo classique - un reglage dont l'optimum tombe a 0.21 alors
    #  que le defaut vaut 1.0 est un reglage mal pose.
    k_base: float = 12.0
    k_map: float = 20.0
    use_mov: bool = True
    #  Demi-vie de retour vers la moyenne quand une equipe ne joue pas.
    idle_halflife_days: float = 400.0
    #  Part de la note effacee quand >=2 joueurs changent.
    roster_shock: float = 0.40
    #  Poids du prior global quand une map n'a jamais ete jouee.
    map_prior_weight: float = 0.65
    lan_bonus: float = 0.0

    #  --- incertitude sur la note (Glicko) ---
    #  Une note de 1800 batie sur 300 maps et une note de 1800 batie sur 15 maps
    #  ne disent pas la meme chose. L'ecart-type de la note le formalise : il
    #  grandit avec l'inactivite, retrecit avec les matchs joues, et dimensionne
    #  automatiquement le pas de mise a jour - une equipe mal connue bouge vite,
    #  une equipe etablie bouge peu.
    use_uncertainty: bool = True
    rd_initial: float = 300.0
    rd_min: float = 40.0
    #  Incertitude de reference : celle a laquelle `k_base` s'applique tel quel.
    rd_reference: float = 100.0
    #  Duree au bout de laquelle une equipe inactive redevient totalement inconnue.
    rd_growth_days: float = 700.0

    #  --- second horizon ---
    #  Une note "rapide" (pas multiplie) suit la forme du moment. L'ecart entre
    #  la note rapide et la note lente mesure une trajectoire : une equipe qui
    #  monte et une equipe qui descend peuvent avoir la meme note aujourd'hui.
    fast_multiplier: float = 3.0

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EloConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"parametres inconnus : {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path) -> "EloConfig":
        import json
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path) -> None:
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilite qu'une equipe notee `rating_a` batte une equipe notee `rating_b`."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / SCALE))


def g_factor(rd: float) -> float:
    """Attenuation de Glicko : face a un adversaire mal connu, on apprend moins."""
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (math.pi**2))


def natural_step(rd: float, rd_opponent: float, expected: float = 0.5) -> float:
    """Pas de Glicko, en points Elo par unite de surprise.

    Sert aussi de normalisation : rapporte au pas obtenu a l'incertitude de
    reference, il donne un multiplicateur sans dimension, ce qui rend `k_base`
    directement lisible en points Elo.
    """
    g_opp = g_factor(rd_opponent)
    var = 1.0 / max(Q * Q * g_opp * g_opp * expected * (1 - expected), 1e-12)
    den = 1.0 / (rd * rd) + 1.0 / var
    return (Q / den) * g_opp


def _mov_multiplier(round_diff: int, elo_diff_winner: float) -> float:
    """Attenue le gain quand un favori ecrase un outsider (anti-inflation)."""
    rd = max(1, abs(int(round_diff)))
    return math.log1p(rd) * (2.2 / (0.001 * max(elo_diff_winner, -400.0) + 2.2))


@dataclass
class TeamRating:
    rating: float = BASE_RATING
    fast_rating: float = BASE_RATING
    rd: float = 300.0
    maps_played: int = 0
    last_date: dt.date | None = None
    roster: tuple[str, ...] = ()
    map_ratings: dict[str, float] = field(default_factory=dict)
    map_counts: dict[str, int] = field(default_factory=dict)
    #  Regression en ligne de l'ecart de rounds sur l'ecart de note : permet de
    #  savoir ce qu'un score "aurait du" etre, sans constante magique.
    xy: float = 0.0
    xx: float = 0.0


class EloEngine:
    """Etat mutable des notes. A alimenter strictement dans l'ordre chronologique."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.cfg = config or EloConfig()
        self.teams: dict[str, TeamRating] = {}
        # Coefficient global ecart de note -> ecart de rounds, appris en ligne.
        self._xy = 0.0
        self._xx = 0.0

    # ------------------------------------------------------------------ lecture

    def _team(self, name: str) -> TeamRating:
        return self.teams.setdefault(name, TeamRating(rd=self.cfg.rd_initial))

    def deviation(self, team: str, on: dt.date | None = None) -> float:
        """Ecart-type de la note, elargi par l'inactivite."""
        tr = self._team(team)
        if not self.cfg.use_uncertainty:
            return self.cfg.rd_min
        rd = tr.rd
        if on is not None and tr.last_date is not None:
            idle = max(0, (on - tr.last_date).days)
            growth = (self.cfg.rd_initial**2 - self.cfg.rd_min**2) / self.cfg.rd_growth_days
            rd = math.sqrt(rd * rd + growth * idle)
        return float(min(self.cfg.rd_initial, rd))

    def certainty(self, team_a: str, team_b: str, on: dt.date | None = None) -> float:
        """Dans [0, 1] : a quel point l'ecart de notes est digne de confiance."""
        rd = math.sqrt(self.deviation(team_a, on) ** 2 + self.deviation(team_b, on) ** 2)
        return float(g_factor(rd))

    def trajectory(self, team: str, on: dt.date | None = None) -> float:
        """Note rapide moins note lente : la pente, pas le niveau."""
        tr = self._team(team)
        if tr.last_date is None:
            return 0.0
        decay = 1.0
        if on is not None:
            idle = max(0, (on - tr.last_date).days)
            decay = 0.5 ** (idle / 90.0)   # une trajectoire s'evente vite
        return float((tr.fast_rating - tr.rating) * decay)

    def rounds_per_100_elo(self) -> float:
        """Combien de rounds d'ecart vaut 100 points de note, appris sur les donnees."""
        if self._xx < 1e-6:
            return 2.5   # valeur de depart plausible, remplacee des les premiers matchs
        return float(self._xy / self._xx)

    def expected_round_diff(self, team_a: str, team_b: str, on: dt.date | None = None) -> float:
        gap = (self.rating(team_a, on) - self.rating(team_b, on)) / 100.0
        return gap * self.rounds_per_100_elo()

    def rating(self, team: str, on: dt.date | None = None) -> float:
        """Note globale, amortie vers la moyenne si l'equipe est inactive."""
        tr = self._team(team)
        if on is None or tr.last_date is None:
            return tr.rating
        idle = max(0, (on - tr.last_date).days)
        if idle <= 0:
            return tr.rating
        decay = 0.5 ** (idle / self.cfg.idle_halflife_days)
        return BASE_RATING + (tr.rating - BASE_RATING) * decay

    def map_rating(self, team: str, map_name: str, on: dt.date | None = None) -> float:
        """Note sur une map : melange note de map observee et note globale."""
        tr = self._team(team)
        glob = self.rating(team, on)
        played = tr.map_counts.get(map_name, 0)
        if played == 0:
            return glob
        specific = tr.map_ratings.get(map_name, glob)
        # Plus l'equipe a joue la map, plus sa note specifique pese.
        w = played / (played + self.cfg.map_prior_weight * 10.0)
        return w * specific + (1.0 - w) * glob

    def maps_played(self, team: str) -> int:
        return self._team(team).maps_played

    def win_probability(
        self, team_a: str, team_b: str, on: dt.date | None = None, map_name: str | None = None
    ) -> float:
        if map_name:
            ra = self.map_rating(team_a, map_name, on)
            rb = self.map_rating(team_b, map_name, on)
        else:
            ra, rb = self.rating(team_a, on), self.rating(team_b, on)
        return expected_score(ra, rb)

    def leaderboard(self, on: dt.date | None = None, min_maps: int = 20) -> list[tuple[str, float, int]]:
        rows = [
            (name, self.rating(name, on), tr.maps_played)
            for name, tr in self.teams.items()
            if tr.maps_played >= min_maps
        ]
        rows.sort(key=lambda r: -r[1])
        return rows

    # ----------------------------------------------------------------- ecriture

    def _apply_roster_shock(self, team: str, roster: tuple[str, ...]) -> None:
        if not roster:
            return
        tr = self._team(team)
        if not tr.roster:
            tr.roster = roster
            return
        changed = len(set(roster) ^ set(tr.roster)) // 2
        if changed >= 2:
            pull = min(1.0, self.cfg.roster_shock * (changed / 2.0))
            tr.rating += (BASE_RATING - tr.rating) * pull
            tr.fast_rating += (tr.rating - tr.fast_rating) * pull
            # Un effectif remanie n'est plus la meme equipe : on redevient
            # partiellement ignorant, ce que l'ecart-type doit refleter.
            tr.rd = min(self.cfg.rd_initial, tr.rd + pull * (self.cfg.rd_initial - tr.rd))
            for m in list(tr.map_ratings):
                tr.map_ratings[m] += (tr.rating - tr.map_ratings[m]) * pull
        tr.roster = roster

    def observe(self, result: MapResult) -> None:
        """Integre une map jouee dans les notes."""
        a, b = result.team_a, result.team_b
        self._apply_roster_shock(a, result.roster_a)
        self._apply_roster_shock(b, result.roster_b)

        ra = self.rating(a, result.date)
        rb = self.rating(b, result.date)
        score_a = float(result.a_won)
        ta, tb = self._team(a), self._team(b)

        # Regression en ligne : ce que l'ecart de notes laisse attendre comme
        # ecart de rounds. Mise a jour avant usage, sur le match courant.
        gap = (ra - rb) / 100.0
        self._xy += gap * result.round_diff
        self._xx += gap * gap

        rd_a = self.deviation(a, result.date)
        rd_b = self.deviation(b, result.date)

        weight = result.tier_weight
        if self.cfg.use_mov:
            elo_diff_winner = (ra - rb) if result.a_won else (rb - ra)
            weight *= _mov_multiplier(result.round_diff, elo_diff_winner)

        if self.cfg.use_uncertainty:
            # Glicko : le pas de chaque equipe depend de SA propre incertitude et
            # de celle de son adversaire. Une equipe etablie qui bat une inconnue
            # n'apprend presque rien ; l'inconnue, elle, apprend beaucoup. Un Elo
            # a somme nulle impose au contraire le meme deplacement aux deux, ce
            # qui est faux des que les deux ne sont pas aussi bien connues.
            ga, gb = g_factor(rd_a), g_factor(rd_b)
            exp_a = 1.0 / (1.0 + 10.0 ** (-gb * (ra - rb) / SCALE))
            exp_b = 1.0 - exp_a
            var_a = 1.0 / max(Q * Q * gb * gb * exp_a * (1 - exp_a), 1e-12)
            var_b = 1.0 / max(Q * Q * ga * ga * exp_b * (1 - exp_b), 1e-12)
            den_a = 1.0 / (rd_a * rd_a) + 1.0 / var_a
            den_b = 1.0 / (rd_b * rd_b) + 1.0 / var_b
            # Normalisation : le pas est exprime en multiples de celui qu'on
            # appliquerait a une equipe d'incertitude de reference. `k_base` est
            # donc bien un K en points Elo.
            ref = natural_step(self.cfg.rd_reference, self.cfg.rd_reference)
            k = self.cfg.k_base * weight / max(ref, 1e-9)
            delta_a = k * (Q / den_a) * gb * (score_a - exp_a)
            delta_b = k * (Q / den_b) * ga * ((1 - score_a) - exp_b)
            ta.rating = ra + delta_a
            tb.rating = rb + delta_b
            ta.rd = max(self.cfg.rd_min, math.sqrt(1.0 / den_a))
            tb.rd = max(self.cfg.rd_min, math.sqrt(1.0 / den_b))
        else:
            exp_a = expected_score(ra, rb)
            delta = self.cfg.k_base * weight * (score_a - exp_a)
            ta.rating = ra + delta
            tb.rating = rb - delta
            delta_a, delta_b = delta, -delta

        # Second horizon : meme signal, pas plus grand, memoire plus courte.
        fa = ta.fast_rating if ta.last_date else ra
        fb = tb.fast_rating if tb.last_date else rb
        ta.fast_rating = fa + self.cfg.fast_multiplier * delta_a
        tb.fast_rating = fb + self.cfg.fast_multiplier * delta_b
        # La note rapide est rappelee vers la note lente, sinon elle derive.
        ta.fast_rating += (ta.rating - ta.fast_rating) * 0.10
        tb.fast_rating += (tb.rating - tb.fast_rating) * 0.10

        ta.last_date = tb.last_date = result.date
        ta.maps_played += 1
        tb.maps_played += 1

        # Notes specifiques a la map, sur le meme principe.
        mname = result.map_name
        mra = ta.map_ratings.get(mname, ra)
        mrb = tb.map_ratings.get(mname, rb)
        exp_map_a = expected_score(mra, mrb)
        km = self.cfg.k_map * result.tier_weight
        if self.cfg.use_mov:
            md = (mra - mrb) if result.a_won else (mrb - mra)
            km *= _mov_multiplier(result.round_diff, md)
        dm = km * (score_a - exp_map_a)
        ta.map_ratings[mname] = mra + dm
        tb.map_ratings[mname] = mrb - dm
        ta.map_counts[mname] = ta.map_counts.get(mname, 0) + 1
        tb.map_counts[mname] = tb.map_counts.get(mname, 0) + 1
