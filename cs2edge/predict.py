"""Orchestration : charger, entrainer, predire un match a venir."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FeatureExtractor, build_datasets
from .markets import project_veto, series_distribution
from .model import MatchPredictor, series_win_probability
from .ratings import EloConfig
from .schema import ACTIVE_DUTY, Match, MapResult, group_matches, normalize_team, read_csv

# Plafond de confiance. Aucun modele base sur des resultats passes ne peut
# honnetement depasser ce seuil sur un match unique.
MAX_CONFIDENCE = 0.96
# En dessous de ce volume de maps, on ne fait pas confiance a la note.
RELIABLE_MAPS = 60
# Poids du modele match face a la lecture par maps. Valeur mesuree par backtest
# (`cs2edge backtest` affiche l'optimum sur vos propres donnees) : les deux
# estimateurs se valent et se completent, l'optimum est plat autour de 0.5.
MATCH_MODEL_WEIGHT = 0.5
# Ecart entre les deux estimateurs a partir duquel il vaut la peine de prevenir.
# Fixe au 95e centile observe : a 0.15, l'avertissement se declenchait sur un
# match sur cinq et n'informait plus personne.
DIVERGENCE_NOTABLE = 0.25


def probability_from_pool(
    pool_probs: list[tuple[str, float]], best_of: int, momentum: float = 0.0
) -> float:
    """Probabilite de serie a partir des 7 probabilites de map, via le veto.

    Remplace l'ancienne moyenne du milieu du pool : le veto n'est pas un tirage
    au hasard, il est joue par deux equipes qui connaissent leurs terrains.
    """
    plan = project_veto(pool_probs, best_of)
    return series_distribution(plan.order, best_of, momentum).p_a


def confidence_label(maps_played: int, days_since_last: int, h2h_n: int) -> str:
    """Niveau de confiance dans les *donnees*, pas dans le resultat.

    Distinction essentielle : une prediction a 85% sur deux equipes bien connues
    et une prediction a 85% sur une equipe vue six fois ne valent pas la meme
    chose, alors qu'elles affichent le meme nombre.
    """
    if maps_played < 25 or days_since_last > 60:
        return "faible"
    if maps_played < RELIABLE_MAPS or h2h_n == 0:
        return "moyenne"
    return "correcte"


def reliability(maps_played: int, days_since_last: int, certainty: float | None = None) -> float:
    """Facteur dans [0.55, 1] : quelle part de la confiance brute on garde.

    Si l'ecart-type des notes est disponible (`certainty`), il remplace la regle
    ad hoc fondee sur le nombre de maps et l'anciennete : c'est exactement la
    quantite que ces deux indicateurs essayaient d'approcher, en mieux, puisque
    elle tient compte de la qualite des adversaires rencontres et pas seulement
    de leur nombre.
    """
    if certainty is not None:
        return float(min(1.0, max(0.55, 0.55 + 0.45 * (certainty - 0.6) / 0.35)))
    volume = min(1.0, maps_played / RELIABLE_MAPS)
    freshness = 0.5 ** (max(0, days_since_last - 30) / 90.0)
    return 0.55 + 0.45 * volume * freshness


def combine_probability(
    p_match: float,
    p_from_maps: float,
    maps_played_min: int,
    days_idle_max: int,
    weight: float = None,
    certainty: float | None = None,
) -> tuple[float, float]:
    """Melange, retrecit et plafonne. Point de passage unique.

    Le backtest et la prediction en direct appellent tous deux cette fonction :
    c'est la seule facon de garantir que le chiffre mesure est bien le chiffre
    servi. Toute logique de combinaison ecrite en double finit par diverger.
    """
    w = MATCH_MODEL_WEIGHT if weight is None else weight
    p_blend = w * p_match + (1.0 - w) * p_from_maps
    p = 0.5 + (p_blend - 0.5) * reliability(maps_played_min, days_idle_max, certainty)
    return float(min(max(p, 1.0 - MAX_CONFIDENCE), MAX_CONFIDENCE)), float(p_blend)


@dataclass
class MapEdge:
    map_name: str
    elo_edge: float          # en points Elo, vu de A
    p_a: float               # proba que A gagne cette map
    a_experience: int
    b_experience: int


@dataclass
class Prediction:
    team_a: str
    team_b: str
    date: dt.date
    best_of: int
    lan: bool
    p_a: float               # proba finale que A gagne le match
    p_a_match_model: float   # modele match direct
    p_a_from_maps: float     # reconstitue depuis les probas par map
    confidence: str
    drivers: list[tuple[str, float]]
    map_edges: list[MapEdge]
    data_quality: dict
    warnings: list[str] = field(default_factory=list)

    @property
    def favorite(self) -> str:
        return self.team_a if self.p_a >= 0.5 else self.team_b

    @property
    def p_favorite(self) -> float:
        return max(self.p_a, 1 - self.p_a)

    def fair_odds(self) -> tuple[float, float]:
        return 1.0 / max(self.p_a, 1e-6), 1.0 / max(1 - self.p_a, 1e-6)

    def report(self) -> str:
        oa, ob = self.fair_odds()
        lines = [
            f"{self.team_a}  vs  {self.team_b}"
            f"   [bo{self.best_of}{', LAN' if self.lan else ', online'}, {self.date}]",
            "",
            f"  {self.team_a:<16} {self.p_a:>6.1%}   cote juste {oa:.2f}",
            f"  {self.team_b:<16} {1 - self.p_a:>6.1%}   cote juste {ob:.2f}",
            "",
            f"  Favori : {self.favorite} ({self.p_favorite:.1%}) - confiance {self.confidence}",
            f"  Modele match {self.p_a_match_model:.1%} | reconstitue par maps "
            f"{self.p_a_from_maps:.1%} (vu de {self.team_a})",
            "",
            "  Ce qui pese (contribution au logit, vu de " + self.team_a + ") :",
        ]
        for name, val in self.drivers:
            bar = "#" * min(20, int(abs(val) * 12))
            lines.append(f"    {name:<22} {val:+.3f}  {bar}")
        lines += ["", "  Pool de maps (avantage en points Elo, vu de " + self.team_a + ") :"]
        for e in self.map_edges:
            flag = ""
            if min(e.a_experience, e.b_experience) < 4:
                flag = "  (peu de donnees)"
            lines.append(
                f"    {e.map_name:<10} {e.elo_edge:+7.1f}   p={e.p_a:.1%}"
                f"   [{e.a_experience}v{e.b_experience} maps]{flag}"
            )
        q = self.data_quality
        lines += [
            "",
            f"  Historique : {self.team_a} {q['maps_a']} maps (derniere {q['last_a']}), "
            f"{self.team_b} {q['maps_b']} maps (derniere {q['last_b']}), "
            f"confrontations directes {q['h2h']}",
        ]
        for w in self.warnings:
            lines.append(f"  /!\\ {w}")
        return "\n".join(lines)


class Analyzer:
    """Point d'entree unique : donnees -> etat -> modele -> prediction."""

    def __init__(
        self,
        elo_config: EloConfig | None = None,
        warmup_maps: int = 12,
        momentum: float = 0.0,
    ) -> None:
        self.elo_config = elo_config
        self.warmup_maps = warmup_maps
        # Effet d'enchainement entre maps. A mesurer sur vos donnees
        # (`markets.estimate_momentum`) avant d'etre suppose non nul.
        self.momentum = momentum
        self.maps: list[MapResult] = []
        self.matches: list[Match] = []
        self.fx: FeatureExtractor | None = None
        self.match_ds = None
        self.map_ds = None
        self.match_model: MatchPredictor | None = None
        self.map_model: MatchPredictor | None = None

    # ------------------------------------------------------------------ setup

    def load_csv(self, path: str | Path) -> "Analyzer":
        return self.load_maps(read_csv(str(path)))

    def load_maps(self, maps: list[MapResult]) -> "Analyzer":
        self.maps = maps
        self.matches = group_matches(maps)
        self.match_ds, self.map_ds, self.fx = build_datasets(
            self.matches, self.elo_config, self.warmup_maps
        )
        return self

    def train(self, **kwargs) -> "Analyzer":
        if self.match_ds is None:
            raise RuntimeError("charger des donnees avant d'entrainer")
        self.match_model = MatchPredictor(**kwargs).fit(
            self.match_ds.X, self.match_ds.y, self.match_ds.dates, self.match_ds.feature_names
        )
        map_kwargs = dict(kwargs)
        map_kwargs.setdefault("use_gbdt", False)
        self.map_model = MatchPredictor(**map_kwargs).fit(
            self.map_ds.X, self.map_ds.y, self.map_ds.dates, self.map_ds.feature_names
        )
        return self

    # ------------------------------------------------------------------ usage

    def known_teams(self) -> list[str]:
        return sorted(self.fx.elo.teams) if self.fx else []

    def resolve_team(self, name: str) -> str:
        canonical = normalize_team(name)
        if self.fx and canonical in self.fx.elo.teams:
            return canonical
        candidates = [t for t in self.known_teams() if canonical in t or t in canonical]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise ValueError(f"'{name}' est ambigu : {candidates[:8]}")
        raise ValueError(f"equipe inconnue : '{name}' (voir la commande `teams`)")

    def predict(
        self,
        team_a: str,
        team_b: str,
        best_of: int = 3,
        lan: bool = False,
        on: dt.date | None = None,
    ) -> Prediction:
        if self.match_model is None or self.fx is None:
            raise RuntimeError("modele non entraine")
        a, b = self.resolve_team(team_a), self.resolve_team(team_b)
        if a == b:
            raise ValueError("une equipe ne joue pas contre elle-meme")
        on = on or max(m.date for m in self.matches)

        x = self.fx.match_features(a, b, on, best_of, lan)
        p_match = self.match_model.predict_one(x)

        edges: list[MapEdge] = []
        for m in ACTIVE_DUTY:
            xm = self.fx.map_features(a, b, on, m)
            edges.append(
                MapEdge(
                    map_name=m,
                    elo_edge=self.fx.elo.map_rating(a, m, on) - self.fx.elo.map_rating(b, m, on),
                    p_a=self.map_model.predict_one(xm),
                    a_experience=self.fx.elo.teams[a].map_counts.get(m, 0),
                    b_experience=self.fx.elo.teams[b].map_counts.get(m, 0),
                )
            )
        edges.sort(key=lambda e: -e.elo_edge)

        k = max(1, best_of)
        p_from_maps = probability_from_pool(
            [(e.map_name, e.p_a) for e in edges], best_of, self.momentum
        )


        ta, tb = self.fx.elo.teams[a], self.fx.elo.teams[b]
        h2h_key = (min(a, b), max(a, b))
        n_h2h = len(self.fx.h2h.get(h2h_key, ()))
        quality = {
            "maps_a": ta.maps_played, "maps_b": tb.maps_played,
            "last_a": ta.last_date.isoformat() if ta.last_date else "jamais",
            "last_b": tb.last_date.isoformat() if tb.last_date else "jamais",
            "h2h": n_h2h,
        }

        thin = min(ta.maps_played, tb.maps_played)
        stale = max(
            (on - ta.last_date).days if ta.last_date else 999,
            (on - tb.last_date).days if tb.last_date else 999,
        )

        # Le modele match voit la forme, le h2h, le repos ; la lecture par maps
        # voit le terrain. Ni l'un ni l'autre ne domine : le melange bat les deux.
        # Puis retrecissement si l'historique est mince ou perime, et plafond dur :
        # un bo3 de CS2 n'est jamais joue d'avance.
        p_final, p_blend = combine_probability(p_match, p_from_maps, thin, stale)

        warnings: list[str] = []
        if thin < 25:
            warnings.append(
                f"historique mince ({thin} maps pour la moins connue) : proba peu fiable"
            )
        if stale > 60:
            warnings.append(
                f"une des equipes n'a pas joue depuis {stale} jours : note amortie vers la moyenne"
            )
        if abs(p_match - p_from_maps) > DIVERGENCE_NOTABLE:
            warnings.append(
                "modele match et lecture par maps divergent : le veto sera decisif"
            )
        if abs(p_final - p_blend) > 0.02:
            reason = (
                f"plafond de confiance {MAX_CONFIDENCE:.0%} : sur un match unique, "
                "aucun historique ne justifie mieux"
                if max(p_blend, 1 - p_blend) > MAX_CONFIDENCE
                else "historique trop mince ou trop ancien pour assumer cette assurance"
            )
            warnings.append(f"probabilite ramenee de {p_blend:.1%} a {p_final:.1%} ({reason})")
        if k >= 3 and abs(p_final - 0.5) < 0.04:
            warnings.append("match a pile ou face : aucune valeur a chercher ici")

        confidence = confidence_label(thin, stale, n_h2h)

        return Prediction(
            team_a=a, team_b=b, date=on, best_of=best_of, lan=lan,
            p_a=float(p_final), p_a_match_model=float(p_match),
            p_a_from_maps=float(p_from_maps), confidence=confidence,
            drivers=self.match_model.explain(x), map_edges=edges,
            data_quality=quality, warnings=warnings,
        )
