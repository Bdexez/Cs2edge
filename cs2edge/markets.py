"""Du niveau map au niveau serie, en passant par le veto.

Le modele precedent convertissait une probabilite de map en probabilite de
serie en supposant les maps independantes et interchangeables. C'est faux, et
l'erreur n'est pas neutre : dans un bo3, la map 1 est choisie par A, la map 2
par B, et la decisive est ce qui reste apres les bans. Les trois maps ne sont
donc pas tirees au hasard dans le pool - elles sont systematiquement biaisees,
la premiere en faveur de A, la deuxieme en faveur de B.

Consequence directe : la probabilite d'un 2-0 est nettement plus basse que ce
que donne le modele i.i.d., et celle d'un 2-1 nettement plus haute. C'est
exactement la ou se trouvent les marches derives (score exact, handicap map,
total de maps), donc c'est exactement la ou l'approximation coute cher.

Sequence de veto standard (Valve / tournois majeurs) :
  bo1 : A ban, B ban, A ban, B ban, A ban, B ban -> il reste 1 map
  bo3 : A ban, B ban, A pick, B pick, A ban, B ban -> il reste la decisive
  bo5 : A ban, B ban, A pick, B pick, A pick, B pick -> il reste la decisive
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .validation import finite_number, positive_integer

EPS = 1e-9


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _validate_format(best_of: int) -> None:
    positive_integer("best_of", best_of)
    if best_of not in (1, 3, 5):
        raise ValueError(f"format non gere : bo{best_of}")


def _validate_maps(maps: list[tuple[str, float]]) -> None:
    names = []
    for name, p in maps:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("chaque map doit avoir un nom non vide")
        finite_number("probabilite de map", p, maximum=1.0)
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("les maps doivent etre distinctes")


@dataclass
class VetoPlan:
    """Maps retenues, dans l'ordre de jeu, avec la proba que A gagne chacune."""

    order: list[tuple[str, float]]
    banned_by_a: list[str] = field(default_factory=list)
    banned_by_b: list[str] = field(default_factory=list)
    picked_by_a: list[str] = field(default_factory=list)
    picked_by_b: list[str] = field(default_factory=list)
    decider: str | None = None

    def describe(self) -> str:
        bits = []
        if self.picked_by_a:
            bits.append(f"pick A : {', '.join(self.picked_by_a)}")
        if self.picked_by_b:
            bits.append(f"pick B : {', '.join(self.picked_by_b)}")
        if self.decider:
            bits.append(f"decisive probable : {self.decider}")
        return " | ".join(bits)


def project_veto(map_probs: list[tuple[str, float]], best_of: int) -> VetoPlan:
    """Deroule le veto en supposant que chacun joue son interet.

    Hypothese : chaque equipe bannit la map ou elle est la plus faible parmi
    celles qui restent, et choisit celle ou elle est la plus forte. C'est une
    simplification - les equipes bannissent parfois selon la preparation
    adverse plutot que selon leur propre force - mais elle capture l'essentiel
    et se trompe dans le bon sens : elle rend la serie plus equilibree, pas moins.
    """
    _validate_format(best_of)
    _validate_maps(map_probs)
    # Trie decroissant : le plus favorable a A en tete.
    pool = sorted(map_probs, key=lambda kv: -kv[1])
    if not pool or (best_of >= 3 and len(pool) != 7):
        raise ValueError(f"le veto bo{best_of} exige "
                         + ("un pool de sept maps" if best_of >= 3 else "un pool non vide"))

    plan = VetoPlan(order=[])
    # bo1 : uniquement des bans, en alternance, jusqu'a ce qu'il reste une map.
    if best_of == 1:
        turn_a = True
        while len(pool) > 1:
            if turn_a:
                plan.banned_by_a.append(pool.pop()[0])   # A retire son pire terrain
            else:
                plan.banned_by_b.append(pool.pop(0)[0])  # B retire le meilleur pour A
            turn_a = not turn_a
        plan.decider = pool[0][0]
        plan.order = [pool[0]]
        return plan

    # bo3 / bo5 : deux bans d'ouverture, puis des picks alternes.
    plan.banned_by_a.append(pool.pop()[0])
    plan.banned_by_b.append(pool.pop(0)[0])

    n_picks = best_of - 1  # la derniere map est la decisive
    picks: list[tuple[str, float]] = []
    for k in range(n_picks):
        if k % 2 == 0:
            picks.append(pool.pop(0))    # A choisit son meilleur terrain restant
            plan.picked_by_a.append(picks[-1][0])
        else:
            picks.append(pool.pop())     # B choisit le pire terrain pour A
            plan.picked_by_b.append(picks[-1][0])

    # bo3 : deux bans supplementaires avant la decisive.
    if best_of == 3:
        plan.banned_by_a.append(pool.pop()[0])
        plan.banned_by_b.append(pool.pop(0)[0])
    decider = pool[int(len(pool) // 2)] if len(pool) > 1 else pool[0]
    plan.decider = decider[0]
    plan.order = picks + [decider]
    return plan


@dataclass
class SeriesDistribution:
    """Loi complete du score de la serie."""

    scores: dict[tuple[int, int], float]
    order: list[tuple[str, float]]
    best_of: int

    @property
    def p_a(self) -> float:
        need = self.best_of // 2 + 1
        return sum(p for (wa, _wb), p in self.scores.items() if wa == need)

    def p_score(self, wins_a: int, wins_b: int) -> float:
        return self.scores.get((wins_a, wins_b), 0.0)

    def p_maps_played(self, n: int) -> float:
        return sum(p for (wa, wb), p in self.scores.items() if wa + wb == n)

    def p_clean_sweep(self, for_a: bool = True) -> float:
        need = self.best_of // 2 + 1
        return self.p_score(need, 0) if for_a else self.p_score(0, need)


def series_distribution(
    order: list[tuple[str, float]], best_of: int, momentum: float = 0.0
) -> SeriesDistribution:
    """Enumere toutes les sequences de maps possibles.

    `momentum` decale le logit de la map suivante selon le resultat de la
    precedente. Valeur par defaut 0 : l'effet doit etre *mesure* sur des
    donnees reelles (`estimate_momentum`) avant d'etre suppose. Beaucoup de
    modeles publics l'imposent a l'intuition, ce qui revient a inventer du
    signal.
    """
    _validate_format(best_of)
    _validate_maps(order)
    if len(order) != best_of:
        raise ValueError(f"une serie bo{best_of} exige {best_of} maps ordonnees")
    finite_number("momentum", momentum, minimum=-math.inf)
    need = best_of // 2 + 1
    scores: dict[tuple[int, int], float] = {}

    def walk(idx: int, wa: int, wb: int, prob: float, shift: float) -> None:
        if wa == need or wb == need:
            scores[(wa, wb)] = scores.get((wa, wb), 0.0) + prob
            return
        raw = order[idx][1]
        p = raw if raw in (0.0, 1.0) else _sigmoid(_logit(raw) + shift)
        walk(idx + 1, wa + 1, wb, prob * p, shift + momentum)
        walk(idx + 1, wa, wb + 1, prob * (1 - p), shift - momentum)

    walk(0, 0, 0, 1.0, 0.0)
    total = sum(scores.values())
    if abs(total - 1.0) > 1e-6:  # normalisation defensive
        scores = {k: v / total for k, v in scores.items()}
    return SeriesDistribution(scores=scores, order=order, best_of=best_of)


def estimate_momentum(matches, map_model, extractor) -> dict:
    """Mesure l'effet d'enchainement sur des donnees reelles.

    Question posee : sachant tout ce que le modele sait deja des deux equipes
    et de la map, gagner la map precedente apporte-t-il une information
    supplementaire ? On compare le taux de victoire observe sur la map n+1
    apres une victoire sur la map n, a ce que le modele predisait.

    Retourne le decalage en logit a passer a `series_distribution`.
    """
    from .schema import ACTIVE_DUTY  # import local : evite un cycle

    after_win = {"pred": 0.0, "obs": 0.0, "n": 0}
    after_loss = {"pred": 0.0, "obs": 0.0, "n": 0}

    for match in sorted(matches, key=lambda m: (m.date, m.match_id)):
        maps = sorted(match.maps, key=lambda x: x.map_index)
        if len(maps) >= 2 and extractor.elo.maps_played(match.team_a) > 20:
            for i in range(1, len(maps)):
                prev, cur = maps[i - 1], maps[i]
                x = extractor.map_features(match.team_a, match.team_b, match.date, cur.map_name)
                p = map_model.predict_one(x)
                bucket = after_win if prev.a_won else after_loss
                bucket["pred"] += p
                bucket["obs"] += cur.a_won
                bucket["n"] += 1
        extractor.observe_match(match)

    out = {}
    for label, b in (("apres_victoire", after_win), ("apres_defaite", after_loss)):
        if b["n"] < 30:
            out[label] = None
            continue
        out[label] = {
            "n": b["n"],
            "predit": b["pred"] / b["n"],
            "observe": b["obs"] / b["n"],
            "ecart": (b["obs"] - b["pred"]) / b["n"],
        }
    if out.get("apres_victoire") and out.get("apres_defaite"):
        w, l = out["apres_victoire"], out["apres_defaite"]
        # Decalage symetrique : moitie de l'ecart total entre les deux etats.
        shift = 0.5 * (
            _logit(w["observe"]) - _logit(w["predit"]) - _logit(l["observe"]) + _logit(l["predit"])
        )
        n = w["n"] + l["n"]
        # Erreur type approximative sur ce decalage, pour savoir s'il est reel.
        se = math.sqrt(1.0 / max(1, w["n"]) + 1.0 / max(1, l["n"])) * 2.0
        out["momentum"] = shift
        out["erreur_type"] = se
        out["significatif"] = abs(shift) > 2 * se
        out["n_total"] = n
    return out


# ---------------------------------------------------------------- marches

@dataclass
class MarketLine:
    market: str
    selection: str
    probability: float

    @property
    def fair_odds(self) -> float:
        return 1.0 / max(self.probability, EPS)


def build_markets(
    dist: SeriesDistribution, team_a: str, team_b: str
) -> list[MarketLine]:
    """Derive tous les marches courants de la loi du score."""
    need = dist.best_of // 2 + 1
    lines = [
        MarketLine("vainqueur", team_a, dist.p_a),
        MarketLine("vainqueur", team_b, 1 - dist.p_a),
    ]

    for (wa, wb), p in sorted(dist.scores.items(), key=lambda kv: -kv[1]):
        lines.append(MarketLine("score exact", f"{team_a} {wa}-{wb}" if wa > wb
                                else f"{team_b} {wb}-{wa}", p))

    if dist.best_of >= 3:
        # En BO5, -1.5 couvre aussi 3-1 ; seul -2.5 exige un sweep.
        for margin in range(1, need):
            handicap = margin + 0.5
            pa = sum(p for (wa, wb), p in dist.scores.items() if wa - wb > handicap)
            pb = sum(p for (wa, wb), p in dist.scores.items() if wb - wa > handicap)
            lines.extend([
                MarketLine(f"handicap -{handicap}", team_a, pa),
                MarketLine(f"handicap -{handicap}", team_b, pb),
                MarketLine(f"handicap +{handicap}", team_a, 1 - pb),
                MarketLine(f"handicap +{handicap}", team_b, 1 - pa),
            ])

        line = dist.best_of - 0.5  # 2.5 en bo3, 4.5 en bo5
        p_over = sum(p for (wa, wb), p in dist.scores.items() if wa + wb > line)
        lines.append(MarketLine(f"total maps +{line}", "oui", p_over))
        lines.append(MarketLine(f"total maps -{line}", "oui", 1 - p_over))

    for i, (map_name, p) in enumerate(dist.order, 1):
        label = f"map {i} ({map_name})"
        lines.append(MarketLine(label, team_a, p))
        lines.append(MarketLine(label, team_b, 1 - p))
    return lines


# ---------------------------------------------------------------------------
# Calibration des marches derives.
#
# Premiere tentative, abandonnee : un parametre de "dispersion" corrigeant
# l'hypothese de veto parfait. Il s'est revele instable d'un fold a l'autre
# (0.10 a 1.05) sans corriger le defaut mesure - signe qu'il ne pointait pas la
# bonne cause. Un parametre qui n'ameliore pas ce qu'il vise ne merite pas de
# rester dans le code.
#
# Le defaut reel etait ailleurs : la loi du score est ancree sur une probabilite
# de vainqueur volontairement prudente (retrecie, plafonnee), ce qui tire toute
# la distribution vers l'equilibre et sous-estime les series courtes. La reponse
# correcte n'est pas de deformer le veto, c'est de calibrer la seconde marginale
# comme on calibre deja la premiere, puis d'imposer les deux ensemble.
# ---------------------------------------------------------------------------


def fit_decider_calibration(samples: list[tuple[float, int]]) -> float:
    """Temperature sur "la serie va-t-elle a la derniere map".

    Meme methode que pour le vainqueur : un seul parametre, ajuste hors
    echantillon, qui preserve l'ordre. `samples` = (proba predite, 1 si la
    serie est allee a son terme).
    """
    if len(samples) < 60:
        return 1.0
    z = [_logit(p) for p, _ in samples]
    y = [float(o) for _, o in samples]
    if len(set(y)) < 2:
        return 1.0

    def nll(temp: float) -> float:
        total = 0.0
        for zi, yi in zip(z, y):
            p = min(max(_sigmoid(zi / temp), 1e-12), 1 - 1e-12)
            total -= yi * math.log(p) + (1 - yi) * math.log(1 - p)
        return total

    lo, hi = 0.3, 4.0
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


def enforce_margins(
    dist: SeriesDistribution, p_a: float, p_full_length: float, iterations: int = 40
) -> SeriesDistribution:
    """Impose deux marges a la loi du score sans casser sa structure interne.

    On connait deux quantites mieux que le modele brut : la probabilite de
    victoire (calibree par le modele match) et celle d'aller a la derniere map
    (calibree ci-dessus). Le reste - comment se repartissent les scores a
    l'interieur de chaque cas - reste ce que le modele en dit.

    Ajustement proportionnel iteratif sur le tableau 2x2 (vainqueur x longueur).
    C'est l'outil fait pour ca : il atteint les deux marges tout en preservant
    l'association entre les deux dimensions.
    """
    need = dist.best_of // 2 + 1
    full = dist.best_of

    def cell(wins_a_side: bool, is_full: bool) -> list[tuple[int, int]]:
        return [
            k for k, _ in dist.scores.items()
            if ((k[0] == need) == wins_a_side) and ((k[0] + k[1] == full) == is_full)
        ]

    cells = {
        (w, f): cell(w, f) for w in (True, False) for f in (True, False)
    }
    if any(not keys for keys in cells.values()):
        # bo1, ou format degenere : la longueur n'est pas une dimension libre.
        # On n'impose alors que la marge du vainqueur.
        scores = {
            k: (p_a if k[0] == need else 1 - p_a) * (
                v / max(sum(q for kk, q in dist.scores.items() if (kk[0] == need) == (k[0] == need)), 1e-12)
            )
            for k, v in dist.scores.items()
        }
        tot = sum(scores.values())
        return SeriesDistribution(
            scores={k: v / tot for k, v in scores.items()},
            order=dist.order, best_of=dist.best_of,
        )
    mass = {
        key: max(sum(dist.scores.get(k, 0.0) for k in keys), 1e-9)
        for key, keys in cells.items()
    }

    row_target = {True: max(p_a, 1e-9), False: max(1 - p_a, 1e-9)}
    col_target = {True: max(p_full_length, 1e-9), False: max(1 - p_full_length, 1e-9)}

    for _ in range(iterations):
        for w in (True, False):
            cur = mass[(w, True)] + mass[(w, False)]
            f = row_target[w] / cur
            mass[(w, True)] *= f
            mass[(w, False)] *= f
        for fl in (True, False):
            cur = mass[(True, fl)] + mass[(False, fl)]
            f = col_target[fl] / cur
            mass[(True, fl)] *= f
            mass[(False, fl)] *= f

    scores: dict[tuple[int, int], float] = {}
    for key, keys in cells.items():
        if not keys:
            continue
        inner = sum(dist.scores.get(k, 0.0) for k in keys)
        for k in keys:
            share = (dist.scores.get(k, 0.0) / inner) if inner > 1e-12 else 1.0 / len(keys)
            scores[k] = mass[key] * share
    total = sum(scores.values())
    return SeriesDistribution(
        scores={k: v / total for k, v in scores.items()},
        order=dist.order,
        best_of=dist.best_of,
    )
