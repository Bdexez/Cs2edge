"""Le pronostiqueur : decider quoi annoncer, et quand se taire.

Un estimateur de probabilites et un pronostiqueur ne font pas le meme metier.
Le premier repond a toutes les questions ; le second doit choisir lesquelles
meritent une reponse. La selectivite *est* le produit : un pronostiqueur qui se
prononce sur tous les matchs n'apporte rien, parce que son lecteur ne sait pas
distinguer ses convictions de ses devinettes.

Deux modes, volontairement separes, parce qu'on les confond tout le temps :

  - mode ANALYSE (sans cotes) : "qui gagne, et a quel point j'en suis sur".
    On optimise le taux de reussite. Annoncer le favori evident est bon ici.

  - mode MARCHE (avec cotes) : "ou le marche se trompe-t-il".
    On optimise la valeur. Le taux de reussite y est structurellement PLUS BAS,
    parce que la valeur se trouve surtout sur les outsiders. Un pronostiqueur
    qui vante son taux de reussite en mode marche vend une illusion.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .markets import (
    MarketLine, build_markets, enforce_margins, fit_decider_calibration,
    project_veto, series_distribution,
)
from .odds import devig_power, expected_value, kelly_fraction
from .predict import Analyzer, Prediction

# Seuils de tri. Valides par `cs2edge tipster-backtest` : chaque palier doit
# tenir sa promesse sur l'historique, sinon il ment a son lecteur.
TIER_SOLIDE = 0.70
TIER_INTERESSANT = 0.60
# On se tait en dessous du premier palier. Un troisieme palier "serre" avait ete
# essaye entre 0.57 et 0.60 : bande trop etroite, 62 pronostics, ecart de +7.6
# points entre promesse et realisation - c'est-a-dire du bruit presente comme un
# jugement. Deux paliers qui tiennent valent mieux que trois dont un ment.
SEUIL_ABSTENTION = TIER_INTERESSANT
# En mode marche : ecart minimal avec la cote debarrassee de sa marge.
EDGE_MIN = 0.04
# Au-dela, on ne croit plus le modele mais le marche (compo de derniere minute).
EDGE_ABSURDE = 0.22


@dataclass
class Fixture:
    """Un match a venir."""

    date: dt.date
    team_a: str
    team_b: str
    best_of: int = 3
    lan: bool = False
    event: str = ""
    odds_a: float | None = None
    odds_b: float | None = None
    odds_over: float | None = None   # total maps au-dessus de la ligne
    odds_under: float | None = None

    @property
    def label(self) -> str:
        return f"{self.team_a} vs {self.team_b}"


@dataclass
class Pick:
    """Un pronostic emis, avec tout ce qu'il faut pour le juger apres coup."""

    date: dt.date
    team_a: str
    team_b: str
    best_of: int
    event: str
    market: str
    selection: str
    probability: float
    tier: str
    verdict: str
    rationale: list[str] = field(default_factory=list)
    odds: float | None = None
    edge: float | None = None
    ev: float | None = None
    stake: float | None = None

    @property
    def fair_odds(self) -> float:
        return 1.0 / max(self.probability, 1e-9)

    @property
    def is_actionable(self) -> bool:
        return self.verdict in ("PRONOSTIC", "VALEUR")


@dataclass
class Analysis:
    """Tout ce que le pronostiqueur a vu sur un match, retenu ou non."""

    fixture: Fixture
    prediction: Prediction
    veto: object
    distribution: object
    markets: list[MarketLine]
    picks: list[Pick]
    skipped: str | None = None   # raison de l'abstention

    @property
    def acted(self) -> bool:
        return any(p.is_actionable for p in self.picks)


def _tier(p: float) -> str:
    return "solide" if p >= TIER_SOLIDE else "interessant"


def shape_series(veto, best_of: int, p_a: float, momentum: float, decider_temp: float):
    """Loi du score coherente avec les deux quantites que l'on calibre.

    Le veto donne la structure, le modele match donne le vainqueur, la
    calibration donne la longueur de la serie. `enforce_margins` reconcilie les
    trois. Sans cette etape, le pronostic "vainqueur" et le pronostic "score
    exact" issus du meme match pourraient se contredire - defaut classique des
    grilles de pronostics, et immediatement visible pour un lecteur attentif.
    """
    import math

    dist = series_distribution(veto.order, best_of, momentum)
    if best_of < 3:
        return enforce_margins(dist, p_a, 1.0)
    raw = min(max(dist.p_maps_played(best_of), 1e-6), 1 - 1e-6)
    p_full = 1.0 / (1.0 + math.exp(-(math.log(raw / (1 - raw)) / max(decider_temp, 1e-6))))
    return enforce_margins(dist, p_a, p_full)


def select_analysis_picks(
    date: dt.date,
    team_a: str,
    team_b: str,
    best_of: int,
    event: str,
    p_a: float,
    dist,
    abstention: float = SEUIL_ABSTENTION,
) -> list[Pick]:
    """Regles de selection en mode analyse, sans prose ni etat.

    Fonction pure, appelee a la fois par le pronostiqueur en direct et par son
    propre backtest. C'est la seule facon d'etre sur que le bilan publie porte
    bien sur les regles reellement appliquees - une logique de decision ecrite
    en deux exemplaires finit toujours par diverger, et le bilan devient faux
    sans que personne ne s'en apercoive.
    """
    picks: list[Pick] = []
    favorite = team_a if p_a >= 0.5 else team_b
    p_fav = max(p_a, 1 - p_a)
    if p_fav < abstention:
        return picks

    picks.append(
        Pick(
            date=date, team_a=team_a, team_b=team_b, best_of=best_of, event=event,
            market="vainqueur", selection=favorite, probability=p_fav,
            tier=_tier(p_fav), verdict="PRONOSTIC",
        )
    )

    # Marche derive : c'est la que la structure du veto apporte le plus, parce
    # que les modeles i.i.d. s'y trompent systematiquement.
    if best_of >= 3 and dist is not None:
        line = best_of - 0.5
        p_over = dist.p_maps_played(best_of)
        side, p = ("+", p_over) if p_over >= 0.5 else ("-", 1 - p_over)
        if p >= TIER_SOLIDE:
            picks.append(
                Pick(
                    date=date, team_a=team_a, team_b=team_b, best_of=best_of,
                    event=event, market=f"total maps {side}{line}", selection="oui",
                    probability=p, tier=_tier(p), verdict="PRONOSTIC",
                )
            )
    return picks


class Tipster:
    """Transforme des matchs a venir en pronostics assumes."""

    def __init__(
        self,
        analyzer: Analyzer,
        momentum: float = 0.0,
        edge_min: float = EDGE_MIN,
        abstention: float = SEUIL_ABSTENTION,
        decider_temperature: float = 1.0,
    ) -> None:
        self.an = analyzer
        self.momentum = momentum
        # Calibration du marche "la serie va-t-elle a la derniere map".
        # A estimer sur vos donnees : `cs2edge tipster-backtest` l'affiche.
        self.decider_temperature = decider_temperature
        self.edge_min = edge_min
        self.abstention = abstention

    # ------------------------------------------------------------- raisonnement

    def _rationale(self, pred: Prediction, veto, dist) -> list[str]:
        """Explication en clair, appuyee sur des nombres, pas sur des adjectifs."""
        a, b = pred.team_a, pred.team_b
        fav = pred.favorite
        out: list[str] = []

        elo_a = self.an.fx.elo.rating(a, pred.date)
        elo_b = self.an.fx.elo.rating(b, pred.date)
        gap = elo_a - elo_b
        sens = a if gap > 0 else b
        out.append(
            f"Ecart de niveau : {abs(gap):.0f} points Elo en faveur de {sens} "
            f"({elo_a:.0f} contre {elo_b:.0f})."
        )

        # Ce qui pousse *contre* l'ecart de niveau merite d'etre dit.
        for name, contrib in pred.drivers[:3]:
            if name == "elo_diff":
                continue
            who = a if contrib > 0 else b
            libelle = {
                "round_form_diff": "la forme recente au differentiel de rounds",
                "form_diff": "le taux de victoire recent",
                "sos_diff": "le niveau des adversaires affrontes",
                "pool_mean_diff": "le pool de maps",
                "pool_top3_diff": "la marge de manoeuvre au veto",
                "h2h_diff": "les confrontations directes",
                "streak_diff": "la serie en cours",
                "rest_diff": "les jours de repos",
                "workload_diff": "la charge de matchs recente",
                "elo_diff_x_lan": "le contexte LAN",
                "elo_diff_x_bo3": "le format long",
            }.get(name, name)
            if abs(contrib) > 0.08:
                out.append(f"{libelle.capitalize()} penche vers {who}.")
                break

        if veto.order:
            roles = []
            for i, (name, p) in enumerate(veto.order):
                if name in veto.picked_by_a:
                    role = f"pick {a}"
                elif name in veto.picked_by_b:
                    role = f"pick {b}"
                else:
                    role = "decisive"
                roles.append(f"{name} ({role}, {p:.0%})")
            # Ces pourcentages sont la lecture brute du terrain. La loi du score
            # affichee plus bas est en plus recalee sur le modele match et sur la
            # calibration de longueur de serie : les deux ne se multiplient donc
            # pas exactement, et il vaut mieux le dire que de laisser croire.
            out.append(
                f"Terrain projete apres veto (lecture brute) : {', '.join(roles)}."
            )

        if pred.best_of >= 3:
            p3 = dist.p_maps_played(pred.best_of)
            if p3 > 0.55:
                out.append(
                    f"Le veto rend les deux camps solides sur leur pick : "
                    f"{p3:.0%} de chances d'aller a la derniere map."
                )
            elif p3 < 0.40:
                out.append(
                    f"Pool desequilibre : {1 - p3:.0%} de chances que la serie "
                    f"se termine sans derniere map."
                )

        for w in pred.warnings:
            out.append(f"Reserve : {w}.")
        return out

    # ----------------------------------------------------------------- analyse

    def analyse(self, fx: Fixture) -> Analysis:
        pred = self.an.predict(fx.team_a, fx.team_b, fx.best_of, fx.lan, fx.date)
        pool = [(e.map_name, e.p_a) for e in pred.map_edges]
        veto = project_veto(pool, fx.best_of)
        dist = shape_series(
            veto, fx.best_of, pred.p_a, self.momentum, self.decider_temperature
        )

        markets = build_markets(dist, pred.team_a, pred.team_b)
        rationale = self._rationale(pred, veto, dist)

        analysis = Analysis(
            fixture=fx, prediction=pred, veto=veto, distribution=dist,
            markets=markets, picks=[],
        )

        if pred.confidence == "faible":
            analysis.skipped = "donnees insuffisantes sur au moins une des deux equipes"
            return analysis

        market_mode = fx.odds_a is not None and fx.odds_b is not None
        if market_mode:
            analysis.picks = self._value_picks(fx, pred, dist, rationale)
        else:
            analysis.picks = self._analysis_picks(fx, pred, dist, rationale)
        if not analysis.picks:
            analysis.skipped = (
                "aucun ecart exploitable avec le marche" if market_mode
                else f"match trop serre ({pred.p_favorite:.0%}) pour un pronostic assume"
            )
        return analysis

    # ------------------------------------------------------- mode sans cotes

    def _analysis_picks(self, fx, pred, dist, rationale) -> list[Pick]:
        picks = select_analysis_picks(
            fx.date, pred.team_a, pred.team_b, fx.best_of, fx.event,
            pred.p_a, dist, self.abstention,
        )
        for pk in picks:
            if pk.market == "vainqueur":
                pk.rationale = rationale
            else:
                pk.rationale = [
                    "Loi du score issue du veto projete : "
                    + ", ".join(
                        f"{a}-{b} {q:.0%}"
                        for (a, b), q in sorted(dist.scores.items(), key=lambda kv: -kv[1])
                    )
                    + "."
                ]
        return picks

    # ------------------------------------------------------- mode avec cotes

    def _value_picks(self, fx, pred, dist, rationale) -> list[Pick]:
        picks: list[Pick] = []
        ma, mb = devig_power(fx.odds_a, fx.odds_b)
        for team, p_model, p_market, odds in (
            (pred.team_a, pred.p_a, ma, fx.odds_a),
            (pred.team_b, 1 - pred.p_a, mb, fx.odds_b),
        ):
            edge = p_model - p_market
            if edge < self.edge_min:
                continue
            verdict = "VALEUR"
            extra = list(rationale)
            if edge > EDGE_ABSURDE:
                verdict = "SUSPECT"
                extra.append(
                    f"Ecart de {edge:+.0%} avec le marche : trop gros pour etre credible. "
                    "Le marche sait probablement quelque chose que l'historique ignore "
                    "(remplacant, roster change, forfait). Verifier la compo avant d'agir."
                )
            picks.append(
                Pick(
                    date=fx.date, team_a=pred.team_a, team_b=pred.team_b,
                    best_of=fx.best_of, event=fx.event, market="vainqueur",
                    selection=team, probability=p_model, tier=_tier(p_model),
                    verdict=verdict, rationale=extra, odds=odds, edge=edge,
                    ev=expected_value(p_model, odds),
                    stake=kelly_fraction(p_model, odds),
                )
            )

        if fx.best_of >= 3 and fx.odds_over and fx.odds_under:
            line = fx.best_of - 0.5
            p_over = dist.p_maps_played(fx.best_of)
            mo, mu = devig_power(fx.odds_over, fx.odds_under)
            for label, p_model, p_market, odds in (
                (f"total maps +{line}", p_over, mo, fx.odds_over),
                (f"total maps -{line}", 1 - p_over, mu, fx.odds_under),
            ):
                edge = p_model - p_market
                if edge < self.edge_min:
                    continue
                picks.append(
                    Pick(
                        date=fx.date, team_a=pred.team_a, team_b=pred.team_b,
                        best_of=fx.best_of, event=fx.event, market=label,
                        selection="oui", probability=p_model, tier=_tier(p_model),
                        verdict="VALEUR", odds=odds, edge=edge,
                        ev=expected_value(p_model, odds),
                        stake=kelly_fraction(p_model, odds),
                        rationale=[
                            "Marche derive de la loi du score : les cotes de total de maps "
                            "sont souvent posees a partir d'un modele i.i.d. qui ignore "
                            "la structure du veto."
                        ],
                    )
                )
        return picks

    def card(self, fixtures: list[Fixture]) -> list[Analysis]:
        out = []
        for fx in sorted(fixtures, key=lambda f: (f.date, f.team_a)):
            try:
                out.append(self.analyse(fx))
            except ValueError as exc:
                out.append(
                    Analysis(
                        fixture=fx, prediction=None, veto=None, distribution=None,
                        markets=[], picks=[], skipped=str(exc),
                    )
                )
        return out



# ---------------------------------------------------------------------------
# Backtest du pronostiqueur lui-meme.
#
# Le backtest du modele repond a "mes probabilites sont-elles justes". Celui-ci
# repond a une autre question, la seule qui interesse un lecteur : "quand cet
# outil annonce quelque chose, qu'est-ce que ca vaut". Les deux ne coincident
# pas, parce que le pronostiqueur choisit ses sujets - et ce choix peut tres
# bien degrader le resultat plutot que l'ameliorer.
# ---------------------------------------------------------------------------


def _flag(stat: dict) -> str:
    """N'alerte que si l'ecart depasse le bruit d'echantillonnage."""
    if stat["n"] < 30:
        return "   (n trop faible pour conclure)"
    return "   <- ecart reel, a corriger" if stat["significatif"] else ""


@dataclass
class SettledPick:
    pick: Pick
    won: bool


@dataclass
class TipsterRecord:
    settled: list[SettledPick]
    n_fixtures: int
    n_skipped: int
    dispersions: list[float] = field(default_factory=list)

    def by(self, key) -> dict:
        groups: dict[str, list[SettledPick]] = {}
        for s in self.settled:
            groups.setdefault(key(s.pick), []).append(s)
        out = {}
        for name, items in sorted(groups.items()):
            n = len(items)
            hits = sum(1 for s in items if s.won)
            promised = sum(s.pick.probability for s in items) / n
            observed = hits / n
            # Erreur type de la frequence observee : sans elle, on prend un
            # ecart de bruit pour un defaut de calibration. C'est le travers
            # classique des bilans de pronostiqueurs, qui commentent des
            # echantillons de vingt lignes comme s'ils demontraient quelque chose.
            se = (promised * (1 - promised) / n) ** 0.5
            out[name] = {
                "n": n,
                "annonce": promised,
                "observe": observed,
                "ecart": observed - promised,
                "erreur_type": se,
                "significatif": n >= 30 and abs(observed - promised) > 2 * se,
            }
        return out

    def report(self) -> str:
        lines = [
            f"Bilan du pronostiqueur : {self.n_fixtures} matchs presentes, "
            f"{self.n_skipped} ecartes ({self.n_skipped / max(1, self.n_fixtures):.0%}), "
            f"{len(self.settled)} pronostics emis.",
            "",
            "Par palier de confiance (un palier qui ne tient pas sa promesse ment) :",
            f"  {'palier':<14}{'n':>6}{'annonce':>10}{'observe':>10}{'ecart':>9}    marge",
        ]
        order = {"solide": 0, "interessant": 1}
        for name, s in sorted(self.by(lambda p: p.tier).items(), key=lambda kv: order.get(kv[0], 9)):
            lines.append(
                f"  {name:<14}{s['n']:>6}{s['annonce']:>9.1%}{s['observe']:>10.1%}"
                f"{s['ecart']:>+9.1%}  +/-{2 * s['erreur_type']:.1%}{_flag(s)}"
            )
        lines += [
            "",
            "Par marche :",
            f"  {'marche':<20}{'n':>6}{'annonce':>10}{'observe':>10}{'ecart':>9}    marge",
        ]
        for name, s in self.by(lambda p: p.market).items():
            lines.append(
                f"  {name:<20}{s['n']:>6}{s['annonce']:>9.1%}{s['observe']:>10.1%}"
                f"{s['ecart']:>+9.1%}  +/-{2 * s['erreur_type']:.1%}{_flag(s)}"
            )
        if self.dispersions:
            lo, hi = min(self.dispersions), max(self.dispersions)
            lines += [
                "",
                f"Calibration de la longueur de serie, reestimee a chaque fold : "
                f"T = {lo:.2f} a {hi:.2f} (1.00 = le modele de veto tombe juste ; "
                f"au-dessus, il annonce des series trop longues).",
            ]
        return "\n".join(lines)


def _settle(pick: Pick, meta: dict, y_a: float) -> bool:
    """Un pronostic est-il tombe juste ? Determine a partir du resultat reel."""
    if pick.market == "vainqueur":
        winner = pick.team_a if y_a == 1 else pick.team_b
        return pick.selection == winner
    if pick.market.startswith("total maps"):
        wa, wb = (int(x) for x in meta["score"].split("-"))
        played = wa + wb
        line = float(pick.market.split()[-1].lstrip("+-"))
        return played > line if "+" in pick.market else played < line
    raise ValueError(f"marche non reglable : {pick.market}")


def _fit_decider_on(match_ds, map_model, upto: int, lookback: int = 800,
                    momentum: float = 0.0) -> float:
    """Calibre la longueur de serie sur les matchs *anterieurs* a `upto`."""
    from .schema import ACTIVE_DUTY

    samples = []
    for r in range(max(0, upto - lookback), upto):
        meta = match_ds.meta[r]
        bo = meta.get("best_of", 3)
        if bo < 3:
            continue
        ps = map_model.predict_proba(match_ds.pool_X[r])
        plan = project_veto(list(zip(ACTIVE_DUTY, ps.tolist())), bo)
        dist = series_distribution(plan.order, bo, momentum)
        wa, wb = (int(x) for x in meta["score"].split("-"))
        samples.append((dist.p_maps_played(bo), int(wa + wb == bo)))
    return fit_decider_calibration(samples)


def simulate_tipster(
    match_ds,
    map_ds,
    initial_frac: float = 0.4,
    refit_every: int = 150,
    min_train: int = 200,
    abstention: float = SEUIL_ABSTENTION,
    momentum: float = 0.0,
    decider_temperature: float | None = None,
    **model_kwargs,
):
    """Rejoue le pronostiqueur sur l'historique, en n'utilisant que le passe.

    `decider_temperature=None` : la calibration est reestimee a chaque fold sur
    les seules series anterieures, comme elle le serait en production. La figer
    a une valeur ajustee sur toute la periode donnerait un bilan flatteur et faux.
    """
    import numpy as np

    from .backtest import _pool_to_series
    from .model import MatchPredictor
    from .predict import combine_probability, confidence_label
    from .schema import ACTIVE_DUTY

    n = len(match_ds)
    start = max(min_train, int(n * initial_frac))
    if n - start < 50:
        raise ValueError(f"historique trop court ({n} matchs) pour juger le pronostiqueur")

    settled: list[SettledPick] = []
    dispersions: list[float] = []
    n_skipped = 0
    i = start
    while i < n:
        j = min(n, i + refit_every)
        model = MatchPredictor(**model_kwargs).fit(
            match_ds.X[:i], match_ds.y[:i], match_ds.dates[:i], match_ds.feature_names
        )
        cutoff = match_ds.dates[i]
        k = sum(1 for d in map_ds.dates if d < cutoff)
        mk = dict(model_kwargs)
        mk["use_gbdt"] = False
        map_model = MatchPredictor(**mk).fit(
            map_ds.X[:k], map_ds.y[:k], map_ds.dates[:k], map_ds.feature_names
        )
        p_match_all = model.predict_proba(match_ds.X[i:j])
        t_fold = decider_temperature
        if t_fold is None:
            t_fold = _fit_decider_on(match_ds, map_model, i, momentum=momentum)
        dispersions.append(t_fold)

        for r in range(i, j):
            meta = match_ds.meta[r]
            bo = meta.get("best_of", 3)
            conf = confidence_label(
                meta.get("maps_played_min", 999),
                meta.get("days_idle_max", 0),
                meta.get("h2h_n", 0),
            )
            if conf == "faible":
                n_skipped += 1
                continue

            pool_ps = map_model.predict_proba(match_ds.pool_X[r])
            pool = list(zip(ACTIVE_DUTY, pool_ps.tolist()))
            p_maps = _pool_to_series(map_model, match_ds.pool_X[r], bo, momentum)
            p_final, _ = combine_probability(
                float(p_match_all[r - i]), p_maps,
                meta.get("maps_played_min", 999), meta.get("days_idle_max", 0),
            )
            plan = project_veto(pool, bo)
            dist = shape_series(plan, bo, p_final, momentum, t_fold)
            picks = select_analysis_picks(
                match_ds.dates[r], meta["team_a"], meta["team_b"], bo,
                meta.get("event", ""), p_final, dist, abstention,
            )
            if not picks:
                n_skipped += 1
                continue
            for pk in picks:
                settled.append(SettledPick(pick=pk, won=_settle(pk, meta, match_ds.y[r])))
        i = j

    return TipsterRecord(
        settled=settled, n_fixtures=n - start, n_skipped=n_skipped,
        dispersions=dispersions,
    )


# ---------------------------------------------------------------------------
# Entree : la liste des matchs a venir.
# ---------------------------------------------------------------------------

FIXTURE_COLUMNS = (
    "date,team_a,team_b[,best_of,lan,event,odds_a,odds_b,odds_over,odds_under]"
)


def read_fixtures(path: str | "Path") -> list[Fixture]:
    """Lit un CSV de matchs a venir.

    Colonnes : date,team_a,team_b puis, facultatif, best_of, lan, event et les
    cotes. Les cotes sont facultatives et changent le mode de fonctionnement :
    sans elles le pronostiqueur repond "qui gagne", avec elles il repond "ou le
    marche se trompe". Ce ne sont pas les memes pronostics.
    """
    import csv

    from .schema import normalize_team, parse_date

    def _f(row: dict, key: str) -> float | None:
        raw = (row.get(key) or "").strip()
        if not raw:
            return None
        value = float(raw)
        if value <= 1.0:
            raise ValueError(f"cote decimale invalide pour {key} : {value}")
        return value

    out: list[Fixture] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = {"date", "team_a", "team_b"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"colonnes manquantes : {sorted(missing)} ({FIXTURE_COLUMNS})")
        for i, row in enumerate(reader):
            try:
                out.append(
                    Fixture(
                        date=parse_date(row["date"]),
                        team_a=normalize_team(row["team_a"]),
                        team_b=normalize_team(row["team_b"]),
                        best_of=int(float(row.get("best_of") or 3)),
                        lan=str(row.get("lan", "")).strip().lower() in {"1", "true", "oui", "yes", "lan"},
                        event=(row.get("event") or "").strip(),
                        odds_a=_f(row, "odds_a"),
                        odds_b=_f(row, "odds_b"),
                        odds_over=_f(row, "odds_over"),
                        odds_under=_f(row, "odds_under"),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise ValueError(f"ligne {i + 2} invalide : {exc}") from exc
    return out


def format_card(analyses: list[Analysis], verbose: bool = False) -> str:
    """Met en page la carte du jour, pronostics d'abord, abstentions ensuite."""
    def _conviction(a: Analysis) -> float:
        """En mode marche, l'ecart avec le marche ; sinon, la probabilite."""
        actionable = [p for p in a.picks if p.is_actionable or p.verdict == "SUSPECT"]
        if not actionable:
            return -1.0
        if any(p.edge is not None for p in actionable):
            return max(p.edge for p in actionable if p.edge is not None)
        return max(p.probability for p in actionable)

    retenus = sorted([a for a in analyses if a.acted], key=_conviction, reverse=True)
    ecartes = [a for a in analyses if not a.acted]
    lines: list[str] = []

    if retenus:
        lines.append(f"PRONOSTICS ({len(retenus)} match{'s' if len(retenus) > 1 else ''})")
        lines.append("=" * 72)
    for a in retenus:
        fx = a.fixture
        lines.append("")
        header = f"{fx.team_a} vs {fx.team_b}   bo{fx.best_of}"
        header += ", LAN" if fx.lan else ", online"
        if fx.event:
            header += f"   [{fx.event}]"
        lines.append(header)
        lines.append(f"  {fx.date}")
        for pk in a.picks:
            if not pk.is_actionable and pk.verdict != "SUSPECT":
                continue
            head = f"  -> {pk.market} : {pk.selection}   {pk.probability:.0%}"
            head += f"   [{pk.tier}]"
            if pk.odds:
                head += (
                    f"   cote {pk.odds:.2f} (juste {pk.fair_odds:.2f})"
                    f"   edge {pk.edge:+.1%}   mise {pk.stake:.1%}"
                )
            else:
                head += f"   cote juste {pk.fair_odds:.2f}"
            if pk.verdict == "SUSPECT":
                head += "   << SUSPECT"
            lines.append(head)
            for r in pk.rationale:
                lines.append(f"       . {r}")
        if verbose and a.distribution is not None:
            scores = ", ".join(
                f"{x}-{y} {p:.0%}"
                for (x, y), p in sorted(a.distribution.scores.items(), key=lambda kv: -kv[1])
            )
            lines.append(f"       loi du score : {scores}")

    if ecartes:
        lines += ["", f"ECARTES ({len(ecartes)})", "=" * 72]
        for a in ecartes:
            fx = a.fixture
            lines.append(f"  {fx.team_a} vs {fx.team_b:<18} {a.skipped}")

    if not lines:
        return "Aucun match a analyser."
    lines += [
        "",
        "-" * 72,
        "Rappel : ces probabilites sont calibrees, pas certaines. Un pronostic a "
        "70%\nse trompe trois fois sur dix - c'est le comportement attendu, pas une "
        "anomalie.",
    ]
    return "\n".join(lines)


def fit_decider_temperature(analyzer, momentum: float = 0.0) -> float:
    """Calibre la longueur de serie sur tout l'historique disponible.

    Utilise pour servir des pronostics : ici on veut la meilleure estimation
    possible a l'instant present, donc on regarde tout le passe. Le backtest,
    lui, la reestime a chaque fold pour ne jamais utiliser le futur.

    `momentum` doit etre celui qui sera reellement applique en service : la
    temperature corrige le biais residuel de la loi du score *apres* momentum,
    donc les deux doivent etre coherents, sinon la calibration compense un
    momentum qui ne sera pas la (ou l'inverse).
    """
    if analyzer.match_ds is None or analyzer.map_model is None:
        raise RuntimeError("charger et entrainer avant de calibrer")
    return _fit_decider_on(
        analyzer.match_ds, analyzer.map_model, len(analyzer.match_ds), momentum=momentum
    )
