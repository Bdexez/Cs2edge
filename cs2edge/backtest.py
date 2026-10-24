"""Backtest walk-forward.

Le seul test honnete : n'entrainer que sur le passe, predire le futur, et
reentrainer periodiquement comme on le ferait en production. Une validation
croisee aleatoire donnerait ici des scores flatteurs et faux, parce que les
notes Elo d'un match donne encodent deja des matchs posterieurs.
"""

from __future__ import annotations

import datetime as dt
import math
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from .features import Dataset
from .metrics import format_summary, log_loss, paired_block_bootstrap, reliability_table, summarize
from .model import MatchPredictor, map_predictor
from .ratings import expected_score
from .schema import ACTIVE_DUTY
from .temporal import forward_splits
from .validation import finite_number, positive_integer


@dataclass
class BacktestResult:
    predictions: np.ndarray          # melange servi par `predict`
    y: np.ndarray
    dates: list[dt.date]
    meta: list[dict]
    elo_only: np.ndarray
    folds: int
    metrics: dict
    elo_metrics: dict
    match_only: np.ndarray | None = None
    maps_only: np.ndarray | None = None
    #  Melange avant garde-fou : sert a distinguer ce que coute le melange de
    #  ce que coute le retrecissement de securite, deux choses differentes.
    blended_raw: np.ndarray | None = None
    blend: float = 0.5
    best_blend: float | None = None
    comparison: dict | None = None

    def report(self) -> str:
        lines = [
            f"Backtest walk-forward : {self.folds} reentrainements, "
            f"{len(self.y)} matchs evalues "
            f"({self.dates[0]} -> {self.dates[-1]})",
            format_summary(f"melange (w={self.blend:.2f})", self.metrics),
        ]
        if self.match_only is not None:
            lines.append(format_summary("modele match seul", summarize(self.y, self.match_only)))
        if self.maps_only is not None:
            lines.append(format_summary("lecture par maps", summarize(self.y, self.maps_only)))
        if self.blended_raw is not None:
            lines.append(
                format_summary("melange sans garde-fou", summarize(self.y, self.blended_raw))
            )
        lines.append(format_summary("Elo seul (reference)", self.elo_metrics))
        if self.comparison is not None:
            c = self.comparison
            lines.append(f"Gain log-loss face a Elo : {c['gain']:+.4f} (positif = mieux).")
            if c["low"] is None:
                lines.append("IC indisponible : moins de deux blocs calendaires observes.")
            else:
                lines.append(f"IC 95% apparie [{c['low']:+.4f}, {c['high']:+.4f}], "
                             f"{c['blocks']} blocs de {c['block_days']} jours, "
                             f"{c['samples']} tirages.")
                if c["low"] <= 0 <= c["high"]:
                    lines.append("L'intervalle inclut zero : l'avantage sur Elo reste incertain.")
                lines.append("Intervalle conditionnel a cette periode ; ne mesure pas un changement de meta.")
        if self.best_blend is not None:
            lines.append(
                f"\nPoids optimal mesure sur cette periode : w={self.best_blend:.2f} "
                f"(w=1 : modele match seul, w=0 : lecture par maps). "
                f"Defaut du code : {self.blend:.2f}."
            )
        lines += ["", "Fiabilite (proba annoncee vs realisee) :"]
        for row in reliability_table(self.y, self.predictions):
            gap = row["observed"] - row["predicted"]
            lines.append(
                f"  {row['bin']:<10} n={row['n']:<5} annonce {row['predicted']:.1%} "
                f"observe {row['observed']:.1%}  ecart {gap:+.1%}"
            )
        return "\n".join(lines)


def _elo_reference(ds: Dataset) -> np.ndarray:
    """Baseline : Elo brut converti en probabilite, sans modele appris."""
    i = ds.feature_names.index("elo_diff")
    return np.array([expected_score(x[i] * 100.0, 0.0) for x in ds.X])


def _pool_to_series(
    map_model: MatchPredictor, pool_x: np.ndarray, best_of: int, momentum: float = 0.0
) -> float:
    """Probabilites des 7 maps -> probabilite de serie, via le veto projete."""
    from .predict import probability_from_pool  # import local : evite un cycle

    ps = map_model.predict_proba(np.asarray(pool_x, dtype=float))
    return probability_from_pool(list(zip(ACTIVE_DUTY, ps.tolist())), best_of, momentum)


def walk_forward(
    ds: Dataset,
    map_ds: Dataset | None = None,
    initial_frac: float = 0.4,
    refit_every: int = 150,
    min_train: int = 200,
    blend: float = 0.5,
    momentum: float = 0.0,
    evaluation_start: dt.date | None = None,
    bootstrap_samples: int = 0,
    block_days: int = 7,
    **model_kwargs,
) -> BacktestResult:
    """Entraine sur [0, i), predit [i, i+refit_every), avance, recommence.

    Si `map_ds` est fourni, evalue aussi la lecture par maps et leur melange,
    c'est-a-dire exactement ce que sert `Analyzer.predict`.
    """
    finite_number("blend", blend, maximum=1.0)
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples doit etre un entier positif ou nul")
    positive_integer("block_days", block_days)
    n = len(ds)
    splits = forward_splits(ds.dates, initial_frac, min_train, refit_every, evaluation_start)
    start = splits[0][0]

    from .predict import combine_probability  # import local : evite un cycle

    use_maps = map_ds is not None and ds.pool_X is not None
    p_match = np.zeros(n - start)
    p_maps = np.zeros(n - start) if use_maps else None
    folds = 0
    for i, j in splits:
        model = MatchPredictor(**model_kwargs)
        model.fit(ds.X[:i], ds.y[:i], ds.dates[:i], ds.feature_names)
        p_match[i - start : j - start] = model.predict_proba(ds.X[i:j])

        if use_maps:
            # Le modele map ne voit que les maps jouees avant la date du 1er match
            # predit : meme frontiere temporelle que le modele match.
            cutoff = ds.dates[i]
            k = sum(1 for d in map_ds.dates if d < cutoff)
            # Logistique seule au niveau map. Le GBDT a ete essaye ici (son
            # dataset est pourtant le plus gros, 4758 lignes contre 2416) :
            # +0.0005 de log-loss pour +0.0011 d'erreur type appariee, soit
            # rien, contre le double de temps de calcul.
            map_model = map_predictor(**model_kwargs)
            map_model.fit(map_ds.X[:k], map_ds.y[:k], map_ds.dates[:k], map_ds.feature_names)
            for r in range(i, j):
                p_maps[r - start] = _pool_to_series(
                    map_model, ds.pool_X[r], ds.meta[r].get("best_of", 3), momentum
                )
        folds += 1

    y = ds.y[start:]
    elo_only = _elo_reference(ds)[start:]
    if use_maps:
        # Meme fonction de combinaison que `Analyzer.predict` : retrecissement
        # selon la fiabilite des donnees, puis plafond de confiance. Le chiffre
        # mesure ici est donc exactement celui qui sera servi.
        blended_raw = blend * p_match + (1.0 - blend) * p_maps
        preds = np.array([
            combine_probability(
                p_match[r], p_maps[r],
                ds.meta[start + r].get("maps_played_min", 999),
                ds.meta[start + r].get("days_idle_max", 0),
                weight=blend,
                certainty=ds.meta[start + r].get("certainty"),
            )[0]
            for r in range(len(y))
        ])
        grid = np.arange(0.0, 1.001, 0.05)

        def _ll(w: float) -> float:
            pw = np.array([
                combine_probability(
                    p_match[r], p_maps[r],
                    ds.meta[start + r].get("maps_played_min", 999),
                    ds.meta[start + r].get("days_idle_max", 0),
                    weight=w,
                    certainty=ds.meta[start + r].get("certainty"),
                )[0]
                for r in range(len(y))
            ])
            return log_loss(y, pw)

        best = float(min(grid, key=_ll))
    else:
        preds, best, blended_raw = p_match, None, None

    return BacktestResult(
        predictions=preds,
        y=y,
        dates=ds.dates[start:],
        meta=ds.meta[start:],
        elo_only=elo_only,
        folds=folds,
        metrics=summarize(y, preds),
        elo_metrics=summarize(y, elo_only),
        match_only=p_match if use_maps else None,
        maps_only=p_maps,
        blended_raw=blended_raw,
        blend=blend,
        best_blend=best,
        comparison=(paired_block_bootstrap(y, preds, elo_only, ds.dates[start:],
                                          samples=bootstrap_samples, block_days=block_days)
                    if bootstrap_samples else None),
    )


def bet_backtest(
    result: BacktestResult,
    odds_lookup: dict[str, tuple[float, float]],
    min_edge: float = 0.04,
) -> dict:
    """Rejoue le backtest contre des cotes historiques, si on en dispose.

    `odds_lookup` : match_id -> (cote team_a, cote team_b).
    Sans cotes reelles, cette fonction ne renvoie rien d'exploitable : ne pas
    substituer des cotes moyennes ou reconstituees, le biais serait total.
    """
    from .odds import find_value, simulate_flat

    plays = []
    for p, y, meta in zip(result.predictions, result.y, result.meta):
        odds = odds_lookup.get(meta["match_id"])
        if not odds:
            continue
        bet = find_value(float(p), odds[0], odds[1], meta["team_a"], meta["team_b"], min_edge)
        if bet is None:
            continue
        won = int((bet.side == meta["team_a"] and y == 1) or (bet.side == meta["team_b"] and y == 0))
        plays.append((bet.model_p, bet.odds, won))
    out = simulate_flat(plays)
    out["matches_with_odds"] = sum(1 for m in result.meta if m["match_id"] in odds_lookup)
    return out


# ---------------------------------------------------------------------------
# Reglage des constantes.
#
# Les valeurs par defaut sont ajustees sur des mondes simules realistes. Elles
# ne sont pas universelles : le bon K depend de la vitesse a laquelle le niveau
# des equipes bouge, qui n'est pas la meme selon la periode, le circuit, ou le
# niveau de jeu couvert par vos donnees. Cette fonction refait le reglage sur
# vos donnees, avec un decoupage temporel qui empeche de se mentir a soi-meme.
# ---------------------------------------------------------------------------

TUNING_GRID: dict[str, tuple] = {
    "k_base": (8.0, 11.0, 14.0, 18.0, 24.0),
    "k_map": (12, 20, 26, 34),
    "rd_growth_days": (250, 700, 2000),
    "fast_multiplier": (0.0, 2.0, 3.0, 5.0),
    "idle_halflife_days": (400, 900, 2000),
    "map_prior_weight": (0.3, 0.65, 1.2),
}


def tune(
    matches,
    grid: dict | None = None,
    holdout_frac: float = 0.25,
    warmup_maps: int = 12,
    verbose: bool = True,
    **wf_kwargs,
):
    """Recherche par coordonnees sur les constantes du systeme de notation.

    Protocole : le reglage ne voit que le debut de l'historique, la validation
    porte sur la fin, jamais touchee. Un reglage valide sur les memes matchs que
    ceux qui l'ont produit ne prouve rien - c'est la facon la plus courante de
    fabriquer une amelioration qui n'existe pas.

    Retourne (config retenue, rapport).
    """
    from .features import build_datasets
    from .ratings import EloConfig

    if not math.isfinite(holdout_frac) or not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac doit etre strictement entre 0 et 1")
    grid = TUNING_GRID if grid is None else grid
    matches = sorted(matches, key=lambda m: (m.date, m.match_id))
    cut = int(len(matches) * (1 - holdout_frac))
    if not 0 < cut < len(matches):
        raise ValueError("historique trop court pour reserver une periode de validation")
    validation_date = matches[cut].date
    cut = bisect_left([m.date for m in matches], validation_date)
    fit_matches = matches[:cut]

    def evaluate(cfg: EloConfig, subset, start_date=None) -> BacktestResult:
        mds, pds, _ = build_datasets(subset, cfg, warmup_maps)
        kwargs = {**wf_kwargs, "use_gbdt": False}
        if start_date is not None:
            kwargs["evaluation_start"] = start_date
        return walk_forward(mds, map_ds=pds, **kwargs)

    def score(cfg: EloConfig, subset) -> float:
        return evaluate(cfg, subset).metrics["log_loss"]

    current = EloConfig()
    baseline_fit = score(current, fit_matches)
    history = [("depart", None, baseline_fit)]
    best = baseline_fit

    for name, values in grid.items():
        trials = []
        for v in values:
            cfg = EloConfig.from_dict({**current.to_dict(), name: v})
            try:
                ll = score(cfg, fit_matches)
            except (ValueError, ZeroDivisionError):
                continue
            trials.append((ll, v))
            history.append((name, v, ll))
        if not trials:
            continue
        ll, v = min(trials)
        if ll < best - 1e-6:
            best = ll
            current = EloConfig.from_dict({**current.to_dict(), name: v})
            if verbose:
                print(f"  {name:<22} -> {v}   (log-loss {ll:.4f})")
        elif verbose:
            print(f"  {name:<22} -> inchange")

    # Tout le passe construit les ratings ; seules les cibles reservees comptent.
    baseline_validation = evaluate(EloConfig(), matches, validation_date)
    tuned_validation = evaluate(current, matches, validation_date)
    report = {
        "logloss_reglage_avant": baseline_fit,
        "logloss_reglage_apres": best,
        "logloss_validation_avant": baseline_validation.metrics["log_loss"],
        "logloss_validation_apres": tuned_validation.metrics["log_loss"],
        "essais": history,
        "n_matchs_reglage": cut,
        "n_matchs_total": len(matches),
        "n_matchs_validation": len(tuned_validation.y),
        "validation_debut": tuned_validation.dates[0].isoformat(),
        "validation_fin": tuned_validation.dates[-1].isoformat(),
        "reglage_match_ids": [m.match_id for m in fit_matches],
        "validation_match_ids": [m["match_id"] for m in tuned_validation.meta],
    }
    report["gain_reel"] = (
        report["logloss_validation_avant"] - report["logloss_validation_apres"]
    )
    return current, report
