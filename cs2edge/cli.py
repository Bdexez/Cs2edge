"""Interface en ligne de commande."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from . import __version__
from .backtest import tune, walk_forward
from .metrics import format_summary, summarize
from .model import MatchPredictor
from .odds import find_value, overround
from .predict import MATCH_MODEL_WEIGHT, Analyzer
from .ratings import EloConfig
from .pronostic import (
    Fixture, Tipster, fit_decider_temperature, format_card, read_fixtures,
    simulate_tipster,
)
from . import journal as journal_mod
from .schema import group_matches, parse_date, write_csv
from .sources import simulate_history

DEFAULT_DATA = "data/sample_matches.csv"


def _analyzer(args, train: bool = True) -> Analyzer:
    if getattr(args, "model", None):
        if args.data is not None or args.warmup is not None or args.no_gbdt or args.elo_config:
            raise ValueError("--model fournit deja les donnees et la configuration ; "
                             "retirer --data, --warmup, --no-gbdt et --elo-config")
        an = Analyzer.load(args.model)
        print(f"Analyseur charge depuis {args.model} : {len(an.maps)} maps, "
              f"historique jusqu'au {an.match_model.last_date}", file=sys.stderr)
        return an
    path = Path(args.data or DEFAULT_DATA)
    if not path.exists():
        sys.exit(
            f"fichier introuvable : {path}\n"
            f"  -> genere un jeu de test : cs2edge sample --out {path}\n"
            f"  -> ou fournis un CSV : match_id,date,team_a,team_b,map,rounds_a,rounds_b"
        )
    elo = EloConfig.load(args.elo_config) if getattr(args, "elo_config", None) else None
    if elo is not None:
        print(f"Constantes de notation chargees depuis {args.elo_config}", file=sys.stderr)
    an = Analyzer(elo_config=elo, warmup_maps=12 if args.warmup is None else args.warmup).load_csv(path)
    if not an.matches:
        raise ValueError(f"aucun resultat exploitable dans {path}")
    print(
        f"{len(an.maps)} maps, {len(an.matches)} matchs, {len(an.known_teams())} equipes "
        f"({an.matches[0].date} -> {an.matches[-1].date}) | "
        f"{len(an.match_ds)} matchs exploitables apres rodage",
        file=sys.stderr,
    )
    if train:
        an.train(use_gbdt=not args.no_gbdt)
    return an


def cmd_sample(args) -> int:
    rows = simulate_history(n_teams=args.teams, years=args.years, seed=args.seed,
                            start=args.start, regime=args.regime)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(str(out), rows)
    matches = group_matches(rows)
    print(f"{len(rows)} maps / {len(matches)} matchs simules -> {out}")
    print("Donnees synthetiques : utiles pour tester la chaine, pas pour parier.")
    return 0


def cmd_teams(args) -> int:
    an = _analyzer(args, train=False)
    for t in an.known_teams():
        print(f"{t:<20} {an.fx.elo.maps_played(t):>5} maps")
    return 0


def cmd_ratings(args) -> int:
    an = _analyzer(args, train=False)
    on = max(m.date for m in an.matches)
    board = an.fx.elo.leaderboard(on=on, min_maps=args.min_maps)
    print(f"Classement Elo au {on} (min {args.min_maps} maps)\n")
    print(f"{'#':<4}{'equipe':<20}{'elo':>8}{'maps':>7}   meilleures maps")
    for i, (team, rating, played) in enumerate(board[: args.top], 1):
        tr = an.fx.elo.teams[team]
        best = sorted(
            ((m, an.fx.elo.map_rating(team, m, on)) for m in tr.map_counts if tr.map_counts[m] >= 3),
            key=lambda kv: -kv[1],
        )[:3]
        tail = ", ".join(f"{m} {r:.0f}" for m, r in best)
        print(f"{i:<4}{team:<20}{rating:>8.0f}{played:>7}   {tail}")
    return 0


def cmd_train(args) -> int:
    an = _analyzer(args)
    ds = an.match_ds
    p_in = an.match_model.predict_proba(ds.X)
    print("\nAjustement en echantillon (optimiste par construction, sert de controle) :")
    print(format_summary("modele", summarize(ds.y, p_in)))
    print("\nPoids appris (features standardisees, vus de l'equipe A) :")
    for name, w in an.match_model.logistic.coefficients():
        bar = "#" * min(30, int(abs(w) * 10))
        print(f"  {name:<22} {w:+.4f}  {bar}")
    print(f"\nTemperature de calibration : {an.match_model.temperature:.3f}")
    print(f"GBDT actif : {an.match_model.gbdt is not None}")
    out = Path(args.bundle or args.out or "models/match.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.bundle:
        an.save(out)
        print(f"\nAnalyseur autonome enregistre -> {out}")
    else:
        an.match_model.save(out)
        print(f"\nModele match enregistre -> {out}")
    print("Pour une mesure honnete de la performance : cs2edge backtest")
    return 0


def cmd_backtest(args) -> int:
    an = _analyzer(args, train=False)
    res = walk_forward(
        an.match_ds,
        map_ds=None if args.no_maps else an.map_ds,
        initial_frac=args.initial_frac,
        refit_every=args.refit_every,
        blend=args.blend,
        bootstrap_samples=args.bootstrap_samples,
        block_days=args.block_days,
        use_gbdt=not args.no_gbdt,
    )
    print()
    print(res.report())
    print(
        "\nLecture : `skill` = reduction du log-loss face au pile ou face. "
        "En CS2 pro, 0.10-0.20 est deja bon ; au-dela, suspecter une fuite de donnees.\n"
        "Le backtest applique le meme melange et les memes garde-fous que `predict`."
    )
    if args.by_year:
        print("\nPar annee :")
        years = sorted({d.year for d in res.dates})
        for y in years:
            idx = [i for i, d in enumerate(res.dates) if d.year == y]
            if len(idx) < 20:
                continue
            s = summarize(res.y[idx], res.predictions[idx])
            print(format_summary(f"  {y}", s))
    return 0


def cmd_predict(args) -> int:
    an = _analyzer(args)
    on = parse_date(args.date) if args.date else None
    pred = an.predict(args.team_a, args.team_b, best_of=args.bo, lan=args.lan, on=on)
    print()
    print(pred.report())

    if args.odds_a and args.odds_b:
        print()
        print(f"  Marche : {args.odds_a:.2f} / {args.odds_b:.2f} "
              f"(marge {overround(args.odds_a, args.odds_b) - 1:.1%})")
        bet = find_value(
            pred.p_a, args.odds_a, args.odds_b, pred.team_a, pred.team_b, min_edge=args.min_edge
        )
        if bet is None:
            print(f"  Aucun ecart >= {args.min_edge:.0%} avec le marche : pas de valeur.")
        else:
            print(f"  {bet}")
            if abs(bet.edge) > 0.25:
                print(
                    "  /!\\ ecart enorme avec le marche. Dans 9 cas sur 10 cela veut dire\n"
                    "      que le marche sait quelque chose que les resultats passes ignorent\n"
                    "      (remplacant, roster fraichement change, forfait), pas que le modele\n"
                    "      a trouve une faille. Verifier la compo avant toute conclusion."
                )
            if pred.confidence == "faible":
                print("  /!\\ confiance faible sur cette prediction : l'ecart est probablement du bruit.")
    return 0


def _tipster(an: Analyzer, args) -> Tipster:
    # Momentum inter-maps : estime sur les donnees, applique seulement s'il est
    # significatif (voir Analyzer.apply_estimated_momentum). Un flag permet de le
    # forcer a zero ou a une valeur mesuree ailleurs.
    mode = getattr(args, "momentum", "auto")
    if mode == "auto":
        diag = an.apply_estimated_momentum()
        m = diag.get("momentum")
        if an.momentum:
            print(
                f"Momentum inter-maps : {an.momentum:+.3f} en logit (significatif, "
                f"n={diag.get('n_total', 0)}) - applique.",
                file=sys.stderr,
            )
        elif m is not None:
            print(
                f"Momentum inter-maps mesure {m:+.3f} mais non significatif "
                f"(+/-{2 * diag.get('erreur_type', 0):.3f}) : laisse a zero.",
                file=sys.stderr,
            )
    elif mode != "0":
        an.momentum = float(mode)
        print(f"Momentum inter-maps force a {an.momentum:+.3f} en logit.", file=sys.stderr)

    temp = (
        1.0 if getattr(args, "no_calibration", False)
        else fit_decider_temperature(an, momentum=an.momentum)
    )
    print(f"Calibration de la longueur de serie : T = {temp:.3f}", file=sys.stderr)
    kw = {}
    if getattr(args, "edge_min", None) is not None:
        kw["edge_min"] = args.edge_min
    if getattr(args, "abstention", None) is not None:
        kw["abstention"] = args.abstention
    return Tipster(an, momentum=an.momentum, decider_temperature=temp,
                  calibrate_decider=not getattr(args, "no_calibration", False), **kw)


def cmd_fixtures(args) -> int:
    """Genere une liste de matchs a venir a partir des equipes les mieux notees."""
    import csv
    import itertools

    an = _analyzer(args, train=False)
    on = args.date or max(dt.date.today(), max(m.date for m in an.matches))
    board = [t for t, _r, _n in an.fx.elo.leaderboard(on=on, min_maps=40)][: args.teams]
    pairs = list(itertools.combinations(board, 2))[: args.count]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["date", "team_a", "team_b", "best_of", "lan", "event",
                    "odds_a", "odds_b", "odds_over", "odds_under",
                    "odds_a_sweep", "odds_b_sweep"])
        for i, (a, b) in enumerate(pairs):
            w.writerow([
                (on + dt.timedelta(days=1 + i // 3)).isoformat(), a, b, 3, 1,
                "Exemple", "", "", "", "", "", "",
            ])
    print(f"{len(pairs)} matchs a venir -> {out}")
    print("Colonnes de cotes laissees vides : le pronostiqueur repondra "
          "'qui gagne'.\nLes remplir bascule en mode marche ('ou le marche se trompe').")
    return 0


def cmd_slate(args) -> int:
    an = _analyzer(args)
    fixtures = read_fixtures(args.fixtures)
    card = _tipster(an, args).card(fixtures)
    print()
    print(format_card(card, verbose=args.verbose))
    if args.record:
        added = journal_mod.record_card(card, args.record)
        print(f"\n{added} ligne(s) archivee(s) dans {args.record} (pronostics et abstentions).")
        print("Ils pourront etre pointes apres coup : cs2edge settle --journal "
              f"{args.record}")
    return 0


def cmd_settle(args) -> int:
    an = _analyzer(args, train=False)
    stats = journal_mod.settle(args.journal, an.matches)
    print(f"Pointage : {stats['pointes']} nouveau(x) resultat(s), "
          f"{stats['en_attente']} en attente, "
          f"{stats['introuvables']} introuvable(s), {stats['ambigus']} ambigu(s).")
    return cmd_journal(args)


def cmd_journal(args) -> int:
    record = journal_mod.scorecard(args.journal)
    if record is None:
        print("Aucun pronostic pointe pour le moment.")
        return 0
    print()
    print(record.report())
    money = journal_mod.financial_summary(args.journal)
    if money:
        print()
        print("Bilan financier (uniquement les lignes ou une cote a ete enregistree) :")
        print(f"  {money['paris']} paris, cote moyenne {money['cote_moyenne']:.2f}, "
              f"reussite {money['taux_reussite']:.1%}")
        print(f"  mise fixe : {money['profit_mise_fixe']:+.2f} unites, "
              f"ROI {money['roi_mise_fixe']:+.1%}")
        print(f"  Kelly     : {money['profit_kelly']:+.3f} pour "
              f"{money['mise_kelly_totale']:.2f} misees, ROI {money['roi_kelly']:+.1%}")
        print("  Sur moins de ~200 paris, ce ROI est domine par le hasard : "
              "ne pas en tirer de conclusion.")
    return 0


def cmd_tipster_backtest(args) -> int:
    an = _analyzer(args, train=False)
    rec = simulate_tipster(
        an.match_ds, an.map_ds,
        initial_frac=args.initial_frac,
        abstention=args.abstention,
        momentum=args.momentum,
        use_gbdt=not args.no_gbdt,
    )
    print()
    print(rec.report())
    print(
        "\nCe bilan repond a 'quand cet outil parle, qu'est-ce que ca vaut',\n"
        "pas a 'mes probabilites sont-elles justes' (voir `cs2edge backtest`).\n"
        "Un ecart n'est un defaut que s'il depasse sa marge (deux erreurs types) :\n"
        "ces lignes-la sont marquees '<- ecart reel, a corriger' et signalent un\n"
        "palier ou un marche qui ment a son lecteur. En dessous, l'ecart est du\n"
        "bruit d'echantillonnage et ne demontre rien."
    )
    return 0


def cmd_tune(args) -> int:
    an = _analyzer(args, train=False)
    print(
        f"\nReglage temporel avec {args.holdout:.0%} de fin d'historique reservee, "
        "frontieres alignees sur les journees.\n"
        "Les constantes par defaut sont ajustees sur des mondes simules realistes ;\n"
        "le bon reglage depend de la vitesse a laquelle bouge le niveau des equipes,\n"
        "qui n'est pas la meme selon la periode et le circuit couverts par vos donnees.\n"
    )
    cfg, report = tune(an.matches, holdout_frac=args.holdout, warmup_maps=an.warmup_maps)
    print(f"\n{report['n_matchs_reglage']} matchs de reglage ; "
          f"{report['n_matchs_validation']} matchs de validation "
          f"({report['validation_debut']} -> {report['validation_fin']}).")
    print(
        f"\nSur la periode de reglage  : {report['logloss_reglage_avant']:.4f} "
        f"-> {report['logloss_reglage_apres']:.4f}"
    )
    print(
        f"Sur la periode de validation : {report['logloss_validation_avant']:.4f} "
        f"-> {report['logloss_validation_apres']:.4f}   "
        f"(gain reel {report['gain_reel']:+.4f})"
    )
    if report["gain_reel"] <= 0.0005:
        print(
            "\nLe gain ne survit pas a la validation : gardez les constantes par defaut.\n"
            "C'est le resultat le plus frequent quand l'historique est court."
        )
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(out)
    print(f"\nConstantes enregistrees -> {out}")
    print(f"Pour les utiliser : cs2edge --elo-config {out} backtest")
    return 0


def _common_options(parser: argparse.ArgumentParser) -> None:
    # Une option absente ne doit pas ecraser celle lue avant la sous-commande.
    parser.add_argument("--data", default=argparse.SUPPRESS,
                        help=f"CSV de resultats (defaut {DEFAULT_DATA})")
    parser.add_argument("--warmup", type=int, default=argparse.SUPPRESS,
                        help="maps minimum par equipe (defaut 12)")
    parser.add_argument("--no-gbdt", action="store_true", default=argparse.SUPPRESS,
                        help="regression logistique seule")
    parser.add_argument("--elo-config", default=argparse.SUPPRESS,
                        help="JSON de constantes de notation (voir `tune`)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cs2edge",
        description="Analyse de matchs CS2 professionnels et estimation des probabilites de victoire.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemples :\n"
            "  cs2edge sample                       # genere un jeu de donnees de test\n"
            "  cs2edge backtest --by-year           # mesure la performance reelle\n"
            "  cs2edge ratings --top 15             # classement Elo courant\n"
            "  cs2edge predict vitality spirit --bo 3 --lan\n"
            "  cs2edge predict navi g2 --odds-a 1.75 --odds-b 2.10\n"
            "\n"
            "pronostiqueur :\n"
            "  cs2edge fixtures --out data/fixtures.csv    # matchs a venir (exemple)\n"
            "  cs2edge slate --fixtures data/fixtures.csv --record data/journal.csv\n"
            "  cs2edge settle --journal data/journal.csv   # pointe les resultats\n"
            "  cs2edge tipster-backtest                    # le bilan historique\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"cs2edge {__version__}")

    # Options communes, acceptees avant comme apres la sous-commande.
    common = argparse.ArgumentParser(add_help=False)
    _common_options(common)
    _common_options(p)
    p.set_defaults(data=None, warmup=None, no_gbdt=False, elo_config=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="genere un historique synthetique", parents=[common])
    s.add_argument("--out", default=DEFAULT_DATA)
    s.add_argument("--teams", type=int, default=32)
    s.add_argument("--years", type=float, default=3.0)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--start", type=parse_date, help="date initiale fixe pour reproduire le jeu")
    s.add_argument("--regime", choices=("facile", "realiste"), default="realiste",
                   help="regime du generateur (defaut realiste)")
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("teams", parents=[common], help="liste les equipes connues")
    s.set_defaults(func=cmd_teams)

    s = sub.add_parser("ratings", parents=[common], help="classement Elo et forces par map")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--min-maps", type=int, default=20)
    s.set_defaults(func=cmd_ratings)

    s = sub.add_parser("train", parents=[common], help="entraine et sauvegarde le modele")
    target = s.add_mutually_exclusive_group()
    target.add_argument("--out", help="modele match JSON seul (defaut models/match.json)")
    target.add_argument("--bundle", help="instantane autonome de l'analyseur, reutilisable avec --model")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("backtest", parents=[common], help="validation walk-forward (la seule qui compte)")
    s.add_argument("--initial-frac", type=float, default=0.4)
    s.add_argument("--refit-every", type=int, default=150)
    s.add_argument("--by-year", action="store_true")
    s.add_argument("--blend", type=float, default=MATCH_MODEL_WEIGHT,
                   help="poids du modele match dans le melange (1 = lui seul)")
    s.add_argument("--no-maps", action="store_true",
                   help="n'evaluer que le modele match")
    s.add_argument("--bootstrap-samples", type=int, default=1000,
                   help="tirages pour l'IC apparie face a Elo (defaut 1000, 0 desactive)")
    s.add_argument("--block-days", type=int, default=7,
                   help="taille des blocs calendaires de l'IC (defaut 7 jours)")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("predict", parents=[common], help="probabilites pour un match a venir")
    s.add_argument("--model", help="instantane d'analyseur produit par train --bundle")
    s.add_argument("team_a")
    s.add_argument("team_b")
    s.add_argument("--bo", type=int, default=3, choices=[1, 3, 5])
    s.add_argument("--lan", action="store_true")
    s.add_argument("--date", help="date du match (defaut : aujourd'hui ; passe reconstruit si necessaire)")
    s.add_argument("--odds-a", type=float, help="cote decimale sur team_a")
    s.add_argument("--odds-b", type=float, help="cote decimale sur team_b")
    s.add_argument("--min-edge", type=float, default=0.04)
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("fixtures", parents=[common],
                       help="genere une liste de matchs a venir (exemple)")
    s.add_argument("--out", default="data/fixtures.csv")
    s.add_argument("--teams", type=int, default=8)
    s.add_argument("--count", type=int, default=10)
    s.add_argument("--date", type=parse_date, help="date initiale du calendrier (defaut aujourd'hui)")
    s.set_defaults(func=cmd_fixtures)

    s = sub.add_parser("slate", parents=[common],
                       help="la carte de pronostics des matchs a venir")
    s.add_argument("--model", help="instantane d'analyseur produit par train --bundle")
    s.add_argument("--fixtures", required=True, help="CSV des matchs a venir")
    s.add_argument("--record", help="archive les pronostics dans ce journal")
    s.add_argument("--verbose", action="store_true", help="affiche la loi du score")
    s.add_argument("--no-calibration", action="store_true",
                   help="desactive la calibration de la longueur de serie")
    s.add_argument("--momentum", default="auto",
                   help="effet d'enchainement entre maps : 'auto' (mesure, applique "
                        "si significatif), '0' (aucun), ou un decalage en logit")
    s.add_argument("--abstention", type=float, default=None,
                   help="probabilite minimale pour se prononcer sans cotes (defaut 0.60)")
    s.add_argument("--edge-min", type=float, default=None,
                   help="ecart minimal avec le marche pour un pari avec cotes (defaut 0.04)")
    s.set_defaults(func=cmd_slate)

    s = sub.add_parser("settle", parents=[common],
                       help="pointe les pronostics archives contre les resultats")
    s.add_argument("--journal", default="data/journal.csv")
    s.set_defaults(func=cmd_settle)

    s = sub.add_parser("journal", parents=[common],
                       help="bilan des pronostics deja pointes")
    s.add_argument("--journal", default="data/journal.csv")
    s.set_defaults(func=cmd_journal)

    s = sub.add_parser("tune", parents=[common],
                       help="regle les constantes de notation sur vos donnees")
    s.add_argument("--out", default="models/elo.json")
    s.add_argument("--holdout", type=float, default=0.25,
                   help="part finale de l'historique reservee a la validation")
    s.set_defaults(func=cmd_tune)

    s = sub.add_parser("tipster-backtest", parents=[common],
                       help="bilan historique du pronostiqueur (paliers, marches)")
    s.add_argument("--initial-frac", type=float, default=0.4)
    s.add_argument("--abstention", type=float, default=0.60,
                   help="probabilite minimale pour se prononcer")
    s.add_argument("--momentum", type=float, default=0.0,
                   help="decalage momentum en logit a appliquer (defaut 0 ; mesurer "
                        "d'abord avec `slate --momentum auto`)")
    s.set_defaults(func=cmd_tipster_backtest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(f"erreur : {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
