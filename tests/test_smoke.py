"""Tests de bout en bout. Lancer : python -m tests.test_smoke (ou pytest)."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np

from cs2edge.features import FeatureExtractor, build_datasets
from cs2edge.metrics import log_loss, summarize
from cs2edge.model import MatchPredictor, series_win_probability
from cs2edge.odds import devig_power, expected_value, kelly_fraction
from cs2edge.predict import MAX_CONFIDENCE, Analyzer
from cs2edge.ratings import EloEngine, expected_score
from cs2edge.schema import MapResult, group_matches, normalize_map, normalize_team, read_csv, write_csv
from cs2edge.sources import simulate_history


def test_normalisation():
    assert normalize_team("Natus Vincere") == "navi"
    assert normalize_team("  G2   Esports ") == "g2"
    assert normalize_team("FaZe Clan") == "faze"
    assert normalize_map("de_Dust2") == "dust2"


def test_group_matches_realigne_les_equipes():
    """Une map ou l'ordre des equipes est inverse doit etre remise dans le sens du match."""
    maps = [
        MapResult("m1", dt.date(2025, 1, 1), "a", "b", "nuke", 13, 7, map_index=0),
        MapResult("m1", dt.date(2025, 1, 1), "b", "a", "mirage", 13, 4, map_index=1),
    ]
    match = group_matches(maps)[0]
    assert match.team_a == "a" and match.team_b == "b"
    assert match.maps[1].rounds_a == 4 and match.maps[1].rounds_b == 13
    assert (match.maps_a, match.maps_b) == (1, 1)


def test_elo_symetrie_entre_equipes_egalement_connues():
    e = EloEngine()
    d = dt.date(2025, 1, 1)
    e.observe(MapResult("m", d, "a", "b", "nuke", 13, 9))
    # Deux equipes aussi mal connues l'une que l'autre bougent d'autant.
    assert abs((e.rating("a") + e.rating("b")) - 3000.0) < 1e-6
    assert e.rating("a") > 1500.0 > e.rating("b")
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-12


def test_l_equipe_la_moins_connue_bouge_le_plus():
    """Choix assume : les mises a jour ne sont PAS a somme nulle.

    Quand une equipe etablie affronte une inconnue, le resultat en dit beaucoup
    sur l'inconnue et presque rien sur l'etablie. Un Elo classique leur impose
    pourtant le meme deplacement, ce qui est faux.
    """
    e = EloEngine()
    d = dt.date(2025, 1, 1)
    for i in range(60):
        e.observe(MapResult(f"w{i}", d + dt.timedelta(days=i), "etablie", "sparring", "nuke", 13, 11))
    rd_etablie = e.deviation("etablie", d + dt.timedelta(days=61))
    assert rd_etablie < e.deviation("inconnue", d + dt.timedelta(days=61))

    avant_e, avant_i = e.rating("etablie"), e.rating("inconnue")
    e.observe(MapResult("x", d + dt.timedelta(days=61), "etablie", "inconnue", "nuke", 13, 4))
    bouge_e = abs(e.rating("etablie") - avant_e)
    bouge_i = abs(e.rating("inconnue") - avant_i)
    assert bouge_i > 2 * bouge_e, (bouge_e, bouge_i)


def test_certitude_et_trajectoire():
    e = EloEngine()
    d = dt.date(2025, 1, 1)
    for i in range(80):
        e.observe(MapResult(f"m{i}", d + dt.timedelta(days=i), "solide", "autre", "nuke", 13, 10))
    # Plus on a de matchs, plus l'ecart de notes est digne de confiance.
    assert e.certainty("solide", "autre", d + dt.timedelta(days=81)) > e.certainty(
        "solide", "jamais_vue", d + dt.timedelta(days=81)
    )
    # Une equipe qui gagne en serie a une note rapide au-dessus de sa note lente.
    assert e.trajectory("solide", d + dt.timedelta(days=81)) > 0
    # Et cette trajectoire s'evente si l'equipe ne joue plus.
    loin = e.trajectory("solide", d + dt.timedelta(days=400))
    assert 0 <= loin < e.trajectory("solide", d + dt.timedelta(days=81))


def test_elo_amorti_par_inactivite():
    e = EloEngine()
    d = dt.date(2025, 1, 1)
    for i in range(20):
        e.observe(MapResult(f"m{i}", d, "a", "b", "nuke", 13, 5))
    frais = e.rating("a", d)
    rouille = e.rating("a", d + dt.timedelta(days=800))
    assert 1500.0 < rouille < frais


def test_aucune_fuite_temporelle():
    """Les features d'un match ne doivent rien savoir de son propre resultat."""
    maps = simulate_history(n_teams=12, years=1.0, seed=3)
    matches = group_matches(maps)
    fx = FeatureExtractor()
    for m in matches[:60]:
        before = fx.match_features(m.team_a, m.team_b, m.date, m.best_of, m.lan)
        again = fx.match_features(m.team_a, m.team_b, m.date, m.best_of, m.lan)
        assert np.allclose(before, again)  # extraction pure
        fx.observe_match(m)
        after = fx.match_features(m.team_a, m.team_b, m.date, m.best_of, m.lan)
        assert not np.allclose(before, after)  # l'etat a bien bouge apres coup


def test_features_antisymetriques():
    maps = simulate_history(n_teams=12, years=1.0, seed=4)
    matches = group_matches(maps)
    fx = FeatureExtractor()
    for m in matches[:80]:
        fx.observe_match(m)
    m = matches[90]
    ab = fx.match_features(m.team_a, m.team_b, m.date, 3, True)
    ba = fx.match_features(m.team_b, m.team_a, m.date, 3, True)
    assert np.allclose(ab, -ba, atol=1e-9)


def test_modele_antisymetrique():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 6))
    y = (rng.random(600) < 1 / (1 + np.exp(-X @ np.array([1.0, .4, -.3, 0, .2, 0])))).astype(float)
    m = MatchPredictor(use_gbdt=False).fit(X, y, feature_names=tuple("abcdef"))
    p = m.predict_proba(X[:20])
    q = m.predict_proba(-X[:20])
    assert np.allclose(p + q, 1.0, atol=1e-12)


def test_probabilite_de_serie():
    assert abs(series_win_probability(0.5, 3) - 0.5) < 1e-12
    assert series_win_probability(0.6, 1) == 0.6
    # Un format long amplifie l'avantage du meilleur.
    assert series_win_probability(0.6, 5) > series_win_probability(0.6, 3) > 0.6
    assert series_win_probability(0.4, 5) < series_win_probability(0.4, 3) < 0.4


def test_devig_et_mise():
    pa, pb = devig_power(1.65, 2.30)
    assert abs(pa + pb - 1.0) < 1e-9 and pa > pb
    assert expected_value(0.5, 2.0) == 0.0
    assert kelly_fraction(0.5, 2.0) == 0.0          # pas d'edge, pas de mise
    assert kelly_fraction(0.9, 1.2) <= 0.05         # plafond respecte
    assert kelly_fraction(0.3, 2.0) == 0.0          # edge negatif


def test_metriques():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert abs(log_loss(y, np.full(4, 0.5)) - np.log(2)) < 1e-9
    s = summarize(y, np.array([0.9, 0.1, 0.8, 0.2]))
    assert s["accuracy"] == 1.0 and s["skill_score"] > 0.5


def test_chaine_complete_et_sauvegarde():
    maps = simulate_history(n_teams=16, years=1.5, seed=11)
    an = Analyzer().load_maps(maps).train(use_gbdt=False)
    teams = an.known_teams()
    pred = an.predict(teams[0], teams[1], best_of=3, lan=True)

    assert 0.0 < pred.p_a < 1.0
    assert max(pred.p_a, 1 - pred.p_a) <= MAX_CONFIDENCE + 1e-9
    assert len(pred.map_edges) == 7
    assert pred.report()

    # Inverser les equipes doit exactement inverser la probabilite.
    rev = an.predict(teams[1], teams[0], best_of=3, lan=True)
    assert abs(pred.p_a + rev.p_a - 1.0) < 1e-9

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        an.match_model.save(path)
        loaded = MatchPredictor.load(path)
        assert np.allclose(
            loaded.predict_proba(an.match_ds.X[:30]),
            an.match_model.predict_proba(an.match_ds.X[:30]),
        )
        csv = Path(tmp) / "d.csv"
        write_csv(str(csv), maps)
        assert len(read_csv(str(csv))) == len(maps)


def test_backtest_bat_le_hasard():
    from cs2edge.backtest import walk_forward

    maps = simulate_history(n_teams=20, years=2.5, seed=5)
    ds, _map_ds, _fx = build_datasets(group_matches(maps))
    res = walk_forward(ds, use_gbdt=False)
    assert res.metrics["log_loss"] < res.metrics["baseline_log_loss"]
    assert res.metrics["skill_score"] > 0.10
    assert res.report()


def test_backtest_melange_expose_les_trois_estimateurs():
    from cs2edge.backtest import walk_forward

    maps = simulate_history(n_teams=20, years=2.5, seed=6)
    mds, pds, _ = build_datasets(group_matches(maps))
    res = walk_forward(mds, map_ds=pds, use_gbdt=False)
    assert res.match_only is not None and res.maps_only is not None
    assert 0.0 <= res.best_blend <= 1.0

    from cs2edge.metrics import log_loss as ll

    # Le melange doit battre chacun de ses deux composants.
    assert ll(res.y, res.blended_raw) < min(ll(res.y, res.match_only), ll(res.y, res.maps_only))
    # Le garde-fou de securite degrade forcement un peu le log-loss : il echange
    # de la performance contre de la prudence sur les cas mal documentes. On
    # verifie seulement que le prix reste modeste.
    assert ll(res.y, res.predictions) - ll(res.y, res.blended_raw) < 0.02


def test_equipe_inconnue_message_clair():
    an = Analyzer().load_maps(simulate_history(n_teams=12, years=1.0, seed=2)).train(use_gbdt=False)
    try:
        an.predict("equipe qui n existe pas", an.known_teams()[0])
    except ValueError as exc:
        assert "inconnue" in str(exc)
    else:
        raise AssertionError("une equipe inconnue doit lever une erreur explicite")



# ---------------------------------------------------------------- pronostiqueur


def test_veto_produit_le_bon_nombre_de_maps():
    from cs2edge.markets import project_veto

    pool = [(f"m{i}", p) for i, p in enumerate([0.8, 0.72, 0.6, 0.5, 0.42, 0.3, 0.2])]
    for bo in (1, 3, 5):
        plan = project_veto(pool, bo)
        assert len(plan.order) == bo
        assert len({m for m, _ in plan.order}) == bo  # pas de doublon
    # En bo3 : A joue son meilleur terrain restant, B le pire pour A.
    plan = project_veto(pool, 3)
    assert plan.order[0][1] > plan.order[1][1]


def test_structure_du_veto_change_la_loi_du_score():
    """Sur un pool polarise, le veto rallonge les series par rapport a l'i.i.d."""
    from cs2edge.markets import project_veto, series_distribution
    from cs2edge.model import series_win_probability

    pool = [("a", .90), ("b", .85), ("c", .60), ("d", .50), ("e", .40), ("f", .15), ("g", .10)]
    dist = series_distribution(project_veto(pool, 3).order, 3)
    moyenne = sum(p for _, p in pool[1:-1]) / 5
    assert abs(dist.p_a - series_win_probability(moyenne, 3)) < 0.02   # meme vainqueur
    assert dist.p_score(2, 0) < moyenne ** 2 - 0.05                    # moins de 2-0
    assert dist.p_maps_played(3) > 2 * moyenne * (1 - moyenne) + 0.10  # plus de 3 maps


def test_loi_du_score_est_une_loi():
    from cs2edge.markets import project_veto, series_distribution

    pool = [(f"m{i}", p) for i, p in enumerate([0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25])]
    for bo in (1, 3, 5):
        d = series_distribution(project_veto(pool, bo).order, bo)
        assert abs(sum(d.scores.values()) - 1.0) < 1e-9
        assert all(p >= 0 for p in d.scores.values())
        assert abs(d.p_a + (1 - d.p_a) - 1.0) < 1e-12


def test_marges_imposees_exactement():
    from cs2edge.markets import enforce_margins, project_veto, series_distribution

    pool = [(f"m{i}", p) for i, p in enumerate([0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25])]
    d = series_distribution(project_veto(pool, 3).order, 3)
    for p_a, p_full in ((0.65, 0.30), (0.40, 0.80), (0.90, 0.15)):
        e = enforce_margins(d, p_a, p_full)
        assert abs(e.p_a - p_a) < 1e-6
        assert abs(e.p_maps_played(3) - p_full) < 1e-6
        assert abs(sum(e.scores.values()) - 1.0) < 1e-9
    # bo1 : la longueur n'est pas une dimension libre, seul le vainqueur compte.
    d1 = series_distribution(project_veto(pool, 1).order, 1)
    assert abs(enforce_margins(d1, 0.7, 1.0).p_a - 0.7) < 1e-6


def test_momentum_nul_sur_donnees_independantes():
    """L'estimateur ne doit pas inventer d'effet la ou il n'y en a pas."""
    from cs2edge.markets import estimate_momentum
    from cs2edge.features import FeatureExtractor

    maps = simulate_history(n_teams=16, years=2.0, seed=21)
    an = Analyzer().load_maps(maps).train(use_gbdt=False)
    res = estimate_momentum(an.matches, an.map_model, FeatureExtractor())
    # Le simulateur tire chaque map independamment : l'effet doit etre
    # indiscernable du bruit.
    assert res["significatif"] is False


def test_pronostiqueur_se_tait_quand_c_est_serre():
    from cs2edge.pronostic import SEUIL_ABSTENTION, select_analysis_picks

    assert select_analysis_picks(dt.date(2026, 1, 1), "a", "b", 3, "", 0.52, None) == []
    picks = select_analysis_picks(dt.date(2026, 1, 1), "a", "b", 3, "", 0.80, None)
    assert len(picks) == 1 and picks[0].selection == "a" and picks[0].tier == "solide"
    # Symetrie : le favori change avec la probabilite, pas avec l'ordre de saisie.
    inv = select_analysis_picks(dt.date(2026, 1, 1), "a", "b", 3, "", 0.20, None)
    assert inv[0].selection == "b" and abs(inv[0].probability - 0.80) < 1e-12
    assert SEUIL_ABSTENTION >= 0.55


def test_momentum_branche_ne_s_active_pas_sur_donnees_independantes():
    """Le momentum ne doit s'appliquer que s'il est mesure significatif."""
    maps = simulate_history(n_teams=16, years=2.0, seed=23)
    an = Analyzer().load_maps(maps).train(use_gbdt=False)
    diag = an.apply_estimated_momentum()
    assert diag.get("significatif") is False
    assert an.momentum == 0.0   # rien invente la ou il n'y a pas de signal


def test_handicap_emis_seulement_sur_pool_desequilibre_et_regle_juste():
    from cs2edge.markets import project_veto, series_distribution
    from cs2edge.pronostic import _settle, select_analysis_picks

    # Pool tres desequilibre : le favori (A) domine partout -> 2-0 tres probable.
    pool = [(f"m{i}", p) for i, p in enumerate([0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.9])]
    dist = series_distribution(project_veto(pool, 3).order, 3)
    picks = select_analysis_picks(dt.date(2026, 1, 1), "a", "b", 3, "", dist.p_a, dist)
    hcap = [p for p in picks if p.market.startswith("handicap -")]
    assert hcap and hcap[0].market == "handicap -1.5"      # bo3 : sweep = -1.5
    assert hcap[0].selection == "a" and hcap[0].tier == "solide"
    # Regle : couvert seulement si le favori gagne 2-0.
    assert _settle(hcap[0], {"score": "2-0"}, y_a=1.0) is True
    assert _settle(hcap[0], {"score": "2-1"}, y_a=1.0) is False   # a lache une map
    assert _settle(hcap[0], {"score": "1-2"}, y_a=0.0) is False   # a perdu

    # Pool moderement desequilibre : sweep ~0.66 -> handicap au palier interessant
    # (emis des le seuil d'abstention, pas seulement au palier solide).
    moyen = [(f"m{i}", p) for i, p in enumerate([0.9, 0.88, 0.85, 0.8, 0.78, 0.75, 0.7])]
    dm = series_distribution(project_veto(moyen, 3).order, 3)
    hm = [p for p in select_analysis_picks(dt.date(2026, 1, 1), "a", "b", 3, "", dm.p_a, dm)
          if p.market.startswith("handicap -")]
    assert hm and hm[0].tier == "interessant" and 0.60 <= hm[0].probability < 0.70

    # bo5 : ne pas lacher de map = 3-0 = handicap -2.5 (pas -1.5).
    b5 = select_analysis_picks(dt.date(2026, 1, 1), "a", "b", 5, "", dist.p_a, dist)
    h5 = [p for p in b5 if p.market.startswith("handicap -")]
    if h5:  # emis seulement si le 3-0 depasse le palier solide
        assert h5[0].market == "handicap -2.5"
        assert _settle(h5[0], {"score": "3-0"}, y_a=1.0) is True
        assert _settle(h5[0], {"score": "3-1"}, y_a=1.0) is False  # une map lachee

    # Pool equilibre : pas de sweep assez probable -> aucun pick handicap.
    even = [(f"m{i}", p) for i, p in enumerate([0.58, 0.55, 0.52, 0.5, 0.48, 0.45, 0.42])]
    d2 = series_distribution(project_veto(even, 3).order, 3)
    assert not any(
        p.market.startswith("handicap -")
        for p in select_analysis_picks(dt.date(2026, 1, 1), "a", "b", 3, "", d2.p_a, d2)
    )


def test_carte_de_pronostics_complete():
    from cs2edge.pronostic import Fixture, Tipster, format_card

    maps = simulate_history(n_teams=18, years=2.0, seed=13)
    an = Analyzer().load_maps(maps).train(use_gbdt=False)
    teams = [t for t in an.known_teams() if an.fx.elo.maps_played(t) > 60][:4]
    on = max(m.date for m in an.matches)
    fixtures = [
        Fixture(on, teams[0], teams[1], 3, True, "T"),
        Fixture(on, teams[2], teams[3], 3, False, "T", odds_a=1.80, odds_b=2.05),
    ]
    card = Tipster(an).card(fixtures)
    assert len(card) == 2
    assert format_card(card, verbose=True)
    for a in card:
        if a.prediction is None:
            continue
        # Le pronostic vainqueur et la loi du score ne doivent jamais se contredire.
        assert abs(a.distribution.p_a - a.prediction.p_a) < 1e-6
        for pk in a.picks:
            assert 0.5 <= pk.probability <= 1.0 or pk.market != "vainqueur"


def test_handicap_en_mode_valeur():
    """Une cote de sweep declenche le marche handicap, meme sans cote vainqueur."""
    from types import SimpleNamespace

    from cs2edge.markets import project_veto, series_distribution
    from cs2edge.odds import expected_value
    from cs2edge.pronostic import Fixture, Tipster

    pool = [(f"m{i}", p) for i, p in enumerate([0.9, 0.88, 0.85, 0.8, 0.78, 0.75, 0.7])]
    dist = series_distribution(project_veto(pool, 3).order, 3)
    pred = SimpleNamespace(team_a="a", team_b="b", p_a=dist.p_a)
    tip = Tipster.__new__(Tipster)
    tip.edge_min = 0.04

    # Cote genereuse sur le sweep de a -> valeur ; cote serree -> rien.
    p_sweep = dist.p_clean_sweep(True)
    genereuse = 1.0 / p_sweep * 1.30
    fx = Fixture(dt.date(2026, 1, 1), "a", "b", 3, False, "T", odds_a_sweep=genereuse)
    picks = tip._value_picks(fx, pred, dist, [])
    assert picks and picks[0].market == "handicap -1.5" and picks[0].selection == "a"
    assert expected_value(picks[0].probability, picks[0].odds) > 0

    serree = 1.0 / p_sweep * 0.98   # sous le seuil de rentabilite
    fx2 = Fixture(dt.date(2026, 1, 1), "a", "b", 3, False, "T", odds_a_sweep=serree)
    assert tip._value_picks(fx2, pred, dist, []) == []


def test_journal_archive_pointe_et_ne_double_pas():
    import tempfile

    from cs2edge import journal as jm
    from cs2edge.pronostic import Fixture, Tipster

    maps = simulate_history(n_teams=18, years=2.0, seed=31)
    an = Analyzer().load_maps(maps).train(use_gbdt=False)
    passes = an.matches[-40:]
    fixtures = [Fixture(m.date, m.team_a, m.team_b, m.best_of, m.lan, m.event) for m in passes]
    picks = [p for a in Tipster(an).card(fixtures) for p in a.picks if p.is_actionable]
    assert picks, "le pronostiqueur devrait se prononcer sur au moins un match"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "journal.csv"
        assert jm.record(picks, path) == len(picks)
        assert jm.record(picks, path) == 0          # idempotent
        assert jm.scorecard(path) is None           # rien de pointe encore

        stats = jm.settle(path, an.matches)
        assert stats["pointes"] == len(picks) and stats["introuvables"] == 0
        assert jm.settle(path, an.matches)["pointes"] == 0   # pointage idempotent

        rec = jm.scorecard(path)
        assert rec is not None and len(rec.settled) == len(picks)
        assert rec.report()
        rows = jm.load(path)
        assert all(r["resultat"] in ("gagne", "perdu") for r in rows)


def test_bilan_ne_crie_pas_au_loup_sur_petit_echantillon():
    from cs2edge.pronostic import Pick, SettledPick, TipsterRecord

    petits = [
        SettledPick(
            pick=Pick(dt.date(2026, 1, 1), "a", "b", 3, "", "vainqueur", "a", 0.80,
                      "solide", "PRONOSTIC"),
            won=(i < 5),
        )
        for i in range(10)
    ]
    rec = TipsterRecord(settled=petits, n_fixtures=10, n_skipped=0)
    stat = rec.by(lambda p: p.tier)["solide"]
    # 50% observe contre 80% annonce, mais n=10 : aucune conclusion possible.
    assert stat["significatif"] is False
    assert "n trop faible" in rec.report()


def test_pronostiqueur_tient_ses_paliers():
    """Chaque palier doit realiser ce qu'il annonce, aux fluctuations pres."""
    from cs2edge.pronostic import simulate_tipster

    maps = simulate_history(n_teams=22, years=3.0, seed=41)
    mds, pds, _ = build_datasets(group_matches(maps))
    rec = simulate_tipster(mds, pds, use_gbdt=False)
    assert rec.settled and rec.n_skipped > 0   # il doit s'abstenir parfois
    for name, stat in rec.by(lambda p: p.tier).items():
        if stat["n"] < 30:
            continue
        assert not stat["significatif"], (
            f"le palier '{name}' annonce {stat['annonce']:.1%} et realise "
            f"{stat['observe']:.1%} sur {stat['n']} pronostics"
        )


def test_lecture_des_fixtures():
    import tempfile

    from cs2edge.pronostic import read_fixtures

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "f.csv"
        path.write_text(
            "date,team_a,team_b,best_of,lan,event,odds_a,odds_b\n"
            "2026-09-07,Natus Vincere,G2 Esports,3,1,IEM,1.75,2.10\n"
            "2026-09-08,FaZe Clan,MOUZ,1,0,IEM,,\n",
            encoding="utf-8",
        )
        fx = read_fixtures(path)
        assert fx[0].team_a == "navi" and fx[0].team_b == "g2"
        assert fx[0].odds_a == 1.75 and fx[0].lan is True
        assert fx[1].odds_a is None and fx[1].best_of == 1

        bad = Path(tmp) / "bad.csv"
        bad.write_text("date,team_a,team_b\n2026-09-07,a,b\n", encoding="utf-8")
        assert read_fixtures(bad)[0].best_of == 3      # defaut applique
        worse = Path(tmp) / "worse.csv"
        worse.write_text("date,team_a\n2026-09-07,a\n", encoding="utf-8")
        try:
            read_fixtures(worse)
        except ValueError as exc:
            assert "team_b" in str(exc)
        else:
            raise AssertionError("une colonne manquante doit lever une erreur")


if __name__ == "__main__":
    import time
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"  ok    {fn.__name__:<40} {time.time() - t0:.2f}s")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ECHEC {fn.__name__:<40} {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passes")
    raise SystemExit(1 if failed else 0)
