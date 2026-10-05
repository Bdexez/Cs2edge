import datetime as dt
from dataclasses import replace
import unittest
from unittest.mock import patch

from cs2edge.backtest import walk_forward
from cs2edge.predict import Analyzer, combine_probability
from cs2edge.pronostic import Fixture, Tipster, fit_decider_temperature, simulate_tipster
from cs2edge.schema import group_matches
from cs2edge.sources import simulate_history


class PredictionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        maps = simulate_history(n_teams=8, years=.5, seed=11, start=dt.date(2023, 1, 1))
        # Une rencontre par jour pour isoler ici la parite du pipeline.
        cls.maps = [replace(mp, date=dt.date(2023, 1, 1) + dt.timedelta(days=i))
                    for i, match in enumerate(group_matches(maps)) for mp in match.maps]
        cls.an = Analyzer().load_maps(cls.maps).train(use_gbdt=False)

    def test_live_probability_uses_recorded_certainty(self):
        an = self.an
        a, b = an.known_teams()[:2]
        on = max(m.date for m in an.matches) + dt.timedelta(days=1)
        with patch.object(an.fx.elo, "certainty", return_value=.65):
            p = an.predict(a, b, on=on)
        q = p.data_quality
        expected, _ = combine_probability(p.p_a_match_model, p.p_a_from_maps,
                                          q["maps_played_min"], q["days_idle_max"], certainty=.65)
        self.assertAlmostEqual(p.p_a, expected, places=12)

    def test_first_fold_matches_actual_pre_match_analyzer_and_tipster(self):
        full = self.an
        ds, maps = full.match_ds, full.map_ds
        start = max(200, int(len(ds)*.6))
        self.assertGreaterEqual(len(ds)-start, 50)
        cutoff = ds.dates[start]
        past = Analyzer().load_maps([m for m in self.maps if m.date < cutoff]).train(use_gbdt=False)
        meta = ds.meta[start]
        fx = Fixture(cutoff, meta["team_a"], meta["team_b"], meta["best_of"],
                     meta["lan"], meta["event"], match_id=meta["match_id"])
        live = past.predict(fx.team_a, fx.team_b, fx.best_of, fx.lan, fx.date)
        result = walk_forward(ds, maps, initial_frac=.6, refit_every=len(ds), use_gbdt=False)
        self.assertAlmostEqual(result.predictions[0], live.p_a, places=12)
        replay = simulate_tipster(ds, maps, initial_frac=.6, refit_every=len(ds),
                                 abstention=.5, use_gbdt=False, decider_temperature=1.0)
        winner = next(s.pick for s in replay.settled
                      if s.pick.match_id == fx.match_id and s.pick.market == "vainqueur")
        self.assertAlmostEqual(winner.probability, live.p_favorite, places=12)
        card = Tipster(past, abstention=.5).analyse(fx)
        self.assertTrue(all(p.match_id == fx.match_id for p in card.picks))

    def test_gbdt_option_never_changes_map_model_family(self):
        # Le test du contrat ne depend pas de la presence de scikit-learn.
        with patch("cs2edge.model.MatchPredictor._fit_gbdt", return_value=None):
            an = Analyzer().load_maps(self.maps).train(use_gbdt=True)
        self.assertTrue(an.match_model.use_gbdt)
        self.assertFalse(an.map_model.use_gbdt)

    def test_historical_prediction_is_invariant_to_future_results(self):
        full = self.an
        cutoff = full.match_ds.dates[-60]
        a, b = full.known_teams()[:2]
        expected = Analyzer().load_maps([m for m in self.maps if m.date < cutoff]).train(use_gbdt=False)
        actual = full.predict(a, b, on=cutoff)
        reference = expected.predict(a, b, on=cutoff)
        self.assertAlmostEqual(actual.p_a, reference.p_a, places=12)
        self.assertLess(actual.data_quality["data_through"], cutoff.isoformat())
        changed = [replace(m, rounds_a=m.rounds_b, rounds_b=m.rounds_a)
                   if m.date >= cutoff else m for m in self.maps]
        modified = Analyzer().load_maps(changed).train(use_gbdt=False)
        self.assertAlmostEqual(modified.predict(a, b, on=cutoff).p_a, actual.p_a, places=12)

    def test_historical_card_rebuilds_derived_calibration_and_rationale(self):
        cutoff = self.an.match_ds.dates[-60]
        a, b = self.an.known_teams()[:2]
        fx = Fixture(cutoff, a, b)
        past = self.an.as_of(cutoff)
        reference = Tipster(past, decider_temperature=fit_decider_temperature(past),
                           calibrate_decider=True).analyse(fx)
        actual = Tipster(self.an, decider_temperature=4.0, calibrate_decider=True).analyse(fx)
        self.assertEqual(actual.distribution.scores, reference.distribution.scores)
        self.assertEqual(actual.prediction.drivers, reference.prediction.drivers)

    def test_default_date_detects_old_history(self):
        a, b = self.an.known_teams()[:2]
        pred = self.an.predict(a, b)
        self.assertEqual(pred.date, dt.date.today())
        self.assertEqual(pred.confidence, "faible")
        self.assertTrue(any("jours" in warning for warning in pred.warnings))

    def test_reloading_data_invalidates_trained_models(self):
        an = Analyzer().load_maps(self.maps).train(use_gbdt=False)
        an.load_maps(self.maps[:-20])
        with self.assertRaisesRegex(RuntimeError, "non entraine"):
            an.predict("a", "b")

    def test_date_before_first_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aucun resultat anterieur"):
            self.an.predict("a", "b", on=dt.date(2020, 1, 1))


if __name__ == "__main__":
    unittest.main()
