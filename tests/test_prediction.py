import datetime as dt
from dataclasses import replace
import unittest
from unittest.mock import patch

from cs2edge.backtest import walk_forward
from cs2edge.predict import Analyzer, combine_probability
from cs2edge.pronostic import Fixture, Tipster, simulate_tipster
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


if __name__ == "__main__":
    unittest.main()
