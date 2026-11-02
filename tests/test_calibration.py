import datetime as dt
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from cs2edge.calibration import prequential_maps
from cs2edge.model import MatchPredictor
from cs2edge.predict import Analyzer
from cs2edge.pronostic import _fit_decider_on, fit_decider_temperature
from cs2edge.sources import simulate_history


class PrequentialCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = simulate_history(n_teams=8, years=.7, seed=17, start=dt.date(2023, 1, 1))
        cls.an = Analyzer().load_maps(cls.maps).train(use_gbdt=False)

    def test_every_prediction_excludes_the_entire_target_day(self):
        calls = []
        original = MatchPredictor.fit
        def capture(model, X, y, dates=None, feature_names=()):
            calls.append(max(dates))
            return original(model, X, y, dates, feature_names)
        with patch.object(MatchPredictor, "fit", capture):
            history = prequential_maps(self.an.match_ds, self.an.map_ds, refit_every=80)
        available = 0
        for i, through in enumerate(history.trained_through):
            if through:
                available += 1
                self.assertLess(dt.date.fromisoformat(through), self.an.match_ds.dates[i])
                self.assertIn(dt.date.fromisoformat(through), calls)
                self.assertTrue(np.isfinite(history.pools[i]).all())
            else:
                self.assertTrue(np.isnan(history.pools[i]).all())
        self.assertGreater(available, 100)

    def test_changing_future_labels_cannot_change_earlier_calibration(self):
        ds, pds = self.an.match_ds, self.an.map_ds
        cutoff = ds.dates[-100]
        upto = next(i for i, day in enumerate(ds.dates) if day == cutoff)
        before = prequential_maps(ds, pds)
        changed = Analyzer().load_maps([replace(mp, rounds_a=mp.rounds_b, rounds_b=mp.rounds_a)
                                        if mp.date >= cutoff else mp for mp in self.maps])
        after = prequential_maps(changed.match_ds, changed.map_ds)
        np.testing.assert_array_equal(before.pools[:upto], after.pools[:upto])
        self.assertEqual(_fit_decider_on(ds, before, upto),
                         _fit_decider_on(changed.match_ds, after, upto))

    def test_live_calibration_reuses_cached_predictions_and_invalidates_on_training(self):
        an = Analyzer().load_maps(self.maps).train(use_gbdt=False)
        first = an.calibration_history()
        with patch("cs2edge.calibration.prequential_maps", side_effect=AssertionError("recalcul")):
            self.assertIs(an.calibration_history(), first)
            temperature = fit_decider_temperature(an)
        self.assertEqual(temperature, _fit_decider_on(an.match_ds, first, len(an.match_ds)))
        an.train(use_gbdt=False)
        self.assertIsNone(an._calibration_history)


if __name__ == "__main__":
    unittest.main()
