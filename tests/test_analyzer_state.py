import datetime as dt
import unittest
from unittest.mock import patch

import numpy as np

from cs2edge.model import MatchPredictor
from cs2edge.predict import Analyzer
from cs2edge.sources import simulate_history


class AnalyzerStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = simulate_history(n_teams=8, years=.5, seed=11, start=dt.date(2023, 1, 1))
        cls.on = max(m.date for m in cls.maps) + dt.timedelta(days=1)

    def setUp(self):
        self.an = Analyzer(momentum=.12).load_maps(self.maps).train(use_gbdt=False)
        self.a, self.b = self.an.known_teams()[:2]

    def test_failed_grouping_keeps_the_original_history_and_prediction(self):
        expected = self.an.predict(self.a, self.b, on=self.on)
        with self.assertRaisesRegex(ValueError, "map dupliquee"):
            self.an.load_maps([self.maps[0], self.maps[0]])
        self.assertEqual(self.an.maps, self.maps)
        self.assertEqual(self.an.predict(self.a, self.b, on=self.on), expected)

    def test_failed_feature_extraction_keeps_the_original_analyzer(self):
        expected = self.an.predict(self.a, self.b, on=self.on)
        with patch("cs2edge.predict.build_datasets", side_effect=ValueError("extraction impossible")):
            with self.assertRaisesRegex(ValueError, "extraction impossible"):
                self.an.load_maps(self.maps[:-10])
        self.assertEqual(self.an.maps, self.maps)
        self.assertEqual(self.an.predict(self.a, self.b, on=self.on), expected)

    def test_failed_second_fit_preserves_both_models_and_training_options(self):
        expected = self.an.predict(self.a, self.b, on=self.on)
        models = (self.an.match_model, self.an.map_model)
        original = MatchPredictor.fit

        def fail_map(model, *args, **kwargs):
            if not model.use_gbdt:
                raise ValueError("entrainement map impossible")
            return original(model, *args, **kwargs)

        with patch.object(MatchPredictor, "fit", fail_map), \
                patch.object(MatchPredictor, "_fit_gbdt", return_value=None):
            with self.assertRaisesRegex(ValueError, "entrainement map impossible"):
                self.an.train(use_gbdt=True, l2=9)
        self.assertIs(self.an.match_model, models[0])
        self.assertIs(self.an.map_model, models[1])
        self.assertEqual(self.an._training_options, {"use_gbdt": False})
        self.assertEqual(self.an.predict(self.a, self.b, on=self.on), expected)

    def activate_auto_momentum(self):
        with patch.object(self.an, "estimate_momentum",
                          return_value={"significatif": True, "momentum": .4}):
            self.an.apply_estimated_momentum()
        self.assertEqual(self.an.momentum, .4)

    def test_reload_invalidates_estimated_momentum_and_train_estimates_it_again(self):
        self.activate_auto_momentum()
        self.an.load_maps(self.maps[:-10])
        self.assertEqual(self.an.momentum, 0.0)
        self.assertTrue(self.an._momentum_auto)
        with patch("cs2edge.markets.estimate_momentum",
                   return_value={"significatif": True, "momentum": -.2}) as estimate:
            self.an.train(use_gbdt=False)
        self.assertEqual(estimate.call_count, 1)
        self.assertEqual(self.an.momentum, -.2)
        self.assertIs(estimate.call_args.kwargs["history"], self.an.calibration_history())

    def test_retraining_refreshes_auto_momentum_with_new_options(self):
        self.activate_auto_momentum()
        with patch("cs2edge.markets.estimate_momentum",
                   return_value={"significatif": False, "momentum": .1}) as estimate:
            self.an.train(use_gbdt=False, l2=4)
        self.assertEqual(estimate.call_count, 1)
        self.assertEqual(self.an.momentum, 0.0)
        reference = Analyzer().load_maps(self.maps).train(use_gbdt=False, l2=4)
        np.testing.assert_array_equal(self.an.calibration_history().pools,
                                      reference.calibration_history().pools)
        self.assertIs(estimate.call_args.kwargs["history"], self.an.calibration_history())

    def test_failed_auto_calibration_keeps_the_previous_models_and_momentum(self):
        self.activate_auto_momentum()
        expected = self.an.predict(self.a, self.b, on=self.on)
        models = (self.an.match_model, self.an.map_model)
        with patch("cs2edge.markets.estimate_momentum", side_effect=ValueError("calibration impossible")):
            with self.assertRaisesRegex(ValueError, "calibration impossible"):
                self.an.train(use_gbdt=False, l2=4)
        self.assertIs(self.an.match_model, models[0])
        self.assertIs(self.an.map_model, models[1])
        self.assertEqual(self.an._training_options, {"use_gbdt": False})
        self.assertEqual(self.an.predict(self.a, self.b, on=self.on), expected)

    def test_explicit_momentum_survives_reload_and_training(self):
        with patch("cs2edge.markets.estimate_momentum", side_effect=AssertionError("mode manuel")):
            self.an.load_maps(self.maps[:-10]).train(use_gbdt=False)
        self.assertEqual(self.an.momentum, .12)
        self.assertFalse(self.an._momentum_auto)


if __name__ == "__main__":
    unittest.main()
