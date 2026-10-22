import unittest

import numpy as np

from cs2edge import metrics
from cs2edge.model import LogisticModel, MatchPredictor, recency_weights


class NumericContractsTests(unittest.TestCase):
    def test_metrics_reject_silent_broadcasting_and_invalid_probabilities(self):
        functions = (metrics.log_loss, metrics.brier, metrics.accuracy,
                     metrics.expected_calibration_error, metrics.reliability_table, metrics.summarize)
        bad_pairs = (([1, 0], [[.9], [.1]]), ([1, 0], [.9]), ([], []),
                     ([1, 0], [.9, np.nan]), ([1, 0], [1.1, 0]),
                     ([1, .5], [.9, .1]), ([[1], [0]], [.9, .1]))
        for fn in functions:
            for y, p in bad_pairs:
                with self.subTest(metric=fn.__name__, y=y, p=p), self.assertRaises(ValueError):
                    fn(y, p)
        self.assertAlmostEqual(metrics.log_loss([1, 0], [.9, .1]), -np.log(.9))
        self.assertEqual(metrics.brier([1, 0], [1, 0]), 0)
        self.assertTrue(np.isfinite(metrics.log_loss([0, 1], [1, 0])))

    def test_calibration_requires_positive_integer_bins(self):
        for fn in (metrics.expected_calibration_error, metrics.reliability_table):
            for bins in (0, -1, 1.5, True):
                with self.subTest(bins=bins), self.assertRaises(ValueError):
                    fn([1, 0], [.9, .1], bins=bins)

    def test_training_rejects_inconsistent_arrays_and_weights(self):
        X = np.array([[1., 2.], [-1., -2.]])
        for x, y in ((X, [1]), (X, [[1], [0]]), (X, [1, np.nan]),
                     (X * np.nan, [1, 0]), (np.empty((2, 0)), [1, 0])):
            for model in (LogisticModel(), MatchPredictor(use_gbdt=False)):
                with self.subTest(model=type(model).__name__), self.assertRaises(ValueError):
                    model.fit(x, y)
        for weights in ([1], [[1], [1]], [0, 0], [1, -1], [np.inf, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                LogisticModel().fit(X, [1, 0], weights)
        model = LogisticModel().fit(X, [1, 0], [1, 0])
        for bad in ([[1]], [[np.nan, 1]], np.empty((0, 2))):
            with self.assertRaises(ValueError):
                model.predict_proba(bad)

    def test_model_parameters_fail_before_fitting(self):
        X = np.arange(120, dtype=float).reshape(60, 2)
        y = np.tile([1, 0], 30)
        for options in (dict(l2=-1), dict(gbdt_weight=2), dict(recency_halflife=0),
                        dict(calibration_frac=np.nan)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MatchPredictor(**options).fit(X, y)
        for options in (dict(max_iter=0), dict(tol=0), dict(l2=np.inf)):
            with self.assertRaises(ValueError):
                LogisticModel(**options).fit(X, y)
        with self.assertRaises(ValueError):
            recency_weights([], halflife=0)


if __name__ == "__main__":
    unittest.main()
