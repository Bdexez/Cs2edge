import datetime as dt
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from cs2edge.features import build_datasets
from cs2edge.model import MatchPredictor, LogisticModel
from cs2edge.schema import MapResult, group_matches
from cs2edge.temporal import forward_splits


class TemporalTests(unittest.TestCase):
    def test_results_within_day_cannot_change_other_features_that_day(self):
        d = dt.date(2026, 1, 1)
        maps = [MapResult("one", d, "a", "b", "mirage", 13, 5, best_of=1),
                MapResult("two", d, "a", "b", "nuke", 5, 13, best_of=1),
                MapResult("three", d + dt.timedelta(days=1), "a", "b", "nuke", 13, 5, best_of=1)]
        before, _, _ = build_datasets(group_matches(maps), warmup_maps=0)
        after, _, _ = build_datasets(group_matches([replace(maps[0], rounds_a=5, rounds_b=13),
                                                   *maps[1:]]), warmup_maps=0)
        np.testing.assert_array_equal(before.X[:2], after.X[:2])
        np.testing.assert_array_equal(before.pool_X[:2], after.pool_X[:2])
        self.assertFalse(np.allclose(before.X[2], after.X[2]))

    def test_folds_never_split_a_date(self):
        dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i//7) for i in range(500)]
        splits = forward_splits(dates, .4, 200, 33)
        self.assertEqual(splits[-1][1], len(dates))
        evaluated = []
        for start, end in splits:
            self.assertLess(dates[start-1], dates[start])
            if end < len(dates):
                self.assertLess(dates[end-1], dates[end])
            evaluated.extend(range(start, end))
        self.assertEqual(evaluated, list(range(splits[0][0], len(dates))))

    def test_invalid_refit_cannot_loop_forever(self):
        dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(500)]
        for refit in (0, -1, 1.5):
            with self.subTest(refit=refit), self.assertRaisesRegex(ValueError, "refit_every"):
                forward_splits(dates, .4, 200, refit)

    def test_calibration_does_not_share_training_day(self):
        dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i//7) for i in range(250)]
        X = np.random.default_rng(7).normal(size=(250, 2))
        y = (np.arange(250) % 2).astype(float)
        seen = []
        original = LogisticModel.fit
        def capture(model, X, y, **kwargs):
            seen.append(len(y))
            return original(model, X, y, **kwargs)
        with patch.object(LogisticModel, "fit", capture):
            MatchPredictor(use_gbdt=False).fit(X, y, dates)
        boundary = seen[0]
        self.assertLess(dates[boundary-1], dates[boundary])
        self.assertLessEqual(boundary, 200)

    def test_unsorted_dates_are_rejected(self):
        dates = [dt.date(2025, 1, 1), dt.date(2024, 1, 1)]
        with self.assertRaisesRegex(ValueError, "chronologiquement"):
            forward_splits(dates, .4, 200, 150)


if __name__ == "__main__":
    unittest.main()
