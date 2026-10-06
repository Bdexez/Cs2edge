import datetime as dt
import unittest

import numpy as np

from cs2edge.backtest import walk_forward
from cs2edge.features import build_datasets
from cs2edge.schema import group_matches
from cs2edge.sources import simulate_history, simulate_with_truth


class SimulationTests(unittest.TestCase):
    def test_both_regimes_are_reproducible_and_have_distinct_maps(self):
        for regime in ("facile", "realiste"):
            for seed in (3, 7, 11):
                with self.subTest(regime=regime, seed=seed):
                    options = dict(n_teams=8, years=.5, seed=seed, regime=regime,
                                   start=dt.date(2023, 1, 1))
                    maps = simulate_history(**options)
                    self.assertEqual(maps, simulate_history(**options))
                    matches = group_matches(maps)
                    self.assertTrue(matches)
                    for match in matches:
                        self.assertTrue(match.is_complete)
                        self.assertEqual(len({m.map_name for m in match.maps}), len(match.maps))
                    ds, pds, _ = build_datasets(matches)
                    result = walk_forward(ds, pds, initial_frac=.6, use_gbdt=False)
                    self.assertTrue(np.all(np.isfinite(result.predictions)))
                    self.assertTrue(np.all((result.predictions > 0) & (result.predictions < 1)))
                    self.assertTrue(np.isfinite(result.metrics["log_loss"]))

    def test_truth_sampling_does_not_change_generated_history(self):
        options = dict(n_teams=4, years=.1, seed=7, start=dt.date(2023, 1, 1))
        expected = simulate_history(**options)
        actual, truth = simulate_with_truth(**options, veto_samples=20)
        self.assertEqual(actual, expected)
        self.assertEqual(set(truth), {m.match_id for m in actual})
        self.assertTrue(all(0 <= p <= 1 for p in truth.values()))

    def test_invalid_inputs_are_rejected(self):
        for options in (dict(n_teams=1), dict(n_teams=41), dict(years=0),
                        dict(years=float("nan")), dict(regime="missing")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                simulate_history(**options)


if __name__ == "__main__":
    unittest.main()
