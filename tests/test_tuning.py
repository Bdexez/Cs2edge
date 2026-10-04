import datetime as dt
from dataclasses import replace
import unittest
from unittest.mock import patch

from cs2edge import backtest
from cs2edge.schema import group_matches
from cs2edge.sources import simulate_history


class TuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = group_matches(simulate_history(n_teams=8, years=.9, seed=17,
                                                    start=dt.date(2023, 1, 1)))

    def test_validation_scores_only_reserved_ids_and_keeps_past_history(self):
        calls = []
        original = backtest.walk_forward
        def observed(ds, *args, **kwargs):
            result = original(ds, *args, **kwargs)
            calls.append((ds, result))
            return result
        with patch.object(backtest, "walk_forward", observed):
            _, report = backtest.tune(self.matches, grid={"k_base": (12,)}, verbose=False)
        self.assertFalse(set(report["reglage_match_ids"]) & set(report["validation_match_ids"]))
        for ds, result in calls[-2:]:
            self.assertEqual([m["match_id"] for m in result.meta], report["validation_match_ids"])
            self.assertLess(ds.dates[0], result.dates[0])
            self.assertGreater(len(ds), len(result.y))
        self.assertAlmostEqual(report["logloss_validation_apres"], calls[-1][1].metrics["log_loss"])
        self.assertEqual(report["n_matchs_validation"], len(calls[-1][1].y))

    def test_future_results_cannot_change_tuned_configuration(self):
        cfg, report = backtest.tune(self.matches, grid={"k_base": (8, 12)}, verbose=False)
        reserved = set(report["validation_match_ids"])
        changed = [replace(m, maps=[replace(mp, rounds_a=mp.rounds_b, rounds_b=mp.rounds_a)
                                   for mp in m.maps]) if m.match_id in reserved else m
                   for m in self.matches]
        again, second = backtest.tune(changed, grid={"k_base": (8, 12)}, verbose=False)
        self.assertEqual(cfg.to_dict(), again.to_dict())
        self.assertEqual(report["essais"], second["essais"])
        self.assertNotEqual(report["logloss_validation_apres"], second["logloss_validation_apres"])

    def test_invalid_holdout_fractions_are_rejected(self):
        for fraction in (0, 1, -.1, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, "holdout_frac"):
                backtest.tune(self.matches, holdout_frac=fraction, grid={})


if __name__ == "__main__":
    unittest.main()
