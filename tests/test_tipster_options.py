import datetime as dt
import unittest

from cs2edge.pronostic import Tipster, select_analysis_picks, simulate_tipster


class TipsterOptionsTests(unittest.TestCase):
    def test_invalid_options_fail_before_any_prediction(self):
        for options in (dict(abstention=float("nan")), dict(abstention=.4), dict(abstention=1.1),
                        dict(momentum=float("inf")), dict(edge_min=-.01),
                        dict(decider_temperature=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Tipster(None, **options)
        self.assertEqual(Tipster(None, abstention=.5, edge_min=0, momentum=-1).abstention, .5)

    def test_pure_selection_cannot_emit_a_nan_probability_or_bypass_abstention(self):
        args = (dt.date(2026, 1, 1), "a", "b", 3, "")
        with self.assertRaises(ValueError):
            select_analysis_picks(*args, float("nan"), None)
        with self.assertRaises(ValueError):
            select_analysis_picks(*args, .8, None, abstention=float("nan"))

    def test_backtest_rejects_the_same_options_before_training(self):
        for options in (dict(abstention=float("nan")), dict(momentum=float("inf")),
                        dict(decider_temperature=-1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                simulate_tipster([], None, **options)


if __name__ == "__main__":
    unittest.main()
