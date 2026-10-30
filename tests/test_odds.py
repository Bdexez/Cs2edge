import math
import unittest

from cs2edge.odds import devig_power, expected_value, find_value, kelly_fraction, simulate_flat


class PowerDevigTests(unittest.TestCase):
    def test_both_outputs_follow_the_same_power_even_for_extreme_prices(self):
        for odds in ((1.1, 1.2), (100, 200), (1.01, 1.02), (1.75, 2.1),
                     (math.nextafter(1., 2.), 2.)):
            with self.subTest(odds=odds):
                pa, pb = devig_power(*odds)
                exponent = math.log(pa)/math.log(1/odds[0])
                self.assertAlmostEqual((1/odds[1])**exponent, pb, places=11)
                self.assertAlmostEqual(pa+pb, 1., places=14)
                reversed_probs = devig_power(*reversed(odds))
                self.assertAlmostEqual(pa, reversed_probs[1], places=14)

    def test_fair_prices_are_unchanged_and_equal_prices_stay_equal(self):
        for pa in (.01, .25, .5, .9):
            result = devig_power(1/pa, 1/(1-pa))
            self.assertAlmostEqual(result[0], pa, places=12)
        for odds in (1.01, 1.5, 2, 10, 1000):
            self.assertEqual(devig_power(odds, odds), (.5, .5))


class OddsValidationTests(unittest.TestCase):
    def test_invalid_probability_cannot_produce_a_maximum_stake(self):
        for p in (float("nan"), float("inf"), -.1, 1.1):
            for fn in (lambda: kelly_fraction(p, 2), lambda: expected_value(p, 2),
                       lambda: find_value(p, 2, 2)):
                with self.subTest(p=p), self.assertRaises(ValueError):
                    fn()

    def test_invalid_prices_and_staking_parameters_are_rejected(self):
        for odds in (1, 0, float("nan"), float("inf")):
            for fn in (kelly_fraction, expected_value):
                with self.subTest(odds=odds, fn=fn), self.assertRaises(ValueError):
                    fn(.7, odds)
        for options in (dict(cap=-1), dict(scale=-1), dict(cap=float("nan")),
                        dict(scale=float("inf")), dict(cap=1.1)):
            with self.assertRaises(ValueError):
                kelly_fraction(.7, 2, **options)
        self.assertEqual(kelly_fraction(.7, 2, scale=0), 0)
        self.assertEqual(kelly_fraction(.7, 2, cap=0), 0)

    def test_invalid_backtest_cannot_report_a_plausible_roi(self):
        for bets in ([(.7, 2., 2)], [(float("nan"), 2., 1)], [(.7, float("inf"), 0)]):
            with self.assertRaises(ValueError):
                simulate_flat(bets)
        for stake in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                simulate_flat([(.7, 2., 1)], stake=stake)
        result = simulate_flat([(.7, 2., 1), (.7, 2., 0)], stake=5)
        self.assertEqual(result["profit"], 0)
        self.assertEqual(result["turnover"], 10)


if __name__ == "__main__":
    unittest.main()
