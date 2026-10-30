import math
import unittest

from cs2edge.odds import devig_power


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


if __name__ == "__main__":
    unittest.main()
