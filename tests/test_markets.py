import unittest

from cs2edge.markets import build_markets, series_distribution


class HandicapTests(unittest.TestCase):
    def test_bo5_minus_one_and_a_half_includes_three_one(self):
        dist = series_distribution([(str(i), 0.5) for i in range(5)], 5)
        markets = {(m.market, m.selection): m.probability
                   for m in build_markets(dist, "a", "b")}
        self.assertAlmostEqual(markets["handicap -1.5", "a"], 0.3125)
        self.assertAlmostEqual(markets["handicap -2.5", "a"], 0.125)
        self.assertAlmostEqual(markets["handicap +1.5", "a"], 0.6875)

    def test_every_handicap_matches_covered_scores(self):
        for bo in (3, 5):
            dist = series_distribution([(str(i), 0.35 + i * 0.1) for i in range(bo)], bo)
            markets = build_markets(dist, "a", "b")
            keys = [(m.market, m.selection) for m in markets]
            self.assertEqual(len(keys), len(set(keys)))
            for market in markets:
                if not market.market.startswith("handicap"):
                    continue
                with self.subTest(bo=bo, market=market):
                    handicap = float(market.market.split()[-1])
                    expected = sum(p for (a, b), p in dist.scores.items()
                                   if (a-b if market.selection == "a" else b-a) + handicap > 0)
                    self.assertAlmostEqual(market.probability, expected)


if __name__ == "__main__":
    unittest.main()
