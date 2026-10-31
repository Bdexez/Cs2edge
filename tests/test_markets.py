import unittest

from cs2edge.markets import build_markets, project_veto, series_distribution


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


class SeriesInputTests(unittest.TestCase):
    def test_short_or_oversized_pool_produces_a_clear_veto_error(self):
        for size in (0, 3, 5, 6, 8):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "pool"):
                project_veto([(str(i), .5) for i in range(size)], 3)

    def test_invalid_series_can_never_create_nonterminal_scores(self):
        for bo, order in ((2, [("a", .5)]*2), (3, [("a", .5)]),
                          (3, [("a", .5)]*3), (1, [("a", float("nan"))]),
                          (1, [("a", 1.1)]), (1, [("", .5)])):
            with self.subTest(bo=bo, order=order), self.assertRaises(ValueError):
                series_distribution(order, bo)

    def test_extreme_momentum_stays_finite_and_deterministic_maps_stay_exact(self):
        for momentum in (-1000, 0, 1000):
            dist = series_distribution([("a", .3), ("b", .4), ("c", .6)], 3, momentum)
            self.assertAlmostEqual(sum(dist.scores.values()), 1.)
            self.assertTrue(all(p >= 0 for p in dist.scores.values()))
            self.assertTrue(all(max(score) == 2 for score in dist.scores))
        for probability in (0., 1.):
            dist = series_distribution([("a", probability), ("b", probability), ("c", probability)], 3)
            self.assertEqual(dist.p_a, probability)
        with self.assertRaises(ValueError):
            series_distribution([("a", .5)], 1, momentum=float("inf"))


if __name__ == "__main__":
    unittest.main()
