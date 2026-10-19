import datetime as dt
import unittest

from cs2edge.features import FeatureExtractor, _Recent
from cs2edge.schema import MapResult


class RecentStateTests(unittest.TestCase):
    def test_streak_restarts_at_one_after_every_change_of_winner(self):
        recent = _Recent()
        expected = (-1, 1, 2, -1, -2, 1)
        for i, (won, streak) in enumerate(zip((0, 1, 1, 0, 0, 1), expected)):
            recent.add(dt.date(2026, 1, 1), won, 1, 1500, "mirage")
            with self.subTest(step=i):
                self.assertEqual(recent.streak, streak)

    def test_change_of_winner_is_visible_in_match_features(self):
        fx = FeatureExtractor()
        day = dt.date(2026, 1, 1)
        fx.observe_map(MapResult("one", day, "a", "b", "mirage", 5, 13, best_of=1))
        fx.observe_map(MapResult("two", day, "a", "b", "nuke", 13, 5, best_of=1))
        feature = fx.match_features("a", "b", day)[fx.MATCH_FEATURES.index("streak_diff")]
        self.assertAlmostEqual(feature, 2/3)


if __name__ == "__main__":
    unittest.main()
