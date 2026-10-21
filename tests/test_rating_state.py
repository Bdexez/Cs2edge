import copy
import datetime as dt
import unittest

from cs2edge.ratings import EloConfig, EloEngine, TeamRating
from cs2edge.schema import MapResult


class RatingOrientationTests(unittest.TestCase):
    def test_updates_do_not_depend_on_team_order_at_unequal_uncertainty(self):
        day = dt.date(2026, 1, 1)
        for winner_a in (False, True):
            for use_mov in (False, True):
                with self.subTest(winner_a=winner_a, use_mov=use_mov):
                    first = EloEngine(EloConfig(use_mov=use_mov))
                    first.teams = {
                        "a": TeamRating(rating=1700, rd=40, last_date=day),
                        "b": TeamRating(rating=1500, rd=300, last_date=day),
                    }
                    reverse = copy.deepcopy(first)
                    a, b = (13, 5) if winner_a else (5, 13)
                    first.observe(MapResult("m", day, "a", "b", "mirage", a, b, best_of=1))
                    reverse.observe(MapResult("m", day, "b", "a", "mirage", b, a, best_of=1))
                    for team in ("a", "b"):
                        self.assertAlmostEqual(first.teams[team].rating, reverse.teams[team].rating, places=10)
                        self.assertAlmostEqual(first.teams[team].rd, reverse.teams[team].rd, places=10)
                        self.assertAlmostEqual(first.teams[team].fast_rating,
                                               reverse.teams[team].fast_rating, places=10)
                        self.assertEqual(first.teams[team].map_ratings, reverse.teams[team].map_ratings)


class MapInactivityTests(unittest.TestCase):
    def setUp(self):
        self.day = dt.date(2026, 1, 1)
        self.engine = EloEngine(EloConfig(map_prior_weight=0, use_mov=False))
        self.engine.observe(MapResult("nuke", self.day, "a", "b", "nuke", 13, 5, best_of=1))

    def test_map_rating_decays_by_its_own_last_played_date(self):
        before = self.engine.map_rating("a", "nuke", self.day)
        future = self.day + dt.timedelta(days=400)
        self.engine.observe(MapResult("mirage", future, "a", "b", "mirage", 13, 5, best_of=1))
        self.assertEqual(self.engine.teams["a"].map_last_dates["nuke"], self.day)
        self.assertAlmostEqual(self.engine.map_rating("a", "nuke", future), 1500+(before-1500)/2)

    def test_reading_future_rating_does_not_mutate_state(self):
        original = copy.deepcopy(self.engine.teams)
        future = self.day + dt.timedelta(days=10000)
        self.assertAlmostEqual(self.engine.map_rating("a", "nuke", future), 1500, places=5)
        self.assertEqual(self.engine.teams, original)

    def test_next_update_uses_the_decayed_map_rating(self):
        future = self.day + dt.timedelta(days=400)
        expected = self.engine.map_rating("a", "nuke", future)
        self.engine.cfg.k_map = 0
        self.engine.observe(MapResult("return", future, "a", "b", "nuke", 13, 5, best_of=1))
        self.assertAlmostEqual(self.engine.teams["a"].map_ratings["nuke"], expected)


if __name__ == "__main__":
    unittest.main()
