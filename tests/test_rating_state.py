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


if __name__ == "__main__":
    unittest.main()
