import datetime as dt
from dataclasses import replace
import unittest

from cs2edge.pronostic import Pick, SettledPick, TipsterRecord


class RecordStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.pick = Pick(dt.date(2026, 1, 1), "a", "b", 3, "", "vainqueur", "a",
                         .8, "solide", "PRONOSTIC", match_id="same")

    def test_many_picks_on_one_match_are_not_many_independent_observations(self):
        record = TipsterRecord([SettledPick(self.pick, False)]*100, 1, 0)
        stat = record.by(lambda p: p.tier)["solide"]
        self.assertEqual(stat["n"], 100)
        self.assertEqual(stat["n_matchs"], 1)
        self.assertAlmostEqual(stat["erreur_type"], .4)
        self.assertFalse(stat["significatif"])
        self.assertIn("n trop faible", record.report())

    def test_duplicate_markets_do_not_shrink_uncertainty(self):
        picks = [SettledPick(replace(self.pick, match_id=str(i)), False) for i in range(30)]
        single = TipsterRecord(picks, 30, 0).by(lambda p: p.tier)["solide"]
        doubled = TipsterRecord(picks + picks, 30, 0).by(lambda p: p.tier)["solide"]
        self.assertAlmostEqual(single["erreur_type"], doubled["erreur_type"])
        self.assertTrue(single["significatif"])
        self.assertEqual(single["n_matchs"], doubled["n_matchs"])


if __name__ == "__main__":
    unittest.main()
