import csv
import datetime as dt
from pathlib import Path
import tempfile
import unittest

from cs2edge.odds import implied
from cs2edge.pronostic import Fixture, read_fixtures


class FixtureValidationTests(unittest.TestCase):
    def test_api_normalizes_team_names_and_booleans(self):
        fixture = Fixture("2026-01-01", "Natus Vincere", "G2 Esports", lan="false")
        self.assertEqual((fixture.team_a, fixture.team_b), ("navi", "g2"))
        self.assertEqual(fixture.date, dt.date(2026, 1, 1))
        self.assertFalse(fixture.lan)

    def test_invalid_fixture_fields_are_rejected_in_csv(self):
        for key, value in (("best_of", "3.9"), ("best_of", "2"), ("odds_a", "nan"),
                           ("odds_a", "inf"), ("odds_a", "1"), ("team_a", "b"),
                           ("team_a", ""), ("lan", "maybe")):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as tmp:
                row = {"date": "2026-01-01", "team_a": "a", "team_b": "b", key: value}
                path = Path(tmp) / "fixtures.csv"
                with path.open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                with self.assertRaisesRegex(ValueError, "ligne 2"):
                    read_fixtures(path)

    def test_non_finite_odds_are_rejected_by_api_too(self):
        for value in (float("nan"), float("inf"), -1, 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Fixture(dt.date(2026, 1, 1), "a", "b", odds_a=value)
                with self.assertRaises(ValueError):
                    implied(value)


if __name__ == "__main__":
    unittest.main()
