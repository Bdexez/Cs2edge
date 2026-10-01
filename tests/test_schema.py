import csv
import datetime as dt
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

from cs2edge.features import build_datasets
from cs2edge.schema import MapResult, group_matches, read_csv, write_csv
from cs2edge.sources import from_records


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(match_id="m", date="2026-01-01", team_a="Natus Vincere",
                        team_b="G2 Esports", map="de_mirage", rounds_a=13,
                        rounds_b=5, best_of=1, lan="false",
                        roster_a="ONE, Two", roster_b=["Three", "four"])

    def test_csv_round_trip_preserves_every_field(self):
        result = from_records([self.row])[0]
        self.assertFalse(result.lan)
        self.assertEqual(result.roster_a, ("one", "two"))
        self.assertEqual(result.roster_b, ("four", "three"))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "maps.csv")
            write_csv(path, [result])
            self.assertEqual(read_csv(path), [result])

    def test_api_and_csv_share_validation(self):
        for key, value in (("rounds_a", -1), ("rounds_a", "13.9"),
                           ("rounds_a", "nan"), ("rounds_a", "inf"),
                           ("rounds_a", 5), ("best_of", 2), ("best_of", 0),
                           ("team_a", ""), ("team_a", "g2"), ("map", ""),
                           ("map_index", "0.5"), ("lan", "maybe")):
            row = {**self.row, key: value}
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "enregistrement 1"):
                    from_records([row])
                path = Path(tmp) / "bad.csv"
                with path.open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                with self.assertRaisesRegex(ValueError, "ligne 2"):
                    read_csv(str(path))

    def test_missing_map_is_not_replaced_with_unknown(self):
        self.row.pop("map")
        with self.assertRaises(ValueError):
            from_records([self.row])


class SeriesTests(unittest.TestCase):
    def setUp(self):
        self.first = MapResult("m", dt.date(2026, 1, 1), "a", "b", "mirage", 13, 5)

    def test_partial_series_has_no_final_label(self):
        match = group_matches([self.first])[0]
        self.assertFalse(match.is_complete)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _ = match.winner
        ds, _, state = build_datasets([match], warmup_maps=0)
        self.assertEqual(len(ds), 0)
        self.assertEqual(state.elo.maps_played("a"), 1)

    def test_inconsistent_maps_are_rejected(self):
        for changes in (dict(team_b="c"), dict(date=dt.date(2026, 1, 2)),
                        dict(best_of=5), dict(event="other"), dict(lan=True)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "incoherentes"):
                group_matches([self.first, replace(self.first, map_index=1, **changes)])

    def test_maps_after_final_score_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "apres la fin"):
            group_matches([replace(self.first, map_index=i) for i in range(3)])

    def test_legacy_missing_indices_preserve_row_order(self):
        match = group_matches([self.first, replace(self.first, map_name="nuke")])[0]
        self.assertTrue(match.is_complete)
        self.assertEqual([m.map_name for m in match.maps], ["mirage", "nuke"])


if __name__ == "__main__":
    unittest.main()
