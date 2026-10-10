from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cs2edge import journal
from cs2edge.pronostic import Analysis, Fixture, Pick, read_fixtures
from cs2edge.schema import MapResult, group_matches


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "journal.csv"
        self.day = dt.date(2026, 1, 1)
        self.pick = Pick(self.day, "a", "b", 1, "", "vainqueur", "a", .8,
                         "solide", "PRONOSTIC")
        self.results = group_matches([
            MapResult("one", self.day, "a", "b", "mirage", 13, 5, 1, event="first"),
            MapResult("two", self.day, "a", "b", "mirage", 5, 13, 1, event="second"),
        ])

    def test_duplicate_in_batch_is_written_once(self):
        self.assertEqual(journal.record([self.pick, self.pick], self.path), 1)
        self.assertEqual(journal.record([self.pick], self.path), 0)
        self.assertEqual(len(journal.load(self.path)), 1)

    def test_match_id_disambiguates_and_survives_reload(self):
        pick = replace(self.pick, match_id="one", data_through="2025-12-31")
        journal.record([pick], self.path)
        for results in (self.results, self.results[::-1]):
            with self.subTest(order=[m.match_id for m in results]):
                path = self.path.with_name(results[0].match_id + ".csv")
                journal.record([pick], path)
                self.assertEqual(journal.settle(path, results)["pointes"], 1)
                row = journal.load(path)[0]
                self.assertEqual(row["resultat"], "gagne")
                self.assertEqual(row["donnees_jusquau"], "2025-12-31")
                self.assertIsNotNone(dt.datetime.fromisoformat(row["emis_a"]).tzinfo)
                self.assertEqual(journal.scorecard(path).settled[0].pick.match_id, "one")

    def test_unknown_id_never_falls_back_to_similar_match(self):
        journal.record([replace(self.pick, match_id="missing")], self.path)
        self.assertEqual(journal.settle(self.path, self.results)["pointes"], 0)

    def test_ambiguous_legacy_match_stays_pending(self):
        journal.record([self.pick], self.path)
        stats = journal.settle(self.path, self.results)
        self.assertEqual((stats["pointes"], stats["ambigus"], stats["en_attente"]), (0, 1, 1))
        self.assertEqual(journal.load(self.path)[0]["resultat"], "")

    def test_event_disambiguates_legacy_rows(self):
        journal.record([replace(self.pick, event="first")], self.path)
        self.assertEqual(journal.settle(self.path, self.results)["pointes"], 1)
        self.assertEqual(journal.load(self.path)[0]["resultat"], "gagne")

    def test_partial_series_cannot_settle(self):
        journal.record([replace(self.pick, best_of=3)], self.path)
        partial = group_matches([MapResult("partial", self.day, "a", "b", "nuke", 13, 5)])
        self.assertEqual(journal.settle(self.path, partial)["pointes"], 0)

    def test_legacy_header_can_be_appended_without_duplicates(self):
        row = journal._row(self.pick, self.day)
        old_fields = journal.COLUMNS[:17]
        row["pick_id"] = "old-format-id"
        with self.path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=old_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        self.assertEqual(journal.record([self.pick], self.path), 0)
        self.assertEqual(journal.record([replace(self.pick, match_id="two")], self.path), 1)
        self.assertEqual(len(journal.load(self.path)), 2)

    def test_failed_atomic_replace_preserves_previous_journal(self):
        journal.record([self.pick], self.path)
        previous = self.path.read_bytes()
        with patch("cs2edge.journal.os.replace", side_effect=OSError("interruption")):
            with self.assertRaises(OSError):
                journal.record([replace(self.pick, match_id="two")], self.path)
        self.assertEqual(self.path.read_bytes(), previous)

    def test_concurrent_records_preserve_all_distinct_picks(self):
        def write(i):
            return journal.record([self.pick, replace(self.pick, match_id=str(i))], self.path)
        with ThreadPoolExecutor(max_workers=6) as pool:
            added = list(pool.map(write, range(6)))
        rows = journal.load(self.path)
        self.assertEqual(sum(added), 7)
        self.assertEqual(len({r["pick_id"] for r in rows}), 7)

    def test_fixtures_accept_stable_match_id(self):
        path = self.path.with_name("fixtures.csv")
        path.write_text("date,team_a,team_b,match_id\n2026-01-01,a,b,source-42\n")
        self.assertEqual(read_fixtures(path)[0].match_id, "source-42")

    def test_card_archives_abstentions_and_counts_distinct_fixtures(self):
        first = replace(self.pick, match_id="one")
        picks = [first, replace(first, market="total maps -1.5", selection="oui")]
        card = [Analysis(Fixture(self.day, "a", "b", 1, match_id="one"),
                         None, None, None, [], picks),
                Analysis(Fixture(self.day, "c", "d", 1, match_id="skip"),
                         None, None, None, [], [], skipped="historique insuffisant")]
        self.assertEqual(journal.record_card(card, self.path), 3)
        self.assertEqual(journal.record_card(card, self.path), 0)
        row = next(r for r in journal.load(self.path) if r["verdict"] == "ABSTENTION")
        self.assertEqual(row["motif"], "historique insuffisant")
        stats = journal.settle(self.path, self.results)
        self.assertEqual(stats["en_attente"], 0)
        result = journal.scorecard(self.path)
        self.assertEqual((result.n_fixtures, result.n_skipped), (2, 1))
        self.assertTrue(result.coverage_known)
        self.assertEqual(len(result.settled), 2)

    def test_abstention_only_card_produces_a_coverage_report(self):
        card = [Analysis(Fixture(self.day, "a", "b", 1), None, None, None, [], [], skipped="serre")]
        journal.record_card(card, self.path)
        stats = journal.settle(self.path, self.results)
        self.assertEqual((stats["pointes"], stats["en_attente"]), (0, 0))
        record = journal.scorecard(self.path)
        self.assertEqual((record.n_fixtures, record.n_skipped), (1, 1))
        self.assertIn("100%", record.report())

    def test_legacy_coverage_is_reported_as_unknown(self):
        journal.record([replace(self.pick, match_id="one")], self.path)
        journal.settle(self.path, self.results)
        record = journal.scorecard(self.path)
        self.assertFalse(record.coverage_known)
        self.assertIn("Couverture des abstentions indisponible", record.report())


if __name__ == "__main__":
    unittest.main()
