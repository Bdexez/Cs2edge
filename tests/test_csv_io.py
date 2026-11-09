import csv
import datetime as dt
from pathlib import Path
import tempfile
import unittest

from cs2edge import journal
from cs2edge.pronostic import Pick, read_fixtures
from cs2edge.schema import read_csv


class CsvStructureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "input.csv"
        self.pick = Pick(dt.date(2026, 1, 1), "a", "b", 1, "", "vainqueur", "a", .8,
                         "solide", "PRONOSTIC")

    def cases(self):
        return (
            (read_csv, ["match_id", "date", "team_a", "team_b", "map", "rounds_a", "rounds_b"],
             ["one", "2026-01-01", "a", "b", "mirage", "13", "5"]),
            (read_fixtures, ["date", "team_a", "team_b"], ["2026-01-01", "a", "b"]),
            (journal.load, journal.COLUMNS, list(journal._row(self.pick, self.pick.date).values())),
        )

    def write(self, header, values, *, encoding="utf-8"):
        with self.path.open("w", newline="", encoding=encoding) as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerow(values)

    def test_duplicate_headers_are_rejected_by_every_reader(self):
        for read, header, values in self.cases():
            with self.subTest(reader=read.__module__):
                self.write(header + [header[0]], values + [values[0]])
                with self.assertRaisesRegex(ValueError, "colonnes dupliquees"):
                    read(self.path)

    def test_truncated_and_overlong_records_are_rejected(self):
        for read, header, values in self.cases():
            for malformed in (values[:-1], values + ["unexpected"]):
                with self.subTest(reader=read.__module__, cells=len(malformed)):
                    self.write(header, malformed)
                    with self.assertRaisesRegex(ValueError, "ligne 2.*nombre de champs"):
                        read(self.path)

    def test_unterminated_quotes_are_rejected(self):
        for read, header, values in self.cases():
            with self.subTest(reader=read.__module__):
                self.path.write_text(",".join(header) + '\n"' + ",".join(map(str, values)))
                with self.assertRaisesRegex(ValueError, "CSV invalide.*ligne 2"):
                    read(self.path)

    def test_bom_and_quoted_multiline_fields_are_preserved(self):
        for read, header, values in self.cases():
            with self.subTest(reader=read.__module__):
                # Les colonnes supplementaires restent autorisees dans les imports.
                if "event" not in header:
                    header = header + ["event"]
                    values = values + [""]
                values[header.index("event")] = "Finale, LAN\nParis"
                self.write(header, values, encoding="utf-8-sig")
                result = read(self.path)[0]
                self.assertEqual(result["event"] if isinstance(result, dict) else result.event,
                                 "Finale, LAN\nParis")

    def test_invalid_journal_is_never_overwritten_by_record_or_settle(self):
        for content in ("", "pick_id,date\nbad,2026-01-01\n"):
            for update in (lambda: journal.record([self.pick], self.path),
                           lambda: journal.settle(self.path, [])):
                with self.subTest(content=content, update=update):
                    self.path.write_text(content)
                    before = self.path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "colonnes manquantes"):
                        update()
                    self.assertEqual(self.path.read_bytes(), before)

    def test_damaged_journal_records_are_preserved_on_update(self):
        header = journal.COLUMNS
        values = list(journal._row(self.pick, self.pick.date).values())
        for malformed in (values[:-1], values + ["unexpected"]):
            for update in (lambda: journal.record([self.pick], self.path),
                           lambda: journal.settle(self.path, [])):
                with self.subTest(cells=len(malformed), update=update):
                    self.write(header, malformed)
                    before = self.path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "nombre de champs"):
                        update()
                    self.assertEqual(self.path.read_bytes(), before)

    def test_domain_errors_use_physical_line_numbers_after_multiline_fields(self):
        self.path.write_text('date,team_a,team_b,event\n2026-01-01,a,b,"LAN\nParis"\n'
                             '2026-01-01,a,a,Finale\n')
        with self.assertRaisesRegex(ValueError, "ligne 4.*deux equipes distinctes"):
            read_fixtures(self.path)

    def test_nonexistent_journal_still_starts_empty(self):
        self.assertEqual(journal.load(self.path), [])
        self.assertEqual(journal.record([self.pick], self.path), 1)


if __name__ == "__main__":
    unittest.main()
