import contextlib
import datetime as dt
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cs2edge.cli import main
from cs2edge.journal import load as load_journal
from cs2edge.model import MatchPredictor
from cs2edge.predict import Analyzer
from cs2edge.schema import write_csv
from cs2edge.sources import simulate_history


class CliSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        data = cls.root / "history.csv"
        maps = simulate_history(n_teams=8, years=.5, seed=11, start=dt.date(2023, 1, 1))
        cls.on = (max(m.date for m in maps)+dt.timedelta(days=1)).isoformat()
        write_csv(str(data), maps)
        cls.bundle = cls.root / "analyzer.cs2edge"
        cls.output = io.StringIO()
        with contextlib.redirect_stdout(cls.output), contextlib.redirect_stderr(io.StringIO()):
            result = main(["train", "--data", str(data), "--no-gbdt", "--warmup", "4",
                           "--bundle", str(cls.bundle)])
        if result != 0:
            raise AssertionError("echec de train --bundle")
        cls.an = Analyzer.load(cls.bundle)
        cls.a, cls.b = cls.an.known_teams()[:2]
        data.unlink()

    def test_train_then_predict_from_another_directory_without_source_csv_or_refit(self):
        cwd = Path.cwd()
        out = io.StringIO()
        try:
            os.chdir(self.root)
            with patch.object(MatchPredictor, "fit", side_effect=AssertionError("refit")), \
                    patch.object(Analyzer, "load_csv", side_effect=AssertionError("CSV")), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["predict", self.a, self.b, "--model", str(self.bundle),
                                       "--date", self.on]), 0)
        finally:
            os.chdir(cwd)
        expected = self.an.predict(self.a, self.b, on=dt.date.fromisoformat(self.on))
        self.assertIn(expected.report(), out.getvalue())
        self.assertIn("Analyseur autonome", self.output.getvalue())

    def test_model_cannot_silently_ignore_training_options(self):
        for flags in (["--data", "ignored.csv"], ["--warmup", "4"], ["--no-gbdt"],
                      ["--elo-config", "ignored.json"]):
            with self.subTest(flags=flags), self.assertRaisesRegex(SystemExit, "--model fournit deja"):
                main(["predict", self.a, self.b, "--model", str(self.bundle), *flags])

    def test_slate_can_load_snapshot_and_archive_without_refitting(self):
        fixtures = self.root / "fixtures.csv"
        journal = self.root / "journal.csv"
        fixtures.write_text("date,team_a,team_b,best_of,match_id\n"
                            f"{self.on},{self.a},{self.b},3,future-one\n")
        with patch.object(MatchPredictor, "fit", side_effect=AssertionError("refit")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["slate", "--model", str(self.bundle), "--fixtures", str(fixtures),
                                   "--momentum", "0", "--no-calibration", "--abstention", ".5",
                                   "--record", str(journal)]), 0)
        rows = load_journal(journal)
        self.assertTrue(rows)
        self.assertTrue(all(row["match_id"] == "future-one" for row in rows))

    def test_missing_model_reports_a_clear_cli_error(self):
        with self.assertRaisesRegex(SystemExit, "erreur :"):
            main(["predict", self.a, self.b, "--model", str(self.root / "missing.cs2edge")])


if __name__ == "__main__":
    unittest.main()
